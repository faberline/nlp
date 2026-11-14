#!/usr/bin/env python3
"""Generates the tiny deterministic Seq2Seq test model under
translate/models/translate-test-tiny/ and translate/models/manifest.json.

Licence: CC0-1.0. Fixed-seed synthetic weights exist solely to verify the offline
Seq2Seq runtime (`nlp.models.seq2seq.v1`: manifest verification, SHA-256 checks,
ByteLevel BPE tokenizer, greedy/beam decoding, and ONNX vs Core ML numerical parity).

Works out-of-the-box with Python 3 + `numpy` alone (emitting deterministic ONNX IR v8 /
opset 13 and Core ML spec v4 NeuralNetwork protobufs directly), and additionally runs
`onnx.checker` if `onnx` is installed in the active environment.
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "crates" / "faberline-nlp-translate" / "models"
MODEL_ID = "translate-test-tiny"
MODEL_DIR = OUT_DIR / MODEL_ID

LICENCE = "CC0-1.0"
SEED = 20261114
SRC_MAX_LEN = 80
TGT_MAX_LEN = 32
D_MODEL = 8
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
VOCAB = 4 + 256  # 260
SIZE_LIMIT = 120_000

SUPPORTED_VARIANTS = ["zh-Hant", "zh-Hans", "en-US", "en-GB", "ja", "ko"]

# Must match FIXTURES in translate/tests/translate_coreml.rs.
FIXTURES = [
    ("fx01", "<2en-US> [SUBJ=lin_yuan:she] [REG=polite] 她推開門。"),
    ("fx02", "<2zh-Hant> She opened the door quietly."),
    ("fx03", "<2ja> [SUBJ=lin_yuan:she] 今天天氣很好。"),
    ("fx04", "<2ko> [[PH0]]走了進來。"),
    ("fx05", "<2zh-Hant> 彼女は静かにドアを開けた。"),
    ("fx06", "<2en-US> 그녀는 조용히 문을 열었다."),
    ("fx07", "<2en-GB> 我們使用 [[PH0]] 模型。"),
    ("fx08", "<2en-US> [REG=casual] 哈哈，今天好累😋"),
    ("fx09", "<2en-US> [FACT=negated] 他沒有去學校。"),
    ("fx10", ""),
    ("fx11", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqr"),
    ("fx12", "好"),
]


def byte_symbols():
    """GPT-2 byte-to-printable-char table used by HuggingFace ByteLevel pre-tokenizer."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def make_tokenizer_json(path: Path):
    symbols = byte_symbols()
    vocab = {
        "[PAD]": PAD_ID,
        "[UNK]": UNK_ID,
        "[BOS]": BOS_ID,
        "[EOS]": EOS_ID,
    }
    for b in range(256):
        vocab[symbols[b]] = 4 + b
    assert len(vocab) == VOCAB

    added_tokens = [
        {
            "id": tid,
            "content": tok,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for tok, tid in [
            ("[PAD]", PAD_ID),
            ("[UNK]", UNK_ID),
            ("[BOS]", BOS_ID),
            ("[EOS]", EOS_ID),
        ]
    ]

    doc = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": True,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [],
        },
    }
    write_json(path, doc)


def encode_bytes_bpe(text: str) -> list[int]:
    """Exact token IDs produced by ByteLevel BPE with empty merges on non-special input."""
    return [4 + b for b in text.encode("utf-8")]


def decode_bytes_bpe(ids: list[int]) -> str:
    raw = bytearray()
    for tid in ids:
        if tid == EOS_ID:
            break
        if tid in (PAD_ID, UNK_ID, BOS_ID):
            continue
        if 4 <= tid < VOCAB:
            raw.append(tid - 4)
    return raw.decode("utf-8", errors="replace")


def make_weights():
    rng = np.random.default_rng(SEED)
    enc_emb = rng.normal(0.0, 0.35, size=(VOCAB, D_MODEL)).astype(np.float32)
    enc_w = rng.normal(0.0, 0.35, size=(D_MODEL, D_MODEL)).astype(np.float32)
    enc_b = rng.normal(0.0, 0.15, size=(D_MODEL,)).astype(np.float32)
    # Dedicated non-empty sentinel channel d=7 so step>=6 cleanly triggers EOS.
    enc_emb[:, 7] = 0.0
    enc_w[:, 7] = 0.0
    enc_b[7] = 1.0

    dec_emb = rng.normal(0.0, 0.35, size=(VOCAB, D_MODEL)).astype(np.float32)
    dec_emb[:, 7] = 0.0

    # Cross-mixing matrix [1, TGT_MAX_LEN, SRC_MAX_LEN]: mixes active source tokens into
    # each decoder position; for step < 6 every partial sum over s in 0..SRC_MAX_LEN is
    # strictly negative (suppressing EOS), and from step 6 onward strictly >= +2.0 (clean EOS).
    mix_table = rng.normal(-0.08, 0.15, size=(1, TGT_MAX_LEN, SRC_MAX_LEN)).astype(np.float32)
    for step in range(TGT_MAX_LEN):
        if step < 6:
            mix_table[0, step, 0] = -2.2
            # Guarantee every prefix sum 1..=SRC_MAX_LEN is <= -1.2.
            csum = np.cumsum(mix_table[0, step, :])
            if float(np.max(csum)) > -1.2:
                mix_table[0, step, 0] -= float(np.max(csum)) + 1.2
        else:
            mix_table[0, step, :] = rng.normal(0.05, 0.08, size=(SRC_MAX_LEN,)).astype(np.float32)
            mix_table[0, step, 0] = 2.2

    proj_w = rng.normal(0.0, 0.45, size=(D_MODEL, VOCAB)).astype(np.float32)
    proj_w[6, :] = 0.0
    proj_w[7, :] = 0.0
    proj_w[7, EOS_ID] = 4.5

    # Bias printable lowercase ASCII letters + space above non-UTF8 bytes; empty input
    # (`encoder_hidden == 0`) emits EOS immediately at step 0 via BOS -> EOS.
    proj_b = np.full((VOCAB,), -6.0, dtype=np.float32)
    for ch in b"abcdefghijklmnopqrstuvwxyz ":
        proj_b[4 + ch] = 0.5
    proj_b[EOS_ID] = -1.0
    dec_emb[:, 6] = 0.0
    dec_emb[BOS_ID, 6] = 1.0
    proj_w[6, EOS_ID] = 3.5

    return enc_emb, enc_w, enc_b, dec_emb, mix_table, proj_w, proj_b


# ---------------------------------------------------------------------------
# Minimal deterministic Protobuf wire encoder (zero external dependencies)
# ---------------------------------------------------------------------------


def pb_varint(v: int) -> bytes:
    if v < 0:
        v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def pb_tag(field_no: int, wire_type: int) -> bytes:
    return pb_varint((field_no << 3) | wire_type)


def pb_uint(field_no: int, v: int) -> bytes:
    return pb_tag(field_no, 0) + pb_varint(v)


def pb_bytes(field_no: int, payload: bytes) -> bytes:
    return pb_tag(field_no, 2) + pb_varint(len(payload)) + payload


def pb_str(field_no: int, s: str) -> bytes:
    return pb_bytes(field_no, s.encode("utf-8"))


def pb_packed_varints(field_no: int, vals: list[int]) -> bytes:
    return pb_bytes(field_no, b"".join(pb_varint(v) for v in vals))


# ---------------------------------------------------------------------------
# ONNX IR v8 / opset 13 protobuf builders
# ---------------------------------------------------------------------------


def onnx_value_info(name: str, elem_type: int, dims: list[int]) -> bytes:
    shape_pb = b"".join(pb_bytes(1, pb_uint(1, d)) for d in dims)
    tensor_type_pb = pb_uint(1, elem_type) + pb_bytes(2, shape_pb)
    type_pb = pb_bytes(1, tensor_type_pb)
    return pb_str(1, name) + pb_bytes(2, type_pb)


def onnx_initializer_f32(name: str, arr: np.ndarray) -> bytes:
    arr_c = np.ascontiguousarray(arr, dtype="<f4")
    dims_pb = b"".join(pb_uint(1, d) for d in arr_c.shape)
    return (
        dims_pb
        + pb_uint(2, 1)  # TensorProto.FLOAT = 1
        + pb_str(8, name)
        + pb_bytes(9, arr_c.tobytes())
    )


def onnx_node(op_type: str, inputs: list[str], outputs: list[str], attrs: bytes = b"") -> bytes:
    return (
        b"".join(pb_str(1, i) for i in inputs)
        + b"".join(pb_str(2, o) for o in outputs)
        + pb_str(4, op_type)
        + attrs
    )


def onnx_attr_int(name: str, val: int) -> bytes:
    # AttributeProto: name=1, i=3, type=20 (INT=2)
    return pb_bytes(5, pb_str(1, name) + pb_uint(3, val) + pb_uint(20, 2))


def onnx_model(graph_name: str, nodes: list[bytes], inits: list[bytes], inputs: list[bytes], outputs: list[bytes]) -> bytes:
    graph = (
        b"".join(pb_bytes(1, n) for n in nodes)
        + pb_str(2, graph_name)
        + b"".join(pb_bytes(5, t) for t in inits)
        + b"".join(pb_bytes(11, vi) for vi in inputs)
        + b"".join(pb_bytes(12, vo) for vo in outputs)
    )
    opset = pb_str(1, "") + pb_uint(2, 13)
    return (
        pb_uint(1, 8)  # ir_version = 8
        + pb_str(2, "faberline-nlp training/make_translate_test_model.py")
        + pb_bytes(7, graph)
        + pb_bytes(8, opset)
    )


def make_onnx_files(enc_path: Path, dec_path: Path, weights):
    enc_emb, enc_w, enc_b, dec_emb, mix_table, proj_w, proj_b = weights

    # Encoder: Gather(enc_emb, input_ids) -> MatMul(enc_w) -> Add(enc_b)
    enc_bytes = onnx_model(
        "translate_test_tiny_encoder",
        nodes=[
            onnx_node("Gather", ["enc_emb", "input_ids"], ["emb"], onnx_attr_int("axis", 0)),
            onnx_node("MatMul", ["emb", "enc_w"], ["mm"]),
            onnx_node("Add", ["mm", "enc_b"], ["encoder_hidden"]),
        ],
        inits=[
            onnx_initializer_f32("enc_emb", enc_emb),
            onnx_initializer_f32("enc_w", enc_w),
            onnx_initializer_f32("enc_b", enc_b),
        ],
        inputs=[onnx_value_info("input_ids", 7, [1, SRC_MAX_LEN])],
        outputs=[onnx_value_info("encoder_hidden", 1, [1, SRC_MAX_LEN, D_MODEL])],
    )
    enc_path.write_bytes(enc_bytes)

    # Decoder step:
    #   dec_emb = Gather(dec_emb_table, decoder_input_ids, axis=0)      [1, TGT, D]
    #   ctx     = MatMul(mix_table, encoder_hidden)                     [1, TGT, D]
    #   comb    = Add(dec_emb, ctx)                                     [1, TGT, D]
    #   mm      = MatMul(comb, proj_w)                                  [1, TGT, V]
    #   logits  = Add(mm, proj_b)                                       [1, TGT, V]
    dec_bytes = onnx_model(
        "translate_test_tiny_decoder",
        nodes=[
            onnx_node("Gather", ["dec_emb_table", "decoder_input_ids"], ["dec_emb"], onnx_attr_int("axis", 0)),
            onnx_node("MatMul", ["mix_table", "encoder_hidden"], ["ctx"]),
            onnx_node("Add", ["dec_emb", "ctx"], ["comb"]),
            onnx_node("MatMul", ["comb", "proj_w"], ["mm"]),
            onnx_node("Add", ["mm", "proj_b"], ["logits"]),
        ],
        inits=[
            onnx_initializer_f32("dec_emb_table", dec_emb),
            onnx_initializer_f32("mix_table", mix_table),
            onnx_initializer_f32("proj_w", proj_w),
            onnx_initializer_f32("proj_b", proj_b),
        ],
        inputs=[
            onnx_value_info("decoder_input_ids", 7, [1, TGT_MAX_LEN]),
            onnx_value_info("encoder_hidden", 1, [1, SRC_MAX_LEN, D_MODEL]),
        ],
        outputs=[onnx_value_info("logits", 1, [1, TGT_MAX_LEN, VOCAB])],
    )
    dec_path.write_bytes(dec_bytes)

    try:
        import onnx  # type: ignore

        onnx.checker.check_model(onnx.load(str(enc_path)), full_check=True)
        onnx.checker.check_model(onnx.load(str(dec_path)), full_check=True)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Core ML spec v4 NeuralNetwork protobuf builders
# ---------------------------------------------------------------------------

COREML_ARRAY_INT32 = 131104
COREML_ARRAY_FLOAT32 = 65568


def coreml_feature(name: str, shape: list[int], dtype: int) -> bytes:
    arr_type = pb_packed_varints(1, shape) + pb_uint(2, dtype)
    feat_type = pb_bytes(5, arr_type)
    return pb_str(1, name) + pb_bytes(3, feat_type)


def coreml_weight_params(arr: np.ndarray) -> bytes:
    arr_c = np.ascontiguousarray(arr, dtype="<f4")
    return pb_bytes(1, arr_c.tobytes())


def coreml_layer(name: str, inputs: list[str], outputs: list[str], tag_no: int, params: bytes) -> bytes:
    return (
        pb_str(1, name)
        + b"".join(pb_str(2, i) for i in inputs)
        + b"".join(pb_str(3, o) for o in outputs)
        + pb_bytes(tag_no, params)
    )


def coreml_load_constant_nd(name: str, output: str, arr: np.ndarray) -> bytes:
    params = pb_packed_varints(1, list(arr.shape)) + pb_bytes(2, coreml_weight_params(arr))
    return coreml_layer(name, [], [output], 1070, params)


def coreml_gather_axis0(name: str, table: str, indices: str, output: str) -> bytes:
    return coreml_layer(name, [table, indices], [output], 930, b"")


def coreml_batched_matmul_const(name: str, inp: str, output: str, w: np.ndarray, b: np.ndarray) -> bytes:
    rows, cols = w.shape
    # Core ML BatchedMatMulLayerParams stores static weight matrix transposed [cols, rows].
    w_t = np.ascontiguousarray(w.T, dtype="<f4")
    params = (
        pb_uint(5, rows)
        + pb_uint(6, cols)
        + pb_uint(7, 1)
        + pb_bytes(8, coreml_weight_params(w_t))
        + pb_bytes(9, coreml_weight_params(b))
    )
    return coreml_layer(name, [inp], [output], 1045, params)


def coreml_batched_matmul_two(name: str, a: str, b: str, output: str) -> bytes:
    return coreml_layer(name, [a, b], [output], 1045, b"")


def coreml_add_broadcastable(name: str, a: str, b: str, output: str) -> bytes:
    # AddBroadcastableLayerParams `addBroadcastable = 880` in NeuralNetworkLayer oneof.
    return coreml_layer(name, [a, b], [output], 880, b"")


def coreml_model(inputs: list[bytes], outputs: list[bytes], layers: list[bytes]) -> bytes:
    desc = b"".join(pb_bytes(1, i) for i in inputs) + b"".join(pb_bytes(10, o) for o in outputs)
    nn = (
        b"".join(pb_bytes(1, lyr) for lyr in layers)
        + pb_uint(5, 1)  # arrayInputShapeMapping = EXACT_ARRAY_MAPPING
        + pb_uint(6, 1)
    )
    return pb_uint(1, 4) + pb_bytes(2, desc) + pb_bytes(500, nn)


def make_coreml_files(enc_path: Path, dec_path: Path, weights):
    enc_emb, enc_w, enc_b, dec_emb, mix_table, proj_w, proj_b = weights

    enc_ml = coreml_model(
        inputs=[coreml_feature("input_ids", [1, SRC_MAX_LEN], COREML_ARRAY_INT32)],
        outputs=[coreml_feature("encoder_hidden", [1, SRC_MAX_LEN, D_MODEL], COREML_ARRAY_FLOAT32)],
        layers=[
            coreml_load_constant_nd("enc_emb_const", "enc_table", enc_emb),
            coreml_gather_axis0("enc_gather", "enc_table", "input_ids", "emb"),
            coreml_batched_matmul_const("enc_dense", "emb", "encoder_hidden", enc_w, enc_b),
        ],
    )
    enc_path.write_bytes(enc_ml)

    dec_ml = coreml_model(
        inputs=[
            coreml_feature("decoder_input_ids", [1, TGT_MAX_LEN], COREML_ARRAY_INT32),
            coreml_feature("encoder_hidden", [1, SRC_MAX_LEN, D_MODEL], COREML_ARRAY_FLOAT32),
        ],
        outputs=[coreml_feature("logits", [1, TGT_MAX_LEN, VOCAB], COREML_ARRAY_FLOAT32)],
        layers=[
            coreml_load_constant_nd("dec_emb_const", "dec_table", dec_emb),
            coreml_gather_axis0("dec_gather", "dec_table", "decoder_input_ids", "dec_emb"),
            coreml_load_constant_nd("mix_const", "mix_table", mix_table),
            coreml_batched_matmul_two("cross_mix", "mix_table", "encoder_hidden", "ctx"),
            coreml_add_broadcastable("combine", "dec_emb", "ctx", "comb"),
            coreml_batched_matmul_const("proj", "comb", "logits", proj_w, proj_b),
        ],
    )
    dec_path.write_bytes(dec_ml)


# ---------------------------------------------------------------------------
# Reference NumPy execution & expected.json
# ---------------------------------------------------------------------------


def numpy_encode(ids: list[int], weights) -> np.ndarray:
    enc_emb, enc_w, enc_b, *_ = weights
    padded = np.full(SRC_MAX_LEN, PAD_ID, dtype=np.int64)
    padded[: len(ids)] = np.asarray(ids, dtype=np.int64)
    hidden = (enc_emb[padded] @ enc_w + enc_b).astype(np.float32)
    hidden[len(ids) :] = 0.0
    return hidden  # [SRC_MAX_LEN, D_MODEL]


def numpy_decode_step(enc_hidden: np.ndarray, dec_ids: list[int], weights) -> np.ndarray:
    _, _, _, dec_emb, mix_table, proj_w, proj_b = weights
    padded_dec = np.full(TGT_MAX_LEN, PAD_ID, dtype=np.int64)
    padded_dec[: len(dec_ids)] = np.asarray(dec_ids, dtype=np.int64)
    d_emb = dec_emb[padded_dec]  # [TGT_MAX_LEN, D_MODEL]
    ctx = (mix_table[0] @ enc_hidden).astype(np.float32)  # [TGT_MAX_LEN, D_MODEL]
    comb = (d_emb + ctx).astype(np.float32)
    logits = (comb @ proj_w + proj_b).astype(np.float32)  # [TGT_MAX_LEN, VOCAB]
    return logits[len(dec_ids) - 1]


def numpy_greedy_decode(enc_hidden: np.ndarray, weights) -> tuple[list[int], list[float]]:
    dec_ids = [BOS_ID]
    first_step_logits = None
    out_tokens: list[int] = []
    for _ in range(TGT_MAX_LEN - 1):
        step_logits = numpy_decode_step(enc_hidden, dec_ids, weights)
        if first_step_logits is None:
            first_step_logits = step_logits
        best = int(np.argmax(step_logits))
        if best == EOS_ID:
            break
        out_tokens.append(best)
        dec_ids.append(best)
    assert first_step_logits is not None
    top8 = [round(float(x), 6) for x in first_step_logits[:8]]
    return out_tokens, top8


def make_expected(weights) -> dict:
    fixtures = []
    for fx_id, text in FIXTURES:
        ids = encode_bytes_bpe(text)
        if len(ids) > SRC_MAX_LEN:
            sys.exit(f"{fx_id}: {len(ids)} tokens exceed src_max_len {SRC_MAX_LEN}")
        enc_hidden = numpy_encode(ids, weights)
        out_tokens, step0_head = numpy_greedy_decode(enc_hidden, weights)
        decoded_text = decode_bytes_bpe(out_tokens)
        enc_head = [round(float(x), 6) for x in enc_hidden[0]]
        fixtures.append(
            {
                "id": fx_id,
                "text": text,
                "src_ids": ids,
                "enc_pos0": enc_head,
                "step0_logits_head": step0_head,
                "greedy_ids": out_tokens,
                "greedy_text": decoded_text,
            }
        )
    return {
        "schema": "nlp.models.seq2seq.expected.v1",
        "model": MODEL_ID,
        "src_max_len": SRC_MAX_LEN,
        "tgt_max_len": TGT_MAX_LEN,
        "d_model": D_MODEL,
        "vocab": VOCAB,
        "fixtures": fixtures,
    }


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    make_tokenizer_json(MODEL_DIR / "tokenizer.json")
    weights = make_weights()
    make_onnx_files(MODEL_DIR / "encoder.onnx", MODEL_DIR / "decoder_step.onnx", weights)
    make_coreml_files(MODEL_DIR / "encoder.mlmodel", MODEL_DIR / "decoder_step.mlmodel", weights)
    write_json(MODEL_DIR / "expected.json", make_expected(weights))

    file_names = [
        "encoder.onnx",
        "decoder_step.onnx",
        "encoder.mlmodel",
        "decoder_step.mlmodel",
        "tokenizer.json",
        "expected.json",
    ]
    files = []
    total = 0
    for name in file_names:
        p = MODEL_DIR / name
        sz = p.stat().st_size
        total += sz
        digest = sha256_of(p)
        files.append({"path": name, "sha256": digest, "bytes": sz})
        print(f"{name:22s}  {sz:6d} bytes  sha256 {digest}")
    if total > SIZE_LIMIT:
        sys.exit(f"model files total {total} bytes, over limit {SIZE_LIMIT}")
    print(f"total {total} bytes")

    manifest = {
        "schema": "nlp.models.seq2seq.v1",
        "models": [
            {
                "id": MODEL_ID,
                "licence": LICENCE,
                "dir": MODEL_ID,
                "src_max_len": SRC_MAX_LEN,
                "tgt_max_len": TGT_MAX_LEN,
                "d_model": D_MODEL,
                "vocab": VOCAB,
                "pad_id": PAD_ID,
                "bos_id": BOS_ID,
                "eos_id": EOS_ID,
                "supported_variants": SUPPORTED_VARIANTS,
                "encoder_onnx": "encoder.onnx",
                "decoder_onnx": "decoder_step.onnx",
                "tokenizer": "tokenizer.json",
                "expected": "expected.json",
                "coreml": {
                    "encoder": "encoder.mlmodel",
                    "decoder": "decoder_step.mlmodel",
                },
                "files": files,
                "receipt": None,
            }
        ],
    }
    write_json(OUT_DIR / "manifest.json", manifest)
    print(f"wrote {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
