#!/usr/bin/env python3
"""Generates the tiny test model under models/test-tiny/ and models/manifest.json.

Licence: CC0-1.0. The weights are random (numpy, fixed seed), so the model has no meaning.
It exists to test the model runtime (manifest, hashes, tokenizer offsets, tract and Core ML backends).

Run by hand, not by cargo or CI:
    python3 -m venv training/.venv
    training/.venv/bin/pip install numpy onnx tokenizers coremltools
    training/.venv/bin/python training/make_test_model.py

Packages: numpy, onnx, tokenizers, coremltools. No torch.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "models"
MODEL_DIR = OUT_DIR / "test-tiny"

MODEL_ID = "test-tiny"
LICENCE = "CC0-1.0"
SEED = 20261110
MAX_LEN = 64
PAD_ID = 0
EMBED_DIM = 8
LABELS = 5
VOCAB = 2 + 256
SIZE_LIMIT = 100_000

# Must equal FIXTURES in tests/doc_model.rs (the Rust test checks this against expected.json).
FIXTURES = [
    ("fx01", "她推開門，屋裡很安靜。"),
    ("fx02", "She opened the door and the room was quiet."),
    ("fx03", "她說 hello，然後笑了。"),
    ("fx04", "本文提出一個簡單的方法。"),
    ("fx05", "We propose a simple method."),
    ("fx06", "我們使用 BERT 模型。"),
    ("fx07", "哈哈，今天好累😋"),
    ("fx08", "lol that was so good!"),
    ("fx09", "明天 meeting 幾點？"),
    ("fx10", ""),
    ("fx11", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab"),
    ("fx12", "好"),
]


def byte_symbols():
    """The GPT-2 byte-to-printable-char table used by the ByteLevel pre-tokenizer."""
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


def make_tokenizer(path):
    symbols = byte_symbols()
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for b in range(256):
        vocab[symbols[b]] = 2 + b
    assert len(vocab) == VOCAB
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False)
    tok.decoder = decoders.ByteLevel()
    tok.save(str(path), pretty=True)
    return Tokenizer.from_file(str(path))


def make_weights():
    rng = np.random.default_rng(SEED)
    embedding = rng.normal(0.0, 0.5, size=(VOCAB, EMBED_DIM)).astype(np.float32)
    dense_w = rng.normal(0.0, 0.5, size=(EMBED_DIM, LABELS)).astype(np.float32)
    dense_b = rng.normal(0.0, 0.5, size=(LABELS,)).astype(np.float32)
    return embedding, dense_w, dense_b


def make_onnx(path, embedding, dense_w, dense_b):
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, MAX_LEN])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, MAX_LEN, LABELS])
    initializers = [
        numpy_helper.from_array(embedding, name="embedding"),
        numpy_helper.from_array(dense_w, name="dense_w"),
        numpy_helper.from_array(dense_b, name="dense_b"),
    ]
    nodes = [
        helper.make_node("Gather", ["embedding", "input_ids"], ["emb"], axis=0),
        helper.make_node("MatMul", ["emb", "dense_w"], ["mm"]),
        helper.make_node("Add", ["mm", "dense_b"], ["logits"]),
    ]
    graph = helper.make_graph(nodes, "test_tiny", [input_ids], [logits], initializer=initializers)
    model = helper.make_model(
        graph,
        producer_name="faberline-nlp training/make_test_model.py",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))


def make_coreml(path, embedding, dense_w, dense_b):
    """The same graph as the ONNX model, as a Core ML neural network (no torch)."""
    import coremltools as ct
    from coremltools.models import datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder
    from coremltools.proto import FeatureTypes_pb2

    builder = NeuralNetworkBuilder(
        [("input_ids", datatypes.Array(MAX_LEN))],
        [("logits", datatypes.Array(1, MAX_LEN, LABELS))],
        disable_rank5_shape_mapping=True,
    )
    builder.add_load_constant_nd("embedding", "table", embedding, shape=list(embedding.shape))
    builder.add_gather("gather", ["table", "input_ids"], "emb", axis=0)
    builder.add_batched_mat_mul(
        "dense",
        ["emb"],
        "logits",
        weight_matrix_rows=EMBED_DIM,
        weight_matrix_columns=LABELS,
        W=dense_w,
        bias=dense_b,
    )
    spec = builder.spec
    ids = spec.description.input[0].type.multiArrayType
    ids.dataType = FeatureTypes_pb2.ArrayFeatureType.INT32
    del ids.shape[:]
    ids.shape.extend([1, MAX_LEN])
    logits = spec.description.output[0].type.multiArrayType
    logits.dataType = FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    del logits.shape[:]
    logits.shape.extend([1, MAX_LEN, LABELS])
    ct.models.MLModel(spec).save(str(path))


def numpy_logits(ids, embedding, dense_w, dense_b):
    """Logits for the real positions only, computed without tract."""
    padded = np.full(MAX_LEN, PAD_ID, dtype=np.int64)
    padded[: len(ids)] = np.asarray(ids, dtype=np.int64)
    hidden = embedding[padded]
    out = (hidden @ dense_w + dense_b).astype(np.float32)
    return out[: len(ids)]


def make_expected(tok, weights):
    embedding, dense_w, dense_b = weights
    fixtures = []
    for fx_id, text in FIXTURES:
        enc = tok.encode(text)
        ids = [int(i) for i in enc.ids]
        if len(ids) > MAX_LEN:
            sys.exit(f"{fx_id}: {len(ids)} tokens exceed max_len {MAX_LEN}")
        spans = [[int(s), int(e)] for s, e in enc.offsets]
        for s, e in spans:
            if not (0 <= s <= e <= len(text)):
                sys.exit(f"{fx_id}: span {s},{e} outside the text")
        logits = numpy_logits(ids, embedding, dense_w, dense_b)
        rounded = [[round(float(x), 6) for x in row] for row in logits]
        argmax = [int(np.argmax(row)) for row in logits]
        fixtures.append(
            {
                "id": fx_id,
                "text": text,
                "ids": ids,
                "spans": spans,
                "logits": rounded,
                "argmax": argmax,
            }
        )
    return {
        "schema": "nlp.models.expected.v1",
        "model": MODEL_ID,
        "max_len": MAX_LEN,
        "labels": LABELS,
        "fixtures": fixtures,
    }


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tok = make_tokenizer(MODEL_DIR / "tokenizer.json")
    weights = make_weights()
    make_onnx(MODEL_DIR / "model.onnx", *weights)
    make_coreml(MODEL_DIR / "model.mlmodel", *weights)
    write_json(MODEL_DIR / "expected.json", make_expected(tok, weights))

    files = []
    total = 0
    for name in ["model.onnx", "model.mlmodel", "tokenizer.json", "expected.json"]:
        path = MODEL_DIR / name
        size = path.stat().st_size
        total += size
        files.append({"path": name, "sha256": sha256_of(path), "bytes": size})
        print(f"{name}  {size} bytes  sha256 {files[-1]['sha256']}")
    if total > SIZE_LIMIT:
        sys.exit(f"model files total {total} bytes, over the {SIZE_LIMIT} byte limit")
    print(f"total {total} bytes")

    manifest = {
        "schema": "nlp.models.v1",
        "models": [
            {
                "id": MODEL_ID,
                "layers": ["test"],
                "licence": LICENCE,
                "dir": MODEL_ID,
                "max_len": MAX_LEN,
                "pad_id": PAD_ID,
                "model": "model.onnx",
                "tokenizer": "tokenizer.json",
                "expected": "expected.json",
                "coreml": {"model": "model.mlmodel", "vocab": VOCAB},
                "files": files,
                "receipt": None,
            }
        ],
    }
    write_json(OUT_DIR / "manifest.json", manifest)
    print(f"wrote {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
