#!/usr/bin/env python3
"""Dual-export `doc-syntax-v1` from float32 `.npz` weights to ONNX (opset 13) + Core ML (`.mlmodel`).

Enforces all 5 runtime invariants of `src/doc/model/` and `src/doc/syntax.rs`:
  1. Single-file Core ML `.mlmodel` (`neuralnetwork` float32, NOT `.mlpackage`) AND `model.onnx`
     (opset 13, `ir_version = 8`) built from the exact same float32 weights.
  2. Single input `"input_ids"` of fixed shape `[1, 256]` (`INT64` in ONNX, `INT32` in Core ML),
     `pad_id = 0`, `[UNK] = 1`, `[CLS] = 2`, `[SEP] = 3`, with internal pad-mask attention bias.
  3. **EXACTLY ONE `Gather` node** in `model.onnx` (`src/doc/model/onnx.rs` `embedding_rows()`),
     with fused positional + token-type constant `[1, 256, 384]` added via `Add`.
  4. Packed output `"logits"` of shape `[1, 256, 421]` (`FLOAT32`):
     `upos (17) || arc (256) || r_dep (37) || r_head (37) || r_bi_dep (37) || r_bi_head (37)`.
  5. Cross-backend numerical parity `max |onnx - coreml| <= 1e-4` across multilingual fixtures
     (`zh-TW`, `zh-CN`, `en`, `mixed-TW`, `mixed-CN`), empty/single-char edge cases, and all-pad.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
import onnxruntime as ort
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.proto import FeatureTypes_pb2
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_ID = "doc-syntax-v1"
LICENCE = "MIT"
MAX_LEN = 256
PAD_ID = 0
VOCAB = 32768
D_MODEL = 384
N_HEADS = 12
D_HEAD = D_MODEL // N_HEADS
D_FF = 1536
D_HEAD_MLP = 192
LABELS = 17 + MAX_LEN + 4 * 37  # 421
LN_EPS = 1e-5

# Multilingual verification fixtures covering zh-TW, zh-CN, en, mixed-TW, mixed-CN, and edge cases.
FIXTURES: list[tuple[str, str, str]] = [
    ("fx01", "zh-TW", "她推開門，屋裡很安靜。"),
    ("fx02", "en", "She opened the door and the room was quiet."),
    ("fx03", "mixed-TW", "她說 hello，然後笑了。"),
    ("fx04", "zh-TW", "本文提出一個簡單的方法。"),
    ("fx05", "en", "We propose a simple method."),
    ("fx06", "mixed-TW", "我們使用 BERT 模型。"),
    ("fx07", "zh-TW", "哈哈，今天好累。"),
    ("fx08", "en", "My neighbor fixed the wooden gate this morning."),
    ("fx09", "mixed-TW", "明天 meeting 幾點開始？"),
    ("fx10", "en", ""),
    ("fx11", "en", "The parser didn't fail on Alice's test suite."),
    ("fx12", "zh-TW", "好"),
    ("fx13", "zh-CN", "研究团队通过公开网络下载了两组数据库。"),
    ("fx14", "zh-CN", "小王在地铁站买了一份外卖和咖啡。"),
    ("fx15", "zh-CN", "老张把木门修好之后，院子里非常安静。"),
    ("fx16", "mixed-CN", "我们通过 API 服务器加载这个 JSON 文件。"),
]


def build_onnx_model(weights: dict[str, np.ndarray], out_path: Path) -> dict:
    num_layers = int(weights["num_layers"][0])
    inits: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []

    def add_const(name: str, arr: np.ndarray) -> str:
        inits.append(numpy_helper.from_array(arr, name=name))
        return name

    # Shared scalar/shape constants
    c_zero = add_const("c_zero_f32", np.array(0.0, dtype=np.float32))
    c_one = add_const("c_one_f32", np.array(1.0, dtype=np.float32))
    c_half = add_const("c_half_f32", np.array(0.5, dtype=np.float32))
    c_inv_sqrt2 = add_const(
        "c_inv_sqrt2_f32", np.array(1.0 / math.sqrt(2.0), dtype=np.float32)
    )
    c_mask_scale = add_const("c_mask_scale_f32", np.array(10000.0, dtype=np.float32))
    c_eps = add_const("c_ln_eps_f32", np.array(LN_EPS, dtype=np.float32))
    c_arc_scale = add_const(
        "c_arc_scale_f32", np.array(1.0 / math.sqrt(D_HEAD_MLP), dtype=np.float32)
    )

    sh_bias_4d = add_const("sh_bias_4d", np.array([1, 1, 1, MAX_LEN], dtype=np.int64))
    sh_valid_3d = add_const("sh_valid_3d", np.array([1, MAX_LEN, 1], dtype=np.int64))
    sh_4d = add_const("sh_mha_4d", np.array([1, MAX_LEN, N_HEADS, D_HEAD], dtype=np.int64))
    sh_3d = add_const("sh_mha_3d", np.array([1, MAX_LEN, D_MODEL], dtype=np.int64))

    def add_ln(x_in: str, w: np.ndarray, b: np.ndarray, pfx: str) -> str:
        wn = add_const(f"{pfx}_w", w.astype(np.float32))
        bn = add_const(f"{pfx}_b", b.astype(np.float32))
        nodes.append(helper.make_node("ReduceMean", [x_in], [f"{pfx}_mean"], axes=[-1], keepdims=1))
        nodes.append(helper.make_node("Sub", [x_in, f"{pfx}_mean"], [f"{pfx}_centered"]))
        nodes.append(
            helper.make_node("Mul", [f"{pfx}_centered", f"{pfx}_centered"], [f"{pfx}_sq"])
        )
        nodes.append(
            helper.make_node("ReduceMean", [f"{pfx}_sq"], [f"{pfx}_var"], axes=[-1], keepdims=1)
        )
        nodes.append(helper.make_node("Add", [f"{pfx}_var", c_eps], [f"{pfx}_var_eps"]))
        nodes.append(helper.make_node("Sqrt", [f"{pfx}_var_eps"], [f"{pfx}_std"]))
        nodes.append(helper.make_node("Div", [f"{pfx}_centered", f"{pfx}_std"], [f"{pfx}_norm"]))
        nodes.append(helper.make_node("Mul", [f"{pfx}_norm", wn], [f"{pfx}_scaled"]))
        nodes.append(helper.make_node("Add", [f"{pfx}_scaled", bn], [f"{pfx}_out"]))
        return f"{pfx}_out"

    def add_gelu(x_in: str, pfx: str) -> str:
        nodes.append(helper.make_node("Mul", [x_in, c_inv_sqrt2], [f"{pfx}_scaled"]))
        nodes.append(helper.make_node("Erf", [f"{pfx}_scaled"], [f"{pfx}_erf"]))
        nodes.append(helper.make_node("Add", [f"{pfx}_erf", c_one], [f"{pfx}_one_p"]))
        nodes.append(helper.make_node("Mul", [x_in, f"{pfx}_one_p"], [f"{pfx}_prod"]))
        nodes.append(helper.make_node("Mul", [f"{pfx}_prod", c_half], [f"{pfx}_out"]))
        return f"{pfx}_out"

    def add_linear(x_in: str, w_out_in: np.ndarray, b: np.ndarray, pfx: str) -> str:
        wt = add_const(f"{pfx}_wt", np.ascontiguousarray(w_out_in.T.astype(np.float32)))
        bn = add_const(f"{pfx}_b", b.astype(np.float32))
        nodes.append(helper.make_node("MatMul", [x_in, wt], [f"{pfx}_mm"]))
        nodes.append(helper.make_node("Add", [f"{pfx}_mm", bn], [f"{pfx}_out"]))
        return f"{pfx}_out"

    # 1. Internal attention mask bias from `input_ids`: (clip(float(ids), 0, 1) * 10000 - 10000)
    nodes.append(helper.make_node("Cast", ["input_ids"], ["ids_f32"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Clip", ["ids_f32", c_zero, c_one], ["valid_f32"]))
    nodes.append(helper.make_node("Reshape", ["valid_f32", sh_valid_3d], ["valid_3d"]))
    nodes.append(helper.make_node("Mul", ["valid_f32", c_mask_scale], ["valid_scaled"]))
    nodes.append(helper.make_node("Sub", ["valid_scaled", c_mask_scale], ["attn_bias_1d"]))
    nodes.append(helper.make_node("Reshape", ["attn_bias_1d", sh_bias_4d], ["attn_bias"]))

    # 2. Single Gather for token embeddings + constant [1, 256, 384] pos_type_bias
    w_emb_name = add_const("word_embeddings", weights["word_embeddings"].astype(np.float32))
    pos_name = add_const("pos_type_bias", weights["pos_type_bias"].astype(np.float32))
    nodes.append(helper.make_node("Gather", [w_emb_name, "input_ids"], ["tok_emb"], axis=0))
    nodes.append(helper.make_node("Add", ["tok_emb", pos_name], ["emb_plus_pos"]))
    cur = add_ln("emb_plus_pos", weights["emb_ln_w"], weights["emb_ln_b"], "emb_ln")

    # 3. Transformer encoder layers
    q_scale = 1.0 / math.sqrt(D_HEAD)
    for i in range(num_layers):
        p = f"l{i}"
        qw_s = (weights[f"{p}_q_w"] * q_scale).astype(np.float32)
        qb_s = (weights[f"{p}_q_b"] * q_scale).astype(np.float32)
        q_lin = add_linear(cur, qw_s, qb_s, f"{p}_q")
        k_lin = add_linear(cur, weights[f"{p}_k_w"], weights[f"{p}_k_b"], f"{p}_k")
        v_lin = add_linear(cur, weights[f"{p}_v_w"], weights[f"{p}_v_b"], f"{p}_v")

        nodes.append(helper.make_node("Reshape", [q_lin, sh_4d], [f"{p}_q4"]))
        nodes.append(helper.make_node("Transpose", [f"{p}_q4"], [f"{p}_qh"], perm=[0, 2, 1, 3]))
        nodes.append(helper.make_node("Reshape", [k_lin, sh_4d], [f"{p}_k4"]))
        nodes.append(
            helper.make_node("Transpose", [f"{p}_k4"], [f"{p}_kh_t"], perm=[0, 2, 3, 1])
        )
        nodes.append(helper.make_node("Reshape", [v_lin, sh_4d], [f"{p}_v4"]))
        nodes.append(helper.make_node("Transpose", [f"{p}_v4"], [f"{p}_vh"], perm=[0, 2, 1, 3]))

        nodes.append(helper.make_node("MatMul", [f"{p}_qh", f"{p}_kh_t"], [f"{p}_qk"]))
        nodes.append(helper.make_node("Add", [f"{p}_qk", "attn_bias"], [f"{p}_scores"]))
        nodes.append(helper.make_node("Softmax", [f"{p}_scores"], [f"{p}_probs"], axis=-1))
        nodes.append(helper.make_node("MatMul", [f"{p}_probs", f"{p}_vh"], [f"{p}_ctx4"]))
        nodes.append(
            helper.make_node("Transpose", [f"{p}_ctx4"], [f"{p}_ctx_t"], perm=[0, 2, 1, 3])
        )
        nodes.append(helper.make_node("Reshape", [f"{p}_ctx_t", sh_3d], [f"{p}_ctx"]))

        o_lin = add_linear(f"{p}_ctx", weights[f"{p}_o_w"], weights[f"{p}_o_b"], f"{p}_o")
        nodes.append(helper.make_node("Add", [cur, o_lin], [f"{p}_res1"]))
        ln1 = add_ln(f"{p}_res1", weights[f"{p}_ln1_w"], weights[f"{p}_ln1_b"], f"{p}_ln1")

        fc1 = add_linear(ln1, weights[f"{p}_fc1_w"], weights[f"{p}_fc1_b"], f"{p}_fc1")
        act = add_gelu(fc1, f"{p}_gelu")
        fc2 = add_linear(act, weights[f"{p}_fc2_w"], weights[f"{p}_fc2_b"], f"{p}_fc2")
        nodes.append(helper.make_node("Add", [ln1, fc2], [f"{p}_res2"]))
        cur = add_ln(f"{p}_res2", weights[f"{p}_ln2_w"], weights[f"{p}_ln2_b"], f"{p}_ln2")

    # Zero out padded positions (`valid_3d == 0`) while keeping real tokens (`valid_3d == 1`) exact
    nodes.append(helper.make_node("Mul", [cur, "valid_3d"], ["cur_valid"]))
    cur = "cur_valid"

    # 4. Packed [1, 256, 421] Syntax + UPOS heads
    u_h = add_gelu(
        add_linear(cur, weights["upos_fc1_w"], weights["upos_fc1_b"], "upos_fc1"), "upos_act"
    )
    upos_logits = add_linear(u_h, weights["upos_fc2_w"], weights["upos_fc2_b"], "upos_fc2")

    a_dep = add_gelu(
        add_linear(cur, weights["arc_dep_w"], weights["arc_dep_b"], "arc_dep"), "arc_dep_act"
    )
    a_head_raw = add_gelu(
        add_linear(cur, weights["arc_head_w"], weights["arc_head_b"], "arc_head"), "arc_head_act"
    )
    nodes.append(helper.make_node("Mul", [a_head_raw, "valid_3d"], ["a_head"]))
    nodes.append(helper.make_node("Mul", [a_dep, c_arc_scale], ["a_dep_scaled"]))
    nodes.append(helper.make_node("Transpose", ["a_head"], ["a_head_t"], perm=[0, 2, 1]))
    nodes.append(helper.make_node("MatMul", ["a_dep_scaled", "a_head_t"], ["arc_logits"]))

    rh_dep = add_gelu(
        add_linear(cur, weights["rel_dep_in_w"], weights["rel_dep_in_b"], "rel_dep_in"),
        "rel_dep_act",
    )
    rh_head = add_gelu(
        add_linear(cur, weights["rel_head_in_w"], weights["rel_head_in_b"], "rel_head_in"),
        "rel_head_act",
    )
    r_dep = add_linear(rh_dep, weights["r_dep_w"], weights["r_dep_b"], "r_dep")
    r_head = add_linear(rh_head, weights["r_head_w"], weights["r_head_b"], "r_head")
    r_bi_dep = add_linear(rh_dep, weights["r_bi_dep_w"], weights["r_bi_dep_b"], "r_bi_dep")
    r_bi_head = add_linear(rh_head, weights["r_bi_head_w"], weights["r_bi_head_b"], "r_bi_head")

    nodes.append(
        helper.make_node(
            "Concat",
            [upos_logits, "arc_logits", r_dep, r_head, r_bi_dep, r_bi_head],
            ["logits"],
            axis=-1,
        )
    )

    inp_vi = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, MAX_LEN])
    out_vi = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, MAX_LEN, LABELS])
    graph = helper.make_graph(nodes, "doc_syntax_v1", [inp_vi], [out_vi], initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="faberline-nlp training/syntax/export_dual.py",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)

    gather_count = sum(1 for n in model.graph.node if n.op_type == "Gather")
    assert gather_count == 1, f"expected Gather count == 1, got {gather_count}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return {
        "ir_version": int(model.ir_version),
        "opset": int(model.opset_import[0].version),
        "gather_nodes": gather_count,
        "total_nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }


def build_coreml_model(weights: dict[str, np.ndarray], out_path: Path) -> None:
    num_layers = int(weights["num_layers"][0])
    builder = NeuralNetworkBuilder(
        [("input_ids", datatypes.Array(MAX_LEN))],
        [("logits", datatypes.Array(1, MAX_LEN, LABELS))],
        disable_rank5_shape_mapping=True,
    )

    def add_linear(x_in: str, w_out_in: np.ndarray, b: np.ndarray, pfx: str) -> str:
        wt = np.ascontiguousarray(w_out_in.T.astype(np.float32))
        builder.add_batched_mat_mul(
            name=f"{pfx}_bmm",
            input_names=[x_in],
            output_name=f"{pfx}_out",
            weight_matrix_rows=wt.shape[0],
            weight_matrix_columns=wt.shape[1],
            W=wt,
            bias=b.astype(np.float32),
        )
        return f"{pfx}_out"

    def add_ln(x_in: str, w: np.ndarray, b: np.ndarray, pfx: str) -> str:
        # Explicit ReduceMean -> Sub -> Square -> ReduceMean -> Add(eps) -> Sqrt -> Div -> Mul -> Add
        # so Core ML executes the exact same float32 formula as ONNX opset 13!
        builder.add_reduce_mean(
            name=f"{pfx}_mean",
            input_name=x_in,
            output_name=f"{pfx}_mean",
            axes=[-1],
            keepdims=True,
        )
        builder.add_broadcast_to_dynamic(
            name=f"{pfx}_mean_bc",
            input_names=[f"{pfx}_mean", x_in],
            output_name=f"{pfx}_mean_b",
        ) if False else None
        # NeuralNetworkBuilder `add_subtract_broadcastable` / `add_divide_broadcastable` / `add_multiply_broadcastable` / `add_add_broadcastable`:
        builder.add_subtract_broadcastable(
            name=f"{pfx}_sub",
            input_names=[x_in, f"{pfx}_mean"],
            output_name=f"{pfx}_centered",
        )
        builder.add_multiply_broadcastable(
            name=f"{pfx}_sq",
            input_names=[f"{pfx}_centered", f"{pfx}_centered"],
            output_name=f"{pfx}_sq",
        )
        builder.add_reduce_mean(
            name=f"{pfx}_var",
            input_name=f"{pfx}_sq",
            output_name=f"{pfx}_var",
            axes=[-1],
            keepdims=True,
        )
        builder.add_activation(
            name=f"{pfx}_var_eps",
            non_linearity="LINEAR",
            input_name=f"{pfx}_var",
            output_name=f"{pfx}_var_eps",
            params=[1.0, LN_EPS],
        )
        builder.add_unary(
            name=f"{pfx}_std",
            input_name=f"{pfx}_var_eps",
            output_name=f"{pfx}_std",
            mode="sqrt",
        )
        builder.add_divide_broadcastable(
            name=f"{pfx}_norm",
            input_names=[f"{pfx}_centered", f"{pfx}_std"],
            output_name=f"{pfx}_norm",
        )
        builder.add_load_constant_nd(
            name=f"{pfx}_w_c",
            output_name=f"{pfx}_w",
            constant_value=w.astype(np.float32).reshape(1, 1, -1),
            shape=[1, 1, w.shape[0]],
        )
        builder.add_load_constant_nd(
            name=f"{pfx}_b_c",
            output_name=f"{pfx}_b",
            constant_value=b.astype(np.float32).reshape(1, 1, -1),
            shape=[1, 1, b.shape[0]],
        )
        builder.add_multiply_broadcastable(
            name=f"{pfx}_scaled",
            input_names=[f"{pfx}_norm", f"{pfx}_w"],
            output_name=f"{pfx}_scaled",
        )
        builder.add_add_broadcastable(
            name=f"{pfx}_out",
            input_names=[f"{pfx}_scaled", f"{pfx}_b"],
            output_name=f"{pfx}_out",
        )
        return f"{pfx}_out"

    inv_sqrt2_f32 = float(np.float32(1.0 / math.sqrt(2.0)))

    def add_gelu(x_in: str, pfx: str) -> str:
        builder.add_activation(
            name=f"{pfx}_scaled",
            non_linearity="LINEAR",
            input_name=x_in,
            output_name=f"{pfx}_scaled",
            params=[inv_sqrt2_f32, 0.0],
        )
        builder.add_erf(
            name=f"{pfx}_erf",
            input_name=f"{pfx}_scaled",
            output_name=f"{pfx}_erf",
        )
        builder.add_activation(
            name=f"{pfx}_one_p",
            non_linearity="LINEAR",
            input_name=f"{pfx}_erf",
            output_name=f"{pfx}_one_p",
            params=[1.0, 1.0],
        )
        builder.add_multiply_broadcastable(
            name=f"{pfx}_prod",
            input_names=[x_in, f"{pfx}_one_p"],
            output_name=f"{pfx}_prod",
        )
        builder.add_activation(
            name=f"{pfx}_out",
            non_linearity="LINEAR",
            input_name=f"{pfx}_prod",
            output_name=f"{pfx}_out",
            params=[0.5, 0.0],
        )
        return f"{pfx}_out"

    # 1. Internal attention mask bias from `input_ids`
    builder.add_clip(
        name="valid_clip",
        input_name="input_ids",
        output_name="valid_f32",
        min_value=0.0,
        max_value=1.0,
    )
    builder.add_reshape_static(
        name="valid_3d_reshape",
        input_name="valid_f32",
        output_name="valid_3d",
        output_shape=[1, MAX_LEN, 1],
    )
    builder.add_activation(
        name="mask_linear",
        non_linearity="LINEAR",
        input_name="valid_f32",
        output_name="attn_bias_1d",
        params=[10000.0, -10000.0],
    )
    builder.add_reshape_static(
        name="mask_reshape",
        input_name="attn_bias_1d",
        output_name="attn_bias",
        output_shape=[1, 1, 1, MAX_LEN],
    )

    # 2. Embedding Gather + constant [1, 256, 384] pos_type_bias
    w_emb = weights["word_embeddings"].astype(np.float32)
    pos_b = weights["pos_type_bias"].astype(np.float32)
    builder.add_load_constant_nd(
        name="word_embeddings_c",
        output_name="word_embeddings",
        constant_value=w_emb,
        shape=list(w_emb.shape),
    )
    builder.add_gather(
        name="tok_gather",
        input_names=["word_embeddings", "input_ids"],
        output_name="tok_emb",
        axis=0,
    )
    builder.add_load_constant_nd(
        name="pos_type_bias_c",
        output_name="pos_type_bias",
        constant_value=pos_b,
        shape=[1, MAX_LEN, D_MODEL],
    )
    builder.add_add_broadcastable(
        name="emb_add_pos",
        input_names=["tok_emb", "pos_type_bias"],
        output_name="emb_plus_pos",
    )
    cur = add_ln("emb_plus_pos", weights["emb_ln_w"], weights["emb_ln_b"], "emb_ln")

    # 3. Transformer encoder layers
    q_scale = 1.0 / math.sqrt(D_HEAD)
    for i in range(num_layers):
        p = f"l{i}"
        qw_s = (weights[f"{p}_q_w"] * q_scale).astype(np.float32)
        qb_s = (weights[f"{p}_q_b"] * q_scale).astype(np.float32)
        q_lin = add_linear(cur, qw_s, qb_s, f"{p}_q")
        k_lin = add_linear(cur, weights[f"{p}_k_w"], weights[f"{p}_k_b"], f"{p}_k")
        v_lin = add_linear(cur, weights[f"{p}_v_w"], weights[f"{p}_v_b"], f"{p}_v")

        builder.add_reshape_static(f"{p}_q4_r", q_lin, f"{p}_q4", [1, MAX_LEN, N_HEADS, D_HEAD])
        builder.add_transpose(f"{p}_qh_t", [0, 2, 1, 3], f"{p}_q4", f"{p}_qh")

        builder.add_reshape_static(f"{p}_k4_r", k_lin, f"{p}_k4", [1, MAX_LEN, N_HEADS, D_HEAD])
        builder.add_transpose(f"{p}_kh_t", [0, 2, 1, 3], f"{p}_k4", f"{p}_kh")

        builder.add_reshape_static(f"{p}_v4_r", v_lin, f"{p}_v4", [1, MAX_LEN, N_HEADS, D_HEAD])
        builder.add_transpose(f"{p}_vh_t", [0, 2, 1, 3], f"{p}_v4", f"{p}_vh")

        builder.add_batched_mat_mul(
            name=f"{p}_qk_bmm",
            input_names=[f"{p}_qh", f"{p}_kh"],
            output_name=f"{p}_qk",
            transpose_b=True,
        )
        builder.add_add_broadcastable(
            name=f"{p}_mask_add",
            input_names=[f"{p}_qk", "attn_bias"],
            output_name=f"{p}_scores",
        )
        builder.add_softmax_nd(
            name=f"{p}_sm",
            input_name=f"{p}_scores",
            output_name=f"{p}_probs",
            axis=-1,
        )
        builder.add_batched_mat_mul(
            name=f"{p}_pv_bmm",
            input_names=[f"{p}_probs", f"{p}_vh"],
            output_name=f"{p}_ctx4",
        )
        builder.add_transpose(f"{p}_ctx_tr", [0, 2, 1, 3], f"{p}_ctx4", f"{p}_ctx_t")
        builder.add_reshape_static(f"{p}_ctx_r", f"{p}_ctx_t", f"{p}_ctx", [1, MAX_LEN, D_MODEL])

        o_lin = add_linear(f"{p}_ctx", weights[f"{p}_o_w"], weights[f"{p}_o_b"], f"{p}_o")
        builder.add_add_broadcastable(f"{p}_res1_add", [cur, o_lin], f"{p}_res1")
        ln1 = add_ln(f"{p}_res1", weights[f"{p}_ln1_w"], weights[f"{p}_ln1_b"], f"{p}_ln1")

        fc1 = add_linear(ln1, weights[f"{p}_fc1_w"], weights[f"{p}_fc1_b"], f"{p}_fc1")
        act = add_gelu(fc1, f"{p}_gelu")
        fc2 = add_linear(act, weights[f"{p}_fc2_w"], weights[f"{p}_fc2_b"], f"{p}_fc2")
        builder.add_add_broadcastable(f"{p}_res2_add", [ln1, fc2], f"{p}_res2")
        cur = add_ln(f"{p}_res2", weights[f"{p}_ln2_w"], weights[f"{p}_ln2_b"], f"{p}_ln2")

    builder.add_multiply_broadcastable(
        name="cur_valid_mul",
        input_names=[cur, "valid_3d"],
        output_name="cur_valid",
    )
    cur = "cur_valid"

    # 4. Packed [1, 256, 421] heads
    u_h = add_gelu(
        add_linear(cur, weights["upos_fc1_w"], weights["upos_fc1_b"], "upos_fc1"), "upos_act"
    )
    upos_logits = add_linear(u_h, weights["upos_fc2_w"], weights["upos_fc2_b"], "upos_fc2")

    a_dep = add_gelu(
        add_linear(cur, weights["arc_dep_w"], weights["arc_dep_b"], "arc_dep"), "arc_dep_act"
    )
    a_head_raw = add_gelu(
        add_linear(cur, weights["arc_head_w"], weights["arc_head_b"], "arc_head"), "arc_head_act"
    )
    builder.add_multiply_broadcastable(
        name="a_head_valid_mul",
        input_names=[a_head_raw, "valid_3d"],
        output_name="a_head",
    )
    builder.add_activation(
        name="arc_dep_scale",
        non_linearity="LINEAR",
        input_name=a_dep,
        output_name="a_dep_scaled",
        params=[1.0 / math.sqrt(D_HEAD_MLP), 0.0],
    )
    builder.add_batched_mat_mul(
        name="arc_bmm",
        input_names=["a_dep_scaled", "a_head"],
        output_name="arc_logits",
        transpose_b=True,
    )

    rh_dep = add_gelu(
        add_linear(cur, weights["rel_dep_in_w"], weights["rel_dep_in_b"], "rel_dep_in"),
        "rel_dep_act",
    )
    rh_head = add_gelu(
        add_linear(cur, weights["rel_head_in_w"], weights["rel_head_in_b"], "rel_head_in"),
        "rel_head_act",
    )
    r_dep = add_linear(rh_dep, weights["r_dep_w"], weights["r_dep_b"], "r_dep")
    r_head = add_linear(rh_head, weights["r_head_w"], weights["r_head_b"], "r_head")
    r_bi_dep = add_linear(rh_dep, weights["r_bi_dep_w"], weights["r_bi_dep_b"], "r_bi_dep")
    r_bi_head = add_linear(rh_head, weights["r_bi_head_w"], weights["r_bi_head_b"], "r_bi_head")

    builder.add_concat_nd(
        name="pack_logits",
        input_names=[upos_logits, "arc_logits", r_dep, r_head, r_bi_dep, r_bi_head],
        output_name="logits",
        axis=-1,
    )

    spec = builder.spec
    ids_desc = spec.description.input[0].type.multiArrayType
    ids_desc.dataType = FeatureTypes_pb2.ArrayFeatureType.INT32
    del ids_desc.shape[:]
    ids_desc.shape.extend([1, MAX_LEN])

    out_desc = spec.description.output[0].type.multiArrayType
    out_desc.dataType = FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    del out_desc.shape[:]
    out_desc.shape.extend([1, MAX_LEN, LABELS])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ct.models.MLModel(spec).save(str(out_path))


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_and_make_expected(
    tok: Tokenizer, onnx_path: Path, mlmodel_path: Path
) -> tuple[dict, dict]:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    mlmodel = ct.models.MLModel(str(mlmodel_path), compute_units=ct.ComputeUnit.CPU_ONLY)

    # 1. All-pad [0]*256 finiteness & parity check (mirrors `coreml.rs` load check)
    zeros_64 = np.zeros((1, MAX_LEN), dtype=np.int64)
    zeros_32 = np.zeros((1, MAX_LEN), dtype=np.int32)
    ort_z = sess.run(["logits"], {"input_ids": zeros_64})[0]
    cml_z = mlmodel.predict({"input_ids": zeros_32})["logits"]
    assert np.all(np.isfinite(ort_z)) and np.all(np.isfinite(cml_z)), "non-finite all-pad output!"
    all_pad_diff = float(np.max(np.abs(ort_z - cml_z)))

    fixtures_out: list[dict] = []
    max_diff = all_pad_diff
    onnx_ms: list[float] = []
    coreml_ms: list[float] = []

    for fx_id, variant, text in FIXTURES:
        enc = tok.encode(text)
        ids = [int(i) for i in enc.ids]
        assert len(ids) <= MAX_LEN, f"{fx_id} exceeds max_len={MAX_LEN}"
        spans = [[int(s), int(e)] for s, e in enc.offsets]
        for s, e in spans:
            assert 0 <= s <= e <= len(text), f"{fx_id}: invalid span {(s, e)}"

        padded_64 = np.full((1, MAX_LEN), PAD_ID, dtype=np.int64)
        padded_64[0, : len(ids)] = ids
        padded_32 = padded_64.astype(np.int32)

        t0 = time.perf_counter()
        ort_out = sess.run(["logits"], {"input_ids": padded_64})[0][0, : len(ids)]
        onnx_ms.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        cml_out = mlmodel.predict({"input_ids": padded_32})["logits"][0, : len(ids)]
        coreml_ms.append((time.perf_counter() - t1) * 1000.0)

        assert np.all(np.isfinite(ort_out)) and np.all(np.isfinite(cml_out))
        diff = float(np.max(np.abs(ort_out - cml_out))) if len(ids) > 0 else 0.0
        max_diff = max(max_diff, diff)

        rounded = [[round(float(x), 5) for x in row] for row in ort_out]
        argmax = [int(np.argmax(row)) for row in ort_out]
        fixtures_out.append(
            {
                "id": fx_id,
                "variant": variant,
                "text": text,
                "ids": ids,
                "spans": spans,
                "logits": rounded,
                "argmax": argmax,
            }
        )

    assert max_diff <= 1e-4, f"cross-backend max_abs_diff {max_diff:.6e} > 1e-4!"

    expected_doc = {
        "schema": "nlp.models.expected.v1",
        "model": MODEL_ID,
        "max_len": MAX_LEN,
        "labels": LABELS,
        "fixtures": fixtures_out,
    }
    parity_report = {
        "max_abs_diff": max_diff,
        "all_pad_max_abs_diff": all_pad_diff,
        "fixture_count": len(FIXTURES),
        "onnx_median_ms": round(float(np.median(onnx_ms)), 3),
        "coreml_cpu_median_ms": round(float(np.median(coreml_ms)), 3),
    }
    return expected_doc, parity_report


def build_doc_syntax_manifest_entry(model_dir: Path, receipt_rel: str | None) -> dict:
    files = []
    for name in ["model.onnx", "model.mlmodel", "tokenizer.json", "expected.json"]:
        p = model_dir / name
        files.append(
            {
                "path": name,
                "sha256": sha256_of(p),
                "bytes": p.stat().st_size,
            }
        )
    return {
        "id": MODEL_ID,
        "layers": ["pos", "syntax"],
        "licence": LICENCE,
        "dir": MODEL_ID,
        "max_len": MAX_LEN,
        "pad_id": PAD_ID,
        "model": "model.onnx",
        "tokenizer": "tokenizer.json",
        "expected": "expected.json",
        "coreml": {"model": "model.mlmodel", "vocab": VOCAB},
        "files": files,
        "receipt": receipt_rel,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "training/.work/syntax/doc_syntax_v1_weights.npz",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "training/.work/syntax/tokenizer.json",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=Path("/tmp/nlp-models-stage"),
    )
    parser.add_argument(
        "--receipt",
        type=str,
        default="docs/receipts/syntax-2026-11-11.json",
    )
    args = parser.parse_args()

    weights_npz = np.load(args.weights)
    weights = {k: weights_npz[k] for k in weights_npz.files}
    tok = Tokenizer.from_file(str(args.tokenizer))

    model_out = args.stage_dir / MODEL_ID
    model_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.tokenizer, model_out / "tokenizer.json")

    onnx_path = model_out / "model.onnx"
    mlmodel_path = model_out / "model.mlmodel"

    onnx_info = build_onnx_model(weights, onnx_path)
    build_coreml_model(weights, mlmodel_path)

    expected_doc, parity = verify_and_make_expected(tok, onnx_path, mlmodel_path)
    (model_out / "expected.json").write_text(
        json.dumps(expected_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Also stage `test-tiny` alongside `doc-syntax-v1` in `/tmp/nlp-models-stage/` so any test
    # loading from `stage_dir` sees the complete repo manifest!
    repo_manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    test_tiny_entry = next(m for m in repo_manifest["models"] if m["id"] == "test-tiny")
    if (ROOT / "models/test-tiny").is_dir():
        dst_tiny = args.stage_dir / "test-tiny"
        if dst_tiny.exists():
            shutil.rmtree(dst_tiny)
        shutil.copytree(ROOT / "models/test-tiny", dst_tiny)

    syntax_entry = build_doc_syntax_manifest_entry(model_out, args.receipt)
    staged_manifest = {
        "schema": "nlp.models.v1",
        "models": [test_tiny_entry, syntax_entry],
    }
    (args.stage_dir / "manifest.json").write_text(
        json.dumps(staged_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "stage_dir": str(args.stage_dir),
        "onnx": onnx_info,
        "parity": parity,
        "files": syntax_entry["files"],
    }
    (ROOT / "training/.work/syntax/export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
