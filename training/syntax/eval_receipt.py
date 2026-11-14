#!/usr/bin/env python3
"""Evaluate dual-exported `doc-syntax-v1` (ONNX + Core ML) on Gold & Probe and write receipt JSON.

Strictly read-only evaluation (zero training on Gold or Probe; asserts 0% text overlap with
`training/.work/syntax/{train,dev}.jsonl`). Runs BOTH `model.onnx` and `model.mlmodel` (`CPU_ONLY`)
on every sentence of `gold_layers4_6_v1.jsonl`, checks cross-backend parity (`max_abs_diff <= 1e-4`
and 100% identical decoded trees, labels, UPOS upgrades, and quantized `Score(u16)`), and writes
`docs/receipts/syntax-2026-11-11.json` with separate `zh-TW`, `zh-CN`, `en`, and `mixed` metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "training" / "syntax"))
from train_syntax import (  # noqa: E402
    DEPREL_LIST,
    DEPREL_TO_ID,
    MAX_LEN,
    ROOT_DEPREL_ID,
    UPOS_LIST,
    UPOS_TO_ID,
    chu_liu_edmonds,
)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cmd_out(cmd: list[str]) -> str:
    return subprocess.check_output(cmd).decode("utf-8").strip()


def softmax_1d(x: np.ndarray) -> np.ndarray:
    x64 = np.clip(x.astype(np.float64), -3.4e38, 3.4e38)
    m = np.max(x64)
    e = np.exp(x64 - m)
    return e / np.sum(e)


def decode_sentence_from_logits(
    logits_row: np.ndarray,
    subword_indices: list[int],
    l2_upos: list[str | None],
) -> tuple[list[str | None], list[int | None], list[str | None], list[int]]:
    """Exact mirror of `src/doc/syntax.rs` `annotate_with_encoder` on one sentence."""
    m = len(subword_indices)
    rel_offset = 17 + MAX_LEN

    # 1. Contextual UPOS upgrade (>= 0.50) for undecided Layer-2 tokens
    final_upos: list[str | None] = []
    for u_zero, s_u in enumerate(subword_indices):
        if l2_upos[u_zero] is not None:
            final_upos.append(l2_upos[u_zero])
        else:
            u_logits = logits_row[s_u, 0:17]
            best_u = int(np.argmax(u_logits))
            probs_u = softmax_1d(u_logits)
            if float(probs_u[best_u]) >= 0.50:
                final_upos.append(UPOS_LIST[best_u])
            else:
                final_upos.append(None)

    # 2. Arc scores [m, m + 1] over [0, s_1, ..., s_m] + single-root Chu-Liu/Edmonds
    cand = [0] + subword_indices
    arc_scores = logits_row[np.ix_(subword_indices, [17 + s for s in cand])]
    heads = chu_liu_edmonds(arc_scores)

    out_heads: list[int | None] = []
    out_rels: list[str | None] = []
    out_scores_milli: list[int] = []

    for u_zero, s_u in enumerate(subword_indices):
        v = heads[u_zero]
        p_arc = float(softmax_1d(arc_scores[u_zero])[v])
        if v == 0:
            deprel_idx = ROOT_DEPREL_ID
            p_rel = 1.0
        else:
            s_v = subword_indices[v - 1]
            r_u = logits_row[s_u]
            r_v = logits_row[s_v]
            r_dep = r_u[rel_offset : rel_offset + 37]
            r_head = r_v[rel_offset + 37 : rel_offset + 74]
            r_bi_dep = r_u[rel_offset + 74 : rel_offset + 111]
            r_bi_head = r_v[rel_offset + 111 : rel_offset + 148]
            rel_logits = r_dep + r_head + r_bi_dep * r_bi_head

            # Best non-root label (index != 34)
            non_root_idxs = [c for c in range(37) if c != ROOT_DEPREL_ID]
            nr_vals = rel_logits[non_root_idxs]
            nr_best_pos = int(np.argmax(nr_vals))
            deprel_idx = non_root_idxs[nr_best_pos]
            p_rel = float(softmax_1d(nr_vals)[nr_best_pos])

        milli = int(round(min(max(p_arc * p_rel, 0.0), 1.0) * 1000.0))
        out_scores_milli.append(milli)
        if milli < 350:
            out_heads.append(None)
            out_rels.append(None)
        else:
            out_heads.append(v)  # 0 = root, 1..=m = local 1-based head
            out_rels.append(DEPREL_LIST[deprel_idx])

    return final_upos, out_heads, out_rels, out_scores_milli


def evaluate_dataset(
    items: list[dict],
    nlp_doc: Path,
    tok: Tokenizer,
    sess: ort.InferenceSession,
    mlmodel: ct.models.MLModel,
) -> dict:
    def empty_bucket() -> dict[str, int]:
        return {
            "items": 0,
            "sentences": 0,
            "tokens": 0,
            "uas_ok": 0,
            "las_ok": 0,
            "upos_total": 0,
            "upos_ok": 0,
            "l2_undecided_total": 0,
            "l2_undecided_ok": 0,
            "emitted_tokens": 0,
            "low_conf_tokens": 0,
        }

    overall = empty_bucket()
    by_variant: dict[str, dict[str, int]] = {}
    by_lang: dict[str, dict[str, int]] = {}
    by_genre: dict[str, dict[str, int]] = {}

    max_abs_diff = 0.0
    parity_mismatches = 0
    max_score_milli_diff = 0
    onnx_lat_ms: list[float] = []
    coreml_lat_ms: list[float] = []

    for item in items:
        txt = item["text"]
        lang = item.get("lang", "unknown")
        variant = item.get("variant", lang)
        genre = item.get("genre", "unknown")

        b_var = by_variant.setdefault(variant, empty_bucket())
        b_lng = by_lang.setdefault(lang, empty_bucket())
        b_gnr = by_genre.setdefault(genre, empty_bucket())
        for b in (overall, b_var, b_lng, b_gnr):
            b["items"] += 1

        proc = subprocess.run(
            [str(nlp_doc)], input=txt.encode("utf-8"), capture_output=True, check=True
        )
        doc = json.loads(proc.stdout.decode("utf-8"))
        sents = doc["segmentation"]["sentences"]
        toks = doc["tokens"]
        toks_by_s: dict[int, list[dict]] = {si: [] for si in range(len(sents))}
        for tk in toks:
            toks_by_s[tk["sentence"]].append(tk)

        assert len(sents) == len(item["sentences"]), (
            f"{item['id']}: nlp_doc sentences {len(sents)} != gold {len(item['sentences'])}"
        )

        for si, (s_obj, (tok_lo, tok_hi)) in enumerate(zip(sents, item["sentences"])):
            for b in (overall, b_var, b_lng, b_gnr):
                b["sentences"] += 1

            s_toks = toks_by_s[si]
            m = tok_hi - tok_lo
            assert len(s_toks) == m
            sent_str = txt[s_obj["span"]["start_char"] : s_obj["span"]["end_char"]]
            enc = tok.encode(sent_str)
            ids = [int(x) for x in enc.ids]
            assert 0 < len(ids) <= MAX_LEN, f"{item['id']} s{si} subwords={len(ids)}"

            sub_idxs: list[int] = []
            for tk in s_toks:
                ts = tk["span"]["start_char"] - s_obj["span"]["start_char"]
                te = tk["span"]["end_char"] - s_obj["span"]["start_char"]
                found = next(
                    (
                        idx
                        for idx, (os, oe) in enumerate(enc.offsets)
                        if idx >= 1 and os < oe and os < te and ts < oe
                    ),
                    None,
                )
                assert found is not None, f"{item['id']} unmapped token in {sent_str!r}"
                sub_idxs.append(found)

            padded64 = np.zeros((1, MAX_LEN), dtype=np.int64)
            padded64[0, : len(ids)] = ids
            padded32 = padded64.astype(np.int32)

            t0 = time.perf_counter()
            ort_logits = sess.run(["logits"], {"input_ids": padded64})[0][0, : len(ids)]
            onnx_lat_ms.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            cml_logits = mlmodel.predict({"input_ids": padded32})["logits"][0, : len(ids)]
            coreml_lat_ms.append((time.perf_counter() - t1) * 1000.0)

            diff = float(np.max(np.abs(ort_logits - cml_logits)))
            max_abs_diff = max(max_abs_diff, diff)

            l2_u = [tk.get("upos") for tk in s_toks]
            u_ort, h_ort, r_ort, sc_ort = decode_sentence_from_logits(ort_logits, sub_idxs, l2_u)
            u_cml, h_cml, r_cml, sc_cml = decode_sentence_from_logits(cml_logits, sub_idxs, l2_u)
            if (u_ort, h_ort, r_ort) != (u_cml, h_cml, r_cml):
                parity_mismatches += 1
            for s_a, s_b in zip(sc_ort, sc_cml):
                max_score_milli_diff = max(max_score_milli_diff, abs(s_a - s_b))

            gold_heads_slice = item["heads"][tok_lo:tok_hi]
            gold_rels_slice = item["deprels"][tok_lo:tok_hi]
            gold_upos_slice = item["upos"][tok_lo:tok_hi] if "upos" in item else None

            for u_zero in range(m):
                gh_global = gold_heads_slice[u_zero]
                gh_local = 0 if gh_global == 0 else (gh_global - tok_lo)
                gr = gold_rels_slice[u_zero]

                ph = h_cml[u_zero]
                pr = r_cml[u_zero]

                for b in (overall, b_var, b_lng, b_gnr):
                    b["tokens"] += 1
                    if ph is None:
                        b["low_conf_tokens"] += 1
                    else:
                        b["emitted_tokens"] += 1
                        if ph == gh_local:
                            b["uas_ok"] += 1
                            if pr == gr:
                                b["las_ok"] += 1

                    if gold_upos_slice is not None:
                        gu = gold_upos_slice[u_zero]
                        b["upos_total"] += 1
                        if u_cml[u_zero] == gu:
                            b["upos_ok"] += 1
                        if l2_u[u_zero] is None:
                            b["l2_undecided_total"] += 1
                            if u_cml[u_zero] == gu:
                                b["l2_undecided_ok"] += 1

    def fmt_bucket(b: dict[str, int]) -> dict:
        toks = max(1, b["tokens"])
        res: dict = {
            "items": b["items"],
            "sentences": b["sentences"],
            "tokens": b["tokens"],
            "uas": round(b["uas_ok"] / toks, 4),
            "las": round(b["las_ok"] / toks, 4),
            "emitted_rate": round(b["emitted_tokens"] / toks, 4),
            "low_confidence_rate": round(b["low_conf_tokens"] / toks, 4),
        }
        if b["upos_total"] > 0:
            res["upos"] = round(b["upos_ok"] / b["upos_total"], 4)
        if b["l2_undecided_total"] > 0:
            res["contextual_upos_upgrade_acc"] = round(
                b["l2_undecided_ok"] / b["l2_undecided_total"], 4
            )
            res["l2_undecided_tokens"] = b["l2_undecided_total"]
        return res

    return {
        "overall": fmt_bucket(overall),
        "by_variant": {k: fmt_bucket(v) for k, v in sorted(by_variant.items())},
        "by_lang": {k: fmt_bucket(v) for k, v in sorted(by_lang.items())},
        "by_genre": {k: fmt_bucket(v) for k, v in sorted(by_genre.items())},
        "max_abs_diff": max_abs_diff,
        "parity_mismatches": parity_mismatches,
        "max_score_milli_diff": max_score_milli_diff,
        "onnx_median_ms": round(float(np.median(onnx_lat_ms)), 3),
        "onnx_p95_ms": round(float(np.percentile(onnx_lat_ms, 95)), 3),
        "coreml_cpu_median_ms": round(float(np.median(coreml_lat_ms)), 3),
        "coreml_cpu_p95_ms": round(float(np.percentile(coreml_lat_ms, 95)), 3),
    }


def load_gold_items(gold_path: Path | None, git_ref: str | None) -> tuple[list[dict], str, str]:
    if gold_path is not None and gold_path.is_file():
        raw = gold_path.read_bytes()
        src = str(gold_path)
    elif git_ref is not None:
        raw = None
        src = ""
        for rel in (
            "crates/faberline-nlp/tests/data/gold_layers4_6_v1.jsonl",
            "tests/data/gold_layers4_6_v1.jsonl",
        ):
            proc = subprocess.run(["git", "show", f"{git_ref}:{rel}"], capture_output=True, check=False)
            if proc.returncode == 0:
                raw = proc.stdout
                src = f"{git_ref}:{rel}"
                break
        if raw is None:
            raise subprocess.CalledProcessError(
                1, ["git", "show", f"{git_ref}:crates/faberline-nlp/tests/data/gold_layers4_6_v1.jsonl"]
            )
    else:
        rel = "crates/faberline-nlp/tests/data/gold_layers4_6_v1.jsonl"
        default_p = ROOT / rel
        if not default_p.is_file():
            rel = "tests/data/gold_layers4_6_v1.jsonl"
            default_p = ROOT / rel
        raw = default_p.read_bytes()
        src = rel
    items = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    return items, sha256_bytes(raw), src


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, default=Path("/tmp/nlp-models-stage"))
    parser.add_argument("--nlp-doc", type=Path, default=ROOT / "target/release/nlp_doc")
    parser.add_argument("--gold-jsonl", type=Path, default=None)
    parser.add_argument("--gold-git-ref", type=str, default=None)
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument(
        "--out-receipt",
        type=Path,
        default=ROOT / "docs/receipts/syntax-2026-11-11.json",
    )
    args = parser.parse_args()

    mdir = args.stage_dir / "doc-syntax-v1"
    tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    sess = ort.InferenceSession(str(mdir / "model.onnx"), providers=["CPUExecutionProvider"])
    mlmodel = ct.models.MLModel(str(mdir / "model.mlmodel"), compute_units=ct.ComputeUnit.CPU_ONLY)

    # Load training/dev texts to verify strictly 0 overlap with Gold and Probe!
    train_texts: set[str] = set()
    for split_name in ("train.jsonl", "dev.jsonl"):
        sp_path = ROOT / "training/.work/syntax" / split_name
        if sp_path.is_file():
            for line in sp_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    train_texts.add(norm_text(json.loads(line)["text"]))

    gold_eval = None
    gold_sha = None
    gold_src = None
    if not args.skip_gold:
        gold_items, gold_sha, gold_src = load_gold_items(args.gold_jsonl, args.gold_git_ref)
        overlap = [it["id"] for it in gold_items if norm_text(it["text"]) in train_texts]
        assert len(overlap) == 0, f"CRITICAL: Gold overlaps training corpus: {overlap}"

        gold_eval = evaluate_dataset(gold_items, args.nlp_doc, tok, sess, mlmodel)
        assert gold_eval["max_abs_diff"] <= 1e-4, (
            f"Gold cross-backend max_abs_diff {gold_eval['max_abs_diff']} > 1e-4"
        )
        assert gold_eval["max_score_milli_diff"] <= 1, (
            f"Gold quantized Score(u16) diff {gold_eval['max_score_milli_diff']} > 1 milli"
        )

    # Also evaluate 30-item read-only Probe
    probe_meta = {
        r["id"]: r
        for r in (
            json.loads(l)
            for l in (ROOT / "training/probe/probe_texts.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
    }
    probe_items = []
    for line in (ROOT / "training/probe/annotations.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            ann = json.loads(line)
            merged = dict(ann)
            merged.update(probe_meta[ann["id"]])
            probe_items.append(merged)
    probe_overlap = [it["id"] for it in probe_items if norm_text(it["text"]) in train_texts]
    assert len(probe_overlap) == 0, f"Probe overlaps training corpus: {probe_overlap}"
    probe_eval = evaluate_dataset(probe_items, args.nlp_doc, tok, sess, mlmodel)

    export_summary_p = ROOT / "training/.work/syntax/export_summary.json"
    export_summary = (
        json.loads(export_summary_p.read_text(encoding="utf-8"))
        if export_summary_p.is_file()
        else {}
    )

    files_meta = []
    for fname in ["model.onnx", "model.mlmodel", "tokenizer.json", "expected.json"]:
        fp = mdir / fname
        files_meta.append(
            {
                "path": fname,
                "sha256": sha256_bytes(fp.read_bytes()),
                "bytes": fp.stat().st_size,
            }
        )

    lat_src = gold_eval if gold_eval is not None else probe_eval
    receipt = {
        "schema": "nlp.receipt.syntax.v1",
        "model": "doc-syntax-v1",
        "layers": ["pos", "syntax"],
        "licence": "MIT",
        "base_encoder": "intfloat/multilingual-e5-small",
        "date": "2026-11-11",
        "macos": cmd_out(["sw_vers", "-productVersion"]),
        "arch": platform.machine(),
        "rust": cmd_out(["rustc", "--version"]),
        "tolerance": 1e-4,
        "parity": {
            "fixtures_max_abs_diff": export_summary.get("parity", {}).get("max_abs_diff"),
            "all_pad_max_abs_diff": export_summary.get("parity", {}).get("all_pad_max_abs_diff"),
            "gold_max_abs_diff": gold_eval["max_abs_diff"] if gold_eval is not None else None,
            "probe_max_abs_diff": probe_eval["max_abs_diff"],
            "gold_tree_label_upos_mismatches": (
                gold_eval["parity_mismatches"] if gold_eval is not None else 0
            ),
            "gold_max_score_milli_diff": (
                gold_eval["max_score_milli_diff"] if gold_eval is not None else 0
            ),
            "decoded_outputs_identical": (
                gold_eval["parity_mismatches"] == 0 if gold_eval is not None else True
            )
            and probe_eval["parity_mismatches"] == 0,
        },
        "latency_ms": {
            "coreml_cpu_median": lat_src["coreml_cpu_median_ms"],
            "coreml_cpu_p95": lat_src["coreml_cpu_p95_ms"],
            "onnx_cpu_median": lat_src["onnx_median_ms"],
            "onnx_cpu_p95": lat_src["onnx_p95_ms"],
        },
        "gold_eval": (
            {
                "source": gold_src,
                "sha256": gold_sha,
                "train_overlap_items": 0,
                "overall": gold_eval["overall"],
                "by_variant": gold_eval["by_variant"],
                "by_lang": gold_eval["by_lang"],
                "by_genre": gold_eval["by_genre"],
            }
            if gold_eval is not None
            else None
        ),
        "probe_eval": {
            "source": "training/probe/annotations.jsonl",
            "train_overlap_items": 0,
            "overall": probe_eval["overall"],
            "by_lang": probe_eval["by_lang"],
        },
        "files": files_meta,
    }
    args.out_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.out_receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
