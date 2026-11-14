#!/usr/bin/env python3
"""Standalone Gold Evaluation Scorer for Layers 2, 4, 5, and 6 (`nlp.doc.v1`).

Usage:
  python3 training/eval_gold.py --gold tests/data/gold_layers4_6_v1.jsonl --pred <pred.jsonl>
  python3 training/eval_gold.py --self-test

Accepts prediction JSONL in either:
  1. Gold/probe annotation format (`id`, `upos`, `heads`, `deprels`, optional `frames`, `coref`,
     `zero_mentions`), where `heads` are 1-based token indices (`0` = sentence root) and spans in
     `coref` are 1-based inclusive `[start, end]` pairs; or
  2. `nlp.doc.v1` `Document` JSONL output from `nlp_doc` (`schema == "nlp.doc.v1"`), where
     `tokens[i].upos` holds UPOS and `syntax` holds `{token, head, deprel}` with 0-based indices
     (`head: null` = sentence root).

Uses only the Python 3 standard library; no network and no third-party packages.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

DEFAULT_GOLD = Path("tests/data/gold_layers4_6_v1.jsonl")

UPOS_SET = frozenset({
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
})

DEPRELS_SET = frozenset({
    "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp",
    "clf", "compound", "conj", "cop", "csubj", "dep", "det", "discourse",
    "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark",
    "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct",
    "reparandum", "root", "vocative", "xcomp",
})

ROLES_SET = frozenset({"ARG0", "ARG1", "ARGM"})


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            items.append(obj)
    return items


def normalize_prediction(pred: dict[str, Any], expected_n: int) -> dict[str, Any]:
    """Convert either annotation-format dict or `nlp.doc.v1` Document dict to canonical lists."""
    if pred.get("schema") == "nlp.doc.v1" or (
        "tokens" in pred
        and isinstance(pred["tokens"], list)
        and pred["tokens"]
        and isinstance(pred["tokens"][0], dict)
    ):
        toks = pred.get("tokens", [])
        if len(toks) != expected_n:
            raise ValueError(
                f"token count mismatch in nlp.doc.v1 prediction: got {len(toks)}, expected {expected_n}"
            )
        upos = [t.get("upos") for t in toks]
        heads: list[int | None] = [None] * expected_n
        deprels: list[str | None] = [None] * expected_n
        for syn in pred.get("syntax", []):
            idx = syn["token"]
            if not (0 <= idx < expected_n):
                raise ValueError(f"syntax token index {idx} out of range 0..{expected_n}")
            raw_head = syn.get("head")
            heads[idx] = 0 if raw_head is None else int(raw_head) + 1
            deprels[idx] = syn.get("deprel")
            if upos[idx] is None and syn.get("upos") is not None:
                upos[idx] = syn.get("upos")
        frames_out: list[dict[str, Any]] = []
        for f in pred.get("frames", []):
            if not isinstance(f, dict) or "predicate" not in f:
                continue
            args_out = []
            for a in f.get("args", []):
                if isinstance(a, dict) and "role" in a and "head" in a:
                    args_out.append({
                        **a,
                        "head": int(a["head"]) + 1,
                    })
            frames_out.append({
                **f,
                "predicate": int(f["predicate"]) + 1,
                "args": args_out,
            })

        coref_out: list[list[list[int]]] = []
        for cl in pred.get("coref", []):
            if not isinstance(cl, list):
                continue
            cl_out: list[list[int]] = []
            for m in cl:
                if isinstance(m, dict) and "start_token" in m and "end_token" in m:
                    cl_out.append([int(m["start_token"]) + 1, int(m["end_token"]) + 1])
                elif isinstance(m, (list, tuple)) and len(m) == 2:
                    cl_out.append([int(m[0]), int(m[1])])
            if cl_out:
                coref_out.append(cl_out)

        zeros_out: list[dict[str, Any]] = []
        for z in pred.get("zero_mentions", []):
            if not isinstance(z, dict) or "zero_at" not in z or "role" not in z:
                continue
            ant = z.get("antecedent")
            if isinstance(ant, dict) and "start_token" in ant and "end_token" in ant:
                zeros_out.append({
                    **z,
                    "zero_at": int(z["zero_at"]) + 1,
                    "antecedent": [int(ant["start_token"]) + 1, int(ant["end_token"]) + 1],
                })
            elif isinstance(ant, (list, tuple)) and len(ant) == 2:
                if "span" in z or "sentence" in z:
                    zeros_out.append({
                        **z,
                        "zero_at": int(z["zero_at"]) + 1,
                        "antecedent": [int(ant[0]) + 1, int(ant[1]) + 1],
                    })
                else:
                    zeros_out.append({
                        **z,
                        "zero_at": int(z["zero_at"]),
                        "antecedent": [int(ant[0]), int(ant[1])],
                    })

        return {
            "upos": upos,
            "heads": heads,
            "deprels": deprels,
            "frames": frames_out,
            "coref": coref_out,
            "zero_mentions": zeros_out,
        }

    upos_raw = pred.get("upos")
    upos = list(upos_raw) if isinstance(upos_raw, list) else [None] * expected_n
    heads_raw = pred.get("heads")
    heads = list(heads_raw) if isinstance(heads_raw, list) else [None] * expected_n
    deprels_raw = pred.get("deprels")
    deprels = list(deprels_raw) if isinstance(deprels_raw, list) else [None] * expected_n

    if len(upos) != expected_n or len(heads) != expected_n or len(deprels) != expected_n:
        raise ValueError(
            f"prediction array length mismatch (expected {expected_n}): "
            f"upos={len(upos)}, heads={len(heads)}, deprels={len(deprels)}"
        )

    return {
        "upos": upos,
        "heads": heads,
        "deprels": deprels,
        "frames": pred.get("frames", []),
        "coref": pred.get("coref", []),
        "zero_mentions": pred.get("zero_mentions", []),
    }


def align_predictions(
    gold_items: list[dict[str, Any]], pred_items: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(pred_items) != len(gold_items):
        by_id = {p["id"]: p for p in pred_items if "id" in p}
        if len(by_id) == len(pred_items) and all(g["id"] in by_id for g in gold_items):
            return [
                (g, normalize_prediction(by_id[g["id"]], len(g["tokens"])))
                for g in gold_items
            ]
        raise ValueError(
            f"prediction count ({len(pred_items)}) does not match gold count ({len(gold_items)})"
        )

    # Match by id when all predictions carry ids; otherwise align positionally.
    if all("id" in p for p in pred_items):
        by_id = {p["id"]: p for p in pred_items}
        if set(by_id.keys()) != {g["id"] for g in gold_items}:
            raise ValueError("prediction IDs do not match gold IDs")
        return [
            (g, normalize_prediction(by_id[g["id"]], len(g["tokens"])))
            for g in gold_items
        ]

    return [
        (g, normalize_prediction(p, len(g["tokens"])))
        for g, p in zip(gold_items, pred_items)
    ]


def safe_div(num: float, den: float, empty_val: float = 1.0) -> float:
    return num / den if den > 0 else empty_val


def prf1(tp: float, p_den: float, r_den: float) -> tuple[float, float, float]:
    if p_den == 0 and r_den == 0:
        return 1.0, 1.0, 1.0
    if p_den == 0 or r_den == 0 or tp == 0:
        p = tp / p_den if p_den > 0 else 0.0
        r = tp / r_den if r_den > 0 else 0.0
        return p, r, 0.0
    p = tp / p_den
    r = tp / r_den
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def max_weight_bipartite_matching(weights: list[list[float]]) -> float:
    """Exact Kuhn-Munkres (Hungarian) maximum-weight assignment on a rectangular matrix."""
    if not weights or not weights[0]:
        return 0.0
    n_rows = len(weights)
    n_cols = len(weights[0])
    n = max(n_rows, n_cols)

    # Hungarian algorithm minimizes cost; negate weights to maximize similarity.
    cost = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n_rows):
        for j in range(n_cols):
            cost[i + 1][j + 1] = -weights[i][j]

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0][j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    total = 0.0
    for j in range(1, n_cols + 1):
        row = p[j]
        if 1 <= row <= n_rows:
            total += weights[row - 1][j - 1]
    return total


def cluster_sets(raw_clusters: Any) -> list[set[tuple[int, int]]]:
    out: list[set[tuple[int, int]]] = []
    if not isinstance(raw_clusters, list):
        return out
    for cl in raw_clusters:
        if not isinstance(cl, list):
            continue
        s: set[tuple[int, int]] = set()
        for m in cl:
            if isinstance(m, (list, tuple)) and len(m) == 2:
                s.add((int(m[0]), int(m[1])))
        if s:
            out.append(s)
    return out


def compute_coref_conll(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, float]:
    """Compute CoNLL-2012 MUC, B^3, CEAF_e (phi4), and CoNLL average F1."""
    muc_r_num = 0.0
    muc_r_den = 0.0
    muc_p_num = 0.0
    muc_p_den = 0.0

    b3_r_num = 0.0
    b3_r_den = 0.0
    b3_p_num = 0.0
    b3_p_den = 0.0

    ceaf_sim = 0.0
    ceaf_r_den = 0.0
    ceaf_p_den = 0.0

    for gold, pred in pairs:
        k_clusters = cluster_sets(gold.get("coref", []))
        r_clusters = cluster_sets(pred.get("coref", []))

        # 1. MUC (Vilain et al., 1995)
        r_all = set().union(*r_clusters) if r_clusters else set()
        for k in k_clusters:
            partitions = sum(1 for r in r_clusters if k & r)
            partitions += sum(1 for m in k if m not in r_all)
            muc_r_num += len(k) - partitions
            muc_r_den += len(k) - 1

        k_all = set().union(*k_clusters) if k_clusters else set()
        for r in r_clusters:
            partitions = sum(1 for k in k_clusters if r & k)
            partitions += sum(1 for m in r if m not in k_all)
            muc_p_num += len(r) - partitions
            muc_p_den += len(r) - 1

        # 2. B^3 (Bagga & Baldwin, 1998; Pradhan et al., 2014 Eq. 8-9)
        for k in k_clusters:
            b3_r_den += len(k)
            for r in r_clusters:
                inter = len(k & r)
                if inter:
                    b3_r_num += (inter * inter) / len(k)

        for r in r_clusters:
            b3_p_den += len(r)
            for k in k_clusters:
                inter = len(r & k)
                if inter:
                    b3_p_num += (inter * inter) / len(r)

        # 3. CEAF_e with phi4 Dice similarity (Luo, 2005; Pradhan et al., 2014)
        ceaf_r_den += len(k_clusters)
        ceaf_p_den += len(r_clusters)
        if k_clusters and r_clusters:
            sim_matrix = [
                [2.0 * len(k & r) / (len(k) + len(r)) for r in r_clusters]
                for k in k_clusters
            ]
            ceaf_sim += max_weight_bipartite_matching(sim_matrix)

    # MUC precision and recall have separate numerators when twinless mentions exist.
    if muc_p_den == 0 and muc_r_den == 0:
        muc_p = muc_r = muc_f1 = 1.0
    else:
        muc_p = muc_p_num / muc_p_den if muc_p_den > 0 else 0.0
        muc_r = muc_r_num / muc_r_den if muc_r_den > 0 else 0.0
        muc_f1 = (2.0 * muc_p * muc_r / (muc_p + muc_r)) if (muc_p + muc_r) > 0 else 0.0

    if b3_p_den == 0 and b3_r_den == 0:
        b3_p = b3_r = b3_f1 = 1.0
    else:
        b3_p = b3_p_num / b3_p_den if b3_p_den > 0 else 0.0
        b3_r = b3_r_num / b3_r_den if b3_r_den > 0 else 0.0
        b3_f1 = (2.0 * b3_p * b3_r / (b3_p + b3_r)) if (b3_p + b3_r) > 0 else 0.0

    _, _, ceafe_f1 = prf1(ceaf_sim, ceaf_p_den, ceaf_r_den)
    conll_f1 = (muc_f1 + b3_f1 + ceafe_f1) / 3.0

    return {
        "muc_f1": round(muc_f1, 6),
        "b3_f1": round(b3_f1, 6),
        "ceafe_f1": round(ceafe_f1, 6),
        "coref_conll_f1": round(conll_f1, 6),
    }


def extract_frame_tuples(item_id: str, frames: Any) -> set[tuple[str, int, str, int]]:
    out: set[tuple[str, int, str, int]] = set()
    if not isinstance(frames, list):
        return out
    for f in frames:
        if not isinstance(f, dict) or "predicate" not in f:
            continue
        pred_idx = int(f["predicate"])
        for arg in f.get("args", []):
            if isinstance(arg, dict) and "role" in arg and "head" in arg:
                out.add((item_id, pred_idx, str(arg["role"]), int(arg["head"])))
    return out


def extract_zero_tuples(item_id: str, zeros: Any) -> set[tuple[str, int, str, int, int]]:
    out: set[tuple[str, int, str, int, int]] = set()
    if not isinstance(zeros, list):
        return out
    for z in zeros:
        if (
            isinstance(z, dict)
            and "zero_at" in z
            and "role" in z
            and isinstance(z.get("antecedent"), (list, tuple))
            and len(z["antecedent"]) == 2
        ):
            out.add((
                item_id,
                int(z["zero_at"]),
                str(z["role"]),
                int(z["antecedent"][0]),
                int(z["antecedent"][1]),
            ))
    return out


def score_slice(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    items_count = len(pairs)
    sents_count = sum(len(g["sentences"]) for g, _ in pairs)
    total_tokens = 0
    non_punct_tokens = 0
    upos_decided = 0
    upos_correct = 0
    syn_decided = 0
    uas_correct = 0
    las_correct = 0
    uas_np_correct = 0
    las_np_correct = 0

    gold_frames: set[tuple[str, int, str, int]] = set()
    pred_frames: set[tuple[str, int, str, int]] = set()
    gold_zeros: set[tuple[str, int, str, int, int]] = set()
    pred_zeros: set[tuple[str, int, str, int, int]] = set()

    for gold, pred in pairs:
        gid = gold["id"]
        g_upos = gold["upos"]
        g_heads = gold["heads"]
        g_rels = gold["deprels"]

        p_upos = pred["upos"]
        p_heads = pred["heads"]
        p_rels = pred["deprels"]

        for i in range(len(gold["tokens"])):
            total_tokens += 1
            is_punct = g_upos[i] == "PUNCT"
            if not is_punct:
                non_punct_tokens += 1

            if p_upos[i] is not None:
                upos_decided += 1
                if p_upos[i] == g_upos[i]:
                    upos_correct += 1

            if p_heads[i] is not None:
                syn_decided += 1
                if p_heads[i] == g_heads[i]:
                    uas_correct += 1
                    if not is_punct:
                        uas_np_correct += 1
                    if p_rels[i] == g_rels[i]:
                        las_correct += 1
                        if not is_punct:
                            las_np_correct += 1

        gold_frames |= extract_frame_tuples(gid, gold.get("frames", []))
        pred_frames |= extract_frame_tuples(gid, pred.get("frames", []))
        gold_zeros |= extract_zero_tuples(gid, gold.get("zero_mentions", []))
        pred_zeros |= extract_zero_tuples(gid, pred.get("zero_mentions", []))

    frame_tp = len(gold_frames & pred_frames)
    frame_p, frame_r, frame_f1 = prf1(frame_tp, len(pred_frames), len(gold_frames))

    zero_tp = len(gold_zeros & pred_zeros)
    _, _, zero_f1 = prf1(zero_tp, len(pred_zeros), len(gold_zeros))

    coref = compute_coref_conll(pairs)

    return {
        "items": items_count,
        "sentences": sents_count,
        "tokens": total_tokens,
        "uas": round(safe_div(uas_correct, total_tokens, 0.0), 6),
        "las": round(safe_div(las_correct, total_tokens, 0.0), 6),
        "upos_acc": round(safe_div(upos_correct, total_tokens, 0.0), 6),
        "frame_f1": round(frame_f1, 6),
        "coref_conll_f1": coref["coref_conll_f1"],
        "uas_no_punct": round(safe_div(uas_np_correct, non_punct_tokens, 0.0), 6),
        "las_no_punct": round(safe_div(las_np_correct, non_punct_tokens, 0.0), 6),
        "syntax_coverage": round(safe_div(syn_decided, total_tokens, 0.0), 6),
        "upos_coverage": round(safe_div(upos_decided, total_tokens, 0.0), 6),
        "frame_p": round(frame_p, 6),
        "frame_r": round(frame_r, 6),
        "muc_f1": coref["muc_f1"],
        "b3_f1": coref["b3_f1"],
        "ceafe_f1": coref["ceafe_f1"],
        "zero_mention_f1": round(zero_f1, 6),
    }


def evaluate(
    gold_items: list[dict[str, Any]], pred_items: list[dict[str, Any]]
) -> dict[str, Any]:
    pairs = align_predictions(gold_items, pred_items)
    by_lang = {
        lang: score_slice([(g, p) for g, p in pairs if g["lang"] == lang])
        for lang in ("zh", "en", "mixed")
    }
    variants = ("zh-TW", "zh-CN", "en", "mixed-TW", "mixed-CN")
    by_variant = {
        var: score_slice([(g, p) for g, p in pairs if g.get("variant") == var])
        for var in variants
        if any(g.get("variant") == var for g, _ in pairs)
    }
    by_genre = {
        genre: score_slice([(g, p) for g, p in pairs if g["genre"] == genre])
        for genre in ("novel", "paper", "casual")
    }
    by_cell = {
        f"{genre}/{lang}": score_slice(
            [(g, p) for g, p in pairs if g["genre"] == genre and g["lang"] == lang]
        )
        for genre in ("novel", "paper", "casual")
        for lang in ("zh", "en", "mixed")
    }
    return {
        "overall": score_slice(pairs),
        "by_lang": by_lang,
        "by_variant": by_variant,
        "by_genre": by_genre,
        "by_cell": by_cell,
    }


def format_table(report: dict[str, Any]) -> str:
    header = (
        f"{'Slice':<14} {'Items':>5} {'Sents':>5} {'Tokens':>6} "
        f"{'UAS':>7} {'LAS':>7} {'UPOS':>7} {'FrameF1':>8} {'CorefF1':>8}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    def add_row(label: str, s: dict[str, Any]) -> None:
        rows.append(
            f"{label:<14} {s['items']:>5} {s['sentences']:>5} {s['tokens']:>6} "
            f"{s['uas']:>7.4f} {s['las']:>7.4f} {s['upos_acc']:>7.4f} "
            f"{s['frame_f1']:>8.4f} {s['coref_conll_f1']:>8.4f}"
        )

    add_row("overall", report["overall"])
    rows.append(sep)
    for lang in ("zh", "en", "mixed"):
        add_row(f"lang:{lang}", report["by_lang"][lang])
    if report.get("by_variant"):
        rows.append(sep)
        for var, sub in report["by_variant"].items():
            add_row(f"var:{var}", sub)
    rows.append(sep)
    for genre in ("novel", "paper", "casual"):
        add_row(f"genre:{genre}", report["by_genre"][genre])
    return "\n".join(rows)


def run_self_test(gold_path: Path) -> None:
    # 1. Analytical check on CoNLL coreference math (MUC, B^3, CEAF_e)
    toy_gold = [{
        "id": "t1",
        "genre": "novel",
        "lang": "en",
        "tokens": ["A", "B", "C", "D"],
        "sentences": [[0, 4]],
        "upos": ["PROPN", "PROPN", "PRON", "PRON"],
        "heads": [0, 1, 1, 1],
        "deprels": ["root", "conj", "dep", "dep"],
        "frames": [],
        "coref": [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],
    }]
    # Merge two 2-mention clusters into one 4-mention cluster:
    # MUC: R = (1+1)/(1+1) = 1.0, P = (4-2)/(4-1) = 2/3 -> F1 = 0.8
    # B^3: R = 1.0, P = 0.5 -> F1 = 2/3
    # CEAF_e (phi4): best match pairs 1 pred cluster with 1 gold cluster -> 2*2/(2+4)=2/3;
    #   R = (2/3)/2 = 1/3, P = (2/3)/1 = 2/3 -> F1 = 4/9
    toy_merged = copy.deepcopy(toy_gold)
    toy_merged[0]["coref"] = [[[1, 1], [2, 2], [3, 3], [4, 4]]]
    toy_res = score_slice(align_predictions(toy_gold, toy_merged))
    assert abs(toy_res["muc_f1"] - 0.8) < 1e-5, f"unexpected MUC F1: {toy_res}"
    assert abs(toy_res["b3_f1"] - (2.0 / 3.0)) < 1e-5, f"unexpected B3 F1: {toy_res}"
    assert abs(toy_res["ceafe_f1"] - (4.0 / 9.0)) < 1e-5, f"unexpected CEAFe F1: {toy_res}"

    # 2. Full gold-vs-gold & perturbation checks on gold_layers4_6_v1.jsonl
    assert gold_path.exists(), f"gold file missing: {gold_path}"
    gold = load_jsonl(gold_path)
    assert len(gold) >= 300, f"expected >= 300 gold items, got {len(gold)}"

    perfect = evaluate(gold, copy.deepcopy(gold))
    ov = perfect["overall"]
    for key in ("uas", "las", "upos_acc", "frame_f1", "coref_conll_f1"):
        assert ov[key] == 1.0, f"gold-vs-gold {key} = {ov[key]} != 1.0"
    for slice_group in ("by_lang", "by_variant", "by_genre", "by_cell"):
        for name, sub in perfect[slice_group].items():
            for key in ("uas", "las", "upos_acc", "frame_f1", "coref_conll_f1"):
                assert sub[key] == 1.0, f"gold-vs-gold {slice_group}.{name}.{key} = {sub[key]} != 1.0"

    # 3. Round-trip conversion from `nlp.doc.v1` Document format
    doc_preds = []
    for g in gold:
        doc_preds.append({
            "schema": "nlp.doc.v1",
            "tokens": [{"upos": u} for u in g["upos"]],
            "syntax": [
                {
                    "token": i,
                    "head": None if h == 0 else h - 1,
                    "deprel": r,
                }
                for i, (h, r) in enumerate(zip(g["heads"], g["deprels"]))
            ],
        })
    doc_eval = evaluate(gold, doc_preds)["overall"]
    assert doc_eval["uas"] == 1.0, f"doc round-trip uas={doc_eval['uas']}"
    assert doc_eval["las"] == 1.0, f"doc round-trip las={doc_eval['las']}"
    assert doc_eval["upos_acc"] == 1.0, f"doc round-trip upos_acc={doc_eval['upos_acc']}"
    assert doc_eval["frame_f1"] == 0.0, f"doc round-trip empty frame_f1={doc_eval['frame_f1']}"
    assert doc_eval["coref_conll_f1"] == 0.0, f"doc round-trip empty coref={doc_eval['coref_conll_f1']}"
    assert doc_eval["zero_mention_f1"] == 0.0, (
        f"doc round-trip empty zero_mention_f1={doc_eval['zero_mention_f1']}"
    )

    doc_full_preds = []
    for g in gold:
        doc_full_preds.append({
            "schema": "nlp.doc.v1",
            "tokens": [{"upos": u} for u in g["upos"]],
            "syntax": [
                {
                    "token": i,
                    "head": None if h == 0 else h - 1,
                    "deprel": r,
                }
                for i, (h, r) in enumerate(zip(g["heads"], g["deprels"]))
            ],
            "frames": [
                {
                    "predicate": int(f["predicate"]) - 1,
                    "sentence": 0,
                    "span": {"start_char": 0, "end_char": 1},
                    "args": [
                        {
                            "role": a["role"],
                            "head": int(a["head"]) - 1,
                            "span": {"start_char": 0, "end_char": 1},
                        }
                        for a in f.get("args", [])
                    ],
                }
                for f in g.get("frames", [])
            ],
            "coref": [
                [
                    {
                        "start_token": int(m[0]) - 1,
                        "end_token": int(m[1]) - 1,
                        "head_token": int(m[1]) - 1,
                        "sentence": 0,
                        "span": {"start_char": 0, "end_char": 1},
                    }
                    for m in cl
                ]
                for cl in g.get("coref", [])
            ],
            "zero_mentions": [
                {
                    "zero_at": int(z["zero_at"]) - 1,
                    "sentence": 0,
                    "span": {"start_char": 0, "end_char": 1},
                    "role": z["role"],
                    "antecedent": {
                        "start_token": int(z["antecedent"][0]) - 1,
                        "end_token": int(z["antecedent"][1]) - 1,
                        "head_token": int(z["antecedent"][1]) - 1,
                        "sentence": 0,
                        "span": {"start_char": 0, "end_char": 1},
                    },
                }
                for z in g.get("zero_mentions", [])
            ],
        })
    doc_full_eval = evaluate(gold, doc_full_preds)["overall"]
    for key in ("uas", "las", "upos_acc", "frame_f1", "coref_conll_f1", "zero_mention_f1"):
        assert doc_full_eval[key] == 1.0, f"doc full round-trip {key}={doc_full_eval[key]} != 1.0"

    # 4. Perturb only deprels: UAS stays 1.0, LAS drops
    rel_pert = copy.deepcopy(gold)
    for g in rel_pert[::2]:
        for i, r in enumerate(g["deprels"]):
            if r != "root":
                g["deprels"][i] = "dep" if r != "dep" else "obl"
    rel_res = evaluate(gold, rel_pert)["overall"]
    assert rel_res["uas"] == 1.0, f"rel perturbation changed uas: {rel_res['uas']}"
    assert 0.3 < rel_res["las"] < 0.85, f"unexpected las on rel perturbation: {rel_res['las']}"
    assert rel_res["upos_acc"] == 1.0

    # 5. Perturb heads, UPOS, frames, and coref simultaneously: all metrics drop predictably
    all_pert = copy.deepcopy(gold)
    for g in all_pert[::2]:
        g["upos"] = ["NOUN" if u != "NOUN" else "VERB" for u in g["upos"]]
        g["heads"] = [0 if i == 0 else 1 for i in range(len(g["tokens"]))]
        g["deprels"] = ["root" if i == 0 else "dep" for i in range(len(g["tokens"]))]
        g["frames"] = []
        g["coref"] = []
    pert_res = evaluate(gold, all_pert)["overall"]
    assert 0.4 < pert_res["uas"] < 0.9, f"unexpected perturbed uas: {pert_res['uas']}"
    assert pert_res["las"] < pert_res["uas"], "expected las < uas under head+rel perturbation"
    assert 0.4 < pert_res["upos_acc"] < 0.75, f"unexpected perturbed upos: {pert_res['upos_acc']}"
    assert 0.3 < pert_res["frame_f1"] < 0.8, f"unexpected perturbed frame_f1: {pert_res['frame_f1']}"
    assert 0.3 < pert_res["coref_conll_f1"] < 0.8, (
        f"unexpected perturbed coref_conll_f1: {pert_res['coref_conll_f1']}"
    )

    print("eval_gold.py self-test PASSED:")
    print(format_table(perfect))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score predictions against gold_layers4_6_v1.jsonl.")
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD,
        help="Path to gold JSONL file (default: tests/data/gold_layers4_6_v1.jsonl)",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="Path to prediction JSONL file (annotation format or nlp.doc.v1 Document format)",
    )
    parser.add_argument(
        "--format",
        choices=("both", "json", "table"),
        default="both",
        help="Output format on stdout (default: both)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional file path to write JSON evaluation report",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in verification on gold-vs-gold and perturbed predictions",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test(args.gold)
        return

    if args.pred is None:
        parser.error("--pred is required unless --self-test is set")

    gold_items = load_jsonl(args.gold)
    pred_items = load_jsonl(args.pred)
    report = evaluate(gold_items, pred_items)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.format in ("both", "table"):
        print(format_table(report))
    if args.format == "both":
        print()
    if args.format in ("both", "json"):
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
