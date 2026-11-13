#!/usr/bin/env python3
"""Validate and merge two-pass silver annotations with resumable batch tracking.

Issue #27: enforces valid JSON, exact Layer 1/2 token and sentence-range equality, 1 root per
sentence, no self-loops or cycles, the 37 UD v2 relation labels (`DEPRELS`, no subtypes),
optional 17 UD v2 `UPOS` tags, `ARG0`/`ARG1`/`ARGM` frame roles within the predicate's sentence,
coreference span bounds (`[start, end]` 1-based inclusive, >= 2 mentions per cluster), known
flags (`layer2_token_error`), and a two-pass agreement filter between Pass A and Pass B.

Modes:
  1. Work directory resume & merge:
       python3 training/silver/validate_merge.py --work-dir <dir> --output <merged.jsonl>
  2. Dry-run verification against recorded two-pass fixtures (no model, no network):
       python3 training/silver/validate_merge.py --dry-run training/silver/sample_dryrun.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "nlp.silver.v1"

UPOS_TAGS = frozenset(
    [
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    ]
)

DEPRELS = frozenset(
    [
        "acl",
        "advcl",
        "advmod",
        "amod",
        "appos",
        "aux",
        "case",
        "cc",
        "ccomp",
        "clf",
        "compound",
        "conj",
        "cop",
        "csubj",
        "dep",
        "det",
        "discourse",
        "dislocated",
        "expl",
        "fixed",
        "flat",
        "goeswith",
        "iobj",
        "list",
        "mark",
        "nmod",
        "nsubj",
        "nummod",
        "obj",
        "obl",
        "orphan",
        "parataxis",
        "punct",
        "reparandum",
        "root",
        "vocative",
        "xcomp",
    ]
)

ROLES = frozenset(["ARG0", "ARG1", "ARGM"])
KNOWN_FLAGS = frozenset(["layer2_token_error"])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_strict_nonneg_int(val: object) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and val >= 0


def check_annotation(ann: object, ref_tokens: list[str], ref_sentences: list[list[int]], ref_upos: list[str | None] | None = None) -> str | None:
    """Return None when structurally valid and matching Layer 1/2 tokens, else error message."""
    if not isinstance(ann, dict):
        return "annotation is not a JSON object"

    tokens = ann.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        return "tokens missing"
    if tokens != ref_tokens:
        return "tokens differ from layer 1/2 output"
    n = len(tokens)
    if n == 0:
        return "tokens empty"

    sents_raw = ann.get("sentences")
    if not isinstance(sents_raw, list):
        return "sentences missing"
    sents: list[list[int]] = []
    for s in sents_raw:
        if not isinstance(s, list) or len(s) != 2 or not is_strict_nonneg_int(s[0]) or not is_strict_nonneg_int(s[1]):
            return "sentences: invalid range"
        sents.append([s[0], s[1]])
    if sents != ref_sentences:
        return "sentence ranges differ from layer 1/2 output"
    if not sents or sents[0][0] != 0 or sents[-1][1] != n:
        return "sentences do not cover the tokens"
    for idx, (b, e) in enumerate(sents):
        if b >= e or (idx > 0 and sents[idx - 1][1] != b):
            return "sentences do not cover the tokens"

    heads = ann.get("heads")
    if not isinstance(heads, list) or not all(is_strict_nonneg_int(h) for h in heads):
        return "heads missing or not integers"
    deprels = ann.get("deprels")
    if not isinstance(deprels, list) or not all(isinstance(r, str) for r in deprels):
        return "deprels missing or not strings"
    if len(heads) != n or len(deprels) != n:
        return "tokens, heads, deprels differ in length"

    if "upos" in ann and ann["upos"] is not None:
        upos = ann["upos"]
        if not isinstance(upos, list) or len(upos) != n:
            return "upos length differs from tokens"
        for idx, tag in enumerate(upos):
            if not isinstance(tag, str) or tag not in UPOS_TAGS:
                return f"token {idx + 1}: upos {tag!r} not in the scheme"
            if ref_upos is not None and ref_upos[idx] is not None and tag != ref_upos[idx]:
                return f"token {idx + 1}: upos {tag!r} disagrees with layer 2 {ref_upos[idx]!r}"

    for b, e in sents:
        roots = 0
        for i in range(b, e):
            rel = deprels[i]
            head = heads[i]
            if rel not in DEPRELS:
                return f'token {i + 1}: label "{rel}" not in the scheme'
            if (head == 0) != (rel == "root"):
                return f"token {i + 1}: root label and head 0 disagree"
            if head == 0:
                roots += 1
            elif head == i + 1 or head < b + 1 or head > e:
                return f"token {i + 1}: head {head} outside the sentence"
        if roots != 1:
            return f"sentence has {roots} roots"
        for i in range(b, e):
            cur = i + 1
            steps = 0
            while cur != 0:
                cur = heads[cur - 1]
                steps += 1
                if steps > n:
                    return f"cycle through token {i + 1}"

    if "flags" in ann and ann["flags"] is not None:
        flags = ann["flags"]
        if not isinstance(flags, list):
            return "flags not a list"
        for f in flags:
            if not isinstance(f, str) or f not in KNOWN_FLAGS:
                return f"unknown flag {f}"

    frames = ann.get("frames")
    if not isinstance(frames, list):
        return "frames missing"
    for f in frames:
        if not isinstance(f, dict) or not is_strict_nonneg_int(f.get("predicate")):
            return "predicate missing"
        p = f["predicate"]
        if p < 1 or p > n:
            return f"predicate {p} out of range"
        sent_bounds = next(((b, e) for b, e in sents if b < p <= e), None)
        assert sent_bounds is not None
        sb, se = sent_bounds
        args = f.get("args")
        if not isinstance(args, list):
            return "args missing"
        for g in args:
            if not isinstance(g, dict) or not is_strict_nonneg_int(g.get("head")):
                return "argument head missing"
            h = g["head"]
            if h < 1 or h > n or h == p:
                return f"argument head {h} invalid for predicate {p}"
            if h <= sb or h > se:
                return f"argument head {h} is not in the sentence of predicate {p}"
            role = g.get("role")
            if not isinstance(role, str) or role not in ROLES:
                return "unknown role"

    coref = ann.get("coref")
    if not isinstance(coref, list):
        return "coref missing"
    for cluster in coref:
        if not isinstance(cluster, list):
            return "cluster not a list"
        if len(cluster) < 2:
            return "cluster with fewer than two mentions"
        seen = set()
        for m in cluster:
            if not isinstance(m, list) or len(m) != 2 or not is_strict_nonneg_int(m[0]) or not is_strict_nonneg_int(m[1]):
                return "invalid mention span"
            s, e = m[0], m[1]
            if s < 1 or s > e or e > n:
                return f"mention [{s},{e}] out of range"
            if (s, e) in seen:
                return f"duplicate mention [{s},{e}] in cluster"
            seen.add((s, e))

    return None


def canonical_frames(ann: dict) -> list[dict]:
    items = []
    for f in ann["frames"]:
        args = sorted(
            ({"role": g["role"], "head": g["head"]} for g in f["args"]),
            key=lambda x: (x["head"], x["role"]),
        )
        # Deduplicate identical (role, head) pairs if any
        dedup = []
        for a in args:
            if not dedup or dedup[-1] != a:
                dedup.append(a)
        items.append({"predicate": f["predicate"], "args": dedup})
    items.sort(key=lambda x: x["predicate"])
    return items


def canonical_coref(ann: dict) -> list[list[list[int]]]:
    clusters = []
    for c in ann["coref"]:
        spans = sorted(([m[0], m[1]] for m in c), key=lambda m: (m[0], m[1]))
        clusters.append(spans)
    clusters.sort(key=lambda c: (c[0][0], c[0][1], len(c)))
    return clusters


def evaluate_pair(
    ref_item: dict,
    pass_a: object,
    pass_b: object,
) -> tuple[str, str, dict | None]:
    """Validate and compare Pass A and Pass B for one item.

    Returns `(status, reason, merged_record_or_None)` where `status` is one of:
      - `"kept"`: both passes valid and agreed
      - `"invalid"`: at least one pass missing or structurally invalid
      - `"dropped"`: both passes structurally valid, but disagreed
    """
    ref_tokens = ref_item["tokens"]
    ref_sents = ref_item["sentences"]
    ref_upos = ref_item.get("layer2_upos")

    err_a = check_annotation(pass_a, ref_tokens, ref_sents, ref_upos)
    if err_a is not None:
        return "invalid", f"pass_a: {err_a}", None
    err_b = check_annotation(pass_b, ref_tokens, ref_sents, ref_upos)
    if err_b is not None:
        return "invalid", f"pass_b: {err_b}", None

    assert isinstance(pass_a, dict) and isinstance(pass_b, dict)

    if pass_a["heads"] != pass_b["heads"] or pass_a["deprels"] != pass_b["deprels"]:
        return "dropped", "disagreed: syntax", None

    upos_a = pass_a.get("upos")
    upos_b = pass_b.get("upos")
    if upos_a is not None and upos_b is not None and upos_a != upos_b:
        return "dropped", "disagreed: upos", None

    flags_a = sorted(set(pass_a.get("flags") or []))
    flags_b = sorted(set(pass_b.get("flags") or []))
    if flags_a != flags_b:
        return "dropped", "disagreed: flags", None

    has_tok_err = "layer2_token_error" in flags_a
    frames_a = canonical_frames(pass_a)
    frames_b = canonical_frames(pass_b)
    coref_a = canonical_coref(pass_a)
    coref_b = canonical_coref(pass_b)

    if not has_tok_err:
        if frames_a != frames_b:
            return "dropped", "disagreed: frames", None
        if coref_a != coref_b:
            return "dropped", "disagreed: coref", None

    merged = {
        "id": ref_item["id"],
        "genre": ref_item.get("genre", "unspecified"),
        "lang": ref_item.get("lang", "unspecified"),
        "text": ref_item["text"],
        "tokens": ref_tokens,
        "sentences": ref_sents,
        "heads": pass_a["heads"],
        "deprels": pass_a["deprels"],
        "frames": [] if has_tok_err else frames_a,
        "coref": [] if has_tok_err else coref_a,
        "usable_layers": ["syntax"] if has_tok_err else ["syntax", "frames", "coref"],
    }
    merged_upos = upos_a if upos_a is not None else upos_b
    if merged_upos is not None:
        merged["upos"] = merged_upos
    if flags_a:
        merged["flags"] = flags_a
    return "kept", "agreed", merged


def load_jsonl_by_id(path: Path) -> tuple[dict[str, object], dict[str, str]]:
    records: dict[str, object] = {}
    parse_errors: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            parse_errors[f"__line_{lineno}"] = f"invalid JSON at line {lineno}: {exc}"
            continue
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            records[obj["id"]] = obj
        else:
            parse_errors[f"__line_{lineno}"] = f"missing string id at line {lineno}"
    return records, parse_errors


def run_dryrun(path: Path) -> int:
    if not path.is_file():
        sys.exit(f"dry-run file not found: {path}")
    counts = {"total": 0, "kept": 0, "invalid": 0, "dropped": 0}
    mismatches: list[str] = []

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        counts["total"] += 1
        status, reason, _ = evaluate_pair(item, item.get("pass_a"), item.get("pass_b"))
        counts[status] += 1
        want_status = item["expected_status"]
        want_reason = item["expected_reason"]
        if status != want_status or want_reason not in reason:
            mismatches.append(
                f"line {lineno} ({item['id']}): got ({status!r}, {reason!r}), "
                f"expected ({want_status!r}, substring {want_reason!r})"
            )

    print(json.dumps({"dry_run": str(path), "counts": counts, "mismatches": len(mismatches)}, indent=2))
    if mismatches:
        for m in mismatches:
            print(f"ERROR: {m}", file=sys.stderr)
        return 1
    return 0


def run_work_dir(work_dir: Path, output_path: Path | None) -> int:
    manifest_path = work_dir / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        sys.exit(f"unexpected manifest schema {manifest.get('schema')!r}")

    summary = {"batches_total": len(manifest["batches"]), "batches_merged": 0, "batches_pending": 0, "kept": 0, "invalid": 0, "dropped": 0}
    all_merged_lines: list[str] = []

    for batch in manifest["batches"]:
        in_path = work_dir / batch["input_file"]
        a_path = work_dir / batch["pass_a_file"]
        b_path = work_dir / batch["pass_b_file"]
        m_path = work_dir / batch["merged_file"]

        if not in_path.is_file() or sha256_file(in_path) != batch["input_sha256"]:
            sys.exit(f"{batch['batch_id']}: input file missing or sha256 mismatch")

        if not a_path.is_file() or not b_path.is_file():
            batch["status"] = "partial" if (a_path.is_file() or b_path.is_file()) else "pending"
            summary["batches_pending"] += 1
            continue

        a_sha = sha256_file(a_path)
        b_sha = sha256_file(b_path)

        if (
            batch.get("status") == "merged"
            and batch.get("pass_a_sha256") == a_sha
            and batch.get("pass_b_sha256") == b_sha
            and m_path.is_file()
            and batch.get("merged_sha256") == sha256_file(m_path)
            and isinstance(batch.get("counts"), dict)
        ):
            c = batch["counts"]
            summary["batches_merged"] += 1
            summary["kept"] += c["kept"]
            summary["invalid"] += c["invalid"]
            summary["dropped"] += c["dropped"]
            all_merged_lines.extend(
                l for l in m_path.read_text(encoding="utf-8").splitlines() if l.strip()
            )
            continue

        ref_rows = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        map_a, parse_err_a = load_jsonl_by_id(a_path)
        map_b, parse_err_b = load_jsonl_by_id(b_path)

        b_counts = {"total": len(ref_rows), "kept": 0, "invalid": 0, "dropped": 0}
        merged_rows: list[dict] = []

        if parse_err_a or parse_err_b:
            # Any malformed JSON line invalidates unmatched items cleanly without crashing resume.
            pass

        for ref in ref_rows:
            rid = ref["id"]
            if rid not in map_a or rid not in map_b:
                b_counts["invalid"] += 1
                continue
            status, _, merged_rec = evaluate_pair(ref, map_a[rid], map_b[rid])
            b_counts[status] += 1
            if status == "kept" and merged_rec is not None:
                merged_rows.append(merged_rec)

        m_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged_rows)
        m_path.write_text(m_text, encoding="utf-8")

        batch["status"] = "merged"
        batch["pass_a_sha256"] = a_sha
        batch["pass_b_sha256"] = b_sha
        batch["merged_sha256"] = sha256_file(m_path)
        batch["counts"] = b_counts

        summary["batches_merged"] += 1
        summary["kept"] += b_counts["kept"]
        summary["invalid"] += b_counts["invalid"]
        summary["dropped"] += b_counts["dropped"]
        all_merged_lines.extend(json.dumps(r, ensure_ascii=False) for r in merged_rows)

    manifest["summary"] = summary
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if output_path is not None and summary["batches_pending"] == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(line + "\n" for line in all_merged_lines),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", type=Path, help="Run offline validation & agreement check on dry-run JSONL")
    parser.add_argument("--work-dir", type=Path, help="Resumable silver work directory")
    parser.add_argument("--output", type=Path, help="Merged silver output JSONL path")
    args = parser.parse_args()

    if args.dry_run is not None:
        sys.exit(run_dryrun(args.dry_run))
    if args.work_dir is not None:
        sys.exit(run_work_dir(args.work_dir, args.output))
    parser.error("pass either --dry-run <jsonl> or --work-dir <dir>")


if __name__ == "__main__":
    main()
