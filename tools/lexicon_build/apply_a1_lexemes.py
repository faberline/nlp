#!/usr/bin/env python3
"""Deterministically apply A1 segmentation & closed-class lexemes to ja.tsv.

Applies:
  - Class 1: Remove godan onbinkei VERB-mask stubs lacking `infl=` that shadow
    existing `infl=godan_*` verbs in `ja.tsv`.
  - Classes 2-6: Merge `tools/lexicon_build/data/a1_lexemes_mit.tsv` into
    `crates/faberline-nlp/lexicon/data/ja.tsv` in UTF-8 byte order.
  - Optional: Refresh `docs/receipts/lexicon-2026-11-15.json` matching the
    schema of `docs/receipts/lexicon-2026-11-14.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
JA_TSV = REPO_ROOT / "crates" / "faberline-nlp" / "lexicon" / "data" / "ja.tsv"
A1_TSV = REPO_ROOT / "tools" / "lexicon_build" / "data" / "a1_lexemes_mit.tsv"
RECEIPT_BASE = REPO_ROOT / "docs" / "receipts" / "lexicon-2026-11-14.json"
RECEIPT_OUT = REPO_ROOT / "docs" / "receipts" / "lexicon-2026-11-15.json"

GODAN_ONBIN_MAP: dict[str, tuple[str, str]] = {
    "godan_ka": ("く", "い"),
    "godan_iku": ("く", "っ"),
    "godan_ga": ("ぐ", "い"),
    "godan_ta": ("つ", "っ"),
    "godan_na": ("ぬ", "ん"),
    "godan_ba": ("ぶ", "ん"),
    "godan_ma": ("む", "ん"),
    "godan_ra": ("る", "っ"),
    "godan_wa": ("う", "っ"),
}


def load_tsv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    headers: list[str] = []
    rows: list[list[str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\r\n")
            if not stripped or stripped.startswith("#"):
                if stripped.startswith("#") and not rows:
                    headers.append(stripped)
                continue
            cols = stripped.split("\t")
            if len(cols) != 8:
                raise ValueError(f"Expected 8 columns in {path}, got {len(cols)}: {cols}")
            rows.append(cols)
    return headers, rows


def find_onbinkei_stubs(rows: list[list[str]]) -> list[tuple[str, str, str]]:
    """Find VERB-mask rows without `infl=` matching a godan onbinkei of an existing verb."""
    verb_infls: dict[str, set[str]] = {}
    for cols in rows:
        for part in cols[7].split(","):
            if part.startswith("infl="):
                verb_infls.setdefault(cols[0], set()).add(part[len("infl="):])

    stubs: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for cols in rows:
        surf = cols[0]
        if "infl=" in cols[7]:
            continue
        if "VERB" not in cols[4].split("|"):
            continue
        if len(surf) < 2 or surf in seen:
            continue
        for infl_type, (dict_end, onbin_end) in GODAN_ONBIN_MAP.items():
            if surf.endswith(onbin_end):
                cand = surf[:-1] + dict_end
                if infl_type in verb_infls.get(cand, ()):
                    stubs.append((surf, cand, infl_type))
                    seen.add(surf)
                    break
    return sorted(stubs)


def pos_coverage(rows: list[list[str]]) -> dict[str, tuple[int, int]]:
    total: Counter[str] = Counter()
    glossed: Counter[str] = Counter()
    for cols in rows:
        key = cols[3] if cols[3] != "-" else cols[4]
        total[key] += 1
        if cols[6] != "-":
            glossed[key] += 1
    return {k: (total[k], glossed[k]) for k in sorted(total)}


def merge_meta(existing_meta: str, new_meta: str) -> str:
    parts = [p for p in existing_meta.replace(";", ",").split(",") if p and p != "-"]
    for p in new_meta.replace(";", ",").split(","):
        if p and p != "-" and p not in parts:
            parts.append(p)
    return ";".join(parts) if parts else "-"


def write_tsv_atomic(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    ordered = sorted(rows, key=lambda c: (c[0].encode("utf-8"), c[1].encode("utf-8")))
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for h in headers:
            f.write(h + "\n")
        for cols in ordered:
            f.write("\t".join(cols) + "\n")
    os.replace(tmp, path)


def sha256_and_bytes(path: Path) -> tuple[int, str]:
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def count_tsv_entries(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n


def refresh_receipt(lexicon_bin: Path | None) -> None:
    base = json.loads(RECEIPT_BASE.read_text(encoding="utf-8"))
    lex_dir = REPO_ROOT / "crates" / "faberline-nlp"

    ja_entries = count_tsv_entries(JA_TSV)
    total_entries = (
        base["variants"]["zh-Hant"]
        + base["variants"]["zh-Hans"]
        + base["variants"]["en-US"]
        + base["variants"]["en-GB"]
        + base["variants"]["en"]
        + ja_entries
        + base["variants"]["ko"]
    )
    base["date"] = "2026-11-15"
    base["total_entries"] = total_entries
    base["variants"]["ja"] = ja_entries

    for item in base["tsv_files"]:
        if item["path"] == "lexicon/data/ja.tsv":
            size, digest = sha256_and_bytes(lex_dir / item["path"])
            item["entries"] = ja_entries
            item["bytes"] = size
            item["sha256"] = digest

    if lexicon_bin is not None and lexicon_bin.is_file():
        bin_bytes, bin_sha = sha256_and_bytes(lexicon_bin)
        base["binary_bytes"] = bin_bytes
        base["lexicon_bin_bytes"] = bin_bytes
        base["lexicon_bin_sha256"] = bin_sha

    RECEIPT_OUT.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote receipt to {RECEIPT_OUT} (ja={ja_entries}, total={total_entries})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing ja.tsv")
    parser.add_argument("--refresh-receipt", action="store_true", help="Write docs/receipts/lexicon-2026-11-15.json")
    parser.add_argument("--lexicon-bin", type=Path, default=None, help="Optional compiled lexicon_v1.bin path")
    args = parser.parse_args()

    headers, rows = load_tsv_rows(JA_TSV)
    before_cov = pos_coverage(rows)
    before_total = len(rows)
    before_glossed = sum(g for _, g in before_cov.values())

    # Class 1: remove godan onbinkei stubs
    stubs = find_onbinkei_stubs(rows)
    stub_surfs = {s[0] for s in stubs}
    rows = [c for c in rows if c[0] not in stub_surfs]

    # Classes 2-6: apply a1_lexemes_mit.tsv
    _, a1_rows = load_tsv_rows(A1_TSV)
    by_surface: dict[str, list[int]] = {}
    for idx, c in enumerate(rows):
        by_surface.setdefault(c[0], []).append(idx)

    inserted = 0
    updated = 0
    for cols in a1_rows:
        surf = cols[0]
        if surf in by_surface:
            for idx in by_surface[surf]:
                rows[idx][6] = cols[6]
                rows[idx][7] = merge_meta(rows[idx][7], cols[7])
            updated += 1
        else:
            rows.append(list(cols))
            by_surface[surf] = [len(rows) - 1]
            inserted += 1

    after_cov = pos_coverage(rows)
    after_total = len(rows)
    after_glossed = sum(g for _, g in after_cov.values())

    print(f"Class 1 removed stubs ({len(stubs)}): {[s[0] for s in stubs]}")
    print(f"Classes 2-6 applied: {inserted} inserted, {updated} updated (total {len(a1_rows)})")
    print(f"Total rows: {before_total} -> {after_total} ({after_total - before_total:+d})")
    print(f"Total glossed: {before_glossed} -> {after_glossed} ({after_glossed - before_glossed:+d})")
    print("\nPOS\tBefore_Total\tBefore_Glossed\tAfter_Total\tAfter_Glossed")
    for k in sorted(set(before_cov) | set(after_cov)):
        bt, bg = before_cov.get(k, (0, 0))
        at, ag = after_cov.get(k, (0, 0))
        if (bt, bg) != (at, ag):
            print(f"{k}\t{bt}\t{bg}\t{at}\t{ag}")

    if not args.dry_run:
        write_tsv_atomic(JA_TSV, headers, rows)
        print(f"\nUpdated {JA_TSV}")
        if args.refresh_receipt:
            refresh_receipt(args.lexicon_bin)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
