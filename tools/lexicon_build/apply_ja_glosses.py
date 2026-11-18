#!/usr/bin/env python3
"""Apply first-party MIT Japanese -> Taiwan zh-Hant content glosses (A2) to ja.tsv.

Reads:
  - crates/faberline-nlp/lexicon/data/ja.tsv
  - tools/lexicon_build/data/first_party_mit_ja_zh.tsv

Strict A2 rules:
  1. Preserve ja.tsv UTF-8 byte row ordering `(c[0].encode("utf-8"), c[1].encode("utf-8"))`
     matching apply_a1_lexemes.py / build_ja.py so diffs only touch modified/inserted rows.
  2. Never modify `cost` or any column other than `counterpart` (col 6) and `meta` (col 7)
     on existing rows; only fill existing unglossed (`cols[6] == '-'`) rows.
  3. Only insert missing rows (<= 35 total) for the explicit closed paradigms and two
     approved everyday nouns:
     (a) Step 4 Keigo verbs (`拝見する`)
     (b) 4-kana reduplicated ABAB / mimetic adverbs (`ざあざあ`, `ざわざわ`, `くたくた`,
         `にゃあにゃあ`, `ちゅんちゅん`, `ねばねば`, `ぽっちゃり`, `さらっと`, `しーん`).
     (c) The NOUN entries `かばん` and `みかん`.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
JA_TSV = REPO_ROOT / "crates" / "faberline-nlp" / "lexicon" / "data" / "ja.tsv"
GLOSS_TSV = Path(__file__).resolve().parent / "data" / "first_party_mit_ja_zh.tsv"
A4_GLOSS_TSV = Path(__file__).resolve().parent / "data" / "a4_first_party_mit_ja_zh.tsv"

ALLOWED_INSERT_SURFACES = {
    # Approved first-party everyday noun entries.
    "かばん",
    "みかん",
    # Step 4 Keigo verb missing from ja.tsv
    "拝見する",
    # Closed 4-kana reduplicated ABAB / mimetic adverbs missing from ja.tsv
    "ざあざあ",
    "ざわざわ",
    "くたくた",
    "にゃあにゃあ",
    "ちゅんちゅん",
    "ねばねば",
    "ぽっちゃり",
    "さらっと",
    "しーん",
}


def merge_meta(existing_meta: str, infl: str, src_tag: str = "src=first_party_mit") -> str:
    parts: list[str] = []
    if existing_meta and existing_meta != "-":
        for p in existing_meta.split(";"):
            p = p.strip()
            if p and p not in parts:
                parts.append(p)
    if infl and infl != "-" and not any(p.startswith("infl=") for p in parts):
        parts.append(f"infl={infl}")
    if src_tag not in parts:
        parts.append(src_tag)
    return ";".join(parts) if parts else "-"


def main() -> int:
    header_lines: list[str] = []
    rows: list[list[str]] = []

    with open(JA_TSV, encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip() or raw.startswith("#"):
                header_lines.append(raw)
                continue
            cols = raw.split("\t")
            if len(cols) != 8:
                raise ValueError(f"Expected 8 columns in {JA_TSV}, got {len(cols)}: {raw!r}")
            rows.append(cols)

    # Index existing rows by surface.
    by_surface: dict[str, list[list[str]]] = {}
    for cols in rows:
        by_surface.setdefault(cols[0], []).append(cols)

    # Load first-party MIT glosses.
    gloss_entries: list[tuple[str, str, str, str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    with open(GLOSS_TSV, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            if not raw.strip() or raw.startswith("#"):
                continue
            cols = raw.split("\t")
            if len(cols) != 5:
                raise ValueError(f"{GLOSS_TSV}:{line_no}: expected 5 cols, got {len(cols)}: {raw!r}")
            surf, upos, cp, cost, infl = (c.strip() for c in cols)
            key = (surf, upos)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            gloss_entries.append((surf, upos, cp, cost, infl))

    updated_count = 0
    inserted_rows: list[list[str]] = []

    for surf, upos, cp, cost, infl in gloss_entries:
        candidates = by_surface.get(surf, [])
        is_na_adj = upos == "ADJ" and infl == "-"
        matched = [
            r
            for r in candidates
            if r[3] == upos
            or upos in r[4].split("|")
            or (is_na_adj and (r[3] == "NOUN" or "NOUN" in r[4].split("|")))
        ]
        if matched:
            for r in matched:
                if r[6] == "-" and cp != "-":
                    r[6] = cp
                    r[7] = merge_meta(r[7], "-")
                    updated_count += 1
        elif surf in ALLOWED_INSERT_SURFACES:
            meta = merge_meta("-", infl)
            new_row = [surf, "ja", cost, upos, upos, "-", cp, meta]
            rows.append(new_row)
            by_surface.setdefault(surf, []).append(new_row)
            inserted_rows.append(new_row)

    # Apply A4 Sudachi-ranked first-party MIT glosses by exact (surface, variant, decided_upos, upos_mask).
    a4_updated = 0
    if A4_GLOSS_TSV.is_file():
        by_exact_key: dict[tuple[str, str, str, str], list[list[str]]] = {}
        for r in rows:
            by_exact_key.setdefault((r[0], r[1], r[3], r[4]), []).append(r)
        with open(A4_GLOSS_TSV, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                raw = line.rstrip("\n")
                if not raw.strip() or raw.startswith("#") or raw.startswith("group\t"):
                    continue
                cols = raw.split("\t")
                if len(cols) != 7:
                    raise ValueError(f"{A4_GLOSS_TSV}:{line_no}: expected 7 cols, got {len(cols)}: {raw!r}")
                _grp, _rk, surf, var, dupos, mask, cp = (c.strip() for c in cols)
                for r in by_exact_key.get((surf, var, dupos, mask), []):
                    if r[6] == "-" and cp != "-":
                        r[6] = cp
                        r[7] = merge_meta(r[7], "-")
                        a4_updated += 1
        updated_count += a4_updated

    # Preserve exact UTF-8 byte row order matching apply_a1_lexemes.py / build_ja.py.
    rows.sort(key=lambda c: (c[0].encode("utf-8"), c[1].encode("utf-8")))

    tmp_path = JA_TSV.with_suffix(f".tsv.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        for h in header_lines:
            f.write(h + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    os.replace(tmp_path, JA_TSV)

    # Print stats.
    total = len(rows)
    glossed = sum(1 for r in rows if r[6] != "-")
    upos_total: Counter[str] = Counter()
    upos_glossed: Counter[str] = Counter()
    for r in rows:
        u = r[3] if r[3] != "-" else r[4]
        upos_total[u] += 1
        if r[6] != "-":
            upos_glossed[u] += 1

    print(
        f"Applied {len(gloss_entries)} entries: "
        f"updated={updated_count}, inserted={len(inserted_rows)}"
    )
    print(f"Total: {total}, With gloss: {glossed} ({glossed / total * 100:.2f}%)")
    for u in ["VERB", "ADV", "ADJ", "NOUN", "PROPN", "PRON", "DET", "AUX"]:
        t = upos_total[u]
        g = upos_glossed[u]
        pct = (g / t * 100) if t else 0.0
        print(f"  {u}: {g}/{t} ({pct:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
