#!/usr/bin/env python3
"""Apply reviewed Japanese -> Taiwan zh-Hant glosses to ja.tsv.

Reads:
  - crates/faberline-nlp/lexicon/data/ja.tsv
  - tools/lexicon_build/data/first_party_mit_ja_zh.tsv
  - tools/lexicon_build/data/a4_first_party_mit_ja_zh.tsv
  - tools/lexicon_build/data/cldr_territory_ja_zh_hant.tsv
  - tools/lexicon_build/data/reviewed_345_ja_zh.tsv

Strict A2 rules:
  1. Preserve ja.tsv UTF-8 byte row ordering `(c[0].encode("utf-8"), c[1].encode("utf-8"))`
     matching apply_a1_lexemes.py / build_ja.py so diffs only touch modified/inserted rows.
  2. Never modify `cost` or any column other than `counterpart` (col 6) and `meta` (col 7)
     on existing rows; fill unglossed (`cols[6] == '-'`) rows. Reviewed exceptions
     replace the `午後` identity gloss and four A4 glosses with first-party
     source glosses after matching the previously shipped rows. Five audited
     mixed-POS corrections are applied after the existing overlays, matching
     each previously shipped row exactly.
  3. Only insert missing rows (<= 35 total) for the explicit closed paradigms and
     approved everyday nouns:
     (a) Step 4 Keigo verbs (`拝見する`)
     (b) 4-kana reduplicated ABAB / mimetic adverbs (`ざあざあ`, `ざわざわ`, `くたくた`,
         `にゃあにゃあ`, `ちゅんちゅん`, `ねばねば`, `ぽっちゃり`, `さらっと`, `しーん`).
     (c) The NOUN entries `かばん`, `みかん`, `係員`, `店長`, `申込書`,
         and `朝ご飯`.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
JA_TSV = REPO_ROOT / "crates" / "faberline-nlp" / "lexicon" / "data" / "ja.tsv"
GLOSS_TSV = Path(__file__).resolve().parent / "data" / "first_party_mit_ja_zh.tsv"
A4_GLOSS_TSV = Path(__file__).resolve().parent / "data" / "a4_first_party_mit_ja_zh.tsv"
CLDR_TERRITORY_TSV = Path(__file__).resolve().parent / "data" / "cldr_territory_ja_zh_hant.tsv"
REVIEWED_345_TSV = Path(__file__).resolve().parent / "data" / "reviewed_345_ja_zh.tsv"

# Match every field of the previously shipped A4 rows before replacing their glosses.
REVIEWED_A4_GLOSS_REPLACEMENTS = {
    ("体する", "ja", "1180", "VERB", "VERB", "-", "體會", "infl=sahen;src=first_party_mit"),
    ("遂せる", "ja", "1162", "VERB", "VERB", "-", "能實現", "infl=ichidan;src=first_party_mit"),
    ("タンブン", "ja", "533", "-", "NOUN|VERB", "-", "佈施", "src=first_party_mit"),
    ("コウジュ", "ja", "589", "-", "NOUN|VERB", "-", "口授", "src=first_party_mit"),
}

ALLOWED_INSERT_SURFACES = {
    # Approved first-party everyday noun entries.
    "かばん",
    "みかん",
    "係員",
    "店長",
    "申込書",
    "朝ご飯",
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


def apply_cldr_territory_glosses(by_surface: dict[str, list[list[str]]]) -> int:
    updated = 0
    source = "src=cldr_territory"
    with CLDR_TERRITORY_TSV.open(encoding="utf-8") as table:
        for line_no, line in enumerate(table, 1):
            raw = line.rstrip("\n")
            if not raw or raw.startswith("#"):
                continue
            _code, surface, counterpart = raw.split("\t")
            matches = [
                row
                for row in by_surface.get(surface, [])
                if row[1] == "ja"
                and row[3:5] == ["PROPN", "PROPN"]
                and "sem=location" in row[7].split(";")
            ]
            context = f"{CLDR_TERRITORY_TSV}:{line_no}: {surface}"
            if len(matches) != 1:
                raise ValueError(f"{context}: expected one existing Japanese location entry")
            (row,) = matches
            tagged = merge_meta(row[7], "-", source)
            if row[6] == counterpart and row[7] == tagged:
                continue
            if row[6] != "-" or any(part.startswith("src=") for part in row[7].split(";")):
                raise ValueError(f"{context}: would replace an existing gloss or source")
            row[6] = counterpart
            row[7] = tagged
            updated += 1
    return updated


def apply_reviewed_345(by_surface: dict[str, list[list[str]]]) -> int:
    """Apply five audited corrections after the existing gloss overlays."""
    updated = 0
    seen: set[str] = set()
    with REVIEWED_345_TSV.open(encoding="utf-8") as table:
        for line_no, line in enumerate(table, 1):
            raw = line.rstrip("\n")
            if not raw or raw.startswith("#"):
                continue
            fields = raw.split("\t")
            if len(fields) != 9:
                raise ValueError(f"{REVIEWED_345_TSV}:{line_no}: expected 9 columns")
            old, new = fields[:8], fields[8]
            surface = old[0]
            if surface in seen:
                raise ValueError(f"duplicate reviewed surface: {surface}")
            seen.add(surface)
            matches = by_surface.get(surface, [])
            if len(matches) != 1:
                raise ValueError(f"expected one row for {surface}")
            row = matches[0]
            corrected = old[:6] + [new, merge_meta(old[7], "-")]
            if row == corrected:
                continue
            # The base builder emits blank counterparts. Earlier overlays may
            # fill them, so accept only the exact blank or shipped identity.
            blank = row[:6] == old[:6] and row[6] == "-" and row[7] in ("-", old[7])
            if row != old and not blank:
                raise ValueError(f"unexpected row for {surface}: {row!r}")
            row[:] = corrected
            updated += 1
    if seen != {"供応", "敷衍", "パンチ", "気絶", "暗算"}:
        raise ValueError("reviewed corrections must contain exactly the five audited surfaces")
    return updated


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
                reviewed_identity = (
                    surf == "午後"
                    and upos == "NOUN"
                    and r[:6] == ["午後", "ja", "756", "-", "ADV|NOUN", "-"]
                    and r[6] == surf
                    and cp not in ("-", surf)
                )
                if (r[6] == "-" or reviewed_identity) and cp != "-":
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
                    if cp in ("-", r[6]):
                        continue
                    if r[6] == "-" or tuple(r) in REVIEWED_A4_GLOSS_REPLACEMENTS:
                        r[6] = cp
                        r[7] = merge_meta(r[7], "-")
                        a4_updated += 1
        updated_count += a4_updated

    updated_count += apply_cldr_territory_glosses(by_surface)
    updated_count += apply_reviewed_345(by_surface)

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
