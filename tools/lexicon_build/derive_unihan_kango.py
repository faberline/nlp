#!/usr/bin/env python3
"""Derive Japanese-to-Traditional-Chinese kango glosses using Unicode-3.0 Unihan
shinjitai-to-Traditional glyph mappings in `crates/faberline-nlp/src/lexicon/ja.rs`.

Reads `ja.tsv` and `zh_hant.tsv`, finds pure-Han entries (length >= 2) with
`counterpart == '-'`, maps shinjitai glyphs to Traditional Han, verifies that
the resulting word exists in `zh_hant.tsv` with a compatible UPOS, and excludes
known Japanese-Chinese false friends.

Outputs `tools/lexicon_build/data/unihan_kango_glosses.tsv`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
JA_RS_PATH = REPO_ROOT / "crates/faberline-nlp/src/lexicon/ja.rs"
JA_TSV_PATH = REPO_ROOT / "crates/faberline-nlp/lexicon/data/ja.tsv"
ZH_HANT_TSV_PATH = REPO_ROOT / "crates/faberline-nlp/lexicon/data/zh_hant.tsv"
FALSE_FRIENDS_TSV_PATH = REPO_ROOT / "tools/lexicon_build/data/ja_zh_false_friends.tsv"
OUT_TSV_PATH = REPO_ROOT / "tools/lexicon_build/data/unihan_kango_glosses.tsv"

PURE_HAN_RE = re.compile(r"^[\u4E00-\u9FFF\u3400-\u4DBF]{2,}$")
SAHEN_HAN_RE = re.compile(r"^([\u4E00-\u9FFF\u3400-\u4DBF]{2,})する$")


def load_false_friends(path: Path) -> frozenset[str]:
    entries: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if cols[0]:
                entries.add(cols[0])
    return frozenset(entries)


def load_shinjitai_table(ja_rs_path: Path) -> dict[str, str]:
    text = ja_rs_path.read_text(encoding="utf-8")
    m = re.search(
        r"pub fn map_ja_shinjitai_char\(c: char\) -> char \{(.*?)\n\}",
        text,
        re.DOTALL,
    )
    if not m:
        raise RuntimeError(f"Could not find map_ja_shinjitai_char in {ja_rs_path}")
    pairs = re.findall(r"'(.)' => '(.)',", m.group(1))
    if not pairs:
        raise RuntimeError(f"No pairs parsed from map_ja_shinjitai_char in {ja_rs_path}")
    return dict(pairs)


def map_shinjitai(s: str, table: dict[str, str]) -> str:
    return "".join(table.get(c, c) for c in s)


def load_zh_hant_upos(zh_hant_path: Path) -> dict[str, set[str]]:
    zh_upos: dict[str, set[str]] = {}
    with zh_hant_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 5:
                continue
            surf, upos_mask = cols[0], cols[4]
            zh_upos.setdefault(surf, set()).update(upos_mask.split("|"))
    return zh_upos


def derive_glosses() -> tuple[list[tuple[str, str, str, str]], dict[str, int]]:
    table = load_shinjitai_table(JA_RS_PATH)
    zh_upos = load_zh_hant_upos(ZH_HANT_TSV_PATH)
    false_friends = load_false_friends(FALSE_FRIENDS_TSV_PATH)

    results: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    class_counts = {"class1_kango": 0, "class2_sahen_verb": 0}

    with JA_TSV_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 8:
                continue
            surf, _var, _cost, dupos, mask, _lemma, cp, meta = cols[:8]
            if cp != "-":
                continue

            ja_pos = set(mask.split("|"))
            if dupos == "PROPN" or "PROPN" in ja_pos:
                continue

            if PURE_HAN_RE.match(surf):
                if surf in false_friends:
                    continue
                trad = map_shinjitai(surf, table)
                if trad in false_friends or trad not in zh_upos:
                    continue
                if not (ja_pos & zh_upos[trad]):
                    continue
                cls_key = "class1_kango"
            else:
                m_sahen = SAHEN_HAN_RE.match(surf)
                if not m_sahen or dupos != "VERB" or "infl=sahen" not in meta.split(";"):
                    continue
                stem = m_sahen.group(1)
                if surf in false_friends or stem in false_friends:
                    continue
                trad = map_shinjitai(stem, table)
                if trad in false_friends or trad not in zh_upos:
                    continue
                if "VERB" not in zh_upos[trad]:
                    continue
                cls_key = "class2_sahen_verb"

            key = (surf, dupos)
            if key in seen:
                continue
            seen.add(key)
            class_counts[cls_key] += 1
            results.append((surf, dupos, trad, "unihan_zh_hant"))

    results.sort()
    return results, class_counts


def write_glosses(rows: list[tuple[str, str, str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# surface\tdecided_upos\tcounterpart\tsrc\n")
        for surf, dupos, trad, src in rows:
            f.write(f"{surf}\t{dupos}\t{trad}\t{src}\n")


def apply_glosses(gloss_path: Path, ja_tsv_path: Path) -> int:
    glosses: dict[tuple[str, str], tuple[str, str]] = {}
    with gloss_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            surf, dupos, cp, src = line.split("\t")
            glosses[(surf, dupos)] = (cp, src)

    updated = 0
    out_lines: list[str] = []
    with ja_tsv_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw or raw.startswith("#"):
                out_lines.append(line)
                continue
            cols = raw.split("\t")
            if len(cols) < 8:
                out_lines.append(line)
                continue
            key = (cols[0], cols[3])
            if cols[6] == "-" and key in glosses:
                cp, src = glosses[key]
                cols[6] = cp
                src_tag = f"src={src}"
                if cols[7] == "-":
                    cols[7] = src_tag
                elif src_tag not in cols[7].split(";"):
                    cols[7] = f"{cols[7]};{src_tag}"
                updated += 1
                out_lines.append("\t".join(cols) + "\n")
            else:
                out_lines.append(line)

    with ja_tsv_path.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(out_lines)
    return updated


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply generated glosses to crates/faberline-nlp/lexicon/data/ja.tsv",
    )
    args = parser.parse_args(argv)

    rows, class_counts = derive_glosses()
    write_glosses(rows, OUT_TSV_PATH)
    print(
        f"Wrote {len(rows)} entries to {OUT_TSV_PATH.relative_to(REPO_ROOT)} "
        f"(class1_kango={class_counts['class1_kango']}, class2_sahen_verb={class_counts['class2_sahen_verb']})"
    )

    if args.apply:
        updated = apply_glosses(OUT_TSV_PATH, JA_TSV_PATH)
        print(f"Applied {updated} glosses to {JA_TSV_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
