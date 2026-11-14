#!/usr/bin/env python3
"""Build Japanese morphological & inflection lexicons from NAIST-jdic (BSD-3-Clause).

Outputs (written atomically via os.replace):
  - lexicon/data/ja.tsv         (8-column canonical lexicon)
  - lexicon/data/ja_inflect.tsv (6-column conjugation table)
"""

from __future__ import annotations

import csv
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_DIR = Path(
    "/Users/chrischeng/.gemini/antigravity/brain/"
    "fcdf5ec7-a63d-4308-93f6-0436229a04ea/scratch/lex_build_ja"
)
NLP_CRATE_ROOT = REPO_ROOT / "crates" / "faberline-nlp"
OUT_JA_TSV = NLP_CRATE_ROOT / "lexicon" / "data" / "ja.tsv"
OUT_INFLECT_TSV = NLP_CRATE_ROOT / "lexicon" / "data" / "ja_inflect.tsv"

TARGET_TOTAL_ENTRIES = 68000

# Characters allowed in clean Japanese lexical surfaces.
RE_VALID_JA = re.compile(
    r"^[\u3041-\u3096\u309D-\u309F\u30A1-\u30FA\u30FC-\u30FF"
    r"\u4E00-\u9FFF\u3400-\u4DBF々ヶヵ]+$"
)
RE_HIRAGANA_ONLY = re.compile(r"^[\u3041-\u3096\u309D-\u309F]+$")
RE_KATAKANA_ONLY = re.compile(r"^[\u30A1-\u30FA\u30FC-\u30FF]+$")

# Core particles & auxiliaries with calibrated costs (250..=550) and exact UD masks.
CORE_OVERRIDES: dict[str, tuple[int, str, str, str, str]] = {
    # surface: (cost, decided_upos, upos_mask, lemma, meta)
    "は": (260, "ADP", "ADP", "-", "-"),
    "が": (270, "-", "ADP|SCONJ", "-", "-"),
    "を": (260, "ADP", "ADP", "-", "-"),
    "に": (270, "ADP", "ADP", "-", "-"),
    "で": (280, "-", "ADP|SCONJ", "-", "-"),
    "と": (290, "-", "ADP|SCONJ", "-", "-"),
    "の": (260, "-", "ADP|PART|SCONJ", "-", "-"),
    "へ": (300, "ADP", "ADP", "-", "-"),
    "から": (310, "-", "ADP|SCONJ", "-", "-"),
    "まで": (320, "ADP", "ADP", "-", "-"),
    "より": (350, "ADP", "ADP", "-", "-"),
    "も": (290, "ADP", "ADP", "-", "-"),
    "か": (320, "PART", "PART", "-", "-"),
    "ね": (340, "PART", "PART", "-", "-"),
    "よ": (340, "PART", "PART", "-", "-"),
    "わ": (420, "PART", "PART", "-", "-"),
    "ぞ": (430, "PART", "PART", "-", "-"),
    "ぜ": (430, "PART", "PART", "-", "-"),
    "な": (380, "PART", "PART", "-", "-"),
    "こそ": (400, "ADP", "ADP", "-", "-"),
    "さえ": (400, "ADP", "ADP", "-", "-"),
    "でも": (360, "-", "ADP|CCONJ", "-", "-"),
    "しか": (390, "ADP", "ADP", "-", "-"),
    "ばかり": (420, "PART", "PART", "-", "-"),
    "だけ": (380, "PART", "PART", "-", "-"),
    "ほど": (410, "PART", "PART", "-", "-"),
    "くらい": (430, "PART", "PART", "-", "-"),
    "ぐらい": (430, "PART", "PART", "-", "-"),
    "て": (280, "SCONJ", "SCONJ", "-", "-"),
    "ば": (320, "SCONJ", "SCONJ", "-", "-"),
    "たら": (350, "SCONJ", "SCONJ", "-", "-"),
    "なら": (360, "SCONJ", "SCONJ", "-", "-"),
    "ながら": (380, "SCONJ", "SCONJ", "-", "-"),
    "けれど": (410, "SCONJ", "SCONJ", "-", "-"),
    "けれども": (420, "SCONJ", "SCONJ", "-", "-"),
    "のに": (370, "SCONJ", "SCONJ", "-", "-"),
    "ので": (350, "SCONJ", "SCONJ", "-", "-"),
    "だ": (280, "AUX", "AUX", "-", "-"),
    "です": (290, "AUX", "AUX", "-", "-"),
    "ます": (290, "AUX", "AUX", "-", "-"),
    "た": (270, "AUX", "AUX", "-", "-"),
    "ない": (310, "-", "ADJ|AUX", "-", "infl=adj_i"),
    "無い": (400, "-", "ADJ|AUX", "-", "infl=adj_i"),
    "れる": (360, "AUX", "AUX", "-", "-"),
    "られる": (360, "AUX", "AUX", "-", "-"),
    "せる": (380, "AUX", "AUX", "-", "-"),
    "させる": (380, "AUX", "AUX", "-", "-"),
    "たい": (350, "AUX", "AUX", "-", "-"),
    "らしい": (420, "AUX", "AUX", "-", "-"),
    "である": (350, "AUX", "AUX", "-", "-"),
    "ある": (340, "-", "AUX|VERB", "-", "infl=godan_ra"),
    "いる": (340, "VERB", "VERB", "-", "infl=ichidan"),
    "する": (320, "VERB", "VERB", "-", "infl=sahen"),
    "来る": (360, "VERB", "VERB", "-", "infl=kahen"),
    "くる": (360, "VERB", "VERB", "-", "infl=kahen"),
    "行く": (360, "VERB", "VERB", "-", "infl=godan_iku"),
    "いく": (380, "VERB", "VERB", "-", "infl=godan_iku"),
    "良い": (380, "ADJ", "ADJ", "-", "infl=adj_i"),
    "いい": (360, "ADJ", "ADJ", "-", "infl=adj_i"),
    "よい": (400, "ADJ", "ADJ", "-", "infl=adj_i"),
    # Key pronouns / determiners / conjunctions / adverbs / proper nouns for real texts
    "吾輩": (420, "PRON", "PRON", "-", "-"),
    "東京駅": (360, "PROPN", "PROPN", "-", "sem=location"),
    "ただ": (400, "-", "ADV|CCONJ", "-", "-"),
    "まだ": (420, "ADV", "ADV", "-", "-"),
    "かも": (450, "PART", "PART", "-", "-"),
    "これ": (420, "PRON", "PRON", "-", "-"),
    "それ": (420, "PRON", "PRON", "-", "-"),
    "あれ": (420, "-", "INTJ|PRON", "-", "-"),
    "本当": (550, "NOUN", "NOUN", "-", "-"),
}

CTYPE_TO_PARADIGM = {
    "五段・カ行イ音便": "godan_ka",
    "五段・カ行促音便": "godan_ka",
    "五段・カ行促音便ユク": "godan_iku",
    "五段・ガ行": "godan_ga",
    "五段・サ行": "godan_sa",
    "五段・タ行": "godan_ta",
    "五段・ナ行": "godan_na",
    "五段・バ行": "godan_ba",
    "五段・マ行": "godan_ma",
    "五段・ラ行": "godan_ra",
    "五段・ラ行特殊": "godan_ra",
    "五段・ワ行促音便": "godan_wa",
    "五段・ワ行ウ音便": "godan_wa",
    "一段": "ichidan",
    "一段・得ル": "ichidan",
    "一段・クレル": "ichidan",
    "サ変・スル": "sahen",
    "サ変・−スル": "sahen",
    "カ変・来ル": "kahen",
    "カ変・クル": "kahen",
    "形容詞・アウオ段": "adj_i",
    "形容詞・イ段": "adj_i",
}


def find_naist_csv() -> Path:
    matches = sorted(SCRATCH_DIR.glob("mecab-naist-jdic-*/naist-jdic.csv"))
    if not matches:
        raise FileNotFoundError(
            f"naist-jdic.csv not found under {SCRATCH_DIR}. "
            "Download mecab-naist-jdic tarball into scratch dir first."
        )
    return matches[0]


def quantize_cost(mecab_cost: int, surface_len: int, is_closed: bool) -> int:
    c = round(100 + (mecab_cost - 1000) * 1900 / 8000)
    c = max(120, min(2000, c))
    if not is_closed and surface_len == 1:
        c = max(c, 750)
    elif not is_closed and surface_len >= 2:
        c = min(c, 1650)
    return c


def map_pos(row: list[str]) -> tuple[str | None, str | None, str | None, bool]:
    """Map NAIST-jdic POS columns to (upos, sem, infl, is_closed)."""
    p1, p2, p3, _p4 = row[4], row[5], row[6], row[7]
    ctype, cform = row[8], row[9]

    if p1 == "助詞":
        if p2 in ("格助詞", "係助詞"):
            return "ADP", None, None, True
        if p2 == "接続助詞":
            return "SCONJ", None, None, True
        if p2 in ("終助詞", "副助詞", "副助詞／並立助詞／終助詞", "並立助詞", "連体化", "副詞化", "特殊"):
            return "PART", None, None, True
        return "PART", None, None, True

    if p1 == "助動詞":
        if cform not in ("*", "基本形"):
            return None, None, None, False
        return "AUX", None, None, True

    if p1 == "連体詞":
        return "DET", None, None, True

    if p1 == "接続詞":
        return "CCONJ", None, None, True

    if p1 == "感動詞":
        return "INTJ", None, None, True

    if p1 == "副詞":
        return "ADV", None, None, False

    if p1 == "動詞":
        if cform != "基本形" or p2 == "接尾":
            return None, None, None, False
        paradigm = CTYPE_TO_PARADIGM.get(ctype)
        if not paradigm:
            return None, None, None, False
        return "VERB", None, paradigm, False

    if p1 == "形容詞":
        if cform != "基本形" or p2 == "接尾":
            return None, None, None, False
        paradigm = CTYPE_TO_PARADIGM.get(ctype)
        if not paradigm:
            return None, None, None, False
        return "ADJ", None, paradigm, False

    if p1 == "名詞":
        if p2 == "代名詞":
            return "PRON", None, None, True
        if p2 == "数":
            return "NUM", "quantity", None, True
        if p2 == "固有名詞":
            sem = None
            if p3 == "人名":
                sem = "person"
            elif p3 == "地域":
                sem = "location"
            elif p3 == "組織":
                sem = "organization"
            return "PROPN", sem, None, False
        if p2 == "サ変接続":
            return "NOUN|VERB", None, None, False
        if p2 == "形容動詞語幹":
            return "ADJ", None, None, False
        if p2 == "一般":
            return "NOUN", None, None, False
        if p2 == "副詞可能":
            return "NOUN|ADV", None, None, False
        if p2 == "接尾" and p3 == "助数詞":
            return "NOUN", "quantity", None, True
        return None, None, None, False

    return None, None, None, False


def build_inflect_rules() -> list[tuple[str, int, str, int, str, str]]:
    """Return all productive Japanese conjugation rules."""
    rules: list[tuple[str, int, str, int, str, str]] = []

    def add(paradigm: str, drop: int, tail: str, delta: int, upos: str, split: str) -> None:
        rules.append((paradigm, drop, tail, delta, upos, split))

    # Godan paradigms (drop=1)
    godan_specs = [
        # (paradigm, i_stem, a_stem, e_stem, o_stem, ta_head, ta_suf, te_head, te_suf)
        ("godan_ka", "き", "か", "け", "こ", "い", "た", "い", "て"),
        ("godan_iku", "き", "か", "け", "こ", "っ", "た", "っ", "て"),
        ("godan_ga", "ぎ", "が", "げ", "ご", "い", "だ", "い", "で"),
        ("godan_sa", "し", "さ", "せ", "そ", "し", "た", "し", "て"),
        ("godan_ta", "ち", "た", "て", "と", "っ", "た", "っ", "て"),
        ("godan_na", "に", "な", "ね", "の", "ん", "だ", "ん", "で"),
        ("godan_ba", "び", "ば", "べ", "ぼ", "ん", "だ", "ん", "で"),
        ("godan_ma", "み", "ま", "め", "も", "ん", "だ", "ん", "で"),
        ("godan_ra", "り", "ら", "れ", "ろ", "っ", "た", "っ", "て"),
        ("godan_wa", "い", "わ", "え", "お", "っ", "た", "っ", "て"),
    ]

    for p, ist, ast, est, ost, th, ts, teh, tes in godan_specs:
        # Ren'yokei
        add(p, 1, ist, 60, "VERB", f"*{ist}/<LEMMA>/VERB")
        # Ta / Te / Tari / Tara
        add(p, 1, f"{th}{ts}", 40, "VERB", f"*{th}/<LEMMA>/VERB+{ts}/た/AUX")
        add(p, 1, f"{teh}{tes}", 40, "VERB", f"*{teh}/<LEMMA>/VERB+{tes}/て/SCONJ")
        add(p, 1, f"{th}{ts}り", 60, "VERB", f"*{th}/<LEMMA>/VERB+{ts}り/たり/PART")
        add(p, 1, f"{th}{ts}ら", 50, "VERB", f"*{th}/<LEMMA>/VERB+{ts}ら/たら/SCONJ")
        # Te-iru / Te-ita / Te-imasu / Te-imashita / Te-inai
        add(p, 1, f"{teh}{tes}いる", 40, "VERB", f"*{teh}/<LEMMA>/VERB+{tes}/て/SCONJ+いる/いる/VERB")
        add(p, 1, f"{teh}{tes}いた", 40, "VERB", f"*{teh}/<LEMMA>/VERB+{tes}/て/SCONJ+い/いる/VERB+た/た/AUX")
        add(p, 1, f"{teh}{tes}います", 45, "VERB", f"*{teh}/<LEMMA>/VERB+{tes}/て/SCONJ+い/いる/VERB+ます/ます/AUX")
        add(p, 1, f"{teh}{tes}いました", 45, "VERB", f"*{teh}/<LEMMA>/VERB+{tes}/て/SCONJ+い/いる/VERB+まし/ます/AUX+た/た/AUX")
        add(p, 1, f"{teh}{tes}いない", 45, "VERB", f"*{teh}/<LEMMA>/VERB+{tes}/て/SCONJ+い/いる/VERB+ない/ない/AUX")
        add(p, 1, f"{teh}{tes}る", 60, "VERB", f"*{teh}/<LEMMA>/VERB+{tes}/て/SCONJ+る/いる/VERB")
        add(p, 1, f"{teh}{tes}た", 60, "VERB", f"*{teh}/<LEMMA>/VERB+{tes}/て/SCONJ+た/いる/VERB")
        # Polite (-masu)
        add(p, 1, f"{ist}ます", 40, "VERB", f"*{ist}/<LEMMA>/VERB+ます/ます/AUX")
        add(p, 1, f"{ist}ました", 40, "VERB", f"*{ist}/<LEMMA>/VERB+まし/ます/AUX+た/た/AUX")
        add(p, 1, f"{ist}ません", 45, "VERB", f"*{ist}/<LEMMA>/VERB+ませ/ます/AUX+ん/ん/AUX")
        add(p, 1, f"{ist}ませんでした", 50, "VERB", f"*{ist}/<LEMMA>/VERB+ませ/ます/AUX+ん/ん/AUX+でし/です/AUX+た/た/AUX")
        add(p, 1, f"{ist}ましょう", 50, "VERB", f"*{ist}/<LEMMA>/VERB+ましょ/ます/AUX+う/う/AUX")
        # Negative (-nai / -zu / -nu)
        add(p, 1, f"{ast}ない", 40, "VERB", f"*{ast}/<LEMMA>/VERB+ない/ない/AUX")
        add(p, 1, f"{ast}なかった", 45, "VERB", f"*{ast}/<LEMMA>/VERB+なかっ/ない/AUX+た/た/AUX")
        add(p, 1, f"{ast}なく", 55, "VERB", f"*{ast}/<LEMMA>/VERB+なく/ない/AUX")
        add(p, 1, f"{ast}なくて", 50, "VERB", f"*{ast}/<LEMMA>/VERB+なく/ない/AUX+て/て/SCONJ")
        add(p, 1, f"{ast}ず", 60, "VERB", f"*{ast}/<LEMMA>/VERB+ず/ず/AUX")
        add(p, 1, f"{ast}ぬ", 70, "VERB", f"*{ast}/<LEMMA>/VERB+ぬ/ぬ/AUX")
        # Desiderative (-tai)
        add(p, 1, f"{ist}たい", 45, "VERB", f"*{ist}/<LEMMA>/VERB+たい/たい/AUX")
        add(p, 1, f"{ist}たかった", 50, "VERB", f"*{ist}/<LEMMA>/VERB+たかっ/たい/AUX+た/た/AUX")
        add(p, 1, f"{ist}たくない", 50, "VERB", f"*{ist}/<LEMMA>/VERB+たく/たい/AUX+ない/ない/AUX")
        # Conditional / Potential / Passive / Causative / Volitional / Imperative
        add(p, 1, f"{est}ば", 50, "VERB", f"*{est}/<LEMMA>/VERB+ば/ば/SCONJ")
        add(p, 1, f"{est}る", 55, "VERB", f"*{est}/<LEMMA>/VERB+る/れる/AUX")
        add(p, 1, f"{est}た", 60, "VERB", f"*{est}/<LEMMA>/VERB+た/れる/AUX")
        add(p, 1, f"{est}ない", 55, "VERB", f"*{est}/<LEMMA>/VERB+ない/れる/AUX")
        add(p, 1, f"{ast}れる", 50, "VERB", f"*{ast}/<LEMMA>/VERB+れる/れる/AUX")
        add(p, 1, f"{ast}れた", 50, "VERB", f"*{ast}/<LEMMA>/VERB+れ/れる/AUX+た/た/AUX")
        add(p, 1, f"{ast}れない", 55, "VERB", f"*{ast}/<LEMMA>/VERB+れ/れる/AUX+ない/ない/AUX")
        add(p, 1, f"{ast}せる", 55, "VERB", f"*{ast}/<LEMMA>/VERB+せる/せる/AUX")
        add(p, 1, f"{ast}せた", 55, "VERB", f"*{ast}/<LEMMA>/VERB+せ/せる/AUX+た/た/AUX")
        add(p, 1, f"{ast}せない", 60, "VERB", f"*{ast}/<LEMMA>/VERB+せ/せる/AUX+ない/ない/AUX")
        add(p, 1, f"{ost}う", 55, "VERB", f"*{ost}/<LEMMA>/VERB+う/う/AUX")
        add(p, 1, est, 80, "VERB", f"*{est}/<LEMMA>/VERB")

    # Ichidan (drop=1)
    add("ichidan", 1, "た", 40, "VERB", "*/<LEMMA>/VERB+た/た/AUX")
    add("ichidan", 1, "て", 40, "VERB", "*/<LEMMA>/VERB+て/て/SCONJ")
    add("ichidan", 1, "たり", 60, "VERB", "*/<LEMMA>/VERB+たり/たり/PART")
    add("ichidan", 1, "たら", 50, "VERB", "*/<LEMMA>/VERB+たら/たら/SCONJ")
    add("ichidan", 1, "ている", 40, "VERB", "*/<LEMMA>/VERB+て/て/SCONJ+いる/いる/VERB")
    add("ichidan", 1, "ていた", 40, "VERB", "*/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+た/た/AUX")
    add("ichidan", 1, "ています", 45, "VERB", "*/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+ます/ます/AUX")
    add("ichidan", 1, "ていました", 45, "VERB", "*/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+まし/ます/AUX+た/た/AUX")
    add("ichidan", 1, "ていない", 45, "VERB", "*/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+ない/ない/AUX")
    add("ichidan", 1, "てる", 60, "VERB", "*/<LEMMA>/VERB+て/て/SCONJ+る/いる/VERB")
    add("ichidan", 1, "てた", 60, "VERB", "*/<LEMMA>/VERB+て/て/SCONJ+た/いる/VERB")
    add("ichidan", 1, "ます", 40, "VERB", "*/<LEMMA>/VERB+ます/ます/AUX")
    add("ichidan", 1, "ました", 40, "VERB", "*/<LEMMA>/VERB+まし/ます/AUX+た/た/AUX")
    add("ichidan", 1, "ません", 45, "VERB", "*/<LEMMA>/VERB+ませ/ます/AUX+ん/ん/AUX")
    add("ichidan", 1, "ませんでした", 50, "VERB", "*/<LEMMA>/VERB+ませ/ます/AUX+ん/ん/AUX+でし/です/AUX+た/た/AUX")
    add("ichidan", 1, "ましょう", 50, "VERB", "*/<LEMMA>/VERB+ましょ/ます/AUX+う/う/AUX")
    add("ichidan", 1, "ない", 40, "VERB", "*/<LEMMA>/VERB+ない/ない/AUX")
    add("ichidan", 1, "なかった", 45, "VERB", "*/<LEMMA>/VERB+なかっ/ない/AUX+た/た/AUX")
    add("ichidan", 1, "なく", 55, "VERB", "*/<LEMMA>/VERB+なく/ない/AUX")
    add("ichidan", 1, "なくて", 50, "VERB", "*/<LEMMA>/VERB+なく/ない/AUX+て/て/SCONJ")
    add("ichidan", 1, "ず", 60, "VERB", "*/<LEMMA>/VERB+ず/ず/AUX")
    add("ichidan", 1, "ぬ", 70, "VERB", "*/<LEMMA>/VERB+ぬ/ぬ/AUX")
    add("ichidan", 1, "たい", 45, "VERB", "*/<LEMMA>/VERB+たい/たい/AUX")
    add("ichidan", 1, "たかった", 50, "VERB", "*/<LEMMA>/VERB+たかっ/たい/AUX+た/た/AUX")
    add("ichidan", 1, "たくない", 50, "VERB", "*/<LEMMA>/VERB+たく/たい/AUX+ない/ない/AUX")
    add("ichidan", 1, "れば", 50, "VERB", "*れ/<LEMMA>/VERB+ば/ば/SCONJ")
    add("ichidan", 1, "られる", 50, "VERB", "*/<LEMMA>/VERB+られる/られる/AUX")
    add("ichidan", 1, "られた", 50, "VERB", "*/<LEMMA>/VERB+られ/られる/AUX+た/た/AUX")
    add("ichidan", 1, "られない", 55, "VERB", "*/<LEMMA>/VERB+られ/られる/AUX+ない/ない/AUX")
    add("ichidan", 1, "させる", 55, "VERB", "*/<LEMMA>/VERB+させる/せる/AUX")
    add("ichidan", 1, "させた", 55, "VERB", "*/<LEMMA>/VERB+させ/せる/AUX+た/た/AUX")
    add("ichidan", 1, "させない", 60, "VERB", "*/<LEMMA>/VERB+させ/せる/AUX+ない/ない/AUX")
    add("ichidan", 1, "よう", 55, "VERB", "*/<LEMMA>/VERB+よう/よう/AUX")
    add("ichidan", 1, "ろ", 75, "VERB", "*ろ/<LEMMA>/VERB")
    add("ichidan", 1, "よ", 80, "VERB", "*よ/<LEMMA>/VERB")

    # Sahen (drop=2, する -> "")
    add("sahen", 2, "し", 65, "VERB", "*し/<LEMMA>/VERB")
    add("sahen", 2, "した", 40, "VERB", "*し/<LEMMA>/VERB+た/た/AUX")
    add("sahen", 2, "して", 40, "VERB", "*し/<LEMMA>/VERB+て/て/SCONJ")
    add("sahen", 2, "したり", 60, "VERB", "*し/<LEMMA>/VERB+たり/たり/PART")
    add("sahen", 2, "したら", 50, "VERB", "*し/<LEMMA>/VERB+たら/たら/SCONJ")
    add("sahen", 2, "している", 40, "VERB", "*し/<LEMMA>/VERB+て/て/SCONJ+いる/いる/VERB")
    add("sahen", 2, "していた", 40, "VERB", "*し/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+た/た/AUX")
    add("sahen", 2, "しています", 45, "VERB", "*し/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+ます/ます/AUX")
    add("sahen", 2, "していました", 45, "VERB", "*し/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+まし/ます/AUX+た/た/AUX")
    add("sahen", 2, "していない", 45, "VERB", "*し/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+ない/ない/AUX")
    add("sahen", 2, "します", 40, "VERB", "*し/<LEMMA>/VERB+ます/ます/AUX")
    add("sahen", 2, "しました", 40, "VERB", "*し/<LEMMA>/VERB+まし/ます/AUX+た/た/AUX")
    add("sahen", 2, "しません", 45, "VERB", "*し/<LEMMA>/VERB+ませ/ます/AUX+ん/ん/AUX")
    add("sahen", 2, "しませんでした", 50, "VERB", "*し/<LEMMA>/VERB+ませ/ます/AUX+ん/ん/AUX+でし/です/AUX+た/た/AUX")
    add("sahen", 2, "しましょう", 50, "VERB", "*し/<LEMMA>/VERB+ましょ/ます/AUX+う/う/AUX")
    add("sahen", 2, "しない", 40, "VERB", "*し/<LEMMA>/VERB+ない/ない/AUX")
    add("sahen", 2, "しなかった", 45, "VERB", "*し/<LEMMA>/VERB+なかっ/ない/AUX+た/た/AUX")
    add("sahen", 2, "しなく", 55, "VERB", "*し/<LEMMA>/VERB+なく/ない/AUX")
    add("sahen", 2, "しなくて", 50, "VERB", "*し/<LEMMA>/VERB+なく/ない/AUX+て/て/SCONJ")
    add("sahen", 2, "せず", 55, "VERB", "*せ/<LEMMA>/VERB+ず/ず/AUX")
    add("sahen", 2, "せぬ", 65, "VERB", "*せ/<LEMMA>/VERB+ぬ/ぬ/AUX")
    add("sahen", 2, "したい", 45, "VERB", "*し/<LEMMA>/VERB+たい/たい/AUX")
    add("sahen", 2, "したかった", 50, "VERB", "*し/<LEMMA>/VERB+たかっ/たい/AUX+た/た/AUX")
    add("sahen", 2, "したくない", 50, "VERB", "*し/<LEMMA>/VERB+たく/たい/AUX+ない/ない/AUX")
    add("sahen", 2, "すれば", 50, "VERB", "*すれ/<LEMMA>/VERB+ば/ば/SCONJ")
    add("sahen", 2, "される", 45, "VERB", "*さ/<LEMMA>/VERB+れる/れる/AUX")
    add("sahen", 2, "された", 45, "VERB", "*さ/<LEMMA>/VERB+れ/れる/AUX+た/た/AUX")
    add("sahen", 2, "されない", 50, "VERB", "*さ/<LEMMA>/VERB+れ/れる/AUX+ない/ない/AUX")
    add("sahen", 2, "させる", 55, "VERB", "*さ/<LEMMA>/VERB+せる/せる/AUX")
    add("sahen", 2, "させた", 55, "VERB", "*さ/<LEMMA>/VERB+せ/せる/AUX+た/た/AUX")
    add("sahen", 2, "させない", 60, "VERB", "*さ/<LEMMA>/VERB+せ/せる/AUX+ない/ない/AUX")
    add("sahen", 2, "しよう", 55, "VERB", "*し/<LEMMA>/VERB+よう/よう/AUX")
    add("sahen", 2, "しろ", 75, "VERB", "*しろ/<LEMMA>/VERB")
    add("sahen", 2, "せよ", 80, "VERB", "*せよ/<LEMMA>/VERB")

    # Kahen (drop=2, 来る/くる -> "")
    for stem_ki, stem_ko, stem_ku in [("来", "来", "来"), ("き", "こ", "く")]:
        add("kahen", 2, f"{stem_ki}た", 40, "VERB", f"*{stem_ki}/<LEMMA>/VERB+た/た/AUX")
        add("kahen", 2, f"{stem_ki}て", 40, "VERB", f"*{stem_ki}/<LEMMA>/VERB+て/て/SCONJ")
        add("kahen", 2, f"{stem_ki}たり", 60, "VERB", f"*{stem_ki}/<LEMMA>/VERB+たり/たり/PART")
        add("kahen", 2, f"{stem_ki}たら", 50, "VERB", f"*{stem_ki}/<LEMMA>/VERB+たら/たら/SCONJ")
        add("kahen", 2, f"{stem_ki}ている", 40, "VERB", f"*{stem_ki}/<LEMMA>/VERB+て/て/SCONJ+いる/いる/VERB")
        add("kahen", 2, f"{stem_ki}ていた", 40, "VERB", f"*{stem_ki}/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+た/た/AUX")
        add("kahen", 2, f"{stem_ki}ています", 45, "VERB", f"*{stem_ki}/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+ます/ます/AUX")
        add("kahen", 2, f"{stem_ki}ていました", 45, "VERB", f"*{stem_ki}/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+まし/ます/AUX+た/た/AUX")
        add("kahen", 2, f"{stem_ki}ていない", 45, "VERB", f"*{stem_ki}/<LEMMA>/VERB+て/て/SCONJ+い/いる/VERB+ない/ない/AUX")
        add("kahen", 2, f"{stem_ki}ます", 40, "VERB", f"*{stem_ki}/<LEMMA>/VERB+ます/ます/AUX")
        add("kahen", 2, f"{stem_ki}ました", 40, "VERB", f"*{stem_ki}/<LEMMA>/VERB+まし/ます/AUX+た/た/AUX")
        add("kahen", 2, f"{stem_ki}ません", 45, "VERB", f"*{stem_ki}/<LEMMA>/VERB+ませ/ます/AUX+ん/ん/AUX")
        add("kahen", 2, f"{stem_ki}ませんでした", 50, "VERB", f"*{stem_ki}/<LEMMA>/VERB+ませ/ます/AUX+ん/ん/AUX+でし/です/AUX+た/た/AUX")
        add("kahen", 2, f"{stem_ki}ましょう", 50, "VERB", f"*{stem_ki}/<LEMMA>/VERB+ましょ/ます/AUX+う/う/AUX")
        add("kahen", 2, f"{stem_ko}ない", 40, "VERB", f"*{stem_ko}/<LEMMA>/VERB+ない/ない/AUX")
        add("kahen", 2, f"{stem_ko}なかった", 45, "VERB", f"*{stem_ko}/<LEMMA>/VERB+なかっ/ない/AUX+た/た/AUX")
        add("kahen", 2, f"{stem_ki}たい", 45, "VERB", f"*{stem_ki}/<LEMMA>/VERB+たい/たい/AUX")
        add("kahen", 2, f"{stem_ku}れば", 50, "VERB", f"*{stem_ku}れ/<LEMMA>/VERB+ば/ば/SCONJ")
        add("kahen", 2, f"{stem_ko}られる", 50, "VERB", f"*{stem_ko}/<LEMMA>/VERB+られる/られる/AUX")
        add("kahen", 2, f"{stem_ko}られた", 50, "VERB", f"*{stem_ko}/<LEMMA>/VERB+られ/られる/AUX+た/た/AUX")
        add("kahen", 2, f"{stem_ko}られない", 55, "VERB", f"*{stem_ko}/<LEMMA>/VERB+られ/られる/AUX+ない/ない/AUX")
        add("kahen", 2, f"{stem_ko}させる", 55, "VERB", f"*{stem_ko}/<LEMMA>/VERB+させる/せる/AUX")
        add("kahen", 2, f"{stem_ko}よう", 55, "VERB", f"*{stem_ko}/<LEMMA>/VERB+よう/よう/AUX")
        add("kahen", 2, f"{stem_ko}い", 75, "VERB", f"*{stem_ko}い/<LEMMA>/VERB")

    # Adj-i (drop=1, 美しい -> 美し)
    add("adj_i", 1, "く", 50, "ADJ", "*く/<LEMMA>/ADJ")
    add("adj_i", 1, "くて", 45, "ADJ", "*く/<LEMMA>/ADJ+て/て/SCONJ")
    add("adj_i", 1, "くない", 45, "ADJ", "*く/<LEMMA>/ADJ+ない/ない/AUX")
    add("adj_i", 1, "くなかった", 50, "ADJ", "*く/<LEMMA>/ADJ+なかっ/ない/AUX+た/た/AUX")
    add("adj_i", 1, "かった", 50, "ADJ", "*かっ/<LEMMA>/ADJ+た/た/AUX")
    add("adj_i", 1, "かったり", 65, "ADJ", "*かっ/<LEMMA>/ADJ+たり/たり/PART")
    add("adj_i", 1, "かったら", 55, "ADJ", "*かっ/<LEMMA>/ADJ+たら/たら/SCONJ")
    add("adj_i", 1, "ければ", 55, "ADJ", "*けれ/<LEMMA>/ADJ+ば/ば/SCONJ")
    add("adj_i", 1, "かろう", 65, "ADJ", "*かろ/<LEMMA>/ADJ+う/う/AUX")
    add("adj_i", 1, "いです", 45, "ADJ", "*い/<LEMMA>/ADJ+です/です/AUX")
    add("adj_i", 1, "かったです", 50, "ADJ", "*かっ/<LEMMA>/ADJ+た/た/AUX+です/です/AUX")

    return rules


def verify_inflect_rules(rules: list[tuple[str, int, str, int, str, str]]) -> None:
    for paradigm, drop, tail, delta, upos, split in rules:
        assert drop in (1, 2), f"Invalid drop={drop} in {paradigm}"
        assert tail, f"Empty tail in {paradigm}"
        assert 0 <= delta <= 400, f"Invalid delta={delta} in {paradigm}"
        assert upos in ("VERB", "ADJ"), f"Invalid head_upos={upos}"
        pieces = split.split("+")
        assert pieces[0].startswith("*"), f"First piece must start with '*': {split}"
        reconstructed = []
        for idx, piece in enumerate(pieces):
            parts = piece.split("/")
            assert len(parts) == 3, f"Piece must have 3 parts: {piece}"
            surf, _lemma, p_upos = parts
            assert p_upos, f"Missing UPOS in {piece}"
            if idx == 0:
                reconstructed.append(surf[1:])
            else:
                assert "*" not in surf, f"'*' only allowed in first piece: {split}"
                reconstructed.append(surf)
        joined = "".join(reconstructed)
        assert joined == tail, f"Mismatch in {paradigm}: {joined!r} != {tail!r}"


def write_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = SCRATCH_DIR / f".tmp_{path.name}.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
    # Note: os.replace requires same filesystem; if /tmp or scratch is on same volume it works,
    # otherwise write temp next to target or use shutil+replace. Both are on /Users/chrischeng.
    try:
        os.replace(tmp_path, path)
    except OSError:
        local_tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        local_tmp.write_text(tmp_path.read_text(encoding="utf-8"), encoding="utf-8")
        tmp_path.unlink(missing_ok=True)
        os.replace(local_tmp, path)


def main() -> int:
    csv_path = find_naist_csv()
    print(f"Reading NAIST-jdic from {csv_path}...", file=sys.stderr)

    # 1. Build and verify ja_inflect.tsv
    rules = build_inflect_rules()
    verify_inflect_rules(rules)
    inflect_lines = [
        "# Japanese productive conjugation rules (BSD-3-Clause, derived for NAIST-jdic paradigms)",
        "# paradigm\tstem_drop_chars\tconjugated_tail\tcost_delta\thead_upos\tsplit_morphemes",
    ]
    for paradigm, drop, tail, delta, upos, split in rules:
        inflect_lines.append(f"{paradigm}\t{drop}\t{tail}\t{delta}\t{upos}\t{split}")
    write_atomic(OUT_INFLECT_TSV, inflect_lines)
    print(f"Wrote {len(rules)} rules to {OUT_INFLECT_TSV}", file=sys.stderr)

    # 2. Aggregate NAIST-jdic entries by surface
    # surface -> list of (mecab_cost, upos_str, sem, infl, is_closed)
    grouped: dict[str, list[tuple[int, str, str | None, str | None, bool]]] = defaultdict(list)
    with open(csv_path, encoding="euc-jp", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            surf = row[0]
            if not RE_VALID_JA.match(surf):
                continue
            if len(surf) > 16:
                continue
            upos, sem, infl, is_closed = map_pos(row)
            if upos is None:
                continue
            try:
                mcost = int(row[3])
            except ValueError:
                continue
            grouped[surf].append((mcost, upos, sem, infl, is_closed))

    # 3. Merge each surface into canonical entry
    # Keep: (priority_tier, min_mecab_cost, surface, cost, decided_upos, upos_mask, lemma, counterpart, meta)
    candidates: list[tuple[int, int, str, int, str, str, str, str, str]] = []

    for surf, items in grouped.items():
        if surf in CORE_OVERRIDES:
            continue
        min_cost = min(it[0] for it in items)
        # Filter out rare homographs much more expensive than primary entry
        kept = [it for it in items if it[0] <= min_cost + 1500]

        # If both NOUN and PROPN exist and NOUN is significantly cheaper, drop PROPN homograph
        has_noun = any(it[1] == "NOUN" for it in kept)
        if has_noun:
            noun_min = min(it[0] for it in kept if it[1] == "NOUN")
            kept = [
                it for it in kept
                if not (it[1] == "PROPN" and it[0] > noun_min + 800)
            ]

        upos_set: set[str] = set()
        sems: set[str] = set()
        infls: list[tuple[int, str]] = []
        is_closed = any(it[4] for it in kept)

        for mcost, upos_str, sem, infl, _ in kept:
            for u in upos_str.split("|"):
                upos_set.add(u)
            if sem:
                sems.add(sem)
            if infl:
                infls.append((mcost, infl))

        if not upos_set:
            continue

        # Filter out single-char katakana/hiragana unless closed-class
        if len(surf) == 1 and not is_closed and (RE_HIRAGANA_ONLY.match(surf) or RE_KATAKANA_ONLY.match(surf)):
            continue

        sorted_upos = sorted(upos_set)
        upos_mask = "|".join(sorted_upos)
        decided = sorted_upos[0] if len(sorted_upos) == 1 else "-"

        meta_parts: list[str] = []
        if infls:
            infls.sort(key=lambda x: x[0])
            meta_parts.append(f"infl={infls[0][1]}")
        if len(sems) == 1:
            meta_parts.append(f"sem={next(iter(sems))}")
        meta = ",".join(meta_parts) if meta_parts else "-"

        cost = quantize_cost(min_cost, len(surf), is_closed)

        # Tier 0: closed-class, true verbs (with infl), true adjectives (with infl), adverbs
        # Tier 1: common nouns, sahen nouns, na-adjectives
        # Tier 2: proper nouns
        if is_closed or infl is not None or bool(upos_set & {"ADV", "PRON", "DET", "CCONJ", "INTJ", "ADP", "AUX", "PART", "SCONJ"}):
            tier = 0
        elif "PROPN" in upos_set and len(upos_set) == 1:
            tier = 2
        else:
            tier = 1

        candidates.append((tier, min_cost, surf, cost, decided, upos_mask, "-", "-", meta))

    # 4. Select ~TARGET_TOTAL_ENTRIES: all tier 0, top tier 1, top tier 2
    tier0 = [c for c in candidates if c[0] == 0]
    tier1 = sorted(
        (c for c in candidates if c[0] == 1),
        key=lambda x: (
            x[1] - (2500 if len(x[2]) == 1 else 1800 if len(x[2]) == 2 else 800 if len(x[2]) == 3 else 0),
            x[2],
        ),
    )
    tier2 = sorted((c for c in candidates if c[0] == 2), key=lambda x: (x[1], x[2]))

    selected: dict[str, tuple[int, str, str, str, str, str]] = {}
    for surf, (cost, decided, mask, lemma, meta) in CORE_OVERRIDES.items():
        selected[surf] = (cost, decided, mask, lemma, "-", meta)

    for c in tier0:
        selected.setdefault(c[2], (c[3], c[4], c[5], c[6], c[7], c[8]))

    remaining = max(0, TARGET_TOTAL_ENTRIES - len(selected))
    n_tier1 = min(len(tier1), int(remaining * 0.78))
    n_tier2 = min(len(tier2), remaining - n_tier1)

    for c in tier1[:n_tier1]:
        selected.setdefault(c[2], (c[3], c[4], c[5], c[6], c[7], c[8]))
    for c in tier2[:n_tier2]:
        selected.setdefault(c[2], (c[3], c[4], c[5], c[6], c[7], c[8]))

    # 5. Sort strictly by UTF-8 byte order of col 1 (surface) then col 2 (variant="ja")
    rows_sorted = sorted(selected.items(), key=lambda kv: (kv[0].encode("utf-8"), b"ja"))

    ja_lines = [
        "# Japanese Morphological & Inflection Lexicon (NAIST-jdic BSD-3-Clause)",
        "# surface\tvariant\tcost\tdecided_upos\tupos_mask\tlemma\tcounterpart\tmeta",
    ]
    for surf, (cost, decided, mask, lemma, counterpart, meta) in rows_sorted:
        ja_lines.append(f"{surf}\tja\t{cost}\t{decided}\t{mask}\t{lemma}\t{counterpart}\t{meta}")

    write_atomic(OUT_JA_TSV, ja_lines)
    print(
        f"Wrote {len(rows_sorted)} entries to {OUT_JA_TSV} "
        f"(tier0={len(tier0)}, tier1={n_tier1}, tier2={n_tier2}, overrides={len(CORE_OVERRIDES)})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
