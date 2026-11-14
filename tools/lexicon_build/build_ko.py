#!/usr/bin/env python3
"""Build Korean lexical & morphological tables (`lexicon/data/ko.tsv` and `lexicon/data/ko_morph.tsv`)
from Apache-2.0 licensed `mecab-ko-dic-2.1.1-20180720`.

Hard invariants enforced before atomic `os.replace()`:
1. UTF-8, LF line endings, exact 8 columns in `ko.tsv`, exact 6 columns in `ko_morph.tsv`.
2. `ko.tsv` sorted strictly by UTF-8 byte order of `(surface, variant)`, deduplicated by `(surface, variant)`.
3. `cost` is an integer in `100..=2000` (`ko.tsv`) / `100..=1000` (`ko_morph.tsv`).
4. Valid UD v2 UPOS tags; if `decided_upos != "-"`, `upos_mask == decided_upos`.
5. Every `decomp` piece has a non-empty surface and `"".join(piece_surfs) == surface` character-for-character.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DEFAULT_DIC_DIR = (
    "/Users/chrischeng/.gemini/antigravity/brain/"
    "fcdf5ec7-a63d-4308-93f6-0436229a04ea/scratch/lex_build_ko/"
    "mecab-ko-dic-2.1.1-20180720"
)
DEFAULT_SCRATCH_DIR = (
    "/Users/chrischeng/.gemini/antigravity/brain/"
    "fcdf5ec7-a63d-4308-93f6-0436229a04ea/scratch/lex_build_ko"
)

HANGUL_SYLLABLE_RE = re.compile(r"^[\uAC00-\uD7A3]+$")

UD_TAGS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
}

SEJONG_TO_UD = {
    "NNG": "NOUN",
    "NNB": "NOUN",
    "NNBC": "NOUN",
    "NNP": "PROPN",
    "NP": "PRON",
    "NR": "NUM",
    "MM": "DET",
    "MAG": "ADV",
    "MAJ": "CCONJ",
    "IC": "INTJ",
    "VV": "VERB",
    "VA": "ADJ",
    "VX": "AUX",
    "VCP": "AUX",
    "VCN": "AUX",
    "XSV": "VERB",
    "XSA": "ADJ",
    "XSN": "PART",
    "XPN": "PART",
    "XR": "NOUN",
    "JKS": "ADP",
    "JKC": "ADP",
    "JKG": "ADP",
    "JKO": "ADP",
    "JKB": "ADP",
    "JKV": "ADP",
    "JKQ": "ADP",
    "JC": "CCONJ",
    "JX": "PART",
    "EP": "PART",
    "EF": "PART",
    "EC": "SCONJ",
    "ETN": "PART",
    "ETM": "PART",
}

SEM_MAP = {
    "인명": "person",
    "지명": "location",
    "장소": "location",
    "기관": "organization",
    "시간": "time",
    "성분부사|시간부사": "time",
    "수량": "quantity",
}


def quantize_cost(raw_cost: int, lo: int = 100, hi: int = 2000) -> int:
    q = int(round(400 + (raw_cost / 8000.0) * 1400))
    return max(lo, min(hi, q))


def is_hangul_syllables(s: str) -> bool:
    return bool(s) and bool(HANGUL_SYLLABLE_RE.match(s))


def map_sejong_pos(pos: str) -> str:
    if pos in SEJONG_TO_UD:
        return SEJONG_TO_UD[pos]
    for p in pos.split("+"):
        if p in SEJONG_TO_UD:
            return SEJONG_TO_UD[p]
    return "X"


def clean_lemma(morph_surf: str, pos: str) -> str:
    """Convert a mecab-ko-dic morpheme surface + Sejong POS into a clean lemma."""
    base = morph_surf.strip()
    if pos in ("VV", "VA", "VX", "VCP", "VCN", "XSV", "XSA"):
        if base in ("하", "되", "보", "오", "가", "사", "자", "주", "두", "쓰", "타", "나", "지", "서"):
            return base + "다"
        if not base.endswith("다"):
            return base + "다"
    return base


def build_decomp(surface: str, expr: str, fallback_pos: str) -> str:
    """Align `mecab-ko-dic` Inflect.csv `expression` onto `surface` so that
    every piece surface is non-empty and `''.join(piece_surfs) == surface`.
    """
    parts = [p for p in expr.split("+") if p]
    morphemes: List[Tuple[str, str, str]] = []
    for p in parts:
        fields = p.split("/")
        if len(fields) >= 2:
            m_surf, m_pos = fields[0], fields[1]
            ud = map_sejong_pos(m_pos)
            lem = clean_lemma(m_surf, m_pos)
            morphemes.append((m_surf, lem, ud))

    if not morphemes:
        ud = map_sejong_pos(fallback_pos)
        return f"{surface}/{surface}/{ud}"

    n_chars = len(surface)
    if n_chars == 1 or len(morphemes) == 1:
        _, lem0, ud0 = morphemes[0]
        return f"{surface}/{lem0}/{ud0}"

    # Peel exact Hangul-syllable suffixes from the right while leaving >= 1 char for earlier morphemes.
    right_pieces: List[Tuple[str, str, str]] = []
    rem = surface
    m_idx = len(morphemes) - 1
    while m_idx > 0 and len(rem) > 1:
        m_surf, m_lem, m_ud = morphemes[m_idx]
        if is_hangul_syllables(m_surf) and len(rem) > len(m_surf) and rem.endswith(m_surf):
            right_pieces.append((m_surf, m_lem, m_ud))
            rem = rem[: -len(m_surf)]
            m_idx -= 1
        elif m_pos_is_ef_da(m_surf) and len(rem) > 1 and rem.endswith("다"):
            right_pieces.append(("다", "다", "PART"))
            rem = rem[:-1]
            m_idx -= 1
        else:
            break

    # Peel exact Hangul-syllable prefix from the left if there are still >= 2 morphemes in `rem`.
    left_pieces: List[Tuple[str, str, str]] = []
    l_idx = 0
    while l_idx < m_idx and len(rem) > 1:
        m_surf, m_lem, m_ud = morphemes[l_idx]
        if is_hangul_syllables(m_surf) and len(rem) > len(m_surf) and rem.startswith(m_surf):
            left_pieces.append((m_surf, m_lem, m_ud))
            rem = rem[len(m_surf):]
            l_idx += 1
        else:
            break

    # Remaining middle morphemes [l_idx .. m_idx] all fuse into `rem`.
    head_lem = morphemes[l_idx][1]
    head_ud = morphemes[l_idx][2]
    mid_piece = (rem, head_lem, head_ud)

    all_pieces = left_pieces + [mid_piece] + list(reversed(right_pieces))
    decomp = "+".join(f"{s}/{l}/{u}" for s, l, u in all_pieces)
    assert "".join(s for s, _, _ in all_pieces) == surface, (decomp, surface)
    assert all(s for s, _, _ in all_pieces), (decomp, surface)
    return decomp


def m_pos_is_ef_da(m_surf: str) -> bool:
    return m_surf.endswith("다") and len(m_surf) >= 2 and not is_hangul_syllables(m_surf[:-1])


def read_csv_rows(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def build_ko_lexicon(dic_dir: Path) -> List[Tuple[str, str, int, str, str, str, str, str]]:
    # Map surface -> list of candidate records:
    # (cost, ud_pos, lemma, sem, infl, decomp, priority)
    # priority: 0 = curated, 1 = closed/predicate/propn, 2 = NNG/MAG, 3 = Inflect
    candidates: Dict[str, List[Tuple[int, str, str, str, str, str, int]]] = defaultdict(list)

    def add_cand(
        surface: str,
        raw_cost: int,
        ud_pos: str,
        lemma: str = "-",
        sem: str = "",
        infl: str = "",
        decomp: str = "",
        priority: int = 1,
    ) -> None:
        if not is_hangul_syllables(surface):
            return
        candidates[surface].append((quantize_cost(raw_cost), ud_pos, lemma, sem, infl, decomp, priority))

    # 1. Closed-class & predicate CSVs (100% of pure Hangul)
    simple_files = [
        ("NNB.csv", "NOUN", "", "n", 1),
        ("NNBC.csv", "NOUN", "", "n", 1),
        ("NP.csv", "PRON", "", "n", 1),
        ("NR.csv", "NUM", "quantity", "n", 1),
        ("MM.csv", "DET", "", "inv", 1),
        ("MAJ.csv", "CCONJ", "", "inv", 1),
        ("IC.csv", "INTJ", "", "inv", 1),
        ("VCP.csv", "AUX", "", "vx", 1),
        ("VCN.csv", "AUX", "", "vx", 1),
        ("VX.csv", "AUX", "", "vx", 1),
        ("NNP.csv", "PROPN", "", "n", 1),
        ("Group.csv", "PROPN", "organization", "n", 1),
        ("Place-station.csv", "PROPN", "location", "n", 1),
    ]
    for fn, default_ud, default_sem, infl, prio in simple_files:
        for row in read_csv_rows(dic_dir / fn):
            if len(row) < 6:
                continue
            surf = row[0]
            if not is_hangul_syllables(surf):
                continue
            raw_cost = int(row[3])
            sem = default_sem or SEM_MAP.get(row[5], "")
            if default_ud == "AUX" and surf in ("아니", "않", "못하", "말"):
                lemma = surf + "다"
                add_cand(surf, raw_cost, default_ud, lemma, sem, infl, "", prio)
                add_cand(surf + "다", raw_cost + 50, default_ud, "-", sem, infl, "", prio)
            else:
                add_cand(surf, raw_cost, default_ud, "-", sem, infl, "", prio)

    # 2. VV.csv and VA.csv (stems + citation forms `<stem>다`)
    for fn, ud_pos, infl in [("VV.csv", "VERB", "vv"), ("VA.csv", "ADJ", "va")]:
        rows = read_csv_rows(dic_dir / fn)
        stem_set = {r[0] for r in rows if len(r) >= 6 and is_hangul_syllables(r[0])}
        for row in rows:
            if len(row) < 6:
                continue
            surf = row[0]
            if not is_hangul_syllables(surf):
                continue
            raw_cost = int(row[3])
            # Skip obvious irregular ㅂ-allomorphs ending in 워/와 when stem[:-1]+'ㅂ' exists
            is_b_allo = (
                len(surf) >= 2
                and surf[-1] in ("워", "와")
                and chr(ord(surf[-2]) + 17) in stem_set  # rough check; exact check below
            )
            # Exact check for ㅂ-irregular stem: e.g. 가까워 -> 가깝
            b_stem = ""
            if len(surf) >= 2 and surf[-1] in ("워", "와"):
                prev_code = ord(surf[-2])
                if 0xAC00 <= prev_code <= 0xD7A3 and (prev_code - 0xAC00) % 28 == 0:
                    cand_b = surf[:-2] + chr(prev_code + 17)  # + jongseong ㅂ (17)
                    if cand_b in stem_set:
                        b_stem = cand_b
                        is_b_allo = True

            if is_b_allo and b_stem:
                add_cand(surf, raw_cost, ud_pos, b_stem + "다", "", "irr_b", "", 1)
            else:
                add_cand(surf, raw_cost, ud_pos, surf + "다", "", infl, "", 1)
                add_cand(surf + "다", raw_cost + 50, ud_pos, "-", "", infl, "", 1)

    # 3. Top NNG, MAG, Place, Place-address, Person by lowest cost
    top_budgets = [
        ("NNG.csv", "NOUN", "", "n", 18000, 2),
        ("MAG.csv", "ADV", "", "inv", 3000, 2),
        ("Place-address.csv", "PROPN", "location", "n", 2000, 1),
        ("Place.csv", "PROPN", "location", "n", 3000, 1),
        ("Person.csv", "PROPN", "person", "n", 2000, 2),
    ]
    for fn, ud_pos, default_sem, infl, limit, prio in top_budgets:
        best_by_surf: Dict[str, Tuple[int, str]] = {}
        for row in read_csv_rows(dic_dir / fn):
            if len(row) < 6:
                continue
            surf = row[0]
            if not is_hangul_syllables(surf):
                continue
            # Filter noisy 1-char NNGs with very high cost
            raw_cost = int(row[3])
            if fn == "NNG.csv" and len(surf) == 1 and raw_cost > 4500:
                continue
            sem = default_sem or SEM_MAP.get(row[5], "")
            if surf not in best_by_surf or raw_cost < best_by_surf[surf][0]:
                best_by_surf[surf] = (raw_cost, sem)
        ranked = sorted(best_by_surf.items(), key=lambda kv: (kv[1][0], kv[0]))[:limit]
        for surf, (raw_cost, sem) in ranked:
            add_cand(surf, raw_cost, ud_pos, "-", sem, infl, "", prio)

    # 4. Top ~15,000 Inflect.csv pre-analyzed fused/contracted/irregular surface forms
    inflect_best: Dict[str, Tuple[int, str, str, str]] = {}
    for row in read_csv_rows(dic_dir / "Inflect.csv"):
        if len(row) < 12:
            continue
        surf = row[0]
        pos = row[4]
        if not is_hangul_syllables(surf):
            continue
        if pos in ("UNA", "NA") or "UNA" in pos or "NA" in pos:
            continue
        raw_cost = int(row[3])
        first_pos = row[9]
        expr = row[11]
        if surf not in inflect_best or raw_cost < inflect_best[surf][0]:
            inflect_best[surf] = (raw_cost, pos, first_pos, expr)

    ranked_inflect = sorted(inflect_best.items(), key=lambda kv: (kv[1][0], kv[0]))[:15000]
    for surf, (raw_cost, pos, first_pos, expr) in ranked_inflect:
        ud_pos = map_sejong_pos(first_pos if first_pos != "*" else pos)
        decomp = build_decomp(surf, expr, pos)
        # Head lemma is the first piece's lemma in decomp
        first_piece = decomp.split("+")[0]
        head_lem = first_piece.split("/")[1]
        lemma = "-" if head_lem == surf else head_lem
        infl = "vv" if ud_pos == "VERB" else ("va" if ud_pos == "ADJ" else ("vx" if ud_pos == "AUX" else "n"))
        add_cand(surf, raw_cost, ud_pos, lemma, "", infl, decomp, 3)

    # 5. Curated essential high-frequency words & contracted/irregular forms
    curated = [
        # Test sentence & core forms
        ("샀다", 200, "VERB", "사다", "", "vv", "샀/사다/VERB+다/다/PART"),
        ("했다", 150, "VERB", "하다", "", "vv", "했/하다/VERB+다/다/PART"),
        ("됐다", 180, "VERB", "되다", "", "vv", "됐/되다/VERB+다/다/PART"),
        ("왔다", 200, "VERB", "오다", "", "vv", "왔/오다/VERB+다/다/PART"),
        ("갔다", 200, "VERB", "가다", "", "vv", "갔/가다/VERB+다/다/PART"),
        ("봤다", 200, "VERB", "보다", "", "vv", "봤/보다/VERB+다/다/PART"),
        ("먹었다", 220, "VERB", "먹다", "", "vv", "먹/먹다/VERB+었/었/PART+다/다/PART"),
        ("열렸다", 240, "VERB", "열리다", "", "vv", "열렸/열리다/VERB+다/다/PART"),
        ("아름다운", 250, "ADJ", "아름답다", "", "irr_b", "아름다/아름답다/ADJ+운/ㄴ/PART"),
        ("예쁜", 250, "ADJ", "예쁘다", "", "irr_eu", "예/예쁘다/ADJ+쁜/ㄴ/PART"),
        ("같은", 180, "ADJ", "같다", "", "va", "같/같다/ADJ+은/은/PART"),
        ("있는", 150, "ADJ", "있다", "", "va", "있/있다/ADJ+는/는/PART"),
        ("없는", 160, "ADJ", "없다", "", "va", "없/없다/ADJ+는/는/PART"),
        ("하는", 150, "VERB", "하다", "", "vv", "하/하다/VERB+는/는/PART"),
        ("되는", 180, "VERB", "되다", "", "vv", "되/되다/VERB+는/는/PART"),
        ("할", 160, "VERB", "하다", "", "vv", "할/하다/VERB"),
        ("될", 180, "VERB", "되다", "", "vv", "될/되다/VERB"),
        ("인", 180, "AUX", "이다", "", "vx", "인/이다/AUX"),
        ("아닌", 200, "AUX", "아니다", "", "vx", "아/아니다/AUX+닌/ㄴ/PART"),
        ("내", 200, "PRON", "나", "", "n", "내/나/PRON"),
        ("네", 220, "PRON", "너", "", "n", "네/너/PRON"),
        ("제", 200, "PRON", "저", "", "n", "제/저/PRON"),
        ("뭘", 240, "PRON", "무엇", "", "n", "뭘/무엇/PRON"),
        ("건", 220, "NOUN", "것", "", "n", "건/것/NOUN"),
        ("걸", 220, "NOUN", "것", "", "n", "걸/것/NOUN"),
        ("서울역", 300, "PROPN", "-", "location", "n", ""),
        ("서울", 200, "PROPN", "-", "location", "n", ""),
        ("오후", 250, "NOUN", "-", "time", "n", ""),
        ("오전", 250, "NOUN", "-", "time", "n", ""),
        ("회의", 250, "NOUN", "-", "", "n", ""),
        ("책", 220, "NOUN", "-", "", "n", ""),
        ("권", 250, "NOUN", "-", "", "n", ""),
        ("안", 200, "ADV", "-", "", "inv", ""),
        ("못", 200, "ADV", "-", "", "inv", ""),
        ("더", 180, "ADV", "-", "", "inv", ""),
        ("잘", 180, "ADV", "-", "", "inv", ""),
        ("가장", 200, "ADV", "-", "", "inv", ""),
        ("매우", 220, "ADV", "-", "", "inv", ""),
        ("아주", 220, "ADV", "-", "", "inv", ""),
        ("너무", 200, "ADV", "-", "", "inv", ""),
        ("그러나", 180, "CCONJ", "-", "", "inv", ""),
        ("그리고", 160, "CCONJ", "-", "", "inv", ""),
        ("하지만", 180, "CCONJ", "-", "", "inv", ""),
        ("그래서", 180, "CCONJ", "-", "", "inv", ""),
        ("안녕하세요", 220, "INTJ", "-", "", "inv", ""),
        ("아니요", 240, "INTJ", "-", "", "inv", ""),
    ]
    for surf, cost, ud_pos, lemma, sem, infl, decomp in curated:
        # Store curated directly with priority 0 (raw_cost adjusted so quantize_cost isn't needed or pass raw)
        candidates[surf].append((cost, ud_pos, lemma, sem, infl, decomp, 0))

    # Resolve each surface into a single canonical row
    rows_out: List[Tuple[str, str, int, str, str, str, str, str]] = []
    for surf, cands in candidates.items():
        # Sort by (priority, cost)
        cands_sorted = sorted(cands, key=lambda x: (x[6], x[0]))
        best_cost, best_ud, best_lemma, best_sem, best_infl, best_decomp, best_prio = cands_sorted[0]

        # Collect UPOS tags from competitive candidates (priority <= 1 or cost within +300 of best)
        upos_set: Set[str] = {best_ud}
        for c_cost, c_ud, _, _, _, _, c_prio in cands_sorted[1:]:
            if best_prio == 0 and c_prio > 1:
                continue
            if c_prio <= 1 and c_cost <= best_cost + 450:
                upos_set.add(c_ud)
            elif c_cost <= best_cost + 200:
                upos_set.add(c_ud)

        # Explicit homograph guard for key function words
        if surf == "이":
            upos_set.update({"DET", "NOUN", "NUM"})
        elif surf in ("그", "저"):
            upos_set.update({"DET", "PRON"})
        elif surf in ("새",):
            upos_set.update({"DET", "NOUN"})
        elif surf in ("네", "제"):
            upos_set.update({"DET", "PRON"})

        if len(upos_set) == 1:
            decided_upos = best_ud
            upos_mask = best_ud
        else:
            decided_upos = "-"
            upos_mask = "|".join(sorted(upos_set))

        assert all(t in UD_TAGS for t in upos_mask.split("|")), (surf, upos_mask)
        if decided_upos != "-":
            assert decided_upos == upos_mask, (surf, decided_upos, upos_mask)

        meta_parts: List[str] = []
        if best_sem:
            meta_parts.append(f"sem={best_sem}")
        if best_infl:
            meta_parts.append(f"infl={best_infl}")
        if best_decomp:
            pieces = [p.split("/")[0] for p in best_decomp.split("+")]
            assert "".join(pieces) == surf and all(pieces), (surf, best_decomp)
            meta_parts.append(f"decomp={best_decomp}")
        meta = ";".join(meta_parts) if meta_parts else "-"

        rows_out.append((surf, "ko", best_cost, decided_upos, upos_mask, best_lemma, "-", meta))

    # Sort strictly by UTF-8 byte order of (surface, variant)
    rows_out.sort(key=lambda r: (r[0].encode("utf-8"), r[1].encode("utf-8")))
    return rows_out


def build_ko_morph() -> List[Tuple[str, str, str, int, str, str]]:
    """Build curated 6-column Josa, Copula, XSV, XSA, and Eomi suffix table."""
    entries: List[Tuple[str, str, str, int, str, str]] = [
        # --- JOSA (Case & topic/focus particles) ---
        ("이", "josa", "batchim", 140, "KEEP_STEM", "이/이/ADP"),
        ("가", "josa", "no_batchim", 140, "KEEP_STEM", "가/가/ADP"),
        ("은", "josa", "batchim", 130, "KEEP_STEM", "은/은/PART"),
        ("는", "josa", "no_batchim", 130, "KEEP_STEM", "는/는/PART"),
        ("을", "josa", "batchim", 120, "KEEP_STEM", "을/을/ADP"),
        ("를", "josa", "no_batchim", 120, "KEEP_STEM", "를/를/ADP"),
        ("의", "josa", "any", 125, "KEEP_STEM", "의/의/ADP"),
        ("에", "josa", "any", 130, "KEEP_STEM", "에/에/ADP"),
        ("에서", "josa", "any", 135, "KEEP_STEM", "에서/에서/ADP"),
        ("에서는", "josa", "any", 145, "KEEP_STEM", "에서/에서/ADP+는/는/PART"),
        ("에서도", "josa", "any", 150, "KEEP_STEM", "에서/에서/ADP+도/도/PART"),
        ("에서만", "josa", "any", 155, "KEEP_STEM", "에서/에서/ADP+만/만/PART"),
        ("에서부터", "josa", "any", 160, "KEEP_STEM", "에서/에서/ADP+부터/부터/ADP"),
        ("에는", "josa", "any", 145, "KEEP_STEM", "에/에/ADP+는/는/PART"),
        ("에도", "josa", "any", 145, "KEEP_STEM", "에/에/ADP+도/도/PART"),
        ("에만", "josa", "any", 150, "KEEP_STEM", "에/에/ADP+만/만/PART"),
        ("에게", "josa", "any", 145, "KEEP_STEM", "에게/에게/ADP"),
        ("에게는", "josa", "any", 155, "KEEP_STEM", "에게/에게/ADP+는/는/PART"),
        ("에게도", "josa", "any", 155, "KEEP_STEM", "에게/에게/ADP+도/도/PART"),
        ("에게서", "josa", "any", 160, "KEEP_STEM", "에게서/에게서/ADP"),
        ("한테", "josa", "any", 165, "KEEP_STEM", "한테/한테/ADP"),
        ("한테서", "josa", "any", 175, "KEEP_STEM", "한테서/한테서/ADP"),
        ("께", "josa", "any", 170, "KEEP_STEM", "께/께/ADP"),
        ("께서", "josa", "any", 175, "KEEP_STEM", "께서/께서/ADP"),
        ("으로", "josa", "batchim", 140, "KEEP_STEM", "으로/으로/ADP"),
        ("으로는", "josa", "batchim", 155, "KEEP_STEM", "으로/으로/ADP+는/는/PART"),
        ("으로도", "josa", "batchim", 155, "KEEP_STEM", "으로/으로/ADP+도/도/PART"),
        ("으로서", "josa", "batchim", 165, "KEEP_STEM", "으로서/으로서/ADP"),
        ("으로써", "josa", "batchim", 165, "KEEP_STEM", "으로써/으로써/ADP"),
        ("으로부터", "josa", "batchim", 170, "KEEP_STEM", "으로/으로/ADP+부터/부터/ADP"),
        ("로", "josa", "rieul_or_no_batchim", 140, "KEEP_STEM", "로/로/ADP"),
        ("로는", "josa", "rieul_or_no_batchim", 155, "KEEP_STEM", "로/로/ADP+는/는/PART"),
        ("로도", "josa", "rieul_or_no_batchim", 155, "KEEP_STEM", "로/로/ADP+도/도/PART"),
        ("로서", "josa", "rieul_or_no_batchim", 165, "KEEP_STEM", "로서/로서/ADP"),
        ("로써", "josa", "rieul_or_no_batchim", 165, "KEEP_STEM", "로써/로써/ADP"),
        ("로부터", "josa", "rieul_or_no_batchim", 170, "KEEP_STEM", "로/로/ADP+부터/부터/ADP"),
        ("와", "josa", "no_batchim", 150, "KEEP_STEM", "와/와/CCONJ"),
        ("과", "josa", "batchim", 150, "KEEP_STEM", "과/과/CCONJ"),
        ("랑", "josa", "no_batchim", 170, "KEEP_STEM", "랑/랑/CCONJ"),
        ("이랑", "josa", "batchim", 170, "KEEP_STEM", "이랑/이랑/CCONJ"),
        ("하고", "josa", "any", 170, "KEEP_STEM", "하고/하고/CCONJ"),
        ("도", "josa", "any", 135, "KEEP_STEM", "도/도/PART"),
        ("만", "josa", "any", 140, "KEEP_STEM", "만/만/PART"),
        ("까지", "josa", "any", 145, "KEEP_STEM", "까지/까지/ADP"),
        ("까지는", "josa", "any", 160, "KEEP_STEM", "까지/까지/ADP+는/는/PART"),
        ("까지도", "josa", "any", 160, "KEEP_STEM", "까지/까지/ADP+도/도/PART"),
        ("부터", "josa", "any", 145, "KEEP_STEM", "부터/부터/ADP"),
        ("부터는", "josa", "any", 160, "KEEP_STEM", "부터/부터/ADP+는/는/PART"),
        ("마저", "josa", "any", 180, "KEEP_STEM", "마저/마저/PART"),
        ("조차", "josa", "any", 180, "KEEP_STEM", "조차/조차/PART"),
        ("조차도", "josa", "any", 190, "KEEP_STEM", "조차/조차/PART+도/도/PART"),
        ("밖에", "josa", "any", 175, "KEEP_STEM", "밖에/밖에/PART"),
        ("처럼", "josa", "any", 160, "KEEP_STEM", "처럼/처럼/ADP"),
        ("만큼", "josa", "any", 165, "KEEP_STEM", "만큼/만큼/ADP"),
        ("보다", "josa", "any", 155, "KEEP_STEM", "보다/보다/ADP"),
        ("마다", "josa", "any", 165, "KEEP_STEM", "마다/마다/PART"),
        ("이나", "josa", "batchim", 165, "KEEP_STEM", "이나/이나/CCONJ"),
        ("나", "josa", "no_batchim", 165, "KEEP_STEM", "나/나/CCONJ"),
        ("이라도", "josa", "batchim", 175, "KEEP_STEM", "이라도/이라도/PART"),
        ("라도", "josa", "no_batchim", 175, "KEEP_STEM", "라도/라도/PART"),
        ("이든지", "josa", "batchim", 185, "KEEP_STEM", "이든지/이든지/PART"),
        ("든지", "josa", "no_batchim", 185, "KEEP_STEM", "든지/든지/PART"),

        # --- COP (이다 copula forms on nominals) ---
        ("이다", "cop", "any", 160, "KEEP_STEM", "이/이다/AUX+다/다/PART"),
        ("입니다", "cop", "any", 165, "KEEP_STEM", "입/이다/AUX+니다/니다/PART"),
        ("이었다", "cop", "batchim", 170, "KEEP_STEM", "이었/이다/AUX+다/다/PART"),
        ("였다", "cop", "no_batchim", 170, "KEEP_STEM", "였/이다/AUX+다/다/PART"),
        ("인", "cop", "any", 175, "KEEP_STEM", "인/이다/AUX"),
        ("이고", "cop", "any", 175, "KEEP_STEM", "이/이다/AUX+고/고/SCONJ"),
        ("이며", "cop", "any", 180, "KEEP_STEM", "이/이다/AUX+며/며/SCONJ"),
        ("이면", "cop", "any", 180, "KEEP_STEM", "이/이다/AUX+면/면/SCONJ"),
        ("이지만", "cop", "any", 180, "KEEP_STEM", "이/이다/AUX+지만/지만/SCONJ"),
        ("이야", "cop", "batchim", 185, "KEEP_STEM", "이/이다/AUX+야/야/PART"),
        ("야", "cop", "no_batchim", 185, "KEEP_STEM", "야/이다/AUX"),

        # --- XSV (하다 / 되다 / 시키다 verbalizers on verbal nouns) ---
        ("하다", "xsv", "any", 150, "VERB", "하/하다/VERB+다/다/PART"),
        ("한다", "xsv", "any", 155, "VERB", "한/하다/VERB+다/다/PART"),
        ("했다", "xsv", "any", 155, "VERB", "했/하다/VERB+다/다/PART"),
        ("하고", "xsv", "any", 160, "VERB", "하/하다/VERB+고/고/SCONJ"),
        ("하며", "xsv", "any", 165, "VERB", "하/하다/VERB+며/며/SCONJ"),
        ("하면", "xsv", "any", 165, "VERB", "하/하다/VERB+면/면/SCONJ"),
        ("하여", "xsv", "any", 170, "VERB", "하/하다/VERB+여/여/SCONJ"),
        ("해", "xsv", "any", 165, "VERB", "해/하다/VERB"),
        ("해서", "xsv", "any", 165, "VERB", "해/하다/VERB+서/서/SCONJ"),
        ("하는", "xsv", "any", 155, "VERB", "하/하다/VERB+는/는/PART"),
        ("한", "xsv", "any", 165, "VERB", "한/하다/VERB"),
        ("할", "xsv", "any", 165, "VERB", "할/하다/VERB"),
        ("합니다", "xsv", "any", 160, "VERB", "합/하다/VERB+니다/니다/PART"),
        ("했습니다", "xsv", "any", 160, "VERB", "했/하다/VERB+습니다/습니다/PART"),
        ("되다", "xsv", "any", 160, "VERB", "되/되다/VERB+다/다/PART"),
        ("된다", "xsv", "any", 160, "VERB", "된/되다/VERB+다/다/PART"),
        ("됐다", "xsv", "any", 165, "VERB", "됐/되다/VERB+다/다/PART"),
        ("되었다", "xsv", "any", 165, "VERB", "되/되다/VERB+었/었/PART+다/다/PART"),
        ("되고", "xsv", "any", 165, "VERB", "되/되다/VERB+고/고/SCONJ"),
        ("되면", "xsv", "any", 165, "VERB", "되/되다/VERB+면/면/SCONJ"),
        ("되는", "xsv", "any", 160, "VERB", "되/되다/VERB+는/는/PART"),
        ("된", "xsv", "any", 165, "VERB", "된/되다/VERB"),
        ("될", "xsv", "any", 165, "VERB", "될/되다/VERB"),
        ("됩니다", "xsv", "any", 165, "VERB", "됩/되다/VERB+니다/니다/PART"),
        ("시키다", "xsv", "any", 180, "VERB", "시키/시키다/VERB+다/다/PART"),
        ("시켰다", "xsv", "any", 185, "VERB", "시켰/시키다/VERB+다/다/PART"),

        # --- XSA (스럽다 / 롭다 / 답다 adjectivizers) ---
        ("스럽다", "xsa", "any", 180, "ADJ", "스럽/스럽다/ADJ+다/다/PART"),
        ("스러운", "xsa", "any", 180, "ADJ", "스러/스럽다/ADJ+운/ㄴ/PART"),
        ("스럽게", "xsa", "any", 185, "ADJ", "스럽/스럽다/ADJ+게/게/SCONJ"),
        ("롭다", "xsa", "any", 185, "ADJ", "롭/롭다/ADJ+다/다/PART"),
        ("로운", "xsa", "any", 185, "ADJ", "로/롭다/ADJ+운/ㄴ/PART"),
        ("롭게", "xsa", "any", 190, "ADJ", "롭/롭다/ADJ+게/게/SCONJ"),
        ("답다", "xsa", "any", 185, "ADJ", "답/답다/ADJ+다/다/PART"),
        ("다운", "xsa", "any", 185, "ADJ", "다/답다/ADJ+운/ㄴ/PART"),

        # --- EOMI (Verbal / Adjectival endings) ---
        ("다", "eomi", "any", 140, "KEEP_STEM", "다/다/PART"),
        ("는다", "eomi", "batchim", 145, "KEEP_STEM", "는/는/PART+다/다/PART"),
        ("았다", "eomi", "any", 150, "KEEP_STEM", "았/았/PART+다/다/PART"),
        ("었다", "eomi", "any", 150, "KEEP_STEM", "었/었/PART+다/다/PART"),
        ("겠다", "eomi", "any", 155, "KEEP_STEM", "겠/겠/PART+다/다/PART"),
        ("시었다", "eomi", "any", 170, "KEEP_STEM", "시/시/PART+었/었/PART+다/다/PART"),
        ("셨다", "eomi", "any", 160, "KEEP_STEM", "셨/시/PART+다/다/PART"),
        ("고", "eomi", "any", 145, "KEEP_STEM", "고/고/SCONJ"),
        ("며", "eomi", "any", 155, "KEEP_STEM", "며/며/SCONJ"),
        ("면서", "eomi", "any", 160, "KEEP_STEM", "면서/면서/SCONJ"),
        ("어서", "eomi", "any", 150, "KEEP_STEM", "어서/어서/SCONJ"),
        ("아서", "eomi", "any", 150, "KEEP_STEM", "아서/아서/SCONJ"),
        ("니까", "eomi", "no_batchim", 160, "KEEP_STEM", "니까/니까/SCONJ"),
        ("으니까", "eomi", "batchim", 160, "KEEP_STEM", "으니까/으니까/SCONJ"),
        ("지만", "eomi", "any", 150, "KEEP_STEM", "지만/지만/SCONJ"),
        ("는데", "eomi", "any", 150, "KEEP_STEM", "는데/는데/SCONJ"),
        ("은데", "eomi", "batchim", 155, "KEEP_STEM", "은데/은데/SCONJ"),
        ("면", "eomi", "no_batchim", 150, "KEEP_STEM", "면/면/SCONJ"),
        ("으면", "eomi", "batchim", 150, "KEEP_STEM", "으면/으면/SCONJ"),
        ("자", "eomi", "any", 165, "KEEP_STEM", "자/자/PART"),
        ("어", "eomi", "any", 160, "KEEP_STEM", "어/어/PART"),
        ("아", "eomi", "any", 160, "KEEP_STEM", "아/아/PART"),
        ("지", "eomi", "any", 155, "KEEP_STEM", "지/지/PART"),
        ("네", "eomi", "any", 165, "KEEP_STEM", "네/네/PART"),
        ("구나", "eomi", "any", 170, "KEEP_STEM", "구나/구나/PART"),
        ("군", "eomi", "any", 170, "KEEP_STEM", "군/군/PART"),
        ("습니다", "eomi", "batchim", 145, "KEEP_STEM", "습니다/습니다/PART"),
        ("어요", "eomi", "any", 155, "KEEP_STEM", "어요/어요/PART"),
        ("아요", "eomi", "any", 155, "KEEP_STEM", "아요/아요/PART"),
        ("기", "eomi", "any", 160, "NOUN", "기/기/PART"),
        ("음", "eomi", "no_batchim", 170, "NOUN", "음/음/PART"),
        ("은", "eomi", "batchim", 150, "KEEP_STEM", "은/은/PART"),
        ("는", "eomi", "any", 145, "KEEP_STEM", "는/는/PART"),
        ("을", "eomi", "batchim", 150, "KEEP_STEM", "을/을/PART"),
    ]
    for surf, kind, cond, cost, effect, decomp in entries:
        pieces = [p.split("/")[0] for p in decomp.split("+")]
        assert "".join(pieces) == surf and all(pieces), (surf, decomp)
        assert 100 <= cost <= 1000, (surf, cost)
    entries.sort(key=lambda r: (r[0].encode("utf-8"), r[1].encode("utf-8"), r[2].encode("utf-8")))
    return entries


def write_tsv_atomic(path: Path, tmp_path: Path, header_lines: List[str], rows: List[Tuple]) -> None:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for line in header_lines:
            f.write(line + "\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Korean lexicon TSVs from mecab-ko-dic.")
    parser.add_argument("--dic-dir", default=DEFAULT_DIC_DIR, help="Path to extracted mecab-ko-dic directory")
    parser.add_argument("--scratch-dir", default=DEFAULT_SCRATCH_DIR, help="Scratch directory for atomic staging")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]), help="Repository root")
    args = parser.parse_args()

    dic_dir = Path(args.dic_dir)
    scratch_dir = Path(args.scratch_dir)
    repo_root = Path(args.repo_root)

    if not (dic_dir / "NNG.csv").exists():
        print(f"ERROR: mecab-ko-dic CSVs not found in {dic_dir}", file=sys.stderr)
        return 1

    ko_rows = build_ko_lexicon(dic_dir)
    morph_rows = build_ko_morph()

    ko_header = [
        "# Korean (ko) Morphological & Lexical Dictionary — Apache-2.0 (mecab-ko-dic-2.1.1-20180720)",
        "# surface\tvariant\tcost\tdecided_upos\tupos_mask\tlemma\tcounterpart\tmeta",
    ]
    morph_header = [
        "# Korean (ko) Josa, Copula, XSV/XSA & Eomi Suffix Table — Apache-2.0",
        "# surface_suffix\tkind\tcond\tcost\teojeol_upos_effect\tdecomp",
    ]

    nlp_crate_root = repo_root / "crates" / "faberline-nlp"
    ko_out = nlp_crate_root / "lexicon" / "data" / "ko.tsv"
    morph_out = nlp_crate_root / "lexicon" / "data" / "ko_morph.tsv"

    write_tsv_atomic(ko_out, scratch_dir / "ko.tsv.tmp", ko_header, ko_rows)
    write_tsv_atomic(morph_out, scratch_dir / "ko_morph.tsv.tmp", morph_header, morph_rows)

    print(f"Wrote {len(ko_rows)} rows to {ko_out} ({ko_out.stat().st_size} bytes)")
    print(f"Wrote {len(morph_rows)} rows to {morph_out} ({morph_out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
