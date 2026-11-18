#!/usr/bin/env python3
"""Build lexicon/data/en.tsv from Princeton WordNet 3.1, SCOWL/VarCon,
src/doc/pos.rs, and allowlisted Public Domain Project Gutenberg English texts.

Strictly permissive commercial-safe sources only:
- Princeton WordNet 3.1 (WordNet BSD-style license)
- SCOWL / VarCon (MIT/BSD-like permissive license)
- In-repo src/doc/pos.rs & gold sets (first-party MIT/CC0)
- Public Domain Project Gutenberg (12 books, 660 docs)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

NLP_CRATE_ROOT = REPO_ROOT / "crates" / "faberline-nlp"
POS_RS_PATH = NLP_CRATE_ROOT / "src" / "doc" / "pos.rs"
POS_GOLD_PATH = NLP_CRATE_ROOT / "tests" / "data" / "pos_gold.json"
GOLD_LAYERS_PATH = NLP_CRATE_ROOT / "tests" / "data" / "gold_layers4_6_v1.jsonl"
OUT_TSV_PATH = NLP_CRATE_ROOT / "lexicon" / "data" / "en.tsv"

ALLOWED_PG_IDS = frozenset(
    {"76", "2701", "64317", "541", "16643", "205", "5116", "408", "575", "5827", "130", "470"}
)

WORD_RE = re.compile(r"^[a-z]+(?:[-'][a-z]+)*$")
CLITICS = {"n't", "'ll", "'ve", "'re", "'m", "'s", "'d"}

WN_POS_TO_UD = {
    "noun": "NOUN",
    "verb": "VERB",
    "adj": "ADJ",
    "adv": "ADV",
}

LEX_FILENUM_TO_SEM = {
    18: "person",        # noun.person
    15: "location",      # noun.location
    14: "organization",  # noun.group
    28: "time",          # noun.time
    23: "quantity",      # noun.quantity
    6: "artifact",       # noun.artifact
}

CLOSED_LEMMAS = {
    "is": "be", "am": "be", "are": "be", "was": "be", "were": "be",
    "been": "be", "being": "be", "'re": "be", "'m": "be",
    "has": "have", "had": "have", "having": "have",
    "does": "do", "did": "do", "doing": "do", "done": "do",
    "n't": "not", "'ll": "will", "'ve": "have",
    "me": "i", "my": "i", "myself": "i",
    "him": "he", "his": "he", "himself": "he",
    "her": "she", "hers": "she", "herself": "she",
    "its": "it", "itself": "it",
    "us": "we", "our": "we", "ours": "we", "ourselves": "we",
    "them": "they", "their": "they", "theirs": "they", "themselves": "they",
    "your": "you", "yours": "you", "yourself": "you", "yourselves": "you",
    "whom": "who", "whose": "who",
    "these": "this", "those": "that",
    "an": "a",
}

TARGET_MIN = 50_000
TARGET_MAX = 70_000
TARGET_GOAL = 66_000

# First-party reviewed POS facts; an empty set withholds an unsupported
# inferred assignment while still allowing independently sourced exact POS.
REVIEWED_POS_EVIDENCE: dict[str, frozenset[str]] = {
    "weftwise": frozenset({"ADJ", "ADV"}),
    "weftwize": frozenset(),
}

# These VarCon pairs connect different substances, not spelling variants.
# Retain the source's variant tags while withholding the incorrect links.
REVIEWED_BAD_COUNTERPARTS: frozenset[tuple[str, str]] = frozenset({
    ("ceratin", "creatin"),
    ("creatin", "ceratin"),
    ("ceratins", "creatins"),
    ("creatins", "ceratins"),
})


def remove_reviewed_bad_counterparts(
    variants: dict[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    """Withhold only reviewed incorrect links, preserving their variant tags."""
    return {
        word: (variant, "-" if (word, counterpart) in REVIEWED_BAD_COUNTERPARTS else counterpart)
        for word, (variant, counterpart) in variants.items()
    }


def apply_reviewed_pos_evidence(
    word: str,
    direct_pos: set[str],
    evidence: dict[str, frozenset[str]],
) -> set[str] | None:
    """Return reviewed and direct POS, or None when no review applies."""
    if word not in evidence:
        return None
    return set(evidence[word]) | direct_pos


def is_valid_token(w: str) -> bool:
    return bool(WORD_RE.match(w) or w in CLITICS)


def parse_pos_rs() -> tuple[dict[str, str], set[str]]:
    """Extract EN_DECIDED and EN_AMBIGUOUS directly from src/doc/pos.rs."""
    text = POS_RS_PATH.read_text(encoding="utf-8")

    decided: dict[str, str] = {}
    m_dec = re.search(
        r"const EN_DECIDED:\s*&\[\(Upos,\s*&str\)\]\s*=\s*&\[(.*?)\];",
        text,
        re.DOTALL,
    )
    if not m_dec:
        raise RuntimeError("Could not find EN_DECIDED in src/doc/pos.rs")
    for upos_raw, words_str in re.findall(
        r'\(\s*Upos::(\w+),\s*"([^"]+)"\s*,?\s*\)', m_dec.group(1), re.DOTALL
    ):
        upos = upos_raw.upper()
        for w in words_str.split():
            if w != "&":
                decided[w] = upos

    ambiguous: set[str] = set()
    m_amb = re.search(
        r"const EN_AMBIGUOUS:\s*&\[&str\]\s*=\s*&\[(.*?)\];",
        text,
        re.DOTALL,
    )
    if not m_amb:
        raise RuntimeError("Could not find EN_AMBIGUOUS in src/doc/pos.rs")
    for words_str in re.findall(r'"([^"]+)"', m_amb.group(1)):
        for w in words_str.split():
            ambiguous.add(w)

    return decided, ambiguous


def load_gold_upos() -> dict[str, set[str]]:
    """Load all English token -> set of gold UPOS tags from test gold files."""
    gold: dict[str, set[str]] = defaultdict(set)

    pos_gold = json.loads(POS_GOLD_PATH.read_text(encoding="utf-8"))
    for item in pos_gold:
        if item.get("lang") == "en":
            for tok, upos in item["tokens"]:
                norm = tok.lower().replace("\u2019", "'")
                if is_valid_token(norm):
                    gold[norm].add(upos)

    with GOLD_LAYERS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("lang") == "en":
                for tok, upos in zip(d["tokens"], d["upos"]):
                    norm = tok.lower().replace("\u2019", "'")
                    if is_valid_token(norm):
                        gold[norm].add(upos)

    return gold


def load_corpus_freqs(
    corpus_path: Path | None, gutenberg_dir: Path | None
) -> tuple[Counter[str], int]:
    """Count token frequencies strictly from the 12 allowlisted Project Gutenberg PD books."""
    counts: Counter[str] = Counter()
    docs = 0
    tok_re = re.compile(r"[a-z]+(?:[-'][a-z]+)*|n't|'[a-z]+")
    if corpus_path is not None:
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                src = d.get("source", "")
                if not src.startswith("gutenberg:"):
                    continue
                parts = src.split(":", 2)
                if len(parts) < 2 or parts[1] not in ALLOWED_PG_IDS:
                    continue
                docs += 1
                text = d.get("text", "").lower().replace("\u2019", "'")
                for m in tok_re.finditer(text):
                    w = m.group(0)
                    if is_valid_token(w):
                        counts[w] += 1
    elif gutenberg_dir is not None:
        for pg_id in sorted(ALLOWED_PG_IDS):
            pg_file = gutenberg_dir / f"pg{pg_id}.txt"
            if not pg_file.exists():
                raise FileNotFoundError(f"Missing allowlisted Gutenberg file: {pg_file}")
            docs += 1
            text = pg_file.read_text(encoding="utf-8", errors="replace").lower().replace("\u2019", "'")
            for m in tok_re.finditer(text):
                w = m.group(0)
                if is_valid_token(w):
                    counts[w] += 1
    return counts, docs


def load_cntlist(wn_dict_dir: Path) -> Counter[str]:
    """Load WordNet cntlist frequencies per lemma."""
    counts: Counter[str] = Counter()
    cnt_path = wn_dict_dir / "cntlist"
    if not cnt_path.exists():
        return counts
    with cnt_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                cnt = int(parts[0])
            except ValueError:
                continue
            lemma = parts[1].split("%", 1)[0].lower()
            if is_valid_token(lemma):
                counts[lemma] += cnt
    return counts


def load_noun_lex_filenums(wn_dict_dir: Path) -> tuple[dict[str, int], set[str]]:
    """Read data.noun to map synset_offset -> lex_filenum and detect instance-only nouns."""
    offset_to_lex: dict[str, int] = {}
    instance_offsets: set[str] = set()
    data_noun = wn_dict_dir / "data.noun"
    with data_noun.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("  "):
                continue
            head = line.split(" | ", 1)[0]
            parts = head.split()
            if len(parts) < 4:
                continue
            offset = parts[0]
            lex_filenum = int(parts[1])
            offset_to_lex[offset] = lex_filenum
            if "@i" in parts:
                instance_offsets.add(offset)
    return offset_to_lex, instance_offsets


def load_wordnet(
    wn_dict_dir: Path, offset_to_lex: dict[str, int], instance_offsets: set[str]
) -> tuple[
    dict[str, set[str]],
    dict[str, int],
    dict[str, str],
    set[str],
    dict[str, dict[str, list[str]]],
    set[str],
]:
    """Load WordNet index.{noun,verb,adj,adv} and *.exc files."""
    wn_pos: dict[str, set[str]] = defaultdict(set)
    wn_senses: dict[str, int] = Counter()
    wn_sem: dict[str, str] = {}
    instance_only_nouns: set[str] = set()

    for pos_name, ud_tag in WN_POS_TO_UD.items():
        idx_path = wn_dict_dir / f"index.{pos_name}"
        with idx_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("  "):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                word = parts[0]
                if not is_valid_token(word):
                    continue
                synset_cnt = int(parts[2])
                offsets = parts[-synset_cnt:]
                wn_pos[word].add(ud_tag)
                wn_senses[word] += synset_cnt

                if pos_name == "noun" and offsets:
                    if all(off in instance_offsets for off in offsets):
                        instance_only_nouns.add(word)
                    sem_counts: Counter[str] = Counter()
                    first_sem = LEX_FILENUM_TO_SEM.get(offset_to_lex.get(offsets[0], -1))
                    for off in offsets:
                        lf = offset_to_lex.get(off, -1)
                        sc = LEX_FILENUM_TO_SEM.get(lf)
                        if sc:
                            sem_counts[sc] += 1
                    if len(sem_counts) == 1 and sum(sem_counts.values()) == len(offsets):
                        wn_sem[word] = next(iter(sem_counts))
                    elif (
                        first_sem is not None
                        and sem_counts[first_sem] >= len(offsets) / 2.0
                        and len(sem_counts) == 1
                    ):
                        wn_sem[word] = first_sem

    # Load irregular morphology from *.exc
    # Order matters: adj before adv so better/best -> good (not well)
    exc_map: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    irr_bases: set[str] = set()
    for pos_name in ["noun", "verb", "adj", "adv"]:
        ud_tag = WN_POS_TO_UD[pos_name]
        exc_path = wn_dict_dir / f"{pos_name}.exc"
        with exc_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                inflected = parts[0]
                if not is_valid_token(inflected):
                    continue
                for base in parts[1:]:
                    if is_valid_token(base):
                        exc_map[inflected][ud_tag].append(base)
                        irr_bases.add(base)

    return wn_pos, wn_senses, wn_sem, instance_only_nouns, exc_map, irr_bases


def load_varcon(varcon_path: Path) -> tuple[dict[str, tuple[str, str]], set[str]]:
    """Parse varcon.txt for clean US<->GB pairs.

    Returns:
      - variant_map: word -> (variant_tag, counterpart)
      - shared_words: words marked A and B on the same varcon line
    """
    raw_pairs: list[tuple[str, str]] = []
    shared_words: set[str] = set()

    with varcon_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            line = line.split("|", 1)[0].strip()
            parts = [p.strip() for p in line.split(" / ")]
            us_words: list[str] = []
            gb_words: list[str] = []
            for p in parts:
                if ": " not in p:
                    continue
                tags_s, w = p.split(": ", 1)
                w = w.strip()
                if not is_valid_token(w) or w.endswith("'s"):
                    continue
                tags = set(tags_s.split())
                if "A" in tags and "B" in tags:
                    shared_words.add(w)
                elif "A" in tags and "B" not in tags:
                    us_words.append(w)
                elif "B" in tags and "A" not in tags:
                    gb_words.append(w)
            if us_words and gb_words and us_words[0] != gb_words[0]:
                raw_pairs.append((us_words[0], gb_words[0]))

    variant_map: dict[str, tuple[str, str]] = {}
    for us_w, gb_w in raw_pairs:
        if us_w not in shared_words and gb_w not in shared_words:
            if us_w not in variant_map and gb_w not in variant_map:
                variant_map[us_w] = ("en-US", gb_w)
                variant_map[gb_w] = ("en-GB", us_w)
        elif us_w in shared_words and gb_w not in shared_words:
            if gb_w not in variant_map:
                variant_map[gb_w] = ("en-GB", us_w)

    return variant_map, shared_words


def infer_regular_lemma(
    word: str, known_bases: dict[str, set[str]]
) -> tuple[str | None, set[str]]:
    """Infer base lemma and possible UPOS for inflected forms (e.g. VarCon plurals/past)."""
    candidates: list[tuple[str, set[str]]] = []

    def check(base: str, allowed_pos: set[str]) -> None:
        if base in known_bases:
            overlap = known_bases[base] & allowed_pos
            # An -ising form can coincide with a noun in -sis after stripping
            # -ing. That spelling overlap does not establish a noun lemma.
            if word.endswith(("ising", "izing")) and base.endswith("sis") and overlap == {"NOUN"}:
                return
            if overlap:
                candidates.append((base, overlap))

    if word.endswith("ies") and len(word) > 4:
        check(word[:-3] + "y", {"NOUN", "VERB"})
    if word.endswith("es") and len(word) > 3:
        allowed_es = {"VERB"} if word.endswith(("sises", "sizes")) else {"NOUN", "VERB"}
        check(word[:-2], allowed_es)
    if word.endswith("s") and len(word) > 2 and not word.endswith("ss"):
        allowed_s = {"VERB"} if word.endswith(("sises", "sizes")) else {"NOUN", "VERB"}
        check(word[:-1], allowed_s)
    if word.endswith("ied") and len(word) > 4:
        check(word[:-3] + "y", {"VERB", "ADJ"})
    if word.endswith("ed") and len(word) > 3:
        check(word[:-2], {"VERB", "ADJ"})
        check(word[:-1], {"VERB", "ADJ"})
        if len(word) > 4 and word[-3] == word[-4]:
            check(word[:-3], {"VERB", "ADJ"})
    if word.endswith("ing") and len(word) > 4:
        check(word[:-3], {"VERB", "NOUN", "ADJ"})
        check(word[:-3] + "e", {"VERB", "NOUN", "ADJ"})
        if len(word) > 5 and word[-4] == word[-5]:
            check(word[:-4], {"VERB", "NOUN", "ADJ"})
    if word.endswith("ier") and len(word) > 4:
        check(word[:-3] + "y", {"ADJ", "ADV"})
    if word.endswith("er") and len(word) > 3:
        check(word[:-2], {"ADJ", "ADV"})
        check(word[:-1], {"ADJ", "ADV"})
    if word.endswith("iest") and len(word) > 5:
        check(word[:-4] + "y", {"ADJ", "ADV"})
    if word.endswith("est") and len(word) > 4:
        check(word[:-3], {"ADJ", "ADV"})
        check(word[:-2], {"ADJ", "ADV"})

    if not candidates:
        return None, set()
    # Prefer the longest matching base
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    return candidates[0]


_HYPHEN_ADJ_SUFFIXES = (
    "-based", "-free", "-related", "-like", "-style", "-wide", "-bound",
    "-proof", "-rich", "-poor", "-friendly", "-oriented", "-driven", "-led",
    "-made", "-born", "-aged", "-sized", "-colored", "-coloured", "-shaped",
    "-lined", "-backed", "-headed", "-sided", "-handed", "-footed", "-eyed",
    "-legged",
)
_HYPHEN_ADV_SUFFIXES = ("-fully", "-lessly", "-wise")
_PREFIXES = (
    "un", "re", "non", "pre", "post", "de", "dis", "over", "under",
    "mis", "sub", "inter", "anti", "counter", "semi", "multi", "super", "trans",
)


def _varcon_orth_variants(w: str) -> list[str]:
    """Generate productive -ise/-ize, -yse/-yze, and ae/e orthographic equivalents."""
    swaps = [
        ("isations", "izations"), ("isation", "ization"),
        ("isabilities", "izabilities"), ("isability", "izability"),
        ("isables", "izables"), ("isable", "izable"),
        ("isabler", "izabler"), ("isablest", "izablest"),
        ("isers", "izers"), ("iser", "izer"),
        ("isings", "izings"), ("ising", "izing"),
        ("ised", "ized"), ("ises", "izes"), ("ise", "ize"),
        ("ysings", "yzings"), ("ysing", "yzing"),
        ("ysed", "yzed"), ("yses", "yzes"),
        ("ysers", "yzers"), ("yser", "yzer"), ("yse", "yze"),
        ("aemias", "emias"), ("aemia", "emia"), ("aemic", "emic"),
        ("aevally", "evally"), ("aeval", "eval"),
    ]
    out: list[str] = []
    for a, b in swaps:
        if w.endswith(a):
            out.append(w[: -len(a)] + b)
        if w.endswith(b):
            out.append(w[: -len(b)] + a)
    return out


def _map_lemma_to_word_orthography(word: str, alt: str, alt_lem: str) -> str:
    """Map an inferred lemma from `alt` back to `word`'s orthographic style if distinct."""
    if not alt_lem or alt_lem == "-" or alt_lem == alt:
        return "-"
    for cand in _varcon_orth_variants(alt_lem):
        if cand != alt_lem:
            return cand
    return alt_lem


def infer_paired_wize_adverb(word: str, counterpart: str, known_bases: dict[str, set[str]]) -> bool:
    """Recognize only a paired -wise spelling supported by a known noun stem."""
    if not word.endswith("wize") or len(word) <= 5:
        return False
    stem = word[:-4]
    return counterpart == stem + "wise" and "NOUN" in known_bases.get(stem, set())


def infer_derivational_pos_and_lemma(
    word: str, known_bases: dict[str, set[str]], depth: int = 0
) -> tuple[set[str], str]:
    """Multi-pass morphological/orthographic POS & lemma inference for out-of-WordNet tokens."""
    # 1. De-hyphenation & hyphenated compound rules
    if "-" in word:
        dehyphen = word.replace("-", "")
        if dehyphen in known_bases:
            return set(known_bases[dehyphen]), "-"
        if word.endswith(_HYPHEN_ADV_SUFFIXES):
            return {"ADV"}, "-"
        if word.endswith(_HYPHEN_ADJ_SUFFIXES):
            return {"ADJ"}, "-"
        tail = word.rsplit("-", 1)[-1]
        if len(tail) >= 3 and tail in known_bases:
            return set(known_bases[tail]), "-"
        if tail.endswith(("ed", "en")) and len(tail) >= 4:
            return {"ADJ"}, "-"

    # 2. Productive VarCon -ize/-ise, -yze/-yse, ae/e orthographic transfer
    for alt in _varcon_orth_variants(word):
        if alt in known_bases:
            return set(known_bases[alt]), "-"
        alt_base, alt_pos = infer_regular_lemma(alt, known_bases)
        if alt_base and alt_pos:
            mapped = _map_lemma_to_word_orthography(word, alt, alt_base)
            return alt_pos, (mapped if mapped != word else "-")

    # 3. Inflected derivational suffixes
    if word.endswith(("izes", "ises", "ifies")) and len(word) > 6:
        lem = word[:-3] + "y" if word.endswith("ifies") else word[:-1]
        return {"VERB"}, lem
    if word.endswith(("ized", "ised", "ified")) and len(word) > 6:
        if word.startswith("un") and len(word) > 7:
            return {"ADJ"}, "-"
        lem = word[:-3] + "y" if word.endswith("ified") else word[:-1]
        return {"ADJ", "VERB"}, lem
    if word.endswith(("izing", "ising", "ifying")) and len(word) > 7:
        if word.startswith("un") and len(word) > 8:
            return {"ADJ"}, "-"
        lem = word[:-3] + "y" if word.endswith("ifying") else word[:-3] + "e"
        return {"ADJ", "VERB"}, lem
    if (
        word.endswith(
            (
                "izations", "isations", "ments", "nesses", "isms", "ists",
                "ities", "abilities", "ibilities", "emias", "aemias",
            )
        )
        and len(word) > 7
    ):
        if word.endswith(("ities", "abilities", "ibilities")):
            return {"NOUN"}, word[:-3] + "y"
        if word.endswith("nesses"):
            return {"NOUN"}, word[:-2]
        return {"NOUN"}, word[:-1]
    if word.endswith(("izers", "isers", "ifiers", "fiers")) and len(word) > 6:
        return {"NOUN"}, word[:-1]
    if word.endswith(("abler", "ibler")) and len(word) > 7:
        return {"ADJ"}, word[:-1]
    if word.endswith(("ablest", "iblest")) and len(word) > 8:
        return {"ADJ"}, word[:-2]

    # 4. Base derivational suffixes
    if word.endswith(("able", "ible", "ative", "ful", "less", "ous", "ish", "emic", "aemic")) and len(word) > 6:
        return {"ADJ"}, "-"
    if word.endswith("ic") and len(word) > 5:
        return {"ADJ"}, "-"
    if word.endswith(("ally", "ably", "ibly", "fully", "lessly", "ously", "ively")) and len(word) > 6:
        return {"ADV"}, "-"
    if word.endswith("ly") and len(word) > 5:
        stem = word[:-2]
        if stem in known_bases:
            if "ADJ" in known_bases[stem]:
                return {"ADV"}, "-"
            if known_bases[stem] == {"NOUN"}:
                return {"ADJ"}, "-"
            return {"ADV"}, "-"
        if word.endswith("ily") and (word[:-3] + "y") in known_bases:
            return {"ADV"}, "-"
        if len(stem) >= 5:
            return {"ADV"}, "-"
    if (
        word.endswith(
            (
                "ness", "ity", "tion", "sion", "ment", "ship", "hood", "ism", "ist",
                "emia", "aemia", "graphy", "logy", "metry", "scopy",
                "izer", "iser", "ifier", "fier",
            )
        )
        and len(word) > 6
    ):
        return {"NOUN"}, "-"
    if word.endswith("ers") and len(word) > 6:
        sg = word[:-1]
        st = sg[:-2]
        st_e = sg[:-1]
        for cand in (st, st_e, *_varcon_orth_variants(st), *_varcon_orth_variants(st_e)):
            if cand in known_bases and "VERB" in known_bases[cand]:
                return {"NOUN"}, sg
    if word.endswith(("er", "or")) and len(word) > 5:
        stem = word[:-2]
        stem_e = word[:-1]
        for st in (stem, stem_e, *_varcon_orth_variants(stem), *_varcon_orth_variants(stem_e)):
            if st in known_bases and "VERB" in known_bases[st]:
                return {"NOUN"}, "-"
    if word.endswith("wise") and len(word) > 5:
        # The spelling partner may supply a known noun stem.
        # Do not infer POS from -wise alone or fall through to the -ise rule.
        if "NOUN" in known_bases.get(word[:-4], set()):
            return {"ADV"}, "-"
        return set(), "-"
    if word.endswith(("ize", "ise", "ify")) and len(word) > 5:
        return {"VERB"}, "-"
    if word.endswith("ate") and len(word) > 6:
        return {"VERB"}, "-"

    # 5. Productive prefix stripping
    if depth < 2:
        for pfx in _PREFIXES:
            for sep in ("-", ""):
                full_pfx = pfx + sep
                if word.startswith(full_pfx) and len(word) - len(full_pfx) >= 4:
                    stem = word[len(full_pfx) :]
                    if pfx == "un" and word.endswith(("ed", "ing")):
                        return {"ADJ"}, "-"
                    if stem in known_bases:
                        return set(known_bases[stem]), "-"
                    r_base, r_pos = infer_regular_lemma(stem, known_bases)
                    if r_base and r_pos:
                        return r_pos, (full_pfx + r_base if full_pfx + r_base != word else "-")
                    sub_pos, sub_lem = infer_derivational_pos_and_lemma(stem, known_bases, depth + 1)
                    if sub_pos:
                        lem = full_pfx + sub_lem if sub_lem != "-" else "-"
                        return sub_pos, (lem if lem != word else "-")

    return set(), "-"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build crates/faberline-nlp/lexicon/data/en.tsv")
    p.add_argument(
        "--wn-dict-dir",
        type=Path,
        default=Path(os.environ["LEX_WN_DICT_DIR"]) if "LEX_WN_DICT_DIR" in os.environ else None,
        help="Path to Princeton WordNet 3.1 dict/ directory (or env LEX_WN_DICT_DIR)",
    )
    p.add_argument(
        "--varcon",
        type=Path,
        default=Path(os.environ["LEX_VARCON_PATH"]) if "LEX_VARCON_PATH" in os.environ else None,
        help="Path to SCOWL/VarCon varcon.txt (or env LEX_VARCON_PATH)",
    )
    p.add_argument(
        "--clean-corpus-jsonl",
        type=Path,
        default=Path(os.environ["LEX_EN_CORPUS_JSONL"]) if "LEX_EN_CORPUS_JSONL" in os.environ else None,
        help="Path to JSONL corpus filtered strictly to allowlisted Gutenberg PD books (or env LEX_EN_CORPUS_JSONL)",
    )
    p.add_argument(
        "--gutenberg-dir",
        type=Path,
        default=Path(os.environ["LEX_EN_GUTENBERG_DIR"]) if "LEX_EN_GUTENBERG_DIR" in os.environ else None,
        help="Directory containing pg<id>.txt for the 12 allowlisted Gutenberg PD books (or env LEX_EN_GUTENBERG_DIR)",
    )
    p.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path(os.environ.get("LEX_TMP_DIR", tempfile.gettempdir())),
        help="Temporary directory for atomic write (or env LEX_TMP_DIR)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.wn_dict_dir is None or args.varcon is None:
        print(
            "ERROR: --wn-dict-dir (or LEX_WN_DICT_DIR) and --varcon (or LEX_VARCON_PATH) are required",
            file=sys.stderr,
        )
        return 1
    wn_dict_dir: Path = args.wn_dict_dir
    varcon_path: Path = args.varcon
    tmp_dir: Path = args.tmp_dir

    for required in [wn_dict_dir / "index.noun", varcon_path, POS_RS_PATH]:
        if not required.exists():
            print(f"ERROR: missing required file {required}", file=sys.stderr)
            return 1

    en_decided, en_ambiguous = parse_pos_rs()
    gold_upos = load_gold_upos()
    corpus_freqs, docs = load_corpus_freqs(args.clean_corpus_jsonl, args.gutenberg_dir)
    cntlist_freqs = load_cntlist(wn_dict_dir)
    offset_to_lex, instance_offsets = load_noun_lex_filenums(wn_dict_dir)
    wn_pos, wn_senses, wn_sem, instance_only_nouns, exc_map, irr_bases = load_wordnet(
        wn_dict_dir, offset_to_lex, instance_offsets
    )
    variant_map, shared_words = load_varcon(varcon_path)
    variant_map = remove_reviewed_bad_counterparts(variant_map)

    # Build candidate vocabulary (Product Vocabulary Policy)
    must_keep: set[str] = set()
    must_keep.update(en_decided.keys())
    must_keep.update(en_ambiguous)
    must_keep.update(gold_upos.keys())
    must_keep.update(exc_map.keys())
    must_keep.update(irr_bases)
    must_keep.update(variant_map.keys())
    must_keep.update(shared_words)
    must_keep.update(w for w, c in corpus_freqs.items() if c >= 1)
    must_keep.update(w for w, c in cntlist_freqs.items() if c >= 1)

    # Filter remaining WordNet words to hit TARGET_GOAL (66k)
    extra_candidates: list[tuple[tuple[int, int, int, int], str]] = []
    for w in wn_pos:
        if w in must_keep:
            continue
        is_non_instance = 0 if w in instance_only_nouns else 1
        is_sem_headword = 1 if (w in wn_sem and w not in instance_only_nouns) else 0
        has_non_noun = 1 if (wn_pos[w] - {"NOUN"}) else 0
        senses = wn_senses.get(w, 1)
        score = (is_non_instance, is_sem_headword, has_non_noun, senses)
        extra_candidates.append((score, w))

    extra_candidates.sort(key=lambda x: (-x[0][0], -x[0][1], -x[0][2], -x[0][3], x[1]))
    need_extra = max(0, TARGET_GOAL - len(must_keep))
    vocab = set(must_keep)
    for _, w in extra_candidates[:need_extra]:
        vocab.add(w)

    # Precompute raw frequencies for cost quantization
    raw_freq: dict[str, float] = {}
    for w in vocab:
        f = corpus_freqs.get(w, 0) * 1.0 + cntlist_freqs.get(w, 0) * 2.0
        if w in en_decided or w in en_ambiguous:
            f = max(f, 500.0)
        if w in gold_upos:
            f = max(f, 5.0)
        raw_freq[w] = f

    # Mirror counterpart frequency so en-GB gets comparable cost to en-US
    for w in vocab:
        if w in variant_map:
            cp = variant_map[w][1]
            if cp in raw_freq:
                raw_freq[w] = max(raw_freq[w], raw_freq[cp])

    max_log = math.log1p(max(raw_freq.values(), default=1.0))

    def compute_cost(w: str) -> int:
        f = raw_freq.get(w, 0.0)
        if f <= 0.0:
            return 1650 if w in variant_map else 1700
        ratio = 1.0 - (math.log1p(f) / max_log)
        cost = int(round(200 + ratio * 1400))
        return max(200, min(1800, cost))

    # Build entries
    rows: list[tuple[str, str, int, str, str, str, str, str]] = []
    for w in sorted(vocab, key=lambda s: s.encode("utf-8")):
        if not is_valid_token(w):
            continue

        # Variant & counterpart
        if w in variant_map:
            variant, counterpart = variant_map[w]
        else:
            variant, counterpart = "en", "-"

        # UPOS mask & lemma
        pos_set: set[str] = set(wn_pos.get(w, set()))
        lemma = "-"
        is_irr = False
        is_closed = w in en_decided or w in en_ambiguous

        # Check closed-class lemmas
        if w in CLOSED_LEMMAS:
            lemma = CLOSED_LEMMAS[w]
            if lemma in {"be", "have", "do"}:
                is_irr = True

        # Check irregular .exc
        if w in exc_map:
            is_irr = True
            for ud_tag, bases in exc_map[w].items():
                pos_set.add(ud_tag)
                if lemma == "-" and bases:
                    # If w is not itself a base in wn_pos or is a known irr like went/mice/better
                    if w not in wn_pos or w in {"better", "best", "gone", "mice", "children", "analyses"}:
                        lemma = bases[0]
        if w in irr_bases:
            is_irr = True

        # Propagate from counterpart if missing in WordNet (e.g. colour, analyse, centre)
        if not pos_set and counterpart != "-":
            if counterpart in wn_pos:
                pos_set.update(wn_pos[counterpart])

        # Infer regular lemma for inflected forms (including VarCon participles/inflections)
        if lemma == "-" and (
            w not in wn_pos
            or (w in variant_map and not ({"NOUN", "VERB"} & wn_pos[w] and w.endswith(("er", "or"))))
        ):
            inf_base, inf_pos = infer_regular_lemma(w, wn_pos)
            if inf_base and inf_base != w:
                pos_set.update(inf_pos)
                lemma = inf_base
            elif counterpart != "-" and (
                counterpart not in wn_pos
                or (
                    counterpart in variant_map
                    and not ({"NOUN", "VERB"} & wn_pos[counterpart] and counterpart.endswith(("er", "or")))
                )
            ):
                c_base, c_pos = infer_regular_lemma(counterpart, wn_pos)
                if c_base and c_base != counterpart:
                    pos_set.update(c_pos)
                    lemma = c_base

        # An explicit VarCon pair and known noun stem support the -wize adverb.
        # Exact and regular source evidence retain precedence.
        if not pos_set and infer_paired_wize_adverb(w, counterpart, wn_pos):
            pos_set.add("ADV")

        # Multi-pass morphological/orthographic POS & lemma inference before unknown fallback
        if not pos_set:
            d_pos, d_lem = infer_derivational_pos_and_lemma(w, wn_pos)
            pos_set.update(d_pos)
            if lemma == "-" and d_lem != "-":
                lemma = d_lem
        if not pos_set and counterpart != "-":
            d_pos, d_lem = infer_derivational_pos_and_lemma(counterpart, wn_pos)
            pos_set.update(d_pos)
            if lemma == "-" and d_lem != "-":
                lemma = _map_lemma_to_word_orthography(w, counterpart, d_lem)
                if lemma == "-":
                    lemma = d_lem

        # Align lemma spelling with en-GB variant if lemma is an en-US headword
        if variant == "en-GB" and lemma != "-" and lemma in variant_map:
            if variant_map[lemma][0] == "en-US":
                lemma = variant_map[lemma][1]

        # Closed-class & gold compatibility
        if w in en_decided:
            pos_set.add(en_decided[w])
        if w in gold_upos:
            pos_set.update(gold_upos[w])

        # Evidence belongs to the exact spelling, never to the generic suffix
        # inference or an unverified spelling counterpart.
        reviewed_pos = None
        if w in REVIEWED_POS_EVIDENCE:
            direct_pos = set(wn_pos.get(w, set())) | set(gold_upos.get(w, set()))
            if w in en_decided:
                direct_pos.add(en_decided[w])
            reviewed_pos = apply_reviewed_pos_evidence(w, direct_pos, REVIEWED_POS_EVIDENCE)
        if reviewed_pos is not None:
            pos_set = reviewed_pos

        # Determine decided_upos (emit "-", "-" when no reliable POS evidence exists)
        if reviewed_pos is not None:
            decided_upos = next(iter(pos_set)) if len(pos_set) == 1 else "-"
        elif w in en_ambiguous:
            decided_upos = "-"
        elif w in en_decided and (w not in gold_upos or gold_upos[w] == {en_decided[w]}):
            decided_upos = en_decided[w]
            pos_set = {decided_upos}
        elif len(pos_set) == 1:
            decided_upos = next(iter(pos_set))
        else:
            decided_upos = "-"

        upos_mask = decided_upos if decided_upos != "-" else ("|".join(sorted(pos_set)) if pos_set else "-")
        if lemma == w:
            lemma = "-"

        # Semantic & inflection meta
        meta_parts: list[str] = []
        sem = wn_sem.get(w)
        if not sem and counterpart != "-":
            sem = wn_sem.get(counterpart)
        if sem and "NOUN" in pos_set:
            meta_parts.append(f"sem={sem}")
        if is_irr:
            meta_parts.append("infl=irr")
        elif is_closed:
            meta_parts.append("infl=closed")
        meta = ",".join(meta_parts) if meta_parts else "-"

        cost = compute_cost(w)
        rows.append((w, variant, cost, decided_upos, upos_mask, lemma, counterpart, meta))

    # Strict sort by UTF-8 bytes of (surface, variant) and deduplicate
    rows.sort(key=lambda r: (r[0].encode("utf-8"), r[1].encode("utf-8")))
    deduped: list[tuple[str, str, int, str, str, str, str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for r in rows:
        key = (r[0], r[1])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(r)

    # Verify ZERO disagreements with gold
    surface_decided = {r[0]: r[3] for r in deduped}
    conflicts = []
    for tok, tags in gold_upos.items():
        dec = surface_decided.get(tok, "-")
        if dec != "-" and (len(tags) > 1 or dec not in tags):
            conflicts.append((tok, dec, sorted(tags)))
    if conflicts:
        print(f"ERROR: {len(conflicts)} gold conflicts: {conflicts[:10]}", file=sys.stderr)
        return 1

    # Write atomically via scratch temp file -> os.replace()
    OUT_TSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = tmp_dir / "en.tsv.tmp"
    with tmp_out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# faberline-nlp English (en-US / en-GB / en) morphological & semantic lexicon\n")
        f.write("# Sources: Princeton WordNet 3.1, SCOWL/VarCon, src/doc/pos.rs, PD Project Gutenberg (12 books)\n")
        f.write("# Format: surface\\tvariant\\tcost\\tdecided_upos\\tupos_mask\\tlemma\\tcounterpart\\tmeta\n")
        for r in deduped:
            f.write(
                f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]}\t{r[6]}\t{r[7]}\n"
            )
    os.replace(tmp_out, OUT_TSV_PATH)

    # Print summary stats
    var_counts = Counter(r[1] for r in deduped)
    dec_count = sum(1 for r in deduped if r[3] != "-")
    sem_count = sum(1 for r in deduped if "sem=" in r[7])
    irr_count = sum(1 for r in deduped if "infl=irr" in r[7])
    lemma_count = sum(1 for r in deduped if r[5] != "-")
    print(f"Wrote {len(deduped)} entries to {OUT_TSV_PATH} (corpus docs={docs})")
    print(f"  Variants: {dict(var_counts)}")
    print(f"  Decided UPOS: {dec_count}, Ambiguous('-'): {len(deduped) - dec_count}")
    print(f"  With lemma != '-': {lemma_count}, sem=*: {sem_count}, infl=irr: {irr_count}")
    print(f"  Gold conflicts: 0 (checked {len(gold_upos)} unique gold tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
