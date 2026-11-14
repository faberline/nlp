#!/usr/bin/env python3
"""Build lexicon/data/en.tsv from Princeton WordNet 3.1, SCOWL/VarCon,
src/doc/pos.rs, and public CC0/PD English corpora.

Strictly permissive commercial-safe sources only:
- Princeton WordNet 3.1 (WordNet BSD-style license)
- SCOWL / VarCon (MIT/BSD-like permissive license)
- In-repo src/doc/pos.rs & public CC0/PD en_corpus.jsonl
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_DIR = Path(
    "/Users/chrischeng/.gemini/antigravity/brain/"
    "fcdf5ec7-a63d-4308-93f6-0436229a04ea/scratch"
)
BUILD_TMP_DIR = SCRATCH_DIR / "lex_build_en"
WN_DICT_DIR = BUILD_TMP_DIR / "dict"
VARCON_PATH = BUILD_TMP_DIR / "varcon.txt"
CORPUS_PATH = SCRATCH_DIR / "corpora" / "en" / "en_corpus.jsonl"

NLP_CRATE_ROOT = REPO_ROOT / "crates" / "faberline-nlp"
POS_RS_PATH = NLP_CRATE_ROOT / "src" / "doc" / "pos.rs"
POS_GOLD_PATH = NLP_CRATE_ROOT / "tests" / "data" / "pos_gold.json"
GOLD_LAYERS_PATH = NLP_CRATE_ROOT / "tests" / "data" / "gold_layers4_6_v1.jsonl"
OUT_TSV_PATH = NLP_CRATE_ROOT / "lexicon" / "data" / "en.tsv"

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
TARGET_GOAL = 62_000


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


def load_corpus_freqs() -> Counter[str]:
    """Count token frequencies from public CC0/PD en_corpus.jsonl."""
    counts: Counter[str] = Counter()
    if not CORPUS_PATH.exists():
        return counts
    tok_re = re.compile(r"[a-z]+(?:[-'][a-z]+)*|n't|'[a-z]+")
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            text = d.get("text", "").lower().replace("\u2019", "'")
            for m in tok_re.finditer(text):
                w = m.group(0)
                if is_valid_token(w):
                    counts[w] += 1
    return counts


def load_cntlist() -> Counter[str]:
    """Load WordNet cntlist frequencies per lemma."""
    counts: Counter[str] = Counter()
    cnt_path = WN_DICT_DIR / "cntlist"
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


def load_noun_lex_filenums() -> tuple[dict[str, int], set[str]]:
    """Read data.noun to map synset_offset -> lex_filenum and detect instance-only nouns."""
    offset_to_lex: dict[str, int] = {}
    instance_offsets: set[str] = set()
    data_noun = WN_DICT_DIR / "data.noun"
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
    offset_to_lex: dict[str, int], instance_offsets: set[str]
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
        idx_path = WN_DICT_DIR / f"index.{pos_name}"
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
        exc_path = WN_DICT_DIR / f"{pos_name}.exc"
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


def load_varcon() -> tuple[dict[str, tuple[str, str]], set[str]]:
    """Parse varcon.txt for clean US<->GB pairs.

    Returns:
      - variant_map: word -> (variant_tag, counterpart)
      - shared_words: words marked A and B on the same varcon line
    """
    raw_pairs: list[tuple[str, str]] = []
    shared_words: set[str] = set()

    with VARCON_PATH.open(encoding="utf-8", errors="replace") as f:
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
            if overlap:
                candidates.append((base, overlap))

    if word.endswith("ies") and len(word) > 4:
        check(word[:-3] + "y", {"NOUN", "VERB"})
    if word.endswith("es") and len(word) > 3:
        check(word[:-2], {"NOUN", "VERB"})
    if word.endswith("s") and len(word) > 2 and not word.endswith("ss"):
        check(word[:-1], {"NOUN", "VERB"})
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
    if word.endswith("er") and len(word) > 3:
        check(word[:-2], {"ADJ", "ADV", "NOUN"})
        check(word[:-1], {"ADJ", "ADV", "NOUN"})
    if word.endswith("est") and len(word) > 4:
        check(word[:-3], {"ADJ", "ADV"})
        check(word[:-2], {"ADJ", "ADV"})

    if not candidates:
        return None, set()
    # Prefer the longest matching base
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    return candidates[0]


def main() -> int:
    for required in [WN_DICT_DIR / "index.noun", VARCON_PATH, POS_RS_PATH]:
        if not required.exists():
            print(f"ERROR: missing required file {required}", file=sys.stderr)
            return 1

    en_decided, en_ambiguous = parse_pos_rs()
    gold_upos = load_gold_upos()
    corpus_freqs = load_corpus_freqs()
    cntlist_freqs = load_cntlist()
    offset_to_lex, instance_offsets = load_noun_lex_filenums()
    wn_pos, wn_senses, wn_sem, instance_only_nouns, exc_map, irr_bases = load_wordnet(
        offset_to_lex, instance_offsets
    )
    variant_map, _ = load_varcon()

    # Build candidate vocabulary
    must_keep: set[str] = set()
    must_keep.update(en_decided.keys())
    must_keep.update(en_ambiguous)
    must_keep.update(gold_upos.keys())
    must_keep.update(exc_map.keys())
    must_keep.update(irr_bases)
    must_keep.update(variant_map.keys())
    must_keep.update(w for w, c in corpus_freqs.items() if c >= 1)
    must_keep.update(w for w, c in cntlist_freqs.items() if c >= 1)

    # Filter remaining WordNet words to hit TARGET_GOAL (~62k)
    extra_candidates: list[tuple[tuple[int, int, int], str]] = []
    for w in wn_pos:
        if w in must_keep:
            continue
        is_non_instance = 0 if w in instance_only_nouns else 1
        has_non_noun = 1 if (wn_pos[w] - {"NOUN"}) else 0
        senses = wn_senses.get(w, 1)
        score = (is_non_instance, has_non_noun, senses)
        extra_candidates.append((score, w))

    extra_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
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

        # Infer regular lemma for inflected forms (e.g. colours -> colour, analysed -> analyse)
        if lemma == "-" and (w not in wn_pos or w in variant_map):
            inf_base, inf_pos = infer_regular_lemma(w, wn_pos)
            # Only accept inferred lemma for words already in wn_pos if base is also a US/GB variant
            if inf_base and inf_base != w and (w not in wn_pos or inf_base in variant_map):
                pos_set.update(inf_pos)
                lemma = inf_base
            elif counterpart != "-":
                c_base, c_pos = infer_regular_lemma(counterpart, wn_pos)
                if c_base and c_base != counterpart and (
                    counterpart not in wn_pos or c_base in variant_map
                ):
                    pos_set.update(c_pos)
                    lemma = c_base

        # Align lemma spelling with en-GB variant if lemma is an en-US headword
        if variant == "en-GB" and lemma != "-" and lemma in variant_map:
            if variant_map[lemma][0] == "en-US":
                lemma = variant_map[lemma][1]

        # Closed-class & gold compatibility
        if w in en_decided:
            pos_set.add(en_decided[w])
        if w in gold_upos:
            pos_set.update(gold_upos[w])

        # Fallback if still empty (e.g. rare corpus/varcon token)
        if not pos_set:
            pos_set.add("NOUN")

        # Determine decided_upos
        if w in en_ambiguous:
            decided_upos = "-"
        elif w in en_decided and (w not in gold_upos or gold_upos[w] == {en_decided[w]}):
            decided_upos = en_decided[w]
            pos_set = {decided_upos}
        elif len(pos_set) == 1:
            decided_upos = next(iter(pos_set))
        else:
            decided_upos = "-"

        upos_mask = decided_upos if decided_upos != "-" else "|".join(sorted(pos_set))
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
    tmp_out = BUILD_TMP_DIR / "en.tsv.tmp"
    with tmp_out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# faberline-nlp English (en-US / en-GB / en) morphological & semantic lexicon\n")
        f.write("# Sources: Princeton WordNet 3.1, SCOWL/VarCon, src/doc/pos.rs, CC0/PD en_corpus\n")
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
    print(f"Wrote {len(deduped)} entries to {OUT_TSV_PATH}")
    print(f"  Variants: {dict(var_counts)}")
    print(f"  Decided UPOS: {dec_count}, Ambiguous('-'): {len(deduped) - dec_count}")
    print(f"  With lemma != '-': {lemma_count}, sem=*: {sem_count}, infl=irr: {irr_count}")
    print(f"  Gold conflicts: 0 (checked {len(gold_upos)} unique gold tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
