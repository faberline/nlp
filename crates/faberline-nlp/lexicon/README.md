# First-Party Multilingual Lexicon (`faberline_nlp::lexicon`)

Permissively licensed, compile-time compiled multilingual lexicon covering 4 languages and 6 variants (`zh-Hant`, `zh-Hans`, `en-US`, `en-GB`, `ja`, `ko`).

## Directory Layout

- `data/` — Canonical UTF-8 TSV source tables:
  - `zh_hant.tsv` (8-column): Traditional Chinese (`zh-Hant`) vocabulary, frequency costs, UPOS masks, cross-strait `zh-Hans` counterparts, and semantic classes.
  - `zh_hans.tsv` (8-column): Simplified Chinese (`zh-Hans`) vocabulary, frequency costs, UPOS masks, cross-strait `zh-Hant` counterparts, and semantic classes.
  - `en.tsv` (8-column): English (`en-US`, `en-GB`, `en`) base lemmas, irregular forms, US↔GB counterparts, and semantic classes.
  - `ja.tsv` (8-column): Japanese (`ja`) dictionary base forms, function words, proper nouns, inflection paradigms, and semantic classes.
  - `ja_inflect.tsv` (6-column): Japanese conjugation rules (`godan_*`, `ichidan`, `sahen`, `kahen`, `adj_i`) with UD-aligned morpheme splits.
  - `ko.tsv` (8-column): Korean (`ko`) stems, function words, and pre-analyzed irregular inflections.
  - `ko_morph.tsv` (6-column): Korean Josa, Copula, XSV/XSA, and Eomi suffix rules with batchim conditions and exact character-span decompositions.
- `LICENSES/` — Upstream permissive license texts (`MIT`, `Apache-2.0`, `BSD-3-Clause`, `WordNet`, `SCOWL`). No `CC BY-SA`, `CC BY-NC`, `CC BY-ND`, or `GPL` sources are used.

## Build-Time Compilation

`build.rs` compiles `lexicon/data/*.tsv` in pure Rust (zero Python, zero network, zero external crate dependencies) into a deterministic little-endian binary blob `$OUT_DIR/lexicon_v1.bin` embedded via `include_bytes!`.

- **Format (`FNLPLX01`)**: 80-byte header + per-language 4096-bucket sorted key index + 20-byte packed entries + `ja_inflect` / `ko_morph` tables + deduplicated UTF-8 string pool.
- **Never-Guess Rule**: If multiple entries for the same `(lang_group, surface)` have differing `upos_mask`s or `upos_mask` has more than 1 bit set, `decided_upos` is `None`.
