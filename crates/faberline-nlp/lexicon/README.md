# First-Party Multilingual Lexicon (`faberline_nlp::lexicon`)

Permissively licensed, compile-time compiled multilingual lexicon covering 4 languages and 6 variants (`zh-Hant`, `zh-Hans`, `en-US`, `en-GB`, `ja`, `ko`).

## Directory Layout

- `data/` — Canonical UTF-8 TSV source tables:
  - `zh_hant.tsv` (8-column): Traditional Chinese (`zh-Hant`) vocabulary, frequency costs, UPOS masks, cross-strait `zh-Hans` counterparts, and semantic classes.
  - `zh_hans.tsv` (8-column): Simplified Chinese (`zh-Hans`) vocabulary, frequency costs, UPOS masks, cross-strait `zh-Hant` counterparts, and semantic classes.
  - `en.tsv` (8-column): English (`en-US`, `en-GB`, `en`) base lemmas, irregular forms, US↔GB counterparts, and semantic classes.
  - `ja.tsv` (8-column): Japanese (`ja`) dictionary base forms, function words, proper nouns, inflection paradigms, and semantic classes.
  - `ja_inflect.tsv` (6-column): Japanese conjugation rules (`godan_*`, `ichidan`, `sahen`, `kahen`, `adj_i`) with UD-aligned morpheme splits.
  - `ja_zh_hant_senses.tsv` (7-column): First-party Taiwan Traditional Chinese senses keyed by Japanese headword and UPOS.
  - `ko.tsv` (8-column): Korean (`ko`) stems, function words, and pre-analyzed irregular inflections.
  - `ko_morph.tsv` (6-column): Korean Josa, Copula, XSV/XSA, and Eomi suffix rules with batchim conditions and exact character-span decompositions.
- `LICENSES/` — Upstream permissive license texts (`MIT`, `Apache-2.0`, `BSD-3-Clause`, `Unicode-3.0`, `WordNet`, `SCOWL`). No `CC BY-SA`, `CC BY-NC`, `CC BY-ND`, or `GPL` sources are used.
  - `LICENSE-UNICODE-3.0.txt`: Unicode License V3 (`Unicode-3.0`) covering Unicode Unihan `kTraditionalVariant` / Japanese shinjitai-to-Traditional glyph mappings (`src/lexicon/ja.rs` and `tools/lexicon_build/derive_unihan_kango.py`; `249` `unihan_zh_hant` rows in `ja.tsv`, `342` false-friend blocks in `tools/lexicon_build/data/ja_zh_false_friends.tsv`).
  - `LICENSE-SUDACHIDICT-APACHE-2.0.txt` & `NOTICE-SUDACHIDICT.txt`: SudachiDict (`20261015`, `Apache-2.0`, including `UniDic` and `NEologd` notices) used offline for frequency/cost ranking only (`tools/lexicon_build/select_a4_sudachi_ranks.py` and `tools/lexicon_build/data/a4_sudachi_ranks.tsv`; no SudachiDict CSV or glosses are checked in). All `2,500` A4 `ja -> zh-Hant` glosses (`1,500` `VERB` + `1,000` `MULTI`) are authored first-party under `MIT` (`src=first_party_mit` in `tools/lexicon_build/data/a4_first_party_mit_ja_zh.tsv`).

## Build-Time Compilation

`build.rs` compiles `lexicon/data/*.tsv` in pure Rust (zero Python, zero network, zero external crate dependencies) into a deterministic little-endian binary blob `$OUT_DIR/lexicon_v1.bin` embedded via `include_bytes!`.

- **Format (`FNLPLX01`)**: 80-byte header + per-language 4096-bucket sorted key index + 20-byte packed entries + `ja_inflect` / `ko_morph` tables + deduplicated UTF-8 string pool.
- **Never-Guess Rule**: If multiple entries for the same `(lang_group, surface)` have differing `upos_mask`s or `upos_mask` has more than 1 bit set, `decided_upos` is `None`.

## Japanese to Taiwan Traditional Chinese senses

Each row in `ja_zh_hant_senses.tsv` declares a headword, UD part of speech, dictionary anchor, stable sense ID, gloss, usage note, and source. A dash means the headword itself is the anchor. A verb ending in `する` can name its base stem when the dictionary stores the stem instead of the full verb. The build checks the anchor's part of speech and rejects duplicate sense IDs for the same headword and part of speech.

The labels and usage notes are original MIT material. They use Taiwan Traditional Chinese. External dictionaries can help establish the boundaries of a Japanese sense. The source table stores no copied definitions or examples.

The build emits a separate read-only table. Use `lexicon::ja::lookup_ja_zh_hant_senses` to retrieve all records for an exact headword and part of speech. The existing `lookup_ja_gloss` call still returns its single default. Sentence translation and tokenization still use their existing rules.

```sh
nlp doc --ja-zh-senses 通す VERB
nlp doc --ja-zh-senses ポスト NOUN
```

This command returns `nlp.ja-zh-hant-senses.v1` JSON on stdout and does not read stdin. An absent key returns an empty `senses` array and a null `legacy_default`. A present key can also have a null default when the older dictionary has no exact headword, as with `試験する`.
