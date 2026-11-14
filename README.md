# Faberline NLP

A Rust library for finding candidate events and ordering them in Chinese narrative text. Results point back to spans in the original text. An end offset is exclusive. Offsets count Unicode scalar values from the start of each input.

Call `analyze` with the source text, its title, and a `Lexicon`. The caller supplies character names, alternate names, places, objects, rank terms, and publisher footer markers. The result includes events, text sections, and items that need review.

Call `timeline::build` with the texts and their candidate events. The function sorts the input by document number. Document numbers must be unique. The result includes the original event order, proposed story order, and the source spans for each relation.

These observations are candidates. They do not prove a person's identity, an
event, or the intended story order. The caller owns the work-specific vocabulary
and must verify source positions against the original input.

Some ambiguous speech cases now leave the speaker, actor, or object unknown.
Callers must not rely on earlier heuristic labels being present in every case.

## Document pipeline (`nlp.doc.v1`)

`faberline_nlp::doc::analyze_document` (and the `nlp_doc` CLI) produces a deterministic
`nlp.doc.v1` JSON `Document` with exact Unicode scalar `Span` offsets (`[start_char, end_char)`),
using no network:

- **Layer 1 & 1b (`segmentation` & `opaque`) — Chinese, English, Japanese, and Korean**:
  - Tags paragraphs, sentences, and clauses with `lang`: `"zh"`, `"en"`, `"ja"`, `"ko"`,
    `"mixed"`, or `"und"`. Pure-Han spans disambiguate `"ja"` vs `"zh"` using curated
    Japanese-exclusive Shinjitai/Kokuji characters (`駅`, `図`, `読`, `発`, `観`, `気`, `込`, `畑`,
    `〆`, ...) and paragraph-level Japanese context inheritance (`「約束。」彼は言った。`). Korean
    parenthetical `(Hanja)` glosses (`도덕성(道德性)`) keep `"ko"`, and isolated 1-letter ASCII
    Latin variables in `ja`/`ko`-dominant prose (`速さが V である`, `점 P 에서`) keep `"ja"`/`"ko"`
    rather than flipping to `"mixed"`.
  - Language-aware segmentation: boundary trimming strips invisible format spaces (`U+200B`,
    `U+2060`, `U+FEFF`) while preserving `U+200D` ZWJ and `U+FE0F` VS16; tight halfwidth ASCII `.`
    between Han/CJK-closer and Han/CJK-opener (`项目.它主要`) splits sentences; Japanese *tōten*
    `、` splits clauses while Chinese enumeration comma `、` does not; Kana-heavy lines break
    paragraphs after CJK/Japanese terminators and brackets (`‥`, `⋯`, `】〕〉》］｝`); Korean
    dotted dates (`2026. 11. 13.`) stay inside one sentence and tight Hangul commas (`먹고,마시고`)
    split clauses; English quoted `?`/`!` before a lowercase attribution verb (`"Really?" she
    asked.`) and ellipsis runs before clause separators (`「……，但是」`) do not split sentences; and
    expanded title/organization/degree/time abbreviations (`Inc.`, `Ltd.`, `Ph.D.`, `a.m.`,
    `p.m.`, ...) are protected.
  - Opaque spans (`code`, `formula`, `heading`, `list_item`, `citation`, `url`, `emoji`) never
    split inside sentences, covering Markdown, Chinese, English, Japanese (`第1話`, `第2巻`,
    `12話 雨の駅`, `プロローグ`, excluding particle-attached prose such as `第1話は...`), and
    Korean (`제1장`, `제12화`, `1화`, `프롤로그`, excluding particle-attached prose such as
    `제1장은...`) headings, multilingual author-year citations (`（田中・佐藤、2020）`,
    `(김철수, 2020a)`), `※ ` list items, and multi-char PUA noise runs (while single-char PUA gaiji
    stay plain text).
- **Layer 2 (`tokens`, `upos` & first-party `lexicon`)**:
  - Tokenizes Chinese using our first-party `zh-Hant` + `zh-Hans` lexicon Viterbi segmenter
    (`lexicon::zh::segment_zh`) by default (with `--legacy-jieba` / `options.legacy_jieba` to select
    legacy `jieba-rs` + Taiwan lexicon + optional `t2s` projection; `--first-party-dict` remains
    accepted as a compatible flag), English (UAX #29 + clitics and abbreviations), Japanese
    (first-party Viterbi segmenter `lexicon::ja::segment_ja` with inflectional stem+tail splitting +
    longest-first `options.names`/`lexicon`), and Korean (longest-first `options.names`/`lexicon`
    + UAX #29 *eojeol* 어절 boundaries).
  - Punctuation, numbers, symbols, and inline opaque spans receive rule-decided `upos` tags across
    all four languages; Japanese and Korean tokens receive real UD v2 `upos` tags from the embedded
    first-party lexicon (emitting `ambiguous` or `no_model` when undecided), while scripts outside
    Han/Kana/Hangul/Latin emit `{"layer": "segmentation", "reason": "unsupported_language"}`.
  - Opt-in `--morphology` / `options.enrich_morphology` populates `lemma`, `canonical_variant`
    (`zh-Hant` ↔ `zh-Hans`, `en-US` ↔ `en-GB`), `upos_candidates`, `sem_class`, and Korean
    `morphemes` on `Token`.
- **Downstream layers (`entities`, `syntax`, `relations`)**:
  - Rule-based Layer 3 (`entities`, including `ja`/`ko` dates, Japanese eras, Korean dotted dates,
    times, quantities with Korean particle trimming, numbers, URLs, emails, and `options.names` /
    `PROPN` semantic-class names) and Layer 7 (`relations`, including `ja`/`ko` explicit discourse
    connectors for `Contrast`, `Cause`, `Condition`, `Sequence`, and `Parallel`) operate across
    `zh`, `en`, `ja`, `ko`, and `mixed` sentences.
  - Optional model-backed Layer 4 (`analyze_document_with_model` / `nlp_doc --models <DIR>`)
    operates on `zh`, `en`, and `zh`/`en` `mixed` sentences and skips sentences containing Japanese
    or Korean script.

See [docs/doc-v1.md](docs/doc-v1.md) for the full `nlp.doc.v1` specification,
[docs/narrative-v1.md](docs/narrative-v1.md) for the downstream `nlp.narrative.v1` narrative graph
layer (`crates/faberline-nlp-narrative/`), and [docs/translate-v1.md](docs/translate-v1.md) for the offline post-NLP
translation subsystem (`crates/faberline-nlp-translate/`, `nlp.translate.v1`).

## Experimental source stream

`nlp_source_stream` emits chapter-by-chapter JSON Lines with source hashes and
positions. Optional flags add cross-chapter candidates and source-scoped quote
and transfer observations. The program uses a caller-supplied vocabulary or an
empty vocabulary. It currently uses the Chinese (Taiwan) tokenizer.

The stream is experimental. Input directories must be trusted and must not
change during a run. See the [source stream contract](docs/source-stream.md)
for the format, validation steps, and known scope and resource limits.

## English CNL (Retired)

`faberline-nlp-cnl` has been retired from `faberline/nlp` and handed over to
[faberline/som](https://github.com/faberline/som/issues/217), which owns both SOM CNL syntax and
semantics.

## Model token limits

Enable the `model-tokenizer` Cargo feature to count model tokens or limit the length of model input. A model token is a unit of text in the model's encoding. Reuse the tokenizer across calls.

```rust
use faberline_nlp::tokenizer::TiktokenTokenizer;

let tokenizer = TiktokenTokenizer::cl100k_base().expect("load the cl100k_base encoding");
let input = tokenizer.truncate("hello world", 1);
assert_eq!(tokenizer.count(&input), 1);
```

This module adapts the Mamba AgentKit tokenizer. See [LICENSE-MAMBA](LICENSE-MAMBA).

```sh
cargo test --locked --workspace --all-features
```
