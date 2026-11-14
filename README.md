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
    `〆`, ...) and paragraph-level Japanese context inheritance (`「約束。」彼は言った。`), while
    Hangul + Hanja without Kana tags as `"mixed"`.
  - Language-aware segmentation: Japanese *tōten* `、` splits clauses while Chinese enumeration
    comma `、` does not; Kana-heavy lines break paragraphs after CJK/Japanese terminators and
    brackets (`‥`, `⋯`, `】〕〉》］｝`); Korean dotted dates (`2026. 11. 13.`) stay inside one
    sentence and tight Hangul commas (`먹고,마시고`) split clauses; English quoted `?`/`!` before a
    lowercase attribution verb (`"Really?" she asked.`) and ellipsis runs before clause separators
    (`「……，但是」`) do not split sentences; and expanded title/organization/degree/time
    abbreviations (`Inc.`, `Ltd.`, `Ph.D.`, `a.m.`, `p.m.`, ...) are protected.
  - Opaque spans (`code`, `formula`, `heading`, `list_item`, `citation`, `url`, `emoji`) never
    split inside sentences, covering Markdown, Chinese, English, Japanese (`第1話`, `第2巻`,
    `12話 雨の駅`, `プロローグ`, excluding particle-attached prose such as `第1話は...`), and
    Korean (`제1장`, `제12화`, `1화`, `프롤로그`, excluding particle-attached prose such as
    `제1장은...`) headings, multilingual author-year citations (`（田中・佐藤、2020）`,
    `(김철수, 2020a)`), and `※ ` list items.
- **Layer 2 (`tokens`, `upos` & first-party `lexicon`)**:
  - Tokenizes Chinese (`jieba` + Taiwan lexicon + optional `t2s` projection, or opt-in
    `--first-party-dict` / `options.first_party_zh_segmenter` using our compiled `zh-Hant` +
    `zh-Hans` lexicon), English (UAX #29 + clitics and abbreviations), Japanese (first-party
    Viterbi segmenter `lexicon::ja::segment_ja` with inflectional stem+tail splitting +
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
  - Rule-based Layer 3 (`entities`) and Layer 7 (`relations`), plus optional model-backed Layer 4
    (`analyze_document_with_model` / `nlp_doc --models <DIR>`), operate on `zh`, `en`, and `zh`/`en`
    `mixed` sentences and cleanly skip sentences containing Japanese or Korean script.

See [docs/doc-v1.md](docs/doc-v1.md) for the full `nlp.doc.v1` specification.

## Experimental source stream

`nlp_source_stream` emits chapter-by-chapter JSON Lines with source hashes and
positions. Optional flags add cross-chapter candidates and source-scoped quote
and transfer observations. The program uses a caller-supplied vocabulary or an
empty vocabulary. It currently uses the Chinese (Taiwan) tokenizer.

The stream is experimental. Input directories must be trusted and must not
change during a run. See the [source stream contract](docs/source-stream.md)
for the format, validation steps, and known scope and resource limits.

## English CNL

Use `cnl::parse` to read English SOM CNL v1. The grammar covers prompt versions 1 through 7. Pass the whole request as a string.

```rust
let source = std::fs::read_to_string("request.cnl")?;
let document = faberline_nlp::cnl::parse(&source)?;
assert_eq!(document.text, source.as_str());
```

The result is a tree of grammar rules. Each node gives its text and a byte range in the original input. A syntax error gives a line and column.

Use `faberline-nlp-cnl` from the `cnl/` directory if you only need the English parser. Call `check_syntax` to check the grammar without building a tree.

`faberline-nlp-cnl` owns the SOM CNL grammar (`cnl/src/som_cnl.pest`), the syntax tree and the syntax error locations. It is the only place where CNL syntax changes. nlp holds no business rules, DSL or code generation. Those belong to [som](https://github.com/faberline/som). If som needs a syntax change, open an nlp issue first. After it merges, som moves its pinned revision.

Use the [SOM examples](https://github.com/faberline/som/tree/05c63d08275d59ea7110cd8f2b30786070b91968/som-rs/contracts/compiler-v1/examples) for sample requests.

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
