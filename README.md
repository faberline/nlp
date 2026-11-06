# Faberline NLP

A Rust library for finding candidate events and ordering them in Chinese narrative text. Results point back to spans in the original text. An end offset is exclusive. Offsets count Unicode scalar values from the start of each input.

Call `analyze` with the source text, its title, and a `Lexicon`. The caller supplies character names, alternate names, places, objects, rank terms, and publisher footer markers. The result includes events, text sections, and items that need review.

Call `timeline::build` with the texts and their candidate events. The function sorts the input by document number. Document numbers must be unique. The result includes the original event order, proposed story order, and the source spans for each relation.

AlterVerse Lab is the first consumer. It handles book import, chapter checkpoints, and storyboard exports. It runs the full novel comparison with its local corpus.

```sh
cargo test --locked
```

## English CNL

Use `cnl::parse` to read English SOM CNL v1. The grammar covers prompt versions 1 through 7. Pass the whole request as a string.

```rust
let source = std::fs::read_to_string("request.cnl")?;
let document = faberline_nlp::cnl::parse(&source)?;
assert_eq!(document.text, source.as_str());
```

The result is a tree of grammar rules. Each node gives its text and a byte range in the original input. A syntax error gives a line and column.

Use the [SOM examples](https://github.com/faberline/som/tree/05c63d08275d59ea7110cd8f2b30786070b91968/som-rs/contracts/compiler-v1/examples) for sample requests. SOM still checks the business rules. It builds the DSL and generates the code.
