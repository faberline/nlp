# Faberline NLP

A Rust library for finding candidate events and ordering them in Chinese narrative text. Results point back to spans in the original text. An end offset is exclusive. Offsets count Unicode scalar values from the start of each input.

Call `analyze` with the source text, its title, and a `Lexicon`. The caller supplies character names, alternate names, places, objects, rank terms, and publisher footer markers. The result includes events, text sections, and items that need review.

Call `timeline::build` with the texts and their candidate events. The function sorts the input by document number. Document numbers must be unique. The result includes the original event order, proposed story order, and the source spans for each relation.

AlterVerse Lab is the first consumer. It handles book import, chapter checkpoints, and storyboard exports. It runs the full novel comparison with its local corpus.

```sh
cargo test --locked
```
