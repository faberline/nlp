# ja → zh-Hant evaluation set

This folder holds the challenge set and the ratchet baseline for the deterministic
Japanese → Traditional Chinese (Taiwan) engine. The runner is
`crates/faberline-nlp-translate/tests/ja_zh_eval.rs`.

## Files

- `challenge.tsv`: the sentences and the references.
- `baseline.tsv`: the last accepted scores. The runner writes it. Do not edit it by hand.

## `challenge.tsv` format

Tab separated, UTF-8, one header line:

```
id	set	genre	phenomenon	ja	zh_hant	zh_hant_alt
```

- `id`: unique row id (`jz001`, …).
- `set`: `dev` or `test` (about 70/30, split inside each phenomenon).
- `genre`: free label (`daily`, `dialogue`, `fiction`, `news`, `business`, `expository`).
- `phenomenon`: one of `case_particles`, `topic_subject`, `passive`, `causative`,
  `giving_receiving`, `negation`, `tense_aspect`, `modality`, `conditionals`,
  `relative_clauses`, `zero_pronoun`, `keigo_register`, `final_particles`,
  `counters_numbers`, `dates`, `katakana_names`, `onomatopoeia`, `idioms`,
  `polysemous_verbs`.
- `ja`: one Japanese sentence.
- `zh_hant`: the main reference, natural Traditional Chinese as written in Taiwan.
- `zh_hant_alt`: a second acceptable reference. It may be empty.

The runner rejects a row whose source or primary reference is empty after whitespace removal
(an empty reference would give an empty output a chrF of 100). A blank alt reference is ignored.

No tab or newline inside a field.

## Metric

- chrF per sentence: character n-grams 1 to 6, beta 2, whitespace ignored, 0–100.
  Precision and recall are averaged over the n-gram orders the reference has.
  The score of a row is the best score over its references.
- Exact match: the trimmed output equals one reference.
- `kana_left`: the output still contains hiragana, katakana or halfwidth katakana
  (fullwidth `ー` and `・` do not count).
- `passthrough`: the output equals the source, or the engine reports `TranslationTrack::Passthrough`.
- Averages are reported overall, per set, and per phenomenon for `dev` rows only.

The engine runs with an empty glossary, `TargetVariant::ZhHant` and no model directory,
through `analyze_and_translate`, one sentence at a time.

## The dev/test rule

- `dev` rows are for development. Read their outputs, write rules and lexicon entries for them.
- `test` rows are held out. Nobody reads `test` outputs or failures to write rules or lexicon
  entries. The runner never prints `test` sentences, outputs or per-row scores; it prints only
  the `set/test` aggregate (and the `overall` number, which includes it).
- Do not move a row between `dev` and `test` to change a score.

## How to run

```bash
cargo test -p faberline-nlp-translate --test ja_zh_eval -- --nocapture
```

Per-row details for `dev` rows:

```bash
JA_ZH_EVAL_VERBOSE=1 cargo test -p faberline-nlp-translate --test ja_zh_eval -- --nocapture
```

The test fails when a chrF number in `baseline.tsv` (overall, a set, or a `dev` phenomenon)
is lower in this run by more than 0.5. It also fails when `challenge.tsv` changed after the
baseline was written (the baseline stores its SHA-256), and when `baseline.tsv` is not exactly
one finite chrF row for each aggregate key (`overall`, `set/dev`, `set/test`, one `dev/<phenomenon>`
per phenomenon), without duplicates or unknown keys.

## How to update the baseline

After an engine change that raises the scores, or after a change to `challenge.tsv`:

```bash
JA_ZH_EVAL_WRITE_BASELINE=1 cargo test -p faberline-nlp-translate --test ja_zh_eval -- --nocapture
```

Commit `baseline.tsv` with the change. It records the engine revision (`git rev-parse HEAD`
at the time of writing) and the hash of `challenge.tsv`. Writing fails when an engine input
(`crates/faberline-nlp-translate/src/`, `crates/faberline-nlp/src/`, `crates/faberline-nlp/lexicon/`,
`crates/faberline-nlp-narrative/src/`) has uncommitted or untracked changes, so the recorded
revision always names the engine that produced the numbers. Commit the engine change first.
Uncommitted evaluation files do not block writing. A baseline that lowers a number needs
a reason in the PR description.

## Licence rule for sentences

Every sentence and every reference in this set is written for this repository by its
contributors. Do not copy sentences or translations from other corpora, dictionaries, books,
subtitles or websites. In particular, no material from CC BY-SA, NC, ND or GPL sources.
References are written before running the engine, and are never edited to match engine output.
