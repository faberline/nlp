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
- `set`: `dev` or `test`, split inside each phenomenon. The `test` rows are fixed; new rows go to `dev`.
- `genre`: free label (`daily`, `dialogue`, `fiction`, `news`, `business`, `expository`).
- `phenomenon`: one of `case_particles`, `topic_subject`, `passive`, `causative`,
  `giving_receiving`, `negation`, `tense_aspect`, `modality`, `conditionals`,
  `relative_clauses`, `zero_pronoun`, `keigo_register`, `final_particles`,
  `counters_numbers`, `dates`, `katakana_names`, `onomatopoeia`, `idioms`,
  `polysemous_verbs`, `dialogue`, `polite_forms`.
- `ja`: one Japanese sentence.
- `zh_hant`: the main reference, natural Traditional Chinese as written in Taiwan.
- `zh_hant_alt`: a second acceptable reference. It may be empty.

The runner rejects a row whose source or primary reference is empty after whitespace removal
(an empty reference would give an empty output a chrF of 100). A blank alt reference is ignored.

No tab or newline inside a field.

### Current size

160 rows: 130 `dev`, 30 `test`.

| phenomenon | dev | test |
|---|---|---|
| `giving_receiving`, `zero_pronoun` | 15 each | 2 / 1 |
| `dialogue` | 15 | none yet |
| `negation` | 13 | 1 |
| `polite_forms` | 8 | none yet |
| the other 16 phenomena | 4 each | 1 or 2 each |

The two larger `dev` groups give a steadier signal for the engine work on giving and
receiving verbs and on omitted subjects. Each `zero_pronoun` row has at least one reference
without a subject, because Chinese can drop the subject too.

`dialogue` covers casual speech: colloquial endings and sentence-final particles, contractions
(`ちゃう`, `てる`, `じゃん`, `なきゃ`), vocatives, ellipsis and onomatopoeia in speech. It has no
`test` rows yet; when they are added, they are written new and never taken from `dev`.

`polite_forms` covers the polite verb endings `ません`, `ませんでした`, `ましょう` and `ませんか`
(negative, past negative, proposal, invitation or request). `negation` also has rows for the
existence negative (`ない`, `なかった`, `ありません`, `ありませんでした`) and the copular negative
(`ではない`, `じゃない`, `ではありません`, `じゃありません`, `ではありませんでした`), plain and polite.
`polite_forms` has no `test` rows yet; the same rule as for `dialogue` applies.

## Metric

- chrF per sentence: character n-grams 1 to 6, beta 2, whitespace ignored, 0–100.
  Precision and recall are averaged over the n-gram orders the reference has.
  The score of a row is the best score over its references.
- Exact match: the trimmed output equals one reference.
- `kana_left`: the output still contains hiragana, katakana or halfwidth katakana
  (fullwidth `ー` and `・` do not count).
- `passthrough`: the output equals the source, or the engine reports `TranslationTrack::Passthrough`.
- Averages are reported overall, per set, and per phenomenon for `dev` rows only.
- Report-only lines (`report\t…`), also stored as `#` comments in `baseline.tsv`, are not
  part of the ratchet:
  - `dev_without_katakana_names`: dev chrF without `katakana_names`. With an empty glossary
    the engine cannot transliterate names, so that group measures the glossary more than the rules.
  - `kana_left_dev` and `kana_left_overall`: rows with kana left in the output.

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

The runner compares current chrF values with the stored baseline. For `overall` and `set/dev`,
it rounds the current value to the same two decimal places stored in the baseline and fails on
any decrease. For every other aggregate key, it allows a decrease of up to 0.5 points. This
includes `set/test` and each `dev/<phenomenon>` key. The runner reports no per-row test details.

The runner also fails when `challenge.tsv` changed after the baseline was written (the baseline
stores its SHA-256), or when `baseline.tsv` does not contain exactly one finite chrF row for each
aggregate key (`overall`, `set/dev`, `set/test`, and one `dev/<phenomenon>` per phenomenon), with
no duplicate or unknown keys.

## How to update the baseline

First merge the engine and data changes that the baseline will reflect. Then start from a clean
checkout of the named `main` commit and measure that state. Generate a baseline only after those
changes are on `main`:

```bash
JA_ZH_EVAL_WRITE_BASELINE=1 cargo test -p faberline-nlp-translate --test ja_zh_eval -- --nocapture
```

Submit the resulting `baseline.tsv` change in its own PR, with no other changed files. The file
records the engine revision (`git rev-parse HEAD`) and the hash of `challenge.tsv` used for the
measurement.

The writer checks that these engine input paths have no uncommitted or untracked changes:
`crates/faberline-nlp-translate/src/`, `crates/faberline-nlp/src/`,
`crates/faberline-nlp/lexicon/`, and `crates/faberline-nlp-narrative/src/`. It then records the
current `HEAD`. This check does not require a clean worktree or a named `main` branch, and
uncommitted evaluation files do not block writing. Those facts do not authorize a baseline from
unmerged engine or data changes. Use the clean named `main` state described above. A reason in a
baseline PR cannot make a failing engine or data PR pass by lowering the baseline.

## Licence rule for sentences

Every sentence and every reference in this set is written for this repository by its
contributors. Do not copy sentences or translations from other corpora, dictionaries, books,
subtitles or websites. In particular, no material from CC BY-SA, NC, ND or GPL sources.
References are written before running the engine, and are never edited to match engine output.
