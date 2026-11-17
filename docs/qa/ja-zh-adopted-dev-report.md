# Adopted development-visible report

Run `cargo test -p faberline-nlp-translate --test ja_zh_visible_dev_eval -- --nocapture` to print the report. The report is provisional and report-only. It is not a quality gate, a human accuracy measure, or acceptance of an N1–N5 level. The level labels are approximate. The references are AI-assisted and need human calibration.

The runner requires all three adopted banks. It verifies their pinned SHA-256 hashes, schemas, counts, exact ID and level membership, development visibility, held-out flags, licence and provenance, and reciprocal pairs before it translates any row. It rejects duplicate IDs across banks. A missing or invalid bank stops the run. It evaluates each ID once and prints actual membership and counts by bank and level, plus membership and counts by level across banks. It prints the Git HEAD revision and marks the dirty state as not checked; use a clean, frozen revision to attribute a result to a committed engine.

For each bank and level, the runner reports mean chrF and exact matches against `zh_hant` only. It does not choose the best score from `allowed_variants`. It also reports each bank hash. These numbers are supplemental signals. They do not replace or change the existing challenge, baseline, or ratchet.

The pinned membership is:

| Bank | N5 | N4 | N3 | N2 | N1 | Total |
| --- | --- | --- | --- | --- | --- | --- |
| `nlevels_dev_v1` | 4 | 4 | 4 | 4 | 4 | 20 |
| `honorific_san_dev_v1` | 6 | 6 | 0 | 0 | 0 | 12 |
| `nlevels_expansion_dev_v1` | 10 | 10 | 10 | 10 | 10 | 50 |
| Total | 20 | 20 | 14 | 14 | 14 | 82 |

The runner prints the exact IDs observed during each successful run. No sentence or per-item translation is printed.
