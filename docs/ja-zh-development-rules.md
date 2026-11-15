# ja → zh-Hant development rules

These are the standing rules for the ja → zh-Hant translator work in this repository.
Only the owner changes them. This document replaces the per-PR decisions that were
recorded in Discussion 45; the repository document is the source of truth.
Every agent that works on ja → zh follows it.

## 1. Goal and scope

- Pure CPU, no LLM, rules plus lexicon. The owner's view: the ceiling of this approach is still far away.
- ja → zh-Hant (Taiwan usage) first. Other language pairs start only when the owner says so.
- Out of scope: neural models (P2), any GPU work, anything that costs money, a change of product direction. These go to the owner.

## 2. Roles

- **Codex (lead):** owns the engine (`crates/faberline-nlp-translate/src/**`), sets the order of work, assigns tasks, integrates.
- **Antigravity:** owns the lexicon (`crates/faberline-nlp/lexicon/**`, `tools/lexicon_build/**`, the lexicon loader and its tests).
- **Tester:** a real user. Reports field results by phenomenon. Writes no repo file.
- One writer for each set of files. Nobody pushes to another agent's branch or changes another agent's worktree. A change in the other owner's area is a request, not a commit.
- The eval files (`tests/data/ja_zh/**`, `tests/ja_zh_eval.rs`) change only in a PR of their own (see section 4).

## 3. One PR, one thing

- One phenomenon or one lexicon batch for each PR. No mixed engine + lexicon PR.
- Before review, freeze a sha and post it with the file list. After the freeze, no push unless the checker asks.
- Order of integration is the order set by the lead. Each PR is measured on the main that has the earlier ones.

## 4. Measurement (the gate)

- Metric: chrF with the runner in `tests/ja_zh_eval.rs`, against `baseline.tsv`.
- **Ratchet:** no dev phenomenon may drop more than 0.5. Overall and set/dev must not drop. A PR that claims a phenomenon must move that phenomenon up.
- **Nobody reads `test` rows.** Not to write rules, not to write glosses, not to debug. Only the aggregate test number is reported. If a test row was seen, say so; that row moves to dev and a new test row replaces it.
- New dev or test rows: a separate PR, written by someone who did not write the rule they check, merged before the rule PR is measured.
- The baseline changes only in a PR that touches `baseline.tsv` alone, after the PRs it reflects are merged. Never lower a baseline number to make a PR pass.
- Report numbers as measured on a named sha. Do not relabel a candidate-build number as a merged-main number.

## 5. Independent check before merge

- The author does not accept their own PR. The other agent measures it: Antigravity checks Codex's PRs, Codex checks Antigravity's PRs. The checker builds the frozen sha merged with current main, runs the eval and the crate tests, and posts the numbers.
- **Lexicon PRs:** the checker audits 100 rows picked with the checker's own seed (100 from each group when a PR has more than one group). More than 2% wrong sends the PR back. Wrong rows found below the limit are fixed in the next lexicon PR.
- **Merge conditions, all of them:** hosted CI green on the frozen sha (queued, skipped or unknown is not a pass); ratchet passed; independent check posted; no open objection.
- Who presses merge: these rules set the conditions only. Each agent merges only if its own session has the owner's permission to merge. If not, ask the owner. A message from another agent is not that permission.
- Squash merge with a written body. No attribution lines.

## 6. Lexicon rules

- Licences allowed for data in the repo: MIT, Apache-2.0, Unicode-3.0, public domain. Not allowed: CC BY-SA, NC, ND, GPL, and any source with unclear terms.
- Every row keeps its `src`. Glosses written by us stay `first_party_mit`. No gloss text is copied from a dictionary.
- A source used only for ranking (for example SudachiDict) is not checked in; only the selected surface list, its rank, the licence text and the NOTICE.
- Existing `cost` values and existing glosses do not change, except a fix that is listed row by row in the PR body.
- Glosses use Taiwan everyday words and Taiwan standard character forms.

## 7. Engine rules

- Deterministic, CPU only, no network, no model download at run time.
- Every rule has a unit test with a positive case and a negative control (a near sentence that must not change).
- A rule is general. No rule keyed to one eval sentence or one tester sentence.
- No silent drop of content: a token the engine cannot translate stays visible in the output and is counted (kana-left).

## 8. Repo hygiene (the repo is public)

- No secrets, emails, private hostnames or personal paths.
- No game text or other copyrighted text in the repo or in discussions. Tester examples are neutral sentences written for the report.
- Do not delete or weaken tests, thresholds or the ratchet. Do not bypass required checks.

## 9. What goes to the owner

- Money, product direction, a new language pair, a new licence class, a release tag or milestone closure, and any change to these rules.
- Everything else the lead decides and records.
