# Clock time development controls v1

Status: proposed. Independent review and adoption are pending. Do not run these rows through the engine before adoption and merge. The data-only adoption record belongs in [issue #131](https://github.com/faberline/nlp/issues/131). The rule dependency is [issue #130](https://github.com/faberline/nlp/issues/130). The safe standing development export is associated with [issue #86](https://github.com/faberline/nlp/issues/86).

The bank contains exactly 13 new rows, CLOCK-C01 through CLOCK-C13. The JSON file holds the full sentences, references, acceptable variants, semantic obligations, unknowns and links. CLOCK-C IDs in the matrix refer to that file. Existing IDs refer to the source bindings below; they are reused, not copied into the new bank.

Japanese sentences and provisional Taiwan Traditional Chinese references were authored with AI assistance for this bank. They were written before any new-row engine execution. No external sentence or gloss source was used. The bank and each row use first_party_mit. Level labels are approximate planning labels, not official JLPT items or human certification. Duration and fraction rows are negative diagnostics, not a higher-level development program.

The [clock time development bank](../../crates/faberline-nlp-translate/tests/data/ja_zh/clock_time_dev_v1.json) uses schema `faberline.nlp.ja_zh.clock_time_dev.v1` and version `1.0.0`. Its SHA-256 is `8e096047bfcae7658c6e84cb0e6a42d55ba6d10c4b52cb3ba3e3f3fe8c377050`.

## Coverage matrix

- **hour**: N5X-03 / CLOCK-C01.
  Stable: meeting, AM, whole, predicate. Changed: 九 to 二. Limit: single other value; midnight, noon, irregular readings untested.
- **clock half and time of day**: CLOCK-C01 / CLOCK-C02, CLOCK-C03 / CLOCK-C04, CLOCK-C01 / CLOCK-C03, CLOCK-C02 / CLOCK-C04.
  Stable: meeting, hour, predicate. Changed: one of 半 or 午前/午後. Limit: one hour in new bank; earlier N5X-03/04 cover nine.
- **numeric representation**: CLOCK-C04 / CLOCK-C05.
  Stable: meeting, PM, time and event. Changed: 二 to ２. Limit: fullwidth digit only; no claim for all forms.
- **event**: CLOCK-C04 / CLOCK-C06.
  Stable: clock and start predicate. Changed: meeting to lesson. Limit: noun change alone; no distinct syntax claim.
- **clock half versus elapsed and fraction**: CLOCK-C04 / CLOCK-C07, CLOCK-C07 / CLOCK-C08, CLOCK-C07 / CLOCK-C09.
  Stable: meeting and numeral two for first contrast; duration and predicate for second. Changed: first: 時/時間, PM and predicate; second: 半; third: noun and full construction. Limit: first and third are near semantic controls, not minimal pairs; duration/fraction are negative guards.
- **temporal versus recipient and destination**: CLOCK-C10 / CLOCK-C11, CLOCK-C12 / CLOCK-C13.
  Stable: recipient plus letter-writing or destination plus movement. Changed: addition/removal of PM clock phrase and an occurrence of に. Limit: cannot isolate role with a single substitution; paired sentences have different number of particles; role preservation required in C10/C12.

## Reused development controls

- N5X-03, N5X-04 — `nlevels_expansion_dev_v1.json`. existing same event and morning nine whole/half; reuse, no copied rows. Limit: single hour and same event.
- N5X-01, N5X-02 — `nlevels_expansion_dev_v1.json`. non-temporal location に. Limit: existential event differs from clock.
- N5X-06, N5X-09, N5X-10 — `nlevels_expansion_dev_v1.json`. distant numeral, counter and floor safeguards. Limit: different referents and constructions.
- L3-01, L3-02 — `nlevels_dev_v1.json`. event-defined temporal boundary safeguards. Limit: no numeric clock, complex confounded clauses.
- jz002, jz020, jz073, jz075, jz078, jz080 — `issue86-standing-dev-export-059c10a.json`. recipient, duration, abstract time, counters and dated PM clock safeguards. Limit: not close clock-vs-duration or role pairs; jz080 includes date and から; standing references may be provisional.

## Source bindings

The repository bank sources are bound to main commit `3fa828813a5955a12d076d74e2be23f0dc638338` and tree `29d28a8b0239cbf97241428a316af8ef86b80a58`. The standing export is a safe development-only export, not the challenge file. Its source head is recorded below.
- `nlevels_dev_v1.json`: version or source head `1.0.0-dev.1`; SHA-256 `a24d91502b3d361eec1ff2bf97381d0accbf27d7516ba2188e77b3b4e96695c0`; 20 rows.
- `honorific_san_dev_v1.json`: version or source head `1.0.0`; SHA-256 `280da5014f9069994bce1f23e8cdc10c3cad3200450278c9b4e3618231134229`; 12 rows.
- `nlevels_expansion_dev_v1.json`: version or source head `faberline.nlp.ja_zh.nlevels_expansion_dev.v1`; SHA-256 `22d0753fd384ae5f07a118c6417c53b213cc66c1637a8369b8c0315e69ab4fcc`; 50 rows.
- `issue86-standing-dev-export-059c10a.json`: version or source head `059c10a260ec75d92ee0867c1649ddc0a0db10be`; SHA-256 `31776ea83cc2046ef84f905f5ba2bdfee7261aaa68124cc3405cdad05441555d`; 130 rows.

All 212 permitted source rows were compared with the planned 13 Japanese sentences using NFKC and whitespace-normalized exact equality. No exact matches were found. This does not detect paraphrases or prove semantic uniqueness. No held-out rows or original challenge file were read.

## Limits and next review

The first clock-versus-duration comparison changes the time expression and predicate. The fraction comparison changes the noun and construction. Neither is a strict minimal pair. The role comparisons add or remove a clock phrase and one particle. They retain the recipient or destination and the event. The event comparison changes only the event noun; it does not establish coverage of a different syntax.

Contact between the future clock rule and locked or glossary spans is unknown. Before accepting the future rule, an independent check must identify actual contact and bind applicable portable controls to their sources, versions and hashes. If contact cannot be determined, it remains unknown. This bank does not establish locked-span safety.

An independent checker must review every sentence, reference, variant, semantic obligation, unknown and contrast. Static checks do not establish semantic acceptance. All seven applicable exact-head hosted checks must pass before formal PR diff review. Existing crate and evaluation checks and the original 24-item ratchet remain required. This bank is not registered with an evaluator and is not included in the existing 82-row report. This data proposal does not claim rule improvement, general clock safety or N5 acceptance.
