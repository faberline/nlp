# N5 adopted seed coverage and gaps

This record fixes the current N5 seed entries for later planning. It does not define the full N5 acceptance bank. The small seed cannot prove full-level quality.

## Source and adopted banks

The historical inventory source is main commit `7edd2e03468b3f2f02cd20d33cdff2acda21005c`, tree `7e1b428ede304e8be37b0d0a80d4050c729a0c16`. The table records that snapshot. Its さん bank hash is the original, not the later metadata revision.

| Adopted bank | Source file | Schema and version | Rows | SHA-256 | Licence | Adoption record |
|---|---|---|---:|---|---|---|
| N1–N5 development bank | [nlevels_dev_v1.json](../../crates/faberline-nlp-translate/tests/data/ja_zh/nlevels_dev_v1.json) | `faberline.nlp.ja_zh.nlevels_dev.v1`, `1.0.0-dev.1` | 20 | `a24d91502b3d361eec1ff2bf97381d0accbf27d7516ba2188e77b3b4e96695c0` | `first_party_mit` | [PR #71](https://github.com/faberline/nlp/pull/71), merged |
| Neutral `さん` development bank | [honorific_san_dev_v1.json at 7edd2e](https://github.com/faberline/nlp/blob/7edd2e03468b3f2f02cd20d33cdff2acda21005c/crates/faberline-nlp-translate/tests/data/ja_zh/honorific_san_dev_v1.json) | `faberline.nlp.ja_zh.honorific_san_dev.v1`, `1.0.0` | 12 | `870e9e759b65d9f75c277517792315f1cbd0b63ae147c5685690f0eab4f62177` | `first_party_mit` | [PR #73](https://github.com/faberline/nlp/pull/73), merged |

The [current metadata revision](../../crates/faberline-nlp-translate/tests/data/ja_zh/honorific_san_dev_v1.json) of the 12-row さん bank has SHA-256 `280da5014f9069994bce1f23e8cdc10c3cad3200450278c9b4e3618231134229`. It changes pair descriptions only. The preceding main `0996e8bad59175d1e71529568486b85c7f3c102b` still has the original hash in the table. The report-only evaluator binds the current file hash. Its review and adoption are recorded by [issue #121](https://github.com/faberline/nlp/issues/121) and its separate PR.

[PR #78](https://github.com/faberline/nlp/pull/78) reports adopted development-bank counts by approximate level. The public PR links record the changes. Their author heads are not merge commit IDs.

All 32 rows are development-visible and have `heldout: false`. The project level labels are approximate, not JLPT certification. The Chinese references are provisional and are not gold references.

| Approximate level | N1–N5 bank | `さん` bank | Total |
|---|---:|---:|---:|
| N5 | 4 | 6 | 10 |
| N4 | 4 | 6 | 10 |
| N3 | 4 | 0 | 4 |
| N2 | 4 | 0 | 4 |
| N1 | 4 | 0 | 4 |
| **Total** | **20** | **12** | **32** |

## Fixed N5 seed entries

Use these ten IDs, in this order, as the current N5 seed. The L5 IDs come from [nlevels_dev_v1.json](../../crates/faberline-nlp-translate/tests/data/ja_zh/nlevels_dev_v1.json). The HS IDs come from [honorific_san_dev_v1.json](../../crates/faberline-nlp-translate/tests/data/ja_zh/honorific_san_dev_v1.json). Preserve their source bank and IDs.

| ID | Meaning and completeness obligations | Control and limit |
|---|---|---|
| L5-01 | Keep the speaker, every-morning habit, drinking action, and water. | Positive member of the L5-01/L5-02 polarity pair. One speaker, action, and object only. |
| L5-02 | Keep the speaker, habitual time, water, and negated drinking. Keep nonpast time. | Negative member of that pair. One negated verb in one context. |
| L5-03 | Keep the named traveler, yesterday, past travel, and school destination. Add no gender. | Positive member of the L5-03/L5-04 past-polarity pair. It does not contrast destination with origin or location. |
| L5-04 | Keep the named traveler, yesterday, past negation, and school destination. Add no gender. | Negative member of that pair. It does not test other destinations or routes. |
| HS-01 | Keep the named person as presently at the reception desk. Preserve positive present existence/location. Add no gender or gendered title. | HS-02 changes polarity and changes `が` to `は`, so the pair is not minimal and does not isolate polarity. |
| HS-02 | Keep the named person as not being at the reception desk now. Do not imply that the person does not exist elsewhere. Add no gender. | Both rows use a personal-name honorific, so both are positive examples for that behavior. They also change polarity and `が` to `は`. Neither is a feature-specific near-negative control. |
| HS-09 | Keep the mother relationship, evening time, return, and home destination. | Near-negative lexical control: `さん` is part of a kinship term, not a personal-name suffix. Not a matched name pair. |
| HS-10 | Keep the clerk as actor, the order as object, confirmation, and past time. Add no direction or gender. | Near-negative role control: `さん` follows an occupation. Not a minimal contrast with a personal-name sentence. |
| HS-11 | Keep the saury identity (秋刀魚), grilling action, dinner, and past event. Treat the whole ordinary word as lexical content. | Near-negative substring control: the word contains `さん`, but it is not an honorific. Sentence structure differs from HS-12. |
| HS-12 | Keep the teacher's relative-clause role as the person responsible for the class, the teacher as giver, the named recipient, the materials, after-class time, and the past transfer event. Add no gender to either person. | Retains its approximate N5 label. Its relative clause and giver/recipient roles are not isolated basic N5 proof. It is a name-and-role case, paired with an ordinary-word control, not a structural minimal pair. |

HS-01, HS-02, and HS-12 contain personal names. The other three `さん` entries are a kinship term, an occupation, and an ordinary word. Do not treat all six as name cases. Do not silently change labels or row content. If a level relabel is justified, use a separate versioned proposal.

## Coverage in the selected ten

These tags may overlap. They describe this seed only. They are not new exclusive groups or evaluation gates. Coverage in the rest of adopted development data is unknown here.

| Project area | Selected-ten status | Evidence and limit |
|---|---|---|
| Polarity | Observed, thin | Positive and negative forms appear in three contexts. HS-01/02 also vary the subject/topic marker. |
| Time | Observed, thin | Habitual present, past, evening, and after-class time occur. Relative time is present; explicit calendar dates are absent from these ten. |
| Topic and subject | Observed, thin | Speaker, named person, and occupation roles occur in narrow contexts. HS-01/02 do not isolate marker choice. |
| Places and destinations | Observed, thin | School and home destinations and reception location occur. This is not a matched destination/origin/location set. |
| Common verbs | Observed, thin | Drinking, going, existing, returning, confirming, cooking, and giving occur in limited contexts. |
| Numbers and calendar dates | Absent in selected ten | No explicit numbers or calendar dates occur in these ten. This is not evidence of a product failure or of gaps in other adopted data. |
| Names and neutral address | Observed, thin | Three of six `さん` entries use personal names. The other three are lexical controls. |

## Data and rule boundaries

Data adoption is separate from engineering. At the historical inventory source, the PR #74 proposal was not adopted. [Issue #86](https://github.com/faberline/nlp/issues/86) records its review and current status. Do not count or execute its rows before adoption. [Issue #93](https://github.com/faberline/nlp/issues/93) splits observed rule roots. [Issue #95](https://github.com/faberline/nlp/issues/95) owns lexicon work; it does not authorize evaluation data in a lexicon change.

The queued data follow-up [#120](https://github.com/faberline/nlp/issues/120) covers explicit numbers and calendar dates after #85 and #86. [#121](https://github.com/faberline/nlp/issues/121) reviews existing `さん` control metadata. Its revision preserves labels, content, and IDs. Any proposed reclassification needs a separate versioned proposal.

[Issue #89](https://github.com/faberline/nlp/issues/89) owns standing-reference triage. [Issue #90](https://github.com/faberline/nlp/issues/90) owns adoption of the separate grouping map. Keep the original 24 evaluation rules, baseline, and ratchet unchanged. A new data author must differ from the rule author. Sentences and references must be first-party work, use MIT, Apache-2.0, Unicode-3.0, or public-domain data, and be independently reviewed and adopted in a data-only PR before a rule that uses them is measured.

Every new rule needs a positive unit case and a near-negative control. There is no optional waiver. Confounded pairs alone cannot isolate a claimed gain. References are provisional. A string match, CI result, or score cannot replace independent checks of meaning, completeness, and natural Traditional Chinese for Taiwan.

Before [N5 acceptance issue #99](https://github.com/faberline/nlp/issues/99) freezes its acceptance bank, record its expanded membership, source hashes, reference policy, and controls. Resolve all current and later blockers. This record fixes seed entry criteria only; it does not show that the seed is large enough or set the final acceptance scope.

No current-main translation output or quality was examined for this record. All N levels remain unaccepted. After a milestone passes acceptance, the project requires a new full CPU NLP release with its seven CLI commands and embedded dictionary for macOS ARM, macOS Intel, and Windows. Keep historical v0.1.0 unchanged.
