# Payment fare stem development proposal

**Status:** Proposed first-party development data. Independent review and formal adoption are pending. No new row has been translated.

This proposal adds eight N5-oriented diagnostic cases for the #410 payment-stem boundary. It is a data-only proposal. It does not change engine code or the adopted evaluation set.

The source main is `176932ea354d7a2904d3a9f1c87c033396778e73`. Its tree is `25fa5c53c03e5ff2e0567bb3e04385f402bd2288`. Every sample is marked `development-visible` and `heldout: false`. The new Japanese sentences and Taiwan Traditional Chinese references are first-party original text under MIT.

## Existing adopted controls

The proposal references `PAY-N5-05` and `PAY-N5-06` by ID in `payment_potential_dev_v1.json` (SHA-256 `55d2958174002be4b8e15f224921efe62d6221edbb309ff168d43970854e2250`). It does not copy either adopted row. Those two adopted cases cover ordinary affirmative card and cash payment.

## Proposed coverage

| ID | Case | Boundary to review |
| --- | --- | --- |
| `PFST-N5-01` | Completed past card payment | Keep the completed past event and unknown payer. |
| `PFST-N5-02` | Non-past cash negative | Keep the negative payment proposition. Do not infer inability or prohibition. |
| `PFST-N5-03` | Past cash negative | Keep both past time and cash-specific negation. Do not infer no payment by another method. |
| `PFST-N5-04` | Positive card question | Keep question force. Do not assert payment. |
| `PFST-N5-05` | Past-negative card question | Keep past, negative scope, and question force. Leave confirmation versus rhetorical force unresolved. |
| `PFST-N5-06` | Reported direct speech with `支払います` | Keep the utterance fact separate from whether a payment later happens or who pays. |
| `PFST-N5-07` | Configured `支払い` noun control | Apply the caller target `付款` as a noun. Do not add a finite payment action. |
| `PFST-N5-08` | Inline-code opaque control | Keep the literal unchanged. Fare, medium and stem terms inside it must have no hits. |

## Caller glossary and span evidence

Each item records its own fixture. The caller glossary keeps `切符代` as `票款`, uses `カードで` or `現金で` for the payment method, and configures `支払い` as `結清` for verb cases. The noun control maps the same surface to `付款`. The noun control keeps the `支払い` entry configured as `付款` for its noun reading and also uses a bare `現金` entry. The opaque control includes all protected terms inside backticks and expects no hits. The card source does not specify a credit card. The `用信用卡` target is a caller choice, not a fact from Japanese.

`lock_expectation` records exact source and target scalar and UTF-8 byte slices for each expected hit. It records explicit no-hit boundaries where a surface is inactive or opaque. These are author expectations only. Runtime behavior is unobserved. Reviewers should verify the full sentence meaning, Taiwan wording, quote and question punctuation, clause scope, exact slices, and source/target alignment.

## Adoption gate

A reviewer independent of the author should check the first-party provenance, all references and meanings, the per-row fixtures, duplicate status, every expected active hit and no-hit, and the full source/target spans. Merge and formally adopt the data before using these rows to measure or tune the #410 rule. Keep the later engine PR separate.
