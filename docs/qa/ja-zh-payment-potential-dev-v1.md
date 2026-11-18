# Payment ability and means controls

This bank contains six original public development examples. The first-party examples and Taiwan Traditional Chinese references use the MIT license. The N5 label is approximate. [Issue #184 scope](https://github.com/faberline/nlp/issues/184#issuecomment-6992515652)

| Rows | What to check |
| --- | --- |
| `PAY-N5-01` and `PAY-N5-02` | The speaker checks the shop’s payment methods. Both questions ask about customers using a phone to pay. The second asks whether customers cannot do so. |
| `PAY-N5-03` and `PAY-N5-04` | Both statements keep the older brother as payer and this month’s rent as the payment object. The verb states whether he can pay. The reason and payment method remain open. |
| `PAY-N5-05` and `PAY-N5-06` | The ticket counter is the place where the ticket is bought. The second sentence states an ordinary payment action. Check that action and the payment means separately. The first row specifies a card, and the second specifies cash. |

The third-person shopping context supports the literal reading of the negative question. Other contexts can change what the same ending does. The ability statements leave the actual payment open.

The existing `POLITE-N5-05` and `POLITE-N5-06` rows check affirmative and negative ability questions about the same dog swimming. The bank links their file hash and IDs so each row can be replayed from its original source. [Adoption record for the swimming controls](https://github.com/faberline/nlp/issues/181#issuecomment-6991875537)

Use the same source settings as the swimming bank. The input has no source hint. The target is `zh-Hant`, with an empty glossary and no model directory. The fixture records the `Onnx` backend label and the full translation options. [Payment ability and means bank](../../crates/faberline-nlp-translate/tests/data/ja_zh/payment_potential_dev_v1.json)

Review the Japanese meaning, Taiwan wording, and reference variants before adoption. This bank records source sentences and expected meanings. Runtime results are recorded during acceptance.
