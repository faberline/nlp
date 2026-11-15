# Japanese さん to Taiwan Traditional Chinese development QA bank

This supplemental bank has 12 original Japanese sentences. All rows are development-visible. Every row states `heldout: false`. The bank focuses on gender-neutral translation of さん attached to personal names and nearby controls that must keep their meaning.

## Authorship and references

Luna (gpt-6-luna) is the independent QA author. The Japanese sentences were composed with AI assistance for this bank. No exam, textbook, game, or other external sentence text was copied. Taiwan Traditional Chinese references and variants are AI-assisted candidate drafts. They are not model gold. No human review claim is made here.

Some personal names have more than one reasonable Chinese rendering. For アレックス, 亞歷克斯 is a provisional Chinese transliteration. Alex is an allowed target form, but the Japanese spelling does not establish a unique Roman spelling. The notes for HS-05 and HS-06 preserve this uncertainty.

The N5 and N4 labels are approximate sentence-level QA labels. They are open to calibration. They do not certify JLPT levels.

## Items and paired controls

| Pair | IDs | What the pair checks |
|---|---|---|
| 1 | HS-01 / HS-02 | Same 佐藤さん name and location, with positive versus negative presence. |
| 2 | HS-03 / HS-04 | Same さくらさん name and call recipient, with past versus next-week action. |
| 3 | HS-05 / HS-06 | Same アレックスさん name and document, with completed reading versus not-yet reading. |
| 4 | HS-07 / HS-08 | Explicit male description versus a teacher title. Preserve stated gender; do not infer gender from 先生. |
| 5 | HS-09 / HS-10 | お母さん is a kinship word; 店員さん is an occupation with a polite suffix. Preserve mother and clerk meanings. |
| 6 | HS-11 / HS-12 | さん in さんま is part of a fish name; 佐藤さん uses the personal-name honorific. Keep the teacher role in 先生. |

The participant checks preserve omitted or unspecified gender. Other checks cover polarity, time, role, recipient, and the difference between personal-name honorifics and ordinary lexical or role terms.

## Scope and adoption

Formal adoption and independent review belong in a separate data-only pull request. Record the exact review and adoption state there before any later rule uses these rows for acceptance. The bank author does not accept their own work.

Twelve sentences are a focused smoke set. Passing them does not show broad translation quality or complete handling of Japanese honorifics. The bank does not change an existing baseline or held-out data.
