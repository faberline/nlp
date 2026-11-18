# Kana and bag question controls

This file records two original development-visible examples. They use the same simple Japanese sentence about an older sister carrying a bag.

`KBQ-N5-01` is a statement. `KBQ-N5-02` adds only the final particle `か`. The Japanese period stays the same. The Taiwan Chinese references mark the statement with `。` and the question with `嗎？`.

The noun is written as `かばん` in both rows. The pair helps distinguish that noun from a sentence-final question particle. It does not show how the tokenizer or translator handles either sentence.

The adopted `N5X-05` example contains `か` inside `みかん`. It is a useful nonfinal-kana control. The adopted `N4X-05` example asks a permission question with final `か`. It is a useful question-form control, but it is not an N5 example. Neither reused row is a complete match for this pair.

The two rows are original first-party data with an MIT license. Their references are provisional. An independent reviewer must check the Japanese meaning and Taiwan wording before adoption. Do not treat these rows as runtime results or as proof of a correct translation.
