# Existence and location DEV proposal v1

Status: **PROPOSED / UNADOPTED**. These rows are not registered for measurement. An independent checker must review every row after all seven applicable checks pass on the frozen head.

This data-only proposal contains ten first-party cases. The Japanese and complete Taiwan Traditional Chinese references are provisional AI-assisted text. They are not human gold or level certification. The author did not write the issue 94 rule. All new text is `first_party_mit`.

| Pair | Control relationship |
| --- | --- |
| EL-A01 / EL-A02 | Same cat, yard and markers; opposite existence polarity. |
| EL-R01 / EL-R02 | Same shelf-top relation and box; past absence versus past placing action. The actor in the second case is implicit. |
| EL-G01 / EL-G02 | A typed glossary place participates in existence; a typed glossary actor is a recipient of an action. |
| EL-O01 / EL-O02 | An opaque participant is absent; the same opaque literal is a recipient of an action. Its content has no assumed meaning. |
| EL-S01 / EL-S02 | Both clauses reverse polarity; negation stays with its own predicate. |

EL-R01 explicitly anchors past time in the reference. EL-G02 uses a gender-neutral target name for the recipient. EL-S01 and EL-S02 preserve the nonpast action contrast without asserting ability or completed action.

The complete fixtures and all source and reference spans are in the JSON. Scalar and UTF-8 offsets are zero-based and half-open. Opaque spans include backtick delimiters. Actual tokenizer, opaque detection, glossary matching, placeholder restoration and target positioning have not been run or verified. Static source inspection establishes the public fixture shape, not runtime reachability.

The 259 adopted DEV records were compared by exact Japanese and NFKC Japanese, including all pairs within this proposal. There were no text matches. The comparison includes complete fixture equality when text matches. Same text with different fixtures would be retained only with an explicit technical reason. Aliases CB-B10, CB-B25 and CB-L01 are not extra rows. No held-out rows or old API outputs were used.

Existing N5X-01/02 and HS-01/02 provide partial animate controls; HS changes particle and includes title and lexical confounds. jz155 provides past relational absence; jz056, jz099 and jz124 give distinct relational or action guards. CLOCK-C10 through C13 and jz002 protect recipients and destinations. CB-L02 and CB-L08 are lock controls for clock events, not existence coverage. N2-D03, N2-D04 and N1-D06 add complex negation guards but do not isolate this simple clause contrast. Existing IDs and their original fixtures remain unchanged.

These cases must not enter an active report or be used for rule measurement before independent adoption and merge. They do not change the original fixed seven cases, existing thresholds, baseline, or ratchet. CI and scores do not replace independent checks of complete meaning, naturalness, roles, polarity, tense, provenance, and every literal and span.
