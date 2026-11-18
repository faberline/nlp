# Japanese name boundary DEV controls

This file records the authoring-stage snapshot for three proposed first-party #140 controls. The bank status is `PROPOSED_UNADOPTED` in this snapshot. It does not change an engine rule, lexicon, evaluator, baseline, or metric. Do not measure a rule against these rows until the data is independently reviewed and merged. Issue #140 should record any later independent adoption and merge with the exact commit and bank digest; this snapshot does not make a timeless claim about later status.

The rows are `HB-A01`, `HB-O01`, and `HB-L01` in `crates/faberline-nlp-translate/tests/data/ja_zh/honorific_boundary_dev_v1.json`.

- `HB-A01` uses an actor glossary entry for the full name `山田花子`. The protected surface ends before the separate `さん`. The reference keeps the configured name and does not add gender or a title. This differs from `EL-G02`, whose actor surface includes `さん`.
- `HB-O01` uses an inline opaque marker followed by external `さん`. The marker remains unknown. The reference keeps it unchanged and adds no identity or gender claim. Its expected source and reference spans are stored as Unicode scalar and UTF-8 byte offsets.
- `HB-L01` uses the complete all-hiragana pronoun `みなさん` before `は`. The repository Japanese lexicon tags the exact spelling as `PRON`. The source obligation is to keep the collective meaning instead of treating the word as a name plus a separate suffix. This case is not a minimal pair with the existing mother or fish-word examples.

All rows use development-visible data, `heldout=false`, and `first_party_mit`. References are provisional. The source offsets and reference positions are author expectations only. Actual tokenizer contact, lock restoration, naturalness, and semantic correctness remain unverified until the independent checker reviews the bank and a later authorized measurement runs after adoption.

The bank uses the existing portable Onnx API fixture defaults. `HB-A01` adds one typed actor glossary entry whose canonical value and surface are the full name without `さん`; the other glossary lists are empty. The other rows use an empty glossary. All rows set `models_directory` to null.
