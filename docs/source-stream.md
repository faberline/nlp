# Experimental source stream

`nlp_source_stream` emits source-linked observations as UTF-8 JSON Lines.
Each line is one JSON object. The stream uses source order; it does not
establish story order. Its observations do not establish a person's identity,
a grammatical role, an event, or a quotation's speaker.

This experimental CLI currently uses the Chinese (Taiwan) tokenizer. English
stream extraction is not validated. Some narrative heuristics still contain
genre-specific terms; callers should not treat them as a domain-independent
semantic model.

## Input

Pass a UTF-8 manifest with `--manifest`, and select `--all` or a comma-separated
list with `--chapters`. Chapter numbers are positive and unique. The program
processes selected chapters in number order. Each selected chapter must have
`status: "ok"`. Chapter files must resolve inside the manifest directory.
The manifest and each chapter are limited to 16 MiB.

The manifest, vocabulary, and chapter directory must be trusted. Other
processes must not modify them while the program runs. Path canonicalization
checks for a static path escape, but it does not prevent a time-of-check to
time-of-use (TOCTOU) race. The program cannot safely handle untrusted
concurrent changes to these inputs or their paths.

This manifest describes three synthetic files:

```json
{
  "count": 3,
  "chapters": [
    {"order": 1, "title": "First", "file": "chapter-1.txt", "status": "ok"},
    {"order": 2, "title": "Second", "file": "chapter-2.txt", "status": "ok"},
    {"order": 3, "title": "Third", "file": "chapter-3.txt", "status": "ok"}
  ]
}
```

The caller may pass a JSON vocabulary with `--work-lexicon`. Omitted categories
are empty. For example:

```json
{
  "actors": [{"surface": "小甲", "canonical": "person-a"}],
  "objects": [{"surface": "劍", "canonical": "object-a"}],
  "places": ["木屋"],
  "footers": ["<end>"],
  "line_footers": ["<line>"]
}
```

The format also accepts `ranks`, `formed_ranks`, and `rank_blocked_prefixes`.
Actor, object, and rank entries have `surface` and `canonical` fields. Entries
must be nonempty, have no outer whitespace, and be unique within each category.
Each `formed_ranks` item must match a rank surface. These labels are caller
inputs, not verified identities. Without a vocabulary file, the program uses an
empty vocabulary. This differs from the earlier work-specific prototype, so
its default output is not promised to match that prototype byte for byte.

For example, with the files above and caller-created chapter text:

```sh
cargo run --bin nlp_source_stream -- \
  --manifest manifest.json --work-lexicon work.json --all \
  --cross-chapter-candidates --scoped-written-observations --written-quotes
```

The optional flags `--written-time-cues`, `--written-transfer-cues`,
`--written-transfer-role-candidates`, `--written-transfer-role-alternatives`,
`--written-predicate-observations`, and `--spelling-review` request their named
observations. Spelling also has optional `--spelling-start-char`,
`--spelling-end-char`, `--spelling-work-budget`, and
`--spelling-max-alternatives` controls. A spelling range needs both endpoints
and exactly one selected chapter. Spelling observations are not corrections.

## Versions and source positions

The `stream_start.schema` value selects the record format:

| Flags | Schema |
| --- | --- |
| Neither cross-chapter nor scoped-written flag | `avl-source-stream-v1` |
| `--cross-chapter-candidates` only | `avl-source-stream-v2-cross-candidates` |
| `--scoped-written-observations`, with or without cross-chapter | `avl-source-stream-v3-scoped-written` |

In v3, `cross_chapter_candidates_enabled` states whether the cross-chapter flag
is also on. Older consumers must reject schemas and record types they do not
understand. A v2 or v3 consumer must validate its own input before accepting
records; the old v1 verifier does not verify these formats.

Byte and Unicode scalar positions refer to the complete original chapter, not
to filtered or concatenated text. Ranges are half-open: the start is included
and the end is excluded. A scalar is a Unicode code point; it is not always a
visible character. Both coordinate systems must select the same UTF-8 text.
The chapter SHA-256 binds a position to the bytes the producer read. A hash is
not proof of source authenticity. `producer_binding` is `not_recorded`.

## Cross-chapter records

When enabled, each selected chapter, including an empty chapter, has a
`cross_chapter_summary`. It contains `chapter`, `chapter_sha256`, an
`observation`, and `candidate_records`, the count of following candidates for
that chapter. Each `cross_chapter_candidate` has `prior_name` and
`current_pronoun` endpoints. Each endpoint includes its chapter number, byte
range, scalar range, and chapter SHA-256. The basis is
`cooccurrence_in_configured_history_window`; identity, role, association,
occurrence, and narrative scope remain `unknown`.

Names are exact caller-vocabulary surfaces. Pronouns are only the configured
written forms that the tokenizer finds. A pair is not a resolved reference.
Different surfaces with the same caller label are not thereby proven to be the
same person. Candidate order is not a confidence ranking.

The policy examines at most the previous two chapter numbers. Its summary
records the actual policy and limits: 65,536 vocabulary terms, 1 MiB of term
bytes, 256 names and 256 pronouns per chapter, 32 pairs per pronoun, and 4,096
pairs per chapter. Input over a hard input limit is an error. A `complete`
inspection status means complete only within the stated policy; coverage is
still `unknown`. `pair_limit_reached` records omitted pairs.

`missing_chapter_ranges` uses inclusive chapter-number endpoints.
`history_incomplete`, `partial_name_chapters`, and
`history_recheck_required` report missing or limited history.
`earlier_chapters_outside_window` does not imply those chapters exist or were
read. `recheck_from_scalar` is a conservative hint for the current chapter,
not a resume cursor. Rechecking that chapter alone cannot repair missing or
truncated history.

## Scoped quotations and transfer observations

With `--scoped-written-observations`, quotation and transfer flags use separate,
contiguous admitted source segments. Filtered gaps are hard boundaries;
cross-gap associations are `not_examined`. The mode emits one
`written_scoped_summary` per selected chapter. Its `segments` list maps each
`segment_id` to a range in the original chapter. It also records counts for
the enabled quotation and transfer record types. No observation category is
enabled by the scope flag alone.

When quotations are enabled, `written_scoped_quote_segment` carries one
segment's report and its chapter SHA-256. Quote and cessation indices are local
to that chapter and segment. Do not join indices across segment reports. The
report retains nested quotes, diagnostics, speech cues, cessation cues, and
written claims with source ranges. Detected crossed delimiters can yield a
`crossed_delimiter` diagnostic for an unexamined quote-scan suffix. Some quote
marks also have apostrophe interpretations, so not every apparent crossing is
necessarily detected. The diagnostic concerns quote scanning; other analysis
may already have run.

Enabled transfer flags emit `written_scoped_transfer_cue`,
`written_scoped_transfer_role_candidate`, or
`written_scoped_transfer_role_alternatives` records. Their observations retain
`segment_id` and original-source ranges. Written direction is not a confirmed
event direction. Identity, roles, occurrence, association, and narrative scope
remain unknown.

The scope flag applies only to these quotation and transfer paths. Without it,
the legacy quotation and transfer paths inspect the full input and can include
excluded footer text. The ordinary chapter, cross-chapter, time, predicate,
and spelling paths are not covered by this scoped-mode claim. In particular,
the scoped chapter-footer matcher folds ASCII case in both text and marker,
while legacy chapter and cross-chapter matching can miss an uppercase marker.
Line-footer and built-in HTML marker matching is case-sensitive. These rules
do not constitute general footer or HTML recognition.

## Receipt and consumer checks

The final `stream_complete` is a receipt. `prior_records` counts the lines
before it. `prior_records_sha256` hashes the exact bytes of those preceding
lines, including their newline characters and excluding the receipt. The
receipt contains selected and completed chapter counts. When cross-chapter is
enabled, it also contains summary and candidate totals. With scoped mode it
contains `scoped_written_summaries` and the scope status.

All records before a valid receipt are provisional. A consumer must obtain
the entire stream, require a successful process exit, and check the prefix hash,
record counts, chapter completion records, per-chapter candidate counts,
missing ranges, limits, and vocabulary. It must read the exact original chapter
bytes and validate every source range and both candidate endpoints against the
chapter hashes. Historical endpoints refer to the snapshot read earlier; the
producer does not reopen that chapter to confirm it has not changed. A matching
hash does not authenticate the producer, the source, or the vocabulary.

The 16 MiB chapter limit does not imply fixed memory use or bounded JSONL line
length. Some components retain the previous full chapter. Scoped processing
collects segment data and reports; many short segments or a large report can
use substantial memory. Empty output does not prove that no relationship or
event exists.

## Use of `nlp.doc.v1`

For zh-TW chapters the stream reads sentences, actor name hits and written pronouns from the
`nlp.doc.v1` layers, only for lines that pass an allow list: Han ideographs, `。！？，、；：「」『』`,
a single inner space, a U+3000 paragraph indent and a final newline. The line must also pass a
context rule: each run of `。！？`, with the `」』` and spaces after it, is followed by a Han
ideograph or by the end of the line (otherwise ICU and layer 1 end sentences at different
places, for example in `「好。」，他說。`). Any other line (ASCII letters, digits or punctuation,
`…`, other brackets, emoji, tabs, opaque spans) keeps the older splitter and tokenizer, so its
records and ranges are those of the older path. The name terms are the lexicon actors, objects,
ranks and places. Sentences and names come from the analyzer; written pronouns come from the
cross-chapter reader, which uses the same allow list. See `doc-v1.md`, section
"Readers of `nlp.doc.v1`".
