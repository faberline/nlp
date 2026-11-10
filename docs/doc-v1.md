# nlp.doc.v1, layer 1

`faberline_nlp::doc` returns a JSON `Document` with paragraphs, sentences, clauses and a
language tag per span. It uses no model and no network. Opaque spans (code, formula, heading,
list, citation, URL, emoji) are not yet covered (#24).

CLI: `nlp_doc` reads UTF-8 text from stdin and prints one JSON document plus a newline. On a
read error, including invalid UTF-8, it prints `nlp_doc: <error>` to stderr and exits with 1.

## Schema

```json
{
  "schema": "nlp.doc.v1",
  "text_chars": 3,
  "text_sha256": "<lowercase hex of SHA-256 over the UTF-8 input bytes>",
  "options": {},
  "segmentation": {
    "paragraphs": [{"span": {"start_char": 0, "end_char": 3}, "lang": "en"}],
    "sentences": [{"span": {...}, "lang": "en", "paragraph": 0}],
    "clauses": [{"span": {...}, "lang": "en", "sentence": 0}]
  },
  "unknown": [{"layer": "segmentation", "span": {...}, "reason": "unsupported_language"}]
}
```

- `lang` is one of `zh`, `en`, `mixed`, `und`.
- `reason` is one of `no_model`, `ambiguous`, `low_confidence`, `unsupported_language`,
  `not_applicable`. Layer 1 only produces `unsupported_language`.
- `sentence.paragraph` indexes `paragraphs`; `clause.sentence` indexes `sentences`.
- All lists are sorted by start. `unknown` is sorted by span start.

## Offsets

Spans count Unicode scalar values from the start of the exact input. `start_char` is included
and `end_char` is excluded. Every span is trimmed of whitespace at both ends (`char::is_whitespace`),
and empty spans are dropped. `\r` is only whitespace. A line ends at `\n`.

## Rules

### Paragraphs

1. A blank line (only whitespace) ends the current paragraph.
2. Inside a block, a line starts a new paragraph when:
   a. it begins with U+3000, a tab, or two ASCII spaces; or
   b. the previous line has more Han letters than other letters, and its last non-whitespace
      character is one of `。！？…」』”’）`.

### Sentences

Scan each paragraph left to right. A sentence ends after the characters named below.

1. A run of `。！？…` and ASCII `!` `?` (the run may include ASCII `.` and `!` `?`), then any closers
   `」』”’）)"`. A CJK terminator in the run always ends the sentence. For ASCII-only runs, the sentence
   ends only if the next char is whitespace or the paragraph ends.
2. A single ASCII `.`, then any closers `)”’」』"`. The sentence ends only if the next char is
   whitespace or the paragraph ends, and the word before the dot is not an abbreviation. The word is
   the maximal run of ASCII letters and dots before the dot. Abbreviations, compared with ASCII
   case folding: `mr mrs ms mx dr prof rev sr jr st vs cf fig eq no e.g i.e`, and any single
   uppercase letter. `etc` and `al` also require the next non-whitespace char to be uppercase or
   absent.
3. A run of two or more ASCII dots ends the sentence only if whitespace or the paragraph end
   follows and the next non-whitespace char is not a lowercase ASCII letter. Closers are not
   skipped here.
4. A `\n` ends the sentence unless it is a soft wrap. A soft wrap needs all of: the trimmed line
   before the `\n` has at least 40 chars; its last char is an ASCII letter, digit or `,`; and the
   first non-whitespace char of the next line is an ASCII lowercase letter.
5. A run of two or more spaces (` ` or U+3000) inside a line ends the sentence. A run at the end
   of a line, before `\n`, does not; rule 4 applies there.

### Clauses

Split a sentence at `，` `；` `：` always. Split at ASCII `,` `;` `:` only when the next char is
whitespace or the sentence ends, so `1,000` and `10:30` are not split. `、` is not a separator.
The separator belongs to no clause. The final terminator stays in the last clause.

### Language

A span's language comes from its letters. Han letters are U+4E00–9FFF, U+3400–4DBF,
U+20000–2A6DF and U+F900–FAFF. Latin letters are ASCII letters and U+00C0–024F, except U+00D7
and U+00F7. Other letters are `char::is_alphabetic` and neither of the above.

- No letters: `und`, no unknown.
- Other letters >= Han + Latin: `und`, and the sentence gets `unsupported_language`.
- Han > 0 and Latin > 0: `mixed`.
- Han > 0: `zh`.
- Otherwise: `en`.

## Known limits

- Hard-wrapped Chinese text splits at each line break. Rule 4 only keeps a wrap when the next
  line starts lowercase Latin, so Chinese line wraps become sentence boundaries.
- No opaque spans yet (#24).
- Japanese, Korean and other scripts are `und` with `unsupported_language`.
- Only the layer 1 rules above are implemented. There is no model, no network and no
  abbreviation list beyond the one given.
