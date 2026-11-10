# nlp.doc.v1, layer 1

`faberline_nlp::doc` returns a JSON `Document` with paragraphs, sentences, clauses and a
language tag per span, and the opaque spans (code, formula, heading, list item, citation, URL,
emoji) that sentences never split. It uses no model and no network.

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
    "clauses": [{"span": {...}, "lang": "en", "sentence": 0}],
    "opaque": [{"kind": "url", "span": {...}}]
  },
  "tokens": [{"span": {...}, "lang": "en", "sentence": 0, "upos": "PRON"}],
  "unknown": [{"layer": "segmentation", "span": {...}, "reason": "unsupported_language"}]
}
```

- `lang` is one of `zh`, `en`, `mixed`, `und`.
- `reason` is one of `no_model`, `ambiguous`, `low_confidence`, `unsupported_language`,
  `not_applicable`. Layer 1 produces `unsupported_language`. Layer 2 produces `no_model` and
  `ambiguous`, with `layer: "pos"`.
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
2. A single ASCII `.`, then any closers `)”’」』）"`. The sentence ends only if the next char is
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

## Opaque spans

`segmentation.opaque` lists the opaque spans, sorted by start. They never overlap and are never
empty. Each has a `kind` and a `span`. Kinds: `code`, `formula`, `heading`, `list_item`,
`citation`, `url`, `emoji`. Block kinds (`heading`, `list_item`, fenced `code`, display
`formula`) cover whole lines. Inline kinds (`code`, `formula`, `citation`, `url`, `emoji`) stay
inside one line. When there are none, `opaque` is `[]`.

Masking: before splitting, the text is masked. Each block span becomes spaces, except its
newlines, so its lines are blank. They end paragraphs and add no sentence. Each inline span
becomes `x`, except an emoji run that ends a sentence: its last char becomes `。`. Paragraphs,
sentences and clauses are computed on the masked text, so no boundary falls inside an opaque
span. Languages are still computed on the original text.

Detection works line by line. A line is a `\n`-separated line, and the line tests ignore its
leading whitespace. Each line is tried for a block span first, in this order: code, formula,
heading, list item. A block span skips the inline scan of its lines. Other lines are scanned
for inline spans. Every span is trimmed of whitespace.

Blocks:

1. Code: the line starts (after at most 3 spaces) with at least 3 backticks or at least 3 `~`, and the rest of that line has no more of the same
   fence char (so ```` ```x``` ```` and `~~~好開心~~~` are normal text).
   It ends at the next line made only of the same fence char, with at least as many chars as the
   opening, and only whitespace after. The span runs from the first fence char to the last char
   of the closing fence. An unclosed fence runs to the end of the text.
2. Formula: the line starts with `$$`. It ends at the first line whose last non-whitespace text
   is `$$` after the opening, and the opening line may be that line (`$$x$$`). The span runs from
   the opening `$$` to the end of that closing `$$`. Without a closing `$$` the line is normal
   text. A blank line before the closing `$$` also makes it normal text.
3. Heading: the whole trimmed line is one of:
   - Markdown: 1 to 6 `#`, then a space or tab, then at least one non-space char.
   - Chinese: `第`, then digits or any of `一二三四五六七八九十百零〇兩`, then one of `章節回卷篇部`,
     then the end of the line, or whitespace / `：` / `:` / `、` and a title. The line is at most
     40 chars and has none of `。！？`.
   - English: `chapter`, `part` or `section` (ASCII case-insensitive), one space, then one or
     more digits or `IVXLC`, then the end of the line, or one of `:` `.` `-` `—` and a title. The
     line is at most 60 chars.
4. List item: the trimmed line starts with `- `, `* `, `+ `, `• ` or `・ ` (the marker, then a
   space, tab or U+3000), or with 1 to 3 ASCII digits, then `.` or `)`, then a space. The span
   is the trimmed line. Continuation lines are not included.

Inline spans are found left to right in a line. At each position the kinds are tried in this
order, and a match continues after its span:

1. Code: a run of N backticks, then the next run of exactly N backticks on the same line, with at
   least one char between. Without that closing run, the whole backtick run is plain text.
2. URL: starts with `http://`, `https://` or `www.` (scheme ASCII case-insensitive). It goes on
   over ASCII graphic chars, stopping at whitespace and at `<`, `>` and `"`. Trailing
   `.,;:!?)]}'` are dropped. At least one char must follow the scheme.
3. Citation:
   - Numeric: `[`, then only ASCII digits, `,`, `-`, `–` and spaces (at least one digit), then `]`.
   - Author-year: `(` or `（`, then content with no `(`, `（`, `)` or `）`, then `)` or `）` on
     the same line. The content is at most 80 chars and is split at `;` and `；`. Each part,
     trimmed, is at most 60 chars, starts with an uppercase ASCII letter or a Han char, and ends
     with a year: four digits from `1500` to `2099`, optionally followed by one lowercase ASCII
     letter, and preceded by a space, `,` or `，`.
4. Formula: `$`, then a char that is not whitespace, not a digit and not `$`. The next `$` on the
   line closes it. That `$` must follow a non-whitespace char and must not be followed by an
   ASCII digit. `$5 and $10` is not a formula.
5. Emoji: a maximal run of emoji chars (U+1F000–1FAFF, U+2600–27BF, U+2B50, U+2B55, and the
   joiners U+200D, U+FE0F, U+20E3) with at least one char that is not a joiner. An emoji run ends
   a sentence when it is attached (the char before it exists and is not whitespace) and the char
   after it is whitespace or the end of the text.

Known limits of opaque spans:

- A Markdown-looking line such as `- Hello,` in dialogue is a list item.
- A year alone in parentheses, `(2020)`, is not a citation.
- Headings, list items and display formulas never have sentences.
- Continuation lines of a list item are normal text.
- No Markdown tables, quotes, links or footnotes.

## Tokens and POS

`tokens` is a list of tokens in text order, sorted by start. A token has a `span`, a `lang`,
the index of its `sentence`, and an optional `upos` (Universal POS, UD v2 names in upper case).
A token's `lang` is `zh` when it has a Han char, `en` when it has a Latin letter (a letter that
is not Han), and `und` otherwise. The Han set is U+3400–4DBF, U+4E00–9FFF, U+F900–FAFF, U+3005
(々) and U+3007 (〇). Extension B is not in it. `upos` is absent when no
rule or word decides it. Tokens never cross a sentence boundary, and no token is whitespace.

### Tokenization

Each sentence is split by these rules, in order:

1. Every opaque span inside the sentence is one token. Block spans are never tokens.
2. Han runs (maximal runs of the Han set) are matched against the Taiwan term list
   (`臺灣`, `資料庫`, `記憶體` and so on), longest match first. The text between matches is
   segmented by jieba with its HMM.
3. All other runs use Unicode word boundaries (UAX 29). Then:
   a. English clitics split off: `n't` (`didn't` gives `did`, `n't`; `can't` gives `ca`,
      `n't`), and `'s`, `'ll`, `'re`, `'ve`, `'m`, `'d`.
   b. A hyphen (`-` or `‐`, U+2010) between two words joins them: `well-known` is one token.
      Both words must contain a letter, so `3-4` gives `3`, `-`, `4` and `COVID-19` gives
      `COVID`, `-`, `19`.
   c. A dot after a listed abbreviation, a single uppercase letter (the abbreviation list of
      the sentence rules) or a word with a dot inside (`U.S`) joins it: `Dr.` and `U.S.` are
      one token each. `etc.` keeps its dot too, although `etc` is not in the list.
   d. A run of one char from `. … - — ~ ! ? ！ ？` is one token: `...`, `!!!`, `---`. A
      different char starts a new token, so `?!!` is `?` and `!!`.
4. Whitespace is dropped. Every other char belongs to exactly one token.

### POS tagging

The tag of each token is decided by the first rule that matches:

1. Opaque `url`, `citation` and `code` are `X`. Opaque `formula` and `emoji` are `SYM`.
2. Only punctuation characters (ASCII and CJK punctuation) are `PUNCT`.
3. Only symbol characters (for example `$ % + = @ ° × ± © € →`) are `SYM`.
4. A number is `NUM`: ASCII digits, with `.` or `,` allowed between them (`3.5`, `1,000`), or
   a run of Chinese numerals (`〇零一二三四五六七八九十百千萬万億亿兩两`).
5. The lexicon. English words are looked up in lower case, Chinese words exactly. A word has
   one tag.
6. A word in the ambiguous list gets no tag, with reason `ambiguous`.
7. Anything else gets no tag, with reason `no_model`.

Every undecided token has exactly one `unknown` entry with `layer: "pos"` and the token's span.
The lexicon is hand-written and small. It has no statistical model, so most content words
(nouns, verbs, adjectives) are `no_model`.

Sentences with `unsupported_language` have no tokens, and the sentence's `pos` entries are
absent.

Known limits of tokens and POS:

- jieba splits some Taiwan-usage words that are not in the term list. Those tokens may have
  the wrong boundary. Seen so far: jieba joins `門關` in `把門關上了`, where `門` and `關` are
  separate words (pinned by a test); `臺灣讀` was also reported as a wrong cut in review.
- Abbreviations other than the single capital letters and the listed words split at each dot.
- `etc.` at the end of a sentence keeps its dot, so the sentence has no separate `.` token.
- An abbreviation dot swallows a sentence-final period: `She said no. Then I left.` gives
  `no.` as one token (`no_model`), and `I said I.` gives `I.` as one token.
- Most content words are `no_model` until the model layers. The lexicon only has closed-class
  words and a few numerals.
- `&` is `und` by the token language rule, so it never matches the CCONJ entry in the
  lexicon. The entry stays, and it is a known spec issue.
- Japanese, Korean and other scripts get no tokens.

## Known limits

- Hard-wrapped Chinese text splits at each line break. Rule 4 only keeps a wrap when the next
  line starts lowercase Latin, so Chinese line wraps become sentence boundaries.
- Japanese, Korean and other scripts are `und` with `unsupported_language`.
- Layer 2 has no statistical tagger. Its UPOS comes from the rules and the lexicon in
  "Tokens and POS". There is no model and no network.
