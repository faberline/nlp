# nlp.doc.v1, layers 1 and 3

`faberline_nlp::doc` returns a JSON `Document` with paragraphs, sentences, clauses and a
language tag per span, the opaque spans (code, formula, heading, list item, citation, URL,
emoji) that sentences never split, and the entities found by rules. It uses no model and no
network.

CLI: `nlp_doc` reads UTF-8 text from stdin and prints one JSON document plus a newline. With no
flags it runs `analyze_document` (rules only, no model). With `--models <DIR>` and optional
`--backend <coreml|onnx>` (defaulting to `coreml` on macOS when compiled in, otherwise `onnx`),
it runs `analyze_document_with_model` to populate Layer 4 (`syntax`). On a read or model
integrity/runtime error, it prints `nlp_doc: <error>` to stderr and exits with 1.

## Schema

```json
{
  "schema": "nlp.doc.v1",
  "text_chars": 3,
  "text_sha256": "<lowercase hex of SHA-256 over the UTF-8 input bytes>",
  "options": {"names": ["Alice Lee"]},
  "segmentation": {
    "paragraphs": [{"span": {"start_char": 0, "end_char": 3}, "lang": "en"}],
    "sentences": [{"span": {...}, "lang": "en", "paragraph": 0}],
    "clauses": [{"span": {...}, "lang": "en", "sentence": 0}],
    "opaque": [{"kind": "url", "span": {...}}]
  },
  "tokens": [{"span": {...}, "lang": "en", "sentence": 0, "upos": "PRON"}],
  "entities": [{"kind": "date", "span": {...}, "sentence": 0, "value": "2026-11-10"}],
  "syntax": [{"token": 0, "sentence": 0, "span": {...}, "head": null, "deprel": "root", "score": 0.94}],
  "relations": [{"kind": "sequence", "level": "clause", "connector": {...}, "arg1": {...}, "arg2": {...}}],
  "unknown": [{"layer": "segmentation", "span": {...}, "reason": "unsupported_language"}]
}
```

- `options.names` is optional and empty by default (see Entities).
- `lang` is one of `zh`, `en`, `mixed`, `und`.
- `reason` is one of `no_model`, `ambiguous`, `low_confidence`, `unsupported_language`,
  `not_applicable`. Layer 1 produces `unsupported_language`. Layer 2 produces `no_model` and
  `ambiguous` (and `low_confidence` when a syntax model runs with UPOS confidence `< 0.50`),
  with `layer: "pos"`. Layer 3 produces `ambiguous`, with `layer: "entities"`. Layer 4 produces
  `no_model`, `low_confidence` and `not_applicable`, with `layer: "syntax"`.
- `sentence.paragraph` indexes `paragraphs`; `clause.sentence` indexes `sentences`.
  `entity.sentence` and `syntax.sentence` index `sentences`. `syntax.token` and `syntax.head`
  index `tokens` (`head` is `null` for `"root"`).
- All lists are sorted by start. `unknown` is sorted by span start, then end, then layer.
- `entities` and `relations` are always present, `[]` when nothing is found. `syntax` is omitted
  when empty (`analyze_document` never emits it). When present, the order of the lists is
  `segmentation`, `tokens`, `entities`, `syntax`, `relations`, `unknown`.

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
   absent. An abbreviated month (`jan feb mar apr jun jul aug sep sept oct nov dec`, ASCII
   case-insensitive; `May` is not one) does not end the sentence when one space and then an ASCII
   digit follow, so `Nov. 10, 2026 was fine.` is one sentence. `It was Dec. Then we ran.` still
   splits.
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

## Relations

`relations` is a list of discourse relations between layer-1 units, sorted by the start of
`connector`. It reads only `segmentation`; it does not use tokens. It uses no model and no
network, and order and cause come only from the connector words and the text order. A relation
has `kind`, `level`, `connector`, an optional `correlate`, `arg1` and `arg2`:

- `kind` is one of `sequence`, `cause`, `contrast`, `parallel`, `condition`.
- `level` is `clause` when `arg1` and `arg2` are layer-1 clauses, and `sentence` when they are
  layer-1 sentences. A sentence relation comes from a connector of arg2 (see Connectors) that
  starts its sentence, or a `because` or `因為` that is the only clause of its sentence.
- `connector` is the first connector in the text. `correlate` is the second connector of a pair
  (`因為 … 所以`, `如果 … 就`, `一方面 … 另一方面`, `先 … 再`, `if … then`), and is absent when
  there is none.
- Roles: `sequence` has `arg1` earlier and `arg2` later. `contrast` and `parallel` keep text
  order. `cause`: `arg1` is the reason and `arg2` the result, so `B, because A` has `arg1` =
  `because A`. `condition`: `arg1` is the `if` clause and `arg2` the consequence.
- A bare connector is one that ends its clause, such as `然後，` or `However,`. A bare backward
  connector that is not first in its sentence takes the clause after it as `arg2`, if there is
  one; `arg1` is still the clause before it. A bare `because` or `因為` that is not first takes
  the clause before it as `arg2` (see Connectors).

### Connectors

A connector counts only at the start of a clause. Before matching, whitespace, opening marks
(`「『“‘（([“"'`), emoji and symbol chars (arrows, dingbats, pictographs, `© ® ™`) are skipped, so
`😋 But we lost.` still starts with `But`. The scan stops at any other opaque span (code, formula,
citation, heading, list item, URL): a connector inside one is not a connector, as in
`(But 2020) says so.` The longest entry wins. English entries match ASCII
case-insensitively and need a word boundary after them (no letter, digit, `'`, `’` or `-`
follows). Multi-word English entries use exactly one space.

| kind | connector sits in | zh | en |
|---|---|---|---|
| sequence | arg2 | 然後 接著 隨後 | then, afterwards, after that |
| cause (result) | arg2 | 所以 因此 | so, therefore, thus |
| contrast | arg2 | 但是 然而 可是 但 | but, however |
| parallel | arg2 | 另一方面 另外 此外 而且 並且 同樣 | also, moreover, furthermore, likewise, similarly, in addition |
| cause (reason) | arg1 | 因為 (correlate 所以 因此 so therefore thus) | because (correlate 所以 因此 so therefore thus) |
| condition | arg1 | 如果 假如 要是 若 (correlate 就 那麼 便 then) | if (correlate 就 那麼 便 then) |
| sequence | arg1 | 先 (correlate 然後 接著 再 隨後) | none |
| parallel | arg1 | 一方面 (correlate 另一方面) | none |

The correlate lists are shared by both languages. Every cause opener (`因為`, `because`) takes
`所以 因此 so therefore thus`, and every condition opener (`如果 假如 要是 若`, `if`) takes
`就 那麼 便 then`.

Connectors of arg2 take arg1 from the clause before them (or the sentence before, when they
start a sentence).

Connectors of arg1 (openers) follow these rules. The first rule that fits is used:

1. A `because` or `因為` that is not first in its sentence takes the clause before it as `arg2`,
   also when bare.
2. A `because` or `因為` that is the only clause of its sentence takes the sentence before it in
   the same paragraph as `arg2`, also when bare. The level is `sentence`:
   `We stayed. Because it rained.` gives `arg1` = `Because it rained.` and `arg2` = `We stayed.`
   When no sentence before it can be its `arg2` (it is the first sentence of a paragraph, or the
   first sentence after an unsupported sentence; `Because.` alone), it gives an `unknown` and no
   relation.
3. Any other opener that is bare (`如果，我們回家。`, `Because, we stayed.`), or that has no clause
   after it in its sentence (`If it rains.`), gives an `unknown` and no relation.
4. Any other opener reads the start of the clause after it:
   - A correlate of the opener: that clause is `arg2`, and the correlate consumes it (its own
     connector gives no second relation). A correlate that is the whole clause
     (`If it rains, then, we stay.`) takes the clause after it as `arg2` instead; if no clause
     follows, or that clause starts with a connector, the opener gives an `unknown` and there is
     no relation.
   - Another connector (`If it rains, but we go`): the opener gives an `unknown` and no relation.
   - No connector: that clause is `arg2`, and there is no correlate.

`先` and `一方面` give a relation only with their correlate, and never an `unknown`: in every
other case of rules 3 and 4 they give nothing.

A correlate must be a whole word. A stop list of following chars rejects `便` before `宜 利 當 條
於 民 秘`, `再` before `見 次 度 三 也 說 加`, and `就` before `近 業 讀 地`, so `便宜`, `再見` and
`就近` are not correlates. (It is a stop list of following chars, not a longest-word match.)

A `because` or `因為` clause followed by a cause-result connector is one pair, not two readings:
`Because A, so B.`, `Because it rained, therefore we stayed.` and `因為下雨，therefore we stayed.`
give one cause relation with `connector` = `Because` (or `因為`), `correlate` = `so` (or
`therefore`), `arg1` = the `Because …` clause and `arg2` = the `so …` clause. There is no
`unknown` over the opener and no relation for the result connector. The correlates are `所以`,
`因此`, `so`, `therefore` and `thus`, and `so` as a correlate follows the `so` rules below (`so
far` is not one, and `So,` is not one).

Special cases:

- `但` is not a connector before `願 凡 求 見`. `若` is not one before `干 無 非 是 即`. `先` is
  not one before `生 前 後`. `if only` is not a connector.
- `however` counts only when `,` follows it (`However,`), not in `However hard he tried`.
- `so` is not a connector before `far much many long as that good bad little few called on`.
  `So,` (a comma follows) gives an `unknown` and no relation.
- `then` after a clause that starts with `if` is the correlate of that `if`, not a sequence. This
  also holds when `then` starts the next sentence of the same paragraph: in
  `If it rains. Then we stay home.` the `then` gives nothing, and the `if` gives an `unknown`.

### Unknown

A connector gives an `unknown` with `layer: "relations"`, `reason: "ambiguous"`, and the
connector's span, and no relation, in these cases:

- It has several readings: zh `同時`; en `while`, `since`.
- Its argument is not clear:
  - A connector of arg2 that starts its sentence, when no sentence before it can be its `arg1`
    (the first sentence of a paragraph, or the first sentence after an unsupported sentence).
  - A bare connector of arg2 that is not first in its sentence and has no clause after it
    (`We left, but.`). A `then` after an `if` clause is a correlate and gives nothing.
  - An opener in the cases of opener rules 2, 3 and 4 above: `Because.` alone, `如果，我們回家。`,
    `If, then we go.`, `If it rains.`, `If it rains, but we go`, `If it rains, then.`

  `先` and `一方面` never give an `unknown`. After a bare `先` or `一方面` the next clause is read
  on its own: `先，然後吃飯。` gives a sequence for `然後` with `arg1` = `先`.
- A discourse marker: `So,`.

A sentence with `unsupported_language` has no relations, and no relation uses it as an argument.
A new paragraph also breaks the sentence relation.

### Relations limits

- Connectors in the middle of a clause are not seen: `He left because it rained`, `A but B`
  without a comma, `如果下雨我們就…` with `就` mid-clause, `We tested it, and then we ran it`.
- Only one unit before (or after) a connector is its argument. No implicit relations: a relation
  needs a connector.
- No `雖然…但是` or `although` concession pair beyond the contrast of `但是`.
- Simplified-only connector forms are not matched, such as `因为` for `因為`. Forms that are the
  same in both scripts still match: `因为下雨，所以比赛取消了。` gives a cause relation for `所以`, and
  no relation for `因为`. zh `又`, `而`, `並` alone are not connectors.
- Argument spans are always clauses or sentences, never smaller.
- A bare connector can leave an argument that is only the connector: `We stayed home, because.`
  (`arg1` is `because.`) and `We slept. So.` (`arg2` is `So.`). Such a relation is still
  reported. A bare correlate is the exception (see above). A bare correlate followed by a clause
  that starts with another connector gives no relation for the opener. The correlate clause can
  still be an argument of that connector. A backward connector with a clause of its own takes it
  as `arg1`: `先洗手，然後，但是吃飯。` gives a contrast with `arg1` = `然後`. A `because` or `因為`
  takes it as `arg2`: `If it rains, then, because we stay.` gives `arg2` = `then`. A connector that
  gives no relation leaves the clause without a role. An opener (`如果`, `if`, `先`, `一方面`) in the
  clause after the bare correlate does not use the correlate clause, also when it gives a relation
  (`先洗手，然後，如果餓了，就吃飯。` gives only the condition relation of `如果`).
- A `because` or `因為` that is not first in its sentence takes the clause before it as `arg2`,
  also when bare: `We left, because, it rained.` gives `arg1` = `because` and `arg2` = `We left`.
- The correlate tables are shared by both languages. A correlate of the other language still
  pairs (`因為下雨，so we left.` gives `so`).
- Only `However` and `So` look at the comma after them. `However,` is a bare connector, and
  `However，` with a full-width comma is not one.
- Correlates may cross languages: `If 天氣好，就…` pairs the English `If` with the zh `就`.

## Entities (layer 3)

`entities` lists entities found by rules in supported sentences. A sentence whose language is
unsupported gets no entities. Each entity has a `kind`, a `span`, the `sentence` index, and
`value` and `unit` when the kind has them. Kinds: `number`, `date`, `relative_date`, `time`,
`quantity`, `url`, `email`, `name`. The list is sorted by start.

Values (`value` is omitted for kinds that have none):

- `date`: `YYYY-MM-DD`, `YYYY-MM`, `YYYY`, or `--MM-DD` when the year is not given.
- `time`: 24-hour `HH:MM`. Seconds are part of the span but not the value.
- `number`: a plain decimal string with no grouping, such as `1000`, `34.5` or `150000000`.
- `quantity`: the number as in `number`, and `unit` as written. Currency prefixes are
  lowercased in `unit` (`NT$` gives `nt$`). `%` is the unit for percent.
- `relative_date`, `url`, `email`, `name`: no value.

Scanners, run at each position of a supported sentence:

- Relative dates: `今天 明天 昨天 後天 前天 今晚 昨晚 今年 明年 去年 這週 本週 上週 下週 這個月
  上個月 下個月`, and `today tomorrow yesterday tonight`, and `last|this|next` followed by
  `week|month|year` (ASCII case-insensitive, one space).
- Dates, all checked for a real calendar day (leap years included):
  - Chinese: `Y年M月D日` (or `號` for `日`), `Y年M月`, `M月D日` (or `號`), and `Y年` alone. `Y` is
    four ASCII or Chinese digits, or `民國N` (N+1911). Month 1 to 12 is needed. `M月` alone and
    `D日` alone are not dates. Their digits are numbers: `11月` gives the number 11, and `10日`
    gives the number 10. `十一月` alone gives nothing, since a Chinese numeral needs a measure word.
  - English: `Month D[st|nd|rd|th][,] YYYY`, `Month YYYY`, `Month D`, `D Month YYYY`, `D Month`.
    Full names and 3-letter names, and `Sept`. An abbreviated month may have a dot before the day
    or the year (`Nov. 10`, `10 Nov. 2026`).
  - Numeric: `YYYY-MM-DD`, `YYYY/MM/DD`, `YYYY.MM.DD`, `YYYY-MM` (month 1 to 12). `A/B/YYYY` is
    read by the one part over 12, which is the day (`13/4/2026`) or the month (`4/13/2026`). If
    both parts are 12 or less, both readings are tried: when both are real dates the candidate is
    ambiguous, when one is it is that date, and when neither is it is not a date. So `3/4/2026` is
    ambiguous, and `0/5/2026` gives the numbers 0, 5 and 2026. If both are over 12 there is no
    date; the digits are numbers.
  - An invalid full date such as `2026-13-45`, `2026/2/30`, `2023年2月29日`, `二〇二三年二月二十九日`
    or `29 Feb 2023` is not a date, and no shorter date inside it is read instead (`2月29日`,
    `Feb 2023`, `2/30`). Its ASCII digits are numbers (`2023年2月29日` gives 2023, 2 and 29).
    A date written in Chinese numerals such as `二〇二三年二月二十九日` gives nothing, since a
    Chinese numeral needs a measure word.
- Times: `H:MM` with an optional `:SS`, `H點`, `H點M分`, `H點半`, `H am|a.m.|pm|p.m.` (the space
  before the marker is optional; seconds must be 0 to 59, so `3:30:99 pm` is not a time), `noon` (12:00), and the Chinese markers `凌晨 上午 早上 下午
  晚上 傍晚 中午` placed before the hour. The hour must be attached to `點`: `我們15點見` gives the
  time `15:00`, but `我們 5 點` with a space gives the number 5. Hour 0 and hours 13 to 24 with no marker are 24-hour. `12 am` is
  `00:00` and `12 pm` is `12:00`, but `上午 早上 晚上 傍晚` with 12 are ambiguous. `midnight` is
  ambiguous. An hour of 1 to 12 with no marker is ambiguous, so `3:30` alone is ambiguous. A bare
  `一點` with no marker and no `半` or `M分` is "a bit" (`他有一點累`), so it gives no time and no
  unknown.
- Quantities: a number, then optionally one space, then a unit. Units such as `kg km 分鐘 天
  days` need a space or none. `g m l` are units only when attached (`5m`, not `5 m`). A currency
  prefix is `US$ NT$ $ ¥ € £`, and `百分之N` is a percent. A bare English `one` is not a
  quantity.
- Numbers: ASCII digits with comma groups of exactly three digits and an optional decimal part.
  `萬 万` and `億 亿` multiply the number before them (`3萬`, `1.5億`). An ASCII number followed
  by Chinese numerals is read as `<number><chain>` where the chain ends in a multiplier: `5千元`
  is 5000, `3萬5千元` is 35000, `3千萬人` is 30000000. A decimal takes one multiplier. If the run
  cannot be read (`3萬5元`, `3五千`, `3萬4萬`), the whole run gives no number, and no bare
  multiplier from it is a number either.
- A decimal point in Chinese is `點` and then digits read one by one: `三點五公斤` is 3.5 `公斤`,
  `百分之三點五` is 3.5 `%`. `三點`, `三點五分` and `三點半` stay times.
- A lone English `hundred` is not a number, like a lone `thousand`.
- English number words follow the order of a number. A word that breaks the order ends the
  number before it, and the word starts a new number: `two three` gives 2 and 3, `nineteen
  eighty-four` gives 19 and 84, `twenty twenty` gives 20 and 20, and `two hundred three hundred`
  gives 200 and 300, and `Five thousand two thousand` gives 5000 and 2000. When the breaking word
  is `hundred`, the unit words right before it start the new number; when it is `thousand`,
  `million` or `billion`, all the words after the last `thousand`, `million` or `billion` do:
  `Five thousand two hundred thousand` gives 5000 and 200000. Valid forms keep their value:
  `two thousand five hundred` is 2500, `two hundred three` is 203 and `twenty-one` is 21. A lone
  `one` is not a number, so `one two three` gives an ambiguous `one` and the numbers 2 and 3.
- Chinese numerals (`〇零一二三四五六七八九十百千萬万億亿兩两`):
  - All digits, at least two: digit by digit (`二〇二六`). 兩 and 两 are not digits here: `一兩天`
    and `兩三個` give no number. A lone `兩個` is 2.
  - Otherwise the standard reading (`一百零五`, `一百〇五`, `三萬五千`, `三億五千萬`).
  - A multiplier with no digit before it reads as 1 (`十` is 10, `百` is 100).
  - A run cannot start with 萬 or 億, and `三億萬` is not a number.
  - Two plain digits in a row are allowed only as digit by digit, or after 零 or 〇.
  - A multiplier must be smaller than the one before it in its group: `十二十`, `三十四十`,
    `五百六百` are not numbers.
  - 萬 and 億 groups must decrease: `三萬四萬`, `一萬二億`, `三萬億` are not numbers.
  - A run counts only with a measure word after it (`個`, `人`, ...).
  - A run starts only after a non-numeral and not after `第`, so `第三十個` gives no number and
    the `十` in `三十個` gives no number of its own.
  - Times and dates that start with a Chinese numeral follow the same start rule. So `三十五點`
    and `第三點` give no time, and `三十一月五日` gives no date.
- Email: a local part of `A-Za-z0-9._%+-`, then a domain of at least two labels, where the last
  label is 2 or more ASCII letters.
- URL: taken from the opaque `url` spans.
- Name: each entry of `options.names`, matched exactly and case-sensitively, leftmost-longest. An
  ASCII name must not touch an ASCII letter or digit on either side. An entry that contains a
  line feed or NUL (`\u{0}`) is ignored.

Overlap: entities never start or end inside an inline opaque span. Candidates are sorted by
start, then longer first, then by priority: `name` 0, `email` and `url` 1, `quantity` 2, `time`
3, `date` 4, `relative_date` 5, ambiguous 6, `number` 7. A candidate is kept only if it does not
overlap a kept one.

Ambiguous candidates are not entities. Each one is an `unknown` with layer `entities` and reason
`ambiguous`. It takes part in overlap resolution, so it can block a `number` inside its span.

Known limits of entities:

- The sentence splitter ends a sentence after `a.m.` or `p.m.` when whitespace follows, as it
  does for any dot. The time entity still resolves: `3:30 p.m.` gives `15:30` and `10 a.m.` gives
  `10:00`. `Sept. 3rd` and `Nov. 10` resolve as dates, because of the abbreviated-month rule in
  layer 1 (a digit must follow the dot).
- No model. Names come only from `options.names`. Other proper nouns are not entities.
- Dates, times and numbers without a marker or unit are only numbers or ambiguous.
- No signs or ranges: `-5` gives the number 5, and `3-5` gives 3 and 5.
- No weekdays: `Monday` gives nothing.
- No ordinals: `1st` and `第三` give nothing.
- Durations written with `年` are numbers: `3年` gives the number 3.
- No time zones: `3pm EST` gives the time `15:00`, and `EST` gives nothing.
- Chinese numerals need a measure word: `三十五` gives nothing, and `三十五個` gives 35.
- `and` ends an English number, so `one hundred and twenty-three` gives 100 and 23.
- A run of two plain Chinese digits such as `二三個` or `三四個` is read digit by digit (23, 34),
  though `三四個` can be an approximate range. Only a run of 兩/两 or a run with a multiplier is
  refused.
- The gold file `tests/data/entity_gold.json` was written with the same rules, so the receipt
  `docs/receipts/entities-2026-11-10.json` is a regression check, not an independent measure.

## Known limits

- Hard-wrapped Chinese text splits at each line break. Rule 4 only keeps a wrap when the next
  line starts lowercase Latin, so Chinese line wraps become sentence boundaries.
- Japanese, Korean and other scripts are `und` with `unsupported_language`.
- Layer 2 has no statistical tagger. Its UPOS comes from the rules and the lexicon in
  "Tokens and POS". There is no model and no network.
- Layer 7 (Relations) sees only connectors at the start of a clause. See "Relations limits".
- Layers 1 and 3 are rules only. There is no model, no network and no abbreviation list beyond
  the one given.

## Model runtime

`faberline_nlp::doc::model` loads a trained model from a directory and runs it on the CPU. It is
the base for layers 4 to 6. The rule layers (1, 2, 3 and 7) do not use it. Nothing here is
downloaded by cargo, and cargo runs no Python.

### Layout and manifest

A model directory holds `manifest.json` and one subdirectory per model:

```json
{ "schema": "nlp.models.v1",
  "models": [ { "id": "test-tiny", "layers": ["test"], "licence": "CC0-1.0",
                "dir": "test-tiny", "max_len": 64, "pad_id": 0,
                "model": "model.onnx", "tokenizer": "tokenizer.json", "expected": "expected.json",
                "coreml": { "model": "model.mlmodel", "vocab": 258 },
                "files": [ {"path": "model.onnx", "sha256": "<lowercase hex>", "bytes": 8728}, ... ],
                "receipt": null } ] }
```

- `model`, `tokenizer` and `expected` must each be listed in `files`. The three paths must
  differ. Every path is relative to `<dir>` and has no `..` and no root.
- `max_len` must be from 1 to 4096. For ONNX it must equal the input length that the file declares
  (the second dimension of its input), and a different value is refused at load. An input length
  that the ONNX file does not fix is accepted. For Core ML, the pad run at load refuses a
  `max_len` that the model does not take (see "Core ML backend").
- `bytes` is optional. When present it is checked too.
- `coreml` is optional. When present, its `model` must be listed in `files` too, and it must not
  name the same path as `model`, `tokenizer` or `expected`. `vocab` must be above 0. The Core ML
  backend refuses an id outside `0..vocab`. `vocab` is read from the manifest only. It is not
  checked against the embedding table in the `.mlmodel` file, so a wrong value is not caught.

### Hash check

`verify` reads the manifest and checks every listed file for existence and sha256 before any model
file is parsed. With the `onnx` feature, `load` calls `verify` first. Each file is read once. Its
bytes are hashed and kept in `Verified`, and the parsers read those bytes. Nothing is read from disk
again, so a file replaced after `verify` has no effect. `Verified` is public (`info` and the bytes of the
model, tokenizer and expected files). `verify(models_dir, id)` returns it, and
`load_verified(&Verified, backend)` loads a model from it. `SubwordTokenizer::from_verified(&Verified)`
builds the tokenizer from it and replaces the earlier `from_file`: no model or tokenizer file is
read by path any more.

Only regular files are read. A listed file that is a symlink is refused, and so is anything that is
not a regular file. A model directory that is a symlink is followed. The size from the file metadata
is compared with `bytes` before the read, so a large or truncated file is refused without being read
in full.

The errors are `UnknownModel`, `Missing` (a listed file is absent), `HashMismatch` (`file` is
`path (bytes)` for a byte count mismatch), `Manifest` (bad schema, an absolute path or `..` in `dir`
or `files`, a `model`, `tokenizer` or `expected` not listed in `files`, two of `model`, `tokenizer`
and `expected` naming the same path, or a `max_len` outside 1 to 4096) and `Load` (a refused file,
a `max_len` that differs from the model's input length, or a parse failure). A hash mismatch is
never reported as `unknown`.

### Backends and the engine choice

`Backend::Onnx` runs on `tract-onnx` (feature `onnx`, which also enables `tokenizers`). tract is
pure Rust, under MIT or Apache-2.0, and needs no native download at build time. The alternatives
were rejected for this layer. `ort` links or downloads the ONNX Runtime native library. `candle`
has no ONNX loader that covers encoders reliably.

`Backend::Coreml` runs on `objc2-core-ml` 0.3.2 (feature `coreml`, macOS only, which also enables
`tokenizers` so standalone `--features coreml` builds include `SubwordTokenizer` without `onnx`).
The bindings crate is generated from the Apple headers and is maintained in the `objc2` project. Its
licence is MIT OR Apache-2.0 OR Zlib, and so are `objc2` 0.6 and `objc2-foundation` 0.3. We write no
FFI by hand. The framework calls are still `unsafe` in the bindings, and the few `unsafe` blocks in
`src/doc/model/coreml.rs` say why each is sound. The three `objc2` crates are declared under
`[target.'cfg(target_os = "macos")'.dependencies]`. On Linux the feature adds only `tokenizers` and
`tempfile`, and no Core ML code is built. On Linux, or without the feature, `Backend::Coreml` returns
`NoBackend`. The default `cargo test` never needs Core ML.

Tokenization uses `tokenizers` 0.23.2 with its `fancy-regex` feature (enabled by either `onnx` or
`coreml`). `fancy-regex` is pure Rust, so no C source is built and no network is needed. The default
`onig` feature is off, so `onig` is not in the dependency tree. `tokenizers` pulls in `rayon`
transitively, and this layer does not call it: `run` and `encode` stay sequential and start no thread
pool.

### Determinism

The tract model is loaded with the fixed input shape `[1, max_len]` of `i64`. Each sequence runs
alone on the CPU in f32. The batch is never run as one tensor, so a batch gives the same bits as
single runs. Each sequence is padded with `pad_id` to `max_len`, and the output is cut to its own
length. A sequence longer than `max_len` is `TooLong`. An empty sequence gives shape `[0, labels]`.
An id outside `0..vocab` is a `Run` error, checked before any sequence runs. `vocab` is the row count
of the model's one embedding table (`Gather`), read when the model is loaded. Without this check,
tract would panic. A `pad_id` outside the vocabulary is refused at load.

### Core ML backend

Core ML loads only compiled models, from a file URL. So `load` writes the verified `.mlmodel` bytes
to a private temporary directory (`tempfile`) and compiles them there with the synchronous
`compileModelAtURL`. That call is deprecated, but it needs no run loop. The compiled model is loaded
with `MLComputeUnitsCPUOnly`, and then the compiled copy is removed. The temporary directory that
holds the `.mlmodel` file lives as long as the model. No GPU and no Neural Engine are used.

The Core ML path takes the ids that the tokenizer gives, as the ONNX path does. After the model
is loaded, `load` runs one sequence of `max_len` pad ids. A failure there is `Load`, with the
message `pad run: ...`. This refuses a `max_len` that differs from the model's input length. A
`pad_id` outside `0..vocab` is refused before the model is compiled.

The run rules are the same as above: one sequence per prediction, padded with `pad_id` to
`max_len`, ids as `Int32`, f32 output cut to the sequence length, and `TooLong` and the id range
check before any run. A mutex holds the one model instance, so predictions are serialised, and
`Model` is `Send + Sync`.

The Core ML test model has the same graph as the ONNX one: a constant embedding table, a gather
(`add_gather`), and a batched matrix multiply with a bias. It is made by `coremltools`
(`NeuralNetworkBuilder`, no torch) from the same numpy weights.

### Cross-backend receipt

`tests/doc_coreml.rs` has two tests. CI runs on Linux and cannot run Core ML, so only
`receipt_is_consistent` runs there.

- `backends_agree` (macOS, features `onnx` and `coreml`) runs the 12 fixtures through both backends.
  It asserts equal shapes and data lengths, equal labels and a logit difference of at most 1e-4.
  With `NLP_WRITE_RECEIPT=1` it writes `docs/receipts/model-runtime-<date>.json`, where `<date>` is
  the UTC date of the run. The receipt holds, per fixture, the token count, label equality and the
  maximum absolute difference. It also holds the overall maximum, the sha256 and byte count of
  `model.onnx` and `model.mlmodel`, and the macOS, architecture and Rust versions.
- `receipt_is_consistent` runs everywhere, without Core ML. It asserts that exactly one
  `model-runtime-*.json` file is in `docs/receipts`. It checks that labels are equal, that the
  maximum is at most 1e-4 and equals the fixture maximum, that the fixtures are those of
  `expected.json`, and that the two hashes equal the manifest.

The repository keeps one receipt. When the model files change, write the receipt again on a Mac
and remove the old one.

To write the receipt:

```
NLP_WRITE_RECEIPT=1 cargo test --locked --features onnx,coreml --test doc_coreml
```

On the run that wrote the current receipt (macOS 26.5, Apple silicon) the maximum difference was
9.5e-7, under 1e-6.

### Tokenizer offsets

`SubwordTokenizer::encode` adds special tokens (none in the test model) and returns ids with
spans. The crate reports UTF-8 byte offsets, and `encode` converts them to Unicode scalar offsets of
the input text. A special token has an empty span. The tokens from one character's bytes all get
that character's span.

### When no model is available

When the feature of the backend asked for is not compiled in, `load` returns `NoBackend` at once
and reads no file, so a layer reports `no_model`. That is the `onnx` feature for `Backend::Onnx`,
and the `coreml` feature on macOS for `Backend::Coreml`. `verify` works without the feature and
gives the integrity errors. With the feature, `load` returns `Missing`, `UnknownModel`, `HashMismatch`, `Manifest` or `Load` for a bad
directory. `ModelError::into_reason` maps `NoBackend`, `UnknownModel` and
`Missing` to `no_model`. A layer then reports
`no_model_unknown(layer, span)`, which is `{"layer": ..., "span": ..., "reason": "no_model"}`.
Other errors are not `unknown`. They are returned to the caller.

To point at a model, pass the directory that holds `manifest.json` to `load`.

### The test model

`models/test-tiny/` is made by `training/make_test_model.py`. Run it by hand on a Mac, inside
`training/.venv` (packages numpy, onnx, tokenizers and coremltools). CI never runs it. It is a byte-level BPE
with no merges (256 byte symbols plus `[PAD]` and `[UNK]`), an embedding of 8 dimensions, and a dense
layer to 5 labels. The weights are random with a fixed seed. The licence is CC0-1.0. `expected.json`
holds the ids, spans and logits for 12 fixture texts, computed by numpy. The tests check tract
against it within 1e-5 and Core ML within 1e-4. The four files total 88,655 bytes.

Real models are not in this repository. They will come as GitHub Release assets, pinned in the
manifest by a script (a later issue).

## Syntax (layer 4)

`analyze_document_with_model(text, options, models_dir, backend)` runs Layers 1–3 and 7 first, then
loads `SYNTAX_MODEL_ID` (`"doc-syntax-v1"`) on `backend` to decode Universal Dependencies v2 trees
into `Document.syntax` and upgrade undecided Layer-2 `upos` tags. Callers that run many documents
can verify and load the model once and call `analyze_document_with_loaded_model(text, options,
model, tokenizer)` (or `doc::syntax::annotate`). Calling `analyze_document(text, options)` never
runs a model and leaves `syntax` empty (omitted from JSON).

### Subword alignment and packed `[len, 421]` output

For each supported sentence, `SubwordTokenizer::encode` runs on `&text[byte_start..byte_end]` of
`sentence.span`. The tokenizer prepends `[CLS]` at subword index `0` (`s_0 = 0`, empty span) and
appends `[SEP]` at the end (empty span). Each Layer-2 token `u` (`1..=m`) maps to its first
overlapping subword `s_u` (`1 <= s_u < subwords.ids.len()`) after adding `sentence.span.start_char`
to the subword's scalar span. If `subwords.ids.len() > model.info().max_len` or any token has no
overlapping subword, the sentence is skipped and recorded in `unknown` as
`{"layer": "syntax", "span": sentence.span, "reason": "not_applicable"}`.

For a model with sequence cap `max_len` (`256` in `doc-syntax-v1`), `Model::run` returns a `[len, C]`
tensor with `C = 17 + max_len + 148` (`421` when `max_len = 256`):

- `0..17`: 17 UPOS logits in `Upos` enum order (`ADJ`, `ADP`, `ADV`, `AUX`, `CCONJ`, `DET`, `INTJ`,
  `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `SCONJ`, `SYM`, `VERB`, `X`).
- `17..17 + max_len`: raw biaffine arc logits. For dependent token `u` (`1..=m`) and candidate head
  `v` (`0..=m`, with `0` = `[CLS]` root at `s_0 = 0`), the arc score is `row(s_u)[17 + s_v]`.
- `17 + max_len..C` (`148` columns): `r_dep` (`37`), `r_head` (`37`), `r_bi_dep` (`37`), and
  `r_bi_head` (`37`) for the 37 alphabetical UD v2 base relation names (`acl` through `xcomp`, with
  `root` at index `34`). For `(dep = u, head = v)`,
  `rel_logit[c] = r_dep[s_u][c] + r_head[s_v][c] + r_bi_dep[s_u][c] * r_bi_head[s_v][c]`.

### Chu-Liu/Edmonds, confidence, and UPOS upgrade

- **Single-root tree decoding**: `doc::syntax::chu_liu_edmonds` finds the exact maximum-weight
  spanning arborescence over `0..=m` subject to node `0` having exactly one child. Every decoded
  sentence is guaranteed to have strictly one `"root"` attachment (`head: None`), no self-loops,
  and no cycles. When `v == 0`, `deprel` is forced to `"root"`; when `v > 0`, `deprel` is the
  argmax over `c != 34` (ties go to the lowest index).
- **Quantized confidence (`Score`)**: `p_arc = softmax(arc_scores_u_over_0..=m)[v]`,
  `p_rel = 1.0` when `v == 0` else `softmax(rel_logits_excluding_34)[best_c]`, and
  `conf = p_arc * p_rel`. `Score(u16)` stores `(conf.clamp(0.0, 1.0) * 1000.0).round() as u16` and
  serializes as `(milli as f64) / 1000.0`, keeping `Document: Eq` and giving byte-identical JSON
  across Core ML and ONNX.
- **Thresholds and `unknown`**:
  - When `score.milli() < 350` (`SYNTAX_THRESHOLD = 0.35`), the token is omitted from `syntax` and
    emits `{"layer": "syntax", "span": tok.span, "reason": "low_confidence"}`. Otherwise a
    `SyntaxToken` is emitted with 0-based `token` and `head` indices into `Document.tokens`.
  - Tokens whose Layer-2 `upos` was already decided by rules or the lexicon keep their tag. Tokens
    whose Layer-2 `upos` was `None` get the model's top UPOS tag (and drop their `"pos"` `unknown`
    entry) when `upos_prob >= 0.50`; otherwise `upos` stays `None` and the token's `"pos"` `unknown`
    reason becomes `low_confidence`.
  - When `backend` is not compiled in, or `doc-syntax-v1` or its files are absent (`NoBackend`,
    `UnknownModel`, `Missing`), each supported sentence emits
    `{"layer": "syntax", "span": sentence.span, "reason": "no_model"}`. Integrity and runtime
    errors (`HashMismatch`, `Manifest`, `Load`, `Run`) are returned to the caller.

## Readers of `nlp.doc.v1`

For `Locale::ChineseTaiwan`, two consumers read the layers instead of their own splitters.

- `NarrativeAnalyzer` (used by `analyze` and the source stream) takes its sentences from layer 1
  and its name hits from the layer 3 name rule. The terms are `TAIWAN_TERMS` plus the caller
  actors, objects, `lexicon.ranks` and places. The name matcher is built once, in
  `NarrativeAnalyzer::new`.
- `CrossChapterCandidates` (zh-TW) takes its pronoun tokens from layer 2.

The output must not change: the old path stays the reference. Both readers use the same
check, `doc_applies`, line by line. A line takes the doc path only when every char is one
of these:

- a Han ideograph (U+4E00 to U+9FFF, U+3400 to U+4DBF);
- one of `。！？，、；：「」『』`;
- one ASCII space or U+3000 that is not next to another space and not first or last in the line;
- a paragraph indent: U+3000 chars at the start of a line that has other text after them;
- a final `\n`.

and the line also passes a context rule. Layer 1 always ends a sentence after a run of
`。！？`. ICU follows UAX 29 (SB8a, SB9 to SB11) and does not end it when the run, its closing
`」』` and its spaces are followed by `，、；：`, another terminator or an opening `「『`
(class Close). So every terminator run, with the `」』` and spaces after it, must be followed by
a Han ideograph or by the end of the line. Example lines that keep the old path: `「好。」，他說。`,
`天黑了。「回家吧」，小甲說。`, `好？ ！`, `小甲。」「阿乙。`. Unbalanced quotes need no rule of
their own: every difference they caused was a terminator run before an opening quote.

Every other line keeps the old path: ASCII letters, digits and punctuation (also `.`, `!`, `?`),
`…`, other brackets and closers (`《》〈〉【】〔〕［］[]`), `｡`, `．`, emoji, tab, `\r`,
U+2028, other scripts, a doubled or trailing space, a leading ASCII space. The reason is that
layer 1 and the ICU segmenter cut such text differently, and that the name rule refuses a long ASCII-edged name before a
shorter valid one. Per reader:

- `NarrativeAnalyzer`: a line outside the allow list, or one with any opaque span, uses ICU
  sentences and `Term` tokens.
- `CrossChapterCandidates`: a line outside the allow list, one with any opaque span (heading,
  list item, code, formula, citation, URL), or one in a language that layer 2 does not support
  uses the generic tokenizer.

Differential tests (`doc_line_tests`, `doc_pronoun_tests`) compare the doc path with the old
path, including the inputs of both reviews of pull request 36. Fixed-seed generated runs compare
the full `Analysis` JSON of `analyze` and `analyze_source_mentions`, and the sentence ranges and
term hits of each doc-path line: 20,001 chapters from the allowed alphabet and 10,000 mixed
chapters (other chars, paragraph indents, CRLF, empty lines, no final newline, 4,000-char
lines). A run of 20,000 generated lines compares the layer 2 pronouns. Each run must report zero
differences. When one fails, the check gets stricter; no layer and no expected value changes.

These stay on the old path too:

- The plain `analyze()` function: it has no analyzer and uses its own quote splitter.
- `timeline::build`: it scans windows with cue lexicons and has no segmenter.
- English and zh-CN: the layers have no zh-CN reading.
