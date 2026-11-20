use faberline_nlp::Span;
use faberline_nlp::doc::{Document, Opaque, SyntaxToken, Token, Upos};

const FORM_NAMES: &[&str] = &[
    "文字数",
    "字数",
    "送り仮名",
    "表記",
    "字形",
    "字面",
    "綴り",
    "つづり",
    "文字列",
    "文字",
    "字",
];
const COMPARED_FORM_NAMES: &[&str] = &["表記", "字形", "字面", "綴り", "つづり"];
const FORM_COUNTERS: &[&str] = &["つの", "種類の"];
const FORM_SCRIPTS: &[&str] = &["カタカナ", "片仮名", "ひらがな", "平仮名", "漢字"];
const WORD_NAMES: &[&str] = &["言葉", "単語", "語句", "文字列", "語"];
const CHARACTER_COUNT_PREDICATES: &[&str] = &[
    "です",
    "でした",
    "でしょう",
    "である",
    "だった",
    "ではありません",
    "ではない",
    "だ",
    "あります",
    "ありません",
    "ありました",
    "ある",
    "あった",
    "あるのです",
    "あるでしょう",
    "になります",
    "になる",
    "になりました",
    "になった",
    "なんです",
    "なの",
];
const FORM_LABELS: &[&str] = &[
    "日本語の",
    "英語の",
    "中国語の",
    "カタカナの",
    "片仮名の",
    "ひらがなの",
    "平仮名の",
    "漢字の",
    "ローマ字の",
];

#[derive(Clone, Copy)]
pub(crate) struct WrittenForm {
    pub(crate) span: Span,
    pub(crate) continuation: bool,
}

#[derive(Clone, Copy, Default)]
struct QuoteSyntax<'a> {
    tokens: &'a [Token],
    dependencies: &'a [SyntaxToken],
    opaque: &'a [Opaque],
}

impl<'a> QuoteSyntax<'a> {
    fn first_predicate(self, first: usize, end: usize) -> Option<(usize, &'a Token)> {
        self.tokens
            .iter()
            .enumerate()
            .skip(first)
            .take_while(|(_, t)| t.span.end_char <= end)
            .find(|(_, t)| {
                matches!(
                    t.upos,
                    Some(Upos::Verb | Upos::Aux | Upos::Sconj | Upos::Cconj)
                )
            })
    }

    fn head(self, token: usize) -> Option<usize> {
        self.edge(token)?.head
    }

    fn edge(self, token: usize) -> Option<&'a SyntaxToken> {
        let i = self
            .dependencies
            .binary_search_by_key(&token, |edge| edge.token)
            .ok()?;
        self.dependencies.get(i)
    }
}

/// Find explicitly written forms on the source paragraph, then give each
/// analysis sentence only its part of the original text to protect.
pub(crate) fn plan_written_form_quotes(chars: &[char], doc: &Document) -> Vec<Vec<WrittenForm>> {
    let segmentation = &doc.segmentation;
    let syntax = QuoteSyntax {
        tokens: &doc.tokens,
        dependencies: &doc.syntax,
        opaque: &segmentation.opaque,
    };
    let mut by_sentence = vec![Vec::new(); segmentation.sentences.len()];
    let sentences = &segmentation.sentences;
    for paragraph in &segmentation.paragraphs {
        let mut quotes = written_form_quotes(chars, paragraph.span, syntax);
        quotes.sort_unstable_by_key(|span| (span.start_char, std::cmp::Reverse(span.end_char)));
        let mut last_end = paragraph.span.start_char;
        for quote in quotes {
            // An explicitly literal outer quote already contains its inner quote.
            if quote.start_char < last_end {
                continue;
            }
            last_end = quote.end_char;
            let first = sentences.partition_point(|sent| sent.span.end_char <= quote.start_char);
            for (index, sent) in sentences.iter().enumerate().skip(first) {
                if quote.end_char <= sent.span.start_char {
                    break;
                }
                let start_char = quote.start_char.max(sent.span.start_char);
                let end_char = quote.end_char.min(sent.span.end_char);
                if start_char < end_char {
                    by_sentence[index].push(WrittenForm {
                        span: Span {
                            start_char,
                            end_char,
                        },
                        continuation: quote.start_char < start_char,
                    });
                }
            }
        }
    }
    by_sentence
}

struct OpenQuote {
    start: usize,
    expected_close: char,
    children: Vec<Span>,
    first_protection: usize,
}

fn written_form_quotes(chars: &[char], paragraph: Span, syntax: QuoteSyntax<'_>) -> Vec<Span> {
    let mut openings = Vec::new();
    let mut protected = Vec::new();
    let mut top_level = Vec::new();
    let raw = &chars[paragraph.start_char..paragraph.end_char];
    for (offset, &ch) in raw.iter().enumerate() {
        let at = paragraph.start_char + offset;
        match ch {
            '「' | '『' => openings.push(OpenQuote {
                start: at,
                expected_close: if ch == '「' { '」' } else { '』' },
                children: Vec::new(),
                first_protection: protected.len(),
            }),
            close @ ('」' | '』') => {
                let Some(frame) = openings.pop() else {
                    continue;
                };
                if close != frame.expected_close {
                    let first = openings.first().unwrap_or(&frame);
                    protected.truncate(first.first_protection);
                    openings.clear();
                    continue;
                }
                if frame.start + 1 == at {
                    continue;
                }
                let inner = Span {
                    start_char: frame.start + 1,
                    end_char: at,
                };
                protect_sibling_forms(chars, inner, &frame.children, true, syntax, &mut protected);
                if let Some(parent) = openings.last_mut() {
                    parent.children.push(inner);
                } else {
                    top_level.push(inner);
                }
            }
            _ => {}
        }
    }
    if let Some(unclosed) = openings.first() {
        protected.truncate(unclosed.first_protection);
    }
    protect_sibling_forms(chars, paragraph, &top_level, false, syntax, &mut protected);
    protected
}

fn protection_span(inner: Span, nested: bool) -> Span {
    if nested {
        // Keep the inner quotation as one atom within the outer quotation.
        Span {
            start_char: inner.start_char - 1,
            end_char: inner.end_char + 1,
        }
    } else {
        inner
    }
}

fn protect_sibling_forms(
    chars: &[char],
    scope: Span,
    quotes: &[Span],
    nested: bool,
    syntax: QuoteSyntax<'_>,
    protected: &mut Vec<Span>,
) {
    let mut first_in_list = 0;
    let mut first_for_comparison = 0;
    let mut labelled = false;
    for (index, quote) in quotes.iter().enumerate() {
        let opening = quote.start_char - 1;
        let before = &chars[scope.start_char..opening];
        let after = skip_space(&chars[quote.end_char + 1..scope.end_char]);
        let join = index.checked_sub(1).and_then(|prev| {
            let after_close = quotes[prev].end_char + 1;
            quote_join(&chars[after_close..opening])
        });
        if let Some(QuoteJoin::Conjunction(right_labelled)) = join {
            labelled &= right_labelled;
        } else {
            first_for_comparison = index;
            if join.is_none() {
                first_in_list = index;
            }
            let before = trim_space_end(before);
            labelled = FORM_LABELS
                .iter()
                .any(|label| strip_suffix(before, label).is_some());
        }
        let measures_form = measures_written_form(after);
        let before_joined = &chars[scope.start_char..quotes[first_for_comparison].start_char - 1];
        let before_list = &chars[scope.start_char..quotes[first_in_list].start_char - 1];
        let list_count = index - first_in_list + 1;
        let counts_words = first_in_list < first_for_comparison
            && (has_counted_name_before(before_list, list_count, WORD_NAMES)
                || strip_prefix(after, "という")
                    .and_then(|tail| skip_space(tail).first())
                    .is_some_and(|&number| matches_list_count(number, list_count)));
        let reading_start = if counts_words {
            first_in_list
        } else {
            first_for_comparison
        };
        let first_quote = quotes[reading_start].start_char - 1;
        let reads_word = names_source_reading(&chars[scope.start_char..first_quote], after, || {
            !ends_in_reading_material(chars, scope.start_char, first_quote, syntax)
                && quotes[reading_start..=index].iter().all(|form| {
                    !ends_in_reading_material(chars, form.start_char, form.end_char, syntax)
                })
        });
        let shared_copy_object = index > first_in_list && after.starts_with(&['を']);
        let copies_form =
            copies_written_form(chars, after, scope.end_char, syntax, shared_copy_object);
        let exact_writing = quotes_exact_writing(chars, scope, *quote, *quote, syntax);
        let translation = source_translation_predicate(chars, scope, *quote, syntax);
        if measures_form
            || reads_word
            || copies_form
            || exact_writing
            || translation.is_some()
            || describes_written_form(after)
            || acts_on_written_form(before, after)
        {
            protected.push(protection_span(*quote, nested));
        }
        if exact_writing {
            protected.extend(
                quotes[first_for_comparison..index]
                    .iter()
                    .filter(|&&word| quotes_exact_writing(chars, scope, *quote, word, syntax))
                    .map(|&word| protection_span(word, nested)),
            );
        }
        if let Some(predicate) = translation {
            protected.extend(
                quotes[first_for_comparison..index]
                    .iter()
                    .filter(|&&word| quote_governs_translation(word, predicate, syntax))
                    .map(|&word| protection_span(word, nested)),
            );
        }

        // The last operand can carry the predicate for a list. A comma alone can
        // also separate an utterance from the following word-reading question.
        let first = if index > first_in_list
            && (measures_form
                || (reads_word && counts_words)
                || (shared_copy_object && copies_form))
        {
            first_in_list
        } else if index > first_for_comparison {
            let count = index - first_for_comparison + 1;
            if !reads_word
                && !compares_written_forms(after)
                && !compares_apposed_forms(before_joined, after, count, labelled)
            {
                continue;
            }
            first_for_comparison
        } else {
            continue;
        };
        protected.extend(
            quotes[first..=index]
                .iter()
                .map(|&span| protection_span(span, nested)),
        );
    }
}

fn names_character_count(chars: &[char]) -> bool {
    let chars = skip_space(chars);
    let chars = ["合計の", "総計の", "総", "全体の", "それぞれの", "各々の"]
        .iter()
        .find_map(|prefix| strip_prefix(chars, prefix))
        .map_or(chars, skip_space);
    ["文字数", "字数"]
        .iter()
        .any(|name| strip_prefix(chars, name).is_some_and(noun_boundary))
}

#[derive(Debug, PartialEq, Eq)]
enum QuoteJoin {
    Conjunction(bool),
    Enumeration,
}

fn quote_join(between: &[char]) -> Option<QuoteJoin> {
    let between = skip_space(between);
    if trim_space_end(between) == ['、'] {
        return Some(QuoteJoin::Enumeration);
    }
    // A single Japanese comma can follow the conjunction と.
    let rest = skip_space(between.strip_prefix(&['と'])?);
    let rest = rest.strip_prefix(&['、']).map(skip_space).unwrap_or(rest);
    if rest.is_empty() {
        Some(QuoteJoin::Conjunction(false))
    } else {
        FORM_LABELS
            .iter()
            .any(|label| strip_prefix(rest, label).is_some_and(|tail| skip_space(tail).is_empty()))
            .then_some(QuoteJoin::Conjunction(true))
    }
}

fn compares_apposed_forms(before: &[char], after: &[char], count: usize, labelled: bool) -> bool {
    if compares_objects(after)
        && (labelled || has_counted_name_before(before, count, COMPARED_FORM_NAMES))
    {
        return true;
    }
    let after = strip_prefix(after, "という")
        .or_else(|| after.strip_prefix(&['、']))
        .or_else(|| after.strip_prefix(&['の']))
        .unwrap_or(after);
    let Some((&number, rest)) = skip_space(after).split_first() else {
        return false;
    };
    matches_list_count(number, count)
        && FORM_COUNTERS.iter().any(|counter| {
            strip_prefix(skip_space(rest), counter).is_some_and(|rest| {
                COMPARED_FORM_NAMES.iter().any(|form| {
                    strip_prefix(skip_space(rest), form)
                        .is_some_and(|tail| compares_objects(skip_space(tail)))
                })
            })
        })
}

fn has_counted_name_before(before: &[char], count: usize, names: &[&str]) -> bool {
    names.iter().any(|form| {
        let Some(before) = strip_suffix(trim_space_end(before), form) else {
            return false;
        };
        FORM_COUNTERS.iter().any(|counter| {
            strip_suffix(trim_space_end(before), counter)
                .and_then(|before| trim_space_end(before).split_last())
                .is_some_and(|(&number, rest)| {
                    matches_list_count(number, count)
                        && !rest.last().is_some_and(|&ch| is_count_character(ch))
                })
        })
    })
}

fn matches_list_count(ch: char, count: usize) -> bool {
    count < 10
        && ("〇一二三四五六七八九".chars().nth(count) == Some(ch)
            || ch.to_digit(10) == Some(count as u32)
            || ch as u32 == '０' as u32 + count as u32)
}

fn compares_objects(after: &[char]) -> bool {
    after.strip_prefix(&['を']).is_some_and(|rest| {
        [
            "比べる",
            "比べま",
            "比べた",
            "比べて",
            "比べな",
            "見比べる",
            "見比べま",
            "見比べた",
            "見比べて",
            "見比べな",
            "比較し",
            "比較する",
        ]
        .iter()
        .any(|verb| strip_prefix(skip_space(rest), verb).is_some())
    })
}

// Protect the original operand of a translation. A quotation followed by
// と can instead give the resulting translation or ordinary speech.
fn source_translation_predicate(
    chars: &[char],
    scope: Span,
    quote: Span,
    syntax: QuoteSyntax<'_>,
) -> Option<usize> {
    let tail = skip_space(&chars[quote.end_char + 1..scope.end_char]);
    let labelled_object = strip_prefix(tail, "という").and_then(after_word_name);
    let operand = labelled_object.unwrap_or(tail);
    let relation = ["だけ", "のみ"]
        .iter()
        .find_map(|focus| strip_prefix(operand, focus))
        .map_or(operand, skip_space);
    if !matches!(relation.first(), Some('を' | 'は' | 'も')) {
        return None;
    }
    let relation_start = scope.end_char - relation.len();
    if labelled_object.is_some() {
        let case = syntax
            .tokens
            .partition_point(|t| t.span.start_char < relation_start);
        if let Some(head) = syntax
            .tokens
            .get(case)
            .filter(|t| t.span.start_char == relation_start)
            .and_then(|_| syntax.head(case))
            .and_then(|word| syntax.head(word))
            .filter(|&index| {
                source_translation_verb(chars, relation_start, scope.end_char, syntax, index)
            })
        {
            return Some(head);
        }
    }
    let first = syntax
        .tokens
        .partition_point(|t| t.span.start_char < quote.start_char);
    let direct = syntax
        .tokens
        .iter()
        .enumerate()
        .skip(first)
        .take_while(|(_, t)| t.span.end_char <= quote.end_char)
        .filter_map(|(index, _)| syntax.head(index))
        .find(|&index| {
            source_translation_verb(chars, relation_start, scope.end_char, syntax, index)
        });
    direct.or_else(|| translation_after_quoted_operand(chars, scope, relation_start, syntax))
}

// A sentence-final mark inside the quotation can end the analyzer's sentence.
// Then its content has no dependency into the outer sentence. The adjacent
// operand marker still belongs to the first predicate in the outer clause.
fn translation_after_quoted_operand(
    chars: &[char],
    scope: Span,
    relation_start: usize,
    syntax: QuoteSyntax<'_>,
) -> Option<usize> {
    let first = syntax
        .tokens
        .partition_point(|t| t.span.start_char <= relation_start);
    let (index, verb) = syntax.first_predicate(first, scope.end_char)?;
    let across_boundary = chars[relation_start..verb.span.start_char]
        .iter()
        .any(|ch| matches!(ch, '。' | '！' | '？' | '!' | '?' | '「' | '『' | '\n'));
    (!across_boundary
        && source_translation_verb(chars, relation_start, scope.end_char, syntax, index))
    .then_some(index)
}

fn quote_governs_translation(quote: Span, predicate: usize, syntax: QuoteSyntax<'_>) -> bool {
    let first = syntax
        .tokens
        .partition_point(|t| t.span.start_char < quote.start_char);
    syntax
        .tokens
        .iter()
        .enumerate()
        .skip(first)
        .take_while(|(_, t)| t.span.end_char <= quote.end_char)
        .any(|(index, _)| syntax.head(index) == Some(predicate))
}

fn source_translation_verb(
    chars: &[char],
    after: usize,
    end: usize,
    syntax: QuoteSyntax<'_>,
    index: usize,
) -> bool {
    let verb = &syntax.tokens[index];
    if verb.upos != Some(Upos::Verb) || verb.span.start_char <= after || verb.span.end_char > end {
        return false;
    }
    let surface: String = chars[verb.span.start_char..verb.span.end_char]
        .iter()
        .collect();
    match verb.lemma.as_deref().unwrap_or(&surface) {
        "訳す" | "訳せる" | "翻訳する" | "直訳する" | "意訳する" => true,
        "訳" | "翻訳" | "直訳" | "意訳" => {
            syntax.tokens.get(index + 1).is_some_and(|suffix| {
                syntax.head(index) == Some(index + 1)
                    && suffix.span.start_char == verb.span.end_char
                    && suffix.span.end_char <= end
                    && suffix.lemma.as_deref() == Some("する")
            })
        }
        "する" | "できる" | "出来る" => index.checked_sub(1).is_some_and(|prev| {
            let noun = &syntax.tokens[prev];
            noun.span.start_char > after
                && noun.span.end_char == verb.span.start_char
                && syntax.head(prev) == Some(index)
                && ["訳", "翻訳", "直訳", "意訳"].iter().any(|form| {
                    strip_prefix(&chars[noun.span.start_char..noun.span.end_char], form)
                        .is_some_and(|rest| rest.is_empty())
                })
        }),
        _ => false,
    }
}

fn compares_written_forms(after: &[char]) -> bool {
    let Some((&particle, rest)) = after.split_first() else {
        return false;
    };
    FORM_NAMES.iter().any(|form| {
        let Some(tail) = strip_prefix(skip_space(rest), form).map(skip_space) else {
            return false;
        };
        match particle {
            'の' => compares_objects(tail),
            'は' => tail.strip_prefix(&['が']).is_some_and(|tail| {
                ["違う", "違い", "違っ", "違わ", "異な", "同じ"]
                    .iter()
                    .any(|comparison| strip_prefix(skip_space(tail), comparison).is_some())
            }),
            _ => false,
        }
    })
}

fn describes_written_form(after: &[char]) -> bool {
    let after = skip_space(after);
    if let Some(note) = strip_prefix(after, "と書いてある") {
        let note = skip_space(note);
        return ["メモの", "紙の", "札の", "カードの"]
            .iter()
            .find_map(|medium| strip_prefix(note, medium))
            .is_some_and(names_character_count);
    }

    let nominal = if let Some(nominal) = strip_prefix(after, "という") {
        nominal
    } else if let Some(nominal) = after.strip_prefix(&['の']) {
        let nominal = skip_space(nominal);
        ["最初の", "最後の", "先頭の", "末尾の"]
            .iter()
            .find_map(|modifier| strip_prefix(nominal, modifier))
            .unwrap_or(nominal)
    } else {
        return false;
    };
    let nominal = without_count(skip_space(nominal));
    names_character_count(nominal)
        || FORM_NAMES
            .iter()
            .any(|form| strip_prefix(nominal, form).is_some_and(noun_boundary))
}

fn acts_on_written_form(before: &[char], after: &[char]) -> bool {
    let Some(object) = after.strip_prefix(&['を']) else {
        return false;
    };
    let amount = after_clause_comma(object);
    let counted = after_character_count(amount);
    let step = counted.and_then(|tail| {
        ["ずつ", "ごとに"]
            .iter()
            .find_map(|marker| strip_prefix(tail, marker))
    });
    let step = step.or_else(|| {
        ["一字一字", "一文字一文字"]
            .iter()
            .find_map(|marker| strip_prefix(amount, marker))
    });
    if step.is_some_and(|tail| {
        ["読", "書", "写", "数", "なぞ"]
            .iter()
            .any(|verb| strip_prefix(after_clause_comma(tail), verb).is_some())
    }) {
        return true;
    }
    let writing_particles = ["では", "でも", "で"];
    if counted
        .and_then(|tail| {
            writing_particles
                .iter()
                .find_map(|particle| strip_prefix(tail, particle))
        })
        .is_some_and(starts_writing)
    {
        return true;
    }
    if !starts_writing(object) {
        return false;
    }
    let before = trim_space_end(before);
    let before = before.strip_suffix(&['、']).map_or(before, trim_space_end);
    writing_particles
        .iter()
        .find_map(|particle| strip_suffix(before, particle))
        .and_then(|count| strip_suffix(trim_space_end(count), "文字"))
        .and_then(|number| trim_space_end(number).last())
        .is_some_and(|&ch| is_count_character(ch))
}

fn copies_written_form(
    chars: &[char],
    after: &[char],
    scope_end: usize,
    syntax: QuoteSyntax<'_>,
    shared_object: bool,
) -> bool {
    let next = if let Some(word) = strip_prefix(after, "という") {
        after_word_name(word).and_then(|operand| operand.strip_prefix(&['を']))
    } else {
        after
            .strip_prefix(&['を'])
            .or_else(|| after.strip_prefix(&['と']))
    };
    let Some(destination) = next.map(after_clause_comma) else {
        return false;
    };
    let destination_start = scope_end - destination.len();
    starts_copying(destination)
        || after_copying_manner(destination).is_some_and(starts_copying)
        || destination
            .iter()
            .take_while(|&&ch| {
                !matches!(
                    ch,
                    'を' | 'が'
                        | 'は'
                        | 'と'
                        | '、'
                        | '。'
                        | '！'
                        | '？'
                        | '!'
                        | '?'
                        | '「'
                        | '」'
                        | '『'
                        | '』'
                        | '\n'
                )
            })
            .enumerate()
            .any(|(index, &ch)| {
                if ch != 'に' || index == 0 {
                    return false;
                }
                let after_particle = skip_space(&destination[index + 1..]);
                let has_comma = after_particle.first() == Some(&'、');
                let tail = after_clause_comma(after_particle);
                let immediate_copy = starts_copying(tail);
                if immediate_copy && !shared_object && !has_comma {
                    return true;
                }
                (immediate_copy || after_copying_manner(tail).is_some_and(starts_copying))
                    && copy_destination_without_predicate(
                        syntax.tokens,
                        destination_start,
                        destination_start + index,
                    )
            })
        || copy_with_local_dependencies(chars, destination_start, scope_end, syntax)
}

/// A plain writing verb can report a message. Keep the original spelling when
/// the writer requests a particular word or explicitly copies it unchanged.
fn quotes_exact_writing(
    chars: &[char],
    scope: Span,
    last: Span,
    quote: Span,
    syntax: QuoteSyntax<'_>,
) -> bool {
    let after = skip_space(&chars[last.end_char + 1..scope.end_char]);
    let Some((&particle @ ('と' | 'を'), tail)) = after.split_first() else {
        return false;
    };
    let start = scope.end_char - tail.len();
    let first = syntax.tokens.partition_point(|t| t.span.start_char < start);
    let Some(marker) = first.checked_sub(1).and_then(|i| syntax.tokens.get(i)) else {
        return false;
    };
    // という can name the word being written, or just describe its meaning.
    let names_written_word = strip_prefix(after, "という")
        .and_then(after_word_name)
        .is_some_and(|rest| matches!(rest.first(), Some('を' | 'が' | 'と' | 'は' | 'も')));
    if (marker.span.start_char + 1 != start || marker.span.end_char != start) && !names_written_word
    {
        return false;
    }
    let Some((predicate, verb)) = syntax.first_predicate(first, scope.end_char) else {
        return false;
    };
    let surface = &chars[verb.span.start_char..verb.span.end_char];
    let copies = verb.lemma.as_deref() == Some("写す") || surface == ['写', 'す'];
    let writes = verb.lemma.as_deref() == Some("書く") || surface == ['書', 'く'];
    let states_inscription = particle == 'と' && verb.lemma.as_deref() == Some("ある");
    if !(copies || writes || states_inscription)
        || chars[start..verb.span.start_char].iter().any(|c| {
            matches!(
                c,
                '「' | '」' | '『' | '』' | '。' | '！' | '？' | '!' | '?' | '\n'
            )
        })
    {
        return false;
    }

    let first_in_quote = syntax
        .tokens
        .partition_point(|t| t.span.start_char < quote.start_char);
    let end_of_quote = syntax
        .tokens
        .partition_point(|t| t.span.start_char < quote.end_char);
    let attached = |i| syntax.head(i) == Some(predicate);
    let words = &syntax.tokens[first_in_quote..end_of_quote];
    let Some(opening_word) = words.first() else {
        return false;
    };
    // Punctuation inside a quote can end the analysis sentence before と or
    // を. The adjacent case particle then begins the outer clause, whose first
    // predicate we already checked. It marks only the immediately preceding quote.
    let split_operand = quote == last
        && words
            .last()
            .is_some_and(|word| word.sentence < verb.sentence)
        && first.checked_sub(1).is_some_and(|i| {
            let case = &syntax.tokens[i];
            case.span.start_char + 1 == start
                && case.span.end_char == start
                && case.sentence == verb.sentence
                && case.upos == Some(Upos::Adp)
        });
    if !split_operand && !(first_in_quote..end_of_quote).any(attached) {
        return false;
    }

    let mut unchanged = false;
    let mut writing_surface = false;
    let mut surface_has_foreign_text = false;
    let mut writes_in_japanese = false;
    for (i, token) in syntax.tokens[..predicate]
        .iter()
        .enumerate()
        .rev()
        .take_while(|(_, t)| {
            t.sentence >= opening_word.sentence && t.span.start_char >= scope.start_char
        })
    {
        if (first_in_quote..end_of_quote).contains(&i) {
            continue;
        }
        let word = &chars[token.span.start_char..token.span.end_char];
        let case = &chars[token.span.end_char..verb.span.start_char];
        // Before an opening quote, the analyser can attach the paper to the
        // quoted adjective or verb. The paper itself lies outside the quote.
        let via_quote = i < first_in_quote
            && syntax
                .head(i)
                .is_some_and(|head| (first_in_quote..end_of_quote).contains(&head));
        if !(attached(i) || via_quote) {
            continue;
        }
        let surface = case.starts_with(&['に'])
            && is_writing_surface(chars, scope.start_char, token.span.end_char, syntax, copies);
        writing_surface |= surface;
        if surface && !copies {
            surface_has_foreign_text |=
                has_foreign_surface_inscription(chars, scope, quote, syntax, i);
        }
        let language_case = inscription_language_case(case);
        let states_language = matches!(language_case, Some('で' | 'に'));
        writes_in_japanese |= states_language && word == ['日', '本', '語'] && attached(i);
        let camera = ["カメラ", "スマホ", "スマートフォン", "携帯", "携帯電話"]
            .iter()
            .any(|name| word.iter().copied().eq(name.chars()));
        if (is_foreign_inscription_language(word) && states_language)
            || (copies && camera && case.starts_with(&['で']))
        {
            // A target language requests translation. An instrument with 写す
            // can take a picture of what the quotation denotes.
            return false;
        }
        if attached(i) && token.upos == Some(Upos::Adv) && word == ['そ', 'の', 'ま', 'ま'] {
            unchanged = true;
        }
    }
    if copies {
        let before = trim_space_end(&chars[scope.start_char..quote.start_char - 1]);
        let written_source = strip_suffix(before, "の").is_some_and(|source| {
            is_writing_surface(
                chars,
                scope.start_char,
                scope.start_char + source.len(),
                syntax,
                true,
            )
        });
        // と introduces what is transcribed onto the page. With を, both a
        // written source and a destination distinguish transcription from a photo.
        return (writing_surface && (particle == 'と' || written_source))
            || (unchanged && (written_source || writing_surface));
    }

    let phrase = &chars[verb.span.start_char..scope.end_char];
    if surface_has_foreign_text
        && !writes_in_japanese
        && !unchanged
        && (states_inscription
            || ["書いてあ", "書いてな", "書かれ"]
                .iter()
                .any(|prefix| strip_prefix(phrase, prefix).is_some()))
    {
        return false;
    }
    let label = particle == 'と' && is_written_phrase(chars, first_in_quote..end_of_quote, syntax);
    let prohibition = particle == 'と'
        && is_written_prohibition(
            chars,
            first_in_quote..end_of_quote,
            syntax,
            quote.start_char,
        );
    if states_inscription {
        // 「…」とある states an inscription when it has an explicit surface.
        return (label || prohibition)
            && writing_surface
            && matches_written_form(
                phrase,
                &[
                    "ある",
                    "あった",
                    "あります",
                    "ありました",
                    "ありません",
                    "ありませんでした",
                ],
                ends_inscription_statement,
            );
    }

    if unchanged {
        return true;
    }
    let instruction = is_writing_instruction(chars, scope, syntax, predicate);
    if label && instruction {
        return true;
    }
    particle == 'と'
        && (label || prohibition)
        && ((writing_surface && instruction)
            || describes_writing(phrase, |tail| {
                (writing_surface && ends_inscription_statement(tail))
                    || modified_writing_surface(chars, scope, syntax, predicate, tail).is_some()
            }))
}

// A command written on a named surface is an inscription even when it has
// an object. Keep this separate from ordinary quoted propositions and speech.
fn is_written_prohibition(
    chars: &[char],
    tokens: std::ops::Range<usize>,
    syntax: QuoteSyntax<'_>,
    quote_start: usize,
) -> bool {
    let Some(marker) = tokens.clone().rev().find(|&i| {
        let word = &syntax.tokens[i];
        !chars[word.span.start_char..word.span.end_char]
            .iter()
            .all(|c| matches!(c, '！' | '!' | '？' | '?' | '。' | '．' | '.'))
    }) else {
        return false;
    };
    let word = &syntax.tokens[marker];
    if word.upos != Some(Upos::Part) || chars[word.span.start_char..word.span.end_char] != ['な'] {
        return false;
    }
    let Some(verb) = syntax.head(marker).filter(|head| tokens.contains(head)) else {
        return false;
    };
    let verb_surface: String = chars[syntax.tokens[verb].span.start_char..word.span.start_char]
        .iter()
        .collect();
    if crate::ja_zh::is_ambiguous_potential(&verb_surface) {
        return false;
    }
    if verb_surface == "降る"
        && (has_ambiguous_quoted_sky_context(chars, tokens.clone(), syntax, verb, quote_start)
            || has_quoted_weather_forecast(chars, tokens.clone(), syntax, verb, quote_start))
    {
        return false;
    }
    let other_subject = tokens.clone().any(|i| {
        let Some(case) = syntax
            .tokens
            .get(i + 1)
            .filter(|_| tokens.contains(&(i + 1)))
        else {
            return false;
        };
        let subject = &syntax.tokens[i];
        let is_subject = syntax
            .dependencies
            .binary_search_by_key(&i, |edge| edge.token)
            .ok()
            .is_some_and(|edge| syntax.dependencies[edge].deprel == "nsubj");
        is_subject
            && syntax.head(i) == Some(verb)
            && syntax.head(i + 1) == Some(i)
            && chars[case.span.start_char..case.span.end_char] == ['が']
            && !crate::ja_zh::is_explicit_second_person(
                &chars[subject.span.start_char..subject.span.end_char]
                    .iter()
                    .collect::<String>(),
            )
    });
    let nonvolitional_subject = tokens.clone().any(|i| {
        let Some(case) = syntax
            .tokens
            .get(i + 1)
            .filter(|_| tokens.contains(&(i + 1)))
        else {
            return false;
        };
        let subject = &syntax.tokens[i];
        let is_subject = syntax
            .dependencies
            .binary_search_by_key(&i, |edge| edge.token)
            .ok()
            .is_some_and(|edge| syntax.dependencies[edge].deprel == "nsubj");
        let case_surface = &chars[case.span.start_char..case.span.end_char];
        let subject_surface: String = chars[subject.span.start_char..subject.span.end_char]
            .iter()
            .collect();
        is_subject
            && syntax.head(i) == Some(verb)
            && syntax.head(i + 1) == Some(i)
            && (case_surface == ['は'] || case_surface == ['が'])
            && crate::ja_zh::has_nonvolitional_subject(
                syntax.tokens[verb]
                    .lemma
                    .as_deref()
                    .unwrap_or(&verb_surface),
                &subject_surface,
            )
    });
    let event_context: String = tokens
        .clone()
        .filter(|&i| {
            let mut current = i;
            for depth in 0..=2 {
                if current == verb {
                    return true;
                }
                if !tokens.contains(&current)
                    || matches!(syntax.tokens[current].upos, Some(Upos::Verb | Upos::Aux))
                {
                    return false;
                }
                if depth == 2
                    && !syntax.edge(i).is_some_and(|edge| {
                        matches!(edge.deprel.as_str(), "case" | "compound" | "fixed")
                    })
                {
                    return false;
                }
                let Some(head) = syntax.head(current) else {
                    return false;
                };
                current = head;
            }
            current == verb
        })
        .flat_map(|i| {
            chars[syntax.tokens[i].span.start_char..syntax.tokens[i].span.end_char].iter()
        })
        .collect();
    let natural_event = tokens.clone().any(|i| {
        let Some(case) = syntax
            .tokens
            .get(i + 1)
            .filter(|_| tokens.contains(&(i + 1)))
        else {
            return false;
        };
        let subject = &syntax.tokens[i];
        let Some(edge) = syntax.edge(i) else {
            return false;
        };
        let attached_to_event = edge.head == Some(verb)
            || edge
                .head
                .filter(|head| tokens.contains(head))
                .is_some_and(|head| {
                    let manner = &syntax.tokens[head];
                    chars[manner.span.start_char..manner.span.end_char] == ['静', 'か']
                        && matches!(manner.upos, Some(Upos::Adj | Upos::Adv))
                        && syntax.edge(head).is_some_and(|parent| {
                            parent.deprel == "advcl" && parent.head == Some(verb)
                        })
                        && tokens.clone().any(|j| {
                            let ending = &syntax.tokens[j];
                            ending.span.start_char == manner.span.end_char
                                && chars[ending.span.start_char..ending.span.end_char] == ['に']
                                && syntax.head(j) == Some(head)
                        })
                });
        let case_surface = &chars[case.span.start_char..case.span.end_char];
        let subject_surface: String = chars[subject.span.start_char..subject.span.end_char]
            .iter()
            .collect();
        edge.deprel == "nsubj"
            && attached_to_event
            && syntax.head(i + 1) == Some(i)
            && (case_surface == ['は'] || case_surface == ['が'])
            && !crate::ja_zh::is_explicit_second_person(&subject_surface)
            && crate::ja_zh::is_natural_event(
                syntax.tokens[verb]
                    .lemma
                    .as_deref()
                    .unwrap_or(&verb_surface),
                &subject_surface,
                &event_context,
            )
    });
    syntax.tokens[verb].upos == Some(Upos::Verb)
        && !other_subject
        && !nonvolitional_subject
        && !natural_event
        && !verb.checked_sub(1).is_some_and(|i| {
            i >= tokens.start
                && syntax.tokens[i].span.end_char <= syntax.tokens[verb].span.start_char
                && chars[syntax.tokens[i].span.end_char..syntax.tokens[verb].span.start_char]
                    .iter()
                    .all(|c| c.is_whitespace())
                && chars[syntax.tokens[i].span.start_char..syntax.tokens[i].span.end_char]
                    == ['よ', 'く']
        })
        && tokens
            .clone()
            .filter(|&i| syntax.tokens[i].upos == Some(Upos::Verb))
            .count()
            == 1
        && marker.checked_sub(1).is_some_and(|i| {
            i >= tokens.start
                && syntax.tokens[i].span.end_char == word.span.start_char
                && if i == verb {
                    let surface = &chars[syntax.tokens[verb].span.start_char..word.span.start_char];
                    surface.last().is_some_and(|c| {
                        matches!(
                            c,
                            'う' | 'く' | 'ぐ' | 'す' | 'つ' | 'ぬ' | 'ぶ' | 'む' | 'る'
                        )
                    }) && !faberline_nlp::lexicon::lookup_ja(&surface.iter().collect::<String>())
                        .is_some_and(|entry| {
                            entry.counterpart.as_deref().is_some_and(|meaning| {
                                meaning.starts_with('能') || meaning.starts_with("可以")
                            })
                        })
                } else {
                    let ending = &syntax.tokens[i];
                    let stem = &syntax.tokens[verb];
                    ending.upos == Some(Upos::Aux)
                        && syntax.head(i) == Some(verb)
                        && stem.span.end_char == ending.span.start_char
                        && chars[ending.span.start_char..ending.span.end_char] == ['る']
                        && faberline_nlp::lexicon::lookup_ja(
                            &chars[stem.span.start_char..ending.span.end_char]
                                .iter()
                                .collect::<String>(),
                        )
                        .is_some_and(|entry| {
                            entry.upos_candidates.contains(&Upos::Verb)
                                && entry.infl == Some("ichidan")
                                && entry.counterpart.as_deref().is_some_and(|meaning| {
                                    !meaning.starts_with('能') && !meaning.starts_with("可以")
                                })
                        })
                }
        })
}

// An adjacent sky statement inside this quotation does not establish whether
// an omitted subject is a weather report or a command. Leave it to the normal
// translation path rather than treating the quotation as a written command.
fn has_ambiguous_quoted_sky_context(
    chars: &[char],
    tokens: std::ops::Range<usize>,
    syntax: QuoteSyntax<'_>,
    verb: usize,
    quote_start: usize,
) -> bool {
    let Some(current) = syntax.tokens.get(verb) else {
        return false;
    };
    if !syntax.tokens.get(tokens.start).is_some_and(|token| {
        quote_start <= token.span.start_char && token.span.start_char <= current.span.start_char
    }) {
        return false;
    }
    let Some(previous_sentence) = current.sentence.checked_sub(1) else {
        return false;
    };
    let surface = |i: usize| {
        let token = syntax.tokens.get(i)?;
        chars.get(token.span.start_char..token.span.end_char)
    };
    let mut prefix_indices = Vec::new();
    let mut prefix_positions = Vec::new();
    for i in tokens.clone() {
        let Some(token) = syntax.tokens.get(i) else {
            return false;
        };
        if token.sentence == current.sentence && token.span.end_char <= current.span.start_char {
            let opaque_index = syntax
                .opaque
                .partition_point(|opaque| opaque.span.end_char <= token.span.start_char);
            if let Some(opaque) = syntax.opaque.get(opaque_index)
                && opaque.span.start_char < token.span.end_char
                && token.span.start_char < opaque.span.end_char
            {
                if opaque.span.start_char <= token.span.start_char
                    && token.span.start_char < token.span.end_char
                    && token.span.end_char <= opaque.span.end_char
                {
                    continue;
                }
                // Never hide a token that is only partly opaque.
                return false;
            }
            prefix_indices.push(i);
            prefix_positions.push(token.span.start_char);
        }
    }
    let Some(same_scopes) = crate::ja_zh::quote_scope_relations_from(
        chars,
        quote_start,
        &prefix_positions,
        current.span.start_char,
        syntax.opaque,
    ) else {
        return false;
    };
    if same_scopes.len() != prefix_indices.len()
        || !prefix_indices
            .into_iter()
            .zip(same_scopes)
            .all(|(i, same_scope)| {
                surface(i).is_some_and(|word| {
                    let Some(token) = syntax.tokens.get(i) else {
                        return false;
                    };
                    !same_scope
                        || (token.upos == Some(Upos::Punct)
                            && (word == ['、']
                                || word == ['「']
                                || word == ['」']
                                || word == ['『']
                                || word == ['』']
                                || word == ['“']
                                || word == ['”']
                                || word == ['‘']
                                || word == ['’']
                                || word == ['"']))
                        || word == ['明', '日']
                        || word == ['今', '日']
                        || word == ['今', '夜']
                        || word == ['は']
                        || word == ['も']
                })
            })
    {
        return false;
    }
    if tokens.clone().any(|i| {
        syntax.tokens.get(i).is_some_and(|token| {
            token.sentence == previous_sentence
                && matches!(token.upos, Some(Upos::Verb | Upos::Aux))
        })
    }) {
        return false;
    }
    tokens.clone().any(|i| {
        let Some(subject) = syntax.tokens.get(i) else {
            return false;
        };
        if subject.sentence != previous_sentence || surface(i) != Some(&['空'][..]) {
            return false;
        }
        let Some(edge) = syntax.edge(i).filter(|edge| edge.deprel == "nsubj") else {
            return false;
        };
        let Some(head) = edge.head.filter(|head| tokens.contains(head)) else {
            return false;
        };
        let Some(predicate) = syntax.tokens.get(head) else {
            return false;
        };
        if predicate.sentence != previous_sentence
            || !syntax.edge(head).is_some_and(|edge| edge.deprel == "root")
            || !(predicate.lemma.as_deref() == Some("暗い")
                || surface(head) == Some(&['暗', 'い'][..]))
        {
            return false;
        }
        crate::ja_zh::quote_scope_relation_from(
            chars,
            quote_start,
            predicate.span.end_char,
            current.span.start_char,
            syntax.opaque,
        ) == Some(true)
    })
}

// A forecast attached to this event gives an explicit weather context. Keep
// a vocative outside this rule, since it can address the rain directly.
fn has_quoted_weather_forecast(
    chars: &[char],
    tokens: std::ops::Range<usize>,
    syntax: QuoteSyntax<'_>,
    verb: usize,
    quote_start: usize,
) -> bool {
    let Some(action) = syntax.tokens.get(verb) else {
        return false;
    };
    if !syntax.tokens.get(tokens.start).is_some_and(|token| {
        quote_start <= token.span.start_char && token.span.start_char <= action.span.start_char
    }) {
        return false;
    }
    let surface = |i: usize| {
        let token = syntax.tokens.get(i)?;
        chars.get(token.span.start_char..token.span.end_char)
    };
    let mut address_positions = Vec::new();
    for i in tokens.clone().filter(|&i| i < verb) {
        let Some(token) = syntax.tokens.get(i) else {
            return false;
        };
        if token.sentence != action.sentence {
            continue;
        }
        let Some(word) = surface(i) else {
            return false;
        };
        let is_second_person =
            crate::ja_zh::is_explicit_second_person(&word.iter().collect::<String>());
        let is_vocative = word == ['よ']
            && syntax.edge(i).is_some_and(|edge| {
                edge.deprel == "discourse"
                    && edge
                        .head
                        .filter(|head| tokens.contains(head))
                        .is_some_and(|head| {
                            syntax.tokens.get(head).is_some_and(|subject| {
                                matches!(subject.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
                            }) && syntax
                                .edge(head)
                                .is_some_and(|subject_edge| subject_edge.deprel == "nmod")
                        })
            });
        if !is_second_person && !is_vocative {
            continue;
        }
        address_positions.push(token.span.start_char);
    }
    let Some(address_scopes) = crate::ja_zh::quote_scope_relations_from(
        chars,
        quote_start,
        &address_positions,
        action.span.start_char,
        syntax.opaque,
    ) else {
        return false;
    };
    if address_scopes.len() != address_positions.len()
        || address_scopes.into_iter().any(|same| same)
    {
        return false;
    }
    let mut forecast_positions = Vec::new();
    for i in tokens.clone() {
        let Some(forecast) = i.checked_add(1).filter(|index| tokens.contains(index)) else {
            continue;
        };
        let Some(particle) = i.checked_add(2).filter(|index| tokens.contains(index)) else {
            continue;
        };
        let Some(topic) = i.checked_add(3).filter(|index| tokens.contains(index)) else {
            continue;
        };
        let Some(forecast_token) = syntax.tokens.get(forecast) else {
            return false;
        };
        if i < verb
            && topic < verb
            && surface(i) == Some(&['天', '気'][..])
            && surface(forecast) == Some(&['予', '報'][..])
            && surface(particle) == Some(&['で'][..])
            && surface(topic) == Some(&['は'][..])
            && forecast_token.sentence == action.sentence
            && syntax
                .edge(i)
                .is_some_and(|edge| edge.deprel == "compound" && edge.head == Some(forecast))
            && syntax.edge(forecast).is_some_and(|edge| {
                matches!(edge.deprel.as_str(), "nsubj" | "obl" | "advmod")
                    && edge.head == Some(verb)
            })
            && syntax
                .edge(particle)
                .is_some_and(|edge| edge.deprel == "case" && edge.head == Some(forecast))
            && syntax
                .edge(topic)
                .is_some_and(|edge| edge.deprel == "case" && edge.head == Some(forecast))
        {
            forecast_positions.push(forecast_token.span.start_char);
        }
    }
    if forecast_positions.is_empty() {
        return false;
    }
    let Some(forecast_scopes) = crate::ja_zh::quote_scope_relations_from(
        chars,
        quote_start,
        &forecast_positions,
        action.span.start_char,
        syntax.opaque,
    ) else {
        return false;
    };
    forecast_scopes.len() == forecast_positions.len()
        && forecast_scopes.into_iter().any(|same| same)
}

/// An inflected word or a noun phrase can be the inscription. A quoted
/// proposition with its own subject or object still reports its meaning.
fn is_written_phrase(
    chars: &[char],
    tokens: std::ops::Range<usize>,
    syntax: QuoteSyntax<'_>,
) -> bool {
    let mut roots = 0;
    for i in tokens.clone() {
        let token = &syntax.tokens[i];
        let Ok(n) = syntax
            .dependencies
            .binary_search_by_key(&i, |edge| edge.token)
        else {
            return false;
        };
        let edge = &syntax.dependencies[n];
        if !edge.head.is_some_and(|head| tokens.contains(&head)) {
            roots += 1;
            continue;
        }
        let surface = &chars[token.span.start_char..token.span.end_char];
        match edge.deprel.as_str() {
            "compound" | "fixed" | "flat" | "amod" | "nummod" | "nmod" | "det" => {}
            "case" if surface == ['の'] => {}
            // に forms an adverb from an adjective, as in 静かに.
            "case"
                if surface == ['に']
                    && edge.head.is_some_and(|head| {
                        let word = &syntax.tokens[head];
                        word.upos == Some(Upos::Adj) || word.upos_candidates.contains(&Upos::Adj)
                    }) => {}
            "aux"
                if ["ない", "なかっ", "た", "だ"]
                    .iter()
                    .any(|s| surface.iter().copied().eq(s.chars())) => {}
            _ => return false,
        }
    }
    roots == 1
}

fn describes_writing(phrase: &[char], accepts_tail: impl Fn(&[char]) -> bool) -> bool {
    let (stem, forms): (_, &[&str]) = if let Some(aspect) =
        strip_prefix(phrase, "書いてい").or_else(|| strip_prefix(phrase, "書かれてい"))
    {
        // いる carries the tense and negation of an ongoing inscription.
        (
            aspect,
            &[
                "る",
                "た",
                "ない",
                "なかった",
                "ます",
                "ました",
                "ません",
                "ませんでした",
            ],
        )
    } else if let Some(aspect) = strip_prefix(phrase, "書いて") {
        (
            aspect,
            &[
                "",
                "ある",
                "あった",
                "ない",
                "なかった",
                "あります",
                "ありました",
                "ありません",
                "ありませんでした",
            ],
        )
    } else {
        (
            phrase,
            &[
                "書く",
                "書いた",
                "書かない",
                "書かなかった",
                "書きます",
                "書きました",
                "書きません",
                "書きませんでした",
                "書き",
                "書かずに",
                "書かないで",
                "書かれる",
                "書かれた",
                "書かれない",
                "書かれなかった",
                "書かれます",
                "書かれました",
                "書かれません",
                "書かれませんでした",
            ],
        )
    };
    matches_written_form(stem, forms, accepts_tail)
}

fn matches_written_form(
    phrase: &[char],
    forms: &[&str],
    accepts_tail: impl Fn(&[char]) -> bool,
) -> bool {
    forms
        .iter()
        .filter_map(|form| strip_prefix(phrase, form))
        .any(|tail| accepts_tail(skip_space(tail)))
}

fn ends_inscription_statement(tail: &[char]) -> bool {
    let sentence_end = ["か", "よ", "ね"].iter().fold(tail, |text, particle| {
        strip_prefix(text, particle).unwrap_or(text)
    });
    matches!(
        sentence_end.first(),
        None | Some('。' | '？' | '！' | '?' | '!' | '、' | 'と' | 'が')
    ) || ["かどうか", "後", "あと", "前", "から", "ので", "けれど"]
        .iter()
        .any(|link| strip_prefix(tail, link).is_some())
}

fn inscription_language_case(tail: &[char]) -> Option<char> {
    let tail = skip_space(tail);
    let tail = strip_prefix(tail, "だけ")
        .or_else(|| strip_prefix(tail, "のみ"))
        .unwrap_or(tail);
    skip_space(tail).first().copied()
}

fn is_foreign_inscription_language(word: &[char]) -> bool {
    word.len() > 1
        && word.ends_with(&['語'])
        && !["日本語", "単語", "用語", "言語", "専門用語"]
            .iter()
            .any(|name| word.iter().copied().eq(name.chars()))
}

fn has_foreign_surface_inscription(
    chars: &[char],
    scope: Span,
    quote: Span,
    syntax: QuoteSyntax<'_>,
    surface: usize,
) -> bool {
    let noun = &syntax.tokens[surface];
    let outside_quote = |token: &Token| {
        token.span.end_char <= quote.start_char || token.span.start_char >= quote.end_char
    };
    syntax.tokens[..surface]
        .iter()
        .enumerate()
        .rev()
        .take_while(|(_, token)| {
            token.sentence == noun.sentence && token.span.start_char >= scope.start_char
        })
        .any(|(index, word)| {
            if !outside_quote(word)
                || !is_foreign_inscription_language(
                    &chars[word.span.start_char..word.span.end_char],
                )
            {
                return false;
            }
            let Some(edge) = syntax.edge(index) else {
                return false;
            };
            let case = inscription_language_case(&chars[word.span.end_char..noun.span.start_char]);
            if edge.head == Some(surface) {
                return edge.deprel == "nmod" && case == Some('の');
            }
            let Some(writer) = edge.head.filter(|&head| index < head && head < surface) else {
                return false;
            };
            let verb = &syntax.tokens[writer];
            outside_quote(verb)
                && verb.lemma.as_deref() == Some("書く")
                && matches!(case, Some('で' | 'に'))
                && describes_writing(&chars[verb.span.start_char..scope.end_char], |tail| {
                    modified_writing_surface(chars, scope, syntax, writer, tail) == Some(surface)
                })
        })
}

fn modified_writing_surface(
    chars: &[char],
    scope: Span,
    syntax: QuoteSyntax<'_>,
    predicate: usize,
    tail: &[char],
) -> Option<usize> {
    let start = scope.end_char - tail.len();
    let first = syntax
        .tokens
        .binary_search_by_key(&start, |t| t.span.start_char)
        .ok()?;
    let noun = syntax
        .edge(first)
        .filter(|edge| matches!(edge.deprel.as_str(), "compound" | "amod" | "det"))
        .and_then(|edge| edge.head.filter(|&head| head > first))
        .unwrap_or(first);
    let surface = &syntax.tokens[noun];
    if surface.sentence != syntax.tokens[predicate].sentence
        || surface.span.end_char > scope.end_char
        || !matches!(surface.upos, Some(Upos::Noun | Upos::Propn))
        || !is_writing_surface(
            chars,
            scope.start_char,
            surface.span.end_char,
            syntax,
            false,
        )
    {
        return None;
    }

    (predicate..first).any(|index| {
        syntax.edge(index).is_some_and(|edge| {
            edge.head == Some(noun)
                && ((edge.deprel == "acl"
                    && (index == predicate || syntax.head(predicate) == Some(index)))
                    // Before a noun, the analyzer can tag the resultant ある as a determiner.
                    || (index + 1 == first
                        && edge.deprel == "det"
                        && chars[syntax.tokens[index].span.start_char..syntax.tokens[index].span.end_char]
                            == ['あ', 'る']))
        })
    }).then_some(noun)
}

fn is_writing_instruction(
    chars: &[char],
    scope: Span,
    syntax: QuoteSyntax<'_>,
    predicate: usize,
) -> bool {
    let verb = &syntax.tokens[predicate];
    let phrase = &chars[verb.span.start_char..scope.end_char];
    if ["書きなさい", "書くな", "書け", "書いて", "書かないで"]
        .iter()
        .filter_map(|ending| strip_prefix(phrase, ending))
        .any(|tail| {
            let tail = skip_space(tail);
            let tail = tail
                .strip_prefix(&['よ'])
                .or_else(|| tail.strip_prefix(&['ね']))
                .unwrap_or(tail);
            match tail.first() {
                None | Some('。' | '！' | '？' | '!' | '?') => true,
                Some('と') => {
                    let start = scope.end_char - tail.len();
                    syntax
                        .tokens
                        .binary_search_by_key(&start, |t| t.span.start_char)
                        .is_ok_and(|index| {
                            let particle = &syntax.tokens[index];
                            particle.span.end_char == start + 1 && particle.upos == Some(Upos::Adp)
                        })
                }
                _ => false,
            }
        })
    {
        return true;
    }
    let Some(head) = syntax
        .head(predicate)
        .and_then(|i| syntax.tokens.get(i))
        .filter(|head| head.span.end_char <= scope.end_char)
    else {
        return false;
    };
    matches!(head.lemma.as_deref(), Some("くださる" | "下さる"))
        || (["書くように", "書かないように"]
            .iter()
            .any(|form| strip_prefix(phrase, form).is_some())
            && matches!(head.lemma.as_deref(), Some("頼む" | "言う" | "命じる")))
}

fn is_writing_surface(
    chars: &[char],
    start: usize,
    end: usize,
    syntax: QuoteSyntax<'_>,
    copying: bool,
) -> bool {
    let source = &chars[start..end];
    // A letter or book can report a message in its own right. A sheet, board
    // or label instead supplies a surface for an inscription.
    (copying && ends_in_reading_material(chars, start, end, syntax))
        || [
            "紙",
            "用紙",
            "ノート",
            "黒板",
            "手帳",
            "申込書",
            "名札",
            "札",
            "ラベル",
            "看板",
            "案内板",
            "掲示板",
            "標示板",
            "標識",
            "ホワイトボード",
        ]
        .iter()
        .filter_map(|noun| strip_suffix(source, noun))
        .any(|prefix| {
            syntax
                .tokens
                .binary_search_by_key(&(start + prefix.len()), |t| t.span.start_char)
                .is_ok()
        })
}

fn copy_with_local_dependencies(
    chars: &[char],
    start: usize,
    end: usize,
    syntax: QuoteSyntax<'_>,
) -> bool {
    let first = syntax
        .tokens
        .partition_point(|token| token.span.start_char < start);
    let Some((predicate, verb)) = syntax.first_predicate(first, end) else {
        return false;
    };
    if verb.upos != Some(Upos::Verb)
        || !starts_copying(&chars[verb.span.start_char..verb.span.end_char])
        || chars[start..verb.span.start_char].iter().any(|&ch| {
            matches!(
                ch,
                '\n' | '。' | '！' | '？' | '!' | '?' | '「' | '」' | '『' | '』'
            )
        })
    {
        return false;
    }

    let before_predicate = first..predicate;
    !before_predicate.is_empty()
        && before_predicate.clone().all(|index| {
            let token = &syntax.tokens[index];
            let surface = &chars[token.span.start_char..token.span.end_char];
            if token.upos == Some(Upos::Punct) {
                return surface == ['、'];
            }
            surface != ['を']
                && token.sentence == verb.sentence
                && attaches_to_copy_predicate(
                    index,
                    before_predicate.clone(),
                    predicate,
                    syntax.dependencies,
                )
        })
}

fn attaches_to_copy_predicate(
    mut token: usize,
    before_predicate: std::ops::Range<usize>,
    predicate: usize,
    dependencies: &[SyntaxToken],
) -> bool {
    // Follow only local attachments. A case particle can attach through its noun,
    // and several manner modifiers can modify one another before reaching the verb.
    for _ in before_predicate.clone() {
        let Ok(index) = dependencies.binary_search_by_key(&token, |edge| edge.token) else {
            return false;
        };
        let edge = &dependencies[index];
        let Some(head) = edge.head else {
            return false;
        };
        if matches!(edge.deprel.as_str(), "csubj" | "obj" | "iobj") {
            return false;
        }
        if head == predicate {
            return matches!(edge.deprel.as_str(), "advmod" | "advcl" | "obl" | "nsubj");
        }
        if !before_predicate.contains(&head) {
            return false;
        }
        token = head;
    }
    false
}

fn after_copying_manner(chars: &[char]) -> Option<&[char]> {
    let chars = skip_space(chars);
    strip_prefix(chars, "そのまま")
        .or_else(|| after_character_count(chars).and_then(|tail| strip_prefix(tail, "ずつ")))
}

fn copy_destination_without_predicate(tokens: &[Token], start: usize, end: usize) -> bool {
    // A clause such as 「…」と言った後で紙に can contain a destination for a later verb.
    // Its earlier verb still governs the quotation.
    !tokens.iter().any(|token| {
        token.span.start_char >= start
            && token.span.end_char <= end
            && matches!(token.upos, Some(Upos::Verb | Upos::Aux | Upos::Sconj))
    })
}

fn starts_copying(chars: &[char]) -> bool {
    let chars = skip_space(chars);
    ["書き写", "書写", "転記"]
        .iter()
        .any(|verb| strip_prefix(chars, verb).is_some())
}

fn starts_writing(chars: &[char]) -> bool {
    let chars = after_clause_comma(chars);
    ["書く", "書き", "書い", "書か", "書け", "書こ"]
        .iter()
        .any(|verb| strip_prefix(chars, verb).is_some())
}

fn after_clause_comma(chars: &[char]) -> &[char] {
    let chars = skip_space(chars);
    chars.strip_prefix(&['、']).map_or(chars, skip_space)
}

fn after_count_modifiers(mut chars: &[char]) -> &[char] {
    loop {
        chars = after_clause_comma(chars);
        let scope = [
            "合計で",
            "合計",
            "全部で",
            "両方合わせて",
            "合わせて",
            "あわせて",
            "それぞれ",
            "各々",
        ]
        .iter()
        .find_map(|phrase| strip_prefix(chars, phrase));
        let Some(rest) = scope.or_else(|| after_included_written_mark(chars)) else {
            return chars;
        };
        chars = rest;
    }
}

fn after_included_written_mark(chars: &[char]) -> Option<&[char]> {
    let tail = after_written_mark(chars)?;
    if let Some(after) = ["込みで", "抜きで", "なしで"]
        .iter()
        .find_map(|marker| strip_prefix(tail, marker))
    {
        return Some(after);
    }
    let relation = written_mark_argument(tail)?;
    ["含めて", "含めると", "入れて", "入れると"]
        .iter()
        .find_map(|predicate| strip_prefix(relation, predicate))
        .or_else(|| after_mark_removal(relation))
}

fn after_written_mark(chars: &[char]) -> Option<&[char]> {
    let chars = skip_space(chars);
    [
        "感嘆符",
        "疑問符",
        "句読点",
        "句点",
        "読点",
        "記号",
        "空白",
        "スペース",
        "括弧",
    ]
    .iter()
    .find_map(|name| strip_prefix(chars, name))
    .map(skip_space)
}

fn written_mark_argument(after: &[char]) -> Option<&[char]> {
    let (&particle, relation) = after.split_first()?;
    matches!(particle, 'を' | 'も').then(|| skip_space(relation))
}

fn after_mark_removal(relation: &[char]) -> Option<&[char]> {
    [
        "引くと",
        "引いて",
        "除くと",
        "除いて",
        "抜くと",
        "抜いて",
        "数えずに",
    ]
    .iter()
    .find_map(|predicate| strip_prefix(relation, predicate))
}

fn after_word_name(nominal: &[char]) -> Option<&[char]> {
    let nominal = skip_space(nominal);
    WORD_NAMES
        .iter()
        .find_map(|word| strip_prefix(nominal, word))
        .map(skip_space)
}

fn ends_in_reading_material(
    chars: &[char],
    start: usize,
    end: usize,
    syntax: QuoteSyntax<'_>,
) -> bool {
    let text = trim_space_end(&chars[start..end]);
    // A title introduced by a genre, or the reading material itself, can refer
    // to how its contents are read. 本 inside 日本 is not a material noun.
    [
        "本",
        "絵本",
        "教科書",
        "雑誌",
        "新聞",
        "記事",
        "小説",
        "物語",
        "落語",
        "詩",
        "漫画",
        "マンガ",
        "作品",
        "書籍",
        "論文",
        "手紙",
        "資料",
        "文章",
        "説明書",
        "参考書",
        "辞書",
        "楽譜",
    ]
    .iter()
    .filter_map(|name| strip_suffix(text, name))
    .any(|prefix| {
        prefix.is_empty()
            || syntax
                .tokens
                .binary_search_by_key(&(start + prefix.len()), |t| t.span.start_char)
                .is_ok()
    })
}

fn names_source_reading(before: &[char], after: &[char], bare_word: impl FnOnce() -> bool) -> bool {
    let labels = WORD_NAMES
        .iter()
        .chain(["漢字", "仮名", "片仮名", "平仮名"].iter());
    let named_word = strip_prefix(after, "という").and_then(|nominal| {
        let nominal = skip_space(nominal);
        let rest = without_count(nominal);
        let nominal = if rest.len() < nominal.len() {
            FORM_COUNTERS
                .iter()
                .find_map(|label| strip_prefix(rest, label))?
        } else {
            nominal
        };
        labels
            .clone()
            .find_map(|word| strip_prefix(skip_space(nominal), word))
            .map(skip_space)
    });
    let explicit_word = named_word.is_some()
        || labels.clone().any(|word| {
            strip_suffix(trim_space_end(before), word).is_some_and(|prefix| {
                // 語 can also end a genre name such as 物語 or 落語 before a title.
                *word != "語"
                    || prefix.last().is_none_or(|&ch| {
                        is_horizontal_space(ch) || "はがのをにで、：".contains(ch)
                    })
            })
        });
    let relation = named_word.unwrap_or(after);
    if let Some(attribute) = relation.strip_prefix(&['の']).map(skip_space) {
        // 読み方 can also mean how to read a book or a newspaper.
        let is_reading = ["読み方", "よみかた", "読み", "よみ"]
            .iter()
            .any(|name| strip_prefix(attribute, name).is_some_and(noun_boundary))
            && (explicit_word || bare_word());
        return is_reading
            || [
                "音読み",
                "訓読み",
                "発音",
                "発音記号",
                "ふりがな",
                "振り仮名",
                "振りがな",
            ]
            .iter()
            .any(|name| strip_prefix(attribute, name).is_some_and(noun_boundary));
    }
    let Some(tail) = ["って", "は", "を", "も"]
        .iter()
        .find_map(|marker| strip_prefix(relation, marker))
    else {
        return false;
    };
    let tail = after_clause_comma(tail);
    let question = strip_prefix(tail, "日本語で").map_or(tail, skip_space);
    let pronounce = strip_prefix(question, "どう").map(skip_space);
    let verb = ["何と", "なんと", "何て", "なんて"]
        .iter()
        .find_map(|word| strip_prefix(question, word))
        .or_else(|| pronounce.filter(|rest| explicit_word || strip_prefix(rest, "発音").is_some()));
    verb.is_some_and(|verb| {
        let verb = skip_space(verb);
        [
            "読む",
            "読みます",
            "読みました",
            "読んだ",
            "発音する",
            "発音します",
            "発音しました",
            "発音した",
        ]
        .iter()
        .filter_map(|form| strip_prefix(verb, form))
        .any(|end| {
            let end = skip_space(end);
            matches!(
                end.first(),
                None | Some('。' | '？' | '?' | '」' | '』' | 'か')
            ) || strip_prefix(end, "のですか").is_some()
                || strip_prefix(end, "のか").is_some()
                || end.strip_prefix(&['の']).is_some_and(|end| {
                    matches!(
                        skip_space(end).first(),
                        None | Some('？' | '?' | '」' | '』')
                    )
                })
        })
    })
}

fn measures_written_form(after: &[char]) -> bool {
    // When the quote names a word, the case marker belongs to that word.
    // A country or another referent introduced by という is a different use.
    let after = if let Some(nominal) = strip_prefix(after, "という") {
        let Some(after) = after_word_name(nominal) else {
            return false;
        };
        after
    } else {
        after
    };
    if after
        .strip_prefix(&['の'])
        .is_some_and(names_character_count)
        || after
            .strip_prefix(&['に'])
            .is_some_and(contains_written_characters)
    {
        return true;
    }
    let Some(topic) = after
        .strip_prefix(&['は'])
        .or_else(|| after.strip_prefix(&['が']))
        .or_else(|| strip_prefix(after, "には"))
        .or_else(|| {
            strip_prefix(after, "から")
                .and_then(after_written_mark)
                .and_then(written_mark_argument)
                .and_then(after_mark_removal)
        })
    else {
        return false;
    };
    let topic = after_count_modifiers(topic);
    let amount = FORM_SCRIPTS
        .iter()
        .find_map(|script| strip_prefix(topic, script))
        .map_or(topic, |tail| {
            let tail = skip_space(tail);
            strip_prefix(tail, "では")
                .or_else(|| tail.strip_prefix(&['で']))
                .map_or(tail, skip_space)
        });
    let amount = after_count_modifiers(amount);
    if let Some(tail) = after_character_count(amount) {
        return matches!(tail.first(), None | Some('。' | '！' | '？' | '?'))
            || character_composition_predicate(tail)
            || CHARACTER_COUNT_PREDICATES.iter().any(|predicate| {
                strip_prefix(tail, predicate).is_some_and(|end| {
                    matches!(
                        skip_space(end).first(),
                        None | Some(
                            '。' | '！'
                                | '？'
                                | '?'
                                | '、'
                                | 'か'
                                | 'ね'
                                | 'よ'
                                | 'と'
                                | '」'
                                | '』'
                        )
                    )
                })
            });
    }
    if let Some(components) = ["どの", "どんな"]
        .iter()
        .find_map(|prefix| strip_prefix(amount, prefix))
    {
        return after_character_unit(components).is_some_and(character_composition_predicate);
    }
    let Some(tail) = amount.strip_prefix(&['何']).and_then(after_character_unit) else {
        return false;
    };
    character_composition_predicate(tail)
        || character_count_is_question(tail)
        || CHARACTER_COUNT_PREDICATES.iter().any(|predicate| {
            strip_prefix(tail, predicate).is_some_and(|question| {
                let question = skip_space(question);
                let particle = question.strip_prefix(&['の']).map_or(question, skip_space);
                // The measure predicate already makes か the question marker.
                // It does not depend on an enclosing verb or that verb's arguments.
                matches!(question.first(), Some('？' | '?'))
                    || particle.strip_prefix(&['か']).is_some_and(|after| {
                        // Keep から, かぎり and かもしれない outside this reading.
                        !["ら", "ぎり", "限り", "もしれ"]
                            .iter()
                            .any(|continuation| strip_prefix(after, continuation).is_some())
                    })
            })
        })
}

fn contains_written_characters(after: &[char]) -> bool {
    ["含まれる", "含まれている", "含まれた"]
        .iter()
        .find_map(|relation| strip_prefix(skip_space(after), relation))
        .and_then(after_character_unit)
        .is_some_and(noun_boundary)
}

fn character_composition_predicate(after: &[char]) -> bool {
    let Some(predicate) = after
        .strip_prefix(&['で'])
        .or_else(|| strip_prefix(after, "から"))
        .map(skip_space)
    else {
        return false;
    };
    ["できて", "できた", "出来て", "出来た", "構成され", "成り立"]
        .iter()
        .any(|form| strip_prefix(predicate, form).is_some())
}

fn character_count_is_question(after: &[char]) -> bool {
    let after = skip_space(after);
    if matches!(after.first(), Some('？' | '?')) {
        return true;
    }
    let Some(after) = after.strip_prefix(&['か']) else {
        return false;
    };
    let after = skip_space(after);
    if matches!(
        after.first(),
        None | Some('。' | '？' | '?' | '、' | 'ね' | 'と')
    ) {
        return true;
    }
    // Without a measure predicate, a local checking or asking verb distinguishes
    // the question marker か from the indefinite amount in 何文字か.
    let after = after.strip_prefix(&['を']).map_or(after, skip_space);
    ["調べ", "確かめ", "確認", "尋ね", "聞い", "聞く", "聞き"]
        .iter()
        .any(|verb| strip_prefix(after, verb).is_some())
}

fn after_character_count(chars: &[char]) -> Option<&[char]> {
    let chars = skip_space(chars);
    let number = FORM_SCRIPTS
        .iter()
        .find_map(|script| strip_prefix(chars, script))
        .map_or(chars, skip_space);
    let tail = without_count(number);
    if tail.len() == number.len() {
        return None;
    }
    after_character_unit(tail)
}

fn after_character_unit(chars: &[char]) -> Option<&[char]> {
    let chars = skip_space(chars);
    ["文字", "字"]
        .iter()
        .find_map(|unit| strip_prefix(chars, unit))
        .map(skip_space)
}

fn strip_prefix<'a>(mut chars: &'a [char], prefix: &str) -> Option<&'a [char]> {
    for expected in prefix.chars() {
        let (&actual, rest) = chars.split_first()?;
        if actual != expected {
            return None;
        }
        chars = rest;
    }
    Some(chars)
}

fn strip_suffix<'a>(mut chars: &'a [char], suffix: &str) -> Option<&'a [char]> {
    for expected in suffix.chars().rev() {
        let (&actual, rest) = chars.split_last()?;
        if actual != expected {
            return None;
        }
        chars = rest;
    }
    Some(chars)
}

fn skip_space(chars: &[char]) -> &[char] {
    let count = chars
        .iter()
        .take_while(|&&ch| is_horizontal_space(ch))
        .count();
    &chars[count..]
}

fn trim_space_end(chars: &[char]) -> &[char] {
    let end = chars
        .iter()
        .rposition(|&ch| !is_horizontal_space(ch))
        .map_or(0, |index| index + 1);
    &chars[..end]
}

fn is_horizontal_space(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '　')
}

fn without_count(chars: &[char]) -> &[char] {
    let count = chars
        .iter()
        .take_while(|&&ch| is_count_character(ch))
        .count();
    skip_space(&chars[count..])
}

fn is_count_character(ch: char) -> bool {
    ch.is_ascii_digit()
        || ('０'..='９').contains(&ch)
        || "〇零一二三四五六七八九十百千万".contains(ch)
}

fn noun_boundary(chars: &[char]) -> bool {
    skip_space(chars)
        .first()
        .is_none_or(|&ch| "はがをにとでのもへだ、。！？」「『』".contains(ch))
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::model::Backend;

    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslateUnknownKind, TranslationGlossary,
        analyze_and_translate,
    };

    fn glossary() -> TranslationGlossary {
        TranslationGlossary {
            places: vec![GlossaryEntry {
                canonical: "ドイツ".to_owned(),
                surfaces: vec!["ドイツ".to_owned()],
                targets: [(TargetVariant::ZhHant, "德國".to_owned())].into(),
                pronoun_class: Default::default(),
                speech_register: Default::default(),
            }],
            ..TranslationGlossary::default()
        }
    }

    #[test]
    fn protects_the_source_operand_of_translation() {
        for (source, kept) in [
            (
                "「ドイツへ行きますか？」を英語に翻訳できますか。ドイツへ行きます。",
                true,
            ),
            (
                "「ドイツ」という語をイタリア語に翻訳できますか。ドイツへ行きます。",
                true,
            ),
            (
                "「ドイツ」を英語ではなくフランス語に訳しました。ドイツへ行きます。",
                true,
            ),
            ("「ドイツへ行きますか？」を見て英語に訳しました。", false),
            (
                "「ドイツ」だけを英語に訳してください。ドイツへ行きます。",
                true,
            ),
            ("「ドイツ」も英語に直訳しました。ドイツへ行きます。", true),
            ("「ドイツ」を英語に訳してください。ドイツへ行きます。", true),
            ("英語に「ドイツ」を翻訳しました。ドイツへ行きます。", true),
            (
                "「ドイツ」という言葉を英語に訳しました。ドイツへ行きます。",
                true,
            ),
            (
                "「ドイツ」と「写真」を英語に翻訳しました。ドイツへ行きます。",
                true,
            ),
            ("「ドイツ」を見て、「写真」を訳しました。", false),
            ("英語を「ドイツ」と訳しました。", false),
            ("「ドイツ」と英語で言いました。", false),
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(translated.target_text.contains("ドイツ"), kept, "{source}");
            assert!(translated.target_text.contains("德國"), "{source}");
            assert_eq!(translated.glossary_hits.len(), 1, "{source}");
        }
    }

    #[test]
    fn preserves_the_characters_that_the_sentence_describes() {
        let glossary = glossary();
        for source in [
            "私は「ドイツ」という三文字を丸で囲みました。",
            "私は「ドイツ」の最後の文字を消しました。",
            "『ドイツ』はカタカナ三文字です。",
            "「ドイツ」の表記をノートに書きました。",
            "彼は「ドイツ」を三文字で書きました。",
            "彼は「ドイツ」をカタカナ３文字で書いた。",
            "彼は「ドイツ」を三文字では書きませんでした。",
            "彼は「ドイツ」を、三文字では書きませんでした。",
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert!(translated.target_text.contains("ドイツ"), "{source}");
            assert!(translated.glossary_hits.is_empty(), "{source}");
        }
    }

    #[test]
    fn written_form_does_not_lock_a_separate_location_or_reply() {
        let glossary = glossary();
        for (source, preserved) in [
            ("出身国を聞かれて、「ドイツです」と答えました。", false),
            ("私は「ドイツ」という国を地図で見ました。", false),
            ("「ドイツ」という国に行きました。", false),
            (
                "「ドイツ」という国について書いた記事は何文字ですか。",
                false,
            ),
            (
                "「ドイツ」という単語を聞いたあと、別の文に含まれる文字を数えました。",
                false,
            ),
            ("「ドイツ」という国の案内を一字ずつ読みました。", false),
            ("「ドイツ」はどこにありますか。", false),
            (
                "「ドイツ」という言葉をニュースで聞きました。「来て」は何文字ですか。",
                false,
            ),
            ("彼は「ドイツ」を三秒で書きました。", false),
            ("彼は三秒で「ドイツ」を書きました。", false),
            ("彼は三日で「ドイツ」を回りました。", false),
            ("彼は三文字で「ドイツ」を書き、ドイツに行きました。", true),
            (
                "彼は「ドイツ」を書き、その隣に三文字の番号を添えました。",
                false,
            ),
            (
                "「ドイツ！」と答えてから、「来て？」の字数を調べました。",
                false,
            ),
            (
                "私は「ドイツ」という三文字を丸で囲み、「ドイツへ行きます」と言いました。",
                true,
            ),
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert!(translated.target_text.contains("德國"), "{source}");
            assert_eq!(
                translated.target_text.contains("ドイツ"),
                preserved,
                "{source}"
            );
            assert_eq!(translated.glossary_hits.len(), 1, "{source}");
        }
    }

    #[test]
    fn preserves_only_the_forms_whose_characters_are_counted() {
        let glossary = glossary();
        for (source, raw_quote) in [
            (
                "ドイツで「ドイツ」という単語の文字数を調べました。",
                "「ドイツ」",
            ),
            (
                "ドイツで「来て」という語が何文字あるか調べました。",
                "「来て」",
            ),
            (
                "ドイツへ行く前に、「ドイツ」という言葉は何文字か調べました。",
                "「ドイツ」",
            ),
            (
                "ドイツで「行く」という単語が何文字かを確認しました。",
                "「行く」",
            ),
            (
                "ドイツで「来て」という語が何文字あるのか調べました。",
                "「来て」",
            ),
            (
                "ドイツで「来て」という語が何文字あるか先生に尋ねました。",
                "「来て」",
            ),
        ] {
            let result = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(result.target_text.matches(raw_quote).count(), 1, "{source}");
            assert_eq!(result.target_text.matches("德國").count(), 1, "{source}");
            assert_eq!(result.glossary_hits.len(), 1, "{source}");
        }

        let source = "ドイツで「ドイツ」という国について書いた文が何文字あるか調べました。";
        let result = analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert!(!result.target_text.contains("ドイツ"), "{source}");
        assert_eq!(result.target_text.matches("德國").count(), 2, "{source}");
    }

    #[test]
    fn preserves_forms_with_an_explicit_written_attribute() {
        let glossary = glossary();
        for (source, written) in [
            ("「来て」という言葉は二文字です。", &["「来て」"][..]),
            (
                "「ドイツ」という単語は何文字でできていますか。",
                &["「ドイツ」"],
            ),
            (
                "「ドイツ」という単語に含まれる文字を数えました。",
                &["「ドイツ」"],
            ),
            (
                "「ドイツ」という単語はどの文字からできていますか。",
                &["「ドイツ」"],
            ),
            (
                "「ドイツ」と「来て」は合計で五文字です。",
                &["「ドイツ」", "「来て」"],
            ),
            ("「来て！」は感嘆符を含めて何文字ですか。", &["「来て！」"]),
            (
                "「ドイツ」と「来て」は合わせて何文字ですか。",
                &["「ドイツ」", "「来て」"],
            ),
            (
                "「ドイツ」と「来て」はそれぞれ何文字ですか。",
                &["「ドイツ」", "「来て」"],
            ),
            (
                "「ドイツ！」と「来て？」はそれぞれ、句読点も含めて何文字ですか。",
                &["「ドイツ！」", "「来て？」"],
            ),
            (
                "「ドイツ」という文字列をそのまま写しました。",
                &["「ドイツ」"],
            ),
            ("「ドイツ」を一字ずつ読みました。", &["「ドイツ」"]),
            ("「ドイツ」を一字一字読みました。", &["「ドイツ」"]),
            (
                "「来て！」は記号を含めると何文字になりますか。",
                &["「来て！」"],
            ),
            ("「来て！」は感嘆符込みで何文字ですか。", &["「来て！」"]),
            (
                "「来て！」から感嘆符を引くと何文字になりますか。",
                &["「来て！」"],
            ),
            (
                "「来て！」は感嘆符を数えずに何文字ですか。",
                &["「来て！」"],
            ),
        ] {
            let result = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            for &original in written {
                assert_eq!(result.target_text.matches(original).count(), 1, "{source}");
            }
            assert!(result.glossary_hits.is_empty(), "{source}");
        }
    }

    #[test]
    fn copies_the_quoted_text_into_a_translated_form() {
        let mut glossary = glossary();
        for (source, target) in [("申込書", "申請表"), ("入居申込書", "入住申請表")]
        {
            glossary.objects.push(GlossaryEntry {
                canonical: source.to_owned(),
                surfaces: vec![source.to_owned()],
                targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
                pronoun_class: Default::default(),
                speech_register: Default::default(),
            });
        }
        for (source, expected) in [
            (
                "「ドイツ」という単語を申込書に書き写しました。",
                &["「ドイツ」", "申請表"][..],
            ),
            ("「申込書」と書き写しました。", &["「申込書」"]),
            (
                "申込書に「申込書」と書き写しました。",
                &["「申込書」", "申請表"],
            ),
            (
                "『入居申込書』と紙に書き写しました。",
                &["『入居申込書』", "紙"],
            ),
            (
                "『入居申込書』と紙に一字ずつ書き写しました。",
                &["『入居申込書』", "紙"],
            ),
            (
                "『ドイツ』と『入居申込書』を紙に書き写しました。",
                &["『ドイツ』", "『入居申込書』", "紙"],
            ),
            (
                "紙に『入居申込書』と『ドイツ』を書き写しました。",
                &["『入居申込書』", "『ドイツ』", "紙"],
            ),
            (
                "『来て！』と『待って？』をノートにそのまま書き写しました。",
                &["『来て！』", "『待って？』", "筆記本"],
            ),
            (
                "『ドイツ』、『来て！』と『入居申込書』を紙に転記しました。",
                &["『ドイツ』", "『来て！』", "『入居申込書』"],
            ),
            (
                "父は『ドイツ』と言い、私は『入居申込書』を紙に書き写しました。",
                &["『德國』", "『入居申込書』", "紙"],
            ),
            (
                "私は四時に「七月八日の午後二時には」とノートにそのまま書き写しました。",
                &["四點", "「七月八日の午後二時には」", "筆記本"],
            ),
            (
                "私は四時に「入居申込書」とノートに、書き写しました。",
                &["四點", "「入居申込書」", "筆記本"],
            ),
            (
                "私は四時に「入居申込書」とノートに、そのまま書き写しました。",
                &["四點", "「入居申込書」", "筆記本"],
            ),
            (
                "『来て』を紙に、丁寧に書き写しました。",
                &["『来て』", "紙"],
            ),
            (
                "『来て』を紙に、小さく書き写しました。",
                &["『来て』", "紙"],
            ),
            (
                "『入居申込書』を紙に、係員が書き写しました。",
                &["『入居申込書』", "紙", "工作人員"],
            ),
            (
                "『入居申込書』を紙に係員が書き写しました。",
                &["『入居申込書』", "紙", "工作人員"],
            ),
            (
                "「帰る」を紙に、ゆっくり正確に書き写しました。",
                &["「帰る」", "紙"],
            ),
            (
                "「来て！」を紙に、丁寧に転記しました。",
                &["「来て！」", "紙"],
            ),
            (
                "『ドイツ』と『入居申込書』を紙に、丁寧に書き写しました。",
                &["『ドイツ』", "『入居申込書』", "紙"],
            ),
            (
                "『ドイツ』と『入居申込書』を紙に、書き写しました。",
                &["『ドイツ』", "『入居申込書』", "紙"],
            ),
            (
                "『ドイツ』と『入居申込書』を紙に、そのまま書き写しました。",
                &["『ドイツ』", "『入居申込書』", "紙"],
            ),
            (
                "紙に、『ドイツ』と『入居申込書』を書き写しました。",
                &["『ドイツ』", "『入居申込書』", "紙"],
            ),
            (
                "『入居申込書』という語を紙にそのまま書き写しました。",
                &["『入居申込書』", "紙"],
            ),
            ("「申込書」とそのまま書き写しました。", &["「申込書」"]),
            ("入居申込書を受け取りました。", &["入住申請表"]),
            (
                "係員は「入居申込書は机の上です」と言いました。",
                &["入住申請表", "說"],
            ),
            (
                "窓口で『入居申込書』と担当者にそのまま伝えました。",
                &["入住申請表"],
            ),
            (
                "『入居申込書』と言った後で紙にそのまま書き写しました。",
                &["入住申請表", "紙"],
            ),
            (
                "父は「入居申込書」と言った後でノートに、そのまま書き写しました。",
                &["入住申請表", "筆記本"],
            ),
            (
                "「来て」と丁寧に伝えてから紙に「行く」を小さく書き写しました。",
                &["「來」", "「行く」", "紙"],
            ),
            (
                "「来て」と言い、紙に宛名をゆっくり書き写しました。",
                &["「來」", "宛名", "紙"],
            ),
            (
                "4月2日に、申込書に『来て！』と書き写しました。",
                &["4月2日", "『来て！』", "申請表"],
            ),
            (
                "4月2日に、紙に『来て！』と書き写しました。",
                &["4月2日", "『来て！』", "紙"],
            ),
            (
                "「ドイツ」という国に行くため、申込書に住所を書き写しました。",
                &["德國", "申請表", "住所"],
            ),
        ] {
            let result = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            for &term in expected {
                assert_eq!(
                    result.target_text.matches(term).count(),
                    1,
                    "{source}: {}",
                    result.target_text
                );
            }
        }
    }

    #[test]
    fn preserves_only_the_quotation_being_written_verbatim() {
        let mut glossary = glossary();
        glossary.objects.push(GlossaryEntry {
            canonical: "写真".to_owned(),
            surfaces: vec!["写真".to_owned()],
            targets: [(TargetVariant::ZhHant, "相片".to_owned())].into(),
            pronoun_class: Default::default(),
            speech_register: Default::default(),
        });
        for (source, fragments) in [
            ("紙に「写真」とそのまま書きました。", &["「写真」"][..]),
            ("「写真」とノートにそのまま書きました。", &["「写真」"]),
            ("私はそのまま「写真」と書きました。", &["「写真」"]),
            ("「写真」と書いてください。", &["「写真」"]),
            (
                "先生は「三日間」とそのまま書きました。父は三日間家に帰りませんでした。",
                &["「三日間」", "三天沒有回到家"],
            ),
            (
                "「写真」と『ドイツ』と書いてください。",
                &["「写真」", "『ドイツ』"],
            ),
            ("「写真」と書かないでください。", &["「写真」"]),
            ("紙に「写真」と書きなさい。", &["「写真」"]),
            ("紙に「写真」と書け。", &["「写真」"]),
            ("先生は紙に「写真」と書くなと言いました。", &["「写真」"]),
            ("紙に「写真」と書くように言われました。", &["「写真」"]),
            ("紙に「写真」と書けます。", &["相片"]),
            ("紙に「写真」と書けばいい。", &["相片"]),
            ("「写真」と書いたように見えます。", &["相片"]),
            ("「写真」と書いてよかったです。", &["相片"]),
            ("日本語で「写真」と書いてください。", &["「写真」"]),
            ("「写真」と黒板に書いて下さい。", &["「写真」"]),
            ("説明書の「写真」をそのまま写しました。", &["「写真」"]),
            (
                "教室で説明書の「写真」をそのまま写しました。",
                &["「写真」"],
            ),
            (
                "辞書の「写真」をノートにそのまま写しました。",
                &["「写真」"],
            ),
            ("カメラで説明書の「写真」をそのまま写しました。", &["相片"]),
            ("英語で「写真」と書いてください。", &["相片"]),
            ("「写真」と英語で書いてください。", &["相片"]),
            ("「写真を買った」と書いてください。", &["相片"]),
            ("「写真」と言った後で住所をそのまま書きました。", &["相片"]),
            (
                "父は「写真」と言い、母は「写真」とそのまま書きました。",
                &["相片", "「写真」"],
            ),
            (
                "先生は「紙に『写真』とそのまま書いてください」と言いました。",
                &["『写真』"],
            ),
        ] {
            let result = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            for &fragment in fragments {
                assert_eq!(
                    result.target_text.matches(fragment).count(),
                    1,
                    "{source}: {}",
                    result.target_text
                );
            }
        }
    }

    #[test]
    fn writing_on_a_surface_distinguishes_an_inscription_from_a_report() {
        for (source, fragments) in [
            (
                "紙に「写真」と書きました。写真を買いました。",
                &["「写真」", "照片"][..],
            ),
            (
                "私は紙に「写真」と書きませんでした。写真を買いました。",
                &["「写真」", "照片"],
            ),
            ("「写真」と紙に書かなかった。", &["「写真」"]),
            ("黒板に「走る」と書いた。", &["「走る」"]),
            ("「赤い」とノートに書きます。", &["「赤い」"]),
            ("紙に「食べない」と書きました。", &["「食べない」"]),
            (
                "妹は十分間本を読んでから、ノートに「写真」と写しました。",
                &["十分鐘", "「写真」"],
            ),
            ("私は「駅」と紙に写しませんでした。", &["「駅」", "沒有"]),
            (
                "先生は紙に「駅」と書いてから、朝まで椅子に座っていました。",
                &["「駅」", "早上"],
            ),
            (
                "ノートに「写真」と書いていませんでした。",
                &["「写真」", "沒有"],
            ),
            ("紙に「赤い」と英語で書いてから、席を立ちました。", &["紅"]),
            ("教科書の「走る」をノートに写しました。", &["「走る」"]),
            (
                "先生はノートに「駅はどこですか？」とそのまま写しました。",
                &["「駅はどこですか？」"],
            ),
            (
                "黒板の「駅はどこですか？」をノートに写しました。",
                &["「駅はどこですか？」"],
            ),
            ("ノートに「駅はどこですか？」と書きました。", &["車站"]),
            ("紙に「静かに」と書きました。", &["「静かに」"]),
            ("紙に「丁寧に」と書きました。", &["「丁寧に」"]),
            (
                "ノートに「十分に」と書いてから、田中さんに本を送りました。",
                &["「十分に」", "給田中"],
            ),
            ("紙に「十分に休んだ」と書きました。", &["休息"]),
            (
                "黒板の「駅はどこですか？」をカメラで写しました。",
                &["車站"],
            ),
            ("妹はカメラでノートの「写真」を写しました。", &["照片"]),
            (
                "ノートに「写真」と言った後で、別の写真を写しました。",
                &["「照片」"],
            ),
            ("母への手紙に「帰った」と書きました。", &["回去"]),
            ("母への手紙に「帰る」と書きました。", &["回去"]),
            (
                "母への手紙に「帰った」とそのまま書きました。",
                &["「帰った」"],
            ),
            ("紙に「青い車」と書きました。", &["「青い車」"]),
            ("紙に「三日間」と書きました。", &["「三日間」"]),
            ("紙に「写真を買った」と書きました。", &["照片"]),
            ("紙に「先生が写真を買いました」と書きました。", &["照片"]),
            ("紙を見て、「写真」と言いました。", &["照片"]),
            ("紙に「写真」と英語で書きました。", &["照片"]),
            ("英語で紙に「赤い」と書きました。", &["紅"]),
            ("紙に英語で「帰る」と書きました。", &["回去"]),
            ("日本語で紙に「赤い」と書きました。", &["「赤い」"]),
            (
                "ドイツは紙に「ドイツ」と書きました。",
                &["「ドイツ」", "德國"],
            ),
        ] {
            let result = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            for &frag in fragments {
                assert_eq!(
                    result.target_text.matches(frag).count(),
                    1,
                    "{source}: {}",
                    result.target_text
                );
            }
        }
    }

    #[test]
    fn written_signs_distinguish_labels_from_quoted_messages() {
        for (source, quoted) in [
            ("看板に「写真」と書いてあります。", "「写真」"),
            ("名札に「写真」と書いてありました。", "「写真」"),
            ("この札には「写真」と書いてありますか。", "「写真」"),
            ("看板には「写真」と書かれていましたか。", "「写真」"),
            ("札に「写真」と書いてあるかどうか確かめました。", "「写真」"),
            ("看板には「写真」とは書いてありませんでした。", "「写真」"),
            ("札に「写真」と書かれていました。", "「写真」"),
            ("案内板には「写真」とあります。", "「写真」"),
            ("案内板には「写真」とありますか？", "「写真」"),
            ("「写真」と書いてある札を母に渡しました。", "「写真」"),
            ("「写真」と書いてある古い札を母に渡しました。", "「写真」"),
            ("「写真」と書いた札を母に渡しました。", "「写真」"),
            ("私は「写真」と書かれた案内板を見ました。", "「写真」"),
            ("「写真」と言ってから、看板を見ました。", "「照片」"),
            ("「写真」と書いてから、札を渡しました。", "「照片」"),
            ("看板に「写真」と英語で書いてあります。", "「照片」"),
            (
                "英語だけで書かれた札には「写真」と書いてあります。",
                "「照片」",
            ),
            (
                "英語だけで書かれている札には「写真」と書いてあります。",
                "「照片」",
            ),
            (
                "英語だけで書いてある札には「写真」と書いてあります。",
                "「照片」",
            ),
            ("英語だけの案内板には「写真」とあります。", "「照片」"),
            (
                "英語で書かれた札には、日本語で「写真」と書いてあります。",
                "「写真」",
            ),
            (
                "英語で書かれた札を閉じ、看板には「写真」と書いてあります。",
                "「写真」",
            ),
            (
                "「写真」という意味の英語だけで書かれた札を見ました。",
                "「照片」",
            ),
            ("紙に「写真」という言葉を書きました。", "「写真」"),
            (
                "紙に「写真」という言葉は書いてありませんでした。",
                "「写真」",
            ),
            ("看板に「写真を買いました」と書いてありました。", "照片"),
            ("看板の前で「写真」と言いました。", "「照片」"),
        ] {
            let result = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert!(
                result.target_text.contains(quoted),
                "{source}: {}",
                result.target_text
            );
        }
    }

    #[test]
    fn a_glossary_only_replaces_the_country_outside_the_measured_word() {
        let mut glossary = glossary();
        glossary.places[0]
            .targets
            .insert(TargetVariant::ZhHant, "柏林代號".to_owned());
        let source = "ドイツで「ドイツ」という単語が何文字あるか忘れました。";
        let result = analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_eq!(result.target_text.matches("「ドイツ」").count(), 1);
        assert_eq!(result.target_text.matches("柏林代號").count(), 1);
        assert_eq!(result.glossary_hits.len(), 1);
    }

    #[test]
    fn preserves_counted_source_with_terminal_punctuation_inside_the_quote() {
        for source in [
            "「昼までに！」は五文字です。",
            "「昼までに！」と書いてあるメモの文字数を数えました。",
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &TranslationGlossary::default(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert!(translated.target_text.contains("昼までに！"), "{source}");
        }
    }

    #[test]
    fn preserves_both_source_forms_in_an_explicit_comparison() {
        let mut glossary = glossary();
        for (source, target) in [("独逸", "獨逸"), ("どいつ", "哪個人"), ("フランス", "法國")]
        {
            glossary.places.push(GlossaryEntry {
                canonical: source.to_owned(),
                surfaces: vec![source.to_owned()],
                targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
                pronoun_class: Default::default(),
                speech_register: Default::default(),
            });
        }
        for (source, other) in [
            ("日本語の「ドイツ」と漢字の「独逸」を比べました。", "独逸"),
            ("日本語の「ドイツ」と、漢字の「独逸」を比べました。", "独逸"),
            ("「ドイツ」と『どいつ』は表記が違います。", "どいつ"),
            ("「ドイツ」と、『どいつ』は表記が違います。", "どいつ"),
            ("『ドイツ』と『フランス』の表記を見比べました。", "フランス"),
            ("二つの表記「ドイツ」と「独逸」を比べました。", "独逸"),
            ("「ドイツ」と「独逸」という二つの表記を比べました。", "独逸"),
            ("「ドイツ」と「独逸」、二つの表記を比べました。", "独逸"),
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert!(translated.target_text.contains("ドイツ"), "{source}");
            assert!(translated.target_text.contains(other), "{source}");
            assert!(translated.glossary_hits.is_empty(), "{source}");
        }

        for source in [
            "「ドイツ」と「フランス」を比べました。",
            "「ドイツ」と「フランス」の人口を比べました。",
            "「ドイツ」と「フランス」を比べるために旅行しました。",
            "太郎は「ドイツ」と答え、花子は「フランス」と答えました。",
            "「ドイツ」と、「フランス」の人口を比べました。",
            "太郎は「ドイツ」と、花子は「フランス」と答えました。",
            "今日は「ドイツ」と、「フランス」について話しました。次に地図の表記を比べました。",
            "二つの国「ドイツ」と「フランス」を比べました。",
            "表記を確認したあと、「ドイツ」と「フランス」の人口を比べました。",
            "三つの表記「ドイツ」と「フランス」を比べました。",
            "十二の表記「ドイツ」と「フランス」を比べました。",
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert!(translated.target_text.contains("德國"), "{source}");
            assert!(translated.target_text.contains("法國"), "{source}");
            assert_eq!(translated.glossary_hits.len(), 2, "{source}");
        }
    }

    #[test]
    fn nested_written_forms_stay_inside_the_outer_quote_and_alignment() {
        let glossary = glossary();
        for (source, written) in [
            (
                "先生は『「ドイツ」という三文字を書きなさい』と言いました。",
                "「ドイツ」",
            ),
            ("父は『「駅」の文字を消しなさい』と言いました。", "「駅」"),
            (
                "先生は『「行く」という単語は何文字ですか』と聞きました。",
                "「行く」",
            ),
            (
                "先生は『「来て！」という語が何文字あるか教えて』と言いました。",
                "「来て！」",
            ),
            (
                "先生は『「来て！」という言葉は三文字です』と説明しました。",
                "「来て！」",
            ),
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            let text = &translated.target_text;
            let quote = text.find(written).unwrap_or_else(|| panic!("{text}"));
            assert_eq!(text.matches(written).count(), 1, "{text}");
            assert!(
                text.find('『').unwrap() < quote && quote < text.find('』').unwrap(),
                "{text}"
            );
            let start = text[..quote].chars().count();
            let end = start + written.chars().count();
            assert!(
                translated.clause_alignments.iter().any(|clause| {
                    clause.target_span.start_char <= start && end <= clause.target_span.end_char
                }),
                "{text}"
            );
            assert!(
                !translated
                    .unknown
                    .iter()
                    .any(|item| { item.kind == TranslateUnknownKind::PlaceholderRepaired }),
                "{text}"
            );
        }
    }

    #[test]
    fn a_word_reading_preserves_only_the_forms_being_pronounced() {
        let glossary = glossary();
        for (source, keep, translate) in [
            (
                "「写真」という語の読み方を教えてください。",
                &["「写真」"][..],
                &[][..],
            ),
            ("「駅」は何と読みますか。", &["「駅」"], &[]),
            ("「駅」って何て読むの？", &["「駅」"], &[]),
            ("「駅」の音読みと訓読みを教えてください。", &["「駅」"], &[]),
            ("「ドイツ」をどう発音しますか。", &["「ドイツ」"], &[]),
            (
                "「ありがとう」、「駅」という漢字の読み方を調べました。",
                &["「駅」"],
                &["謝謝"],
            ),
            (
                "二つの単語「写真」、「駅」の読み方を調べました。",
                &["「写真」", "「駅」"],
                &[],
            ),
            (
                "「写真」と「ドイツ」という二つの単語の読み方を調べました。",
                &["「写真」", "「ドイツ」"],
                &[],
            ),
            (
                "先生は『「写真」という言葉の発音を聞いてください』と言いました。",
                &["「写真」"],
                &["老師"],
            ),
            (
                "ドイツで「ドイツ」という語の読み方を聞きました。",
                &["「ドイツ」"],
                &["德國"],
            ),
            ("「写真」という語の意味を教えてください。", &[], &["照片"]),
            ("「新聞」の読み方を教えてください。", &[], &["報紙"]),
            ("物語「新聞」の読み方を教えてください。", &[], &["報紙"]),
            ("小説「写真」の読み方を教えてください。", &[], &["照片"]),
            (
                "日本語の授業で「ドイツ」の読み方を教えてください。",
                &["「ドイツ」"],
                &[],
            ),
            (
                "「ドイツ」と「申込書」の読み方を教えてください。",
                &["「ドイツ」", "「申込書」"],
                &[],
            ),
            (
                "「写真」と言ってから「駅」という語のよみかたを聞きました。",
                &["「駅」"],
                &["照片"],
            ),
        ] {
            let result = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            for expected in keep.iter().chain(translate) {
                assert_eq!(
                    result.target_text.matches(expected).count(),
                    1,
                    "{source}: {}",
                    result.target_text
                );
            }
            assert_eq!(
                result.glossary_hits.len(),
                usize::from(translate.contains(&"德國")),
                "{source}"
            );
        }
    }

    #[test]
    fn reading_inside_reported_speech_keeps_glossary_hits_on_outer_occurrences() {
        let mut glossary = TranslationGlossary::default();
        for (surface, target) in [("写真", "彩色照片"), ("駅", "中央車站")] {
            glossary.domain_terms.push(GlossaryEntry {
                canonical: surface.to_owned(),
                surfaces: vec![surface.to_owned()],
                targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
                pronoun_class: Default::default(),
                speech_register: Default::default(),
            });
        }
        let source = "先生は学生に『「写真」と「駅」の読み方を調べてください』と言い、そのあと駅で写真を見せました。";
        let result = analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap();
        let (speech, after) = result.target_text.split_once('』').unwrap();
        assert!(
            speech.contains("「写真」") && speech.contains("「駅」"),
            "{}",
            result.target_text
        );
        assert!(!speech.contains("彩色照片") && !speech.contains("中央車站"));
        assert!(
            after.contains("彩色照片") && after.contains("中央車站"),
            "{after}"
        );
        let boundary = source.chars().position(|c| c == '』').unwrap();
        assert_eq!(result.glossary_hits.len(), 2);
        assert!(
            result
                .glossary_hits
                .iter()
                .all(|hit| hit.source_span.start_char > boundary)
        );
    }

    #[test]
    fn a_character_count_question_keeps_the_word_being_measured() {
        let glossary = glossary();
        for (source, raw_quote) in [
            ("「来て」という言葉は何文字ですか。", "「来て」"),
            ("「来て」という言葉は何文字あるでしょうか。", "「来て」"),
            ("「行く」という単語は何文字ですか。", "「行く」"),
            ("「ドイツ」という言葉は何文字ですか。", "「ドイツ」"),
            ("「ドイツ」はカタカナで何文字ですか。", "「ドイツ」"),
            ("「ドイツ」は、カタカナでは何文字？", "「ドイツ」"),
            (
                "「どいつ」という語は、ひらがなで何文字ですか。",
                "「どいつ」",
            ),
            ("「来て」には何字ありますか。", "「来て」"),
            (
                "「ドイツ」という単語が何文字あるか忘れました。",
                "「ドイツ」",
            ),
            (
                "「ドイツ」という単語が何文字あるか聞きました。",
                "「ドイツ」",
            ),
            (
                "「ドイツ」という単語が何文字あるのか知っています。",
                "「ドイツ」",
            ),
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(
                translated.target_text.matches(raw_quote).count(),
                1,
                "{source}: {}",
                translated.target_text,
            );
            assert!(translated.glossary_hits.is_empty(), "{source}");
        }
    }

    #[test]
    fn an_indefinite_quantity_or_causal_particle_is_not_a_count_question() {
        for suffix in [
            "という語が何文字かを書きました。",
            "という語が何文字かの例文を示しました。",
            "という語が何文字ですから書きました。",
            "という語が何文字あるかぎり残します。",
            "という語が何文字あるかもしれません。",
        ] {
            let chars: Vec<_> = suffix.chars().collect();
            assert!(!super::measures_written_form(&chars), "{suffix}");
        }
    }

    #[test]
    fn counting_retains_complete_written_forms_across_analysis_sentences() {
        let glossary = glossary();
        for (source, written) in [
            (
                "「来て！すぐ帰る」の文字数を数えます。",
                &["「来て！すぐ帰る」"][..],
            ),
            ("「A!ドイツ？B」の文字数を数えます。", &["「A!ドイツ？B」"]),
            ("「来て！帰る」の字数を数えました。", &["「来て！帰る」"]),
            ("「来て！帰る」の字数を調べました。", &["「来て！帰る」"]),
            (
                "「ドイツ！」と「来て？」の合計の文字数を調べました。",
                &["「ドイツ！」", "「来て？」"],
            ),
            (
                "「ドイツ！」と「来て？」の総文字数を調べました。",
                &["「ドイツ！」", "「来て？」"],
            ),
            (
                "「ドイツ！」と「来て？」は合計で何文字ですか。",
                &["「ドイツ！」", "「来て？」"],
            ),
            (
                "「ドイツ！」と「来て？」は、合計でカタカナ何文字ですか。",
                &["「ドイツ！」", "「来て？」"],
            ),
            (
                "「ドイツ」と「来て」という言葉は、合計で何文字あるでしょうか。",
                &["「ドイツ」", "「来て」"],
            ),
            (
                "「ドイツ！」と「行く？」の文字数をそれぞれ数えました。",
                &["「ドイツ！」", "「行く？」"],
            ),
            (
                "「ドイツ！」と、「行く？」の文字数をそれぞれ数えました。",
                &["「ドイツ！」", "「行く？」"],
            ),
            (
                "「行く？」と、「ドイツ！」の文字数をそれぞれ数えました。",
                &["「行く？」", "「ドイツ！」"],
            ),
            (
                "「来て！」と、「すぐ帰る」の文字数をそれぞれ数えました。",
                &["「来て！」", "「すぐ帰る」"],
            ),
            (
                "「ドイツ！」、「行く？」、「払える！」の文字数をそれぞれ数えました。",
                &["「ドイツ！」", "「行く？」", "「払える！」"],
            ),
            (
                "「ドイツ！」、「行く？」の字数をそれぞれ調べました。",
                &["「ドイツ！」", "「行く？」"],
            ),
            (
                "「行く？」と、「ドイツ！」の文字数をそれぞれ調べました。",
                &["「行く？」", "「ドイツ！」"],
            ),
            (
                "「ドイツ！」と「来て？」の字数を調べました。",
                &["「ドイツ！」", "「来て？」"],
            ),
            (
                "「来て？」と「ドイツ！」の文字数を数えてください。",
                &["「来て？」", "「ドイツ！」"],
            ),
            (
                "「ドイツ！」と「来て？」の字数は違います。",
                &["「ドイツ！」", "「来て？」"],
            ),
            (
                "「猫！」、「犬？」の字数を調べました。",
                &["「猫！」", "「犬？」"],
            ),
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            for original in written {
                assert!(
                    translated.target_text.contains(original),
                    "{}",
                    translated.target_text
                );
            }
            assert!(translated.glossary_hits.is_empty(), "{source}");
            assert!(
                !translated
                    .unknown
                    .iter()
                    .any(|item| { item.kind == TranslateUnknownKind::PlaceholderRepaired }),
                "{source}"
            );
        }
    }

    #[test]
    fn coordinated_children_keep_their_complete_text_inside_the_parent_quote() {
        let glossary = glossary();
        for (source, expected) in [
            (
                "先生は『「ドイツ！」と、「行く？」の文字数をそれぞれ数えなさい』と言いました。",
                ["「ドイツ！」", "「行く？」"],
            ),
            (
                "父は『「来て！」と、「すぐ帰る」の文字数をそれぞれ数えなさい』と言いました。",
                ["「来て！」", "「すぐ帰る」"],
            ),
            (
                "母は『日本語の「ドイツ」と、漢字の「独逸」を比べなさい』と言いました。",
                ["「ドイツ」", "「独逸」"],
            ),
            (
                "『二つの表記「ドイツ」と「独逸」を比べました』",
                ["「ドイツ」", "「独逸」"],
            ),
            (
                "先生は『「来て！帰る」と、「今すぐ。」の字数をそれぞれ数えなさい』と言いました。",
                ["「来て！帰る」", "「今すぐ。」"],
            ),
            (
                "先生は『「A!ドイツ？B」と、「行く？」の文字数をそれぞれ数えなさい』と言いました。",
                ["「A!ドイツ？B」", "「行く？」"],
            ),
            (
                "先生は『「駅！」、「行く？」の字数をそれぞれ調べてください』と言いました。",
                ["「駅！」", "「行く？」"],
            ),
            (
                "先生は『「ドイツ！」と「来て？」の合計の文字数を調べなさい』と言いました。",
                ["「ドイツ！」", "「来て？」"],
            ),
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            let text = &translated.target_text;
            let start = text.find('『').unwrap_or_else(|| panic!("{text}"));
            let end = text.rfind('』').unwrap_or_else(|| panic!("{text}"));
            let mut previous = start;
            for original in expected {
                let at = text.find(original).unwrap_or_else(|| panic!("{text}"));
                assert!(previous < at && at + original.len() <= end, "{text}");
                assert_eq!(text.matches(original).count(), 1, "{text}");
                previous = at;
            }
            assert!(
                !translated
                    .unknown
                    .iter()
                    .any(|item| item.kind == TranslateUnknownKind::PlaceholderRepaired),
                "{text}"
            );
        }
    }

    #[test]
    fn a_count_inside_one_quote_does_not_protect_other_utterances() {
        use super::{QuoteSyntax, Span, written_form_quotes};

        for source in [
            "先生は『「行く？」の文字数を数えてから、「ドイツ！」と答えなさい』と言いました。",
            "『「ドイツ！」と叫びました』と『「行く？」の文字数を数えました』を読みました。",
        ] {
            let chars: Vec<_> = source.chars().collect();
            let spans = written_form_quotes(
                &chars,
                Span {
                    start_char: 0,
                    end_char: chars.len(),
                },
                QuoteSyntax::default(),
            );
            let protected: Vec<String> = spans
                .into_iter()
                .map(|span| chars[span.start_char..span.end_char].iter().collect())
                .collect();
            assert_eq!(protected, ["「行く？」"], "{source}");
        }
    }

    #[test]
    fn quote_join_accepts_only_one_comma_after_the_conjunction() {
        use super::{QuoteJoin, quote_join};

        for (between, expected) in [
            ("と、", Some(QuoteJoin::Conjunction(false))),
            (" と　、 漢字の　", Some(QuoteJoin::Conjunction(true))),
            ("、", Some(QuoteJoin::Enumeration)),
            ("、と", None),
            ("と、、", None),
            ("と、。", None),
            ("と、\n", None),
            ("と、花子は", None),
        ] {
            assert_eq!(
                quote_join(&between.chars().collect::<Vec<_>>()),
                expected,
                "{between:?}"
            );
        }
    }
}
