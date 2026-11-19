use faberline_nlp::Span;
use faberline_nlp::doc::Segmentation;

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

/// Find explicitly written forms on the source paragraph, then give each
/// analysis sentence only its part of the original text to protect.
pub(crate) fn plan_written_form_quotes(
    chars: &[char],
    segmentation: &Segmentation,
) -> Vec<Vec<WrittenForm>> {
    let mut by_sentence = vec![Vec::new(); segmentation.sentences.len()];
    let sentences = &segmentation.sentences;
    for paragraph in &segmentation.paragraphs {
        let mut quotes = written_form_quotes(chars, paragraph.span);
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

fn written_form_quotes(chars: &[char], paragraph: Span) -> Vec<Span> {
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
                protect_sibling_forms(chars, inner, &frame.children, true, &mut protected);
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
    protect_sibling_forms(chars, paragraph, &top_level, false, &mut protected);
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
    protected: &mut Vec<Span>,
) {
    let mut first_for_count = 0;
    let mut first_for_comparison = 0;
    let mut labelled = false;
    for (index, quote) in quotes.iter().enumerate() {
        let opening = quote.start_char - 1;
        let before = &chars[scope.start_char..opening];
        let after = skip_space(&chars[quote.end_char + 1..scope.end_char]);
        let measures_form = measures_written_form(after);
        if measures_form
            || describes_written_form(after)
            || acts_on_written_form(before, after)
            || copies_written_form(after)
        {
            protected.push(protection_span(*quote, nested));
        }
        let join = index.checked_sub(1).and_then(|prev| {
            let after_close = quotes[prev].end_char + 1;
            quote_join(&chars[after_close..opening])
        });
        if let Some(QuoteJoin::Conjunction(right_labelled)) = join {
            labelled &= right_labelled;
        } else {
            first_for_comparison = index;
            if join.is_none() {
                first_for_count = index;
            }
            let before = trim_space_end(before);
            labelled = FORM_LABELS
                .iter()
                .any(|label| strip_suffix(before, label).is_some());
        }

        // The count noun applies to the whole list within this quotation scope,
        // even if terminal marks inside the operands split the analysis sentences.
        let first = if index > first_for_count && measures_form {
            first_for_count
        } else if index > first_for_comparison {
            let before = &chars[scope.start_char..quotes[first_for_comparison].start_char - 1];
            let count = index - first_for_comparison + 1;
            if !compares_written_forms(after)
                && !compares_apposed_forms(before, after, count, labelled)
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
    if compares_objects(after) && (labelled || has_form_name_before(before, count)) {
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

fn has_form_name_before(before: &[char], count: usize) -> bool {
    COMPARED_FORM_NAMES.iter().any(|form| {
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

fn copies_written_form(after: &[char]) -> bool {
    let operand = if let Some(word) = strip_prefix(after, "という") {
        let Some(operand) = after_word_name(word) else {
            return false;
        };
        operand
    } else if let Some(citation) = after.strip_prefix(&['と']) {
        return starts_copying(citation);
    } else {
        after
    };
    let Some(destination) = operand.strip_prefix(&['を']).map(after_clause_comma) else {
        return false;
    };
    starts_copying(destination)
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
                ch == 'に' && index > 0 && starts_copying(&destination[index + 1..])
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
        glossary.objects.push(GlossaryEntry {
            canonical: "申込書".to_owned(),
            surfaces: vec!["申込書".to_owned()],
            targets: [(TargetVariant::ZhHant, "申請表".to_owned())].into(),
            pronoun_class: Default::default(),
            speech_register: Default::default(),
        });
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
        use super::{Span, written_form_quotes};

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
