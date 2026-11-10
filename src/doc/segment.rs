//! Layer 1 rules on Unicode scalar values. All returned spans are absolute offsets into `c`,
//! trimmed of whitespace, and empty spans are dropped.

use super::Lang;
use crate::Span;

const CJK_END: &[char] = &['。', '！', '？', '…'];
const CLOSERS: &[char] = &['」', '』', '”', '’', '）', ')', '"'];
const PARAGRAPH_END: &[char] = &['。', '！', '？', '…', '」', '』', '”', '’', '）'];
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "mx", "dr", "prof", "rev", "sr", "jr", "st", "vs", "cf", "fig", "eq", "no",
    "e.g", "i.e",
];
/// Abbreviated month names (`May` is not one). A dot after one does not end the sentence when
/// a digit follows, so `Nov. 10, 2026` stays one sentence.
const ABBREVIATED_MONTHS: &[&str] = &[
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
];

pub(super) fn paragraphs(c: &[char]) -> Vec<Span> {
    let mut out = Vec::new();
    let mut open: Option<Span> = None;
    let mut prev = (0, 0);
    for (ls, le) in lines(c) {
        if c[ls..le].iter().all(|ch| ch.is_whitespace()) {
            out.extend(open.take().and_then(|s| trim(c, s)));
            continue;
        }
        open = match open {
            Some(p) if !starts_block(&c[ls..le]) && !ends_block(&c[prev.0..prev.1]) => {
                Some(Span { end_char: le, ..p })
            }
            other => {
                out.extend(other.and_then(|s| trim(c, s)));
                Some(Span {
                    start_char: ls,
                    end_char: le,
                })
            }
        };
        prev = (ls, le);
    }
    out.extend(open.and_then(|s| trim(c, s)));
    out
}

pub(super) fn sentences(c: &[char], paragraph: Span) -> Vec<Span> {
    let end = paragraph.end_char;
    let mut out = Vec::new();
    let mut start = paragraph.start_char;
    let mut i = start;
    while i < end {
        let (cut, next) = match c[i] {
            '。' | '！' | '？' | '…' | '!' | '?' => terminal_run(c, i, end),
            '.' => dots(c, paragraph, i),
            '\n' => newline(c, paragraph, i),
            ' ' | '\u{3000}' => spaces(c, paragraph, i),
            _ => (None, i + 1),
        };
        if let Some((stop, resume)) = cut {
            out.extend(trim(
                c,
                Span {
                    start_char: start,
                    end_char: stop,
                },
            ));
            start = resume;
        }
        i = next;
    }
    out.extend(trim(
        c,
        Span {
            start_char: start,
            end_char: end,
        },
    ));
    out
}

pub(super) fn clauses(c: &[char], sentence: Span) -> Vec<Span> {
    let mut out = Vec::new();
    let mut start = sentence.start_char;
    for (k, &ch) in c[sentence.start_char..sentence.end_char].iter().enumerate() {
        let i = sentence.start_char + k;
        let split = match ch {
            '，' | '；' | '：' => true,
            ',' | ';' | ':' => i + 1 == sentence.end_char || c[i + 1].is_whitespace(),
            _ => false,
        };
        if split {
            out.extend(trim(
                c,
                Span {
                    start_char: start,
                    end_char: i,
                },
            ));
            start = i + 1;
        }
    }
    out.extend(trim(
        c,
        Span {
            start_char: start,
            end_char: sentence.end_char,
        },
    ));
    out
}

pub(super) fn lang(c: &[char], span: Span) -> (Lang, bool) {
    let (han, latin, other) = counts(&c[span.start_char..span.end_char]);
    if han + latin + other == 0 {
        (Lang::Und, false)
    } else if other >= han + latin {
        (Lang::Und, true)
    } else if han > 0 && latin > 0 {
        (Lang::Mixed, false)
    } else if han > 0 {
        (Lang::Zh, false)
    } else {
        (Lang::En, false)
    }
}

/// Start and end index of each line; the newline itself is excluded.
fn lines(c: &[char]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, &ch) in c.iter().enumerate() {
        if ch == '\n' {
            out.push((start, i));
            start = i + 1;
        }
    }
    out.push((start, c.len()));
    out
}

fn trim(c: &[char], s: Span) -> Option<Span> {
    let mut start = s.start_char;
    let mut end = s.end_char;
    while start < end && c[start].is_whitespace() {
        start += 1;
    }
    while end > start && c[end - 1].is_whitespace() {
        end -= 1;
    }
    (start < end).then_some(Span {
        start_char: start,
        end_char: end,
    })
}

fn starts_block(line: &[char]) -> bool {
    matches!(line, ['\u{3000}' | '\t', ..] | [' ', ' ', ..])
}

/// A line ends a block with a terminator when it is Han-heavy.
fn ends_block(line: &[char]) -> bool {
    let Some(last) = line.iter().rev().find(|ch| !ch.is_whitespace()) else {
        return false;
    };
    let (han, _, other) = counts(line);
    PARAGRAPH_END.contains(last) && han > other
}

/// Returns (han, latin, other letters). Other letters are alphabetic and neither Han nor Latin.
fn counts(chars: &[char]) -> (usize, usize, usize) {
    let (mut han, mut latin, mut other) = (0, 0, 0);
    for &ch in chars {
        if is_han(ch) {
            han += 1;
        } else if is_latin(ch) {
            latin += 1;
        } else if ch.is_alphabetic() {
            other += 1;
        }
    }
    (han, latin, other)
}

pub(super) fn is_han(ch: char) -> bool {
    matches!(
        ch,
        '\u{4e00}'..='\u{9fff}'
            | '\u{3400}'..='\u{4dbf}'
            | '\u{20000}'..='\u{2a6df}'
            | '\u{f900}'..='\u{faff}'
    )
}

fn is_latin(ch: char) -> bool {
    ch.is_ascii_alphabetic()
        || (('\u{c0}'..='\u{24f}').contains(&ch) && ch != '\u{d7}' && ch != '\u{f7}')
}

fn run_end(c: &[char], from: usize, end: usize, pred: impl Fn(char) -> bool) -> usize {
    let mut j = from;
    while j < end && pred(c[j]) {
        j += 1;
    }
    j
}

fn skip_closers(c: &[char], from: usize, end: usize) -> usize {
    run_end(c, from, end, |ch| CLOSERS.contains(&ch))
}

fn at_break(c: &[char], at: usize, end: usize) -> bool {
    at == end || c[at].is_whitespace()
}

fn next_nonspace(c: &[char], from: usize, end: usize) -> Option<char> {
    c[from..end].iter().copied().find(|ch| !ch.is_whitespace())
}

/// Rule 1: a run of CJK or ASCII terminators, then closers. Returns (cut, next index).
fn terminal_run(c: &[char], i: usize, end: usize) -> (Option<(usize, usize)>, usize) {
    let j = run_end(c, i, end, |ch| {
        CJK_END.contains(&ch) || matches!(ch, '!' | '?' | '.')
    });
    let k = skip_closers(c, j, end);
    let cjk = c[i..j].iter().any(|ch| CJK_END.contains(ch));
    let ends = cjk || at_break(c, k, end);
    (ends.then_some((k, k)), k)
}

/// Rules 2 and 3: a single ASCII dot, or a run of two or more.
fn dots(c: &[char], paragraph: Span, i: usize) -> (Option<(usize, usize)>, usize) {
    let end = paragraph.end_char;
    let j = run_end(c, i, end, |ch| ch == '.');
    if j - i >= 2 {
        let ends = at_break(c, j, end)
            && next_nonspace(c, j, end).is_none_or(|ch| !ch.is_ascii_lowercase());
        return (ends.then_some((j, j)), j);
    }
    let k = skip_closers(c, j, end);
    if !at_break(c, k, end) {
        return (None, j);
    }
    let word_start = (paragraph.start_char..i)
        .rev()
        .take_while(|&m| c[m].is_ascii_alphabetic() || c[m] == '.')
        .last()
        .unwrap_or(i);
    let word: String = c[word_start..i].iter().collect();
    if is_abbreviation(&word) {
        return (None, j);
    }
    let month = word.to_ascii_lowercase();
    if ABBREVIATED_MONTHS.contains(&month.as_str())
        && k + 1 < end
        && c[k] == ' '
        && c[k + 1].is_ascii_digit()
    {
        return (None, j);
    }
    let lower = word.to_ascii_lowercase();
    if matches!(lower.as_str(), "etc" | "al")
        && !next_nonspace(c, k, end).is_none_or(char::is_uppercase)
    {
        return (None, j);
    }
    (Some((k, k)), k)
}

/// A single uppercase ASCII letter, or a listed abbreviation with ASCII case folding. Layer 1
/// uses it before a dot, and the tokenizer uses it to keep the dot in the token.
pub(super) fn is_abbreviation(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    let mut letters = word.chars();
    let single_upper =
        matches!((letters.next(), letters.next()), (Some(ch), None) if ch.is_ascii_uppercase());
    single_upper || ABBREVIATIONS.contains(&lower.as_str())
}

/// Rule 4: a newline ends the sentence unless it is a soft wrap.
fn newline(c: &[char], paragraph: Span, i: usize) -> (Option<(usize, usize)>, usize) {
    let line_start = (paragraph.start_char..i)
        .rev()
        .find(|&m| c[m] == '\n')
        .map_or(paragraph.start_char, |m| m + 1);
    let soft_wrap = trim(
        c,
        Span {
            start_char: line_start,
            end_char: i,
        },
    )
    .is_some_and(|line| {
        let last = c[line.end_char - 1];
        line.end_char - line.start_char >= 40 && (last.is_ascii_alphanumeric() || last == ',')
    }) && next_nonspace(c, i + 1, paragraph.end_char)
        .is_some_and(|ch| ch.is_ascii_lowercase());
    if soft_wrap {
        (None, i + 1)
    } else {
        (Some((i, i + 1)), i + 1)
    }
}

/// Rule 5: two or more spaces inside a line end the sentence.
fn spaces(c: &[char], paragraph: Span, i: usize) -> (Option<(usize, usize)>, usize) {
    let end = paragraph.end_char;
    let j = run_end(c, i, end, |ch| ch == ' ' || ch == '\u{3000}');
    if j - i >= 2 && j < end && c[j] != '\n' {
        (Some((i, j)), j)
    } else {
        (None, j)
    }
}
