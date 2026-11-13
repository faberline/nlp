//! Layer 1 rules on Unicode scalar values. All returned spans are absolute offsets into `c`,
//! trimmed of whitespace, and empty spans are dropped.

use super::Lang;
use crate::Span;

const CJK_END: &[char] = &['。', '．', '！', '？', '…', '‥', '⋯', '｡'];
const CLOSERS: &[char] = &[
    '」', '』', '”', '’', '）', ')', '"', '\'', '»', ']', '】', '〕', '〉', '》', '］', '｝',
];
const PARAGRAPH_END: &[char] = &[
    '。', '．', '！', '？', '…', '‥', '⋯', '」', '』', '”', '’', '）', '】', '〕', '〉', '》',
    '］', '｝',
];
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "mx", "dr", "prof", "rev", "sr", "jr", "st", "vs", "cf", "fig", "eq", "no",
    "e.g", "i.e", "sec", "ch", "vol", "vols", "eds", "trans", "pp",
];
/// Reference abbreviations that suppress a sentence break only when followed by an ASCII digit.
const NUMERIC_ABBREVIATIONS: &[&str] = &["p", "ed", "art", "pt", "para", "paras"];
/// Unconditional title and organization abbreviations matched only when capitalized so lowercase
/// common words (`gen`, `rep`, `lt`, `inc`) are never merged or suppressed at sentence end.
const CAP_ABBREVIATIONS: &[&str] = &[
    "gen", "col", "maj", "capt", "lt", "sgt", "gov", "sen", "rep", "univ", "dept", "assn", "bros",
    "inc", "ltd", "corp",
];
/// Dotted time and degree abbreviations whose trailing dot stays in the token, and whose sentence
/// boundary is suppressed unless an uppercase new sentence follows.
const CONT_ABBREVIATIONS: &[&str] = &["a.m", "p.m", "ph.d", "m.d", "b.a", "m.a", "b.s", "m.s"];
/// Jurisdiction abbreviations (`U.S.`, `U.K.`, `E.U.`, `U.N.`) whose trailing dot stays in the
/// token and suppresses sentence break before proper nouns unless a sentence-starter follows.
const GEO_ABBREVIATIONS: &[&str] = &["u.s", "u.k", "e.u", "u.n"];
/// Uppercase closed-class sentence-starters that end a sentence after a `GEO_ABBREVIATIONS` dot.
const SENTENCE_STARTERS: &[&str] = &[
    "The",
    "This",
    "That",
    "These",
    "Those",
    "It",
    "He",
    "She",
    "They",
    "We",
    "I",
    "In",
    "On",
    "At",
    "For",
    "From",
    "By",
    "With",
    "However",
    "Moreover",
    "Furthermore",
    "Therefore",
    "Thus",
    "Meanwhile",
    "Although",
    "Because",
    "When",
    "While",
    "If",
    "As",
    "But",
    "And",
    "Or",
    "Yet",
    "So",
    "Our",
    "Their",
    "His",
    "Her",
    "Its",
    "A",
    "An",
    "Here",
    "There",
    "Later",
    "Then",
];
/// Capitalized continuation-guarded abbreviations without an inner dot.
const CONT_CAP_ABBREVIATIONS: &[&str] = &["co"];
/// Abbreviated month names (`May` is not one). A dot after one does not end the sentence when
/// a digit follows, so `Nov. 10, 2026` stays one sentence.
const ABBREVIATED_MONTHS: &[&str] = &[
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
];
const MONTH_NAMES: &[&str] = &[
    "jan",
    "january",
    "feb",
    "february",
    "mar",
    "march",
    "apr",
    "april",
    "may",
    "jun",
    "june",
    "jul",
    "july",
    "aug",
    "august",
    "sep",
    "sept",
    "september",
    "oct",
    "october",
    "nov",
    "november",
    "dec",
    "december",
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
            '．' if is_decimal_fullwidth_dot(c, paragraph.start_char, i, end) => (None, i + 1),
            ch if CJK_END.contains(&ch) || matches!(ch, '!' | '?') => terminal_run(c, i, end),
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

#[allow(dead_code)]
pub(super) fn clauses(c: &[char], sentence: Span) -> Vec<Span> {
    clauses_with_lang(c, sentence, false)
}

pub(super) fn clauses_with_lang(c: &[char], sentence: Span, sent_ja: bool) -> Vec<Span> {
    let mut out = Vec::new();
    let mut start = sentence.start_char;
    let mut paren_depth: usize = 0;
    for (k, &ch) in c[sentence.start_char..sentence.end_char].iter().enumerate() {
        let i = sentence.start_char + k;
        if matches!(ch, '(' | '（') {
            if has_matching_close_paren(c, i, sentence.end_char) {
                paren_depth += 1;
            }
            continue;
        }
        if matches!(ch, ')' | '）') {
            paren_depth = paren_depth.saturating_sub(1);
            continue;
        }
        let split = match ch {
            '，' if paren_depth == 0 => true,
            '；' | '：' => true,
            '、' => sent_ja && paren_depth == 0,
            ',' if paren_depth == 0
                && !is_us_date_comma(c, sentence.start_char, i, sentence.end_char) =>
            {
                i + 1 == sentence.end_char
                    || c[i + 1].is_whitespace()
                    || (i > sentence.start_char && is_hangul(c[i - 1]) && is_hangul(c[i + 1]))
            }
            ';' | ':' => i + 1 == sentence.end_char || c[i + 1].is_whitespace(),
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
    lang_in_context(c, span, false)
}

pub(super) fn lang_in_context(c: &[char], span: Span, context_ja: bool) -> (Lang, bool) {
    let s = counts(&c[span.start_char..span.end_char]);
    let sup = s.han + s.kana + s.hangul + s.latin;
    if sup + s.other == 0 {
        return (Lang::Und, false);
    }
    if s.other >= sup {
        return (Lang::Und, true);
    }
    let has_ja = s.kana > 0 || (s.han > 0 && (s.has_ja_exclusive || (context_ja && s.hangul == 0)));
    let has_zh = s.han > 0 && !has_ja;
    let has_ko = s.hangul > 0;
    let has_en = s.latin > 0;
    let families =
        usize::from(has_ja) + usize::from(has_zh) + usize::from(has_ko) + usize::from(has_en);
    let lang = if families >= 2 {
        Lang::Mixed
    } else if has_ja {
        Lang::Ja
    } else if has_ko {
        Lang::Ko
    } else if has_zh {
        Lang::Zh
    } else {
        Lang::En
    };
    (lang, false)
}

pub(super) fn has_ja_cue(c: &[char], span: Span) -> bool {
    let slice = &c[span.start_char..span.end_char];
    slice
        .iter()
        .copied()
        .any(|ch| is_kana(ch) || is_ja_exclusive_kanji(ch))
        || has_ja_slang_tag(slice)
}

pub(super) fn has_ja_or_ko(c: &[char], span: Span) -> bool {
    c[span.start_char..span.end_char]
        .iter()
        .copied()
        .any(|ch| is_kana(ch) || is_hangul(ch) || is_ja_exclusive_kanji(ch))
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

/// A line ends a block with a CJK/Japanese terminator when Han + Kana outnumber Latin + other.
fn ends_block(line: &[char]) -> bool {
    let Some(last) = line.iter().rev().find(|ch| !ch.is_whitespace()) else {
        return false;
    };
    let s = counts(line);
    PARAGRAPH_END.contains(last) && (s.han + s.kana) > (s.latin + s.other)
}

struct ScriptCounts {
    han: usize,
    kana: usize,
    hangul: usize,
    latin: usize,
    other: usize,
    has_ja_exclusive: bool,
}

fn counts(chars: &[char]) -> ScriptCounts {
    let mut out = ScriptCounts {
        han: 0,
        kana: 0,
        hangul: 0,
        latin: 0,
        other: 0,
        has_ja_exclusive: false,
    };
    for &ch in chars {
        if is_han(ch) {
            out.han += 1;
            if is_ja_exclusive_kanji(ch) {
                out.has_ja_exclusive = true;
            }
        } else if is_kana(ch) {
            out.kana += 1;
        } else if is_hangul(ch) {
            out.hangul += 1;
        } else if is_latin(ch) {
            out.latin += 1;
        } else if ch.is_alphabetic() {
            out.other += 1;
        }
    }
    if !out.has_ja_exclusive && has_ja_slang_tag(chars) {
        out.has_ja_exclusive = true;
    }
    out
}

fn has_ja_slang_tag(chars: &[char]) -> bool {
    chars.windows(3).any(|w| {
        matches!(
            w,
            ['（', '笑' | '泣' | '汗' | '涙' | '爆', '）']
                | ['(', '笑' | '泣' | '汗' | '涙' | '爆', ')']
        )
    })
}

fn is_decimal_fullwidth_dot(c: &[char], bound: usize, i: usize, end: usize) -> bool {
    let is_digit = |ch: char| ch.is_ascii_digit() || ('０'..='９').contains(&ch);
    i > bound && i + 1 < end && is_digit(c[i - 1]) && is_digit(c[i + 1])
}

fn has_matching_close_paren(c: &[char], open_at: usize, end: usize) -> bool {
    let mut balance: usize = 1;
    for &ch in &c[open_at + 1..end] {
        if matches!(ch, '(' | '（') {
            balance += 1;
        } else if matches!(ch, ')' | '）') {
            balance -= 1;
            if balance == 0 {
                return true;
            }
        }
    }
    false
}

fn is_us_date_comma(c: &[char], bound: usize, i: usize, end: usize) -> bool {
    if i + 6 > end
        || c[i + 1] != ' '
        || !c[i + 2..i + 6].iter().all(|ch| ch.is_ascii_digit())
        || (i + 6 < end && c[i + 6].is_ascii_digit())
    {
        return false;
    }
    let Some(day_start) = digits_before(c, bound, i, 2) else {
        return false;
    };
    if day_start <= bound || c[day_start - 1] != ' ' {
        return false;
    }
    let word_end = if day_start >= bound + 2 && c[day_start - 2] == '.' {
        day_start - 2
    } else {
        day_start - 1
    };
    let mut word_start = word_end;
    while word_start > bound && c[word_start - 1].is_ascii_alphabetic() {
        word_start -= 1;
    }
    if word_start == word_end {
        return false;
    }
    let lower: String = c[word_start..word_end]
        .iter()
        .map(|ch| ch.to_ascii_lowercase())
        .collect();
    MONTH_NAMES.contains(&lower.as_str())
}

pub(super) fn is_han(ch: char) -> bool {
    matches!(
        ch,
        '\u{4e00}'..='\u{9fff}'
            | '\u{3400}'..='\u{4dbf}'
            | '\u{20000}'..='\u{2ceaf}'
            | '\u{f900}'..='\u{faff}'
            | '\u{3005}'..='\u{3007}'
    )
}

pub(super) fn is_kana(ch: char) -> bool {
    matches!(
        ch,
        '\u{3041}'..='\u{3096}'
            | '\u{309d}'..='\u{309f}'
            | '\u{30a1}'..='\u{30fa}'
            | '\u{30fc}'..='\u{30ff}'
            | '\u{31f0}'..='\u{31ff}'
            | '\u{ff66}'..='\u{ff9f}'
    )
}

pub(super) fn is_hangul(ch: char) -> bool {
    matches!(
        ch,
        '\u{ac00}'..='\u{d7a3}'
            | '\u{1100}'..='\u{11ff}'
            | '\u{3131}'..='\u{318e}'
            | '\u{a960}'..='\u{a97c}'
            | '\u{d7b0}'..='\u{d7fb}'
            | '\u{ffa0}'..='\u{ffdc}'
    )
}

#[allow(dead_code)]
pub(super) fn is_latin(ch: char) -> bool {
    ch.is_ascii_alphabetic()
        || (('\u{c0}'..='\u{24f}').contains(&ch) && ch != '\u{d7}' && ch != '\u{f7}')
        || matches!(ch, '\u{ff21}'..='\u{ff3a}' | '\u{ff41}'..='\u{ff5a}')
}

pub(super) fn is_ja_exclusive_kanji(ch: char) -> bool {
    matches!(
        ch,
        '駅' | '図'
            | '読'
            | '発'
            | '観'
            | '験'
            | '権'
            | '様'
            | '戦'
            | '続'
            | '転'
            | '広'
            | '庁'
            | '労'
            | '変'
            | '実'
            | '気'
            | '楽'
            | '帰'
            | '歩'
            | '毎'
            | '黒'
            | '経'
            | '絵'
            | '総'
            | '訳'
            | '鉄'
            | '鉱'
            | '顕'
            | '込'
            | '枠'
            | '峠'
            | '畑'
            | '辻'
            | '雫'
            | '〆'
    )
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

fn next_nonspace_same_line(c: &[char], from: usize, end: usize) -> Option<char> {
    for &ch in &c[from..end] {
        if ch == '\n' || ch == '\r' {
            return None;
        }
        if !ch.is_whitespace() {
            return Some(ch);
        }
    }
    None
}

fn next_word_same_line(c: &[char], from: usize, end: usize) -> Option<String> {
    let mut i = from;
    while i < end {
        if c[i] == '\n' || c[i] == '\r' {
            return None;
        }
        if !c[i].is_whitespace() {
            break;
        }
        i += 1;
    }
    if i >= end {
        return None;
    }
    let w_end = run_end(c, i, end, |ch| ch.is_ascii_alphabetic());
    Some(c[i..w_end].iter().collect())
}

fn starts_with_str(slice: &[char], pat: &str) -> bool {
    let mut it = slice.iter().copied();
    for p in pat.chars() {
        if it.next() != Some(p) {
            return false;
        }
    }
    true
}

fn is_korean_quotative_continuation(c: &[char], k: usize, end: usize) -> bool {
    if k >= end {
        return false;
    }
    let mut ns = k;
    while ns < end && c[ns].is_whitespace() {
        if c[ns] == '\n' || c[ns] == '\r' {
            return false;
        }
        ns += 1;
    }
    if ns == end {
        return false;
    }
    if ns == k {
        const ATTACHED: &[&str] = &[
            "라고",
            "이라며",
            "냐고",
            "자고",
            "다고",
            "고 ",
            "고,",
            "고.",
        ];
        if ATTACHED.iter().any(|pat| starts_with_str(&c[k..end], pat)) {
            return true;
        }
    }
    const FOLLOWED: &[&str] = &[
        "하고",
        "하며",
        "하면",
        "하는",
        "하자",
        "하니",
        "하여",
        "해서",
        "하더니",
        "라고",
        "이라며",
    ];
    for pat in FOLLOWED {
        if starts_with_str(&c[ns..end], pat) {
            let after = ns + pat.chars().count();
            if after == end
                || c[after].is_whitespace()
                || matches!(c[after], ',' | '.' | '!' | '?' | '，' | '。')
            {
                return true;
            }
        }
    }
    false
}

fn is_inline_parenthetical_tag(c: &[char], k: usize, end: usize) -> bool {
    let mut ns = k;
    while ns < end && c[ns].is_whitespace() {
        if c[ns] == '\n' || c[ns] == '\r' {
            return false;
        }
        ns += 1;
    }
    if ns >= end {
        return false;
    }
    let close = match c[ns] {
        '(' => ')',
        '（' => '）',
        _ => return false,
    };
    let limit = (ns + 6).min(end.saturating_sub(1));
    for (m, &ch) in c[..=limit].iter().enumerate().skip(ns + 1) {
        if ch == '\n' || ch == '\r' {
            return false;
        }
        if m >= ns + 2 && ch == close {
            return true;
        }
    }
    false
}

/// Rule 1: a run of CJK or ASCII terminators, then closers. Returns (cut, next index).
fn terminal_run(c: &[char], i: usize, end: usize) -> (Option<(usize, usize)>, usize) {
    let j = run_end(c, i, end, |ch| {
        CJK_END.contains(&ch) || matches!(ch, '!' | '?' | '.')
    });
    let k = skip_closers(c, j, end);
    let only_ellipsis = c[i..j].iter().all(|ch| matches!(ch, '…' | '‥' | '⋯'));
    if only_ellipsis
        && j == k
        && k < end
        && matches!(c[k], '，' | '、' | '；' | '：' | ',' | ';' | ':')
    {
        return (None, k);
    }
    if only_ellipsis && is_inline_parenthetical_tag(c, k, end) {
        return (None, k);
    }
    if k > j
        && c[j..k]
            .iter()
            .any(|ch| matches!(ch, '"' | '”' | '\'' | '’' | '」' | '』'))
        && is_korean_quotative_continuation(c, k, end)
    {
        return (None, k);
    }
    let cjk = c[i..j].iter().any(|ch| CJK_END.contains(ch));
    if !cjk
        && k > j
        && c[j..k]
            .iter()
            .any(|ch| matches!(ch, '"' | '”' | '\'' | '’'))
        && next_nonspace_same_line(c, k, end).is_some_and(|ch| ch.is_ascii_lowercase())
    {
        return (None, k);
    }
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
    if k > j
        && c[j..k]
            .iter()
            .any(|ch| matches!(ch, '"' | '”' | '\'' | '’'))
        && is_korean_quotative_continuation(c, k, end)
    {
        return (None, k);
    }
    if is_korean_date_inner_dot(c, paragraph, i, j, k) {
        return (None, j);
    }
    if is_korean_date_trailing_dot(c, paragraph, i, k) {
        return (None, k);
    }
    let word_start = (paragraph.start_char..i)
        .rev()
        .take_while(|&m| c[m].is_ascii_alphabetic() || c[m] == '.')
        .last()
        .unwrap_or(i);
    let raw_word: String = c[word_start..i].iter().collect();
    let word = raw_word.trim_start_matches('.');
    let lower = word.to_ascii_lowercase();
    let first_upper = word
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_uppercase());
    let single_upper = first_upper && word.len() == 1;
    if single_upper
        || ABBREVIATIONS.contains(&lower.as_str())
        || (first_upper && CAP_ABBREVIATIONS.contains(&lower.as_str()))
    {
        return (None, j);
    }
    if (ABBREVIATED_MONTHS.contains(&lower.as_str())
        || NUMERIC_ABBREVIATIONS.contains(&lower.as_str()))
        && k + 1 < end
        && c[k] == ' '
        && c[k + 1].is_ascii_digit()
    {
        return (None, j);
    }
    if GEO_ABBREVIATIONS.contains(&lower.as_str()) {
        let should_split = match next_word_same_line(c, k, end) {
            None => true,
            Some(w) => SENTENCE_STARTERS.contains(&w.as_str()),
        };
        if !should_split {
            return (None, j);
        }
        return (Some((k, k)), k);
    }
    let cont_abbrev = matches!(lower.as_str(), "etc" | "al")
        || CONT_ABBREVIATIONS.contains(&lower.as_str())
        || (first_upper && CONT_CAP_ABBREVIATIONS.contains(&lower.as_str()));
    if cont_abbrev && !next_nonspace(c, k, end).is_none_or(char::is_uppercase) {
        return (None, j);
    }
    (Some((k, k)), k)
}

/// Scans a maximal ASCII digit run ending at `dot_at` within `bound..dot_at`, returning its start
/// index when its length is in `1..=max_digits` and it is not preceded by an ASCII letter.
fn digits_before(c: &[char], bound: usize, dot_at: usize, max_digits: usize) -> Option<usize> {
    let mut s = dot_at;
    while s > bound && c[s - 1].is_ascii_digit() {
        s -= 1;
    }
    let len = dot_at - s;
    if (1..=max_digits).contains(&len) && (s == bound || !c[s - 1].is_ascii_alphabetic()) {
        Some(s)
    } else {
        None
    }
}

/// Inner dot of a Korean dotted date (`2026. 11. 13.` or `11. 13.`): `<1..=4 digits>. <1..=2 digits>.`
fn is_korean_date_inner_dot(c: &[char], paragraph: Span, i: usize, j: usize, k: usize) -> bool {
    let bound = paragraph.start_char;
    let end = paragraph.end_char;
    if j != k || j >= end || c[j] != ' ' {
        return false;
    }
    let Some(d1_start) = digits_before(c, bound, i, 4) else {
        return false;
    };
    if d1_start >= bound + 2
        && c[d1_start - 1] == ' '
        && c[d1_start - 2] == '.'
        && (i - d1_start > 2 || digits_before(c, bound, d1_start - 2, 4).is_none())
    {
        return false;
    }
    let m_end = run_end(c, j + 1, end, |ch| ch.is_ascii_digit());
    let m_len = m_end - (j + 1);
    (1..=2).contains(&m_len)
        && m_end < end
        && c[m_end] == '.'
        && (m_end + 1 == end || (!c[m_end + 1].is_ascii_digit() && c[m_end + 1] != '.'))
}

/// Trailing dot of `<1..=4 digits>. <1..=2 digits>.` (or 3-part `YYYY. MM. DD.`) followed on the
/// same line by Hangul, an ASCII digit, or an ASCII lowercase letter.
fn is_korean_date_trailing_dot(c: &[char], paragraph: Span, i: usize, k: usize) -> bool {
    let bound = paragraph.start_char;
    let Some(d2_start) = digits_before(c, bound, i, 2) else {
        return false;
    };
    if d2_start < bound + 2 || c[d2_start - 1] != ' ' || c[d2_start - 2] != '.' {
        return false;
    }
    let d1_dot = d2_start - 2;
    let Some(d1_start) = digits_before(c, bound, d1_dot, 4) else {
        return false;
    };
    if d1_start >= bound + 2
        && c[d1_start - 1] == ' '
        && c[d1_start - 2] == '.'
        && (d1_dot - d1_start > 2 || digits_before(c, bound, d1_start - 2, 4).is_none())
    {
        return false;
    }
    next_nonspace_same_line(c, k, paragraph.end_char)
        .is_some_and(|ch| is_hangul(ch) || ch.is_ascii_digit() || ch.is_ascii_lowercase())
}

/// A single uppercase ASCII letter, or a listed abbreviation. Layer 1 uses it before a dot, and
/// the tokenizer uses it to keep the dot in the token.
pub(super) fn is_abbreviation(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    let mut letters = word.chars();
    let first_upper = letters.next().is_some_and(|ch| ch.is_ascii_uppercase());
    let single_upper = first_upper && letters.next().is_none();
    single_upper
        || ABBREVIATIONS.contains(&lower.as_str())
        || CONT_ABBREVIATIONS.contains(&lower.as_str())
        || GEO_ABBREVIATIONS.contains(&lower.as_str())
        || (first_upper
            && (CAP_ABBREVIATIONS.contains(&lower.as_str())
                || CONT_CAP_ABBREVIATIONS.contains(&lower.as_str())))
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
