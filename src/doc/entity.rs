//! Layer 3 of `nlp.doc.v1`: entities by rules. See `docs/doc-v1.md`.
//!
//! Scanners run over each supported sentence. Inline opaque spans are hidden behind a sentinel
//! char that no rule matches, so no candidate can start or end inside one.

use super::opaque::Found;
use super::{Entity, EntityKind, OpaqueKind, Reason, Unknown};
use crate::Span;
use aho_corasick::{AhoCorasick, MatchKind};
use std::cell::RefCell;
use std::cmp::Reverse;

const SENTINEL: char = '\u{0}';
const MEASURE: &str = "個人位次件隻張本條台輛間場筆項回遍倍歲名所家句片顆枚篇部";
const ZH_RELATIVE: &[&str] = &[
    "今天",
    "明天",
    "昨天",
    "後天",
    "前天",
    "今晚",
    "昨晚",
    "今年",
    "明年",
    "去年",
    "這週",
    "本週",
    "上週",
    "下週",
    "這個月",
    "上個月",
    "下個月",
];
const EN_RELATIVE: &[&str] = &["today", "tomorrow", "yesterday", "tonight"];
const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];
const EN_SMALL: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const EN_TENS: [&str; 8] = [
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const CURRENCY: &[&str] = &["US$", "NT$", "$", "¥", "€", "£"];
const PERCENT_PREFIX: &str = "百分之";
/// Units, and whether the unit must follow the number with no space.
const UNITS: &[(&str, bool)] = &[
    ("公斤", false),
    ("公克", false),
    ("克", false),
    ("公里", false),
    ("公尺", false),
    ("公分", false),
    ("毫米", false),
    ("公升", false),
    ("毫升", false),
    ("元", false),
    ("美元", false),
    ("台幣", false),
    ("秒", false),
    ("分鐘", false),
    ("小時", false),
    ("天", false),
    ("週", false),
    ("個月", false),
    ("%", false),
    ("kg", false),
    ("mg", false),
    ("km", false),
    ("cm", false),
    ("mm", false),
    ("ml", false),
    ("lb", false),
    ("lbs", false),
    ("oz", false),
    ("mph", false),
    ("seconds", false),
    ("second", false),
    ("sec", false),
    ("minutes", false),
    ("minute", false),
    ("min", false),
    ("hours", false),
    ("hour", false),
    ("hr", false),
    ("days", false),
    ("day", false),
    ("weeks", false),
    ("week", false),
    ("months", false),
    ("month", false),
    ("years", false),
    ("year", false),
    ("dollars", false),
    ("dollar", false),
    ("usd", false),
    ("ntd", false),
    ("euros", false),
    ("euro", false),
    ("g", true),
    ("m", true),
    ("l", true),
];

#[derive(Clone, Copy)]
enum Marker {
    /// `strict` is true for `am`, `a.m.` and `凌晨`, where 12 means 00:00.
    Morning(bool),
    /// `strict` is true for `pm`, `p.m.` and `下午`, where 12 means 12:00.
    Afternoon(bool),
    Noon,
}

/// Markers placed before the hour.
const ZH_MARKERS: &[(&str, Marker)] = &[
    ("凌晨", Marker::Morning(true)),
    ("上午", Marker::Morning(false)),
    ("早上", Marker::Morning(false)),
    ("下午", Marker::Afternoon(true)),
    ("晚上", Marker::Afternoon(false)),
    ("傍晚", Marker::Afternoon(false)),
    ("中午", Marker::Noon),
];
/// Markers placed after the hour.
const AM_PM: &[(&str, Marker)] = &[
    ("am", Marker::Morning(true)),
    ("a.m.", Marker::Morning(true)),
    ("pm", Marker::Afternoon(true)),
    ("p.m.", Marker::Afternoon(true)),
];

#[derive(Clone, Copy)]
enum Tok {
    Add(u64),
    Mul(u64),
    Big(u64),
}

/// A candidate. `kind` None is a not-decided span, reported as an ambiguous unknown.
struct Cand {
    span: Span,
    sentence: usize,
    kind: Option<EntityKind>,
    value: Option<String>,
    unit: Option<String>,
}

/// A sentence being scanned. `t` is the text with inline opaque spans replaced by the sentinel.
struct Sent<'a> {
    t: &'a [char],
    s: usize,
    e: usize,
    idx: usize,
    /// Spans of invalid full dates. No date candidate may overlap one, so a shorter date inside
    /// is not read instead.
    bad_dates: RefCell<Vec<(usize, usize)>>,
}

impl Sent<'_> {
    /// True when `i` does not continue an ASCII word or number.
    fn bound(&self, i: usize) -> bool {
        i == self.s || !self.t[i - 1].is_ascii_alphanumeric()
    }

    /// True when a Chinese numeral char may start a candidate at `i`: it is not inside a numeral
    /// run and not after 第. Every scanner that starts at a Chinese numeral char checks this.
    fn zh_start_ok(&self, i: usize) -> bool {
        i == self.s
            || !(is_zh_num(self.t[i - 1])
                || self.t[i - 1] == '第'
                || self.t[i - 1].is_ascii_digit())
    }

    /// Notes an invalid full date, and returns None for the caller.
    fn bad_date(&self, start: usize, end: usize) -> Option<Cand> {
        self.bad_dates.borrow_mut().push((start, end));
        None
    }

    fn cand(
        &self,
        start: usize,
        end: usize,
        kind: Option<EntityKind>,
        value: Option<String>,
        unit: Option<String>,
    ) -> Cand {
        Cand {
            span: Span {
                start_char: start,
                end_char: end,
            },
            sentence: self.idx,
            kind,
            value,
            unit,
        }
    }
}

struct Num {
    value: String,
    end: usize,
    /// The English word `one` alone. It is not a number.
    bare_one: bool,
}

pub(super) fn extract(
    c: &[char],
    found: &[Found],
    sentences: &[(usize, Span)],
    names: &[String],
) -> (Vec<Entity>, Vec<Unknown>) {
    let mut t = c.to_vec();
    for f in found.iter().filter(|f| !f.block) {
        t[f.item.span.start_char..f.item.span.end_char].fill(SENTINEL);
    }
    let names: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|n| !n.is_empty() && !n.contains(['\n', SENTINEL]))
        .collect();
    let ac = (!names.is_empty()).then(|| {
        AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&names)
            .expect("name patterns")
    });
    let mut cands = Vec::new();
    for &(idx, span) in sentences {
        let x = Sent {
            t: &t,
            s: span.start_char,
            e: span.end_char,
            idx,
            bad_dates: RefCell::new(Vec::new()),
        };
        let mut dead = Vec::new();
        for i in x.s..x.e {
            scan_at(&x, i, &mut cands);
            if t[i].is_ascii_digit()
                && x.bound(i)
                && let Some((None, end)) = mixed_at(&t, i, x.e)
            {
                dead.push((i, end));
            }
        }
        let bad = x.bad_dates.take();
        cands.retain(|c| {
            let hit = |&(s, e): &(usize, usize)| c.span.start_char < e && s < c.span.end_char;
            // Inside an invalid full date, no date and no undecided candidate (such as `2/30`)
            // is read. Its digits stay numbers.
            let dated = matches!(c.kind, Some(EntityKind::Date) | None);
            !(dead.iter().any(hit) || (dated && bad.iter().any(hit)))
        });
        emails(&x, &mut cands);
        if let Some(ac) = &ac {
            names_in(ac, &x, &mut cands);
        }
        for f in found
            .iter()
            .filter(|f| !f.block && f.item.kind == OpaqueKind::Url)
        {
            let sp = f.item.span;
            if sp.start_char >= x.s && sp.end_char <= x.e {
                cands.push(x.cand(
                    sp.start_char,
                    sp.end_char,
                    Some(EntityKind::Url),
                    None,
                    None,
                ));
            }
        }
    }

    let (mut entities, mut unknown) = (Vec::new(), Vec::new());
    for c in resolve(cands) {
        match c.kind {
            Some(kind) => entities.push(Entity {
                kind,
                span: c.span,
                sentence: c.sentence,
                value: c.value,
                unit: c.unit,
            }),
            None => unknown.push(Unknown {
                layer: "entities".to_owned(),
                span: c.span,
                reason: Reason::Ambiguous,
            }),
        }
    }
    (entities, unknown)
}

/// Keeps the candidates that do not overlap an earlier kept one: first by start, then longer
/// first, then by kind priority.
fn resolve(mut cands: Vec<Cand>) -> Vec<Cand> {
    cands.sort_by_key(|c| (c.span.start_char, Reverse(c.span.end_char), rank(c.kind)));
    let mut kept: Vec<Cand> = Vec::new();
    for c in cands {
        let clear = kept
            .iter()
            .all(|k| k.span.end_char <= c.span.start_char || c.span.end_char <= k.span.start_char);
        if clear {
            kept.push(c);
        }
    }
    kept
}

fn rank(kind: Option<EntityKind>) -> u8 {
    match kind {
        Some(EntityKind::Name) => 0,
        Some(EntityKind::Email | EntityKind::Url) => 1,
        Some(EntityKind::Quantity) => 2,
        Some(EntityKind::Time) => 3,
        Some(EntityKind::Date) => 4,
        Some(EntityKind::RelativeDate) => 5,
        None => 6,
        Some(EntityKind::Number) => 7,
    }
}

fn scan_at(x: &Sent, i: usize, out: &mut Vec<Cand>) {
    out.extend(relative_at(x, i));
    out.extend(date_at(x, i));
    out.extend(time_at(x, i));
    out.extend(quantity_at(x, i));
    out.extend(number_cand(x, i));
}

fn relative_at(x: &Sent, i: usize) -> Option<Cand> {
    let (t, e) = (x.t, x.e);
    for word in ZH_RELATIVE {
        if let Some(end) = matches_at(t, i, e, word) {
            return Some(x.cand(i, end, Some(EntityKind::RelativeDate), None, None));
        }
    }
    if !x.bound(i) {
        return None;
    }
    let (word, end) = word_at(t, i, e)?;
    if EN_RELATIVE.contains(&word.as_str()) {
        return Some(x.cand(i, end, Some(EntityKind::RelativeDate), None, None));
    }
    if matches!(word.as_str(), "last" | "this" | "next") && end < e && t[end] == ' ' {
        let (second, end2) = word_at(t, end + 1, e)?;
        if matches!(second.as_str(), "week" | "month" | "year") {
            return Some(x.cand(i, end2, Some(EntityKind::RelativeDate), None, None));
        }
    }
    None
}

fn date_at(x: &Sent, i: usize) -> Option<Cand> {
    let ch = x.t[i];
    if !x.bound(i) {
        return None;
    }
    if ch.is_ascii_uppercase() {
        return en_month_date(x, i);
    }
    if ch.is_ascii_digit() {
        return zh_date(x, i)
            .or_else(|| en_day_month(x, i))
            .or_else(|| numeric_date(x, i));
    }
    if zh_digit(ch).is_some() || matches!(ch, '民' | '十' | '百' | '千') {
        if is_zh_num(ch) && !x.zh_start_ok(i) {
            return None;
        }
        return zh_date(x, i);
    }
    None
}

/// `2026年11月10日`, `民國115年3月5日`, `11月10日`, `2026年11月`, `2026年`.
fn zh_date(x: &Sent, i: usize) -> Option<Cand> {
    let (t, e) = (x.t, x.e);
    let year = zh_year(x, i);
    let mut end = year.map_or(i, |(_, k)| k);
    let (mut month, mut day) = (None, None);
    if let Some((m, k)) = int_at(t, end, e)
        && k < e
        && t[k] == '月'
    {
        month = Some(m);
        end = k + 1;
        if let Some((d, k2)) = int_at(t, end, e)
            && k2 < e
            && (t[k2] == '日' || t[k2] == '號')
        {
            day = Some(d);
            end = k2 + 1;
        }
    }
    let year = year.map(|(y, _)| y);
    let ok = match (year, month, day) {
        (Some(y), Some(m), Some(d)) => {
            valid(Some(y), m, d).then(|| format!("{y:04}-{m:02}-{d:02}"))
        }
        (Some(y), Some(m), None) => (1..=12).contains(&m).then(|| format!("{y:04}-{m:02}")),
        (None, Some(m), Some(d)) => valid(None, m, d).then(|| format!("--{m:02}-{d:02}")),
        (Some(y), None, None) => Some(format!("{y:04}")),
        _ => return None,
    };
    let Some(value) = ok else {
        return x.bad_date(i, end);
    };
    Some(x.cand(i, end, Some(EntityKind::Date), Some(value), None))
}

/// A year before 年: four digits (ASCII or digit-by-digit), or 民國 and a number.
fn zh_year(x: &Sent, i: usize) -> Option<(u64, usize)> {
    let (t, e) = (x.t, x.e);
    if t[i] == '民' {
        let after = matches_at(t, i, e, "民國")?;
        let (n, k) = int_at(t, after, e)?;
        return (k < e && t[k] == '年').then_some((n.checked_add(1911)?, k + 1));
    }
    if i + 5 > e || t[i + 4] != '年' {
        return None;
    }
    let four = &t[i..i + 4];
    let value = if four.iter().all(char::is_ascii_digit) {
        four.iter().collect::<String>().parse().ok()?
    } else if four.iter().all(|&c| zh_digit(c).is_some()) {
        zh_value(four)?
    } else {
        return None;
    };
    Some((value, i + 5))
}

/// `November 10, 2026`, `November 10 2026`, `Nov 10`, `May 2026`, `Sept. 3rd`.
fn en_month_date(x: &Sent, i: usize) -> Option<Cand> {
    let (t, e) = (x.t, x.e);
    let (word, wend) = word_at(t, i, e)?;
    let month = month_of(&word)?;
    let mut j = wend;
    let abbreviation = word.len() == 3 || word == "sept";
    if abbreviation && j + 2 < e && t[j] == '.' && t[j + 1] == ' ' && t[j + 2].is_ascii_digit() {
        j += 1;
    }
    if j >= e || t[j] != ' ' {
        return None;
    }
    j += 1;
    if let Some((day, dend)) = day_at(t, j, e) {
        let comma = dend < e && t[dend] == ',';
        let p = if comma { dend + 1 } else { dend };
        if p < e
            && t[p] == ' '
            && let Some((year, yend)) = year4_at(t, p + 1, e)
        {
            return full(x, i, yend, year, month, day);
        }
        return valid(None, month, day).then(|| {
            x.cand(
                i,
                dend,
                Some(EntityKind::Date),
                Some(format!("--{month:02}-{day:02}")),
                None,
            )
        });
    }
    let (year, yend) = year4_at(t, j, e)?;
    Some(x.cand(
        i,
        yend,
        Some(EntityKind::Date),
        Some(format!("{year:04}-{month:02}")),
        None,
    ))
}

/// `10 November 2026`, `5 May`.
fn en_day_month(x: &Sent, i: usize) -> Option<Cand> {
    let (t, e) = (x.t, x.e);
    let (day, dend) = day_at(t, i, e)?;
    if !(dend + 1 < e && t[dend] == ' ' && t[dend + 1].is_ascii_uppercase()) {
        return None;
    }
    let (word, wend) = word_at(t, dend + 1, e)?;
    let month = month_of(&word)?;
    // `10 Nov. 2026`: the dot after an abbreviated month is part of the date, as in en_month_date.
    let abbreviation = word.len() == 3 || word == "sept";
    let dot = abbreviation
        && wend + 2 < e
        && t[wend] == '.'
        && t[wend + 1] == ' '
        && t[wend + 2].is_ascii_digit();
    let sp = if dot { wend + 1 } else { wend };
    if sp < e
        && t[sp] == ' '
        && let Some((year, yend)) = year4_at(t, sp + 1, e)
    {
        return full(x, i, yend, year, month, day);
    }
    valid(None, month, day).then(|| {
        x.cand(
            i,
            wend,
            Some(EntityKind::Date),
            Some(format!("--{month:02}-{day:02}")),
            None,
        )
    })
}

/// `2026-11-10`, `2026/11/10`, `2026.11.10`, `2026-11`, `13/4/2026`, `3/4` (not decided).
fn numeric_date(x: &Sent, i: usize) -> Option<Cand> {
    let (t, e) = (x.t, x.e);
    let j = digits_end(t, i, e);
    if j - i == 4 {
        let year: u64 = t[i..j].iter().collect::<String>().parse().ok()?;
        if j >= e || !matches!(t[j], '-' | '/' | '.') {
            return None;
        }
        let sep = t[j];
        let (m, k) = digits_at(t, j + 1, e, 2)?;
        if k < e && t[k] == sep {
            let (d, k2) = digits_at(t, k + 1, e, 2)?;
            return full(x, i, k2, year, m, d);
        }
        if sep == '-' && (1..=12).contains(&m) {
            return Some(x.cand(
                i,
                k,
                Some(EntityKind::Date),
                Some(format!("{year:04}-{m:02}")),
                None,
            ));
        }
        return None;
    }
    let (a, _) = digits_at(t, i, e, 2)?;
    if j >= e || t[j] != '/' {
        return None;
    }
    let (b, k) = digits_at(t, j + 1, e, 2)?;
    if k < e
        && t[k] == '/'
        && let Some((year, k2)) = year4_at(t, k + 1, e)
    {
        // A part over 12 is the day (`13/4`) or the month (`4/13`). When both are 12 or less,
        // both readings are tried: both real dates is ambiguous, one is that date, neither is a
        // bad date and its digits are numbers.
        return match (a > 12, b > 12) {
            (true, false) => full(x, i, k2, year, b, a),
            (false, true) => full(x, i, k2, year, a, b),
            (false, false) => match (valid(Some(year), a, b), valid(Some(year), b, a)) {
                (true, true) => Some(x.cand(i, k2, None, None, None)),
                (true, false) => full(x, i, k2, year, a, b),
                (false, true) => full(x, i, k2, year, b, a),
                (false, false) => x.bad_date(i, k2),
            },
            (true, true) => x.bad_date(i, k2),
        };
    }
    if k < e && (t[k].is_ascii_digit() || t[k] == '/') {
        return None;
    }
    Some(x.cand(i, k, None, None, None))
}

fn full(x: &Sent, start: usize, end: usize, year: u64, month: u64, day: u64) -> Option<Cand> {
    if !valid(Some(year), month, day) {
        return x.bad_date(start, end);
    }
    Some(x.cand(
        start,
        end,
        Some(EntityKind::Date),
        Some(format!("{year:04}-{month:02}-{day:02}")),
        None,
    ))
}

fn valid(year: Option<u64>, month: u64, day: u64) -> bool {
    let leap = year.is_none_or(|y| (y % 4 == 0 && y % 100 != 0) || y % 400 == 0);
    let days = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => return false,
    };
    (1..=days).contains(&day)
}

fn month_of(word: &str) -> Option<u64> {
    MONTHS
        .iter()
        .position(|m| *m == word || ((word.len() == 3 || word == "sept") && m.starts_with(word)))
        .map(|k| k as u64 + 1)
}

/// One or two digits, then an optional `st nd rd th`.
fn day_at(t: &[char], i: usize, e: usize) -> Option<(u64, usize)> {
    let (day, j) = digits_at(t, i, e, 2)?;
    let suffix = ["st", "nd", "rd", "th"]
        .iter()
        .find_map(|s| matches_at(t, j, e, s));
    Some((day, suffix.unwrap_or(j)))
}

fn time_at(x: &Sent, i: usize) -> Option<Cand> {
    let (t, e) = (x.t, x.e);
    if !x.bound(i) || (is_zh_num(t[i]) && !x.zh_start_ok(i)) {
        return None;
    }
    if let Some((word, end)) = word_at(t, i, e) {
        match word.as_str() {
            "noon" => {
                return Some(x.cand(i, end, Some(EntityKind::Time), Some("12:00".into()), None));
            }
            "midnight" => return Some(x.cand(i, end, None, None, None)),
            _ => {}
        }
    }
    for (marker, group) in ZH_MARKERS {
        if let Some(h0) = matches_at(t, i, e, marker) {
            return clock(x, i, h0, Some(*group));
        }
    }
    clock(x, i, i, None)
}

/// An hour (from `h0`) with `:MM`, `點`, `點半`, `點M分`, or an am/pm marker after it.
fn clock(x: &Sent, start: usize, h0: usize, mut marker: Option<Marker>) -> Option<Cand> {
    let (t, e) = (x.t, x.e);
    let (h, k) = int_at(t, h0, e)?;
    let mut minute = 0;
    let mut end;
    if k + 3 <= e
        && t[k] == ':'
        && t[k + 1..k + 3].iter().all(char::is_ascii_digit)
        && !(k + 3 < e && t[k + 3].is_ascii_digit())
    {
        minute = t[k + 1..k + 3].iter().collect::<String>().parse().ok()?;
        end = k + 3;
        if end + 3 <= e
            && t[end] == ':'
            && t[end + 1..end + 3].iter().all(char::is_ascii_digit)
            && !(end + 3 < e && t[end + 3].is_ascii_digit())
        {
            let second: u64 = t[end + 1..end + 3]
                .iter()
                .collect::<String>()
                .parse()
                .ok()?;
            if second > 59 {
                return None;
            }
            end += 3;
        }
        if let Some((m, after)) = after_marker(t, end, e) {
            marker = Some(m);
            end = after;
        }
    } else if k < e && (t[k] == '點' || t[k] == '点') {
        end = k + 1;
        // `一點` alone with no marker is "a bit" (`有一點累`), not one o'clock.
        let a_bit = marker.is_none() && h0 + 1 == k && t[h0] == '一';
        if end < e && t[end] == '半' {
            minute = 30;
            end += 1;
        } else if let Some((m, j)) = int_at(t, end, e)
            && j < e
            && t[j] == '分'
        {
            minute = m;
            end = j + 1;
        } else if a_bit {
            return None;
        }
    } else {
        let (m, after) = after_marker(t, k, e)?;
        marker = Some(m);
        end = after;
    }
    if h > 24 || minute > 59 {
        return None;
    }
    let (kind, value) = match decide(h, minute, marker) {
        Some(v) => (Some(EntityKind::Time), Some(v)),
        None => (None, None),
    };
    Some(x.cand(start, end, kind, value, None))
}

fn after_marker(t: &[char], k: usize, e: usize) -> Option<(Marker, usize)> {
    let p = if k < e && t[k] == ' ' { k + 1 } else { k };
    AM_PM
        .iter()
        .find_map(|&(word, m)| matches_at(t, p, e, word).map(|end| (m, end)))
}

/// The 24-hour time, or None when the text does not decide it (see the decision table).
fn decide(h: u64, m: u64, marker: Option<Marker>) -> Option<String> {
    let hour = match marker {
        None if h >= 13 || h == 0 => h,
        None => return None,
        Some(_) if h >= 13 => return None,
        Some(Marker::Morning(strict)) => match h {
            1..=11 => h,
            12 if strict => 0,
            _ => return None,
        },
        Some(Marker::Afternoon(strict)) => match h {
            1..=11 => h + 12,
            12 if strict => 12,
            _ => return None,
        },
        Some(Marker::Noon) => match h {
            11 | 12 => h,
            1 | 2 => h + 12,
            _ => return None,
        },
    };
    Some(format!("{hour:02}:{m:02}"))
}

fn quantity_at(x: &Sent, i: usize) -> Option<Cand> {
    let (t, e) = (x.t, x.e);
    if let Some(j) = matches_at(t, i, e, PERCENT_PREFIX) {
        let num = number_at(x, j)?;
        return Some(x.cand(
            i,
            num.end,
            Some(EntityKind::Quantity),
            Some(num.value),
            Some("%".to_owned()),
        ));
    }
    for cur in CURRENCY {
        if let Some(j) = matches_at(t, i, e, cur)
            && (!cur.is_ascii() || x.bound(i))
        {
            let num = number_at(x, j)?;
            return Some(x.cand(
                i,
                num.end,
                Some(EntityKind::Quantity),
                Some(num.value),
                Some(cur.to_ascii_lowercase()),
            ));
        }
    }
    let num = number_at(x, i)?;
    if num.bare_one {
        return None;
    }
    let (j, attached) = if num.end < e && t[num.end] == ' ' {
        (num.end + 1, false)
    } else {
        (num.end, true)
    };
    let (unit, uend) = unit_at(t, j, e, attached)?;
    Some(x.cand(
        i,
        uend,
        Some(EntityKind::Quantity),
        Some(num.value),
        Some(unit.to_owned()),
    ))
}

/// The longest unit at `j`. Units marked as attached-only need no space before them.
fn unit_at(t: &[char], j: usize, e: usize, attached: bool) -> Option<(&'static str, usize)> {
    UNITS
        .iter()
        .filter(|(_, only_attached)| attached || !only_attached)
        .filter_map(|&(unit, _)| matches_at(t, j, e, unit).map(|end| (unit, end)))
        .max_by_key(|(unit, _)| unit.chars().count())
}

fn number_cand(x: &Sent, i: usize) -> Option<Cand> {
    let num = number_at(x, i)?;
    let (t, ch) = (x.t, x.t[i]);
    if num.bare_one {
        return Some(x.cand(i, num.end, None, None, None));
    }
    let ok = if ch.is_ascii_digit() {
        !(num.end < x.e && t[num.end].is_ascii_alphabetic())
    } else if ch.is_ascii_alphabetic() {
        true
    } else {
        // Chinese numerals need a measure word. A run that starts after 第 is rejected in number_at.
        num.end < x.e && MEASURE.contains(t[num.end])
    };
    ok.then(|| x.cand(i, num.end, Some(EntityKind::Number), Some(num.value), None))
}

fn number_at(x: &Sent, i: usize) -> Option<Num> {
    // A prefix such as `$` that ends the text or the sentence has no number after it.
    if i >= x.e {
        return None;
    }
    let (t, e) = (x.t, x.e);
    let ch = t[i];
    if ch.is_ascii_digit() {
        if !x.bound(i) {
            return None;
        }
        if let Some((value, end)) = mixed_at(t, i, e) {
            // None is a run that cannot be read: no number, and extract drops what is inside.
            return Some(Num {
                value: value?,
                end,
                bare_one: false,
            });
        }
        let (value, end) = ascii_number_at(t, i, e);
        return Some(Num {
            value,
            end,
            bare_one: false,
        });
    }
    if zh_digit(ch).is_some() || matches!(ch, '十' | '百' | '千') {
        // A run starts only where the shared guard allows it, so `十` in `三十` is not a number.
        if !x.zh_start_ok(i) {
            return None;
        }
        let end = zh_run_end(t, i, e);
        let value = zh_value(&t[i..end])?;
        // `三點五` is 3.5: 點 and then Chinese digits, read one by one.
        if let Some(frac_end) = zh_fraction_end(t, end, e) {
            let frac: String = t[end + 1..frac_end]
                .iter()
                .filter_map(|&ch| zh_digit(ch))
                .map(|d| char::from(b'0' + d as u8))
                .collect();
            return Some(Num {
                value: format!("{value}.{frac}"),
                end: frac_end,
                bare_one: false,
            });
        }
        return Some(Num {
            value: value.to_string(),
            end,
            bare_one: false,
        });
    }
    if ch.is_ascii_alphabetic() && x.bound(i) {
        return en_number_at(t, i, e);
    }
    None
}

/// The end of `點` and one or more digits (`〇零一二三四五六七八九`) at `j`. 兩 is not a digit here.
fn zh_fraction_end(t: &[char], j: usize, e: usize) -> Option<usize> {
    if j >= e || !matches!(t[j], '點' | '点') {
        return None;
    }
    let end = (j + 1..e)
        .find(|&k| zh_digit(t[k]).is_none() || matches!(t[k], '兩' | '两'))
        .unwrap_or(e);
    (end > j + 1).then_some(end)
}

/// An ASCII number with a Chinese numeral char right after it: `5千`, `3萬5千`, `1.5億`,
/// `3千萬`. Returns the end of the whole run, and its value when the rule reads it. The rule is
/// `<ascii number>` then a chain of numerals that ends in a multiplier (`十百千萬億`). A decimal
/// may have one multiplier only. Anything else (`3萬5`, `3五千`, `3萬4萬`) has no value.
/// None when no numeral follows the ASCII number.
fn mixed_at(t: &[char], i: usize, e: usize) -> Option<(Option<String>, usize)> {
    let (value, j) = ascii_number_at(t, i, e);
    if j >= e || !is_zh_num(t[j]) {
        return None;
    }
    let (mut toks, mut zeros) = (Vec::new(), Vec::new());
    let mut k = j;
    while k < e {
        if let Some(tok) = zh_tok(t[k]) {
            zeros.push(matches!(t[k], '零' | '〇'));
            toks.push(tok);
            k += 1;
        } else if t[k].is_ascii_digit() {
            let m = digits_end(t, k, e);
            let Ok(n) = t[k..m].iter().collect::<String>().parse() else {
                return Some((None, m));
            };
            zeros.push(false);
            toks.push(Tok::Add(n));
            k = m;
        } else {
            break;
        }
    }
    let ends_in_multiplier = matches!(toks.last(), Some(Tok::Mul(_) | Tok::Big(_)));
    let read = if !ends_in_multiplier {
        None
    } else if value.contains('.') {
        match toks[..] {
            [Tok::Mul(m) | Tok::Big(m)] => Some(shift(&value, m.ilog10() as usize)),
            _ => None,
        }
    } else {
        let first = value.parse().ok();
        first
            .and_then(|n| {
                toks.insert(0, Tok::Add(n));
                zeros.insert(0, false);
                ordered_value(&toks, &zeros)
            })
            .map(|v| v.to_string())
    };
    Some((read, k))
}

/// `1,000`, `3.5`.
fn ascii_number_at(t: &[char], i: usize, e: usize) -> (String, usize) {
    let mut j = digits_end(t, i, e);
    let mut out: String = t[i..j].iter().collect();
    while j + 4 <= e
        && t[j] == ','
        && t[j + 1..j + 4].iter().all(char::is_ascii_digit)
        && (j + 4 == e || !t[j + 4].is_ascii_digit())
    {
        out.extend(&t[j + 1..j + 4]);
        j += 4;
    }
    if j + 1 < e && t[j] == '.' && t[j + 1].is_ascii_digit() {
        let k = digits_end(t, j + 1, e);
        out.push('.');
        out.extend(&t[j + 1..k]);
        j = k;
    }
    (out, j)
}

/// Multiplies a decimal string by 10^k, keeping it decimal only when a fraction remains.
fn shift(s: &str, k: usize) -> String {
    let (int, frac) = s.split_once('.').unwrap_or((s, ""));
    let mut digits = format!("{int}{frac}");
    let point = int.len() + k;
    if digits.len() < point {
        digits.push_str(&"0".repeat(point - digits.len()));
    }
    let (ip, fp) = digits.split_at(point);
    let ip = ip.trim_start_matches('0');
    let ip = if ip.is_empty() { "0" } else { ip };
    let fp = fp.trim_end_matches('0');
    if fp.is_empty() {
        ip.to_owned()
    } else {
        format!("{ip}.{fp}")
    }
}

/// English number words, joined by one space or hyphen. Returns None if the first word is not one.
fn en_number_at(t: &[char], i: usize, e: usize) -> Option<Num> {
    let (mut toks, mut words, mut ends) = (Vec::new(), Vec::new(), Vec::new());
    let (mut j, mut last) = (i, i);
    let (mut hundred, mut big) = (false, 0);
    while let Some((word, end)) = word_at(t, j, e)
        && let Some(tok) = en_tok(&word)
    {
        // A word that breaks the order ends the number before it, and reads as a new number.
        if !en_follows(toks.last().copied(), tok, hundred, big) {
            // A multiplier that cannot join this number belongs to the words right before it
            // (`two hundred three hundred` is 200 and 300), so those words are given back.
            if !matches!(tok, Tok::Add(_))
                && let Some(keep) = toks.iter().rposition(|k| !matches!(k, Tok::Add(_)))
            {
                toks.truncate(keep + 1);
                words.truncate(keep + 1);
                ends.truncate(keep + 1);
                last = ends[keep];
            }
            break;
        }
        match tok {
            Tok::Mul(_) => hundred = true,
            Tok::Big(m) => {
                hundred = false;
                big = m;
            }
            Tok::Add(_) => {}
        }
        toks.push(tok);
        words.push(word);
        ends.push(end);
        last = end;
        if last + 1 < e && (t[last] == '-' || t[last] == ' ') {
            j = last + 1;
        } else {
            break;
        }
    }
    if toks.is_empty() {
        return None;
    }
    let value = combine(&toks)?;
    Some(Num {
        value: value.to_string(),
        end: last,
        bare_one: words.len() == 1 && words[0] == "one",
    })
}

/// Whether English number word `tok` may follow `prev` (None at the start) in one number. A group
/// starts at the beginning or after thousand, million or billion. A unit (0 to 9) follows a group
/// start, a tens word or hundred, and only 1 to 9 after a tens word or hundred (`twenty-one`,
/// `two hundred three`). A teen or tens word
/// follows a group start or hundred. Hundred follows a non-zero unit, teen or tens word, once per
/// group. Thousand, million or billion follows a non-zero unit, teen, tens or hundred, and is
/// smaller than the one before. `hundred` says whether this group has one, and `big` is the last
/// thousand, million or billion (0 for none).
fn en_follows(prev: Option<Tok>, tok: Tok, hundred: bool, big: u64) -> bool {
    match tok {
        Tok::Add(v) if v < 10 => match prev {
            None | Some(Tok::Big(_)) => true,
            Some(Tok::Add(p)) => (20..=90).contains(&p) && v > 0,
            Some(Tok::Mul(_)) => v > 0,
        },
        Tok::Add(_) => matches!(prev, None | Some(Tok::Big(_)) | Some(Tok::Mul(_))),
        Tok::Mul(_) => !hundred && matches!(prev, Some(Tok::Add(p)) if p > 0),
        Tok::Big(m) => {
            let after_number = match prev {
                Some(Tok::Add(p)) => p > 0,
                Some(Tok::Mul(_)) => true,
                _ => false,
            };
            after_number && (big == 0 || m < big)
        }
    }
}

fn en_tok(word: &str) -> Option<Tok> {
    if let Some(v) = EN_SMALL.iter().position(|w| *w == word) {
        return Some(Tok::Add(v as u64));
    }
    if let Some(k) = EN_TENS.iter().position(|w| *w == word) {
        return Some(Tok::Add((k as u64 + 2) * 10));
    }
    Some(match word {
        "hundred" => Tok::Mul(100),
        "thousand" => Tok::Big(1_000),
        "million" => Tok::Big(1_000_000),
        "billion" => Tok::Big(1_000_000_000),
        _ => return None,
    })
}

fn is_zh_num(ch: char) -> bool {
    zh_digit(ch).is_some() || "十百千萬万億亿".contains(ch)
}

fn zh_run_end(t: &[char], i: usize, e: usize) -> usize {
    (i..e).find(|&k| !is_zh_num(t[k])).unwrap_or(e)
}

fn zh_digit(ch: char) -> Option<u64> {
    Some(match ch {
        '〇' | '零' => 0,
        '一' => 1,
        '二' | '兩' | '两' => 2,
        '三' => 3,
        '四' => 4,
        '五' => 5,
        '六' => 6,
        '七' => 7,
        '八' => 8,
        '九' => 9,
        _ => return None,
    })
}

/// Digit-by-digit when every char is a digit and there are at least two, otherwise the standard
/// reading (`一百零五` is 105, `三萬五千` is 35000).
fn zh_value(run: &[char]) -> Option<u64> {
    // 兩 and 两 are a count (`一兩天`, `兩三個`), so they are never read digit by digit.
    if run.len() >= 2
        && run
            .iter()
            .all(|&ch| zh_digit(ch).is_some() && !matches!(ch, '兩' | '两'))
    {
        let digits: String = run
            .iter()
            .filter_map(|&ch| zh_digit(ch))
            .map(|d| char::from(b'0' + d as u8))
            .collect();
        return digits.parse().ok();
    }
    let toks: Vec<Tok> = run.iter().map(|&ch| zh_tok(ch)).collect::<Option<_>>()?;
    let zeros: Vec<bool> = run.iter().map(|&ch| matches!(ch, '零' | '〇')).collect();
    ordered_value(&toks, &zeros)
}

/// The value of numeral tokens, or None when their order is not a number. `zeros[k]` is true
/// when token `k` is 零 or 〇.
fn ordered_value(toks: &[Tok], zeros: &[bool]) -> Option<u64> {
    if matches!(toks.first()?, Tok::Big(_)) {
        return None;
    }
    // Two plain digits in a row must be digit-by-digit (`二三十` is rejected), except after 零
    // (`一百零五`). A multiplier must be smaller than the one before it in its group, so
    // `十二十`, `三十四十` and `五百六百` are rejected instead of merged into one wrong number.
    // Group sizes must strictly decrease (億, then 萬, then the rest), so `三萬四萬`, `一萬二億`
    // and `三萬億` are rejected. The check is here, not in `combine`, so English is unchanged.
    let mut last_mul: Option<u64> = None;
    let mut last_big: Option<u64> = None;
    for (k, tok) in toks.iter().enumerate() {
        match tok {
            Tok::Add(_) if k > 0 && matches!(toks[k - 1], Tok::Add(_)) && !zeros[k - 1] => {
                return None;
            }
            Tok::Mul(m) => {
                if last_mul.is_some_and(|p| *m >= p) {
                    return None;
                }
                last_mul = Some(*m);
            }
            Tok::Big(m) => {
                if last_big.is_some_and(|p| *m >= p) {
                    return None;
                }
                last_big = Some(*m);
                last_mul = None;
            }
            Tok::Add(_) => {}
        }
    }
    combine(toks)
}

fn zh_tok(ch: char) -> Option<Tok> {
    if let Some(v) = zh_digit(ch) {
        return Some(Tok::Add(v));
    }
    Some(match ch {
        '十' => Tok::Mul(10),
        '百' => Tok::Mul(100),
        '千' => Tok::Mul(1000),
        '萬' | '万' => Tok::Big(10_000),
        '億' | '亿' => Tok::Big(100_000_000),
        _ => return None,
    })
}

/// `Mul` scales the digits before it inside a group, `Big` closes a group of 萬 or 億 size.
fn combine(toks: &[Tok]) -> Option<u64> {
    let (mut result, mut section, mut number) = (0u64, 0u64, 0u64);
    for tok in toks {
        match *tok {
            Tok::Add(v) => number = number.checked_add(v)?,
            Tok::Mul(m) => {
                section = section.checked_add(number.max(1).checked_mul(m)?)?;
                number = 0;
            }
            Tok::Big(m) => {
                // 萬 or 億 needs a group before it, so `萬`, `三億萬` and `億億` are not numbers.
                let group = section.checked_add(number)?;
                if group == 0 {
                    return None;
                }
                result = result.checked_add(group.checked_mul(m)?)?;
                section = 0;
                number = 0;
            }
        }
    }
    result.checked_add(section)?.checked_add(number)
}

fn emails(x: &Sent, out: &mut Vec<Cand>) {
    let t = x.t;
    for at in x.s + 1..x.e {
        if t[at] != '@' {
            continue;
        }
        let mut ls = at;
        while ls > x.s && (t[ls - 1].is_ascii_alphanumeric() || "._%+-".contains(t[ls - 1])) {
            ls -= 1;
        }
        if ls == at || t[ls] == '.' || t[at - 1] == '.' {
            continue;
        }
        let mut de = at + 1;
        while de < x.e && (t[de].is_ascii_alphanumeric() || t[de] == '-' || t[de] == '.') {
            de += 1;
        }
        while de > at + 1 && t[de - 1] == '.' {
            de -= 1;
        }
        let domain: String = t[at + 1..de].iter().collect();
        let labels: Vec<&str> = domain.split('.').collect();
        let last = labels.last().copied().unwrap_or("");
        let ok = labels.len() >= 2
            && labels
                .iter()
                .all(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
            && last.len() >= 2
            && last.chars().all(|c| c.is_ascii_alphabetic());
        if ok {
            out.push(x.cand(ls, de, Some(EntityKind::Email), None, None));
        }
    }
}

fn names_in(ac: &AhoCorasick, x: &Sent, out: &mut Vec<Cand>) {
    let t = x.t;
    let text: String = t[x.s..x.e].iter().collect();
    for m in ac.find_iter(&text) {
        let start = x.s + text[..m.start()].chars().count();
        let end = start + text[m.start()..m.end()].chars().count();
        let first_ok = !(t[start].is_ascii_alphanumeric()
            && start > 0
            && t[start - 1].is_ascii_alphanumeric());
        let last_ok = !(t[end - 1].is_ascii_alphanumeric()
            && end < t.len()
            && t[end].is_ascii_alphanumeric());
        if first_ok && last_ok {
            out.push(x.cand(start, end, Some(EntityKind::Name), None, None));
        }
    }
}

/// True when `word` occurs at `i` (ASCII case-insensitive) and does not run into a letter.
fn matches_at(t: &[char], i: usize, e: usize, word: &str) -> Option<usize> {
    let n = word.chars().count();
    if i + n > e
        || !word
            .chars()
            .zip(&t[i..i + n])
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
    {
        return None;
    }
    let end = i + n;
    if word.ends_with(|ch: char| ch.is_ascii_alphabetic())
        && end < e
        && t[end].is_ascii_alphabetic()
    {
        return None;
    }
    Some(end)
}

/// The lowercased ASCII word starting at `i`, and its end.
fn word_at(t: &[char], i: usize, e: usize) -> Option<(String, usize)> {
    if i >= e || !t[i].is_ascii_alphabetic() {
        return None;
    }
    let j = (i..e).find(|&k| !t[k].is_ascii_alphabetic()).unwrap_or(e);
    Some((t[i..j].iter().collect::<String>().to_ascii_lowercase(), j))
}

fn digits_end(t: &[char], i: usize, e: usize) -> usize {
    (i..e).find(|&k| !t[k].is_ascii_digit()).unwrap_or(e).max(i)
}

/// 1 to `max` ASCII digits, not followed by another digit.
fn digits_at(t: &[char], i: usize, e: usize, max: usize) -> Option<(u64, usize)> {
    let j = digits_end(t, i, e);
    if j == i || j - i > max {
        return None;
    }
    Some((t[i..j].iter().collect::<String>().parse().ok()?, j))
}

/// Four ASCII digits.
fn year4_at(t: &[char], i: usize, e: usize) -> Option<(u64, usize)> {
    digits_at(t, i, e, 4).filter(|&(_, j)| j - i == 4)
}

/// Digits (ASCII, or a Chinese numeral run) as an integer.
fn int_at(t: &[char], i: usize, e: usize) -> Option<(u64, usize)> {
    if i >= e {
        return None;
    }
    if t[i].is_ascii_digit() {
        return digits_at(t, i, e, usize::MAX);
    }
    if is_zh_num(t[i]) && !matches!(t[i], '萬' | '万' | '億' | '亿') {
        let j = zh_run_end(t, i, e);
        return Some((zh_value(&t[i..j])?, j));
    }
    None
}
