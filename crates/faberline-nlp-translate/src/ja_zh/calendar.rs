//! Bind a calendar day to its case particles before noun-phrase grouping.
use super::{Atom, Upos, normalize_decimal_digits, selects_time_value};
use crate::opaque::parse_cjk_or_ascii_num;

const DATE: &str = "__ja_calendar_day";

pub(super) fn is_date(atom: &Atom) -> bool {
    atom.lemma.as_deref() == Some(DATE)
}

pub(super) fn is_question(atom: &Atom) -> bool {
    is_date(atom) && matches!(atom.surface.as_str(), "幾號" | "星期幾")
}

fn is_numeral(atom: &Atom) -> bool {
    !atom.is_locked
        && (atom.upos == Some(Upos::Num)
            || atom.upos == Some(Upos::Noun)
                && !atom.surface.is_empty()
                && atom.surface.chars().all(|c| ('０'..='９').contains(&c)))
}

fn calendar_number(text: &str) -> Option<u32> {
    if text.is_empty()
        || text.chars().count() > 7
        || !text.chars().all(|c| {
            matches!(c, '0'..='9' | '０'..='９' | '〇' | '零' | '一' | '二' | '三' | '四' |
            '五' | '六' | '七' | '八' | '九' | '十' | '百' | '千')
        })
    {
        return None;
    }
    parse_cjk_or_ascii_num(text)
}

// In 二十五日, 二 can attach to 十, while 十 and 五 both attach to 日.
fn number_before(atoms: &[Atom], end: usize) -> Option<(usize, String)> {
    let head = atoms[end].doc_tok_idx?;
    let mut start = end;
    while let Some(i) = start.checked_sub(1) {
        let part = &atoms[i];
        if !is_numeral(part)
            || part.span.end_char != atoms[start].span.start_char
            || !matches!(part.deprel.as_str(), "nmod" | "nummod" | "compound")
            || !part
                .head
                .is_some_and(|h| part.doc_tok_idx.is_some_and(|p| p < h && h <= head))
        {
            break;
        }
        start = i;
    }
    (start != end).then(|| {
        (
            start,
            atoms[start..end]
                .iter()
                .map(|a| a.surface.as_str())
                .collect(),
        )
    })
}

fn month_before(atoms: &[Atom], start: usize, day: &Atom) -> Option<(usize, String)> {
    let i = start.checked_sub(1)?;
    let month = &atoms[i];
    if month.is_locked
        || month.span.end_char != atoms[start].span.start_char
        || month.head != day.head
        || month.deprel != "advmod"
    {
        return None;
    }
    let number = month.surface.strip_suffix('月')?;
    let (start, number) = if number.is_empty() {
        let n = i.checked_sub(1)?;
        let num = &atoms[n];
        if !is_numeral(num)
            || num.span.end_char != month.span.start_char
            || !(num.head == month.doc_tok_idx || num.head == day.head)
        {
            return None;
        }
        (n, num.surface.as_str())
    } else {
        (i, number)
    };
    ((1..=12).contains(&calendar_number(number)?))
        .then(|| (start, format!("{}月", normalize_decimal_digits(number))))
}

fn linked_case(case: &Atom, day: &Atom, adjacent_to: &Atom, surface: &str) -> bool {
    !case.is_locked
        && case.upos == Some(Upos::Adp)
        && case.surface == surface
        && case.deprel == "case"
        && day.doc_tok_idx.is_some()
        && case.head == day.doc_tok_idx
        && case.span.start_char == adjacent_to.span.end_char
}

// A year that modifies this month belongs inside the same date phrase.
// Leave other attached modifiers and era years to their normal renderer.
fn dated_month(atoms: &[Atom], start: usize, month: String) -> Option<(usize, String)> {
    let Some((year, before)) = atoms[..start].split_last() else {
        return Some((start, month));
    };
    if year.head != atoms[start].doc_tok_idx
        || year.span.end_char != atoms[start].span.start_char
        || year.deprel != "advmod"
    {
        return Some((start, month));
    }
    if year.is_locked {
        return None;
    }
    let relative = match year.surface.as_str() {
        "去年" => "去年",
        "今年" => "今年",
        "来年" => "明年",
        "年" => {
            let mut first = before.len();
            while first > 0
                && is_numeral(&atoms[first - 1])
                && atoms[first - 1].span.end_char == atoms[first].span.start_char
            {
                first -= 1;
            }
            let digits: String = before[first..].iter().map(|a| a.surface.as_str()).collect();
            if !(1000..=9999).contains(&calendar_number(&digits)?) {
                return None;
            }
            return Some((
                first,
                format!("{}年{month}", normalize_decimal_digits(&digits)),
            ));
        }
        _ => return None,
    };
    Some((start - 1, format!("{relative}{month}")))
}

fn expression(atoms: &[Atom], end: usize) -> Option<(usize, String)> {
    let day = &atoms[end];
    if day.is_locked || day.upos != Some(Upos::Noun) {
        return None;
    }
    if day.surface == "日間" {
        let (start, n) = number_before(atoms, end)?;
        let count = if n == "二" {
            "兩".to_owned()
        } else {
            normalize_decimal_digits(&n)
        };
        return Some((start, format!("{count}天")));
    }
    if day.surface != "日" || !matches!(day.deprel.as_str(), "obl" | "iobj") {
        return None;
    }
    let case = atoms.get(end + 1)?;
    let deadline = linked_case(case, day, day, "まで")
        && atoms
            .get(end + 2)
            .is_some_and(|ni| linked_case(ni, day, case, "に"));
    if !deadline && !linked_case(case, day, day, "に") {
        return None;
    }
    let verb = atoms
        .iter()
        .find(|a| a.doc_tok_idx == day.head && !a.is_locked && a.upos == Some(Upos::Verb))?;
    if !deadline && selects_time_value(verb.lemma.as_deref().unwrap_or(&verb.surface)) {
        return None;
    }
    // Counted repetitions turn 何日 into an interval: 何日に一回, 何日に三度.
    let frequency = !deadline
        && atoms[end + 2..].iter().any(|a| {
            a.head == verb.doc_tok_idx
                && !a.is_locked
                && ["回", "度"].iter().any(|suffix| {
                    a.surface
                        .strip_suffix(suffix)
                        .is_some_and(|n| n.is_empty() || calendar_number(n).is_some())
                })
        });
    let linked_modifier = |a: &Atom, next: &Atom| {
        !a.is_locked
            && a.deprel == "compound"
            && a.head == day.doc_tok_idx
            && a.span.end_char == next.span.start_char
    };
    let i = end.checked_sub(1)?;
    let prev = &atoms[i];
    let (q, text) =
        if prev.surface == "曜" && prev.upos == Some(Upos::Noun) && linked_modifier(prev, day) {
            (i.checked_sub(1)?, "星期幾")
        } else {
            (i, "幾號")
        };
    if atoms[q].surface == "何"
        && atoms[q].upos == Some(Upos::Pron)
        && linked_modifier(&atoms[q], &atoms[q + 1])
    {
        return (!(text == "幾號" && frequency)).then(|| (q, text.to_owned()));
    }
    let (start, n) = number_before(atoms, end)?;
    let value = calendar_number(&n)?;
    if !(1..=31).contains(&value) {
        return None;
    }
    let n = normalize_decimal_digits(&n);
    if let Some((start, month)) = month_before(atoms, start, day) {
        let (start, month) = dated_month(atoms, start, month)?;
        return Some((start, format!("{month}{n}日")));
    }
    // A bare numbered day followed by a count can express a rate (二日に一回).
    if frequency || (!deadline && value == 1 && atoms[end + 2..].iter().any(is_numeral)) {
        return None;
    }
    Some((start, format!("{n}號")))
}

pub(super) fn normalize(atoms: &mut Vec<Atom>) {
    let mut end = atoms.len();
    while end > 0 {
        end -= 1;
        if let Some((start, text)) = expression(atoms, end) {
            let mut day = atoms[end].clone();
            day.lemma = (day.surface == "日").then(|| DATE.to_owned());
            day.surface = text;
            day.is_locked = true;
            day.span.start_char = atoms[start].span.start_char;
            atoms.splice(start..=end, [day]);
            end = start;
        }
    }
}
