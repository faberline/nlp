//! Resolve dates that share an explicit month within a list or across a sentence boundary.
use super::{
    Atom, DATE, DATE_LIST, Upos, adjacent_count, calendar_number, is_numeral, linked_case,
    month_text, normalize_decimal_digits,
};
use faberline_nlp::doc::Document;

struct NumberedDay {
    start: usize,
    end: usize,
    month: Option<String>,
    number: String,
}

// At list boundaries, the parser can attach the first month to the second
// month and a comma-terminated numeral directly to the verb. Read the date
// spelling only inside a contiguous noun phrase of calendar units.
fn day_at(atoms: &[Atom], start: usize) -> Option<NumberedDay> {
    let mut i = start;
    let first = atoms.get(i)?;
    let mut month = month_text(&first.surface);
    if month.is_some() {
        i += 1;
    } else if is_numeral(first) && atoms.get(i + 1).is_some_and(|a| a.surface == "月") {
        month = Some(month_text(&format!("{}月", first.surface))?);
        i += 2;
    }
    if month.is_some() && atoms.get(i).is_some_and(|a| a.surface == "の") {
        if !linked_case(&atoms[i], &atoms[i - 1], &atoms[i - 1], "の") {
            return None;
        }
        i += 1;
    }
    let number_start = i;
    while atoms.get(i).is_some_and(is_numeral) {
        i += 1;
    }
    let day = atoms.get(i)?;
    let joined = day.surface.strip_suffix('日')?;
    if !matches!(day.upos, Some(Upos::Noun | Upos::Adv))
        || atoms[start..=i].iter().any(|a| a.is_locked)
        || atoms[start..=i]
            .windows(2)
            .any(|w| w[0].span.end_char != w[1].span.start_char)
    {
        return None;
    }
    let mut raw: String = atoms[number_start..i]
        .iter()
        .map(|a| a.surface.as_str())
        .collect();
    raw.push_str(joined);
    (1..=31)
        .contains(&calendar_number(&raw)?)
        .then(|| NumberedDay {
            start,
            end: i,
            month,
            number: normalize_decimal_digits(&raw),
        })
}

fn render(day: &NumberedDay) -> String {
    format!(
        "{}{}日",
        day.month.as_deref().unwrap_or_default(),
        day.number
    )
}

pub(super) fn spelled_report_day(spelling: &str) -> Option<String> {
    let raw = spelling.strip_suffix('日')?;
    let (month, number) = if let Some((month, day)) = raw.split_once('月') {
        (
            Some(super::month_text(&format!("{month}月"))?),
            day.strip_prefix('の').unwrap_or(day),
        )
    } else {
        (None, raw)
    };
    if !(1..=31).contains(&super::calendar_number(number)?) {
        return None;
    }
    let number = super::normalize_decimal_digits(number);
    Some(match month {
        Some(month) => format!("{month}{number}日"),
        None => format!("{number}號"),
    })
}

pub(super) fn report_day(words: &[Atom]) -> Option<String> {
    if let [wh, rest @ .., day] = words
        && !wh.is_locked
        && wh.surface == "何"
        && wh.head == day.doc_tok_idx
        && !day.is_locked
        && day.surface == "日"
    {
        return match rest {
            [] => Some("幾號".to_owned()),
            [weekday]
                if !weekday.is_locked
                    && weekday.surface == "曜"
                    && weekday.head == day.doc_tok_idx =>
            {
                Some("星期幾".to_owned())
            }
            _ => None,
        };
    }
    if let [day] = words
        && super::is_date(day)
    {
        return Some(day.surface.clone());
    }
    if words.iter().any(|a| a.is_locked)
        || !words
            .windows(2)
            .all(|p| p[0].span.end_char == p[1].span.start_char)
    {
        return None;
    }
    spelled_report_day(&words.iter().map(|a| a.surface.as_str()).collect::<String>())
}

pub(super) fn dated_atom(atom: &Atom) -> bool {
    !atom.is_locked
        && matches!(atom.upos, Some(Upos::Noun | Upos::Adv))
        && spelled_report_day(&atom.surface).is_some()
}

fn bind(atoms: &mut Vec<Atom>, start: usize, end: usize, text: String, tag: &str) {
    let mut day = atoms[end].clone();
    day.is_locked = true;
    day.surface = text;
    day.lemma = Some(tag.to_owned());
    day.span.start_char = atoms[start].span.start_char;
    atoms.splice(start..=end, [day]);
}

fn bind_month_dates(atoms: &mut Vec<Atom>) {
    let mut i = 0;
    while i < atoms.len() {
        let start = i;
        i += 1;
        let Some(first) = day_at(atoms, start).filter(|day| day.month.is_some()) else {
            continue;
        };
        let mut text = render(&first);
        let mut end = first.end;
        let mut examples = false;
        while let Some(join) = atoms.get(end + 1) {
            let connector = match join.surface.as_str() {
                "と" if linked_case(join, &atoms[end], &atoms[end], "と") => "和",
                "や" if linked_case(join, &atoms[end], &atoms[end], "や") => {
                    examples = true;
                    "、"
                }
                "、" | "・" if !join.is_locked => "、",
                _ => break,
            };
            let Some(next) = day_at(atoms, end + 2) else {
                break;
            };
            if join.span.start_char != atoms[end].span.end_char
                || join.span.end_char != atoms[next.start].span.start_char
            {
                break;
            }
            text.push_str(connector);
            text.push_str(&render(&next));
            end = next.end;
        }
        let Some(case) = atoms.get(end + 1) else {
            continue;
        };
        let explicit_case = matches!(case.surface.as_str(), "に" | "まで")
            && linked_case(case, &atoms[end], &atoms[end], &case.surface);
        if end == first.end {
            // A single token such as 七日 still carries its written month.
            // Standalone 七日 and duration units do not establish a date.
            if atoms[end].surface != "日"
                && matches!(atoms[end].deprel.as_str(), "obl" | "iobj")
                && explicit_case
                && let Some((start, text)) = super::dated_month(atoms, start, text)
            {
                bind(atoms, start, end, text, DATE);
            }
            continue;
        }
        // A detached list can be resumed by それぞれ + counted ずつ.
        let distributive = atoms
            .get(end + 2)
            .filter(|a| !a.is_locked && a.surface == "それぞれ" && a.deprel == "advmod");
        if !explicit_case {
            let Some(each) = distributive.filter(|a| {
                !case.is_locked
                    && case.surface == "、"
                    && atoms[end].head == a.doc_tok_idx
                    && atoms[end + 3..]
                        .iter()
                        .any(|p| !p.is_locked && p.surface == "ずつ")
            }) else {
                continue;
            };
            // Adopt the shared predicate. Use the two source markers for the
            // normalized temporal case before the clause splitter sees the comma.
            let head = each.head;
            let end_char = each.span.end_char;
            let date = &mut atoms[end];
            date.head = head;
            date.upos = Some(Upos::Noun);
            date.deprel = "obl".to_owned();
            let day_id = date.doc_tok_idx;
            let marker = &mut atoms[end + 1];
            marker.upos = Some(Upos::Adp);
            marker.surface = "に".to_owned();
            marker.deprel = "case".to_owned();
            marker.head = day_id;
            marker.span.end_char = end_char;
            atoms.remove(end + 2);
        }
        if examples {
            text.push('等');
        }
        bind(atoms, start, end, text, DATE_LIST);
    }
}

#[derive(Default)]
pub(crate) struct DateContext {
    single: Option<(usize, u32, String)>,
}

impl DateContext {
    pub(in crate::ja_zh) fn normalize(
        &mut self,
        atoms: &mut Vec<Atom>,
        index: usize,
        chars: &[char],
        doc: &Document,
    ) {
        let previous = self.single.take();
        if let Some((last, value, text)) = previous
            && last + 1 == index
            && doc.segmentation.sentences[last].paragraph
                == doc.segmentation.sentences[index].paragraph
            && doc.segmentation.sentences[last].span.end_char
                <= doc.segmentation.sentences[index].span.start_char
            && chars[doc.segmentation.sentences[last].span.end_char
                ..doc.segmentation.sentences[index].span.start_char]
                .iter()
                .all(|c| c.is_whitespace())
            && let Some(day) = day_at(atoms, 0)
                .filter(|d| d.month.is_none() && calendar_number(&d.number) == Some(value))
            && atoms
                .get(day.end + 1)
                .is_some_and(|c| linked_case(c, &atoms[day.end], &atoms[day.end], "に"))
            && adjacent_count(atoms, day.end + 1).is_none()
            && !atoms[day.end + 2..].iter().any(|a| {
                !a.is_locked
                    && (a.surface.starts_with('毎')
                        || matches!(a.surface.as_str(), "いつも" | "普段" | "繰り返し")
                        || ["度", "回"].iter().any(|unit| {
                            a.surface
                                .strip_suffix(unit)
                                .is_some_and(|n| n.is_empty() || calendar_number(n).is_some())
                        }))
            })
        {
            bind(atoms, 0, day.end, text, DATE);
        }
        bind_month_dates(atoms);
        for start in 0..atoms.len() {
            let Some(day) = day_at(atoms, start).filter(|d| d.month.is_some()) else {
                continue;
            };
            let token = &atoms[day.end];
            let tail = super::super::core_non_punct(&atoms[day.end + 1..]);
            if token.deprel != "root"
                || !matches!(tail, [only,cop] if
                !only.is_locked && only.surface=="だけ" && only.head==token.doc_tok_idx &&
                !cop.is_locked && matches!(cop.surface.as_str(),"です"|"だ") && cop.deprel=="cop" && cop.head==token.doc_tok_idx)
            {
                continue;
            }
            let text = render(&day);
            if !atoms
                .iter()
                .any(|a| matches!(a.surface.as_str(), "?" | "？"))
            {
                self.single =
                    calendar_number(&day.number).map(|value| (index, value, text.clone()));
            }
            bind(atoms, day.start, day.end, text, DATE);
            break;
        }
    }
}
