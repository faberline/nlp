//! Resolve dates that share an explicit month within a list or across a sentence boundary.
use super::{
    Atom, DATE, DATE_LIST, Upos, adjacent_count, calendar_number, is_numeral, linked_case,
    month_text, normalize_decimal_digits,
};
use crate::lock::{LockOrigin, SpanLockTable};
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

// The analyzer can split 友だち into 友 + a copula + ち even before the
// recipient particle. Recover the nominal head and its dependent time.
fn bind_friend_recipient(atoms: &mut Vec<Atom>) {
    let mut i = 0;
    while let Some([friend, cop, suffix, case]) = atoms.get(i..i + 4) {
        if ![friend, cop, suffix].iter().any(|a| a.is_locked)
            && friend.surface == "友"
            && cop.surface == "だ"
            && suffix.surface == "ち"
            && friend.head == suffix.doc_tok_idx
            && cop.head == friend.doc_tok_idx
            && cop.deprel == "cop"
            && friend.span.end_char == cop.span.start_char
            && cop.span.end_char == suffix.span.start_char
            && matches!(case.surface.as_str(), "に" | "へ")
            && linked_case(case, suffix, suffix, &case.surface)
        {
            let mut noun = suffix.clone();
            noun.surface = "友達".to_owned();
            noun.lemma = Some(noun.surface.clone());
            noun.upos = Some(Upos::Noun);
            noun.span.start_char = friend.span.start_char;
            let misparsed = friend.doc_tok_idx;
            for a in &mut atoms[..i] {
                if a.head == misparsed
                    && matches!(a.deprel.as_str(), "advmod" | "obl" | "iobj")
                    && (a.surface == "日"
                        || month_text(&a.surface).is_some()
                        || dated_atom(a)
                        || a.is_locked)
                {
                    a.head = noun.head;
                }
            }
            atoms.splice(i..i + 3, [noun]);
        }
        i += 1;
    }
}

// A comma after the case of a concrete date can separate a temporal
// argument from the event it still modifies. Keep them in the same clause.
fn protected_date(atoms: &[Atom], start: usize, chars: &[char]) -> Option<(usize, String)> {
    let first = &atoms[start];
    if !(first.is_locked || is_numeral(first) || month_text(&first.surface).is_some()) {
        return None;
    }
    for (offset, part) in atoms[start..].iter().enumerate() {
        let end = start + offset;
        if !part.is_locked
            && !(is_numeral(part)
                || month_text(&part.surface).is_some()
                || matches!(part.surface.as_str(), "月" | "日" | "の")
                || dated_atom(part))
        {
            break;
        }
        if offset > 0 && atoms[end - 1].span.end_char != part.span.start_char {
            break;
        }
        if !atoms.get(end + 1).is_some_and(|c| {
            !c.is_locked
                && matches!(
                    c.surface.as_str(),
                    "に" | "まで" | "の" | "は" | "と" | "か"
                )
        }) {
            continue;
        }
        let parts = &atoms[start..=end];
        if !parts.iter().any(|a| a.is_locked) {
            return None;
        }
        let source: String = chars
            .get(parts[0].span.start_char..part.span.end_char)?
            .iter()
            .collect();
        if !source.contains('月') || spelled_report_day(&source).is_none() {
            return None;
        }
        return Some((
            end,
            parts
                .iter()
                .map(|a| {
                    if a.is_locked {
                        a.surface.clone()
                    } else if a.surface == "の" {
                        String::new()
                    } else {
                        month_text(&a.surface)
                            .unwrap_or_else(|| normalize_decimal_digits(&a.surface))
                    }
                })
                .collect(),
        ));
    }
    None
}

struct TopicClock {
    index: usize,
    pause: Option<usize>,
    question_end: usize,
    advice: Option<(usize, usize)>,
}

// A verbal noun plus する forms one predicate. In an advice question the
// event's ば is followed by the matrix predicate いいですか.
fn clock_question_head(atoms: &[Atom], owner: usize) -> (usize, Option<(usize, usize)>) {
    let Some(i) = atoms
        .iter()
        .position(|a| a.doc_tok_idx == Some(owner) && !a.is_locked)
    else {
        return (owner, None);
    };
    let action = &atoms[i];
    if let [ba, good, cop, ka, ..] = &atoms[i + 1..]
        && action.upos == Some(Upos::Verb)
        && action.deprel == "advcl"
        && action.head == good.doc_tok_idx
        && ba.surface == "ば"
        && ba.deprel == "mark"
        && ba.head == Some(owner)
        && matches!(good.surface.as_str(), "いい" | "よい")
        && good.upos == Some(Upos::Adj)
        && cop.surface == "です"
        && cop.deprel == "cop"
        && cop.head == good.doc_tok_idx
        && ka.surface == "か"
        && ka.deprel == "discourse"
        && ka.head == good.doc_tok_idx
        && atoms[i..=i + 4]
            .windows(2)
            .all(|p| !p[1].is_locked && p[0].span.end_char == p[1].span.start_char)
        && let Some(matrix) = good.doc_tok_idx
    {
        return (matrix, Some((i, i + 3)));
    }
    if let Some(light) = atoms.get(i + 1)
        && !light.is_locked
        && light.lemma.as_deref() == Some("する")
        && action.upos == Some(Upos::Verb)
        && action.deprel == "advcl"
        && action.span.end_char == light.span.start_char
        && action.head == light.doc_tok_idx
        && let Some(matrix) = light.doc_tok_idx
    {
        return (matrix, None);
    }
    (owner, None)
}

// Separate は markers introduce separate topics even if an omitted verb
// makes the parser assign both dates and both destinations to one predicate.
fn has_parallel_day_topics(
    atoms: &[Atom],
    owner: usize,
    chars: &[char],
    doc: &Document,
) -> Option<bool> {
    let mut previous = None;
    let mut parallel = false;
    for wa in atoms
        .iter()
        .filter(|a| !a.is_locked && a.surface == "は" && a.deprel == "case")
    {
        let Some(day) = doc.syntax.iter().find(|n| {
            Some(n.token) == wa.head
                && n.head == Some(owner)
                && n.deprel == "nsubj"
                && n.span.end_char == wa.span.start_char
        }) else {
            continue;
        };
        let mut start = day.span.start_char;
        if chars.get(start..day.span.end_char) == Some(&['日'])
            && let Some(number) = day.token.checked_sub(1).and_then(|i| doc.tokens.get(i))
            && number.upos == Some(Upos::Num)
            && number.span.end_char == start
        {
            start = number.span.start_char;
        }
        let surface = chars
            .get(start..day.span.end_char)?
            .iter()
            .collect::<String>();
        if spelled_report_day(&surface).is_none() {
            continue;
        }
        if let Some(last) = previous {
            if !chars
                .get(last..start)?
                .iter()
                .any(|c| matches!(c, '、' | '，' | ','))
            {
                return None;
            }
            parallel = true;
        }
        previous = Some(wa.span.end_char);
    }
    Some(parallel)
}

// A は-marked day and a separately marked clock can describe the same event.
// Their shared predicate licenses both arguments.
fn topic_clock(
    atoms: &[Atom],
    start: usize,
    day_end: usize,
    choice: bool,
    chars: &[char],
    doc: &Document,
) -> Option<TopicClock> {
    let wa = atoms.get(day_end + 1)?;
    if wa.is_locked || !(wa.surface == "は" || choice && wa.surface == "に") || wa.deprel != "case"
    {
        return None;
    }
    let day_surface = chars.get(atoms[start].span.start_char..wa.span.start_char)?;
    if !day_surface.contains(&'月') || atoms[day_end].lemma.as_deref() == Some(DATE_LIST) {
        return None;
    }
    // A following は can mark an entire date list. Its final day alone is
    // not the topic.
    let before = &atoms[..start];
    let is_join = |a: &Atom| {
        matches!(a.surface.as_str(), "と" | "や" | "・" | "、" | "か")
            || a.upos == Some(Upos::Cconj)
    };
    if before.last().is_some_and(is_join)
        && before
            .iter()
            .rev()
            .find(|a| !is_join(a))
            .and_then(|a| chars[..a.span.end_char].last())
            == Some(&'日')
    {
        return None;
    }
    let day = doc.syntax.iter().find(|node| Some(node.token) == wa.head)?;
    if !(day.deprel == "nsubj" || choice && matches!(day.deprel.as_str(), "obl" | "iobj"))
        || day.span.end_char != wa.span.start_char
    {
        return None;
    }
    let owner = day.head?;
    if doc.tokens.get(owner)?.upos != Some(Upos::Verb) {
        return None;
    }
    let (clock, pause) = (day_end + 2..atoms.len()).chain(0..start).find_map(|i| {
        let time = &atoms[i];
        if chars.get(time.span.start_char..time.span.end_char) != Some(&['何', '時'])
            || !doc.syntax.iter().any(|node| {
                node.span == time.span
                    && node.head == Some(owner)
                    && matches!(node.deprel.as_str(), "obl" | "iobj")
            })
        {
            return None;
        }
        let case = atoms.get(i + 1)?;
        let after_case = if linked_case(case, time, time, "に") {
            i + 2
        } else if linked_case(case, time, time, "まで")
            && atoms
                .get(i + 2)
                .is_some_and(|ni| linked_case(ni, time, case, "に"))
        {
            i + 3
        } else {
            return None;
        };
        // Further particles attached to 何時 can express unrestricted hours.
        if atoms[after_case..].iter().any(|a| {
            a.head == time.doc_tok_idx
                && matches!(a.upos, Some(Upos::Adp | Upos::Part | Upos::Sconj))
        }) {
            return None;
        }
        let mut pause = None;
        if i < start {
            let parallel = has_parallel_day_topics(atoms, owner, chars, doc)?;
            for (offset, word) in atoms.get(after_case..start)?.iter().enumerate() {
                if word.upos == Some(Upos::Verb) {
                    return None;
                }
                if word.upos == Some(Upos::Punct) {
                    // A pause inside direct speech can attach to the reporting
                    // verb. The date, clock, and final question must still
                    // belong to the same event inside or outside the quote.
                    if pause.is_some()
                        || word.is_locked
                        || !matches!(word.surface.as_str(), "、" | "，" | ",")
                        || word.deprel != "punct"
                    {
                        return None;
                    }
                    // Keep the commas that distinguish day–argument
                    // pairs when this clock introduces several topics.
                    if !parallel {
                        pause = Some(after_case + offset);
                    }
                }
            }
        }
        Some((i, pause))
    })?;
    // Use the question's source syntax even when the final punctuation was
    // taken out of the translation atoms. A bare ？ still belongs to its verb.
    let (matrix, advice) = clock_question_head(atoms, owner);
    if advice.is_some_and(|(action, _)| action <= clock.max(day_end)) {
        return None;
    }
    let end = doc.syntax.iter().find(|node| {
        if node.head != Some(matrix)
            || node.span.start_char < atoms[clock].span.end_char.max(wa.span.end_char)
        {
            return false;
        }
        match chars.get(node.span.start_char..node.span.end_char) {
            Some(['？' | '?']) => node.deprel == "punct",
            Some(['か']) => {
                node.deprel == "discourse"
                    && chars[node.span.end_char..]
                        .iter()
                        .find(|c| !c.is_whitespace())
                        .is_none_or(|c| {
                            crate::ja_zh::is_quote_terminal(*c)
                                || matches!(c, '」' | '』' | '”' | '"')
                        })
            }
            _ => false,
        }
    })?;
    // An enclosing written quotation belongs to the source-form renderer.
    // A source question on the page is not a request for advice to the writer.
    if (matrix != owner || wa.surface == "に")
        && matches!(chars.get(end.span.end_char), Some('」' | '』' | '”' | '"'))
        && doc
            .syntax
            .iter()
            .find(|n| n.token == matrix)
            .and_then(|n| n.head)
            .and_then(|i| doc.tokens.get(i))
            .and_then(|t| t.lemma.as_deref())
            .is_some_and(|verb| matches!(verb, "書く" | "写す" | "読み取る" | "読む"))
    {
        return None;
    }
    let fronted = clock < start;
    let object_words = if fronted {
        &atoms[day_end + 2..]
    } else {
        &atoms[day_end + 2..clock]
    };
    let object_quotes: Vec<_> = object_words
        .windows(4)
        .filter_map(|words| {
            let [open, name, close, wo] = words else {
                return None;
            };
            (matches!(
                (open.surface.as_str(), close.surface.as_str()),
                ("「", "」") | ("『", "』")
            ) && (name.is_locked || matches!(name.upos, Some(Upos::Noun | Upos::Propn)))
                && !wo.is_locked
                && wo.surface == "を"
                && (wo.head == Some(owner) || name.head == Some(owner)))
            .then_some((open.span.start_char, close.span.start_char))
        })
        .collect();
    if (wa.span.end_char.min(atoms[clock].span.start_char)..end.span.start_char).any(|at| {
        let c = chars[at];
        crate::ja_zh::is_quote_terminal(c)
            || matches!(c, '「' | '」' | '『' | '』' | '“' | '”' | '"')
                && !object_quotes
                    .iter()
                    .any(|&(left, right)| at == left || at == right)
    }) {
        return None;
    }
    Some(TopicClock {
        index: clock,
        pause,
        question_end: end.token,
        advice,
    })
}

// Alternatives share は at the end of the last date. Read each complete
// month and day before assigning the topic to the clock question.
fn date_alternatives(
    atoms: &[Atom],
    first_end: usize,
    first: &str,
    chars: &[char],
    doc: &Document,
) -> (usize, String) {
    let mut end = first_end;
    let mut label = first.to_owned();
    let next_date = |start| {
        atoms.get(start)?;
        day_at(atoms, start)
            .filter(|d| d.month.is_some())
            .map(|d| (d.end, render(&d)))
            .or_else(|| protected_date(atoms, start, chars))
    };
    // A glossary can replace all date tokens. The particle still attaches
    // to the original day token at the end of that same source span.
    let attached = |case: &Atom, day: &Atom| {
        case.head.is_some()
            && (case.head == day.doc_tok_idx
                || day.is_locked
                    && case.head.and_then(|i| doc.tokens.get(i)).is_some_and(|t| {
                        day.span.start_char <= t.span.start_char
                            && day.span.end_char == t.span.end_char
                    }))
    };
    let after_comma = |i| {
        i + usize::from(
            atoms
                .get(i)
                .is_some_and(|a: &Atom| !a.is_locked && a.surface == "、"),
        )
    };
    loop {
        let separator = after_comma(end + 1);
        let Some(join) = atoms.get(separator) else {
            break;
        };
        // と normally adds dates. のいずれか一日 explicitly selects just one.
        if join.surface == "と"
            && !join.is_locked
            && join.deprel == "case"
            && attached(join, &atoms[end])
            && let Some((next, text)) = next_date(separator + 1)
            && let Some(no) = atoms.get(next + 1)
            && !no.is_locked
            && no.surface == "の"
            && no.deprel == "case"
            && attached(no, &atoms[next])
        {
            let tail = &atoms[next + 2..];
            if let Some(ni) = tail.iter().take(5).position(|a| a.surface == "に")
                && ni > 0
                && tail[..ni]
                    .iter()
                    .map(|a| a.surface.as_str())
                    .collect::<String>()
                    .strip_suffix('日')
                    .and_then(|s| {
                        s.strip_prefix("いずれか")
                            .or_else(|| s.strip_prefix("どちらか"))
                    })
                    .and_then(calendar_number)
                    == Some(1)
                && tail[..=ni].iter().all(|a| !a.is_locked)
                && atoms[end..=next + 2 + ni]
                    .windows(2)
                    .all(|p| p[0].span.end_char == p[1].span.start_char)
                && linked_case(&tail[ni], &tail[ni - 1], &tail[ni - 1], "に")
            {
                label.push('或');
                label.push_str(&text);
                label.push_str("其中一天");
                return (next + 1 + ni, label);
            }
        }
        if join.is_locked
            || !(join.surface == "か" && attached(join, &atoms[end])
                || join.upos == Some(Upos::Cconj)
                    && join.deprel == "cc"
                    && matches!(
                        join.surface.as_str(),
                        "または" | "又は" | "あるいは" | "或いは" | "もしくは" | "若しくは"
                    ))
        {
            break;
        }
        let next = after_comma(separator + 1);
        let Some((next_end, text)) = next_date(next) else {
            break;
        };
        label.push('或');
        label.push_str(&text);
        end = next_end;
    }
    (end, label)
}

fn bind_detached_dates(
    atoms: &mut Vec<Atom>,
    chars: &[char],
    doc: &Document,
    locks: &SpanLockTable,
) {
    let mut i = 0;
    while i < atoms.len() {
        let start = i;
        i += 1;
        let (start, end, text) =
            if let Some(day) = day_at(atoms, start).filter(|d| d.month.is_some()) {
                let Some((start, text)) = super::dated_month(atoms, start, render(&day)) else {
                    continue;
                };
                (start, day.end, text)
            } else if super::is_date(&atoms[start]) {
                (start, start, atoms[start].surface.clone())
            } else if let Some((end, text)) = protected_date(atoms, start, chars) {
                (start, end, text)
            } else {
                continue;
            };
        let (topic_end, topic_text) = date_alternatives(atoms, end, &text, chars, doc);
        let choice = topic_end != end;
        if let Some(found) = topic_clock(atoms, start, topic_end, choice, chars, doc) {
            let clock = found.index;
            if clock < start
                && let Some(ka) = atoms.iter_mut().find(|a| {
                    a.doc_tok_idx == Some(found.question_end) && !a.is_locked && a.surface == "か"
                })
            {
                // A quoted object after the date can split a single event
                // into rendered segments. Keep its verified question open.
                ka.lemma = Some(super::FRONTED_CLOCK_KA.to_owned());
            }
            let mut time = atoms[clock].clone();
            if !time.is_locked {
                time.surface = "幾點".to_owned();
            }
            if let Some((verb, cop)) = found.advice {
                let action_id = atoms[verb].doc_tok_idx;
                atoms[cop + 1].head = action_id;
                atoms[verb].deprel = "root".to_owned();
                atoms[verb].head = None;
                atoms.drain(verb + 1..=cop);
                time.surface.insert_str(0, "應該");
            }
            let temporal_choice = choice && atoms[topic_end + 1].surface == "に";
            if temporal_choice {
                atoms[topic_end + 1].surface = "は".to_owned();
                atoms[topic_end].deprel = "nsubj".to_owned();
            }
            time.is_locked = true;
            time.lemma = Some(super::CLOCK_QUESTION.to_owned());
            let period_index = if let Some(period) = clock.checked_sub(1).and_then(|i| atoms.get(i))
                && period.span.end_char == time.span.start_char
                && (period.head == time.head
                    || period.head == time.doc_tok_idx
                    || clock < start && period.head == atoms[start].doc_tok_idx)
                && let Some(name) = match chars.get(period.span.start_char..period.span.end_char) {
                    Some(['午', '前']) => Some("上午"),
                    Some(['午', '後']) => Some("下午"),
                    _ => None,
                } {
                time.surface.insert_str(
                    0,
                    if period.is_locked {
                        &period.surface
                    } else {
                        name
                    },
                );
                time.span.start_char = period.span.start_char;
                clock - 1
            } else {
                clock
            };
            atoms.splice(period_index..=clock, [time]);
            let shift = if clock < start {
                clock - period_index
            } else {
                0
            };
            bind(atoms, start - shift, topic_end - shift, topic_text, DATE);
            let after_date_case = start - shift + 2;
            if temporal_choice
                && atoms
                    .get(after_date_case)
                    .is_some_and(|a| !a.is_locked && a.surface == "、" && a.deprel == "punct")
            {
                atoms.remove(after_date_case);
            }
            if let Some(comma) = found.pause {
                atoms.remove(comma - shift);
            }
            continue;
        }
        let day = &atoms[end];
        // In 八月九日の切符 the date modifies the ticket. The month can
        // have a separate parser arc, but the genitive case belongs to 日.
        if day.deprel == "nmod"
            && atoms
                .get(end + 1)
                .is_some_and(|no| linked_case(no, day, day, "の"))
            && let Some(pos) = atoms[end + 2..].iter().position(|noun| {
                noun.doc_tok_idx.is_some()
                    && noun.doc_tok_idx == day.head
                    && (noun.is_locked || matches!(noun.upos, Some(Upos::Noun | Upos::Propn)))
            })
        {
            let object = end + 2 + pos;
            if atoms[object].deprel == "obj"
                && let [wo, pause, time, ..] = &atoms[object + 1..]
                && linked_case(wo, &atoms[object], &atoms[object], "を")
                && !pause.is_locked
                && matches!(pause.surface.as_str(), "、" | "，" | ",")
                && crate::ja_zh::is_clock_atom(time)
                && atoms[object].head == time.head
            {
                atoms.remove(object + 2);
            }
            bind(atoms, start, end, text, DATE);
            continue;
        }
        let Some(case) = atoms.get(end + 1).filter(|c| {
            matches!(c.surface.as_str(), "まで" | "に") && linked_case(c, day, day, &c.surface)
        }) else {
            continue;
        };
        let comma = if case.surface == "まで" {
            if !atoms
                .get(end + 2)
                .is_some_and(|ni| linked_case(ni, day, case, "に"))
            {
                continue;
            }
            end + 3
        } else {
            end + 2
        };
        if !atoms
            .get(comma)
            .is_some_and(|a| !a.is_locked && matches!(a.surface.as_str(), "、" | "，" | ","))
        {
            continue;
        }
        let following = &atoms[comma + 1..];
        // A forced multi-token recipient can hide the original token to
        // which the parser attached the date. Its case identifies the event.
        let head = day.head.and_then(|head| {
            if following
                .iter()
                .any(|a| a.doc_tok_idx == Some(head) && a.upos == Some(Upos::Verb))
            {
                return Some(head);
            }
            // The written form can hide a verb that already licenses this date
            // in the source analysis. Keep the date without opening the lock.
            if let Some(verb) = doc
                .tokens
                .get(head)
                .filter(|token| token.upos == Some(Upos::Verb))
                && following.iter().any(|atom| {
                    atom.is_locked
                        && atom.span.start_char <= verb.span.start_char
                        && verb.span.end_char <= atom.span.end_char
                        && locks.locks.iter().any(|lock| {
                            lock.source_span == atom.span && lock.origin == LockOrigin::QuotedSource
                        })
                })
            {
                return Some(head);
            }
            let recipient = following.windows(2).find_map(|pair| {
                let [noun, case] = pair else { return None };
                if !noun.is_locked
                    || !matches!(case.surface.as_str(), "に" | "へ")
                    || !linked_case(case, noun, noun, &case.surface)
                {
                    return None;
                }
                let token = doc.tokens.get(head)?;
                (token.span.start_char >= noun.span.start_char
                    && token.span.end_char <= noun.span.end_char)
                    .then_some(noun)
            })?;
            following
                .iter()
                .find(|a| a.doc_tok_idx == recipient.head && a.upos == Some(Upos::Verb))?
                .doc_tok_idx
        });
        let Some(head) = head else { continue };
        let list = day.lemma.as_deref() == Some(DATE_LIST);
        atoms[end].head = Some(head);
        atoms.remove(comma);
        bind(atoms, start, end, text, if list { DATE_LIST } else { DATE });
    }
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
        locks: &SpanLockTable,
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
        bind_friend_recipient(atoms);
        super::dated_clock::normalize(atoms, chars, doc, locks);
        for a in atoms.iter_mut().filter(|a| a.is_locked) {
            let Some(raw) = chars.get(a.span.start_char..a.span.end_char) else {
                continue;
            };
            let source: String = raw.iter().collect();
            let mut dates = source.split(['と', 'や', '、', '・']);
            if dates
                .next()
                .is_some_and(|date| date.contains('月') && spelled_report_day(date).is_some())
                && dates.clone().next().is_some()
                && dates.all(|date| spelled_report_day(date).is_some())
            {
                a.lemma = Some(DATE_LIST.to_owned());
            }
        }
        bind_month_dates(atoms);
        bind_detached_dates(atoms, chars, doc, locks);
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

#[cfg(test)]
mod dated_question_tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

    fn translate(source: &str) -> String {
        analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap()
        .target_text
    }

    #[test]
    fn date_keeps_the_kind_of_clock_in_a_request_for_advice() {
        for (case, expected) in [
            ("に", "11月12日應該幾點"),
            ("までに", "11月12日最晚應該幾點"),
        ] {
            let s = format!("何時{case}、11月12日は市役所に申請書を提出すればいいですか。");
            let out = translate(&s);
            assert!(
                out.starts_with(expected) && out.contains("申請書") && out.ends_with('？'),
                "{out}"
            );
        }
    }

    #[test]
    fn verbal_noun_keeps_each_dates_location_in_the_question() {
        assert_eq!(
            translate("何時までに、七月九日は駅で、七月十一日は港で集合しますか。"),
            "最晚幾點，七月九日在車站，七月十一日在港口集合？"
        );
    }

    #[test]
    fn reported_advice_leaves_a_written_source_quotation_to_its_renderer() {
        let question = "何時までに、11月12日は市役所に申請書を提出すればいいですか";
        let spoken = translate(&format!("先生は「{question}」と言いました。"));
        assert!(spoken.contains("11月12日最晚應該幾點"), "{spoken}");
        let written = translate(&format!("「{question}」と一字ずつ紙に書きました。"));
        assert!(!written.contains("11月12日最晚應該幾點"), "{written}");
    }

    #[test]
    fn date_selection_is_distinct_from_joint_event_dates() {
        for (dates, expected) in [
            (
                "11月19日と11月20日のいずれか一日に",
                "11月19日或11月20日其中一天",
            ),
            ("11月19日か11月20日に", "11月19日或11月20日"),
        ] {
            let out = translate(&format!(
                "何時までに、{dates}、会議室へ名簿を届ければいいですか。"
            ));
            assert!(
                out.starts_with(expected)
                    && out.contains("最晚應該幾點")
                    && out.contains("會議室")
                    && out.contains("名簿")
                    && !out.contains('嗎'),
                "{out}"
            );
        }
        let both = translate("何時までに、11月19日と11月20日に、会議室へ名簿を届けますか。");
        assert!(
            both.contains("11月19日和11月20日") && !both.contains('或'),
            "{both}"
        );
    }
}
