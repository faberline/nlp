//! Bind a calendar day and a clock only when their source relation makes
//! them one time argument of the event.
use super::{Atom, Upos, dates::spelled_report_day, month_text, normalize_decimal_digits};
use crate::ja_zh::{CLOCK_ATOM, LockOrigin, SpanLockTable, is_clock_atom, render_clock_surface};
use faberline_nlp::doc::Document;

pub(in crate::ja_zh) const QUESTION: &str = "__ja_dated_clock_question";

fn label(
    parts: &[Atom],
    source: &[char],
    date_end: usize,
    clock_start: usize,
) -> Option<(String, bool)> {
    let first = parts.first()?;
    let last = parts.last()?;
    let date: String = source[first.span.start_char..date_end].iter().collect();
    let time: String = source[clock_start..last.span.end_char].iter().collect();
    let date_question = date.strip_suffix("何日").and_then(month_text);
    let date_text = date_question
        .as_ref()
        .map(|m| format!("{m}幾號"))
        .or_else(|| spelled_report_day(&date))?;
    let time_question = match time.as_str() {
        "何時" => Some("幾點"),
        "午前何時" => Some("上午幾點"),
        "午後何時" => Some("下午幾點"),
        _ => None,
    };
    let time_text = time_question
        .map(str::to_owned)
        .or_else(|| render_clock_surface(&time).map(|(s, _)| s))?;
    let asks = date_question.is_some() || time_question.is_some();
    if parts.len() == 1 && first.is_locked {
        return Some((first.surface.clone(), asks));
    }
    let date_parts = parts.partition_point(|a| a.span.end_char <= date_end);
    let time_parts = parts.partition_point(|a| a.span.start_char < clock_start);
    if date_parts == 0
        || parts[date_parts - 1].span.end_char != date_end
        || parts.get(time_parts)?.span.start_char != clock_start
    {
        return None;
    }
    let day = if parts[..date_parts].iter().any(|a| a.is_locked) {
        parts[..date_parts]
            .iter()
            .map(|a| {
                if a.is_locked {
                    a.surface.clone()
                } else {
                    normalize_decimal_digits(a.surface.trim_matches('の'))
                }
            })
            .collect()
    } else {
        date_text
    };
    let clock = match &parts[time_parts..] {
        [a] if is_clock_atom(a) || a.is_locked => {
            let original_period =
                [("午前", "上午"), ("午後", "下午")]
                    .into_iter()
                    .find_map(|(ja, zh)| {
                        time.strip_prefix(ja)?;
                        Some(format!("{zh}{}", a.surface.strip_prefix(ja)?))
                    });
            original_period.unwrap_or_else(|| a.surface.clone())
        }
        words if words.iter().all(|a| !a.is_locked) => time_text,
        _ => return None,
    };
    Some((day + &clock, asks))
}

fn bind_case(
    atoms: &mut Vec<Atom>,
    case: usize,
    source: &[char],
    doc: &Document,
    locks: &SpanLockTable,
) {
    let marker = &atoms[case];
    if marker.is_locked
        || marker.deprel != "case"
        || !matches!(marker.surface.as_str(), "まで" | "に" | "から")
    {
        return;
    }
    let end = marker.span.start_char;
    if !end
        .checked_sub(1)
        .and_then(|i| source.get(i))
        .is_some_and(|c| matches!(c, '時' | '半'))
    {
        return;
    }
    let fused_ni = marker.surface == "に"
        && marker
            .doc_tok_idx
            .and_then(|id| doc.tokens.get(id))
            .is_some_and(|t| t.span.start_char < end && t.span.end_char == marker.span.end_char);
    let Some(clock_id) = (if fused_ni {
        marker.doc_tok_idx
    } else {
        marker.head
    }) else {
        return;
    };
    let Some(clock) = doc.syntax.iter().find(|n| n.token == clock_id) else {
        return;
    };
    let copular_end = clock.deprel == "root"
        && marker.surface == "まで"
        && atoms[case + 1..]
            .iter()
            .any(|a| a.deprel == "cop" && a.head == Some(clock_id));
    if !copular_end
        && (!matches!(clock.deprel.as_str(), "obl" | "advmod" | "iobj") || clock.head.is_none())
    {
        return;
    }
    if case == 0 || atoms[case - 1].span.end_char != end || (!fused_ni && clock.span.end_char > end)
    {
        return;
    }
    // Genitive 日 modifies the clock. Without の, the adverbial day can
    // modify its period. Do not cross a separate argument or predicate.
    for i in (0..case).rev() {
        let a = &atoms[i];
        if i + 1 < case && a.span.end_char != atoms[i + 1].span.start_char {
            break;
        }
        if !a.is_locked
            && !(matches!(
                a.upos,
                Some(Upos::Num | Upos::Noun | Upos::Adv | Upos::Pron)
            ) || a.surface == "の")
        {
            break;
        }
        let raw = &source[a.span.start_char..end];
        let Some(pos) = raw.iter().position(|&c| c == '日') else {
            continue;
        };
        if !raw[..pos].contains(&'月') {
            continue;
        }
        // A date joined to another date can share its clock. Preserve the
        // whole coordination for the date-list rule instead of binding a suffix.
        if let [.., previous, join] = &atoms[..i]
            && matches!(join.surface.as_str(), "と" | "や" | "・")
            && previous.surface.ends_with('日')
        {
            continue;
        }
        let after_day = a.span.start_char + pos + 1;
        let genitive = source.get(after_day) == Some(&'の');
        let start_clock = after_day + usize::from(genitive);
        let whole = a.is_locked
            && i + 1 == case
            && locks.locks.iter().any(|l| {
                l.placeholder == a.surface && matches!(l.origin, LockOrigin::Glossary { .. })
            });
        if !whole && start_clock > clock.span.start_char {
            continue;
        }
        let attached = doc.syntax.iter().any(|day| {
            day.span.end_char == after_day
                && day.sentence == clock.sentence
                && (if genitive {
                    day.deprel == "nmod"
                } else {
                    day.deprel == "advmod"
                })
                && day
                    .head
                    .and_then(|h| doc.tokens.get(h))
                    .is_some_and(|h| start_clock <= h.span.start_char && h.span.end_char <= end)
        });
        if !whole && !attached {
            continue;
        }
        let Some((text, asks)) = label(&atoms[i..case], source, after_day, start_clock) else {
            continue;
        };
        let mut time = atoms[case - 1].clone();
        time.surface = text;
        time.span.start_char = a.span.start_char;
        time.lemma = Some(if asks { QUESTION } else { CLOCK_ATOM }.to_owned());
        time.upos = Some(Upos::Noun);
        time.is_locked = true;
        time.doc_tok_idx = Some(clock_id);
        time.head = clock.head;
        time.deprel = clock.deprel.clone();
        // A comma between the marked time and its governor is a pause.
        // Keep the question word with the verb whose か closes the question.
        let comma = case
            + 1
            + usize::from(
                marker.surface == "まで"
                    && atoms.get(case + 1).is_some_and(|ni| {
                        !ni.is_locked
                            && ni.surface == "に"
                            && ni.deprel == "case"
                            && ni.head == marker.head
                            && ni.span.start_char == marker.span.end_char
                    }),
            );
        if atoms
            .get(comma)
            .is_some_and(|a| !a.is_locked && matches!(a.surface.as_str(), "、" | "，" | ","))
            && atoms[comma + 1..].iter().any(|a| {
                a.doc_tok_idx.is_some() && a.doc_tok_idx == clock.head && a.upos == Some(Upos::Verb)
            })
        {
            atoms.remove(comma);
        }
        atoms.splice(i..case, [time]);
        return;
    }
}

pub(super) fn normalize(
    atoms: &mut Vec<Atom>,
    source: &[char],
    doc: &Document,
    locks: &SpanLockTable,
) {
    for i in (1..atoms.len()).rev() {
        if i < atoms.len() {
            bind_case(atoms, i, source, doc, locks);
        }
    }
}
