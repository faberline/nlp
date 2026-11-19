use super::*;
use crate::ja_zh::{counted_object, translate_single_ja_term};

struct Measure {
    start: usize,
    unit: usize,
    number: String,
    kind: &'static str,
}

fn measure_at(atoms: &[Atom], i: usize, day: &Atom) -> Option<Measure> {
    let unit = &atoms[i];
    let head = unit.head?;
    if unit.is_locked
        || unit.upos != Some(Upos::Noun)
        || !(Some(head) == day.head
            || atoms.iter().any(|a| {
                !a.is_locked
                    && a.doc_tok_idx == Some(head)
                    && a.head == day.head
                    && matches!(a.upos, Some(Upos::Noun | Upos::Propn))
            }))
    {
        return None;
    }
    // Includes counters without a Chinese classifier in the current lexicon.
    // Their presence can still make the day an interval rather than a date.
    let kind = [
        "人",
        "名",
        "冊",
        "枚",
        "個",
        "台",
        "件",
        "匹",
        "杯",
        "本",
        "つ",
        "通",
        "着",
        "箱",
        "組",
        "袋",
        "頭",
        "羽",
        "粒",
        "問",
        "円",
        "ページ",
        "時間",
    ]
    .into_iter()
    .find(|suffix| unit.surface.ends_with(suffix))?;
    let joined = unit.surface.strip_suffix(kind)?;
    let (start, number) = if joined.is_empty() {
        number_before(atoms, i)?
    } else {
        (i, joined.to_owned())
    };
    calendar_number(&number)?;
    Some(Measure {
        start,
        unit: i,
        number,
        kind,
    })
}

pub(super) fn has_measure(atoms: &[Atom], day: usize) -> bool {
    (day + 2..atoms.len()).any(|i| {
        if measure_at(atoms, i, &atoms[day]).is_some() {
            return true;
        }
        // Unknown units still carry an overt amount. Keep a bare 日
        // undecided even when the target classifier cannot be rendered.
        let unit = &atoms[i];
        let Some(prev) = i.checked_sub(1).and_then(|n| atoms.get(n)) else {
            return false;
        };
        !unit.is_locked
            && unit.upos == Some(Upos::Noun)
            && !matches!(
                unit.surface.as_str(),
                "年" | "月" | "日" | "曜日" | "時" | "分"
            )
            && is_numeral(prev)
            && prev.deprel == "nummod"
            && prev.head == unit.doc_tok_idx
            && prev.span.end_char == unit.span.start_char
            && (unit.head == atoms[day].head
                || atoms.iter().any(|a| {
                    !a.is_locked && a.doc_tok_idx == unit.head && a.head == atoms[day].head
                }))
    })
}

// Classify a counted argument within its own event. Only an explicit rate or
// a distributive count without a competing recipient gives a recurring event.
pub(super) fn bind(atoms: &mut Vec<Atom>, end: usize, contrast: bool) {
    let day = &atoms[end];
    if day.is_locked
        || day.surface != "日"
        || day.upos != Some(Upos::Noun)
        || !matches!(day.deprel.as_str(), "obl" | "iobj")
        || !atoms
            .get(end + 1)
            .is_some_and(|c| linked_case(c, day, day, "に"))
    {
        return;
    }
    let Some(verb) = atoms.iter().find(|a| {
        !a.is_locked
            && a.doc_tok_idx == day.head
            && a.upos == Some(Upos::Verb)
            && a.deprel == "root"
    }) else {
        return;
    };
    let mut measures = (end + 2..atoms.len()).filter_map(|i| measure_at(atoms, i, day));
    let Some(q) = measures.next() else { return };
    if measures.next().is_some() {
        return;
    }
    let unit = &atoms[q.unit];
    let marker = atoms.get(q.unit + 1).filter(|a| {
        !a.is_locked
            && a.upos == Some(Upos::Part)
            && a.head == unit.doc_tok_idx
            && a.deprel == "discourse"
            && a.span.start_char == unit.span.end_char
    });
    let only = marker.is_some_and(|a| a.surface == "だけ");
    let each = marker.is_some_and(|a| a.surface == "ずつ");
    let Some(DayInterval {
        start,
        dated,
        atom: interval,
    }) = day_interval(atoms, end)
    else {
        return;
    };
    let competing_recipient = atoms.iter().any(|a| {
        !a.is_locked && a.surface == "に" && a.deprel == "case" && a.head != day.doc_tok_idx
    });
    let people = matches!(q.kind, "人" | "名");
    if !(dated && (only || people && each && !competing_recipient)
        || !dated && (contrast || each && !competing_recipient))
    {
        return;
    }

    let mut result = unit.clone();
    let mut amount = String::new();
    let count_end = q.unit + 1 + usize::from(only || each);
    let related_argument = |arg: &&Atom| {
        !arg.is_locked
            && arg.head == verb.doc_tok_idx
            && (unit.head == verb.doc_tok_idx || unit.head == arg.doc_tok_idx)
    };
    if people {
        // An indefinite participant of 来る arrives at each interval. Keep
        // the person count with that participant rather than counting trips.
        if verb.lemma.as_deref() != Some("来る") {
            return;
        }
        let mut subjects = atoms
            .iter()
            .filter(related_argument)
            .filter(|a| a.deprel == "nsubj" && a.doc_tok_idx != unit.doc_tok_idx);
        if let Some(subject) = subjects.next() {
            if subjects.next().is_some() || subject.upos != Some(Upos::Noun) {
                return;
            }
            result.lemma = Some(
                if each && dated {
                    DISTRIBUTED_SUBJECT
                } else if only {
                    ONLY_MEASURED_SUBJECT
                } else {
                    MEASURED_SUBJECT
                }
                .to_owned(),
            );
            counted_object::push_quantity(&mut amount, &q.number, "位");
        } else {
            result.lemma = Some(
                if each && dated {
                    DISTRIBUTED_SUBJECT
                } else if only {
                    ONLY_REPETITIONS
                } else {
                    REPETITIONS
                }
                .to_owned(),
            );
            counted_object::push_quantity(&mut amount, &q.number, "人");
        }
    } else {
        let mut objects = atoms
            .iter()
            .filter(related_argument)
            .filter(|a| a.deprel == "obj");
        let Some(object) = objects.next() else { return };
        if objects.next().is_some() {
            return;
        }
        let noun = translate_single_ja_term(&object.surface, object.lemma.as_deref(), object.upos);
        let Some(classifier) = counted_object::classifier(q.kind, &noun) else {
            return;
        };
        counted_object::push_quantity(&mut amount, &q.number, classifier);
        result.lemma = Some(if only { ONLY_REPETITIONS } else { REPETITIONS }.to_owned());
    }
    // A following の or an unhandled focus particle still belongs to its
    // noun phrase. Leave both it and its quantity on the normal path.
    if atoms
        .get(count_end)
        .is_some_and(|a| matches!(a.upos, Some(Upos::Part | Upos::Adp)))
    {
        return;
    }
    result.surface = amount;
    result.is_locked = true;
    result.head = day.head;
    result.span.start_char = atoms[q.start].span.start_char;
    result.span.end_char = atoms[count_end - 1].span.end_char;
    atoms.splice(q.start..count_end, [result]);
    if !dated {
        atoms.splice(start..end + 2, [interval]);
    }
}
