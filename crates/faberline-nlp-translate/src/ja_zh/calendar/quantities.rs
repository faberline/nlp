use super::{
    Atom, DATE_LIST, DISTRIBUTED_SUBJECT, DayInterval, EACH_COUNT, EACH_SUBJECT, MEASURED_SUBJECT,
    ONLY_MEASURED_SUBJECT, ONLY_REPETITIONS, REPETITIONS, Upos, calendar_number, countable_day,
    day_interval, is_numeral, linked_case, number_before,
};
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
        || !(unit.upos == Some(Upos::Noun)
            || unit.upos == Some(Upos::Adv)
                && unit.surface.ends_with('杯')
                && unit.deprel == "advmod")
        || !(Some(head) == day.head
            || atoms.iter().any(|a| {
                a.doc_tok_idx == Some(head)
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
        let prev = &atoms[i - 1];
        let adjacent_focus = i == day + 3
            && atoms.get(i + 1).is_some_and(|focus| {
                !focus.is_locked
                    && focus.head == unit.doc_tok_idx
                    && focus.deprel == "discourse"
                    && matches!(focus.surface.as_str(), "だけ" | "ずつ")
            });
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
            && (adjacent_focus
                || unit.head == atoms[day].head
                || atoms.iter().any(|a| {
                    !a.is_locked && a.doc_tok_idx == unit.head && a.head == atoms[day].head
                }))
    })
}

// Classify a counted argument within its own event. Only an explicit rate or
// a distributive count without a competing recipient gives a recurring event.
pub(super) fn bind(atoms: &mut Vec<Atom>, end: usize, contrast: bool) {
    let day = &atoms[end];
    if !countable_day(day)
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
            && matches!(a.deprel.as_str(), "root" | "acl")
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
    // A floated count after a subject can distribute people within a single
    // visit. A genitive or relative modifier can likewise select a day.
    let ambiguous_distribution = atoms[end + 2..q.start]
        .iter()
        .any(|a| a.deprel == "nsubj" && a.head == day.head)
        || start
            .checked_sub(1)
            .is_some_and(|p| atoms[p].surface == "の" || atoms[p].deprel == "acl");
    let list = day.lemma.as_deref() == Some(DATE_LIST);
    if !(dated && (only || list && each || people && each && !competing_recipient)
        || !dated && (contrast || each && !competing_recipient && !ambiguous_distribution))
    {
        return;
    }

    let mut result = unit.clone();
    let mut amount = String::new();
    let count_end = q.unit + 1 + usize::from(only || each);
    let related_argument = |arg: &&Atom| {
        arg.head == verb.doc_tok_idx
            && (unit.head == verb.doc_tok_idx || unit.head == arg.doc_tok_idx)
    };
    if people {
        // An indefinite participant of 来る arrives at each interval. Keep
        // the person count with that participant rather than counting trips.
        if verb.lemma.as_deref().unwrap_or(&verb.surface) != "来る" {
            return;
        }
        let mut subjects = atoms
            .iter()
            .filter(related_argument)
            .filter(|a| a.deprel == "nsubj" && a.doc_tok_idx != unit.doc_tok_idx);
        if let Some(subject) = subjects.next() {
            if subjects.next().is_some()
                || !(subject.upos == Some(Upos::Noun)
                    || subject.is_locked && subject.upos == Some(Upos::Propn))
            {
                return;
            }
            result.lemma = Some(
                if each && list {
                    EACH_SUBJECT
                } else if each && dated {
                    DISTRIBUTED_SUBJECT
                } else if only {
                    ONLY_MEASURED_SUBJECT
                } else {
                    MEASURED_SUBJECT
                }
                .to_owned(),
            );
            counted_object::push_quantity(&mut amount, &q.number, "位");
            if dated
                && !list
                && each
                && calendar_number(&q.number) == Some(1)
                && subject.span.end_char <= unit.span.start_char
            {
                result.lemma = Some(super::INDIVIDUALS.to_owned());
                amount.push_str("接一位地");
            }
        } else {
            result.lemma = Some(
                if each && list {
                    EACH_COUNT
                } else if each && dated {
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
        let noun = if object.is_locked {
            String::new()
        } else {
            translate_single_ja_term(&object.surface, object.lemma.as_deref(), object.upos)
        };
        let Some(classifier) = counted_object::classifier(q.kind, &noun) else {
            return;
        };
        counted_object::push_quantity(&mut amount, &q.number, classifier);
        result.lemma = Some(
            if only {
                ONLY_REPETITIONS
            } else if list && each {
                EACH_COUNT
            } else {
                REPETITIONS
            }
            .to_owned(),
        );
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

pub(super) const HABIT: &str = "__ja_habitual_topic";

// Attach an alternative rate to the measured event inside the relative
// clause before punctuation can separate the two halves of its contrast.
pub(in crate::ja_zh) fn bind_relative_contrast(atoms: &mut Vec<Atom>) {
    for start in 0..atoms.len() {
        let Some(end) = [3, 4].into_iter().map(|len| start + len).find(|&end| {
            atoms
                .get(start..end)
                .is_some_and(super::contrasts_recurrence)
        }) else {
            continue;
        };
        let neg = &atoms[end - 1];
        let number = end
            + usize::from(
                atoms
                    .get(end)
                    .is_some_and(|a| !a.is_locked && a.surface == "、"),
            );
        let Some(day_idx) = (number..atoms.len()).find(|&i| !is_numeral(&atoms[i])) else {
            continue;
        };
        let day = &atoms[day_idx];
        if day.is_locked
            || day.surface != "日"
            || number == day_idx
            || !atoms
                .get(day_idx + 1)
                .is_some_and(|c| linked_case(c, day, day, "に"))
            || !(neg.head == day.doc_tok_idx || neg.head == day.head)
            || !atoms.iter().any(|a| {
                a.doc_tok_idx == day.head
                    && a.upos == Some(Upos::Verb)
                    && a.deprel == "acl"
                    && atoms.iter().any(|n| {
                        n.doc_tok_idx == a.head && matches!(n.upos, Some(Upos::Noun | Upos::Propn))
                    })
            })
        {
            continue;
        }
        let mut scoped = atoms.clone();
        bind(&mut scoped, day_idx, true);
        let Some(interval) = scoped
            .iter_mut()
            .find(|a| a.doc_tok_idx == day.doc_tok_idx && super::is_interval(a))
        else {
            continue;
        };
        let rate = &atoms[start];
        let text = translate_single_ja_term(&rate.surface, rate.lemma.as_deref(), rate.upos);
        interval.surface = format!("不是{text}，而是{}", interval.surface);
        // A topic before the relative contrast can be attached to なく.
        // Restore the matrix governor when the relative has its own subject.
        let mut first = start;
        let before = start - usize::from(start > 0 && atoms[start - 1].surface == "、");
        if let [.., topic] = &atoms[..before]
            && !topic.is_locked
            && topic.surface == "は"
            && topic.deprel == "case"
            && let Some(subject) = atoms[..before]
                .iter()
                .position(|a| a.doc_tok_idx == topic.head && a.deprel == "nsubj")
            && let Some(matrix) = atoms
                .iter()
                .find(|a| a.upos == Some(Upos::Verb) && a.deprel == "root")
            && (atoms[subject].head == neg.doc_tok_idx || atoms[subject].head == matrix.doc_tok_idx)
            && atoms[day_idx + 1..]
                .iter()
                .any(|a| a.deprel == "nsubj" && a.head == day.head)
        {
            scoped[subject].head = matrix.doc_tok_idx;
            first = before;
        }
        scoped.drain(first..number);
        *atoms = scoped;
        return;
    }
}

// 普段は and its kana spelling are time modifiers of the following event.
// The analyzer can split ふだん into ふ + だ + ん.
pub(super) fn normalize_habit(atoms: &mut Vec<Atom>) {
    let end = match atoms.as_slice() {
        [a, topic, ..] if a.surface == "普段" && topic.surface == "は" => 1,
        [a, da, n, topic, ..]
            if a.surface == "ふ"
                && da.surface == "だ"
                && n.surface == "ん"
                && topic.surface == "は" =>
        {
            3
        }
        _ => return,
    };
    let first = &atoms[0];
    if atoms[..=end].iter().any(|a| a.is_locked)
        || atoms[..=end]
            .windows(2)
            .any(|p| p[0].span.end_char != p[1].span.start_char)
        || !atoms[1..=end].iter().all(|a| a.head == first.doc_tok_idx)
        || !atoms[end + 1..]
            .iter()
            .any(|a| a.doc_tok_idx == first.head && a.upos == Some(Upos::Verb))
        || !atoms[end + 1..]
            .iter()
            .any(|a| a.surface == "日" && a.head == first.head)
    {
        return;
    }
    let mut habit = first.clone();
    habit.upos = Some(Upos::Adv);
    habit.deprel = "advmod".to_owned();
    habit.surface = "平常".to_owned();
    habit.lemma = Some(HABIT.to_owned());
    habit.span.end_char = atoms[end].span.end_char;
    habit.is_locked = true;
    atoms.splice(..=end, [habit]);
}
