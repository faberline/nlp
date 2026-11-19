//! Distinguish a restored time setting from a physical destination of 戻る.
use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, Segment, Span, TargetVariant, Upos,
    bounded_clock_number, clock_case_governor, clock_number, deadline_weekday, explicit_clock_case,
    is_building_floor, is_daypart_surface, is_existing_spatial_head, is_motion_actor,
    is_plain_clock_ni, is_time_setting, linked_joint_case, predicates_has_passive,
    render_bunsetsu_np, setting_time_label,
};

pub(super) struct ClockHandReturn {
    pub(super) dial: Span,
    pub(super) action: String,
    pub(super) object: Option<usize>,
    pub(super) event_time: Option<usize>,
}

fn attached_case(noun: &Atom, marker: &Atom) -> bool {
    !noun.is_locked
        && !marker.is_locked
        && noun.doc_tok_idx.is_some()
        && marker.head == noun.doc_tok_idx
        && marker.deprel == "case"
        && marker.upos == Some(Upos::Adp)
        && noun.span.end_char == marker.span.start_char
}

fn has_hand_prefix(noun: &Atom, source: &[Atom], prefixes: &[&str]) -> bool {
    noun.surface == "針"
        && noun.doc_tok_idx.is_some()
        && source.iter().any(|prefix| {
            !prefix.is_locked
                && prefix.head == noun.doc_tok_idx
                && prefix.deprel == "compound"
                && prefix.span.end_char == noun.span.start_char
                && prefixes.contains(&prefix.surface.as_str())
        })
}

fn clock_hand(noun: &Atom, source: &[Atom]) -> bool {
    let Some(id) = noun.doc_tok_idx else {
        return false;
    };
    if noun.is_locked || noun.upos != Some(Upos::Noun) {
        return false;
    }
    if matches!(
        noun.surface.as_str(),
        "長針" | "短針" | "時針" | "分針" | "秒針"
    ) {
        return true;
    }
    has_hand_prefix(noun, source, &["長", "短", "時", "分", "秒"])
        || noun.surface == "針"
            && source.windows(2).any(|pair| {
                pair[0].head == Some(id)
                    && pair[0].deprel == "nmod"
                    && matches!(
                        pair[0].surface.as_str(),
                        "時計" | "腕時計" | "掛け時計" | "置き時計"
                    )
                    && !pair[0].is_locked
                    && pair[1].surface == "の"
                    && attached_case(&pair[0], &pair[1])
            })
}

fn whole_dial_number(source: &[Atom], last: usize, final_part: &str) -> Option<(usize, u32)> {
    let atom = &source[last];
    if atom.is_locked {
        return None;
    }
    let mut first = last;
    while first > 0 {
        let previous = &source[first - 1];
        let next = &source[first];
        if previous.is_locked
            || previous.upos != Some(Upos::Num)
            || !matches!(previous.deprel.as_str(), "nmod" | "nummod")
            || next.doc_tok_idx.is_none()
            || previous.head != next.doc_tok_idx
            || previous.span.end_char != next.span.start_char
        {
            break;
        }
        first -= 1;
    }
    let mut text: String = source[first..last]
        .iter()
        .map(|a| a.surface.as_str())
        .collect();
    text.push_str(final_part);
    let number = clock_number(&text)?;
    (1..=12).contains(&number).then_some((first, number))
}

fn dial_hour(label: &str) -> Option<u32> {
    let numeral = label.strip_suffix('點')?;
    let hour = if numeral == "兩" {
        2
    } else {
        clock_number(numeral)?
    };
    (1..=12).contains(&hour).then_some(hour)
}

fn second_count(source: &[Atom], unit: usize) -> Option<(usize, u32)> {
    let mark = &source[unit];
    let suffix = mark.surface.strip_suffix('秒')?;
    let head = mark.doc_tok_idx?;
    let mut first = unit;
    while let Some(index) = first.checked_sub(1) {
        let number = &source[index];
        let next = &source[first];
        if number.is_locked
            || number.span.end_char != next.span.start_char
            || !matches!(number.deprel.as_str(), "nmod" | "nummod" | "compound")
            || !number
                .head
                .is_some_and(|token| token == head || Some(token) == next.doc_tok_idx)
            || bounded_clock_number(&number.surface, u32::MAX).is_none()
        {
            break;
        }
        first = index;
    }
    let number: String = source[first..unit]
        .iter()
        .map(|atom| atom.surface.as_str())
        .chain(std::iter::once(suffix))
        .collect();
    Some((first, bounded_clock_number(&number, u32::MAX)?))
}

fn is_clock_time_qualifier(time: &Atom) -> bool {
    use faberline_nlp::lexicon::{SemanticClass, lookup_ja};

    let word = time.surface.as_str();
    is_daypart_surface(word)
        || word
            .strip_suffix(['前', '後'])
            .is_some_and(is_daypart_surface)
        || word
            .strip_prefix(['今', '昨', '明', '翌', '毎'])
            .is_some_and(|part| matches!(part, "朝" | "晩" | "夕" | "夜"))
        || word
            .strip_prefix(['前', '先', '今', '昨', '来', '翌', '毎'])
            .is_some_and(|period| matches!(period, "週" | "月" | "年"))
        || deadline_weekday(word).is_some()
        || matches!(
            word,
            "一昨日"
                | "昨日"
                | "今日"
                | "明日"
                | "明後日"
                | "本日"
                | "当日"
                | "前日"
                | "翌日"
                | "先日"
                | "毎日"
                | "いつも"
                | "普段"
        )
        || lookup_ja(word).is_some_and(|entry| entry.sem_class == Some(SemanticClass::Time))
}

fn simultaneous_prefix(atoms: &[Atom]) -> Option<(&Atom, &[Atom])> {
    match atoms {
        [time, rest @ ..]
            if !time.is_locked
                && matches!(
                    time.surface.as_str(),
                    "そのとき" | "その時" | "そのときに" | "その時に" | "同時に"
                ) =>
        {
            Some((time, rest))
        }
        [first, second, rest @ ..]
            if !first.is_locked
                && !second.is_locked
                && first.span.end_char == second.span.start_char
                && first.surface == "その"
                && matches!(second.surface.as_str(), "とき" | "時" | "ときに" | "時に") =>
        {
            Some((second, rest))
        }
        [time, ni, rest @ ..]
            if time.surface == "同時" && ni.surface == "に" && attached_case(time, ni) =>
        {
            Some((time, rest))
        }
        _ => None,
    }
}

fn dial_case_governor(noun: &Atom, marker: &Atom, relation: &str, doc: &Document) -> Option<usize> {
    if noun.lemma.as_deref() == Some(super::CLOCK_ATOM) {
        let head = marker.head?;
        if marker.is_locked
            || marker.upos != Some(Upos::Adp)
            || noun.span.end_char != marker.span.start_char
            || !doc
                .syntax
                .iter()
                .any(|node| node.token == head && node.deprel == relation)
        {
            return None;
        }
        clock_case_governor(noun, marker, doc)
    } else if noun.deprel == relation && attached_case(noun, marker) {
        noun.head
    } else {
        None
    }
}

fn indicated_hour(atoms: &[Atom], end: usize, doc: &Document) -> Option<(usize, u32)> {
    let value_index = if atoms[end].surface == "位置" {
        let [.., number, no, position] = &atoms[..=end] else {
            return None;
        };
        if no.surface != "の"
            || position.doc_tok_idx.is_none()
            || dial_case_governor(number, no, "nmod", doc) != position.doc_tok_idx
            || no.span.end_char != position.span.start_char
        {
            return None;
        }
        end - 2
    } else {
        end
    };
    let value = &atoms[value_index];
    if value.lemma.as_deref() == Some(super::CLOCK_ATOM) {
        return Some((value_index, dial_hour(&value.surface)?));
    }
    if value.surface == "時" {
        let index = value_index.checked_sub(1)?;
        let number = &atoms[index];
        if value.doc_tok_idx.is_none()
            || !matches!(number.deprel.as_str(), "nmod" | "nummod")
            || number.head != value.doc_tok_idx
            || number.span.end_char != value.span.start_char
        {
            return None;
        }
        return whole_dial_number(atoms, index, &number.surface);
    }
    whole_dial_number(
        atoms,
        value_index,
        value.surface.strip_suffix('時').unwrap_or(&value.surface),
    )
}

fn hand_kind(hand: &Atom, source: &[Atom]) -> Option<&'static str> {
    ["長", "短", "時", "分", "秒"].into_iter().find(|&kind| {
        hand.surface.strip_suffix('針') == Some(kind) || has_hand_prefix(hand, source, &[kind])
    })
}

fn repeated_clock_owner(clock: &Atom, marker: &Atom, clause: &[Atom]) -> bool {
    if clock.deprel != "nmod"
        || !matches!(
            clock.surface.as_str(),
            "時計" | "腕時計" | "掛け時計" | "置き時計"
        )
        || marker.surface != "の"
        || !attached_case(clock, marker)
    {
        return false;
    }
    let mut modifiers = clause.iter().filter(|atom| {
        atom.head == clock.doc_tok_idx
            && matches!(atom.deprel.as_str(), "det" | "nmod" | "acl" | "amod")
    });
    modifiers.next().is_some_and(|atom| {
        !atom.is_locked
            && atom.surface == "同じ"
            && atom.deprel == "det"
            && atom.span.end_char <= clock.span.start_char
    }) && modifiers.next().is_none()
}

fn same_pointing_hand(subject: &Atom, clause: &[Atom], hand: &Atom, source: &[Atom]) -> bool {
    let kind = hand_kind(subject, clause);
    if subject.is_locked
        || !(subject.surface == "針" && kind.is_none()
            || kind.is_some() && kind == hand_kind(hand, source))
    {
        return false;
    }
    let mut modifiers = clause.iter().enumerate().filter(|(_, atom)| {
        atom.head == subject.doc_tok_idx && matches!(atom.deprel.as_str(), "nmod" | "acl" | "amod")
    });
    match modifiers.next() {
        None => true,
        Some((index, clock)) => {
            modifiers.next().is_none()
                && clause
                    .get(index + 1)
                    .is_some_and(|marker| repeated_clock_owner(clock, marker, clause))
        }
    }
}

fn dial_face_start(source: &[Atom], first: usize) -> Option<usize> {
    let [prefix @ .., dial, particle] = &source[..first] else {
        return None;
    };
    if particle.surface != "の"
        || particle.span.end_char != source[first].span.start_char
        || !attached_case(dial, particle)
        || dial.head.is_none()
        || dial.head != source[first].doc_tok_idx
    {
        return None;
    }
    if dial.surface == "文字盤" {
        return Some(first - 2);
    }
    let stem = prefix.last()?;
    (stem.surface == "文字"
        && dial.surface == "盤"
        && !stem.is_locked
        && stem.head == dial.doc_tok_idx
        && stem.deprel == "compound"
        && stem.span.end_char == dial.span.start_char)
        .then_some(first - 3)
}

fn indication_polarity(tail: &[Atom], source: &[char], after_verb: usize) -> Option<bool> {
    // Suffix tokenization can split a modal across two rendering clauses.
    let raw_tail: String = source
        .get(after_verb..)?
        .iter()
        .copied()
        .take_while(|&ch| {
            !matches!(
                ch,
                '、' | '，' | '。' | '！' | '？' | '!' | '?' | '」' | '』' | '\n'
            )
        })
        .collect();
    if [
        "でしょう",
        "でしょうか",
        "だろう",
        "だろうか",
        "なら",
        "たら",
        "ならば",
    ]
    .iter()
    .any(|&suffix| raw_tail.ends_with(suffix))
    {
        return None;
    }
    let mut points_at = true;
    for atom in tail {
        let word = atom.lemma.as_deref().unwrap_or(&atom.surface);
        if atom.surface == "か" && atom.deprel == "discourse"
            || matches!(atom.surface.as_str(), "なら" | "たら" | "ば")
                && atom.upos == Some(Upos::Sconj)
            || matches!(
                atom.surface.as_str(),
                "だろ" | "だろう" | "でしょ" | "でしょう"
            )
            || atom.upos == Some(Upos::Verb) && !matches!(word, "いる" | "おる")
        {
            return None;
        }
        if (matches!(word, "ない" | "ぬ" | "ず")
            && matches!(atom.deprel.as_str(), "aux" | "advmod" | "mark"))
            || word == "ん" && atom.upos == Some(Upos::Aux) && atom.deprel == "aux"
        {
            if !points_at {
                return None;
            }
            points_at = false;
        }
    }
    Some(points_at)
}

fn reading_at_return(
    following: &[Segment],
    returning: &Atom,
    hand: &Atom,
    source: &[Atom],
    doc: &Document,
    chars: &[char],
) -> Option<(u32, bool)> {
    let [
        Segment::Clause {
            atoms,
            trailing_punct,
            ..
        },
        next @ ..,
    ] = following
    else {
        return None;
    };
    let (when, rest) = simultaneous_prefix(atoms)?;
    let (rest, end_punct) =
        if rest.is_empty() && matches!(trailing_punct.as_deref(), Some("、" | "，")) {
            let [
                Segment::Clause {
                    atoms,
                    trailing_punct,
                    ..
                },
                ..,
            ] = next
            else {
                return None;
            };
            (atoms.as_slice(), trailing_punct.as_deref())
        } else {
            (rest, trailing_punct.as_deref())
        };
    let pointing = returning.head?;
    let indicator = rest.iter().position(|atom| {
        !atom.is_locked
            && atom.doc_tok_idx == Some(pointing)
            && atom.upos == Some(Upos::Verb)
            && atom.lemma.as_deref() == Some("指す")
    })?;
    if matches!(end_punct, Some("？" | "?")) || returning.deprel != "advcl" {
        return None;
    }
    let points_at =
        indication_polarity(&rest[indicator + 1..], chars, rest[indicator].span.end_char)?;
    let mut subjects = rest.iter().filter(|atom| atom.deprel == "nsubj");
    let subject = subjects.next();
    if subjects.next().is_some()
        || subject.is_some_and(|subject| {
            subject.head != Some(pointing) || !same_pointing_hand(subject, rest, hand, source)
        })
    {
        return None;
    }
    let mut readings = rest.windows(2).enumerate().filter(|(_, pair)| {
        let [noun, marker] = pair else { return false };
        marker.surface == "を" && dial_case_governor(noun, marker, "obj", doc) == Some(pointing)
    });
    let (index, _) = readings.next()?;
    if readings.next().is_some() {
        return None;
    }
    let (first, hour) = indicated_hour(rest, index, doc)?;
    let first = dial_face_start(rest, first).unwrap_or(first);
    let reading_span = Span {
        start_char: rest[first].span.start_char,
        end_char: rest[index].span.end_char,
    };
    when.head
        .is_some_and(|owner| {
            owner == pointing
                || subject.is_some_and(|subject| subject.doc_tok_idx == Some(owner))
                || doc
                    .tokens
                    .get(owner)
                    .is_some_and(|token| contains_span(reading_span, token.span))
        })
        .then_some((hour, points_at))
}

fn stated_dial_position(
    source: &[Atom],
    doc: &Document,
    head: usize,
    at: usize,
    second_hand: bool,
) -> Option<(Span, String)> {
    let [prefix @ .., modifier, no, position] = &source[..=at] else {
        return None;
    };
    let case = source.get(at + 1)?;
    let focus = source.get(at + 2).filter(|focus| {
        !focus.is_locked
            && matches!(focus.surface.as_str(), "は" | "も")
            && focus.upos == Some(Upos::Adp)
            && focus.deprel == "case"
            && focus.head == case.head
            && focus.span.start_char == case.span.end_char
    });
    if position.is_locked
        || position.surface != "位置"
        || position.head != Some(head)
        || !(matches!(position.deprel.as_str(), "obl" | "advmod")
            || position.deprel == "nsubj" && focus.is_some())
        || !matches!(case.surface.as_str(), "に" | "へ")
        || !attached_case(position, case)
        || no.is_locked
        || no.surface != "の"
        || no.upos != Some(Upos::Adp)
        || no.deprel != "case"
        || modifier.span.end_char != no.span.start_char
        || no.span.end_char != position.span.start_char
    {
        return None;
    }
    // Clock normalization joins 六時 before rendering. Verify its original
    // nominal head through the source genitive marker and dependency.
    let normalized_hour = if modifier.lemma.as_deref() == Some(super::CLOCK_ATOM) {
        let head = no.head?;
        let node = doc.syntax.iter().find(|node| node.token == head)?;
        if node.head != position.doc_tok_idx
            || node.deprel != "nmod"
            || !contains_span(modifier.span, doc.tokens.get(head)?.span)
        {
            return None;
        }
        Some(dial_hour(&modifier.surface)?)
    } else {
        if modifier.head != position.doc_tok_idx
            || modifier.deprel != "nmod"
            || !attached_case(modifier, no)
        {
            return None;
        }
        None
    };
    let (mut first, mut label) = if let Some(hour) = normalized_hour {
        (at - 2, format!("{hour}點鐘的位置"))
    } else if modifier.surface == "元" {
        (at - 2, "原位".to_owned())
    } else if second_hand && modifier.surface.ends_with('秒') {
        let (begin, second) = second_count(source, at - 2).filter(|&(_, value)| value < 60)?;
        (begin, format!("{second}秒的位置"))
    } else if modifier.surface == "時" {
        let last = prefix.len().checked_sub(1)?;
        let value = &source[last];
        if value.upos != Some(Upos::Num)
            || value.head != modifier.doc_tok_idx
            || value.span.end_char != modifier.span.start_char
        {
            return None;
        }
        let (begin, number) = whole_dial_number(source, last, &value.surface)?;
        (begin, format!("{number}點鐘的位置"))
    } else if let Some(hour) = modifier.surface.strip_suffix('時') {
        let (begin, number) = whole_dial_number(source, at - 2, hour)?;
        (begin, format!("{number}點鐘的位置"))
    } else {
        if modifier.upos != Some(Upos::Num) {
            return None;
        }
        let (begin, number) = whole_dial_number(source, at - 2, &modifier.surface)?;
        (begin, format!("{number}的位置"))
    };
    if let Some(start) = dial_face_start(source, first) {
        first = start;
        label = format!("鐘面上{label}");
    }
    Some((
        Span {
            start_char: source[first].span.start_char,
            end_char: focus.unwrap_or(case).span.end_char,
        },
        label,
    ))
}

pub(super) fn clock_hand_return(
    groups: &[Bunsetsu],
    predicates: &[Bunsetsu],
    source: &[Atom],
    doc: &Document,
    following: &[Segment],
    chars: &[char],
    target: TargetVariant,
) -> Option<ClockHandReturn> {
    let mut verbs = predicates
        .iter()
        .flat_map(|predicate| &predicate.atoms)
        .filter(|a| a.upos == Some(Upos::Verb));
    let verb = verbs.next()?;
    if verb.is_locked
        || predicates_has_passive(predicates)
        || predicates
            .iter()
            .flat_map(|predicate| &predicate.atoms)
            .any(|a| matches!(a.lemma.as_deref(), Some("せる" | "させる")))
    {
        return None;
    }
    let (role, relation, action) = match verb.lemma.as_deref().unwrap_or(&verb.surface) {
        "戻る" => (BunsetsuRole::Subj, "nsubj", "回到"),
        "戻す" => (BunsetsuRole::Obj, "obj", "撥回"),
        _ => return None,
    };
    let head = verb.doc_tok_idx?;
    let action = match verbs.next() {
        None => action,
        Some(plan)
            if role == BunsetsuRole::Subj
                && !plan.is_locked
                && plan.surface == "予定"
                && plan.doc_tok_idx.is_some()
                && verb.head == plan.doc_tok_idx
                && verb.deprel == "advcl"
                && verb.span.end_char == plan.span.start_char
                && verbs.next().is_none()
                && source.windows(2).any(|pair| {
                    pair[0].doc_tok_idx == plan.doc_tok_idx
                        && !pair[1].is_locked
                        && matches!(pair[1].surface.as_str(), "です" | "だ")
                        && pair[1].head == plan.doc_tok_idx
                        && pair[1].deprel == "aux"
                        && pair[0].span.end_char == pair[1].span.start_char
                }) =>
        {
            "預定回到"
        }
        _ => return None,
    };
    let mut hands = groups
        .iter()
        .enumerate()
        .filter(|(_, group)| group.role == role)
        .filter_map(|(index, group)| {
            source
                .iter()
                .find(|noun| {
                    noun.head == Some(head)
                        && noun.deprel == relation
                        && clock_hand(noun, source)
                        && group
                            .atoms
                            .iter()
                            .any(|part| contains_span(part.span, noun.span))
                })
                .map(|noun| (index, noun))
        });
    let (hand, hand_noun) = hands.next()?;
    if hands.next().is_some() {
        return None;
    }
    let second_hand = hand_noun.surface == "秒針" || has_hand_prefix(hand_noun, source, &["秒"]);
    let mut positions = source.iter().enumerate().filter_map(|(idx, _)| {
        let (span, label) = stated_dial_position(source, doc, head, idx, second_hand)?;
        let mut parts = groups
            .iter()
            .filter(|group| {
                group
                    .atoms
                    .iter()
                    .any(|a| a.span.start_char < span.end_char && span.start_char < a.span.end_char)
            })
            .peekable();
        (parts.peek().is_some()
            && parts.all(|group| {
                group
                    .atoms
                    .iter()
                    .all(|atom| contains_span(span, atom.span))
            }))
        .then_some((span, label))
    });
    let point = |group: &Bunsetsu| {
        (is_plain_clock_ni(group)
            && explicit_clock_case(group)
                .is_some_and(|(clock, case)| clock_case_governor(clock, case, doc) == Some(head)))
            || nominal_case(group, head)
                .is_some_and(|(time, case)| case == "に" && is_daypart_surface(&time.surface))
    };
    let competing_destination = |group: &Bunsetsu| {
        matches!(group.case_particle.as_deref(), Some("に" | "へ")) && !point(group)
    };
    let (dial, label, event_time) = if let Some((span, label)) = positions.next() {
        if positions.next().is_some() {
            return None;
        }
        let outside = groups.iter().enumerate().filter(|(_, group)| {
            !group
                .atoms
                .iter()
                .all(|atom| contains_span(span, atom.span))
        });
        if outside
            .clone()
            .any(|(_, group)| competing_destination(group))
        {
            return None;
        }
        let mut events = outside.filter_map(|(idx, group)| point(group).then_some(idx));
        let when = events.next();
        if events.next().is_some() {
            return None;
        }
        (span, label, when)
    } else {
        // A second destination may be a physical place or locked text.
        // Either way it prevents treating a clock reading as the dial setting.
        if groups.iter().any(|group| {
            group.case_particle.as_deref() == Some("から") || competing_destination(group)
        }) {
            return None;
        }
        let mut points = groups.iter().filter(|group| point(group));
        let value = points.next()?;
        if points.next().is_some() {
            return None;
        }
        let span = Span {
            start_char: value.atoms.first()?.span.start_char,
            end_char: value.atoms.last()?.span.end_char,
        };
        // A date, daypart or frequency can qualify the clock time of the return.
        // Preserve that reading when there is no stated dial position.
        if source.iter().any(|time| {
            !time.is_locked
                && matches!(time.deprel.as_str(), "advmod" | "obl")
                && matches!(time.upos, Some(Upos::Adv | Upos::Noun))
                && time.head.is_some_and(|token| {
                    token == head
                        || doc
                            .tokens
                            .get(token)
                            .is_some_and(|token| contains_span(span, token.span))
                })
                && is_clock_time_qualifier(time)
        }) {
            return None;
        }
        if second_hand
            && source.iter().enumerate().any(|(index, amount)| {
                !amount.is_locked
                    && amount.head == Some(head)
                    && matches!(amount.deprel.as_str(), "obl" | "advmod")
                    && second_count(source, index).is_some()
            })
        {
            // A standalone count describes the return, outside any dial position.
            return None;
        }
        let label = setting_time_label(value, target);
        let hour = if label == "正午" {
            12
        } else {
            dial_hour(&label)?
        };
        // A separately stated reading can rule out a destination on the dial.
        if role == BunsetsuRole::Subj
            && reading_at_return(following, verb, hand_noun, source, doc, chars)
                .is_some_and(|(reading, points_at)| points_at != (reading == hour))
        {
            return None;
        }
        let label = if label == "正午" {
            "正午的位置".to_owned()
        } else {
            format!("{label}鐘的位置")
        };
        (span, label, None)
    };
    Some(ClockHandReturn {
        dial,
        action: format!("{action}{label}"),
        object: (role == BunsetsuRole::Obj).then_some(hand),
        event_time,
    })
}

pub(super) fn contains_span(outer: Span, inner: Span) -> bool {
    outer.start_char <= inner.start_char && inner.end_char <= outer.end_char
}

pub(super) fn is_place(noun: &Atom) -> bool {
    use faberline_nlp::lexicon::{SemanticClass, lookup_ja};
    !noun.is_locked
        && (is_existing_spatial_head(noun)
            || is_building_floor(noun)
            || matches!(noun.surface.as_str(), "店" | "岸" | "島" | "庭")
            || lookup_ja(&noun.surface).is_some_and(|entry| {
                entry.decided_upos == noun.upos && entry.sem_class == Some(SemanticClass::Location)
            }))
}

// A relative clause and a linked clause render before the main verb.
// Keep a daypart with the local 戻る if it has a destination or a mover.
pub(super) fn local_motion(verb: &Atom, clause: &[Bunsetsu], source: &[Atom]) -> bool {
    let Some(head) = verb.doc_tok_idx else {
        return false;
    };
    // In 戻りましたが the analyzer can attach the finite return to the next
    // verb as nsubj. Its own arguments still belong to the local predicate.
    let mut predicates = clause.iter().filter(|b| b.role == BunsetsuRole::Predicate);
    let Some(local) = predicates.next() else {
        return false;
    };
    if predicates.next().is_some()
        || !local
            .atoms
            .iter()
            .any(|a| !a.is_locked && a.doc_tok_idx == Some(head))
    {
        return false;
    }
    let spatial_path = clause.iter().any(|group| {
        nominal_case(group, head).is_some_and(|(noun, case)| {
            case == "へ" && !super::source_time_head(noun)
                || matches!(case, "に" | "まで" | "から") && is_place(noun)
        })
    });
    if spatial_path {
        return true;
    }
    let mover = |a: &Atom| {
        is_motion_actor(a)
            || !a.is_locked
                && a.upos == Some(Upos::Noun)
                && matches!(a.surface.as_str(), "客" | "車")
    };
    let mut subjects = source
        .windows(2)
        .filter(|w| {
            w[0].head == Some(head)
                && w[0].deprel == "nsubj"
                && w[0].doc_tok_idx.is_some()
                && w[1].head == w[0].doc_tok_idx
                && w[1].deprel == "case"
                && matches!(w[1].surface.as_str(), "が" | "は" | "も")
        })
        .map(|w| &w[0])
        .peekable();
    if subjects.peek().is_some() {
        return subjects.any(mover);
    }
    // Only an unfilled subject can be supplied by the following relative noun.
    verb.deprel == "acl"
        && source
            .iter()
            .any(|a| a.doc_tok_idx.is_some() && a.doc_tok_idx == verb.head && mover(a))
}

// In 夕方に古い店に戻る the analyzer can attach the time to 古い. Age
// identifies the place here. A temporary property such as 暗い may itself
// hold only at the named time.
pub(super) fn time_before_place_modifier(time: &Atom, source: &[Atom]) -> bool {
    let Some(adjective) = source.iter().find(|a| {
        a.doc_tok_idx.is_some()
            && a.doc_tok_idx == time.head
            && !a.is_locked
            && a.deprel == "amod"
            && a.upos == Some(Upos::Adj)
            && matches!(a.surface.as_str(), "古い" | "新しい")
    }) else {
        return false;
    };
    let Some(place) = source.iter().find(|a| {
        a.doc_tok_idx.is_some()
            && a.doc_tok_idx == adjective.head
            && a.deprel == "obl"
            && is_place(a)
    }) else {
        return false;
    };
    source.iter().any(|a| {
        a.doc_tok_idx.is_some()
            && a.doc_tok_idx == place.head
            && !a.is_locked
            && a.lemma.as_deref().unwrap_or(&a.surface) == "戻る"
    }) && source
        .iter()
        .any(|case| linked_joint_case(place, case) && matches!(case.surface.as_str(), "に" | "へ"))
}

fn schedule_noun(noun: &Atom, source: &[Atom]) -> bool {
    !noun.is_locked
        && (matches!(
            noun.surface.as_str(),
            "時間" | "時刻" | "日時" | "日程" | "日付" | "予定"
        ) || matches!(noun.surface.as_str(), "枠" | "帯") && is_time_setting(noun, source))
}

// In 元に戻る the former state is explicit. Another time on the same
// verb describes when it was restored. A possessor's 元 can mean their side.
fn original_schedule_value(noun: &Atom, source: &[Atom]) -> bool {
    let Some(id) = noun.doc_tok_idx else {
        return false;
    };
    let has_genitive = |n: &Atom| {
        source
            .iter()
            .any(|a| a.deprel == "nmod" && a.head == n.doc_tok_idx)
    };
    if noun.surface == "元" {
        return !noun.is_locked && !has_genitive(noun);
    }
    schedule_noun(noun, source)
        && source.windows(2).any(|w| {
            !w[0].is_locked
                && w[0].surface == "元"
                && w[0].deprel == "nmod"
                && w[0].head == Some(id)
                && !has_genitive(&w[0])
                && !w[1].is_locked
                && w[1].surface == "の"
                && w[1].deprel == "case"
                && w[0].doc_tok_idx.is_some()
                && w[1].head == w[0].doc_tok_idx
        })
}

pub(super) fn has_original_value(source: &[Atom], head: usize) -> bool {
    source.windows(2).any(|w| {
        w[0].head == Some(head)
            && original_schedule_value(&w[0], source)
            && w[1].surface == "に"
            && linked_joint_case(&w[0], &w[1])
    })
}

// Strip a focusing は / も only after checking that both case markers share
// the noun. A focused destination may carry an nsubj dependency.
fn nominal_case(group: &Bunsetsu, head: usize) -> Option<(&Atom, &str)> {
    let (atoms, focused) = match group.atoms.as_slice() {
        [prefix @ .., case, topic]
            if matches!(topic.surface.as_str(), "は" | "も")
                && !topic.is_locked
                && topic.head == case.head =>
        {
            (&group.atoms[..prefix.len() + 1], true)
        }
        atoms => (atoms, false),
    };
    let [.., noun, case] = atoms else { return None };
    if noun.is_locked
        || case.is_locked
        || noun.head != Some(head)
        || noun.doc_tok_idx.is_none()
        || case.head != noun.doc_tok_idx
        || case.deprel != "case"
        || case.upos != Some(Upos::Adp)
        || noun.span.end_char != case.span.start_char
        || !(matches!(noun.deprel.as_str(), "obl" | "advmod") || focused && noun.deprel == "nsubj")
    {
        return None;
    }
    Some((noun, &case.surface))
}

// A time or a place can be the result of 戻る. Match its own governor and
// subject before removing the case marker; an event time keeps its usual role.
pub(super) fn resolve(
    groups: &[Bunsetsu],
    predicates: &[Bunsetsu],
    source: &[Atom],
    doc: &Document,
    target: TargetVariant,
) -> Option<(usize, String)> {
    let [predicate] = predicates else { return None };
    let mut verbs = predicate
        .atoms
        .iter()
        .filter(|a| a.upos == Some(Upos::Verb));
    let verb = verbs.next()?;
    if verb.is_locked
        || verb.lemma.as_deref().unwrap_or(&verb.surface) != "戻る"
        || verbs.next().is_some()
        || predicates_has_passive(predicates)
        || predicate
            .atoms
            .iter()
            .any(|a| matches!(a.lemma.as_deref(), Some("せる" | "させる")))
    {
        return None;
    }
    let head = verb.doc_tok_idx?;
    let schedule = source
        .iter()
        .any(|a| a.head == Some(head) && a.deprel == "nsubj" && schedule_noun(a, source));
    let explicit_original = schedule && has_original_value(source, head);
    let mut endpoints = groups.iter().enumerate().filter_map(|(index, group)| {
        if schedule && !explicit_original && is_plain_clock_ni(group) {
            let (clock, case) = explicit_clock_case(group)?;
            return (clock_case_governor(clock, case, doc) == Some(head)).then(|| {
                (
                    index,
                    format!("恢復到{}", setting_time_label(group, target)),
                )
            });
        }
        let (noun, case) = nominal_case(group, head)?;
        if !matches!(case, "に" | "へ") {
            return None;
        }
        if schedule && case == "に" {
            if explicit_original && original_schedule_value(noun, source) {
                if noun.surface == "元" && group.atoms[0].doc_tok_idx == noun.doc_tok_idx {
                    return Some((index, "恢復原狀".to_owned()));
                }
                let label = render_bunsetsu_np(group, target);
                return label
                    .strip_prefix("元的")
                    .map(|time| (index, format!("恢復到原本的{time}")));
            }
            if !explicit_original && is_daypart_surface(&noun.surface) {
                return Some((
                    index,
                    format!("恢復到{}", setting_time_label(group, target)),
                ));
            }
        }
        is_place(noun).then(|| (index, format!("回到{}", render_bunsetsu_np(group, target))))
    });
    let endpoint = endpoints.next()?;
    endpoints.next().is_none().then_some(endpoint)
}
