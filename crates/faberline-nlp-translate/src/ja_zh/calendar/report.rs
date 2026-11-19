//! Keep an unquoted と complement with its reporting verb. The subordinate
//! event gets a local root so its day, subject and quantity render together.
use super::{is_date, linked_case, number_before};
use crate::ja_zh::{
    Atom, Bunsetsu, BunsetsuRole, Document, Span, SpanLockTable, TargetVariant, Upos,
    core_non_punct, deadline_time_question, direct_quote_time, is_floor_facility, is_punct_atom,
    render_bunsetsu_np, render_direct_quote_person, render_temporal_made_atom,
};

pub(in crate::ja_zh) struct Report {
    pub(in crate::ja_zh) text: String,
    pub(in crate::ja_zh) spans: Vec<Span>,
}

// An argument-free prenominal ある in ある学校 is the indefinite determiner.
fn indefinite_modifier(a: &Atom, phrase: &[Atom]) -> bool {
    !a.is_locked
        && a.surface == "ある"
        && a.deprel == "acl"
        && !phrase.iter().any(|x| {
            x.head == a.doc_tok_idx && matches!(x.deprel.as_str(), "obj" | "obl" | "nsubj")
        })
        && phrase.iter().any(|n| {
            n.doc_tok_idx == a.head
                && a.span.end_char == n.span.start_char
                && matches!(n.upos, Some(Upos::Noun | Upos::Propn))
        })
}

// In 来たと若い店員が答えた, the analyzer can attach 来た to the adjective
// in the speaker phrase. Another verb may instead own the quotation.
fn nominal_path(mut parent: Option<usize>, target: Option<usize>, inside: &[Atom]) -> bool {
    while let Some(index) = parent {
        if parent == target {
            return true;
        }
        let Some(a) = inside.iter().find(|a| a.doc_tok_idx == parent) else {
            return false;
        };
        if !(indefinite_modifier(a, inside)
            || matches!(
                a.deprel.as_str(),
                "amod" | "nmod" | "compound" | "det" | "nsubj" | "obl"
            ) && matches!(
                a.upos,
                Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::Det | Upos::Adj)
            ))
        {
            return false;
        }
        parent = a.head.filter(|next| *next > index);
    }
    false
}

fn nominal(noun: &[Atom], doc: &Document, locks: &SpanLockTable) -> Option<String> {
    let (head, modifiers) = noun.split_last()?;
    if let [first, tail @ ..] = noun
        && indefinite_modifier(first, noun)
    {
        return Some(format!("某{}", nominal(tail, doc, locks)?));
    }
    if let Some(i) = modifiers
        .iter()
        .position(|a| !a.is_locked && a.surface == "と" && a.deprel == "case")
        && i > 0
        && modifiers[i].head == noun[i - 1].doc_tok_idx
        && (noun[i - 1].head == head.head
            || nominal_path(noun[i - 1].head, head.doc_tok_idx, &noun[i + 1..]))
    {
        return Some(format!(
            "{}和{}",
            nominal(&noun[..i], doc, locks)?,
            nominal(&noun[i + 1..], doc, locks)?
        ));
    }
    if !(head.is_locked || matches!(head.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron)))
        || !modifiers
            .iter()
            .all(|a| a.head.is_some() && noun.iter().any(|inside| inside.doc_tok_idx == a.head))
    {
        return None;
    }
    if modifiers
        .iter()
        .any(|a| !a.is_locked && !matches!(a.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron)))
    {
        let sentence = doc.tokens.get(head.doc_tok_idx?)?.sentence;
        if let Some(text) = crate::ja_zh::nominal_possession::noun_phrase(
            noun,
            doc,
            sentence,
            TargetVariant::ZhHant,
        ) {
            return Some(text);
        }
        if !modifiers.iter().all(|a| {
            a.is_locked
                || matches!(
                    a.upos,
                    Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::Det | Upos::Adj)
                )
                || a.surface == "の" && a.deprel == "case"
        }) {
            return None;
        }
    }
    if let [person] = noun
        && person.is_locked
    {
        return Some(person.surface.clone());
    }
    if let [modifiers @ .., person] = noun
        && noun.iter().all(|a| !a.is_locked)
        && let Some((_, name)) = crate::ja_zh::classroom_role(modifiers, person)
    {
        return Some(name.to_owned());
    }
    Some(
        render_direct_quote_person(noun, doc, locks).unwrap_or_else(|| {
            render_bunsetsu_np(
                &Bunsetsu {
                    atoms: noun.to_vec(),
                    role: BunsetsuRole::Other,
                    case_particle: None,
                    sconj_tail: None,
                    sfp_tail: None,
                },
                TargetVariant::ZhHant,
            )
        }),
    )
}

fn is_listener(words: &[Atom], doc: &Document, locks: &SpanLockTable) -> bool {
    words
        .last()
        .is_some_and(crate::ja_zh::is_demonstrative_person_noun)
        || render_direct_quote_person(words, doc, locks).is_some()
}

#[derive(Default)]
struct Frame {
    person: Option<String>,
    time: Option<String>,
    recipient: Option<String>,
    also: bool,
    wh: bool,
    place: Option<String>,
}

fn reporting_time(words: &[Atom], doc: &Document, chars: &[char]) -> Option<String> {
    // A glossary placeholder keeps the syntax of its final source token.
    // Classify a covered numeric date from its source spelling and keep the
    // placeholder intact for restoration, including its forced spelling.
    if let [opaque] = words
        && opaque.is_locked
        && !is_date(opaque)
    {
        let start = doc
            .tokens
            .partition_point(|t| t.span.end_char <= opaque.span.start_char);
        let end = opaque.doc_tok_idx?;
        let tokens = doc.tokens.get(start..=end)?;
        if tokens.first()?.span.start_char != opaque.span.start_char
            || tokens.last()?.span.end_char != opaque.span.end_char
        {
            return None;
        }
        let spelling: String = chars
            .get(opaque.span.start_char..opaque.span.end_char)?
            .iter()
            .collect();
        return super::dates::spelled_report_day(&spelling)
            .map(|_| format!("在{}", opaque.surface));
    }
    if let Some(date) = super::dates::report_day(words) {
        return Some(format!("在{date}"));
    }
    let time = |a: &Atom| {
        deadline_time_question(a)
            .map(str::to_owned)
            .or_else(|| direct_quote_time(a, doc))
            .or_else(|| render_temporal_made_atom(a))
    };
    match words {
        [a] => time(a),
        [day, link, part]
            if !link.is_locked
                && link.surface == "の"
                && link.head == day.doc_tok_idx
                && day.head == part.doc_tok_idx =>
        {
            if !day.is_locked && !part.is_locked && part.surface == "日" && day.surface == "次" {
                return Some("隔天".to_owned());
            }
            Some(format!("{}{}", time(day)?, time(part)?))
        }
        _ => None,
    }
}

impl Frame {
    // The case and its governor establish the role, including for occupations
    // and names for which the source dictionary supplies no person label.
    fn argument(
        &mut self,
        args: &[Atom],
        governor: Option<usize>,
        doc: &Document,
        locks: &SpanLockTable,
        chars: &[char],
    ) -> Option<()> {
        let (case, words) = args.split_last()?;
        if case.is_locked || !matches!(case.deprel.as_str(), "case" | "discourse") {
            return None;
        }
        let owner = words
            .iter()
            .find(|a| a.doc_tok_idx == case.head && a.head == governor)?;
        let wh = words.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.surface.as_str(),
                    "誰" | "だれ" | "どなた" | "何" | "いつ" | "どこ" | "何時"
                )
        }) || words.iter().any(|a| super::question(a).is_some());
        match (case.surface.as_str(), owner.deprel.as_str()) {
            ("は" | "が" | "も", "nsubj" | "obl")
                if self.person.is_none() && (owner.deprel == "nsubj" || case.surface == "も") =>
            {
                let (noun, only) = if let [noun @ .., focus] = words
                    && !focus.is_locked
                    && focus.surface == "だけ"
                    && focus.head == owner.doc_tok_idx
                {
                    (noun, true)
                } else {
                    (words, false)
                };
                let person = nominal(noun, doc, locks)?;
                self.person = Some(if only {
                    format!("只有{person}")
                } else {
                    person
                });
                self.also = case.surface == "も";
            }
            ("に" | "へ", "obl" | "iobj") => {
                if case.surface == "に"
                    && let Some(time) = reporting_time(words, doc, chars)
                {
                    if self.time.is_some() {
                        return None;
                    }
                    self.time = Some(time);
                } else {
                    if self.recipient.is_some() {
                        return None;
                    }
                    // Japanese also marks the question being answered with に.
                    // Its form alone does not identify a listener.
                    if owner.deprel != "iobj"
                        && !owner.is_locked
                        && !matches!(owner.surface.as_str(), "誰" | "だれ" | "どなた")
                        && !is_listener(words, doc, locks)
                    {
                        return None;
                    }
                    self.recipient = Some(format!("向{}", nominal(words, doc, locks)?));
                }
            }
            ("で", "obl") if self.place.is_none() && is_floor_facility(owner, doc) => {
                self.place = Some(format!("在{}", nominal(words, doc, locks)?));
            }
            _ => return None,
        }
        self.wh |= wh;
        Some(())
    }
}

pub(in crate::ja_zh) fn take(
    atoms: &mut Vec<Atom>,
    doc: &Document,
    locks: &SpanLockTable,
    chars: &[char],
) -> Option<Report> {
    let core = core_non_punct(atoms);
    let report = core
        .iter()
        .position(|a| !a.is_locked && a.deprel == "root" && a.upos == Some(Upos::Verb))?;
    let verb = &core[report];
    let spoken = match verb.lemma.as_deref().unwrap_or(&verb.surface) {
        "答える" => "回答",
        "言う" => "說",
        "伝える" => "轉告",
        "説明する" => "說明",
        "述べる" => "表示",
        _ => return None,
    };
    if atoms
        .iter()
        .any(|a| matches!(a.surface.as_str(), "「" | "『" | "」" | "』"))
    {
        return None;
    }
    let mut tail = &core[report + 1..];
    let particle = tail
        .last()
        .is_some_and(|a| !a.is_locked && a.surface == "か" && a.head == verb.doc_tok_idx);
    let question = particle
        || atoms
            .last()
            .is_some_and(|a| matches!(a.surface.as_str(), "？" | "?"));
    if particle {
        tail = &tail[..tail.len() - 1];
    }
    // Explanatory のです modifies the outer assertion, including in a
    // negative question. It does not negate the reported proposition.
    if let [inflection @ .., no, cop] = tail
        && !no.is_locked
        && no.surface == "の"
        && no.deprel == "mark"
        && !cop.is_locked
        && matches!(cop.surface.as_str(), "だ" | "です")
        && no.head == verb.doc_tok_idx
        && cop.head == verb.doc_tok_idx
    {
        tail = inflection;
    }
    if !tail
        .iter()
        .all(|a| !a.is_locked && a.upos == Some(Upos::Aux) && a.head == verb.doc_tok_idx)
    {
        return None;
    }
    let negative = match tail
        .iter()
        .map(|a| a.surface.as_str())
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] | ["た"] | ["だ"] | ["ます"] | ["まし", "た"] | ["ました"] => "",
        ["なかっ", "た"] | ["なかった"] | ["ませ", "ん", "でし", "た"] | ["ませんでした"] => {
            "沒有"
        }
        ["ない"] | ["ませ", "ん"] | ["ません"] => "不",
        _ => return None,
    };
    let (marker, pred, after_start) =
        core[..report].iter().enumerate().rev().find_map(|(i, m)| {
            if m.is_locked || m.surface != "と" || m.deprel != "case" {
                return None;
            }
            let contrast = core.get(i + 1).is_some_and(|wa| {
                !wa.is_locked
                    && wa.surface == "は"
                    && wa.deprel == "case"
                    && wa.span.start_char == m.span.end_char
                    && (wa.head == m.head || wa.head == m.doc_tok_idx)
            });
            let pred = core[..i].iter().rposition(|a| {
                !a.is_locked
                    && a.doc_tok_idx == m.head
                    && a.upos == Some(Upos::Verb)
                    && (matches!(a.deprel.as_str(), "obl" | "ccomp")
                        || contrast && a.deprel == "nsubj")
            })?;
            nominal_path(core[pred].head, verb.doc_tok_idx, &core[i + 1..report]).then_some((
                i,
                pred,
                i + 1 + usize::from(contrast),
            ))
        })?;
    let to = &core[marker];
    let event = &core[pred];
    if !core[pred + 1..marker]
        .iter()
        .all(|a| !a.is_locked && a.upos == Some(Upos::Aux) && a.head == event.doc_tok_idx)
    {
        return None;
    }
    if !(0..pred).any(|i| {
        let day = &core[i];
        (is_date(day) || super::dates::dated_atom(day) || !day.is_locked && day.surface == "日")
            && day.head == event.doc_tok_idx
            && core
                .get(i + 1)
                .is_some_and(|ni| linked_case(ni, day, day, "に"))
            && (is_date(day)
                || super::dates::dated_atom(day)
                || number_before(core, i).is_some()
                || i.checked_sub(1).is_some_and(|p| {
                    !core[p].is_locked && core[p].surface == "何" && core[p].head == day.doc_tok_idx
                }))
    }) {
        return None;
    }
    let mut frame = Frame::default();
    let mut after = &core[after_start..report];
    while !after.is_empty() {
        if is_punct_atom(&after[0]) {
            after = &after[1..];
            continue;
        }
        if let Some((end, governor)) = after.iter().enumerate().find_map(|(end, a)| {
            if !a.is_locked
                && matches!(a.surface.as_str(), "は" | "が" | "に" | "へ" | "も" | "で")
                && matches!(a.deprel.as_str(), "case" | "discourse")
                && let Some(owner) = after[..end]
                    .iter()
                    .find(|owner| owner.doc_tok_idx == a.head)
                && nominal_path(owner.head, verb.doc_tok_idx, &after[end + 1..])
            {
                Some((end, owner.head))
            } else {
                None
            }
        }) && frame
            .argument(&after[..=end], governor, doc, locks, chars)
            .is_some()
        {
            after = &after[end + 1..];
            continue;
        }
        if after[0].head == verb.doc_tok_idx
            && matches!(after[0].deprel.as_str(), "advmod" | "obl")
            && !after
                .get(1)
                .is_some_and(|p| p.head == after[0].doc_tok_idx && p.deprel == "case")
            && let Some(time) = reporting_time(&after[..1], doc, chars)
        {
            if frame.time.is_some() {
                return None;
            }
            frame.time = Some(time);
            frame.wh |= deadline_time_question(&after[0]).is_some();
            after = &after[1..];
            continue;
        }
        return None;
    }
    let mut start = 0;
    let mut spans = Vec::new();
    // A comma can delimit the entire outer frame. In that frame the parser
    // sometimes attaches dates and recipients to the nearer quoted verb.
    let opening_end = core[..pred]
        .iter()
        .position(|a| matches!(a.surface.as_str(), "、" | "，" | ","))
        .filter(|&i| {
            core[i + 1..pred].iter().any(|a| {
                (is_date(a) || a.surface == "日" || super::dates::dated_atom(a))
                    && a.head == event.doc_tok_idx
            })
        });
    // The reporter can precede the complement. Extract only the actual
    // matrix subject and keep a quoted subject with its own predicate.
    while let Some(case) = core[start..pred]
        .iter()
        .position(|a| {
            !a.is_locked
                && a.deprel == "case"
                && matches!(a.surface.as_str(), "は" | "が" | "に" | "へ" | "も" | "で")
        })
        .map(|n| start + n)
    {
        let Some(owner) = core[start..case]
            .iter()
            .find(|a| a.doc_tok_idx == core[case].head)
        else {
            break;
        };
        // An initial topic can attach to the nearer, quoted predicate. An
        // overt reporter after と leaves this topic inside the complement.
        let before_comma = core
            .get(case + 1)
            .is_some_and(|a| matches!(a.surface.as_str(), "、" | "，" | ","));
        let another_subject = core[case + 1..pred]
            .iter()
            .any(|a| a.deprel == "nsubj" && a.head == event.doc_tok_idx);
        let front_topic = frame.person.is_none()
            && owner.deprel == "nsubj"
            && (core[case].surface == "は" || core[case].surface == "が" && another_subject)
            && (owner.head == event.doc_tok_idx || before_comma);
        // With a speaker in the opening frame, X に followed by the frame's
        // comma can introduce a listener before the entire quotation.
        let front_listener = start > 0
            && frame.person.is_some()
            && frame.recipient.is_none()
            && matches!(core[case].surface.as_str(), "に" | "へ")
            && owner.head == event.doc_tok_idx
            && opening_end.is_some_and(|end| case < end)
            && is_listener(&core[start..case], doc, locks);
        let front_time = (before_comma
            || opening_end.is_some_and(|end| {
                case < end
                    && (frame.person.is_some()
                        || core[case + 1..end].iter().any(|a| a.deprel == "nsubj"))
            }))
            && core[case].surface == "に"
            && owner.head == event.doc_tok_idx
            && reporting_time(&core[start..case], doc, chars).is_some()
            && core[case + 1..pred]
                .iter()
                .any(|a| (is_date(a) || a.surface == "日") && a.head == event.doc_tok_idx);
        if !(owner.head == verb.doc_tok_idx || front_topic || front_listener || front_time)
            || frame
                .argument(&core[start..=case], owner.head, doc, locks, chars)
                .is_none()
        {
            break;
        }
        start = case + 1;
        if core.get(start).is_some_and(is_punct_atom) {
            start += 1;
        }
    }
    if start > 0 {
        spans.push(Span {
            start_char: core[0].span.start_char,
            end_char: core[start - 1].span.end_char,
        });
    }
    // A place, time, recipient or manner owned by the reporting verb may
    // not be silently moved into the subordinate event.
    if core[start..pred]
        .iter()
        .any(|a| a.head == verb.doc_tok_idx && !is_punct_atom(a))
    {
        return None;
    }
    spans.push(Span {
        start_char: to.span.start_char,
        end_char: core.last()?.span.end_char,
    });
    let mut final_punctuation = atoms[core.len()..].to_vec();
    if question
        && let Some(last) = final_punctuation.last_mut()
        && matches!(last.surface.as_str(), "。" | "？" | "?")
    {
        last.surface = "？".to_owned();
    }
    let mut body = core[start..marker].to_vec();
    let content_wh = body.iter().any(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "誰" | "だれ" | "どなた" | "何" | "いつ" | "何時" | "どこ"
            )
            || super::question(a).is_some()
    });
    body[pred - start].deprel = "root".to_owned();
    body[pred - start].head = None;
    body.extend(final_punctuation);
    *atoms = body;
    let join = if !question && negative.is_empty() {
        "，"
    } else if spoken == "回答" {
        "說"
    } else {
        ""
    };
    let person = frame.person.unwrap_or_default();
    let time = frame.time.unwrap_or_default();
    let recipient = frame.recipient.unwrap_or_default();
    let place = frame.place.unwrap_or_default();
    let also = if frame.also { "也" } else { "" };
    Some(Report {
        text: format!(
            "{person}{also}{}{time}{place}{negative}{recipient}{spoken}{join}",
            if question && !frame.wh && !content_wh {
                "是否"
            } else {
                ""
            }
        ),
        spans,
    })
}
