//! Keep an unquoted と complement with its reporting verb. The subordinate
//! event gets a local root so its day, subject and quantity render together.
use super::{is_date, linked_case, number_before};
use crate::ja_zh::{
    Atom, Document, Span, SpanLockTable, Upos, core_non_punct, is_punct_atom,
    render_direct_quote_person,
};

pub(in crate::ja_zh) struct Report {
    pub(in crate::ja_zh) text: String,
    pub(in crate::ja_zh) spans: Vec<Span>,
}

fn speaker(
    args: &[Atom],
    head: Option<usize>,
    doc: &Document,
    locks: &SpanLockTable,
) -> Option<String> {
    let (case, noun) = args.split_last()?;
    if case.is_locked
        || !matches!(case.surface.as_str(), "が" | "は")
        || !noun
            .iter()
            .any(|a| a.doc_tok_idx == case.head && a.head == head && a.deprel == "nsubj")
    {
        return None;
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
    render_direct_quote_person(noun, doc, locks)
}

pub(in crate::ja_zh) fn take(
    atoms: &mut Vec<Atom>,
    doc: &Document,
    locks: &SpanLockTable,
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
    let marker = core[..report]
        .iter()
        .rposition(|m| !m.is_locked && m.surface == "と" && m.deprel == "case")?;
    let to = &core[marker];
    let pred = core[..marker].iter().rposition(|a| {
        !a.is_locked
            && a.doc_tok_idx == to.head
            && a.head == verb.doc_tok_idx
            && matches!(a.deprel.as_str(), "obl" | "ccomp")
            && a.upos == Some(Upos::Verb)
    })?;
    let event = &core[pred];
    if !core[pred + 1..marker]
        .iter()
        .all(|a| !a.is_locked && a.upos == Some(Upos::Aux) && a.head == event.doc_tok_idx)
    {
        return None;
    }
    if !(0..pred).any(|i| {
        let day = &core[i];
        (is_date(day) || !day.is_locked && day.surface == "日")
            && day.head == event.doc_tok_idx
            && core
                .get(i + 1)
                .is_some_and(|ni| linked_case(ni, day, day, "に"))
            && (is_date(day)
                || number_before(core, i).is_some()
                || i.checked_sub(1).is_some_and(|p| {
                    !core[p].is_locked && core[p].surface == "何" && core[p].head == day.doc_tok_idx
                }))
    }) {
        return None;
    }
    let after = &core[marker + 1..report];
    let mut person = if after.is_empty() {
        String::new()
    } else {
        speaker(after, verb.doc_tok_idx, doc, locks)?
    };
    let mut start = 0;
    let mut spans = Vec::new();
    // The reporter can precede the complement. Extract only the actual
    // matrix subject and keep a quoted subject with its own predicate.
    if after.is_empty()
        && let Some(case) = core[..pred].iter().position(|a| !a.is_locked && a.deprel=="case")
        && matches!(core[case].surface.as_str(),"が"|"は")
        && let Some(owner) = core[..case].iter().find(|a| a.doc_tok_idx==core[case].head && a.deprel=="nsubj")
        // A sentence-initial X は can be the reporter even when the parser
        // attaches the topic to the nearer quoted verb.
        && (owner.head==verb.doc_tok_idx || core[case].surface=="は")
        && let Some(who)=speaker(&core[..=case],owner.head,doc,locks)
    {
        person = who;
        start = case + 1;
        if core
            .get(start)
            .is_some_and(|p| matches!(p.surface.as_str(), "、" | "，" | ","))
        {
            start += 1;
        }
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
    Some(Report {
        text: format!(
            "{person}{}{negative}{spoken}{join}",
            if question { "是否" } else { "" }
        ),
        spans,
    })
}
