//! A simple deadline bounds completion of the letter-writing action.

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, LOCKED_OBJECT_ATOM, Upos, core_non_punct,
    deadline_governor, group_into_bunsetsus, linked_letter_case, normalize_clock_atoms,
};

fn is_letter_lock(lock: &LockedSpan, chars: &[char]) -> bool {
    lock.mandatory
        && matches!(
            lock.origin,
            LockOrigin::Glossary {
                kind: GlossaryKind::Object,
                ..
            }
        )
        && matches!(
            chars.get(lock.source_span.start_char..lock.source_span.end_char),
            Some(['手', '紙'] | ['手', '紙', 'を'])
        )
}

pub(super) fn may_cover_object(chars: &[char], locks: &SpanLockTable) -> bool {
    locks.locks.iter().any(|lock| is_letter_lock(lock, chars))
}

pub(super) fn restore_case(
    sentence: usize,
    doc: &Document,
    atoms: &[Atom],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<SpanLockTable> {
    if !may_cover_object(chars, locks) {
        return None;
    }
    let normalized = normalize_clock_atoms(atoms);
    let groups = group_into_bunsetsus(&normalized, doc, sentence);
    let predicates = groups
        .iter()
        .filter(|group| group.role == BunsetsuRole::Predicate);
    let letter = completion_letter(&groups, predicates, &normalized, doc, chars)?;
    let mut render = locks.clone();
    let lock = render.locks.iter_mut().find(|lock| {
        is_letter_lock(lock, chars)
            && lock.source_span.start_char == letter.span.start_char
            && matches!(
                lock.source_span.end_char.checked_sub(letter.span.end_char),
                Some(0 | 1)
            )
    })?;
    lock.source_span = letter.span;
    Some(render)
}

pub(super) fn completion_object<'a, 'p>(
    groups: &[Bunsetsu],
    predicates: impl Iterator<Item = &'p Bunsetsu>,
    source: &'a [Atom],
    doc: &Document,
    chars: &[char],
) -> Option<&'a str> {
    completion_letter(groups, predicates, source, doc, chars).map(|letter| {
        if letter.is_locked {
            letter.surface.as_str()
        } else {
            "信"
        }
    })
}

fn completion_letter<'a, 'p>(
    groups: &[Bunsetsu],
    mut predicates: impl Iterator<Item = &'p Bunsetsu>,
    source: &'a [Atom],
    doc: &Document,
    chars: &[char],
) -> Option<&'a Atom> {
    let predicate = predicates.next()?;
    if predicates.next().is_some() || predicate.sconj_tail.is_some() || predicate.sfp_tail.is_some()
    {
        return None;
    }
    let verb = match core_non_punct(&predicate.atoms) {
        [verb] if verb.surface == "書く" => verb,
        [verb, polite]
            if !polite.is_locked
                && polite.surface == "ます"
                && polite.deprel == "aux"
                && polite.head == verb.doc_tok_idx =>
        {
            verb
        }
        _ => return None,
    };
    let action = verb.doc_tok_idx?;
    if verb.is_locked
        || verb.upos != Some(Upos::Verb)
        || verb.lemma.as_deref() != Some("書く")
        || verb.deprel != "root"
        || !groups
            .iter()
            .any(|group| group.atoms.len() == 3 && deadline_governor(group, doc) == Some(action))
        || groups.iter().any(|group| group.role == BunsetsuRole::IObj)
        || source
            .iter()
            .any(|atom| matches!(atom.surface.as_str(), "？" | "?"))
    {
        return None;
    }
    source.windows(2).find_map(|pair| {
        let [letter, case] = pair else { return None };
        let is_letter = if letter.is_locked {
            letter.lemma.as_deref() == Some(LOCKED_OBJECT_ATOM)
                && chars.get(letter.span.start_char..letter.span.end_char) == Some(&['手', '紙'])
        } else {
            letter.surface == "手紙"
        };
        (is_letter
            && letter.deprel == "obj"
            && letter.head == Some(action)
            && case.surface == "を"
            && linked_letter_case(letter, case))
        .then_some(letter)
    })
}
