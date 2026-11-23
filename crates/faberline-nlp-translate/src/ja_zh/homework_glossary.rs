//! Retain the written object case of an action on homework.

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{Atom, Span, Upos, dasu_case_link};

fn covers_case(lock: &LockedSpan, chars: &[char]) -> bool {
    lock.mandatory
        && matches!(
            lock.origin,
            LockOrigin::Glossary {
                kind: GlossaryKind::Object,
                ..
            }
        )
        && chars.get(lock.source_span.start_char..lock.source_span.end_char)
            == Some(&['宿', '題', 'を'])
}

pub(super) fn may_restore_case(chars: &[char], locks: &SpanLockTable) -> bool {
    locks.locks.iter().any(|lock| covers_case(lock, chars))
}

fn object_span(source: &[Atom], span: Span) -> Option<Span> {
    source.windows(2).find_map(|pair| {
        let [noun, case] = pair else { return None };
        let action = noun.head?;
        (noun.span.start_char == span.start_char
            && case.span.end_char == span.end_char
            && noun.surface == "宿題"
            && noun.upos == Some(Upos::Noun)
            && noun.deprel == "obj"
            && case.surface == "を"
            && noun.span.end_char == case.span.start_char
            && dasu_case_link(noun, case)
            && source.iter().any(|verb| {
                verb.doc_tok_idx == Some(action)
                    && verb.upos == Some(Upos::Verb)
                    && verb.lemma.as_deref() == Some("出す")
            }))
        .then_some(noun.span)
    })
}

pub(super) fn restore_case(
    source: &[Atom],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<SpanLockTable> {
    let mut render = None;
    for (index, lock) in locks.locks.iter().enumerate() {
        if covers_case(lock, chars)
            && let Some(span) = object_span(source, lock.source_span)
        {
            // The full caller match remains in the original glossary table.
            render.get_or_insert_with(|| locks.clone()).locks[index].source_span = span;
        }
    }
    render
}
