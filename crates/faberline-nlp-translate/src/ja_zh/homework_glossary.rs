//! Retain the written object case of an action on homework.

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{Atom, Span, Upos, dasu_case_link};

fn covers_homework(lock: &LockedSpan, chars: &[char]) -> bool {
    lock.mandatory
        && matches!(
            (
                &lock.origin,
                chars.get(lock.source_span.start_char..lock.source_span.end_char),
            ),
            (
                LockOrigin::Glossary {
                    kind: GlossaryKind::Object | GlossaryKind::DomainTerm,
                    ..
                },
                Some(['宿', '題', 'を']),
            ) | (
                LockOrigin::Glossary {
                    kind: GlossaryKind::DomainTerm,
                    ..
                },
                Some(['宿', '題']),
            )
        )
}

pub(super) fn may_restore_case(chars: &[char], locks: &SpanLockTable) -> bool {
    locks.locks.iter().any(|lock| covers_homework(lock, chars))
}

fn object_span(source: &[Atom], span: Span) -> Option<Span> {
    source.windows(2).find_map(|pair| {
        let [noun, case] = pair else { return None };
        let action = noun.head?;
        (noun.span.start_char == span.start_char
            && (noun.span.end_char == span.end_char || case.span.end_char == span.end_char)
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
        if covers_homework(lock, chars)
            && let Some(span) = object_span(source, lock.source_span)
        {
            // Render the proved object role. Report the original glossary match.
            let rendered = &mut render.get_or_insert_with(|| locks.clone()).locks[index];
            rendered.source_span = span;
            if let LockOrigin::Glossary { kind, .. } = &mut rendered.origin {
                *kind = GlossaryKind::Object;
            }
        }
    }
    render
}
