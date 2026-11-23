//! Retain the object case and polite question of a classroom start.

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{Atom, Span, Upos, dasu_case_link, is_punct_atom};

#[derive(Clone, Copy)]
pub(super) struct PredicateLock {
    pub(super) index: usize,
    pub(super) includes_question: bool,
}

pub(super) fn is_start_root(atom: &Atom) -> bool {
    !atom.is_locked
        && atom.upos == Some(Upos::Verb)
        && atom.deprel == "root"
        && atom.lemma.as_deref() == Some("始める")
}

pub(super) fn lesson_head(lesson: &Atom, case: &Atom) -> Option<usize> {
    if !lesson.is_locked
        && lesson.surface == "授業"
        && lesson.upos == Some(Upos::Noun)
        && lesson.deprel == "obj"
        && case.surface == "を"
        && dasu_case_link(lesson, case)
    {
        lesson.head
    } else {
        None
    }
}

fn polite_aux(verb: &Atom, polite: &Atom) -> bool {
    is_start_root(verb)
        && verb.doc_tok_idx.is_some()
        && !polite.is_locked
        && polite.surface == "ます"
        && polite.upos == Some(Upos::Aux)
        && polite.head == verb.doc_tok_idx
        && polite.deprel == "aux"
        && verb.span.end_char == polite.span.start_char
}

pub(super) fn question_tail(verb: &Atom, polite: &Atom, marker: &Atom) -> bool {
    polite_aux(verb, polite)
        && !marker.is_locked
        && marker.surface == "か"
        && marker.upos == Some(Upos::Part)
        && marker.head == verb.doc_tok_idx
        && marker.deprel == "discourse"
        && marker.doc_tok_idx.is_some()
        && polite.span.end_char == marker.span.start_char
}

fn is_kind(lock: &LockedSpan, kind: GlossaryKind) -> bool {
    lock.mandatory && matches!(lock.origin, LockOrigin::Glossary { kind: k, .. } if k == kind)
}

fn covers_case(lock: &LockedSpan, chars: &[char]) -> bool {
    is_kind(lock, GlossaryKind::Object)
        && chars.get(lock.source_span.start_char..lock.source_span.end_char)
            == Some(&['授', '業', 'を'])
}

pub(super) fn may_restore(text: &str, chars: &[char], locks: &SpanLockTable) -> bool {
    text.contains("授業")
        && locks.locks.iter().any(|lock| {
            covers_case(lock, chars)
                || is_kind(lock, GlossaryKind::DomainTerm)
                    && matches!(
                        chars.get(lock.source_span.start_char..lock.source_span.end_char),
                        Some(['始', 'め', 'ま', 'す', 'か'] | ['は', 'じ', 'め', 'ま', 'す', 'か'])
                    )
        })
}

fn finite_lock(
    source: &[Atom],
    root: usize,
    locks: &SpanLockTable,
    case_restored: bool,
) -> Option<PredicateLock> {
    let [verb, polite, tail @ ..] = &source[root..] else {
        return None;
    };
    if !polite_aux(verb, polite) {
        return None;
    }
    // Reuse the full-question lock only in the main clause with a visible
    // terminal marker. A lock inside a quotation or connector is different.
    if source[..root]
        .iter()
        .any(|atom| is_punct_atom(atom) || atom.upos == Some(Upos::Sconj))
    {
        return None;
    }
    let (question, punctuation) = match tail {
        [marker, rest @ ..] if question_tail(verb, polite, marker) => (Some(marker), rest),
        rest => (None, rest),
    };
    match punctuation {
        [] => {}
        [punct] if !punct.is_locked && matches!(punct.surface.as_str(), "。" | "？" | "?") => {}
        _ => return None,
    }
    locks.locks.iter().find_map(|lock| {
        if !is_kind(lock, GlossaryKind::DomainTerm)
            || lock.source_span.start_char != verb.span.start_char
        {
            return None;
        }
        let includes_question =
            question.is_some_and(|marker| lock.source_span.end_char == marker.span.end_char);
        (includes_question || case_restored && lock.source_span.end_char == polite.span.end_char)
            .then_some(PredicateLock {
                index: lock.index,
                includes_question,
            })
    })
}

pub(super) fn restore_locks(
    source: &[Atom],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<(SpanLockTable, Option<PredicateLock>)> {
    let (lesson, case, root) = source.windows(2).find_map(|pair| {
        let [noun, case] = pair else { return None };
        let head = lesson_head(noun, case)?;
        let root = source
            .iter()
            .position(|atom| atom.doc_tok_idx == Some(head) && is_start_root(atom))?;
        Some((noun, case, root))
    })?;
    let locked_case = locks.locks.iter().position(|lock| {
        lock.source_span
            == (Span {
                start_char: lesson.span.start_char,
                end_char: case.span.end_char,
            })
            && covers_case(lock, chars)
    });
    let predicate = finite_lock(source, root, locks, locked_case.is_some());
    if locked_case.is_none() && predicate.is_none() {
        return None;
    }
    let mut render = locks.clone();
    if let Some(index) = locked_case {
        // Only rendering uses the reduced span. The hit still owns 授業を.
        render.locks[index].source_span = lesson.span;
    }
    Some((render, predicate))
}

pub(super) fn owns_question_punctuation(lock: &LockedSpan) -> bool {
    lock.replacement.trim_end().ends_with(['？', '?'])
}

pub(super) fn owns_question_ending(lock: &LockedSpan) -> bool {
    lock.replacement.trim_end().ends_with(['嗎', '吗', '呢']) || owns_question_punctuation(lock)
}
