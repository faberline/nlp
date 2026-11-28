//! Keep the sequence adjunct attached to the action it modifies.

use crate::lock::{LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Bunsetsu, Document, Segment, Span, Token, Upos, build_sentence_atoms, dasu_case_link,
    group_into_bunsetsus, joint_action, polite_joint, predicate_glossary, split_ja_atoms,
};
use predicate_glossary::{Coverage, PredicateTerms};

fn domain_term(lock: &LockedSpan) -> bool {
    predicate_glossary::kind(lock) == Some(GlossaryKind::DomainTerm)
}

fn governs_first_step(source: &[Atom], governor: usize) -> bool {
    let Some((index, action)) = source.iter().enumerate().find(|(_, atom)| {
        !atom.is_locked && atom.doc_tok_idx == Some(governor) && atom.upos == Some(Upos::Verb)
    }) else {
        return false;
    };
    if action.deprel == "root" {
        return true;
    }
    // A bound verbal noun can own the adjunct while する carries inflection.
    let Some(light_verb) = source.get(index + 1) else {
        return false;
    };
    light_verb.doc_tok_idx.is_some_and(|head| {
        !light_verb.is_locked
            && light_verb.upos == Some(Upos::Verb)
            && light_verb.deprel == "root"
            && joint_action(&source[..=index], light_verb, head).doc_tok_idx == Some(governor)
    })
}

fn argument<'a>(
    group: &'a Bunsetsu,
    source: &[Atom],
    chars: &[char],
    locks: &[LockedSpan],
) -> Option<(&'a Atom, &'a Atom)> {
    let [first, ni] = group.atoms.as_slice() else {
        return None;
    };
    let governor = first.head?;
    let initial = if first.is_locked {
        chars.get(first.span.start_char..first.span.end_char) == Some(&['初', 'め'])
            && locks.iter().any(|lock| {
                domain_term(lock)
                    && lock.source_span == first.span
                    && lock.placeholder == first.surface
            })
    } else {
        first.surface == "初め" && first.upos == Some(Upos::Noun)
    };
    (initial
        && ni.surface == "に"
        && matches!(first.deprel.as_str(), "obl" | "advmod")
        && dasu_case_link(first, ni)
        && source
            .iter()
            .all(|atom| atom.head != first.doc_tok_idx || atom.doc_tok_idx == ni.doc_tok_idx)
        && governs_first_step(source, governor))
    .then_some((first, ni))
}

pub(super) fn render(
    group: &Bunsetsu,
    source: &[Atom],
    chars: &[char],
    locks: &[LockedSpan],
) -> Option<String> {
    let (first, _) = argument(group, source, chars, locks)?;
    Some(if first.is_locked {
        first.surface.clone()
    } else {
        "先".to_owned()
    })
}

pub(crate) fn may_restore(text: &str, locks: &SpanLockTable) -> bool {
    text.contains("初めに") && locks.locks.iter().any(domain_term)
}

pub(crate) fn restore(
    sentence: usize,
    doc: &Document,
    tokens: &[&Token],
    chars: &[char],
    render_locks: &mut SpanLockTable,
    terms: &mut Option<PredicateTerms>,
) -> Option<()> {
    let span = doc.segmentation.sentences.get(sentence)?.span;
    if doc
        .segmentation
        .opaque
        .iter()
        .any(|part| part.span.start_char < span.end_char && span.start_char < part.span.end_char)
    {
        return None;
    }
    let plain = build_sentence_atoms(sentence, chars, doc, tokens, &SpanLockTable::default());
    let clauses = split_ja_atoms(&plain, doc, span, Some(chars));
    let [Segment::Clause { atoms, .. }] = clauses.as_slice() else {
        return None;
    };
    let groups = group_into_bunsetsus(atoms, doc, sentence);
    let mut first_steps = groups
        .iter()
        .filter_map(|group| argument(group, atoms, chars, &[]));
    let (first, ni) = first_steps.next()?;
    if first_steps.next().is_some() {
        return None;
    }
    let phrase = Span {
        start_char: first.span.start_char,
        end_char: ni.span.end_char,
    };
    let first_lock = render_locks.locks.iter_mut().find(|lock| {
        domain_term(lock) && (lock.source_span == first.span || lock.source_span == phrase)
    });
    let adjusted_first = first_lock.is_some();
    if let Some(lock) = first_lock {
        // The rendering view exposes the particle. The caller's lock still owns
        // the complete source span and is used when the placeholder is restored.
        lock.source_span = first.span;
    }

    let mut adjusted_verb = false;
    if terms.is_none()
        && let Some(index) = atoms.iter().position(|atom| {
            atom.doc_tok_idx.is_some() && atom.doc_tok_idx == first.head && atom.deprel == "root"
        })
        && let Some(verb) = atoms.get(index)
        && polite_joint::is_polite_ending(verb, &atoms[index + 1..])
        && let Some(lock_index) = render_locks
            .locks
            .iter()
            .position(|lock| domain_term(lock) && lock.source_span == verb.span)
    {
        let lock = render_locks.locks.remove(lock_index);
        let mut predicate = PredicateTerms::default();
        predicate.add(verb.span, &lock, Coverage::Stem);
        *terms = Some(predicate);
        adjusted_verb = true;
    }
    (adjusted_first || adjusted_verb).then_some(())
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::analyze_document;

    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, doc_options_for_translate};

    use super::*;

    #[test]
    fn a_bound_verbal_noun_can_supply_the_action() {
        let source = "みんなで作業します。";
        let chars: Vec<_> = source.chars().collect();
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let doc = analyze_document(
            source,
            &doc_options_for_translate(&options, &TranslationGlossary::new()),
        );
        let tokens: Vec<_> = doc.tokens.iter().collect();
        let mut atoms = build_sentence_atoms(0, &chars, &doc, &tokens, &SpanLockTable::default());
        let index = atoms
            .iter()
            .position(|atom| atom.surface == "作業")
            .unwrap();
        let action = atoms[index].doc_tok_idx.unwrap();
        assert_eq!(atoms[index].deprel, "advcl");
        assert!(governs_first_step(&atoms, action));
        atoms[index].head = None;
        assert!(!governs_first_step(&atoms, action));
    }
}
