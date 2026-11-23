//! Keep a bare 初めに attached to the first action.

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Bunsetsu, Document, Segment, Span, Upos, dasu_case_link, group_into_bunsetsus,
    joint_action, split_ja_atoms,
};

fn domain_term(lock: &LockedSpan) -> bool {
    lock.mandatory
        && matches!(
            lock.origin,
            LockOrigin::Glossary {
                kind: GlossaryKind::DomainTerm,
                ..
            }
        )
}

fn initial_noun(span: Span, chars: &[char]) -> bool {
    chars.get(span.start_char..span.end_char) == Some(&['初', 'め'])
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
        initial_noun(first.span, chars)
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

pub(super) fn may_restore(chars: &[char], locks: &SpanLockTable) -> bool {
    locks.locks.iter().any(|lock| {
        domain_term(lock)
            && matches!(
                chars.get(lock.source_span.start_char..lock.source_span.end_char),
                Some(['初', 'め'] | ['初', 'め', 'に'])
            )
    })
}

pub(super) fn restore_case(
    sentence: usize,
    atoms: &[Atom],
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<SpanLockTable> {
    if !may_restore(chars, locks) {
        return None;
    }
    let sent = doc.segmentation.sentences.get(sentence)?.span;
    let clauses = split_ja_atoms(atoms, doc, sent, Some(chars));
    let [Segment::Clause { atoms, .. }] = clauses.as_slice() else {
        return None;
    };
    let groups = group_into_bunsetsus(atoms, doc, sentence);
    let mut render = locks.clone();
    let mut restored = false;
    for (first, ni) in groups
        .iter()
        .filter_map(|group| argument(group, atoms, chars, &locks.locks))
    {
        let phrase = Span {
            start_char: first.span.start_char,
            end_char: ni.span.end_char,
        };
        if let Some(lock) = render.locks.iter_mut().find(|lock| {
            domain_term(lock) && (lock.source_span == first.span || lock.source_span == phrase)
        }) {
            // Expose the particle to the renderer. The original lock still owns the whole match.
            lock.source_span = first.span;
            restored = true;
        }
    }
    restored.then_some(render)
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::{analyze_document, model::Backend};

    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
        doc_options_for_translate, translate_document,
    };

    fn entry(source: &str, target: &str) -> GlossaryEntry {
        GlossaryEntry {
            canonical: source.into(),
            surfaces: vec![source.into()],
            targets: [(TargetVariant::ZhHant, target.into())].into(),
            ..GlossaryEntry::default()
        }
    }

    fn check(source: &str, glossary: &TranslationGlossary, expected: &str) {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let analyzed =
            analyze_and_translate(source, &options, glossary, None, Backend::Onnx).unwrap();
        let doc = analyze_document(source, &doc_options_for_translate(&options, glossary));
        let direct = translate_document(
            source,
            &doc,
            #[cfg(feature = "narrative")]
            None,
            &options,
            glossary,
            None,
        )
        .unwrap();
        assert_eq!(analyzed.target_text, expected);
        assert_eq!(direct.target_text, expected);
        assert_eq!(direct.glossary_hits, analyzed.glossary_hits);
        assert_eq!(direct.clause_alignments, analyzed.clause_alignments);
        let entries = glossary.actors.iter().chain(&glossary.domain_terms);
        assert_eq!(analyzed.glossary_hits.len(), entries.clone().count());
        let slice = |text: &str, span: super::Span| -> String {
            text.chars()
                .skip(span.start_char)
                .take(span.end_char - span.start_char)
                .collect()
        };
        for entry in entries {
            let hit = analyzed
                .glossary_hits
                .iter()
                .find(|hit| hit.canonical == entry.canonical)
                .unwrap();
            assert_eq!(slice(source, hit.source_span), entry.canonical);
            assert_eq!(
                slice(&analyzed.target_text, hit.target_span),
                entry.targets[&TargetVariant::ZhHant]
            );
        }
    }

    #[test]
    fn first_step_keeps_a_caller_term_and_its_linked_question() {
        for surface in ["初め", "初めに"] {
            let mut glossary = TranslationGlossary::new();
            glossary.domain_terms.push(entry(surface, "首先"));
            glossary.domain_terms.push(entry("書き", "填寫"));
            check("初めに名前を書きますか。", &glossary, "首先填寫名字嗎？");
        }
    }

    #[test]
    fn first_step_keeps_the_group_subject_before_the_action_order() {
        let mut glossary = TranslationGlossary::new();
        glossary.domain_terms.push(entry("初めに", "首先"));
        glossary.domain_terms.push(entry("歌い", "合唱歌曲"));
        glossary.actors.push(entry("みんなで", "全組同學"));
        check(
            "みんなで初めに歌いますか。",
            &glossary,
            "全組同學首先一起合唱歌曲嗎？",
        );
    }

    #[test]
    fn first_step_accepts_a_bound_action_and_preserves_a_modified_noun() {
        check(
            "初めに作業します。",
            &TranslationGlossary::new(),
            "先作業。",
        );
        let mut glossary = TranslationGlossary::new();
        glossary.domain_terms.push(entry("初め", "開場"));
        check(
            "会議の初めに名前を書きます。",
            &glossary,
            "給會議的開場寫名字。",
        );
    }
}
