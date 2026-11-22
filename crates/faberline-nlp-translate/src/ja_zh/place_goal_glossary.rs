//! Recover a household destination particle covered by a caller's place lock.

use faberline_nlp::Span;
use faberline_nlp::doc::{Document, Upos, analyze_document};

use super::{Atom, LOCKED_PLACE_ATOM, household_entry};
use crate::GlossaryKind;
use crate::lock::{LockOrigin, SpanLockTable};

pub(super) fn restore_destination(
    atoms: &mut Vec<Atom>,
    chars: &[char],
    sentence: Span,
    doc: &Document,
    locks: &SpanLockTable,
) {
    let mut places = locks.locks.iter().filter(|lock| {
        lock.mandatory
            && matches!(
                lock.origin,
                LockOrigin::Glossary {
                    kind: GlossaryKind::Place,
                    ..
                }
            )
            && lock.source_span.start_char >= sentence.start_char
            && lock.source_span.end_char <= sentence.end_char
            && chars.get(lock.source_span.start_char..lock.source_span.end_char)
                == Some(&['家', 'に'])
    });
    let Some(lock) = places.next() else {
        return;
    };
    if places.next().is_some() {
        return;
    }
    let Some(index) = atoms.iter().position(|atom| {
        atom.is_locked && atom.span == lock.source_span && atom.surface == lock.placeholder
    }) else {
        return;
    };
    let Some((case, entry)) = proven_destination(atoms, chars, sentence, doc, lock.source_span)
    else {
        return;
    };

    // The lock table retains the full hit and supplies the single replacement.
    let mut destination = atoms[index].clone();
    destination.span.end_char = case.start_char;
    destination.lemma = Some(LOCKED_PLACE_ATOM.to_owned());
    destination.doc_tok_idx = None;
    destination.head = Some(entry);
    destination.deprel = "obl".to_owned();
    let marker = Atom {
        span: case,
        surface: "に".to_owned(),
        lemma: None,
        upos: Some(Upos::Adp),
        doc_tok_idx: None,
        head: None,
        deprel: "case".to_owned(),
        is_locked: false,
    };
    atoms.splice(index..=index, [destination, marker]);
}

fn proven_destination(
    atoms: &[Atom],
    chars: &[char],
    sentence: Span,
    doc: &Document,
    span: Span,
) -> Option<(Span, usize)> {
    if !doc.options.enrich_morphology || !doc.options.cpu_syntax {
        return None;
    }
    let text = chars.get(sentence.start_char..sentence.end_char)?;
    let mut options = doc.options.clone();
    options.names.retain(|hint| hint != "家に");
    options.lexicon.retain(|hint| hint != "家に");
    let parsed = analyze_document(&text.iter().collect::<String>(), &options);
    if parsed.segmentation.sentences.len() != 1 {
        return None;
    }
    let case = Span {
        start_char: span.end_char - 1,
        end_char: span.end_char,
    };
    let absolute = |span: Span| Span {
        start_char: sentence.start_char + span.start_char,
        end_char: sentence.start_char + span.end_char,
    };
    let particle = parsed.syntax.iter().find(|link| {
        absolute(link.span) == case
            && link.deprel == "case"
            && parsed
                .tokens
                .get(link.token)
                .is_some_and(|token| token.upos == Some(Upos::Adp))
    })?;
    let noun = parsed.syntax.get(particle.head?)?;
    if noun.deprel != "obl"
        || absolute(noun.span).start_char != span.start_char
        || absolute(noun.span).end_char != case.start_char
        || parsed.tokens.get(noun.token)?.upos != Some(Upos::Noun)
    {
        return None;
    }
    let entry = parsed.syntax.get(noun.head?)?;
    let entry_span = absolute(entry.span);
    if entry_span.start_char < span.end_char
        || !household_entry::allows_physical_entry(&parsed, text, entry.token)
    {
        return None;
    }
    let entry_atom = atoms.iter().find(|atom| {
        !atom.is_locked
            && atom.span == entry_span
            && atom.upos == Some(Upos::Verb)
            && atom.lemma.as_deref() == Some("入る")
    })?;
    let competing_destination = parsed.syntax.iter().any(|link| {
        link.token != particle.token
            && link.deprel == "case"
            && link
                .head
                .and_then(|head| parsed.syntax.get(head))
                .is_some_and(|other| other.head == Some(entry.token) && other.deprel == "obl")
            && matches!(
                text.get(link.span.start_char..link.span.end_char),
                Some(['に'] | ['へ'] | ['ま', 'で'])
            )
    });
    if competing_destination {
        return None;
    }
    Some((case, entry_atom.doc_tok_idx?))
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::{analyze_document, model::Backend};

    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
        doc_options_for_translate, translate_document,
    };

    fn entry(surface: &str, target: &str) -> GlossaryEntry {
        GlossaryEntry {
            canonical: surface.into(),
            surfaces: vec![surface.into()],
            targets: [(TargetVariant::ZhHant, target.into())].into(),
            ..GlossaryEntry::default()
        }
    }

    #[test]
    fn caller_place_keeps_the_syntactic_destination_and_full_source_span() {
        let source = "靴を脱がないで、家に入りました。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let mut glossary = TranslationGlossary::new();
        let doc = analyze_document(source, &doc_options_for_translate(&options, &glossary));
        for term in ["家", "家に"] {
            glossary.places = vec![entry(term, "舊屋")];
            for objects in [vec![], vec![entry("靴", "鞋履")]] {
                glossary.objects = objects;
                let analyzed =
                    analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                        .unwrap();
                let direct = translate_document(
                    source,
                    &doc,
                    #[cfg(feature = "narrative")]
                    None,
                    &options,
                    &glossary,
                    None,
                )
                .unwrap();
                for (entrypoint, result) in [
                    ("analyze_and_translate", analyzed),
                    ("translate_document", direct),
                ] {
                    assert!(
                        result.target_text.ends_with("進入舊屋了。"),
                        "{entrypoint} {term} ({} objects): {}",
                        glossary.objects.len(),
                        result.target_text
                    );
                    assert_eq!(result.target_text.matches("舊屋").count(), 1);
                    let hit = result
                        .glossary_hits
                        .iter()
                        .find(|hit| hit.canonical == term)
                        .unwrap();
                    let at = |text: &str, span: faberline_nlp::Span| -> String {
                        text.chars()
                            .skip(span.start_char)
                            .take(span.end_char - span.start_char)
                            .collect()
                    };
                    assert_eq!(at(source, hit.source_span), term);
                    assert_eq!(at(&result.target_text, hit.target_span), "舊屋");
                    if !glossary.objects.is_empty() {
                        assert!(result.target_text.contains("鞋履"));
                        assert_eq!(result.glossary_hits.len(), 2);
                    }
                }
            }
        }
    }
}
