//! Keep caller verb terms separate from the syntax that establishes an omission.

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Document, Segment, Span, Token, Upos, build_sentence_atoms, carrying, naide_adverbial,
    naide_ending, split_ja_atoms,
};

pub(crate) struct ProtectedOmission {
    stems: Vec<LockedSpan>,
    whole_result: Option<LockedSpan>,
}

impl ProtectedOmission {
    pub(super) fn stem(&self, atoms: &[Atom]) -> Option<&str> {
        self.stems
            .iter()
            .find(|lock| {
                atoms.iter().any(|atom| {
                    !atom.is_locked
                        && atom.upos == Some(Upos::Verb)
                        && atom.span == lock.source_span
                })
            })
            .map(|lock| lock.placeholder.as_str())
    }

    pub(super) fn result(&self, span: Span) -> Option<&str> {
        self.whole_result
            .as_ref()
            .filter(|lock| lock.source_span == span)
            .map(|lock| lock.placeholder.as_str())
    }
}

fn kind(lock: &LockedSpan) -> Option<GlossaryKind> {
    match lock.origin {
        LockOrigin::Glossary { kind, .. } if lock.mandatory => Some(kind),
        _ => None,
    }
}

pub(crate) fn may_restore(text: &str, locks: &SpanLockTable) -> bool {
    text.contains("ないで")
        && !text.contains(['「', '」', '『', '』', '"', '“', '”', '?', '？'])
        && locks
            .locks
            .iter()
            .any(|lock| kind(lock) == Some(GlossaryKind::DomainTerm))
}

pub(crate) fn restore(
    sentence: usize,
    doc: &Document,
    tokens: &[&Token],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<(SpanLockTable, ProtectedOmission)> {
    let sentence_span = doc.segmentation.sentences.get(sentence)?.span;
    let atoms = build_sentence_atoms(sentence, chars, doc, tokens, &SpanLockTable::new());
    let segments = split_ja_atoms(&atoms, doc, sentence_span, Some(chars));
    let [
        Segment::Clause {
            atoms: before,
            trailing_punct: Some(comma),
            ..
        },
        Segment::Clause {
            atoms: after,
            trailing_punct,
            span: result_span,
        },
    ] = segments.as_slice()
    else {
        return None;
    };
    if !matches!(comma.as_str(), "、" | "，" | ",") {
        return None;
    }
    let omission = naide_adverbial(before, after, trailing_punct.as_deref(), chars)?;
    let (hold, _, _) = naide_ending(before)?;
    let [motion, ending @ ..] = after.as_slice() else {
        return None;
    };
    let motion_id = motion.doc_tok_idx?;
    if !omission.past
        || !omission.join_unstated_subject
        || hold.head != Some(motion_id)
        || hold.lemma.as_deref() != Some("持つ")
        || before
            .iter()
            .filter(|atom| atom.upos == Some(Upos::Verb))
            .count()
            != 1
        || !carrying::is_motion(motion.lemma.as_deref())
        || ending.is_empty()
        || ending.iter().any(|atom| atom.upos != Some(Upos::Aux))
    {
        return None;
    }
    let (_, noun, particle) = carrying::portable_object(doc, chars, omission.action, motion_id)?;
    let object_with_case = Span {
        start_char: noun.start_char,
        end_char: particle.end_char,
    };

    let mut render = SpanLockTable::new();
    let mut stems = Vec::new();
    let mut whole_result = None;
    for lock in &locks.locks {
        let span = lock.source_span;
        match kind(lock) {
            Some(GlossaryKind::DomainTerm) if span == hold.span || span == motion.span => {
                stems.push(lock.clone());
            }
            Some(GlossaryKind::DomainTerm) if span == *result_span => {
                whole_result = Some(lock.clone());
            }
            Some(GlossaryKind::Object | GlossaryKind::DomainTerm)
                if span == noun || span == object_with_case =>
            {
                render.locks.push(lock.clone());
            }
            _ => return None,
        }
    }
    // Keep the complete source analysis for grammar. Insert the protected verb
    // placeholders only when rendering the corresponding action.
    if stems.is_empty() && whole_result.is_none() {
        return None;
    }
    Some((
        render,
        ProtectedOmission {
            stems,
            whole_result,
        },
    ))
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::{analyze_document, model::Backend};

    use crate::{
        GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
        analyze_and_translate, doc_options_for_translate, translate_document,
    };

    fn term(surface: &str, target: &str) -> GlossaryEntry {
        GlossaryEntry {
            canonical: surface.to_owned(),
            surfaces: vec![surface.to_owned()],
            targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
            ..GlossaryEntry::default()
        }
    }

    #[test]
    fn carrying_omission_keeps_protected_predicates_in_their_own_clauses() {
        let source = "傘を持たないで、出かけました。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let doc = analyze_document(
            source,
            &doc_options_for_translate(&options, &TranslationGlossary::new()),
        );
        for object in [
            None,
            Some((GlossaryKind::Object, "傘")),
            Some((GlossaryKind::Object, "傘を")),
            Some((GlossaryKind::DomainTerm, "傘")),
            Some((GlossaryKind::DomainTerm, "傘を")),
        ] {
            for omitted in [false, true] {
                for result in [
                    None,
                    Some(("出かけ", "外出")),
                    Some(("出かけました", "已出門")),
                ] {
                    let mut glossary = TranslationGlossary::new();
                    let mut terms = Vec::new();
                    if let Some((kind, surface)) = object {
                        let entry = term(surface, "雨具");
                        match kind {
                            GlossaryKind::Object => glossary.objects.push(entry),
                            _ => glossary.domain_terms.push(entry),
                        }
                        terms.push((kind, surface, "雨具"));
                    }
                    if omitted {
                        glossary.domain_terms.push(term("持た", "攜帶"));
                        terms.push((GlossaryKind::DomainTerm, "持た", "攜帶"));
                    }
                    if let Some((surface, alias)) = result {
                        glossary.domain_terms.push(term(surface, alias));
                        terms.push((GlossaryKind::DomainTerm, surface, alias));
                    }
                    let live =
                        analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                            .expect("translate with built-in CPU analysis");
                    let direct = translate_document(
                        source,
                        &doc,
                        #[cfg(feature = "narrative")]
                        None,
                        &options,
                        &glossary,
                        None,
                    )
                    .expect("translate supplied native analysis");
                    let noun = if object.is_some() { "雨具" } else { "傘" };
                    let action = if omitted { "攜帶" } else { "帶" };
                    let exit = match result {
                        None => "出門了",
                        Some(("出かけ", _)) => "外出了",
                        _ => "已出門",
                    };
                    assert_eq!(
                        live.target_text,
                        format!("沒有{action}{noun}就{exit}。"),
                        "{terms:?}"
                    );
                    assert_eq!(live.target_text, direct.target_text, "{terms:?}");
                    assert_eq!(live.glossary_hits, direct.glossary_hits, "{terms:?}");
                    assert_eq!(live.clause_alignments.len(), 2);
                    assert_eq!(live.glossary_hits.len(), terms.len());
                    assert!(live.unknown.is_empty());
                    let cut = |text: &str, span: faberline_nlp::Span| {
                        text.chars()
                            .skip(span.start_char)
                            .take(span.end_char - span.start_char)
                            .collect::<String>()
                    };
                    for (hit, (kind, surface, alias)) in live.glossary_hits.iter().zip(terms) {
                        assert_eq!(hit.kind, kind);
                        assert_eq!(cut(source, hit.source_span), surface);
                        assert_eq!(cut(&live.target_text, hit.target_span), alias);
                    }
                }
            }
        }
    }

    #[test]
    fn protected_departure_keeps_affirmative_and_boundary_controls_separate() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let mut glossary = TranslationGlossary::new();
        glossary.domain_terms.push(term("出かけました", "出門了"));
        let translate = |source| {
            analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                .expect("translate on CPU")
        };
        let connected = translate("傘を持たないで出かけました。");
        assert_eq!(connected.target_text, "沒有帶傘就出門了。");
        assert_eq!(connected.glossary_hits.len(), 1);
        for source in [
            "傘を持たないで、出かけましたか。",
            "私は傘を持たないで、弟は出かけました。",
            "「傘を持たないで、出かけました」と言いました。",
        ] {
            let translated = translate(source);
            assert!(!translated.target_text.contains('就'), "{source}");
            assert_eq!(translated.target_text.matches("出門了").count(), 1);
            assert_eq!(translated.glossary_hits.len(), 1);
        }
    }
}
