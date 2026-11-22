//! Recover argument cases covered by glossary locks in a polite prohibition.

use faberline_nlp::doc::analyze_document;

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Bunsetsu, BunsetsuRole, ClauseDiscourseCtx, Document, LOCKED_OBJECT_ATOM,
    LOCKED_PLACE_ATOM, SemanticClass, Span, Upos, build_sentence_atoms, core_non_punct,
    dasu_case_link, is_place_location_surface, is_polite_prohibition_ending,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum CoveredCase {
    Object,
    Venue,
}

pub(super) fn restore_argument_cases(
    groups: &mut [Bunsetsu],
    source: &[Atom],
    doc: &Document,
    ctx: &ClauseDiscourseCtx<'_>,
) {
    if ctx.locks.is_empty() || !doc.options.cpu_syntax {
        return;
    }
    let core = core_non_punct(source);
    let Some((first, last)) = core.first().zip(core.last()) else {
        return;
    };
    let start = first.span.start_char;
    let Some(clause) = ctx.source_chars.get(start..last.span.end_char) else {
        return;
    };
    if !clause.ends_with(&['な', 'い', 'で', 'く', 'だ', 'さ', 'い']) {
        return;
    }
    let candidates: Vec<_> = groups
        .iter_mut()
        .filter_map(|group| covered_case(group, ctx).map(|(case, lock)| (group, case, lock)))
        .collect();
    if candidates.is_empty() {
        return;
    }

    // Nearby glossary hints can mask both the case and the original governor.
    let mut options = doc.options.clone();
    options.lexicon.clear();
    options.names.clear();
    let parsed = analyze_document(&clause.iter().collect::<String>(), &options);
    if parsed.segmentation.sentences.len() != 1 {
        return;
    }
    let tokens: Vec<_> = parsed.tokens.iter().collect();
    let plain = build_sentence_atoms(0, clause, &parsed, &tokens, &SpanLockTable::new());
    let Some(action) = prohibition_action(&plain) else {
        return;
    };
    let absolute = |span: Span| Span {
        start_char: span.start_char + start,
        end_char: span.end_char + start,
    };
    let action_span = absolute(action.span);
    let direct_action = core.iter().any(|atom| {
        !atom.is_locked
            && atom.span == action_span
            && atom.upos == Some(Upos::Verb)
            && atom.lemma == action.lemma
    });
    let whole_predicate = last.is_locked
        && last.span.start_char == action_span.start_char
        && ctx.locks.iter().any(|lock| {
            lock.mandatory
                && lock.source_span == last.span
                && lock.placeholder == last.surface
                && matches!(
                    lock.origin,
                    LockOrigin::Glossary {
                        kind: GlossaryKind::DomainTerm,
                        ..
                    }
                )
        });
    if !direct_action && !whole_predicate {
        return;
    }

    let venues = candidates
        .iter()
        .filter(|(_, case, _)| *case == CoveredCase::Venue)
        .count();
    for (group, case, lock) in candidates {
        if case == CoveredCase::Venue
            && (venues != 1 || lock.replacement.trim_start().starts_with(['在', '於']))
        {
            continue;
        }
        let span = lock.source_span;
        if span.start_char >= start
            && span.end_char <= last.span.end_char
            && native_argument(
                case,
                Span {
                    start_char: span.start_char - start,
                    end_char: span.end_char - start,
                },
                action,
                &plain,
            )
        {
            // The original lock still owns the complete match and verbatim replacement.
            match case {
                CoveredCase::Object => group.role = BunsetsuRole::Obj,
                CoveredCase::Venue => {
                    group.role = BunsetsuRole::Obl;
                    group.case_particle = Some("で".to_owned());
                }
            }
        }
    }
}

fn prohibition_action(atoms: &[Atom]) -> Option<&Atom> {
    let core = core_non_punct(atoms);
    let [prefix @ .., _, _, _] = core else {
        return None;
    };
    let action = prefix.last()?;
    (action.upos == Some(Upos::Verb) && is_polite_prohibition_ending(action, &core[prefix.len()..]))
        .then_some(action)
}

fn covered_case<'a>(
    group: &Bunsetsu,
    ctx: &ClauseDiscourseCtx<'a>,
) -> Option<(CoveredCase, &'a LockedSpan)> {
    let [atom] = group.atoms.as_slice() else {
        return None;
    };
    if !atom.is_locked || group.case_particle.is_some() {
        return None;
    }
    let surface = ctx
        .source_chars
        .get(atom.span.start_char..atom.span.end_char)?;
    let case = match (atom.lemma.as_deref(), surface) {
        (Some(LOCKED_OBJECT_ATOM), [_, .., 'を']) => CoveredCase::Object,
        (Some(LOCKED_PLACE_ATOM), [noun @ .., 'で']) => {
            let noun: String = noun.iter().collect();
            if !is_place_location_surface(&noun)
                && !faberline_nlp::lexicon::lookup_ja(&noun)
                    .is_some_and(|entry| entry.sem_class == Some(SemanticClass::Location))
            {
                return None;
            }
            CoveredCase::Venue
        }
        _ => return None,
    };
    let lock = ctx
        .locks
        .iter()
        .find(|lock| lock.source_span == atom.span && lock.placeholder == atom.surface)?;
    Some((case, lock))
}

fn native_argument(case: CoveredCase, span: Span, action: &Atom, atoms: &[Atom]) -> bool {
    let start = atoms.partition_point(|atom| atom.span.start_char < span.start_char);
    let end = atoms.partition_point(|atom| atom.span.start_char < span.end_char);
    let [modifiers @ .., noun, marker] = &atoms[start..end] else {
        return false;
    };
    let (particle, relation) = match case {
        CoveredCase::Object => ("を", "obj"),
        CoveredCase::Venue => ("で", "obl"),
    };
    atoms[start].span.start_char == span.start_char
        && marker.span.end_char == span.end_char
        && marker.surface == particle
        && noun.deprel == relation
        && noun.head == action.doc_tok_idx
        && matches!(noun.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
        && dasu_case_link(noun, marker)
        && match case {
            CoveredCase::Venue => modifiers.is_empty(),
            CoveredCase::Object => {
                atoms
                    .iter()
                    .filter(|atom| atom.head == action.doc_tok_idx && atom.deprel == "obj")
                    .count()
                    == 1
                    && atoms[start..end]
                        .windows(2)
                        .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
                    && modifiers.iter().all(|atom| {
                        atom.head == noun.doc_tok_idx
                            && matches!(
                                atom.deprel.as_str(),
                                "compound" | "amod" | "nummod" | "det"
                            )
                    })
            }
        }
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::model::Backend;

    use crate::{
        GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslatedDocument,
        TranslationGlossary, analyze_and_translate,
    };

    fn entry(surface: &str, alias: &str) -> GlossaryEntry {
        GlossaryEntry {
            canonical: surface.into(),
            surfaces: vec![surface.into()],
            targets: [(TargetVariant::ZhHant, alias.into())].into(),
            ..GlossaryEntry::default()
        }
    }

    fn translated(source: &str, glossary: &TranslationGlossary) -> TranslatedDocument {
        analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            glossary,
            None,
            Backend::Onnx,
        )
        .unwrap()
    }

    fn translate(source: &str, surface: &str, alias: &str, typed: bool) -> TranslatedDocument {
        let mut glossary = TranslationGlossary::new();
        let entries = if typed {
            &mut glossary.places
        } else {
            &mut glossary.domain_terms
        };
        entries.push(entry(surface, alias));
        translated(source, &glossary)
    }

    #[test]
    fn a_venue_lock_keeps_its_full_glossary_span_under_a_polite_prohibition() {
        for (source, term, alias, expected) in [
            (
                "ここで写真を撮らないでください。",
                "ここで",
                "此處",
                "請不要在此處拍攝照片。",
            ),
            (
                "公園でタバコを吸わないでください。",
                "公園で",
                "指定公園",
                "請不要在指定公園抽菸。",
            ),
        ] {
            let result = translate(source, term, alias, true);
            assert_eq!(result.target_text, expected, "{source}");
            let [hit] = result.glossary_hits.as_slice() else {
                panic!("expected one venue hit");
            };
            assert_eq!(hit.kind, GlossaryKind::Place);
            assert_eq!(hit.canonical, term);
            assert_eq!(hit.source_span.start_char..hit.source_span.end_char, 0..3);
            assert_eq!(
                hit.target_span.start_char..hit.target_span.end_char,
                4..4 + alias.chars().count()
            );
        }
    }

    #[test]
    fn venue_repair_leaves_other_roles_and_explicit_venue_markers_alone() {
        for (source, term, alias, typed, expected) in [
            (
                "ここで写真を撮らないでください。",
                "ここで",
                "在此處",
                true,
                "請不要在此處拍攝照片。",
            ),
            (
                "ここで写真を撮らないでください。",
                "ここで",
                "此處",
                false,
                "請不要此處拍攝照片。",
            ),
            (
                "カメラで写真を撮らないでください。",
                "カメラで",
                "器材",
                true,
                "請不要器材拍攝照片。",
            ),
            (
                "公園でタバコを吸いません。",
                "公園で",
                "指定公園",
                true,
                "指定公園不抽菸。",
            ),
        ] {
            assert_eq!(
                translate(source, term, alias, typed).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn a_full_object_lock_keeps_the_complete_noun_phrase_after_the_action() {
        for (source, term, alias, expected) in [
            (
                "ここで本を読まないでください。",
                "本を",
                "書本",
                "請不要在這裡讀書本。",
            ),
            (
                "ここでこの写真を撮らないでください。",
                "この写真を",
                "這張相片",
                "請不要在這裡拍攝這張相片。",
            ),
        ] {
            let mut glossary = TranslationGlossary::new();
            glossary.objects.push(entry(term, alias));
            let result = translated(source, &glossary);
            assert_eq!(result.target_text, expected);
            let [hit] = result.glossary_hits.as_slice() else {
                panic!("expected one object hit");
            };
            assert_eq!(hit.kind, GlossaryKind::Object);
            assert_eq!(hit.canonical, term);
            assert_eq!(
                hit.source_span.start_char..hit.source_span.end_char,
                3..3 + term.chars().count()
            );
            assert_eq!(
                hit.target_span.start_char..hit.target_span.end_char,
                expected.chars().count() - alias.chars().count() - 1..expected.chars().count() - 1
            );
        }
    }

    #[test]
    fn a_full_object_lock_can_coexist_with_venue_and_predicate_locks() {
        let source = "ここで写真を撮らないでください。";
        for (verb, alias, expected) in [
            ("撮ら", "拍", "請不要在此處拍相片。"),
            ("撮らないでください", "請勿拍攝", "在此處請勿拍攝相片。"),
        ] {
            let mut glossary = TranslationGlossary::new();
            glossary.places.push(entry("ここで", "此處"));
            glossary.objects.push(entry("写真を", "相片"));
            glossary.domain_terms.push(entry(verb, alias));
            let result = translated(source, &glossary);
            assert_eq!(result.target_text, expected);
            assert_eq!(result.glossary_hits.len(), 3);
            for hit in &result.glossary_hits {
                let actual: String = result
                    .target_text
                    .chars()
                    .skip(hit.target_span.start_char)
                    .take(hit.target_span.end_char - hit.target_span.start_char)
                    .collect();
                let expected = match hit.kind {
                    GlossaryKind::Place => "此處",
                    GlossaryKind::Object => "相片",
                    GlossaryKind::DomainTerm => alias,
                    _ => panic!("unexpected kind"),
                };
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn object_repair_requires_the_typed_role_and_prohibition() {
        let mut glossary = TranslationGlossary::new();
        glossary.objects.push(entry("写真を", "相片"));
        assert_eq!(
            translated("ここで写真を撮りません。", &glossary).target_text,
            "在這裡不相片拍攝。"
        );
        assert_eq!(
            translate("ここで写真を撮らないでください。", "写真を", "相片", false).target_text,
            "請不要在這裡相片拍攝。"
        );
    }
}
