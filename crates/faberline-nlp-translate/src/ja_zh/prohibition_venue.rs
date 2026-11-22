//! Recover the venue case when a Place lock covers both the noun and its で.

use faberline_nlp::doc::analyze_document;

use super::{
    Atom, Bunsetsu, BunsetsuRole, ClauseDiscourseCtx, Document, LOCKED_PLACE_ATOM, SemanticClass,
    Span, Upos, core_non_punct, is_place_location_surface, is_polite_prohibition_ending,
};

pub(super) fn restore(
    groups: &mut [Bunsetsu],
    source: &[Atom],
    doc: &Document,
    ctx: &ClauseDiscourseCtx<'_>,
) {
    if ctx.locks.is_empty() || !doc.options.cpu_syntax {
        return;
    }
    let core = core_non_punct(source);
    let [prefix @ .., _, _, request] = core else {
        return;
    };
    let Some(action) = prefix.last() else {
        return;
    };
    if action.is_locked
        || action.upos != Some(Upos::Verb)
        || !is_polite_prohibition_ending(action, &core[prefix.len()..])
    {
        return;
    }
    let mut candidates = groups.iter_mut().filter_map(|group| {
        let [atom] = group.atoms.as_slice() else {
            return None;
        };
        if !atom.is_locked
            || atom.lemma.as_deref() != Some(LOCKED_PLACE_ATOM)
            || group.case_particle.is_some()
        {
            return None;
        }
        let word = ctx
            .source_chars
            .get(atom.span.start_char..atom.span.end_char)?;
        let [noun @ .., 'で'] = word else {
            return None;
        };
        let noun: String = noun.iter().collect();
        if !is_place_location_surface(&noun)
            && !faberline_nlp::lexicon::lookup_ja(&noun)
                .is_some_and(|entry| entry.sem_class == Some(SemanticClass::Location))
        {
            return None;
        }
        let lock = ctx
            .locks
            .iter()
            .find(|lock| lock.source_span == atom.span && lock.placeholder == atom.surface)?;
        Some((group, lock))
    });
    let Some((group, lock)) = candidates.next() else {
        return;
    };
    if candidates.next().is_some() || lock.replacement.trim_start().starts_with(['在', '於']) {
        return;
    }

    let start = core[0].span.start_char;
    let clause: String = ctx.source_chars[start..request.span.end_char]
        .iter()
        .collect();
    // Prove the venue with the native parse. Nearby locks may also include
    // particles and change where the parser attaches the venue.
    let mut options = doc.options.clone();
    options.lexicon.clear();
    options.names.clear();
    let parsed = analyze_document(&clause, &options);
    let relative = |span: Span| Span {
        start_char: span.start_char - start,
        end_char: span.end_char - start,
    };
    let span = relative(lock.source_span);
    let action_span = relative(action.span);
    let venue_case = parsed.tokens.windows(2).enumerate().any(|(index, tokens)| {
        let place = &tokens[0];
        let marker = &tokens[1];
        if place.span.start_char != span.start_char
            || place.span.end_char != marker.span.start_char
            || marker.span.end_char != span.end_char
            || marker.span.start_char + 1 != marker.span.end_char
            || !matches!(place.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
            || marker.upos != Some(Upos::Adp)
        {
            return false;
        }
        let Some(place_link) = parsed.syntax.get(index) else {
            return false;
        };
        let Some(case_link) = parsed.syntax.get(index + 1) else {
            return false;
        };
        place_link.deprel == "obl"
            && case_link.deprel == "case"
            && case_link.head == Some(index)
            && place_link
                .head
                .and_then(|head| parsed.tokens.get(head))
                .is_some_and(|verb| {
                    verb.sentence == place.sentence
                        && verb.span == action_span
                        && verb.lemma == action.lemma
                        && verb.upos == Some(Upos::Verb)
                })
    });
    if venue_case {
        // Keep the complete source and target spans on the original lock.
        group.role = BunsetsuRole::Obl;
        group.case_particle = Some("で".to_owned());
    }
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::model::Backend;

    use crate::{
        GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslatedDocument,
        TranslationGlossary, analyze_and_translate,
    };

    fn translate(source: &str, surface: &str, alias: &str, typed: bool) -> TranslatedDocument {
        let mut glossary = TranslationGlossary::new();
        let entries = if typed {
            &mut glossary.places
        } else {
            &mut glossary.domain_terms
        };
        entries.push(GlossaryEntry {
            canonical: surface.into(),
            surfaces: vec![surface.into()],
            targets: [(TargetVariant::ZhHant, alias.into())].into(),
            ..GlossaryEntry::default()
        });
        analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap()
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
}
