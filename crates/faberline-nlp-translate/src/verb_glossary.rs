//! Keep a verified Japanese stem's grammar separate from its rendering lock.

use std::collections::HashSet;

use faberline_nlp::doc::{Document, Lang, Upos, analyze_document};

use crate::lock::{LockOrigin, LockedSpan};
use crate::{GlossaryKind, TranslationGlossary};

pub(crate) fn is_stem_lock(lock: &LockedSpan, doc: &Document, sentence: usize) -> bool {
    if !lock.mandatory
        || !matches!(
            lock.origin,
            LockOrigin::Glossary {
                kind: GlossaryKind::DomainTerm,
                ..
            }
        )
    {
        return false;
    }
    let index = doc
        .tokens
        .partition_point(|token| token.span.start_char < lock.source_span.start_char);
    doc.tokens.get(index).is_some_and(|token| {
        token.sentence == sentence && token.span == lock.source_span && has_ending(doc, index)
    })
}

fn has_ending(doc: &Document, index: usize) -> bool {
    let stem = &doc.tokens[index];
    let Some(ending) = doc.tokens.get(index + 1) else {
        return false;
    };
    stem.lang == Lang::Ja
        && stem.upos == Some(Upos::Verb)
        && stem.lemma.is_some()
        && ending.sentence == stem.sentence
        && ending.span.start_char == stem.span.end_char
        && doc.syntax.get(index).is_some_and(|link| {
            link.token == index
                && link.span == stem.span
                && link.sentence == stem.sentence
                && matches!(link.deprel.as_str(), "root" | "advcl" | "conj")
        })
        && doc.syntax.get(index + 1).is_some_and(|link| {
            link.token == index + 1
                && link.span == ending.span
                && link.head == Some(index)
                && link.sentence == stem.sentence
                && matches!(
                    (ending.upos, link.deprel.as_str()),
                    (Some(Upos::Aux), "aux") | (Some(Upos::Sconj), "mark")
                )
        })
}

pub(crate) fn structural_document(
    text: &str,
    chars: &[char],
    doc: &Document,
    glossary: &TranslationGlossary,
) -> Option<Document> {
    if !doc.options.enrich_morphology || !doc.options.cpu_syntax || glossary.domain_terms.is_empty()
    {
        return None;
    }
    // The compiled glossary can resolve sibling variants and source-surface fallbacks.
    let domain: HashSet<&str> = glossary
        .domain_terms
        .iter()
        .flat_map(|entry| entry.surfaces.iter().map(String::as_str))
        .collect();
    let protected: HashSet<&str> = glossary
        .objects
        .iter()
        .chain(&glossary.actors)
        .chain(&glossary.places)
        .flat_map(|entry| entry.surfaces.iter().map(String::as_str))
        .chain(doc.options.names.iter().map(String::as_str))
        .collect();
    let candidates: HashSet<&str> = doc
        .options
        .lexicon
        .iter()
        .map(String::as_str)
        .filter(|term| {
            !term.is_empty()
                && domain.contains(term)
                && !protected.contains(term)
                && text.contains(*term)
        })
        .collect();
    if candidates.is_empty() {
        return None;
    }

    let mut options = doc.options.clone();
    options
        .lexicon
        .retain(|word| !candidates.contains(word.as_str()));
    let plain = analyze_document(text, &options);
    let matched: HashSet<_> = plain
        .tokens
        .iter()
        .enumerate()
        .filter(|(index, _)| has_ending(&plain, *index))
        .map(|(_, token)| (token.span.start_char, token.span.end_char))
        .collect();
    let eligible: HashSet<_> = candidates
        .iter()
        .copied()
        .filter(|term| {
            let word: Vec<_> = term.chars().collect();
            let mut occurrences = chars
                .windows(word.len())
                .enumerate()
                .filter(|(_, slice)| *slice == word.as_slice());
            let known = |start| matched.contains(&(start, start + word.len()));
            occurrences.next().is_some_and(|(start, _)| known(start))
                && occurrences.all(|(start, _)| known(start))
        })
        .collect();
    if eligible.is_empty() {
        return None;
    }
    if candidates == eligible {
        return Some(plain);
    }
    let mut options = doc.options.clone();
    options
        .lexicon
        .retain(|word| !eligible.contains(word.as_str()));
    Some(analyze_document(text, &options))
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::model::Backend;

    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
        doc_options_for_translate, translate_document,
    };

    use super::analyze_document;

    #[test]
    fn smoking_stem_gloss_keeps_the_prohibition_and_tobacco_object() {
        let text = "ここでたばこを吸わないでください。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let entry = |surface: &str, target: &str| GlossaryEntry {
            canonical: surface.into(),
            surfaces: vec![surface.into()],
            targets: [(TargetVariant::ZhHant, target.into())].into(),
            ..GlossaryEntry::default()
        };
        let mut glossary = TranslationGlossary::new();
        glossary.domain_terms.push(entry("吸わ", "吸"));
        let mut doc_options = doc_options_for_translate(&options, &glossary);
        doc_options.lexicon.push("たばこ".into());
        let doc = analyze_document(text, &doc_options);
        let hinted = translate_document(
            text,
            &doc,
            #[cfg(feature = "narrative")]
            None,
            &options,
            &glossary,
            None,
        )
        .unwrap();
        assert_eq!(hinted.target_text, "請不要在這裡吸菸。");
        assert_eq!(hinted.glossary_hits.len(), 1);

        glossary.objects.push(entry("たばこを", "香菸"));
        let locked = analyze_and_translate(text, &options, &glossary, None, Backend::Onnx).unwrap();
        assert_eq!(locked.target_text, "請不要在這裡吸香菸。");
        assert_eq!(locked.glossary_hits.len(), 2);
        for (canonical, source, target) in [("吸わ", 7..9, 6..7), ("たばこを", 3..7, 7..9)] {
            let hit = locked
                .glossary_hits
                .iter()
                .find(|hit| hit.canonical == canonical)
                .unwrap();
            assert_eq!(hit.source_span.start_char..hit.source_span.end_char, source);
            assert_eq!(hit.target_span.start_char..hit.target_span.end_char, target);
        }
    }

    #[test]
    fn stem_glosses_keep_linked_endings_and_their_own_source_span() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        for (text, surface, target, expected) in [
            (
                "傘を持たないで、出かけました。",
                "持た",
                "手持",
                "沒有手持傘就出門了。",
            ),
            (
                "窓を開けて、外を見ました。",
                "見",
                "察看",
                "打開窗戶，察看了外面。",
            ),
            (
                "朝ご飯は食べて、学校には行きませんでした。",
                "行き",
                "前往",
                "飯早上吃，沒有前往到學校。",
            ),
            (
                "窓を開けて、外を見ました。",
                "見ました",
                "察看了",
                "打開窗戶，察看了外面。",
            ),
        ] {
            let mut glossary = TranslationGlossary::new();
            glossary.domain_terms.push(GlossaryEntry {
                canonical: surface.into(),
                surfaces: vec![surface.into()],
                targets: [(TargetVariant::ZhHant, target.into())].into(),
                ..GlossaryEntry::default()
            });
            let analyzed =
                analyze_and_translate(text, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(analyzed.target_text, expected);
            let doc = analyze_document(text, &doc_options_for_translate(&options, &glossary));
            let direct = translate_document(
                text,
                &doc,
                #[cfg(feature = "narrative")]
                None,
                &options,
                &glossary,
                None,
            )
            .unwrap();
            assert_eq!(direct.target_text, expected);
            assert_eq!(direct.glossary_hits, analyzed.glossary_hits);
            let [hit] = direct.glossary_hits.as_slice() else {
                panic!("expected one glossary hit")
            };
            let source_start = text[..text.find(surface).unwrap()].chars().count();
            let target_start = expected[..expected.find(target).unwrap()].chars().count();
            assert_eq!(hit.source_span.start_char, source_start);
            assert_eq!(
                hit.source_span.end_char,
                source_start + surface.chars().count()
            );
            assert_eq!(hit.target_span.start_char, target_start);
            assert_eq!(
                hit.target_span.end_char,
                target_start + target.chars().count()
            );
        }
    }

    #[test]
    fn stem_gloss_keeps_past_when_target_uses_sibling_variant() {
        let text = "窓を開けて、外を見ました。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let mut glossary = TranslationGlossary::new();
        glossary.domain_terms.push(GlossaryEntry {
            canonical: "見".into(),
            surfaces: vec!["見".into()],
            targets: [(TargetVariant::ZhHans, "检视".into())].into(),
            ..GlossaryEntry::default()
        });

        let result = analyze_and_translate(text, &options, &glossary, None, Backend::Onnx).unwrap();
        assert_eq!(result.target_text, "打開窗戶，檢視了外面。");
        let [hit] = result.glossary_hits.as_slice() else {
            panic!("expected one glossary hit")
        };
        let locked_source: String = text
            .chars()
            .skip(hit.source_span.start_char)
            .take(hit.source_span.end_char - hit.source_span.start_char)
            .collect();
        let locked_target: String = result
            .target_text
            .chars()
            .skip(hit.target_span.start_char)
            .take(hit.target_span.end_char - hit.target_span.start_char)
            .collect();
        assert_eq!(locked_source, "見");
        assert_eq!(locked_target, "檢視");
    }
}
