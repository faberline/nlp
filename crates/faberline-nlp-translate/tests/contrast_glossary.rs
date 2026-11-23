use faberline_nlp::{
    Span,
    doc::{analyze_document, model::Backend},
};
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
    analyze_and_translate, doc_options_for_translate, translate_document,
};

fn entry(term: &str, target: &str) -> GlossaryEntry {
    GlossaryEntry {
        canonical: term.to_owned(),
        surfaces: vec![term.to_owned()],
        targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
        ..Default::default()
    }
}

fn span_text(text: &str, span: Span) -> String {
    text.chars()
        .skip(span.start_char)
        .take(span.end_char - span.start_char)
        .collect()
}

#[test]
fn paired_topics_keep_their_polarity_under_glossary_locks() {
    // Adopted D08 in naide_clause_dev_v1.
    let source = "朝ご飯は食べて、学校には行きませんでした。";
    let native_meal = match faberline_nlp::lexicon::ja::lookup_ja_gloss(
        "朝ご飯",
        Some(faberline_nlp::doc::Upos::Noun),
    ) {
        None => "早上有吃飯",
        Some("早餐") => "早餐有吃",
        Some(gloss) => panic!("unexpected breakfast gloss: {gloss}"),
    };
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let doc = analyze_document(
        source,
        &doc_options_for_translate(&options, &TranslationGlossary::new()),
    );
    for meal in [None, Some("朝ご飯"), Some("朝ご飯は")] {
        for place in [None, Some("学校"), Some("学校に"), Some("学校には")] {
            for stem in [None, Some("行き")] {
                let mut glossary = TranslationGlossary::new();
                glossary
                    .objects
                    .extend(meal.map(|term| entry(term, "早餐")));
                glossary
                    .places
                    .extend(place.map(|term| entry(term, "新校舍")));
                glossary
                    .domain_terms
                    .extend(stem.map(|term| entry(term, "前往")));
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
                let first = if meal.is_some() {
                    "早餐有吃"
                } else {
                    native_meal
                };
                let action = if stem.is_some() { "前往" } else { "去" };
                let destination = if place.is_some() {
                    "新校舍"
                } else {
                    "學校"
                };
                let expected = format!("{first}，但沒有{action}{destination}。");
                assert_eq!(analyzed.target_text, expected);
                assert_eq!(direct.target_text, expected);
                assert_eq!(direct.glossary_hits, analyzed.glossary_hits);
                let expected_hits = [
                    (GlossaryKind::Object, meal, "早餐"),
                    (GlossaryKind::Place, place, "新校舍"),
                    (GlossaryKind::DomainTerm, stem, "前往"),
                ];
                assert_eq!(
                    analyzed.glossary_hits.len(),
                    expected_hits
                        .iter()
                        .filter(|(_, term, _)| term.is_some())
                        .count()
                );
                for (kind, term, target) in expected_hits {
                    if let Some(term) = term {
                        let hit = analyzed
                            .glossary_hits
                            .iter()
                            .find(|hit| hit.kind == kind)
                            .unwrap();
                        assert_eq!(span_text(source, hit.source_span), term);
                        assert_eq!(span_text(&expected, hit.target_span), target);
                    }
                }
            }
        }
    }
}

#[test]
fn whole_meal_and_trip_locks_keep_polarity_across_source_boundaries() {
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    for clause_boundary in ["、", "，", ",", ""] {
        for (source_end, first_clause, target_end) in [
            ("。", "早餐有享用，但", "。"),
            ("？", "早餐享用，", "？"),
            ("?", "早餐享用，", "？"),
            ("か。", "早餐享用，", "嗎？"),
            ("か", "早餐享用，", "嗎？"),
        ] {
            let source =
                format!("朝ご飯は食べて{clause_boundary}学校には行きませんでした{source_end}");
            let doc = analyze_document(
                &source,
                &doc_options_for_translate(&options, &TranslationGlossary::new()),
            );
            for (alias, negative) in [
                ("前往", "沒有前往"),
                ("沒有前往", "沒有前往"),
                ("未曾前往", "未曾前往"),
            ] {
                let glossary = TranslationGlossary {
                    objects: vec![entry("朝ご飯は", "早餐")],
                    places: vec![entry("学校には", "新校舍")],
                    domain_terms: vec![entry("食べて", "享用"), entry("行きませんでした", alias)],
                    ..Default::default()
                };
                let expected = format!("{first_clause}{negative}新校舍{target_end}");
                let result =
                    analyze_and_translate(&source, &options, &glossary, None, Backend::Onnx)
                        .unwrap();
                let direct = translate_document(
                    &source,
                    &doc,
                    #[cfg(feature = "narrative")]
                    None,
                    &options,
                    &glossary,
                    None,
                )
                .unwrap();
                assert_eq!(result.target_text, expected, "{source}");
                assert_eq!(direct.target_text, expected);
                assert_eq!(direct.glossary_hits, result.glossary_hits);
                assert_eq!(direct.clause_alignments, result.clause_alignments);
                let alignments = &result.clause_alignments;
                assert_eq!(
                    alignments
                        .iter()
                        .map(|clause| clause.source_span)
                        .collect::<Vec<_>>(),
                    doc.segmentation
                        .clauses
                        .iter()
                        .map(|clause| clause.span)
                        .collect::<Vec<_>>(),
                    "{source}"
                );
                let first = alignments.first().unwrap();
                let last = alignments.last().unwrap();
                assert_eq!(first.target_span.start_char, 0);
                assert_eq!(last.target_span.end_char, expected.chars().count());
                for pair in alignments.windows(2) {
                    assert_eq!(pair[0].target_span.end_char, pair[1].target_span.start_char);
                }
                let terms = [
                    ("朝ご飯は", "早餐"),
                    ("食べて", "享用"),
                    ("学校には", "新校舍"),
                    ("行きませんでした", alias),
                ];
                assert_eq!(result.glossary_hits.len(), terms.len());
                for (hit, (term, replacement)) in result.glossary_hits.iter().zip(terms) {
                    assert_eq!(span_text(&source, hit.source_span), term);
                    assert_eq!(span_text(&expected, hit.target_span), replacement);
                }
            }
        }
    }
}
