use faberline_nlp::{
    Span,
    doc::{analyze_document, model::Backend},
};
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
    analyze_and_translate, doc_options_for_translate, translate_document,
};

fn entry(source: &str, target: &str) -> GlossaryEntry {
    GlossaryEntry {
        canonical: source.to_owned(),
        surfaces: vec![source.to_owned()],
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
fn gaze_keeps_the_caller_marker_and_verb_in_source_order() {
    // Adopted D07 in naide_clause_dev_v1.
    let source = "窓を開けて、外を見ました。";
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let doc = analyze_document(
        source,
        &doc_options_for_translate(&options, &TranslationGlossary::new()),
    );
    for outside in [
        None,
        Some((GlossaryKind::Object, "外", "戶外")),
        Some((GlossaryKind::Object, "外を", "戶外")),
        Some((GlossaryKind::Place, "外", "庭院")),
        Some((GlossaryKind::Place, "外を", "庭院")),
    ] {
        for stem in [None, Some((GlossaryKind::DomainTerm, "見", "察看"))] {
            for window in [None, Some((GlossaryKind::Object, "窓", "側窗"))] {
                let expected_hits: Vec<_> = [outside, stem, window].into_iter().flatten().collect();
                let mut glossary = TranslationGlossary::new();
                for &(kind, term, alias) in &expected_hits {
                    let group = match kind {
                        GlossaryKind::Object => &mut glossary.objects,
                        GlossaryKind::Place => &mut glossary.places,
                        GlossaryKind::DomainTerm => &mut glossary.domain_terms,
                        _ => unreachable!(),
                    };
                    group.push(entry(term, alias));
                }
                let result =
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
                let first = if window.is_some() {
                    "打開側窗"
                } else {
                    "打開窗戶"
                };
                let action = if stem.is_some() { "察看" } else { "看" };
                let tail = match outside {
                    Some((_, _, alias)) => format!("{action}了{alias}"),
                    None => format!("往外{action}了"),
                };
                let expected = format!("{first}，{tail}。");
                assert_eq!(result.target_text, expected);
                assert_eq!(direct.target_text, expected);
                assert_eq!(direct.glossary_hits, result.glossary_hits);
                assert_eq!(result.glossary_hits.len(), expected_hits.len());
                for (kind, term, alias) in expected_hits {
                    let hit = result
                        .glossary_hits
                        .iter()
                        .find(|hit| hit.kind == kind && hit.canonical == term)
                        .unwrap();
                    assert_eq!(span_text(source, hit.source_span), term);
                    assert_eq!(span_text(&expected, hit.target_span), alias);
                }
            }
        }
    }
}
