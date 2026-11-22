use faberline_nlp::doc::{analyze_document, model::Backend};
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

#[test]
fn omitted_carrying_keeps_the_item_and_the_callers_verb() {
    // Adopted D02 in naide_clause_dev_v1.
    let source = "傘を持たないで、出かけました。";
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let doc = analyze_document(
        source,
        &doc_options_for_translate(&options, &TranslationGlossary::new()),
    );
    for object in [None, Some("傘"), Some("傘を")] {
        for verb in [None, Some("手持"), Some("攜帶")] {
            let mut glossary = TranslationGlossary::new();
            glossary
                .objects
                .extend(object.map(|term| entry(term, "雨具")));
            glossary
                .domain_terms
                .extend(verb.map(|target| entry("持た", target)));
            let analyzed =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
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
            let item = if object.is_some() { "雨具" } else { "傘" };
            let action = verb.unwrap_or("帶");
            assert_eq!(
                analyzed.target_text,
                format!("沒有{action}{item}就出門了。")
            );
            assert_eq!(analyzed.target_text, direct.target_text);
            assert_eq!(analyzed.glossary_hits, direct.glossary_hits);
            let expected_hits = object
                .map(|term| (GlossaryKind::Object, term, "雨具"))
                .into_iter()
                .chain(verb.map(|target| (GlossaryKind::DomainTerm, "持た", target)));
            assert_eq!(
                analyzed.glossary_hits.len(),
                usize::from(object.is_some()) + usize::from(verb.is_some())
            );
            for (kind, from, to) in expected_hits {
                let hit = analyzed
                    .glossary_hits
                    .iter()
                    .find(|hit| hit.kind == kind)
                    .unwrap();
                let span_text = |text: &str, span: faberline_nlp::Span| {
                    text.chars()
                        .skip(span.start_char)
                        .take(span.end_char - span.start_char)
                        .collect::<String>()
                };
                assert_eq!(span_text(source, hit.source_span), from);
                assert_eq!(span_text(&analyzed.target_text, hit.target_span), to);
            }
        }
    }
}
