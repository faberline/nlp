use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
    analyze_and_translate, doc_options_for_translate, translate_document,
};

fn check(source: &str, object: (&str, &str), place: Option<(&str, &str)>, expected: &str) {
    let entry = |(term, target): (&str, &str)| GlossaryEntry {
        canonical: term.to_owned(),
        surfaces: vec![term.to_owned()],
        targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
        ..Default::default()
    };
    let mut glossary = TranslationGlossary::new();
    glossary.objects.push(entry(object));
    glossary.places.extend(place.map(entry));
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let analyzed = analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
    let doc = faberline_nlp::doc::analyze_document(
        source,
        &doc_options_for_translate(&options, &TranslationGlossary::new()),
    );
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
    for result in [analyzed, direct] {
        assert_eq!(result.target_text, expected);
        let expected_hits = [(GlossaryKind::Object, object)]
            .into_iter()
            .chain(place.map(|pair| (GlossaryKind::Place, pair)));
        assert_eq!(result.glossary_hits.len(), 1 + usize::from(place.is_some()));
        for (kind, (term, replacement)) in expected_hits {
            let hit = result
                .glossary_hits
                .iter()
                .find(|h| h.kind == kind)
                .unwrap();
            let at = |text: &str, span: faberline_nlp::Span| {
                text.chars()
                    .skip(span.start_char)
                    .take(span.end_char - span.start_char)
                    .collect::<String>()
            };
            assert_eq!(at(source, hit.source_span), term);
            assert_eq!(at(&result.target_text, hit.target_span), replacement);
        }
    }
}

#[test]
fn omitted_actions_keep_their_objects_and_the_later_destinations() {
    // D01–D03 in the adopted naide_clause_dev_v1 set.
    check(
        "朝ご飯を食べないで、学校へ行きました。",
        ("朝ご飯を", "早餐"),
        Some(("学校へ", "新校舍")),
        "沒有吃早餐就去了新校舍。",
    );
    check(
        "傘を持たないで、出かけました。",
        ("傘を", "雨具"),
        None,
        "沒有帶雨具就出門了。",
    );
    check(
        "靴を脱がないで、家に入りました。",
        ("靴を", "鞋履"),
        Some(("家に", "舊屋")),
        "沒有脫下鞋履就進入舊屋了。",
    );
}
