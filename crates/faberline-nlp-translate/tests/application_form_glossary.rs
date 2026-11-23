use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
    analyze_and_translate,
};

#[test]
fn application_form_glossary_preserves_filing_and_calendar_questions() {
    // DLC-01 and DLC-02 in date_noon_deadline_dev_v1.
    let cases = [
        (
            "今月は何日までに講座の申込書を出しますか。",
            "這個月最晚幾號交",
        ),
        ("今月は何日に講座の申込書を出しますか。", "這個月幾號交"),
    ];
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    for (source, question) in cases {
        for (term, qualifier) in [
            ("申込書", "課程"),
            ("申込書を", "課程"),
            ("講座の申込書", ""),
            ("講座の申込書を", ""),
        ] {
            let mut glossary = TranslationGlossary::new();
            glossary.objects.push(GlossaryEntry {
                canonical: term.to_owned(),
                surfaces: vec![term.to_owned()],
                targets: [(TargetVariant::ZhHant, "指定表單".to_owned())].into(),
                ..Default::default()
            });
            let translated =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(
                translated.target_text,
                format!("{question}{qualifier}指定表單？"),
                "{source} with {term}"
            );
            let [hit] = translated.glossary_hits.as_slice() else {
                panic!("expected one glossary hit for {term}");
            };
            assert_eq!(hit.kind, GlossaryKind::Object);
            let source_slice: String = source
                .chars()
                .skip(hit.source_span.start_char)
                .take(hit.source_span.end_char - hit.source_span.start_char)
                .collect();
            let target_slice: String = translated
                .target_text
                .chars()
                .skip(hit.target_span.start_char)
                .take(hit.target_span.end_char - hit.target_span.start_char)
                .collect();
            assert_eq!(source_slice, term);
            assert_eq!(target_slice, "指定表單");
        }
    }
}
