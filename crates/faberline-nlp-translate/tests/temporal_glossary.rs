use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
    analyze_and_translate,
};

fn assert_time_gloss(source: &str, term: &str, replacement: &str, expected: &str) {
    let mut glossary = TranslationGlossary::new();
    glossary.domain_terms.push(entry(term, replacement));
    let document = analyze_and_translate(
        source,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &glossary,
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_eq!(document.target_text, expected, "{source} with {term}");
    let [hit] = document.glossary_hits.as_slice() else {
        panic!("expected exactly one glossary hit for {term}");
    };
    assert_eq!(hit.kind, GlossaryKind::DomainTerm);
    assert_eq!(span_text(source, hit.source_span), term);
    assert_eq!(
        span_text(&document.target_text, hit.target_span),
        replacement
    );
}

fn entry(term: &str, replacement: &str) -> GlossaryEntry {
    GlossaryEntry {
        canonical: term.to_owned(),
        surfaces: vec![term.to_owned()],
        targets: [(TargetVariant::ZhHant, replacement.to_owned())].into(),
        ..Default::default()
    }
}

fn span_text(text: &str, span: faberline_nlp::Span) -> String {
    text.chars()
        .skip(span.start_char)
        .take(span.end_char - span.start_char)
        .collect()
}

#[test]
fn protected_time_distinguishes_deadline_point_and_continuation() {
    // The same clock in adopted TMD-N5-01, TMD-N5-03 and TMD-N5-05.
    for (source, term, expected) in [
        (
            "私は六時まで日本語を勉強します。",
            "六時まで",
            "我學習日本語到六點整。",
        ),
        (
            "私は六時までに手紙を書きます。",
            "六時までに",
            "我最晚在六點整寫信。",
        ),
        ("私は六時に手紙を書きます。", "六時に", "我在六點整寫信。"),
    ] {
        assert_time_gloss(source, term, "六點整", expected);
    }
}

#[test]
fn protected_calendar_question_preserves_the_date_and_its_action() {
    // Adopted DLC-01 and DLC-02 keep the same filing action.
    for (source, term, expected) in [
        (
            "今月は何日までに講座の申込書を出しますか。",
            "何日までに",
            "這個月最晚幾日交課程報名表？",
        ),
        (
            "今月は何日に講座の申込書を出しますか。",
            "何日に",
            "這個月幾日交課程報名表？",
        ),
    ] {
        assert_time_gloss(source, term, "幾日", expected);
    }
}

#[test]
fn protected_date_and_application_keep_their_separate_roles() {
    // Both locks refer to the filing action in adopted DLC-01 and DLC-02.
    for (source, dates, prefix) in [
        (
            "今月は何日までに講座の申込書を出しますか。",
            &["何日", "何日まで", "何日までに"][..],
            "這個月最晚幾日交",
        ),
        (
            "今月は何日に講座の申込書を出しますか。",
            &["何日", "何日に"][..],
            "這個月幾日交",
        ),
    ] {
        for date in dates {
            for (object, rendered) in [
                ("申込書", "課程指定表單"),
                ("申込書を", "課程指定表單"),
                ("講座の申込書", "指定表單"),
                ("講座の申込書を", "指定表單"),
            ] {
                let mut glossary = TranslationGlossary::new();
                glossary.domain_terms.push(entry(date, "幾日"));
                glossary.objects.push(entry(object, "指定表單"));
                let document = analyze_and_translate(
                    source,
                    &TranslateOptions::new(TargetVariant::ZhHant),
                    &glossary,
                    None,
                    Backend::Onnx,
                )
                .unwrap();
                assert_eq!(
                    document.target_text,
                    format!("{prefix}{rendered}？"),
                    "{source} with {date} and {object}"
                );
                let [time, form] = document.glossary_hits.as_slice() else {
                    panic!("expected one date hit and one object hit");
                };
                for (hit, kind, term, replacement) in [
                    (time, GlossaryKind::DomainTerm, *date, "幾日"),
                    (form, GlossaryKind::Object, object, "指定表單"),
                ] {
                    assert_eq!(hit.kind, kind);
                    assert_eq!(span_text(source, hit.source_span), term);
                    assert_eq!(
                        span_text(&document.target_text, hit.target_span),
                        replacement
                    );
                }
                assert!(time.target_span.end_char <= form.target_span.start_char);
            }
        }
    }
}
