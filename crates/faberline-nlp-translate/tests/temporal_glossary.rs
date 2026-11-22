use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
    analyze_and_translate,
};

fn assert_time_gloss(source: &str, term: &str, replacement: &str, expected: &str) {
    let mut glossary = TranslationGlossary::new();
    glossary.domain_terms.push(GlossaryEntry {
        canonical: term.to_owned(),
        surfaces: vec![term.to_owned()],
        targets: [(TargetVariant::ZhHant, replacement.to_owned())].into(),
        ..Default::default()
    });
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
    let range = |text: &str, start: usize, end: usize| {
        text.chars()
            .skip(start)
            .take(end - start)
            .collect::<String>()
    };
    assert_eq!(
        range(source, hit.source_span.start_char, hit.source_span.end_char),
        term
    );
    assert_eq!(
        range(
            &document.target_text,
            hit.target_span.start_char,
            hit.target_span.end_char
        ),
        replacement
    );
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
