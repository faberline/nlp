use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
    analyze_and_translate,
};

fn assert_glosses(source: &str, terms: &[(GlossaryKind, &str, &str)], expected: &str) {
    let mut glossary = TranslationGlossary::new();
    for &(kind, term, replacement) in terms {
        let entries = match kind {
            GlossaryKind::DomainTerm => &mut glossary.domain_terms,
            GlossaryKind::Object => &mut glossary.objects,
            _ => panic!("unexpected glossary kind: {kind:?}"),
        };
        entries.push(GlossaryEntry {
            canonical: term.to_owned(),
            surfaces: vec![term.to_owned()],
            targets: [(TargetVariant::ZhHant, replacement.to_owned())].into(),
            ..Default::default()
        });
    }
    let document = analyze_and_translate(
        source,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &glossary,
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_eq!(document.target_text, expected, "{source} with {terms:?}");
    assert_eq!(document.glossary_hits.len(), terms.len());
    let range = |text: &str, start: usize, end: usize| {
        text.chars()
            .skip(start)
            .take(end - start)
            .collect::<String>()
    };
    for &(kind, term, replacement) in terms {
        let hit = document
            .glossary_hits
            .iter()
            .find(|hit| hit.kind == kind && hit.canonical == term)
            .unwrap();
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
            "我最晚在六點整寫完信。",
        ),
        ("私は六時に手紙を書きます。", "六時に", "我在六點整寫信。"),
    ] {
        assert_glosses(
            source,
            &[(GlossaryKind::DomainTerm, term, "六點整")],
            expected,
        );
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
        assert_glosses(
            source,
            &[(GlossaryKind::DomainTerm, term, "幾日")],
            expected,
        );
    }
}

#[test]
fn protected_letter_is_completed_only_at_a_deadline() {
    for (source, marker, native_time, protected_time) in [
        (
            "私は六時までに手紙を書きます。",
            "六時までに",
            "六點",
            "六點整",
        ),
        (
            "私は日曜日までに手紙を書きます。",
            "日曜日までに",
            "星期日",
            "週日",
        ),
    ] {
        for term in ["手紙", "手紙を"] {
            let object = (GlossaryKind::Object, term, "信函");
            assert_glosses(
                source,
                &[object],
                &format!("我最晚在{native_time}寫完信函。"),
            );
            assert_glosses(
                source,
                &[(GlossaryKind::DomainTerm, marker, protected_time), object],
                &format!("我最晚在{protected_time}寫完信函。"),
            );
        }
    }
    assert_glosses(
        "私は六時に手紙を書きます。",
        &[
            (GlossaryKind::DomainTerm, "六時に", "六點整"),
            (GlossaryKind::Object, "手紙", "信函"),
        ],
        "我在六點整寫信函。",
    );
    assert_glosses(
        "私は六時までに手紙を書きます。",
        &[
            (GlossaryKind::DomainTerm, "六時までに", "六點整"),
            (GlossaryKind::DomainTerm, "書き", "撰寫"),
        ],
        "我最晚在六點整撰寫信。",
    );
}

#[test]
fn protected_date_and_application_keep_their_separate_roles() {
    // Both glossary entries refer to the filing action in adopted DLC-01 and DLC-02.
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
            for (form, qualifier) in [
                ("申込書", "課程"),
                ("申込書を", "課程"),
                ("講座の申込書", ""),
                ("講座の申込書を", ""),
            ] {
                assert_glosses(
                    source,
                    &[
                        (GlossaryKind::DomainTerm, date, "幾日"),
                        (GlossaryKind::Object, form, "指定表單"),
                    ],
                    &format!("{prefix}{qualifier}指定表單？"),
                );
            }
        }
    }
}
