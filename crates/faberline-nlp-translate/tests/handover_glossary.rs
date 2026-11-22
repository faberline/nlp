use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
    analyze_and_translate,
};

const HS_12: &str = "授業を担当する先生は、授業後に佐藤さんへ資料を渡しました。";

fn check_glossary(terms: &[(GlossaryKind, &str, &str)], expected: &str) {
    let mut glossary = TranslationGlossary::new();
    for &(kind, term, target) in terms {
        let entries = match kind {
            GlossaryKind::Actor => &mut glossary.actors,
            GlossaryKind::Object => &mut glossary.objects,
            GlossaryKind::DomainTerm => &mut glossary.domain_terms,
            GlossaryKind::Place => &mut glossary.places,
        };
        entries.push(GlossaryEntry {
            canonical: term.to_owned(),
            surfaces: vec![term.to_owned()],
            targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
            ..Default::default()
        });
    }
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let translated =
        analyze_and_translate(HS_12, &options, &glossary, None, Backend::Onnx).unwrap();
    assert_eq!(translated.target_text, expected);
    assert_eq!(translated.glossary_hits.len(), terms.len());
    for &(kind, term, target) in terms {
        let hit = translated
            .glossary_hits
            .iter()
            .find(|hit| hit.kind == kind && hit.canonical == term)
            .unwrap();
        let source_slice: String = HS_12
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
        assert_eq!(target_slice, target);
    }
}

#[test]
fn handover_preserves_protected_arguments_and_course_time_together() {
    for time in ["授業後", "授業後に"] {
        check_glossary(
            &[
                (GlossaryKind::Actor, "先生", "任課老師"),
                (GlossaryKind::Actor, "佐藤さんへ", "佐藤同學"),
                (GlossaryKind::Object, "資料を", "指定資料"),
                (GlossaryKind::DomainTerm, time, "下課之後"),
            ],
            "負責授課的任課老師下課之後把指定資料交給佐藤同學。",
        );
    }
}

#[test]
fn complete_subject_replacement_does_not_repeat_its_relative_clause() {
    for teacher in ["授業を担当する先生", "授業を担当する先生は"] {
        check_glossary(
            &[(GlossaryKind::Actor, teacher, "任課老師")],
            "任課老師下課後把資料交給佐藤。",
        );
    }
}
