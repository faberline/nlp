use faberline_nlp::doc::Document;
use faberline_nlp_translate::{
    TargetVariant, TranslateOptions, TranslationGlossary, translate_document,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Cases {
    rows: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    case: String,
    stage: String,
    source: String,
    doc: Document,
}

fn render(case: &Case) -> String {
    translate_document(
        &case.source,
        &case.doc,
        #[cfg(feature = "narrative")]
        None,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &TranslationGlossary::new(),
        None,
    )
    .expect("the frozen Document must match its source")
    .target_text
}

#[test]
fn both_source_structures_keep_the_existing_deadline_assertions() {
    let cases: Cases =
        serde_json::from_str(include_str!("fixtures/voice-negative-documents.json")).unwrap();
    let mut checked = 0;
    let mut variants = std::collections::BTreeSet::new();
    for case in cases
        .rows
        .iter()
        .filter(|c| matches!(c.case.as_str(), "test-come" | "test-see" | "test-past"))
    {
        checked += 1;
        assert!(variants.insert((case.case.as_str(), case.stage.as_str())));
        let output = render(case);
        if case.case == "test-past" {
            assert_eq!(output, "到六點為止沒有來到會場。", "{}", case.stage);
        } else {
            let cutoff = if case.case == "test-come" {
                "最晚在六點"
            } else {
                "最晚在明天"
            };
            assert!(
                output.contains(cutoff) && !output.contains("為止") && !output.contains("沒辦法"),
                "{} {}: {output}",
                case.case,
                case.stage,
            );
        }
    }
    assert_eq!(checked, 6);
}

#[test]
fn deadline_text_controls_across_source_structures() {
    let cases: Cases =
        serde_json::from_str(include_str!("fixtures/voice-negative-documents.json")).unwrap();
    let mut checked = 0;
    let mut variants = std::collections::BTreeSet::new();
    for case in cases
        .rows
        .iter()
        .filter(|c| matches!(c.case.as_str(), "F01" | "F02" | "F03" | "F04" | "F05"))
    {
        checked += 1;
        assert!(variants.insert((case.case.as_str(), case.stage.as_str())));
        let output = render(case);
        match case.case.as_str() {
            "F01" => assert!(
                output.contains("到六點為止") && !output.contains("最晚在六點"),
                "{}: {output}",
                case.stage,
            ),
            "F02" => assert!(
                output.contains("最晚在六點") && !output.contains("到六點為止"),
                "{}: {output}",
                case.stage,
            ),
            "F03" => {
                let cutoff = output.find("最晚在六點");
                let movie = output.find("電影");
                let station = output.find("車站");
                assert!(
                    matches!((cutoff, movie, station), (Some(a), Some(b), Some(c)) if a < b && b < c),
                    "{}: {output}",
                    case.stage,
                );
            }
            "F04" => assert!(
                output.contains("最晚在明天") && output.contains("資料"),
                "{}: {output}",
                case.stage,
            ),
            "F05" => assert!(
                output.contains("最晚在明天") && !output.contains("到明天為止"),
                "{}: {output}",
                case.stage,
            ),
            _ => unreachable!(),
        }
    }
    assert_eq!(checked, 10);
}

#[test]
fn an_explicit_patient_keeps_its_negative_text() {
    let case: Case =
        serde_json::from_str(include_str!("fixtures/voice-negative-passive.json")).unwrap();
    let output = render(&case);
    assert!(output.contains("荷物") && output.contains('不'), "{output}");
}
