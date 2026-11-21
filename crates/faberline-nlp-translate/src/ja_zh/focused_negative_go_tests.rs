use faberline_nlp::doc::model::Backend;

use crate::{
    GlossaryEntry, TargetVariant, TranslateOptions, TranslatedDocument, TranslationGlossary,
    analyze_and_translate,
};

fn translate(source: &str, glossary: &TranslationGlossary) -> TranslatedDocument {
    analyze_and_translate(
        source,
        &TranslateOptions::new(TargetVariant::ZhHant),
        glossary,
        None,
        Backend::Onnx,
    )
    .unwrap()
}

#[test]
fn focus_keeps_the_physical_goal_of_a_negated_go() {
    let glossary = TranslationGlossary::new();
    for (source, expected) in [
        ("駅には行きませんでした。", "沒有去車站。"),
        ("図書館へは行きませんでした。", "沒有去圖書館。"),
        ("駅へは行かない。", "不去車站。"),
        ("図書館へは行きませんでしたか。", "沒有去圖書館嗎？"),
    ] {
        assert_eq!(
            translate(source, &glossary).target_text,
            expected,
            "{source}"
        );
    }
    let contrast = translate("映画は見て、図書館へは行きませんでした。", &glossary);
    assert!(contrast.target_text.ends_with("，沒有去圖書館。"));
}

#[test]
fn other_case_markers_and_nonnegative_actions_keep_their_roles() {
    let glossary = TranslationGlossary::new();
    for (source, expected) in [
        ("駅に行きませんでした。", "沒有去車站。"),
        ("図書館へ行きませんでした。", "沒有去圖書館。"),
        ("駅には行きました。", "去到車站了。"),
        ("橋までは行きませんでした。", "到橋沒有去。"),
        ("散歩には行きませんでした。", "沒有去散步。"),
    ] {
        assert_eq!(
            translate(source, &glossary).target_text,
            expected,
            "{source}"
        );
    }
}

#[test]
fn a_focused_place_from_the_glossary_keeps_its_lock() {
    let mut glossary = TranslationGlossary::new();
    glossary.places.push(GlossaryEntry {
        canonical: "図書館".into(),
        surfaces: vec!["図書館".into()],
        targets: [(TargetVariant::ZhHant, "圖書館".into())].into(),
        ..GlossaryEntry::default()
    });
    let result = translate("図書館へは行きませんでした。", &glossary);
    assert_eq!(result.target_text, "往圖書館沒有去。");
    assert_eq!(result.glossary_hits[0].canonical, "図書館");
}
