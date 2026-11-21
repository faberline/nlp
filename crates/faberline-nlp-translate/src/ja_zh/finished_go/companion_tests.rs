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
fn completed_travel_keeps_the_companion_and_place_separate() {
    let glossary = TranslationGlossary::new();
    for (source, expected) in [
        ("姉と一緒に図書館に行きました。", "跟姊姊一起去了圖書館。"),
        ("父と駅へ行きました。", "跟父親去了車站。"),
        ("父と一緒に市場に行った。", "跟父親一起去了市場。"),
        ("一緒に駅へ行った。", "一起去了車站。"),
        (
            "昨日、妹と一緒に学校へ行きました。",
            "昨天，跟妹妹一起去了學校。",
        ),
        ("兄は姉と一緒に駅へ行きました。", "哥哥跟姊姊一起去了車站。"),
    ] {
        assert_eq!(
            translate(source, &glossary).target_text,
            expected,
            "{source}"
        );
    }
}

#[test]
fn coordination_invitations_and_purposes_follow_their_existing_paths() {
    let glossary = TranslationGlossary::new();
    for (source, expected) in [
        ("兄と妹が駅へ行きました。", "哥哥和妹妹往車站去了。"),
        ("兄と一緒に空港へ行きませんか。", "要不要和哥哥一起去機場？"),
        ("姉と映画を見に行った。", "跟姊姊去看電影了。"),
    ] {
        assert_eq!(
            translate(source, &glossary).target_text,
            expected,
            "{source}"
        );
    }
}

#[test]
fn a_glossary_place_retains_its_label() {
    let mut glossary = TranslationGlossary::new();
    glossary.places.push(GlossaryEntry {
        canonical: "駅".into(),
        surfaces: vec!["駅".into()],
        targets: [(TargetVariant::ZhHant, "北站".into())].into(),
        ..GlossaryEntry::default()
    });
    let output = translate("父と駅へ行きました。", &glossary);
    assert_eq!(output.target_text, "跟父親去了北站。");
    assert_eq!(output.glossary_hits[0].canonical, "駅");
}
