//! Multi-layer `ja -> zh-Hant / zh-Hans` deterministic translation integration tests.

use std::collections::BTreeMap;

use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    GlossaryEntry, PronounClass, SpeechRegister, TRANSLATE_SCHEMA, TargetVariant, TranslateOptions,
    TranslatedDocument, TranslationGlossary, TranslationTrack, analyze_and_translate, sha256_hex,
};

fn assert_valid_ja_zh_doc(src: &str, doc: &TranslatedDocument) {
    assert_eq!(doc.schema, TRANSLATE_SCHEMA);
    assert_eq!(doc.source_chars, src.chars().count());
    assert_eq!(doc.target_chars, doc.target_text.chars().count());
    assert_eq!(doc.source_sha256, sha256_hex(src));
    assert_eq!(doc.target_sha256, sha256_hex(&doc.target_text));
    assert_eq!(doc.track, TranslationTrack::VariantRule);
    assert!(
        doc.unknown.is_empty(),
        "expected zero NoModel unknowns for ja->zh, got: {:?}",
        doc.unknown
    );
    for (idx, s) in doc.sentences.iter().enumerate() {
        assert_eq!(s.sentence, idx);
        assert_eq!(s.track, TranslationTrack::VariantRule);
        let sliced: String = doc
            .target_text
            .chars()
            .skip(s.target_span.start_char)
            .take(s.target_span.end_char - s.target_span.start_char)
            .collect();
        assert_eq!(sliced, s.target_text);
    }
}

#[test]
fn everyday_sov_to_svo_and_honorifics_both_scripts() {
    let src = "私は図書館で本を読みます。美咲さんは佐藤先生に手紙を書いた。";

    let hant = analyze_and_translate(
        src,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_ja_zh_doc(src, &hant);
    assert!(
        hant.target_text.contains("圖書館") && hant.target_text.contains("讀書"),
        "hant SOV->SVO failed: {}",
        hant.target_text
    );
    assert!(
        hant.target_text.contains("美咲小姐")
            && hant.target_text.contains("佐藤老師")
            && hant.target_text.contains("信"),
        "hant honorific/false-friend failed: {}",
        hant.target_text
    );

    let hans = analyze_and_translate(
        src,
        &TranslateOptions::new(TargetVariant::ZhHans),
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_ja_zh_doc(src, &hans);
    assert!(
        hans.target_text.contains("图书馆") && hans.target_text.contains("读书"),
        "hans SOV->SVO failed: {}",
        hans.target_text
    );
    assert!(
        hans.target_text.contains("佐藤老师"),
        "hans honorific failed: {}",
        hans.target_text
    );
}

#[test]
fn negation_desiderative_passive_causative_and_multiclause() {
    let src = concat!(
        "私は行かない。",
        "彼はコーヒーを飲みたい。",
        "彼女は犬に噛まれた。",
        "先生は学生に本を読ませる。",
        "雨が降ったから、家に帰ります。しかし、彼女は諦めなかった。"
    );

    let hant = analyze_and_translate(
        src,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_ja_zh_doc(src, &hant);
    let t = &hant.target_text;
    assert!(t.contains("不去"), "negation failed: {t}");
    assert!(t.contains("想喝咖啡"), "desiderative failed: {t}");
    assert!(t.contains("被"), "passive failed: {t}");
    assert!(t.contains("讓"), "causative failed: {t}");
    assert!(
        t.contains("因為") || t.contains("但是"),
        "multiclause failed: {t}"
    );
}

#[test]
fn span_lock_glossary_url_code_math_conservation() {
    let src = "アリスは `config.toml` と $E=mc^2$ を https://faberline.dev/docs で確認した。";
    let mut glossary = TranslationGlossary::new();
    glossary.actors.push(GlossaryEntry {
        canonical: "alice".to_owned(),
        surfaces: vec!["アリス".to_owned()],
        targets: BTreeMap::from([
            (TargetVariant::ZhHant, "愛麗絲".to_owned()),
            (TargetVariant::ZhHans, "爱丽丝".to_owned()),
        ]),
        pronoun_class: PronounClass::FeminineSingular,
        speech_register: SpeechRegister::Plain,
    });

    let hant = analyze_and_translate(
        src,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &glossary,
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_ja_zh_doc(src, &hant);
    let t = &hant.target_text;
    assert!(t.contains("愛麗絲"), "glossary missing: {t}");
    assert!(t.contains("`config.toml`"), "code lock altered: {t}");
    assert!(t.contains("$E=mc^2$"), "math lock altered: {t}");
    assert!(
        t.contains("https://faberline.dev/docs"),
        "url lock altered: {t}"
    );
    assert!(
        !hant.glossary_hits.is_empty(),
        "expected glossary hit for アリス"
    );
}

#[test]
fn narrative_dialogue_and_pro_drop_both_scripts() {
    let src = "美咲は部屋に入った。「おはようございます」と言った。窓を開けた。";
    let mut glossary = TranslationGlossary::new();
    glossary.actors.push(GlossaryEntry {
        canonical: "misaki".to_owned(),
        surfaces: vec!["美咲".to_owned()],
        targets: BTreeMap::from([
            (TargetVariant::ZhHant, "美咲".to_owned()),
            (TargetVariant::ZhHans, "美咲".to_owned()),
        ]),
        pronoun_class: PronounClass::FeminineSingular,
        speech_register: SpeechRegister::Polite,
    });

    let hant = analyze_and_translate(
        src,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &glossary,
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_ja_zh_doc(src, &hant);
    assert!(
        hant.target_text.contains("「早安」"),
        "hant dialogue failed: {}",
        hant.target_text
    );

    let hans = analyze_and_translate(
        src,
        &TranslateOptions::new(TargetVariant::ZhHans),
        &glossary,
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_ja_zh_doc(src, &hans);
    assert!(
        hans.target_text.contains("“早安”"),
        "hans dialogue failed: {}",
        hans.target_text
    );
}

#[test]
fn content_neutral_adult_erotic_and_anatomical_ja_to_zh() {
    let src = concat!(
        "「気持ちいい……もう、我慢できない」と彼女は声を漏らす。",
        "息が荒くなり、快感に震える。",
        "陰茎が膣の奥まで届き、熱く濡れる。",
        "乳首を愛撫され、頭が真っ白になる。",
        "中に出す直前、絶頂の余韻に包まれた。"
    );

    let hant = analyze_and_translate(
        src,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_ja_zh_doc(src, &hant);
    let ht = &hant.target_text;
    for expected in [
        "好舒服",
        "忍不住了",
        "漏出聲音",
        "呼吸變得急促",
        "因快感而顫抖",
        "陰莖",
        "陰道",
        "頂到最深處",
        "濡濕發燙",
        "乳頭",
        "愛撫",
        "腦中一片空白",
        "射在裡面",
        "絕頂",
        "餘韻",
    ] {
        assert!(ht.contains(expected), "hant missing '{expected}' in: {ht}");
    }

    let hans = analyze_and_translate(
        src,
        &TranslateOptions::new(TargetVariant::ZhHans),
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_ja_zh_doc(src, &hans);
    let hs = &hans.target_text;
    for expected in [
        "好舒服",
        "忍不住了",
        "漏出声音",
        "呼吸变得急促",
        "因快感而颤抖",
        "阴茎",
        "阴道",
        "顶到最深处",
        "濡湿发烫",
        "乳头",
        "爱抚",
        "脑中一片空白",
        "射在里面",
        "绝顶",
        "余韵",
    ] {
        assert!(hs.contains(expected), "hans missing '{expected}' in: {hs}");
    }
}
