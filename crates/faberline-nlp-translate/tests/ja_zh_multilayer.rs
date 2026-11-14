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

#[test]
fn polish_multilayer_quotes_acl_genitive_passive_and_obliques() {
    let cases: &[(&str, &[&str], &[&str])] = &[
        (
            "「だめ…っ、そんなに見ないで…」彼女は恥ずかしそうに顔を赤くした。",
            &["「", "」", "不要那樣看", "害羞地"],
            &["“", "”", "不要那样看", "害羞地"],
        ),
        (
            "意識が飛ぶほどの快感に震えた。",
            &["神智飛散", "快感", "顫抖"],
            &["神智飞散", "快感", "颤抖"],
        ),
        (
            "彼の机の上には3冊の本が残っている。",
            &["3本書", "留著"],
            &["3本书", "留着"],
        ),
        (
            "彼女の胸に優しいキスをした。",
            &["溫柔的親吻"],
            &["温柔的亲吻"],
        ),
        (
            "雨が降っているから、諦めなかった。",
            &["因為正在下雨", "沒有放棄"],
            &["因为正在下雨", "没有放弃"],
        ),
    ];

    for (src, hant_expected, hans_expected) in cases {
        let hant = analyze_and_translate(
            src,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(src, &hant);
        for pat in *hant_expected {
            assert!(
                hant.target_text.contains(pat),
                "hant missing '{pat}' in: {}",
                hant.target_text
            );
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
        for pat in *hans_expected {
            assert!(
                hans.target_text.contains(pat),
                "hans missing '{pat}' in: {}",
                hans.target_text
            );
        }
    }
}

#[test]
fn formal_exam_benchmarks_chuken_catti_and_diplomatic_service() {
    let cases: &[(&str, &[&str], &[&str])] = &[
        (
            "姉は音楽を聴きながら宿題をしています。",
            &["姊姊一邊聽音樂，一邊做作業"],
            &["姐姐一边听音乐，一边做作业"],
        ),
        ("彼の財布が盗まれた。", &["被偷了"], &["被偷了"]),
        (
            "先生は私たちに明日早く来るように言いました。",
            &["老師要我們明天早點來"],
            &["老师要我们明天早点来"],
        ),
        (
            "彼は自分のミスに気づくとすぐに、みんなに謝罪した。",
            &["一發現自己的失誤，就立刻向大家道歉了"],
            &["一发现自己的失误，就立刻向大家道歉了"],
        ),
        (
            "どんなに忙しくても、彼は毎日必ず読書をする。",
            &["無論多麼忙碌", "他每天都一定閱讀"],
            &["无论多么忙碌", "他每天都一定阅读"],
        ),
        (
            "彼は長年努力を重ねた末に、ついに目標を達成した。",
            &["經過多年累積努力之後", "終於達成了目標"],
            &["经过多年累积努力之后", "终于达成了目标"],
        ),
        (
            "計画の成否は、今後のチーム全体の緊密な協力にかかっている。",
            &["取決於", "緊密的合作"],
            &["取决于", "紧密的合作"],
        ),
        (
            concat!(
                "円安の流れが加速している。",
                "輸入価格の上昇が物価高に拍車をかけ、家計の負担は一段と重くなっている。",
                "政府・日銀は知恵を集めて暮らしを守らねばならない。"
            ),
            &[
                "日圓貶值",
                "加劇",
                "進一步變重",
                "政府、日本央行匯集智慧，必須守護民生",
            ],
            &[
                "日元贬值",
                "加剧",
                "进一步变重",
                "政府、日本央行汇集智慧，必须守护民生",
            ],
        ),
        (
            concat!(
                "テクノロジーすなわち国力の時代には「国力がスポーツ競技を左右する」と言っても過言ではない。",
                "競技場は各国がテクノロジーを競う場ともなっているのだ。",
                "テクノロジーは今や選手のトレーニングや競技審判に欠かせず、スマートスタジアムまで登場している。",
                "位置情報システム、センサー、画像機器、AI 技術、通信技術の活用はスポーツ界では世界的な流れとなっている。"
            ),
            &[
                "在科技亦即國力的時代",
                "國力左右體育賽事",
                "也不為過",
                "也成為了",
                "不可或缺",
                "連智慧體育場都已登場",
                "定位資訊系統",
                "感測器",
                "通訊技術",
                "已成為世界性的趨勢",
            ],
            &[
                "在科技亦即国力的时代",
                "国力左右体育赛事",
                "也不为过",
                "也成为了",
                "不可或缺",
                "连智能体育场都已登场",
                "定位信息系统",
                "传感器",
                "通信技术",
                "已成为世界性的趋势",
            ],
        ),
        (
            "彼は100メートル泳げるようになりました。",
            &["他能游泳100公尺了"],
            &["他能游泳100米了"],
        ),
        (
            "彼の説得によって、彼女はついにこれまでの頑なな態度を改めた。",
            &["在他的勸說下", "改變了以往頑固的態度"],
            &["在他的劝说下", "改变了以往顽固的态度"],
        ),
        (
            "日本では少子高齢化が急速に進み、観光客の増加やデジタル技術の導入が地域経済を支えている。",
            &["觀光客", "數位技術", "正在支撐地方經濟"],
            &["观光客", "数字技术", "正在支撑地方经济"],
        ),
    ];

    for (src, hant_expected, hans_expected) in cases {
        let hant = analyze_and_translate(
            src,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(src, &hant);
        for pat in *hant_expected {
            assert!(
                hant.target_text.contains(pat),
                "hant missing '{pat}' in: {}",
                hant.target_text
            );
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
        for pat in *hans_expected {
            assert!(
                hans.target_text.contains(pat),
                "hans missing '{pat}' in: {}",
                hans.target_text
            );
        }
    }
}
