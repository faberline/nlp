//! Multi-layer `ja -> zh-Hant / zh-Hans` deterministic translation integration tests.

use std::collections::BTreeMap;

use faberline_nlp::Span;
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
    for clause in &doc.clause_alignments {
        let sentence = &doc.sentences[clause.sentence];
        assert!(clause.source_span.start_char <= clause.source_span.end_char);
        assert!(clause.source_span.end_char <= doc.source_chars);
        assert!(clause.target_span.start_char <= clause.target_span.end_char);
        assert!(clause.target_span.start_char >= sentence.target_span.start_char);
        assert!(
            clause.target_span.end_char <= sentence.target_span.end_char,
            "clause {:?} exceeds sentence {:?}: {}",
            clause.target_span,
            sentence.target_span,
            doc.target_text
        );
    }
}

fn slice_chars(text: &str, span: Span) -> String {
    text.chars()
        .skip(span.start_char)
        .take(span.end_char - span.start_char)
        .collect()
}

fn actor_glossary(alice: &str) -> TranslationGlossary {
    let mut glossary = TranslationGlossary::new();
    for (surface, target) in [("アリス", alice), ("ボブ", "鮑伯")] {
        glossary.actors.push(GlossaryEntry {
            canonical: surface.to_owned(),
            surfaces: vec![surface.to_owned()],
            targets: BTreeMap::from([(TargetVariant::ZhHant, target.to_owned())]),
            ..GlossaryEntry::default()
        });
    }
    glossary
}

#[test]
fn clause_spans_follow_glossary_replacement_and_space_cleanup() {
    let source = "アリスは本を読む、ボブは手紙を書く。";
    for (replacement, expected_name) in [
        ("愛麗絲", "愛麗絲"),
        ("愛麗絲與她的閱讀小隊", "愛麗絲與她的閱讀小隊"),
        ("愛 麗 絲", "愛麗絲"),
    ] {
        let doc = analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &actor_glossary(replacement),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        let clause_texts: Vec<String> = doc
            .clause_alignments
            .iter()
            .map(|c| slice_chars(&doc.target_text, c.target_span))
            .collect();
        assert_eq!(
            clause_texts,
            [format!("{expected_name}讀書，"), "鮑伯寫信。".to_owned()]
        );
        assert_eq!(clause_texts.concat(), doc.target_text);
        for hit in &doc.glossary_hits {
            let expected = if slice_chars(source, hit.source_span) == "アリス" {
                expected_name
            } else {
                "鮑伯"
            };
            assert_eq!(slice_chars(&doc.target_text, hit.target_span), expected);
        }
    }
}

#[test]
fn clause_spans_follow_opaque_restoration_and_optional_latin_spacing() {
    let source = "私は `x` を読む、アリスは本を読む。";
    for spacing in [false, true] {
        let mut options = TranslateOptions::new(TargetVariant::ZhHant);
        options.cjk_latin_spacing = spacing;
        let doc = analyze_and_translate(
            source,
            &options,
            &actor_glossary("Alice"),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        let clause_texts: Vec<String> = doc
            .clause_alignments
            .iter()
            .map(|c| slice_chars(&doc.target_text, c.target_span))
            .collect();
        assert_eq!(
            clause_texts,
            [
                "我讀`x`，".to_owned(),
                if spacing {
                    "Alice 讀書。"
                } else {
                    "Alice讀書。"
                }
                .to_owned(),
            ]
        );
        assert_eq!(clause_texts.concat(), doc.target_text);
        let code = doc
            .opaque_spans
            .iter()
            .find(|op| slice_chars(source, op.source_span) == "`x`")
            .unwrap();
        assert_eq!(
            slice_chars(&doc.target_text, code.target_span.unwrap()),
            "`x`"
        );
        assert_eq!(doc.glossary_hits.len(), 1);
        let hit_span = doc.glossary_hits[0].target_span;
        assert!(hit_span.start_char <= hit_span.end_char);
        assert!(hit_span.end_char <= doc.target_chars);
        assert_eq!(slice_chars(&doc.target_text, hit_span).trim(), "Alice");
    }
}

#[test]
fn clause_spans_follow_length_changing_simplified_variants() {
    for (source, first_clause, second_start) in [
        ("演算法を使うから、本を読む。", "因为使用算法，", 7),
        ("私は五メートル歩いて、本を読む。", "我五米走，", 5),
        ("泡麵を食べるから、本を読む。", "因为吃方便面，", 7),
    ] {
        let doc = analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHans),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        assert_eq!(doc.clause_alignments.len(), 2);
        let clause_texts: Vec<String> = doc
            .clause_alignments
            .iter()
            .map(|c| slice_chars(&doc.target_text, c.target_span))
            .collect();
        assert_eq!(clause_texts, [first_clause, "读书。"]);
        assert_eq!(
            doc.clause_alignments[1].target_span.start_char,
            second_start
        );
        assert_eq!(clause_texts.concat(), doc.target_text);
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

#[test]
fn conversational_and_n3_n2_discourse_patterns_and_holdouts() {
    let paragraph = concat!(
        "昨日、久しぶりに友達と会ったんだけど、昔と全然変わっていなくて驚いた。",
        "彼は「最近、仕事が忙しくて休む暇もない」と言っていたくせに、週末になると必ず旅行に出かけているらしい。",
        "正直、羨ましくないと言えば嘘になるけど、だからといって自分も同じ生活を送りたいわけではない。",
        "もし時間とお金に余裕があったとしても、私は家でのんびり過ごすほうが好きだ。",
        "それに、何もしない時間だって、決して無駄とは限らないと思う。"
    );

    let hant = analyze_and_translate(
        paragraph,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_ja_zh_doc(paragraph, &hant);
    for pat in [
        "久違地跟朋友見面了",
        "跟以前完全沒有改變",
        "感到驚訝了",
        "他明明說著「最近，工作忙碌，連休息的空閒都沒有」",
        "一到週末，卻聽說一定會出門旅行",
        "老實說，要說不羨慕，那是騙人的",
        "但話雖如此自己也並不是想過同樣的生活",
        "就算時間和金錢有餘裕，我也比較喜歡在家悠閒地度過",
        "而且，就算是什麼都不做的時間，我覺得絕不見得是浪費",
    ] {
        assert!(
            hant.target_text.contains(pat),
            "hant missing '{pat}' in: {}",
            hant.target_text
        );
    }

    let hans = analyze_and_translate(
        paragraph,
        &TranslateOptions::new(TargetVariant::ZhHans),
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_ja_zh_doc(paragraph, &hans);
    for pat in [
        "久违地跟朋友见面了",
        "跟以前完全没有改变",
        "感到惊讶了",
        "他明明说着“最近，工作忙碌，连休息的空闲都没有”",
        "一到周末，却听说一定会出门旅行",
        "老实说，要说不羡慕，那是骗人的",
        "但话虽如此自己也并不是想过同样的生活",
        "就算时间和金钱有余裕，我也比较喜欢在家悠闲地度过",
        "而且，就算是什么都不做的时间，我觉得绝不见得是浪费",
    ] {
        assert!(
            hans.target_text.contains(pat),
            "hans missing '{pat}' in: {}",
            hans.target_text
        );
    }

    let holdouts: &[(&str, &[&str], &[&str])] = &[
        (
            "彼は本当のことを知っているくせに、何も言っていないらしい。",
            &["他明明知道真的事情", "卻聽說什麼都沒有說"],
            &["他明明知道真的事情", "却听说什么都没有说"],
        ),
        (
            "もし今日雨が降ったとしても、電車で行くほうがいいと思う。",
            &["就算今天下雨", "我覺得坐電車去比較好"],
            &["就算今天下雨", "我觉得坐电车去比较好"],
        ),
        (
            "お金があるからといって、必ずしも幸せになれるとは限らないし、贅沢な生活を送りたいわけでもない。",
            &[
                "即便因為有金錢",
                "未必能變得幸福",
                "而且也並不是想過奢侈的生活",
            ],
            &[
                "即便因为有金钱",
                "未必能变得幸福",
                "而且也并不是想过奢侈的生活",
            ],
        ),
    ];

    for (src, hant_expected, hans_expected) in holdouts {
        let h_hant = analyze_and_translate(
            src,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(src, &h_hant);
        for pat in *hant_expected {
            assert!(
                h_hant.target_text.contains(pat),
                "holdout hant missing '{pat}' in: {}",
                h_hant.target_text
            );
        }

        let h_hans = analyze_and_translate(
            src,
            &TranslateOptions::new(TargetVariant::ZhHans),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(src, &h_hans);
        for pat in *hans_expected {
            assert!(
                h_hans.target_text.contains(pat),
                "holdout hans missing '{pat}' in: {}",
                h_hans.target_text
            );
        }
    }
}

#[test]
fn graded_register_casual_keigo_and_n1_editorial_with_holdouts() {
    let cases: &[(&str, &str, &str)] = &[
        (
            "今日は早く帰るつもりだったのに、急に仕事を頼まれて、結局いつもより遅くなっちゃった。",
            "本來打算今天早點回家，卻突然被交代工作，結果反而比平常更晚下班了。",
            "本来打算今天早点回家，却突然被交代工作，结果反而比平常更晚下班了。",
        ),
        (
            "別に怒っているわけじゃないけど、せめて一言くらい相談してくれてもよかったんじゃない？",
            "我也不是在生氣，只是你至少可以先跟我商量一下吧？",
            "我也不是在生气，只是你至少可以先跟我商量一下吧？",
        ),
        (
            "彼の意見が間違っているとは言い切れないが、だからといって全面的に賛成できるわけでもない。",
            "不能斷言他的意見有錯，但也不代表我能完全贊同。",
            "不能断言他的意见有错，但也不代表我能完全赞同。",
        ),
        (
            "へえ、そんなに忙しいんだ。それなのに毎日ゲームする時間だけはあるんだね。",
            "喔，原來你這麼忙啊。可是每天倒是都有時間玩遊戲呢。",
            "喔，原来你这么忙啊。可是每天倒是都有时间玩游戏呢。",
        ),
        (
            "昨日電話したとき、もう帰ったって言ってたから、てっきり家にいるものだと思ってた。",
            "昨天打電話時，對方說已經回去了，所以我一直以為他人在家。",
            "昨天打电话时，对方说已经回去了，所以我一直以为他人在家。",
        ),
        (
            "恐れ入りますが、先日ご案内いたしました件につきまして、改めてご確認いただけますと幸いです。",
            "不好意思，關於前幾天向您說明的事項，若能請您再次確認，將不勝感激。",
            "不好意思，关于前几天向您说明的事项，若能请您再次确认，将不胜感激。",
        ),
        (
            "仮に今回の計画が予定どおり進んだとしても、それが長期的な成功につながるとは必ずしも限らない。",
            "即使這次計畫如期進行，也不一定代表它能帶來長期的成功。",
            "即使这次计划如期进行，也不一定代表它能带来长期的成功。",
        ),
        (
            "人は失って初めてその大切さに気づくものだが、気づいたときにはすでに手遅れになっていることも少なくない。",
            "人往往直到失去之後才明白其珍貴，但等到察覺時，很多時候已經太遲了。",
            "人往往直到失去之后才明白其珍贵，但等到察觉时，很多时候已经太迟了。",
        ),
        (
            "彼が何も知らなかったとは考えにくいものの、だからといって意図的に事実を隠していたと断定することもできない。",
            "雖然很難相信他對此一無所知，但也不能因此斷定他刻意隱瞞事實。",
            "虽然很难相信他对此一无所知，但也不能因此断定他刻意隐瞒事实。",
        ),
        (
            "技術が進歩すればするほど、人間の仕事が楽になると考えられがちだ。しかし、実際には新しい技術が生み出す課題に対応するため、以前よりも複雑な判断を求められる場面が増えている。重要なのは、技術そのものの優劣ではなく、それをどのように活用し、どこまで人間が責任を持つべきかを見極めることだ。",
            "人們往往認為科技越進步，工作就越輕鬆。然而，實際上，為了應對新技術帶來的挑戰，需要做出比以前更加複雜判斷的情況也越來越多。重要的不是技術本身孰優孰劣，而是如何運用技術，以及如何判斷人類應當承擔多少責任。",
            "人们往往认为科技越进步，工作就越轻松。然而，实际上，为了应对新技术带来的挑战，需要做出比以前更加复杂判断的情况也越来越多。重要的不是技术本身孰优孰劣，而是如何运用技术，以及如何判断人类应当承担多少责任。",
        ),
    ];

    for (src, expected_hant, expected_hans) in cases {
        let hant = analyze_and_translate(
            src,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(src, &hant);
        assert_eq!(
            hant.target_text, *expected_hant,
            "zh-Hant mismatch for {src}"
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
        assert_eq!(
            hans.target_text, *expected_hans,
            "zh-Hans mismatch for {src}"
        );
    }

    let holdouts: &[(&str, &[&str], &[&str])] = &[
        (
            "明日は家で休むつもりだったのに、急に会議を入れられて、結局普段より忙しくなっちゃった。",
            &[
                "本來打算明天在家休息",
                "卻突然被安排會議",
                "結果反而比平常更忙了",
            ],
            &[
                "本来打算明天在家休息",
                "却突然被安排会议",
                "结果反而比平常更忙了",
            ],
        ),
        (
            "別に反対しているわけじゃないけど、せめて事前に連絡してくれてもよかったんじゃない？",
            &["我也不是在反對", "只是你至少可以事先跟我聯絡吧"],
            &["我也不是在反对", "只是你至少可以事先跟我联系吧"],
        ),
        (
            "恐れ入りますが、昨日ご送付いたしました資料につきまして、改めてご検討いただけますと幸いです。",
            &[
                "不好意思，關於昨天向您寄送的資料",
                "若能請您再次審閱，將不勝感激",
            ],
            &[
                "不好意思，关于昨天向您寄送的资料",
                "若能请您再次审阅，将不胜感激",
            ],
        ),
        (
            "彼女が理由を知らなかったとは考えにくいものの、だからといって故意に嘘をついていたと断定することもできない。",
            &["雖然很難相信她不知道理由", "但也不能因此斷定她故意說謊"],
            &["虽然很难相信她不知道理由", "但也不能因此断定她故意说谎"],
        ),
        (
            "便利になればなるほど、人は考えることをやめがちだ。大切なのは、道具そのものの性能ではなく、それをどのように使い、どこまで自分が判断するべきかを見極めることだ。",
            &[
                "越方便，人就越容易停止思考",
                "重要的不是工具本身的性能",
                "如何使用工具",
                "自己應當判斷",
            ],
            &[
                "越方便，人就越容易停止思考",
                "重要的不是工具本身的性能",
                "如何使用工具",
                "自己应当判断",
            ],
        ),
    ];

    for (src, hant_expected, hans_expected) in holdouts {
        let h_hant = analyze_and_translate(
            src,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(src, &h_hant);
        for pat in *hant_expected {
            assert!(
                h_hant.target_text.contains(pat),
                "holdout hant missing '{pat}' in: {}",
                h_hant.target_text
            );
        }

        let h_hans = analyze_and_translate(
            src,
            &TranslateOptions::new(TargetVariant::ZhHans),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(src, &h_hans);
        for pat in *hans_expected {
            assert!(
                h_hans.target_text.contains(pat),
                "holdout hans missing '{pat}' in: {}",
                h_hans.target_text
            );
        }
    }
}
