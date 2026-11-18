//! Multi-layer `ja -> zh-Hant / zh-Hans` deterministic translation integration tests.

use std::collections::BTreeMap;

use faberline_nlp::Span;
use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    GlossaryEntry, PronounClass, SpeechRegister, TRANSLATE_SCHEMA, TargetVariant, TranslateOptions,
    TranslatedDocument, TranslationGlossary, TranslationTrack, analyze_and_translate,
    doc_options_for_translate, render_bilingual_jsonl, sha256_hex,
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
fn homework_dasu_keeps_roles_and_physical_sources() {
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let glossary = TranslationGlossary::new();
    let translate = |source: &str| {
        let doc = analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        doc.target_text
    };

    let bank: serde_json::Value =
        serde_json::from_str(include_str!("data/ja_zh/homework_role_dev_v1.json")).unwrap();
    let items = bank["items"].as_array().unwrap();
    assert_eq!(items.len(), 3);
    for (id, expected) in [
        ("HWR-N5-01", "學生把作業交給老師。"),
        ("HWR-N5-02", "老師出作業給學生。"),
    ] {
        let row = items.iter().find(|row| row["id"] == id).unwrap();
        assert_eq!(translate(row["jp"].as_str().unwrap()), expected, "{id}");
    }

    let physical = items.iter().find(|row| row["id"] == "HWR-N5-03").unwrap();
    let actual = translate(physical["jp"].as_str().unwrap());
    assert!(
        matches!(
            actual.as_str(),
            "我從書桌抽屜拿出作業。" | "我從書桌的抽屜拿出作業。"
        ),
        "{actual}"
    );
    for (source, expected) in [
        ("先生は書類を出します。", "老師拿出書類。"),
        ("私は引き出しから鍵を出します。", "我從抽屜拿出鑰匙。"),
        ("宿題を出します。", "拿出作業。"),
        ("まだ宿題が終わっていない。", "作業還沒有結束。"),
    ] {
        assert_eq!(translate(source), expected, "{source}");
    }
}

#[test]
fn homework_dasu_preserves_modifiers_tense_and_transfer_direction() {
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let glossary = TranslationGlossary::new();
    for (source, expected) in [
        (
            "教室で生徒は先生に宿題を出しました。",
            "學生在教室把作業交給老師了。",
        ),
        (
            "先生は教室で生徒に宿題を出しました。",
            "老師在教室出作業給學生了。",
        ),
        (
            "先生は教室で生徒に宿題を出しませんでした。",
            "老師在教室沒有出作業給學生。",
        ),
        (
            "生徒は数学の宿題を先生に出しました。",
            "學生把數學的作業交給老師了。",
        ),
        (
            "教師は学生に今日の宿題を出しました。",
            "老師出今天的作業給學生了。",
        ),
        (
            "生徒は午後に先生に宿題を出しました。",
            "學生下午把作業交給老師了。",
        ),
        ("大学生は先生に宿題を出します。", "大學生把作業交給老師。"),
        (
            "生徒は鍵を引き出しから出しました。",
            "生徒從抽屜拿出了鑰匙。",
        ),
        ("生徒は先生に鍵を渡しました。", "生徒給老師遞給了鑰匙。"),
        ("生徒は宿題を出しました。", "生徒拿出了作業。"),
        ("先生に宿題を出しませんでした。", "給老師沒有拿出作業。"),
        (
            "会社員は先生に宿題を出しました。",
            "會社員給老師拿出了作業。",
        ),
    ] {
        let doc = analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        assert_eq!(doc.target_text, expected, "{source}");
    }
}

#[test]
fn homework_dasu_keeps_a_locked_homework_modifier_on_its_existing_path() {
    let source = "生徒は数学の宿題を先生に出しました。";
    let mut glossary = TranslationGlossary::new();
    glossary.objects.push(GlossaryEntry {
        canonical: "数学".to_owned(),
        surfaces: vec!["数学".to_owned()],
        targets: BTreeMap::from([(TargetVariant::ZhHant, "數學科".to_owned())]),
        ..GlossaryEntry::default()
    });
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let doc = analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
    assert_valid_ja_zh_doc(source, &doc);
    assert_eq!(doc.target_text, "生徒給老師拿出了數學科的作業。");
    assert_eq!(doc.glossary_hits.len(), 1);
}

#[test]
fn shinjitai_fallback_keeps_word_glosses_and_mixed_script_terms() {
    for (source, target, expected) in [
        ("私は製糸を見る。", TargetVariant::ZhHant, "我看製絲。"),
        ("私は塩を見る。", TargetVariant::ZhHant, "我看鹽。"),
        ("私は塩を見る。", TargetVariant::ZhHans, "我看盐。"),
        ("私は鉄を見る。", TargetVariant::ZhHant, "我看鐵。"),
        ("私は鉄を見る。", TargetVariant::ZhHans, "我看铁。"),
        ("私はガス灯を見る。", TargetVariant::ZhHant, "我看ガス灯。"),
        ("私はガス灯を見る。", TargetVariant::ZhHans, "我看ガス灯。"),
        (
            "私は駅で新聞を読む。",
            TargetVariant::ZhHant,
            "我在車站讀報紙。",
        ),
        (
            "私は駅で新聞を読む。",
            TargetVariant::ZhHans,
            "我在车站读报纸。",
        ),
        ("「私は塩を見る。」", TargetVariant::ZhHant, "「我看鹽。」"),
        ("「私は塩を見る。」", TargetVariant::ZhHans, "“我看盐。”"),
    ] {
        let doc = analyze_and_translate(
            source,
            &TranslateOptions::new(target),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        assert_eq!(doc.target_text, expected, "{source}");
    }
}

#[test]
fn shinjitai_fallback_preserves_locked_names_and_scalar_spans() {
    let source = "私は `𠮷塩` を見る、斎藤は塩を見る。";
    for (target, expected) in [
        (TargetVariant::ZhHant, "我看`𠮷塩`，斎藤看鹽。"),
        (TargetVariant::ZhHans, "我看`𠮷塩`，斎藤看盐。"),
    ] {
        let glossary = TranslationGlossary {
            actors: vec![GlossaryEntry {
                canonical: "斎藤".to_owned(),
                surfaces: vec!["斎藤".to_owned()],
                targets: BTreeMap::from([(target, "斎藤".to_owned())]),
                ..GlossaryEntry::default()
            }],
            ..TranslationGlossary::new()
        };
        let doc = analyze_and_translate(
            source,
            &TranslateOptions::new(target),
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        assert_eq!(doc.target_text, expected);
        assert_eq!(doc.glossary_hits.len(), 1);
        let hit = &doc.glossary_hits[0];
        assert_eq!(slice_chars(source, hit.source_span), "斎藤");
        assert_eq!(slice_chars(&doc.target_text, hit.target_span), "斎藤");
        let opaque = doc
            .opaque_spans
            .iter()
            .find(|op| slice_chars(source, op.source_span) == "`𠮷塩`")
            .unwrap();
        assert_eq!(
            slice_chars(&doc.target_text, opaque.target_span.unwrap()),
            "`𠮷塩`"
        );
        assert_eq!(
            doc.clause_alignments
                .iter()
                .map(|clause| slice_chars(&doc.target_text, clause.target_span))
                .collect::<String>(),
            doc.target_text
        );
        assert!(
            doc.clause_alignments
                .iter()
                .any(|clause| slice_chars(source, clause.source_span).contains("斎藤は塩を見る"))
        );
    }
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
fn existence_and_copula_negatives_keep_their_distinct_meanings() {
    let mut failures = Vec::new();
    for (source, hant, hans) in [
        ("約束はあります。", "有約定。", "有约定。"),
        ("約束はありません。", "沒有約定。", "没有约定。"),
        ("約束はありませんでした。", "沒有約定。", "没有约定。"),
        ("約束はない。", "沒有約定。", "没有约定。"),
        ("約束はなかった。", "沒有約定。", "没有约定。"),
        (
            "ここに椅子はありません。",
            "這裡沒有椅子。",
            "这里没有椅子。",
        ),
        (
            "ここに椅子がありませんでした。",
            "這裡沒有椅子。",
            "这里没有椅子。",
        ),
        ("私は学生ではありません。", "我不是學生。", "我不是学生。"),
        ("私は学生じゃありません。", "我不是學生。", "我不是学生。"),
        ("私は学生ではない。", "我不是學生。", "我不是学生。"),
        ("私は学生じゃない。", "我不是學生。", "我不是学生。"),
        (
            "私は学生ではありませんでした。",
            "我不是學生。",
            "我不是学生。",
        ),
        ("私は本があります。", "我有書。", "我有书。"),
        ("私は本がありません。", "我沒有書。", "我没有书。"),
        ("私はありません。", "我沒有。", "我没有。"),
        ("私はありませんでした。", "我沒有。", "我没有。"),
        ("先生はありません。", "老師沒有。", "老师没有。"),
        ("私の本はありません。", "沒有我的書。", "没有我的书。"),
        ("先生の本はありません。", "沒有老師的書。", "没有老师的书。"),
        ("ここに椅子があります。", "這裡有椅子。", "这里有椅子。"),
        ("私は学生です。", "我是學生。", "我是学生。"),
    ] {
        for (target, expected) in [(TargetVariant::ZhHant, hant), (TargetVariant::ZhHans, hans)] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            if doc.target_text != expected {
                failures.push(format!("{source}: {} != {expected}", doc.target_text));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn existence_and_copula_negatives_preserve_locations_quotes_and_locks() {
    for target in [TargetVariant::ZhHant, TargetVariant::ZhHans] {
        for (source, hant, hans) in [
            (
                "私は学校では本を読みます。",
                "我在學校讀書。",
                "我在学校读书。",
            ),
            (
                "学校では約束はありません。",
                "在學校沒有約定。",
                "在学校没有约定。",
            ),
            (
                "私は「学生ではありません」と言う。",
                "我說「不是學生」。",
                "我说“不是学生”。",
            ),
            (
                "私は `ではありません` を読む。",
                "我讀`ではありません`。",
                "我读`ではありません`。",
            ),
        ] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert_eq!(
                doc.target_text,
                if target == TargetVariant::ZhHant {
                    hant
                } else {
                    hans
                },
                "{source}"
            );
            for opaque in &doc.opaque_spans {
                assert_eq!(
                    slice_chars(source, opaque.source_span),
                    slice_chars(&doc.target_text, opaque.target_span.unwrap())
                );
            }
        }
        let source = "私は学生ではありません。";
        let glossary = TranslationGlossary {
            domain_terms: vec![GlossaryEntry {
                canonical: "学生".to_owned(),
                surfaces: vec!["学生".to_owned()],
                targets: BTreeMap::from([(target, "STUDENT".to_owned())]),
                ..GlossaryEntry::default()
            }],
            ..TranslationGlossary::new()
        };
        let doc = analyze_and_translate(
            source,
            &TranslateOptions::new(target),
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        assert_eq!(doc.target_text, "我不是STUDENT。");
        assert_eq!(doc.glossary_hits.len(), 1);
        let hit = &doc.glossary_hits[0];
        assert_eq!(slice_chars(source, hit.source_span), "学生");
        assert_eq!(slice_chars(&doc.target_text, hit.target_span), "STUDENT");
    }
}

#[test]
fn keigo_descriptive_forms_keep_the_action_tense_and_polarity() {
    let mut failures = Vec::new();
    for (source, hant, hans) in [
        ("先生は本をお読みになる。", "老師讀書。", "老师读书。"),
        ("先生は本をお読みになります。", "老師讀書。", "老师读书。"),
        ("先生は本をお読みになった。", "老師讀了書。", "老师读了书。"),
        (
            "先生は本をお読みになりました。",
            "老師讀了書。",
            "老师读了书。",
        ),
        (
            "先生は本をお読みになりません。",
            "老師不讀書。",
            "老师不读书。",
        ),
        (
            "先生は本をお読みにならない。",
            "老師不讀書。",
            "老师不读书。",
        ),
        (
            "先生は本をお読みになりませんでした。",
            "老師沒有讀書。",
            "老师没有读书。",
        ),
        ("私は先生をお待ちする。", "我等待老師。", "我等待老师。"),
        ("私は先生をお待ちします。", "我等待老師。", "我等待老师。"),
        ("私は先生をお待ちした。", "我等待了老師。", "我等待了老师。"),
        (
            "私は先生をお待ちしました。",
            "我等待了老師。",
            "我等待了老师。",
        ),
        (
            "私は先生をお待ちしない。",
            "我不等待老師。",
            "我不等待老师。",
        ),
        (
            "私は先生をお待ちしませんでした。",
            "我沒有等待老師。",
            "我没有等待老师。",
        ),
        (
            "私は先生に本を読んでいただく。",
            "老師幫我讀書。",
            "老师帮我读书。",
        ),
        (
            "私は先生に本を読んでいただきます。",
            "老師幫我讀書。",
            "老师帮我读书。",
        ),
        (
            "私は先生に本を読んでいただいた。",
            "老師幫我讀了書。",
            "老师帮我读了书。",
        ),
        (
            "私は先生に本を読んでいただきました。",
            "老師幫我讀了書。",
            "老师帮我读了书。",
        ),
        (
            "私は先生に本を読んでいただきません。",
            "老師不幫我讀書。",
            "老师不帮我读书。",
        ),
        (
            "先生は本を読んでくださる。",
            "老師幫我讀書。",
            "老师帮我读书。",
        ),
        (
            "先生は本を読んでくださいます。",
            "老師幫我讀書。",
            "老师帮我读书。",
        ),
        (
            "先生は本を読んでくださった。",
            "老師幫我讀了書。",
            "老师帮我读了书。",
        ),
        (
            "先生は本を読んでくださいました。",
            "老師幫我讀了書。",
            "老师帮我读了书。",
        ),
        (
            "先生は本を読んでくださいません。",
            "老師不幫我讀書。",
            "老师不帮我读书。",
        ),
        (
            "先生は本を読んでくださいませんでした。",
            "老師沒有幫我讀書。",
            "老师没有帮我读书。",
        ),
        (
            "私は先生に本を読んでいただきませんでした。",
            "老師沒有幫我讀書。",
            "老师没有帮我读书。",
        ),
    ] {
        for (target, expected) in [(TargetVariant::ZhHant, hant), (TargetVariant::ZhHans, hans)] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            if doc.target_text != expected {
                failures.push(format!("{source}: {} != {expected}", doc.target_text));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn keigo_lending_keeps_the_beneficiary_as_the_borrower() {
    for (source, hant, hans) in [
        (
            "先生は本を貸してくださる。",
            "老師借給我書。",
            "老师借给我书。",
        ),
        (
            "先生は本を貸してくださった。",
            "老師借給了我書。",
            "老师借给了我书。",
        ),
        (
            "私は先生に本を貸していただきました。",
            "老師借給了我書。",
            "老师借给了我书。",
        ),
        (
            "先生は彼女に本を貸してくださる。",
            "老師借給她書。",
            "老师借给她书。",
        ),
        (
            "先生は本を貸してくれた。",
            "老師借給了我書。",
            "老师借给了我书。",
        ),
        (
            "私は先生に本を貸してもらう。",
            "老師借給我書。",
            "老师借给我书。",
        ),
    ] {
        for (target, expected) in [(TargetVariant::ZhHant, hant), (TargetVariant::ZhHans, hans)] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert_eq!(doc.target_text, expected, "{source}");
        }
    }
}

#[test]
fn keigo_wrappers_preserve_other_verbs_and_explicit_beneficiaries() {
    for (source, plain) in [
        ("先生は手紙をお書きになる。", "先生は手紙を書く。"),
        ("私は本をお持ちする。", "私は本を持つ。"),
        ("私は手紙をお書きします。", "私は手紙を書く。"),
        (
            "先生は私に手紙を書いてくださる。",
            "先生は私に手紙を書いてくれる。",
        ),
        (
            "私は先生に手紙を書いていただいた。",
            "私は先生に手紙を書いてもらった。",
        ),
        (
            "先生は彼女に本を読んでくださいます。",
            "先生は彼女に本を読んでくれる。",
        ),
        (
            "私は先生から本を読んでいただきます。",
            "私は先生から本を読んでもらう。",
        ),
    ] {
        for target in [TargetVariant::ZhHant, TargetVariant::ZhHans] {
            let translate = |text| {
                analyze_and_translate(
                    text,
                    &TranslateOptions::new(target),
                    &TranslationGlossary::new(),
                    None,
                    Backend::Onnx,
                )
                .unwrap()
            };
            let doc = translate(source);
            assert_valid_ja_zh_doc(source, &doc);
            assert_eq!(doc.target_text, translate(plain).target_text, "{source}");
            assert!(
                !doc.target_text.contains("您"),
                "{source}: {}",
                doc.target_text
            );
        }
    }
}

#[test]
fn keigo_does_not_rewrite_lexical_verbs_requests_or_complex_controls() {
    for target in [TargetVariant::ZhHant, TargetVariant::ZhHans] {
        let translate = |text| {
            analyze_and_translate(
                text,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap()
        };
        for source in [
            "先生は本を読んでください。",
            "先生は本を読んでくださいませ。",
            "先生は本を読んでくださいまし。",
            "先生は本を読まないでください。",
        ] {
            let doc = translate(source);
            assert_valid_ja_zh_doc(source, &doc);
            assert!(
                doc.target_text.contains('請') || doc.target_text.contains('请'),
                "{}",
                doc.target_text
            );
            assert!(!doc.target_text.contains("幫我") && !doc.target_text.contains("帮我"));
        }
        let meal = translate("いただきます。");
        assert_eq!(
            meal.target_text,
            if target == TargetVariant::ZhHant {
                "我要開動了。"
            } else {
                "我要开动了。"
            }
        );
        for source in [
            "私は座って寿司をいただいた。",
            "私は本をいただく。",
            "先生は私に本をくださる。",
            "私は先生に本を読ませていただく。",
            "私は先生に本を読んでいただきたい。",
        ] {
            let doc = translate(source);
            assert_valid_ja_zh_doc(source, &doc);
            assert!(
                doc.target_text.starts_with(if source.starts_with("私は") {
                    "我"
                } else if target == TargetVariant::ZhHant {
                    "老師"
                } else {
                    "老师"
                }),
                "{source}: {}",
                doc.target_text
            );
            assert!(!doc.target_text.contains("幫我") && !doc.target_text.contains("帮我"));
        }
        let at_home = translate("家でいただきます。");
        assert!(
            at_home
                .target_text
                .contains(if target == TargetVariant::ZhHant {
                    "我要開動了"
                } else {
                    "我要开动了"
                }),
            "{}",
            at_home.target_text
        );
        for source in ["お金になる。", "お茶にする。", "先生になった。"] {
            let doc = translate(source);
            assert_valid_ja_zh_doc(source, &doc);
            assert!(!doc.target_text.contains("幫我") && !doc.target_text.contains("帮我"));
            if source.ends_with("なる。") || source.ends_with("なった。") {
                assert!(
                    doc.target_text.contains("變成") || doc.target_text.contains("变成"),
                    "{}",
                    doc.target_text
                );
            }
        }
    }
}

#[test]
fn keigo_keeps_glossary_opaque_and_quote_spans() {
    let source = "アリスは `お読みになる` をお読みになりました、私は先生をお待ちします。";
    for target in [TargetVariant::ZhHant, TargetVariant::ZhHans] {
        let mut glossary = actor_glossary("Alice");
        glossary.actors[0]
            .targets
            .insert(TargetVariant::ZhHans, "Alice".to_owned());
        let doc = analyze_and_translate(
            source,
            &TranslateOptions::new(target),
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        for hit in &doc.glossary_hits {
            assert_eq!(slice_chars(&doc.target_text, hit.target_span), "Alice");
        }
        assert_eq!(doc.glossary_hits.len(), 1);
        let code = doc
            .opaque_spans
            .iter()
            .find(|span| slice_chars(source, span.source_span) == "`お読みになる`")
            .unwrap();
        assert_eq!(
            slice_chars(&doc.target_text, code.target_span.unwrap()),
            "`お読みになる`"
        );
        assert_eq!(
            doc.clause_alignments
                .iter()
                .map(|clause| slice_chars(&doc.target_text, clause.target_span))
                .collect::<String>(),
            doc.target_text
        );
        for (locked_source, term) in [
            ("先生は本をお読みになる。", "読み"),
            ("私は先生に本を読んでいただく。", "いただく"),
        ] {
            let mut glossary = TranslationGlossary::new();
            glossary.domain_terms.push(GlossaryEntry {
                canonical: term.to_owned(),
                surfaces: vec![term.to_owned()],
                targets: BTreeMap::from([(target, "LOCKED".to_owned())]),
                ..GlossaryEntry::default()
            });
            let doc = analyze_and_translate(
                locked_source,
                &TranslateOptions::new(target),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(locked_source, &doc);
            assert_eq!(doc.glossary_hits.len(), 1);
            let hit = &doc.glossary_hits[0];
            assert_eq!(slice_chars(locked_source, hit.source_span), term);
            assert_eq!(slice_chars(&doc.target_text, hit.target_span), "LOCKED");
        }
        let quoted = "「私は先生をお待ちします」と言う。";
        let doc = analyze_and_translate(
            quoted,
            &TranslateOptions::new(target),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(quoted, &doc);
        assert!(
            doc.target_text
                .contains(if target == TargetVariant::ZhHant {
                    "我等待老師"
                } else {
                    "我等待老师"
                }),
            "{}",
            doc.target_text
        );
    }
}

#[test]
fn polite_negative_endings_keep_polarity_and_past_tense() {
    let mut failures = Vec::new();
    for (source, hant, hans) in [
        ("寝ません。", "不睡覺。", "不睡觉。"),
        ("寝ませんでした。", "沒有睡覺。", "没有睡觉。"),
        ("寝ます。", "睡覺。", "睡觉。"),
        ("私は本を読みません。", "我不讀書。", "我不读书。"),
        ("私は本を読みませんでした。", "我沒有讀書。", "我没有读书。"),
        ("私は本を読みます。", "我讀書。", "我读书。"),
        ("私は本を読みました。", "我讀了書。", "我读了书。"),
    ] {
        for (target, expected) in [(TargetVariant::ZhHant, hant), (TargetVariant::ZhHans, hans)] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            if doc.target_text != expected {
                failures.push(format!("{source}: {} != {expected}", doc.target_text));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn polite_negative_is_local_to_the_following_temporal_clause() {
    for (source, hant, hans) in [
        ("私は本を読んでから寝ません。", "不睡覺", "不睡觉"),
        ("私は本を読んでから寝ませんでした。", "沒有睡覺", "没有睡觉"),
    ] {
        for (target, expected, after) in [
            (TargetVariant::ZhHant, hant, '後'),
            (TargetVariant::ZhHans, hans, '后'),
        ] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            let (before, following) = doc
                .target_text
                .split_once(after)
                .expect("temporal boundary");
            assert!(!before.contains(['不', '沒', '没']), "{}", doc.target_text);
            assert!(following.contains(expected), "{}", doc.target_text);
            assert!(!following.contains('了'), "{}", doc.target_text);
        }
    }
}

#[test]
fn quoted_and_opaque_negative_syllables_do_not_negate_positive_actions() {
    for source in [
        "「ん」と言います。",
        "私は `ません` を読みます。",
        "彼は本を読むんです。",
    ] {
        for target in [TargetVariant::ZhHant, TargetVariant::ZhHans] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert!(
                !doc.target_text.contains(['不', '沒', '没']),
                "{source}: {}",
                doc.target_text
            );
            if source.contains("`ません`") {
                let code = doc
                    .opaque_spans
                    .iter()
                    .find(|record| slice_chars(source, record.source_span) == "`ません`")
                    .unwrap();
                assert_eq!(
                    slice_chars(&doc.target_text, code.target_span.unwrap()),
                    "`ません`"
                );
            }
        }
    }
}

#[test]
fn te_kara_keeps_temporal_order_separate_from_cause_and_source() {
    for target in [TargetVariant::ZhHant, TargetVariant::ZhHans] {
        let after = if target == TargetVariant::ZhHant {
            '後'
        } else {
            '后'
        };
        for source in [
            "私は本を読んでから寝ます。",
            "私は本を読んでから、寝ます。",
            "私は仕事をしてから帰った。",
            "先生に本を読んでもらってから返します。",
        ] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert!(
                doc.target_text.contains(after),
                "{source}: {}",
                doc.target_text
            );
            assert!(
                !doc.target_text.contains("因"),
                "{source}: {}",
                doc.target_text
            );
            if source.contains("寝ます") {
                let read = if target == TargetVariant::ZhHant {
                    '讀'
                } else {
                    '读'
                };
                let read_at = doc.target_text.find(read).expect("reading action");
                let after_at = doc.target_text.find(after).expect("temporal relation");
                let sleep_at = doc.target_text.find('睡').expect("following action");
                assert!(
                    read_at < after_at && after_at < sleep_at,
                    "{}",
                    doc.target_text
                );
            }
        }
        for source in [
            "私は本を読んでいるから家にいます。",
            "私は本を読んでいないから家にいます。",
        ] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert!(doc.target_text.contains("因"), "{}", doc.target_text);
            assert!(!doc.target_text.contains(after), "{}", doc.target_text);
        }
        let source = "私は本を読んでから寝ない。";
        let doc = analyze_and_translate(
            source,
            &TranslateOptions::new(target),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        let (before, following) = doc
            .target_text
            .split_once(after)
            .expect("temporal boundary");
        assert!(!before.contains('不'), "{}", doc.target_text);
        assert!(following.contains('不'), "{}", doc.target_text);
    }
}

#[test]
fn showing_for_someone_uses_one_recipient_marker() {
    for (source, hant, hans) in [
        (
            "私は彼に写真を見せてあげた。",
            "我給他看了照片。",
            "我给他看了照片。",
        ),
        (
            "彼が私に写真を見せてくれた。",
            "他給我看了照片。",
            "他给我看了照片。",
        ),
        (
            "私は彼に写真を見せてもらった。",
            "他給我看了照片。",
            "他给我看了照片。",
        ),
        (
            "私は彼に写真を見せてあげます。",
            "我給他看照片。",
            "我给他看照片。",
        ),
        ("写真を見せてあげた。", "展示了照片。", "展示了照片。"),
        ("写真を見せてくれた。", "給我看了照片。", "给我看了照片。"),
        (
            "私は彼に写真を見せてあげなかった。",
            "我沒有給他看照片。",
            "我没有给他看照片。",
        ),
        (
            "私は彼に本を読ませてあげた。",
            "我讓他讀了書。",
            "我让他读了书。",
        ),
    ] {
        for (target, expected) in [(TargetVariant::ZhHant, hant), (TargetVariant::ZhHans, hans)] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert_eq!(doc.target_text, expected, "{source}");
        }
    }
}

#[test]
fn showing_keeps_additional_predicate_content_on_the_conservative_path() {
    let source = "私は彼に写真を綺麗に見せてあげた。";
    for (target, modifier) in [
        (TargetVariant::ZhHant, "美麗"),
        (TargetVariant::ZhHans, "美丽"),
    ] {
        let doc = analyze_and_translate(
            source,
            &TranslateOptions::new(target),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        assert!(doc.target_text.contains(modifier), "{}", doc.target_text);
    }
}

#[test]
fn kureru_defaults_toward_the_speaker_but_keeps_explicit_recipients() {
    for (source, hant, hans) in [
        ("本をくれた。", "給了我書。", "给了我书。"),
        ("彼が本をくれた。", "他給了我書。", "他给了我书。"),
        ("彼が本を読んでくれた。", "他幫我讀了書。", "他帮我读了书。"),
        (
            "彼が本を読んでくれました。",
            "他幫我讀了書。",
            "他帮我读了书。",
        ),
        (
            "彼が妹に本を読んでくれた。",
            "他幫妹妹讀了書。",
            "他帮妹妹读了书。",
        ),
        ("彼が本を読んであげた。", "他幫忙讀了書。", "他帮忙读了书。"),
    ] {
        for (target, expected) in [(TargetVariant::ZhHant, hant), (TargetVariant::ZhHans, hans)] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert_eq!(doc.target_text, expected, "{source}");
        }
    }
}

#[test]
fn kureru_keeps_competing_recipients_on_the_conservative_path() {
    for source in [
        "彼が妹へ本を読んでくれた。",
        "彼が妹に母に本を読んでくれた。",
    ] {
        for target in [TargetVariant::ZhHant, TargetVariant::ZhHans] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert!(
                !doc.target_text.contains('我'),
                "{source}: {}",
                doc.target_text
            );
        }
    }
}

#[cfg(feature = "narrative")]
#[test]
fn kureru_in_direct_speech_uses_first_person_instead_of_the_speaker_name() {
    let source = "「彼が本を読んでくれた。」";
    let glossary = TranslationGlossary::new();
    for (target, expected) in [
        (TargetVariant::ZhHant, "他幫我讀了書"),
        (TargetVariant::ZhHans, "他帮我读了书"),
    ] {
        let options = TranslateOptions::new(target);
        let doc = faberline_nlp::doc::analyze_document(
            source,
            &faberline_nlp_translate::doc_options_for_translate(&options, &glossary),
        );
        let mut narrative = faberline_nlp_narrative::analyze_chapter_narrative(
            1,
            None,
            source,
            &doc,
            &glossary.to_work_vocabulary(),
        )
        .unwrap();
        narrative.dialogues[0].speaker_canonical = Some("named_speaker".to_owned());
        let translated = faberline_nlp_translate::translate_document(
            source,
            &doc,
            Some(&narrative),
            &options,
            &glossary,
            None,
        )
        .unwrap();
        assert_valid_ja_zh_doc(source, &translated);
        assert_eq!(
            translated.sentences[0].conditioning.speaker.as_deref(),
            Some("named_speaker")
        );
        assert!(
            translated.target_text.contains(expected),
            "{}",
            translated.target_text
        );
        assert!(!translated.target_text.contains("named_speaker"));
    }
}

#[test]
fn source_kara_before_a_copula_keeps_the_start_point() {
    for source in [
        "授業は午前九時からです。",
        "仕事は明日からです。",
        "発表は六月一日の午後三時からです。",
    ] {
        for (target, start) in [(TargetVariant::ZhHant, '從'), (TargetVariant::ZhHans, '从')] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert!(
                doc.target_text.contains(start),
                "{source}: {}",
                doc.target_text
            );
        }
    }
}

#[test]
fn giving_and_receiving_keep_explicit_transfer_roles() {
    let cases = [
        ("私は彼に本をあげた。", "我給了他書。", "我给了他书。"),
        ("彼が私に本をくれた。", "他給了我書。", "他给了我书。"),
        (
            "私は彼に本をもらった。",
            "我從他收到了書。",
            "我从他收到了书。",
        ),
        (
            "私は彼から本をもらった。",
            "我從他收到了書。",
            "我从他收到了书。",
        ),
        (
            "私は彼に本をもらいました。",
            "我從他收到了書。",
            "我从他收到了书。",
        ),
        ("本をもらった。", "收到了書。", "收到了书。"),
        ("彼が本をくれた。", "他給了我書。", "他给了我书。"),
        (
            "私は彼に本をもらわなかった。",
            "我沒有從他收到書。",
            "我没有从他收到书。",
        ),
    ];
    for (source, expected_hant, expected_hans) in cases {
        for (target, expected) in [
            (TargetVariant::ZhHant, expected_hant),
            (TargetVariant::ZhHans, expected_hans),
        ] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert_eq!(doc.target_text, expected, "{source}");
        }
    }
}

#[test]
fn benefactive_actions_keep_actor_and_beneficiary_distinct() {
    let cases = [
        (
            "彼が私に本を読んでくれた。",
            "他幫我讀了書。",
            "他帮我读了书。",
        ),
        (
            "彼が妹に本を読んでくれた。",
            "他幫妹妹讀了書。",
            "他帮妹妹读了书。",
        ),
        (
            "私は妹に本を読んであげた。",
            "我幫妹妹讀了書。",
            "我帮妹妹读了书。",
        ),
        (
            "私は彼に本を読んでもらった。",
            "他幫我讀了書。",
            "他帮我读了书。",
        ),
        (
            "彼女は弟に手紙を書いてもらった。",
            "弟弟幫她寫了信。",
            "弟弟帮她写了信。",
        ),
        (
            "私は彼に本を読んでもらいました。",
            "他幫我讀了書。",
            "他帮我读了书。",
        ),
        (
            "彼に本を読んでもらった。",
            "他幫忙讀了書。",
            "他帮忙读了书。",
        ),
        (
            "私は本を読んでもらった。",
            "有人幫我讀了書。",
            "有人帮我读了书。",
        ),
        ("彼が本を読んでくれた。", "他幫我讀了書。", "他帮我读了书。"),
    ];
    for (source, expected_hant, expected_hans) in cases {
        for (target, expected) in [
            (TargetVariant::ZhHant, expected_hant),
            (TargetVariant::ZhHans, expected_hans),
        ] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert_eq!(doc.target_text, expected, "{source}");
        }
    }
}

#[test]
fn transfer_roles_preserve_glossary_names_and_opaque_objects() {
    let source = "アリスはボブに `x` をもらった。";
    let mut glossary = TranslationGlossary::new();
    for (surface, hant, hans) in [("アリス", "愛麗絲", "爱丽丝"), ("ボブ", "鮑伯", "鲍伯")]
    {
        glossary.actors.push(GlossaryEntry {
            canonical: surface.to_owned(),
            surfaces: vec![surface.to_owned()],
            targets: BTreeMap::from([
                (TargetVariant::ZhHant, hant.to_owned()),
                (TargetVariant::ZhHans, hans.to_owned()),
            ]),
            ..GlossaryEntry::default()
        });
    }
    for (target, expected) in [
        (TargetVariant::ZhHant, "愛麗絲從鮑伯收到了`x`。"),
        (TargetVariant::ZhHans, "爱丽丝从鲍伯收到了`x`。"),
    ] {
        let doc = analyze_and_translate(
            source,
            &TranslateOptions::new(target),
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(source, &doc);
        assert_eq!(doc.target_text, expected);
        assert_eq!(doc.glossary_hits.len(), 2);
        for hit in &doc.glossary_hits {
            let entry = glossary
                .actors
                .iter()
                .find(|e| e.canonical == hit.canonical)
                .unwrap();
            assert_eq!(
                slice_chars(&doc.target_text, hit.target_span),
                entry.targets[&target]
            );
        }
        let code = doc
            .opaque_spans
            .iter()
            .find(|record| slice_chars(source, record.source_span) == "`x`")
            .unwrap();
        assert_eq!(
            slice_chars(&doc.target_text, code.target_span.unwrap()),
            "`x`"
        );
    }
}

#[test]
fn benefactive_actions_keep_the_named_actor_before_verb_role_rewrites() {
    for source in [
        "私は彼に来てもらった。",
        "私は彼に学校へ行ってもらった。",
        "私は彼に泣いてもらった。",
        "私は彼に気づいてもらった。",
    ] {
        for (target, benefit) in [
            (TargetVariant::ZhHant, "幫我"),
            (TargetVariant::ZhHans, "帮我"),
        ] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert!(
                doc.target_text.starts_with('他'),
                "{source}: {}",
                doc.target_text
            );
            assert!(
                doc.target_text.contains(benefit),
                "{source}: {}",
                doc.target_text
            );
            assert_eq!(
                doc.target_text.matches('他').count(),
                1,
                "{source}: {}",
                doc.target_text
            );
        }
    }
}

#[test]
fn ambiguous_ageru_does_not_assert_giving_without_a_recipient() {
    for source in [
        "私は声をあげた。",
        "私は声をあげる。",
        "私は手をあげた。",
        "私は窓をあげた。",
    ] {
        for target in [TargetVariant::ZhHant, TargetVariant::ZhHans] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert!(
                !doc.target_text.contains(['給', '给']),
                "{source}: {}",
                doc.target_text
            );
        }
    }
}

#[test]
fn complex_benefactive_controls_keep_the_original_subject() {
    for (source, subject) in [
        ("私は彼に手紙を書いてもらいたい。", "我"),
        ("彼女は弟に手紙を書いてもらいたかった。", "她"),
        ("私は先生に発表させてもらった。", "我"),
        ("私は彼から母に本を読んでもらった。", "我"),
    ] {
        for target in [TargetVariant::ZhHant, TargetVariant::ZhHans] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_valid_ja_zh_doc(source, &doc);
            assert!(
                doc.target_text.starts_with(subject),
                "{source}: {}",
                doc.target_text
            );
        }
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
fn direct_reporting_keeps_adopted_roles_and_judgment_scope() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("data/ja_zh/direct_quote_speaker_dev_v1.json")).unwrap();
    let cases = cases["items"].as_array().unwrap();
    assert_eq!(cases.len(), 3);

    for case in cases {
        let src = case["jp"].as_str().unwrap();
        let roles = &case["expected_roles"];
        let speaker = roles["speaker"].as_str().unwrap();
        let listener = roles["listener"].as_str().unwrap();
        let actor = roles["quoted_actor"].as_str().unwrap();
        let narrated_judgment = roles["judgment_source"] == "narrator";
        let doc = analyze_and_translate(
            src,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(src, &doc);
        let (prefix, rest) = doc.target_text.split_once('「').unwrap();
        let (quote, after) = rest.split_once('」').unwrap();
        assert_eq!(prefix, format!("{speaker}告訴{listener}："), "{src}");
        assert!(quote.contains(actor), "{src}: {quote}");
        assert!(quote.ends_with('。'), "{src}: {quote}");
        assert!(!after.starts_with('。'), "{src}: {after}");
        assert_eq!(quote.contains("正式"), !narrated_judgment, "{src}");
        assert_eq!(after.contains("正式"), narrated_judgment, "{src}");
        assert!(!after.contains("告訴"), "{src}: {after}");
        assert_eq!(doc.sentences.len(), 2, "{src}");
    }
}

#[test]
fn direct_reporting_places_attached_roles_and_outer_time_before_the_quote() {
    let translate = |src| {
        let doc = analyze_and_translate(
            src,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_ja_zh_doc(src, &doc);
        doc.target_text
    };
    for src in [
        "先生は学生に「私は家に帰ります」と伝えた。",
        "「私は家に帰ります」と先生は学生に伝えた。",
        "先生は「私は家に帰ります」と学生に伝えた。",
    ] {
        assert_eq!(translate(src), "老師告訴學生：「我回到家。」", "{src}");
    }
    assert_eq!(
        translate("先生は朝、学生に「私は家に帰ります」と伝えた。"),
        "老師早上告訴學生：「我回到家。」"
    );
    let inner_time = translate("先生は学生に「私は朝、家に帰ります」と伝えた。");
    let quote = inner_time
        .strip_prefix("老師告訴學生：「")
        .and_then(|text| text.strip_suffix("。」"))
        .unwrap();
    assert!(quote.contains("早上"), "{inner_time}");
}

#[test]
fn direct_reporting_keeps_terminal_marks_in_their_own_scope() {
    for (quoted, report_end, expected_end) in [
        ("明日また来ます。", '。', "。」"),
        ("明日また来ます？", '。', "？」"),
        ("明日また来ます！", '。', "！」"),
        ("資料を確認します。明日また来ます？", '。', "？」"),
        ("明日また来ます", '？', "」？"),
        ("明日また来ます", '！', "」！"),
    ] {
        let src = format!("佐藤は田中に「{quoted}」と伝えた{report_end}");
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let doc = analyze_and_translate(&src, &options, &glossary, None, Backend::Onnx).unwrap();
        assert_valid_ja_zh_doc(&src, &doc);
        let translated = &doc.target_text;
        assert!(
            translated.starts_with("佐藤告訴田中：「"),
            "{src}: {translated}"
        );
        assert!(translated.ends_with(expected_end), "{src}: {translated}");
        assert_eq!(translated.matches("告訴").count(), 1, "{src}: {translated}");
        assert!(!translated.contains("。。"), "{src}: {translated}");
        if report_end == '。' && quoted.ends_with(['。', '？', '！']) {
            let source_doc = faberline_nlp::doc::analyze_document(
                &src,
                &doc_options_for_translate(&options, &glossary),
            );
            assert!(source_doc.segmentation.sentences.len() > 1, "{src}");
            assert_eq!(doc.sentences.len(), 1, "{src}");
            let jsonl = render_bilingual_jsonl(&doc, &src).unwrap();
            let row: serde_json::Value = serde_json::from_str(&jsonl).unwrap();
            assert_eq!(row["source_text"], src);
            assert_eq!(row["target_text"], *translated);
            assert_eq!(slice_chars(&src, doc.sentences[0].source_span), src);
        }
    }
}

#[test]
fn direct_reporting_keeps_following_alignment_after_consuming_a_tail() {
    let src = "佐藤は田中に「明日また来ます？」と伝えた。田中は学生に資料を見せた。";
    let doc = analyze_and_translate(
        src,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_ja_zh_doc(src, &doc);
    assert_eq!(doc.sentences.len(), 2);
    assert_eq!(
        slice_chars(src, doc.sentences[0].source_span),
        "佐藤は田中に「明日また来ます？」と伝えた。"
    );
    assert_eq!(
        slice_chars(src, doc.sentences[1].source_span),
        "田中は学生に資料を見せた。"
    );
    assert!(
        doc.clause_alignments
            .iter()
            .any(|clause| clause.sentence == 1)
    );
}

#[test]
fn direct_reporting_preserves_locked_text_across_multiple_sentences() {
    let src = "佐藤は田中に「資料を読みます。`x」y`を見ます。窓を開けます」と伝えた。";
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let glossary = TranslationGlossary::new();
    let doc = analyze_and_translate(src, &options, &glossary, None, Backend::Onnx).unwrap();
    assert_valid_ja_zh_doc(src, &doc);
    assert!(
        doc.target_text.starts_with("佐藤告訴田中：「"),
        "{}",
        doc.target_text
    );
    assert_eq!(doc.sentences.len(), 3);
    assert_eq!(doc.opaque_spans.len(), 1);
    let locked = &doc.opaque_spans[0];
    assert_eq!(slice_chars(src, locked.source_span), "`x」y`");
    assert_eq!(
        slice_chars(&doc.target_text, locked.target_span.unwrap()),
        "`x」y`"
    );
    assert_eq!(doc.target_text.matches("告訴").count(), 1);

    for src in [
        "佐藤は田中に「資料を読みます」と伝えなかった。",
        "佐藤は田中に「資料を読みます」と言った。",
        "佐藤は田中に「資料を読みます。",
        "佐藤は田中に資料を読むように伝えた。",
    ] {
        let doc = analyze_and_translate(src, &options, &glossary, None, Backend::Onnx).unwrap();
        assert_valid_ja_zh_doc(src, &doc);
        assert!(!doc.target_text.starts_with("佐藤告訴田中："), "{src}");
    }
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
