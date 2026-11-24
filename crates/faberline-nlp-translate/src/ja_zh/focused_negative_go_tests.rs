use faberline_nlp::doc::{analyze_document, model::Backend};

use crate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslatedDocument,
    TranslationGlossary, analyze_and_translate, doc_options_for_translate, translate_document,
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
fn a_focused_place_from_the_glossary_keeps_its_name_and_negation() {
    let mut glossary = TranslationGlossary::new();
    glossary.places.push(GlossaryEntry {
        canonical: "図書館".into(),
        surfaces: vec!["図書館".into()],
        targets: [(TargetVariant::ZhHant, "北館".into())].into(),
        ..GlossaryEntry::default()
    });
    for (source, expected) in [
        ("図書館へは行きませんでした。", "沒有去北館。"),
        ("図書館には行きませんでした。", "沒有去北館。"),
        ("図書館へは行かない。", "不去北館。"),
    ] {
        let result = translate(source, &glossary);
        assert_eq!(result.target_text, expected, "{source}");
        let [hit] = result.glossary_hits.as_slice() else {
            panic!("expected one place glossary hit");
        };
        assert_eq!(hit.canonical, "図書館");
        let span = hit.target_span;
        let alias: String = result
            .target_text
            .chars()
            .skip(span.start_char)
            .take(span.end_char - span.start_char)
            .collect();
        assert_eq!(alias, "北館");
    }
}

#[test]
fn a_glossary_category_alone_does_not_establish_a_physical_goal() {
    let activity = GlossaryEntry {
        canonical: "散歩".into(),
        surfaces: vec!["散歩".into()],
        targets: [(TargetVariant::ZhHant, "散步".into())].into(),
        ..GlossaryEntry::default()
    };
    let library_as_object = GlossaryEntry {
        canonical: "図書館".into(),
        surfaces: vec!["図書館".into()],
        targets: [(TargetVariant::ZhHant, "北館".into())].into(),
        ..GlossaryEntry::default()
    };
    let glossary = TranslationGlossary {
        places: vec![activity],
        objects: vec![library_as_object],
        ..TranslationGlossary::new()
    };
    assert_eq!(
        translate("散歩には行きませんでした。", &glossary).target_text,
        "沒有去到散步。"
    );
    assert_eq!(
        translate("図書館へは行きませんでした。", &glossary).target_text,
        "往北館沒有去。"
    );
}

#[test]
fn a_full_destination_match_keeps_case_focus_and_glossary_spans() {
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    for (source, term, kind, target) in [
        (
            "図書館へは行きませんでしたか。",
            "図書館へ",
            GlossaryKind::Place,
            "沒有去北館嗎？",
        ),
        (
            "図書館には行きませんでした。",
            "図書館には",
            GlossaryKind::DomainTerm,
            "沒有去北館。",
        ),
        (
            "図書館へは行かない。",
            "図書館へは",
            GlossaryKind::Place,
            "不去北館。",
        ),
    ] {
        let entry = GlossaryEntry {
            canonical: term.into(),
            surfaces: vec![term.into()],
            targets: [(TargetVariant::ZhHant, "北館".into())].into(),
            ..GlossaryEntry::default()
        };
        let mut glossary = TranslationGlossary::new();
        if kind == GlossaryKind::Place {
            glossary.places.push(entry);
        } else {
            glossary.domain_terms.push(entry);
        }
        let result = translate(source, &glossary);
        assert_eq!(result.target_text, target, "{source} / {term}");
        let [hit] = result.glossary_hits.as_slice() else {
            panic!("expected one glossary hit");
        };
        assert_eq!(
            hit.source_span.start_char..hit.source_span.end_char,
            0..term.chars().count()
        );
        assert_eq!(hit.kind, kind);
        assert_eq!(
            result
                .target_text
                .chars()
                .skip(hit.target_span.start_char)
                .take(hit.target_span.end_char - hit.target_span.start_char)
                .collect::<String>(),
            "北館"
        );
        let unhinted = analyze_document(
            source,
            &doc_options_for_translate(&options, &TranslationGlossary::new()),
        );
        let direct = translate_document(
            source,
            &unhinted,
            #[cfg(feature = "narrative")]
            None,
            &options,
            &glossary,
            None,
        )
        .unwrap();
        assert_eq!(result.target_text, direct.target_text);
        assert_eq!(result.glossary_hits, direct.glossary_hits);
    }
}

#[test]
fn a_complete_lock_does_not_make_affirmative_or_activity_goals_negative() {
    for (source, term, alias, expected) in [
        ("図書館へは行きました。", "図書館へは", "北館", "北館去了。"),
        (
            "散歩には行きませんでした。",
            "散歩には",
            "散步",
            "散步沒有去。",
        ),
    ] {
        let glossary = TranslationGlossary {
            places: vec![GlossaryEntry {
                canonical: term.into(),
                surfaces: vec![term.into()],
                targets: [(TargetVariant::ZhHant, alias.into())].into(),
                ..GlossaryEntry::default()
            }],
            ..TranslationGlossary::new()
        };
        assert_eq!(translate(source, &glossary).target_text, expected);
    }
}
