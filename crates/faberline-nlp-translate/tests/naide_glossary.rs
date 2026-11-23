use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
    analyze_and_translate, doc_options_for_translate, translate_document,
};

fn check(source: &str, object: Option<(&str, &str)>, place: Option<(&str, &str)>, expected: &str) {
    check_locks(source, object, place, &[], expected);
}

fn check_locks(
    source: &str,
    object: Option<(&str, &str)>,
    place: Option<(&str, &str)>,
    domain: &[(&str, &str)],
    expected: &str,
) {
    check_locks_in_both_entries(source, object, place, domain, [expected; 2]);
}

fn check_locks_in_both_entries(
    source: &str,
    object: Option<(&str, &str)>,
    place: Option<(&str, &str)>,
    domain: &[(&str, &str)],
    expected: [&str; 2],
) {
    let entry = |(term, target): (&str, &str)| GlossaryEntry {
        canonical: term.to_owned(),
        surfaces: vec![term.to_owned()],
        targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
        ..Default::default()
    };
    let mut glossary = TranslationGlossary::new();
    glossary.objects.extend(object.map(entry));
    glossary.places.extend(place.map(entry));
    glossary
        .domain_terms
        .extend(domain.iter().copied().map(entry));
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let analyzed = analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
    let doc = faberline_nlp::doc::analyze_document(
        source,
        &doc_options_for_translate(&options, &TranslationGlossary::new()),
    );
    let direct = translate_document(
        source,
        &doc,
        #[cfg(feature = "narrative")]
        None,
        &options,
        &glossary,
        None,
    )
    .unwrap();
    for ((entry, result), expected) in [
        ("analyze and translate", analyzed),
        ("preanalyzed input", direct),
    ]
    .into_iter()
    .zip(expected)
    {
        assert_eq!(result.target_text, expected, "{entry}");
        let expected_hits = object
            .map(|pair| (GlossaryKind::Object, pair))
            .into_iter()
            .chain(place.map(|pair| (GlossaryKind::Place, pair)))
            .chain(
                domain
                    .iter()
                    .copied()
                    .map(|pair| (GlossaryKind::DomainTerm, pair)),
            );
        assert_eq!(
            result.glossary_hits.len(),
            usize::from(object.is_some()) + usize::from(place.is_some()) + domain.len()
        );
        for (kind, (term, replacement)) in expected_hits {
            let hit = result
                .glossary_hits
                .iter()
                .find(|h| h.kind == kind && h.canonical == term)
                .unwrap();
            let at = |text: &str, span: faberline_nlp::Span| {
                text.chars()
                    .skip(span.start_char)
                    .take(span.end_char - span.start_char)
                    .collect::<String>()
            };
            assert_eq!(at(source, hit.source_span), term);
            assert_eq!(at(&result.target_text, hit.target_span), replacement);
        }
    }
}

#[test]
fn household_goals_keep_their_role_in_a_domain_glossary() {
    for (source, place) in [
        ("靴を脱がないで、家に入りました。", "家"),
        ("靴を脱がないで家に入りました。", "家に"),
    ] {
        check_locks(
            source,
            None,
            None,
            &[(place, "舊屋")],
            "沒有脫下鞋子就進入舊屋了。",
        );
        check_locks(
            source,
            Some(("靴を", "皮鞋")),
            None,
            &[(place, "舊屋"), ("入りました", "踏進了")],
            "沒有脫下皮鞋就踏進了舊屋。",
        );
    }
}

#[test]
fn omitted_actions_keep_their_objects_and_the_later_destinations() {
    // D01–D03 in the adopted naide_clause_dev_v1 set.
    check(
        "朝ご飯を食べないで、学校へ行きました。",
        Some(("朝ご飯を", "早餐")),
        Some(("学校へ", "新校舍")),
        "沒有吃早餐就去了新校舍。",
    );
    check(
        "傘を持たないで、出かけました。",
        Some(("傘を", "雨具")),
        None,
        "沒有帶雨具就出門了。",
    );
    check(
        "靴を脱がないで、家に入りました。",
        Some(("靴を", "鞋履")),
        Some(("家に", "舊屋")),
        "沒有脫下鞋履就進入舊屋了。",
    );
}

#[test]
fn omitted_action_keeps_departure_source_when_a_glossary_covers_its_case() {
    // jz062 in the adopted standing development cases.
    let source = "朝ご飯を食べずに家を出た。";
    let native_meal = match faberline_nlp::lexicon::ja::lookup_ja_gloss(
        "朝ご飯",
        Some(faberline_nlp::doc::Upos::Noun),
    ) {
        None => "早上沒有吃飯就離開舊屋了。",
        Some("早餐") => "沒有吃早餐就離開舊屋了。",
        Some(gloss) => panic!("unexpected breakfast gloss: {gloss}"),
    };
    for place in ["家", "家を"] {
        check(source, None, Some((place, "舊屋")), native_meal);
        check_locks(source, None, None, &[(place, "舊屋")], native_meal);
        for meal in ["朝ご飯", "朝ご飯を"] {
            check(
                source,
                Some((meal, "早餐")),
                Some((place, "舊屋")),
                "沒有吃早餐就離開舊屋了。",
            );
            check_locks(
                source,
                Some((meal, "早餐")),
                None,
                &[(place, "舊屋")],
                "沒有吃早餐就離開舊屋了。",
            );
        }
    }
}

#[test]
fn departure_source_terms_work_for_other_physical_places() {
    check_locks(
        "水を飲まずに部屋を出た。",
        None,
        None,
        &[("部屋を", "書房")],
        "沒有喝水就離開書房了。",
    );
}

#[test]
fn a_negated_departure_does_not_recover_the_source_role() {
    check_locks_in_both_entries(
        "水を飲まずに家を出なかった。",
        None,
        None,
        &[("家を", "舊屋")],
        ["沒有喝在舊屋出去水。", "不喝在水的舊屋沒有出去。"],
    );
}

#[test]
fn complete_departure_terms_keep_the_omitted_action_and_the_source() {
    let source = "朝ご飯を食べずに家を出た。";
    for (term, replacement, expected) in [
        ("出た", "離開了", "沒有吃早餐就從家離開了。"),
        ("家を出た", "離開住處了", "沒有吃早餐就離開住處了。"),
    ] {
        for text in [
            source,
            "朝ご飯を食べずに、家を出た。",
            "朝ご飯を食べずに，家を出た。",
            "朝ご飯を食べずに,家を出た。",
        ] {
            check_locks(
                text,
                Some(("朝ご飯を", "早餐")),
                None,
                &[(term, replacement)],
                expected,
            );
        }
    }
    check_locks(
        "水を飲まずに部屋を出た。",
        None,
        None,
        &[("部屋を", "書房"), ("出た", "啟程了")],
        "沒有喝水就從書房啟程了。",
    );
    check_locks(
        "水を飲まずに兄の家を出た。",
        None,
        None,
        &[("出た", "離開了")],
        "沒有喝水就從哥哥的家離開了。",
    );
}
