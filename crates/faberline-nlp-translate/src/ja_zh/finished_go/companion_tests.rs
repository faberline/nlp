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

#[test]
fn a_companion_and_a_market_keep_both_roles_under_place_locks() {
    let source = "母と一緒に市場へ行った。";
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let doc = analyze_document(
        source,
        &doc_options_for_translate(&options, &TranslationGlossary::new()),
    );
    for term in ["市場", "市場へ"] {
        let mut glossary = TranslationGlossary::new();
        glossary.places.push(GlossaryEntry {
            canonical: term.into(),
            surfaces: vec![term.into()],
            targets: [(TargetVariant::ZhHant, "公有市場".into())].into(),
            ..GlossaryEntry::default()
        });
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
        let analyzed = translate(source, &glossary);
        assert_eq!(direct.target_text, analyzed.target_text, "{term}");
        assert_eq!(direct.glossary_hits, analyzed.glossary_hits, "{term}");
        assert_eq!(analyzed.target_text, "跟母親一起去了公有市場。", "{term}");
        let [hit] = analyzed.glossary_hits.as_slice() else {
            panic!("expected the original Place match");
        };
        assert_eq!(hit.canonical, term);
        assert_eq!(hit.source_span.start_char, 5);
        assert_eq!(hit.source_span.end_char, 5 + term.chars().count());
        let matched: String = analyzed
            .target_text
            .chars()
            .skip(hit.target_span.start_char)
            .take(hit.target_span.end_char - hit.target_span.start_char)
            .collect();
        assert_eq!(matched, "公有市場");
    }
}

#[test]
fn a_place_lock_cannot_turn_an_invitation_into_a_completed_trip() {
    let output = translate("母と一緒に市場へ行きませんか。", &market_glossary("市場へ"));
    assert_eq!(output.target_text, "要不要和母親一起公有市場去？");
    assert_eq!(output.glossary_hits[0].canonical, "市場へ");
}

#[test]
fn actor_and_place_locks_keep_both_roles_in_the_completed_trip() {
    // Adopted public development case jz004.
    let source = "母と一緒に市場へ行った。";
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let raw = analyze_document(
        source,
        &doc_options_for_translate(&options, &TranslationGlossary::new()),
    );
    for actor in ["母", "母と"] {
        for place in [None, Some("市場"), Some("市場へ")] {
            let mut glossary = place.map_or_else(TranslationGlossary::new, market_glossary);
            glossary.actors.push(GlossaryEntry {
                canonical: actor.into(),
                surfaces: vec![actor.into()],
                targets: [(TargetVariant::ZhHant, "媽媽".into())].into(),
                ..GlossaryEntry::default()
            });
            let analyzed = translate(source, &glossary);
            let direct = translate_document(
                source,
                &raw,
                #[cfg(feature = "narrative")]
                None,
                &options,
                &glossary,
                None,
            )
            .unwrap();
            let goal = if place.is_some() {
                "公有市場"
            } else {
                "市場"
            };
            assert_eq!(analyzed.target_text, format!("跟媽媽一起去了{goal}。"));
            assert_eq!(direct.target_text, analyzed.target_text);
            assert_eq!(direct.glossary_hits, analyzed.glossary_hits);
            let [person, destinations @ ..] = analyzed.glossary_hits.as_slice() else {
                panic!("expected an actor glossary hit");
            };
            assert_eq!(person.kind, GlossaryKind::Actor);
            assert_eq!(person.canonical, actor);
            assert_eq!(
                (person.source_span.start_char, person.source_span.end_char),
                (0, actor.chars().count())
            );
            assert_eq!(
                (person.target_span.start_char, person.target_span.end_char),
                (1, 3)
            );
            match place {
                Some(term) => {
                    let [destination] = destinations else {
                        panic!("expected one place glossary hit");
                    };
                    assert_eq!(destination.kind, GlossaryKind::Place);
                    assert_eq!(destination.canonical, term);
                    assert_eq!(
                        (
                            destination.source_span.start_char,
                            destination.source_span.end_char
                        ),
                        (5, 5 + term.chars().count())
                    );
                    assert_eq!(
                        (
                            destination.target_span.start_char,
                            destination.target_span.end_char
                        ),
                        (7, 11)
                    );
                }
                None => assert!(destinations.is_empty()),
            }
        }
    }
}

#[test]
fn an_actor_lock_does_not_turn_a_coordinated_subject_into_a_companion() {
    let mut glossary = TranslationGlossary::new();
    glossary.actors.push(GlossaryEntry {
        canonical: "兄".into(),
        surfaces: vec!["兄".into()],
        targets: [(TargetVariant::ZhHant, "親人".into())].into(),
        ..GlossaryEntry::default()
    });
    glossary.places.push(GlossaryEntry {
        canonical: "駅へ".into(),
        surfaces: vec!["駅へ".into()],
        targets: [(TargetVariant::ZhHant, "北站".into())].into(),
        ..GlossaryEntry::default()
    });
    let result = translate("兄と妹が駅へ行きました。", &glossary);
    assert_eq!(result.target_text, "親人和妹妹北站去了。");
    assert_eq!(result.glossary_hits.len(), 2);
}

fn market_glossary(term: &str) -> TranslationGlossary {
    let mut glossary = TranslationGlossary::new();
    glossary.places.push(GlossaryEntry {
        canonical: term.into(),
        surfaces: vec![term.into()],
        targets: [(TargetVariant::ZhHant, "公有市場".into())].into(),
        ..GlossaryEntry::default()
    });
    glossary
}
