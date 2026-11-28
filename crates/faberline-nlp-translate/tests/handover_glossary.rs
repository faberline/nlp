use faberline_nlp::doc::model::Backend;
use faberline_nlp::doc::{Document, Opaque, OpaqueKind, analyze_document};
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
    analyze_and_translate, doc_options_for_translate, translate_document,
};

const HS_12: &str = "授業を担当する先生は、授業後に佐藤さんへ資料を渡しました。";
const TWO_HANDOVERS: &str = "先生は健太に本を渡し、美咲に手紙を渡しました。";

fn handover_glossary(terms: &[(GlossaryKind, &str, &str)]) -> TranslationGlossary {
    let mut glossary = TranslationGlossary::new();
    for &(kind, term, target) in terms {
        let entries = match kind {
            GlossaryKind::Actor => &mut glossary.actors,
            GlossaryKind::Object => &mut glossary.objects,
            GlossaryKind::DomainTerm => &mut glossary.domain_terms,
            GlossaryKind::Place => &mut glossary.places,
        };
        entries.push(GlossaryEntry {
            canonical: term.to_owned(),
            surfaces: vec![term.to_owned()],
            targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
            ..Default::default()
        });
    }
    glossary
}

fn check_glossary(source: &str, terms: &[(GlossaryKind, &str, &str)], expected: &str) {
    let glossary = handover_glossary(terms);
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let translated =
        analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
    assert_eq!(translated.target_text, expected);
    assert_eq!(translated.glossary_hits.len(), terms.len());
    for &(kind, term, target) in terms {
        let hit = translated
            .glossary_hits
            .iter()
            .find(|hit| hit.kind == kind && hit.canonical == term)
            .unwrap();
        let source_slice: String = source
            .chars()
            .skip(hit.source_span.start_char)
            .take(hit.source_span.end_char - hit.source_span.start_char)
            .collect();
        let target_slice: String = translated
            .target_text
            .chars()
            .skip(hit.target_span.start_char)
            .take(hit.target_span.end_char - hit.target_span.start_char)
            .collect();
        assert_eq!(source_slice, term);
        assert_eq!(target_slice, target);
    }
}

#[test]
fn handover_preserves_protected_arguments_and_course_time_together() {
    for time in ["授業後", "授業後に"] {
        check_glossary(
            HS_12,
            &[
                (GlossaryKind::Actor, "先生", "任課老師"),
                (GlossaryKind::Actor, "佐藤さんへ", "佐藤同學"),
                (GlossaryKind::Object, "資料を", "指定資料"),
                (GlossaryKind::DomainTerm, time, "下課之後"),
            ],
            "負責授課的任課老師下課之後把指定資料交給佐藤同學。",
        );
    }
}

#[test]
fn complete_subject_replacement_does_not_repeat_its_relative_clause() {
    for teacher in ["授業を担当する先生", "授業を担当する先生は"] {
        check_glossary(
            HS_12,
            &[(GlossaryKind::Actor, teacher, "任課老師")],
            "任課老師下課後把資料交給佐藤。",
        );
    }
}

#[test]
fn partial_subject_replacement_keeps_its_uncovered_argument_in_place() {
    check_glossary(
        HS_12,
        &[(GlossaryKind::Actor, "担当する先生", "任課老師")],
        "任課老師授業，下課後往佐藤遞給了資料。",
    );
}

#[test]
fn separate_handovers_keep_each_glossary_object_with_its_recipient() {
    check_glossary(
        TWO_HANDOVERS,
        &[
            (GlossaryKind::Actor, "先生", "導師"),
            (GlossaryKind::Actor, "健太に", "甲同學"),
            (GlossaryKind::Object, "本を", "指定書籍"),
            (GlossaryKind::Actor, "美咲", "乙同學"),
            (GlossaryKind::Object, "手紙", "指定信件"),
        ],
        "導師把指定書籍交給甲同學，也把指定信件交給乙同學。",
    );
}

#[test]
fn separate_handovers_and_a_distinct_action_keep_two_clause_spans() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("data/ja_zh/two_handover_dev_v1.json")).unwrap();
    let items = cases["items"].as_array().unwrap();
    let glossary = TranslationGlossary::new();
    for (id, first, second) in [
        ("T01", "老師把書交給健太，", "也把信交給美咲。"),
        (
            "T02",
            "店員把紙袋交給客人田中，",
            "也把箱交給另一位客人佐藤。",
        ),
        ("T03", "老師把地圖交給學生葵，", "但葵問了路。"),
    ] {
        let source = items.iter().find(|row| row["id"] == id).unwrap()["jp"]
            .as_str()
            .unwrap();
        for hint in [None, Some(TargetVariant::Ja)] {
            let mut options = TranslateOptions::new(TargetVariant::ZhHant);
            options.source_hint = hint;
            let translated =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(translated.target_text, format!("{first}{second}"), "{id}");
            let [left, right] = translated.clause_alignments.as_slice() else {
                panic!("{id}: expected two clause spans");
            };
            assert_eq!(left.source_span.start_char, 0);
            assert_eq!(right.source_span.start_char, left.source_span.end_char + 1);
            assert_eq!(right.source_span.end_char, source.chars().count());
            assert_eq!(left.target_span.end_char, first.chars().count());
            assert_eq!(right.target_span.start_char, left.target_span.end_char);
            assert_eq!(
                right.target_span.end_char,
                translated.target_text.chars().count()
            );
        }
    }
}

#[test]
fn two_handovers_do_not_repair_a_detached_caller_argument() {
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let glossary = handover_glossary(&[(GlossaryKind::Object, "本を", "指定書籍")]);
    let empty = TranslationGlossary::new();
    let mut source = analyze_document(TWO_HANDOVERS, &doc_options_for_translate(&options, &empty));
    let translate = |source: &Document| {
        translate_document(
            TWO_HANDOVERS,
            source,
            #[cfg(feature = "narrative")]
            None,
            &options,
            &glossary,
            None,
        )
        .unwrap()
    };
    let repaired = "老師把指定書籍交給健太，也把信交給美咲。";
    assert_eq!(translate(&source).target_text, repaired);
    let first_object = source
        .syntax
        .iter_mut()
        .find(|node| node.deprel == "obj")
        .unwrap();
    first_object.head = None;
    let fallback = translate(&source);
    assert_ne!(fallback.target_text, repaired);
    let [hit] = fallback.glossary_hits.as_slice() else {
        panic!("expected the protected caller term to survive");
    };
    let target: String = fallback
        .target_text
        .chars()
        .skip(hit.target_span.start_char)
        .take(hit.target_span.end_char - hit.target_span.start_char)
        .collect();
    assert_eq!(target, "指定書籍");
}

#[test]
fn handover_keeps_caller_source_boundaries() {
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let empty = TranslationGlossary::new();
    let glossary = handover_glossary(&[(GlossaryKind::Actor, "先生", "任課老師")]);
    let mut source = analyze_document(HS_12, &doc_options_for_translate(&options, &empty));
    let translate = |source: &Document| {
        translate_document(
            HS_12,
            source,
            #[cfg(feature = "narrative")]
            None,
            &options,
            &glossary,
            None,
        )
        .unwrap()
    };
    let expected = "負責授課的任課老師下課後把資料交給佐藤。";
    assert_eq!(translate(&source).target_text, expected);

    let slice = |text: &str, span: faberline_nlp::Span| {
        text.chars()
            .skip(span.start_char)
            .take(span.end_char - span.start_char)
            .collect::<String>()
    };
    let object = source
        .syntax
        .iter()
        .find(|arc| arc.deprel == "obj" && slice(HS_12, arc.span) == "資料")
        .unwrap()
        .span;
    let mut opaque = source.clone();
    opaque.segmentation.opaque.push(Opaque {
        kind: OpaqueKind::Code,
        span: object,
    });
    let protected = translate(&opaque);
    assert_ne!(protected.target_text, expected);
    let [record] = protected.opaque_spans.as_slice() else {
        panic!("missing the caller's opaque span");
    };
    assert_eq!(record.source_span, object);
    assert_eq!(
        slice(&protected.target_text, record.target_span.unwrap()),
        "資料"
    );

    let teacher = source
        .syntax
        .iter_mut()
        .find(|arc| arc.deprel == "nsubj" && slice(HS_12, arc.span) == "先生")
        .unwrap();
    teacher.head = None;
    let detached = translate(&source);
    assert_ne!(detached.target_text, expected);
    let [hit] = detached.glossary_hits.as_slice() else {
        panic!("expected the actor lock to survive without reanalysis");
    };
    assert_eq!(slice(HS_12, hit.source_span), "先生");
    assert_eq!(slice(&detached.target_text, hit.target_span), "任課老師");
}
