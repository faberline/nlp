use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
    analyze_and_translate,
};

const HS_12: &str = "授業を担当する先生は、授業後に佐藤さんへ資料を渡しました。";

fn check_glossary(terms: &[(GlossaryKind, &str, &str)], expected: &str) {
    check_source(HS_12, terms, expected);
}

fn check_source(source: &str, terms: &[(GlossaryKind, &str, &str)], expected: &str) {
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
            &[(GlossaryKind::Actor, teacher, "任課老師")],
            "任課老師下課後把資料交給佐藤。",
        );
    }
}

#[test]
fn domain_terms_keep_their_linked_handover_roles_and_original_spans() {
    let domain = GlossaryKind::DomainTerm;
    let terms = [
        (domain, "先生は", "任課老師"),
        (domain, "佐藤さんへ", "佐藤同學"),
        (domain, "資料を", "指定資料"),
        (domain, "授業後に", "下課之後"),
    ];
    for (term, expected) in [
        (terms[0], "負責授課的任課老師下課後把資料交給佐藤。"),
        (terms[1], "負責授課的老師下課後把資料交給佐藤同學。"),
        (terms[2], "負責授課的老師下課後把指定資料交給佐藤。"),
    ] {
        check_glossary(&[term], expected);
    }
    check_glossary(&terms, "負責授課的任課老師下課之後把指定資料交給佐藤同學。");
    check_glossary(
        &[(domain, "授業を担当する先生は", "任課老師")],
        "任課老師下課後把資料交給佐藤。",
    );
}

#[test]
fn domain_terms_do_not_make_places_or_motion_into_handover_roles() {
    for (source, term, label, expected) in [
        (
            "先生は教室へ資料を渡しました。",
            "教室へ",
            "指定教室",
            "老師遞給了指定教室的資料。",
        ),
        ("先生は橋を渡りました。", "橋を", "石橋", "老師石橋渡過了。"),
    ] {
        check_source(source, &[(GlossaryKind::DomainTerm, term, label)], expected);
    }
}

#[test]
fn surname_glossary_keeps_the_linked_honorific_and_recipient() {
    let recipient = HS_12.replace("さんへ", "さんに");
    for source in [HS_12, recipient.as_str()] {
        for kind in [GlossaryKind::Actor, GlossaryKind::DomainTerm] {
            check_source(
                source,
                &[(kind, "佐藤", "張先生")],
                "負責授課的老師下課後把資料交給張先生。",
            );
        }
    }
    check_glossary(
        &[(GlossaryKind::DomainTerm, "佐藤", "張")],
        "負責授課的老師下課後把資料交給張。",
    );
    check_source(
        "鈴木さんは佐藤さんに資料を渡しました。",
        &[
            (GlossaryKind::DomainTerm, "鈴木", "陳老師"),
            (GlossaryKind::DomainTerm, "佐藤", "張先生"),
        ],
        "陳老師把資料交給張先生。",
    );
}

#[test]
fn matching_a_name_does_not_consume_an_independent_glossary_lock() {
    check_glossary(
        &[
            (GlossaryKind::DomainTerm, "佐藤", "張先生"),
            (GlossaryKind::DomainTerm, "さん", "君"),
        ],
        "負責授課的老師下課後往君張先生遞給了資料。",
    );
    check_source(
        "先生は大阪さんへ資料を渡しました。",
        &[(GlossaryKind::DomainTerm, "大阪", "大阪市")],
        "老師往三大阪市遞給了資料。",
    );
    check_source(
        "「佐藤さんへ」と書きました。",
        &[(GlossaryKind::DomainTerm, "佐藤", "張先生")],
        "「往三張先生」寫了。",
    );
}
