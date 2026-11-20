use faberline_nlp::{
    Span,
    doc::{Document, analyze_document, frame::FrameRole, model::Backend},
};
use faberline_nlp_translate::{
    GlossaryKind, TargetVariant, TranslateOptions, TranslatedDocument, TranslationGlossary,
    analyze_and_translate, doc_options_for_translate, translate_document,
};
use serde::Deserialize;

const MEETING_CASES: [&str; 3] = ["P01", "M04", "M08"];

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    glossary: TranslationGlossary,
    negative_controls: Vec<NegativeControl>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    source: String,
    glossary: bool,
    documents: Vec<FrozenDocument>,
}

#[derive(Deserialize)]
struct NegativeControl {
    id: String,
    source: String,
    doc: Document,
    glossary: TranslationGlossary,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Stage {
    Main,
    Candidate,
    CandidateGlossary,
}

#[derive(Deserialize)]
struct FrozenDocument {
    stage: Stage,
    doc: Document,
    baseline_target: String,
}

impl Fixture {
    fn load() -> Self {
        serde_json::from_str(include_str!("fixtures/meeting-voice-documents.json"))
            .expect("valid frozen meeting fixture")
    }

    fn case(&self, id: &str) -> &Case {
        self.cases
            .iter()
            .find(|case| case.id == id)
            .unwrap_or_else(|| panic!("missing fixture case {id}"))
    }
}

impl Case {
    fn document(&self, stage: Stage) -> &FrozenDocument {
        self.documents
            .iter()
            .find(|document| document.stage == stage)
            .unwrap_or_else(|| panic!("{} has no {stage:?} Document", self.id))
    }

    fn render(&self, doc: &Document, glossary: &TranslationGlossary) -> TranslatedDocument {
        let empty_glossary = TranslationGlossary::new();
        let glossary = if self.glossary {
            glossary
        } else {
            &empty_glossary
        };
        translate_document(
            &self.source,
            doc,
            #[cfg(feature = "narrative")]
            None,
            &TranslateOptions::new(TargetVariant::ZhHant),
            glossary,
            None,
        )
        .unwrap_or_else(|error| {
            panic!(
                "{} could not translate its frozen Document: {error}",
                self.id
            )
        })
    }
}

fn contains_negation(text: &str) -> bool {
    text.contains(['不', '沒', '未'])
}

fn assert_no_passive(text: &str, label: &str) {
    assert!(
        !text.contains(['被', '遭', '挨']),
        "{label}: invented a passive construction: {text}"
    );
}

fn assert_meeting_with_counterpart(case_id: &str, text: &str, label: &str, person_locked: bool) {
    let (subjects, counterparts): (&[&str], &[&str]) = match case_id {
        "P01" => (
            &["老師", "先生"],
            &["客人", "顧客", "來客", "訪客", "貴客", "客大人"],
        ),
        "M04" => (&["校長"], &["家長", "監護人", "保護者"]),
        "M08" => (&["支部長", "分部長", "分會長", "分部主任"], &["森川代表"]),
        _ => panic!("no meeting criteria for {case_id}"),
    };
    assert_no_passive(text, label);

    let action_start = ["見面", "會面", "相見"]
        .iter()
        .filter_map(|verb| text.find(*verb))
        .min()
        .unwrap_or_else(|| panic!("{label}: missing meeting: {text}"));
    let before_action = &text[..action_start];
    assert!(
        subjects
            .iter()
            .any(|subject| before_action.contains(*subject)),
        "{label}: missing meeting subject: {text}"
    );
    assert!(
        ['和', '跟', '與', '同'].iter().any(|connector| {
            before_action
                .split_once(*connector)
                .is_some_and(|(_, tail)| {
                    counterparts.iter().any(|name| tail.contains(*name))
                        || (case_id == "M08"
                            && !person_locked
                            && tail
                                .split(['，', '。', '；', '！', '？', '、', '和', '跟', '與', '同'])
                                .next()
                                .is_some_and(|name| name.contains("森川")))
                })
        }),
        "{label}: missing meeting counterpart: {text}"
    );
    assert!(
        contains_negation(before_action),
        "{label}: lost the meeting negation: {text}"
    );
    for modal in ["不能", "無法", "沒辦法", "不願", "不想"] {
        assert!(
            !before_action.contains(modal),
            "{label}: inferred an unsupported modality: {text}"
        );
    }
    if case_id == "M04" {
        assert!(
            before_action.contains("明天") || before_action.contains("明日"),
            "{label}: missing the meeting date: {text}"
        );
    }
}

#[test]
fn old_meeting_documents_keep_negation_without_inventing_passive() {
    let fixture = Fixture::load();
    for id in MEETING_CASES {
        let case = fixture.case(id);
        let translated = case.render(&case.document(Stage::Main).doc, &fixture.glossary);
        let text = &translated.target_text;
        assert_no_passive(text, id);
        assert!(contains_negation(text), "{id}: lost the negation: {text}");
    }
}

#[test]
fn attached_voice_negatives_keep_the_meeting_counterpart() {
    let fixture = Fixture::load();
    for id in MEETING_CASES {
        let case = fixture.case(id);
        let translated = case.render(&case.document(Stage::Candidate).doc, &fixture.glossary);
        assert_meeting_with_counterpart(id, &translated.target_text, id, case.glossary);
    }
}

fn slice_chars(text: &str, span: Span) -> String {
    let chars: Vec<char> = text.chars().collect();
    chars[span.start_char..span.end_char].iter().collect()
}

fn assert_meeting_glossary(
    source_text: &str,
    translated: &TranslatedDocument,
    label: &str,
    place_locked: bool,
    person_surface: Option<&str>,
) {
    let place = place_locked.then_some(("星波会館", "星潮中心", GlossaryKind::Place, 4, 8));
    let person = person_surface.map(|surface| {
        (
            surface,
            "森川代表",
            GlossaryKind::DomainTerm,
            9,
            9 + surface.chars().count(),
        )
    });
    assert_eq!(
        translated.glossary_hits.len(),
        usize::from(place.is_some()) + usize::from(person.is_some()),
        "{label}"
    );
    for (source, target, kind, start_char, end_char) in [place, person].into_iter().flatten() {
        let hit = translated
            .glossary_hits
            .iter()
            .find(|hit| hit.canonical == source)
            .unwrap_or_else(|| panic!("{label}: missing glossary hit {source}"));
        assert_eq!(hit.kind, kind, "{label}: {source}");
        assert_eq!(hit.sentence, Some(0), "{label}: {source}");
        assert_eq!(
            hit.source_span,
            Span {
                start_char,
                end_char,
            },
            "{label}: {source}"
        );
        assert_eq!(slice_chars(source_text, hit.source_span), source);
        assert_eq!(
            slice_chars(&translated.target_text, hit.target_span),
            target,
            "{label}: {source}"
        );
        assert_eq!(
            translated.target_text.matches(target).count(),
            1,
            "{label}: {source}"
        );
    }
}

#[test]
fn meeting_glossary_keeps_each_lock_and_its_alignment() {
    let fixture = Fixture::load();
    let case = fixture.case("M08");
    for stage in [Stage::Main, Stage::Candidate, Stage::CandidateGlossary] {
        let translated = case.render(&case.document(stage).doc, &fixture.glossary);
        let label = format!("M08 {stage:?}");
        assert_meeting_glossary(&case.source, &translated, &label, true, Some("森川さん"));
        if stage == Stage::CandidateGlossary {
            assert_meeting_with_counterpart("M08", &translated.target_text, &label, true);
        }
    }
}

#[test]
fn meeting_glossary_retains_roles_through_public_analysis() {
    let fixture = Fixture::load();
    let case = fixture.case("M08");
    for (glossary_label, place_locked, person_locked) in [
        ("no glossary", false, false),
        ("place only", true, false),
        ("person only", false, true),
        ("place and person", true, true),
    ] {
        let mut glossary = fixture.glossary.clone();
        if !place_locked {
            glossary.places.clear();
        }
        if !person_locked {
            glossary.domain_terms.clear();
        }

        for (source_hint, mode) in [(None, "auto"), (Some(TargetVariant::Ja), "ja")] {
            let label = format!("M08 {glossary_label}, {mode}");
            let mut options = TranslateOptions::new(TargetVariant::ZhHant);
            options.source_hint = source_hint;
            let doc_options = doc_options_for_translate(&options, &glossary);
            let doc = analyze_document(&case.source, &doc_options);
            let translated =
                analyze_and_translate(&case.source, &options, &glossary, None, Backend::Onnx)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));

            assert_meeting_glossary(
                &case.source,
                &translated,
                &label,
                place_locked,
                person_locked.then_some("森川さん"),
            );
            let text = &translated.target_text;
            let contains_meeting_verb = doc
                .tokens
                .iter()
                .any(|token| token.span.start_char == 14 && token.lemma.as_deref() == Some("会う"));
            if contains_meeting_verb {
                assert_meeting_with_counterpart("M08", text, &label, person_locked);
            } else {
                assert_no_passive(text, &label);
                assert!(
                    contains_negation(text),
                    "{label}: lost the negation: {text}"
                );
            }
        }
    }
}

#[test]
fn synthetic_name_without_honorific_keeps_glossary_alignment() {
    let fixture = Fixture::load();
    let control = fixture
        .negative_controls
        .iter()
        .find(|control| control.id == "M08-no-honorific")
        .expect("frozen M08 control without an honorific");
    // The fixture adds a synthetic Name entity to an official Document
    // so we can check glossary alignment with that annotation present.
    let translated = translate_document(
        &control.source,
        &control.doc,
        #[cfg(feature = "narrative")]
        None,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &control.glossary,
        None,
    )
    .unwrap_or_else(|error| panic!("{}: {error}", control.id));
    assert_meeting_glossary(
        &control.source,
        &translated,
        &control.id,
        true,
        Some("森川"),
    );
}

#[test]
fn meeting_frame_cannot_create_a_passive_agent() {
    let fixture = Fixture::load();
    let case = fixture.case("P01");
    let original = &case.document(Stage::Candidate).doc;
    let frame_index = original
        .frames
        .iter()
        .position(|frame| slice_chars(&case.source, frame.span) == "会わ")
        .expect("P01 candidate must have a meeting frame");
    let mut changed_role = original.clone();
    let arg = changed_role.frames[frame_index]
        .args
        .iter_mut()
        .find(|arg| arg.role == FrameRole::Arg0 && slice_chars(&case.source, arg.span) == "様")
        .expect("P01 candidate must label the counterpart ARG0");
    arg.role = FrameRole::Argm;

    let mut removed_frame = original.clone();
    removed_frame.frames.remove(frame_index);

    for (label, doc) in [
        ("P01 with counterpart ARG0 changed to ARGM", &changed_role),
        ("P01 without the meeting frame", &removed_frame),
    ] {
        assert_eq!(doc.tokens, original.tokens, "{label}");
        assert_eq!(doc.syntax, original.syntax, "{label}");
        let translated = case.render(doc, &fixture.glossary);
        assert_meeting_with_counterpart("P01", &translated.target_text, label, case.glossary);
    }
}

#[test]
fn sources_outside_the_meeting_change_keep_their_baseline_text() {
    let fixture = Fixture::load();
    assert_eq!(fixture.cases.len(), 16);
    let mut checked = 0;
    for case in &fixture.cases {
        if MEETING_CASES.contains(&case.id.as_str()) {
            continue;
        }
        assert_eq!(case.documents.len(), 2, "{}", case.id);
        for stage in [Stage::Main, Stage::Candidate] {
            let frozen = case.document(stage);
            let translated = case.render(&frozen.doc, &fixture.glossary);
            // These are regression baselines, including inherited translation gaps.
            assert_eq!(
                translated.target_text, frozen.baseline_target,
                "{} {stage:?}",
                case.id
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 26);
}
