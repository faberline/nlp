use faberline_nlp::{
    Span,
    doc::{Document, EntityKind, Upos, analyze_document, frame::FrameRole, model::Backend},
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
        "P01" => (&["老師", "先生"], &["客人", "顧客", "來客", "訪客", "貴客"]),
        "M04" => (&["校長"], &["家長"]),
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
    let negative = before_action
        .find(['不', '沒', '未'])
        .expect("meeting negation was checked above");
    let connector = before_action
        .find(['和', '跟', '與', '同'])
        .expect("meeting counterpart was checked above");
    assert!(
        negative < connector,
        "{label}: negation must precede the counterpart: {text}"
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
        assert!(
            !before_action.contains("明天，") && !before_action.contains("明日，"),
            "{label}: source-linked meeting date must not be separated: {text}"
        );
    }
    if case_id == "M08" {
        let subject = subjects
            .iter()
            .filter_map(|subject| before_action.find(*subject))
            .min()
            .expect("meeting subject was checked above");
        // With a place lock, use its translated target. Without that lock,
        // accept the existing rendering of the hall while checking its scope.
        let place = before_action
            .find("星潮中心")
            .or_else(|| before_action.find("會館"))
            .unwrap_or_else(|| panic!("{label}: missing meeting location: {text}"));
        let counterpart = before_action
            .find("森川")
            .unwrap_or_else(|| panic!("{label}: missing meeting counterpart: {text}"));
        let location_marker = before_action
            .find('在')
            .unwrap_or_else(|| panic!("{label}: missing location marker: {text}"));
        assert!(
            subject < negative
                && negative < connector
                && connector < counterpart
                && counterpart < location_marker
                && location_marker < place,
            "{label}: meeting location must follow the negated counterpart: {text}"
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
        if id == "M04" {
            assert_sentence_spans(&case.source, &translated, id);
        }
    }
}

fn slice_chars(text: &str, span: Span) -> String {
    let chars: Vec<char> = text.chars().collect();
    chars[span.start_char..span.end_char].iter().collect()
}

fn assert_sentence_spans(source: &str, translated: &TranslatedDocument, label: &str) {
    assert_eq!(translated.source_chars, source.chars().count(), "{label}");
    assert_eq!(
        translated.target_chars,
        translated.target_text.chars().count(),
        "{label}"
    );
    assert_eq!(translated.sentences.len(), 1, "{label}");
    let sentence = &translated.sentences[0];
    assert_eq!(sentence.sentence, 0, "{label}");
    assert_eq!(
        sentence.source_span,
        Span {
            start_char: 0,
            end_char: source.chars().count()
        },
        "{label}"
    );
    assert_eq!(
        sentence.target_span,
        Span {
            start_char: 0,
            end_char: translated.target_text.chars().count()
        },
        "{label}"
    );
    assert_eq!(slice_chars(source, sentence.source_span), source, "{label}");
    assert_eq!(
        slice_chars(&translated.target_text, sentence.target_span),
        sentence.target_text,
        "{label}"
    );
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

    let mut removed_counterpart = original.clone();
    let counterpart_indices: Vec<_> = original.frames[frame_index]
        .args
        .iter()
        .enumerate()
        .filter(|(_, arg)| {
            arg.role == FrameRole::Arg0 && slice_chars(&case.source, arg.span) == "様"
        })
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        counterpart_indices.len(),
        1,
        "P01 has one counterpart argument"
    );
    let counterpart_index = counterpart_indices[0];
    let removed_argument = removed_counterpart.frames[frame_index]
        .args
        .remove(counterpart_index);
    assert!(
        !removed_counterpart.frames[frame_index].args.is_empty(),
        "P01 retains its other frame roles"
    );
    let mut restored_frames = removed_counterpart.frames.clone();
    restored_frames[frame_index]
        .args
        .insert(counterpart_index, removed_argument);
    assert_eq!(
        restored_frames, original.frames,
        "removing the counterpart changes no other frame or role"
    );

    let mut removed_frame = original.clone();
    removed_frame.frames.remove(frame_index);

    for (label, doc) in [
        ("P01 with counterpart ARG0 changed to ARGM", &changed_role),
        (
            "P01 with only the counterpart argument removed",
            &removed_counterpart,
        ),
        ("P01 without the meeting frame", &removed_frame),
    ] {
        assert_eq!(doc.tokens, original.tokens, "{label}");
        assert_eq!(doc.syntax, original.syntax, "{label}");
        let translated = case.render(doc, &fixture.glossary);
        assert_meeting_with_counterpart("P01", &translated.target_text, label, case.glossary);
    }
}

#[test]
fn removing_m04_counterpart_keeps_other_frame_roles() {
    let fixture = Fixture::load();
    let case = fixture.case("M04");
    let original = &case.document(Stage::Candidate).doc;
    let frame_index = original
        .frames
        .iter()
        .position(|frame| slice_chars(&case.source, frame.span) == "会わ")
        .expect("M04 candidate must have a meeting frame");
    let frame = &original.frames[frame_index];
    let matches: Vec<_> = frame
        .args
        .iter()
        .enumerate()
        .filter(|(_, arg)| arg.role == FrameRole::Argm && arg.head == 5)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(matches.len(), 1, "M04 has one counterpart argument");
    let mut changed = original.clone();
    let removed = changed.frames[frame_index].args.remove(matches[0]);
    let remaining = &changed.frames[frame_index].args;
    assert!(remaining.iter().any(|arg| {
        arg.role == FrameRole::Arg1 && slice_chars(&case.source, arg.span) == "校長"
    }));
    assert!(remaining.iter().any(|arg| {
        arg.role == FrameRole::Argm && slice_chars(&case.source, arg.span) == "明日"
    }));
    let mut restored_frames = changed.frames.clone();
    restored_frames[frame_index]
        .args
        .insert(matches[0], removed);
    assert_eq!(restored_frames, original.frames);
    assert_eq!(changed.tokens, original.tokens);
    assert_eq!(changed.syntax, original.syntax);

    for (label, doc) in [
        ("M04 original", original),
        ("M04 without counterpart ARGM", &changed),
    ] {
        let translated = case.render(doc, &fixture.glossary);
        assert_meeting_with_counterpart("M04", &translated.target_text, label, case.glossary);
        assert_eq!(translated.clause_alignments.len(), 2, "{label}");
        let first = &translated.clause_alignments[0];
        let second = &translated.clause_alignments[1];
        assert_eq!((first.clause, first.sentence), (0, 0), "{label}");
        assert_eq!((second.clause, second.sentence), (1, 0), "{label}");
        assert_eq!(
            first.source_span,
            Span {
                start_char: 0,
                end_char: 5
            },
            "{label}"
        );
        assert_eq!(
            second.source_span,
            Span {
                start_char: 6,
                end_char: 17
            },
            "{label}"
        );
        assert_eq!(first.target_span.start_char, 0, "{label}");
        assert_eq!(
            first.target_span.end_char, second.target_span.start_char,
            "{label}"
        );
        assert_eq!(
            second.target_span.end_char, translated.target_chars,
            "{label}"
        );
        assert_eq!(
            slice_chars(&translated.target_text, first.target_span),
            "校長明天",
            "{label}"
        );
        assert_eq!(
            slice_chars(&translated.target_text, second.target_span),
            "不和家長見面。",
            "{label}"
        );
    }
}

#[test]
fn past_negative_meeting_keeps_past_negation_when_chain_is_attached() {
    let source = "先生はお客様に会われませんでした。";
    let glossary = TranslationGlossary::new();
    let mut options = TranslateOptions::new(TargetVariant::ZhHant);
    options.source_hint = Some(TargetVariant::Ja);
    let doc_options = doc_options_for_translate(&options, &glossary);
    let doc = analyze_document(source, &doc_options);
    let translated = analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
        .expect("past negative meeting translation");
    let text = &translated.target_text;

    let attached_chain = doc.tokens.windows(4).enumerate().any(|(index, chain)| {
        let surfaces: Vec<_> = chain
            .iter()
            .map(|token| slice_chars(source, token.span))
            .collect();
        let root = doc.syntax.iter().find(|edge| edge.token == index);
        chain[0].upos == Some(Upos::Verb)
            && chain[0].lemma.as_deref() == Some("会う")
            && root.is_some_and(|edge| edge.deprel == "root")
            && matches!(
                (surfaces[1].as_str(), chain[1].lemma.as_deref()),
                ("れ", Some("れる")) | ("られ", Some("られる"))
            )
            && surfaces[2] == "ませ"
            && chain[2].lemma.as_deref() == Some("ます")
            && surfaces[3] == "ん"
            && chain
                .windows(2)
                .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
            && (1..4).all(|offset| {
                chain[offset].upos == Some(Upos::Aux)
                    && doc.syntax.iter().any(|edge| {
                        edge.token == index + offset
                            && edge.deprel == "aux"
                            && edge.head == Some(index)
                    })
            })
    });
    if !attached_chain {
        assert!(
            contains_negation(text),
            "existing path lost negation: {text}"
        );
        return;
    }
    assert_no_passive(text, "past negative meeting");
    assert!(
        ["老師", "先生"]
            .iter()
            .any(|subject| text.contains(subject)),
        "missing subject: {text}"
    );
    assert!(
        ["客人", "顧客", "來客", "訪客", "貴客"]
            .iter()
            .any(|person| text.contains(person)),
        "missing counterpart: {text}"
    );
    assert!(
        ["見面", "會面", "相見"]
            .iter()
            .any(|verb| text.contains(verb)),
        "missing meeting: {text}"
    );
    let negative = text.find('沒').expect("past negative must use 沒 or 沒有");
    let connector = text
        .find(['和', '跟', '與', '同'])
        .expect("missing counterpart connector");
    assert!(
        negative < connector,
        "past negation must precede counterpart: {text}"
    );
}

#[test]
fn m04_comma_stays_without_source_linked_same_event_evidence() {
    let fixture = Fixture::load();
    let case = fixture.case("M04");
    let original = &case.document(Stage::Candidate).doc;
    let mut variants = Vec::new();
    for (label, token) in [("date head", 2), ("comma head", 3), ("subject head", 0)] {
        let index = original
            .syntax
            .iter()
            .position(|edge| edge.token == token)
            .unwrap();
        assert_eq!(original.syntax[index].head, Some(7), "{label}");
        let mut changed = original.clone();
        changed.syntax[index].head = Some(5);
        let mut restored = changed.clone();
        restored.syntax[index].head = Some(7);
        assert_eq!(&restored, original, "{label} changes only one head");
        variants.push((label, changed));
    }

    let date_index = original
        .entities
        .iter()
        .position(|entity| entity.kind == EntityKind::RelativeDate)
        .unwrap();
    let mut no_date = original.clone();
    let date = no_date.entities.remove(date_index);
    let mut restored = no_date.clone();
    restored.entities.insert(date_index, date);
    assert_eq!(
        &restored, original,
        "only the relative date entity was removed"
    );
    variants.push(("missing relative date", no_date));

    let syntax_index = original
        .syntax
        .iter()
        .position(|edge| edge.token == 2)
        .unwrap();
    let mut mismatched = original.clone();
    mismatched.syntax[syntax_index].span.start_char += 1;
    let mut restored = mismatched.clone();
    restored.syntax[syntax_index].span = original.syntax[syntax_index].span;
    assert_eq!(&restored, original, "only the date dependency span differs");
    variants.push(("inconsistent date dependency", mismatched));

    let mut second_verb = original.clone();
    assert_ne!(second_verb.tokens[4].upos, Some(Upos::Verb));
    second_verb.tokens[4].upos = Some(Upos::Verb);
    let mut restored = second_verb.clone();
    restored.tokens[4].upos = original.tokens[4].upos;
    assert_eq!(
        &restored, original,
        "only the unrelated token's part of speech differs"
    );
    variants.push(("additional verb", second_verb));

    for (label, doc) in variants {
        let translated = case.render(&doc, &fixture.glossary);
        assert!(
            translated.target_text.contains('，'),
            "{label}: source comma lost: {}",
            translated.target_text
        );
        assert_sentence_spans(&case.source, &translated, label);
    }
}

#[test]
fn m04_glossary_locks_keep_comma_and_unicode_alignment() {
    let fixture = Fixture::load();
    let case = fixture.case("M04");
    let doc = &case.document(Stage::Candidate).doc;
    for (label, date_locked, counterpart_locked) in [
        ("date lock", true, false),
        ("counterpart lock", false, true),
        ("both locks", true, true),
    ] {
        let mut entries = Vec::new();
        if date_locked {
            entries.push(serde_json::json!({
                "canonical": "明日", "surfaces": ["明日"],
                "targets": {"zh-Hant": "隔天"}
            }));
        }
        if counterpart_locked {
            entries.push(serde_json::json!({
                "canonical": "保護者", "surfaces": ["保護者"],
                "targets": {"zh-Hant": "家長代表"}
            }));
        }
        let glossary: TranslationGlossary = serde_json::from_value(serde_json::json!({
            "schema": "nlp.translate.glossary.v1", "domain_terms": entries
        }))
        .expect("M04 glossary fixture");
        let translated = translate_document(
            &case.source,
            doc,
            #[cfg(feature = "narrative")]
            None,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &glossary,
            None,
        )
        .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert!(
            translated.target_text.contains('，'),
            "{label}: source comma lost: {}",
            translated.target_text
        );
        assert_sentence_spans(&case.source, &translated, label);
        assert_eq!(
            translated.glossary_hits.len(),
            usize::from(date_locked) + usize::from(counterpart_locked),
            "{label}"
        );
        for (enabled, canonical, target, start, end) in [
            (date_locked, "明日", "隔天", 3, 5),
            (counterpart_locked, "保護者", "家長代表", 6, 9),
        ] {
            if !enabled {
                continue;
            }
            let hit = translated
                .glossary_hits
                .iter()
                .find(|hit| hit.canonical == canonical)
                .unwrap_or_else(|| panic!("{label}: missing {canonical}"));
            assert_eq!(hit.kind, GlossaryKind::DomainTerm, "{label}: {canonical}");
            assert_eq!(hit.sentence, Some(0), "{label}: {canonical}");
            assert_eq!(
                hit.source_span,
                Span {
                    start_char: start,
                    end_char: end
                },
                "{label}: {canonical}"
            );
            assert_eq!(slice_chars(&case.source, hit.source_span), canonical);
            assert_eq!(
                slice_chars(&translated.target_text, hit.target_span),
                target,
                "{label}: {canonical}"
            );
            assert_eq!(
                translated.target_text.matches(target).count(),
                1,
                "{label}: {canonical}"
            );
        }
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
