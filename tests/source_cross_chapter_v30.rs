//! Synthetic source references only. These tests do not establish identity,
//! grammar, quote attribution, occurrence, or extraction completeness.
use faberline_nlp::{
    Lexicon, Span,
    source_cross_chapter::{
        Candidate, ChapterCandidates, CrossChapterCandidates, CrossChapterPolicy, Inspection,
        SourcePosition, Unknown,
    },
    text::Locale,
};

fn policy() -> CrossChapterPolicy {
    CrossChapterPolicy {
        max_chapter_bytes: 32_768,
        max_terms: 64,
        max_term_bytes: 256,
        history_distance: 2,
        max_names_per_chapter: 64,
        max_pronouns_per_chapter: 64,
        max_pairs_per_pronoun: 64,
        max_pairs_per_chapter: 256,
    }
}

fn reference<'a>(text: &'a str, source: &SourcePosition, chapter: u32) -> &'a str {
    assert_eq!(source.chapter, chapter);
    assert!(source.bytes.start < source.bytes.end && source.bytes.end <= text.len());
    assert!(text.is_char_boundary(source.bytes.start) && text.is_char_boundary(source.bytes.end));
    assert_eq!(
        text[..source.bytes.start].chars().count(),
        source.scalars.start_char
    );
    assert_eq!(
        text[..source.bytes.end].chars().count(),
        source.scalars.end_char
    );
    &text[source.bytes.clone()]
}

fn unknown(result: &ChapterCandidates) {
    assert_eq!(result.coverage, Unknown::Unknown);
    assert_eq!(result.source_binding_status, "caller_must_bind_exact_input");
    assert_eq!(result.identity_status, Unknown::Unknown);
    assert_eq!(result.role_status, Unknown::Unknown);
    assert_eq!(result.association_status, Unknown::Unknown);
    assert_eq!(result.occurrence_status, Unknown::Unknown);
    assert_eq!(result.narrative_scope_status, Unknown::Unknown);
    for pronoun in &result.pronouns {
        assert_eq!(pronoun.interpretation_status, Unknown::Unknown);
    }
    for candidate in &result.candidates {
        candidate_unknown(candidate);
    }
}

fn candidate_unknown(candidate: &Candidate) {
    assert_eq!(candidate.basis, "cooccurrence_in_configured_history_window");
    assert_eq!(candidate.identity_status, Unknown::Unknown);
    assert_eq!(candidate.role_status, Unknown::Unknown);
    assert_eq!(candidate.association_status, Unknown::Unknown);
    assert_eq!(candidate.occurrence_status, Unknown::Unknown);
    assert_eq!(candidate.narrative_scope_status, Unknown::Unknown);
}

#[test]
fn three_chapters_keep_original_unicode_locations_across_empty_result() {
    let lexicon = Lexicon {
        actors: &[("陳東", "a")],
        ..Lexicon::default()
    };
    let mut stream = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, policy()).unwrap();
    let one = "𠮷e\u{301}。陳東進入木屋。";
    let two = "天色漸暗。";
    let three = "次日，他離開木屋。";
    let first = stream.push(1, one, "").unwrap();
    assert_eq!(first.names.len(), 1);
    assert_eq!(reference(one, &first.names[0], 1), "陳東");
    assert_eq!(
        first.names[0].scalars,
        Span {
            start_char: 4,
            end_char: 6
        }
    );
    assert_eq!(first.names[0].bytes, 10..16);
    let middle = stream.push(2, two, "").unwrap();
    assert!(middle.names.is_empty());
    assert_eq!(middle.names_status, Inspection::Complete);
    let last = stream.push(3, three, "").unwrap();
    assert_eq!(
        last.pronouns.len(),
        1,
        "must actually inspect the written 他"
    );
    assert_eq!(last.pronouns[0].written_text, "他");
    assert_eq!(reference(three, &last.pronouns[0].source, 3), "他");
    assert_eq!(
        last.pronouns[0].source.scalars,
        Span {
            start_char: 3,
            end_char: 4
        }
    );
    assert_eq!(last.pronouns[0].source.bytes, 9..12);
    assert_eq!(last.candidates.len(), 1);
    assert_eq!(reference(one, &last.candidates[0].prior_name, 1), "陳東");
    assert_eq!(
        reference(three, &last.candidates[0].current_pronoun, 3),
        "他"
    );
    assert_eq!(last.pairs_status, Inspection::Complete);
    assert!(!last.history_incomplete);
    assert!(last.missing_chapter_ranges.is_empty());
    unknown(&first);
    unknown(&middle);
    unknown(&last);
}

#[test]
fn same_caller_label_does_not_merge_names_or_occurrences() {
    let lexicon = Lexicon {
        actors: &[("陳東", "same"), ("陳", "other"), ("岑棠", "same")],
        ..Lexicon::default()
    };
    let mut stream = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, policy()).unwrap();
    let previous = "陳東與岑棠相遇。陳東離開。";
    let first = stream.push(1, previous, "").unwrap();
    let surfaces: Vec<_> = first
        .names
        .iter()
        .map(|source| reference(previous, source, 1))
        .collect();
    assert_eq!(surfaces.iter().filter(|&&s| s == "陳東").count(), 2);
    assert_eq!(surfaces.iter().filter(|&&s| s == "陳").count(), 2);
    assert_eq!(surfaces.iter().filter(|&&s| s == "岑棠").count(), 1);
    let last = stream.push(2, "他站起。", "").unwrap();
    assert_eq!(last.pronouns.len(), 1);
    assert_eq!(last.candidates.len(), 5);
    assert_eq!(
        last.candidates
            .iter()
            .filter(|pair| reference(previous, &pair.prior_name, 1) == "陳東")
            .count(),
        2
    );
    assert_eq!(
        last.candidates
            .iter()
            .filter(|pair| reference(previous, &pair.prior_name, 1) == "岑棠")
            .count(),
        1
    );
    unknown(&last);
}

#[test]
fn excluded_title_html_and_footers_are_not_names_or_pronouns() {
    let lexicon = Lexicon {
        actors: &[("陳東", "a"), ("岑棠", "b")],
        footers: &["<end>"],
        line_footers: &["<line>"],
        ..Lexicon::default()
    };
    let mut stream = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, policy()).unwrap();
    let text = "陳東標題\n𠮷陳東進入木屋。<line>岑棠他\n岑棠離開木屋。<script>陳東他\n<end>陳東他";
    let result = stream.push(1, text, "陳東標題").unwrap();
    let names: Vec<_> = result
        .names
        .iter()
        .map(|source| reference(text, source, 1))
        .collect();
    assert_eq!(names, vec!["陳東", "岑棠"]);
    assert!(result.pronouns.is_empty());
    assert_eq!(result.names_status, Inspection::Complete);
    unknown(&result);
}

#[test]
fn missing_chapters_and_window_bounds_are_explicit() {
    let lexicon = Lexicon {
        actors: &[("陳東", "a")],
        ..Lexicon::default()
    };
    let mut stream = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, policy()).unwrap();
    stream.push(1, "陳東進入木屋。", "").unwrap();
    let gap = stream.push(3, "他離開木屋。", "").unwrap();
    assert_eq!(gap.pronouns.len(), 1);
    assert_eq!(
        gap.candidates.len(),
        1,
        "prior source remains within the numeric window"
    );
    assert!(gap.history_incomplete);
    assert!(gap.history_recheck_required);
    assert!(gap.partial_name_chapters.is_empty());
    assert!(!gap.pair_limit_reached);
    assert_eq!(gap.missing_chapter_ranges, vec![2..=2]);
    assert_eq!(gap.pairs_status, Inspection::Partial);
    assert_eq!(gap.recheck_from_scalar, Some(0));
    let later = stream.push(4, "他返回。", "").unwrap();
    assert_eq!(later.pronouns.len(), 1);
    assert!(
        later.candidates.is_empty(),
        "chapter 1 is outside the two-chapter window"
    );
    assert!(later.history_incomplete);
    assert_eq!(later.missing_chapter_ranges, vec![2..=2]);
    unknown(&gap);
    unknown(&later);
}

#[test]
fn caps_report_partial_without_turning_absence_into_complete_coverage() {
    let lexicon = Lexicon {
        actors: &[("陳東", "same"), ("岑棠", "same")],
        ..Lexicon::default()
    };
    let mut limited_names = policy();
    limited_names.max_names_per_chapter = 1;
    let mut stream =
        CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, limited_names).unwrap();
    let one = stream.push(1, "陳東與岑棠同行。", "").unwrap();
    assert_eq!(one.names.len(), 1);
    assert_eq!(one.names_status, Inspection::Partial);
    assert_eq!(one.recheck_from_scalar, Some(0));
    let two = stream.push(2, "他離開。", "").unwrap();
    assert_eq!(two.pronouns.len(), 1);
    assert!(two.history_incomplete);
    assert!(two.history_recheck_required);
    assert_eq!(two.partial_name_chapters, vec![1]);
    assert!(two.missing_chapter_ranges.is_empty());
    assert!(!two.pair_limit_reached);
    assert_eq!(two.pairs_status, Inspection::Partial);
    unknown(&one);
    unknown(&two);

    let mut limited_pronouns = policy();
    limited_pronouns.max_pronouns_per_chapter = 1;
    let mut stream =
        CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, limited_pronouns).unwrap();
    stream.push(1, "陳東。", "").unwrap();
    let capped = stream.push(2, "他離開。他回來。", "").unwrap();
    assert_eq!(capped.pronouns.len(), 1);
    assert_eq!(capped.pronouns_status, Inspection::Partial);
    assert_eq!(capped.recheck_from_scalar, Some(0));
    unknown(&capped);

    let mut limited_pairs = policy();
    limited_pairs.max_pairs_per_pronoun = 1;
    let mut stream =
        CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, limited_pairs).unwrap();
    stream.push(1, "陳東與岑棠。", "").unwrap();
    let capped = stream.push(2, "他離開。", "").unwrap();
    assert_eq!(capped.pronouns.len(), 1);
    assert_eq!(capped.candidates.len(), 1);
    assert_eq!(capped.pairs_status, Inspection::Partial);
    assert!(capped.pair_limit_reached);
    assert_eq!(capped.recheck_from_scalar, Some(0));
    unknown(&capped);

    let mut limited_total = policy();
    limited_total.max_pairs_per_chapter = 1;
    let mut stream =
        CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, limited_total).unwrap();
    stream.push(1, "陳東。", "").unwrap();
    let capped = stream.push(2, "他離開。她回來。", "").unwrap();
    assert_eq!(capped.pronouns.len(), 2);
    assert_eq!(capped.candidates.len(), 1);
    assert_eq!(capped.pairs_status, Inspection::Partial);
    assert!(capped.pair_limit_reached);
    assert_eq!(capped.recheck_from_scalar, Some(0));
    unknown(&capped);

    // Reaching a cap is not itself proof that anything was omitted.
    let mut exact = policy();
    exact.max_names_per_chapter = 1;
    exact.max_pronouns_per_chapter = 1;
    exact.max_pairs_per_pronoun = 1;
    exact.max_pairs_per_chapter = 1;
    let one_name = Lexicon {
        actors: &[("陳東", "a")],
        ..Lexicon::default()
    };
    let mut stream = CrossChapterCandidates::new(Locale::ChineseTaiwan, one_name, exact).unwrap();
    let first = stream.push(1, "陳東。", "").unwrap();
    let last = stream.push(2, "他離開。", "").unwrap();
    assert_eq!(first.names.len(), 1);
    assert_eq!(first.names_status, Inspection::Complete);
    assert_eq!(last.pronouns.len(), 1);
    assert_eq!(last.pronouns_status, Inspection::Complete);
    assert_eq!(last.candidates.len(), 1);
    assert_eq!(last.pairs_status, Inspection::Complete);
    assert!(!last.pair_limit_reached);
    assert_eq!(last.recheck_from_scalar, None);
    unknown(&last);
}

#[test]
fn chapter_window_discloses_earlier_history_without_claiming_a_gap() {
    let lexicon = Lexicon {
        actors: &[("陳東", "a")],
        ..Lexicon::default()
    };
    let mut limits = policy();
    limits.history_distance = 1;
    let mut stream = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, limits).unwrap();
    stream.push(1, "陳東進入木屋。", "").unwrap();
    stream.push(2, "天色漸暗。", "").unwrap();
    let third = stream.push(3, "他離開木屋。", "").unwrap();
    assert_eq!(third.policy, limits);
    assert_eq!(third.history_distance, 1);
    assert_eq!(third.history_start_chapter, 2);
    assert!(third.earlier_chapters_outside_window);
    assert!(!third.history_incomplete);
    assert!(!third.history_recheck_required);
    assert!(third.partial_name_chapters.is_empty());
    assert!(third.missing_chapter_ranges.is_empty());
    assert!(!third.pair_limit_reached);
    assert!(third.candidates.is_empty());
    assert_eq!(third.pronouns.len(), 1);
    assert_eq!(third.pairs_status, Inspection::Complete);
    unknown(&third);
}

#[test]
fn first_observed_chapter_reports_missing_in_window_not_all_earlier_chapters() {
    let lexicon = Lexicon {
        actors: &[("陳東", "a")],
        ..Lexicon::default()
    };
    let mut stream = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, policy()).unwrap();
    let result = stream.push(5, "他離開。", "").unwrap();
    assert_eq!(result.history_start_chapter, 3);
    assert!(result.earlier_chapters_outside_window);
    assert_eq!(result.missing_chapter_ranges, vec![3..=4]);
    assert!(result.history_recheck_required);
    assert!(result.partial_name_chapters.is_empty());
    assert!(!result.pair_limit_reached);
    assert_eq!(result.pronouns.len(), 1);
    assert!(result.candidates.is_empty());
    assert_eq!(result.pairs_status, Inspection::Partial);
    assert_eq!(result.recheck_from_scalar, Some(0));
    unknown(&result);
}

#[test]
fn quoted_and_simplified_written_tokens_do_not_establish_identity() {
    for (locale, name, previous, next) in [
        (
            Locale::ChineseTaiwan,
            "陳東",
            "陳東進入木屋。",
            "她說：「他離開。」",
        ),
        (
            Locale::ChineseChina,
            "陈东",
            "陈东进入木屋。",
            "她说：“他离开。”",
        ),
    ] {
        let actors = [(name, "shared")];
        let lexicon = Lexicon {
            actors: &actors,
            ..Lexicon::default()
        };
        let mut stream = CrossChapterCandidates::new(locale, lexicon, policy()).unwrap();
        stream.push(1, previous, "").unwrap();
        let current = stream.push(2, next, "").unwrap();
        assert!(current.pronouns.iter().any(|p| p.written_text == "她"));
        assert!(current.pronouns.iter().any(|p| p.written_text == "他"));
        for pronoun in &current.pronouns {
            assert_eq!(reference(next, &pronoun.source, 2), pronoun.written_text);
        }
        for pair in &current.candidates {
            assert_eq!(reference(previous, &pair.prior_name, 1), name);
            assert!(["她", "他"].contains(&reference(next, &pair.current_pronoun, 2)));
        }
        assert!(current.candidates.len() >= 2);
        unknown(&current);
    }
}

#[test]
fn unsupported_english_is_not_reported_as_no_pronouns() {
    let lexicon = Lexicon {
        actors: &[("Alice", "a")],
        ..Lexicon::default()
    };
    let mut stream = CrossChapterCandidates::new(Locale::English, lexicon, policy()).unwrap();
    stream.push(1, "Alice entered the room.", "").unwrap();
    let result = stream.push(2, "He left the room.", "").unwrap();
    assert!(result.pronouns.is_empty());
    assert!(result.candidates.is_empty());
    assert_eq!(result.pronouns_status, Inspection::NotExamined);
    assert_eq!(result.pairs_status, Inspection::NotExamined);
    unknown(&result);
}

#[test]
fn failed_pushes_leave_history_unchanged() {
    let lexicon = Lexicon {
        actors: &[("陳東", "a")],
        ..Lexicon::default()
    };
    let mut limits = policy();
    limits.max_chapter_bytes = 32;
    let mut tested = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, limits).unwrap();
    let mut clean = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, limits).unwrap();
    tested.push(1, "陳東進入木屋。", "").unwrap();
    clean.push(1, "陳東進入木屋。", "").unwrap();
    assert!(tested.push(0, "他。", "").is_err());
    assert!(tested.push(1, "他。", "").is_err());
    assert!(tested.push(2, &"甲".repeat(20), "").is_err());
    let actual = tested.push(2, "他離開。", "").unwrap();
    let expected = clean.push(2, "他離開。", "").unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.pronouns.len(), 1);
    assert_eq!(actual.candidates.len(), 1);
    unknown(&actual);
}

#[test]
fn invalid_policy_and_caller_term_limits_are_rejected() {
    let lexicon = Lexicon {
        actors: &[("陳東", "a"), ("岑棠", "b")],
        ..Lexicon::default()
    };
    let mut zero = policy();
    zero.history_distance = 0;
    assert!(CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, zero).is_err());
    let mut terms = policy();
    terms.max_terms = 1;
    assert!(CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, terms).is_err());
    let mut bytes = policy();
    bytes.max_term_bytes = 5;
    assert!(CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, bytes).is_err());
    let empty_surface = Lexicon {
        actors: &[("", "a")],
        ..Lexicon::default()
    };
    assert!(CrossChapterCandidates::new(Locale::ChineseTaiwan, empty_surface, policy()).is_err());
}
