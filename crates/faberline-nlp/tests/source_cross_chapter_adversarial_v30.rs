//! Synthetic checks of written forms, caps and large chapter ordinals.
//! None of these checks resolves a person, grammar, quote or event.
use faberline_nlp::{
    Lexicon,
    source_cross_chapter::{CrossChapterCandidates, CrossChapterPolicy, Inspection, Unknown},
    text::Locale,
};

fn policy() -> CrossChapterPolicy {
    CrossChapterPolicy {
        max_chapter_bytes: 32_768,
        max_terms: 64,
        max_term_bytes: 512,
        history_distance: 4,
        max_names_per_chapter: 64,
        max_pronouns_per_chapter: 64,
        max_pairs_per_pronoun: 64,
        max_pairs_per_chapter: 256,
    }
}

#[test]
fn token_forms_inside_names_quotes_and_plural_text_remain_unverified() {
    for (locale, prior, current, expected_tag) in [
        (
            Locale::ChineseTaiwan,
            "𠮷e\u{301}。阿他走了。",
            "其他人說：「他們等他。」他。",
            "zh-Hant-TW",
        ),
        (
            Locale::ChineseChina,
            "𠮷e\u{301}。阿他走了。",
            "其他人说：“他们等他。”他。",
            "zh-Hans-CN",
        ),
    ] {
        let lexicon = Lexicon {
            actors: &[("阿他", "unverified")],
            ..Lexicon::default()
        };
        let mut stream = CrossChapterCandidates::new(locale, lexicon, policy()).unwrap();
        let first = stream.push(1, prior, "合成").unwrap();
        assert_eq!(first.names.len(), 1);
        assert_eq!(&prior[first.names[0].bytes.clone()], "阿他");
        let second = stream.push(2, current, "合成").unwrap();
        assert_eq!(second.locale, expected_tag);
        assert_eq!(
            second.pronoun_inspection_scope,
            "chinese_exact_token_form_subset"
        );
        assert_eq!(
            second.pronoun_forms,
            &[
                "他", "她", "它", "他們", "她們", "它們", "他们", "她们", "它们"
            ]
        );
        assert_eq!(second.pronouns_status, Inspection::Complete);
        assert_eq!(second.coverage, Unknown::Unknown);
        assert_eq!(second.identity_status, Unknown::Unknown);
        assert_eq!(second.association_status, Unknown::Unknown);
        // The separate final token is a non-vacuous control. We deliberately
        // do not assume how Jieba segments 其他 or either plural form.
        let final_byte = current.rfind('他').unwrap();
        assert!(
            second
                .pronouns
                .iter()
                .any(|p| p.source.bytes.start == final_byte && p.written_text == "他")
        );
        assert!(!second.candidates.is_empty());
        for pronoun in &second.pronouns {
            let span = &pronoun.source;
            assert_eq!(span.chapter, 2);
            assert_eq!(&current[span.bytes.clone()], pronoun.written_text);
            assert_eq!(
                current[..span.bytes.start].chars().count(),
                span.scalars.start_char
            );
            assert_eq!(
                current[..span.bytes.end].chars().count(),
                span.scalars.end_char
            );
            assert!(
                second
                    .pronoun_forms
                    .contains(&pronoun.written_text.as_str())
            );
            assert_eq!(pronoun.interpretation_status, Unknown::Unknown);
        }
        for candidate in &second.candidates {
            assert_eq!(&prior[candidate.prior_name.bytes.clone()], "阿他");
            assert_eq!(candidate.identity_status, Unknown::Unknown);
            assert_eq!(candidate.role_status, Unknown::Unknown);
            assert_eq!(candidate.association_status, Unknown::Unknown);
            assert_eq!(candidate.occurrence_status, Unknown::Unknown);
            assert_eq!(candidate.narrative_scope_status, Unknown::Unknown);
        }
    }
}

#[test]
fn identical_surface_with_distinct_labels_is_one_hit_at_exact_cap() {
    let lexicon = Lexicon {
        actors: &[("陳東", "first"), ("陳東", "second")],
        ..Lexicon::default()
    };
    let mut limit = policy();
    limit.max_names_per_chapter = 1;
    limit.max_pairs_per_pronoun = 1;
    limit.max_pairs_per_chapter = 1;
    let mut stream = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, limit).unwrap();
    let first = stream.push(1, "陳東。", "").unwrap();
    assert_eq!(first.names.len(), 1);
    assert_eq!(first.names_status, Inspection::Complete);
    assert_eq!(first.recheck_from_scalar, None);
    let second = stream.push(2, "他。", "").unwrap();
    assert_eq!(second.pronouns.len(), 1);
    assert_eq!(second.candidates.len(), 1);
    assert_eq!(second.pairs_status, Inspection::Complete);
    assert!(!second.pair_limit_reached);
    assert_eq!(second.candidates[0].prior_name, first.names[0]);
    assert_eq!(second.candidates[0].identity_status, Unknown::Unknown);
}

#[test]
fn maximum_ordinal_reports_gaps_and_serializes_without_overflow() {
    let lexicon = Lexicon {
        actors: &[("陳東", "a")],
        ..Lexicon::default()
    };
    let mut limit = policy();
    limit.history_distance = 2;
    let mut stream = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, limit).unwrap();
    let first = stream.push(u32::MAX - 1, "陳東。", "").unwrap();
    assert_eq!(
        first.missing_chapter_ranges,
        vec![u32::MAX - 3..=u32::MAX - 2]
    );
    let last = stream.push(u32::MAX, "他。", "").unwrap();
    assert_eq!(last.history_start_chapter, u32::MAX - 2);
    assert_eq!(
        last.missing_chapter_ranges,
        vec![u32::MAX - 2..=u32::MAX - 2]
    );
    assert_eq!(last.pronouns.len(), 1);
    assert_eq!(last.candidates.len(), 1);
    assert_eq!(last.candidates[0].prior_name.chapter, u32::MAX - 1);
    assert_eq!(last.pairs_status, Inspection::Partial);
    assert!(last.history_recheck_required);
    assert!(!last.pair_limit_reached);
    let encoded = serde_json::to_value(&last).unwrap();
    assert_eq!(encoded["chapter"].as_u64(), Some(u32::MAX as u64));
    assert_eq!(
        encoded["missing_chapter_ranges"].as_array().unwrap().len(),
        1
    );
    let gap = &encoded["missing_chapter_ranges"][0];
    assert_eq!(gap["start"].as_u64(), Some((u32::MAX - 2) as u64));
    assert_eq!(gap["end"].as_u64(), Some((u32::MAX - 2) as u64));
    assert_eq!(encoded["coverage"], "unknown");
    assert_eq!(encoded["candidates"][0]["association_status"], "unknown");
}

#[test]
fn cap_omission_after_empty_chapters_is_disclosed_only_if_real() {
    let lexicon = Lexicon {
        actors: &[("甲甲", "a"), ("乙乙", "b")],
        ..Lexicon::default()
    };
    let mut limit = policy();
    limit.max_pairs_per_pronoun = 1;
    let mut stream = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, limit).unwrap();
    stream.push(1, "甲甲。", "").unwrap();
    stream.push(2, "", "").unwrap();
    stream.push(3, "乙乙。", "").unwrap();
    stream.push(4, "", "").unwrap();
    let result = stream.push(5, "他。", "").unwrap();
    assert_eq!(result.pronouns.len(), 1);
    assert_eq!(result.candidates.len(), 1);
    assert_eq!(result.candidates[0].prior_name.chapter, 3);
    assert!(result.missing_chapter_ranges.is_empty());
    assert!(!result.history_incomplete);
    assert!(result.pair_limit_reached);
    assert_eq!(result.pairs_status, Inspection::Partial);
    assert_eq!(result.candidates[0].association_status, Unknown::Unknown);

    let mut exact = CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, limit).unwrap();
    exact.push(1, "甲甲。", "").unwrap();
    exact.push(2, "", "").unwrap();
    exact.push(3, "", "").unwrap();
    let result = exact.push(4, "他。", "").unwrap();
    assert_eq!(result.candidates.len(), 1);
    assert_eq!(result.pairs_status, Inspection::Complete);
    assert!(!result.pair_limit_reached);
}

#[test]
fn english_forms_are_explicitly_not_examined() {
    let lexicon = Lexicon {
        actors: &[("Alice", "a")],
        ..Lexicon::default()
    };
    let mut stream = CrossChapterCandidates::new(Locale::English, lexicon, policy()).unwrap();
    stream.push(1, "Alice left.", "").unwrap();
    let result = stream.push(2, "The other said he left.", "").unwrap();
    assert_eq!(result.locale, "en");
    assert_eq!(result.pronoun_inspection_scope, "not_examined");
    assert!(result.pronoun_forms.is_empty());
    assert_eq!(result.pronouns_status, Inspection::NotExamined);
    assert_eq!(result.pairs_status, Inspection::NotExamined);
    assert_eq!(result.coverage, Unknown::Unknown);
}
