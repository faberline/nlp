//! Synthetic scope checks. Written spans do not establish speakers or events.
use faberline_nlp::{
    Lexicon, NarrativeAnalyzer,
    safe_written_observations::{QuotationForm, Unknown, WrittenQuote},
    scoped_written_observations::{self, Error, ScopedReport},
    text::Locale,
    written_speech_alias_evidence::SourceRange,
};

fn at(source: &str, range: SourceRange) -> &str {
    assert!(range.byte_start < range.byte_end);
    let text = source
        .get(range.byte_start..range.byte_end)
        .expect("original UTF-8 boundaries");
    assert_eq!(
        source[..range.byte_start].chars().count(),
        range.scalars.start_char
    );
    assert_eq!(
        source[..range.byte_end].chars().count(),
        range.scalars.end_char
    );
    text
}

fn quote_ranges(source: &str, quote: &WrittenQuote, parent: SourceRange) {
    let _ = at(source, quote.source);
    assert!(
        parent.byte_start <= quote.source.byte_start && quote.source.byte_end <= parent.byte_end
    );
    assert_eq!(quote.delimiter_status, Unknown::Unknown);
    assert_eq!(quote.speaker_identity, Unknown::Unknown);
    assert_eq!(quote.speaker_role, Unknown::Unknown);
    assert_eq!(quote.occurrence, Unknown::Unknown);
    assert_eq!(quote.narrative_scope, Unknown::Unknown);
    if let Some(inner) = quote.inner {
        let _ = at(source, inner);
        assert!(
            quote.source.byte_start <= inner.byte_start && inner.byte_end <= quote.source.byte_end
        );
    }
    for child in &quote.nested_quotations {
        quote_ranges(source, child, quote.source);
    }
}

fn check_report(source: &str, scoped: &ScopedReport<'_>) {
    assert_eq!(scoped.scope_coverage, "partial_or_unknown");
    assert_eq!(scoped.cross_segment_associations, "not_examined");
    assert_eq!(scoped.source_identity, Unknown::Unknown);
    assert_eq!(scoped.coordinate_binding, "caller_must_bind_exact_input");
    let mut end = 0;
    for (id, segment) in scoped.segments.iter().enumerate() {
        assert_eq!(segment.segment_id, id);
        assert_eq!(segment.index_scope, "segment");
        assert_eq!(segment.boundary_scope, "hard_break");
        assert!(end <= segment.source.byte_start);
        let _ = at(source, segment.source);
        end = segment.source.byte_end;
        let r = &segment.report;
        assert_eq!(r.source_identity, Unknown::Unknown);
        assert_eq!(r.coordinate_binding, "caller_must_bind_exact_input");
        for quote in &r.quotations {
            quote_ranges(source, quote, segment.source);
        }
        for alt in &r.alternative_quote_boundaries {
            let _ = at(source, alt.source);
        }
        for diag in &r.quote_scan_diagnostics {
            let _ = at(source, diag.source);
        }
        for speech in &r.speech {
            let _ = at(source, speech.written.cue);
            let _ = at(source, speech.written.payload);
            if let Some(subject) = speech.written.written_speaker {
                let _ = at(source, subject);
            }
            assert_eq!(speech.written.identity, "unknown");
            assert_eq!(speech.written.occurrence, "unknown");
            assert_eq!(speech.role, Unknown::Unknown);
            assert_eq!(speech.quote_attribution, Unknown::Unknown);
            if let Some(index) = speech.potential_quote_index {
                assert!(index < r.quotations.len());
            }
        }
        for cue in &r.written_negated_cues {
            let _ = at(source, cue.cue);
            let _ = at(source, cue.negation);
            if let Some(subject) = cue.written_subject {
                let _ = at(source, subject);
            }
            assert!(cue.potential_quote_index < r.quotations.len());
            assert_eq!(cue.identity, Unknown::Unknown);
            assert_eq!(cue.occurrence, Unknown::Unknown);
        }
        for cessation in &r.cessation {
            for local in [
                cessation.written.tag,
                cessation.written.subject_surface,
                cessation.written.cue,
            ] {
                let _ = at(
                    source,
                    SourceRange {
                        scalars: local.scalars,
                        byte_start: local.byte_start,
                        byte_end: local.byte_end,
                    },
                );
            }
            assert!(cessation.preceding_quote_index < r.quotations.len());
            assert_eq!(cessation.identity, Unknown::Unknown);
            assert_eq!(cessation.occurrence, Unknown::Unknown);
        }
        for diag in &r.cessation_diagnostics {
            let _ = at(source, diag.candidate_source);
            assert!(diag.preceding_quote_index < r.quotations.len());
        }
        for claim in &r.written_claims {
            let _ = at(source, claim.source);
            assert!(claim.potential_quote_index < r.quotations.len());
            assert!(claim.potential_cessation_index < r.cessation.len());
            assert_eq!(claim.occurrence, Unknown::Unknown);
            assert_eq!(claim.quote_attribution, Unknown::Unknown);
        }
    }
}

fn inspect<'a>(source: &str, lexicon: &'a Lexicon<'a>) -> ScopedReport<'a> {
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, lexicon).unwrap();
    scoped_written_observations::inspect_with_analyzer(source, "", &analyzer, lexicon).unwrap()
}

#[test]
fn hidden_footer_quotes_and_closer_do_not_change_body_report() {
    let vocabulary = Lexicon {
        actors: &[("陳東", "a")],
        objects: &[("令牌", "item")],
        footers: &["<end>"],
        ..Lexicon::default()
    };
    let normal = "𠮷e\u{301}。陳東說道：「我取出了令牌。」<end>甲";
    let stray = "𠮷e\u{301}。陳東說道：「我取出了令牌。」<end>』";
    let hidden_quote = "𠮷e\u{301}。陳東說道：「我取出了令牌。」<end>「隱藏。」";
    let a = inspect(normal, &vocabulary);
    let b = inspect(stray, &vocabulary);
    let c = inspect(hidden_quote, &vocabulary);
    check_report(normal, &a);
    check_report(stray, &b);
    check_report(hidden_quote, &c);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert_eq!(format!("{a:?}"), format!("{c:?}"));
    assert_eq!(a.segments.len(), 1);
    assert_eq!(a.segments[0].report.quotations.len(), 1);
    assert_eq!(
        a.segments[0].report.speech.len(),
        1,
        "visible speech must not disappear"
    );
    assert_eq!(
        at(normal, a.segments[0].report.quotations[0].source),
        "「我取出了令牌。」"
    );
}

#[test]
fn hidden_gap_is_a_hard_boundary_not_a_quote_join() {
    let vocabulary = Lexicon {
        actors: &[("陳東", "a")],
        line_footers: &["<line>"],
        ..Lexicon::default()
    };
    let source = "𠮷。陳東說道：「前文<line>隱藏\n後文。」";
    let scoped = inspect(source, &vocabulary);
    check_report(source, &scoped);
    assert_eq!(scoped.segments.len(), 2);
    let first = &scoped.segments[0].report.quotations;
    let second = &scoped.segments[1].report.quotations;
    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);
    assert_eq!(first[0].form, QuotationForm::IncompleteOrDamaged);
    assert_eq!(second[0].form, QuotationForm::UnmatchedCloser);
    assert_eq!(at(source, first[0].source), "「前文");
    assert_eq!(at(source, second[0].source), "」");
}

#[test]
fn contiguous_visible_lines_keep_the_written_multiline_quote() {
    let vocabulary = Lexicon::default();
    let source = "𠮷e\u{301}。「前文\r\n後文。」";
    let scoped = inspect(source, &vocabulary);
    check_report(source, &scoped);
    assert_eq!(scoped.segments.len(), 1);
    assert_eq!(scoped.segments[0].report.quotations.len(), 1);
    let quote = &scoped.segments[0].report.quotations[0];
    assert_eq!(quote.form, QuotationForm::Complete);
    assert_eq!(at(source, quote.source), "「前文\r\n後文。」");
}

#[test]
fn segment_local_quote_indices_and_all_evidence_are_preserved() {
    let vocabulary = Lexicon {
        actors: &[("陳東", "a"), ("顧梨", "b")],
        objects: &[("令牌", "item"), ("紙條", "paper")],
        line_footers: &["<line>"],
        ..Lexicon::default()
    };
    let source = concat!(
        "𠮷。陳東說道：「我取出了令牌。」<line>隱藏\n",
        "「門邊風起，讓我碰巧取出了一張紙條……」顧梨停了一會，沒再說下去。<line>隱藏\n",
        "「甲『乙』丙。」<line>隱藏\n",
        "陳東沒有說道：「我取出了令牌。」"
    );
    let scoped = inspect(source, &vocabulary);
    check_report(source, &scoped);
    assert_eq!(scoped.segments.len(), 4);
    let first = &scoped.segments[0].report;
    assert_eq!(first.speech.len(), 1);
    assert_eq!(first.speech[0].potential_quote_index, Some(0));
    let second = &scoped.segments[1].report;
    assert_eq!(second.cessation.len(), 1);
    assert_eq!(second.cessation[0].preceding_quote_index, 0);
    assert_eq!(second.written_claims.len(), 1);
    assert_eq!(second.written_claims[0].potential_quote_index, 0);
    assert_eq!(second.written_claims[0].potential_cessation_index, 0);
    assert_eq!(
        at(source, second.written_claims[0].source),
        "讓我碰巧取出了一張紙條……"
    );
    let tag = second.cessation[0].written.tag;
    assert_eq!(
        &source[tag.byte_start..tag.byte_end],
        "顧梨停了一會，沒再說下去。"
    );
    let third = &scoped.segments[2].report;
    assert_eq!(third.quotations.len(), 1);
    assert_eq!(third.quotations[0].nested_quotations.len(), 1);
    assert_eq!(
        at(source, third.quotations[0].nested_quotations[0].source),
        "『乙』"
    );
    let fourth = &scoped.segments[3].report;
    assert_eq!(fourth.written_negated_cues.len(), 1);
    assert_eq!(fourth.written_negated_cues[0].potential_quote_index, 0);
}

#[test]
fn empty_scope_rejects_analyzer_from_a_different_vocabulary() {
    let first = Lexicon {
        actors: &[("陳東", "a")],
        footers: &["<end>"],
        ..Lexicon::default()
    };
    let different = Lexicon {
        actors: &[("岑棠", "b")],
        footers: &["<end>"],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &first).unwrap();
    assert!(matches!(
        scoped_written_observations::inspect_with_analyzer("<end>隱藏", "", &analyzer, &different),
        Err(Error::VocabularyMismatch)
    ));
}
