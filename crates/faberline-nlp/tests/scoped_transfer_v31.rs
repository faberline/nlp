//! Synthetic source-scope checks for unverified written transfer candidates.
use faberline_nlp::{
    Lexicon, NarrativeAnalyzer, WrittenTransferUnknown,
    scoped_transfer::{self, ScopedTransferReport},
    source_scope::SourceScope,
    text::Locale,
    written_speech_alias_evidence::SourceRange,
};

fn lexicon() -> Lexicon<'static> {
    Lexicon {
        actors: &[("陳東", "a"), ("岑棠", "b"), ("寧衡", "c")],
        objects: &[("劍", "sword"), ("令牌", "item")],
        footers: &["<end>"],
        line_footers: &["<line>"],
        ..Lexicon::default()
    }
}

fn original_at(source: &str, span: SourceRange) -> &str {
    assert!(span.byte_start < span.byte_end);
    let fragment = source
        .get(span.byte_start..span.byte_end)
        .expect("original UTF-8 boundaries");
    assert_eq!(
        source[..span.byte_start].chars().count(),
        span.scalars.start_char
    );
    assert_eq!(
        source[..span.byte_end].chars().count(),
        span.scalars.end_char
    );
    fragment
}

fn inside_report_segment(
    source: &str,
    report: &ScopedTransferReport,
    segment_id: usize,
    range: SourceRange,
) {
    let segment = report
        .segments
        .get(segment_id)
        .expect("segment id must be valid");
    original_at(source, *segment);
    original_at(source, range);
    assert!(segment.byte_start <= range.byte_start && range.byte_end <= segment.byte_end);
    assert!(
        segment.scalars.start_char <= range.scalars.start_char
            && range.scalars.end_char <= segment.scalars.end_char
    );
}

fn run(source: &str) -> ScopedTransferReport {
    let terms = lexicon();
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &terms).unwrap();
    let scope = SourceScope::new(source, "合成", &terms);
    scoped_transfer::inspect(&scope, &analyzer).unwrap()
}

fn unknown(report: &ScopedTransferReport) {
    assert_eq!(report.coverage, "partial_heuristic");
    assert_eq!(report.semantic_status, WrittenTransferUnknown::Unknown);
    assert_eq!(report.source_binding_status, "caller_must_bind_exact_input");
    assert_eq!(report.cross_segment_associations, "not_examined");
    for cue in &report.cues {
        assert_eq!(cue.direction_status, "written_form_only");
        for state in [
            cue.identity_status,
            cue.role_status,
            cue.occurrence_status,
            cue.association_status,
            cue.narrative_scope_status,
        ] {
            assert_eq!(state, WrittenTransferUnknown::Unknown);
        }
    }
    for role in &report.roles {
        for state in [
            role.identity_status,
            role.role_status,
            role.occurrence_status,
            role.association_status,
            role.narrative_scope_status,
        ] {
            assert_eq!(state, WrittenTransferUnknown::Unknown);
        }
    }
    for role in &report.alternatives {
        for state in [
            role.boundary_status,
            role.interpretation_status,
            role.identity_status,
            role.role_status,
            role.occurrence_status,
            role.association_status,
            role.narrative_scope_status,
        ] {
            assert_eq!(state, WrittenTransferUnknown::Unknown);
        }
    }
}

#[test]
fn excluded_closer_does_not_change_body_role_candidates() {
    let plain = "𠮷e\u{301}。陳東把劍交給岑棠。<end>甲";
    let closer = "𠮷e\u{301}。陳東把劍交給岑棠。<end>』";
    let a = run(plain);
    let b = run(closer);
    unknown(&a);
    unknown(&b);
    assert_eq!(a.cues.len(), 1);
    assert_eq!(b.cues.len(), 1);
    assert_eq!(a.roles.len(), 1);
    assert_eq!(b.roles.len(), 1);
    let ar = &a.roles[0];
    let br = &b.roles[0];
    assert_eq!(
        (ar.cue, ar.construction, ar.giver, ar.item, ar.recipient),
        (br.cue, br.construction, br.giver, br.item, br.recipient)
    );
    assert_eq!(original_at(plain, ar.cue), "交給");
    assert_eq!(original_at(plain, ar.giver.expect("visible giver")), "陳東");
    assert_eq!(original_at(plain, ar.item.expect("visible item")), "劍");
    assert_eq!(
        original_at(plain, ar.recipient.expect("visible recipient")),
        "岑棠"
    );
    assert_eq!(original_at(closer, br.cue), "交給");
}

#[test]
fn excluded_footer_cues_are_absent_and_later_visible_line_maps_to_original() {
    let source =
        "𠮷e\u{301}。陳東把劍交給岑棠。<line>他交給寧衡。\r\n陳東把劍交給岑棠。<end>他交給寧衡。";
    let terms = lexicon();
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &terms).unwrap();
    let scope = SourceScope::new(source, "合成", &terms);
    assert_eq!(scope.segments().len(), 2);
    let report = scoped_transfer::inspect(&scope, &analyzer).unwrap();
    unknown(&report);
    assert_eq!(
        report.segments,
        scope
            .segments()
            .iter()
            .map(|segment| segment.source)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        report.cues.len(),
        2,
        "only the two visible cues are eligible"
    );
    assert_eq!(report.roles.len(), 2);
    assert_eq!(report.alternatives.len(), 2);
    let line_cut = source.find("<line>").unwrap();
    let chapter_cut = source.find("<end>").unwrap();
    let later_start = source.find("\n陳東").unwrap() + 1;
    for (index, cue) in report.cues.iter().enumerate() {
        inside_report_segment(source, &report, cue.segment_id, cue.cue);
        assert_eq!(original_at(source, cue.cue), "交給");
        if index == 0 {
            assert!(cue.cue.byte_end <= line_cut);
        } else {
            assert!(cue.cue.byte_start >= later_start && cue.cue.byte_end <= chapter_cut);
        }
    }
    for role in &report.roles {
        inside_report_segment(source, &report, role.segment_id, role.cue);
        inside_report_segment(source, &report, role.segment_id, role.construction);
        assert!(
            report
                .cues
                .iter()
                .any(|cue| cue.segment_id == role.segment_id && cue.cue == role.cue)
        );
        assert_eq!(original_at(source, role.cue), "交給");
        original_at(source, role.construction);
        for range in [role.giver, role.item, role.recipient]
            .into_iter()
            .flatten()
        {
            inside_report_segment(source, &report, role.segment_id, range);
            original_at(source, range);
            assert!(
                range.byte_end <= line_cut
                    || (range.byte_start >= later_start && range.byte_end <= chapter_cut)
            );
        }
    }
    for group in &report.alternatives {
        inside_report_segment(source, &report, group.segment_id, group.cue);
        inside_report_segment(source, &report, group.segment_id, group.construction);
        assert!(
            report
                .cues
                .iter()
                .any(|cue| cue.segment_id == group.segment_id && cue.cue == group.cue)
        );
        assert_eq!(original_at(source, group.cue), "交給");
        original_at(source, group.construction);
        for &range in group
            .giver_alternatives
            .iter()
            .chain(&group.recipient_alternatives)
        {
            inside_report_segment(source, &report, group.segment_id, range);
            original_at(source, range);
            assert!(
                range.byte_end <= line_cut
                    || (range.byte_start >= later_start && range.byte_end <= chapter_cut)
            );
        }
    }
    // The previous API still inspects the full text; it remains opt-in raw behavior.
    let raw = analyzer.discover_written_transfer_cues(source);
    assert_eq!(
        raw.len(),
        4,
        "raw API continues to include both footer cues"
    );
}

#[test]
fn excluded_gap_cannot_join_two_written_characters_into_a_cue() {
    let source = "𠮷。陳東把劍交<line>被刪掉\n給岑棠。";
    let terms = lexicon();
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &terms).unwrap();
    let scope = SourceScope::new(source, "合成", &terms);
    assert_eq!(scope.segments().len(), 2);
    let report = scoped_transfer::inspect(&scope, &analyzer).unwrap();
    unknown(&report);
    assert!(
        report.cues.is_empty(),
        "separate source segments must not be concatenated"
    );
    assert!(report.roles.is_empty());
    assert!(report.alternatives.is_empty());
}
