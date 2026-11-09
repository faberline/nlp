//! Synthetic cases for the scoped chapter-footer policy.
//! No source text or marker in this file comes from a real work.

use faberline_nlp::{
    Lexicon,
    source_scope::{SourceScope, SourceSegment},
};

fn assert_original_range(source: &str, segment: &SourceSegment<'_>, expected: &str) {
    let range = segment.source;
    assert!(range.byte_start < range.byte_end);
    assert!(range.scalars.start_char < range.scalars.end_char);
    assert_eq!(source.get(range.byte_start..range.byte_end), Some(expected));
    assert_eq!(segment.text, expected);
    assert_eq!(
        source[..range.byte_start].chars().count(),
        range.scalars.start_char
    );
    assert_eq!(
        source[..range.byte_end].chars().count(),
        range.scalars.end_char
    );
}

#[test]
fn chapter_footer_folds_ascii_in_both_marker_and_source() {
    for (marker, written) in [("<END>", "<end>"), ("<end>", "<END>"), ("<eNd>", "<EnD>")] {
        let source = format!("𠮷e\u{301}甲。{written}不可見。");
        let lexicon = Lexicon {
            footers: &[marker],
            ..Lexicon::default()
        };
        let scope = SourceScope::new(&source, "", &lexicon);
        assert_eq!(
            scope.segments().len(),
            1,
            "marker={marker}, written={written}"
        );
        assert_original_range(&source, &scope.segments()[0], "𠮷e\u{301}甲。");
        assert_eq!(
            scope.segments()[0].source.byte_end,
            source.find(written).unwrap()
        );
        assert_eq!(scope.coverage(), "unknown");
        assert_eq!(scope.cross_segment_status(), "not_examined");
    }
}

#[test]
fn chapter_footer_can_span_lines_without_losing_original_offsets() {
    let source = "合成標題\r\n𠮷e\u{301}甲\r\n乙<eNd\r\n>不可見。";
    let lexicon = Lexicon {
        footers: &["<EnD\r\n>"],
        ..Lexicon::default()
    };
    let scope = SourceScope::new(source, "合成標題", &lexicon);
    assert_eq!(scope.segments().len(), 1);
    let segment = &scope.segments()[0];
    assert_original_range(source, segment, "𠮷e\u{301}甲\r\n乙");
    assert_eq!(segment.source.byte_start, source.find('𠮷').unwrap());
    assert_eq!(segment.source.byte_end, source.find("<eNd\r\n>").unwrap());
}

#[test]
fn only_ascii_is_case_folded_in_a_mixed_unicode_marker() {
    let source = "𠮷e\u{301}甲<énd>乙<Énd>隱藏";
    let lexicon = Lexicon {
        footers: &["<ÉND>"],
        ..Lexicon::default()
    };
    let scope = SourceScope::new(source, "", &lexicon);
    assert_eq!(scope.segments().len(), 1);
    assert_original_range(source, &scope.segments()[0], "𠮷e\u{301}甲<énd>乙");
    assert_eq!(
        scope.segments()[0].source.byte_end,
        source.find("<Énd>").unwrap()
    );
}

#[test]
fn line_markers_remain_case_sensitive_and_break_segments() {
    let source = "𠮷甲<LINE>仍可見\r\n乙<line>隱藏\r\n丙";
    let lexicon = Lexicon {
        line_footers: &["<line>"],
        ..Lexicon::default()
    };
    let scope = SourceScope::new(source, "", &lexicon);
    assert_eq!(scope.segments().len(), 2);
    assert_original_range(source, &scope.segments()[0], "𠮷甲<LINE>仍可見\r\n乙");
    assert_original_range(source, &scope.segments()[1], "丙");
    assert!(scope.segments()[0].source.byte_end < scope.segments()[1].source.byte_start);
}

#[test]
fn html_markers_remain_case_sensitive_and_first_footer_wins() {
    let source = "𠮷甲<SCRIPT>仍可見\n乙<script>隱藏\n丙<END>尾<end>更後";
    let lexicon = Lexicon {
        footers: &["<eNd>", "<EN"],
        ..Lexicon::default()
    };
    let scope = SourceScope::new(source, "", &lexicon);
    assert_eq!(scope.segments().len(), 2);
    assert_original_range(source, &scope.segments()[0], "𠮷甲<SCRIPT>仍可見\n乙");
    assert_original_range(source, &scope.segments()[1], "丙");
    assert_eq!(
        scope.segments()[1].source.byte_end,
        source.find("<END>").unwrap()
    );
}

#[test]
fn empty_chapter_footer_retains_existing_full_exclusion() {
    let lexicon = Lexicon {
        footers: &[""],
        ..Lexicon::default()
    };
    let scope = SourceScope::new("𠮷e\u{301}甲", "", &lexicon);
    assert!(scope.segments().is_empty());
}
