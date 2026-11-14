//! Synthetic comparison against the pre-index `str::match_indices` rule.
use super::{Lexicon, MAX_OBJECT_COVERAGE_STARTS, ObjectCoverage};

fn original_scan(text: &str, lexicon: &Lexicon<'_>, start: usize, end: usize) -> bool {
    lexicon.objects.iter().any(|&(object, _)| {
        !object.is_empty()
            && object.len() > end - start
            && text
                .match_indices(object)
                .any(|(left, _)| left <= start && end <= left + object.len())
    })
}

fn compare_all_ranges(text: &str, objects: &[(&str, &str)]) {
    let lexicon = Lexicon {
        objects,
        ..Lexicon::default()
    };
    let coverage = ObjectCoverage::new(text, &lexicon);
    assert!(
        matches!(&coverage, ObjectCoverage::Indexed(_)),
        "small input must use index"
    );
    let boundaries: Vec<_> = text
        .char_indices()
        .map(|(byte, _)| byte)
        .chain(std::iter::once(text.len()))
        .collect();
    for (left_index, &start) in boundaries.iter().enumerate() {
        for &end in &boundaries[left_index + 1..] {
            assert_eq!(
                coverage.covers(text, &lexicon, start, end),
                original_scan(text, &lexicon, start, end),
                "different coverage: text={text:?}, objects={objects:?}, bytes={start}..{end}"
            );
        }
    }
}

#[test]
fn index_matches_original_scan_for_every_small_utf8_range() {
    let words = ["", "甲", "甲乙", "乙甲", "𠮷甲", "aa", "aaa"];
    for mask in 0..(1usize << words.len()) {
        let objects: Vec<_> = words
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, &word)| (word, "unverified"))
            .collect();
        for text in ["𠮷甲乙甲", "甲甲甲", "aaaaa", "e\u{301}甲乙"] {
            compare_all_ranges(text, &objects);
        }
    }
    // Repeated caller entries do not alter the written-match rule.
    compare_all_ranges(
        "甲乙甲乙",
        &[("甲乙", "one"), ("甲乙", "two"), ("乙", "three")],
    );
}

#[test]
fn same_surface_partial_overlap_and_nonoverlapping_matches_follow_original_rule() {
    let objects = [("甲甲", "unverified")];
    let lexicon = Lexicon {
        objects: &objects,
        ..Lexicon::default()
    };
    let text = "甲甲甲";
    let coverage = ObjectCoverage::new(text, &lexicon);
    let single = "甲".len();
    assert!(
        !coverage.covers(text, &lexicon, 0, 2 * single),
        "equal range is not a longer object"
    );
    assert!(
        coverage.covers(text, &lexicon, single, 2 * single),
        "the first object contains this scalar"
    );
    // str::match_indices returns non-overlapping matches; the possible second
    // overlapping occurrence is intentionally not invented by the index.
    assert!(!coverage.covers(text, &lexicon, 2 * single, 3 * single));
    compare_all_ranges(text, &objects);
}

#[test]
fn coverage_at_and_above_the_unique_start_limit_matches_original_scan() {
    let objects = [("甲", "single"), ("甲甲", "pair")];
    let lexicon = Lexicon {
        objects: &objects,
        ..Lexicon::default()
    };
    let width = "甲".len();
    for count in [MAX_OBJECT_COVERAGE_STARTS, MAX_OBJECT_COVERAGE_STARTS + 1] {
        let text = "甲".repeat(count);
        let coverage = ObjectCoverage::new(&text, &lexicon);
        if count == MAX_OBJECT_COVERAGE_STARTS {
            assert!(
                matches!(&coverage, ObjectCoverage::Indexed(_)),
                "the exact limit stays indexed"
            );
        } else {
            assert!(
                matches!(&coverage, ObjectCoverage::Scan),
                "above the limit must use exact scanning"
            );
        }
        for (start, end) in [
            (0, width),
            (width, 2 * width),
            (0, 2 * width),
            ((count - 2) * width, (count - 1) * width),
            ((count - 1) * width, count * width),
        ] {
            assert_eq!(
                coverage.covers(&text, &lexicon, start, end),
                original_scan(&text, &lexicon, start, end),
                "count={count}, bytes={start}..{end}"
            );
        }
    }
}
