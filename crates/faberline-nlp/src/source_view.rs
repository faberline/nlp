//! Omit a caller-selected marker in a read-only view of source text.
//!
//! The caller decides which original neighbor pairs permit an omission. A match
//! only establishes permission under that policy. The caller must establish
//! whether the marker is an artifact in its source.
//!
//! Offsets count Unicode scalar values from zero. The end of a range is exclusive.

use crate::Span;
use anyhow::{Result, ensure};
use serde::Serialize;
use std::borrow::Cow;
use std::collections::BTreeSet;

/// A source marker and the original neighbor pairs for which the caller permits omission.
pub struct InsertionPolicy {
    marker: char,
    neighbors: BTreeSet<(char, char)>,
}

impl InsertionPolicy {
    /// Allow omission between the given left and right neighbors of `marker`.
    /// An empty list keeps the original text intact.
    ///
    /// # Errors
    ///
    /// Returns an error if a pair contains the marker on either side.
    pub fn new(marker: char, neighbors: impl IntoIterator<Item = (char, char)>) -> Result<Self> {
        let neighbors: BTreeSet<_> = neighbors.into_iter().collect();
        ensure!(
            neighbors
                .iter()
                .all(|&(left, right)| left != marker && right != marker),
            "repair neighbors must exclude the inserted marker"
        );
        Ok(Self { marker, neighbors })
    }

    /// Build a view using the original neighbors of each marker.
    /// The view borrows the original text when no marker matches.
    pub fn apply<'a>(&self, original: &'a str) -> SourceView<'a> {
        let mut view = SourceView {
            text: Cow::Borrowed(original),
            scalar_count: 0,
            repairs: Vec::new(),
        };
        if self.neighbors.is_empty() || !original.contains(self.marker) {
            view.scalar_count = original.chars().count();
            return view;
        }
        let mut text = String::new();
        let mut next_copy = 0;
        let mut chars = original.char_indices().enumerate().peekable();
        let mut previous = None;
        while let Some((index, (byte, current))) = chars.next() {
            view.scalar_count = index + 1;
            if current == self.marker
                && let (Some(left), Some(&(_, (_, right)))) = (previous, chars.peek())
                && self.neighbors.contains(&(left, right))
            {
                let normalized_start = index - view.repairs.len() - 1;
                view.repairs.push(WordRepair {
                    original: span(index - 1, index + 2),
                    inserted: span(index, index + 1),
                    normalized: span(normalized_start, normalized_start + 2),
                    term: [left, right].into_iter().collect(),
                });
                text.push_str(&original[next_copy..byte]);
                next_copy = byte + current.len_utf8();
            }
            previous = Some(current);
        }
        if !view.repairs.is_empty() {
            text.push_str(&original[next_copy..]);
            view.scalar_count -= view.repairs.len();
            view.text = Cow::Owned(text);
        }
        view
    }
}

/// The ranges and text for one omission. Neighbor windows can overlap.
#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct WordRepair {
    /// The original window containing the marker and its two neighbors.
    pub original: Span,
    /// The omitted marker in the original text.
    pub inserted: Span,
    /// The two retained neighbors in the analysis view.
    pub normalized: Span,
    /// The text of the two retained neighbors.
    pub term: String,
}

/// A read-only view with an index of omitted source characters.
pub struct SourceView<'a> {
    text: Cow<'a, str>,
    scalar_count: usize,
    repairs: Vec<WordRepair>,
}

impl SourceView<'_> {
    /// The text after applying the omission policy.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The omissions in original source order.
    pub fn repairs(&self) -> &[WordRepair] {
        &self.repairs
    }

    /// Return the smallest original range that covers the kept scalars.
    /// The range includes any omitted marker between them.
    /// Returns `None` for an empty, reversed, or out-of-bounds range.
    pub fn original_span(&self, normalized: Span) -> Option<Span> {
        if normalized.start_char >= normalized.end_char || normalized.end_char > self.scalar_count {
            return None;
        }
        Some(span(
            self.original_index(normalized.start_char),
            self.original_index(normalized.end_char - 1) + 1,
        ))
    }

    fn original_index(&self, index: usize) -> usize {
        index
            + self
                .repairs
                .partition_point(|repair| repair.normalized.end_char - 1 <= index)
    }
}

fn span(start_char: usize, end_char: usize) -> Span {
    Span {
        start_char,
        end_char,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_original_covering_ranges_and_contiguous_normalized_tokens() -> Result<()> {
        let original = "𠮷激堊射堊出。";
        let policy = InsertionPolicy::new('堊', [('激', '射'), ('射', '出')])?;
        let view = policy.apply(original);
        assert_eq!(view.text(), "𠮷激射出。");
        assert_eq!(view.original_span(span(1, 3)), Some(span(1, 4)));
        assert_eq!(view.original_span(span(2, 4)), Some(span(3, 6)));
        assert_eq!(view.original_span(span(1, 4)), Some(span(1, 6)));
        assert_eq!(view.repairs()[0].inserted, span(2, 3));
        assert_eq!(view.repairs()[1].inserted, span(4, 5));
        assert_eq!(view.original_span(span(3, 3)), None);
        assert_eq!(view.original_span(span(0, 6)), None);
        Ok(())
    }

    #[test]
    fn keeps_unlisted_pairs_and_non_han_boundaries() -> Result<()> {
        let original = "白堊色，ji堊射，激垩射，激堊射。";
        let policy = InsertionPolicy::new('堊', [('激', '射')])?;
        let view = policy.apply(original);
        assert_eq!(view.text(), "白堊色，ji堊射，激垩射，激射。");
        assert_eq!(view.repairs().len(), 1);
        assert!(InsertionPolicy::new('堊', [('堊', '射')]).is_err());
        Ok(())
    }
}
