//! Source ranges admitted by the scoped title and footer policy.
//!
//! Segments borrow the exact input and are joined only when their original
//! bytes are adjacent. A skipped source region is never concatenated away.
//! Source identity is not authenticated; callers must bind results to the
//! exact original bytes. Observations across segment gaps are not examined.

use std::{fmt, ops::Range};

use crate::{
    Lexicon, Span, analyze::admitted_lines_scoped, written_speech_alias_evidence::SourceRange,
};

const CHECKPOINT_STRIDE: usize = 256;

/// An invalid local source range or an arithmetic limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScopeError {
    EmptyOrReversed,
    OutOfBounds,
    InvalidUtf8Boundary,
    CoordinateMismatch,
    OffsetOverflow,
}

impl fmt::Display for ScopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid scoped source range: {self:?}")
    }
}
impl std::error::Error for ScopeError {}

/// One uninterrupted portion of the original input.
///
/// `text` is borrowed from the original source. `source` uses half-open byte
/// and Unicode scalar offsets in the whole original source.
#[derive(Debug)]
pub struct SourceSegment<'a> {
    pub text: &'a str,
    pub source: SourceRange,
    // Local (byte, scalar) checkpoints. There is one per 256 scalars.
    checkpoints: Vec<(usize, usize)>,
    scalar_len: usize,
}

impl<'a> SourceSegment<'a> {
    fn new(text: &'a str, byte_start: usize, scalar_start: usize, scalar_end: usize) -> Self {
        let mut checkpoints = vec![(0, 0)];
        for (scalar, (byte, _)) in text.char_indices().enumerate() {
            if scalar != 0 && scalar % CHECKPOINT_STRIDE == 0 {
                checkpoints.push((byte, scalar));
            }
        }
        Self {
            text,
            source: SourceRange {
                scalars: Span {
                    start_char: scalar_start,
                    end_char: scalar_end,
                },
                byte_start,
                byte_end: byte_start + text.len(),
            },
            checkpoints,
            scalar_len: scalar_end - scalar_start,
        }
    }

    fn byte_at_scalar(&self, scalar: usize) -> Result<usize, ScopeError> {
        if scalar > self.scalar_len {
            return Err(ScopeError::OutOfBounds);
        }
        if scalar == self.scalar_len {
            return Ok(self.text.len());
        }
        let (byte, base) = self.checkpoints[scalar / CHECKPOINT_STRIDE];
        self.text[byte..]
            .char_indices()
            .nth(scalar - base)
            .map(|(offset, _)| byte + offset)
            .ok_or(ScopeError::OutOfBounds)
    }

    fn scalar_at_byte(&self, byte: usize) -> Result<usize, ScopeError> {
        if byte > self.text.len() {
            return Err(ScopeError::OutOfBounds);
        }
        if !self.text.is_char_boundary(byte) {
            return Err(ScopeError::InvalidUtf8Boundary);
        }
        let index = self
            .checkpoints
            .partition_point(|&(offset, _)| offset <= byte)
            - 1;
        let (start, scalar) = self.checkpoints[index];
        Ok(scalar + self.text[start..byte].chars().count())
    }

    /// Validate both local coordinate systems, then map them to original input.
    /// Empty, reversed, out-of-segment, mismatched and non-UTF-8-boundary
    /// ranges are rejected. Returned ranges do not authenticate the source.
    pub fn map_range(&self, scalars: Span, bytes: Range<usize>) -> Result<SourceRange, ScopeError> {
        if scalars.start_char >= scalars.end_char || bytes.start >= bytes.end {
            return Err(ScopeError::EmptyOrReversed);
        }
        if scalars.end_char > self.scalar_len || bytes.end > self.text.len() {
            return Err(ScopeError::OutOfBounds);
        }
        if self.scalar_at_byte(bytes.start)? != scalars.start_char
            || self.scalar_at_byte(bytes.end)? != scalars.end_char
        {
            return Err(ScopeError::CoordinateMismatch);
        }
        Ok(SourceRange {
            scalars: Span {
                start_char: self
                    .source
                    .scalars
                    .start_char
                    .checked_add(scalars.start_char)
                    .ok_or(ScopeError::OffsetOverflow)?,
                end_char: self
                    .source
                    .scalars
                    .start_char
                    .checked_add(scalars.end_char)
                    .ok_or(ScopeError::OffsetOverflow)?,
            },
            byte_start: self
                .source
                .byte_start
                .checked_add(bytes.start)
                .ok_or(ScopeError::OffsetOverflow)?,
            byte_end: self
                .source
                .byte_start
                .checked_add(bytes.end)
                .ok_or(ScopeError::OffsetOverflow)?,
        })
    }

    /// Map a nonempty local scalar range after deriving its local byte bounds.
    pub fn map_span(&self, scalars: Span) -> Result<SourceRange, ScopeError> {
        if scalars.start_char >= scalars.end_char {
            return Err(ScopeError::EmptyOrReversed);
        }
        let start = self.byte_at_scalar(scalars.start_char)?;
        let end = self.byte_at_scalar(scalars.end_char)?;
        self.map_range(scalars, start..end)
    }
}

/// Visible, contiguous portions of one original source under caller policy.
///
/// This describes only the configured title/footer/HTML filtering rules; it
/// does not prove text completeness or inspect relationships across gaps.
/// Sparse checkpoints use memory proportional to visible scalars / 256 plus
/// segment count. The existing admitted-lines helper also scans the input.
#[derive(Debug)]
pub struct SourceScope<'a> {
    segments: Vec<SourceSegment<'a>>,
}

impl<'a> SourceScope<'a> {
    /// Borrow the original input and retain its unmodified coordinate space.
    /// Chapter footer matching folds ASCII in both source and marker. This may
    /// differ from the legacy chapter analyzer for uppercase footer markers.
    /// Empty markers retain the legacy behavior of excluding the entire input.
    pub fn new(source: &'a str, title: &str, lexicon: &Lexicon<'_>) -> Self {
        let mut ranges: Vec<Range<usize>> = Vec::new();
        for line in admitted_lines_scoped(source, title, lexicon) {
            if line.text.is_empty() {
                continue;
            }
            let start = line.start;
            let end = start + line.text.len();
            if let Some(previous) = ranges.last_mut()
                && previous.end == start
            {
                previous.end = end;
            } else {
                ranges.push(start..end);
            }
        }
        let mut previous_byte = 0;
        let mut scalar = 0;
        let mut segments = Vec::with_capacity(ranges.len());
        for range in ranges {
            scalar += source[previous_byte..range.start].chars().count();
            let start_scalar = scalar;
            scalar += source[range.clone()].chars().count();
            let text = &source[range.clone()];
            segments.push(SourceSegment::new(text, range.start, start_scalar, scalar));
            previous_byte = range.end;
        }
        Self { segments }
    }

    pub fn segments(&self) -> &[SourceSegment<'a>] {
        &self.segments
    }
    pub fn coverage(&self) -> &'static str {
        "unknown"
    }
    pub fn source_binding_status(&self) -> &'static str {
        "caller_must_bind_exact_input"
    }
    pub fn cross_segment_status(&self) -> &'static str {
        "not_examined"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_lines_join_but_filtered_text_keeps_original_gap() {
        let source = "標題\n𠮷e\u{301}甲\r\n乙<line>隱藏\n丙。<end>尾文";
        let lexicon = Lexicon {
            line_footers: &["<line>"],
            footers: &["<end>"],
            ..Lexicon::default()
        };
        let scope = SourceScope::new(source, "標題", &lexicon);
        let parts = scope.segments();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].text, "𠮷e\u{301}甲\r\n乙");
        assert_eq!(parts[1].text, "丙。");
        for part in parts {
            assert_eq!(
                &source[part.source.byte_start..part.source.byte_end],
                part.text
            );
            assert_eq!(
                source[..part.source.byte_start].chars().count(),
                part.source.scalars.start_char
            );
            assert_eq!(
                source[..part.source.byte_end].chars().count(),
                part.source.scalars.end_char
            );
        }
        assert_eq!(scope.coverage(), "unknown");
        assert_eq!(scope.cross_segment_status(), "not_examined");
    }

    #[test]
    fn mappings_validate_both_coordinate_systems() {
        let source = "序<line>隱\n𠮷e\u{301}甲乙。";
        let lexicon = Lexicon {
            line_footers: &["<line>"],
            ..Lexicon::default()
        };
        let scope = SourceScope::new(source, "", &lexicon);
        let segment = &scope.segments()[1];
        let local = Span {
            start_char: 3,
            end_char: 5,
        };
        let mapped = segment.map_span(local).unwrap();
        assert_eq!(&source[mapped.byte_start..mapped.byte_end], "甲乙");
        assert_eq!(
            source[..mapped.byte_start].chars().count(),
            mapped.scalars.start_char
        );
        let byte_start = segment.text.find('甲').unwrap();
        assert_eq!(
            mapped,
            segment
                .map_range(local, byte_start..byte_start + "甲乙".len())
                .unwrap()
        );
        assert_eq!(
            segment.map_range(local, 1..byte_start),
            Err(ScopeError::InvalidUtf8Boundary)
        );
        assert_eq!(
            segment.map_range(local, 0..4),
            Err(ScopeError::CoordinateMismatch)
        );
        assert_eq!(
            segment.map_span(Span {
                start_char: 1,
                end_char: 1
            }),
            Err(ScopeError::EmptyOrReversed)
        );
        assert_eq!(
            segment.map_span(Span {
                start_char: 2,
                end_char: usize::MAX
            }),
            Err(ScopeError::OutOfBounds)
        );
    }

    #[test]
    fn checkpoints_map_unicode_at_and_across_stride_boundaries() {
        let source = format!("{}𠮷e\u{301}甲", "文".repeat(255));
        let scope = SourceScope::new(&source, "", &Lexicon::default());
        let segment = &scope.segments()[0];
        for (start, end, expected) in [
            (254, 256, "文𠮷"),
            (255, 258, "𠮷e\u{301}"),
            (258, 259, "甲"),
        ] {
            let range = segment
                .map_span(Span {
                    start_char: start,
                    end_char: end,
                })
                .unwrap();
            assert_eq!(&source[range.byte_start..range.byte_end], expected);
            assert_eq!(source[..range.byte_start].chars().count(), start);
            assert_eq!(source[..range.byte_end].chars().count(), end);
        }
    }

    #[test]
    fn completely_excluded_source_produces_no_segments() {
        let scope = SourceScope::new(
            "<end>hidden",
            "",
            &Lexicon {
                footers: &["<end>"],
                ..Lexicon::default()
            },
        );
        assert!(scope.segments().is_empty());
    }
}
