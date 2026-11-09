//! Bind a bounded quotation inspection to an exact caller-provided source.
//! This verifies offsets against the borrowed string, not its file identity,
//! version, narrative scope, actual occurrence, or speaker identity.
use crate::written_quote_typed::{self as typed, WrittenVocabulary};
use serde::Serialize;

/// Upper bound applies to the original source, before scanning its prefix.
pub const MAX_SOURCE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct BoundRange {
    pub segment: typed::Range,
    pub source: typed::Range,
}
#[derive(Debug, Serialize)]
pub struct BoundTag {
    pub kind: typed::TagKind,
    pub written_lead: BoundRange,
    pub speech_cue: BoundRange,
    pub sentence_terminator: BoundRange,
    pub attribution_status: &'static str,
    pub identity_status: &'static str,
    pub real_occurrence_status: &'static str,
    pub full_context_scope: &'static str,
}
#[derive(Debug, Serialize)]
pub struct BoundUnresolvedTag {
    /// This gap is not a resolved speaker surface or identity.
    pub unresolved_gap: BoundRange,
    pub speech_cue: BoundRange,
    pub status: &'static str,
}
#[derive(Debug, Serialize)]
pub struct BoundQuote {
    pub kind: typed::QuoteKind,
    pub written_quote: BoundRange,
    pub inner_text: BoundRange,
    pub postposed_tag: Option<BoundTag>,
    pub unresolved_postposed_tag: Option<BoundUnresolvedTag>,
    pub speaker_identity_status: &'static str,
    pub real_occurrence_status: &'static str,
    pub full_context_scope: &'static str,
}
#[derive(Debug, Serialize)]
pub struct BoundInspection {
    pub quotations: Vec<BoundQuote>,
    pub segment: typed::Range,
    pub exact_source_byte_len: usize,
    /// Exact only for the borrowed string passed to this call; not a digest.
    pub coordinate_binding: &'static str,
    pub source_identity: &'static str,
    pub story_scope: &'static str,
    pub coverage: &'static str,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectError {
    SourceTooLarge,
    InvalidSegment,
    SegmentNotUtf8Boundary,
    SegmentCutsSupportedQuote,
    PrefixDelimiterAmbiguous,
    SegmentInspection(typed::InspectError),
}

fn closing(ch: char) -> Option<char> {
    match ch {
        '「' => Some('」'),
        '『' => Some('』'),
        '“' => Some('”'),
        '‘' => Some('’'),
        _ => None,
    }
}
fn is_close(ch: char) -> bool {
    matches!(ch, '」' | '』' | '”' | '’')
}

/// Reject boundaries inside supported quote pairs. The scan does not attest
/// that the passage is narration or that unsupported delimiters are absent.
fn validate_quote_boundaries(source: &str, start: usize, end: usize) -> Result<(), InspectError> {
    let mut stack = Vec::new();
    for (offset, ch) in source.char_indices() {
        if (offset == start || offset == end) && !stack.is_empty() {
            return Err(InspectError::SegmentCutsSupportedQuote);
        }
        if offset >= end {
            break;
        }
        if stack.last() == Some(&ch) {
            stack.pop();
        } else if let Some(close) = closing(ch) {
            stack.push(close);
        } else if is_close(ch) {
            return Err(InspectError::PrefixDelimiterAmbiguous);
        }
    }
    // char_indices has no entry at source.len().
    if end == source.len() && !stack.is_empty() {
        return Err(InspectError::SegmentCutsSupportedQuote);
    }
    Ok(())
}

fn bind(range: typed::LocatedRange) -> BoundRange {
    BoundRange {
        segment: range.local,
        source: range.source_offset_hint,
    }
}

/// Inspect a byte segment of an exact borrowed UTF-8 source. The caller does
/// not provide global offsets: both byte and scalar positions are calculated
/// from `source`. A malformed selection fails before any slicing or parsing.
/// A successful result does not authenticate a persistent file or determine
/// whether any quoted speech occurred in a story.
pub fn inspect_source_segment(
    source: &str,
    start_byte: usize,
    end_byte: usize,
    vocabulary: WrittenVocabulary<'_>,
) -> Result<BoundInspection, InspectError> {
    if source.len() > MAX_SOURCE_BYTES {
        return Err(InspectError::SourceTooLarge);
    }
    if start_byte > end_byte || end_byte > source.len() {
        return Err(InspectError::InvalidSegment);
    }
    if !source.is_char_boundary(start_byte) || !source.is_char_boundary(end_byte) {
        return Err(InspectError::SegmentNotUtf8Boundary);
    }
    validate_quote_boundaries(source, start_byte, end_byte)?;
    let scalar_start = source[..start_byte].chars().count();
    let segment = &source[start_byte..end_byte];
    let scalar_end = scalar_start + segment.chars().count();
    let output = typed::inspect(
        typed::Window {
            text: segment,
            byte_offset_hint: start_byte,
            scalar_offset_hint: scalar_start,
        },
        vocabulary,
    )
    .map_err(InspectError::SegmentInspection)?;
    let quotations = output
        .quotations
        .into_iter()
        .map(|q| BoundQuote {
            kind: q.kind,
            written_quote: bind(q.written_quote),
            inner_text: bind(q.inner_text),
            postposed_tag: q.postposed_tag.map(|t| BoundTag {
                kind: t.kind,
                written_lead: bind(t.written_lead),
                speech_cue: bind(t.speech_cue),
                sentence_terminator: bind(t.sentence_terminator),
                attribution_status: t.attribution_status,
                identity_status: t.identity_status,
                real_occurrence_status: t.real_occurrence_status,
                full_context_scope: t.full_context_scope,
            }),
            unresolved_postposed_tag: q.unresolved_postposed_tag.map(|t| BoundUnresolvedTag {
                unresolved_gap: bind(t.unresolved_gap),
                speech_cue: bind(t.speech_cue),
                status: t.status,
            }),
            speaker_identity_status: q.speaker_identity_status,
            real_occurrence_status: q.real_occurrence_status,
            full_context_scope: q.full_context_scope,
        })
        .collect();
    Ok(BoundInspection {
        quotations,
        segment: typed::Range {
            byte_start: start_byte,
            byte_end: end_byte,
            scalar_start,
            scalar_end,
        },
        exact_source_byte_len: source.len(),
        coordinate_binding: "exact_borrowed_source_only",
        source_identity: "unknown",
        story_scope: "unknown",
        coverage: "supported_delimiters_in_segment_only",
    })
}
