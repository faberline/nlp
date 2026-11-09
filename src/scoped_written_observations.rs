//! Source-scoped written observations over contiguous original-text segments.
//!
//! The segments are hard boundaries. Quote and cessation indices are local to
//! each segment and must never be interpreted across segments. All returned
//! offsets refer to the exact original input; the caller must bind that input.
//! This API does not authenticate source, identity, attribution or occurrence.

use serde::Serialize;

use crate::{
    Lexicon, NarrativeAnalyzer,
    safe_written_observations::{self, Report, ReuseError, Unknown, WrittenQuote},
    source_scope::{SourceScope, SourceSegment},
    written_cessation_evidence,
    written_speech_alias_evidence::SourceRange,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    VocabularyMismatch,
    Observation(safe_written_observations::Error),
    InvalidSourceRange,
    InvalidLocalIndex,
}

/// All report indices are local to this segment, not the enclosing result.
#[derive(Debug, Serialize)]
pub struct SegmentReport<'a> {
    pub segment_id: usize,
    pub source: SourceRange,
    pub index_scope: &'static str,
    pub boundary_scope: &'static str,
    pub report: Report<'a>,
}

/// A segmented, unverified source observation; it does not imply full coverage.
#[derive(Debug, Serialize)]
pub struct ScopedReport<'a> {
    pub segments: Vec<SegmentReport<'a>>,
    pub scope_coverage: &'static str,
    pub cross_segment_associations: &'static str,
    pub source_identity: Unknown,
    pub coordinate_binding: &'static str,
}

fn rebase(segment: &SourceSegment<'_>, range: &mut SourceRange) -> Result<(), Error> {
    *range = segment
        .map_range(range.scalars, range.byte_start..range.byte_end)
        .map_err(|_| Error::InvalidSourceRange)?;
    Ok(())
}

fn rebase_cessation(
    segment: &SourceSegment<'_>,
    range: &mut written_cessation_evidence::SourceRange,
) -> Result<(), Error> {
    let mapped = segment
        .map_range(range.scalars, range.byte_start..range.byte_end)
        .map_err(|_| Error::InvalidSourceRange)?;
    range.scalars = mapped.scalars;
    range.byte_start = mapped.byte_start;
    range.byte_end = mapped.byte_end;
    Ok(())
}

fn within(inner: SourceRange, outer: SourceRange) -> bool {
    outer.byte_start <= inner.byte_start
        && inner.byte_end <= outer.byte_end
        && outer.scalars.start_char <= inner.scalars.start_char
        && inner.scalars.end_char <= outer.scalars.end_char
}

fn rebase_quote(segment: &SourceSegment<'_>, quote: &mut WrittenQuote) -> Result<(), Error> {
    rebase(segment, &mut quote.source)?;
    if let Some(inner) = &mut quote.inner {
        rebase(segment, inner)?;
        if !within(*inner, quote.source) {
            return Err(Error::InvalidSourceRange);
        }
    }
    for nested in &mut quote.nested_quotations {
        rebase_quote(segment, nested)?;
        if !within(nested.source, quote.source) {
            return Err(Error::InvalidSourceRange);
        }
    }
    Ok(())
}

fn rebase_report(segment: &SourceSegment<'_>, report: &mut Report<'_>) -> Result<(), Error> {
    for quote in &mut report.quotations {
        rebase_quote(segment, quote)?;
    }
    for alternative in &mut report.alternative_quote_boundaries {
        rebase(segment, &mut alternative.source)?;
    }
    for diagnostic in &mut report.quote_scan_diagnostics {
        rebase(segment, &mut diagnostic.source)?;
    }
    for speech in &mut report.speech {
        rebase(segment, &mut speech.written.cue)?;
        rebase(segment, &mut speech.written.payload)?;
        if let Some(speaker) = &mut speech.written.written_speaker {
            rebase(segment, speaker)?;
        }
        if let Some(index) = speech.potential_quote_index {
            let quote = report
                .quotations
                .get(index)
                .ok_or(Error::InvalidLocalIndex)?;
            if !quote
                .inner
                .is_some_and(|inner| within(speech.written.payload, inner))
            {
                return Err(Error::InvalidLocalIndex);
            }
        }
    }
    for cue in &mut report.written_negated_cues {
        rebase(segment, &mut cue.cue)?;
        rebase(segment, &mut cue.negation)?;
        if let Some(subject) = &mut cue.written_subject {
            rebase(segment, subject)?;
        }
        if cue.potential_quote_index >= report.quotations.len() {
            return Err(Error::InvalidLocalIndex);
        }
    }
    for cessation in &mut report.cessation {
        rebase_cessation(segment, &mut cessation.written.tag)?;
        rebase_cessation(segment, &mut cessation.written.subject_surface)?;
        rebase_cessation(segment, &mut cessation.written.cue)?;
        cessation.written.coordinate_binding = "caller_must_bind_exact_input";
        if cessation.preceding_quote_index >= report.quotations.len() {
            return Err(Error::InvalidLocalIndex);
        }
    }
    for diagnostic in &mut report.cessation_diagnostics {
        rebase(segment, &mut diagnostic.candidate_source)?;
        if diagnostic.preceding_quote_index >= report.quotations.len() {
            return Err(Error::InvalidLocalIndex);
        }
    }
    for claim in &mut report.written_claims {
        rebase(segment, &mut claim.source)?;
        let quote = report
            .quotations
            .get(claim.potential_quote_index)
            .ok_or(Error::InvalidLocalIndex)?;
        let cessation = report
            .cessation
            .get(claim.potential_cessation_index)
            .ok_or(Error::InvalidLocalIndex)?;
        if cessation.preceding_quote_index != claim.potential_quote_index
            || !quote.inner.is_some_and(|inner| within(claim.source, inner))
        {
            return Err(Error::InvalidLocalIndex);
        }
    }
    report.coordinate_binding = "caller_must_bind_exact_input";
    Ok(())
}

/// Inspect contiguous admitted source segments without joining hidden gaps.
///
/// Source ranges in the result index the original chapter. Each `segment_id`
/// has an independent quote/cessation index space. Empty output does not prove
/// absence; relationships crossing segment boundaries are not examined.
/// Returned labels and source offsets do not authenticate a person or source.
pub fn inspect_with_analyzer<'a>(
    source: &str,
    title: &str,
    analyzer: &NarrativeAnalyzer<'_>,
    lexicon: &'a Lexicon<'a>,
) -> Result<ScopedReport<'a>, Error> {
    // Check even for an empty admitted scope, where no segment runs the raw API.
    if !analyzer.matches_lexicon(lexicon) {
        return Err(Error::VocabularyMismatch);
    }
    let scope = SourceScope::new(source, title, lexicon);
    let mut output = Vec::new();
    for (segment_id, segment) in scope.segments().iter().enumerate() {
        // Filtering already happened on the original chapter. Reapplying its
        // title to an interior segment could incorrectly discard its first line.
        let mut report = safe_written_observations::inspect_scoped_with_analyzer(
            segment.text,
            "",
            analyzer,
            lexicon,
        )
        .map_err(|error| match error {
            ReuseError::VocabularyMismatch => Error::VocabularyMismatch,
            ReuseError::Observation(inner) => Error::Observation(inner),
        })?;
        rebase_report(segment, &mut report)?;
        output.push(SegmentReport {
            segment_id,
            source: segment.source,
            index_scope: "segment",
            boundary_scope: "hard_break",
            report,
        });
    }
    Ok(ScopedReport {
        segments: output,
        scope_coverage: "partial_or_unknown",
        cross_segment_associations: "not_examined",
        source_identity: Unknown::Unknown,
        coordinate_binding: "caller_must_bind_exact_input",
    })
}
