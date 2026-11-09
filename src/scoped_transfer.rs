//! Source-scoped, unverified written transfer observations.
//!
//! The scanner never joins separate admitted segments. All returned positions
//! refer to the exact original input, including text that was not examined.
//! A caller must bind those positions to the original source bytes. Nothing in
//! this report authenticates a source, identifies a person, or confirms an event.

use serde::Serialize;

use crate::{
    NarrativeAnalyzer, Span, WrittenTransferDirection, WrittenTransferUnknown,
    source_scope::{ScopeError, SourceScope, SourceSegment},
    written_speech_alias_evidence::SourceRange,
};

/// A written transfer cue, not a verified direction or event.
#[derive(Debug, Serialize)]
pub struct ScopedTransferCue {
    /// Index into the report's admitted original-source segments.
    pub segment_id: usize,
    pub cue: SourceRange,
    pub direction: WrittenTransferDirection,
    pub direction_status: &'static str,
    pub identity_status: WrittenTransferUnknown,
    pub role_status: WrittenTransferUnknown,
    pub occurrence_status: WrittenTransferUnknown,
    pub association_status: WrittenTransferUnknown,
    pub narrative_scope_status: WrittenTransferUnknown,
}

/// Written role-shaped fragments. Their association remains unverified.
#[derive(Debug, Serialize)]
pub struct ScopedTransferRoleCandidate {
    pub segment_id: usize,
    pub cue: SourceRange,
    pub construction: SourceRange,
    pub direction: WrittenTransferDirection,
    pub giver: Option<SourceRange>,
    pub item: Option<SourceRange>,
    pub recipient: Option<SourceRange>,
    pub identity_status: WrittenTransferUnknown,
    pub role_status: WrittenTransferUnknown,
    pub occurrence_status: WrittenTransferUnknown,
    pub association_status: WrittenTransferUnknown,
    pub narrative_scope_status: WrittenTransferUnknown,
}

/// Alternative written boundaries, not resolved participants or roles.
#[derive(Debug, Serialize)]
pub struct ScopedTransferRoleAlternatives {
    pub segment_id: usize,
    pub cue: SourceRange,
    pub construction: SourceRange,
    pub giver_alternatives: Vec<SourceRange>,
    pub recipient_alternatives: Vec<SourceRange>,
    pub boundary_status: WrittenTransferUnknown,
    pub interpretation_status: WrittenTransferUnknown,
    pub identity_status: WrittenTransferUnknown,
    pub role_status: WrittenTransferUnknown,
    pub occurrence_status: WrittenTransferUnknown,
    pub association_status: WrittenTransferUnknown,
    pub narrative_scope_status: WrittenTransferUnknown,
}

#[derive(Debug, Serialize)]
pub struct ScopedTransferReport {
    /// Original-source ranges indexed by each observation's `segment_id`.
    pub segments: Vec<SourceRange>,
    pub cues: Vec<ScopedTransferCue>,
    pub roles: Vec<ScopedTransferRoleCandidate>,
    pub alternatives: Vec<ScopedTransferRoleAlternatives>,
    /// The underlying written-form heuristic has incomplete coverage.
    pub coverage: &'static str,
    /// The interpretation of the text and its overall completeness are unknown.
    pub semantic_status: WrittenTransferUnknown,
    pub source_binding_status: &'static str,
    /// Pairs or constructions spanning a hidden boundary are not examined.
    pub cross_segment_associations: &'static str,
}

fn optional(
    segment: &SourceSegment<'_>,
    span: Option<Span>,
) -> Result<Option<SourceRange>, ScopeError> {
    span.map(|range| segment.map_span(range)).transpose()
}

fn alternatives(
    segment: &SourceSegment<'_>,
    spans: Vec<Span>,
) -> Result<Vec<SourceRange>, ScopeError> {
    spans
        .into_iter()
        .map(|range| segment.map_span(range))
        .collect()
}

/// Inspect each admitted source segment independently.
///
/// Hidden source fragments cannot supply a cue or repair damaged quote state.
/// The existing whole-input methods remain available with their original behavior.
/// A successful return does not establish complete extraction or source identity.
pub fn inspect(
    scope: &SourceScope<'_>,
    analyzer: &NarrativeAnalyzer<'_>,
) -> Result<ScopedTransferReport, ScopeError> {
    let mut cues = Vec::new();
    let mut roles = Vec::new();
    let mut role_alternatives = Vec::new();
    for (segment_id, segment) in scope.segments().iter().enumerate() {
        for item in analyzer.discover_written_transfer_cues(segment.text) {
            cues.push(ScopedTransferCue {
                segment_id,
                cue: segment.map_span(item.cue)?,
                direction: item.direction,
                direction_status: "written_form_only",
                identity_status: WrittenTransferUnknown::Unknown,
                role_status: WrittenTransferUnknown::Unknown,
                occurrence_status: WrittenTransferUnknown::Unknown,
                association_status: WrittenTransferUnknown::Unknown,
                narrative_scope_status: WrittenTransferUnknown::Unknown,
            });
        }
        for item in analyzer.discover_written_transfer_role_candidates(segment.text) {
            roles.push(ScopedTransferRoleCandidate {
                segment_id,
                cue: segment.map_span(item.cue)?,
                construction: segment.map_span(item.construction)?,
                direction: item.direction,
                giver: optional(segment, item.giver)?,
                item: optional(segment, item.item)?,
                recipient: optional(segment, item.recipient)?,
                identity_status: item.identity_status,
                role_status: item.role_status,
                occurrence_status: item.occurrence_status,
                association_status: item.association_status,
                narrative_scope_status: WrittenTransferUnknown::Unknown,
            });
        }
        for item in analyzer.discover_written_transfer_role_alternatives(segment.text) {
            role_alternatives.push(ScopedTransferRoleAlternatives {
                segment_id,
                cue: segment.map_span(item.cue)?,
                construction: segment.map_span(item.construction)?,
                giver_alternatives: alternatives(segment, item.giver_alternatives)?,
                recipient_alternatives: alternatives(segment, item.recipient_alternatives)?,
                boundary_status: item.boundary_status,
                interpretation_status: item.interpretation_status,
                identity_status: item.identity_status,
                role_status: item.role_status,
                occurrence_status: item.occurrence_status,
                association_status: item.association_status,
                narrative_scope_status: WrittenTransferUnknown::Unknown,
            });
        }
    }
    Ok(ScopedTransferReport {
        segments: scope
            .segments()
            .iter()
            .map(|segment| segment.source)
            .collect(),
        cues,
        roles,
        alternatives: role_alternatives,
        coverage: "partial_heuristic",
        semantic_status: WrittenTransferUnknown::Unknown,
        source_binding_status: "caller_must_bind_exact_input",
        cross_segment_associations: "not_examined",
    })
}
