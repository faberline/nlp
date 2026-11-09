//! Source-linked English and Chinese text analysis.
//!
//! Supply a narrative work's vocabulary through [`Lexicon`]. Narrative offsets count Unicode
//! scalar values from the beginning of the exact, unmodified input text. Ranges
//! include the start and exclude the end.

mod analyze;
mod language;
mod recap_time;
pub mod scoped_transfer;
pub mod source_cross_chapter;
pub mod source_scope;
pub mod source_view;
pub mod text;
pub mod timeline;
#[cfg(feature = "model-tokenizer")]
pub mod tokenizer;

pub use analyze::{
    ClauseCandidate, ClauseFrame, ClauseOrderAdvisory, ClauseOrderBinding, ClauseOrderCue,
    ClauseOrderCueKind, ClauseOrderDiagnostic, ClauseOrderEndpoint, ClauseOrderProbe,
    ClauseOrderSide, ClauseOrderWindow, ClausePrecedence, ClauseResultLink, ClauseScope,
    DurationObservation, LinkedClauses, MentionActorObservation, MentionActorUnknown,
    MentionAssociationBasis, MentionLabelBasis, MentionLabelMappingStatus, MentionPredicateBasis,
    MentionPredicateObservation, NarrativeAnalyzer, RankedClauseOrderCue, ReadingAction,
    ScopedClause, SpeechAct, SpeechActKind, VerifiedClauseEvidence, VerifiedClauseHead,
    WrittenTransferCue, WrittenTransferDirection, WrittenTransferRoleAlternatives,
    WrittenTransferRoleCandidate, WrittenTransferUnknown, analyze,
};
pub use faberline_nlp_cnl as cnl;
use serde::{Deserialize, Serialize};

/// Names, aliases, vocabulary and source-format markers for a single work.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lexicon<'a> {
    /// Surface spelling and canonical actor pairs, including aliases.
    pub actors: &'a [(&'a str, &'a str)],
    /// Literal place names used to identify destinations and section changes.
    pub places: &'a [&'a str],
    /// Surface spelling and canonical object pairs.
    pub objects: &'a [(&'a str, &'a str)],
    /// Surface spelling and canonical rank pairs.
    pub ranks: &'a [(&'a str, &'a str)],
    /// Rank spellings that accept an explicit formation verb as an outcome.
    pub formed_ranks: &'a [&'a str],
    /// Work-specific prefixes that make an adjacent rank mention nonliteral.
    pub rank_blocked_prefixes: &'a [&'a str],
    /// Chapter-wide footer markers, matched with ASCII case folding.
    pub footers: &'a [&'a str],
    /// Source-specific markers that stop only the current line.
    pub line_footers: &'a [&'a str],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub start_char: usize,
    pub end_char: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Event {
    /// Legacy parser classification, not a verified event type.
    pub kind: String,
    /// Legacy parser status, not verified occurrence.
    pub status: String,
    /// Legacy parser label; it is not a verified identity.
    pub actor: Option<String>,
    /// Legacy parser label; it is not a verified identity.
    pub object: Option<String>,
    pub action: String,
    pub evidence: Vec<Span>,
    pub recap: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Review {
    pub kind: String,
    pub detail: String,
    pub evidence: Span,
}

/// Output from analyzing a single chapter with its evidence ranges.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Analysis {
    pub events: Vec<Event>,
    pub sections: Vec<TextSection>,
    pub reviews: Vec<Review>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextSection {
    pub evidence: Span,
    pub place: Option<String>,
    pub context: String,
    pub units: Vec<TextUnit>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextUnit {
    pub kind: String,
    /// Legacy parser-selected label, often from the caller vocabulary; not a verified identity.
    pub speaker: Option<String>,
    pub evidence: Span,
}

pub mod written_quote_typed;

pub mod written_quote_bound_segment;

pub mod written_quote_stream_reader;

pub mod resumable_written_quote_reader;

pub mod bounded_quote_chunks;

pub mod written_speech_alias_evidence;

pub mod written_cessation_evidence;

pub mod text_quality;

pub mod safe_written_observations;

pub mod scoped_written_observations;

pub mod automatic_spelling;

pub mod ranked_spelling;

/// Source-linked timeline hints over the current and previous chapter only.
pub mod source_timeline_stream;
