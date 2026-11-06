//! Rule-based chapter and timeline analysis for Chinese narrative text.
//!
//! Supply a work's vocabulary through [`Lexicon`]. Evidence offsets count Unicode
//! scalar values from the beginning of the exact, unmodified input text. Ranges
//! include the start and exclude the end.

mod analyze;
mod language;
pub mod timeline;

pub use analyze::analyze;
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
    pub kind: String,
    pub status: String,
    pub actor: Option<String>,
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
    pub speaker: Option<String>,
    pub evidence: Span,
}
