//! Locked `nlp.narrative.v1` schema types and vocabulary helpers.

use aho_corasick::{AhoCorasick, MatchKind};
use faberline_nlp::Span;
use faberline_nlp::doc::{EntityKind, Options};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const NARRATIVE_SCHEMA: &str = "nlp.narrative.v1";
pub const CHAPTER_NARRATIVE_SCHEMA: &str = "nlp.narrative.chapter.v1";
pub const BOOK_RECEIPT_SCHEMA: &str = "nlp.narrative.receipt.v1";

/// Surface-to-canonical vocabulary entry supplied by the caller.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasEntry {
    pub surface: String,
    pub canonical: String,
}

/// Optional genre-neutral caller vocabulary for entity normalization and segmentation hints.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkVocabulary {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actors: Vec<AliasEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub places: Vec<AliasEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<AliasEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub domain_terms: Vec<String>,
}

impl WorkVocabulary {
    /// Returns `true` when no vocabulary entries or domain terms are present.
    pub fn is_empty(&self) -> bool {
        self.actors.is_empty()
            && self.places.is_empty()
            && self.objects.is_empty()
            && self.domain_terms.is_empty()
    }

    /// Looks up an actor's canonical label by exact surface spelling.
    pub fn lookup_actor(&self, surface: &str) -> Option<&str> {
        lookup_alias(&self.actors, surface)
    }

    /// Looks up a place's canonical label by exact surface spelling.
    pub fn lookup_place(&self, surface: &str) -> Option<&str> {
        lookup_alias(&self.places, surface)
    }

    /// Looks up an object's canonical label by exact surface spelling.
    pub fn lookup_object(&self, surface: &str) -> Option<&str> {
        lookup_alias(&self.objects, surface)
    }

    /// Looks up any actor, place, or object canonical label by exact surface spelling.
    pub fn lookup_any(&self, surface: &str) -> Option<&str> {
        self.lookup_actor(surface)
            .or_else(|| self.lookup_object(surface))
            .or_else(|| self.lookup_place(surface))
    }

    /// Builds `faberline_nlp::doc::Options` populated with actor names and lexicon terms.
    pub fn to_doc_options(&self) -> Options {
        let mut seen_names = HashSet::new();
        let mut names = Vec::new();
        for entry in &self.actors {
            if !entry.surface.is_empty() && seen_names.insert(entry.surface.clone()) {
                names.push(entry.surface.clone());
            }
        }
        let mut seen_lex = HashSet::new();
        let mut lexicon = Vec::new();
        for surface in self
            .places
            .iter()
            .chain(self.objects.iter())
            .map(|e| &e.surface)
            .chain(self.domain_terms.iter())
        {
            if !surface.is_empty() && seen_lex.insert(surface.clone()) {
                lexicon.push(surface.clone());
            }
        }
        Options {
            names,
            normalize_traditional: true,
            lexicon,
            ..Options::default()
        }
    }

    /// Finds leftmost-longest actor surface matches in `text`, returning `(Span, canonical)`.
    pub fn find_actor_spans(&self, text: &str) -> Vec<(Span, String)> {
        find_alias_spans(&self.actors, text)
    }

    /// Finds leftmost-longest place surface matches in `text`, returning `(Span, canonical)`.
    pub fn find_place_spans(&self, text: &str) -> Vec<(Span, String)> {
        find_alias_spans(&self.places, text)
    }

    /// Finds leftmost-longest object surface matches in `text`, returning `(Span, canonical)`.
    pub fn find_object_spans(&self, text: &str) -> Vec<(Span, String)> {
        find_alias_spans(&self.objects, text)
    }
}

fn lookup_alias<'a>(entries: &'a [AliasEntry], surface: &str) -> Option<&'a str> {
    entries
        .iter()
        .find(|e| e.surface == surface)
        .map(|e| e.canonical.as_str())
}

fn find_alias_spans(entries: &[AliasEntry], text: &str) -> Vec<(Span, String)> {
    let filtered: Vec<&AliasEntry> = entries
        .iter()
        .filter(|e| !e.surface.is_empty() && !e.surface.contains('\n'))
        .collect();
    if filtered.is_empty() || text.is_empty() {
        return Vec::new();
    }
    let patterns: Vec<&str> = filtered.iter().map(|e| e.surface.as_str()).collect();
    let Ok(ac) = AhoCorasick::builder()
        .match_kind(MatchKind::LeftmostLongest)
        .build(&patterns)
    else {
        return Vec::new();
    };

    let mut byte_to_char = vec![0usize; text.len() + 1];
    let mut char_count = 0usize;
    for (byte_idx, ch) in text.char_indices() {
        byte_to_char[byte_idx] = char_count;
        char_count += 1;
        let next_byte = byte_idx + ch.len_utf8();
        for slot in &mut byte_to_char[(byte_idx + 1)..next_byte] {
            *slot = char_count;
        }
    }
    byte_to_char[text.len()] = char_count;

    let bytes = text.as_bytes();
    let mut out = Vec::new();
    for m in ac.find_iter(text) {
        let start = m.start();
        let end = m.end();
        let pat_bytes = patterns[m.pattern().as_usize()].as_bytes();
        if let (Some(&first), Some(&last)) = (pat_bytes.first(), pat_bytes.last()) {
            if first.is_ascii_alphanumeric()
                && start > 0
                && bytes[start - 1].is_ascii_alphanumeric()
            {
                continue;
            }
            if last.is_ascii_alphanumeric()
                && end < bytes.len()
                && bytes[end].is_ascii_alphanumeric()
            {
                continue;
            }
        }
        out.push((
            Span {
                start_char: byte_to_char[start],
                end_char: byte_to_char[end],
            },
            filtered[m.pattern().as_usize()].canonical.clone(),
        ));
    }
    out
}

/// Computes lowercase hex SHA-256 over the exact UTF-8 bytes of `text`.
pub fn text_sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Chapter-qualified character span.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocatedSpan {
    pub chapter: u32,
    pub span: Span,
}

impl Ord for LocatedSpan {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.chapter, self.span.start_char, self.span.end_char).cmp(&(
            other.chapter,
            other.span.start_char,
            other.span.end_char,
        ))
    }
}

impl PartialOrd for LocatedSpan {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::hash::Hash for LocatedSpan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.chapter.hash(state);
        self.span.start_char.hash(state);
        self.span.end_char.hash(state);
    }
}

/// Distinguishes narrator-asserted prose from direct quoted speech.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "zone", rename_all = "snake_case")]
pub enum ContextZone {
    Narrative,
    DirectQuote { quote_index: usize },
}

/// Evidence basis for speaker attribution on a [`DialogueSpan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerBasis {
    PostQuoteSpeechVerb,
    PreQuoteSpeechVerb,
    IntraParagraphSingleActor,
    Unknown,
}

/// Paired direct quotation span and optional speaker attribution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DialogueSpan {
    pub quote_index: usize,
    pub span: Span,
    pub content_span: Span,
    pub paragraph: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_verb: Option<Span>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_span: Option<Span>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_canonical: Option<String>,
    pub basis: SpeakerBasis,
}

/// Epistemic / reality status of a candidate event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Factuality {
    Actual,
    Negated,
    Hypothetical,
    ModalOrFuture,
    Quoted,
}

/// Semantic role of an event argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventRole {
    Agent,
    Patient,
    Oblique,
}

/// Structural or discourse basis for an extracted [`EventArgument`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArgumentBasis {
    DirectSyntax,
    DisposalJiang,
    PassiveBei,
    CompoundMerged,
    IntraSentenceTopicChain,
    PronounResolved,
}

/// Source-linked argument attached to a [`CandidateEvent`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventArgument {
    pub role: EventRole,
    pub span: Span,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical: Option<String>,
    pub basis: ArgumentBasis,
    pub score_milli: u16,
}

/// Clause or sentence temporal entity anchor from `nlp.doc.v1`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalAnchor {
    pub entity_index: usize,
    pub kind: EntityKind,
    pub span: Span,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

/// Clause-level candidate event anchored on a predicate token.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateEvent {
    pub event_id: String,
    pub chapter: u32,
    pub paragraph: usize,
    pub sentence: usize,
    pub clause: usize,
    pub clause_span: Span,
    pub predicate_token: usize,
    pub predicate_span: Span,
    pub zone: ContextZone,
    pub factuality: Factuality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factuality_cue: Option<Span>,
    pub arguments: Vec<EventArgument>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub temporal_anchors: Vec<TemporalAnchor>,
}

/// Category of an extracted actor mention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MentionKind {
    ExplicitName,
    NominalPhrase,
    Pronoun,
    ZeroPronounProDrop,
}

/// Source-linked actor mention and optional same-chapter antecedent resolution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActorMention {
    pub chapter: u32,
    pub sentence: usize,
    pub clause: usize,
    pub span: Span,
    pub kind: MentionKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_antecedent: Option<Span>,
}

/// Rhetorical or chronological relationship between two events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Before,
    Simultaneous,
    Cause,
    Contrast,
    Condition,
}

/// Provenance category supporting a [`TimelineEdge`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeEvidenceKind {
    ExplicitDiscourseRelation,
    TemporalAnchorShift,
    CausalConnector,
    IntraSentenceClauseFlow,
}

/// Acceptance status of a [`TimelineEdge`] inside the chronological DAG.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeStatus {
    Supported,
    NonChronologicalFactuality,
    ConflictCycle,
}

/// Directed or structural link between two candidate events.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimelineEdge {
    pub edge_id: String,
    pub kind: EdgeKind,
    pub evidence_kind: EdgeEvidenceKind,
    pub status: EdgeStatus,
    pub from_event: usize,
    pub to_event: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connector_span: Option<LocatedSpan>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_spans: Vec<LocatedSpan>,
    pub confidence_tier: u8,
}

/// Partial-order query answer over the condensed timeline DAG.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChronologyOrder {
    Before,
    After,
    Simultaneous,
    Incomparable,
}

/// Abstention or conflict reason recorded under `Never Guess` and cycle-breaking policies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NarrativeUnknownKind {
    ProDropAmbiguous,
    PronounAmbiguous,
    SpeakerAmbiguous,
    PredicateLowConfidence,
    CycleConflictDemoted,
}

/// Source-linked diagnostic for an abstained or demoted narrative decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NarrativeUnknown {
    pub kind: NarrativeUnknownKind,
    pub chapter: u32,
    pub span: Span,
    pub detail: String,
}

/// Complete `nlp.narrative.chapter.v1` output for a single chapter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChapterNarrative {
    pub schema: String,
    pub chapter: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub text_chars: usize,
    pub text_sha256: String,
    pub dialogues: Vec<DialogueSpan>,
    pub events: Vec<CandidateEvent>,
    pub mentions: Vec<ActorMention>,
    pub edges: Vec<TimelineEdge>,
    pub event_dag_layers: Vec<Option<u32>>,
    pub conflict_cycles_broken: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unknown: Vec<NarrativeUnknown>,
}
