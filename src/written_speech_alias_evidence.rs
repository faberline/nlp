//! Source-linked speech observations and caller vocabulary labels.
//!
//! A caller's alias mapping is preserved as an explicitly supplied label. It
//! is not proof of a person's identity, an actual utterance, or story scope.
//! This adapter invokes the existing whole-input narrative analyzer; it is not
//! a streaming API and does not claim complete speech coverage.
use std::collections::BTreeSet;

use crate::{Lexicon, NarrativeAnalyzer, Span, SpeechAct, SpeechActKind, text::Locale};

/// Bounds the adapter's output and vocabulary work, not the input analyzer.
pub const MAX_ACTS: usize = 4096;
pub const MAX_ACTOR_ENTRIES: usize = 4096;
pub const MAX_ENTRY_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct SourceRange {
    /// Unicode scalar range in the exact source passed to this call.
    pub scalars: Span,
    /// UTF-8 bytes in that same source, end-exclusive.
    pub byte_start: usize,
    pub byte_end: usize,
}

/// Distinct outputs are deliberately called labels, not resolved identities.
#[derive(Debug, Eq, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CallerLabels<'a> {
    NoWrittenSurface,
    Unmapped,
    Unique { label: &'a str },
    Ambiguous { labels: Vec<&'a str> },
}

#[derive(Debug, serde::Serialize)]
pub struct WrittenSpeech<'a> {
    pub kind: SpeechActKind,
    pub cue: SourceRange,
    pub payload: SourceRange,
    /// The source's written subject; never a verified person's identity.
    pub written_speaker: Option<SourceRange>,
    pub caller_labels: CallerLabels<'a>,
    pub attribution_basis: &'static str,
    pub identity: &'static str,
    pub occurrence: &'static str,
    pub narrative_scope: &'static str,
}

#[derive(Debug, serde::Serialize)]
pub struct WrittenSpeechReport<'a> {
    pub speech: Vec<WrittenSpeech<'a>>,
    pub coverage: &'static str,
    pub source_identity: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    AnalyzerInit,
    TooManyActs,
    VocabularyTooLarge,
    InvalidVocabulary,
    InvalidSpan,
    OverlappingEvidence,
}

fn overlaps(left: Span, right: Span) -> bool {
    left.start_char < right.end_char && right.start_char < left.end_char
}

/// Preserve the input analyzer's source ranges and the complete vocabulary
/// mapping for each exact written surface. The raw `SpeechAct.speaker` is
/// deliberately ignored. Each act and its role is still a heuristic source
/// observation; callers must not treat its supplied label as identity proof.
pub fn map_written_speech<'a>(
    source: &str,
    acts: &[SpeechAct],
    lexicon: &'a Lexicon<'a>,
) -> Result<WrittenSpeechReport<'a>, Error> {
    if acts.len() > MAX_ACTS {
        return Err(Error::TooManyActs);
    }
    if lexicon.actors.len() > MAX_ACTOR_ENTRIES {
        return Err(Error::VocabularyTooLarge);
    }
    if lexicon.actors.iter().any(|(surface, label)| {
        surface.is_empty()
            || label.is_empty()
            || surface.len() > MAX_ENTRY_BYTES
            || label.len() > MAX_ENTRY_BYTES
    }) {
        return Err(Error::InvalidVocabulary);
    }

    // Convert only the requested offsets. Work is one source scan plus a sort
    // of O(number of acts) boundaries; no source-sized byte-index vector.
    let mut targets = Vec::with_capacity(acts.len().saturating_mul(6));
    for act in acts {
        let mut spans = vec![act.cue, act.payload];
        if let Some(speaker) = act.speaker_surface {
            spans.push(speaker);
        }
        for span in &spans {
            if span.start_char >= span.end_char {
                return Err(Error::InvalidSpan);
            }
            targets.push(span.start_char);
            targets.push(span.end_char);
        }
    }
    targets.sort_unstable();
    targets.dedup();
    let mut byte_at = Vec::with_capacity(targets.len());
    let mut next = 0;
    let mut end_scalar = 0;
    for (scalar, (byte, _)) in source.char_indices().enumerate() {
        while next < targets.len() && targets[next] == scalar {
            byte_at.push(byte);
            next += 1;
        }
        end_scalar = scalar + 1;
    }
    while next < targets.len() && targets[next] == end_scalar {
        byte_at.push(source.len());
        next += 1;
    }
    if next != targets.len() {
        return Err(Error::InvalidSpan);
    }
    // All ranges are known to be nonempty and within this source before an
    // overlap is diagnosed. This keeps malformed inputs deterministic without
    // rescanning large sources once per act.
    for act in acts {
        if overlaps(act.cue, act.payload)
            || act
                .speaker_surface
                .is_some_and(|speaker| overlaps(speaker, act.cue) || overlaps(speaker, act.payload))
        {
            return Err(Error::OverlappingEvidence);
        }
    }
    let locate = |span: Span| -> Result<SourceRange, Error> {
        let start = targets
            .binary_search(&span.start_char)
            .map_err(|_| Error::InvalidSpan)?;
        let end = targets
            .binary_search(&span.end_char)
            .map_err(|_| Error::InvalidSpan)?;
        Ok(SourceRange {
            scalars: span,
            byte_start: byte_at[start],
            byte_end: byte_at[end],
        })
    };

    let mut speech = Vec::with_capacity(acts.len());
    for act in acts {
        let cue = locate(act.cue)?;
        let payload = locate(act.payload)?;
        let written_speaker = act.speaker_surface.map(&locate).transpose()?;
        let caller_labels = if let Some(surface) = written_speaker {
            let literal = source
                .get(surface.byte_start..surface.byte_end)
                .ok_or(Error::InvalidSpan)?;
            let labels: BTreeSet<_> = lexicon
                .actors
                .iter()
                .filter_map(|&(written, label)| (written == literal).then_some(label))
                .collect();
            match labels.len() {
                0 => CallerLabels::Unmapped,
                1 => CallerLabels::Unique {
                    label: labels.into_iter().next().expect("one label"),
                },
                _ => CallerLabels::Ambiguous {
                    labels: labels.into_iter().collect(),
                },
            }
        } else {
            CallerLabels::NoWrittenSurface
        };
        speech.push(WrittenSpeech {
            kind: act.kind,
            cue,
            payload,
            written_speaker,
            caller_labels,
            attribution_basis: "unverified_analyzer_or_caller_observation",
            identity: "unknown",
            occurrence: "unknown",
            narrative_scope: "unknown",
        });
    }
    Ok(WrittenSpeechReport {
        speech,
        coverage: "partial",
        source_identity: "unknown",
    })
}

/// Run the existing analyzer, then retain both exact written surfaces and all
/// distinct labels the caller associated with each spelling. This does not
/// make the analyzer's local attribution or the caller's labels verified facts.
pub fn analyze_written_speech<'a>(
    source: &str,
    title: &str,
    locale: Locale,
    lexicon: &'a Lexicon<'a>,
) -> Result<WrittenSpeechReport<'a>, Error> {
    if lexicon.actors.len() > MAX_ACTOR_ENTRIES {
        return Err(Error::VocabularyTooLarge);
    }
    let analyzer = NarrativeAnalyzer::new(locale, lexicon).map_err(|_| Error::AnalyzerInit)?;
    let acts = analyzer.discover_speech_acts(source, title);
    map_written_speech(source, &acts, lexicon)
}
