//! Source-first written quotations and a limited postposed speech form.
//! This module uses only std and serde, present in the pinned NLP package.
//! It does not alter legacy TextUnit speaker semantics. A recognized form
//! is source syntax, not proof of person, actual speech, or narrator scope.
use serde::Serialize;

const MAX_SCALARS: usize = 4096;
const MAX_QUOTES: usize = 128;
const MAX_VOCAB: usize = 512;
const MAX_TERM_BYTES: usize = 128;
const MAX_LEAD_SCALARS: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Range {
    pub byte_start: usize,
    pub byte_end: usize,
    pub scalar_start: usize,
    pub scalar_end: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LocatedRange {
    pub local: Range,
    /// A caller-provided offset hint, not an authenticated source position.
    pub source_offset_hint: Range,
}
#[derive(Clone, Copy, Debug)]
pub struct Window<'a> {
    pub text: &'a str,
    pub byte_offset_hint: usize,
    pub scalar_offset_hint: usize,
}
/// Written forms supplied by the caller, never identity proof. Domain titles
/// such as 真人 belong here and are not hard-coded in generic NLP.
#[derive(Clone, Copy, Debug, Default)]
pub struct WrittenVocabulary<'a> {
    pub subject_surfaces: &'a [&'a str],
    pub title_suffixes: &'a [&'a str],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteKind {
    WrittenQuotation,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TagKind {
    AdjacentPostposedSpeechForm,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PostposedTag {
    pub kind: TagKind,
    /// Written lead only: not a resolved or confirmed speaker.
    pub written_lead: LocatedRange,
    pub speech_cue: LocatedRange,
    pub sentence_terminator: LocatedRange,
    pub identity_status: &'static str,
    pub real_occurrence_status: &'static str,
    pub full_context_scope: &'static str,
    pub attribution_status: &'static str,
}
/// A title suffix alone cannot establish where a Chinese lead starts. This
/// range is the whole unresolved gap, and must not be displayed as a speaker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct UnresolvedPostposedTag {
    pub unresolved_gap: LocatedRange,
    pub speech_cue: LocatedRange,
    pub status: &'static str,
}
#[derive(Debug, Serialize)]
pub struct Quotation {
    pub kind: QuoteKind,
    pub written_quote: LocatedRange,
    pub inner_text: LocatedRange,
    pub postposed_tag: Option<PostposedTag>,
    pub unresolved_postposed_tag: Option<UnresolvedPostposedTag>,
    pub speaker_identity_status: &'static str,
    pub real_occurrence_status: &'static str,
    pub full_context_scope: &'static str,
}
#[derive(Debug, Serialize)]
pub struct Inspection {
    pub quotations: Vec<Quotation>,
    pub coverage: &'static str,
    pub source_binding: &'static str,
    pub unexamined_outside_window: &'static str,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectError {
    WindowTooLong,
    TooManyQuotes,
    VocabularyTooLarge,
    OffsetOverflow,
    UnsupportedOrDamagedDelimiter,
}

fn closing(c: char) -> Option<char> {
    match c {
        '「' => Some('」'),
        '『' => Some('』'),
        '“' => Some('”'),
        '‘' => Some('’'),
        _ => None,
    }
}
fn is_closing(c: char) -> bool {
    matches!(c, '」' | '』' | '”' | '’')
}
fn is_han(c: char) -> bool {
    matches!(c as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff |
        0x20000..=0x2fa1f | 0x30000..=0x323af)
}
fn classify_lead(text: &str, vocab: WrittenVocabulary<'_>) -> LeadKind {
    let n = text.chars().count();
    if n == 0 || n > MAX_LEAD_SCALARS || !text.chars().all(is_han) {
        return LeadKind::None;
    }
    if vocab
        .subject_surfaces
        .iter()
        .any(|s| !s.is_empty() && *s == text)
    {
        return LeadKind::ExactWrittenSurface;
    }
    if vocab.title_suffixes.iter().any(|suffix| {
        if suffix.is_empty() {
            return false;
        }
        text.strip_suffix(*suffix)
            .is_some_and(|prefix| !prefix.is_empty())
    }) {
        LeadKind::UnresolvedSuffix
    } else {
        LeadKind::None
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeadKind {
    None,
    ExactWrittenSurface,
    UnresolvedSuffix,
}

/// Inspect one bounded borrowed window. Unsupported nested or damaged
/// delimiters return an error; the caller retains the entire original input.
pub fn inspect(
    window: Window<'_>,
    vocabulary: WrittenVocabulary<'_>,
) -> Result<Inspection, InspectError> {
    if vocabulary.subject_surfaces.len() > MAX_VOCAB
        || vocabulary.title_suffixes.len() > MAX_VOCAB
        || vocabulary
            .subject_surfaces
            .iter()
            .chain(vocabulary.title_suffixes)
            .any(|v| v.len() > MAX_TERM_BYTES)
    {
        return Err(InspectError::VocabularyTooLarge);
    }
    let mut bytes = Vec::new();
    let mut chars = Vec::new();
    for (byte, ch) in window.text.char_indices() {
        if chars.len() == MAX_SCALARS {
            return Err(InspectError::WindowTooLong);
        }
        chars.push(ch);
        bytes.push(byte);
    }
    bytes.push(window.text.len());
    window
        .byte_offset_hint
        .checked_add(window.text.len())
        .ok_or(InspectError::OffsetOverflow)?;
    window
        .scalar_offset_hint
        .checked_add(chars.len())
        .ok_or(InspectError::OffsetOverflow)?;
    let range = |start: usize, end: usize| -> LocatedRange {
        LocatedRange {
            local: Range {
                byte_start: bytes[start],
                byte_end: bytes[end],
                scalar_start: start,
                scalar_end: end,
            },
            source_offset_hint: Range {
                byte_start: window.byte_offset_hint + bytes[start],
                byte_end: window.byte_offset_hint + bytes[end],
                scalar_start: window.scalar_offset_hint + start,
                scalar_end: window.scalar_offset_hint + end,
            },
        }
    };
    let mut spans = Vec::new();
    let mut open: Option<(usize, char)> = None;
    for (index, &ch) in chars.iter().enumerate() {
        if let Some((start, expected)) = open {
            if ch == expected {
                if spans.len() == MAX_QUOTES {
                    return Err(InspectError::TooManyQuotes);
                }
                spans.push((start, index + 1));
                open = None;
            } else if closing(ch).is_some() || is_closing(ch) {
                return Err(InspectError::UnsupportedOrDamagedDelimiter);
            }
        } else if let Some(end) = closing(ch) {
            open = Some((index, end));
        } else if is_closing(ch) {
            return Err(InspectError::UnsupportedOrDamagedDelimiter);
        }
    }
    if open.is_some() {
        return Err(InspectError::UnsupportedOrDamagedDelimiter);
    }
    let mut quotations = Vec::new();
    for (start, end) in spans {
        let mut tag_start = end;
        while tag_start < chars.len() && matches!(chars[tag_start], ' ' | '\t') {
            tag_start += 1;
        }
        let max = chars
            .len()
            .min(tag_start.saturating_add(MAX_LEAD_SCALARS + 5));
        let stop = (tag_start..max)
            .find(|&i| matches!(chars[i], '。' | '！' | '？' | '.' | '!' | '?' | '\n' | '\r'));
        let mut tag = None;
        let mut unresolved_tag = None;
        if let Some(stop) = stop.filter(|&i| matches!(chars[i], '。' | '.')) {
            let clause = &window.text[bytes[tag_start]..bytes[stop]];
            for cue in ["說道", "说道", "問道", "问道", "喊道", "答道", "說", "说"] {
                if let Some(lead) = clause.strip_suffix(cue) {
                    let subject_end = tag_start + lead.chars().count();
                    match classify_lead(lead, vocabulary) {
                        LeadKind::ExactWrittenSurface => {
                            tag = Some(PostposedTag {
                                kind: TagKind::AdjacentPostposedSpeechForm,
                                written_lead: range(tag_start, subject_end),
                                speech_cue: range(subject_end, stop),
                                sentence_terminator: range(stop, stop + 1),
                                identity_status: "unknown",
                                real_occurrence_status: "unknown",
                                full_context_scope: "unknown",
                                attribution_status: "written_form_only",
                            })
                        }
                        LeadKind::UnresolvedSuffix => {
                            unresolved_tag = Some(UnresolvedPostposedTag {
                                unresolved_gap: range(tag_start, subject_end),
                                speech_cue: range(subject_end, stop),
                                status: "unknown",
                            })
                        }
                        LeadKind::None => {}
                    }
                    break;
                }
            }
        }
        quotations.push(Quotation {
            kind: QuoteKind::WrittenQuotation,
            written_quote: range(start, end),
            inner_text: range(start + 1, end - 1),
            postposed_tag: tag,
            unresolved_postposed_tag: unresolved_tag,
            speaker_identity_status: "unknown",
            real_occurrence_status: "unknown",
            full_context_scope: "unknown",
        });
    }
    Ok(Inspection {
        quotations,
        coverage: "supported_delimiters_in_window_only",
        source_binding: "unverified",
        unexamined_outside_window: "unknown",
    })
}
