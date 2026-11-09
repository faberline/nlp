//! Source-backed observations, with identity and role deliberately unresolved.
//!
//! The quote scanner preserves a damaged suffix as a written fragment. The
//! speech analyzer is heuristic and may omit observations; its labels remain
//! caller-provided hints. No quote, speaker, event, or story scope is verified.
use serde::Serialize;

use crate::written_cessation_evidence::{self, SourceIndex, WrittenCessation};
use crate::written_speech_alias_evidence::{
    self, CallerLabels, Error as SpeechError, SourceRange, WrittenSpeech,
};
use crate::{Lexicon, NarrativeAnalyzer, Span, text::Locale};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Unknown {
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuotationForm {
    Complete,
    IncompleteOrDamaged,
    UnmatchedCloser,
}

/// Bounds nesting in the recursive serialized/dropped quote structure.
pub const MAX_QUOTE_DEPTH: usize = 64;

#[derive(Debug, Serialize)]
pub struct QuoteBoundaryAlternative {
    /// A literal prefix ending at one possible right-single-quote boundary.
    pub source: SourceRange,
    pub delimiter_status: Unknown,
    pub reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct QuoteScanDiagnostic {
    /// A conservative suffix that the quote scanner could not fully inspect.
    pub source: SourceRange,
    pub status: &'static str,
    pub reason: &'static str,
    pub delimiter_status: Unknown,
}

#[derive(Debug, Serialize)]
pub struct WrittenQuote {
    /// Literal quote or damaged fragment, including any written delimiters.
    pub source: SourceRange,
    /// Inner text for a complete nonempty quotation; never a resolved speech act.
    pub inner: Option<SourceRange>,
    pub form: QuotationForm,
    /// Delimiters are a written-form interpretation, not a proven boundary.
    pub delimiter_status: Unknown,
    pub delimiter_basis: &'static str,
    /// Nested source candidates. Each retains its own precise source range.
    pub nested_quotations: Vec<WrittenQuote>,
    pub speaker_identity: Unknown,
    pub speaker_role: Unknown,
    pub occurrence: Unknown,
    pub narrative_scope: Unknown,
}

#[derive(Debug, Serialize)]
pub struct SpeechObservation<'a> {
    /// An analyzer hint with literal cue, payload and optional written subject.
    /// `caller_labels` are supplied by the caller, not authenticated names.
    pub written: WrittenSpeech<'a>,
    /// A structural association only: payload lies inside this complete quote.
    /// It does not establish speaker, quotation attribution or occurrence.
    pub potential_quote_index: Option<usize>,
    pub role: Unknown,
    pub quote_attribution: Unknown,
}

/// Literal preposed words only. Their grammatical scope and semantic roles
/// remain unknown even when the words resemble a negated speech expression.
#[derive(Debug, Serialize)]
pub struct WrittenNegatedCue<'a> {
    pub cue: SourceRange,
    pub negation: SourceRange,
    pub written_subject: Option<SourceRange>,
    pub caller_labels: CallerLabels<'a>,
    pub potential_quote_index: usize,
    pub basis: &'static str,
    pub identity: Unknown,
    pub role: Unknown,
    pub occurrence: Unknown,
    pub narrative_scope: Unknown,
    pub quote_attribution: Unknown,
}

#[derive(Debug, Serialize)]
pub struct CessationObservation<'a> {
    /// Exact written tag, subject and cue; supplied labels remain unverified.
    pub written: WrittenCessation<'a>,
    /// The tag follows this quote in the source. Attribution is not established.
    pub preceding_quote_index: usize,
    pub identity: Unknown,
    pub role: Unknown,
    pub occurrence: Unknown,
    pub narrative_scope: Unknown,
    pub quote_attribution: Unknown,
}

#[derive(Debug, Serialize)]
pub struct WrittenClaim {
    /// A source excerpt in the quote selected by the legacy analyzer.
    pub source: SourceRange,
    pub potential_quote_index: usize,
    pub potential_cessation_index: usize,
    pub selection_basis: &'static str,
    pub identity: Unknown,
    pub role: Unknown,
    pub occurrence: Unknown,
    pub narrative_scope: Unknown,
    pub quote_attribution: Unknown,
}

#[derive(Debug, Serialize)]
pub struct CessationDiagnostic {
    pub preceding_quote_index: usize,
    /// For `suffix_limit`, the source after the quote through input end is
    /// unexamined. This is not a recognized tag and may extend beyond the scan budget.
    pub candidate_source: SourceRange,
    pub status: &'static str,
    pub reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Report<'a> {
    pub quotations: Vec<WrittenQuote>,
    pub alternative_quote_boundaries: Vec<QuoteBoundaryAlternative>,
    pub quote_scan_diagnostics: Vec<QuoteScanDiagnostic>,
    pub speech: Vec<SpeechObservation<'a>>,
    pub written_negated_cues: Vec<WrittenNegatedCue<'a>>,
    pub cessation: Vec<CessationObservation<'a>>,
    pub cessation_diagnostics: Vec<CessationDiagnostic>,
    pub written_claims: Vec<WrittenClaim>,
    pub speech_coverage: &'static str,
    pub quote_coverage: &'static str,
    pub cessation_coverage: &'static str,
    pub written_claim_coverage: &'static str,
    pub source_identity: Unknown,
    pub coordinate_binding: &'static str,
}

/// Delimiter observations without running speech or event heuristics.
#[derive(Debug, Serialize)]
pub struct QuoteScanReport {
    pub quotations: Vec<WrittenQuote>,
    pub alternative_quote_boundaries: Vec<QuoteBoundaryAlternative>,
    pub quote_scan_diagnostics: Vec<QuoteScanDiagnostic>,
    pub quote_coverage: &'static str,
    pub source_identity: Unknown,
    pub coordinate_binding: &'static str,
}

/// Inspect written delimiters only. The caller must bind offsets to the exact
/// input; observed delimiters do not establish attribution or occurrence.
pub fn inspect_written_quotes(source: &str) -> QuoteScanReport {
    let (quotations, alternative_quote_boundaries, quote_scan_diagnostics) =
        written_quotes(source, false);
    QuoteScanReport {
        quotations,
        alternative_quote_boundaries,
        quote_scan_diagnostics,
        quote_coverage: "partial_unverified_delimiter_candidates",
        source_identity: Unknown::Unknown,
        coordinate_binding: "caller_must_bind_exact_input",
    }
}

/// Compatibility alias for the isolated streaming candidate.
pub type QuoteForms = QuoteScanReport;
pub fn inspect_quote_forms(source: &str) -> QuoteForms {
    inspect_written_quotes(source)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    AnalyzerInit,
    SpeechEvidence(SpeechError),
    InvalidClaimRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReuseError {
    VocabularyMismatch,
    Observation(Error),
}

fn closing(ch: char) -> Option<char> {
    match ch {
        '「' => Some('」'),
        '『' => Some('』'),
        '“' => Some('”'),
        '‘' => Some('’'),
        '"' => Some('"'),
        _ => None,
    }
}

fn is_closing(ch: char) -> bool {
    matches!(ch, '」' | '』' | '”' | '’')
}

fn range(byte_start: usize, byte_end: usize, start_char: usize, end_char: usize) -> SourceRange {
    SourceRange {
        scalars: Span {
            start_char,
            end_char,
        },
        byte_start,
        byte_end,
    }
}

/// Scan only delimiter structure. On a crossed or unclosed quotation, retain
/// the exact remaining source as a damaged fragment; do not invent a closer.
fn written_quotes(
    source: &str,
    report_crossed: bool,
) -> (
    Vec<WrittenQuote>,
    Vec<QuoteBoundaryAlternative>,
    Vec<QuoteScanDiagnostic>,
) {
    struct Frame {
        expected: char,
        start_byte: usize,
        start_scalar: usize,
        opener_len: usize,
        nested: Vec<WrittenQuote>,
        ambiguous_apostrophe: bool,
    }
    fn row(
        source: SourceRange,
        inner: Option<SourceRange>,
        form: QuotationForm,
        nested_quotations: Vec<WrittenQuote>,
        ambiguous: bool,
    ) -> WrittenQuote {
        WrittenQuote {
            source,
            inner,
            form,
            nested_quotations,
            delimiter_status: Unknown::Unknown,
            delimiter_basis: if ambiguous {
                "ambiguous_apostrophe_candidate"
            } else {
                "unverified_written_delimiters"
            },
            speaker_identity: Unknown::Unknown,
            speaker_role: Unknown::Unknown,
            occurrence: Unknown::Unknown,
            narrative_scope: Unknown::Unknown,
        }
    }
    fn retain_damaged(
        stack: &mut Vec<Frame>,
        rows: &mut Vec<WrittenQuote>,
        end_byte: usize,
        end_scalar: usize,
    ) {
        while let Some(frame) = stack.pop() {
            let quote = row(
                range(frame.start_byte, end_byte, frame.start_scalar, end_scalar),
                None,
                QuotationForm::IncompleteOrDamaged,
                frame.nested,
                true,
            );
            if let Some(parent) = stack.last_mut() {
                parent.nested.push(quote);
            } else {
                rows.push(quote);
            }
        }
    }
    let mut rows = Vec::new();
    let mut alternatives = Vec::new();
    let mut diagnostics = Vec::new();
    let mut stack: Vec<Frame> = Vec::new();
    let mut escaped = false;
    let mut scalar_count = 0;
    for (scalar, (byte, ch)) in source.char_indices().enumerate() {
        scalar_count = scalar + 1;
        if escaped {
            escaped = false;
            continue;
        }
        if stack.last().is_some_and(|frame| frame.expected == '"') && ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '’'
            && source[..byte]
                .chars()
                .next_back()
                .is_some_and(char::is_alphabetic)
        {
            let next = source[byte + ch.len_utf8()..].chars().next();
            if let Some(frame) = stack.last_mut() {
                if frame.expected != '’' {
                    // Inside another quoted form this may be a possessive or
                    // apostrophe. Retain the source and mark interpretation unknown.
                    frame.ambiguous_apostrophe = true;
                    continue;
                }
                if next.is_some_and(char::is_alphabetic) {
                    frame.ambiguous_apostrophe = true;
                    // Record this literal boundary even when the scanner also
                    // considers the same glyph an in-word apostrophe.
                    alternatives.push(QuoteBoundaryAlternative {
                        source: range(
                            frame.start_byte,
                            byte + ch.len_utf8(),
                            frame.start_scalar,
                            scalar + 1,
                        ),
                        delimiter_status: Unknown::Unknown,
                        reason: "ambiguous_right_single_quote",
                    });
                    // For ‘don’t!’ a later right single quote can close this
                    // frame. Bound the lookahead and stop at another opener.
                    let later_close = source[byte + ch.len_utf8()..]
                        .chars()
                        .take(256)
                        .take_while(|&c| !matches!(c, '‘' | '“' | '「' | '『' | '"' | '\n' | '\r'))
                        .any(|c| c == '’');
                    if later_close {
                        continue;
                    }
                    // The first ’ may instead close ‘hello’world. Use this
                    // boundary as an explicitly unverified candidate so later
                    // independent quote sources are not swallowed.
                }
            } else if next.is_some_and(char::is_alphabetic) {
                // An apostrophe inside a word outside any supported quotation.
                continue;
            }
        }
        if let Some(frame) = stack.last() {
            if ch == frame.expected {
                let frame = stack.pop().expect("matching frame");
                let inner = (scalar > frame.start_scalar + 1).then(|| {
                    range(
                        frame.start_byte + frame.opener_len,
                        byte,
                        frame.start_scalar + 1,
                        scalar,
                    )
                });
                let ambiguous = frame.ambiguous_apostrophe;
                let quote = row(
                    range(
                        frame.start_byte,
                        byte + ch.len_utf8(),
                        frame.start_scalar,
                        scalar + 1,
                    ),
                    inner,
                    QuotationForm::Complete,
                    frame.nested,
                    ambiguous,
                );
                if let Some(parent) = stack.last_mut() {
                    parent.ambiguous_apostrophe |= ambiguous;
                    parent.nested.push(quote);
                } else {
                    rows.push(quote);
                }
            } else if let Some(expected) = closing(ch) {
                if stack.len() == MAX_QUOTE_DEPTH {
                    let end_scalar = source.chars().count();
                    diagnostics.push(QuoteScanDiagnostic {
                        source: range(byte, source.len(), scalar, end_scalar),
                        status: "unexamined",
                        reason: "nesting_depth_limit",
                        delimiter_status: Unknown::Unknown,
                    });
                    retain_damaged(&mut stack, &mut rows, source.len(), end_scalar);
                    return (rows, alternatives, diagnostics);
                }
                stack.push(Frame {
                    expected,
                    start_byte: byte,
                    start_scalar: scalar,
                    opener_len: ch.len_utf8(),
                    nested: Vec::new(),
                    ambiguous_apostrophe: false,
                });
            } else if is_closing(ch) {
                // Keep the entire suffix as unexamined source. No guessed close.
                let end_scalar = source.chars().count();
                if report_crossed {
                    diagnostics.push(QuoteScanDiagnostic {
                        // Include the conflicting closer conservatively. No
                        // quote boundary after this position is established.
                        source: range(byte, source.len(), scalar, end_scalar),
                        status: "unexamined",
                        reason: "crossed_delimiter",
                        delimiter_status: Unknown::Unknown,
                    });
                }
                retain_damaged(&mut stack, &mut rows, source.len(), end_scalar);
                return (rows, alternatives, diagnostics);
            }
        } else if let Some(expected) = closing(ch) {
            stack.push(Frame {
                expected,
                start_byte: byte,
                start_scalar: scalar,
                opener_len: ch.len_utf8(),
                nested: Vec::new(),
                ambiguous_apostrophe: false,
            });
        } else if is_closing(ch) {
            rows.push(row(
                range(byte, byte + ch.len_utf8(), scalar, scalar + 1),
                None,
                QuotationForm::UnmatchedCloser,
                Vec::new(),
                true,
            ));
        }
    }
    if !stack.is_empty() {
        retain_damaged(&mut stack, &mut rows, source.len(), scalar_count);
    }
    (rows, alternatives, diagnostics)
}

/// Preserve a bounded literal cue before a quote even when the legacy speech
/// parser rejects negation. No grammatical attachment or role is asserted.
fn written_negated_cues<'a>(
    source: &str,
    quotations: &[WrittenQuote],
    lexicon: &'a Lexicon<'a>,
) -> Vec<WrittenNegatedCue<'a>> {
    let mut output = Vec::new();
    for (quote_index, quote) in quotations.iter().enumerate() {
        if quote.form != QuotationForm::Complete {
            continue;
        }
        let end_byte = quote.source.byte_start;
        let start_byte = source[..end_byte]
            .char_indices()
            .rev()
            .take(128)
            .last()
            .map_or(0, |(at, _)| at);
        let before = &source[start_byte..end_byte];
        let before = before.trim_end_matches([' ', '\t']);
        let Some(before) = before.strip_suffix(['：', ':']) else {
            continue;
        };
        let before = before.trim_end_matches([' ', '\t']);
        let Some(cue) = ["說道", "说道", "問道", "问道", "喊道", "答道", "說", "说"]
            .into_iter()
            .find(|cue| before.ends_with(cue))
        else {
            continue;
        };
        let cue_end = start_byte + before.len();
        let cue_start = cue_end - cue.len();
        let before_cue = before[..before.len() - cue.len()].trim_end_matches([' ', '\t']);
        let Some(negation) = ["沒有", "没有", "並未", "并未", "未曾", "尚未", "不曾", "不"]
            .into_iter()
            .find(|word| before_cue.ends_with(word))
        else {
            continue;
        };
        let neg_end = start_byte + before_cue.len();
        let neg_start = neg_end - negation.len();
        let before_neg =
            before_cue[..before_cue.len() - negation.len()].trim_end_matches([' ', '\t']);
        let surface = lexicon
            .actors
            .iter()
            .map(|(surface, _)| *surface)
            .filter(|surface| !surface.is_empty() && before_neg.ends_with(surface))
            .max_by_key(|surface| surface.len());
        let locate = |start: usize, end: usize| -> SourceRange {
            let scalar_start =
                quote.source.scalars.start_char - source[start..end_byte].chars().count();
            let scalar_end =
                quote.source.scalars.start_char - source[end..end_byte].chars().count();
            range(start, end, scalar_start, scalar_end)
        };
        let (written_subject, caller_labels) = if let Some(surface) = surface {
            let subject_end = start_byte + before_neg.len();
            let subject = locate(subject_end - surface.len(), subject_end);
            let labels: std::collections::BTreeSet<_> = lexicon
                .actors
                .iter()
                .filter_map(|&(written, label)| (written == surface).then_some(label))
                .collect();
            let caller_labels = match labels.len() {
                0 => CallerLabels::Unmapped,
                1 => CallerLabels::Unique {
                    label: labels.into_iter().next().expect("one label"),
                },
                _ => CallerLabels::Ambiguous {
                    labels: labels.into_iter().collect(),
                },
            };
            (Some(subject), caller_labels)
        } else {
            (None, CallerLabels::NoWrittenSurface)
        };
        output.push(WrittenNegatedCue {
            cue: locate(cue_start, cue_end),
            negation: locate(neg_start, neg_end),
            written_subject,
            caller_labels,
            potential_quote_index: quote_index,
            basis: "unverified_adjacent_written_form",
            identity: Unknown::Unknown,
            role: Unknown::Unknown,
            occurrence: Unknown::Unknown,
            narrative_scope: Unknown::Unknown,
            quote_attribution: Unknown::Unknown,
        });
    }
    output
}

// Check bounded same-line source immediately following a complete quote.
// A match is only a written cessation form; it is not a speaker attribution.
fn cessation_observations<'a>(
    source: &str,
    quotations: &[WrittenQuote],
    lexicon: &'a Lexicon<'a>,
) -> (Vec<CessationObservation<'a>>, Vec<CessationDiagnostic>) {
    let index = SourceIndex::new(source);
    let mut found = Vec::new();
    let mut diagnostics = Vec::new();
    // Count the end coordinate once, and only if an unexamined suffix needs it.
    let mut end_scalar = None;
    for (quote_index, quote) in quotations.iter().enumerate() {
        if quote.form != QuotationForm::Complete {
            continue;
        }
        let tail = &source[quote.source.byte_end..];
        // Leading spaces consume the same budget as all other characters.
        // Do not use unbounded trim_start_matches on the remaining source.
        let mut skipped = 0;
        let mut at_leading_space = true;
        let mut terminator = None;
        let mut limit_reached = false;
        for (count, (byte, ch)) in tail.char_indices().enumerate() {
            if count == 256 {
                limit_reached = true;
                break;
            }
            if at_leading_space && matches!(ch, ' ' | '\t') {
                skipped = byte + ch.len_utf8();
                continue;
            }
            at_leading_space = false;
            if matches!(
                ch,
                '\n' | '\r' | '「' | '」' | '『' | '』' | '“' | '”' | '‘' | '’' | '"'
            ) {
                break;
            }
            if matches!(ch, '。' | '.') {
                terminator = Some((byte + ch.len_utf8(), count + 1));
                break;
            }
            if matches!(ch, '！' | '!' | '？' | '?') {
                break;
            }
        }
        if limit_reached {
            let end = *end_scalar.get_or_insert_with(|| source.chars().count());
            diagnostics.push(CessationDiagnostic {
                preceding_quote_index: quote_index,
                candidate_source: range(
                    quote.source.byte_end,
                    source.len(),
                    quote.source.scalars.end_char,
                    end,
                ),
                status: "unexamined",
                reason: "suffix_limit",
            });
            continue;
        }
        let Some((length_bytes, length_chars)) = terminator else {
            continue;
        };
        let start_byte = quote.source.byte_end + skipped;
        let start_char = quote.source.scalars.end_char + skipped; // skipped spaces/tabs are ASCII
        let span = Span {
            start_char,
            end_char: quote.source.scalars.end_char + length_chars,
        };
        match written_cessation_evidence::inspect(&index, span, lexicon.actors) {
            Ok(written) => found.push(CessationObservation {
                written,
                preceding_quote_index: quote_index,
                identity: Unknown::Unknown,
                role: Unknown::Unknown,
                occurrence: Unknown::Unknown,
                narrative_scope: Unknown::Unknown,
                quote_attribution: Unknown::Unknown,
            }),
            Err(written_cessation_evidence::InspectError::IncompleteOrUnsupportedForm) => {}
            Err(error) => diagnostics.push(CessationDiagnostic {
                preceding_quote_index: quote_index,
                candidate_source: range(
                    start_byte,
                    quote.source.byte_end + length_bytes,
                    start_char,
                    quote.source.scalars.end_char + length_chars,
                ),
                status: "unexamined",
                reason: match error {
                    written_cessation_evidence::InspectError::InvalidRange => "invalid_range",
                    written_cessation_evidence::InspectError::TagTooLong => "tag_too_long",
                    written_cessation_evidence::InspectError::VocabularyTooLarge => {
                        "vocabulary_too_large"
                    }
                    written_cessation_evidence::InspectError::AmbiguousSubjectSurface => {
                        "ambiguous_subject_surface"
                    }
                    written_cessation_evidence::InspectError::IncompleteOrUnsupportedForm => {
                        unreachable!()
                    }
                },
            }),
        }
    }
    (found, diagnostics)
}

// Convert only selected scalar boundaries in one scan; do not allocate a
// source-sized offset vector or rescan from the beginning for each claim.
fn claim_ranges(source: &str, spans: &[Span]) -> Result<Vec<SourceRange>, Error> {
    let mut targets = Vec::with_capacity(spans.len().saturating_mul(2));
    for span in spans {
        if span.start_char >= span.end_char {
            return Err(Error::InvalidClaimRange);
        }
        targets.extend([span.start_char, span.end_char]);
    }
    targets.sort_unstable();
    targets.dedup();
    let mut bytes = Vec::with_capacity(targets.len());
    let mut next = 0;
    let mut scalar_count = 0;
    for (scalar, (byte, _)) in source.char_indices().enumerate() {
        while next < targets.len() && targets[next] == scalar {
            bytes.push(byte);
            next += 1;
        }
        scalar_count = scalar + 1;
    }
    while next < targets.len() && targets[next] == scalar_count {
        bytes.push(source.len());
        next += 1;
    }
    if next != targets.len() {
        return Err(Error::InvalidClaimRange);
    }
    spans
        .iter()
        .map(|span| {
            let start = targets
                .binary_search(&span.start_char)
                .map_err(|_| Error::InvalidClaimRange)?;
            let end = targets
                .binary_search(&span.end_char)
                .map_err(|_| Error::InvalidClaimRange)?;
            Ok(SourceRange {
                scalars: *span,
                byte_start: bytes[start],
                byte_end: bytes[end],
            })
        })
        .collect()
}

/// Inspect the exact borrowed input. Source positions use both Unicode scalar
/// and UTF-8 byte offsets. A returned quote does not prove speech, attribution,
/// character identity, or occurrence. This is not a streaming interface.
pub fn inspect<'vocabulary>(
    source: &str,
    title: &str,
    locale: Locale,
    lexicon: &'vocabulary Lexicon<'vocabulary>,
) -> Result<Report<'vocabulary>, Error> {
    let analyzer = NarrativeAnalyzer::new(locale, lexicon).map_err(|_| Error::AnalyzerInit)?;
    inspect_using_analyzer(source, title, &analyzer, lexicon, false)
}

/// Reuse a prebuilt analyzer only with the exact same work vocabulary. This
/// preserves its configured locale and does not authenticate any observation.
pub fn inspect_with_analyzer<'vocabulary>(
    source: &str,
    title: &str,
    analyzer: &NarrativeAnalyzer<'_>,
    lexicon: &'vocabulary Lexicon<'vocabulary>,
) -> Result<Report<'vocabulary>, ReuseError> {
    if !analyzer.matches_lexicon(lexicon) {
        return Err(ReuseError::VocabularyMismatch);
    }
    inspect_using_analyzer(source, title, analyzer, lexicon, false).map_err(ReuseError::Observation)
}

// Only source-scoped inspection opts into the new diagnostic. The raw APIs
// retain their existing serialized observations for callers using older formats.
pub(crate) fn inspect_scoped_with_analyzer<'vocabulary>(
    source: &str,
    title: &str,
    analyzer: &NarrativeAnalyzer<'_>,
    lexicon: &'vocabulary Lexicon<'vocabulary>,
) -> Result<Report<'vocabulary>, ReuseError> {
    if !analyzer.matches_lexicon(lexicon) {
        return Err(ReuseError::VocabularyMismatch);
    }
    inspect_using_analyzer(source, title, analyzer, lexicon, true).map_err(ReuseError::Observation)
}

fn inspect_using_analyzer<'vocabulary>(
    source: &str,
    title: &str,
    analyzer: &NarrativeAnalyzer<'_>,
    lexicon: &'vocabulary Lexicon<'vocabulary>,
    report_crossed: bool,
) -> Result<Report<'vocabulary>, Error> {
    let acts = analyzer.discover_speech_acts(source, title);
    // The adapter checks every range against the original input, ignores the
    // legacy resolved speaker, and labels every vocabulary match as unverified.
    let mapped = written_speech_alias_evidence::map_written_speech(source, &acts, lexicon)
        .map_err(Error::SpeechEvidence)?;
    let (quotations, alternative_quote_boundaries, quote_scan_diagnostics) =
        written_quotes(source, report_crossed);
    let written_negated_cues = written_negated_cues(source, &quotations, lexicon);
    let (cessation, cessation_diagnostics) = cessation_observations(source, &quotations, lexicon);
    let mut claim_candidates = std::collections::BTreeSet::new();
    if !cessation.is_empty() {
        let mut tags = std::collections::BTreeMap::new();
        for (index, item) in cessation.iter().enumerate() {
            let tag = item.written.tag.scalars;
            tags.entry((tag.start_char, tag.end_char))
                .or_insert_with(Vec::new)
                .push(index);
        }
        // The legacy parser decides which text resembles a reported claim;
        // actor/object names and factuality conclusions are never copied.
        let analysis = analyzer.analyze_source_mentions(source, title);
        for event in &analysis.events {
            if event.status != "reported" {
                continue;
            }
            let Some(claim) = event.evidence.first().copied() else {
                continue;
            };
            for tag in event.evidence.iter().skip(1) {
                if let Some(indices) = tags.get(&(tag.start_char, tag.end_char)) {
                    for &cessation_index in indices {
                        let quote_index = cessation[cessation_index].preceding_quote_index;
                        let quote = &quotations[quote_index];
                        if quote.inner.is_some_and(|inner| {
                            inner.scalars.start_char <= claim.start_char
                                && claim.end_char <= inner.scalars.end_char
                        }) {
                            claim_candidates.insert((
                                claim.start_char,
                                claim.end_char,
                                quote_index,
                                cessation_index,
                            ));
                        }
                    }
                }
            }
        }
    }
    let candidates: Vec<_> = claim_candidates.into_iter().collect();
    let spans: Vec<_> = candidates
        .iter()
        .map(|&(start_char, end_char, _, _)| Span {
            start_char,
            end_char,
        })
        .collect();
    let ranges = claim_ranges(source, &spans)?;
    let written_claims = candidates
        .into_iter()
        .zip(ranges)
        .map(
            |((_, _, quote_index, cessation_index), source)| WrittenClaim {
                source,
                potential_quote_index: quote_index,
                potential_cessation_index: cessation_index,
                selection_basis: "unverified_parser_and_source_containment",
                identity: Unknown::Unknown,
                role: Unknown::Unknown,
                occurrence: Unknown::Unknown,
                narrative_scope: Unknown::Unknown,
                quote_attribution: Unknown::Unknown,
            },
        )
        .collect();
    let speech = mapped
        .speech
        .into_iter()
        .map(|written| {
            let payload = written.payload.scalars;
            let mut candidates = quotations
                .iter()
                .enumerate()
                .filter(|(_, quote)| {
                    quote.form == QuotationForm::Complete
                        && quote.inner.is_some_and(|inner| {
                            inner.scalars.start_char <= payload.start_char
                                && payload.end_char <= inner.scalars.end_char
                        })
                })
                .map(|(index, _)| index);
            let first = candidates.next();
            let potential_quote_index = if candidates.next().is_none() {
                first
            } else {
                None
            };
            SpeechObservation {
                written,
                potential_quote_index,
                role: Unknown::Unknown,
                quote_attribution: Unknown::Unknown,
            }
        })
        .collect();
    Ok(Report {
        quotations,
        alternative_quote_boundaries,
        quote_scan_diagnostics,
        speech,
        written_negated_cues,
        cessation,
        cessation_diagnostics,
        written_claims,
        speech_coverage: "partial_heuristic",
        quote_coverage: "partial_unverified_delimiter_candidates",
        cessation_coverage: "partial_bounded_adjacent_written_forms",
        written_claim_coverage: "partial_reported_claims_with_recognized_cessation",
        source_identity: Unknown::Unknown,
        coordinate_binding: "exact_borrowed_input_only",
    })
}
