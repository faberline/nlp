//! Source-backed written cessation evidence, separate from legacy actor fields.
//!
//! A caller-supplied label is vocabulary data, not identity authentication.
//! Matching this local form does not prove that the preceding quotation was
//! spoken by this subject, or that its asserted events actually occurred.
use crate::Span;
use serde::Serialize;

const CHECKPOINT_SCALARS: usize = 256;
const MAX_TAG_SCALARS: usize = 256;
const MAX_VOCABULARY: usize = 512;
const MAX_SURFACE_BYTES: usize = 128;
const MAX_LABEL_BYTES: usize = 256;

/// Reusable index into one exact borrowed source. Build once per input rather
/// than scanning from the beginning for every source range.
pub struct SourceIndex<'source> {
    source: &'source str,
    checkpoints: Vec<usize>,
    scalars: usize,
}

impl<'source> SourceIndex<'source> {
    pub fn new(source: &'source str) -> Self {
        let mut checkpoints = vec![0];
        let mut scalars = 0;
        for (byte, _) in source.char_indices() {
            if scalars > 0 && scalars % CHECKPOINT_SCALARS == 0 {
                checkpoints.push(byte);
            }
            scalars += 1;
        }
        Self {
            source,
            checkpoints,
            scalars,
        }
    }

    fn byte_at(&self, scalar: usize) -> Option<usize> {
        if scalar > self.scalars {
            return None;
        }
        if scalar == self.scalars {
            return Some(self.source.len());
        }
        let checkpoint = scalar / CHECKPOINT_SCALARS;
        let base = *self.checkpoints.get(checkpoint)?;
        let within = scalar % CHECKPOINT_SCALARS;
        self.source[base..]
            .char_indices()
            .nth(within)
            .map(|(offset, _)| base + offset)
    }

    fn source_range(&self, span: Span) -> Result<SourceRange, InspectError> {
        if span.start_char >= span.end_char || span.end_char > self.scalars {
            return Err(InspectError::InvalidRange);
        }
        Ok(SourceRange {
            scalars: span,
            byte_start: self
                .byte_at(span.start_char)
                .ok_or(InspectError::InvalidRange)?,
            byte_end: self
                .byte_at(span.end_char)
                .ok_or(InspectError::InvalidRange)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SourceRange {
    pub scalars: Span,
    pub byte_start: usize,
    pub byte_end: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct CallerLabel<'vocabulary> {
    /// Caller-provided mapping; never a proven identity.
    pub value: &'vocabulary str,
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct WrittenCessation<'vocabulary> {
    pub tag: SourceRange,
    /// Caller-matched written surface; its semantic role remains unknown.
    pub subject_surface: SourceRange,
    /// Exact source text of the stop-continuing-speech cue.
    pub cue: SourceRange,
    /// All distinct labels supplied for that spelling; no choice is made.
    pub caller_labels: Vec<CallerLabel<'vocabulary>>,
    pub written_subject_role_status: &'static str,
    pub identity_status: &'static str,
    pub quotation_attribution_status: &'static str,
    pub occurrence_status: &'static str,
    pub global_scope_status: &'static str,
    pub source_identity_status: &'static str,
    pub coordinate_binding: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InspectError {
    InvalidRange,
    TagTooLong,
    VocabularyTooLarge,
    IncompleteOrUnsupportedForm,
    AmbiguousSubjectSurface,
}

/// Verify one supplied tag span against the exact source. Limits fail before
/// emitting a partial label list. `vocabulary` consists of written spelling and
/// caller label pairs; the label is returned without being authenticated.
pub fn inspect<'vocabulary>(
    index: &SourceIndex<'_>,
    tag_span: Span,
    vocabulary: &'vocabulary [(&'vocabulary str, &'vocabulary str)],
) -> Result<WrittenCessation<'vocabulary>, InspectError> {
    if vocabulary.len() > MAX_VOCABULARY
        || vocabulary.iter().any(|(surface, label)| {
            surface.len() > MAX_SURFACE_BYTES || label.len() > MAX_LABEL_BYTES
        })
    {
        return Err(InspectError::VocabularyTooLarge);
    }
    let tag = index.source_range(tag_span)?;
    if tag_span.end_char - tag_span.start_char > MAX_TAG_SCALARS {
        return Err(InspectError::TagTooLong);
    }
    let written = &index.source[tag.byte_start..tag.byte_end];
    if written.chars().any(|ch| {
        matches!(
            ch,
            '\r' | '\n' | '「' | '」' | '『' | '』' | '“' | '”' | '‘' | '’' | '"'
        )
    }) {
        return Err(InspectError::IncompleteOrUnsupportedForm);
    }
    let body = written
        .strip_suffix('。')
        .or_else(|| written.strip_suffix('.'))
        .ok_or(InspectError::IncompleteOrUnsupportedForm)?;
    let mut subject: Option<&str> = None;
    let mut cue_offset = 0;
    let mut cue_len = 0;
    for &(surface, _) in vocabulary {
        if surface.is_empty() || subject == Some(surface) {
            continue;
        }
        let Some(after_subject) = body.strip_prefix(surface) else {
            continue;
        };
        let Some(after_stop) = after_subject.strip_prefix("停了") else {
            continue;
        };
        let Some(after_duration) = ["一會", "一会", "一下", "片刻"]
            .iter()
            .find_map(|duration| after_stop.strip_prefix(*duration))
        else {
            continue;
        };
        let Some(after_comma) = after_duration
            .strip_prefix('，')
            .or_else(|| after_duration.strip_prefix(','))
        else {
            continue;
        };
        let cue = after_comma.trim_start();
        if !matches!(
            cue,
            "沒再說下去" | "没再说下去" | "沒有再說下去" | "没有再说下去"
        ) {
            continue;
        }
        if subject.is_some() {
            return Err(InspectError::AmbiguousSubjectSurface);
        }
        subject = Some(surface);
        cue_offset = body.len() - cue.len();
        cue_len = cue.len();
    }
    let subject = subject.ok_or(InspectError::IncompleteOrUnsupportedForm)?;
    let scalar_range = |start: usize, end: usize| -> Result<SourceRange, InspectError> {
        let before = written
            .get(..start)
            .ok_or(InspectError::InvalidRange)?
            .chars()
            .count();
        let through = written
            .get(..end)
            .ok_or(InspectError::InvalidRange)?
            .chars()
            .count();
        index.source_range(Span {
            start_char: tag_span.start_char + before,
            end_char: tag_span.start_char + through,
        })
    };
    let mut labels = Vec::new();
    for &(surface, label) in vocabulary {
        if surface == subject
            && !labels
                .iter()
                .any(|row: &CallerLabel<'_>| row.value == label)
        {
            labels.push(CallerLabel {
                value: label,
                status: "caller_supplied_unverified",
            });
        }
    }
    Ok(WrittenCessation {
        tag,
        subject_surface: scalar_range(0, subject.len())?,
        cue: scalar_range(cue_offset, cue_offset + cue_len)?,
        caller_labels: labels,
        written_subject_role_status: "unknown",
        identity_status: "unknown",
        quotation_attribution_status: "unknown",
        occurrence_status: "unknown",
        global_scope_status: "unknown",
        source_identity_status: "unknown",
        coordinate_binding: "exact_borrowed_source_only",
    })
}
