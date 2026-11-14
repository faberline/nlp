//! Span locking (`SpanLockTable`) and placeholder conservation verification.
//!
//! Inline opaque spans (`code`, `formula`, `url`, `citation`, `emoji`), matched glossary entries,
//! and formatted entities are replaced with ASCII-safe sentinel placeholders (`[[PH0]]`, `[[PH1]]`,
//! ...) before neural seq2seq decoding and restored deterministically afterward while recording
//! exact target scalar offsets.

use faberline_nlp::Span;
use faberline_nlp::doc::OpaqueKind;

use crate::schema::{GlossaryKind, TranslateUnknownKind};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LockOrigin {
    InlineOpaque(OpaqueKind),
    Glossary {
        canonical: String,
        kind: GlossaryKind,
    },
    Entity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockedSpan {
    pub index: usize,
    pub placeholder: String,
    pub source_span: Span,
    pub replacement: String,
    pub origin: LockOrigin,
    /// Must be conserved verbatim in target output (always true for `InlineOpaque` and `Glossary`).
    pub mandatory: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SpanLockTable {
    pub locks: Vec<LockedSpan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoredLock {
    pub index: usize,
    pub source_span: Span,
    pub target_span: Span,
    pub origin: LockOrigin,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoreReport {
    pub text: String,
    pub restored: Vec<RestoredLock>,
    pub issues: Vec<(TranslateUnknownKind, Span, String)>,
}

impl SpanLockTable {
    pub fn new() -> Self {
        Self { locks: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.locks.is_empty()
    }

    /// Add candidate locks, resolving overlaps by `(priority, start_char, longer_first)` where
    /// priority `0` = `InlineOpaque`, `1` = `Glossary`, `2` = `Entity`.
    pub fn build(mut candidates: Vec<(u8, Span, String, LockOrigin, bool)>) -> Self {
        candidates.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.start_char.cmp(&b.1.start_char))
                .then_with(|| (b.1.end_char - b.1.start_char).cmp(&(a.1.end_char - a.1.start_char)))
        });

        let mut kept: Vec<(Span, String, LockOrigin, bool)> = Vec::new();
        for (_prio, span, replacement, origin, mandatory) in candidates {
            if span.start_char >= span.end_char {
                continue;
            }
            let overlaps = kept
                .iter()
                .any(|(s, _, _, _)| span.start_char < s.end_char && s.start_char < span.end_char);
            if !overlaps {
                kept.push((span, replacement, origin, mandatory));
            }
        }

        // Re-sort in source text order so placeholder indices `0..k` follow reading order.
        kept.sort_by_key(|(s, _, _, _)| (s.start_char, s.end_char));

        let locks = kept
            .into_iter()
            .enumerate()
            .map(
                |(index, (source_span, replacement, origin, mandatory))| LockedSpan {
                    index,
                    placeholder: format!("[[PH{index}]]"),
                    source_span,
                    replacement,
                    origin,
                    mandatory,
                },
            )
            .collect();

        Self { locks }
    }

    /// Mask `chars[sent_span.start_char..sent_span.end_char]` by substituting every locked span
    /// inside `sent_span` with its `[[PHk]]` sentinel.
    pub fn mask_span(&self, chars: &[char], sent_span: Span) -> String {
        if self.locks.is_empty() {
            return chars[sent_span.start_char..sent_span.end_char]
                .iter()
                .collect();
        }
        let mut out = String::new();
        let mut cursor = sent_span.start_char;
        for lock in &self.locks {
            if lock.source_span.end_char <= sent_span.start_char
                || lock.source_span.start_char >= sent_span.end_char
            {
                continue;
            }
            let s = lock.source_span.start_char.max(cursor);
            let e = lock.source_span.end_char.min(sent_span.end_char);
            if s > cursor {
                for &ch in &chars[cursor..s] {
                    out.push(ch);
                }
            }
            out.push_str(&lock.placeholder);
            cursor = e;
        }
        if cursor < sent_span.end_char {
            for &ch in &chars[cursor..sent_span.end_char] {
                out.push(ch);
            }
        }
        out
    }

    /// Restore all `[[PHk]]` placeholders in `decoded_text`, enforcing 100% conservation of
    /// mandatory (`InlineOpaque` and `Glossary`) spans and recording exact relative target spans
    /// (`0..target_chars` within the returned sentence string).
    pub fn restore(&self, decoded_text: &str) -> RestoreReport {
        self.restore_with_spans(decoded_text, &mut [])
    }

    /// Restore placeholders and move ranges from the masked text into the restored text.
    /// A range boundary inside a placeholder includes the corresponding replacement.
    pub(crate) fn restore_with_spans(
        &self,
        decoded_text: &str,
        tracked_spans: &mut [Span],
    ) -> RestoreReport {
        if self.locks.is_empty() {
            return RestoreReport {
                text: decoded_text.to_owned(),
                restored: Vec::new(),
                issues: Vec::new(),
            };
        }

        let mut seen = vec![false; self.locks.len()];
        let mut restored = Vec::with_capacity(self.locks.len());
        let mut issues = Vec::new();
        let mut out = String::with_capacity(decoded_text.len() + 16);
        let mut out_chars = 0usize;
        let mut input_chars = 0usize;
        let mut replacements = Vec::new();

        let mut rest = decoded_text;
        while let Some(open_pos) = rest.find("[[PH") {
            let prefix = &rest[..open_pos];
            out.push_str(prefix);
            let prefix_chars = prefix.chars().count();
            out_chars += prefix_chars;
            input_chars += prefix_chars;

            let after_open = &rest[open_pos + 4..];
            if let Some(close_rel) = after_open.find("]]") {
                let digits = &after_open[..close_rel];
                if !digits.is_empty()
                    && digits.bytes().all(|b| b.is_ascii_digit())
                    && let Ok(idx) = digits.parse::<usize>()
                    && let Some(lock) = self.locks.get(idx)
                {
                    let rep_chars = lock.replacement.chars().count();
                    let target_span = Span {
                        start_char: out_chars,
                        end_char: out_chars + rep_chars,
                    };
                    let marker_end = input_chars + 4 + digits.len() + 2;
                    if !tracked_spans.is_empty() {
                        replacements.push((
                            Span {
                                start_char: input_chars,
                                end_char: marker_end,
                            },
                            target_span,
                        ));
                    }
                    input_chars = marker_end;
                    out.push_str(&lock.replacement);
                    out_chars += rep_chars;
                    if !seen[idx] {
                        seen[idx] = true;
                        restored.push(RestoredLock {
                            index: idx,
                            source_span: lock.source_span,
                            target_span,
                            origin: lock.origin.clone(),
                        });
                    }
                    rest = &after_open[close_rel + 2..];
                    continue;
                }
            }

            // Not a valid `[[PHk]]` marker; emit `[[PH` literally and advance.
            out.push_str("[[PH");
            out_chars += 4;
            input_chars += 4;
            rest = after_open;
        }
        out.push_str(rest);
        out_chars += rest.chars().count();

        let map_position = |position: usize, is_end: bool| {
            let mut input_end = 0usize;
            let mut output_end = 0usize;
            for &(masked, restored) in &replacements {
                if position <= masked.start_char {
                    break;
                }
                if position < masked.end_char {
                    return if is_end {
                        restored.end_char
                    } else {
                        restored.start_char
                    };
                }
                input_end = masked.end_char;
                output_end = restored.end_char;
            }
            output_end + position - input_end
        };
        for span in tracked_spans {
            span.start_char = map_position(span.start_char, false);
            span.end_char = map_position(span.end_char, true);
        }

        // Conservation check: any mandatory lock missing from neural decode is deterministically
        // appended (before final sentence punctuation if present) so zero code/URL/formula/glossary
        // content can ever vanish silently.
        for (idx, lock) in self.locks.iter().enumerate() {
            if !seen[idx] && lock.mandatory {
                if !out.is_empty() && !out.ends_with(' ') {
                    out.push(' ');
                    out_chars += 1;
                }
                let rep_chars = lock.replacement.chars().count();
                let target_span = Span {
                    start_char: out_chars,
                    end_char: out_chars + rep_chars,
                };
                out.push_str(&lock.replacement);
                out_chars += rep_chars;
                restored.push(RestoredLock {
                    index: idx,
                    source_span: lock.source_span,
                    target_span,
                    origin: lock.origin.clone(),
                });
                issues.push((
                    TranslateUnknownKind::PlaceholderRepaired,
                    lock.source_span,
                    format!(
                        "repaired missing mandatory placeholder {}",
                        lock.placeholder
                    ),
                ));
            }
        }

        restored.sort_by_key(|r| r.index);
        RestoreReport {
            text: out,
            restored,
            issues,
        }
    }
}
