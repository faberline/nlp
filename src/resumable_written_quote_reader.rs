//! Resumable, source-preserving written-quote observations.
//!
//! The caller receives every decoded scalar and its original byte/scalar range.
//! Quote and postposed-tag records describe the written form only. They do not
//! establish person identity, occurrence, narration scope, or file identity.
//! Preserve all scalar steps and the terminal step; a quote observation alone
//! does not prove complete source coverage. Input is read through one bounded
//! buffer and only a bounded delimiter stack and suffix are retained. On an
//! I/O error the reader pauses without discarding state. The caller decides
//! whether to retry or stop; it is responsible for retaining all emitted steps.
//! On a non-EOF terminal error, also read `pending_at_termination` before
//! `into_parts`; the pending source is deliberately not labeled as EOF.
pub use crate::written_quote_stream_reader::{
    Damage, DamageKind, End, EndKind, InitError, MAX_DEPTH, MAX_POST_SCALARS, MAX_TERM_BYTES,
    MAX_VOCAB, Observation, Position, PostUnknownReason, Range, Step, Unknowns, Vocabulary,
};
use std::io::{self, BufReader, Read};

fn unknowns() -> Unknowns {
    Unknowns {
        identity: "unknown",
        occurrence: "unknown",
        scope: "unknown",
        source_identity: "unknown",
    }
}

/// A pause emits no scalar and no quote observation. The reader retains its
/// delimiter stack and pending postposed tag. `raw_prefix` is already consumed
/// from the reader and must not be replayed when retrying on this same object.
#[derive(Debug)]
pub enum ResumableStep {
    Item(Step),
    Paused {
        kind: io::ErrorKind,
        valid_until: Position,
        raw_prefix: Vec<u8>,
    },
}

/// A postposed form was only partly inspected before the caller stopped.
#[derive(Debug)]
pub struct StoppedPendingPostposed {
    pub quote: Range,
    pub scanned: Range,
    pub reason: &'static str,
}

/// An explicit caller stop following an I/O error. This is not EOF.
#[derive(Debug)]
pub struct Stopped {
    pub kind: io::ErrorKind,
    pub valid_until: Position,
    pub raw_prefix: Vec<u8>,
    pub unfinished_from: Option<Position>,
    pub unfinished_depth: usize,
    pub damage: Option<Damage>,
    pub pending_postposed: Option<StoppedPendingPostposed>,
    pub source_identity: &'static str,
    pub global_scope: &'static str,
    pub coverage: &'static str,
}

#[derive(Clone, Copy)]
struct Open {
    closer: char,
    start: Position,
    payload: Position,
    depth: usize,
}
#[derive(Clone, Copy)]
struct Scalar {
    ch: char,
    start: Position,
    end: Position,
}
struct Pending {
    quote: Range,
    suffix: Vec<Scalar>,
}

fn closer(ch: char) -> Option<char> {
    match ch {
        '「' => Some('」'),
        '『' => Some('』'),
        '“' => Some('”'),
        '‘' => Some('’'),
        '"' => Some('"'),
        _ => None,
    }
}
fn is_closer(ch: char) -> bool {
    matches!(ch, '」' | '』' | '”' | '’')
}
fn is_han(ch: char) -> bool {
    matches!(ch as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x2fa1f | 0x30000..=0x323af)
}
fn width(lead: u8) -> Option<usize> {
    match lead {
        0x00..=0x7f => Some(1),
        0xc2..=0xdf => Some(2),
        0xe0..=0xef => Some(3),
        0xf0..=0xf4 => Some(4),
        _ => None,
    }
}

/// `Read` may read ahead into the internal BufReader. On a pause, call
/// `next_step` to retry without losing context, or `stop_after_pause` to end.
/// If the reader is consumed with `into_parts`, replay the returned prefix
/// followed by the returned BufReader; this does not preserve parser state.
pub struct ResumableReader<'a, R: Read> {
    inner: BufReader<R>,
    vocabulary: Vocabulary<'a>,
    position: Position,
    stack: Vec<Open>,
    pending: Option<Pending>,
    damage: Option<Damage>,
    raw_prefix: Vec<u8>,
    escaped: bool,
    terminal_pending: Option<StoppedPendingPostposed>,
    paused: Option<io::ErrorKind>,
    ended: bool,
}
impl<'a, R: Read> ResumableReader<'a, R> {
    /// Validation occurs before any read. No external source offset is trusted.
    pub fn new(reader: R, vocabulary: Vocabulary<'a>) -> Result<Self, (InitError, R)> {
        if vocabulary.subject_surfaces.len() > MAX_VOCAB
            || vocabulary.title_suffixes.len() > MAX_VOCAB
            || vocabulary
                .subject_surfaces
                .iter()
                .chain(vocabulary.title_suffixes)
                .any(|s| s.len() > MAX_TERM_BYTES)
        {
            return Err((InitError::VocabularyTooLarge, reader));
        }
        Ok(Self {
            inner: BufReader::new(reader),
            vocabulary,
            position: Position { byte: 0, scalar: 0 },
            stack: Vec::new(),
            pending: None,
            damage: None,
            raw_prefix: Vec::new(),
            escaped: false,
            terminal_pending: None,
            paused: None,
            ended: false,
        })
    }

    /// Returns buffered unread bytes intact, plus any failed partial scalar.
    pub fn into_parts(self) -> (Vec<u8>, BufReader<R>) {
        (self.raw_prefix, self.inner)
    }

    /// Pending postposed source on a non-EOF terminal error. Read this before
    /// consuming the reader with `into_parts`. It does not claim completeness.
    pub fn pending_at_termination(&self) -> Option<&StoppedPendingPostposed> {
        self.terminal_pending.as_ref()
    }

    fn read_byte(&mut self) -> Result<Option<u8>, io::ErrorKind> {
        let mut byte = [0_u8; 1];
        match self.inner.read(&mut byte) {
            Ok(0) => Ok(None),
            Ok(_) => Ok(Some(byte[0])),
            Err(e) => Err(e.kind()),
        }
    }

    fn pause(&mut self, kind: io::ErrorKind) -> ResumableStep {
        self.paused = Some(kind);
        ResumableStep::Paused {
            kind,
            valid_until: self.position,
            raw_prefix: self.raw_prefix.clone(),
        }
    }

    fn finish(&mut self, kind: EndKind) -> Step {
        self.ended = true;
        self.paused = None;
        let mut observations = Vec::new();
        if let Some(pending) = self.pending.take() {
            match &kind {
                EndKind::Eof => observations.push(unknown(
                    pending,
                    self.position,
                    PostUnknownReason::EndOfInput,
                )),
                _ => {
                    let reason = match &kind {
                        EndKind::InvalidUtf8 => "invalid_utf8",
                        EndKind::TruncatedUtf8 => "truncated_utf8",
                        EndKind::OffsetOverflow => "offset_overflow",
                        EndKind::Io(_) => "io_error",
                        EndKind::Eof => unreachable!(),
                    };
                    self.terminal_pending = Some(StoppedPendingPostposed {
                        quote: pending.quote,
                        scanned: Range {
                            start: pending.quote.end,
                            end: self.position,
                        },
                        reason,
                    });
                }
            }
        }
        Step::End(End {
            kind,
            valid_until: self.position,
            raw_prefix: self.raw_prefix.clone(),
            unfinished_from: self.stack.first().map(|o| o.start),
            unfinished_depth: self.stack.len(),
            damage: self.damage,
            observations,
            source_identity: "unknown",
            global_scope: "unknown",
            coverage: "partial",
        })
    }

    /// Stop after the latest pause. No action occurs if no pause is pending.
    /// A stopped reader returns its failed scalar prefix via `into_parts`.
    pub fn stop_after_pause(&mut self) -> Option<Stopped> {
        if self.ended {
            return None;
        }
        let kind = self.paused?;
        self.paused = None;
        self.ended = true;
        let pending_postposed = self.pending.take().map(|pending| StoppedPendingPostposed {
            quote: pending.quote,
            scanned: Range {
                start: pending.quote.end,
                end: self.position,
            },
            reason: "caller_stopped_after_io",
        });
        Some(Stopped {
            kind,
            valid_until: self.position,
            raw_prefix: self.raw_prefix.clone(),
            unfinished_from: self.stack.first().map(|o| o.start),
            unfinished_depth: self.stack.len(),
            damage: self.damage,
            pending_postposed,
            source_identity: "unknown",
            global_scope: "unknown",
            coverage: "partial",
        })
    }

    /// Read until one scalar is decoded or a read pauses. Every I/O error,
    /// including Interrupted and WouldBlock, pauses without an automatic retry.
    /// Exactly one terminal item follows; later calls return None. `Read::read`
    /// returning zero is treated as EOF.
    pub fn next_step(&mut self) -> Option<ResumableStep> {
        if self.ended {
            return None;
        }
        self.paused = None;
        if self.raw_prefix.is_empty() {
            match self.read_byte() {
                Ok(Some(b)) => self.raw_prefix.push(b),
                Ok(None) => return Some(ResumableStep::Item(self.finish(EndKind::Eof))),
                Err(kind) => return Some(self.pause(kind)),
            }
        }
        let Some(need) = width(self.raw_prefix[0]) else {
            return Some(ResumableStep::Item(self.finish(EndKind::InvalidUtf8)));
        };
        while self.raw_prefix.len() < need {
            match self.read_byte() {
                Ok(Some(b)) => self.raw_prefix.push(b),
                Ok(None) => return Some(ResumableStep::Item(self.finish(EndKind::TruncatedUtf8))),
                Err(kind) => return Some(self.pause(kind)),
            }
        }
        let ch = match std::str::from_utf8(&self.raw_prefix)
            .ok()
            .and_then(|s| s.chars().next())
        {
            Some(ch) => ch,
            None => return Some(ResumableStep::Item(self.finish(EndKind::InvalidUtf8))),
        };
        let at = self.position;
        let Some(byte) = at.byte.checked_add(need) else {
            return Some(ResumableStep::Item(self.finish(EndKind::OffsetOverflow)));
        };
        let Some(scalar) = at.scalar.checked_add(1) else {
            return Some(ResumableStep::Item(self.finish(EndKind::OffsetOverflow)));
        };
        self.position = Position { byte, scalar };
        self.raw_prefix.clear();
        let item = Scalar {
            ch,
            start: at,
            end: self.position,
        };
        let mut observations = Vec::new();
        if let Some(mut pending) = self.pending.take() {
            if closer(ch).is_some() || is_closer(ch) {
                observations.push(unknown(pending, at, PostUnknownReason::UnexpectedDelimiter));
            } else if matches!(ch, '\r' | '\n') {
                observations.push(unknown(pending, at, PostUnknownReason::LineBoundary));
            } else {
                pending.suffix.push(item);
                if matches!(ch, '。' | '！' | '？' | '.' | '!' | '?') {
                    observations.push(classify(pending, self.vocabulary));
                } else if pending.suffix.len() >= MAX_POST_SCALARS {
                    observations.push(unknown(
                        pending,
                        self.position,
                        PostUnknownReason::SuffixLimit,
                    ));
                } else {
                    self.pending = Some(pending);
                }
            }
        }
        let escaped = self.escaped;
        self.escaped = false;
        if !escaped && self.stack.last().is_some_and(|open| open.closer == '"') && ch == '\\' {
            self.escaped = true;
        }
        if self.damage.is_none() && !escaped {
            if self.stack.last().is_some_and(|open| open.closer == ch) {
                let open = self.stack.pop().expect("checked nonempty");
                let quote = Range {
                    start: open.start,
                    end: self.position,
                };
                observations.push(Observation::WrittenQuote {
                    quote,
                    payload: Range {
                        start: open.payload,
                        end: at,
                    },
                    depth: open.depth,
                    unknowns: unknowns(),
                });
                if self.stack.is_empty() {
                    self.pending = Some(Pending {
                        quote,
                        suffix: Vec::new(),
                    });
                } else {
                    // The enclosing quote makes postposed attribution ambiguous.
                    // Emit an explicit unknown instead of silently omitting it.
                    observations.push(Observation::PostposedUnexamined {
                        quote,
                        scanned: Range {
                            start: self.position,
                            end: self.position,
                        },
                        reason: PostUnknownReason::NestedContext,
                    });
                }
            } else if let Some(end) = closer(ch) {
                if self.stack.len() == MAX_DEPTH {
                    let d = Damage {
                        kind: DamageKind::DepthLimit,
                        unexamined_from: self.stack.first().map_or(at, |o| o.start),
                        detected_at: at,
                    };
                    self.damage = Some(d);
                    self.stack.clear();
                    observations.push(Observation::Damage(d));
                } else {
                    self.stack.push(Open {
                        closer: end,
                        start: at,
                        payload: self.position,
                        depth: self.stack.len() + 1,
                    });
                }
            } else if is_closer(ch) {
                let d = Damage {
                    kind: DamageKind::MismatchedCloser,
                    unexamined_from: self.stack.first().map_or(at, |o| o.start),
                    detected_at: at,
                };
                self.damage = Some(d);
                self.stack.clear();
                observations.push(Observation::Damage(d));
            }
        }
        Some(ResumableStep::Item(Step::Scalar {
            ch,
            range: Range {
                start: at,
                end: self.position,
            },
            observations,
        }))
    }
}

fn unknown(pending: Pending, end: Position, reason: PostUnknownReason) -> Observation {
    Observation::PostposedUnexamined {
        quote: pending.quote,
        scanned: Range {
            start: pending.quote.end,
            end,
        },
        reason,
    }
}
fn classify(pending: Pending, vocabulary: Vocabulary<'_>) -> Observation {
    let suffix = &pending.suffix;
    let Some(terminator) = suffix.last().copied() else {
        let end = pending.quote.end;
        return unknown(pending, end, PostUnknownReason::NoSupportedTag);
    };
    if !matches!(terminator.ch, '。' | '.') {
        return unknown(pending, terminator.end, PostUnknownReason::NoSupportedTag);
    }
    let mut left = 0;
    while left + 1 < suffix.len() && matches!(suffix[left].ch, ' ' | '\t') {
        left += 1;
    }
    let body: String = suffix[left..suffix.len() - 1]
        .iter()
        .map(|s| s.ch)
        .collect();
    for cue in ["說道", "说道", "問道", "问道", "喊道", "答道", "說", "说"] {
        if let Some(lead) = body.strip_suffix(cue) {
            let lead_len = lead.chars().count();
            if lead_len == 0 || lead_len > 12 || !lead.chars().all(is_han) {
                break;
            }
            let cue_start = left + lead_len;
            let gap = Range {
                start: suffix[left].start,
                end: suffix[cue_start - 1].end,
            };
            let speech_cue = Range {
                start: suffix[cue_start].start,
                end: suffix[suffix.len() - 2].end,
            };
            let end = Range {
                start: terminator.start,
                end: terminator.end,
            };
            if vocabulary
                .subject_surfaces
                .iter()
                .any(|surface| !surface.is_empty() && *surface == lead)
            {
                return Observation::PostposedForm {
                    quote: pending.quote,
                    written_lead: gap,
                    speech_cue,
                    terminator: end,
                    unknowns: unknowns(),
                };
            }
            if vocabulary.title_suffixes.iter().any(|suffix| {
                !suffix.is_empty()
                    && lead
                        .strip_suffix(*suffix)
                        .is_some_and(|prefix| !prefix.is_empty())
            }) {
                return Observation::UnresolvedPostposedForm {
                    quote: pending.quote,
                    gap,
                    speech_cue,
                    terminator: end,
                    unknowns: unknowns(),
                };
            }
            break;
        }
    }
    unknown(pending, terminator.end, PostUnknownReason::NoSupportedTag)
}
