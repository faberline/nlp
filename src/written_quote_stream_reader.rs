//! Source-preserving, incremental written-quote observations.
//!
//! The caller receives every decoded scalar and its original byte/scalar range.
//! Quote and postposed-tag records describe the written form only. They do not
//! establish person identity, occurrence, narration scope, or file identity.
//! Preserve all scalar steps and the terminal step; a quote observation alone
//! does not prove complete source coverage. Input is read through one bounded
//! buffer and only a bounded delimiter stack and suffix are retained.
use std::io::{self, BufReader, Read};

pub const MAX_DEPTH: usize = 32;
pub const MAX_POST_SCALARS: usize = 24;
pub const MAX_VOCAB: usize = 512;
pub const MAX_TERM_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Position {
    pub byte: usize,
    pub scalar: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vocabulary<'a> {
    /// Exact written surfaces, not authenticated identities.
    pub subject_surfaces: &'a [&'a str],
    /// A suffix alone is not proof of where a written lead begins.
    pub title_suffixes: &'a [&'a str],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DamageKind {
    MismatchedCloser,
    DepthLimit,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Damage {
    pub kind: DamageKind,
    pub unexamined_from: Position,
    pub detected_at: Position,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PostUnknownReason {
    NoSupportedTag,
    UnexpectedDelimiter,
    LineBoundary,
    SuffixLimit,
    EndOfInput,
    NestedContext,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Unknowns {
    pub identity: &'static str,
    pub occurrence: &'static str,
    pub scope: &'static str,
    pub source_identity: &'static str,
}
fn unknowns() -> Unknowns {
    Unknowns {
        identity: "unknown",
        occurrence: "unknown",
        scope: "unknown",
        source_identity: "unknown",
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Observation {
    /// Matched supported delimiters; nested quote context remains unknown.
    WrittenQuote {
        quote: Range,
        payload: Range,
        depth: usize,
        unknowns: Unknowns,
    },
    /// Exact source spelling + supported adjacent cue; this is not an identity.
    PostposedForm {
        quote: Range,
        written_lead: Range,
        speech_cue: Range,
        terminator: Range,
        unknowns: Unknowns,
    },
    /// A suffix matched, but the lead's left boundary is not established.
    UnresolvedPostposedForm {
        quote: Range,
        gap: Range,
        speech_cue: Range,
        terminator: Range,
        unknowns: Unknowns,
    },
    PostposedUnexamined {
        quote: Range,
        scanned: Range,
        reason: PostUnknownReason,
    },
    Damage(Damage),
}

#[derive(Debug)]
pub enum EndKind {
    Eof,
    Io(io::ErrorKind),
    InvalidUtf8,
    TruncatedUtf8,
    OffsetOverflow,
}
#[derive(Debug)]
pub struct End {
    pub kind: EndKind,
    /// End of all successfully emitted Unicode scalars.
    pub valid_until: Position,
    /// Bytes consumed after valid_until, also returned by into_parts.
    pub raw_prefix: Vec<u8>,
    pub unfinished_from: Option<Position>,
    pub unfinished_depth: usize,
    pub damage: Option<Damage>,
    pub observations: Vec<Observation>,
    pub source_identity: &'static str,
    pub global_scope: &'static str,
    pub coverage: &'static str,
}
#[derive(Debug)]
pub enum Step {
    /// `ch` re-encodes to exactly the UTF-8 bytes at `range`.
    Scalar {
        ch: char,
        range: Range,
        observations: Vec<Observation>,
    },
    End(End),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitError {
    VocabularyTooLarge,
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

/// `Read` may read ahead into the internal BufReader. On an input error, call
/// `into_parts` and replay the returned prefix, followed by that BufReader.
/// Previously emitted scalar steps remain the caller's responsibility.
pub struct Reader<'a, R: Read> {
    inner: BufReader<R>,
    vocabulary: Vocabulary<'a>,
    position: Position,
    stack: Vec<Open>,
    pending: Option<Pending>,
    damage: Option<Damage>,
    raw_prefix: Vec<u8>,
    ended: bool,
}
impl<'a, R: Read> Reader<'a, R> {
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
            ended: false,
        })
    }

    /// Returns buffered unread bytes intact, plus any failed partial scalar.
    pub fn into_parts(self) -> (Vec<u8>, BufReader<R>) {
        (self.raw_prefix, self.inner)
    }

    fn read_byte(&mut self) -> Result<Option<u8>, io::ErrorKind> {
        let mut byte = [0_u8; 1];
        let mut interruptions = 0;
        loop {
            match self.inner.read(&mut byte) {
                Ok(0) => return Ok(None),
                Ok(_) => return Ok(Some(byte[0])),
                Err(e) if e.kind() == io::ErrorKind::Interrupted && interruptions < 16 => {
                    interruptions += 1
                }
                Err(e) => return Err(e.kind()),
            }
        }
    }

    fn finish(&mut self, kind: EndKind) -> Step {
        self.ended = true;
        let mut observations = Vec::new();
        if let Some(pending) = self.pending.take() {
            observations.push(unknown(
                pending,
                self.position,
                PostUnknownReason::EndOfInput,
            ));
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

    /// Yields exactly one terminal step, followed by `None` on later calls.
    pub fn next_step(&mut self) -> Option<Step> {
        if self.ended {
            return None;
        }
        let first = match self.read_byte() {
            Ok(Some(b)) => b,
            Ok(None) => return Some(self.finish(EndKind::Eof)),
            Err(kind) => return Some(self.finish(EndKind::Io(kind))),
        };
        self.raw_prefix.push(first);
        let Some(need) = width(first) else {
            return Some(self.finish(EndKind::InvalidUtf8));
        };
        while self.raw_prefix.len() < need {
            match self.read_byte() {
                Ok(Some(b)) => self.raw_prefix.push(b),
                Ok(None) => return Some(self.finish(EndKind::TruncatedUtf8)),
                Err(kind) => return Some(self.finish(EndKind::Io(kind))),
            }
        }
        let ch = match std::str::from_utf8(&self.raw_prefix)
            .ok()
            .and_then(|s| s.chars().next())
        {
            Some(ch) => ch,
            None => return Some(self.finish(EndKind::InvalidUtf8)),
        };
        let at = self.position;
        let Some(byte) = at.byte.checked_add(need) else {
            return Some(self.finish(EndKind::OffsetOverflow));
        };
        let Some(scalar) = at.scalar.checked_add(1) else {
            return Some(self.finish(EndKind::OffsetOverflow));
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
        if self.damage.is_none() {
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
        Some(Step::Scalar {
            ch,
            range: Range {
                start: at,
                end: self.position,
            },
            observations,
        })
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
