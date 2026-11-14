//! Lossless bounded source chunks over the resumable written-quote reader.
//!
//! Chunks are transport units, not sentences. Quote observations may point
//! into earlier chunks; the caller must retain or independently bind the exact
//! source. This API does not establish identity, story scope or completeness.
//! A damage observation may mark an earlier emitted chunk unexamined; consumers
//! must retain and apply its original `unexamined_from` position.
//! On pause, retry on the same object or call `stop_after_pause`. On a terminal
//! event all decoded, buffered text is emitted first. Preserve every chunk,
//! every observation, and the terminal event before consuming `into_parts`.
use std::io::{self, BufReader, Read};

use crate::resumable_written_quote_reader::{
    End, InitError, Observation, Position, Range, ResumableReader, ResumableStep, Step, Stopped,
    StoppedPendingPostposed, Vocabulary,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChunkCut {
    ScalarLimit,
    End,
    Stopped,
}

#[derive(Debug)]
pub struct TriggeredObservations {
    /// The scalar whose processing produced these observations.
    pub trigger: Range,
    pub observations: Vec<Observation>,
}

#[derive(Debug)]
pub struct Chunk {
    pub text: String,
    pub source: Range,
    pub cut: ChunkCut,
    pub observations: Vec<TriggeredObservations>,
    /// The underlying quote scan stops making observations after damage.
    pub quote_unexamined_from: Option<Position>,
    pub source_identity: &'static str,
    pub sentence_boundary: &'static str,
    pub semantic_coverage: &'static str,
}

#[derive(Debug)]
pub enum ChunkEvent {
    Chunk(Chunk),
    /// `read_until` counts decoded scalars, not BufReader read-ahead.
    /// `raw_prefix` has been consumed and must not be replayed on retry.
    Paused {
        kind: io::ErrorKind,
        read_until: Position,
        delivered_until: Position,
        raw_prefix: Vec<u8>,
    },
    End {
        end: End,
        pending_postposed: Option<StoppedPendingPostposed>,
    },
    Stopped(Stopped),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundedInitError {
    InvalidScalarLimit,
    Vocabulary(InitError),
}

enum Terminal {
    End {
        end: End,
        pending_postposed: Option<StoppedPendingPostposed>,
    },
    Stopped(Stopped),
}

/// A bounded transport adapter. The caller owns all emitted chunks.
pub struct BoundedQuoteChunks<'a, R: Read> {
    reader: ResumableReader<'a, R>,
    max_scalars: usize,
    start: Position,
    end: Position,
    text: String,
    scalar_count: usize,
    observations: Vec<TriggeredObservations>,
    unexamined_from: Option<Position>,
    terminal: Option<Terminal>,
    terminal_delivered: bool,
}

impl<'a, R: Read> BoundedQuoteChunks<'a, R> {
    /// `max_scalars` is a chunk limit, not a process-memory or CPU limit.
    /// Initialization never consumes input. Bytes of each chunk are <= 4*N.
    pub fn new(
        reader: R,
        vocabulary: Vocabulary<'a>,
        max_scalars: usize,
    ) -> Result<Self, (BoundedInitError, R)> {
        if max_scalars == 0 || max_scalars.checked_mul(4).is_none() {
            return Err((BoundedInitError::InvalidScalarLimit, reader));
        }
        let reader = ResumableReader::new(reader, vocabulary)
            .map_err(|(error, reader)| (BoundedInitError::Vocabulary(error), reader))?;
        let zero = Position { byte: 0, scalar: 0 };
        Ok(Self {
            reader,
            max_scalars,
            start: zero,
            end: zero,
            text: String::new(),
            scalar_count: 0,
            observations: Vec::new(),
            unexamined_from: None,
            terminal: None,
            terminal_delivered: false,
        })
    }

    fn take_chunk(&mut self, cut: ChunkCut) -> Chunk {
        let chunk = Chunk {
            text: std::mem::take(&mut self.text),
            source: Range {
                start: self.start,
                end: self.end,
            },
            cut,
            observations: std::mem::take(&mut self.observations),
            quote_unexamined_from: self.unexamined_from,
            source_identity: "unknown",
            sentence_boundary: "unknown",
            semantic_coverage: "unknown",
        };
        self.start = self.end;
        self.scalar_count = 0;
        chunk
    }

    /// Request a stop only after a pause. Buffered chunks still come first.
    pub fn stop_after_pause(&mut self) -> bool {
        if self.terminal.is_some() || self.terminal_delivered {
            return false;
        }
        let Some(stopped) = self.reader.stop_after_pause() else {
            return false;
        };
        self.terminal = Some(Terminal::Stopped(stopped));
        true
    }

    /// After a terminal event, recover failed bytes and the underlying reader.
    /// Early calls fail without consuming state. Already emitted chunks remain
    /// the caller's responsibility. Buffered read-ahead remains in BufReader.
    #[allow(clippy::result_large_err)] // Return the intact reader for retry without allocation.
    pub fn into_parts(self) -> Result<(Vec<u8>, BufReader<R>), Self> {
        if !self.terminal_delivered || !self.text.is_empty() {
            return Err(self);
        }
        Ok(self.reader.into_parts())
    }

    /// One pause per I/O error. A later call retries; this method never loops
    /// on an error. Exactly one terminal event is returned before `None`.
    pub fn next_event(&mut self) -> Option<ChunkEvent> {
        if self.terminal_delivered {
            return None;
        }
        loop {
            if self.terminal.is_some() {
                if !self.text.is_empty() {
                    let cut = if matches!(self.terminal.as_ref(), Some(Terminal::Stopped(_))) {
                        ChunkCut::Stopped
                    } else {
                        ChunkCut::End
                    };
                    return Some(ChunkEvent::Chunk(self.take_chunk(cut)));
                }
                self.terminal_delivered = true;
                return Some(match self.terminal.take().expect("terminal checked") {
                    Terminal::End {
                        end,
                        pending_postposed,
                    } => ChunkEvent::End {
                        end,
                        pending_postposed,
                    },
                    Terminal::Stopped(stopped) => ChunkEvent::Stopped(stopped),
                });
            }
            match self
                .reader
                .next_step()
                .expect("reader terminal is forwarded")
            {
                ResumableStep::Paused {
                    kind,
                    valid_until,
                    raw_prefix,
                } => {
                    return Some(ChunkEvent::Paused {
                        kind,
                        read_until: valid_until,
                        delivered_until: self.start,
                        raw_prefix,
                    });
                }
                ResumableStep::Item(Step::Scalar {
                    ch,
                    range,
                    observations,
                }) => {
                    debug_assert_eq!(range.start, self.end);
                    self.text.push(ch);
                    self.end = range.end;
                    self.scalar_count += 1;
                    for observation in &observations {
                        if let Observation::Damage(damage) = observation {
                            self.unexamined_from = Some(damage.unexamined_from);
                        }
                    }
                    if !observations.is_empty() {
                        self.observations.push(TriggeredObservations {
                            trigger: range,
                            observations,
                        });
                    }
                    if self.scalar_count == self.max_scalars {
                        return Some(ChunkEvent::Chunk(self.take_chunk(ChunkCut::ScalarLimit)));
                    }
                }
                ResumableStep::Item(Step::End(end)) => {
                    let pending_postposed =
                        self.reader
                            .pending_at_termination()
                            .map(|p| StoppedPendingPostposed {
                                quote: p.quote,
                                scanned: p.scanned,
                                reason: p.reason,
                            });
                    self.terminal = Some(Terminal::End {
                        end,
                        pending_postposed,
                    });
                }
            }
        }
    }
}
