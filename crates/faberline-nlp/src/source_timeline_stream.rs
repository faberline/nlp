//! Incremental source-mention timeline hints, retaining at most two chapters
//! during a push and the latest chapter afterward. This is not story ordering,
//! event identity, occurrence verification, or a general streaming parser.
//!
//! The caller owns the source binding: chapter ordinals and spans do not by
//! themselves authenticate a file or digest. The caller must emit or persist
//! each returned chunk and bind each source to its exact bytes.

use crate::{
    Event,
    timeline::{self, LocatedSpan, TimelineDocument},
};
use anyhow::{Result, ensure};

pub struct Chapter {
    pub order: u32,
    pub text: String,
    pub events: Vec<Event>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MentionRef {
    pub chapter: u32,
    pub event_index: usize,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Mention {
    pub reference: MentionRef,
    pub evidence: Vec<LocatedSpan>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct RelationHint {
    pub before: MentionRef,
    pub after: MentionRef,
    pub evidence: [LocatedSpan; 3],
}

/// All hints are unverified. No global cycle check or story ordering is made.
#[derive(Debug, Eq, PartialEq)]
pub struct Chunk {
    pub chapter: u32,
    pub mentions: Vec<Mention>,
    pub relations: Vec<RelationHint>,
}

#[derive(Default)]
pub struct SourceTimelineStream {
    previous: Option<Chapter>,
}

impl SourceTimelineStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// The current chapter is retained so a consumer can emit source-backed
    /// records before the next push. It is replaced only after a valid push.
    pub fn current(&self) -> Option<&Chapter> {
        self.previous.as_ref()
    }

    /// Chapters must have strictly increasing positive ordinals. On failure,
    /// prior state is unchanged. Each hint involving the new chapter is emitted
    /// once; `supported` is deliberately omitted because pairwise cycle checks
    /// are not the global story graph's cycle checks.
    pub fn push(&mut self, chapter: Chapter) -> Result<Chunk> {
        ensure!(chapter.order > 0, "chapter ordinal must be positive");
        if let Some(previous) = &self.previous {
            ensure!(
                chapter.order > previous.order,
                "chapter ordinals must increase"
            );
        }
        let mut documents = Vec::with_capacity(2);
        if let Some(previous) = &self.previous {
            documents.push(TimelineDocument {
                order: previous.order,
                text: &previous.text,
                events: &previous.events,
            });
        }
        documents.push(TimelineDocument {
            order: chapter.order,
            text: &chapter.text,
            events: &chapter.events,
        });
        let timeline = timeline::build_source_mentions_with_indefinite_delays(&documents)?;
        let reference = |index: usize| MentionRef {
            chapter: timeline.events[index].chapter,
            event_index: timeline.events[index].index,
        };
        let mentions = timeline
            .events
            .iter()
            .filter(|event| event.chapter == chapter.order)
            .map(|event| Mention {
                reference: MentionRef {
                    chapter: event.chapter,
                    event_index: event.index,
                },
                evidence: event.evidence.clone(),
            })
            .collect();
        let relations = timeline
            .relations
            .iter()
            .filter_map(|relation| {
                let before = reference(relation.before);
                let after = reference(relation.after);
                (before.chapter == chapter.order || after.chapter == chapter.order).then_some(
                    RelationHint {
                        before,
                        after,
                        evidence: relation.evidence,
                    },
                )
            })
            .collect();
        let chunk = Chunk {
            chapter: chapter.order,
            mentions,
            relations,
        };
        self.previous = Some(chapter);
        Ok(chunk)
    }
}
