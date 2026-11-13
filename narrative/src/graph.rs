//! Narrative timeline graph construction, discourse hygiene, deterministic cycle breaking,
//! and cycle-free condensed partial-order querying over `nlp.doc.v1`.

use crate::schema::{
    CandidateEvent, ChapterNarrative, ChronologyOrder, ContextZone, EdgeEvidenceKind, EdgeKind,
    EdgeStatus, Factuality, LocatedSpan, NarrativeUnknown, NarrativeUnknownKind, TimelineEdge,
};
use faberline_nlp::Span;
use faberline_nlp::doc::{Document, EntityKind, Relation, RelationKind, Upos};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

/// Chapter-scoped event reference for cross-chapter timeline queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BookEventRef {
    pub chapter: u32,
    pub event_index: usize,
}

/// Condensed cycle-free partial-order DAG for a single chapter's `Narrative + Actual` events.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChapterTimelineDag {
    num_events: usize,
    is_chronological: Vec<bool>,
    /// Maps each event index `0..num_events` to its condensed equivalence component index.
    component_of: Vec<usize>,
    num_components: usize,
    /// Condensed strict reachability: `reach[comp_a]` has bit `comp_b` set iff `comp_a <* comp_b`.
    reach: BitMatrix,
}

impl ChapterTimelineDag {
    /// Queries the partial-order chronological relation between `event_a` and `event_b`.
    ///
    /// Returns [`ChronologyOrder::Incomparable`] if either index is out of bounds, non-`Narrative`,
    /// non-`Actual`, or has no directed/simultaneous path connecting them.
    pub fn query_order(&self, event_a: usize, event_b: usize) -> ChronologyOrder {
        if event_a >= self.num_events || event_b >= self.num_events {
            return ChronologyOrder::Incomparable;
        }
        if !self.is_chronological[event_a] || !self.is_chronological[event_b] {
            return ChronologyOrder::Incomparable;
        }
        if event_a == event_b {
            return ChronologyOrder::Simultaneous;
        }
        let ca = self.component_of[event_a];
        let cb = self.component_of[event_b];
        if ca == cb {
            ChronologyOrder::Simultaneous
        } else if self.reach.has(ca, cb) {
            ChronologyOrder::Before
        } else if self.reach.has(cb, ca) {
            ChronologyOrder::After
        } else {
            ChronologyOrder::Incomparable
        }
    }

    /// Returns `true` if the condensed chronological graph contains any directed cycle.
    ///
    /// By construction during condensation and deterministic cycle breaking, this is always `false`.
    pub fn has_cycle(&self) -> bool {
        (0..self.num_components).any(|c| self.reach.has(c, c))
    }

    /// Returns the number of events covered by this DAG.
    pub fn len(&self) -> usize {
        self.num_events
    }

    /// Returns `true` when there are zero events in the chapter.
    pub fn is_empty(&self) -> bool {
        self.num_events == 0
    }
}

/// Output of building a chapter's narrative timeline edges, DAG layers, and conflict diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChapterGraphBuild {
    pub edges: Vec<TimelineEdge>,
    pub event_dag_layers: Vec<Option<u32>>,
    pub conflict_cycles_broken: usize,
    pub cycle_unknowns: Vec<NarrativeUnknown>,
    pub dag: ChapterTimelineDag,
}

/// Cross-chapter timeline graph combining intra-chapter condensed DAGs with consecutive-chapter
/// forward narrative boundaries when chapters advance without flashback cues.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BookTimelineGraph {
    chapters: Vec<BookChapterEntry>,
    chapter_index_by_num: HashMap<u32, usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BookChapterEntry {
    chapter: u32,
    dag: ChapterTimelineDag,
    event_is_chronological: Vec<bool>,
    event_is_flashback: Vec<bool>,
    /// True iff this chapter directly advances forward from the immediately preceding chapter
    /// (`chapter == prev.chapter + 1`) without an opening flashback cue.
    advances_from_prev: bool,
}

impl BookTimelineGraph {
    /// Builds a cross-chapter timeline graph from a slice of [`ChapterNarrative`] chapters.
    pub fn from_chapters(chapters: &[ChapterNarrative]) -> Self {
        let mut sorted: Vec<&ChapterNarrative> = chapters.iter().collect();
        sorted.sort_by_key(|ch| ch.chapter);

        let mut entries = Vec::with_capacity(sorted.len());
        let mut chapter_index_by_num = HashMap::with_capacity(sorted.len());

        for (idx, ch) in sorted.iter().enumerate() {
            let dag = build_chapter_dag_from_narrative(ch);
            let event_is_chronological: Vec<bool> =
                ch.events.iter().map(is_chronological_event).collect();
            let event_is_flashback: Vec<bool> = ch
                .events
                .iter()
                .map(event_has_retrospective_anchor_metadata)
                .collect();

            let advances_from_prev = if idx == 0 {
                false
            } else {
                let prev = sorted[idx - 1];
                prev.chapter.checked_add(1) == Some(ch.chapter)
                    && !chapter_opens_with_flashback(ch)
                    && prev.events.iter().any(is_chronological_event)
                    && ch.events.iter().any(is_chronological_event)
            };

            chapter_index_by_num.insert(ch.chapter, idx);
            entries.push(BookChapterEntry {
                chapter: ch.chapter,
                dag,
                event_is_chronological,
                event_is_flashback,
                advances_from_prev,
            });
        }

        Self {
            chapters: entries,
            chapter_index_by_num,
        }
    }

    /// Queries the partial-order chronological relationship between two chapter-qualified events.
    pub fn query_order(&self, a: BookEventRef, b: BookEventRef) -> ChronologyOrder {
        let Some(&idx_a) = self.chapter_index_by_num.get(&a.chapter) else {
            return ChronologyOrder::Incomparable;
        };
        let Some(&idx_b) = self.chapter_index_by_num.get(&b.chapter) else {
            return ChronologyOrder::Incomparable;
        };

        if idx_a == idx_b {
            return self.chapters[idx_a]
                .dag
                .query_order(a.event_index, b.event_index);
        }

        let entry_a = &self.chapters[idx_a];
        let entry_b = &self.chapters[idx_b];

        if !entry_a
            .event_is_chronological
            .get(a.event_index)
            .copied()
            .unwrap_or(false)
            || !entry_b
                .event_is_chronological
                .get(b.event_index)
                .copied()
                .unwrap_or(false)
        {
            return ChronologyOrder::Incomparable;
        }

        if entry_a.event_is_flashback[a.event_index] || entry_b.event_is_flashback[b.event_index] {
            return ChronologyOrder::Incomparable;
        }

        if idx_a < idx_b {
            let all_forward = ((idx_a + 1)..=idx_b).all(|k| self.chapters[k].advances_from_prev);
            if all_forward {
                ChronologyOrder::Before
            } else {
                ChronologyOrder::Incomparable
            }
        } else {
            let all_forward = ((idx_b + 1)..=idx_a).all(|k| self.chapters[k].advances_from_prev);
            if all_forward {
                ChronologyOrder::After
            } else {
                ChronologyOrder::Incomparable
            }
        }
    }
}

/// Builds the full timeline edge set, resolves cycle conflicts deterministically, computes
/// condensed topological layers, and returns the cycle-free [`ChapterTimelineDag`].
pub fn build_chapter_graph(
    chapter: u32,
    text: &str,
    doc: &Document,
    events: &[CandidateEvent],
) -> ChapterGraphBuild {
    let chars: Vec<char> = text.chars().collect();
    let mut raw_edges: Vec<RawEdge> = Vec::new();
    let mut covered_pairs: HashSet<(usize, usize)> = HashSet::new();

    // Tier 1 & Tier 2: validated `doc.relations` with Carbon discourse-hygiene guardrails.
    for rel in &doc.relations {
        if let Some(edge) = build_edge_from_relation(chapter, &chars, doc, events, rel) {
            record_pair(&mut covered_pairs, edge.from_event, edge.to_event);
            raw_edges.push(edge);
        }
    }

    // Tier 1 & Tier 2 supplementary explicit connectors (`之前`, `之所以…是因為`, `同時`, `一邊…一邊…`, `meanwhile`).
    collect_explicit_connector_edges(
        chapter,
        &chars,
        doc,
        events,
        &mut covered_pairs,
        &mut raw_edges,
    );

    // Precompute flashback status per event once for Tier 3 and Tier 4 guards.
    let event_flashback: Vec<bool> = events
        .iter()
        .enumerate()
        .map(|(idx, ev)| event_has_flashback(&chars, doc, events, idx, ev))
        .collect();

    // Tier 3: TemporalAnchorShift between adjacent `Narrative + Actual` events.
    collect_temporal_anchor_shift_edges(
        chapter,
        &chars,
        doc,
        events,
        &event_flashback,
        &mut covered_pairs,
        &mut raw_edges,
    );

    // Tier 4: IntraSentenceClauseFlow between consecutive `Narrative + Actual` clauses in a sentence.
    collect_intra_sentence_flow_edges(
        chapter,
        events,
        &event_flashback,
        &mut covered_pairs,
        &mut raw_edges,
    );

    // Materialize `TimelineEdge`s with strict factuality & quote gating and deterministic IDs.
    let mut edges: Vec<TimelineEdge> = raw_edges
        .into_iter()
        .enumerate()
        .map(|(idx, raw)| {
            let both_chronological = is_chronological_event(&events[raw.from_event])
                && is_chronological_event(&events[raw.to_event]);
            let status = if both_chronological {
                EdgeStatus::Supported
            } else {
                EdgeStatus::NonChronologicalFactuality
            };
            TimelineEdge {
                edge_id: format!("c{chapter:04}-g{idx:04}"),
                kind: raw.kind,
                evidence_kind: raw.evidence_kind,
                status,
                from_event: raw.from_event,
                to_event: raw.to_event,
                connector_span: raw.connector_span,
                evidence_spans: raw.evidence_spans,
                confidence_tier: raw.confidence_tier,
            }
        })
        .collect();

    let (dag, event_dag_layers, conflict_cycles_broken, cycle_unknowns) =
        condense_and_break_cycles(chapter, events, &mut edges);

    ChapterGraphBuild {
        edges,
        event_dag_layers,
        conflict_cycles_broken,
        cycle_unknowns,
        dag,
    }
}

/// Reconstructs a cycle-free [`ChapterTimelineDag`] from an existing [`ChapterNarrative`].
pub fn build_chapter_dag_from_narrative(chapter: &ChapterNarrative) -> ChapterTimelineDag {
    let mut edges_copy = chapter.edges.clone();
    let (dag, _, _, _) =
        condense_and_break_cycles(chapter.chapter, &chapter.events, &mut edges_copy);
    dag
}

struct RawEdge {
    kind: EdgeKind,
    evidence_kind: EdgeEvidenceKind,
    from_event: usize,
    to_event: usize,
    connector_span: Option<LocatedSpan>,
    evidence_spans: Vec<LocatedSpan>,
    confidence_tier: u8,
}

fn record_pair(covered: &mut HashSet<(usize, usize)>, a: usize, b: usize) {
    covered.insert((a, b));
    covered.insert((b, a));
}

fn is_chronological_event(ev: &CandidateEvent) -> bool {
    ev.zone == ContextZone::Narrative && ev.factuality == Factuality::Actual
}

// ---------------------------------------------------------------------------
// Discourse Hygiene & Tier 1/2 Extraction
// ---------------------------------------------------------------------------

const DETERMINER_FOLLOWERS: &[char] = &[
    '一', '兩', '两', '二', '三', '四', '五', '六', '七', '八', '九', '十', '半', '幾', '几', '數',
    '数', '個', '个', '名', '位', '種', '种', '批', '處', '处', '座', '條', '条', '件',
];

const ADVERB_PROXY_EXCLUDED_PREDS: &[&str] = &[
    "幾乎", "紛紛", "接著", "隨即", "隨後", "同時", "還", "才", "就", "都", "也", "只", "更", "卻",
    "却", "居然", "竟然", "果然", "頓時", "立刻",
];

fn build_edge_from_relation(
    chapter: u32,
    chars: &[char],
    doc: &Document,
    events: &[CandidateEvent],
    rel: &Relation,
) -> Option<RawEdge> {
    if !valid_span(chars, rel.arg1)
        || !valid_span(chars, rel.arg2)
        || !valid_span(chars, rel.connector)
        || spans_overlap(rel.arg1, rel.arg2)
    {
        return None;
    }
    if let Some(cor) = rel.correlate
        && !valid_span(chars, cor)
    {
        return None;
    }

    // Reject degenerate / punctuation-only spans after stripping connector & correlate.
    if !has_lexical_content_excluding(chars, rel.arg1, rel.connector, rel.correlate)
        || !has_lexical_content_excluding(chars, rel.arg2, rel.connector, rel.correlate)
    {
        return None;
    }

    let conn_text = span_text(chars, rel.connector);

    // Carbon Layer-2 false-positive filters on `Parallel`.
    if rel.kind == RelationKind::Parallel {
        if matches!(conn_text.as_str(), "同樣" | "同样") {
            let arg1_has_event = events
                .iter()
                .any(|ev| span_contains(rel.arg1, ev.predicate_span));
            let arg1_has_pred_tok = arg_has_non_adverb_predicate_token(chars, doc, rel.arg1);
            if !arg1_has_event || !arg1_has_pred_tok {
                return None;
            }
        }
        if conn_text == "另外" && is_adnominal_lingwai(chars, rel.connector) {
            return None;
        }
    }

    let (kind, evidence_kind, confidence_tier, from_span, to_span) = match rel.kind {
        RelationKind::Sequence => {
            let reversed = is_reversed_sequence_before(chars, &conn_text, rel.arg1);
            let (from_s, to_s) = if reversed {
                (rel.arg2, rel.arg1)
            } else {
                (rel.arg1, rel.arg2)
            };
            (
                EdgeKind::Before,
                EdgeEvidenceKind::ExplicitDiscourseRelation,
                1,
                from_s,
                to_s,
            )
        }
        RelationKind::Parallel => (
            EdgeKind::Simultaneous,
            EdgeEvidenceKind::ExplicitDiscourseRelation,
            1,
            rel.arg1,
            rel.arg2,
        ),
        RelationKind::Contrast => (
            EdgeKind::Contrast,
            EdgeEvidenceKind::ExplicitDiscourseRelation,
            1,
            rel.arg1,
            rel.arg2,
        ),
        RelationKind::Condition => (
            EdgeKind::Condition,
            EdgeEvidenceKind::ExplicitDiscourseRelation,
            1,
            rel.arg1,
            rel.arg2,
        ),
        RelationKind::Cause => {
            let (cause_span, effect_span) =
                orient_causal_spans(chars, &conn_text, rel.arg1, rel.arg2);
            (
                EdgeKind::Cause,
                EdgeEvidenceKind::ExplicitDiscourseRelation,
                2,
                cause_span,
                effect_span,
            )
        }
    };

    let prefer_last_from = matches!(kind, EdgeKind::Before | EdgeKind::Cause);
    let from_event = pick_event_in_span(events, from_span, prefer_last_from)?;
    let to_event = pick_event_in_span(events, to_span, false)?;
    if from_event == to_event {
        return None;
    }

    Some(RawEdge {
        kind,
        evidence_kind,
        from_event,
        to_event,
        connector_span: Some(LocatedSpan {
            chapter,
            span: rel.connector,
        }),
        evidence_spans: vec![
            LocatedSpan {
                chapter,
                span: events[from_event].predicate_span,
            },
            LocatedSpan {
                chapter,
                span: events[to_event].predicate_span,
            },
        ],
        confidence_tier,
    })
}

fn is_adnominal_lingwai(chars: &[char], connector: Span) -> bool {
    let mut i = connector.end_char;
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    chars
        .get(i)
        .is_some_and(|ch| DETERMINER_FOLLOWERS.contains(ch) || ch.is_ascii_digit())
}

fn arg_has_non_adverb_predicate_token(chars: &[char], doc: &Document, arg: Span) -> bool {
    if doc.tokens.is_empty() {
        return true;
    }
    doc.tokens.iter().enumerate().any(|(idx, tok)| {
        span_contains(arg, tok.span)
            && matches!(tok.upos, Some(Upos::Verb | Upos::Adj))
            && !ADVERB_PROXY_EXCLUDED_PREDS.contains(&span_text(chars, tok.span).as_str())
            && !doc
                .syntax
                .iter()
                .find(|syn| syn.token == idx)
                .is_some_and(|syn| syn.deprel == "acl" && syn.head != Some(idx))
    })
}

fn is_reversed_sequence_before(chars: &[char], conn_text: &str, arg1: Span) -> bool {
    let arg1_trimmed = trim_span_text(chars, arg1);
    if matches!(conn_text, "之前") || conn_text.eq_ignore_ascii_case("before") {
        return arg1_trimmed.ends_with("之前") || starts_with_ci(&arg1_trimmed, "before ");
    }
    arg1_trimmed.ends_with("之前")
        || starts_with_ci(&arg1_trimmed, "before ")
        || starts_with_ci(&arg1_trimmed, "prior to ")
}

fn orient_causal_spans(chars: &[char], conn_text: &str, arg1: Span, arg2: Span) -> (Span, Span) {
    let a1 = trim_span_text(chars, arg1);
    let a2 = trim_span_text(chars, arg2);
    let a1_is_forward_cause =
        starts_with_any(&a1, &["因為", "因为", "由於", "由于"]) || starts_with_ci(&a1, "because");
    let a2_is_reason_intro = starts_with_any(
        &a2,
        &[
            "是因為",
            "是因为",
            "就是因為",
            "就是因为",
            "正因為",
            "正因为",
            "皆因",
            "乃因",
            "只因",
            "全因",
            "因為",
            "因为",
            "由於",
            "由于",
        ],
    ) || starts_with_ci(&a2, "because");
    let a1_is_result_header = a1.starts_with("之所以");

    if ((!a1_is_forward_cause || a1_is_result_header) && a2_is_reason_intro)
        || (matches!(conn_text, "是因為" | "是因为") && !a1_is_forward_cause)
    {
        (arg2, arg1)
    } else {
        (arg1, arg2)
    }
}

fn collect_explicit_connector_edges(
    chapter: u32,
    chars: &[char],
    doc: &Document,
    events: &[CandidateEvent],
    covered_pairs: &mut HashSet<(usize, usize)>,
    raw_edges: &mut Vec<RawEdge>,
) {
    // Group event indices by sentence and paragraph for fast local adjacent-clause checks.
    for i in 0..events.len() {
        for j in (i + 1)..events.len() {
            let e1 = &events[i];
            let e2 = &events[j];
            if e1.sentence != e2.sentence {
                continue;
            }
            if e1.clause >= e2.clause {
                continue;
            }
            if covered_pairs.contains(&(i, j)) {
                continue;
            }
            // Only examine immediate sequential clauses (or adjacent clauses separated by non-event).
            let has_intervening = ((i + 1)..j).any(|k| events[k].sentence == e1.sentence);
            if has_intervening {
                continue;
            }

            let c1 = e1.clause_span;
            let c2 = e2.clause_span;
            if !valid_span(chars, c1) || !valid_span(chars, c2) {
                continue;
            }

            // 1. Backward causal connector (`之所以 B，是因為 A` / `B，是因為 A` / `B，只因 A`).
            if let Some(conn_span) = detect_backward_causal_connector(chars, doc, c1, c2) {
                record_pair(covered_pairs, i, j);
                raw_edges.push(RawEdge {
                    kind: EdgeKind::Cause,
                    evidence_kind: EdgeEvidenceKind::CausalConnector,
                    from_event: j,
                    to_event: i,
                    connector_span: Some(LocatedSpan {
                        chapter,
                        span: conn_span,
                    }),
                    evidence_spans: vec![
                        LocatedSpan {
                            chapter,
                            span: e2.predicate_span,
                        },
                        LocatedSpan {
                            chapter,
                            span: e1.predicate_span,
                        },
                    ],
                    confidence_tier: 2,
                });
                continue;
            }

            // 2. Forward causal connector missed by layer 7 (`由於 A，導致/致使/因而 B`).
            if let Some(conn_span) = detect_forward_causal_connector(chars, doc, c1, c2) {
                record_pair(covered_pairs, i, j);
                raw_edges.push(RawEdge {
                    kind: EdgeKind::Cause,
                    evidence_kind: EdgeEvidenceKind::CausalConnector,
                    from_event: i,
                    to_event: j,
                    connector_span: Some(LocatedSpan {
                        chapter,
                        span: conn_span,
                    }),
                    evidence_spans: vec![
                        LocatedSpan {
                            chapter,
                            span: e1.predicate_span,
                        },
                        LocatedSpan {
                            chapter,
                            span: e2.predicate_span,
                        },
                    ],
                    confidence_tier: 2,
                });
                continue;
            }

            // 3. `在 B 之前，A` / `Before B, A` inverted sequence frame.
            if let Some(conn_span) = detect_pre_event_before_frame(chars, doc, c1, c2) {
                record_pair(covered_pairs, i, j);
                raw_edges.push(RawEdge {
                    kind: EdgeKind::Before,
                    evidence_kind: EdgeEvidenceKind::ExplicitDiscourseRelation,
                    from_event: j,
                    to_event: i,
                    connector_span: Some(LocatedSpan {
                        chapter,
                        span: conn_span,
                    }),
                    evidence_spans: vec![
                        LocatedSpan {
                            chapter,
                            span: e2.predicate_span,
                        },
                        LocatedSpan {
                            chapter,
                            span: e1.predicate_span,
                        },
                    ],
                    confidence_tier: 1,
                });
                continue;
            }

            // 4. Simultaneous connectors (`同時`, `一邊…一邊…`, `meanwhile`, `at the same time`).
            if let Some(conn_span) = detect_simultaneous_connector(chars, doc, c1, c2) {
                record_pair(covered_pairs, i, j);
                raw_edges.push(RawEdge {
                    kind: EdgeKind::Simultaneous,
                    evidence_kind: EdgeEvidenceKind::ExplicitDiscourseRelation,
                    from_event: i,
                    to_event: j,
                    connector_span: Some(LocatedSpan {
                        chapter,
                        span: conn_span,
                    }),
                    evidence_spans: vec![
                        LocatedSpan {
                            chapter,
                            span: e1.predicate_span,
                        },
                        LocatedSpan {
                            chapter,
                            span: e2.predicate_span,
                        },
                    ],
                    confidence_tier: 1,
                });
            }
        }
    }

    // Also scan consecutive sentences in the same paragraph for explicit `同時` / `meanwhile` /
    // `之所以…是因為` when missed by Layer-7 `doc.relations`.
    let mut sentence_events: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (idx, ev) in events.iter().enumerate() {
        sentence_events.entry(ev.sentence).or_default().push(idx);
    }
    let sent_ids: Vec<usize> = sentence_events.keys().copied().collect();
    for pair in sent_ids.windows(2) {
        let s1 = pair[0];
        let s2 = pair[1];
        if s2 != s1 + 1 {
            continue;
        }
        let Some(sent1) = doc.segmentation.sentences.get(s1) else {
            continue;
        };
        let Some(sent2) = doc.segmentation.sentences.get(s2) else {
            continue;
        };
        if sent1.paragraph != sent2.paragraph {
            continue;
        }
        let evs1 = &sentence_events[&s1];
        let evs2 = &sentence_events[&s2];
        let Some(&last_1) = evs1.last() else {
            continue;
        };
        let Some(&first_2) = evs2.first() else {
            continue;
        };
        if covered_pairs.contains(&(last_1, first_2)) {
            continue;
        }
        if let Some(conn_span) = detect_simultaneous_connector(chars, doc, sent1.span, sent2.span) {
            record_pair(covered_pairs, last_1, first_2);
            raw_edges.push(RawEdge {
                kind: EdgeKind::Simultaneous,
                evidence_kind: EdgeEvidenceKind::ExplicitDiscourseRelation,
                from_event: last_1,
                to_event: first_2,
                connector_span: Some(LocatedSpan {
                    chapter,
                    span: conn_span,
                }),
                evidence_spans: vec![
                    LocatedSpan {
                        chapter,
                        span: events[last_1].predicate_span,
                    },
                    LocatedSpan {
                        chapter,
                        span: events[first_2].predicate_span,
                    },
                ],
                confidence_tier: 1,
            });
        } else if let Some(conn_span) =
            detect_backward_causal_connector(chars, doc, sent1.span, sent2.span)
        {
            record_pair(covered_pairs, last_1, first_2);
            raw_edges.push(RawEdge {
                kind: EdgeKind::Cause,
                evidence_kind: EdgeEvidenceKind::CausalConnector,
                from_event: first_2,
                to_event: last_1,
                connector_span: Some(LocatedSpan {
                    chapter,
                    span: conn_span,
                }),
                evidence_spans: vec![
                    LocatedSpan {
                        chapter,
                        span: events[first_2].predicate_span,
                    },
                    LocatedSpan {
                        chapter,
                        span: events[last_1].predicate_span,
                    },
                ],
                confidence_tier: 2,
            });
        }
    }
}

fn detect_backward_causal_connector(
    chars: &[char],
    doc: &Document,
    effect_span: Span,
    cause_span: Span,
) -> Option<Span> {
    let eff_start = skip_opening_chars(chars, doc, effect_span);
    if starts_with_chars(chars, eff_start, effect_span.end_char, "因為").is_some()
        || starts_with_chars(chars, eff_start, effect_span.end_char, "因为").is_some()
        || starts_with_chars(chars, eff_start, effect_span.end_char, "由於").is_some()
        || starts_with_chars(chars, eff_start, effect_span.end_char, "由于").is_some()
        || starts_with_ascii_word_ci(chars, eff_start, effect_span.end_char, "because").is_some()
    {
        return None;
    }

    let c_start = skip_opening_chars(chars, doc, cause_span);
    const BACKWARD_PREFIXES: &[&str] = &[
        "就是因為",
        "就是因为",
        "正因為",
        "正因为",
        "是因為",
        "是因为",
        "皆因",
        "乃因",
        "全因",
        "只因",
        "緣於",
        "缘于",
        "因為",
        "因为",
        "由於",
        "由于",
    ];
    for pat in BACKWARD_PREFIXES {
        if let Some(end) = starts_with_chars(chars, c_start, cause_span.end_char, pat) {
            return Some(Span {
                start_char: c_start,
                end_char: end,
            });
        }
    }
    if let Some(end) = starts_with_ascii_word_ci(chars, c_start, cause_span.end_char, "because") {
        return Some(Span {
            start_char: c_start,
            end_char: end,
        });
    }
    None
}

fn detect_forward_causal_connector(
    chars: &[char],
    doc: &Document,
    cause_span: Span,
    effect_span: Span,
) -> Option<Span> {
    let c1_start = skip_opening_chars(chars, doc, cause_span);
    let has_cause_opener = starts_with_chars(chars, c1_start, cause_span.end_char, "由於")
        .is_some()
        || starts_with_chars(chars, c1_start, cause_span.end_char, "由于").is_some();

    let c2_start = skip_opening_chars(chars, doc, effect_span);
    const RESULT_PREFIXES: &[&str] = &[
        "導致", "导致", "致使", "以致", "故而", "因而", "於是", "于是",
    ];
    for pat in RESULT_PREFIXES {
        if let Some(end) = starts_with_chars(chars, c2_start, effect_span.end_char, pat)
            && (has_cause_opener
                || matches!(*pat, "導致" | "导致" | "致使" | "以致" | "故而" | "因而"))
        {
            return Some(Span {
                start_char: c2_start,
                end_char: end,
            });
        }
    }
    None
}

fn detect_pre_event_before_frame(
    chars: &[char],
    doc: &Document,
    c1: Span,
    _c2: Span,
) -> Option<Span> {
    let start = skip_opening_chars(chars, doc, c1);
    let end = trim_closing_chars(chars, start, c1.end_char);
    if end <= start {
        return None;
    }
    if ends_with_chars(chars, start, end, "之前") {
        // Guard against pure retrospective durations like `數日之前` / `很久之前`.
        let inner = trim_span_text(
            chars,
            Span {
                start_char: start,
                end_char: end - 2,
            },
        );
        if !inner.is_empty() && !is_retrospective_cue_text(&format!("{inner}之前")) {
            return Some(Span {
                start_char: end - 2,
                end_char: end,
            });
        }
    }
    if let Some(w_end) = starts_with_ascii_word_ci(chars, start, end, "before") {
        return Some(Span {
            start_char: start,
            end_char: w_end,
        });
    }
    None
}

fn detect_simultaneous_connector(
    chars: &[char],
    doc: &Document,
    c1: Span,
    c2: Span,
) -> Option<Span> {
    let c1_start = skip_opening_chars(chars, doc, c1);
    let c2_start = skip_opening_chars(chars, doc, c2);

    // `一邊…一邊…` / `一面…一面…`
    let c1_text = span_text(chars, c1);
    for pat in ["一邊", "一边", "一面"] {
        if c1_text.contains(pat)
            && let Some(end) = starts_with_chars(chars, c2_start, c2.end_char, pat)
        {
            return Some(Span {
                start_char: c2_start,
                end_char: end,
            });
        }
    }

    // Clause/sentence-initial `同時` / `同时` (excluding `同時期`, `同時代`, `同事`, `同伴`).
    for pat in ["同時", "同时"] {
        if let Some(end) = starts_with_chars(chars, c2_start, c2.end_char, pat) {
            let next_ch = chars.get(end).copied();
            if !matches!(next_ch, Some('期' | '代' | '事' | '伴' | '學' | '学'))
                && c1_start < c1.end_char
            {
                return Some(Span {
                    start_char: c2_start,
                    end_char: end,
                });
            }
        }
    }

    // English `meanwhile` / `at the same time`.
    if let Some(end) = starts_with_ascii_word_ci(chars, c2_start, c2.end_char, "meanwhile") {
        return Some(Span {
            start_char: c2_start,
            end_char: end,
        });
    }
    if let Some(end) = starts_with_ascii_phrase_ci(chars, c2_start, c2.end_char, "at the same time")
    {
        return Some(Span {
            start_char: c2_start,
            end_char: end,
        });
    }

    None
}

// ---------------------------------------------------------------------------
// Tier 3 TemporalAnchorShift & Flashback Guard
// ---------------------------------------------------------------------------

fn collect_temporal_anchor_shift_edges(
    chapter: u32,
    chars: &[char],
    doc: &Document,
    events: &[CandidateEvent],
    event_flashback: &[bool],
    covered_pairs: &mut HashSet<(usize, usize)>,
    raw_edges: &mut Vec<RawEdge>,
) {
    let mut prev_narrative_idx: Option<usize> = None;

    for (idx, ev) in events.iter().enumerate() {
        if !is_chronological_event(ev) {
            continue;
        }
        let Some(prev_idx) = prev_narrative_idx else {
            prev_narrative_idx = Some(idx);
            continue;
        };

        let prev_ev = &events[prev_idx];
        // Require same paragraph or consecutive narrative progress (all intervening events
        // between `prev_ev.paragraph` and `ev.paragraph` had no `Narrative + Actual` events).
        let adjacent_scope = ev.paragraph >= prev_ev.paragraph;
        if adjacent_scope
            && !covered_pairs.contains(&(prev_idx, idx))
            && !event_flashback[idx]
            && !event_flashback[prev_idx]
            && let Some(shift_span) = find_forward_temporal_shift_span(chars, doc, events, idx, ev)
        {
            record_pair(covered_pairs, prev_idx, idx);
            raw_edges.push(RawEdge {
                kind: EdgeKind::Before,
                evidence_kind: EdgeEvidenceKind::TemporalAnchorShift,
                from_event: prev_idx,
                to_event: idx,
                connector_span: None,
                evidence_spans: vec![
                    LocatedSpan {
                        chapter,
                        span: shift_span,
                    },
                    LocatedSpan {
                        chapter,
                        span: prev_ev.predicate_span,
                    },
                    LocatedSpan {
                        chapter,
                        span: ev.predicate_span,
                    },
                ],
                confidence_tier: 3,
            });
        }

        prev_narrative_idx = Some(idx);
    }
}

fn event_has_flashback(
    chars: &[char],
    doc: &Document,
    events: &[CandidateEvent],
    idx: usize,
    ev: &CandidateEvent,
) -> bool {
    // 1. Check explicit `TemporalAnchor`s attached to this event.
    for anchor in &ev.temporal_anchors {
        let a_text = span_text(chars, anchor.span);
        if is_retrospective_cue_text(&a_text)
            || anchor_metadata_is_retrospective(anchor.value.as_deref(), anchor.unit.as_deref())
        {
            return true;
        }
    }

    // 2. Check the event's clause span (and sentence prefix if first event in sentence).
    let scope_span = event_temporal_search_span(doc, events, idx, ev);
    let text = span_text(chars, scope_span);
    if contains_retrospective_cue(&text) {
        return true;
    }

    // 3. Check `doc.entities` inside the scope span (outside `DirectQuote`).
    if ev.zone == ContextZone::Narrative {
        for ent in &doc.entities {
            if matches!(
                ent.kind,
                EntityKind::Date
                    | EntityKind::RelativeDate
                    | EntityKind::Time
                    | EntityKind::Quantity
            ) && span_contains(scope_span, ent.span)
            {
                let e_text = span_text(chars, ent.span);
                if is_retrospective_cue_text(&e_text) {
                    return true;
                }
            }
        }
    }

    false
}

fn find_forward_temporal_shift_span(
    chars: &[char],
    doc: &Document,
    events: &[CandidateEvent],
    idx: usize,
    ev: &CandidateEvent,
) -> Option<Span> {
    // 1. Check attached `TemporalAnchor`s belonging to this clause/sentence prefix.
    for anchor in &ev.temporal_anchors {
        let a_text = span_text(chars, anchor.span);
        if is_forward_shift_cue_text(&a_text) {
            return Some(anchor.span);
        }
    }

    // 2. Check `doc.entities` inside this event's temporal search span.
    let scope_span = event_temporal_search_span(doc, events, idx, ev);
    for ent in &doc.entities {
        if matches!(
            ent.kind,
            EntityKind::Date | EntityKind::RelativeDate | EntityKind::Time | EntityKind::Quantity
        ) && span_contains(scope_span, ent.span)
        {
            let e_text = span_text(chars, ent.span);
            if is_forward_shift_cue_text(&e_text) {
                return Some(ent.span);
            }
        }
    }

    // 3. Direct lexical scan over `scope_span`.
    find_forward_shift_cue_in_span(chars, scope_span)
}

fn event_temporal_search_span(
    doc: &Document,
    events: &[CandidateEvent],
    idx: usize,
    ev: &CandidateEvent,
) -> Span {
    let is_first_in_sentence = idx == 0 || events[idx - 1].sentence != ev.sentence;
    if is_first_in_sentence
        && let Some(sent) = doc.segmentation.sentences.get(ev.sentence)
        && sent.span.start_char <= ev.clause_span.start_char
    {
        return Span {
            start_char: sent.span.start_char,
            end_char: ev.clause_span.end_char,
        };
    }
    ev.clause_span
}

// ---------------------------------------------------------------------------
// Tier 4 IntraSentenceClauseFlow
// ---------------------------------------------------------------------------

fn collect_intra_sentence_flow_edges(
    chapter: u32,
    events: &[CandidateEvent],
    event_flashback: &[bool],
    covered_pairs: &mut HashSet<(usize, usize)>,
    raw_edges: &mut Vec<RawEdge>,
) {
    let mut prev_in_sentence: Option<usize> = None;

    for (idx, ev) in events.iter().enumerate() {
        if !is_chronological_event(ev) {
            continue;
        }
        if let Some(prev_idx) = prev_in_sentence {
            let prev_ev = &events[prev_idx];
            if prev_ev.sentence == ev.sentence
                && prev_ev.clause < ev.clause
                && !covered_pairs.contains(&(prev_idx, idx))
                && !event_flashback[prev_idx]
                && !event_flashback[idx]
            {
                record_pair(covered_pairs, prev_idx, idx);
                raw_edges.push(RawEdge {
                    kind: EdgeKind::Before,
                    evidence_kind: EdgeEvidenceKind::IntraSentenceClauseFlow,
                    from_event: prev_idx,
                    to_event: idx,
                    connector_span: None,
                    evidence_spans: vec![
                        LocatedSpan {
                            chapter,
                            span: prev_ev.predicate_span,
                        },
                        LocatedSpan {
                            chapter,
                            span: ev.predicate_span,
                        },
                    ],
                    confidence_tier: 4,
                });
            }
        }
        prev_in_sentence = Some(idx);
    }
}

// ---------------------------------------------------------------------------
// Equivalence Condensation, Deterministic Cycle Breaking & Longest-Path Layers
// ---------------------------------------------------------------------------

fn condense_and_break_cycles(
    chapter: u32,
    events: &[CandidateEvent],
    edges: &mut [TimelineEdge],
) -> (
    ChapterTimelineDag,
    Vec<Option<u32>>,
    usize,
    Vec<NarrativeUnknown>,
) {
    let n = events.len();
    let is_chronological: Vec<bool> = events.iter().map(is_chronological_event).collect();

    // Sort candidate chronological edges deterministically by `(confidence_tier ASC, edge_index ASC)`.
    let mut candidate_indices: Vec<usize> = edges
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.status == EdgeStatus::Supported
                && matches!(
                    e.kind,
                    EdgeKind::Before | EdgeKind::Cause | EdgeKind::Simultaneous
                )
        })
        .map(|(idx, _)| idx)
        .collect();

    candidate_indices.sort_by_key(|&idx| (edges[idx].confidence_tier, idx));

    let mut uf = UnionFind::new(n);
    let mut active = vec![true; n];
    let mut reach = BitMatrix::new(n);
    let mut accepted_strict: Vec<(usize, usize)> = Vec::new();
    let mut conflict_cycles_broken = 0usize;
    let mut cycle_unknowns = Vec::new();

    for edge_idx in candidate_indices {
        let (u, v, kind) = {
            let e = &edges[edge_idx];
            (e.from_event, e.to_event, e.kind)
        };
        if u >= n || v >= n || !is_chronological[u] || !is_chronological[v] {
            edges[edge_idx].status = EdgeStatus::NonChronologicalFactuality;
            continue;
        }

        let ru = uf.find(u);
        let rv = uf.find(v);

        match kind {
            EdgeKind::Before | EdgeKind::Cause => {
                if ru == rv || reach.has(rv, ru) {
                    // Closing a cycle or contradicting a `Simultaneous` equivalence class.
                    edges[edge_idx].status = EdgeStatus::ConflictCycle;
                    conflict_cycles_broken += 1;
                    let span = edges[edge_idx]
                        .connector_span
                        .map(|ls| ls.span)
                        .unwrap_or(events[v].predicate_span);
                    cycle_unknowns.push(NarrativeUnknown {
                        kind: NarrativeUnknownKind::CycleConflictDemoted,
                        chapter,
                        span,
                        detail: format!(
                            "demoted {} edge {} ({} -> {}) to preserve cycle-free DAG",
                            edges[edge_idx].edge_id,
                            match kind {
                                EdgeKind::Before => "before",
                                EdgeKind::Cause => "cause",
                                _ => "chronological",
                            },
                            events[u].event_id,
                            events[v].event_id
                        ),
                    });
                } else {
                    accepted_strict.push((u, v));
                    if !reach.has(ru, rv) {
                        // Every active component `a` that is `ru` or strictly reaches `ru`
                        // can now strictly reach `rv` and everything `rv` strictly reaches.
                        let rv_closure = reach.row_with_bit(rv);
                        for (a, &is_active) in active.iter().enumerate() {
                            if is_active && (a == ru || reach.has(a, ru)) {
                                reach.or_slice(a, &rv_closure);
                            }
                        }
                    }
                }
            }
            EdgeKind::Simultaneous => {
                if ru == rv {
                    // Already in the same equivalence class.
                } else if reach.has(ru, rv) || reach.has(rv, ru) {
                    // Merging strictly ordered components would collapse a strict order into a cycle.
                    edges[edge_idx].status = EdgeStatus::ConflictCycle;
                    conflict_cycles_broken += 1;
                    let span = edges[edge_idx]
                        .connector_span
                        .map(|ls| ls.span)
                        .unwrap_or(events[v].predicate_span);
                    cycle_unknowns.push(NarrativeUnknown {
                        kind: NarrativeUnknownKind::CycleConflictDemoted,
                        chapter,
                        span,
                        detail: format!(
                            "demoted {} simultaneous edge ({} ~ {}) that contradicted strict order",
                            edges[edge_idx].edge_id, events[u].event_id, events[v].event_id
                        ),
                    });
                } else {
                    // Merge `rv` into `ru` while preserving exact strict reachability.
                    let rv_row = reach.clone_row(rv);
                    reach.or_slice(ru, &rv_row);
                    reach.clear_bit(ru, ru);
                    reach.clear_bit(ru, rv);

                    let ru_new_row = reach.clone_row(ru);
                    for (a, &is_active) in active.iter().enumerate() {
                        if is_active && a != ru && a != rv {
                            let reaches_merged = reach.has(a, ru) || reach.has(a, rv);
                            reach.clear_bit(a, rv);
                            if reaches_merged {
                                reach.set_bit(a, ru);
                                reach.or_slice(a, &ru_new_row);
                            }
                        }
                    }
                    uf.union_keep_left(ru, rv);
                    active[rv] = false;
                }
            }
            EdgeKind::Contrast | EdgeKind::Condition => {}
        }
    }

    // Compact active component IDs to `0..num_components` for the final `ChapterTimelineDag`.
    let mut comp_id_by_rep = vec![usize::MAX; n];
    let mut num_components = 0usize;
    let mut reps = Vec::new();
    for i in 0..n {
        let r = uf.find(i);
        if comp_id_by_rep[r] == usize::MAX {
            comp_id_by_rep[r] = num_components;
            reps.push(r);
            num_components += 1;
        }
    }
    let component_of: Vec<usize> = (0..n).map(|i| comp_id_by_rep[uf.find(i)]).collect();

    let mut compact_reach = BitMatrix::new(num_components);
    for (ca, &ra) in reps.iter().enumerate() {
        for (cb, &rb) in reps.iter().enumerate() {
            if ca != cb && reach.has(ra, rb) {
                compact_reach.set_bit(ca, cb);
            }
        }
    }

    // Compute longest-path topological depths on the condensed `Supported` DAG.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_components];
    let mut indegree = vec![0usize; num_components];
    let mut seen_comp_edges: HashSet<(usize, usize)> = HashSet::new();

    for (u, v) in accepted_strict {
        let cu = component_of[u];
        let cv = component_of[v];
        if cu != cv && seen_comp_edges.insert((cu, cv)) {
            adj[cu].push(cv);
            indegree[cv] += 1;
        }
    }

    let mut comp_layers = vec![0u32; num_components];
    let mut queue: VecDeque<usize> = (0..num_components).filter(|&c| indegree[c] == 0).collect();

    while let Some(u) = queue.pop_front() {
        let next_depth = comp_layers[u].saturating_add(1);
        for &v in &adj[u] {
            if next_depth > comp_layers[v] {
                comp_layers[v] = next_depth;
            }
            indegree[v] -= 1;
            if indegree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    let event_dag_layers: Vec<Option<u32>> = (0..n)
        .map(|i| {
            if is_chronological[i] {
                Some(comp_layers[component_of[i]])
            } else {
                None
            }
        })
        .collect();

    let dag = ChapterTimelineDag {
        num_events: n,
        is_chronological,
        component_of,
        num_components,
        reach: compact_reach,
    };

    (
        dag,
        event_dag_layers,
        conflict_cycles_broken,
        cycle_unknowns,
    )
}

// ---------------------------------------------------------------------------
// Flashback & Forward Temporal Cue Lexicons / Helpers
// ---------------------------------------------------------------------------

const RETROSPECTIVE_PHRASES: &[&str] = &[
    "此前",
    "當初",
    "当初",
    "先前",
    "當年",
    "当年",
    "昔日",
    "昔年",
    "往昔",
    "從前",
    "从前",
    "早先",
    "原先",
    "不久前",
    "早些年",
    "曾幾何時",
    "曾几何时",
    "在此之前",
    "在這之前",
    "在这之前",
    "昨日",
    "昨天",
    "前天",
    "前日",
    "去年",
    "前年",
    "上個月",
    "上个月",
    "上週",
    "上周",
];

const EN_RETROSPECTIVE_PHRASES: &[&str] = &[
    "years ago",
    "months ago",
    "weeks ago",
    "days ago",
    "hours ago",
    "minutes ago",
    "moments ago",
    "long ago",
    "earlier",
    "previously",
    "formerly",
    "in the past",
    "back then",
    "yesterday",
    "last year",
    "last month",
    "last week",
];

const FORWARD_SHIFT_PHRASES: &[&str] = &[
    "次日",
    "翌日",
    "隔日",
    "隔天",
    "隔夜",
    "翌晨",
    "次晨",
    "次年",
    "翌年",
    "第二天",
    "第三天",
    "片刻後",
    "片刻后",
    "半晌後",
    "半晌后",
    "良久後",
    "良久后",
    "許久後",
    "许久后",
    "一會兒後",
    "一会儿后",
    "不久後",
    "不久后",
    "不多時",
    "不多时",
    "沒多久",
    "没多久",
    "轉眼間",
    "转眼间",
    "過了片刻",
    "过了片刻",
    "過了一會",
    "过了一会",
    "過了數日",
    "过了数日",
    "過了幾天",
    "过了几天",
    "明日",
    "明天",
    "後天",
    "后天",
    "明年",
    "下個月",
    "下个月",
    "下週",
    "下周",
];

const EN_FORWARD_SHIFT_PHRASES: &[&str] = &[
    "days later",
    "day later",
    "hours later",
    "hour later",
    "minutes later",
    "minute later",
    "moments later",
    "moment later",
    "weeks later",
    "week later",
    "months later",
    "month later",
    "years later",
    "year later",
    "the next morning",
    "the next day",
    "the next evening",
    "the next night",
    "next morning",
    "next day",
    "the following day",
    "the following morning",
    "shortly after",
    "soon after",
    "tomorrow",
];

const ZH_TIME_UNITS: &[&str] = &[
    "個時辰",
    "个时辰",
    "時辰",
    "时辰",
    "個月",
    "个月",
    "刻鐘",
    "刻钟",
    "分鐘",
    "分钟",
    "小時",
    "小时",
    "柱香",
    "炷香",
    "盞茶",
    "盏茶",
    "星期",
    "載",
    "载",
    "年",
    "月",
    "週",
    "周",
    "天",
    "日",
    "夜",
    "更",
    "刻",
    "息",
    "瞬",
];

const ZH_NUM_CHARS: &str = "一二兩两三四五六七八九十百千萬万半幾几數数多這这那0123456789";

fn chapter_opens_with_flashback(ch: &ChapterNarrative) -> bool {
    if let Some(title) = ch.title.as_deref()
        && (contains_retrospective_cue(title)
            || title.contains("回憶")
            || title.contains("回忆")
            || title.contains("往事")
            || title.to_ascii_lowercase().contains("flashback"))
    {
        return true;
    }
    if let Some(first_actual) = ch.events.iter().find(|ev| is_chronological_event(ev)) {
        return event_has_retrospective_anchor_metadata(first_actual);
    }
    false
}

fn event_has_retrospective_anchor_metadata(ev: &CandidateEvent) -> bool {
    ev.temporal_anchors
        .iter()
        .any(|a| anchor_metadata_is_retrospective(a.value.as_deref(), a.unit.as_deref()))
}

fn anchor_metadata_is_retrospective(value: Option<&str>, unit: Option<&str>) -> bool {
    for s in [value, unit].into_iter().flatten() {
        let trimmed = s.trim();
        if trimmed.starts_with('-')
            || contains_retrospective_cue(trimmed)
            || trimmed.to_ascii_lowercase().contains("flashback")
            || trimmed.to_ascii_lowercase().contains("past")
        {
            return true;
        }
    }
    false
}

fn is_retrospective_cue_text(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }
    if contains_retrospective_cue(t) {
        return true;
    }
    // Check `<num> + <unit> + 前/之前`.
    has_zh_duration_with_suffix(t, &["之前", "前"]).is_some()
}

fn contains_retrospective_cue(text: &str) -> bool {
    if RETROSPECTIVE_PHRASES.iter().any(|p| text.contains(p)) {
        return true;
    }
    if has_zh_duration_with_suffix(text, &["之前", "前"]).is_some() {
        return true;
    }
    let lower = text.to_ascii_lowercase();
    EN_RETROSPECTIVE_PHRASES
        .iter()
        .any(|p| contains_ascii_phrase_boundary(&lower, p))
}

fn is_forward_shift_cue_text(text: &str) -> bool {
    if is_retrospective_cue_text(text) {
        return false;
    }
    if FORWARD_SHIFT_PHRASES.iter().any(|p| text.contains(p)) {
        return true;
    }
    if has_zh_duration_with_suffix(text, &["之後", "之后", "後", "后"]).is_some() {
        return true;
    }
    let lower = text.to_ascii_lowercase();
    EN_FORWARD_SHIFT_PHRASES
        .iter()
        .any(|p| contains_ascii_phrase_boundary(&lower, p))
}

fn find_forward_shift_cue_in_span(chars: &[char], span: Span) -> Option<Span> {
    if !valid_span(chars, span) {
        return None;
    }
    let slice_str = span_text(chars, span);
    if contains_retrospective_cue(&slice_str) {
        return None;
    }

    for phrase in FORWARD_SHIFT_PHRASES {
        if let Some(rel_char) = find_substring_char_offset(&slice_str, phrase) {
            let len = phrase.chars().count();
            return Some(Span {
                start_char: span.start_char + rel_char,
                end_char: span.start_char + rel_char + len,
            });
        }
    }

    if let Some((rel_start, rel_end)) =
        has_zh_duration_with_suffix(&slice_str, &["之後", "之后", "後", "后"])
    {
        return Some(Span {
            start_char: span.start_char + rel_start,
            end_char: span.start_char + rel_end,
        });
    }

    let lower = slice_str.to_ascii_lowercase();
    for phrase in EN_FORWARD_SHIFT_PHRASES {
        if let Some(rel_char) = find_ascii_phrase_char_offset(&lower, phrase) {
            let len = phrase.chars().count();
            return Some(Span {
                start_char: span.start_char + rel_char,
                end_char: span.start_char + rel_char + len,
            });
        }
    }

    None
}

/// Searches `text` for `<zh_numeral> + [個/个] + <time_unit> + <suffix>` and returns `(start_char, end_char)`.
fn has_zh_duration_with_suffix(text: &str, suffixes: &[&str]) -> Option<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    while i < n {
        if ZH_NUM_CHARS.contains(chars[i]) {
            let start = i;
            while i < n && ZH_NUM_CHARS.contains(chars[i]) {
                i += 1;
            }
            if i < n && matches!(chars[i], '個' | '个') {
                i += 1;
            }
            let mut matched_unit = None;
            for &unit in ZH_TIME_UNITS {
                let u_len = unit.chars().count();
                if i + u_len <= n && unit.chars().eq(chars[i..i + u_len].iter().copied()) {
                    matched_unit = Some(u_len);
                    break;
                }
            }
            if let Some(u_len) = matched_unit {
                let after_unit = i + u_len;
                for &suf in suffixes {
                    let s_len = suf.chars().count();
                    if after_unit + s_len <= n
                        && suf
                            .chars()
                            .eq(chars[after_unit..after_unit + s_len].iter().copied())
                    {
                        return Some((start, after_unit + s_len));
                    }
                }
            }
        } else {
            i += 1;
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Low-Level Span, Text, Union-Find, and BitMatrix Utilities
// ---------------------------------------------------------------------------

fn valid_span(chars: &[char], span: Span) -> bool {
    span.start_char <= span.end_char && span.end_char <= chars.len()
}

fn spans_overlap(a: Span, b: Span) -> bool {
    a.start_char < b.end_char && b.start_char < a.end_char
}

fn span_contains(outer: Span, inner: Span) -> bool {
    outer.start_char <= inner.start_char && inner.end_char <= outer.end_char
}

fn span_text(chars: &[char], span: Span) -> String {
    if span.start_char >= span.end_char || span.end_char > chars.len() {
        return String::new();
    }
    chars[span.start_char..span.end_char].iter().collect()
}

fn trim_span_text(chars: &[char], span: Span) -> String {
    if span.start_char >= span.end_char || span.end_char > chars.len() {
        return String::new();
    }
    let s = &chars[span.start_char..span.end_char];
    let l = s
        .iter()
        .position(|ch| !ch.is_whitespace() && !is_punct_or_quote(*ch))
        .unwrap_or(s.len());
    let r = s
        .iter()
        .rposition(|ch| !ch.is_whitespace() && !is_punct_or_quote(*ch))
        .map_or(l, |p| p + 1);
    s[l..r].iter().collect()
}

fn has_lexical_content_excluding(
    chars: &[char],
    arg: Span,
    connector: Span,
    correlate: Option<Span>,
) -> bool {
    if arg.start_char >= arg.end_char || arg.end_char > chars.len() {
        return false;
    }
    for (idx, &ch) in chars
        .iter()
        .enumerate()
        .take(arg.end_char)
        .skip(arg.start_char)
    {
        let in_conn = idx >= connector.start_char && idx < connector.end_char;
        let in_corr = correlate.is_some_and(|c| idx >= c.start_char && idx < c.end_char);
        if !in_conn && !in_corr && !ch.is_whitespace() && !is_punct_or_quote(ch) {
            return true;
        }
    }
    false
}

fn is_punct_or_quote(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '，' | '。'
                | '！'
                | '？'
                | '、'
                | '；'
                | '：'
                | '「'
                | '」'
                | '『'
                | '』'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '（'
                | '）'
                | '《'
                | '》'
                | '〈'
                | '〉'
                | '—'
                | '…'
                | '·'
        )
}

fn pick_event_in_span(events: &[CandidateEvent], span: Span, prefer_last: bool) -> Option<usize> {
    let iter = events.iter().enumerate().filter(|(_, ev)| {
        span_contains(span, ev.predicate_span)
            || (spans_overlap(span, ev.clause_span) && spans_overlap(span, ev.predicate_span))
    });
    let all_matches: Vec<(usize, &CandidateEvent)> = iter.collect();
    if all_matches.is_empty() {
        return None;
    }
    let chronological_matches: Vec<usize> = all_matches
        .iter()
        .filter(|(_, ev)| is_chronological_event(ev))
        .map(|(idx, _)| *idx)
        .collect();
    if !chronological_matches.is_empty() {
        if prefer_last {
            chronological_matches.last().copied()
        } else {
            chronological_matches.first().copied()
        }
    } else if prefer_last {
        all_matches.last().map(|(idx, _)| *idx)
    } else {
        all_matches.first().map(|(idx, _)| *idx)
    }
}

fn skip_opening_chars(chars: &[char], doc: &Document, span: Span) -> usize {
    let mut i = span.start_char.min(chars.len());
    let limit = span.end_char.min(chars.len());
    while i < limit {
        if let Some(o) = doc
            .segmentation
            .opaque
            .iter()
            .find(|o| o.span.start_char <= i && i < o.span.end_char)
        {
            i = o.span.end_char;
        } else if chars[i].is_whitespace()
            || matches!(
                chars[i],
                '「' | '『' | '“' | '‘' | '（' | '(' | '[' | '"' | '\'' | '，' | '、'
            )
        {
            i += 1;
        } else {
            break;
        }
    }
    i.min(limit)
}

fn trim_closing_chars(chars: &[char], start: usize, end: usize) -> usize {
    let mut e = end.min(chars.len());
    while e > start && (chars[e - 1].is_whitespace() || is_punct_or_quote(chars[e - 1])) {
        e -= 1;
    }
    e
}

fn starts_with_chars(chars: &[char], from: usize, limit: usize, pat: &str) -> Option<usize> {
    let len = pat.chars().count();
    let end = from.checked_add(len)?;
    if end <= limit && pat.chars().eq(chars[from..end].iter().copied()) {
        Some(end)
    } else {
        None
    }
}

fn ends_with_chars(chars: &[char], start: usize, end: usize, pat: &str) -> bool {
    let len = pat.chars().count();
    end >= start + len && pat.chars().eq(chars[end - len..end].iter().copied())
}

fn starts_with_ascii_word_ci(
    chars: &[char],
    from: usize,
    limit: usize,
    word: &str,
) -> Option<usize> {
    let len = word.chars().count();
    let end = from.checked_add(len)?;
    if end > limit {
        return None;
    }
    if !word
        .chars()
        .zip(&chars[from..end])
        .all(|(a, b)| a.eq_ignore_ascii_case(b))
    {
        return None;
    }
    if chars
        .get(end)
        .is_some_and(|&ch| ch.is_ascii_alphanumeric() || matches!(ch, '\'' | '’' | '-'))
    {
        return None;
    }
    Some(end)
}

fn starts_with_ascii_phrase_ci(
    chars: &[char],
    from: usize,
    limit: usize,
    phrase: &str,
) -> Option<usize> {
    let len = phrase.chars().count();
    let end = from.checked_add(len)?;
    if end > limit {
        return None;
    }
    if !phrase
        .chars()
        .zip(&chars[from..end])
        .all(|(a, b)| a.eq_ignore_ascii_case(b))
    {
        return None;
    }
    if chars
        .get(end)
        .is_some_and(|&ch| ch.is_ascii_alphanumeric() || matches!(ch, '\'' | '’' | '-'))
    {
        return None;
    }
    Some(end)
}

fn starts_with_any(text: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| text.starts_with(p))
}

fn starts_with_ci(text: &str, prefix: &str) -> bool {
    let p_len = prefix.chars().count();
    let t_chars: Vec<char> = text.chars().take(p_len).collect();
    t_chars.len() == p_len
        && prefix
            .chars()
            .zip(t_chars)
            .all(|(a, b)| a.eq_ignore_ascii_case(&b))
}

fn find_substring_char_offset(haystack: &str, needle: &str) -> Option<usize> {
    let byte_idx = haystack.find(needle)?;
    Some(haystack[..byte_idx].chars().count())
}

fn contains_ascii_phrase_boundary(lower_haystack: &str, lower_needle: &str) -> bool {
    find_ascii_phrase_char_offset(lower_haystack, lower_needle).is_some()
}

fn find_ascii_phrase_char_offset(lower_haystack: &str, lower_needle: &str) -> Option<usize> {
    let mut offset = 0usize;
    while let Some(pos) = lower_haystack[offset..].find(lower_needle) {
        let start_b = offset + pos;
        let end_b = start_b + lower_needle.len();
        let left_ok = lower_haystack[..start_b]
            .chars()
            .next_back()
            .is_none_or(|ch| !ch.is_ascii_alphanumeric());
        let right_ok = lower_haystack[end_b..]
            .chars()
            .next()
            .is_none_or(|ch| !ch.is_ascii_alphanumeric());
        if left_ok && right_ok {
            return Some(lower_haystack[..start_b].chars().count());
        }
        offset = end_b;
    }
    None
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union_keep_left(&mut self, left_rep: usize, right_rep: usize) {
        if left_rep != right_rep {
            self.parent[right_rep] = left_rep;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BitMatrix {
    words_per_row: usize,
    bits: Vec<u64>,
}

impl BitMatrix {
    fn new(n: usize) -> Self {
        let words_per_row = n.div_ceil(64).max(1);
        Self {
            words_per_row,
            bits: vec![0u64; n * words_per_row],
        }
    }

    fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        let start = row * self.words_per_row;
        start..(start + self.words_per_row)
    }

    fn has(&self, row: usize, col: usize) -> bool {
        let word = row * self.words_per_row + (col / 64);
        let bit = 1u64 << (col % 64);
        (self.bits[word] & bit) != 0
    }

    fn set_bit(&mut self, row: usize, col: usize) {
        let word = row * self.words_per_row + (col / 64);
        let bit = 1u64 << (col % 64);
        self.bits[word] |= bit;
    }

    fn clear_bit(&mut self, row: usize, col: usize) {
        let word = row * self.words_per_row + (col / 64);
        let bit = 1u64 << (col % 64);
        self.bits[word] &= !bit;
    }

    fn clone_row(&self, row: usize) -> Vec<u64> {
        self.bits[self.row_range(row)].to_vec()
    }

    fn row_with_bit(&self, row: usize) -> Vec<u64> {
        let mut out = self.clone_row(row);
        out[row / 64] |= 1u64 << (row % 64);
        out
    }

    fn or_slice(&mut self, dst_row: usize, src: &[u64]) {
        let dst_start = dst_row * self.words_per_row;
        for (i, &w) in src.iter().enumerate() {
            self.bits[dst_start + i] |= w;
        }
    }
}
