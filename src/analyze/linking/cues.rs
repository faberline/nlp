use std::collections::HashMap;

use super::{
    ClauseCandidate, ClauseScope, DisposalFrame, SEQUENCE, ScopedClause, Source, Span, Unit,
    UnitScope, actor_compatible, candidate_scope, contained, elapsed_order_cue,
    grounded_action_tail, inside_lexical_term, known_action_head, local_line_gap, nominal_measure,
    predicate, scene_transition, temporal_boundary, temporal_bridge, temporal_endpoint,
    temporal_shift, temporal_subject, trim_clause_end, written_prior_head,
};
use crate::Lexicon;

/// The kind of written order cue. A reference has no local action anchor by itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClauseOrderCueKind {
    Sequence,
    Elapsed,
    Before,
    After,
    Reference,
    FirstThen,
}

/// A cue in the original chapter, with the endpoints the analysis could establish.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClauseOrderCue {
    pub kind: ClauseOrderCueKind,
    pub cue: Span,
    pub paired_cue: Option<Span>,
    /// Written construction from its first marker or action to its last marker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub construction: Option<Span>,
    pub scope: ClauseScope,
    /// Index of the earlier action in the accepted clauses, when resolved.
    pub before: Option<usize>,
    /// Index of the later action in the accepted clauses, when resolved.
    pub after: Option<usize>,
}

/// An order cue and the available source evidence for each side.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RankedClauseOrderCue {
    #[serde(flatten)]
    pub order: ClauseOrderCue,
    /// Up to five source-supported alternatives for the earlier action, strongest first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub before_alternatives: Vec<ClauseOrderEndpoint>,
    /// Up to five source-supported alternatives for the later action, strongest first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub after_alternatives: Vec<ClauseOrderEndpoint>,
}

/// The source evidence for one possible endpoint of an order cue.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClauseOrderEndpoint {
    /// Index into the accepted clauses. Its proposed predicate can include more than the head.
    pub clause: usize,
    pub head: Span,
    pub completion: Option<Span>,
    pub argument: Option<Span>,
    /// This alternative passed the analyzer's built-in head recognition rules.
    pub lexical_head: bool,
    /// The evidence meets the binding rule for this side. Competing options can leave it open.
    pub eligible: bool,
}

/// A caller-verified predicate head in the original, unmodified chapter text.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VerifiedClauseHead {
    /// Index into the input candidates, including any the analyzer rejects.
    pub input_index: usize,
    /// Exact alphabetic predicate head, excluding completion and arguments.
    /// This range counts Unicode scalar values and must be a prefix of the proposed predicate.
    /// The caller must verify that it is used as a predicate in context.
    pub head: Span,
}

/// Caller-verified placement evidence for an advisory endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VerifiedClauseEvidence {
    pub head: VerifiedClauseHead,
    /// Exact subject phrase in the same original chapter, before the head.
    /// The caller must verify that this phrase is the subject of the supplied predicate.
    /// It must agree with any actor range in the candidate and fit the local cue placement.
    pub subject: Option<Span>,
}

/// Chronological side of a written cue, independent of where the action appears in the text.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClauseOrderSide {
    Before,
    After,
}

/// Explain one candidate against one side of an exact written cue.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClauseOrderProbe {
    pub cue: Span,
    pub side: ClauseOrderSide,
    /// Index in the original candidate slice, including rejected proposals.
    pub input_index: usize,
}

/// A window the resolver searched for this side. All ranges count Unicode scalar values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClauseOrderWindow {
    pub unit: Span,
    /// Source from the applicable anchor to the end of the local evidence, or the prefix
    /// up to a cue or trimmed unit end for an end anchor.
    pub window: Span,
    /// Exact written anchor the strict rule requires when the construction fixes the head.
    pub written_head: Option<Span>,
    /// Normalized written head used for matching advisory alternatives.
    pub required_head: Option<Span>,
    /// This window was searched only to offer alternatives.
    pub advisory_only: bool,
}

/// Outcome of the advisory candidate checks, before applying the display limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClauseOrderAdvisory {
    Admitted,
    CueNotFound,
    ReferenceOnly,
    NonNarrated,
    UnknownCandidate,
    CandidateRejected,
    NoWindow,
    OutsideUnit,
    OutsideAnchor,
    ScopeOrLexicalGuard,
    HeadNotRecognized,
    SubjectBridge,
    Tail,
}

/// Strict outcome, separate from whether the candidate reached the advisory ranking.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClauseOrderBinding {
    NotEvaluated,
    AdvisoryOnly,
    /// This construction requires written completion. In a pair either side may supply
    /// the missing evidence.
    MissingCompletion,
    /// A construction lacks source support for an action's full tail.
    UngroundedTail,
    SubjectBridge,
    Competing,
    SameClause,
    Bound,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClauseOrderDiagnostic {
    pub probe: ClauseOrderProbe,
    pub accepted_clause: Option<usize>,
    pub windows: Vec<ClauseOrderWindow>,
    pub advisory: ClauseOrderAdvisory,
    /// Zero-based rank in the returned side list. An admitted candidate can be outside the limit.
    pub rank: Option<usize>,
    pub binding: ClauseOrderBinding,
}

use ClauseOrderCueKind::{After, Before, Elapsed, FirstThen, Reference, Sequence};

const MORE_SEQUENCE: &[&str] = &[
    "緊接其後",
    "紧接其后",
    "緊隨其後",
    "紧随其后",
    "繼而",
    "继而",
    "而後",
    "而后",
    "隨之",
    "随之",
    "接下去",
];
const BEFORE: &[&str] = &["之前", "以前", "以先"];
const AFTER: &[&str] = &["之後", "之后", "以後", "以后", "過後", "过后"];
const CALENDAR: &[&str] = &[
    "次日",
    "次天",
    "翌日",
    "隔日",
    "隔天",
    "第二天",
    "翌晨",
    "翌晚",
    "前一天",
    "先前",
    "此前",
    "此後",
    "此后",
    "事後",
    "事后",
];
const SHORT_ELAPSED: &[&str] = &[
    "不一會兒",
    "不一会儿",
    "不一會",
    "不一会",
    "不多時",
    "不多时",
    "轉瞬間",
    "转瞬间",
    "一轉眼",
    "一转眼",
    "轉眼間",
    "转眼间",
];

#[derive(Clone, Copy)]
enum Placement {
    Leading,
    Interior,
    Detached,
    Attached,
    EventBound {
        opening: Option<Span>,
        head: Span,
        grounded: bool,
        trailing: bool,
    },
    Paired {
        first: Span,
        heads: [Span; 2],
        blocked: [Option<ClauseOrderBinding>; 2],
    },
}

#[derive(Clone, Copy)]
struct WrittenCue {
    kind: ClauseOrderCueKind,
    span: Span,
    placement: Placement,
}

fn source_lead(source: &Source<'_>, unit: &Unit, end: usize) -> usize {
    let start = source.bytes[unit.evidence.start_char];
    let prefix = &source.text[start..end];
    let lead =
        prefix.trim_start_matches(|ch: char| ch.is_whitespace() || super::quote_end(ch).is_some());
    end - lead.len()
}

fn sequence_placement(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    unit: &Unit,
    span: Span,
) -> Placement {
    let start = source.bytes[span.start_char];
    let end = source.bytes[span.end_char];
    let from = source_lead(source, unit, start);
    let unit_end = source.bytes[unit.evidence.end_char];
    let tail = source.text[end..unit_end].trim_start();
    if from == start && trim_clause_end(tail).is_empty() {
        Placement::Detached
    } else if temporal_bridge(source, lexicon, from, start, None).is_some() {
        Placement::Leading
    } else {
        Placement::Interior
    }
}

fn following_lexical_head(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    unit: &Unit,
    cue: Span,
) -> Option<Span> {
    lexical_head(
        source,
        lexicon,
        unit,
        source.bytes[cue.end_char],
        source.bytes[unit.evidence.end_char],
    )
}

fn lexical_head(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    unit: &Unit,
    start: usize,
    end: usize,
) -> Option<Span> {
    let (_, offset, word) = predicate(&source.text[start..end], lexicon)?;
    let head = source.span(start + offset, start + offset + word.len());
    (temporal_bridge(source, lexicon, start, start + offset, None).is_some()
        && !inside_lexical_term(source, lexicon, unit.evidence, head))
    .then_some(head)
}

#[derive(Clone, Copy)]
enum SpatialTail {
    Landmark,
    Possible,
}

fn spatial_order_tail(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    unit: &Unit,
    end: usize,
) -> Option<SpatialTail> {
    let start = source_lead(source, unit, end);
    let before = source.text[start..end].trim();
    let noun = |text: &str| {
        lexicon.places.contains(&text)
            || lexicon.objects.iter().any(|&(word, _)| word == text)
            || matches!(
                text,
                "門" | "门" | "門口" | "门口" | "窗" | "牆" | "墙" | "桌" | "桌子"
            )
    };
    if noun(before) {
        return Some(SpatialTail::Landmark);
    }
    let (offset, marker) = before
        .rmatch_indices(['在', '於', '于', '到', '向', '朝', '往'])
        .next()?;
    let lead = &before[..offset];
    let landmark = before[offset + marker.len()..].trim();
    if landmark.is_empty() || matches!(landmark, "那" | "這" | "这" | "此" | "其") {
        return None;
    }
    let location = if noun(landmark) {
        SpatialTail::Landmark
    } else {
        SpatialTail::Possible
    };
    if marker == "到" {
        return (lead.ends_with([
            '走', '跑', '飛', '飞', '回', '來', '来', '抵', '趕', '赶', '推', '拉', '移', '站',
            '放', '搬', '擺', '摆', '送',
        ]) && predicate(landmark, lexicon).is_none())
        .then_some(location);
    }
    (noun(landmark)
        || (matches!(marker, "向" | "朝" | "往")
            || lead.ends_with(['站', '坐', '停', '躲', '蹲', '靠', '位']))
            && predicate(landmark, lexicon).is_none())
    .then_some(location)
}

fn event_bound(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    unit: &Unit,
    cue: Span,
    trailing: bool,
) -> Option<Placement> {
    let end = source.bytes[cue.start_char];
    let start = source_lead(source, unit, end);
    let before = &source.text[start..end];
    let framed = before
        .match_indices(['在', '於', '于'])
        .find_map(|(offset, marker)| {
            let from = start + offset;
            let to = from + marker.len();
            temporal_bridge(source, lexicon, start, from, None)?;
            let head = lexical_head(source, lexicon, unit, to, end)?;
            let opening = source.span(from, to);
            (!inside_lexical_term(source, lexicon, unit.evidence, opening))
                .then_some((Some(opening), head))
        });
    let (opening, head) = if let Some(framed) = framed {
        framed
    } else {
        if !trailing || spatial_order_tail(source, lexicon, unit, end).is_some() {
            return None;
        }
        (None, lexical_head(source, lexicon, unit, start, end)?)
    };
    let tail = &source.text[source.bytes[head.end_char]..end];
    if super::contains_any(tail, &["向後", "向后", "向前", "往後", "往后", "往前"]) {
        return None;
    }
    let grounded = grounded_action_tail(source, lexicon, ClauseCandidate::new(head), end, true);
    if (opening.is_none() && !grounded)
        || (!trailing && following_lexical_head(source, lexicon, unit, cue).is_none())
    {
        return None;
    }
    Some(Placement::EventBound {
        opening,
        head,
        grounded,
        trailing,
    })
}

fn written_cues(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    unit: &Unit,
    scope: &UnitScope,
) -> Vec<WrittenCue> {
    let text = source.slice(&unit.evidence);
    let base = source.bytes[unit.evidence.start_char];
    let mut found = Vec::new();
    let mut add = |start: usize, end: usize, kind, placement| {
        let span = source.span(base + start, base + end);
        let quoted_label = scope.quotes.iter().any(|quote| {
            quote.start_char + 1 == span.start_char && quote.end_char == span.end_char + 1
        });
        if !quoted_label && !inside_lexical_term(source, lexicon, unit.evidence, span) {
            found.push(WrittenCue {
                kind,
                span,
                placement,
            });
        }
    };

    for word in SEQUENCE
        .iter()
        .chain(MORE_SEQUENCE)
        .chain(SHORT_ELAPSED)
        .chain(CALENDAR)
    {
        for (start, _) in text.match_indices(word) {
            let end = start + word.len();
            let span = source.span(base + start, base + end);
            let placement = sequence_placement(source, lexicon, unit, span);
            if matches!(*word, "接著" | "接着")
                && !matches!(placement, Placement::Detached)
                && following_lexical_head(source, lexicon, unit, span).is_none()
            {
                continue;
            }
            let kind = if CALENDAR.contains(word) {
                Reference
            } else if SHORT_ELAPSED.contains(word) {
                Elapsed
            } else {
                Sequence
            };
            add(start, end, kind, placement);
        }
    }

    let body = trim_clause_end(text);
    for (words, kind) in [(BEFORE, Before), (AFTER, After)] {
        for word in words {
            for (start, _) in text.match_indices(word) {
                let end = start + word.len();
                let span = source.span(base + start, base + end);
                let placement = sequence_placement(source, lexicon, unit, span);
                if text[end..].trim_start().starts_with('的') {
                    add(start, end, Reference, placement);
                    continue;
                }
                if let Some(event) = event_bound(source, lexicon, unit, span, end == body.len()) {
                    add(start, end, kind, event);
                } else if let Some(spatial) =
                    spatial_order_tail(source, lexicon, unit, base + start)
                {
                    if matches!(spatial, SpatialTail::Possible) {
                        add(start, end, Reference, placement);
                    }
                } else if !matches!(placement, Placement::Leading | Placement::Detached)
                    && end == body.len()
                {
                    add(start, end, kind, Placement::Attached);
                } else {
                    let kind = if kind == After && !matches!(*word, "以後" | "以后") {
                        After
                    } else {
                        Reference
                    };
                    add(start, end, kind, placement);
                }
            }
        }
    }

    if let Some(last @ ('前' | '後' | '后')) = body.chars().next_back() {
        let end = body.len();
        let start = end - last.len_utf8();
        if !BEFORE.iter().chain(AFTER).any(|word| body.ends_with(word)) {
            let span = source.span(base + start, base + end);
            if let Some(event @ Placement::EventBound { grounded: true, .. }) =
                event_bound(source, lexicon, unit, span, true)
            {
                add(start, end, if last == '前' { Before } else { After }, event);
            }
        }
    }

    for (start, ch) in text.char_indices() {
        if !"過过不片半少一許许".contains(ch) {
            continue;
        }
        let tail = &text[start..];
        if let Some(len) = tail
            .char_indices()
            .map(|(byte, ch)| byte + ch.len_utf8())
            .take(12)
            .filter(|&end| {
                let phrase = &tail[..end];
                elapsed_order_cue(phrase)
                    || ["之後", "之后"]
                        .iter()
                        .any(|suffix| phrase.strip_suffix(suffix).is_some_and(elapsed_order_cue))
            })
            .last()
        {
            let span = source.span(base + start, base + start + len);
            add(
                start,
                start + len,
                Elapsed,
                sequence_placement(source, lexicon, unit, span),
            );
        }
    }

    if text.contains(['前', '後', '后']) {
        let mut start = unit.evidence.start_char;
        let end = unit.evidence.end_char;
        while let Some((backward, span)) = crate::timeline::written_duration(
            source.slice(&Span {
                start_char: start,
                end_char: end,
            }),
            start,
        ) {
            let short = !backward
                && !source
                    .slice(&span)
                    .contains(['年', '月', '日', '天', '週', '周', '星']);
            add(
                source.bytes[span.start_char] - base,
                source.bytes[span.end_char] - base,
                if short { Elapsed } else { Reference },
                sequence_placement(source, lexicon, unit, span),
            );
            start = span.end_char;
        }
    }
    found.sort_by_key(|cue| (cue.span.start_char, std::cmp::Reverse(cue.span.end_char)));
    let mut end = unit.evidence.start_char;
    found.retain(|cue| {
        if cue.span.start_char < end {
            return false;
        }
        end = cue.span.end_char;
        true
    });
    found
}

fn paired_action(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    unit: &Unit,
    words: &[&str],
) -> Option<(Span, Span)> {
    let text = source.slice(&unit.evidence);
    let base = source.bytes[unit.evidence.start_char];
    words.iter().find_map(|word| {
        text.match_indices(word).find_map(|(offset, _)| {
            let start = base + offset;
            let end = start + word.len();
            temporal_bridge(
                source,
                lexicon,
                source_lead(source, unit, start),
                start,
                None,
            )?;
            let span = source.span(start, end);
            let head = following_lexical_head(source, lexicon, unit, span)?;
            (temporal_bridge(source, lexicon, end, source.bytes[head.start_char], None)
                == Some(false)
                && !inside_lexical_term(source, lexicon, unit.evidence, span))
            .then_some((span, head))
        })
    })
}

fn written_head(source: &Source<'_>, lexical: Span) -> Span {
    let Some((stem, tail)) = source.slice(&lexical).split_once('了') else {
        return lexical;
    };
    if !stem.is_empty()
        && matches!(
            tail,
            "" | "下來" | "下来" | "過去" | "过去" | "上去" | "起來" | "起来"
        )
    {
        let start = source.bytes[lexical.start_char];
        source.span(start, start + stem.len())
    } else {
        lexical
    }
}

fn completion_after_head(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    head: Span,
    end: usize,
) -> bool {
    let head = written_head(source, head);
    let head_end = source.bytes[head.end_char];
    let tail = &source.text[head_end..end];
    let body = trim_clause_end(tail);
    if lexical_argument(source, lexicon, head_end, head_end + body.len()).is_some() {
        return false;
    }
    body.trim_start()
        .strip_prefix('了')
        .is_some_and(|rest| !rest.starts_with(['解', '然', '卻', '却']))
        || (body.ends_with('了')
            && grounded_action_tail(
                source,
                lexicon,
                ClauseCandidate::new(head),
                source.bytes[head.end_char] + body.len(),
                true,
            ))
}

fn head_in_relative_clause(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    unit: &Unit,
    head: Span,
) -> bool {
    let head = written_head(source, head);
    let start = source.bytes[head.end_char];
    let body = trim_clause_end(&source.text[start..source.bytes[unit.evidence.end_char]]);
    let end = start + body.len();
    if !body.contains('的')
        || grounded_action_tail(source, lexicon, ClauseCandidate::new(head), end, true)
    {
        return false;
    }
    // In a relative clause, 的 can follow the completed head and introduce
    // the subject of a different action. Prove that action before blocking the pair.
    body.match_indices('的').any(|(offset, marker)| {
        let from = start + offset + marker.len();
        lexical_head(source, lexicon, unit, from, end).is_some_and(|main| {
            temporal_bridge(source, lexicon, from, source.bytes[main.start_char], None)
                == Some(true)
                && grounded_action_tail(source, lexicon, ClauseCandidate::new(main), end, true)
        })
    })
}

fn written_pair(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    left: &Unit,
    right: &Unit,
) -> Option<WrittenCue> {
    if left.sentence != right.sentence || !source.slice(&left.evidence).ends_with(['，', ',']) {
        return None;
    }
    let (first, left_head, second, right_head, blocked) = if let Some((second, right_head)) =
        paired_action(source, lexicon, right, &["再"])
        && let Some((first, left_head)) = paired_action(source, lexicon, left, &["先"])
    {
        (first, left_head, second, right_head, [None; 2])
    } else {
        let (second, right_head) = paired_action(source, lexicon, right, &["就", "便", "即"])?;
        let (first, left_head) = paired_action(
            source,
            lexicon,
            left,
            &[
                "剛剛", "刚刚", "方一", "甫一", "剛一", "刚一", "剛", "刚", "一",
            ],
        )?;
        let left_text = source.slice(&left.evidence);
        let left_end = source.bytes[left.evidence.start_char] + trim_clause_end(left_text).len();
        let blocked = if !grounded_action_tail(
            source,
            lexicon,
            ClauseCandidate::new(left_head),
            left_end,
            true,
        ) {
            Some(ClauseOrderBinding::UngroundedTail)
        } else if !completion_after_head(source, lexicon, left_head, left_end)
            || !completion_after_head(
                source,
                lexicon,
                right_head,
                source.bytes[right.evidence.end_char],
            )
        {
            Some(ClauseOrderBinding::MissingCompletion)
        } else {
            None
        };
        let mut blocked = [blocked; 2];
        if blocked[1].is_none() && head_in_relative_clause(source, lexicon, right, right_head) {
            blocked[1] = Some(ClauseOrderBinding::UngroundedTail);
        }
        (first, left_head, second, right_head, blocked)
    };
    Some(WrittenCue {
        kind: FirstThen,
        span: second,
        placement: Placement::Paired {
            first,
            heads: [left_head, right_head],
            blocked,
        },
    })
}

#[derive(Clone, Copy)]
enum Anchor {
    End {
        end: usize,
        attached: Option<ClauseOrderCueKind>,
    },
    Start(usize),
    Head {
        written: Span,
        end: usize,
        blocked: Option<ClauseOrderBinding>,
    },
}

#[derive(Clone, Copy)]
struct Search {
    unit: usize,
    anchor: Anchor,
}

#[derive(Default)]
struct CueTrace {
    before: Option<Search>,
    after: Option<Search>,
    earlier_fallback: Option<Search>,
    after_resolved: Option<usize>,
}

#[derive(Default)]
struct Endpoints {
    resolved: Option<usize>,
    alternatives: Vec<ClauseOrderEndpoint>,
}

impl Endpoints {
    fn into_advisory(mut self) -> Vec<ClauseOrderEndpoint> {
        for option in &mut self.alternatives {
            option.eligible = false;
        }
        self.alternatives
    }
}

fn lexical_argument(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    start: usize,
    end: usize,
) -> Option<Span> {
    let span = source.trim(start, end)?;
    let text = source.slice(&span);
    let noun = nominal_measure(text).unwrap_or(text);
    if noun.is_empty()
        || !(lexicon
            .objects
            .iter()
            .chain(lexicon.actors)
            .any(|&(word, _)| word == noun)
            || lexicon.places.contains(&noun))
    {
        return None;
    }
    let end = source.bytes[span.end_char];
    Some(source.span(end - noun.len(), end))
}

fn head_evidence(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    clause: &ScopedClause,
    head: Span,
    end: usize,
) -> (Option<Span>, Option<Span>) {
    let start = source.bytes[head.end_char];
    let body = trim_clause_end(&source.text[start..end]);
    let end = start + body.len();
    let Some(span) = source.trim(start, end) else {
        return (None, clause.patient_evidence);
    };
    let text = source.slice(&span);
    let start = source.bytes[span.start_char];
    let end = source.bytes[span.end_char];
    if let Some(argument) = lexical_argument(source, lexicon, start, end) {
        return (None, Some(argument));
    }
    if let Some(rest) = text.strip_prefix('了') {
        let after = start + '了'.len_utf8();
        if !rest.starts_with(['解', '然', '卻', '却', '去', '來', '来']) {
            let argument =
                lexical_argument(source, lexicon, after, end).or(clause.patient_evidence);
            return (Some(source.span(start, after)), argument);
        }
    }
    if text.ends_with('了') {
        let marker = end - '了'.len_utf8();
        if let Some(argument) = lexical_argument(source, lexicon, start, marker) {
            return (Some(source.span(marker, end)), Some(argument));
        }
    }
    (None, clause.patient_evidence)
}

fn simple_action_tail(tail: &str, lexicon: &Lexicon<'_>) -> bool {
    tail.chars().count() <= 24
        && predicate(tail, lexicon).is_none()
        && !super::contains_any(tail, &["再", "又", "才", "便", "後", "后", "同時", "同时"])
}

fn earlier_modifier_bridge(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    from: usize,
    to: usize,
    subject: Option<Span>,
) -> bool {
    let prefix = source.text[from..to].trim_end();
    ["已經", "已经", "早已"].iter().any(|modifier| {
        prefix.strip_suffix(modifier).is_some_and(|lead| {
            source
                .trim(from, from + lead.len())
                .is_none_or(|span| Some(span) == subject || temporal_subject(source, lexicon, span))
        })
    })
}

fn contains_line_break(text: &str) -> bool {
    text.contains(|ch| matches!(ch, '\n'..='\r' | '\u{85}' | '\u{2028}' | '\u{2029}'))
}

fn single_action_bridge(
    source: &Source<'_>,
    from: usize,
    evidence_end: usize,
    head: Span,
    subject: Span,
) -> bool {
    let head_start = source.bytes[head.start_char];
    let subject_end = source.bytes[subject.end_char];
    // A verified one-character verb can follow 一 as its only written action marker.
    head.end_char == head.start_char + 1
        && matches!(
            source.slice(&head).chars().next(),
            Some(
                '\u{3400}'..='\u{4dbf}'
                | '\u{4e00}'..='\u{9fff}'
                | '\u{20000}'..='\u{2ffff}'
                | '\u{30000}'..='\u{323af}',
            )
        )
        && &source.text[subject_end..head_start] == "一"
        && !contains_line_break(&source.text[from..evidence_end])
}

fn nominalized_tail(tail: &str) -> bool {
    super::contains_any(
        tail,
        &[
            "的時間",
            "的时间",
            "的時機",
            "的时机",
            "的時候",
            "的时候",
            "的日子",
            "的前提",
            "的傳聞",
            "的传闻",
            "的消息",
            "的說法",
            "的说法",
        ],
    )
}

const DESCRIPTION_DELIMITERS: &[char] = &['。', '.', '，', ','];

fn static_scene_description(text: &str) -> bool {
    let Some(body) = text.strip_suffix(DESCRIPTION_DELIMITERS) else {
        return false;
    };
    let Some(state) = [
        "屋內", "屋内", "房內", "房内", "房中", "室內", "室内", "廳內", "厅内", "殿內", "殿内",
        "洞內", "洞内", "四周", "周圍", "周围", "周遭", "四下", "窗外", "天色", "光線", "光线",
    ]
    .iter()
    .find_map(|subject| body.strip_prefix(subject)) else {
        return false;
    };
    let state = state.trim();
    let state = [
        "仍然", "依然", "仍舊", "仍旧", "依舊", "依旧", "還是", "还是", "十分", "很",
    ]
    .iter()
    .find_map(|modifier| state.strip_prefix(modifier))
    .unwrap_or(state)
    .trim_start();
    let state = state.strip_prefix("一片").unwrap_or(state).trim_start();
    matches!(
        state,
        "寂靜"
            | "寂静"
            | "安靜"
            | "安静"
            | "寧靜"
            | "宁静"
            | "平靜"
            | "平静"
            | "靜悄悄"
            | "静悄悄"
            | "昏暗"
            | "陰暗"
            | "阴暗"
            | "漆黑"
            | "明亮"
            | "陰沉"
            | "阴沉"
    )
}

struct Resolver<'a, 'source> {
    source: &'a Source<'source>,
    lexicon: &'a Lexicon<'a>,
    units: &'a [Unit],
    scopes: &'a [UnitScope],
    frames: &'a [Option<DisposalFrame<'a>>],
    clauses: &'a [ScopedClause],
    indices: Vec<Vec<usize>>,
    verified_heads: HashMap<usize, Option<Span>>,
    verified_subjects: HashMap<usize, Option<Span>>,
}

impl Resolver<'_, '_> {
    fn allowed_endpoint(&self, unit: usize, clause: &ScopedClause) -> bool {
        let source = self.source;
        let lexicon = self.lexicon;
        let proposed = clause.candidate.predicate;
        let proposed_end = source.bytes[proposed.end_char];
        let unit_end = source.bytes[clause.clause.end_char];
        let tail = &source.text[proposed_end..unit_end];
        let through_head = &source.text[source.bytes[clause.clause.start_char]..proposed_end];
        temporal_endpoint(
            source,
            clause,
            &self.units[unit],
            &self.scopes[unit],
            self.frames[unit].as_ref(),
        ) && !inside_lexical_term(source, lexicon, clause.clause, proposed)
            && !super::contains_any(
                through_head,
                &[
                    "假裝", "假装", "佯裝", "佯装", "裝作", "装作", "謊稱", "谎称", "假稱", "假称",
                ],
            )
            && !nominalized_tail(tail)
    }

    fn verified_endpoint(
        &self,
        unit: usize,
        index: usize,
        anchor: Anchor,
    ) -> Result<ClauseOrderEndpoint, ClauseOrderAdvisory> {
        let clause = &self.clauses[index];
        let head = self
            .verified_heads
            .get(&clause.input_index)
            .copied()
            .flatten()
            .ok_or(ClauseOrderAdvisory::HeadNotRecognized)?;
        let source = self.source;
        let lexicon = self.lexicon;
        let start = source.bytes[head.start_char];
        let end = source.bytes[head.end_char];
        let unit_end = source.bytes[clause.clause.end_char];
        let evidence_end = match anchor {
            Anchor::End { end, .. } | Anchor::Head { end, .. } => end,
            Anchor::Start(_) => unit_end,
        };
        if end > evidence_end || matches!(anchor, Anchor::Start(from) if from > start) {
            return Err(ClauseOrderAdvisory::OutsideAnchor);
        }
        if !self.allowed_endpoint(unit, clause)
            || inside_lexical_term(source, lexicon, clause.clause, head)
        {
            return Err(ClauseOrderAdvisory::ScopeOrLexicalGuard);
        }
        let tail = &source.text[end..evidence_end];
        if tail.trim_start().starts_with('的') || nominalized_tail(tail) {
            return Err(ClauseOrderAdvisory::ScopeOrLexicalGuard);
        }
        let earlier_end =
            matches!(anchor, Anchor::End { attached, .. } if attached != Some(Before));
        let bridge = |from| {
            if from > start {
                return false;
            }
            let subject = self
                .verified_subjects
                .get(&clause.input_index)
                .and_then(|subject| *subject)
                .and_then(|subject| {
                    let lead = source.trim(from, start)?;
                    let text = source.slice(&subject);
                    (lead.start_char == subject.start_char
                        && contained(subject, lead)
                        && !scene_transition(text)
                        && temporal_shift(text).is_none())
                    .then_some(subject)
                });
            temporal_bridge(
                source,
                lexicon,
                from,
                start,
                subject.as_ref().map(|span| source.slice(span)),
            )
            .is_some()
                || lexical_argument(source, lexicon, from, start).is_some()
                || (earlier_end && earlier_modifier_bridge(source, lexicon, from, start, subject))
                || subject.is_some_and(|span| {
                    // An ungrounded 的 tail could put the verified verb inside a modifier.
                    single_action_bridge(source, from, evidence_end, head, span)
                        && (!tail.contains('的')
                            || grounded_action_tail(
                                source,
                                lexicon,
                                ClauseCandidate::new(head),
                                end + trim_clause_end(tail).len(),
                                true,
                            ))
                })
        };
        let placed = match anchor {
            Anchor::End { attached, .. } => {
                let lead = source.bytes[clause.clause.start_char];
                let opening = source_lead(source, &self.units[unit], start);
                let framed = attached.is_some()
                    && source.text[opening..start]
                        .strip_prefix(['在', '於', '于'])
                        .is_some_and(|rest| bridge(start - rest.len()));
                bridge(lead) || clause.actor_evidence.is_some() || framed
            }
            Anchor::Start(from) => bridge(from),
            Anchor::Head { written, .. } => head == written_head(source, written),
        };
        if !placed {
            return Err(if matches!(anchor, Anchor::Head { .. }) {
                ClauseOrderAdvisory::OutsideAnchor
            } else {
                ClauseOrderAdvisory::SubjectBridge
            });
        }
        if !matches!(anchor, Anchor::Head { .. })
            && !grounded_action_tail(
                source,
                lexicon,
                ClauseCandidate::new(head),
                evidence_end,
                true,
            )
            && !simple_action_tail(trim_clause_end(tail).trim(), lexicon)
        {
            return Err(ClauseOrderAdvisory::Tail);
        }
        let (completion, argument) = head_evidence(source, lexicon, clause, head, evidence_end);
        Ok(ClauseOrderEndpoint {
            clause: index,
            head,
            completion,
            argument,
            lexical_head: false,
            eligible: false,
        })
    }

    fn recognized_endpoint(
        &self,
        unit: usize,
        index: usize,
        anchor: Anchor,
    ) -> Result<(ClauseOrderEndpoint, Option<ClauseOrderBinding>), ClauseOrderAdvisory> {
        let source = self.source;
        let lexicon = self.lexicon;
        let clause = &self.clauses[index];
        let proposed = clause.candidate.predicate;
        let proposed_start = source.bytes[proposed.start_char];
        let proposed_end = source.bytes[proposed.end_char];
        let unit_end = source.bytes[clause.clause.end_char];
        if !self.allowed_endpoint(unit, clause) {
            return Err(ClauseOrderAdvisory::ScopeOrLexicalGuard);
        }
        let evidence_end = match anchor {
            Anchor::End { end, .. } | Anchor::Head { end, .. } => end,
            Anchor::Start(_) => unit_end,
        };
        if proposed_end > evidence_end {
            return Err(ClauseOrderAdvisory::OutsideAnchor);
        }
        let text = source.slice(&proposed);
        let exact = known_action_head(source, lexicon, clause);
        let (head, lexical_head) = if exact {
            (written_head(source, proposed), true)
        } else if let Some((_, offset, word)) = predicate(text, lexicon) {
            if offset != 0 {
                return Err(ClauseOrderAdvisory::HeadNotRecognized);
            }
            let head = source.span(proposed_start, proposed_start + word.len());
            if !grounded_action_tail(
                source,
                lexicon,
                ClauseCandidate::new(head),
                proposed_end,
                true,
            ) || inside_lexical_term(source, lexicon, clause.clause, head)
            {
                return Err(ClauseOrderAdvisory::HeadNotRecognized);
            }
            (written_head(source, head), true)
        } else {
            let lexical_head = predicate(&source.text[proposed_start..evidence_end], lexicon)
                .is_some_and(|(_, offset, word)| {
                    offset == 0
                        && word.len() > text.len()
                        && written_head(
                            source,
                            source.span(proposed_start, proposed_start + word.len()),
                        ) == proposed
                });
            (proposed, lexical_head)
        };
        let (completion, argument) = head_evidence(source, lexicon, clause, head, evidence_end);
        if !lexical_head
            && (completion.is_none()
                || argument.is_none()
                || clause.actor_evidence.is_none()
                || text.chars().count() > 4
                || !text.chars().all(char::is_alphabetic)
                || text.starts_with(['是', '有'])
                || text
                    .chars()
                    .any(|ch| "零一二三四五六七八九十百千萬万兩两歲岁".contains(ch)))
        {
            return Err(ClauseOrderAdvisory::HeadNotRecognized);
        }
        let blocked = match anchor {
            Anchor::End { end, attached } => {
                let strict_grounded =
                    exact && grounded_action_tail(source, lexicon, clause.candidate, end, true);
                let grounded = strict_grounded
                    || grounded_action_tail(source, lexicon, ClauseCandidate::new(head), end, true);
                let lead = source.bytes[clause.clause.start_char];
                let main_head = clause.actor_evidence.is_some()
                    || head.start_char == clause.clause.start_char
                    || (lexical_head
                        && grounded
                        && (temporal_bridge(source, lexicon, lead, proposed_start, None)
                            .is_some()
                            || lexical_argument(source, lexicon, lead, proposed_start).is_some()));
                let tail = trim_clause_end(&source.text[source.bytes[head.end_char]..end]).trim();
                if !main_head {
                    return Err(ClauseOrderAdvisory::SubjectBridge);
                }
                if !grounded && !simple_action_tail(tail, lexicon) {
                    return Err(ClauseOrderAdvisory::Tail);
                }
                if !exact {
                    Some(ClauseOrderBinding::AdvisoryOnly)
                } else if !strict_grounded {
                    Some(ClauseOrderBinding::UngroundedTail)
                } else if !written_prior_head(source, lexicon, clause, false) {
                    Some(ClauseOrderBinding::SubjectBridge)
                } else if attached == Some(Before)
                    && !completion_after_head(source, lexicon, head, end)
                {
                    Some(ClauseOrderBinding::MissingCompletion)
                } else {
                    None
                }
            }
            Anchor::Start(start) => {
                if start > proposed_start {
                    return Err(ClauseOrderAdvisory::OutsideAnchor);
                }
                let direct =
                    temporal_bridge(source, lexicon, start, proposed_start, None).is_some();
                if !direct
                    && (!lexical_head
                        || lexical_argument(source, lexicon, start, proposed_start).is_none())
                {
                    return Err(ClauseOrderAdvisory::SubjectBridge);
                }
                if !exact {
                    Some(ClauseOrderBinding::AdvisoryOnly)
                } else if !direct {
                    Some(ClauseOrderBinding::SubjectBridge)
                } else {
                    None
                }
            }
            Anchor::Head {
                written, blocked, ..
            } => {
                if head != written_head(source, written) {
                    return Err(ClauseOrderAdvisory::OutsideAnchor);
                }
                if exact && proposed == written {
                    blocked
                } else {
                    Some(ClauseOrderBinding::AdvisoryOnly)
                }
            }
        };
        Ok((
            ClauseOrderEndpoint {
                clause: index,
                head,
                completion,
                argument,
                lexical_head,
                eligible: blocked.is_none(),
            },
            blocked,
        ))
    }

    fn select(&self, unit: usize, anchor: Anchor) -> Endpoints {
        let Some(indices) = self.indices.get(unit) else {
            return Endpoints::default();
        };
        let mut alternatives: Vec<_> = indices
            .iter()
            .filter_map(|&index| {
                self.recognized_endpoint(unit, index, anchor)
                    .ok()
                    .map(|(option, _)| option)
            })
            .collect();
        if !self.verified_heads.is_empty() {
            for &index in indices {
                if let Ok(option) = self.verified_endpoint(unit, index, anchor)
                    && !alternatives
                        .iter()
                        .any(|old| old.clause == index && old.head == option.head)
                {
                    alternatives.push(option);
                }
            }
        }
        let mut eligible = alternatives.iter().filter(|option| option.eligible);
        let resolved = eligible
            .next()
            .filter(|_| eligible.next().is_none())
            .map(|option| option.clause);
        alternatives.sort_by_key(|option| {
            use std::cmp::Reverse;
            (
                Reverse(option.eligible),
                Reverse(option.lexical_head),
                Reverse(option.completion.is_some()),
                Reverse(option.argument.is_some()),
                Reverse(option.head == self.clauses[option.clause].candidate.predicate),
                option.head.start_char,
                option.head.end_char,
                option.clause,
            )
        });
        alternatives.truncate(5);
        Endpoints {
            resolved,
            alternatives,
        }
    }

    fn adjacent(&self, left: usize, right: usize) -> bool {
        let (Some(left), Some(right)) = (self.units.get(left), self.units.get(right)) else {
            return false;
        };
        let span = Span {
            start_char: left.evidence.end_char,
            end_char: right.evidence.start_char,
        };
        if right.first_in_line {
            local_line_gap(self.source, left.evidence, right.evidence)
        } else {
            self.source.slice(&span).chars().all(char::is_whitespace)
        }
    }

    fn same_line_adjacent(&self, left: usize, right: usize) -> bool {
        let right_unit = &self.units[right];
        let span = Span {
            start_char: self.units[left].evidence.start_char,
            end_char: right_unit.evidence.end_char,
        };
        !right_unit.first_in_line
            && self.adjacent(left, right)
            && !contains_line_break(self.source.slice(&span))
    }

    fn prior(&self, owner: usize) -> Option<Search> {
        let left = owner.checked_sub(1)?;
        if !self.adjacent(left, owner) {
            return None;
        }
        self.at_unit_end(left)
    }

    fn at_unit_end(&self, left: usize) -> Option<Search> {
        let clause = self.units[left].evidence;
        let text = self.source.slice(&clause);
        if !text.ends_with(['。', '.', '，', ',', '；', ';']) {
            return None;
        }
        Some(Search {
            unit: left,
            anchor: Anchor::End {
                end: self.source.bytes[clause.start_char] + trim_clause_end(text).len(),
                attached: None,
            },
        })
    }

    fn prior_across_description(&self, owner: usize, cue: WrittenCue) -> Option<Search> {
        let earlier = owner.checked_sub(2)?;
        let gap = owner - 1;
        let source = self.source;
        let middle = &self.units[gap];
        let current = &self.units[owner];
        let middle_scope = &self.scopes[gap];
        if !self.same_line_adjacent(earlier, gap)
            || !self.same_line_adjacent(gap, owner)
            || !source
                .slice(&self.units[earlier].evidence)
                .ends_with(DESCRIPTION_DELIMITERS)
            || !static_scene_description(source.slice(&middle.evidence))
            || temporal_boundary(source, middle, middle_scope)
            || temporal_boundary(source, current, &self.scopes[owner])
            || middle_scope.condition.is_some()
            || middle_scope.dream.is_some()
            || middle_scope.report.is_some()
            || !middle_scope.wakes.is_empty()
            || !middle_scope.cues.is_empty()
            || self.frames[gap].is_some()
            || self.indices[gap].iter().any(|&index| {
                let clause = &self.clauses[index];
                self.verified_heads
                    .get(&clause.input_index)
                    .is_some_and(Option::is_some)
                    || known_action_head(source, self.lexicon, clause)
            })
        {
            return None;
        }
        let following = if matches!(cue.placement, Placement::Detached) {
            self.units
                .get(owner + 1)
                .map_or("", |unit| source.slice(&unit.evidence))
        } else {
            source.slice(&Span {
                start_char: cue.span.end_char,
                end_char: current.evidence.end_char,
            })
        }
        .trim_start();
        if scene_transition(following) || temporal_shift(following).is_some() {
            return None;
        }
        self.at_unit_end(earlier)
    }

    fn subject_before_sequence(&self, owner: usize, start: usize, after: &Endpoints) -> bool {
        let unit = &self.units[owner];
        if self.verified_subjects.is_empty()
            || !owner
                .checked_sub(1)
                .is_some_and(|left| self.same_line_adjacent(left, owner))
        {
            return false;
        }
        let source = self.source;
        let Some(prefix) = source.trim(source.bytes[unit.evidence.start_char], start) else {
            return false;
        };
        let text = source.slice(&prefix);
        !scene_transition(text)
            && temporal_shift(text).is_none()
            && !text.contains('的')
            && predicate(text, self.lexicon).is_none()
            && after.alternatives.iter().any(|option| {
                let input = self.clauses[option.clause].input_index;
                self.verified_subjects.get(&input) == Some(&Some(prefix))
                    && self.verified_heads.get(&input) == Some(&Some(option.head))
            })
    }

    fn following(&self, owner: usize, detached: bool) -> Option<Search> {
        let right = owner + 1;
        if !self.adjacent(owner, right)
            || (detached && self.units[right].sentence != self.units[owner].sentence)
            || (!detached
                && !self
                    .source
                    .slice(&self.units[owner].evidence)
                    .ends_with(['，', ',', '；', ';']))
        {
            return None;
        }
        Some(Search {
            unit: right,
            anchor: Anchor::Start(self.source.bytes[self.units[right].evidence.start_char]),
        })
    }

    fn bind(
        &self,
        owner: usize,
        cue: WrittenCue,
        record_trace: impl FnOnce(CueTrace),
    ) -> RankedClauseOrderCue {
        let start = self.source.bytes[cue.span.start_char];
        let end = self.source.bytes[cue.span.end_char];
        // Scope the marker at its end so negation inside 不知過了多久 is not a clause operator.
        let scope_marker = ClauseCandidate::new(Span {
            start_char: cue.span.end_char - 1,
            end_char: cue.span.end_char,
        });
        let mut scope = candidate_scope(self.source, &self.scopes[owner], &scope_marker).0;
        if scope == ClauseScope::Narrated
            && let Placement::Paired { first, .. } = cue.placement
        {
            scope = candidate_scope(
                self.source,
                &self.scopes[owner - 1],
                &ClauseCandidate::new(first),
            )
            .0;
        }
        let search = |unit, anchor| Some(Search { unit, anchor });
        let (before_search, after_search) =
            if scope != ClauseScope::Narrated || cue.kind == Reference {
                (None, None)
            } else {
                match cue.placement {
                    Placement::Leading => (self.prior(owner), search(owner, Anchor::Start(end))),
                    Placement::Interior => (
                        search(
                            owner,
                            Anchor::End {
                                end: start,
                                attached: None,
                            },
                        ),
                        search(owner, Anchor::Start(end)),
                    ),
                    Placement::Detached => (self.prior(owner), self.following(owner, true)),
                    Placement::Attached => {
                        let anchor = search(
                            owner,
                            Anchor::End {
                                end: start,
                                attached: Some(cue.kind),
                            },
                        );
                        let following = self.following(owner, false);
                        if cue.kind == Before {
                            (following, anchor)
                        } else {
                            (anchor, following)
                        }
                    }
                    Placement::EventBound {
                        head,
                        grounded,
                        trailing,
                        ..
                    } => {
                        let blocked = if !grounded {
                            Some(ClauseOrderBinding::UngroundedTail)
                        } else if cue.kind == Before
                            && !completion_after_head(self.source, self.lexicon, head, start)
                        {
                            Some(ClauseOrderBinding::MissingCompletion)
                        } else {
                            None
                        };
                        let anchor = search(
                            owner,
                            Anchor::Head {
                                written: head,
                                end: start,
                                blocked,
                            },
                        );
                        let following = if trailing {
                            self.following(owner, false)
                        } else {
                            search(owner, Anchor::Start(end))
                        };
                        if cue.kind == Before {
                            (following, anchor)
                        } else {
                            (anchor, following)
                        }
                    }
                    Placement::Paired { heads, blocked, .. } => (
                        search(
                            owner - 1,
                            Anchor::Head {
                                written: heads[0],
                                end: self.source.bytes[self.units[owner - 1].evidence.end_char],
                                blocked: blocked[0],
                            },
                        ),
                        search(
                            owner,
                            Anchor::Head {
                                written: heads[1],
                                end: self.source.bytes[self.units[owner].evidence.end_char],
                                blocked: blocked[1],
                            },
                        ),
                    ),
                }
            };
        let select = |search: Search| self.select(search.unit, search.anchor);
        let mut before = before_search.map(select).unwrap_or_default();
        let after = after_search.map(select).unwrap_or_default();
        let mut earlier_fallback = None;
        if scope == ClauseScope::Narrated && cue.kind == Sequence && before.alternatives.is_empty()
        {
            earlier_fallback = match cue.placement {
                Placement::Leading | Placement::Detached => {
                    self.prior_across_description(owner, cue)
                }
                Placement::Interior if self.subject_before_sequence(owner, start, &after) => {
                    self.prior(owner)
                }
                _ => None,
            };
            if let Some(search) = earlier_fallback {
                before.alternatives = select(search).into_advisory();
            }
        }
        record_trace(CueTrace {
            before: before_search,
            after: after_search,
            earlier_fallback,
            after_resolved: after.resolved,
        });
        let (paired_cue, construction_start) = match cue.placement {
            Placement::Paired { first, .. } => (Some(first), Some(first.start_char)),
            Placement::EventBound { opening, head, .. } => {
                (opening, Some(opening.unwrap_or(head).start_char))
            }
            _ => (None, None),
        };
        let order = ClauseOrderCue {
            kind: cue.kind,
            cue: cue.span,
            paired_cue,
            construction: construction_start.map(|start_char| Span {
                start_char,
                end_char: cue.span.end_char,
            }),
            scope,
            before: before.resolved,
            after: after
                .resolved
                .filter(|&after| Some(after) != before.resolved),
        };
        RankedClauseOrderCue {
            order,
            before_alternatives: before.alternatives,
            after_alternatives: after.alternatives,
        }
    }

    fn diagnostic_window(&self, search: Search, advisory_only: bool) -> ClauseOrderWindow {
        let unit = self.units[search.unit].evidence;
        let source = self.source;
        let (window, written_head) = match search.anchor {
            Anchor::Start(start) => (source.span(start, source.bytes[unit.end_char]), None),
            Anchor::End { end, .. } => (source.span(source.bytes[unit.start_char], end), None),
            Anchor::Head { written, end, .. } => (
                source.span(source.bytes[written.start_char], end),
                Some(written),
            ),
        };
        ClauseOrderWindow {
            unit,
            window,
            written_head,
            required_head: written_head.map(|head| self::written_head(source, head)),
            advisory_only,
        }
    }

    fn explain(
        &self,
        cue: &RankedClauseOrderCue,
        trace: &CueTrace,
        diagnostic: &mut ClauseOrderDiagnostic,
    ) {
        let (searches, options, resolved) = match diagnostic.probe.side {
            ClauseOrderSide::Before => (
                [
                    trace.before.map(|search| (search, false)),
                    trace.earlier_fallback.map(|search| (search, true)),
                ],
                &cue.before_alternatives,
                cue.order.before,
            ),
            ClauseOrderSide::After => (
                [trace.after.map(|search| (search, false)), None],
                &cue.after_alternatives,
                cue.order.after,
            ),
        };
        diagnostic.windows = searches
            .into_iter()
            .flatten()
            .map(|(search, advisory_only)| self.diagnostic_window(search, advisory_only))
            .collect();
        let Some(index) = diagnostic.accepted_clause else {
            return;
        };
        if cue.order.scope != ClauseScope::Narrated {
            diagnostic.advisory = ClauseOrderAdvisory::NonNarrated;
            return;
        }
        if cue.order.kind == Reference {
            diagnostic.advisory = ClauseOrderAdvisory::ReferenceOnly;
            return;
        }
        let clause = &self.clauses[index];
        let Some((search, advisory_only)) = searches
            .into_iter()
            .flatten()
            .find(|(search, _)| self.units[search.unit].evidence == clause.clause)
        else {
            diagnostic.advisory = if diagnostic.windows.is_empty() {
                ClauseOrderAdvisory::NoWindow
            } else {
                ClauseOrderAdvisory::OutsideUnit
            };
            return;
        };
        // Reuse the selector checks for this one requested candidate. This also sees
        // candidates below the display limit, including strict competitors.
        let blocked = match self.recognized_endpoint(search.unit, index, search.anchor) {
            Ok((_, blocked)) => blocked,
            Err(builtin_reason) => {
                let verified = self
                    .verified_heads
                    .get(&clause.input_index)
                    .is_some_and(Option::is_some);
                let assessment = if verified {
                    self.verified_endpoint(search.unit, index, search.anchor)
                } else {
                    Err(builtin_reason)
                };
                match assessment {
                    Ok(_) => Some(ClauseOrderBinding::AdvisoryOnly),
                    Err(reason) => {
                        diagnostic.advisory = reason;
                        return;
                    }
                }
            }
        };
        diagnostic.advisory = ClauseOrderAdvisory::Admitted;
        diagnostic.rank = options.iter().position(|option| option.clause == index);
        diagnostic.binding = if advisory_only {
            ClauseOrderBinding::AdvisoryOnly
        } else if let Some(blocked) = blocked {
            blocked
        } else if resolved == Some(index) {
            ClauseOrderBinding::Bound
        } else if diagnostic.probe.side == ClauseOrderSide::After
            && trace.after_resolved == Some(index)
        {
            ClauseOrderBinding::SameClause
        } else {
            ClauseOrderBinding::Competing
        };
    }
}

pub(super) struct DiagnosticInput<'a> {
    pub(super) evidence: &'a [VerifiedClauseEvidence],
    pub(super) probes: &'a [ClauseOrderProbe],
    pub(super) candidate_count: usize,
}

struct Diagnostics {
    by_cue: HashMap<(usize, usize), Vec<usize>>,
    reports: Vec<ClauseOrderDiagnostic>,
}

impl Diagnostics {
    fn new(input: &DiagnosticInput<'_>, clauses: &[ScopedClause]) -> Self {
        let mut by_cue = HashMap::<_, Vec<_>>::new();
        let reports = input
            .probes
            .iter()
            .copied()
            .enumerate()
            .map(|(index, probe)| {
                by_cue
                    .entry((probe.cue.start_char, probe.cue.end_char))
                    .or_default()
                    .push(index);
                let accepted_clause = clauses
                    .binary_search_by_key(&probe.input_index, |clause| clause.input_index)
                    .ok();
                let advisory = if probe.input_index >= input.candidate_count {
                    ClauseOrderAdvisory::UnknownCandidate
                } else if accepted_clause.is_none() {
                    ClauseOrderAdvisory::CandidateRejected
                } else {
                    ClauseOrderAdvisory::CueNotFound
                };
                ClauseOrderDiagnostic {
                    probe,
                    accepted_clause,
                    windows: Vec::new(),
                    advisory,
                    rank: None,
                    binding: ClauseOrderBinding::NotEvaluated,
                }
            })
            .collect();
        Self { by_cue, reports }
    }

    fn bind(
        &mut self,
        resolver: &Resolver<'_, '_>,
        owner: usize,
        cue: WrittenCue,
    ) -> RankedClauseOrderCue {
        let key = (cue.span.start_char, cue.span.end_char);
        let Some(indices) = self.by_cue.get(&key) else {
            return resolver.bind(owner, cue, |_| {});
        };
        let mut trace = CueTrace::default();
        let ranked = resolver.bind(owner, cue, |observed| trace = observed);
        for &index in indices {
            resolver.explain(&ranked, &trace, &mut self.reports[index]);
        }
        ranked
    }
}

fn record_boundary(boundaries: &mut HashMap<usize, Option<Span>>, input_index: usize, span: Span) {
    boundaries
        .entry(input_index)
        .and_modify(|previous| {
            if *previous != Some(span) {
                *previous = None;
            }
        })
        .or_insert(Some(span));
}

pub(super) fn resolve<'a>(
    source: &Source<'_>,
    lexicon: &'a Lexicon<'a>,
    units: &[Unit],
    scopes: &[UnitScope],
    frames: &[Option<DisposalFrame<'a>>],
    clauses: &[ScopedClause],
    evidence: &[VerifiedClauseEvidence],
) -> Vec<RankedClauseOrderCue> {
    resolve_inner::<false>(
        source,
        lexicon,
        units,
        scopes,
        frames,
        clauses,
        DiagnosticInput {
            evidence,
            probes: &[],
            candidate_count: 0,
        },
    )
    .0
}

pub(super) fn resolve_diagnosed<'a>(
    source: &Source<'_>,
    lexicon: &'a Lexicon<'a>,
    units: &[Unit],
    scopes: &[UnitScope],
    frames: &[Option<DisposalFrame<'a>>],
    clauses: &[ScopedClause],
    input: DiagnosticInput<'_>,
) -> (Vec<RankedClauseOrderCue>, Vec<ClauseOrderDiagnostic>) {
    if input.probes.is_empty() {
        return (
            resolve(
                source,
                lexicon,
                units,
                scopes,
                frames,
                clauses,
                input.evidence,
            ),
            Vec::new(),
        );
    }
    resolve_inner::<true>(source, lexicon, units, scopes, frames, clauses, input)
}

fn resolve_inner<'a, const DIAGNOSE: bool>(
    source: &Source<'_>,
    lexicon: &'a Lexicon<'a>,
    units: &[Unit],
    scopes: &[UnitScope],
    frames: &[Option<DisposalFrame<'a>>],
    clauses: &[ScopedClause],
    input: DiagnosticInput<'_>,
) -> (Vec<RankedClauseOrderCue>, Vec<ClauseOrderDiagnostic>) {
    let mut indices = vec![Vec::new(); units.len()];
    for (index, clause) in clauses.iter().enumerate() {
        let unit = units.partition_point(|unit| unit.evidence.end_char <= clause.clause.start_char);
        indices[unit].push(index);
    }
    let mut verified_heads = HashMap::new();
    let mut verified_subjects = HashMap::new();
    for &VerifiedClauseEvidence {
        head: VerifiedClauseHead { input_index, head },
        subject,
    } in input.evidence
    {
        let Ok(index) = clauses.binary_search_by_key(&input_index, |clause| clause.input_index)
        else {
            continue;
        };
        let clause = &clauses[index];
        let proposed = clause.candidate.predicate;
        if head.start_char == proposed.start_char
            && contained(head, proposed)
            && source.slice(&head).chars().all(char::is_alphabetic)
        {
            record_boundary(&mut verified_heads, input_index, head);
            if let Some(subject) = subject
                && subject.end_char <= head.start_char
                && contained(subject, clause.clause)
                && actor_compatible(clause.candidate, subject)
                && source.slice(&subject).trim() == source.slice(&subject)
                && source
                    .slice(&subject)
                    .chars()
                    .all(|ch| ch.is_alphabetic() || ch.is_whitespace())
            {
                record_boundary(&mut verified_subjects, input_index, subject);
            }
        }
    }
    let resolver = Resolver {
        source,
        lexicon,
        units,
        scopes,
        frames,
        clauses,
        indices,
        verified_heads,
        verified_subjects,
    };
    let mut diagnostics = DIAGNOSE.then(|| Diagnostics::new(&input, clauses));
    let mut bind = |owner, cue| match diagnostics.as_mut() {
        Some(diagnostics) => diagnostics.bind(&resolver, owner, cue),
        None => resolver.bind(owner, cue, |_| {}),
    };
    let mut result = Vec::new();
    for (owner, unit) in units.iter().enumerate() {
        result.extend(
            written_cues(source, lexicon, unit, &scopes[owner])
                .into_iter()
                .map(|cue| bind(owner, cue)),
        );
        if owner > 0
            && let Some(cue) = written_pair(source, lexicon, &units[owner - 1], unit)
        {
            result.push(bind(owner, cue));
        }
    }
    result.sort_by_key(|cue| cue.order.cue.start_char);
    (
        result,
        diagnostics
            .map(|diagnostics| diagnostics.reports)
            .unwrap_or_default(),
    )
}
