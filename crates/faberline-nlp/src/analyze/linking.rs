use super::{
    DREAM, DisposalFrame, HYPOTHETICAL, NarrativeAnalyzer, PLANNED, REPORTED, Source, Span,
    THOUGHT, Unit, contains_any, continuation, disposal_clause_frame, earliest,
    ends_with_question_particle, nominal_measure, participant_description, predicate, quote_end,
    split_units, starts_any, temporal_shift, track_quote, trim_clause_end,
};

mod cues;
mod observation;
mod reading;

pub use cues::{
    ClauseOrderAdvisory, ClauseOrderBinding, ClauseOrderCue, ClauseOrderCueKind,
    ClauseOrderDiagnostic, ClauseOrderEndpoint, ClauseOrderProbe, ClauseOrderSide,
    ClauseOrderWindow, RankedClauseOrderCue, VerifiedClauseEvidence, VerifiedClauseHead,
};
pub use observation::DurationObservation;
pub(super) use observation::discover as discover_duration_observations;
pub use reading::ReadingAction;
pub(super) use reading::discover as discover_reading_actions;

pub(super) enum TemporalOutput<'a> {
    Precedence(&'a mut Vec<ClausePrecedence>),
    Cues(&'a mut Vec<ClauseOrderCue>),
    RankedCues(
        &'a mut Vec<RankedClauseOrderCue>,
        &'a [VerifiedClauseEvidence],
    ),
    DiagnosedCues {
        cues: &'a mut Vec<RankedClauseOrderCue>,
        diagnostics: &'a mut Vec<ClauseOrderDiagnostic>,
        evidence: &'a [VerifiedClauseEvidence],
        probes: &'a [ClauseOrderProbe],
    },
}

/// A proposed predicate and optional role ranges in the original chapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClauseCandidate {
    pub predicate: Span,
    pub actor: Option<Span>,
    pub theme: Option<Span>,
    pub patient: Option<Span>,
    /// A proposed result phrase or a proper terminal suffix of the predicate.
    pub result: Option<Span>,
}

impl ClauseCandidate {
    /// Make a candidate with only a predicate range.
    pub fn new(predicate: Span) -> Self {
        Self {
            predicate,
            actor: None,
            theme: None,
            patient: None,
            result: None,
        }
    }
}

/// The local scope of a candidate. Narrated does not mean the event is proven.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClauseScope {
    Narrated,
    Quoted,
    Conditional,
    Planned,
    Negated,
    Reported,
    Dream,
}

/// A candidate anchored to a retained clause. Its input roles remain proposals.
#[derive(Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScopedClause {
    pub input_index: usize,
    pub clause: Span,
    pub candidate: ClauseCandidate,
    /// A locally anchored name, pronoun, or participant description.
    pub actor_evidence: Option<Span>,
    /// The patient in an aligned disposal construction.
    pub patient_evidence: Option<Span>,
    /// A written result complement within the predicate, such as 開 in 推開.
    pub result_complement: Option<Span>,
    /// A result written in this same clause.
    pub result_evidence: Option<Span>,
    pub scope: ClauseScope,
    pub scope_evidence: Option<Span>,
}

/// Source evidence for an explicit result construction between two candidates.
#[derive(Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClauseResultLink {
    /// Index into the accepted clauses for the action.
    pub action: usize,
    /// Index into the accepted clauses for the result.
    pub result_clause: usize,
    /// False for a same-sentence result without a causal connective.
    pub explicit_cause: bool,
    /// The second patient span refers back through a demonstrative or pronoun.
    pub patient_reference: bool,
    pub cue: Span,
    pub patient_before: Span,
    pub patient_after: Span,
    pub result: Option<Span>,
}

/// Written temporal precedence between two accepted clause candidates.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClausePrecedence {
    /// Index into the accepted clauses for the action that comes first.
    pub before: usize,
    /// Index into the accepted clauses for the action that comes later.
    pub after: usize,
    /// Main cue in the original chapter.
    pub cue: Span,
    /// The earlier part of a paired cue, such as 先 in 先 … 再.
    pub paired_cue: Option<Span>,
}

/// A batch of anchored candidates. All ranges count Unicode scalar values.
#[derive(Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LinkedClauses {
    /// Accepted candidates in input order.
    pub clauses: Vec<ScopedClause>,
    /// Input indices with ranges outside a retained clause or overlapping roles.
    pub rejected: Vec<usize>,
    pub results: Vec<ClauseResultLink>,
}

fn overlaps(left: Span, right: Span) -> bool {
    left.start_char < right.end_char && right.start_char < left.end_char
}

fn contained(span: Span, clause: Span) -> bool {
    span.start_char < span.end_char
        && clause.start_char <= span.start_char
        && span.end_char <= clause.end_char
}

fn predicate_suffix(result: Span, predicate: Span) -> bool {
    predicate.start_char < result.start_char
        && result.start_char < result.end_char
        && result.end_char == predicate.end_char
}

fn valid_candidate(source: &Source<'_>, candidate: ClauseCandidate, unit: &Unit) -> bool {
    let ranges = [
        Some(candidate.predicate),
        candidate.actor,
        candidate.theme,
        candidate.patient,
        candidate
            .result
            .filter(|&span| !predicate_suffix(span, candidate.predicate)),
    ];
    ranges.iter().enumerate().all(|(index, range)| {
        range.is_none_or(|span| {
            contained(span, unit.evidence)
                && ranges[..index]
                    .iter()
                    .flatten()
                    .all(|&earlier| !overlaps(span, earlier))
        })
    }) && {
        let predicate = source.slice(&candidate.predicate);
        predicate.chars().any(char::is_alphanumeric)
            && !predicate.chars().any(|ch| quote_end(ch).is_some())
    }
}

fn written_complement(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    clause: Span,
    predicate: Span,
) -> Option<Span> {
    let head = source.slice(&predicate);
    let mut chars = head.chars();
    let stem = chars.next()?;
    let complement = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    let productive = match complement {
        '開' | '开' => "打推拉撬揭撕割扯抖掰".contains(stem),
        '上' => "關关合蓋盖鎖锁扣".contains(stem),
        '破' | '碎' | '裂' => "打敲砸撞擊击踢壓压摔捏".contains(stem),
        '斷' | '断' => "折扯掰撕切剪砍割斬斩".contains(stem),
        '倒' => "推撞絆绊摔吹打".contains(stem),
        '成' => "燒烧煉炼壓压磨捏揉擠挤雕塑擊击剪裁折敲打裂斷断變变化凝融".contains(stem),
        '為' | '为' => stem == '化',
        '出' => "問问查找辨認认".contains(stem),
        _ => false,
    };
    if !productive {
        return None;
    }
    (!inside_lexical_term(source, lexicon, clause, predicate)).then_some(Span {
        start_char: predicate.start_char + 1,
        end_char: predicate.end_char,
    })
}

fn inside_lexical_term(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    clause: Span,
    anchor: Span,
) -> bool {
    let text = source.slice(&clause);
    let head = source.slice(&anchor);
    let start = source.bytes[anchor.start_char] - source.bytes[clause.start_char];
    lexicon
        .objects
        .iter()
        .chain(lexicon.actors)
        .chain(lexicon.ranks)
        .map(|&(word, _)| word)
        .chain(lexicon.places.iter().copied())
        .filter(|word| word.contains(head))
        .any(|word| {
            text.match_indices(word)
                .any(|(offset, _)| offset <= start && start + head.len() <= offset + word.len())
        })
}

#[derive(Clone, Copy)]
enum UnitCandidates {
    None,
    One(usize),
    Multiple,
}

pub(super) fn link(
    analyzer: &NarrativeAnalyzer<'_>,
    chapter_text: &str,
    title: &str,
    candidates: &[ClauseCandidate],
    temporal: Option<TemporalOutput<'_>>,
) -> LinkedClauses {
    if candidates.is_empty()
        && !matches!(
            temporal,
            Some(
                TemporalOutput::Cues(_)
                    | TemporalOutput::RankedCues(..)
                    | TemporalOutput::DiagnosedCues { .. }
            )
        )
    {
        return LinkedClauses::default();
    }
    let mut source = Source::new(chapter_text);
    let units = split_units(&mut source, title, &analyzer.lexicon, Some(analyzer));
    let scopes = scopes_for_units(&source, &units, false);
    let frames: Vec<_> = units
        .iter()
        .enumerate()
        .map(|(index, unit)| {
            disposal_clause_frame(&source, unit, units.get(index + 1), &analyzer.lexicon)
        })
        .collect();
    let mut result = LinkedClauses::default();
    let mut by_unit = vec![UnitCandidates::None; units.len()];
    for (input_index, &candidate) in candidates.iter().enumerate() {
        let index =
            units.partition_point(|unit| unit.evidence.end_char <= candidate.predicate.start_char);
        let Some(unit) = units
            .get(index)
            .filter(|unit| valid_candidate(&source, candidate, unit))
        else {
            result.rejected.push(input_index);
            continue;
        };
        let frame = frames[index]
            .as_ref()
            .filter(|frame| frame.predicate == candidate.predicate);
        let local = if frame.is_none() {
            local_result(&source, &analyzer.lexicon, unit, candidate)
        } else {
            None
        };
        let parsed_patient = frame
            .map(|frame| frame.patient)
            .or(local.and_then(|local| local.disposal.map(|(_, patient)| patient)));
        let patient_evidence =
            parsed_patient.filter(|&patient| compatible_patient(&candidate, patient));
        let scoped_candidate = ClauseCandidate {
            patient: parsed_patient.or(candidate.patient),
            ..candidate
        };
        let (scope, scope_evidence) = candidate_scope(&source, &scopes[index], &scoped_candidate);
        let actor_evidence = if scope == ClauseScope::Quoted {
            None
        } else if let Some(frame) = frame {
            frame.actor_surface
        } else {
            let head = local
                .and_then(|local| local.disposal.map(|(marker, _)| marker))
                .unwrap_or(candidate.predicate);
            explicit_actor(&source, &analyzer.lexicon, unit, head, scope_evidence)
        }
        .filter(|&actor| actor_compatible(candidate, actor));
        let result_evidence = frame
            .filter(|frame| !frame.blocked)
            .and_then(|frame| frame.result)
            .or(local.map(|local| local.result))
            .filter(|&span| contained(span, unit.evidence) && compatible_result(&candidate, span));
        by_unit[index] = match by_unit[index] {
            UnitCandidates::None => UnitCandidates::One(result.clauses.len()),
            _ => UnitCandidates::Multiple,
        };
        result.clauses.push(ScopedClause {
            input_index,
            clause: unit.evidence,
            candidate,
            actor_evidence,
            patient_evidence,
            result_complement: written_complement(
                &source,
                &analyzer.lexicon,
                unit.evidence,
                candidate.predicate,
            ),
            result_evidence,
            scope,
            scope_evidence,
        });
    }
    for (index, pair) in by_unit.windows(2).enumerate() {
        if let [UnitCandidates::One(action), UnitCandidates::One(next)] = *pair
            && !units[index + 1].first_in_line
            && !units[index..index + 2].iter().any(|unit| {
                let sentence = source.slice(&unit.sentence);
                contains_any(sentence, &["？", "?"])
                    || ends_with_question_particle(sentence)
                    || trim_clause_end(sentence).ends_with('吧')
            })
            && let Some(link) = result_link(
                &source,
                &result.clauses,
                &scopes[index],
                frames[index].as_ref(),
                &units[index..index + 2],
                &analyzer.lexicon,
                (action, next),
            )
        {
            result.results.push(link);
        }
    }
    match temporal {
        Some(TemporalOutput::Precedence(precedence)) => precedence.extend(clause_precedence(
            &source,
            &analyzer.lexicon,
            &units,
            &scopes,
            &frames,
            &by_unit,
            &result.clauses,
        )),
        Some(TemporalOutput::Cues(output)) => output.extend(
            cues::resolve(
                &source,
                &analyzer.lexicon,
                &units,
                &scopes,
                &frames,
                &result.clauses,
                &[],
            )
            .into_iter()
            .map(|cue| cue.order),
        ),
        Some(TemporalOutput::RankedCues(output, evidence)) => output.extend(cues::resolve(
            &source,
            &analyzer.lexicon,
            &units,
            &scopes,
            &frames,
            &result.clauses,
            evidence,
        )),
        Some(TemporalOutput::DiagnosedCues {
            cues,
            diagnostics,
            evidence,
            probes,
        }) => {
            (*cues, *diagnostics) = cues::resolve_diagnosed(
                &source,
                &analyzer.lexicon,
                &units,
                &scopes,
                &frames,
                &result.clauses,
                cues::DiagnosticInput {
                    evidence,
                    probes,
                    candidate_count: candidates.len(),
                },
            );
        }
        None => {}
    }
    result
}

const SEQUENCE: &[&str] = &[
    "緊接著",
    "紧接着",
    "緊接着",
    "隨後",
    "随后",
    "隨后",
    "隨即",
    "随即",
    "然後",
    "然后",
    "接著",
    "接着",
    "旋即",
    "下一刻",
    "接下來",
    "接下来",
];

fn elapsed_order_cue(text: &str) -> bool {
    crate::timeline::indefinite_duration(text)
        || ["之後", "之后", "後", "后"].iter().any(|suffix| {
            text.strip_suffix(suffix).is_some_and(|duration| {
                matches!(
                    duration,
                    "片刻" | "半晌" | "少頃" | "少顷" | "不久" | "一會兒" | "一会儿"
                )
            })
        })
}

fn temporal_boundary(source: &Source<'_>, unit: &Unit, scope: &UnitScope) -> bool {
    let sentence = source.slice(&unit.sentence);
    let text = source.slice(&unit.evidence);
    let elapsed = sentence
        .split_once(['，', ','])
        .is_some_and(|(cue, _)| elapsed_order_cue(cue.trim()));
    unit.quote
        || !scope.quotes.is_empty()
        || (temporal_shift(sentence).is_some() && !elapsed)
        || (temporal_shift(text).is_some()
            && !(elapsed && unit.evidence.start_char == unit.sentence.start_char))
        || sentence.contains(['？', '?'])
        || sentence.contains(['\u{2028}', '\u{2029}'])
        || ends_with_question_particle(sentence)
        || trim_clause_end(sentence).ends_with('吧')
        || contains_any(sentence, HYPOTHETICAL)
        || contains_any(
            sentence,
            &[
                "同時", "同时", "一邊", "一边", "一面", "隨著", "随着", "除非", "倘若", "萬一",
                "万一", "每當", "每当", "每次", "每逢",
            ],
        )
        || (starts_any(text, &["當", "当", "正當", "正当"])
            && trim_clause_end(text).ends_with(['時', '时']))
        || scene_transition(text)
}

fn scene_transition(text: &str) -> bool {
    starts_any(
        text,
        &[
            "另一邊",
            "另一边",
            "另一處",
            "另一处",
            "畫面一轉",
            "画面一转",
            "話分兩頭",
            "话分两头",
        ],
    )
}

fn temporal_endpoint(
    source: &Source<'_>,
    clause: &ScopedClause,
    unit: &Unit,
    scope: &UnitScope,
    frame: Option<&DisposalFrame<'_>>,
) -> bool {
    let text = source.slice(&unit.evidence);
    let sentence = source.slice(&unit.sentence);
    let through_head = source.slice(&Span {
        start_char: unit.evidence.start_char,
        end_char: clause.candidate.predicate.end_char,
    });
    let tail = source.slice(&Span {
        start_char: clause.candidate.predicate.end_char,
        end_char: clause.clause.end_char,
    });
    clause.scope == ClauseScope::Narrated
        && !temporal_boundary(source, unit, scope)
        && !frame
            .is_some_and(|frame| frame.predicate == clause.candidate.predicate && frame.blocked)
        && !source.has_speech(sentence)
        && !contains_any(
            sentence,
            &[
                "說，",
                "说，",
                "說,",
                "说,",
                "問，",
                "问，",
                "承認，",
                "承认，",
                "否認，",
                "否认，",
                "聲稱，",
                "声称，",
                "據傳",
                "据传",
                "講，",
                "讲，",
            ],
        )
        && !contains_any(text, THOUGHT)
        && !source.slice(&clause.candidate.predicate).contains('所')
        && !tail.trim_start().starts_with('的')
        && !contains_any(
            tail,
            &[
                "的建議",
                "的建议",
                "的要求",
                "的計畫",
                "的计划",
                "的計劃",
                "的打算",
                "的決定",
                "的决定",
                "的可能",
                "的辦法",
                "的办法",
                "之勢",
                "之势",
            ],
        )
        && !scope.cues.iter().any(|cue| {
            matches!(cue.scope, ClauseScope::Negated | ClauseScope::Planned)
                && cue.span.start_char < clause.candidate.predicate.end_char
        })
        && !contains_any(
            through_head,
            &[
                "說",
                "说",
                "問",
                "问",
                "請",
                "请",
                "務必",
                "务必",
                "記得",
                "记得",
                "等到",
                "可能",
                "可以",
                "將會",
                "将会",
                "若",
                "要不是",
                "拒絕",
                "拒绝",
                "否認",
                "否认",
                "承認",
                "承认",
                "聲稱",
                "声称",
                "似乎",
                "彷彿",
                "仿佛",
                "看起來",
                "看起来",
            ],
        )
}

fn temporal_subject(source: &Source<'_>, lexicon: &super::Lexicon<'_>, span: Span) -> bool {
    let text = source.slice(&span);
    matches!(
        text,
        "我" | "你" | "他" | "她" | "它" | "我們" | "我们" | "他們" | "他们" | "她們" | "她们"
    ) || participant_description(text)
        || source
            .first_name(
                source.bytes[span.start_char],
                source.bytes[span.end_char],
                lexicon,
            )
            .is_some_and(|(offset, alias, _)| offset == 0 && alias == text)
}

fn temporal_bridge(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    start: usize,
    end: usize,
    shared_subject: Option<&str>,
) -> Option<bool> {
    let Some(span) = source.trim(start, end) else {
        return Some(false);
    };
    let text = source.slice(&span);
    if shared_subject == Some(text) || temporal_subject(source, lexicon, span) {
        return Some(true);
    }
    let subject = text
        .strip_suffix(['便', '就', '即', '又', '已'])?
        .trim_end();
    if subject.is_empty() {
        return Some(false);
    }
    let start = source.bytes[span.start_char];
    (shared_subject == Some(subject)
        || temporal_subject(source, lexicon, source.span(start, start + subject.len())))
    .then_some(true)
}

fn leading_order_cue(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    clause: &ScopedClause,
    words: &[&str],
    shared_subject: Option<&str>,
) -> Option<(Span, bool)> {
    let start = source.bytes[clause.clause.start_char];
    let end = source.bytes[clause.candidate.predicate.start_char];
    let lead = &source.text[start..end];
    if source
        .trim(start, end)
        .is_some_and(|span| temporal_subject(source, lexicon, span))
    {
        return None;
    }
    words.iter().find_map(|word| {
        lead.match_indices(word).find_map(|(offset, _)| {
            let cue = start + offset;
            let after = cue + word.len();
            let subject = match (
                source.trim(start, cue),
                temporal_bridge(source, lexicon, after, end, shared_subject),
            ) {
                (None, Some(subject)) => subject,
                (Some(span), Some(false))
                    if shared_subject == Some(source.slice(&span))
                        || temporal_subject(source, lexicon, span) =>
                {
                    true
                }
                _ => return None,
            };
            Some((source.span(cue, after), subject))
        })
    })
}

fn attached_order_cue(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    clause: &ScopedClause,
    cross_line: bool,
) -> Option<(Span, bool)> {
    let text = source.slice(&clause.clause);
    if !text.ends_with(['，', ',', '；', ';']) {
        return None;
    }
    let body = trim_clause_end(text);
    let (word, backward) = [
        ("之後", false),
        ("之后", false),
        ("過後", false),
        ("过后", false),
        ("之前", true),
        ("以前", true),
    ]
    .into_iter()
    .find(|(word, _)| body.ends_with(word))?;
    let cue_end = source.bytes[clause.clause.start_char] + body.len();
    let cue_start = cue_end - word.len();
    grounded_action_tail(source, lexicon, clause.candidate, cue_start, cross_line)
        .then(|| (source.span(cue_start, cue_end), backward))
}

fn grounded_action_tail(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    candidate: ClauseCandidate,
    end: usize,
    require_grounded: bool,
) -> bool {
    let head_end = source.bytes[candidate.predicate.end_char];
    let Some(tail) = source.text.get(head_end..end) else {
        return false;
    };
    let tail = tail.trim_end();
    let body = tail.trim_start();
    if require_grounded && body.starts_with(['著', '着']) {
        return false;
    }
    let object = body
        .strip_prefix(['了', '過', '过', '著', '着'])
        .unwrap_or(body)
        .trim_start();
    if matches!(
        object,
        "" | "一下"
            | "一眼"
            | "一聲"
            | "一声"
            | "一遍"
            | "下來"
            | "下来"
            | "下去"
            | "過來"
            | "过来"
            | "過去"
            | "过去"
            | "起來"
            | "起来"
            | "而出"
            | "而來"
            | "而来"
    ) {
        return true;
    }
    let object_start = head_end + tail.len() - object.len();
    std::iter::once(object)
        .chain(object.strip_suffix('了').map(str::trim_end))
        .any(|phrase| {
            let noun = nominal_measure(phrase).unwrap_or(phrase);
            if noun.is_empty() {
                return false;
            }
            let end = object_start + phrase.len();
            let patient = source.span(end - noun.len(), end);
            (!require_grounded && candidate.patient.or(candidate.theme) == Some(patient))
                || lexicon
                    .objects
                    .iter()
                    .chain(lexicon.actors)
                    .any(|&(word, _)| word == noun)
                || lexicon.places.contains(&noun)
        })
}

fn local_line_gap(source: &Source<'_>, left: Span, right: Span) -> bool {
    let gap = source.slice(&Span {
        start_char: left.end_char,
        end_char: right.start_char,
    });
    let mut breaks = 0;
    for part in gap.split_inclusive('\n') {
        let text = if let Some(line) = part.strip_suffix('\n') {
            breaks += 1;
            line.strip_suffix('\r').unwrap_or(line)
        } else {
            part
        };
        if !text.trim_matches([' ', '\t', '\u{3000}']).is_empty() {
            return false;
        }
    }
    (1..=2).contains(&breaks)
}

fn repeated_paragraph_subject<'a>(
    source: &Source<'a>,
    earlier: &ScopedClause,
    later: &ScopedClause,
) -> Option<&'a str> {
    let lead = |clause: &ScopedClause| {
        source
            .slice(&Span {
                start_char: clause.clause.start_char,
                end_char: clause.candidate.predicate.start_char,
            })
            .trim()
    };
    let before = lead(earlier);
    if !before.starts_with('一') {
        return None;
    }
    let noun = nominal_measure(before)
        .or_else(|| before.strip_prefix('一')?.strip_prefix(['團', '团']))?;
    if !(2..=8).contains(&noun.chars().count()) || !noun.chars().all(char::is_alphanumeric) {
        return None;
    }
    let after = lead(later);
    std::iter::once(after)
        .chain(
            after
                .strip_suffix(['便', '就', '即', '又', '已'])
                .map(str::trim_end),
        )
        .any(|stem| {
            let mention = SEQUENCE
                .iter()
                .find_map(|cue| stem.strip_prefix(cue).or_else(|| stem.strip_suffix(cue)))
                .unwrap_or(stem)
                .trim();
            noun == mention
        })
        .then_some(noun)
}

fn known_action_head(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    clause: &ScopedClause,
) -> bool {
    let head = source.slice(&clause.candidate.predicate);
    clause.result_complement.is_some()
        || written_change(head)
        || predicate(head, lexicon).is_some_and(|(_, offset, verb)| offset == 0 && verb == head)
}

fn written_prior_head(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    clause: &ScopedClause,
    repeated_subject: bool,
) -> bool {
    known_action_head(source, lexicon, clause)
        && (repeated_subject
            || clause.actor_evidence.is_some()
            || clause.candidate.predicate.start_char == clause.clause.start_char)
}

fn written_prior_action(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    clause: &ScopedClause,
    repeated_subject: bool,
) -> bool {
    let text = source.slice(&clause.clause);
    text.ends_with(['。', '.', '，', ',', '；', ';'])
        && grounded_action_tail(
            source,
            lexicon,
            clause.candidate,
            source.bytes[clause.clause.start_char] + trim_clause_end(text).len(),
            true,
        )
        && written_prior_head(source, lexicon, clause, repeated_subject)
}

fn detached_order_cue(source: &Source<'_>, unit: &Unit) -> Option<Span> {
    let text = source.slice(&unit.evidence);
    let cue = text.strip_suffix(['，', ','])?.trim_end();
    let start = source.bytes[unit.evidence.start_char];
    (SEQUENCE.contains(&cue) || elapsed_order_cue(cue) || matches!(cue, "之後" | "之后"))
        .then(|| source.span(start, start + cue.len()))
}

fn clause_precedence(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    units: &[Unit],
    scopes: &[UnitScope],
    frames: &[Option<DisposalFrame<'_>>],
    by_unit: &[UnitCandidates],
    clauses: &[ScopedClause],
) -> Vec<ClausePrecedence> {
    let mut links = Vec::new();
    for (right, slot) in by_unit.iter().enumerate().skip(1) {
        let UnitCandidates::One(after) = *slot else {
            continue;
        };
        let (left, detached) = if matches!(by_unit[right - 1], UnitCandidates::One(_)) {
            (right - 1, None)
        } else if right >= 2
            && matches!(by_unit[right - 1], UnitCandidates::None)
            && units[right - 1].sentence == units[right].sentence
            && !temporal_boundary(source, &units[right - 1], &scopes[right - 1])
            && let Some(cue) = detached_order_cue(source, &units[right - 1])
        {
            (right - 2, Some(cue))
        } else {
            continue;
        };
        let UnitCandidates::One(before) = by_unit[left] else {
            continue;
        };
        if ![(left, before), (right, after)]
            .into_iter()
            .all(|(unit, clause)| {
                temporal_endpoint(
                    source,
                    &clauses[clause],
                    &units[unit],
                    &scopes[unit],
                    frames[unit].as_ref(),
                )
            })
        {
            continue;
        }
        let earlier = &clauses[before];
        let later = &clauses[after];
        let cue_unit_index = if detached.is_some() { right - 1 } else { right };
        let cross_line = units[cue_unit_index].first_in_line;
        if cross_line && !local_line_gap(source, earlier.clause, units[cue_unit_index].evidence) {
            continue;
        }
        let same_sentence = units[left].sentence == units[right].sentence;
        let shared_subject = cross_line
            .then(|| repeated_paragraph_subject(source, earlier, later))
            .flatten();
        let leading =
            leading_order_cue(source, lexicon, later, SEQUENCE, shared_subject).map(|(cue, _)| cue);
        let wrapped = cross_line && source.slice(&earlier.clause).ends_with(['，', ',']);
        let attached = (same_sentence || wrapped)
            .then(|| attached_order_cue(source, lexicon, earlier, cross_line))
            .flatten();
        if (detached.is_some() || attached.is_some())
            && temporal_bridge(
                source,
                lexicon,
                source.bytes[later.clause.start_char],
                source.bytes[later.candidate.predicate.start_char],
                shared_subject,
            )
            .is_none()
        {
            continue;
        }
        let (cue, paired_cue, backward) = match (detached, leading, attached) {
            (Some(cue), None, None) | (None, Some(cue), None) => (cue, None, false),
            (None, None, Some((cue, backward))) => (cue, None, backward),
            (None, None, None) if same_sentence || wrapped => {
                let Some((first, true)) =
                    leading_order_cue(source, lexicon, earlier, &["先"], None)
                else {
                    continue;
                };
                let Some((cue, _)) =
                    leading_order_cue(source, lexicon, later, &["再"], shared_subject)
                else {
                    continue;
                };
                (cue, Some(first), false)
            }
            _ => continue,
        };
        let needs_prior_action = cross_line
            || detached.is_some_and(|cue| {
                let text = source.slice(&cue);
                elapsed_order_cue(text) || matches!(text, "之後" | "之后")
            });
        if needs_prior_action
            && attached.is_none()
            && !written_prior_action(source, lexicon, earlier, shared_subject.is_some())
        {
            continue;
        }
        if cross_line
            && attached.is_some()
            && !written_prior_head(source, lexicon, earlier, shared_subject.is_some())
        {
            continue;
        }
        let cue_unit = if detached.is_some() {
            units[right - 1].evidence
        } else if attached.is_some() {
            earlier.clause
        } else {
            later.clause
        };
        if inside_lexical_term(source, lexicon, earlier.clause, earlier.candidate.predicate)
            || inside_lexical_term(source, lexicon, later.clause, later.candidate.predicate)
            || inside_lexical_term(source, lexicon, cue_unit, cue)
            || paired_cue
                .is_some_and(|cue| inside_lexical_term(source, lexicon, earlier.clause, cue))
        {
            continue;
        }
        let (before, after) = if backward {
            (after, before)
        } else {
            (before, after)
        };
        links.push(ClausePrecedence {
            before,
            after,
            cue,
            paired_cue,
        });
    }
    links
}

fn actor_compatible(candidate: ClauseCandidate, evidence: Span) -> bool {
    candidate.actor.is_none_or(|actor| actor == evidence)
        && [candidate.theme, candidate.patient, candidate.result]
            .into_iter()
            .flatten()
            .all(|role| !overlaps(role, evidence))
}

fn explicit_actor(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    unit: &Unit,
    predicate: Span,
    cue: Option<Span>,
) -> Option<Span> {
    let predicate_start = source.bytes[predicate.start_char];
    let mut start = source.bytes[unit.evidence.start_char];
    let cue = cue
        .filter(|cue| contained(*cue, unit.evidence))
        .map(|cue| source.bytes[cue.start_char]..source.bytes[cue.end_char]);
    if let Some(cue) = &cue
        && cue.start == start
        && cue.end <= predicate_start
    {
        start = predicate_start - source.text[cue.end..predicate_start].trim_start().len();
    }
    let lead = &source.text[start..predicate_start];
    if let Some(tail) = continuation(lead) {
        start = predicate_start - tail.trim_start().len();
    }
    let mut bridge_end = predicate_start;
    if let Some(cue) = &cue
        && cue.start >= start
        && cue.end <= predicate_start
        && source.text[cue.end..predicate_start].trim().is_empty()
    {
        bridge_end = cue.start;
    }
    let lead = source.text[start..bridge_end].trim_end();
    if let Some((0, alias, _)) = source.first_name(start, bridge_end, lexicon)
        && let Some(suffix) = lead.strip_prefix(alias)
        && local_actor_modifier(suffix.trim())
    {
        return Some(source.span(start, start + alias.len()));
    }
    let surface = lead
        .strip_suffix(['便', '就', '又', '也', '先'])
        .unwrap_or(lead)
        .trim_end();
    (matches!(surface, "我" | "你" | "他" | "她" | "它") || participant_description(surface))
        .then(|| source.span(start, start + surface.len()))
}

fn local_actor_modifier(text: &str) -> bool {
    matches!(
        text,
        "" | "已"
            | "已經"
            | "已经"
            | "早已"
            | "一"
            | "便"
            | "就"
            | "先"
            | "又"
            | "也"
            | "瞬間"
            | "瞬间"
    ) || continuation(text).is_some_and(str::is_empty)
}

#[derive(Clone, Copy)]
struct LocalResult {
    result: Span,
    disposal: Option<(Span, Span)>,
}

fn local_result(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    unit: &Unit,
    candidate: ClauseCandidate,
) -> Option<LocalResult> {
    let head = source.slice(&candidate.predicate);
    if head.chars().count() < 2 || !head.ends_with(['成', '為', '为']) {
        return None;
    }
    let clause = source.slice(&unit.evidence);
    if clause.contains(['？', '?']) || ends_with_question_particle(clause) {
        return None;
    }
    let result = explicit_result(source, unit.evidence, candidate.predicate)?;
    let value = source.slice(&result);
    if value.chars().count() > 24
        || !value
            .chars()
            .all(|ch| ch.is_alphanumeric() || ch.is_whitespace())
        || super::ends_any(
            value,
            &[
                "之前", "以前", "之後", "之后", "以後", "以后", "為止", "为止", "吧", "次", "遍",
            ],
        )
    {
        return None;
    }
    let start = source.bytes[unit.evidence.start_char];
    let end = source.bytes[candidate.predicate.start_char];
    let lead = &source.text[start..end];
    let Some((marker, word)) = lead.rmatch_indices(['把', '將', '将']).next() else {
        return written_change(head).then_some(LocalResult {
            result,
            disposal: None,
        });
    };
    let marker = source.span(start + marker, start + marker + word.len());
    let patient = source.trim(source.bytes[marker.end_char], end)?;
    let body = source.slice(&patient);
    let prefix = source.text[start..source.bytes[marker.start_char]].trim();
    if !(local_actor_modifier(prefix)
        || explicit_actor(source, lexicon, unit, marker, None).is_some())
        || !source.text
            [source.bytes[candidate.predicate.end_char]..source.bytes[unit.evidence.end_char]]
            .trim_start()
            .starts_with('了')
        || body.chars().count() > 16
        || !body.chars().all(char::is_alphanumeric)
        || predicate(body, lexicon).is_some()
        || contains_any(
            body,
            &["瞬間", "瞬间", "隨即", "随即", "立刻", "立即", "用力"],
        )
        || !(candidate.patient.or(candidate.theme) == Some(patient)
            || nominal_measure(body).is_some_and(|noun| !noun.is_empty())
            || matches!(body, "他" | "她" | "它")
            || starts_any(
                body,
                &[
                    "兩具", "两具", "一具", "那個", "那个", "這個", "这个", "那些", "這些", "这些",
                ],
            ))
    {
        return None;
    }
    Some(LocalResult {
        result,
        disposal: Some((marker, patient)),
    })
}

fn written_change(head: &str) -> bool {
    matches!(
        head,
        "裂成" | "斷成" | "断成" | "變成" | "变成" | "化成" | "化為" | "化为" | "凝成" | "融成"
    )
}

fn mention(source: &Source<'_>, clause: &ScopedClause, span: Span) -> bool {
    let text = source.slice(&span);
    text.chars().count() <= 24
        && text.chars().all(|ch| ch.is_alphanumeric() || ch == ' ')
        && compatible_patient(&clause.candidate, span)
}

fn next_patient(source: &Source<'_>, clause: &ScopedClause) -> Option<(Span, Span)> {
    let start = source.bytes[clause.clause.start_char];
    let end = source.bytes[clause.candidate.predicate.start_char];
    let prefix = &source.text[start..end];
    let (offset, cue) = earliest(prefix, &["因此", "因而", "從而", "从而"])?;
    let cue_start = start + offset;
    let cue_end = cue_start + cue.len();
    let patient = match (source.trim(start, cue_start), source.trim(cue_end, end)) {
        (Some(span), None) | (None, Some(span)) => span,
        _ => return None,
    };
    mention(source, clause, patient).then(|| (source.span(cue_start, cue_end), patient))
}

fn continuation_patient(source: &Source<'_>, clause: &ScopedClause) -> Option<Span> {
    let start = source.bytes[clause.clause.start_char];
    let end = source.bytes[clause.candidate.predicate.start_char];
    let lead = source.text[start..end].trim_end();
    let text = ["隨即", "随即", "頓時", "顿时", "旋即"]
        .iter()
        .find_map(|cue| lead.strip_suffix(cue))
        .unwrap_or(lead);
    let patient = source.trim(start, start + text.len())?;
    mention(source, clause, patient).then_some(patient)
}

fn compatible_patient(candidate: &ClauseCandidate, patient: Span) -> bool {
    candidate
        .patient
        .or(candidate.theme)
        .is_none_or(|span| span == patient)
        && candidate.actor.is_none_or(|span| !overlaps(span, patient))
        && candidate.result.is_none_or(|span| !overlaps(span, patient))
}

fn compatible_result(candidate: &ClauseCandidate, result: Span) -> bool {
    candidate
        .result
        .is_none_or(|span| predicate_suffix(span, candidate.predicate) || span == result)
        && [candidate.actor, candidate.theme, candidate.patient]
            .into_iter()
            .flatten()
            .all(|role| !overlaps(role, result))
}

fn previous_patient(source: &Source<'_>, clause: &ScopedClause, surface: &str) -> Option<Span> {
    let predicate = clause.candidate.predicate;
    let start = source.bytes[clause.clause.start_char];
    let end = source.bytes[clause.clause.end_char];
    let tail = trim_clause_end(source.text[source.bytes[predicate.end_char]..end].trim_start());
    let aspect_tail = tail
        .strip_prefix(['了', '過', '过', '著', '着'])
        .unwrap_or(tail)
        .trim_start();
    if let Some(patient) = [
        tail,
        aspect_tail,
        nominal_measure(aspect_tail).unwrap_or(aspect_tail),
    ]
    .into_iter()
    .find(|text| *text == surface)
    {
        let trimmed_end = source.bytes[predicate.end_char]
            + trim_clause_end(&source.text[source.bytes[predicate.end_char]..end]).len();
        return Some(source.span(trimmed_end - patient.len(), trimmed_end));
    }
    let lead = source.text[start..source.bytes[predicate.start_char]].trim_end();
    let prefix = lead.strip_suffix(surface)?;
    if prefix.trim_end().ends_with(['把', '將', '将']) {
        let end = start + lead.len();
        return Some(source.span(end - surface.len(), end));
    }
    let end = start + lead.len();
    let theme = source.span(end - surface.len(), end);
    (prefix.trim().is_empty() && clause.candidate.theme == Some(theme)).then_some(theme)
}

const REFERENCE_CLASSIFIERS: &str = "扇本卷條条張张塊块片枚個个只件根盞盏把面台臺名位";

fn demonstrative_nouns(text: &str) -> [Option<&str>; 2] {
    if matches!(text, "此刻" | "此時" | "此时" | "這時" | "这时") {
        return [None; 2];
    }
    let Some(body) = text.strip_prefix(['這', '这', '此']) else {
        return [None; 2];
    };
    let mut chars = body.chars();
    let classifier = chars
        .next()
        .filter(|&ch| REFERENCE_CLASSIFIERS.contains(ch));
    [Some(body), classifier.map(|_| chars.as_str())]
}

fn counted_patient(source: &Source<'_>, clause: Span, patient: Span) -> bool {
    let lead = source.slice(&Span {
        start_char: clause.start_char,
        end_char: patient.start_char,
    });
    let mut chars = lead.chars().rev().filter(|ch| !ch.is_whitespace());
    chars
        .next()
        .is_some_and(|ch| REFERENCE_CLASSIFIERS.contains(ch))
        && chars.next().is_some_and(|ch| {
            "一二三四五六七八九十百千萬万兩两幾几數数多半每各0123456789".contains(ch)
        })
}

fn shared_patient(
    source: &Source<'_>,
    lexicon: &super::Lexicon<'_>,
    before: &ScopedClause,
    frame: Option<&DisposalFrame<'_>>,
    surface: &str,
    can_refer: bool,
    starts_sentence: bool,
) -> Option<(Span, bool)> {
    let matching = |noun: &str| {
        if noun.is_empty() {
            return None;
        }
        let patient = match frame {
            Some(frame) if source.slice(&frame.patient) == noun => frame.patient,
            Some(_) => return None,
            None => previous_patient(source, before, noun)?,
        };
        compatible_patient(&before.candidate, patient).then_some(patient)
    };
    if let Some(patient) = matching(surface) {
        return Some((patient, false));
    }
    if !can_refer {
        return None;
    }
    if let Some(patient) = demonstrative_nouns(surface)
        .into_iter()
        .flatten()
        .find_map(matching)
        .filter(|&patient| !counted_patient(source, before.clause, patient))
    {
        return Some((patient, true));
    }
    let frame = frame.filter(|_| surface == "它" && starts_sentence)?;
    let actor = source.slice(&frame.actor_surface?);
    let patient = source.slice(&frame.patient);
    let tail = source.slice(&Span {
        start_char: before.candidate.predicate.end_char,
        end_char: before.clause.end_char,
    });
    ((matches!(actor, "我" | "你" | "他" | "她") || participant_description(actor))
        && lexicon.objects.iter().any(|&(word, _)| word == patient)
        && !lexicon.actors.iter().any(|&(word, _)| word == patient)
        && !counted_patient(source, before.clause, frame.patient)
        && matches!(trim_clause_end(tail.trim_start()), "" | "了")
        && compatible_patient(&before.candidate, frame.patient))
    .then_some((frame.patient, true))
}

fn explicit_result(source: &Source<'_>, clause: Span, predicate: Span) -> Option<Span> {
    let end = source.bytes[clause.end_char];
    let tail = source.text[source.bytes[predicate.end_char]..end].trim_start();
    let tail = if source.slice(&predicate).ends_with(['成', '為', '为']) {
        tail
    } else {
        tail.strip_prefix(['成', '為', '为'])?
    };
    let value = tail.strip_prefix('了').unwrap_or(tail).trim_start();
    let start = end - value.len();
    let value = trim_clause_end(value);
    if !value.chars().any(char::is_alphanumeric) {
        return None;
    }
    Some(source.span(start, start + value.len()))
}

fn result_link(
    source: &Source<'_>,
    clauses: &[ScopedClause],
    before_scope: &UnitScope,
    frame: Option<&DisposalFrame<'_>>,
    units: &[Unit],
    lexicon: &super::Lexicon<'_>,
    (action, result_clause): (usize, usize),
) -> Option<ClauseResultLink> {
    let before = &clauses[action];
    let after = &clauses[result_clause];
    let [before_unit, after_unit] = units else {
        return None;
    };
    let same_sentence = before_unit.sentence == after_unit.sentence;
    if [before, after].into_iter().any(|clause| {
        clause.scope != ClauseScope::Narrated
            || source
                .slice(&clause.clause)
                .chars()
                .any(|ch| quote_end(ch).is_some())
    }) {
        return None;
    }
    let frame = frame.filter(|frame| frame.predicate == before.candidate.predicate);
    if frame.is_some_and(|frame| frame.blocked) {
        return None;
    }
    let transform = written_change(source.slice(&after.candidate.predicate));
    let (cue, patient_after, explicit_cause, result, needs_reference) =
        if let Some((cue, patient)) = next_patient(source, after) {
            (
                cue,
                patient,
                true,
                explicit_result(source, after.clause, after.candidate.predicate)
                    .filter(|&result| compatible_result(&after.candidate, result)),
                false,
            )
        } else if let Some(result) = frame
            .and_then(|frame| frame.result)
            .filter(|&span| contained(span, after.clause))
        {
            if after.candidate.predicate.end_char > result.start_char {
                return None;
            }
            let start = source.bytes[after.clause.start_char];
            let verb_start = source.bytes[after.candidate.predicate.start_char];
            let patient = source.trim(start, verb_start)?;
            let cue = source.trim(verb_start, source.bytes[result.start_char])?;
            if !compatible_patient(&after.candidate, patient) {
                return None;
            }
            (
                cue,
                patient,
                false,
                compatible_result(&after.candidate, result).then_some(result),
                false,
            )
        } else if same_sentence
            && transform
            && let Some(result) = after.result_evidence
        {
            (
                after.candidate.predicate,
                continuation_patient(source, after)?,
                false,
                Some(result),
                true,
            )
        } else {
            return None;
        };
    let (patient_before, patient_reference) = shared_patient(
        source,
        lexicon,
        before,
        frame,
        source.slice(&patient_after),
        same_sentence && transform && result.is_some(),
        before_unit.evidence.start_char == before_unit.sentence.start_char,
    )?;
    if needs_reference && !patient_reference {
        return None;
    }
    if before.candidate.patient.is_none()
        && patient_before.end_char <= before.candidate.predicate.start_char
        && candidate_scope(
            source,
            before_scope,
            &ClauseCandidate {
                patient: Some(patient_before),
                ..before.candidate
            },
        )
        .0 != ClauseScope::Narrated
    {
        return None;
    }
    Some(ClauseResultLink {
        action,
        result_clause,
        explicit_cause,
        patient_reference,
        cue,
        patient_before,
        patient_after,
        result,
    })
}

#[derive(Clone, Copy)]
struct ScopeCue {
    scope: ClauseScope,
    span: Span,
}

struct UnitScope {
    quotes: Vec<Span>,
    cues: Vec<ScopeCue>,
    wakes: Vec<Span>,
    dream: Option<Span>,
    report: Option<Span>,
    condition: Option<Span>,
}

const LINK_DREAM: &[&str] = &["在睡夢中", "在睡梦中", "睡夢中", "睡梦中"];
const LINK_WAKE: &[&str] = &[
    "從夢中醒來",
    "从梦中醒来",
    "夢醒",
    "梦醒",
    "醒來",
    "醒来",
    "醒了",
    "回到眼前",
    "回過神",
    "回过神",
];
const LINK_CONDITION: &[&str] = &["假使", "如若", "假若", "若"];
const LINK_PLANS: &[&str] = &["打算", "原想", "本想", "正想", "將", "将", "欲", "想", "要"];
const LINK_NEGATIONS: &[&str] = &[
    "從未", "从未", "不曾", "未曾", "尚未", "並未", "并未", "沒有", "没有", "未能", "沒能", "没能",
    "無法", "无法", "不能", "不要", "別", "别", "沒", "没", "未", "不",
];

fn scope_quotes(source: &Source<'_>, units: &[Unit], include_open_quote: bool) -> Vec<Span> {
    let (Some(first), Some(last)) = (units.first(), units.last()) else {
        return Vec::new();
    };
    // A malformed leading quotation may be outside the retained units.
    let base = if include_open_quote {
        0
    } else {
        source.bytes[first.evidence.start_char]
    };
    let text = &source.text[base..source.bytes[last.evidence.end_char]];
    let mut quotes = Vec::new();
    let mut stack = Vec::new();
    let mut opening = None;
    for (offset, ch) in text.char_indices() {
        let depth = stack.len();
        let book_close = match ch {
            '《' => Some('》'),
            '〈' => Some('〉'),
            _ => None,
        };
        if let Some(end) = book_close
            && text[offset + ch.len_utf8()..].contains(end)
        {
            stack.push(end);
        } else if !track_quote(text, offset, ch, &mut stack) {
            if include_open_quote && let Some(end) = quote_end(ch) {
                stack.push(end);
            } else {
                continue;
            }
        }
        if depth == 0 && !stack.is_empty() {
            opening = Some(offset);
        } else if stack.is_empty()
            && let Some(start) = opening.take()
        {
            quotes.push(source.span(base + start, base + offset + ch.len_utf8()));
        }
    }
    if include_open_quote && let Some(start) = opening {
        quotes.push(source.span(base + start, base + text.len()));
    }
    quotes
}

fn scope_cues(
    source: &Source<'_>,
    text: &str,
    base: usize,
    quotes: &[Span],
    words: &[&str],
) -> Vec<Span> {
    words
        .iter()
        .flat_map(|word| text.match_indices(word))
        .filter(|&(offset, word)| {
            (word != "若" || text[..offset].trim().is_empty())
                && (word != "不"
                    || !starts_any(
                        &text[offset..],
                        &[
                            "不但", "不僅", "不仅", "不只", "不光", "不時", "不时", "不由", "不免",
                        ],
                    ))
        })
        .map(|(offset, word)| source.span(base + offset, base + offset + word.len()))
        .filter(|&span| !quotes.iter().any(|&quote| overlaps(span, quote)))
        .collect()
}

fn scopes_for_units(
    source: &Source<'_>,
    units: &[Unit],
    include_open_quote: bool,
) -> Vec<UnitScope> {
    let mut dream = None;
    let mut report = None;
    let mut condition = None;
    let mut sentence = None;
    let mut states = Vec::with_capacity(units.len());
    let quotes = scope_quotes(source, units, include_open_quote);
    for unit in units {
        if sentence != Some(unit.sentence) {
            report = None;
            condition = None;
            sentence = Some(unit.sentence);
        }
        let text = source.slice(&unit.evidence);
        let base = source.bytes[unit.evidence.start_char];
        let first = quotes.partition_point(|quote| quote.end_char <= unit.evidence.start_char);
        let last = quotes.partition_point(|quote| quote.start_char < unit.evidence.end_char);
        let mut local_quotes = quotes[first..last].to_vec();
        if unit.quote && local_quotes.is_empty() {
            local_quotes.push(unit.evidence);
        }
        let mut cues = Vec::new();
        if !unit.quote {
            for (scope, words) in [
                (ClauseScope::Dream, DREAM),
                (ClauseScope::Dream, LINK_DREAM),
                (ClauseScope::Reported, REPORTED),
                (ClauseScope::Conditional, HYPOTHETICAL),
                (ClauseScope::Conditional, LINK_CONDITION),
                (ClauseScope::Planned, PLANNED),
                (ClauseScope::Planned, LINK_PLANS),
                (ClauseScope::Negated, LINK_NEGATIONS),
            ] {
                cues.extend(
                    scope_cues(source, text, base, &local_quotes, words)
                        .into_iter()
                        .map(|span| ScopeCue { scope, span }),
                );
            }
        }
        let wakes = if unit.quote {
            Vec::new()
        } else {
            scope_cues(source, text, base, &local_quotes, LINK_WAKE)
        };
        states.push(UnitScope {
            quotes: local_quotes,
            cues,
            wakes,
            dream,
            report,
            condition,
        });
        let state = states.last().unwrap();
        if unit.quote {
            continue;
        }
        let latest_dream = state
            .cues
            .iter()
            .filter(|cue| matches!(cue.scope, ClauseScope::Dream))
            .map(|cue| cue.span)
            .max_by_key(|span| (span.end_char, std::cmp::Reverse(span.start_char)));
        let latest_wake = state.wakes.iter().map(|span| span.end_char).max();
        if latest_wake.is_some_and(|wake| latest_dream.is_none_or(|cue| wake > cue.end_char))
            || (unit.first_in_line
                && latest_dream.is_none()
                && temporal_shift(text) == Some("main"))
        {
            dream = None;
        } else {
            dream = latest_dream.or(dream);
        }
        for cue in &state.cues {
            // A leading cue can govern the following clauses of this sentence.
            if cue.span.start_char != unit.evidence.start_char {
                continue;
            }
            let slot = match cue.scope {
                ClauseScope::Reported => &mut report,
                ClauseScope::Conditional => &mut condition,
                _ => continue,
            };
            if slot.is_none_or(|old| {
                old.start_char == cue.span.start_char && old.end_char < cue.span.end_char
            }) {
                *slot = Some(cue.span);
            }
        }
    }
    states
}

fn scope_operator_reaches(source: &Source<'_>, cue: Span, candidate: &ClauseCandidate) -> bool {
    let mut gap = source
        .slice(&Span {
            start_char: cue.end_char,
            end_char: candidate.predicate.start_char,
        })
        .trim();
    if let Some(patient) = candidate.patient
        && cue.end_char <= patient.start_char
        && patient.end_char <= candidate.predicate.start_char
        && source
            .slice(&Span {
                start_char: cue.end_char,
                end_char: patient.start_char,
            })
            .trim()
            .strip_prefix(['把', '將', '将'])
            .map(str::trim)
            .is_some_and(|lead| lead.is_empty() || nominal_measure(lead) == Some(""))
    {
        gap = source
            .slice(&Span {
                start_char: patient.end_char,
                end_char: candidate.predicate.start_char,
            })
            .trim();
    }
    matches!(
        gap,
        "" | "再"
            | "先"
            | "去"
            | "立刻"
            | "輕輕"
            | "轻轻"
            | "慢慢"
            | "小心地"
            | "明日"
            | "明天"
            | "稍後"
            | "稍后"
    )
}

fn candidate_scope(
    source: &Source<'_>,
    state: &UnitScope,
    candidate: &ClauseCandidate,
) -> (ClauseScope, Option<Span>) {
    if let Some(quote) = state
        .quotes
        .iter()
        .find(|&&quote| overlaps(quote, candidate.predicate))
    {
        return (
            ClauseScope::Quoted,
            Some(Span {
                start_char: quote.start_char,
                end_char: quote.start_char + 1,
            }),
        );
    }
    let local = |scope| {
        let eligible = state
            .cues
            .iter()
            .filter(|cue| {
                if cue.scope != scope || cue.span.start_char > candidate.predicate.start_char {
                    return false;
                }
                if !matches!(scope, ClauseScope::Negated | ClauseScope::Planned) {
                    return cue.span.end_char <= candidate.predicate.start_char;
                }
                cue.span.end_char > candidate.predicate.start_char
                    || scope_operator_reaches(source, cue.span, candidate)
            })
            .map(|cue| cue.span);
        if matches!(scope, ClauseScope::Dream) {
            eligible.max_by_key(|span| (span.end_char, std::cmp::Reverse(span.start_char)))
        } else {
            eligible.min_by_key(|span| (span.start_char, std::cmp::Reverse(span.end_char)))
        }
    };
    let dream = local(ClauseScope::Dream).or(state.dream).filter(|dream| {
        !state.wakes.iter().any(|&wake| {
            wake.start_char >= dream.end_char && wake.start_char <= candidate.predicate.start_char
        })
    });
    for (scope, cue) in [
        (ClauseScope::Dream, dream),
        (
            ClauseScope::Reported,
            local(ClauseScope::Reported).or(state.report),
        ),
        (
            ClauseScope::Conditional,
            local(ClauseScope::Conditional).or(state.condition),
        ),
        (ClauseScope::Negated, local(ClauseScope::Negated)),
        (ClauseScope::Planned, local(ClauseScope::Planned)),
    ] {
        if cue.is_some() {
            return (scope, cue);
        }
    }
    (ClauseScope::Narrated, None)
}
