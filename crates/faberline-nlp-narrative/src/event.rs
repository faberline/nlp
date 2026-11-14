//! Clause-level candidate event extraction, Layer-2 syntax gates, disposal/passive
//! role realignment, factuality classification, and temporal anchor attachment.

use crate::dialogue::zone_for_span;
use crate::schema::{
    ArgumentBasis, CandidateEvent, ContextZone, DialogueSpan, EventArgument, EventRole, Factuality,
    NarrativeUnknown, NarrativeUnknownKind, TemporalAnchor, WorkVocabulary,
};
use crate::{glued_pronoun_prefix, is_personal_pronoun, span_contains, span_text, spans_overlap};
use faberline_nlp::Span;
use faberline_nlp::doc::{Document, EntityKind, Reason, RelationKind, SyntaxToken, Upos};
use std::collections::HashSet;

/// Output of clause-level event extraction for a chapter.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EventExtraction {
    pub events: Vec<CandidateEvent>,
    pub unknown: Vec<NarrativeUnknown>,
}

const ADVERBIAL_PSEUDO_PREDICATES: &[&str] = &[
    "幾乎",
    "几乎",
    "紛紛",
    "纷纷",
    "接著",
    "接着",
    "隨即",
    "随即",
    "隨後",
    "随后",
    "同時",
    "同时",
    "與此同時",
    "与此同时",
    "還",
    "还",
    "才",
    "就",
    "都",
    "也",
    "只",
    "更",
    "卻",
    "却",
    "居然",
    "竟然",
    "果然",
    "頓時",
    "顿时",
    "立刻",
    "馬上",
    "马上",
    "忽然",
    "突然",
    "漸漸",
    "渐渐",
    "慢慢",
    "緩緩",
    "缓缓",
    "連忙",
    "连忙",
    "趕緊",
    "赶紧",
    "早已",
    "已經",
    "已经",
    "曾經",
    "曾经",
    "不時",
    "不时",
    "時常",
    "时常",
    "另外",
    "此外",
    "而且",
    "並且",
    "并且",
    "因此",
    "所以",
    "於是",
    "于是",
    "然後",
    "然后",
    "然而",
    "但是",
    "可是",
    "不過",
    "不过",
    "低聲",
    "低声",
    "冷聲",
    "冷声",
    "沉聲",
    "沉声",
    "朗聲",
    "朗声",
    "輕聲",
    "轻声",
    "大聲",
    "大声",
    "小聲",
    "小声",
    "淡淡",
    "冷冷",
    "微微",
    "悄悄",
    "偷偷",
    "急忙",
    "暗暗",
    "默默",
    "仔細",
    "仔细",
    "輕輕",
    "轻轻",
    "重重",
    "猛地",
    "用力",
    "趕忙",
    "赶忙",
];

const ADVERB_PROXY_HEADS: &[&str] = &["還", "还", "才", "就", "都", "也", "只", "更", "卻", "却"];

const HYPOTHETICAL_CJK_CUES: &[&str] = &[
    "若是",
    "倘若",
    "如果",
    "假如",
    "萬一",
    "万一",
    "一旦",
    "設若",
    "设若",
    "若",
    "除非",
    "もしも",
    "もし",
    "仮に",
    "万が一",
    "ならば",
    "としたら",
    "とすれば",
];
const HYPOTHETICAL_JA_TOKEN_CUES: &[&str] = &[
    "もしも",
    "もし",
    "仮に",
    "万が一",
    "たら",
    "れば",
    "なら",
    "ならば",
    "としたら",
    "とすれば",
];
const HYPOTHETICAL_EN_CUES: &[&str] = &["if", "unless", "suppose", "supposing", "provided"];
const HYPOTHETICAL_NON_COND_WORDS: &[&str] = &[
    "若無其事",
    "若无其事",
    "宛若",
    "仿若",
    "若有所思",
    "自若",
    "若干",
    "倘或",
];

const NEGATION_CJK_CUES: &[&str] = &[
    "沒有",
    "没有",
    "未曾",
    "並未",
    "并未",
    "並非",
    "并非",
    "毫無",
    "毫无",
    "絕不",
    "绝不",
    "從未",
    "从未",
    "並不",
    "并不",
    "不會",
    "不会",
    "不能",
    "不可",
    "不敢",
    "不想",
    "不知",
    "不得",
    "無法",
    "无法",
    "未能",
    "不用",
    "不必",
    "不曾",
    "不願",
    "不愿",
    "不肯",
    "沒法",
    "没法",
    "沒能",
    "没能",
    "不再",
    "不是",
    "不見",
    "不见",
    "沒見",
    "没见",
    "不",
    "沒",
    "没",
    "未",
    "莫",
    "勿",
    "ない",
    "なかっ",
    "なかった",
    "なく",
    "ず",
    "ぬ",
    "ません",
    "ませんでした",
    "まい",
    "じゃない",
    "ではない",
    "ではなかった",
];
const NEGATION_EN_CUES: &[&str] = &[
    "never",
    "not",
    "n't",
    "didn't",
    "doesn't",
    "don't",
    "won't",
    "wouldn't",
    "couldn't",
    "shouldn't",
    "wasn't",
    "weren't",
    "isn't",
    "aren't",
    "cannot",
    "can't",
    "without",
    "neither",
    "nor",
];
const NON_NEGATED_IDIOMS: &[&str] = &[
    "不久",
    "不錯",
    "不错",
    "不由得",
    "不禁",
    "不覺",
    "不觉",
    "不知不覺",
    "不知不觉",
    "不斷",
    "不断",
    "不停",
    "不時",
    "不时",
    "不料",
    "不愧",
    "不免",
    "不乏",
    "不妨",
    "不勝",
    "不胜",
    "不惜",
    "不顧",
    "不顾",
    "不但",
    "不僅",
    "不仅",
    "不只",
    "不單",
    "不单",
    "不遠",
    "不远",
    "不少",
    "不小",
    "不大",
    "不得不",
    "不得了",
    "不得已",
    "不可思議",
    "不可思议",
    "不可一世",
    "不多",
    "不到",
    "不等",
    "不論",
    "不论",
    "不管",
    "不要緊",
    "不要紧",
    "不至於",
    "不至于",
    "沒錯",
    "没错",
    "沒多久",
    "没多久",
    "未來",
    "未来",
    "未必",
];

const MODAL_FUTURE_CJK_CUES: &[&str] = &[
    "將要",
    "将要",
    "即將",
    "即将",
    "快要",
    "打算",
    "準備",
    "准备",
    "想要",
    "或許",
    "或许",
    "恐怕",
    "未必",
    "可能",
    "也許",
    "也许",
    "大概",
    "應該",
    "应该",
    "必須",
    "必须",
    "欲",
    "よう",
    "だろう",
    "でしょう",
    "かもしれ",
    "かもしれない",
    "かもしれません",
    "たい",
    "たかった",
    "つもり",
    "はず",
    "べき",
    "ほしい",
];
const MODAL_FUTURE_EN_CUES: &[&str] = &[
    "will", "would", "might", "may", "could", "should", "shall", "must", "'ll",
];

const PP_MARKERS_CJK: &[&str] = &[
    "從", "从", "在", "向", "對", "对", "給", "给", "往", "到", "於", "于", "朝", "沿", "順", "顺",
    "與", "与", "和", "跟", "替", "為", "为", "向著", "朝著", "沿著", "順著",
];
const PP_MARKERS_EN: &[&str] = &[
    "in", "on", "at", "to", "from", "with", "into", "onto", "by", "for", "of", "about", "through",
    "toward", "towards", "across", "over", "under", "between", "against",
];

const TEMPORAL_QUANTITY_UNITS: &[&str] = &[
    "秒",
    "分鐘",
    "分钟",
    "小時",
    "小时",
    "時間",
    "分間",
    "秒間",
    "日間",
    "週間",
    "ヶ月間",
    "か月間",
    "年間",
    "個時辰",
    "个时辰",
    "時辰",
    "时辰",
    "刻鐘",
    "刻钟",
    "盞茶",
    "盏茶",
    "炷香",
    "柱香",
    "個甲子",
    "个甲子",
    "甲子",
    "載",
    "载",
    "天",
    "夜",
    "更",
    "週",
    "周",
    "個月",
    "个月",
    "年",
    "日",
    "月",
    "歲",
    "岁",
    "seconds",
    "second",
    "sec",
    "minutes",
    "minute",
    "min",
    "hours",
    "hour",
    "hr",
    "days",
    "day",
    "weeks",
    "week",
    "months",
    "month",
    "years",
    "year",
];

/// Extracts clause-level candidate events from `doc` and `text`.
pub fn extract_events(
    chapter: u32,
    text: &str,
    doc: &Document,
    vocab: &WorkVocabulary,
    dialogues: &[DialogueSpan],
) -> EventExtraction {
    let chars: Vec<char> = text.chars().collect();
    let opaque_spans: Vec<Span> = doc.segmentation.opaque.iter().map(|o| o.span).collect();

    let mut syntax_by_tok: Vec<Option<&SyntaxToken>> = vec![None; doc.tokens.len()];
    let mut sentence_has_syntax = vec![false; doc.segmentation.sentences.len()];
    for st in &doc.syntax {
        if st.token < doc.tokens.len() {
            syntax_by_tok[st.token] = Some(st);
        }
        if st.sentence < sentence_has_syntax.len() {
            sentence_has_syntax[st.sentence] = true;
        }
    }

    let low_conf_pos: Vec<Span> = doc
        .unknown
        .iter()
        .filter(|u| u.reason == Reason::LowConfidence && u.layer == "pos")
        .map(|u| u.span)
        .collect();
    let low_conf_syntax: Vec<Span> = doc
        .unknown
        .iter()
        .filter(|u| u.reason == Reason::LowConfidence && u.layer == "syntax")
        .map(|u| u.span)
        .collect();
    // Distinguish genuine rule-only (`--no-model`) docs from model-analyzed docs where
    // some or all sentences had `syntax` suppressed by `low_confidence`.
    let model_syntax_attempted = !doc.syntax.is_empty() || !low_conf_syntax.is_empty();

    // Precompute actor/place/object spans from `WorkVocabulary` and `doc.entities`.
    let vocab_actors = vocab.find_actor_spans(text);
    let vocab_places = vocab.find_place_spans(text);
    let vocab_objects = vocab.find_object_spans(text);

    // Map each token to its hosting clause index.
    let mut token_clause: Vec<Option<usize>> = vec![None; doc.tokens.len()];
    let mut clause_tokens: Vec<Vec<usize>> = vec![Vec::new(); doc.segmentation.clauses.len()];
    for (t_idx, tok) in doc.tokens.iter().enumerate() {
        if tok.span.end_char > chars.len() || is_token_opaque(&opaque_spans, tok.span) {
            continue;
        }
        for (c_idx, cl) in doc.segmentation.clauses.iter().enumerate() {
            if tok.sentence == cl.sentence && span_contains(cl.span, tok.span) {
                token_clause[t_idx] = Some(c_idx);
                clause_tokens[c_idx].push(t_idx);
                break;
            }
        }
    }

    let mut events = Vec::new();
    let mut unknown = Vec::new();
    let mut clause_has_event = vec![false; doc.segmentation.clauses.len()];

    for (c_idx, cl) in doc.segmentation.clauses.iter().enumerate() {
        if is_span_opaque(&opaque_spans, cl.span) {
            continue;
        }
        let tok_indices = &clause_tokens[c_idx];
        if tok_indices.is_empty() {
            continue;
        }

        let sent_idx = cl.sentence;
        let para_idx = doc
            .segmentation
            .sentences
            .get(sent_idx)
            .map_or(0, |s| s.paragraph);
        let has_syntax = sentence_has_syntax.get(sent_idx).copied().unwrap_or(false);

        // Partition non-punctuation tokens in the clause by zone (typically 1 zone, or
        // Narrative + DirectQuote when an inline quote sits mid-clause).
        let zones = clause_zones(dialogues, doc, tok_indices);

        for zone in zones {
            let zone_toks: Vec<usize> = tok_indices
                .iter()
                .copied()
                .filter(|&ti| {
                    let tok = &doc.tokens[ti];
                    tok.upos != Some(Upos::Punct) && zone_for_span(dialogues, tok.span) == zone
                })
                .collect();
            if zone_toks.is_empty() {
                continue;
            }

            let pred_selection = select_predicate(
                &chars,
                doc,
                &zone_toks,
                &token_clause,
                c_idx,
                model_syntax_attempted,
                &syntax_by_tok,
                &low_conf_pos,
                &low_conf_syntax,
                &vocab_actors,
                &vocab_places,
                &vocab_objects,
            );

            let pred_tok = match pred_selection {
                PredicatePick::Selected(ti) => ti,
                PredicatePick::AbstainLowConfidence(detail) => {
                    unknown.push(NarrativeUnknown {
                        kind: NarrativeUnknownKind::PredicateLowConfidence,
                        chapter,
                        span: cl.span,
                        detail,
                    });
                    continue;
                }
                PredicatePick::None => continue,
            };

            let raw_pred_span = doc.tokens[pred_tok].span;
            let raw_pred_surf = span_text(&chars, raw_pred_span);
            let pred_span = if !model_syntax_attempted
                && let Some(pron_prefix) = glued_pronoun_prefix(&raw_pred_surf, raw_pred_span)
            {
                Span {
                    start_char: pron_prefix.end_char,
                    end_char: raw_pred_span.end_char,
                }
            } else {
                raw_pred_span
            };
            let disposal = detect_disposal(&chars, doc, &zone_toks, pred_tok);
            let passive = detect_passive(&chars, doc, &zone_toks, pred_tok, &syntax_by_tok);

            let mut arguments = if has_syntax {
                extract_args_with_syntax(
                    &chars,
                    doc,
                    vocab,
                    &zone_toks,
                    &token_clause,
                    c_idx,
                    pred_tok,
                    &syntax_by_tok,
                    &vocab_actors,
                    &vocab_places,
                    &vocab_objects,
                )
            } else if !model_syntax_attempted {
                extract_args_no_syntax(
                    &chars,
                    doc,
                    vocab,
                    &zone_toks,
                    pred_tok,
                    disposal,
                    passive,
                    &vocab_actors,
                    &vocab_places,
                    &vocab_objects,
                )
            } else {
                Vec::new()
            };

            apply_disposal_and_passive_realignment(
                &chars,
                doc,
                vocab,
                &zone_toks,
                pred_tok,
                disposal,
                passive,
                has_syntax,
                &syntax_by_tok,
                &vocab_actors,
                &vocab_places,
                &vocab_objects,
                &mut arguments,
            );

            dedup_and_order_arguments(&mut arguments, pred_span);

            let is_disposal = disposal.is_some();
            let (factuality, factuality_cue) = classify_factuality(
                &chars,
                doc,
                &zone_toks,
                cl.span,
                sent_idx,
                zone,
                is_disposal,
            );

            let event_idx = events.len();
            clause_has_event[c_idx] = true;
            events.push(CandidateEvent {
                event_id: format!("c{chapter:04}-e{event_idx:04}"),
                chapter,
                paragraph: para_idx,
                sentence: sent_idx,
                clause: c_idx,
                clause_span: cl.span,
                predicate_token: pred_tok,
                predicate_span: pred_span,
                zone,
                factuality,
                factuality_cue,
                arguments,
                temporal_anchors: Vec::new(),
            });
        }
    }

    attach_temporal_anchors(&chars, doc, dialogues, &clause_has_event, &mut events);

    EventExtraction { events, unknown }
}

enum PredicatePick {
    Selected(usize),
    AbstainLowConfidence(String),
    None,
}

#[derive(Clone, Copy)]
struct DisposalFrame {
    coverb_tok: usize,
    patient_start_tok: usize,
    patient_end_tok: usize,
}

#[derive(Clone, Copy)]
struct PassiveFrame {
    marker_tok: usize,
    agent_start_tok: Option<usize>,
    agent_end_tok: Option<usize>,
}

fn is_token_opaque(opaque_spans: &[Span], span: Span) -> bool {
    opaque_spans.iter().any(|o| spans_overlap(*o, span))
}

fn is_span_opaque(opaque_spans: &[Span], span: Span) -> bool {
    opaque_spans.iter().any(|o| span_contains(*o, span))
}

fn clause_zones(
    dialogues: &[DialogueSpan],
    doc: &Document,
    tok_indices: &[usize],
) -> Vec<ContextZone> {
    let mut zones = Vec::new();
    for &ti in tok_indices {
        let tok = &doc.tokens[ti];
        if tok.upos == Some(Upos::Punct) {
            continue;
        }
        let z = zone_for_span(dialogues, tok.span);
        if !zones.contains(&z) {
            zones.push(z);
        }
    }
    zones
}

#[allow(clippy::too_many_arguments)]
fn select_predicate(
    chars: &[char],
    doc: &Document,
    zone_toks: &[usize],
    token_clause: &[Option<usize>],
    clause_idx: usize,
    model_syntax_attempted: bool,
    syntax_by_tok: &[Option<&SyntaxToken>],
    low_conf_pos: &[Span],
    low_conf_syntax: &[Span],
    vocab_actors: &[(Span, String)],
    vocab_places: &[(Span, String)],
    vocab_objects: &[(Span, String)],
) -> PredicatePick {
    let mut pred_candidates: Vec<usize> = Vec::new();
    let mut suppressed_candidates: Vec<usize> = Vec::new();

    for (pos_in_zone, &ti) in zone_toks.iter().enumerate() {
        let tok = &doc.tokens[ti];
        if !matches!(tok.upos, Some(Upos::Verb | Upos::Adj)) {
            continue;
        }
        let surf = span_text(chars, tok.span);
        if is_pseudo_predicate(&surf) {
            continue;
        }
        // Check `low_confidence` suppression when syntax was expected.
        let is_low_syntax = model_syntax_attempted
            && syntax_by_tok[ti].is_none()
            && low_conf_syntax.iter().any(|s| spans_overlap(*s, tok.span));
        if is_low_syntax {
            suppressed_candidates.push(ti);
            continue;
        }
        // Skip pre-nominal `acl` relative clause modifiers if another predicate exists.
        if is_acl_modifier(
            chars,
            doc,
            zone_toks,
            pos_in_zone,
            ti,
            token_clause,
            clause_idx,
            syntax_by_tok,
        ) {
            continue;
        }
        pred_candidates.push(ti);
    }

    if !pred_candidates.is_empty() {
        return PredicatePick::Selected(choose_best_predicate(
            chars,
            doc,
            &pred_candidates,
            token_clause,
            clause_idx,
            syntax_by_tok,
        ));
    }

    // Check if any token in the clause was suppressed by `low_confidence` POS or syntax.
    let low_conf_hits: Vec<usize> = zone_toks
        .iter()
        .copied()
        .filter(|&ti| {
            let s = doc.tokens[ti].span;
            low_conf_pos.iter().any(|u| spans_overlap(*u, s))
                || low_conf_syntax.iter().any(|u| spans_overlap(*u, s))
        })
        .collect();
    if !suppressed_candidates.is_empty() || (!low_conf_hits.is_empty() && model_syntax_attempted) {
        return PredicatePick::AbstainLowConfidence(
            "candidate predicate suppressed by low_confidence diagnostic".to_owned(),
        );
    }
    // Also abstain when POS itself reported `LowConfidence` on candidate content tokens.
    if zone_toks.iter().any(|&ti| {
        doc.tokens[ti].upos.is_none()
            && low_conf_pos
                .iter()
                .any(|u| spans_overlap(*u, doc.tokens[ti].span))
    }) {
        return PredicatePick::AbstainLowConfidence(
            "candidate predicate UPOS suppressed by low_confidence diagnostic".to_owned(),
        );
    }

    // Fallback for `--no-model` rule-based mode (`doc.syntax` is empty and open-class
    // tokens have `upos: None` with `Reason::NoModel`).
    if !model_syntax_attempted {
        return fallback_predicate_no_model(
            chars,
            doc,
            zone_toks,
            vocab_actors,
            vocab_places,
            vocab_objects,
        );
    }

    PredicatePick::None
}

fn is_pseudo_predicate(surf: &str) -> bool {
    ADVERBIAL_PSEUDO_PREDICATES.contains(&surf)
}

#[allow(clippy::too_many_arguments)]
fn is_acl_modifier(
    chars: &[char],
    doc: &Document,
    zone_toks: &[usize],
    pos_in_zone: usize,
    ti: usize,
    token_clause: &[Option<usize>],
    clause_idx: usize,
    syntax_by_tok: &[Option<&SyntaxToken>],
) -> bool {
    if let Some(st) = syntax_by_tok.get(ti).and_then(|s| *s)
        && st.deprel == "acl"
        && let Some(head_idx) = st.head
        && token_clause.get(head_idx).copied().flatten() == Some(clause_idx)
        && matches!(
            doc.tokens[head_idx].upos,
            Some(Upos::Noun | Upos::Propn | Upos::Pron)
        )
    {
        // Only skip as `acl` if the clause has another Verb/Adj candidate.
        let has_other = zone_toks.iter().any(|&other| {
            other != ti
                && matches!(doc.tokens[other].upos, Some(Upos::Verb | Upos::Adj))
                && !is_pseudo_predicate(&span_text(chars, doc.tokens[other].span))
        });
        if has_other {
            return true;
        }
    }

    // Rule guard: `[Verb/Adj] + 的 + Noun ... [Later Verb]`
    if pos_in_zone + 2 < zone_toks.len() {
        let next_ti = zone_toks[pos_in_zone + 1];
        if span_text(chars, doc.tokens[next_ti].span) == "的" {
            let has_later_pred = zone_toks[(pos_in_zone + 2)..].iter().any(|&later| {
                matches!(doc.tokens[later].upos, Some(Upos::Verb | Upos::Adj))
                    && !is_pseudo_predicate(&span_text(chars, doc.tokens[later].span))
            });
            if has_later_pred {
                return true;
            }
        }
    }
    false
}

fn choose_best_predicate(
    chars: &[char],
    doc: &Document,
    candidates: &[usize],
    token_clause: &[Option<usize>],
    clause_idx: usize,
    syntax_by_tok: &[Option<&SyntaxToken>],
) -> usize {
    if candidates.len() == 1 {
        return candidates[0];
    }

    // Prefer a candidate whose syntax head is `None` (`root`) or outside this clause,
    // or that is a `Verb` over `Adj`, and prefer main verb over modal/aux/manner modifier.
    let mut best = candidates[0];
    let mut best_score = i32::MIN;
    for &ti in candidates {
        let tok = &doc.tokens[ti];
        let surf = span_text(chars, tok.span);
        let mut score = 0i32;

        if tok.upos == Some(Upos::Verb) {
            score += 40;
        } else if tok.upos == Some(Upos::Adj) {
            score += 10;
        }

        if let Some(st) = syntax_by_tok.get(ti).and_then(|s| *s) {
            if st.head.is_none() || st.deprel == "root" {
                score += 100;
            } else if let Some(h) = st.head {
                if token_clause.get(h).copied().flatten() != Some(clause_idx) {
                    score += 80;
                } else if matches!(st.deprel.as_str(), "advcl" | "amod" | "advmod") {
                    score -= 30;
                } else if st.deprel == "xcomp" {
                    // Prefer complement action over control/modal verb (e.g. `打算離開` -> `離開`).
                    let h_surf = span_text(chars, doc.tokens[h].span);
                    if is_modal_or_aux_like(&h_surf) {
                        score += 20;
                    } else {
                        score -= 10;
                    }
                }
            }
        }

        if is_modal_or_aux_like(&surf) {
            score -= 25;
        }
        if is_passive_or_coverb_verb(&surf) {
            score -= 20;
        }

        if score > best_score {
            best_score = score;
            best = ti;
        }
    }
    best
}

fn is_modal_or_aux_like(surf: &str) -> bool {
    let lower = surf.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "打算"
            | "準備"
            | "准备"
            | "想要"
            | "欲"
            | "能"
            | "可以"
            | "應該"
            | "应该"
            | "必須"
            | "必须"
            | "能夠"
            | "能够"
            | "plan"
            | "want"
            | "try"
            | "is"
            | "am"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "have"
            | "has"
            | "had"
            | "do"
            | "does"
            | "did"
    )
}

fn is_passive_or_coverb_verb(surf: &str) -> bool {
    matches!(surf, "被" | "遭" | "由" | "將" | "将" | "把")
}

fn fallback_predicate_no_model(
    chars: &[char],
    doc: &Document,
    zone_toks: &[usize],
    vocab_actors: &[(Span, String)],
    vocab_places: &[(Span, String)],
    vocab_objects: &[(Span, String)],
) -> PredicatePick {
    let mut skip_pp_nominal = false;
    let mut skip_det_nominal = false;
    let mut eligible: Vec<usize> = Vec::new();
    let mut after_subject_or_marker: Vec<usize> = Vec::new();
    let mut seen_initial_subject = false;
    let mut seen_preverbal_marker = false;

    for (pos, &ti) in zone_toks.iter().enumerate() {
        let tok = &doc.tokens[ti];
        let surf = span_text(chars, tok.span);
        let lower = surf.to_ascii_lowercase();

        // Glued pronoun+verb token (e.g. `他端`) contains the verb immediately after the pronoun.
        if glued_pronoun_prefix(&surf, tok.span).is_some() {
            eligible.push(ti);
            after_subject_or_marker.push(ti);
            continue;
        }

        let is_coverb_or_passive = matches!(surf.as_str(), "將" | "将" | "把" | "被" | "遭" | "由")
            && !(surf == "把"
                && pos > 0
                && matches!(
                    doc.tokens[zone_toks[pos - 1]].upos,
                    Some(Upos::Num | Upos::Det)
                ));
        if is_coverb_or_passive {
            // Coverbs (`將`/`把`) and passive markers (`被`/`遭`/`由`) always precede the
            // matrix predicate; any preceding token was a subject or modifier, not the verb.
            eligible.clear();
            after_subject_or_marker.clear();
            skip_pp_nominal = true;
            seen_preverbal_marker = true;
            continue;
        }
        if is_pp_marker(&surf, &lower) {
            skip_pp_nominal = true;
            if eligible.is_empty() {
                seen_preverbal_marker = true;
            }
            continue;
        }

        if tok.upos == Some(Upos::Det) {
            if !eligible.is_empty() || skip_pp_nominal {
                skip_det_nominal = true;
            }
            continue;
        }

        if is_known_non_predicate_token(
            chars,
            doc,
            tok,
            &surf,
            &lower,
            vocab_actors,
            vocab_places,
            vocab_objects,
        ) {
            let is_actor_or_pron = tok.upos == Some(Upos::Pron)
                || tok.upos == Some(Upos::Propn)
                || is_personal_pronoun(&surf)
                || vocab_actors
                    .iter()
                    .any(|(s, _)| spans_overlap(*s, tok.span))
                || doc
                    .entities
                    .iter()
                    .any(|e| e.kind == EntityKind::Name && spans_overlap(e.span, tok.span));
            if is_actor_or_pron && eligible.is_empty() {
                seen_initial_subject = true;
            }
            skip_pp_nominal = false;
            skip_det_nominal = false;
            continue;
        }

        if is_pseudo_predicate(&surf) || is_modal_or_aux_like(&surf) {
            continue;
        }

        // Skip nominal inside a `Det` phrase after a verb or PP (e.g. `the wooden door`, `the room`).
        if skip_det_nominal {
            // Keep skipping contiguous modifiers/nouns until end of NP (at most until `Det`/`Adp`).
            continue;
        }

        // Skip nominal argument immediately inside a coverb/PP (`從袖中`, `將玉佩`, `被狂風`, `into room`).
        if skip_pp_nominal {
            let next_is_loc_tail = zone_toks.get(pos + 1).is_some_and(|&nxt| {
                matches!(
                    span_text(chars, doc.tokens[nxt].span).as_str(),
                    "中" | "裡" | "裏" | "内" | "內" | "上" | "下" | "旁" | "前" | "後"
                )
            });
            let is_vocab_known = vocab_objects
                .iter()
                .any(|(s, _)| spans_overlap(*s, tok.span))
                || vocab_places
                    .iter()
                    .any(|(s, _)| spans_overlap(*s, tok.span))
                || vocab_actors
                    .iter()
                    .any(|(s, _)| spans_overlap(*s, tok.span));
            let has_later_content = zone_toks[(pos + 1)..].iter().any(|&later_ti| {
                let lt = &doc.tokens[later_ti];
                let ls = span_text(chars, lt.span);
                let ll = ls.to_ascii_lowercase();
                !is_known_non_predicate_token(
                    chars,
                    doc,
                    lt,
                    &ls,
                    &ll,
                    vocab_actors,
                    vocab_places,
                    vocab_objects,
                ) && !is_pseudo_predicate(&ls)
                    && !matches!(
                        ls.as_str(),
                        "中" | "裡" | "裏" | "内" | "內" | "上" | "下" | "旁"
                    )
            });
            if !eligible.is_empty() || next_is_loc_tail || is_vocab_known || has_later_content {
                skip_pp_nominal = false;
                continue;
            }
        }
        skip_pp_nominal = false;

        // Skip `的`-modifier noun/adj when followed by `的` + nominal + verb.
        if pos + 2 < zone_toks.len()
            && span_text(chars, doc.tokens[zone_toks[pos + 1]].span) == "的"
        {
            continue;
        }

        eligible.push(ti);
        if seen_initial_subject || seen_preverbal_marker {
            after_subject_or_marker.push(ti);
        }
    }

    if let Some(&first_after) = after_subject_or_marker.first() {
        return PredicatePick::Selected(first_after);
    }

    // If no explicit marker/known-actor preceded, check if clause starts with an untagged subject
    // followed by a verb + object (e.g., `狂風吹開木門` or `Alice opened the door`).
    if eligible.len() >= 2 {
        let first = eligible[0];
        let second = eligible[1];
        let first_pos = zone_toks.iter().position(|&t| t == first).unwrap_or(0);
        let followed_by_aspect = zone_toks.get(first_pos + 1).is_some_and(|&nxt| {
            matches!(
                span_text(chars, doc.tokens[nxt].span).as_str(),
                "了" | "著" | "着" | "過" | "过"
            )
        });
        if followed_by_aspect {
            return PredicatePick::Selected(first);
        }
        let second_pos = zone_toks.iter().position(|&t| t == second).unwrap_or(0);
        let second_followed_by_det = zone_toks.get(second_pos + 1).is_some_and(|&nxt| {
            doc.tokens[nxt].upos == Some(Upos::Det)
                || doc.tokens[nxt].upos == Some(Upos::Adp)
                || vocab_objects
                    .iter()
                    .any(|(s, _)| spans_overlap(*s, doc.tokens[nxt].span))
                || vocab_places
                    .iter()
                    .any(|(s, _)| spans_overlap(*s, doc.tokens[nxt].span))
        });
        if second_followed_by_det
            || (eligible.len() >= 3
                && doc.tokens[first].lang == faberline_nlp::doc::Lang::En
                && chars[doc.tokens[first].span.start_char].is_uppercase())
        {
            return PredicatePick::Selected(second);
        }
    }

    if let Some(&first) = eligible.first() {
        PredicatePick::Selected(first)
    } else {
        PredicatePick::None
    }
}

#[allow(clippy::too_many_arguments)]
fn is_known_non_predicate_token(
    _chars: &[char],
    doc: &Document,
    tok: &faberline_nlp::doc::Token,
    surf: &str,
    lower: &str,
    vocab_actors: &[(Span, String)],
    vocab_places: &[(Span, String)],
    vocab_objects: &[(Span, String)],
) -> bool {
    if matches!(
        tok.upos,
        Some(
            Upos::Pron
                | Upos::Det
                | Upos::Num
                | Upos::Adp
                | Upos::Adv
                | Upos::Cconj
                | Upos::Sconj
                | Upos::Part
                | Upos::Intj
                | Upos::Sym
                | Upos::Punct
                | Upos::X
                | Upos::Propn
                | Upos::Aux
        )
    ) {
        return true;
    }
    if is_personal_pronoun(surf) {
        return true;
    }
    if matches!(
        surf,
        "與" | "与"
            | "和"
            | "跟"
            | "及"
            | "以及"
            | "把"
            | "將"
            | "将"
            | "被"
            | "遭"
            | "由"
            | "沒"
            | "没"
            | "沒有"
            | "没有"
            | "未"
            | "未曾"
            | "並未"
            | "并未"
            | "若是"
            | "如果"
            | "假如"
            | "倘若"
            | "萬一"
            | "万一"
            | "即將"
            | "即将"
            | "將要"
            | "将要"
            | "不久"
            | "不禁"
            | "不由得"
            | "不錯"
            | "不错"
            | "裡"
            | "裏"
            | "中"
            | "內"
            | "内"
            | "後"
            | "后"
            | "前"
            | "為"
            | "为"
            | "所"
            | "着"
            | "著"
            | "過"
            | "过"
            | "呢"
            | "嗎"
    ) {
        return true;
    }
    if matches!(
        lower,
        "did"
            | "do"
            | "does"
            | "have"
            | "has"
            | "had"
            | "to"
            | "so"
            | "when"
            | "while"
            | "as"
            | "than"
            | "that"
            | "this"
            | "these"
            | "those"
            | "there"
            | "here"
            | "then"
    ) {
        return true;
    }
    if doc.entities.iter().any(|e| {
        matches!(
            e.kind,
            EntityKind::Name
                | EntityKind::Date
                | EntityKind::RelativeDate
                | EntityKind::Time
                | EntityKind::Quantity
                | EntityKind::Number
                | EntityKind::Url
                | EntityKind::Email
        ) && spans_overlap(e.span, tok.span)
    }) {
        return true;
    }
    if vocab_actors
        .iter()
        .any(|(s, _)| spans_overlap(*s, tok.span))
        || vocab_places
            .iter()
            .any(|(s, _)| spans_overlap(*s, tok.span))
        || vocab_objects
            .iter()
            .any(|(s, _)| spans_overlap(*s, tok.span))
    {
        return true;
    }
    false
}

fn is_pp_marker(surf: &str, lower: &str) -> bool {
    PP_MARKERS_CJK.contains(&surf) || PP_MARKERS_EN.contains(&lower)
}

#[allow(clippy::too_many_arguments)]
fn extract_args_with_syntax(
    chars: &[char],
    doc: &Document,
    vocab: &WorkVocabulary,
    zone_toks: &[usize],
    token_clause: &[Option<usize>],
    clause_idx: usize,
    pred_tok: usize,
    syntax_by_tok: &[Option<&SyntaxToken>],
    vocab_actors: &[(Span, String)],
    vocab_places: &[(Span, String)],
    vocab_objects: &[(Span, String)],
) -> Vec<EventArgument> {
    // Enforce Argon/Carbon Layer-2 Core Argument Gate:
    // tok_clause == head_clause && arc.score.milli() >= 350 &&
    // matches!(head.upos, Some(Verb | Adj)) && head not in adverb-proxy stoplist &&
    // matches!(dep.upos, Some(Noun | Propn | Pron)).
    let mut direct_args: Vec<(usize, EventRole, Span, ArgumentBasis, u16)> = Vec::new();
    let mut nsubj_indices: Vec<usize> = Vec::new();

    for &ti in zone_toks {
        if ti == pred_tok {
            continue;
        }
        let Some(st) = syntax_by_tok.get(ti).and_then(|s| *s) else {
            continue;
        };
        let Some(head_idx) = st.head else {
            continue;
        };
        if token_clause.get(ti).copied().flatten() != Some(clause_idx)
            || token_clause.get(head_idx).copied().flatten() != Some(clause_idx)
        {
            continue;
        }
        if st.score.milli() < 350 {
            continue;
        }
        let head_tok = &doc.tokens[head_idx];
        if !matches!(head_tok.upos, Some(Upos::Verb | Upos::Adj)) {
            continue;
        }
        let head_surf = span_text(chars, head_tok.span);
        if ADVERB_PROXY_HEADS.contains(&head_surf.as_str()) {
            continue;
        }
        let dep_tok = &doc.tokens[ti];
        if !matches!(dep_tok.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron)) {
            continue;
        }
        // Only accept arguments attached to the selected predicate (or an intra-clause verbal head).
        if head_idx != pred_tok && !matches!(st.deprel.as_str(), "nsubj" | "obj" | "obl" | "iobj") {
            continue;
        }

        let role = match st.deprel.as_str() {
            "nsubj" | "csubj" => Some(EventRole::Agent),
            "nsubj:pass" => Some(EventRole::Patient),
            "obj" => Some(EventRole::Patient),
            "iobj" | "obl" | "obl:patient" | "obl:agent" | "dislocated" => Some(EventRole::Oblique),
            _ => None,
        };
        let Some(role) = role else {
            continue;
        };

        let basis = if st.deprel == "nsubj:pass" || st.deprel == "obl:agent" {
            ArgumentBasis::PassiveBei
        } else {
            ArgumentBasis::DirectSyntax
        };
        let mut arg_span = dep_tok.span;
        let mut arg_basis = basis;

        // Merge contiguous intra-clause nominal modifiers (`compound`, `nmod`, `flat`, `amod`, `appos`).
        let merged = merge_contiguous_nominal_modifiers(
            chars,
            doc,
            zone_toks,
            ti,
            token_clause,
            clause_idx,
            syntax_by_tok,
        );
        if merged != arg_span {
            arg_span = merged;
            if arg_basis == ArgumentBasis::DirectSyntax {
                arg_basis = ArgumentBasis::CompoundMerged;
            }
        }

        if role == EventRole::Agent {
            nsubj_indices.push(direct_args.len());
        }
        direct_args.push((ti, role, arg_span, arg_basis, st.score.milli()));
    }

    // Merge contiguous `nsubj` entries or deduplicate non-contiguous double `nsubj`
    // (keep closest valid subject to `pred_tok`).
    let pred_start = doc.tokens[pred_tok].span.start_char;
    if nsubj_indices.len() >= 2 {
        let mut merged_agents: Vec<(usize, EventRole, Span, ArgumentBasis, u16)> = Vec::new();
        for &idx in &nsubj_indices {
            let item = direct_args[idx];
            if let Some(last) = merged_agents.last_mut()
                && are_contiguous_spans(chars, last.2, item.2)
            {
                last.2 = Span {
                    start_char: last.2.start_char.min(item.2.start_char),
                    end_char: last.2.end_char.max(item.2.end_char),
                };
                last.3 = ArgumentBasis::CompoundMerged;
                last.4 = last.4.min(item.4);
                continue;
            }
            merged_agents.push(item);
        }
        // Keep only the closest `Agent` to the predicate if multiple non-contiguous remain.
        let best_agent = merged_agents
            .into_iter()
            .min_by_key(|a| a.2.end_char.abs_diff(pred_start));

        let nsubj_set: HashSet<usize> = nsubj_indices.into_iter().collect();
        let mut filtered = Vec::new();
        for (i, item) in direct_args.into_iter().enumerate() {
            if !nsubj_set.contains(&i) {
                filtered.push(item);
            }
        }
        if let Some(best) = best_agent {
            filtered.push(best);
        }
        direct_args = filtered;
    }

    direct_args
        .into_iter()
        .map(|(ti, role, span, basis, score_milli)| EventArgument {
            role,
            span,
            token: Some(ti),
            canonical: lookup_canonical(
                chars,
                vocab,
                span,
                vocab_actors,
                vocab_places,
                vocab_objects,
            ),
            basis,
            score_milli,
        })
        .collect()
}

fn merge_contiguous_nominal_modifiers(
    chars: &[char],
    doc: &Document,
    zone_toks: &[usize],
    head_ti: usize,
    token_clause: &[Option<usize>],
    clause_idx: usize,
    syntax_by_tok: &[Option<&SyntaxToken>],
) -> Span {
    let mut span = doc.tokens[head_ti].span;
    // Iteratively absorb adjacent modifiers attached to `head_ti`.
    let mut changed = true;
    while changed {
        changed = false;
        for &ti in zone_toks {
            let tok = &doc.tokens[ti];
            if span_contains(span, tok.span) {
                continue;
            }
            let Some(st) = syntax_by_tok.get(ti).and_then(|s| *s) else {
                continue;
            };
            if st.head != Some(head_ti)
                || st.score.milli() < 350
                || token_clause.get(ti).copied().flatten() != Some(clause_idx)
                || !matches!(
                    st.deprel.as_str(),
                    "compound" | "nmod" | "flat" | "amod" | "appos"
                )
            {
                continue;
            }
            if are_contiguous_spans(chars, tok.span, span)
                || are_contiguous_spans(chars, span, tok.span)
            {
                span = Span {
                    start_char: span.start_char.min(tok.span.start_char),
                    end_char: span.end_char.max(tok.span.end_char),
                };
                changed = true;
            }
        }
    }
    span
}

fn are_contiguous_spans(chars: &[char], left: Span, right: Span) -> bool {
    if left.end_char > right.start_char {
        return false;
    }
    (left.end_char..right.start_char).all(|i| chars[i].is_whitespace())
}

#[allow(clippy::too_many_arguments)]
fn extract_args_no_syntax(
    chars: &[char],
    doc: &Document,
    vocab: &WorkVocabulary,
    zone_toks: &[usize],
    pred_tok: usize,
    disposal: Option<DisposalFrame>,
    passive: Option<PassiveFrame>,
    vocab_actors: &[(Span, String)],
    vocab_places: &[(Span, String)],
    vocab_objects: &[(Span, String)],
) -> Vec<EventArgument> {
    let mut args = Vec::new();
    let pred_pos = zone_toks.iter().position(|&t| t == pred_tok).unwrap_or(0);

    // If `pred_tok` has a glued pronoun prefix (e.g. `他端`), record the pronoun as `Agent`.
    let pred_raw_span = doc.tokens[pred_tok].span;
    let pred_raw_surf = span_text(chars, pred_raw_span);
    if let Some(pron_span) = glued_pronoun_prefix(&pred_raw_surf, pred_raw_span) {
        args.push(EventArgument {
            role: EventRole::Agent,
            span: pron_span,
            token: Some(pred_tok),
            canonical: None,
            basis: ArgumentBasis::DirectSyntax,
            score_milli: 0,
        });
    }

    // Pre-predicate window (before disposal/passive marker if present, or before predicate).
    let pre_limit_pos = match (disposal, passive) {
        (Some(d), _) => d.coverb_tok.min(pred_pos),
        (None, Some(p)) => p.marker_tok.min(pred_pos),
        (None, None) => pred_pos,
    };

    let mut pre_pp_active = false;
    let mut pre_agents: Vec<(usize, Span)> = Vec::new();
    let mut i = 0usize;
    while i < pre_limit_pos {
        let ti = zone_toks[i];
        let tok = &doc.tokens[ti];
        let surf = span_text(chars, tok.span);
        let lower = surf.to_ascii_lowercase();

        if is_pp_marker(&surf, &lower) {
            pre_pp_active = true;
            i += 1;
            continue;
        }
        if matches!(
            surf.as_str(),
            "與" | "与" | "和" | "跟" | "及" | "以及" | "and"
        ) {
            i += 1;
            continue;
        }
        if is_nominal_candidate_no_syntax(
            chars,
            doc,
            tok,
            &surf,
            &lower,
            vocab_actors,
            vocab_places,
            vocab_objects,
        ) {
            // Collect contiguous nominal phrase if not separated by PP/conjunction.
            let (np_span, next_i) = collect_np_span(
                chars,
                doc,
                zone_toks,
                i,
                pre_limit_pos,
                vocab_actors,
                vocab_places,
                vocab_objects,
            );
            let next_surf = zone_toks
                .get(next_i)
                .map(|&nti| span_text(chars, doc.tokens[nti].span));
            if doc.tokens[pred_tok].lang == faberline_nlp::doc::Lang::Ja
                && let Some(ref p_surf) = next_surf
                && matches!(
                    p_surf.as_str(),
                    "は" | "が" | "を" | "に" | "で" | "へ" | "から"
                )
            {
                let role = match p_surf.as_str() {
                    "は" | "が" => EventRole::Agent,
                    "を" => EventRole::Patient,
                    _ => EventRole::Oblique,
                };
                args.push(EventArgument {
                    role,
                    span: np_span,
                    token: Some(ti),
                    canonical: lookup_canonical(
                        chars,
                        vocab,
                        np_span,
                        vocab_actors,
                        vocab_places,
                        vocab_objects,
                    ),
                    basis: ArgumentBasis::DirectSyntax,
                    score_milli: 0,
                });
                i = next_i + 1;
                continue;
            }
            if pre_pp_active {
                args.push(EventArgument {
                    role: EventRole::Oblique,
                    span: np_span,
                    token: Some(ti),
                    canonical: lookup_canonical(
                        chars,
                        vocab,
                        np_span,
                        vocab_actors,
                        vocab_places,
                        vocab_objects,
                    ),
                    basis: ArgumentBasis::DirectSyntax,
                    score_milli: 0,
                });
                pre_pp_active = false;
            } else {
                pre_agents.push((ti, np_span));
            }
            i = next_i;
            continue;
        }
        i += 1;
    }

    if let Some((ti, span)) = pre_agents.last().copied() {
        let role = if passive.is_some() {
            EventRole::Patient
        } else {
            EventRole::Agent
        };
        let basis = if passive.is_some() {
            ArgumentBasis::PassiveBei
        } else {
            ArgumentBasis::DirectSyntax
        };
        args.push(EventArgument {
            role,
            span,
            token: Some(ti),
            canonical: lookup_canonical(
                chars,
                vocab,
                span,
                vocab_actors,
                vocab_places,
                vocab_objects,
            ),
            basis,
            score_milli: 0,
        });
    }

    // Post-predicate window.
    let mut post_pp_active = false;
    let mut post_patient_assigned = disposal.is_some();
    let mut j = pred_pos + 1;
    while j < zone_toks.len() {
        let ti = zone_toks[j];
        let tok = &doc.tokens[ti];
        let surf = span_text(chars, tok.span);
        let lower = surf.to_ascii_lowercase();

        if matches!(
            surf.as_str(),
            "了" | "著"
                | "着"
                | "過"
                | "过"
                | "下去"
                | "下來"
                | "下来"
                | "起來"
                | "起来"
                | "出來"
                | "出来"
                | "進去"
                | "进去"
                | "進來"
                | "进来"
                | "上"
                | "下"
                | "開"
                | "开"
                | "完"
                | "好"
        ) || tok.upos == Some(Upos::Det)
            || tok.upos == Some(Upos::Part)
        {
            j += 1;
            continue;
        }
        if is_pp_marker(&surf, &lower) {
            post_pp_active = true;
            j += 1;
            continue;
        }
        if is_nominal_candidate_no_syntax(
            chars,
            doc,
            tok,
            &surf,
            &lower,
            vocab_actors,
            vocab_places,
            vocab_objects,
        ) {
            let (np_span, next_j) = collect_np_span(
                chars,
                doc,
                zone_toks,
                j,
                zone_toks.len(),
                vocab_actors,
                vocab_places,
                vocab_objects,
            );
            let role = if post_pp_active || post_patient_assigned {
                EventRole::Oblique
            } else {
                post_patient_assigned = true;
                EventRole::Patient
            };
            post_pp_active = false;
            args.push(EventArgument {
                role,
                span: np_span,
                token: Some(ti),
                canonical: lookup_canonical(
                    chars,
                    vocab,
                    np_span,
                    vocab_actors,
                    vocab_places,
                    vocab_objects,
                ),
                basis: ArgumentBasis::DirectSyntax,
                score_milli: 0,
            });
            j = next_j;
            continue;
        }
        j += 1;
    }

    args
}

#[allow(clippy::too_many_arguments)]
fn is_nominal_candidate_no_syntax(
    _chars: &[char],
    doc: &Document,
    tok: &faberline_nlp::doc::Token,
    surf: &str,
    lower: &str,
    vocab_actors: &[(Span, String)],
    vocab_places: &[(Span, String)],
    vocab_objects: &[(Span, String)],
) -> bool {
    if matches!(tok.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron)) {
        return true;
    }
    if is_personal_pronoun(surf) {
        return true;
    }
    if doc
        .entities
        .iter()
        .any(|e| e.kind == EntityKind::Name && spans_overlap(e.span, tok.span))
    {
        return true;
    }
    if vocab_actors
        .iter()
        .any(|(s, _)| spans_overlap(*s, tok.span))
        || vocab_places
            .iter()
            .any(|(s, _)| spans_overlap(*s, tok.span))
        || vocab_objects
            .iter()
            .any(|(s, _)| spans_overlap(*s, tok.span))
    {
        return true;
    }
    if tok.upos.is_none()
        && !is_pseudo_predicate(surf)
        && !is_modal_or_aux_like(surf)
        && !is_passive_or_coverb_verb(surf)
        && !is_pp_marker(surf, lower)
        && !matches!(
            surf,
            "的" | "了"
                | "著"
                | "着"
                | "過"
                | "过"
                | "不"
                | "沒"
                | "没"
                | "沒有"
                | "没有"
                | "未"
                | "很"
                | "太"
                | "更"
                | "最"
                | "卻"
                | "却"
                | "低聲"
                | "低声"
                | "緩緩"
                | "缓缓"
                | "輕輕"
                | "轻轻"
                | "冷冷"
                | "默默"
        )
        && !NEGATION_EN_CUES.contains(&lower)
        && !HYPOTHETICAL_EN_CUES.contains(&lower)
        && !MODAL_FUTURE_EN_CUES.contains(&lower)
        && !doc.entities.iter().any(|e| {
            matches!(
                e.kind,
                EntityKind::Date | EntityKind::RelativeDate | EntityKind::Time | EntityKind::Number
            ) && spans_overlap(e.span, tok.span)
        })
    {
        return true;
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn collect_np_span(
    chars: &[char],
    doc: &Document,
    zone_toks: &[usize],
    start_pos: usize,
    limit_pos: usize,
    vocab_actors: &[(Span, String)],
    vocab_places: &[(Span, String)],
    vocab_objects: &[(Span, String)],
) -> (Span, usize) {
    let first_tok = &doc.tokens[zone_toks[start_pos]];
    let mut span = first_tok.span;

    // If a vocabulary/entity span starts here and covers multiple tokens, snap to it.
    for (v_span, _) in vocab_actors
        .iter()
        .chain(vocab_places.iter())
        .chain(vocab_objects.iter())
    {
        if v_span.start_char == span.start_char && v_span.end_char > span.end_char {
            span.end_char = v_span.end_char;
        }
    }
    for ent in &doc.entities {
        if ent.kind == EntityKind::Name
            && ent.span.start_char == span.start_char
            && ent.span.end_char > span.end_char
        {
            span.end_char = ent.span.end_char;
        }
    }

    let mut pos = start_pos + 1;
    while pos < limit_pos {
        let ti = zone_toks[pos];
        let tok = &doc.tokens[ti];
        if span_contains(span, tok.span) {
            pos += 1;
            continue;
        }
        let surf = span_text(chars, tok.span);
        // Stop merging at locative postpositions only if they're single-char suffixes like `中`/`裡`
        // that attach to the NP (`袖中` -> absorb `中` into `袖中`), or stop at verbs/adverbs/PPs.
        if matches!(surf.as_str(), "中" | "裡" | "裏" | "內" | "内")
            && are_contiguous_spans(chars, span, tok.span)
        {
            span.end_char = tok.span.end_char;
            pos += 1;
            continue;
        }
        // Only merge adjacent English/CJK modifiers (`wooden` + `door`) when contiguous and non-entity-boundary.
        if tok.lang == faberline_nlp::doc::Lang::En
            && tok.upos.is_none()
            && are_contiguous_spans(chars, span, tok.span)
            && !is_known_stop_for_np(&surf)
        {
            span.end_char = tok.span.end_char;
            pos += 1;
            continue;
        }
        break;
    }
    (span, pos)
}

fn is_known_stop_for_np(surf: &str) -> bool {
    let lower = surf.to_ascii_lowercase();
    is_pp_marker(surf, &lower)
        || is_pseudo_predicate(surf)
        || is_modal_or_aux_like(surf)
        || matches!(lower.as_str(), "and" | "or" | "but" | "the" | "a" | "an")
}

fn detect_disposal(
    chars: &[char],
    doc: &Document,
    zone_toks: &[usize],
    pred_tok: usize,
) -> Option<DisposalFrame> {
    let pred_pos = zone_toks.iter().position(|&t| t == pred_tok)?;
    for pos in 0..pred_pos {
        let ti = zone_toks[pos];
        let surf = span_text(chars, doc.tokens[ti].span);
        if !matches!(surf.as_str(), "將" | "将" | "把") {
            continue;
        }
        // Avoid classifier `一把劍` (`Num` or `Det` immediately before `把`).
        if surf == "把"
            && pos > 0
            && matches!(
                doc.tokens[zone_toks[pos - 1]].upos,
                Some(Upos::Num | Upos::Det)
            )
        {
            continue;
        }
        // Check that at least one nominal sits between coverb (`pos`) and predicate (`pred_pos`).
        let start = pos + 1;
        if start >= pred_pos {
            continue;
        }
        let mut np_end = start;
        while np_end < pred_pos {
            let n_ti = zone_toks[np_end];
            let n_tok = &doc.tokens[n_ti];
            let n_surf = span_text(chars, n_tok.span);
            if n_tok.upos == Some(Upos::Adv) || is_pseudo_predicate(&n_surf) {
                break;
            }
            np_end += 1;
        }
        if np_end > start {
            // Confirm at least one token in `start..np_end` can be a nominal (not pure adverb/negation).
            let has_nominal = (start..np_end).any(|k| {
                let t = &doc.tokens[zone_toks[k]];
                let s = span_text(chars, t.span);
                !matches!(t.upos, Some(Upos::Adv | Upos::Part | Upos::Adp | Upos::Aux))
                    && !NEGATION_CJK_CUES.contains(&s.as_str())
                    && !MODAL_FUTURE_CJK_CUES.contains(&s.as_str())
            });
            if has_nominal {
                return Some(DisposalFrame {
                    coverb_tok: pos,
                    patient_start_tok: start,
                    patient_end_tok: np_end,
                });
            }
        }
    }
    None
}

fn detect_passive(
    chars: &[char],
    doc: &Document,
    zone_toks: &[usize],
    pred_tok: usize,
    syntax_by_tok: &[Option<&SyntaxToken>],
) -> Option<PassiveFrame> {
    let pred_pos = zone_toks.iter().position(|&t| t == pred_tok)?;
    for pos in 0..pred_pos {
        let ti = zone_toks[pos];
        let surf = span_text(chars, doc.tokens[ti].span);
        if matches!(surf.as_str(), "被" | "遭" | "由") {
            let mut agent_end = pos + 1;
            while agent_end < pred_pos {
                let n_tok = &doc.tokens[zone_toks[agent_end]];
                let n_surf = span_text(chars, n_tok.span);
                if matches!(
                    n_tok.upos,
                    Some(Upos::Adv | Upos::Verb | Upos::Adp | Upos::Sconj | Upos::Cconj)
                ) || is_pseudo_predicate(&n_surf)
                    || NEGATION_CJK_CUES.contains(&n_surf.as_str())
                    || MODAL_FUTURE_CJK_CUES.contains(&n_surf.as_str())
                {
                    break;
                }
                agent_end += 1;
            }
            let (agent_start_tok, agent_end_tok) = if agent_end > pos + 1 {
                (Some(pos + 1), Some(agent_end))
            } else {
                (None, None)
            };
            return Some(PassiveFrame {
                marker_tok: pos,
                agent_start_tok,
                agent_end_tok,
            });
        }
        if surf == "為" || surf == "为" {
            // Look for `為 <NP> 所 <Verb>`
            if let Some(suo_pos) = ((pos + 1)..pred_pos)
                .find(|&k| span_text(chars, doc.tokens[zone_toks[k]].span) == "所")
            {
                let (a_s, a_e) = if suo_pos > pos + 1 {
                    (Some(pos + 1), Some(suo_pos))
                } else {
                    (None, None)
                };
                return Some(PassiveFrame {
                    marker_tok: pos,
                    agent_start_tok: a_s,
                    agent_end_tok: a_e,
                });
            }
        }
    }

    // Japanese passive: `れる` / `られる` / `れ` / `られ` / `れた` / `られた` / `れて` / `られて` aux.
    if doc.tokens[pred_tok].lang == faberline_nlp::doc::Lang::Ja {
        let ja_pass_pos = zone_toks.iter().enumerate().find_map(|(pos, &ti)| {
            let surf = span_text(chars, doc.tokens[ti].span);
            let is_pass_aux = matches!(
                surf.as_str(),
                "れる" | "られる" | "れ" | "られ" | "れた" | "られた" | "れて" | "られて"
            ) && (pos == pred_pos + 1
                || syntax_by_tok
                    .get(ti)
                    .and_then(|s| *s)
                    .is_some_and(|st| st.deprel == "aux" && st.head == Some(pred_tok)));
            is_pass_aux.then_some(pos)
        });
        if let Some(marker_pos) = ja_pass_pos {
            // Find an animate `に`/`によって`/`により`-marked agent in the clause.
            let mut agent_range: Option<(usize, usize)> = None;
            for (pos, &ti) in zone_toks.iter().enumerate() {
                if pos >= pred_pos {
                    break;
                }
                let is_ja_obl_agent = syntax_by_tok.get(ti).and_then(|s| *s).is_some_and(|st| {
                    st.head == Some(pred_tok)
                        && matches!(st.deprel.as_str(), "obl" | "iobj" | "obl:agent")
                        && zone_toks.iter().any(|&ct| {
                            syntax_by_tok.get(ct).and_then(|s| *s).is_some_and(|cst| {
                                cst.head == Some(ti)
                                    && cst.deprel == "case"
                                    && matches!(
                                        span_text(chars, doc.tokens[ct].span).as_str(),
                                        "に" | "によって" | "により"
                                    )
                            })
                        })
                });
                if is_ja_obl_agent && is_ja_animate_token(chars, doc, ti) {
                    agent_range = Some((pos, pos + 1));
                    break;
                }
            }
            return Some(PassiveFrame {
                marker_tok: marker_pos,
                agent_start_tok: agent_range.map(|r| r.0),
                agent_end_tok: agent_range.map(|r| r.1),
            });
        }
    }

    // Also detect syntax `nsubj:pass` in English/Chinese.
    let has_pass_subj = zone_toks.iter().any(|&ti| {
        syntax_by_tok
            .get(ti)
            .and_then(|s| *s)
            .is_some_and(|st| st.deprel == "nsubj:pass" && st.score.milli() >= 350)
    });
    if has_pass_subj {
        return Some(PassiveFrame {
            marker_tok: pred_pos,
            agent_start_tok: None,
            agent_end_tok: None,
        });
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn apply_disposal_and_passive_realignment(
    chars: &[char],
    doc: &Document,
    vocab: &WorkVocabulary,
    zone_toks: &[usize],
    pred_tok: usize,
    disposal: Option<DisposalFrame>,
    passive: Option<PassiveFrame>,
    has_syntax: bool,
    syntax_by_tok: &[Option<&SyntaxToken>],
    vocab_actors: &[(Span, String)],
    vocab_places: &[(Span, String)],
    vocab_objects: &[(Span, String)],
    arguments: &mut Vec<EventArgument>,
) {
    if let Some(disp) = disposal {
        let coverb_span = doc.tokens[zone_toks[disp.coverb_tok]].span;
        let disp_span = Span {
            start_char: doc.tokens[zone_toks[disp.patient_start_tok]]
                .span
                .start_char,
            end_char: doc.tokens[zone_toks[disp.patient_end_tok - 1]]
                .span
                .end_char,
        };
        let head_ti = zone_toks[disp.patient_end_tok - 1];
        let mut matched_existing = false;

        for arg in arguments.iter_mut() {
            if spans_overlap(arg.span, disp_span) {
                arg.role = EventRole::Patient;
                arg.basis = ArgumentBasis::DisposalJiang;
                matched_existing = true;
            } else if arg.span.start_char >= doc.tokens[pred_tok].span.end_char
                && arg.role == EventRole::Patient
            {
                // Post-verbal nominal when disposal `將`/`把` supplies `Patient` is `Oblique`.
                arg.role = EventRole::Oblique;
            } else if arg.span.end_char <= coverb_span.start_char && arg.role != EventRole::Oblique
            {
                arg.role = EventRole::Agent;
            }
        }
        if !matched_existing {
            let score = if has_syntax {
                syntax_by_tok
                    .get(head_ti)
                    .and_then(|s| *s)
                    .map_or(0, |st| st.score.milli())
            } else {
                0
            };
            arguments.push(EventArgument {
                role: EventRole::Patient,
                span: disp_span,
                token: Some(head_ti),
                canonical: lookup_canonical(
                    chars,
                    vocab,
                    disp_span,
                    vocab_actors,
                    vocab_places,
                    vocab_objects,
                ),
                basis: ArgumentBasis::DisposalJiang,
                score_milli: score,
            });
        }
    }

    if let Some(pass) = passive {
        let marker_span = doc.tokens[zone_toks[pass.marker_tok]].span;
        // Any pre-marker `Agent` becomes `Patient` (`PassiveBei`).
        for arg in arguments.iter_mut() {
            if arg.span.end_char <= marker_span.start_char && arg.role == EventRole::Agent {
                arg.role = EventRole::Patient;
                arg.basis = ArgumentBasis::PassiveBei;
            }
        }
        if let (Some(a_start), Some(a_end)) = (pass.agent_start_tok, pass.agent_end_tok) {
            let agent_span = Span {
                start_char: doc.tokens[zone_toks[a_start]].span.start_char,
                end_char: doc.tokens[zone_toks[a_end - 1]].span.end_char,
            };
            let head_ti = zone_toks[a_end - 1];
            let mut matched = false;
            for arg in arguments.iter_mut() {
                if spans_overlap(arg.span, agent_span) {
                    if arg.span.start_char < agent_span.start_char
                        && spans_overlap(arg.span, marker_span)
                    {
                        arg.span = agent_span;
                        arg.token = Some(head_ti);
                    }
                    arg.role = EventRole::Agent;
                    arg.basis = ArgumentBasis::PassiveBei;
                    matched = true;
                }
            }
            if !matched {
                let score = if has_syntax {
                    syntax_by_tok
                        .get(head_ti)
                        .and_then(|s| *s)
                        .map_or(0, |st| st.score.milli())
                } else {
                    0
                };
                arguments.push(EventArgument {
                    role: EventRole::Agent,
                    span: agent_span,
                    token: Some(head_ti),
                    canonical: lookup_canonical(
                        chars,
                        vocab,
                        agent_span,
                        vocab_actors,
                        vocab_places,
                        vocab_objects,
                    ),
                    basis: ArgumentBasis::PassiveBei,
                    score_milli: score,
                });
            }
        }
    }
}

fn dedup_and_order_arguments(arguments: &mut Vec<EventArgument>, pred_span: Span) {
    // Keep highest-priority entry per overlapping span, then ensure at most one Agent.
    arguments.sort_by_key(|a| (a.span.start_char, a.span.end_char));
    let mut deduped: Vec<EventArgument> = Vec::new();
    for arg in arguments.drain(..) {
        if let Some(prev) = deduped.last_mut()
            && spans_overlap(prev.span, arg.span)
        {
            if basis_priority(arg.basis) > basis_priority(prev.basis) {
                *prev = arg;
            }
            continue;
        }
        deduped.push(arg);
    }

    // If multiple Agents remain, keep the closest to the predicate.
    let agent_positions: Vec<usize> = deduped
        .iter()
        .enumerate()
        .filter(|(_, a)| a.role == EventRole::Agent)
        .map(|(i, _)| i)
        .collect();
    if agent_positions.len() > 1 {
        let best = *agent_positions
            .iter()
            .min_by_key(|&&i| deduped[i].span.end_char.abs_diff(pred_span.start_char))
            .unwrap();
        let mut filtered = Vec::with_capacity(deduped.len());
        for (i, arg) in deduped.into_iter().enumerate() {
            if arg.role != EventRole::Agent || i == best {
                filtered.push(arg);
            }
        }
        *arguments = filtered;
    } else {
        *arguments = deduped;
    }
}

fn basis_priority(b: ArgumentBasis) -> u8 {
    match b {
        ArgumentBasis::DisposalJiang | ArgumentBasis::PassiveBei => 4,
        ArgumentBasis::CompoundMerged => 3,
        ArgumentBasis::PronounResolved => 2,
        ArgumentBasis::DirectSyntax => 1,
        ArgumentBasis::IntraSentenceTopicChain => 0,
    }
}

fn lookup_canonical(
    chars: &[char],
    vocab: &WorkVocabulary,
    span: Span,
    vocab_actors: &[(Span, String)],
    vocab_places: &[(Span, String)],
    vocab_objects: &[(Span, String)],
) -> Option<String> {
    let surf = span_text(chars, span);
    if let Some(c) = vocab.lookup_any(&surf) {
        return Some(c.to_owned());
    }
    for (s, c) in vocab_actors
        .iter()
        .chain(vocab_objects.iter())
        .chain(vocab_places.iter())
    {
        if *s == span || span_contains(span, *s) {
            return Some(c.clone());
        }
    }
    if let Some(canon) = crate::normalize_ja_actor_key(vocab, &surf) {
        return Some(canon);
    }
    None
}

fn is_ja_animate_token(chars: &[char], doc: &Document, tok_idx: usize) -> bool {
    let tok = &doc.tokens[tok_idx];
    if doc.entities.iter().any(|e| {
        matches!(
            e.kind,
            EntityKind::Date | EntityKind::RelativeDate | EntityKind::Time
        ) && spans_overlap(e.span, tok.span)
    }) {
        return false;
    }
    if doc
        .entities
        .iter()
        .any(|e| e.kind == EntityKind::Name && spans_overlap(e.span, tok.span))
    {
        return true;
    }
    if matches!(tok.upos, Some(Upos::Propn | Upos::Pron)) {
        return true;
    }
    let surf = span_text(chars, tok.span);
    crate::strip_ja_honorific(&surf) != surf || crate::is_personal_pronoun(&surf)
}

fn classify_factuality(
    chars: &[char],
    doc: &Document,
    zone_toks: &[usize],
    clause_span: Span,
    sent_idx: usize,
    zone: ContextZone,
    is_disposal: bool,
) -> (Factuality, Option<Span>) {
    if matches!(zone, ContextZone::DirectQuote { .. }) {
        return (Factuality::Quoted, None);
    }

    // 1. Hypothetical / conditional protasis cue (intra-clause or governing `Condition` protasis).
    if let Some(cue) = find_hypothetical_cue(chars, doc, zone_toks, clause_span) {
        return (Factuality::Hypothetical, Some(cue));
    }
    for rel in &doc.relations {
        if rel.kind == RelationKind::Condition
            && spans_overlap(rel.arg2, clause_span)
            && let Some(sent) = doc.segmentation.sentences.get(sent_idx)
            && span_contains(sent.span, rel.arg1)
            && !matches!(
                span_text(chars, rel.connector).as_str(),
                "否則" | "否则" | "otherwise"
            )
        {
            return (Factuality::Hypothetical, Some(rel.connector));
        }
    }
    // Also detect intra-sentence `若是 ...，...便/就...` even if `doc.relations` didn't emit Condition.
    if let Some(sent) = doc.segmentation.sentences.get(sent_idx) {
        let prefix_span = Span {
            start_char: sent.span.start_char,
            end_char: clause_span.start_char.min(sent.span.end_char),
        };
        if prefix_span.start_char < prefix_span.end_char
            && let Some(cue) = find_hypothetical_cue_in_span(chars, doc, prefix_span)
        {
            return (Factuality::Hypothetical, Some(cue));
        }
    }

    // 2. Negation cues (careful not to trigger on non-negated idioms).
    if let Some(cue) = find_negation_cue(chars, doc, zone_toks) {
        return (Factuality::Negated, Some(cue));
    }

    // 3. Modal / future cues (excluding disposal `將 + NP + V`).
    if let Some(cue) = find_modal_future_cue(chars, doc, zone_toks, is_disposal) {
        return (Factuality::ModalOrFuture, Some(cue));
    }

    (Factuality::Actual, None)
}

fn find_hypothetical_cue(
    chars: &[char],
    doc: &Document,
    zone_toks: &[usize],
    clause_span: Span,
) -> Option<Span> {
    for &ti in zone_toks {
        let tok = &doc.tokens[ti];
        let surf = span_text(chars, tok.span);
        let lower = surf.to_ascii_lowercase();
        if HYPOTHETICAL_EN_CUES.contains(&lower.as_str()) {
            return Some(tok.span);
        }
        if HYPOTHETICAL_CJK_CUES.contains(&surf.as_str())
            || HYPOTHETICAL_JA_TOKEN_CUES.contains(&surf.as_str())
        {
            if surf == "若" && is_non_conditional_ruo(chars, tok.span, clause_span) {
                continue;
            }
            return Some(tok.span);
        }
        if tok.lang == faberline_nlp::doc::Lang::Ja
            && matches!(
                tok.upos,
                Some(Upos::Verb | Upos::Adj | Upos::Aux | Upos::Sconj)
            )
            && surf != "ながら"
            && (surf.ends_with("たら") || surf.ends_with("れば"))
        {
            return Some(tok.span);
        }
    }
    find_hypothetical_cue_in_span(chars, doc, clause_span)
}

fn find_hypothetical_cue_in_span(chars: &[char], doc: &Document, span: Span) -> Option<Span> {
    let text = span_text(chars, span);
    for &cue in HYPOTHETICAL_CJK_CUES {
        if cue.chars().count() >= 2
            && let Some(byte_pos) = text.find(cue)
        {
            let char_off = text[..byte_pos].chars().count();
            return Some(Span {
                start_char: span.start_char + char_off,
                end_char: span.start_char + char_off + cue.chars().count(),
            });
        }
    }
    for tok in &doc.tokens {
        if !span_contains(span, tok.span) {
            continue;
        }
        let surf = span_text(chars, tok.span);
        let lower = surf.to_ascii_lowercase();
        if HYPOTHETICAL_EN_CUES.contains(&lower.as_str())
            || (surf == "若" && !is_non_conditional_ruo(chars, tok.span, span))
        {
            return Some(tok.span);
        }
    }
    None
}

fn is_non_conditional_ruo(chars: &[char], tok_span: Span, clause_span: Span) -> bool {
    let window = Span {
        start_char: tok_span
            .start_char
            .saturating_sub(2)
            .max(clause_span.start_char),
        end_char: (tok_span.end_char + 3).min(clause_span.end_char),
    };
    let w_text = span_text(chars, window);
    HYPOTHETICAL_NON_COND_WORDS
        .iter()
        .any(|w| w_text.contains(w))
}

fn find_negation_cue(chars: &[char], doc: &Document, zone_toks: &[usize]) -> Option<Span> {
    for (pos, &ti) in zone_toks.iter().enumerate() {
        let tok = &doc.tokens[ti];
        let surf = span_text(chars, tok.span);
        let lower = surf.to_ascii_lowercase();

        if NEGATION_EN_CUES.contains(&lower.as_str()) {
            return Some(tok.span);
        }
        if tok.lang == faberline_nlp::doc::Lang::Ja
            && matches!(tok.upos, Some(Upos::Verb | Upos::Aux))
            && (surf.ends_with("ない")
                || surf.ends_with("なかった")
                || surf.ends_with("ません")
                || surf.ends_with("ませんでした"))
        {
            return Some(tok.span);
        }
        if NEGATION_CJK_CUES.contains(&surf.as_str()) {
            // Guard against idioms (`不久`, `不錯`, `不由得`, `不禁`, `沒多久`, `未來`, `未必`, etc.)
            // whether jieba kept them as one token or split `不` + `久`.
            let look_ahead = Span {
                start_char: tok.span.start_char,
                end_char: (tok.span.start_char + 4).min(chars.len()),
            };
            let ahead_text = span_text(chars, look_ahead);
            if NON_NEGATED_IDIOMS
                .iter()
                .any(|idiom| ahead_text.starts_with(idiom))
            {
                continue;
            }
            // Also check if `不`/`沒` is inside a longer idiom token starting earlier.
            if pos > 0 {
                let prev_span = doc.tokens[zone_toks[pos - 1]].span;
                let joined = span_text(
                    chars,
                    Span {
                        start_char: prev_span.start_char,
                        end_char: tok.span.end_char,
                    },
                );
                if NON_NEGATED_IDIOMS
                    .iter()
                    .any(|idiom| joined.ends_with(idiom))
                {
                    continue;
                }
            }
            return Some(tok.span);
        }
    }
    None
}

fn find_modal_future_cue(
    chars: &[char],
    doc: &Document,
    zone_toks: &[usize],
    is_disposal: bool,
) -> Option<Span> {
    for (pos, &ti) in zone_toks.iter().enumerate() {
        let tok = &doc.tokens[ti];
        let surf = span_text(chars, tok.span);
        let lower = surf.to_ascii_lowercase();

        if MODAL_FUTURE_EN_CUES.contains(&lower.as_str()) {
            return Some(tok.span);
        }
        if lower == "plan"
            && zone_toks.get(pos + 1).is_some_and(|&nxt| {
                span_text(chars, doc.tokens[nxt].span).eq_ignore_ascii_case("to")
            })
        {
            return Some(Span {
                start_char: tok.span.start_char,
                end_char: doc.tokens[zone_toks[pos + 1]].span.end_char,
            });
        }
        if MODAL_FUTURE_CJK_CUES.contains(&surf.as_str()) {
            return Some(tok.span);
        }
        if tok.lang == faberline_nlp::doc::Lang::Ja
            && matches!(tok.upos, Some(Upos::Verb | Upos::Aux | Upos::Adj))
            && (surf.ends_with("たい")
                || surf.ends_with("たかった")
                || surf.ends_with("だろう")
                || surf.ends_with("でしょう"))
        {
            return Some(tok.span);
        }
        // Handle multi-char cues split across tokens (e.g. `即` + `將`).
        for &cue in MODAL_FUTURE_CJK_CUES {
            if cue.chars().count() >= 2 {
                let cue_len = cue.chars().count();
                if tok.span.start_char + cue_len <= chars.len() {
                    let candidate = span_text(
                        chars,
                        Span {
                            start_char: tok.span.start_char,
                            end_char: tok.span.start_char + cue_len,
                        },
                    );
                    if candidate == cue {
                        return Some(Span {
                            start_char: tok.span.start_char,
                            end_char: tok.span.start_char + cue_len,
                        });
                    }
                }
            }
        }
        if !is_disposal && matches!(surf.as_str(), "將" | "将") {
            return Some(tok.span);
        }
    }
    None
}

fn attach_temporal_anchors(
    chars: &[char],
    doc: &Document,
    dialogues: &[DialogueSpan],
    clause_has_event: &[bool],
    events: &mut [CandidateEvent],
) {
    // Collect all temporal entities from `doc.entities`.
    let temporal_ents: Vec<(usize, &faberline_nlp::doc::Entity)> = doc
        .entities
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            matches!(
                e.kind,
                EntityKind::Date | EntityKind::RelativeDate | EntityKind::Time
            ) || (e.kind == EntityKind::Quantity
                && e.unit
                    .as_deref()
                    .is_some_and(|u| TEMPORAL_QUANTITY_UNITS.contains(&u)))
        })
        .collect();

    if temporal_ents.is_empty() {
        return;
    }

    // Track which entities belong to an eventless clause in each sentence (e.g. leading `數日後，`).
    let mut sent_shared_anchors: Vec<Vec<TemporalAnchor>> =
        vec![Vec::new(); doc.segmentation.sentences.len()];
    for &(ent_idx, ent) in &temporal_ents {
        if !matches!(zone_for_span(dialogues, ent.span), ContextZone::Narrative) {
            continue;
        }
        let hosting_clause = doc
            .segmentation
            .clauses
            .iter()
            .enumerate()
            .find(|(_, c)| c.sentence == ent.sentence && spans_overlap(c.span, ent.span))
            .map(|(ci, _)| ci);
        let in_eventless_clause = hosting_clause.is_some_and(|ci| !clause_has_event[ci]);
        if in_eventless_clause && ent.sentence < sent_shared_anchors.len() {
            sent_shared_anchors[ent.sentence].push(TemporalAnchor {
                entity_index: ent_idx,
                kind: ent.kind,
                span: ent.span,
                value: ent.value.clone(),
                unit: ent.unit.clone(),
            });
        }
    }

    for ev in events.iter_mut() {
        let mut local = Vec::new();
        for &(ent_idx, ent) in &temporal_ents {
            if ent.sentence != ev.sentence || !spans_overlap(ev.clause_span, ent.span) {
                continue;
            }
            let ent_zone = zone_for_span(dialogues, ent.span);
            if ev.zone == ContextZone::Narrative && ent_zone != ContextZone::Narrative {
                continue;
            }
            if ev.zone != ContextZone::Narrative && ent_zone != ev.zone {
                continue;
            }
            local.push(TemporalAnchor {
                entity_index: ent_idx,
                kind: ent.kind,
                span: ent.span,
                value: ent.value.clone(),
                unit: ent.unit.clone(),
            });
        }
        if !local.is_empty() {
            ev.temporal_anchors = local;
        } else if ev.zone == ContextZone::Narrative
            && let Some(shared) = sent_shared_anchors.get(ev.sentence)
        {
            ev.temporal_anchors = shared.clone();
        }
        if ev.zone == ContextZone::Narrative {
            attach_ja_fallback_temporal_anchors(chars, doc, ev);
        }
    }
}

fn attach_ja_fallback_temporal_anchors(chars: &[char], doc: &Document, ev: &mut CandidateEvent) {
    let Some(sent) = doc.segmentation.sentences.get(ev.sentence) else {
        return;
    };
    if sent.lang != faberline_nlp::doc::Lang::Ja {
        return;
    }
    const JA_POINT_IN_TIME: &[&str] = &[
        "翌朝",
        "その夜",
        "翌晩",
        "次の日",
        "翌日",
        "今朝",
        "今夜",
        "昨夜",
    ];
    for tok in &doc.tokens {
        if tok.sentence != ev.sentence || !spans_overlap(ev.clause_span, tok.span) {
            continue;
        }
        if ev
            .temporal_anchors
            .iter()
            .any(|a| spans_overlap(a.span, tok.span))
        {
            continue;
        }
        let s = span_text(chars, tok.span);
        if JA_POINT_IN_TIME.contains(&s.as_str()) {
            ev.temporal_anchors.push(TemporalAnchor {
                entity_index: doc.entities.len(),
                kind: EntityKind::RelativeDate,
                span: tok.span,
                value: Some(s),
                unit: None,
            });
        }
    }
}
