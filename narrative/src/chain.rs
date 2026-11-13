//! Actor mention extraction, conservative intra-sentence topic-chain (`流水句`)
//! inheritance, and `Never Guess` pronoun/pro-drop ambiguity diagnostics.

use crate::dialogue::zone_for_span;
use crate::schema::{
    ActorMention, ArgumentBasis, CandidateEvent, ContextZone, DialogueSpan, EventArgument,
    EventRole, Factuality, MentionKind, NarrativeUnknown, NarrativeUnknownKind, WorkVocabulary,
};
use crate::{
    collect_actor_candidates, glued_pronoun_prefix, is_inside_opaque, is_personal_pronoun,
    span_contains, span_text, spans_overlap,
};
use faberline_nlp::Span;
use faberline_nlp::doc::Document;
use std::collections::{HashMap, HashSet};

/// Output of actor mention extraction and intra-sentence chain resolution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChainExtraction {
    pub mentions: Vec<ActorMention>,
    pub unknown: Vec<NarrativeUnknown>,
}

const PARALLEL_SHIFT_MARKERS: &[&str] = &[
    "與此同時",
    "与此同时",
    "同時",
    "同时",
    "另一方面",
    "meanwhile",
];

/// Extracts actor mentions and applies conservative intra-sentence topic-chain and pronoun
/// resolution under the `Never Guess` policy.
pub fn resolve_chains(
    chapter: u32,
    text: &str,
    doc: &Document,
    vocab: &WorkVocabulary,
    dialogues: &[DialogueSpan],
    events: &mut [CandidateEvent],
) -> ChainExtraction {
    let chars: Vec<char> = text.chars().collect();
    let opaque_spans: Vec<Span> = doc.segmentation.opaque.iter().map(|o| o.span).collect();
    let actor_cands = collect_actor_candidates(chapter, text, &chars, doc, vocab, &opaque_spans);

    let mut mentions = Vec::new();
    let mut unknown = Vec::new();

    // 1. ExplicitName mentions from `EntityKind::Name` / `WorkVocabulary::actors` / `Upos::Propn`.
    for cand in &actor_cands {
        let (sent, clause) = locate_sent_clause(doc, cand.span);
        mentions.push(ActorMention {
            chapter,
            sentence: sent,
            clause,
            span: cand.span,
            kind: MentionKind::ExplicitName,
            canonical: cand.canonical.clone(),
            resolved_antecedent: None,
        });
    }

    // 2. Pronoun mentions from tokens (including classical forms `其`, `老夫`, `在下`, `眾人`, etc.).
    for tok in &doc.tokens {
        if tok.span.end_char > chars.len() || is_inside_opaque(&opaque_spans, tok.span.start_char) {
            continue;
        }
        let surf = span_text(&chars, tok.span);
        let pron_span = if is_personal_pronoun(&surf) {
            tok.span
        } else if let Some(prefix) = glued_pronoun_prefix(&surf, tok.span) {
            prefix
        } else {
            continue;
        };
        if mentions.iter().any(|m| spans_overlap(m.span, pron_span)) {
            continue;
        }
        let (sent, clause) = locate_sent_clause(doc, pron_span);
        mentions.push(ActorMention {
            chapter,
            sentence: sent,
            clause,
            span: pron_span,
            kind: MentionKind::Pronoun,
            canonical: None,
            resolved_antecedent: None,
        });
    }

    // 3. NominalPhrase mentions for animate/role nouns acting as `Agent` in events.
    for ev in events.iter() {
        for arg in &ev.arguments {
            if arg.role != EventRole::Agent {
                continue;
            }
            if mentions.iter().any(|m| spans_overlap(m.span, arg.span)) {
                continue;
            }
            mentions.push(ActorMention {
                chapter,
                sentence: ev.sentence,
                clause: ev.clause,
                span: arg.span,
                kind: MentionKind::NominalPhrase,
                canonical: arg.canonical.clone(),
                resolved_antecedent: None,
            });
        }
    }

    mentions.sort_by_key(|m| (m.span.start_char, m.span.end_char));

    // 4. Resolve singular third-person narrative pronouns using the active window
    // (same sentence or immediately preceding sentence in the same paragraph).
    let mut ambiguous_pronoun_spans: Vec<Span> = Vec::new();
    let mut resolved_pronouns: HashMap<(usize, usize), (Span, Option<String>)> = HashMap::new();

    for m_idx in 0..mentions.len() {
        if mentions[m_idx].kind != MentionKind::Pronoun {
            continue;
        }
        let m_span = mentions[m_idx].span;
        if !matches!(zone_for_span(dialogues, m_span), ContextZone::Narrative) {
            continue;
        }
        let surf = span_text(&chars, m_span);
        if !is_third_person_singular_pronoun(&surf) {
            continue;
        }

        let m_sent = mentions[m_idx].sentence;
        let m_para = sentence_paragraph(doc, m_sent);

        // Collect preceding distinct narrative actor antecedents in active window.
        let mut window_cands: Vec<(Span, Option<String>, String)> = Vec::new();
        for prev in &mentions[..m_idx] {
            if prev.span.end_char > m_span.start_char {
                continue;
            }
            if !matches!(zone_for_span(dialogues, prev.span), ContextZone::Narrative) {
                continue;
            }
            if matches!(prev.kind, MentionKind::ZeroPronounProDrop) {
                continue;
            }
            if prev.kind == MentionKind::Pronoun {
                continue;
            }
            let prev_para = sentence_paragraph(doc, prev.sentence);
            if prev_para != m_para {
                continue;
            }
            if prev.sentence != m_sent && prev.sentence + 1 != m_sent {
                continue;
            }
            let key = prev
                .canonical
                .clone()
                .unwrap_or_else(|| span_text(&chars, prev.span));
            if let Some(existing) = window_cands.iter_mut().find(|(_, _, k)| k == &key) {
                // Keep most recent span for that actor identity.
                existing.0 = prev.span;
            } else {
                window_cands.push((prev.span, prev.canonical.clone(), key));
            }
        }

        if window_cands.len() >= 2 {
            ambiguous_pronoun_spans.push(m_span);
            unknown.push(NarrativeUnknown {
                kind: NarrativeUnknownKind::PronounAmbiguous,
                chapter,
                span: m_span,
                detail: format!(
                    "pronoun '{}' has {} competing actor candidates in active window",
                    surf,
                    window_cands.len()
                ),
            });
        } else if window_cands.len() == 1 {
            let (ant_span, ant_canon, _) = window_cands.remove(0);
            mentions[m_idx].resolved_antecedent = Some(ant_span);
            mentions[m_idx].canonical = ant_canon.clone();
            resolved_pronouns.insert((m_span.start_char, m_span.end_char), (ant_span, ant_canon));
        }
    }

    // Apply resolved pronouns onto matching event arguments.
    for ev in events.iter_mut() {
        for arg in &mut ev.arguments {
            if let Some((_, ant_canon)) =
                resolved_pronouns.get(&(arg.span.start_char, arg.span.end_char))
            {
                arg.canonical = ant_canon.clone();
                if arg.basis == ArgumentBasis::DirectSyntax {
                    arg.basis = ArgumentBasis::PronounResolved;
                }
            }
        }
    }

    // 5. Conservative Intra-Sentence Topic-Chain (`流水句`) Subject Inheritance.
    let mut events_by_sentence: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, ev) in events.iter().enumerate() {
        events_by_sentence.entry(ev.sentence).or_default().push(idx);
    }

    let mut extra_pro_drop_mentions: Vec<ActorMention> = Vec::new();
    let mut sent_keys: Vec<usize> = events_by_sentence.keys().copied().collect();
    sent_keys.sort_unstable();

    for sent_idx in sent_keys {
        let mut ev_indices = events_by_sentence.remove(&sent_idx).unwrap_or_default();
        ev_indices.sort_by_key(|&i| events[i].clause_span.start_char);

        let mut active_agent: Option<EventArgument> = None;
        let mut sentence_actors: Vec<String> = Vec::new();

        for &ev_idx in &ev_indices {
            let ev_zone = events[ev_idx].zone;
            let ev_fact = events[ev_idx].factuality;
            let ev_clause_span = events[ev_idx].clause_span;
            let ev_sentence = events[ev_idx].sentence;
            let ev_clause = events[ev_idx].clause;

            if ev_zone != ContextZone::Narrative || ev_fact != Factuality::Actual {
                continue;
            }

            // Collect explicit/nominal actors present inside this clause.
            let clause_actors = actors_in_clause(&chars, &mentions, ev_clause_span, dialogues);

            let clause_text = span_text(&chars, ev_clause_span);
            let is_parallel_shift = PARALLEL_SHIFT_MARKERS
                .iter()
                .any(|m| clause_text.to_ascii_lowercase().contains(m));
            let has_passive = events[ev_idx]
                .arguments
                .iter()
                .any(|a| a.basis == ArgumentBasis::PassiveBei);

            if is_parallel_shift {
                active_agent = None;
                sentence_actors.clear();
            }

            let existing_agent = events[ev_idx]
                .arguments
                .iter()
                .find(|a| a.role == EventRole::Agent)
                .cloned();

            if let Some(agent) = existing_agent {
                if has_passive {
                    // Passive clause introduces its own passive agent (or agentless passive);
                    // update active_agent only if clear single agent.
                    record_actors(&mut sentence_actors, &clause_actors);
                    if sentence_actors.len() == 1 {
                        active_agent = Some(agent);
                    } else {
                        active_agent = None;
                    }
                    continue;
                }

                let is_ambig_pronoun = ambiguous_pronoun_spans
                    .iter()
                    .any(|s| spans_overlap(*s, agent.span));
                record_actors(&mut sentence_actors, &clause_actors);

                if is_ambig_pronoun || clause_actors.len() >= 2 || sentence_actors.len() >= 2 {
                    active_agent = None;
                } else {
                    // If the agent was a resolved pronoun, propagate the antecedent span.
                    let mut propagated = agent.clone();
                    if let Some((ant_span, ant_canon)) =
                        resolved_pronouns.get(&(agent.span.start_char, agent.span.end_char))
                    {
                        propagated.span = *ant_span;
                        propagated.canonical = ant_canon.clone();
                    }
                    active_agent = Some(propagated);
                }
            } else {
                // Agentless `Narrative + Actual` clause.
                if is_parallel_shift || has_passive {
                    record_actors(&mut sentence_actors, &clause_actors);
                    active_agent = None;
                    continue;
                }

                if sentence_actors.len() >= 2 {
                    unknown.push(NarrativeUnknown {
                        kind: NarrativeUnknownKind::ProDropAmbiguous,
                        chapter,
                        span: ev_clause_span,
                        detail: format!(
                            "agentless narrative clause has {} competing preceding actors in sentence",
                            sentence_actors.len()
                        ),
                    });
                } else if sentence_actors.len() == 1
                    && let Some(ref ant) = active_agent
                {
                    let inherited = EventArgument {
                        role: EventRole::Agent,
                        span: ant.span,
                        token: ant.token,
                        canonical: ant.canonical.clone(),
                        basis: ArgumentBasis::IntraSentenceTopicChain,
                        score_milli: ant.score_milli,
                    };
                    events[ev_idx].arguments.push(inherited);
                    extra_pro_drop_mentions.push(ActorMention {
                        chapter,
                        sentence: ev_sentence,
                        clause: ev_clause,
                        span: Span {
                            start_char: ev_clause_span.start_char,
                            end_char: ev_clause_span.start_char,
                        },
                        kind: MentionKind::ZeroPronounProDrop,
                        canonical: ant.canonical.clone(),
                        resolved_antecedent: Some(ant.span),
                    });
                }

                record_actors(&mut sentence_actors, &clause_actors);
                if sentence_actors.len() >= 2 {
                    active_agent = None;
                }
            }
        }
    }

    mentions.extend(extra_pro_drop_mentions);
    mentions.sort_by_key(|m| (m.span.start_char, m.span.end_char, mention_rank(m.kind)));
    unknown.sort_by_key(|u| (u.span.start_char, u.span.end_char));

    ChainExtraction { mentions, unknown }
}

fn mention_rank(kind: MentionKind) -> u8 {
    match kind {
        MentionKind::ZeroPronounProDrop => 0,
        MentionKind::ExplicitName => 1,
        MentionKind::Pronoun => 2,
        MentionKind::NominalPhrase => 3,
    }
}

fn is_third_person_singular_pronoun(surf: &str) -> bool {
    let lower = surf.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "他" | "她"
            | "它"
            | "牠"
            | "其"
            | "he"
            | "him"
            | "his"
            | "she"
            | "her"
            | "hers"
            | "it"
            | "its"
    )
}

fn locate_sent_clause(doc: &Document, span: Span) -> (usize, usize) {
    for (c_idx, cl) in doc.segmentation.clauses.iter().enumerate() {
        if span_contains(cl.span, span) || spans_overlap(cl.span, span) {
            return (cl.sentence, c_idx);
        }
    }
    for (s_idx, sent) in doc.segmentation.sentences.iter().enumerate() {
        if span_contains(sent.span, span) || spans_overlap(sent.span, span) {
            return (s_idx, 0);
        }
    }
    (0, 0)
}

fn sentence_paragraph(doc: &Document, sent_idx: usize) -> usize {
    doc.segmentation
        .sentences
        .get(sent_idx)
        .map_or(0, |s| s.paragraph)
}

fn actors_in_clause(
    chars: &[char],
    mentions: &[ActorMention],
    clause_span: Span,
    dialogues: &[DialogueSpan],
) -> Vec<String> {
    let mut keys = Vec::new();
    let mut seen = HashSet::new();
    for m in mentions {
        if !span_contains(clause_span, m.span) {
            continue;
        }
        if !matches!(zone_for_span(dialogues, m.span), ContextZone::Narrative) {
            continue;
        }
        if matches!(m.kind, MentionKind::ZeroPronounProDrop) {
            continue;
        }
        if m.kind == MentionKind::Pronoun && m.resolved_antecedent.is_none() {
            let surf = span_text(chars, m.span);
            if is_third_person_singular_pronoun(&surf) {
                continue;
            }
        }
        let key = m
            .canonical
            .clone()
            .unwrap_or_else(|| span_text(chars, m.span));
        if seen.insert(key.clone()) {
            keys.push(key);
        }
    }
    keys
}

fn record_actors(target: &mut Vec<String>, additions: &[String]) {
    for k in additions {
        if !target.contains(k) {
            target.push(k.clone());
        }
    }
}
