//! Stage 4a: Narrative, discourse, and clause-window conditioning over `nlp.doc.v1` and
//! optional `nlp.narrative.v1` (`ChapterNarrative`).

use faberline_nlp::Span;
use faberline_nlp::doc::{Clause, Document, Lang, RelationKind};
#[cfg(feature = "narrative")]
use faberline_nlp_narrative::{ChapterNarrative, Factuality, MentionKind};

use crate::schema::{PronounClass, SentenceConditioning, TargetVariant, TranslationGlossary};

/// A clause-aligned sub-window of a sentence, together with the trailing separator text (if any)
/// between this window and the next window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClauseWindow {
    pub span: Span,
    pub trailing_sep: String,
    pub clause_indices: Vec<usize>,
}

/// Extract `SentenceConditioning` for `sentence_idx` from `doc` (`relations`) and optional
/// `narrative` (`dialogues`, `events`, `mentions`), cross-referencing `glossary` for pronoun class
/// and speaker register.
pub fn build_sentence_conditioning(
    sentence_idx: usize,
    sent_span: Span,
    doc: &Document,
    #[cfg(feature = "narrative")] narrative: Option<&ChapterNarrative>,
    #[cfg(not(feature = "narrative"))] _narrative: Option<&()>,
    glossary: &TranslationGlossary,
    target: TargetVariant,
) -> SentenceConditioning {
    let mut cond = SentenceConditioning::default();

    // 1. Discourse relation from `nlp.doc.v1` Layer 7 (`relations`).
    for rel in &doc.relations {
        let touches = spans_overlap(rel.arg1, sent_span) || spans_overlap(rel.arg2, sent_span);
        if touches {
            cond.discourse_relation = Some(relation_name(rel.kind).to_owned());
            break;
        }
    }

    // 2. Narrative conditioning (`dialogues`, `mentions`, `events`) when compiled and provided.
    #[cfg(feature = "narrative")]
    if let Some(nar) = narrative {
        // Dialogue speaker & speech register
        for dlg in &nar.dialogues {
            if spans_overlap(dlg.span, sent_span) {
                if let Some(sp) = &dlg.speaker_canonical {
                    cond.speaker = Some(sp.clone());
                    if let Some(actor) = glossary.find_actor(sp)
                        && actor.speech_register != crate::schema::SpeechRegister::Unspecified
                    {
                        cond.speech_register = Some(actor.speech_register);
                    }
                }
                break;
            }
        }

        // Zero-pronoun pro-drop & resolved pronoun antecedent
        for m in &nar.mentions {
            if m.sentence != sentence_idx {
                continue;
            }
            if matches!(
                m.kind,
                MentionKind::ZeroPronounProDrop | MentionKind::Pronoun
            ) && let Some(canon) = &m.canonical
            {
                if m.kind == MentionKind::ZeroPronounProDrop && cond.resolved_pro_drop.is_none() {
                    cond.resolved_pro_drop = Some(canon.clone());
                }
                if cond.pronoun_hint.is_none()
                    && let Some(actor) = glossary.find_actor(canon)
                {
                    if target.is_en() {
                        cond.pronoun_hint =
                            actor.pronoun_class.en_subject_pronoun().map(str::to_owned);
                    } else {
                        cond.pronoun_hint = actor.pronoun_class.control_code().map(str::to_owned);
                    }
                }
            }
        }

        // Primary non-trivial event factuality (`negated`, `hypothetical`, `modal_or_future`, else `actual`)
        let mut chosen_fact: Option<Factuality> = None;
        for ev in &nar.events {
            if ev.sentence != sentence_idx {
                continue;
            }
            match ev.factuality {
                Factuality::Negated | Factuality::Hypothetical | Factuality::ModalOrFuture => {
                    chosen_fact = Some(ev.factuality);
                    break;
                }
                other if chosen_fact.is_none() => {
                    chosen_fact = Some(other);
                }
                _ => {}
            }
        }
        if let Some(f) = chosen_fact {
            cond.factuality = Some(factuality_name(f).to_owned());
        }
    }

    #[cfg(not(feature = "narrative"))]
    {
        let _ = (sentence_idx, glossary, target);
    }

    // 3. Layer 6 `doc.zero_mentions` fallback when narrative did not resolve pro-drop,
    //    plus Japanese-to-Chinese factuality/register cues for Track B & Track A.
    if cond.resolved_pro_drop.is_none()
        && let Some(zm) = doc
            .zero_mentions
            .iter()
            .find(|z| z.sentence == sentence_idx)
        && let Some(head_tok) = doc.tokens.get(zm.antecedent.head_token)
    {
        let ant_text = head_tok
            .lemma
            .as_deref()
            .or(head_tok.canonical_variant.as_deref())
            .unwrap_or("");
        if !ant_text.is_empty() {
            cond.resolved_pro_drop = Some(ant_text.to_owned());
            if cond.pronoun_hint.is_none()
                && let Some(actor) = glossary.find_actor(ant_text)
            {
                if target.is_en() {
                    cond.pronoun_hint = actor.pronoun_class.en_subject_pronoun().map(str::to_owned);
                } else {
                    cond.pronoun_hint = actor.pronoun_class.control_code().map(str::to_owned);
                }
            }
        }
    }

    if target.is_zh()
        && doc
            .segmentation
            .sentences
            .get(sentence_idx)
            .is_some_and(|s| s.lang == Lang::Ja)
        && cond.factuality.is_none()
    {
        let has_neg = doc.tokens.iter().any(|t| {
            t.sentence == sentence_idx
                && matches!(
                    t.lemma.as_deref(),
                    Some("ない" | "ぬ" | "ん" | "ません" | "なかった")
                )
        });
        if has_neg {
            cond.factuality = Some("negated".to_owned());
        }
    }

    cond
}

/// Map a `PronounClass` or control-code hint (`"3sg.m"`, `"3sg.f"`, etc.) to a Chinese pronoun.
pub fn zh_pronoun_for_class(class: PronounClass) -> Option<&'static str> {
    match class {
        PronounClass::MasculineSingular => Some("他"),
        PronounClass::FeminineSingular => Some("她"),
        PronounClass::NeuterSingular => Some("它"),
        PronounClass::AnimatePlural => Some("他們"),
        PronounClass::FirstSingular => Some("我"),
        PronounClass::SecondSingular => Some("你"),
        PronounClass::Unspecified => None,
    }
}

/// Map a control-code `pronoun_hint` (`"3sg.m"`, `"3sg.f"`, etc.) to a Chinese pronoun.
pub fn zh_pronoun_for_hint(hint: &str) -> Option<&'static str> {
    match hint {
        "3sg.m" | "he" => Some("他"),
        "3sg.f" | "she" => Some("她"),
        "3sg.n" | "it" => Some("它"),
        "3pl" | "they" => Some("他們"),
        "1sg" | "I" => Some("我"),
        "2sg" | "you" => Some("你"),
        _ => None,
    }
}

/// Format compact deterministic control prefix tokens prepended to a sentence window before
/// Seq2Seq subword encoding.
pub fn format_control_prefix(target: TargetVariant, cond: &SentenceConditioning) -> String {
    let mut prefix = String::with_capacity(48);
    prefix.push_str(target.control_tag());

    if let Some(sub) = &cond.resolved_pro_drop {
        prefix.push_str(" [SUBJ=");
        prefix.push_str(sub);
        if let Some(ph) = &cond.pronoun_hint {
            prefix.push(':');
            prefix.push_str(ph);
        }
        prefix.push(']');
    } else if let Some(ph) = &cond.pronoun_hint {
        prefix.push_str(" [PRON=");
        prefix.push_str(ph);
        prefix.push(']');
    }

    if let Some(reg) = cond.speech_register.and_then(|r| r.control_code()) {
        prefix.push_str(" [REG=");
        prefix.push_str(reg);
        prefix.push(']');
    } else if let Some(spk) = &cond.speaker {
        prefix.push_str(" [SPK=");
        prefix.push_str(spk);
        prefix.push(']');
    }

    if let Some(fact) = &cond.factuality
        && fact != "actual"
    {
        prefix.push_str(" [FACT=");
        prefix.push_str(fact);
        prefix.push(']');
    }

    if let Some(rel) = &cond.discourse_relation {
        prefix.push_str(" [REL=");
        prefix.push_str(rel);
        prefix.push(']');
    }

    prefix.push(' ');
    prefix
}

/// Partition `sent_span` strictly along `doc.segmentation.clauses` boundaries whenever the scalar
/// budget of `sent_span` exceeds `max_window_chars`, preserving exact inter-clause separators.
pub fn partition_clause_windows(
    chars: &[char],
    sentence_idx: usize,
    sent_span: Span,
    all_clauses: &[Clause],
    max_window_chars: usize,
) -> Vec<ClauseWindow> {
    let sent_len = sent_span.end_char.saturating_sub(sent_span.start_char);
    let sent_clauses: Vec<(usize, &Clause)> = all_clauses
        .iter()
        .enumerate()
        .filter(|(_, c)| c.sentence == sentence_idx)
        .collect();

    if sent_len <= max_window_chars || sent_clauses.len() <= 1 {
        return vec![ClauseWindow {
            span: sent_span,
            trailing_sep: String::new(),
            clause_indices: sent_clauses.iter().map(|(i, _)| *i).collect(),
        }];
    }

    let mut windows: Vec<ClauseWindow> = Vec::new();
    let mut group_start = sent_span.start_char;
    let mut group_end = sent_clauses[0].1.span.end_char;
    let mut group_ids = vec![sent_clauses[0].0];

    for pair in sent_clauses.windows(2) {
        let prev_clause = pair[0].1;
        let (next_idx, next_clause) = pair[1];
        let proposed_len = next_clause.span.end_char.saturating_sub(group_start);

        if proposed_len > max_window_chars && !group_ids.is_empty() {
            let sep: String = chars[prev_clause.span.end_char..next_clause.span.start_char]
                .iter()
                .collect();
            windows.push(ClauseWindow {
                span: Span {
                    start_char: group_start,
                    end_char: group_end,
                },
                trailing_sep: sep,
                clause_indices: std::mem::take(&mut group_ids),
            });
            group_start = next_clause.span.start_char;
        }
        group_end = next_clause.span.end_char;
        group_ids.push(next_idx);
    }

    windows.push(ClauseWindow {
        span: Span {
            start_char: group_start,
            end_char: sent_span.end_char,
        },
        trailing_sep: String::new(),
        clause_indices: group_ids,
    });

    windows
}

fn spans_overlap(a: Span, b: Span) -> bool {
    a.start_char < b.end_char && b.start_char < a.end_char
}

fn relation_name(kind: RelationKind) -> &'static str {
    match kind {
        RelationKind::Sequence => "sequence",
        RelationKind::Cause => "cause",
        RelationKind::Contrast => "contrast",
        RelationKind::Parallel => "parallel",
        RelationKind::Condition => "condition",
    }
}

#[cfg(feature = "narrative")]
fn factuality_name(f: Factuality) -> &'static str {
    match f {
        Factuality::Actual => "actual",
        Factuality::Negated => "negated",
        Factuality::Hypothetical => "hypothetical",
        Factuality::ModalOrFuture => "modal_or_future",
        Factuality::Quoted => "quoted",
    }
}
