//! Multi-layer deterministic `ja -> zh-Hant / zh-Hans` translation engine.
//!
//! Consumes all `nlp.doc.v1` layers (`segmentation`, `tokens`, `entities`, `syntax`, `frames`,
//! `coref` / `zero_mentions`, `relations`) plus `SentenceConditioning` and `SpanLockTable` to
//! reorder Japanese SOV clauses into Chinese SVO, synthesize predicates, resolve pro-drop, and
//! render uncensored, idiomatic `zh-Hant` or `zh-Hans` prose and dialogue.

use std::collections::HashMap;

mod calendar;
pub(crate) use calendar::DateContext;
mod counted_object;
mod payment_potential;

use faberline_nlp::Span;
use faberline_nlp::doc::{Document, FrameRole, SemanticClass, Token, Upos};

use crate::conditioning::zh_pronoun_for_hint;
use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::{ClauseAlignment, GlossaryKind, SentenceConditioning, TargetVariant};
use crate::variant::{convert_zh_text_to_target, convert_zh_text_to_target_with_spans};

/// Returns `true` if `text` contains Hiragana or Katakana and no Hangul.
pub(crate) fn has_kana_no_hangul(text: &str) -> bool {
    let mut has_kana = false;
    for ch in text.chars() {
        if is_hangul(ch) {
            return false;
        }
        if is_kana(ch) {
            has_kana = true;
        }
    }
    has_kana
}

fn is_kana(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3040..=0x309F | 0x30A0..=0x30FF | 0xFF65..=0xFF9F
    )
}

fn is_hangul(ch: char) -> bool {
    matches!(
        ch as u32,
        0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F
    )
}

fn is_han(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2CEAF | 0xF900..=0xFAFF
    )
}

/// Lookup Japanese gloss in `faberline_nlp::lexicon::ja::lookup_ja_gloss`, preferring lemma fallback when surface misses.
fn lookup_ja_lexicon_gloss(
    surface: &str,
    lemma: Option<&str>,
    upos: Option<Upos>,
) -> Option<String> {
    if let Some(cp) = faberline_nlp::lexicon::ja::lookup_ja_gloss(surface, upos)
        && !cp.is_empty()
    {
        return Some(cp.to_owned());
    }
    if let Some(lem) = lemma
        && lem != surface
        && let Some(cp) = faberline_nlp::lexicon::ja::lookup_ja_gloss(lem, upos)
        && !cp.is_empty()
    {
        return Some(cp.to_owned());
    }
    None
}

/// An atomic unit inside a clause: either a locked `[[PHk]]` placeholder or a `Document` token.
#[derive(Clone, Debug)]
struct Atom {
    span: Span,
    surface: String,
    lemma: Option<String>,
    upos: Option<Upos>,
    doc_tok_idx: Option<usize>,
    head: Option<usize>,
    deprel: String,
    is_locked: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JointManner {
    Together,
    Everyone,
    Counted,
    Companion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidenceForm {
    Simple,
    State,
    Plan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BunsetsuRole {
    Connector,
    JointManner {
        manner: JointManner,
        invitation: bool,
        wh_question: bool,
    },
    JointCompanion,
    TravelCompanion,
    ResidencePlace {
        form: ResidenceForm,
        past: bool,
        wh: bool,
        only: bool,
    },
    AdditionalResidencePlace,
    ResidenceDuration,
    ResidenceTogether,
    ResidenceCount,
    UnresolvedResidenceArgument,
    Subj,
    Obl,
    ExplicitDeadline {
        modal: &'static str,
        additive: bool,
    },
    DeadlineWithDetachedNegation,
    MotionPurpose,
    PurposeModifier,
    ProposalModifier,
    ProposalCompanion,
    PurposeCompanion,
    PurposePartner(&'static str),
    PurposeRecipient,
    Adv,
    IObj,
    Obj,
    Predicate,
    Punct,
    Other,
}

#[derive(Clone, Debug)]
struct Bunsetsu {
    atoms: Vec<Atom>,
    role: BunsetsuRole,
    case_particle: Option<String>,
    sconj_tail: Option<String>,
    sfp_tail: Option<String>,
}

pub(crate) struct DirectQuoteOpening {
    prefix_span: Span,
    roles: DirectQuoteRoles,
    pub(crate) close: Option<char>,
}

pub(crate) struct DirectQuotePrefix {
    span: Span,
    text: String,
}

pub(crate) enum DirectQuoteEnd {
    Continues,
    Closed,
    Reports(Span, DirectQuotePrefix),
    Other,
}

#[derive(Default)]
pub(crate) struct DirectQuotePlacement {
    pub(crate) opening: Option<DirectQuotePrefix>,
    pub(crate) report: Option<Span>,
    pub(crate) detached_from: Option<usize>,
}

/// Translate a single Japanese (or Ja-dominant Mixed) sentence into `target` (`zh-Hant` / `zh-Hans`),
/// returning the masked target string (with `[[PHk]]` placeholders intact for `SpanLockTable::restore`)
/// and `ClauseAlignment` entries relative to sentence start (0-based within the sentence).
#[allow(clippy::too_many_arguments)]
pub(crate) fn translate_ja_sentence(
    sentence_idx: usize,
    chars: &[char],
    doc: &Document,
    sent_tokens: &[&Token],
    lock_table: &SpanLockTable,
    conditioning: &SentenceConditioning,
    target: TargetVariant,
    direct_quote: Option<&DirectQuotePlacement>,
    dates: &mut DateContext,
) -> (String, Vec<ClauseAlignment>) {
    let sent_span = doc.segmentation.sentences[sentence_idx].span;
    let mut atoms = normalized_sentence_atoms(sentence_idx, chars, doc, sent_tokens, lock_table);
    if target == TargetVariant::ZhHant {
        normalize_drawer_extraction(&mut atoms, doc);
        dates.normalize(&mut atoms, sentence_idx, chars, doc);
    }

    if atoms.is_empty() {
        let masked = lock_table.mask_span(chars, sent_span);
        let converted = convert_zh_text_to_target(&masked, target);
        return (converted, Vec::new());
    }

    // Check if the sentence has `doc.segmentation.clauses` for this sentence.
    let doc_clauses: Vec<(usize, Span)> = doc
        .segmentation
        .clauses
        .iter()
        .enumerate()
        .filter(|(_, c)| c.sentence == sentence_idx)
        .map(|(i, c)| (i, c.span))
        .collect();

    let report = (target == TargetVariant::ZhHant && direct_quote.is_none())
        .then(|| calendar::report::take(&mut atoms, doc, lock_table))
        .flatten();

    // Split atoms into clauses at `、` / `，` / quoted spans / SCONJ tails.
    let segments = split_atoms_into_segments(&atoms, doc, sent_span);

    let mut out = String::new();
    let mut seg_target_spans: Vec<(Span, Span)> = Vec::with_capacity(segments.len());
    if let Some(report) = report {
        out.push_str(&report.text);
        let target_span = Span {
            start_char: 0,
            end_char: out.chars().count(),
        };
        seg_target_spans.extend(report.spans.into_iter().map(|span| (span, target_span)));
    }
    let mut is_first_non_quote_clause = true;
    let mut prev_was_quote = false;
    let mut emitted_pre_quote_verb = false;
    let mut sentence_has_overt_topic = false;
    let mut prev_was_nagara = false;
    let mut prev_concessive_dou = false;
    let mut in_kuseni_follow = false;
    let mut in_toshitemo_follow = false;
    let mut in_noni_follow = false;
    let mut had_noni_in_sentence = false;
    let mut had_comm_event_in_sentence = false;
    let sentence_has_work_return = atoms_has_any(&atoms, &["帰る", "帰っ", "帰り"])
        && atoms_has_any(&atoms, &["仕事", "残業", "会社", "会議"]);
    let mut mirative_opening = false;
    let mut in_proportional_follow = false;
    let mut sentence_topic_pronoun: Option<&'static str> = None;
    let mut sonomono_antecedent: Option<String> = None;
    let mut skip_next_seg = false;
    let mut moved_report = direct_quote.and_then(|quote| quote.report);
    let opening = direct_quote.and_then(|quote| quote.opening.as_ref());
    let mut direct_quote_prefix_segments = 0;

    for (seg_idx, seg) in segments.iter().enumerate() {
        if direct_quote_prefix_segments > 0 {
            direct_quote_prefix_segments -= 1;
            continue;
        }
        if skip_next_seg {
            skip_next_seg = false;
            continue;
        }
        let tgt_start = out.chars().count();
        if target == TargetVariant::ZhHant
            && moved_report.is_none()
            && let Some((prefix, report, quote_index)) =
                local_direct_quote(&segments[seg_idx..], doc, lock_table)
        {
            out.push_str(&prefix);
            moved_report = Some(report);
            is_first_non_quote_clause = false;
            sentence_has_overt_topic = true;
            prev_was_quote = false;
            emitted_pre_quote_verb = false;
            if quote_index > 0 {
                let target_span = Span {
                    start_char: tgt_start,
                    end_char: out.chars().count(),
                };
                for prefix_segment in &segments[seg_idx..seg_idx + quote_index] {
                    if let Segment::Clause { span, .. } = prefix_segment {
                        seg_target_spans.push((*span, target_span));
                    }
                }
                direct_quote_prefix_segments = quote_index - 1;
                continue;
            }
        }
        let next_post_quote = segments.get(seg_idx + 1).and_then(|nx| match nx {
            Segment::Clause { atoms: ca, .. } => {
                Some((is_kagon_clause(ca), classify_post_quote_clause(ca)))
            }
            _ => None,
        });
        match seg {
            Segment::Quote {
                open_q,
                inner_atoms,
                close_q,
                span,
            } => {
                if let Some(prefix) = opening.filter(|quote| quote.span == *span) {
                    out.push_str(&prefix.text);
                }
                emitted_pre_quote_verb = false;
                if let Some((is_kagon, post_kind)) = next_post_quote {
                    if is_kagon {
                        out.push('說');
                        emitted_pre_quote_verb = true;
                    } else if post_kind == PostQuoteKind::QuoteKuseni {
                        out.push_str(if target == TargetVariant::ZhHans {
                            "明明说着"
                        } else {
                            "明明說著"
                        });
                        emitted_pre_quote_verb = true;
                    } else if post_kind == PostQuoteKind::QuotePlainSay && sentence_has_overt_topic
                    {
                        out.push('說');
                        emitted_pre_quote_verb = true;
                    }
                }
                let open_zh = localize_open_quote(*open_q, target);
                let close_zh = localize_close_quote(*close_q, target);
                out.push(open_zh);
                let inner_segs = split_atoms_into_segments(inner_atoms, doc, *span);
                let after_quote = &segments[seg_idx + 1..];
                let adjacent = if target == TargetVariant::ZhHant && after_quote.is_empty() {
                    payment_potential::adjacent_report(
                        inner_atoms,
                        *span,
                        doc,
                        sentence_idx,
                        chars,
                        lock_table,
                    )
                } else {
                    None
                };
                for iseg in &inner_segs {
                    if let Segment::Clause {
                        atoms: catoms,
                        trailing_punct,
                        ..
                    } = iseg
                    {
                        let quoted = (target == TargetVariant::ZhHant)
                            .then(|| {
                                payment_potential::quoted_clause(
                                    inner_atoms,
                                    catoms,
                                    seg_idx.checked_sub(1).and_then(|i| segments.get(i)),
                                    adjacent.as_deref().unwrap_or(after_quote),
                                    chars,
                                    target,
                                )
                            })
                            .flatten();
                        let rendered = quoted.unwrap_or_else(|| {
                            translate_clause_atoms(
                                catoms,
                                doc,
                                sentence_idx,
                                None,
                                false,
                                &ClauseDiscourseCtx {
                                    inside_quote: true,
                                    ..Default::default()
                                },
                                target,
                            )
                            .0
                        });
                        out.push_str(&rendered);
                        if let Some(p) = trailing_punct {
                            out.push_str(&localize_punct(p));
                        }
                    } else if let Segment::Punct(p, _) = iseg {
                        out.push_str(&localize_punct(p));
                    }
                }
                if moved_report.is_some()
                    && out.ends_with('嗎')
                    && payment_potential::ends_in_question(inner_atoms)
                {
                    out.push('？');
                }
                emit_quote_close(
                    &mut out,
                    close_zh,
                    segments.get(seg_idx + 1),
                    moved_report,
                    false,
                );
                prev_was_quote = true;
                let tgt_end = out.chars().count();
                seg_target_spans.push((
                    *span,
                    Span {
                        start_char: tgt_start,
                        end_char: tgt_end,
                    },
                ));
            }
            Segment::OpenQuote {
                open_q,
                inner_atoms,
                span,
            } => {
                if let Some(prefix) = opening.filter(|quote| quote.span == *span) {
                    out.push_str(&prefix.text);
                }
                out.push(localize_open_quote(*open_q, target));
                let inner_segs = split_atoms_into_segments(inner_atoms, doc, *span);
                for iseg in &inner_segs {
                    if let Segment::Clause {
                        atoms: catoms,
                        trailing_punct,
                        ..
                    } = iseg
                    {
                        let rendered = translate_clause_atoms(
                            catoms,
                            doc,
                            sentence_idx,
                            None,
                            false,
                            &ClauseDiscourseCtx {
                                inside_quote: true,
                                ..Default::default()
                            },
                            target,
                        )
                        .0;
                        out.push_str(&rendered);
                        if let Some(p) = trailing_punct {
                            out.push_str(&localize_punct(p));
                        }
                    } else if let Segment::Punct(p, _) = iseg {
                        out.push_str(&localize_punct(p));
                    }
                }
                prev_was_quote = false;
                let tgt_end = out.chars().count();
                seg_target_spans.push((
                    *span,
                    Span {
                        start_char: tgt_start,
                        end_char: tgt_end,
                    },
                ));
            }
            Segment::CloseQuote(close_q, span) => {
                let closes_after_previous_terminal =
                    out.is_empty() && terminal_before_quote(chars, span.start_char);
                emit_quote_close(
                    &mut out,
                    localize_close_quote(*close_q, target),
                    segments.get(seg_idx + 1),
                    moved_report,
                    closes_after_previous_terminal,
                );
                prev_was_quote = true;
                let tgt_end = out.chars().count();
                seg_target_spans.push((
                    *span,
                    Span {
                        start_char: tgt_start,
                        end_char: tgt_end,
                    },
                ));
            }
            Segment::Clause {
                atoms: catoms,
                trailing_punct,
                span,
            } => {
                if moved_report == Some(*span) {
                    // The reporting verb belongs to the opening frame. A period belongs
                    // inside the quote, but an outside question or exclamation stays here.
                    moved_report = None;
                    prev_was_quote = false;
                    emitted_pre_quote_verb = false;
                    if let Some(p) = trailing_punct.as_deref().filter(|p| *p != "。") {
                        out.push_str(&localize_punct(p));
                    }
                    seg_target_spans.push((
                        *span,
                        Span {
                            start_char: tgt_start,
                            end_char: out.chars().count(),
                        },
                    ));
                    continue;
                }

                if let Some(prefix) = opening.filter(|quote| quote.span == *span) {
                    out.push_str(&prefix.text);
                    is_first_non_quote_clause = false;
                    sentence_has_overt_topic = true;
                    prev_was_quote = false;
                    emitted_pre_quote_verb = false;
                    seg_target_spans.push((
                        *span,
                        Span {
                            start_char: tgt_start,
                            end_char: out.chars().count(),
                        },
                    ));
                    continue;
                }

                if prev_was_quote && is_kagon_clause(catoms) {
                    prev_was_quote = false;
                    emitted_pre_quote_verb = false;
                    out.push_str("也不為過");
                    if let Some(p) = trailing_punct {
                        out.push_str(&localize_punct(p));
                    }
                    let tgt_end = out.chars().count();
                    seg_target_spans.push((
                        *span,
                        Span {
                            start_char: tgt_start,
                            end_char: tgt_end,
                        },
                    ));
                    continue;
                }
                if prev_was_quote && emitted_pre_quote_verb {
                    let post_kind = classify_post_quote_clause(catoms);
                    if post_kind == PostQuoteKind::QuoteKuseni {
                        prev_was_quote = false;
                        emitted_pre_quote_verb = false;
                        in_kuseni_follow = true;
                        if let Some(p) = trailing_punct {
                            out.push_str(&localize_punct(p));
                        } else {
                            out.push('，');
                        }
                        let tgt_end = out.chars().count();
                        seg_target_spans.push((
                            *span,
                            Span {
                                start_char: tgt_start,
                                end_char: tgt_end,
                            },
                        ));
                        continue;
                    }
                    if post_kind == PostQuoteKind::QuotePlainSay {
                        prev_was_quote = false;
                        emitted_pre_quote_verb = false;
                        if let Some(p) = trailing_punct {
                            out.push_str(&localize_punct(p));
                        }
                        let tgt_end = out.chars().count();
                        seg_target_spans.push((
                            *span,
                            Span {
                                start_char: tgt_start,
                                end_char: tgt_end,
                            },
                        ));
                        continue;
                    }
                }
                emitted_pre_quote_verb = false;

                let trimmed_atoms = if prev_was_quote
                    && catoms
                        .first()
                        .is_some_and(|a| !a.is_locked && a.surface == "と")
                {
                    &catoms[1..]
                } else {
                    catoms.as_slice()
                };
                prev_was_quote = false;
                let cur_is_nagara = trimmed_atoms.last().is_some_and(|a| {
                    !a.is_locked && matches!(a.surface.as_str(), "ながら" | "つつ")
                });
                let has_univ_adv = trimmed_atoms.iter().any(|a| {
                    !a.is_locked && matches!(a.surface.as_str(), "どんなに" | "いくら" | "どれほど")
                });
                let ends_with_temo = trimmed_atoms
                    .last()
                    .is_some_and(|a| !a.is_locked && matches!(a.surface.as_str(), "ても" | "でも"))
                    || (trimmed_atoms.len() >= 2
                        && !trimmed_atoms[trimmed_atoms.len() - 1].is_locked
                        && trimmed_atoms[trimmed_atoms.len() - 1].surface == "も"
                        && !trimmed_atoms[trimmed_atoms.len() - 2].is_locked
                        && matches!(
                            trimmed_atoms[trimmed_atoms.len() - 2].surface.as_str(),
                            "て" | "で"
                        ));
                let cur_concessive_dou = has_univ_adv && ends_with_temo;
                let cur_ends_kuseni = trimmed_atoms
                    .last()
                    .is_some_and(|a| !a.is_locked && a.surface == "くせに")
                    || (trimmed_atoms.len() >= 2
                        && !trimmed_atoms[trimmed_atoms.len() - 2].is_locked
                        && trimmed_atoms[trimmed_atoms.len() - 2].surface == "くせ"
                        && !trimmed_atoms[trimmed_atoms.len() - 1].is_locked
                        && trimmed_atoms[trimmed_atoms.len() - 1].surface == "に");
                let cur_ends_toshitemo = trimmed_atoms
                    .last()
                    .is_some_and(|a| !a.is_locked && a.surface == "としても")
                    || (trimmed_atoms.len() >= 2
                        && !trimmed_atoms[trimmed_atoms.len() - 2].is_locked
                        && trimmed_atoms[trimmed_atoms.len() - 2].surface == "として"
                        && !trimmed_atoms[trimmed_atoms.len() - 1].is_locked
                        && trimmed_atoms[trimmed_atoms.len() - 1].surface == "も");
                let cur_ends_noni = !cur_ends_kuseni
                    && trimmed_atoms
                        .last()
                        .is_some_and(|a| !a.is_locked && a.surface == "のに");
                if cur_ends_noni {
                    had_noni_in_sentence = true;
                }
                let cur_is_subordinate = !cur_ends_kuseni
                    && !cur_ends_noni
                    && trimmed_atoms.last().is_some_and(|a| {
                        !a.is_locked
                            && (is_clause_split_sconj(&a.surface)
                                || matches!(a.surface.as_str(), "と" | "でも" | "としても"))
                    });

                // Proportional + passive cognitive tendency combiner (`[XばVほど]、[Yと考えられがちだ]`).
                if let Some(prop_zh) = try_render_proportional_clause(trimmed_atoms) {
                    is_first_non_quote_clause = false;
                    if let Some(Segment::Clause {
                        atoms: next_atoms,
                        trailing_punct: next_punct,
                        span: next_span,
                    }) = segments.get(seg_idx + 1)
                        && let Some(inner_cognitive) =
                            try_extract_passive_cognitive_gachi(next_atoms)
                    {
                        let inner_zh = translate_proportional_cognitive_inner(
                            inner_cognitive,
                            doc,
                            sentence_idx,
                            target,
                        );
                        let combined = format!("人們往往認為{prop_zh}，{inner_zh}");
                        out.push_str(&combined);
                        if let Some(p) = next_punct {
                            out.push_str(&localize_punct(p));
                        }
                        let tgt_end = out.chars().count();
                        seg_target_spans.push((
                            *span,
                            Span {
                                start_char: tgt_start,
                                end_char: tgt_end,
                            },
                        ));
                        seg_target_spans.push((
                            *next_span,
                            Span {
                                start_char: tgt_start,
                                end_char: tgt_end,
                            },
                        ));
                        skip_next_seg = true;
                        continue;
                    }
                    out.push_str(&prop_zh);
                    if let Some(p) = trailing_punct {
                        out.push_str(&localize_punct(p));
                    }
                    in_proportional_follow = true;
                    let tgt_end = out.chars().count();
                    seg_target_spans.push((
                        *span,
                        Span {
                            start_char: tgt_start,
                            end_char: tgt_end,
                        },
                    ));
                    continue;
                }

                // Editorial `しかし、実際には...` -> `然而，...`.
                let editorial_shikashi = trimmed_atoms.iter().filter(|a| !is_punct_atom(a)).count()
                    == 1
                    && trimmed_atoms
                        .iter()
                        .find(|a| !is_punct_atom(a))
                        .is_some_and(|a| !a.is_locked && a.surface == "しかし")
                    && segments.get(seg_idx + 1).is_some_and(|nx| match nx {
                        Segment::Clause { atoms: na, .. } => {
                            na.first()
                                .is_some_and(|fa| !fa.is_locked && fa.surface == "実際には")
                                || (na.len() >= 3
                                    && na[0].surface == "実際"
                                    && na[1].surface == "に"
                                    && na[2].surface == "は")
                        }
                        _ => false,
                    });

                if let Some(tp) = detect_clause_topic_pronoun(trimmed_atoms) {
                    sentence_topic_pronoun = Some(tp);
                }
                if let Some(ante) = extract_sonomono_antecedent(trimmed_atoms) {
                    sonomono_antecedent = Some(ante);
                }
                let is_mirative_interj =
                    trimmed_atoms.iter().filter(|a| !is_punct_atom(a)).all(|a| {
                        !a.is_locked && matches!(a.surface.as_str(), "へえ" | "へぇ" | "ほう")
                    }) && trimmed_atoms.iter().any(|a| !is_punct_atom(a));

                let allow_pro_drop = if is_first_non_quote_clause {
                    is_first_non_quote_clause = false;
                    !sentence_has_overt_topic
                } else {
                    false
                };
                let cond_opt = Some(conditioning);
                let ctx = ClauseDiscourseCtx {
                    contrasts_recurrence: seg_idx.checked_sub(1).is_some_and(|i|
                        matches!(&segments[i], Segment::Clause { atoms, .. } if calendar::contrasts_recurrence(atoms))),
                    return_trip: None,
                    inside_quote: ends_continued_quote(
                        segments.get(seg_idx + 1),
                        doc,
                        sentence_idx,
                        chars,
                    ),
                    in_nagara_follow: prev_was_nagara,
                    in_concessive_follow: prev_concessive_dou,
                    in_kuseni_follow,
                    in_toshitemo_follow,
                    in_noni_follow,
                    had_noni_in_sentence,
                    had_comm_event_in_sentence,
                    sentence_has_work_return,
                    mirative_opening,
                    in_proportional_follow,
                    sentence_topic_pronoun,
                    sonomono_antecedent: sonomono_antecedent.clone(),
                };
                let receiving_request = if target == TargetVariant::ZhHant
                    && nominal_receiving_request_position(&segments, seg_idx, doc, sentence_idx)
                {
                    try_translate_nominal_receiving_request(trimmed_atoms, doc, sentence_idx)
                } else {
                    None
                };
                let trailing_punct = if receiving_request.is_some() {
                    Some("？")
                } else {
                    trailing_punct.as_deref()
                };
                let (mut rendered, had_subj_only) = if editorial_shikashi {
                    ("然而".to_owned(), false)
                } else if let Some(rendered) = receiving_request {
                    (rendered, false)
                } else {
                    translate_clause_atoms(
                        trimmed_atoms,
                        doc,
                        sentence_idx,
                        cond_opt,
                        allow_pro_drop,
                        &ctx,
                        target,
                    )
                };
                if is_mirative_interj {
                    mirative_opening = true;
                }
                if clause_has_comm_event(trimmed_atoms) {
                    had_comm_event_in_sentence = true;
                }
                if had_subj_only {
                    sentence_has_overt_topic = true;
                }
                prev_was_nagara = cur_is_nagara;
                prev_concessive_dou = cur_concessive_dou;
                if cur_ends_kuseni {
                    in_kuseni_follow = true;
                } else if !cur_is_subordinate {
                    in_kuseni_follow = false;
                }
                in_noni_follow = cur_ends_noni;
                in_toshitemo_follow = cur_ends_toshitemo;
                in_proportional_follow = false;

                let next_starts_with_contrast =
                    segments.get(seg_idx + 1).is_some_and(|nx| match nx {
                        Segment::Clause { atoms: na, .. } => na.first().is_some_and(|fa| {
                            !fa.is_locked
                                && matches!(
                                    fa.surface.as_str(),
                                    "だからといって"
                                        | "でも"
                                        | "しかし"
                                        | "だが"
                                        | "とはいえ"
                                        | "けれど"
                                        | "けれども"
                                        | "ところが"
                                        | "それでも"
                                        | "それなのに"
                                )
                        }),
                        _ => false,
                    });
                if next_starts_with_contrast {
                    if let Some(stripped) = rendered.strip_suffix("，但是") {
                        rendered = stripped.to_owned();
                    } else if let Some(stripped) = rendered.strip_suffix("，不過") {
                        rendered = stripped.to_owned();
                    } else if let Some(stripped) = rendered.strip_suffix("，但") {
                        rendered = stripped.to_owned();
                    }
                }

                out.push_str(&rendered);
                let next_quote_framed = match (segments.get(seg_idx + 1), segments.get(seg_idx + 2))
                {
                    (Some(Segment::Quote { .. }), Some(Segment::Clause { atoms: nx2, .. })) => {
                        is_kagon_clause(nx2)
                            || matches!(
                                classify_post_quote_clause(nx2),
                                PostQuoteKind::QuoteKuseni | PostQuoteKind::QuotePlainSay
                            )
                    }
                    _ => false,
                };
                let suppress_cleft_comma = is_cleft_topic_clause(trimmed_atoms)
                    && segments.get(seg_idx + 1).is_some_and(|nx| match nx {
                        Segment::Clause { atoms: na, .. } => clause_contains_dewa_naku(na),
                        _ => false,
                    });
                if let Some(p) = trailing_punct {
                    if !(suppress_cleft_comma
                        || (rendered.ends_with('？') && matches!(p, "。" | "？" | "?"))
                        || ((p == "、" || p == ",")
                            && (rendered.ends_with("，就立刻")
                                || rendered.ends_with("，但是")
                                || rendered.ends_with("，不過")
                                || rendered.ends_with("，而且")
                                || rendered.ends_with("，但")
                                || rendered.ends_with("，只是")
                                || rendered.ends_with("，所以")
                                || rendered.ends_with("，而是")
                                || rendered.ends_with("，以及")
                                || rendered.ends_with('，'))))
                    {
                        out.push_str(&localize_punct(p));
                    }
                } else if matches!(segments.get(seg_idx + 1), Some(Segment::Quote { .. }))
                    && !next_quote_framed
                    && !had_subj_only
                    && !rendered.is_empty()
                    && !rendered.ends_with(['，', '。', '：'])
                {
                    out.push('，');
                }
                let tgt_end = out.chars().count();
                seg_target_spans.push((
                    *span,
                    Span {
                        start_char: tgt_start,
                        end_char: tgt_end,
                    },
                ));
            }
            Segment::Punct(p, span) => {
                prev_was_quote = false;
                out.push_str(&localize_punct(p));
                let tgt_end = out.chars().count();
                seg_target_spans.push((
                    *span,
                    Span {
                        start_char: tgt_start,
                        end_char: tgt_end,
                    },
                ));
            }
        }
    }

    let final_zh = if target == TargetVariant::ZhHans {
        let mut target_spans: Vec<Span> = seg_target_spans.iter().map(|(_, span)| *span).collect();
        let converted = convert_zh_text_to_target_with_spans(&out, target, &mut target_spans);
        for ((_, span), converted_span) in seg_target_spans.iter_mut().zip(target_spans) {
            *span = converted_span;
        }
        converted
    } else {
        out
    };
    let final_chars = final_zh.chars().count();

    // Build `ClauseAlignment` entries aligned with `doc.segmentation.clauses` when available,
    // or one alignment per translated segment.
    let mut alignments = Vec::new();
    if !doc_clauses.is_empty() {
        for (ci, cspan) in doc_clauses {
            let matching: Vec<&(Span, Span)> = seg_target_spans
                .iter()
                .filter(|(s, _)| s.start_char < cspan.end_char && cspan.start_char < s.end_char)
                .collect();
            let (t_start, t_end) =
                if let (Some(first), Some(last)) = (matching.first(), matching.last()) {
                    (first.1.start_char, last.1.end_char)
                } else {
                    (0, final_chars)
                };
            alignments.push(ClauseAlignment {
                clause: ci,
                sentence: sentence_idx,
                source_span: cspan,
                target_span: Span {
                    start_char: t_start,
                    end_char: t_end,
                },
            });
        }
    }

    (final_zh, alignments)
}

/// Translate an inline Japanese fragment (such as a heading or list item body) that already has
/// `[[PHk]]` placeholders masked.
pub(crate) fn translate_ja_fragment(masked: &str, target: TargetVariant) -> String {
    if masked.trim().is_empty() {
        return masked.to_owned();
    }
    let doc = faberline_nlp::doc::analyze_document(
        masked,
        &faberline_nlp::doc::Options {
            enrich_morphology: true,
            cpu_syntax: true,
            ..Default::default()
        },
    );
    let chars: Vec<char> = masked.chars().collect();
    let mut out = String::new();
    let mut cursor = 0usize;
    let mut dates = DateContext::default();
    for (si, sent) in doc.segmentation.sentences.iter().enumerate() {
        if sent.span.start_char > cursor {
            for &c in &chars[cursor..sent.span.start_char] {
                out.push(c);
            }
        }
        let sent_tokens: Vec<&Token> = doc.tokens.iter().filter(|t| t.sentence == si).collect();
        let lock_table = SpanLockTable::build(Vec::new());
        let cond = SentenceConditioning::default();
        let (tr, _) = translate_ja_sentence(
            si,
            &chars,
            &doc,
            &sent_tokens,
            &lock_table,
            &cond,
            target,
            None,
            &mut dates,
        );
        out.push_str(&tr);
        cursor = sent.span.end_char;
    }
    if cursor < chars.len() {
        for &c in &chars[cursor..] {
            out.push(c);
        }
    }
    if out.is_empty() {
        convert_zh_text_to_target(masked, target)
    } else {
        out
    }
}

// Marker used only for atoms synthesized by the clock recognizer.  Protected
// placeholders retain their own identity and are never classified by this marker.
const CLOCK_ATOM: &str = "__ja_clock_atom";
// A typed compound time created only while assembling a continuation endpoint.
const TEMPORAL_MADE_COMPOUND_ATOM: &str = "__ja_temporal_made_compound";
// An opaque component in an explicitly period-qualified clock-shaped phrase.
// Its value remains unknown and its placeholder remains untouched.
const OPAQUE_PERIOD_CLOCK: &str = "__ja_opaque_period_clock";
const OPAQUE_EVENT_OBLIQUE: &str = "__ja_opaque_event_oblique";

fn is_clock_atom(atom: &Atom) -> bool {
    matches!(
        atom.lemma.as_deref(),
        Some(CLOCK_ATOM | OPAQUE_PERIOD_CLOCK)
    )
}

fn normalize_decimal_digits(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            '０'..='９' => char::from_u32(ch as u32 - '０' as u32 + '0' as u32).unwrap_or(ch),
            _ => ch,
        })
        .collect()
}

fn clock_number(text: &str) -> Option<u32> {
    let normalized = normalize_decimal_digits(text);
    if normalized.chars().all(|ch| ch.is_ascii_digit()) {
        return normalized.parse::<u32>().ok().filter(|&n| n <= 24);
    }
    fn digit(s: &str) -> Option<u32> {
        match s {
            "零" | "〇" => Some(0),
            "一" => Some(1),
            "二" => Some(2),
            "三" => Some(3),
            "四" => Some(4),
            "五" => Some(5),
            "六" => Some(6),
            "七" => Some(7),
            "八" => Some(8),
            "九" => Some(9),
            _ => None,
        }
    }
    if let Some((tens, ones)) = normalized.split_once('十') {
        if ones.contains('十') {
            return None;
        }
        let tens = if tens.is_empty() { 1 } else { digit(tens)? };
        let ones = if ones.is_empty() { 0 } else { digit(ones)? };
        if tens == 0 || ones == 0 && !normalized.ends_with('十') {
            return None;
        }
        return (tens * 10 + ones <= 24).then_some(tens * 10 + ones);
    }
    digit(&normalized)
}

fn clock_hour_text(hour: u32) -> String {
    const HOURS: [&str; 25] = [
        "零",
        "一",
        "兩",
        "三",
        "四",
        "五",
        "六",
        "七",
        "八",
        "九",
        "十",
        "十一",
        "十二",
        "十三",
        "十四",
        "十五",
        "十六",
        "十七",
        "十八",
        "十九",
        "二十",
        "二十一",
        "二十二",
        "二十三",
        "二十四",
    ];
    HOURS[hour as usize].to_owned()
}

// Accept a whole token sequence only. This rejects 時間, 時限, fractions,
// counters and unsupported minutes instead of interpreting a prefix as a clock.
fn render_clock_surface(text: &str) -> Option<(String, bool)> {
    let (period, rest) = if let Some(s) = text.strip_prefix("午前") {
        (Some("am"), s)
    } else if let Some(s) = text.strip_prefix("午後") {
        (Some("pm"), s)
    } else {
        (None, text)
    };
    let (number, suffix) = rest.split_once('時')?;
    let (half, particle) = match suffix {
        "" => (false, false),
        "半" => (true, false),
        "に" => (false, true),
        "半に" => (true, true),
        _ => return None,
    };
    // Bare 一時 can mean a short while. Keep its original meaning unresolved
    // unless an explicit period or 半 supplies a clock-specific distinction.
    if period.is_none() && number == "一" && !half {
        return None;
    }
    let hour = clock_number(number)?;
    if hour == 24 && (half || period.is_some()) {
        return None;
    }
    if period.is_some() && hour > 12 {
        return None;
    }
    if period == Some("am") && hour == 12 && half {
        return Some(("十二點半（可能是中午，也可能是凌晨）".to_owned(), particle));
    }
    let (prefix, display_hour) = match (period, hour, half) {
        (Some("am"), 0, _) => ("午夜", 0),
        (Some("am"), 12, false) => ("中午", 12),
        (Some("pm"), 0, _) => ("中午", 12),
        (Some("pm"), 12, _) => ("午夜", 12),
        (Some("am"), 1..=4, _) => ("凌晨", hour),
        (Some("am"), _, _) => ("上午", hour),
        (Some("pm"), _, _) => ("下午", hour),
        _ => ("", hour),
    };
    Some((
        format!(
            "{prefix}{}點{}",
            clock_hour_text(display_hour),
            if half { "半" } else { "" }
        ),
        particle,
    ))
}

// A glossary is transparent source text with a mandatory replacement. Inline
// opaque contents are never inspected. Match only whole clock components.
fn normalize_glossary_clocks(atoms: &[Atom], chars: &[char], locks: &SpanLockTable) -> Vec<Atom> {
    let mut out = Vec::with_capacity(atoms.len());
    let mut i = 0;
    while i < atoms.len() {
        let mut source = String::new();
        let mut protected: Vec<(usize, usize, String)> = Vec::new();
        let mut candidate = None;
        for j in i..atoms.len().min(i + 6) {
            let atom = &atoms[j];
            if j > i && atoms[j - 1].span.end_char != atom.span.start_char {
                break;
            }
            let start = source.chars().count();
            if atom.is_locked {
                let Some(lock) = locks.locks.iter().find(|l| l.placeholder == atom.surface) else {
                    break;
                };
                if !matches!(&lock.origin, LockOrigin::Glossary { .. }) {
                    break;
                }
                source.extend(chars[lock.source_span.start_char..lock.source_span.end_char].iter());
                protected.push((start, source.chars().count(), atom.surface.clone()));
            } else {
                source.push_str(&atom.surface);
            }
            if !protected.is_empty()
                && let Some((rendered, particle)) = render_clock_surface(&source)
                && let Some(composed) =
                    compose_glossary_clock(&source, &rendered, particle, &protected)
            {
                candidate = Some((j, composed, particle));
            }
            if source.chars().count() > 12 {
                break;
            }
        }
        if let Some((end, _, particle)) = &candidate
            && !particle
            && atoms.get(end + 1).is_some_and(|next| {
                atoms[*end].span.end_char == next.span.start_char
                    && !next.is_locked
                    && clock_continuation(&next.surface)
            })
        {
            candidate = None;
        }
        if let Some((end, rendered, particle)) = candidate {
            let last = &atoms[end];
            let clock_end = last.span.end_char - usize::from(particle);
            out.push(Atom {
                span: Span {
                    start_char: atoms[i].span.start_char,
                    end_char: clock_end,
                },
                surface: rendered,
                lemma: Some(CLOCK_ATOM.to_owned()),
                upos: Some(Upos::Noun),
                doc_tok_idx: None,
                head: None,
                deprel: String::new(),
                is_locked: true,
            });
            if particle {
                out.push(Atom {
                    span: Span {
                        start_char: clock_end,
                        end_char: last.span.end_char,
                    },
                    surface: "に".to_owned(),
                    lemma: None,
                    upos: Some(Upos::Adp),
                    doc_tok_idx: last.doc_tok_idx,
                    head: last.head,
                    deprel: "case".to_owned(),
                    is_locked: false,
                });
            }
            i = end + 1;
        } else {
            out.push(atoms[i].clone());
            i += 1;
        }
    }
    out
}

fn compose_glossary_clock(
    source: &str,
    rendered: &str,
    particle: bool,
    protected: &[(usize, usize, String)],
) -> Option<String> {
    let source = if particle {
        source.strip_suffix('に')?
    } else {
        source
    };
    let length = source.chars().count();
    if protected.len() == 1 && protected[0].0 == 0 && protected[0].1 == length {
        return Some(protected[0].2.clone());
    }
    if rendered.contains('（') {
        return None;
    }
    let period_len = if source.starts_with("午前") || source.starts_with("午後") {
        2
    } else {
        0
    };
    let unit_pos = source.chars().position(|c| c == '時')?;
    let number: String = source
        .chars()
        .skip(period_len)
        .take(unit_pos - period_len)
        .collect();
    let hour = clock_number(&number)?;
    let output_hour = if source.starts_with("午後") && hour == 0 {
        12
    } else {
        hour
    };
    let hour_text = clock_hour_text(output_hour);
    let (front, tail) = rendered.split_once('點')?;
    let period_text = front.strip_suffix(&hour_text)?;
    let spans = [
        (0, period_len, period_text.to_owned()),
        (period_len, unit_pos, hour_text),
        (unit_pos, unit_pos + 1, "點".to_owned()),
        (unit_pos + 1, length, tail.to_owned()),
    ];
    let mut used = 0;
    let mut output = String::new();
    for (a, b, text) in spans {
        if a == b {
            continue;
        }
        if let Some((_, _, placeholder)) = protected
            .iter()
            .find(|(start, end, _)| *start == a && *end == b)
        {
            output.push_str(placeholder);
            used += 1;
        } else {
            if protected
                .iter()
                .any(|(start, end, _)| *start < b && a < *end)
            {
                return None;
            }
            output.push_str(&text);
        }
    }
    (used == protected.len()).then_some(output)
}

fn clock_continuation(text: &str) -> bool {
    text.chars().next().is_some_and(|ch| {
        matches!(ch,
        '間' | '限' | '分' | '半' | '０'..='９' | '0'..='9' | '零' | '〇' |
        '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十')
    })
}

// Preserve an opaque prefix immediately adjoining a visible clock. An initial
// opaque prefix, or one after a visible topic marker, stays before that clock.
// For an entirely opaque oblique, only an intransitive event boundary plus the
// frame's own ARGM head permits an oblique role; this is not a time assertion.
fn preserve_clock_adjacent_locks(atoms: &[Atom], doc: &Document, sentence_idx: usize) -> Vec<Atom> {
    let mut out = Vec::with_capacity(atoms.len());
    let mut i = 0;
    while i < atoms.len() {
        let a = &atoms[i];
        // A visible period and an opaque hour form one ordered clock phrase.
        // Preserve the period text: the protected hour might be a boundary.
        if !a.is_locked
            && matches!(a.surface.as_str(), "午前" | "午後")
            && let Some(next) = atoms.get(i + 1)
            && a.span.end_char == next.span.start_char
            && is_clock_atom(next)
            && next.surface.starts_with("[[PH")
        {
            let mut joined = next.clone();
            joined.span.start_char = a.span.start_char;
            joined.surface = format!("{}{}", a.surface, next.surface);
            out.push(joined);
            i += 2;
            continue;
        }
        let opaque = a.is_locked && a.surface.starts_with("[[PH") && a.lemma.is_none();
        if opaque
            && (i == 0 || atoms[i - 1].surface == "は")
            && let Some(next) = atoms.get(i + 1)
            && a.span.end_char == next.span.start_char
            && is_clock_atom(next)
        {
            let mut joined = next.clone();
            joined.span.start_char = a.span.start_char;
            joined.surface = format!("{}{}", a.surface, next.surface);
            out.push(joined);
            i += 2;
            continue;
        }
        let mut item = a.clone();
        if opaque
            && i > 0
            && atoms[i - 1].surface == "は"
            && atoms
                .get(i + 1)
                .is_some_and(|n| n.surface == "に" && !n.is_locked)
            && let Some(idx) = a.doc_tok_idx
            && doc.frames.iter().any(|frame| {
                frame.sentence == sentence_idx
                    && frame
                        .args
                        .iter()
                        .any(|arg| arg.head == idx && arg.role == FrameRole::Argm)
                    && doc
                        .tokens
                        .get(frame.predicate)
                        .and_then(|t| t.lemma.as_deref())
                        .is_some_and(|lemma| matches!(lemma, "始まる" | "終わる"))
            })
        {
            item.lemma = Some(OPAQUE_EVENT_OBLIQUE.to_owned());
        }
        out.push(item);
        i += 1;
    }
    out
}

// The same source clock can be one atom, several numeric atoms, or a final
// 時に atom. Preserve a separate に so existing case rendering remains local.
fn normalize_clock_atoms(atoms: &[Atom]) -> Vec<Atom> {
    let mut out = Vec::with_capacity(atoms.len());
    let mut i = 0;
    while i < atoms.len() {
        // An explicit visible period and 時半 provide temporal context. The
        // protected component remains unknown; do not infer a numeric value.
        if i > 0
            && !atoms[i - 1].is_locked
            && matches!(atoms[i - 1].surface.as_str(), "午前" | "午後")
            && atoms[i - 1].span.end_char == atoms[i].span.start_char
            && atoms[i].is_locked
            && atoms[i].surface.starts_with("[[PH")
        {
            let mut suffix = String::new();
            let mut found = None;
            for j in (i + 1)..atoms.len().min(i + 4) {
                if atoms[j].is_locked || atoms[j - 1].span.end_char != atoms[j].span.start_char {
                    break;
                }
                suffix.push_str(&atoms[j].surface);
                if suffix == "時半" || suffix == "時半に" {
                    found = Some((j, suffix.ends_with('に')));
                }
            }
            if let Some((end, particle)) = found
                && !particle
                && atoms.get(end + 1).is_some_and(|next| {
                    atoms[end].span.end_char == next.span.start_char
                        && !next.is_locked
                        && next.surface.starts_with('分')
                })
            {
                found = None;
            }
            if let Some((end, particle)) = found {
                let last = &atoms[end];
                let clock_end = last.span.end_char - usize::from(particle);
                out.push(Atom {
                    span: Span {
                        start_char: atoms[i].span.start_char,
                        end_char: clock_end,
                    },
                    surface: format!("{}時半", atoms[i].surface),
                    lemma: Some(OPAQUE_PERIOD_CLOCK.to_owned()),
                    upos: Some(Upos::Noun),
                    doc_tok_idx: None,
                    head: None,
                    deprel: String::new(),
                    is_locked: true,
                });
                if particle {
                    out.push(Atom {
                        span: Span {
                            start_char: clock_end,
                            end_char: last.span.end_char,
                        },
                        surface: "に".to_owned(),
                        lemma: None,
                        upos: Some(Upos::Adp),
                        doc_tok_idx: last.doc_tok_idx,
                        head: last.head,
                        deprel: "case".to_owned(),
                        is_locked: false,
                    });
                }
                i = end + 1;
                continue;
            }
        }
        let mut joined = String::new();
        let mut best = None;
        for j in i..atoms.len().min(i + 6) {
            let atom = &atoms[j];
            if atom.is_locked || (j > i && atoms[j - 1].span.end_char != atom.span.start_char) {
                break;
            }
            joined.push_str(&atom.surface);
            if let Some((rendered, particle)) = render_clock_surface(&joined) {
                best = Some((j, rendered, particle));
            }
            if joined.chars().count() > 12 {
                break;
            }
        }
        if let Some((end, _, particle)) = &best
            && !particle && atoms.get(end + 1).is_some_and(|next| {
                atoms[*end].span.end_char == next.span.start_char
                    && !next.is_locked
                    && next.surface.chars().next().is_some_and(|ch| {
                        matches!(ch, '間' | '限' | '分' | '半' | '０'..='９' | '0'..='9' | '零' | '〇' | '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十')
                    })
            })
        {
            best = None;
        }
        if let Some((end, rendered, particle)) = best {
            let last = &atoms[end];
            let clock_end = last.span.end_char - usize::from(particle);
            out.push(Atom {
                span: Span {
                    start_char: atoms[i].span.start_char,
                    end_char: clock_end,
                },
                surface: rendered,
                lemma: Some(CLOCK_ATOM.to_owned()),
                upos: Some(Upos::Noun),
                doc_tok_idx: None,
                head: None,
                deprel: String::new(),
                is_locked: true,
            });
            if particle {
                out.push(Atom {
                    span: Span {
                        start_char: clock_end,
                        end_char: last.span.end_char,
                    },
                    surface: "に".to_owned(),
                    lemma: None,
                    upos: Some(Upos::Adp),
                    doc_tok_idx: last.doc_tok_idx,
                    head: last.head,
                    deprel: "case".to_owned(),
                    is_locked: false,
                });
            }
            i = end + 1;
        } else {
            out.push(atoms[i].clone());
            i += 1;
        }
    }
    out
}

fn build_sentence_atoms(
    sentence_idx: usize,
    chars: &[char],
    doc: &Document,
    sent_tokens: &[&Token],
    lock_table: &SpanLockTable,
) -> Vec<Atom> {
    let sent_span = doc.segmentation.sentences[sentence_idx].span;
    // Map doc token global index by token span start for syntax lookup.
    let mut syn_by_span_start: HashMap<usize, (usize, Option<usize>, &str)> = HashMap::new();
    for syn in &doc.syntax {
        if syn.sentence == sentence_idx {
            syn_by_span_start.insert(
                syn.span.start_char,
                (syn.token, syn.head, syn.deprel.as_str()),
            );
        }
    }

    let sent_locks: Vec<_> = lock_table
        .locks
        .iter()
        .filter(|l| {
            l.source_span.start_char >= sent_span.start_char
                && l.source_span.end_char <= sent_span.end_char
        })
        .collect();

    let mut atoms = Vec::with_capacity(sent_tokens.len());
    let mut lock_idx = 0usize;
    let mut tok_idx = 0usize;

    while tok_idx < sent_tokens.len() || lock_idx < sent_locks.len() {
        let next_lock = sent_locks.get(lock_idx);
        let next_tok = sent_tokens.get(tok_idx);

        match (next_lock, next_tok) {
            (Some(lk), Some(tk)) if lk.source_span.start_char <= tk.span.start_char => {
                // Consume all tokens covered by this lock, keeping the last token's syntax info.
                let mut doc_tok_idx = None;
                let mut head = None;
                let mut deprel = String::new();
                while let Some(t) = sent_tokens.get(tok_idx) {
                    if t.span.start_char >= lk.source_span.end_char {
                        break;
                    }
                    if let Some(&(gi, hd, dr)) = syn_by_span_start.get(&t.span.start_char) {
                        doc_tok_idx = Some(gi);
                        head = hd;
                        deprel = dr.to_owned();
                    }
                    tok_idx += 1;
                }
                let whole_lock = lk.mandatory
                    && tk.span.start_char == lk.source_span.start_char
                    && doc_tok_idx
                        .is_some_and(|id| doc.tokens[id].span.end_char == lk.source_span.end_char);
                atoms.push(Atom {
                    span: lk.source_span,
                    surface: lk.placeholder.clone(),
                    lemma: match (&lk.origin, whole_lock) {
                        (
                            LockOrigin::Glossary {
                                kind: GlossaryKind::Place,
                                ..
                            },
                            true,
                        ) => Some(LOCKED_PLACE_ATOM.to_owned()),
                        (
                            LockOrigin::Glossary {
                                kind: GlossaryKind::Actor,
                                ..
                            },
                            true,
                        ) => Some(LOCKED_RESIDENT_ATOM.to_owned()),
                        _ => None,
                    },
                    upos: Some(Upos::Propn),
                    doc_tok_idx,
                    head,
                    deprel,
                    is_locked: true,
                });
                lock_idx += 1;
            }
            (Some(lk), None) => {
                atoms.push(Atom {
                    span: lk.source_span,
                    surface: lk.placeholder.clone(),
                    lemma: None,
                    upos: Some(Upos::Propn),
                    doc_tok_idx: None,
                    head: None,
                    deprel: String::new(),
                    is_locked: true,
                });
                lock_idx += 1;
            }
            (_, Some(tk)) => {
                let surf: String = chars[tk.span.start_char..tk.span.end_char].iter().collect();
                let (doc_tok_idx, head, deprel) = syn_by_span_start
                    .get(&tk.span.start_char)
                    .map(|&(gi, hd, dr)| (Some(gi), hd, dr.to_owned()))
                    .unwrap_or((None, None, String::new()));
                atoms.push(Atom {
                    span: tk.span,
                    surface: surf,
                    lemma: tk.lemma.clone(),
                    upos: tk.upos,
                    doc_tok_idx,
                    head,
                    deprel,
                    is_locked: false,
                });
                tok_idx += 1;
            }
            (None, None) => break,
        }
    }

    // Re-attach any `[[PHk]]` literal tokens if `translate_ja_fragment` was called on pre-masked text.
    merge_literal_placeholders(atoms)
}

fn normalized_sentence_atoms(
    sentence_idx: usize,
    chars: &[char],
    doc: &Document,
    tokens: &[&Token],
    locks: &SpanLockTable,
) -> Vec<Atom> {
    let atoms = build_sentence_atoms(sentence_idx, chars, doc, tokens, locks);
    let atoms = normalize_keigo_atoms(&atoms);
    let atoms = normalize_glossary_clocks(&atoms, chars, locks);
    let atoms = normalize_clock_atoms(&atoms);
    preserve_clock_adjacent_locks(&atoms, doc, sentence_idx)
}

// The inflection analyzer can read 引き出し as the stem of 引き出す. A linked
// から source on an extraction predicate instead identifies the drawer noun.
fn normalize_drawer_extraction(atoms: &mut [Atom], doc: &Document) {
    for index in 0..atoms.len().saturating_sub(1) {
        let drawer = &atoms[index];
        let source = &atoms[index + 1];
        if drawer.is_locked
            || drawer.surface != "引き出し"
            || drawer.deprel != "obl"
            || drawer.head.is_none()
            || source.is_locked
            || source.surface != "から"
            || source.upos != Some(Upos::Adp)
            || source.deprel != "case"
            || drawer.doc_tok_idx.is_none()
            || source.head != drawer.doc_tok_idx
        {
            continue;
        }
        let predicate = drawer.head;
        let drawer_id = drawer.doc_tok_idx;
        let after_start = index + 2;
        let clause_end = atoms[after_start..]
            .iter()
            .position(|atom| is_punct_atom(atom) || atom.is_locked)
            .map_or(atoms.len(), |boundary| after_start + boundary);
        let after = &atoms[after_start..clause_end];
        if !after.iter().any(|atom| {
            atom.doc_tok_idx == predicate
                && atom.upos == Some(Upos::Verb)
                && atom.lemma.as_deref() == Some("出す")
        }) {
            continue;
        }
        let recovered_object = if after
            .iter()
            .any(|atom| atom.head == predicate && atom.deprel == "obj")
        {
            None
        } else {
            // In object-first order, the analyzer may attach Nを to the
            // apparent 引き出す stem. Recover it only next to the drawer
            // and its established source relation.
            let Some(object_idx) = index.checked_sub(2) else {
                continue;
            };
            let object = &atoms[object_idx];
            let wo = &atoms[object_idx + 1];
            if object.is_locked
                || !matches!(object.upos, Some(Upos::Noun | Upos::Propn))
                || object.deprel != "obj"
                || (object.head != predicate && object.head != drawer_id)
                || wo.surface != "を"
                || !dasu_case_link(object, wo)
                || wo.span.end_char != drawer.span.start_char
                || after
                    .iter()
                    .find(|atom| atom.upos == Some(Upos::Verb))
                    .is_none_or(|atom| atom.doc_tok_idx != predicate)
            {
                continue;
            }
            Some(object_idx)
        };
        // Preserve an already grounded floor location when its apparent
        // verbal governor becomes the drawer noun.
        let clause_start = atoms[..index]
            .iter()
            .rposition(|atom| atom.is_locked || is_punct_atom(atom))
            .map_or(0, |boundary| boundary + 1);
        for floor in clause_start..index.saturating_sub(1) {
            if atoms[floor].head == drawer_id
                && atoms[floor + 1].surface == "で"
                && building_floor_start(&atoms[clause_start..clause_end], floor - clause_start, doc)
                    .is_some()
            {
                atoms[floor].head = predicate;
            }
        }
        if let Some(object_idx) = recovered_object {
            atoms[object_idx].head = predicate;
        }
        let drawer = &mut atoms[index];
        drawer.surface = "抽屜".to_owned();
        drawer.lemma = None;
        drawer.upos = Some(Upos::Noun);
    }
}

fn merge_literal_placeholders(atoms: Vec<Atom>) -> Vec<Atom> {
    let mut out: Vec<Atom> = Vec::with_capacity(atoms.len());
    let mut i = 0usize;
    while i < atoms.len() {
        if !atoms[i].is_locked && atoms[i].surface == "[" {
            // Try to match `[` `[` `PH` `0` `]` `]`
            let mut combined = String::new();
            let mut j = i;
            let start = atoms[i].span.start_char;
            let mut found_close = false;
            while j < atoms.len() && j < i + 8 {
                combined.push_str(&atoms[j].surface);
                if combined.starts_with("[[PH") && combined.ends_with("]]") {
                    out.push(Atom {
                        span: Span {
                            start_char: start,
                            end_char: atoms[j].span.end_char,
                        },
                        surface: combined,
                        lemma: None,
                        upos: Some(Upos::Propn),
                        doc_tok_idx: atoms[j].doc_tok_idx,
                        head: atoms[j].head,
                        deprel: atoms[j].deprel.clone(),
                        is_locked: true,
                    });
                    i = j + 1;
                    found_close = true;
                    break;
                }
                j += 1;
            }
            if found_close {
                continue;
            }
        }
        out.push(atoms[i].clone());
        i += 1;
    }
    out
}

#[derive(Debug)]
enum Segment {
    Quote {
        open_q: char,
        inner_atoms: Vec<Atom>,
        close_q: char,
        span: Span,
    },
    OpenQuote {
        open_q: char,
        inner_atoms: Vec<Atom>,
        span: Span,
    },
    CloseQuote(char, Span),
    Clause {
        atoms: Vec<Atom>,
        trailing_punct: Option<String>,
        span: Span,
    },
    Punct(String, Span),
}

fn localize_open_quote(open_q: char, target: TargetVariant) -> char {
    if target == TargetVariant::ZhHans {
        if open_q == '『' { '‘' } else { '“' }
    } else if open_q == '『' {
        '『'
    } else {
        '「'
    }
}

fn localize_close_quote(close_q: char, target: TargetVariant) -> char {
    if target == TargetVariant::ZhHans {
        if close_q == '』' { '’' } else { '”' }
    } else if close_q == '』' {
        '』'
    } else {
        '」'
    }
}

fn is_te_subsidiary_atom(a: &Atom) -> bool {
    !a.is_locked
        && (giving_receiving_kind(a).is_some()
            || matches!(
                a.surface.as_str(),
                "いる"
                    | "いた"
                    | "い"
                    | "います"
                    | "いました"
                    | "る"
                    | "た"
                    | "ない"
                    | "なかっ"
                    | "なかった"
                    | "しまう"
                    | "しまっ"
                    | "しまった"
                    | "くる"
                    | "き"
                    | "きた"
                    | "いく"
                    | "いっ"
                    | "いった"
                    | "みる"
                    | "み"
                    | "みた"
                    | "くれる"
                    | "くれ"
                    | "くれた"
                    | "もらう"
                    | "もらっ"
                    | "もらった"
                    | "あげる"
                    | "あげ"
                    | "あげた"
                    | "ほしい"
                    | "ください"
                    | "くださる"
                    | "おく"
                    | "お"
                    | "ある"
                    | "あっ"
                    | "あった"
                    | "も"
                    | "は"
                    | "から"
            ))
}

fn is_te_form_link(base: &Atom, te: &Atom, auxiliary: &Atom) -> bool {
    !base.is_locked
        && !te.is_locked
        && !auxiliary.is_locked
        && base.upos == Some(Upos::Verb)
        && te.upos == Some(Upos::Sconj)
        && matches!(te.surface.as_str(), "て" | "で")
        && base.span.end_char == te.span.start_char
        && te.span.end_char == auxiliary.span.start_char
}

struct TeHelpRequest<'a> {
    action: &'a [Atom],
    person: Option<TeHelpPerson<'a>>,
    letter_addressee: Option<TeHelpLetterAddressee<'a>>,
    letter_writers: Option<(TeHelpPerson<'a>, &'a str)>,
    manner: Option<TeHelpManner<'a>>,
    child_suffix: Option<usize>,
    humble: bool,
    negative_form: bool,
}

struct TeHelpManner<'a> {
    start: usize,
    end: usize,
    count: Option<&'a str>,
    companion: Option<&'a Atom>,
}

struct TeHelpPerson<'a> {
    phrase: &'a [Atom],
    start: usize,
    end: usize,
    role: TeHelpPersonRole,
}

struct TeHelpLetterAddressee<'a> {
    members: Vec<&'a [Atom]>,
    count: Option<&'a str>,
    illustrative: bool,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TeHelpPersonRole {
    Recipient,
    Performer,
    UnresolvedLetterParticipant,
}

fn te_help_person<'a>(
    arguments: &'a [Atom],
    particle_index: usize,
    verb: &Atom,
    helper_idx: usize,
    letter_role: Option<TeHelpPersonRole>,
) -> Option<TeHelpPerson<'a>> {
    let noun_index = particle_index.checked_sub(1)?;
    let noun = &arguments[noun_index];
    let particle = &arguments[particle_index];
    let noun_idx = noun.doc_tok_idx?;
    let action_idx = verb.doc_tok_idx?;
    if noun.is_locked
        || particle.is_locked
        || !matches!(noun.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
        || !is_demonstrative_person_noun(noun)
        || !matches!(noun.deprel.as_str(), "obl" | "iobj")
        || !matches!(noun.head, Some(head) if head == action_idx || head == helper_idx)
        || particle.head != Some(noun_idx)
        || particle.deprel != "case"
        || noun.span.end_char != particle.span.start_char
    {
        return None;
    }
    let start = noun_index
        .checked_sub(1)
        .filter(|&index| attached_demonstrative(&arguments[index], noun).is_some())
        .unwrap_or(noun_index);
    if arguments[..start].iter().any(|atom| {
        atom.head == Some(noun_idx) && matches!(atom.deprel.as_str(), "det" | "compound" | "nmod")
    }) {
        return None;
    }

    let mut objects = arguments
        .iter()
        .filter(|atom| atom.deprel == "obj" && atom.head == Some(action_idx));
    let object = objects.next();
    let has_object = object.is_some();
    let lemma = verb.lemma.as_deref().unwrap_or(&verb.surface);
    let has_sending_recipient = lemma == "送る"
        && noun.deprel == "iobj"
        && object.is_some_and(|item| {
            !item.is_locked
                && matches!(
                    item.surface.as_str(),
                    "手紙"
                        | "メール"
                        | "電子メール"
                        | "葉書"
                        | "はがき"
                        | "小包"
                        | "荷物"
                        | "書類"
                        | "資料"
                        | "写真"
                        | "ファイル"
                        | "メッセージ"
                )
        })
        && objects.next().is_none();
    let role = if noun.head == Some(helper_idx) {
        TeHelpPersonRole::Performer
    } else if has_sending_recipient || (has_object && matches!(lemma, "返す" | "渡す" | "手渡す"))
    {
        TeHelpPersonRole::Recipient
    } else if has_object && lemma == "描く" {
        // Ordinary drawing takes the drawing as its object, not a recipient.
        TeHelpPersonRole::Performer
    } else {
        letter_role?
    };
    Some(TeHelpPerson {
        phrase: &arguments[start..particle_index],
        start,
        end: particle_index + 1,
        role,
    })
}

fn letter_address_nominal(atom: &Atom) -> bool {
    !atom.is_locked
        && atom.doc_tok_idx.is_some()
        && matches!(atom.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
}

fn letter_address_member_start(
    arguments: &[Atom],
    last: usize,
    address_idx: usize,
) -> Option<usize> {
    let name = &arguments[last];
    if !letter_address_nominal(name) || name.surface == "人" {
        return None;
    }
    let mut start = last;
    while let Some(modifier) = start.checked_sub(2) {
        let [noun, no, following] = &arguments[modifier..=start] else {
            return None;
        };
        if no.surface != "の" || no.span.end_char != following.span.start_char {
            break;
        }
        if !letter_address_nominal(noun)
            || !linked_letter_case(noun, no)
            || noun.deprel != "nmod"
            || (noun.head != following.doc_tok_idx && noun.head != Some(address_idx))
        {
            return None;
        }
        start = modifier;
    }
    if let Some(demonstrative) = start.checked_sub(1)
        && attached_demonstrative(&arguments[demonstrative], &arguments[start]).is_some()
    {
        start = demonstrative;
    }
    Some(start)
}

fn linked_letter_case(noun: &Atom, particle: &Atom) -> bool {
    !particle.is_locked
        && particle.upos == Some(Upos::Adp)
        && particle.deprel == "case"
        && noun.doc_tok_idx.is_some()
        && particle.head == noun.doc_tok_idx
        && noun.span.end_char == particle.span.start_char
}

fn te_help_letter_addressee(
    arguments: &[Atom],
    marker: usize,
    action_idx: usize,
    letter_idx: usize,
    separate_writers: bool,
) -> Option<TeHelpLetterAddressee<'_>> {
    let [address, tail @ ..] = &arguments[marker..] else {
        return None;
    };
    let address_idx = address.doc_tok_idx?;
    let has_suffix = matches!(tail, [suffix, ..] if suffix.surface == "て"
        && suffix.upos == Some(Upos::Sconj)
        && suffix.deprel == "mark"
        && suffix.head == Some(address_idx));
    let case_index = marker + 1 + usize::from(has_suffix);
    let particle = arguments.get(case_index)?;
    let attached_to_letter =
        particle.surface == "の" && address.deprel == "nmod" && address.head == Some(letter_idx);
    let attached_to_writing =
        particle.surface == "に" && address.deprel == "obl" && address.head == Some(action_idx);
    if particle.upos != Some(Upos::Adp)
        || particle.deprel != "case"
        || particle.head != Some(address_idx)
        || !(attached_to_letter || attached_to_writing)
        || te_help_has_following_focus(arguments, case_index + 1)
    {
        return None;
    }

    let mut start = marker.checked_sub(1)?;
    let mut count = None;
    let mut members = Vec::new();
    let mut conjunction = None;
    if let Some(number_idx) = marker.checked_sub(2)
        && let [number, people] = &arguments[number_idx..marker]
        && people.surface == "人"
        && people.deprel == "compound"
        && people.head == Some(address_idx)
        && number.upos == Some(Upos::Num)
        && number.deprel == "nummod"
        && number.head == Some(address_idx)
        && is_japanese_numeral(&number.surface)
    {
        count = Some(number.surface.as_str());
        start = number_idx;
    }
    let last_member = if count.is_some() {
        if start >= 2 && arguments[start - 1].surface == "の" {
            let (name, no) = (&arguments[start - 2], &arguments[start - 1]);
            if name.deprel != "nmod"
                || name.head != Some(address_idx)
                || !linked_letter_case(name, no)
            {
                return None;
            }
            Some(start - 2)
        } else {
            None
        }
    } else {
        let name = &arguments[start];
        if name.deprel != "compound" || name.head != Some(address_idx) {
            return None;
        }
        Some(start)
    };
    if let Some(last) = last_member {
        start = letter_address_member_start(arguments, last, address_idx)?;
        members.push(&arguments[start..=last]);
        let mut next_head = arguments[last].doc_tok_idx;
        while let Some(previous) = start.checked_sub(2) {
            let (name, joint) = (&arguments[previous], &arguments[previous + 1]);
            if !matches!(joint.surface.as_str(), "と" | "や")
                || joint.span.end_char != arguments[start].span.start_char
            {
                break;
            }
            let linked_conjunct =
                name.deprel == "conj" && (name.head == Some(address_idx) || name.head == next_head);
            // The parser may attach a と conjunct to 書く. Only extend that
            // span with a separate writer group or an explicit addressee count.
            let scoped_conjunct = joint.surface == "と"
                && (separate_writers || count.is_some())
                && name.deprel == "obl"
                && name.head == Some(action_idx);
            if !(linked_conjunct || scoped_conjunct)
                || !linked_letter_case(name, joint)
                || conjunction.is_some_and(|seen| seen != joint.surface.as_str())
            {
                return None;
            }
            conjunction = Some(joint.surface.as_str());
            start = letter_address_member_start(arguments, previous, address_idx)?;
            members.push(&arguments[start..=previous]);
            next_head = name.doc_tok_idx;
        }
        members.reverse();
    }
    let end = case_index + 1;
    if count.is_some_and(|number| {
        !members.is_empty()
            && super::opaque::parse_cjk_or_ascii_num(number).map(|n| n as usize)
                != Some(members.len())
    }) || arguments[start..end].iter().any(|atom| atom.is_locked)
        || !arguments[start..end]
            .windows(2)
            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
        || arguments.iter().enumerate().any(|(index, atom)| {
            !(start..end).contains(&index)
                && matches!(
                    atom.deprel.as_str(),
                    "det" | "compound" | "nmod" | "nummod" | "conj"
                )
                && (atom.head == Some(address_idx)
                    || members.iter().any(|member| {
                        member
                            .iter()
                            .any(|noun| noun.doc_tok_idx.is_some() && atom.head == noun.doc_tok_idx)
                    }))
        })
    {
        return None;
    }
    Some(TeHelpLetterAddressee {
        members,
        count,
        illustrative: conjunction == Some("や"),
        start,
        end,
    })
}

fn te_help_letter_writers(
    arguments: &[Atom],
    start: usize,
    action_idx: usize,
    helper_idx: usize,
) -> Option<(TeHelpPerson<'_>, &str)> {
    let [name, plural, tail @ ..] = &arguments[start..] else {
        return None;
    };
    let prefix = usize::from(tail.first().is_some_and(|atom| atom.surface == "お"));
    let [number, people, particle, ..] = &tail[prefix..] else {
        return None;
    };
    let plural_links_to_action = plural.head == Some(action_idx)
        || arguments.iter().any(|atom| {
            atom.doc_tok_idx.is_some()
                && atom.doc_tok_idx == plural.head
                && !atom.is_locked
                && atom.upos == Some(Upos::Adv)
                && atom.deprel == "advmod"
                && atom.head == Some(action_idx)
        });
    let end = start + 5 + prefix;
    if !is_demonstrative_person_noun(name)
        || name.doc_tok_idx.is_none()
        || arguments[..start].iter().any(|atom| {
            atom.head == name.doc_tok_idx
                && matches!(atom.deprel.as_str(), "det" | "compound" | "nmod")
        })
        || name.deprel != "obl"
        || !matches!(name.head, Some(head) if head == action_idx || head == helper_idx)
        || plural.surface != "方"
        || plural.deprel != "advmod"
        || !plural_links_to_action
        || (prefix != 0 && tail[0].head != Some(helper_idx))
        || number.upos != Some(Upos::Num)
        || number.deprel != "nummod"
        || !is_japanese_numeral(&number.surface)
        || people.surface != "人"
        || people.doc_tok_idx.is_none()
        || number.head != people.doc_tok_idx
        || people.deprel != "obl"
        || !matches!(people.head, Some(head) if head == action_idx || head == helper_idx)
        || particle.surface != "に"
        || !linked_joint_case(people, particle)
        || arguments[start..end].iter().any(|atom| atom.is_locked)
        || !arguments[start..end]
            .windows(2)
            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
        || te_help_has_following_focus(arguments, end)
    {
        return None;
    }
    Some((
        TeHelpPerson {
            phrase: &arguments[start..start + 1],
            start,
            end,
            role: TeHelpPersonRole::Performer,
        },
        number.surface.as_str(),
    ))
}

fn te_help_letter_companion(
    arguments: &[Atom],
    start: usize,
    action_idx: usize,
) -> Option<TeHelpManner<'_>> {
    let [name, to, number, people, de, ..] = &arguments[start..] else {
        return None;
    };
    let end = start + 5;
    if to.surface != "と"
        || !is_demonstrative_person_noun(name)
        || name.head != Some(action_idx)
        || !linked_joint_case(name, to)
        || !matches!(
            normalize_decimal_digits(&number.surface).as_str(),
            "二" | "2"
        )
        || people.head != Some(action_idx)
        || !linked_joint_people(number, people, de)
        || to.span.end_char != number.span.start_char
        || te_help_has_following_focus(arguments, end)
    {
        return None;
    }
    Some(TeHelpManner {
        start,
        end,
        count: Some(number.surface.as_str()),
        companion: Some(name),
    })
}

// Some lexical child compounds contain ども even when the tokenizer gives it
// a connective tag. The nominal modifier and its の must both be source-linked.
fn te_help_child_suffix(arguments: &[Atom], action_idx: usize) -> Option<usize> {
    arguments.windows(3).enumerate().find_map(|(index, group)| {
        let child = &group[0];
        let suffix = &group[1];
        let genitive = &group[2];
        let child_idx = child.doc_tok_idx?;
        (!group.iter().any(|atom| atom.is_locked)
            && child.surface == "子"
            && child.upos == Some(Upos::Noun)
            && child.deprel == "nmod"
            && suffix.surface == "ども"
            && suffix.upos == Some(Upos::Sconj)
            && suffix.deprel == "mark"
            && suffix.head == Some(child_idx)
            && genitive.surface == "の"
            && genitive.upos == Some(Upos::Adp)
            && genitive.deprel == "case"
            && genitive.head == Some(child_idx)
            && child.span.end_char == suffix.span.start_char
            && suffix.span.end_char == genitive.span.start_char
            && arguments[index + 3..].iter().any(|object| {
                object.doc_tok_idx.is_some()
                    && child.head == object.doc_tok_idx
                    && object.deprel == "obj"
                    && object.head == Some(action_idx)
            }))
        .then_some(index + 1)
    })
}

fn te_help_has_following_focus(arguments: &[Atom], end: usize) -> bool {
    arguments.get(end).is_some_and(|atom| {
        matches!(atom.surface.as_str(), "も" | "は")
            && arguments[end - 1].span.end_char == atom.span.start_char
    })
}

fn te_help_request<'a>(atoms: &'a [Atom], doc: &Document) -> Option<TeHelpRequest<'a>> {
    let (body, helper, polite, question, negative_form) = match atoms {
        [body @ .., helper, mase, n, question] if mase.surface == "ませ" && n.surface == "ん" => {
            (
                body,
                helper,
                &atoms[body.len() + 1..atoms.len() - 1],
                question,
                true,
            )
        }
        [body @ .., helper, masu, question] if masu.surface == "ます" => {
            (body, helper, std::slice::from_ref(masu), question, false)
        }
        _ => return None,
    };
    let humble = match (helper.surface.as_str(), helper.lemma.as_deref()) {
        ("いただけ", Some("いただける")) => true,
        ("もらえ", Some("もらえる")) => false,
        _ => return None,
    };
    let helper_idx = helper.doc_tok_idx?;
    let [arguments @ .., verb, link] = body else {
        return None;
    };
    let action_idx = verb.doc_tok_idx?;
    if !is_te_form_link(verb, link, helper)
        || helper.upos != Some(Upos::Verb)
        || helper.deprel != "root"
        || verb.head != Some(helper_idx)
        || verb.deprel != "advcl"
        || link.head != Some(action_idx)
        || link.deprel != "mark"
        || polite.iter().any(|atom| {
            atom.is_locked
                || atom.upos != Some(Upos::Aux)
                || atom.head != Some(helper_idx)
                || atom.deprel != "aux"
        })
        || question.is_locked
        || question.surface != "か"
        || question.upos != Some(Upos::Part)
        || question.head != Some(helper_idx)
        || question.deprel != "discourse"
        || atoms[body.len()..]
            .windows(2)
            .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
        // Show and lend have their own recipient. Do not turn either into
        // an action the speaker would perform with the listener's help.
        || matches!(verb.lemma.as_deref(), Some("見せる" | "貸す"))
        || verb.surface.ends_with("ない")
    {
        return None;
    }

    let mut manners = (0..arguments.len()).filter_map(|start| {
        let (size, count) = match &arguments[start..] {
            [word, ni, ..]
                if word.surface == "一緒"
                    && ni.surface == "に"
                    && word.head == Some(action_idx)
                    && linked_joint_case(word, ni) =>
            {
                (2, None)
            }
            [number, people, de, ..]
                if people.head == Some(action_idx) && linked_joint_people(number, people, de) =>
            {
                (3, Some(number.surface.as_str()))
            }
            _ => return None,
        };
        let end = start + size;
        if start.checked_sub(1).is_some_and(|index| {
            let before = &arguments[index];
            before.span.end_char == arguments[start].span.start_char
                && (before.surface == "と" && before.deprel == "case"
                    || count.is_some() && is_japanese_numeral(&before.surface))
        }) || matches!(&arguments[end..], [companion, to, ..]
            if to.surface == "と"
                && companion.head == Some(action_idx)
                && linked_joint_case(companion, to))
            || te_help_has_following_focus(arguments, end)
        {
            return None;
        }
        Some(TeHelpManner {
            start,
            end,
            count,
            companion: None,
        })
    });
    let mut manner = manners.next();
    if manners.next().is_some() {
        return None;
    }

    // A normalized clock can be stored as a locked atom. Verify its original
    // endpoint and both particles before exempting the span from the dative check.
    let mut deadlines = arguments
        .windows(3)
        .enumerate()
        .filter_map(|(start, group)| {
            let time = &group[0];
            let end = start + group.len();
            ((!time.is_locked || time.lemma.as_deref() == Some(CLOCK_ATOM))
                && deadline_time_question(time).is_none()
                && !te_help_has_following_focus(arguments, end)
                && has_temporal_deadline(&arguments[..end], doc)
                && deadline_case_governor(&group[2], doc) == Some(action_idx))
            .then_some(start..end)
        });
    let deadline = deadlines.next();
    if deadlines.next().is_some() {
        return None;
    }

    let mut objects = arguments
        .iter()
        .filter(|atom| atom.deprel == "obj" && atom.head == Some(action_idx));
    let letter_idx = if verb.lemma.as_deref() == Some("書く") {
        objects
            .next()
            .filter(|object| {
                !object.is_locked && object.surface == "手紙" && objects.next().is_none()
            })
            .and_then(|object| object.doc_tok_idx)
    } else {
        None
    };
    let letter_marker = if let Some(letter_idx) = letter_idx {
        let mut markers = arguments.iter().enumerate().filter(|(_, atom)| {
            matches!(atom.surface.as_str(), "宛" | "宛て")
                && ((atom.deprel == "nmod" && atom.head == Some(letter_idx))
                    || (atom.deprel == "obl" && atom.head == Some(action_idx)))
        });
        let marker = markers.next().map(|(index, _)| (index, letter_idx));
        if markers.next().is_some() {
            return None;
        }
        marker
    } else {
        None
    };
    let letter_writers = if letter_marker.is_some() {
        let mut groups = (0..arguments.len())
            .filter_map(|start| te_help_letter_writers(arguments, start, action_idx, helper_idx));
        let writers = groups.next();
        if groups.next().is_some()
            || (writers.is_some() && manner.as_ref().is_some_and(|bound| bound.count.is_some()))
        {
            return None;
        }
        writers
    } else {
        None
    };
    let letter_addressee = if let Some((marker, letter_idx)) = letter_marker {
        Some(te_help_letter_addressee(
            arguments,
            marker,
            action_idx,
            letter_idx,
            letter_writers.is_some(),
        )?)
    } else {
        None
    };
    if letter_addressee.is_some() && manner.is_none() && letter_writers.is_none() {
        let mut companions = (0..arguments.len())
            .filter_map(|start| te_help_letter_companion(arguments, start, action_idx));
        manner = companions.next();
        if companions.next().is_some() {
            return None;
        }
    }
    if let Some(addressee) = &letter_addressee
        && (manner
            .as_ref()
            .is_some_and(|manner| manner.start < addressee.end && addressee.start < manner.end)
            || letter_writers.as_ref().is_some_and(|(writers, _)| {
                writers.start < addressee.end && addressee.start < writers.end
            }))
    {
        return None;
    }
    if letter_idx.is_some()
        && let Some(bound) = &mut manner
        && bound.count.is_some()
        && bound.companion.is_none()
        && let Some(prior) = bound.start.checked_sub(1)
    {
        let prefix = &arguments[prior];
        if !prefix.is_locked
            && prefix.surface == "お"
            && prefix.head == Some(helper_idx)
            && prefix.span.end_char == arguments[bound.start].span.start_char
        {
            bound.start = prior;
        }
    }
    let is_modifier = |index| {
        manner
            .as_ref()
            .is_some_and(|bound| (bound.start..bound.end).contains(&index))
            || deadline
                .as_ref()
                .is_some_and(|bound| bound.contains(&index))
            || letter_addressee
                .as_ref()
                .is_some_and(|bound| (bound.start..bound.end).contains(&index))
            || letter_writers
                .as_ref()
                .is_some_and(|(bound, _)| (bound.start..bound.end).contains(&index))
    };

    let child_suffix = te_help_child_suffix(arguments, action_idx);
    let letter_role = if letter_addressee.is_some() {
        Some(TeHelpPersonRole::Performer)
    } else if letter_idx.is_some() && manner.as_ref().is_some_and(|bound| bound.count.is_some()) {
        Some(TeHelpPersonRole::UnresolvedLetterParticipant)
    } else {
        None
    };
    let mut datives = arguments.iter().enumerate().filter(|&(index, atom)| {
        atom.upos == Some(Upos::Adp) && atom.surface == "に" && !is_modifier(index)
    });
    let person = if let Some((index, _)) = datives.next() {
        if datives.next().is_some()
            || letter_writers.is_some()
            || manner
                .as_ref()
                .is_some_and(|bound| bound.companion.is_some())
        {
            return None;
        }
        Some(te_help_person(
            arguments,
            index,
            verb,
            helper_idx,
            letter_role,
        )?)
    } else {
        None
    };
    for (index, atom) in arguments.iter().enumerate() {
        if is_modifier(index)
            || person
                .as_ref()
                .is_some_and(|bound| (bound.start..bound.end).contains(&index))
        {
            continue;
        }
        if atom.is_locked {
            if atom.deprel != "obj" || atom.head != Some(action_idx) {
                return None;
            }
            continue;
        }
        if (matches!(atom.upos, Some(Upos::Verb | Upos::Sconj)) && Some(index) != child_suffix)
            || matches!(atom.deprel.as_str(), "nsubj" | "iobj")
            || (atom.upos == Some(Upos::Aux)
                && matches!(
                    atom.surface.as_str(),
                    "ない" | "なく" | "なかった" | "なかっ"
                ))
            || (atom.upos == Some(Upos::Adp)
                && matches!(atom.surface.as_str(), "は" | "が" | "へ" | "から"))
        {
            return None;
        }
    }

    let mut lead_start = 0;
    for (comma, separator) in arguments
        .iter()
        .enumerate()
        .filter(|(_, atom)| is_punct_atom(atom))
    {
        let lead = &arguments[lead_start..comma];
        if !matches!(separator.surface.as_str(), "、" | "，" | ",")
            || lead.is_empty()
            || (lead_start > 0 && letter_addressee.is_none())
            || !lead.iter().enumerate().all(|(offset, adverb)| {
                let index = lead_start + offset;
                is_modifier(index)
                    || (letter_role.is_some()
                        && person
                            .as_ref()
                            .is_some_and(|bound| (bound.start..bound.end).contains(&index)))
                    || (!adverb.is_locked
                        && adverb.upos == Some(Upos::Adv)
                        && adverb.deprel == "advmod"
                        && (adverb.head == Some(action_idx)
                            || arguments[comma + 1..].iter().any(|next| {
                                next.doc_tok_idx.is_some()
                                    && adverb.head == next.doc_tok_idx
                                    && next.deprel == "advmod"
                                    && next.head == Some(action_idx)
                            })))
            })
        {
            return None;
        }
        lead_start = comma + 1;
    }

    Some(TeHelpRequest {
        action: &body[..body.len() - 1],
        person,
        letter_addressee,
        letter_writers,
        manner,
        child_suffix,
        humble,
        negative_form,
    })
}

/// Remove descriptive honorific wrappers before clause splitting. Keep all finite
/// auxiliaries, source spans and locks so tense, negation and alignment stay local.
fn normalize_keigo_atoms(atoms: &[Atom]) -> Vec<Atom> {
    let mut out = Vec::with_capacity(atoms.len());
    let mut i = 0;
    while i < atoms.len() {
        let a = &atoms[i];
        if !a.is_locked
            && a.surface == "お"
            && a.upos == Some(Upos::Intj)
            && let Some(base) = atoms.get(i + 1)
            && !base.is_locked
            && base.upos == Some(Upos::Verb)
            && base
                .lemma
                .as_deref()
                .is_some_and(|lemma| lemma != base.surface)
        {
            let tail = if atoms
                .get(i + 2)
                .is_some_and(|ni| !ni.is_locked && ni.surface == "に" && ni.upos == Some(Upos::Adp))
            {
                atoms
                    .get(i + 3)
                    .filter(|tail| {
                        !tail.is_locked
                            && ((tail.upos == Some(Upos::Verb)
                                && tail.lemma.as_deref().unwrap_or(&tail.surface) == "なる")
                                || (tail.surface == "なら"
                                    && tail.upos == Some(Upos::Sconj)
                                    && atoms.get(i + 4).is_some_and(|neg| {
                                        !neg.is_locked
                                            && neg.upos == Some(Upos::Aux)
                                            && tail.span.end_char == neg.span.start_char
                                            && matches!(neg.surface.as_str(), "ない" | "なかっ")
                                    })))
                    })
                    .map(|tail| (tail, 4))
            } else {
                atoms
                    .get(i + 2)
                    .filter(|tail| {
                        !tail.is_locked
                            && tail.upos == Some(Upos::Verb)
                            && tail.lemma.as_deref().unwrap_or(&tail.surface) == "する"
                    })
                    .map(|tail| (tail, 3))
            };
            if let Some((tail, consumed)) = tail
                && atoms[i..i + consumed]
                    .windows(2)
                    .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
            {
                let mut verb = base.clone();
                verb.span = Span {
                    start_char: a.span.start_char,
                    end_char: tail.span.end_char,
                };
                verb.doc_tok_idx = tail.doc_tok_idx;
                verb.head = tail.head;
                verb.deprel.clone_from(&tail.deprel);
                out.push(verb);
                i += consumed;
                continue;
            }
        }

        // The tokenizer emits `ください` for the stem of `くださる`, too.
        // Only a following finite polite ending makes this a statement.
        let after_te = i >= 2 && is_te_form_link(&atoms[i - 2], &atoms[i - 1], a);
        if after_te && !a.is_locked && a.lemma.as_deref() == Some("くださる") {
            let mut verb = a.clone();
            if a.surface == "ください"
                && let Some(next) = atoms.get(i + 1)
                && !next.is_locked
                && a.span.end_char == next.span.start_char
            {
                if next.upos == Some(Upos::Aux)
                    && (next.surface == "ます"
                        || (next.surface == "まし"
                            && atoms.get(i + 2).is_some_and(|ta| {
                                !ta.is_locked
                                    && ta.surface == "た"
                                    && ta.upos == Some(Upos::Aux)
                                    && next.span.end_char == ta.span.start_char
                            })))
                {
                    verb.surface = "くださる".to_owned();
                    out.push(verb);
                    i += 1;
                    continue;
                }
                // `くださいました` can be split as `ください + ま + し + た`.
                if next.surface == "ま"
                    && next.upos == Some(Upos::Intj)
                    && let Some(shi) = atoms.get(i + 2)
                    && !shi.is_locked
                    && shi.surface == "し"
                    && shi.upos == Some(Upos::Verb)
                    && let Some(ta) = atoms.get(i + 3)
                    && !ta.is_locked
                    && ta.surface == "た"
                    && ta.upos == Some(Upos::Aux)
                    && next.span.end_char == shi.span.start_char
                    && shi.span.end_char == ta.span.start_char
                {
                    verb.surface = "くださる".to_owned();
                    out.push(verb);
                    let mut polite = next.clone();
                    polite.surface = "まし".to_owned();
                    polite.lemma = Some("ます".to_owned());
                    polite.upos = Some(Upos::Aux);
                    polite.span.end_char = shi.span.end_char;
                    out.push(polite);
                    i += 3;
                    continue;
                }
            } else if a.surface == "くださいませ" {
                let negative = atoms.get(i + 1).is_some_and(|neg| {
                    !neg.is_locked
                        && neg.surface == "ん"
                        && neg.upos == Some(Upos::Aux)
                        && a.span.end_char == neg.span.start_char
                });
                let negative_past = atoms.get(i + 1..i + 4).is_some_and(|tail| {
                    tail.iter().all(|a| !a.is_locked)
                        && tail[0].surface == "んで"
                        && tail[0].upos == Some(Upos::Sconj)
                        && tail[1].surface == "し"
                        && tail[1].upos == Some(Upos::Verb)
                        && tail[2].surface == "た"
                        && tail[2].upos == Some(Upos::Aux)
                        && a.span.end_char == tail[0].span.start_char
                        && tail
                            .windows(2)
                            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
                });
                if !negative && !negative_past {
                    out.push(a.clone());
                    i += 1;
                    continue;
                }
                verb.surface = "くださる".to_owned();
                verb.span.end_char -= 2;
                out.push(verb);
                let mut polite = a.clone();
                polite.span.start_char = polite.span.end_char - 2;
                polite.surface = "ませ".to_owned();
                polite.lemma = Some("ます".to_owned());
                polite.upos = Some(Upos::Aux);
                out.push(polite);
                // `くださいませんでした` may end in `んで + し + た`.
                if negative_past {
                    let mut neg = atoms[i + 1].clone();
                    neg.surface = "ん".to_owned();
                    neg.lemma = None;
                    neg.upos = Some(Upos::Aux);
                    neg.span.end_char = neg.span.start_char + 1;
                    let mut copula = atoms[i + 2].clone();
                    copula.span.start_char = neg.span.end_char;
                    copula.surface = "でし".to_owned();
                    copula.lemma = Some("です".to_owned());
                    copula.upos = Some(Upos::Aux);
                    out.push(neg);
                    out.push(copula);
                    i += 3;
                    continue;
                }
                i += 1;
                continue;
            }
        }
        out.push(a.clone());
        i += 1;
    }
    out
}

fn has_temporal_te_kara_tail(atoms: &[Atom]) -> bool {
    let Some(i) = atoms.iter().rposition(|a| !is_punct_atom(a)) else {
        return false;
    };
    i >= 2
        && !atoms[i].is_locked
        && atoms[i].surface == "から"
        && !atoms[i - 1].is_locked
        && matches!(atoms[i - 1].surface.as_str(), "て" | "で")
        && atoms[i - 1].upos == Some(Upos::Sconj)
        && !atoms[i - 2].is_locked
        && atoms[i - 2].upos == Some(Upos::Verb)
}

// Japanese can write a full stop after the question particle. Only replace
// the stop belonging to this clause. A quoted or embedded か cannot change
// the punctuation of a later clause.
fn direct_ka_before_period(clause: &[Atom], end: &Atom) -> bool {
    let Some((ka, body)) = clause.split_last() else {
        return false;
    };
    end.surface == "。"
        && !ka.is_locked
        && ka.surface == "か"
        && ka.upos == Some(Upos::Part)
        && ka.deprel == "discourse"
        && ka.span.end_char == end.span.start_char
        && ka.head.is_some()
        && body.iter().any(|a| {
            !a.is_locked
                && a.doc_tok_idx == ka.head
                && (matches!(a.upos, Some(Upos::Verb | Upos::Adj)) || a.deprel == "root")
        })
}

#[derive(Clone, Copy, Debug)]
struct ReturnTrip {
    manner: usize,
    returning: usize,
    action: &'static str,
}

fn is_nominal_adverb(a: &Atom, doc: &Document) -> bool {
    a.doc_tok_idx
        .and_then(|i| doc.tokens.get(i))
        .is_some_and(|t| {
            t.upos_candidates.contains(&Upos::Adv) && t.upos_candidates.contains(&Upos::Noun)
        })
}

// A final と can mark a travelling companion. An earlier と can instead
// share the next person's case, as in 私と妹は or 母と妹と.
fn return_companion_owner(source: &[Atom], marker: usize) -> Option<usize> {
    let case = &source[marker];
    if case.surface != "と"
        || case.is_locked
        || case.deprel != "case"
        || case.upos != Some(Upos::Adp)
    {
        return None;
    }
    let noun = source[..marker]
        .iter()
        .rev()
        .take(2)
        .find(|a| a.doc_tok_idx.is_some() && a.doc_tok_idx == case.head)?;
    let mut owner = noun.head?;
    if noun.is_locked
        || noun.deprel != "obl"
        || !(linked_joint_case(noun, case)
            || marker >= 2 && indefinite_argument(&source[marker - 2..=marker], owner).is_some())
    {
        return None;
    }
    // In 妹と古い家へ the parser can attach 妹 to the adjective of the
    // destination. Its own direction marker identifies the motion verb.
    if let Some(adj) = source.get(marker + 1)
        && adj.doc_tok_idx == Some(owner)
        && !adj.is_locked
        && adj.upos == Some(Upos::Adj)
        && adj.deprel == "amod"
        && let Some([place, direction]) = source.get(marker + 2..marker + 4)
        && adj.head == place.doc_tok_idx
        && linked_joint_case(place, direction)
        && matches!(direction.surface.as_str(), "へ" | "に" | "まで")
        && is_existing_spatial_head(place)
    {
        owner = place.head?;
    }
    let next = source[marker + 1..]
        .windows(2)
        .take_while(|p| !is_punct_atom(&p[0]) && p[0].upos != Some(Upos::Verb))
        .find(|p| {
            !p[0].is_locked
                && p[0].head == Some(owner)
                && p[0].doc_tok_idx.is_some()
                && !p[1].is_locked
                && p[1].head == p[0].doc_tok_idx
                && p[1].deprel == "case"
        });
    if next.is_some_and(|p| {
        p[0].deprel == "nsubj"
            || is_companion_person(&p[0]) && !matches!(p[1].surface.as_str(), "の" | "から")
    }) {
        return None;
    }
    // The motion verb itself can close Nと before the destination appears.
    // This also retains an unlisted name or an animal without guessing its type.
    (is_companion_person(noun)
        || next.is_none()
            && source[marker + 1..]
                .iter()
                .take_while(|a| !is_punct_atom(a))
                .find(|a| a.upos == Some(Upos::Verb))
                .is_some_and(|verb| {
                    verb.doc_tok_idx == Some(owner) && travel_manner(verb).is_some()
                }))
    .then_some(owner)
}

fn travel_manner(verb: &Atom) -> Option<&'static str> {
    match verb.lemma.as_deref().unwrap_or(&verb.surface) {
        "歩く" => Some("走"),
        "走る" | "駆ける" => Some("跑"),
        "泳ぐ" => Some("游"),
        "飛ぶ" => Some("飛"),
        _ => None,
    }
}

fn return_goal(group: &Bunsetsu, returning: usize) -> bool {
    matches!(group.atoms.as_slice(), [.., noun, case]
        if !noun.is_locked && !case.is_locked && noun.head == Some(returning)
        && noun.doc_tok_idx.is_some() && case.head == noun.doc_tok_idx && case.deprel == "case"
        && matches!(case.surface.as_str(), "へ" | "に" | "まで")
        && !source_time_head(noun)
        && (physical_goal_head(group) == Some(returning)
            || case.surface == "へ" || is_spatial_made_bunsetsu(group)
            || is_place_location_bunsetsu(&group.atoms)
            || matches!(noun.surface.as_str(), "店" | "岸" | "島" | "庭")))
}

// Keep each return argument's source role before adjacent nouns are folded.
// The original governor is still available when the two verbs need separate clauses.
fn mark_return_arguments(groups: &mut [Bunsetsu], source: &[Atom], doc: &Document) {
    for action in source.iter().filter(|a| {
        !a.is_locked
            && a.upos == Some(Upos::Verb)
            && (matches!(a.lemma.as_deref().unwrap_or(&a.surface), "帰る" | "戻る")
                || travel_manner(a).is_some()
                    && a.deprel == "advcl"
                    && a.head
                        .and_then(|i| doc.tokens.get(i))
                        .is_some_and(|verb| matches!(verb.lemma.as_deref(), Some("帰る" | "戻る"))))
    }) {
        let Some(owner) = action.doc_tok_idx else {
            continue;
        };
        for group in &mut *groups {
            let Some(tail) = group.atoms.last() else {
                continue;
            };
            if let [adv] = group.atoms.as_slice()
                && adv.is_locked
                && adv.upos == Some(Upos::Adv)
                && adv.doc_tok_idx.is_some()
                && let Some(adj) = source.iter().find(|a| a.doc_tok_idx == adv.doc_tok_idx)
                && !adj.is_locked
                && adj.head == Some(owner)
                && adj.deprel == "advcl"
                && adj.upos == Some(Upos::Adj)
                && adj.surface.ends_with('く')
            {
                group.role = BunsetsuRole::Adv;
                if adj.lemma.as_deref() == Some("早い")
                    && source.iter().any(|a| {
                        a.head == adj.doc_tok_idx
                            && a.deprel == "advmod"
                            && (source_time_head(a) || is_nominal_adverb(a, doc))
                    })
                {
                    group.atoms[0].surface = "早早地".to_owned();
                }
            } else if matches!(group.atoms.as_slice(), [noun, case] if noun.head == Some(owner)
                && noun.surface == "一緒" && case.surface == "に" && linked_joint_case(noun, case))
            {
                group.role = BunsetsuRole::Adv;
            } else if tail.surface == "と"
                && tail.doc_tok_idx.is_some()
                && let Some(i) = source
                    .iter()
                    .position(|a| a.doc_tok_idx == tail.doc_tok_idx)
                && return_companion_owner(source, i) == Some(owner)
            {
                if i >= 2
                    && let Some((noun, case)) = indefinite_argument(&source[i - 2..=i], owner)
                {
                    group.atoms = vec![noun];
                    group.atoms.extend(case);
                    group.sfp_tail = None;
                }
                group.role = BunsetsuRole::TravelCompanion;
            }
        }
    }
}

// A linked manner and return describe one trip. A second subject, a
// duration of walking or two destinations can instead distinguish two events.
fn return_trip_at(atoms: &[Atom], link: usize, doc: &Document) -> Option<ReturnTrip> {
    let manner = atoms.get(link.checked_sub(1)?)?;
    let te = atoms.get(link)?;
    let action = travel_manner(manner)?;
    let first = manner.doc_tok_idx?;
    let last = manner.head?;
    if manner.is_locked
        || te.is_locked
        || manner.upos != Some(Upos::Verb)
        || manner.deprel != "advcl"
        || te.upos != Some(Upos::Sconj)
        || !matches!(te.surface.as_str(), "て" | "で")
        || te.deprel != "mark"
        || te.head != Some(first)
        || manner.span.end_char != te.span.start_char
    {
        return None;
    }
    let start = atoms[..link]
        .iter()
        .rposition(is_punct_atom)
        .map_or(0, |i| i + 1);
    let clause = &atoms[start..];
    let verb = atoms[link + 1..]
        .iter()
        .take_while(|a| !is_punct_atom(a))
        .find(|a| a.doc_tok_idx == Some(last))?;
    if verb.is_locked
        || verb.upos != Some(Upos::Verb)
        || verb.deprel != "root"
        || !matches!(
            verb.lemma.as_deref().unwrap_or(&verb.surface),
            "帰る" | "戻る"
        )
    {
        return None;
    }
    let mut goals = 0;
    let mut origins = 0;
    let mut time_owner = None;
    let mut companion_owner = None;
    for (i, a) in clause.iter().take_while(|a| !is_punct_atom(a)).enumerate() {
        if a.lemma.as_deref() == Some(CLOCK_ATOM) && !a.surface.contains("[[PH") {
            if clause.get(i + 1).is_some_and(|p| p.surface == "まで") {
                return None;
            }
            // A normalized clock can cover several tokens, including part
            // of 時に. Recover its governing verb from the source span.
            let owner = doc.syntax.iter().find_map(|node| {
                node.head.filter(|head| {
                    (*head == first || *head == last)
                        && node.span.start_char < a.span.end_char
                        && a.span.start_char < node.span.end_char
                })
            })?;
            if time_owner.replace(owner).is_some_and(|prev| prev != owner) {
                return None;
            }
            continue;
        }
        if (a.is_locked && (a.lemma.as_deref() != Some(CLOCK_ATOM) || a.surface.contains("[[PH")))
            || (a.upos == Some(Upos::Verb)
                && a.doc_tok_idx != Some(first)
                && a.doc_tok_idx != Some(last))
            || (a.head == Some(last)
                && a.deprel == "nsubj"
                && a.span.start_char > te.span.start_char)
        {
            return None;
        }
        if let Some(owner) =
            return_companion_owner(clause, i).filter(|&owner| owner == first || owner == last)
            && companion_owner
                .replace(owner)
                .is_some_and(|prev| prev != owner)
        {
            return None;
        }
        let Some(head) = a.head else { continue };
        // Nominal adverbs such as 今朝 can appear bare, as a topic or
        // with a case. Their source role survives even without a time gloss.
        let time = render_temporal_made_atom(a).is_some();
        if (time || is_nominal_adverb(a, doc))
            && matches!(a.deprel.as_str(), "advmod" | "obl" | "iobj" | "nsubj")
        {
            let mut owner = Some(head);
            // 昨日ゆっくり歩く can attach 昨日 to ゆっくり first.
            if a.deprel == "advmod"
                && doc
                    .tokens
                    .get(head)
                    .is_some_and(|t| matches!(t.upos, Some(Upos::Adv | Upos::Adj)))
            {
                owner = doc
                    .syntax
                    .iter()
                    .find(|n| n.token == head && matches!(n.deprel.as_str(), "advmod" | "advcl"))
                    .and_then(|n| n.head);
            }
            if let Some(owner) = owner.filter(|&owner| owner == first || owner == last)
                && time_owner.replace(owner).is_some_and(|prev| prev != owner)
            {
                return None;
            }
        }
        if head != first && head != last {
            continue;
        }
        if a.deprel == "obj" {
            return None;
        }
        if a.doc_tok_idx.is_none() {
            continue;
        }
        let Some(case) = clause
            .get(i + 1)
            .filter(|case| case.head == a.doc_tok_idx && case.deprel == "case")
        else {
            continue;
        };
        // A separate activity site or means of transport need not belong to
        // the return (公園で走って家に戻る, 車で走って家に帰る).
        if case.surface == "で" {
            return None;
        }
        if !matches!(a.deprel.as_str(), "obl" | "iobj") {
            continue;
        }
        if case.surface == "から" {
            origins += 1;
        }
        if matches!(case.surface.as_str(), "に" | "へ" | "まで") {
            if time {
                if case.surface == "まで" {
                    return None;
                }
            } else if !(a.surface == "一緒" && case.surface == "に" && linked_joint_case(a, case))
            {
                goals += 1;
            }
        }
    }
    (goals <= 1 && origins <= 1).then_some(ReturnTrip {
        manner: first,
        returning: last,
        action,
    })
}

fn split_atoms_into_segments(atoms: &[Atom], doc: &Document, parent_span: Span) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current: Vec<Atom> = Vec::new();
    let mut i = 0usize;

    let flush_clause = |buf: &mut Vec<Atom>, punct: Option<String>, out: &mut Vec<Segment>| {
        if !buf.is_empty() {
            let s = buf
                .first()
                .map(|a| a.span.start_char)
                .unwrap_or(parent_span.start_char);
            let e = buf.last().map(|a| a.span.end_char).unwrap_or(s);
            out.push(Segment::Clause {
                atoms: std::mem::take(buf),
                trailing_punct: punct,
                span: Span {
                    start_char: s,
                    end_char: e,
                },
            });
        } else if let Some(p) = punct {
            out.push(Segment::Punct(p, parent_span));
        }
    };

    // Keep a verified whole request together. The ordinary splitter can
    // otherwise treat its potential helper as a separate lexical action.
    let request_atoms = core_non_punct(atoms);
    let ending = &atoms[request_atoms.len()..];
    if (ending.is_empty()
        || matches!(ending, [punct] if !punct.is_locked && matches!(punct.surface.as_str(), "。" | "？" | "?")))
        && te_help_request(request_atoms, doc).is_some()
    {
        let punctuation = (!ending.is_empty()).then(|| "？".to_owned());
        flush_clause(&mut request_atoms.to_vec(), punctuation, &mut segments);
        return segments;
    }

    while i < atoms.len() {
        let a = &atoms[i];
        if a.upos == Some(Upos::Num)
            && atoms
                .get(i + 1)
                .is_some_and(|day| !day.is_locked && day.surface == "日")
            && calendar::contrasts_recurrence(&current)
        {
            flush_clause(&mut current, None, &mut segments);
        }
        if !a.is_locked && matches!(a.surface.as_str(), "「" | "『") {
            flush_clause(&mut current, None, &mut segments);
            let open_q = a.surface.chars().next().unwrap_or('「');
            let close_target = if open_q == '『' { "』" } else { "」" };
            let q_start = a.span.start_char;
            let mut inner = Vec::new();
            let mut j = i + 1;
            let mut q_end = a.span.end_char;
            let mut found_close = false;
            while j < atoms.len() {
                if !atoms[j].is_locked && atoms[j].surface == close_target {
                    q_end = atoms[j].span.end_char;
                    j += 1;
                    found_close = true;
                    break;
                }
                inner.push(atoms[j].clone());
                q_end = atoms[j].span.end_char;
                j += 1;
            }
            if found_close {
                segments.push(Segment::Quote {
                    open_q,
                    inner_atoms: inner,
                    close_q: close_target.chars().next().unwrap_or('」'),
                    span: Span {
                        start_char: q_start,
                        end_char: q_end,
                    },
                });
            } else {
                segments.push(Segment::OpenQuote {
                    open_q,
                    inner_atoms: inner,
                    span: Span {
                        start_char: q_start,
                        end_char: q_end,
                    },
                });
            }
            i = j;
            continue;
        }

        if !a.is_locked && matches!(a.surface.as_str(), "」" | "』") {
            flush_clause(&mut current, None, &mut segments);
            let close_q = a.surface.chars().next().unwrap_or('」');
            segments.push(Segment::CloseQuote(close_q, a.span));
            i += 1;
            continue;
        }

        // Clause-ending comma `、` / `，` or sentence-ending punctuation `。` / `！` / `？`
        if !a.is_locked
            && matches!(
                a.surface.as_str(),
                "、" | "，" | "。" | "！" | "？" | "!" | "?" | "…" | "……"
            )
        {
            let punctuation = if direct_ka_before_period(&current, a) {
                "？".to_owned()
            } else {
                a.surface.clone()
            };
            flush_clause(&mut current, Some(punctuation), &mut segments);
            i += 1;
            continue;
        }

        current.push(a.clone());

        // SCONJ clause boundary: `から`, `ので`, `けれど`, `けれども`, `けど`, `のに`, `たら`, `なら`, `ば`, `ながら`, `ても`, `ものの`
        // Guard obligation `ねばならない` / `なければならない`, proportional `ば...ほど`, and evaluative `てもいい/よかった`.
        let is_obligation_nara = (a.surface == "なら"
            && atoms.get(i + 1).is_some_and(|nx| {
                !nx.is_locked
                    && matches!(
                        nx.surface.as_str(),
                        "ない" | "なかっ" | "なかった" | "ぬ" | "ず"
                    )
            }))
            || (a.surface == "ば"
                && atoms
                    .get(i + 1)
                    .is_some_and(|nx| !nx.is_locked && nx.surface == "なら"));
        let is_proportional_ba = a.surface == "ば"
            && atoms[i + 1..]
                .iter()
                .take(4)
                .any(|nx| !nx.is_locked && matches!(nx.surface.as_str(), "ほど" | "なるほど"));
        let is_temo_eval = matches!(a.surface.as_str(), "ても" | "でも")
            && atoms.get(i + 1).is_some_and(|nx| {
                !nx.is_locked
                    && matches!(
                        nx.surface.as_str(),
                        "いい"
                            | "よい"
                            | "よかっ"
                            | "よかった"
                            | "良かっ"
                            | "良かった"
                            | "構わ"
                            | "大丈夫"
                    )
            });
        let is_kagon_sconj = matches!(a.surface.as_str(), "ても" | "も")
            && atoms[i + 1..]
                .iter()
                .any(|later| !later.is_locked && later.surface == "過言");
        // Keep source-plus-copula endings on the boundary path. Grouping `からです`
        // into one bunsetsu would classify the source phrase as a predicate and lose `從`.
        let next_is_copula = atoms.get(i + 1).is_some_and(|next| {
            !next.is_locked
                && matches!(
                    next.lemma.as_deref().unwrap_or(&next.surface),
                    "だ" | "です"
                )
        });
        let is_source_kara = a.surface == "から"
            && a.upos == Some(Upos::Adp)
            && i > 0
            && !next_is_copula
            && !has_temporal_te_kara_tail(&current)
            && !matches!(atoms[i - 1].upos, Some(Upos::Verb | Upos::Adj | Upos::Aux));
        if !a.is_locked
            && !is_obligation_nara
            && !is_proportional_ba
            && !is_temo_eval
            && !is_kagon_sconj
            && !is_source_kara
            && (is_clause_split_sconj(&a.surface)
                || (a.surface == "し" && a.upos == Some(Upos::Sconj))
                || (a.surface == "だけど" && current.len() >= 2))
            && i + 1 < atoms.len()
            && !is_punct_atom(&atoms[i + 1])
        {
            flush_clause(&mut current, Some("，".to_owned()), &mut segments);
            i += 1;
            continue;
        }

        // Temporal `ときには` clause boundary after a predicate (`気づいたときには`).
        if !a.is_locked
            && a.surface == "ときには"
            && i > 0
            && atoms[..i].iter().any(|prev| {
                !prev.is_locked && matches!(prev.upos, Some(Upos::Verb | Upos::Adj | Upos::Aux))
            })
            && i + 1 < atoms.len()
            && !is_punct_atom(&atoms[i + 1])
        {
            flush_clause(&mut current, Some("，".to_owned()), &mut segments);
            i += 1;
            continue;
        }

        // Concessive `["くせ", "に"]` (excluding `くせになる`) and hypothetical `["として", "も"]`.
        if !a.is_locked
            && a.surface == "くせ"
            && atoms
                .get(i + 1)
                .is_some_and(|nx| !nx.is_locked && nx.surface == "に")
            && !atoms.get(i + 2).is_some_and(|nx2| {
                !nx2.is_locked && matches!(nx2.surface.as_str(), "なる" | "なり" | "なっ")
            })
        {
            current.push(atoms[i + 1].clone());
            if i + 2 < atoms.len() && !is_punct_atom(&atoms[i + 2]) {
                flush_clause(&mut current, Some("，".to_owned()), &mut segments);
            }
            i += 2;
            continue;
        }
        if !a.is_locked
            && a.surface == "として"
            && atoms
                .get(i + 1)
                .is_some_and(|nx| !nx.is_locked && nx.surface == "も")
        {
            current.push(atoms[i + 1].clone());
            if i + 2 < atoms.len() && !is_punct_atom(&atoms[i + 2]) {
                flush_clause(&mut current, Some("，".to_owned()), &mut segments);
            }
            i += 2;
            continue;
        }

        // Conditional / habitual non-past `Verb + と` boundary, guarding quotative/cognitive/light verbs and `とは`/`とも`.
        if !a.is_locked
            && a.surface == "と"
            && i > 0
            && !atoms[i - 1].is_locked
            && atoms[i - 1].upos == Some(Upos::Verb)
            && !matches!(
                atoms[i - 1].surface.as_str(),
                "た" | "だ" | "ました" | "だった" | "でした"
            )
            && let Some(nx) = atoms.get(i + 1)
            && !is_punct_atom(nx)
        {
            let nx_blocks_to = !nx.is_locked
                && (matches!(
                    nx.surface.as_str(),
                    "すぐ"
                        | "すぐに"
                        | "は"
                        | "も"
                        | "する"
                        | "し"
                        | "なる"
                        | "なっ"
                        | "いう"
                        | "幸い"
                ) || nx.surface.starts_with("言")
                    || nx.surface.starts_with("思")
                    || nx.surface.starts_with("考")
                    || nx.surface.starts_with("感")
                    || nx.surface.starts_with("信")
                    || nx.surface.starts_with("聞")
                    || nx.surface.starts_with("書")
                    || matches!(
                        nx.lemma.as_deref(),
                        Some(
                            "言う"
                                | "思う"
                                | "考える"
                                | "感じる"
                                | "信じる"
                                | "聞く"
                                | "書く"
                                | "する"
                                | "なる"
                        )
                    ));
            if !nx_blocks_to {
                flush_clause(&mut current, Some("，".to_owned()), &mut segments);
                i += 1;
                continue;
            }
        }

        // Concessive `[て] + [も]` (e.g. `忙しくても`): keep `[て, も]` together, and don't split before `いい`/`よかった`.
        if !a.is_locked
            && a.upos == Some(Upos::Sconj)
            && a.surface == "て"
            && atoms
                .get(i + 1)
                .is_some_and(|nx| !nx.is_locked && nx.surface == "も")
        {
            let before_kagon = i > 0
                && !atoms[i - 1].is_locked
                && atoms[i - 1].surface == "言っ"
                && atoms[i + 2..]
                    .iter()
                    .any(|later| !later.is_locked && later.surface == "過言");
            let before_eval_adj = atoms.get(i + 2).is_some_and(|nx2| {
                !nx2.is_locked
                    && matches!(
                        nx2.surface.as_str(),
                        "いい"
                            | "よい"
                            | "よかっ"
                            | "よかった"
                            | "良かっ"
                            | "良かった"
                            | "構わ"
                            | "大丈夫"
                    )
            });
            if before_eval_adj {
                current.push(atoms[i + 1].clone());
                i += 2;
                continue;
            }
            if !before_kagon {
                current.push(atoms[i + 1].clone());
                if i + 2 < atoms.len() && !is_punct_atom(&atoms[i + 2]) {
                    flush_clause(&mut current, Some("，".to_owned()), &mut segments);
                }
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }

        // Conjunctive `て`/`で` clause boundary (preserving `ないで`, `なくては`/`なくても`, `によって`, `て初めて`, and `て + subsidiary verb`).
        let prev_is_neg = i > 0
            && !atoms[i - 1].is_locked
            && (matches!(atoms[i - 1].surface.as_str(), "ない" | "なかっ")
                || (atoms[i - 1].surface == "なく"
                    && atoms.get(i + 1).is_some_and(|nx| {
                        (!nx.is_locked && matches!(nx.surface.as_str(), "は" | "も"))
                            || is_te_subsidiary_atom(nx)
                    })));
        let prev_is_yotte = i > 0 && !atoms[i - 1].is_locked && atoms[i - 1].surface == "よっ";
        if !a.is_locked
            && !prev_is_neg
            && !prev_is_yotte
            && a.upos == Some(Upos::Sconj)
            && matches!(a.surface.as_str(), "て" | "で")
            && let Some(next_atom) = atoms.get(i + 1)
            && !is_punct_atom(next_atom)
            && next_atom.surface != "初めて"
            && !is_te_subsidiary_atom(next_atom)
            && return_trip_at(atoms, i, doc).is_none()
            && atoms[i + 1..].iter().any(|later| {
                !later.is_locked
                    && (matches!(later.upos, Some(Upos::Verb | Upos::Adj))
                        || matches!(later.surface.as_str(), "だ" | "です" | "だった" | "でした"))
            })
        {
            flush_clause(&mut current, Some("，".to_owned()), &mut segments);
        }
        i += 1;
    }

    flush_clause(&mut current, None, &mut segments);
    segments
}

fn is_kagon_clause(atoms: &[Atom]) -> bool {
    let s: String = atoms
        .iter()
        .filter(|a| !is_punct_atom(a))
        .map(|a| a.surface.as_str())
        .collect();
    matches!(
        s.as_str(),
        "と言っても過言ではない"
            | "といっても過言ではない"
            | "と言っても過言じゃない"
            | "といっても過言じゃない"
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PostQuoteKind {
    Other,
    QuoteKuseni,
    QuotePlainSay,
}

fn classify_post_quote_clause(atoms: &[Atom]) -> PostQuoteKind {
    let core: Vec<&str> = atoms
        .iter()
        .filter(|a| !is_punct_atom(a))
        .map(|a| a.surface.as_str())
        .collect();
    if core.len() < 2 || core[0] != "と" {
        return PostQuoteKind::Other;
    }
    let rest = &core[1..];
    if !rest[0].starts_with('言') {
        return PostQuoteKind::Other;
    }
    if rest.windows(2).any(|w| w == ["くせ", "に"])
        && rest[1..].iter().all(|s| {
            matches!(
                *s,
                "て" | "い" | "いる" | "いた" | "た" | "くせ" | "に" | "くせに"
            )
        })
    {
        return PostQuoteKind::QuoteKuseni;
    }
    if rest.contains(&"くせに")
        && rest[1..]
            .iter()
            .all(|s| matches!(*s, "て" | "い" | "いる" | "いた" | "た" | "くせに"))
    {
        return PostQuoteKind::QuoteKuseni;
    }
    if rest[1..].iter().all(|s| {
        matches!(
            *s,
            "て" | "い" | "いる" | "いた" | "た" | "ます" | "まし" | "ました"
        )
    }) {
        return PostQuoteKind::QuotePlainSay;
    }
    PostQuoteKind::Other
}

fn is_quote_mark(atom: &Atom) -> bool {
    !atom.is_locked && matches!(atom.surface.as_str(), "「" | "」" | "『" | "』")
}

// In a later sentence the final quoted clause and its closing mark are
// separate segments. Require the corresponding open mark in this paragraph.
fn ends_continued_quote(
    next: Option<&Segment>,
    doc: &Document,
    sentence_idx: usize,
    chars: &[char],
) -> bool {
    let Some(Segment::CloseQuote(close, span)) = next else {
        return false;
    };
    let sent = &doc.segmentation.sentences[sentence_idx];
    let start = doc.segmentation.paragraphs[sent.paragraph].span.start_char;
    let mut closers = vec![*close];
    for i in (start..span.start_char).rev() {
        let ch = chars[i];
        if !matches!(ch, '「' | '」' | '『' | '』')
            || doc
                .segmentation
                .opaque
                .iter()
                .any(|opaque| opaque.span.start_char <= i && i < opaque.span.end_char)
        {
            continue;
        }
        if matches!(ch, '」' | '』') {
            closers.push(ch);
        } else {
            if !matches!((ch, closers.pop()), ('「', Some('」')) | ('『', Some('』'))) {
                return false;
            }
            if closers.is_empty() {
                return i < sent.span.start_char;
            }
        }
    }
    false
}

fn is_quote_terminal(ch: char) -> bool {
    matches!(ch, '。' | '？' | '！' | '?' | '!')
}

fn terminal_before_quote(chars: &[char], close_start: usize) -> bool {
    chars[..close_start]
        .iter()
        .rev()
        .find(|ch| !ch.is_whitespace())
        .copied()
        .is_some_and(is_quote_terminal)
}

fn emit_quote_close(
    out: &mut String,
    close: char,
    next: Option<&Segment>,
    moved_report: Option<Span>,
    previous_terminal: bool,
) {
    if let Some(Segment::Clause {
        span,
        trailing_punct: Some(punctuation),
        ..
    }) = next
        && Some(*span) == moved_report
        && punctuation == "。"
        && !previous_terminal
        && !out.chars().next_back().is_some_and(is_quote_terminal)
    {
        out.push('。');
    }
    out.push(close);
}

fn is_direct_quote_prefix_separator(punctuation: Option<&str>) -> bool {
    matches!(punctuation, None | Some("、" | "，"))
}

fn direct_quote_report_parts(atoms: &[Atom]) -> Option<(&[Atom], &Atom)> {
    let (marker, rest) = atoms.split_first()?;
    if marker.is_locked || marker.surface != "と" {
        return None;
    }
    let (args, verb) = match rest {
        [args @ .., verb, past]
            if !past.is_locked && past.surface == "た" && verb.surface == "伝え" =>
        {
            (args, verb)
        }
        [args @ .., verb] if verb.surface == "伝えた" => (args, verb),
        _ => return None,
    };
    if verb.is_locked || verb.lemma.as_deref() != Some("伝える") {
        return None;
    }
    if !args.is_empty() && (verb.doc_tok_idx.is_none() || marker.head != verb.doc_tok_idx) {
        return None;
    }
    Some((args, verb))
}

#[derive(Clone, Default)]
struct DirectQuoteRoles {
    speaker: Option<String>,
    listener: Option<String>,
    time: Option<String>,
}

impl DirectQuoteRoles {
    fn prefix(self) -> Option<String> {
        let speaker = self.speaker?;
        let listener = self.listener?;
        let time = self.time.unwrap_or_default();
        Some(format!("{speaker}{time}告訴{listener}："))
    }
}

fn direct_quote_time(atom: &Atom, doc: &Document) -> Option<String> {
    if atom.is_locked {
        return None;
    }
    if is_clock_atom(atom) {
        return Some(atom.surface.clone());
    }
    let class = atom
        .doc_tok_idx
        .and_then(|index| doc.tokens.get(index))
        .filter(|token| token.span == atom.span)
        .and_then(|token| token.sem_class);
    if class == Some(SemanticClass::Time)
        || matches!(
            atom.surface.as_str(),
            "朝" | "昼" | "夜" | "午前" | "午後" | "今日" | "明日" | "昨日"
        )
    {
        let text = translate_single_ja_term(&atom.surface, atom.lemma.as_deref(), atom.upos);
        return (!text.is_empty()).then_some(text);
    }
    None
}

fn read_direct_quote_roles(
    atoms: &[Atom],
    doc: &Document,
    locks: &SpanLockTable,
    report_verb: Option<&Atom>,
    roles: &mut DirectQuoteRoles,
) -> Option<()> {
    let mut args = atoms;
    while !args.is_empty() {
        if let Some(time) = direct_quote_time(&args[0], doc) {
            if roles.time.is_some()
                || report_verb.is_some_and(|verb| {
                    args[0].head != verb.doc_tok_idx || args[0].deprel != "advmod"
                })
            {
                return None;
            }
            roles.time = Some(time);
            args = &args[1..];
            continue;
        }
        let marker = args.iter().position(|atom| {
            !atom.is_locked && matches!(atom.surface.as_str(), "は" | "が" | "に")
        })?;
        let listener = args[marker].surface == "に";
        let person = &args[..marker];
        if let Some(verb) = report_verb
            && !person.iter().any(|atom| {
                atom.head == verb.doc_tok_idx
                    && if listener {
                        atom.deprel == "iobj" || atom.deprel.split(':').next() == Some("obl")
                    } else {
                        atom.deprel.split(':').next() == Some("nsubj")
                    }
            })
        {
            return None;
        }
        let role = if listener {
            &mut roles.listener
        } else {
            &mut roles.speaker
        };
        if role.is_some() {
            return None;
        }
        *role = Some(render_direct_quote_person(person, doc, locks)?);
        args = &args[marker + 1..];
    }
    Some(())
}

fn local_direct_quote(
    segments: &[Segment],
    doc: &Document,
    locks: &SpanLockTable,
) -> Option<(String, Span, usize)> {
    let quote_index = segments
        .iter()
        .take(3)
        .position(|segment| matches!(segment, Segment::Quote { .. }))?;
    let [
        Segment::Quote { inner_atoms, .. },
        Segment::Clause {
            atoms: report,
            span,
            ..
        },
        ..,
    ] = &segments[quote_index..]
    else {
        return None;
    };
    if inner_atoms.iter().any(is_quote_mark) {
        return None;
    }
    if let Some(prefix) = payment_potential::reported_question(
        &segments[..quote_index],
        inner_atoms,
        report,
        doc,
        locks,
    ) {
        return Some((prefix, *span, quote_index));
    }
    let (after, verb) = direct_quote_report_parts(report)?;
    let mut roles = DirectQuoteRoles::default();
    for before in &segments[..quote_index] {
        let Segment::Clause {
            atoms,
            trailing_punct,
            ..
        } = before
        else {
            return None;
        };
        if !is_direct_quote_prefix_separator(trailing_punct.as_deref()) {
            return None;
        }
        read_direct_quote_roles(atoms, doc, locks, None, &mut roles)?;
    }
    read_direct_quote_roles(after, doc, locks, Some(verb), &mut roles)?;
    Some((roles.prefix()?, *span, quote_index))
}

fn render_direct_quote_person(
    atoms: &[Atom],
    doc: &Document,
    locks: &SpanLockTable,
) -> Option<String> {
    let person_like = |atom: &Atom| {
        if atom.is_locked {
            return locks.locks.iter().any(|lock| {
                lock.source_span == atom.span
                    && matches!(
                        lock.origin,
                        LockOrigin::Glossary {
                            kind: GlossaryKind::Actor,
                            ..
                        }
                    )
            });
        }
        let class = atom
            .doc_tok_idx
            .and_then(|index| doc.tokens.get(index))
            .filter(|token| token.span == atom.span)
            .and_then(|token| token.sem_class);
        class.map_or_else(
            || is_person_like_atom(atom),
            |kind| kind == SemanticClass::Person,
        )
    };
    if !atoms.iter().any(person_like)
        || is_time_or_place_bunsetsu(atoms)
        || !atoms.iter().all(|atom| {
            atom.is_locked || matches!(atom.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
        })
    {
        return None;
    }
    let person = render_bunsetsu_np(
        &Bunsetsu {
            atoms: atoms.to_vec(),
            role: BunsetsuRole::Other,
            case_particle: None,
            sconj_tail: None,
            sfp_tail: None,
        },
        TargetVariant::ZhHant,
    );
    (!person.is_empty()).then_some(person)
}

fn complete_cross_sentence_direct_quote(
    opening: &DirectQuoteOpening,
    report: &[Atom],
    doc: &Document,
    locks: &SpanLockTable,
) -> Option<DirectQuotePrefix> {
    let (after, verb) = direct_quote_report_parts(report)?;
    if after.iter().any(|atom| atom.is_locked) {
        // A placeholder from the reporting sentence cannot move to the opening
        // sentence. The two sentences restore locks from different tables.
        return None;
    }
    let mut roles = opening.roles.clone();
    read_direct_quote_roles(after, doc, locks, Some(verb), &mut roles)?;
    Some(DirectQuotePrefix {
        span: opening.prefix_span,
        text: roles.prefix()?,
    })
}

pub(crate) fn direct_quote_opening(
    sentence_idx: usize,
    chars: &[char],
    doc: &Document,
    tokens: &[&Token],
    locks: &SpanLockTable,
) -> Option<DirectQuoteOpening> {
    let atoms = normalized_sentence_atoms(sentence_idx, chars, doc, tokens, locks);
    let segments =
        split_atoms_into_segments(&atoms, doc, doc.segmentation.sentences[sentence_idx].span);
    let (quote, before) = segments.split_last()?;
    let (inner_atoms, span, close) = match quote {
        Segment::OpenQuote {
            open_q,
            inner_atoms,
            span,
        } => (
            inner_atoms,
            span,
            Some(if *open_q == '「' { '」' } else { '』' }),
        ),
        Segment::Quote {
            inner_atoms, span, ..
        } if terminal_before_quote(chars, span.end_char - 1) => (inner_atoms, span, None),
        _ => return None,
    };
    if inner_atoms.iter().any(is_quote_mark) {
        return None;
    }
    let mut roles = DirectQuoteRoles::default();
    let prefix_span = match before {
        [] => *span,
        [
            ..,
            Segment::Clause {
                atoms,
                trailing_punct,
                span,
            },
        ] if is_direct_quote_prefix_separator(trailing_punct.as_deref()) => {
            read_direct_quote_roles(atoms, doc, locks, None, &mut roles)?;
            *span
        }
        _ => return None,
    };
    Some(DirectQuoteOpening {
        prefix_span,
        roles,
        close,
    })
}

pub(crate) fn direct_quote_end(
    sentence_idx: usize,
    chars: &[char],
    doc: &Document,
    tokens: &[&Token],
    locks: &SpanLockTable,
    opening: &DirectQuoteOpening,
) -> DirectQuoteEnd {
    let atoms = normalized_sentence_atoms(sentence_idx, chars, doc, tokens, locks);
    let sent_span = doc.segmentation.sentences[sentence_idx].span;
    let Some(close_quote) = opening.close else {
        if let Some(Segment::Clause { atoms, span, .. }) =
            split_atoms_into_segments(&atoms, doc, sent_span).first()
            && let Some(prefix) = complete_cross_sentence_direct_quote(opening, atoms, doc, locks)
        {
            return DirectQuoteEnd::Reports(*span, prefix);
        }
        return DirectQuoteEnd::Other;
    };
    let Some(delimiter) = atoms.iter().find(|atom| is_quote_mark(atom)) else {
        return DirectQuoteEnd::Continues;
    };
    if !delimiter.surface.starts_with(close_quote) {
        return DirectQuoteEnd::Other;
    }
    let segments = split_atoms_into_segments(&atoms, doc, sent_span);
    for segments in segments.windows(2) {
        if let [
            Segment::CloseQuote(_, close_span),
            Segment::Clause {
                atoms: report,
                span,
                ..
            },
        ] = segments
            && *close_span == delimiter.span
            && let Some(prefix) = complete_cross_sentence_direct_quote(opening, report, doc, locks)
        {
            return DirectQuoteEnd::Reports(*span, prefix);
        }
    }
    if matches!(segments.last(), Some(Segment::CloseQuote(_, span)) if *span == delimiter.span)
        && terminal_before_quote(chars, delimiter.span.start_char)
    {
        DirectQuoteEnd::Closed
    } else {
        DirectQuoteEnd::Other
    }
}

#[derive(Clone, Default)]
struct ClauseDiscourseCtx {
    contrasts_recurrence: bool,
    return_trip: Option<ReturnTrip>,
    inside_quote: bool,
    in_nagara_follow: bool,
    in_concessive_follow: bool,
    in_kuseni_follow: bool,
    in_toshitemo_follow: bool,
    in_noni_follow: bool,
    had_noni_in_sentence: bool,
    had_comm_event_in_sentence: bool,
    sentence_has_work_return: bool,
    mirative_opening: bool,
    in_proportional_follow: bool,
    sentence_topic_pronoun: Option<&'static str>,
    sonomono_antecedent: Option<String>,
}

fn atoms_has_any(atoms: &[Atom], targets: &[&str]) -> bool {
    atoms.iter().any(|a| {
        !a.is_locked
            && (targets.contains(&a.surface.as_str())
                || a.lemma.as_deref().is_some_and(|l| targets.contains(&l)))
    })
}

fn detect_clause_topic_pronoun(atoms: &[Atom]) -> Option<&'static str> {
    for w in atoms.windows(2) {
        if !w[0].is_locked && !w[1].is_locked && matches!(w[1].surface.as_str(), "が" | "は") {
            if w[0].surface == "彼" {
                return Some("他");
            }
            if w[0].surface == "彼女" {
                return Some("她");
            }
        }
    }
    None
}

fn extract_sonomono_antecedent(atoms: &[Atom]) -> Option<String> {
    for i in 1..atoms.len() {
        let cur = &atoms[i];
        let prev = &atoms[i - 1];
        if !prev.is_locked
            && matches!(prev.upos, Some(Upos::Noun | Upos::Propn))
            && !cur.is_locked
            && (cur.surface == "そのもの"
                || (cur.surface == "その"
                    && atoms
                        .get(i + 1)
                        .is_some_and(|nx| !nx.is_locked && nx.surface == "もの")))
        {
            return Some(translate_single_ja_term(
                &prev.surface,
                prev.lemma.as_deref(),
                prev.upos,
            ));
        }
    }
    None
}

fn clause_has_comm_event(atoms: &[Atom]) -> bool {
    atoms.iter().any(|a| {
        !a.is_locked
            && (matches!(
                a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                "電話" | "連絡" | "連絡する" | "話す" | "聞く"
            ) || matches!(
                a.surface.as_str(),
                "電話" | "連絡" | "連絡し" | "話" | "話し" | "聞い"
            ))
    })
}

fn is_cleft_topic_clause(atoms: &[Atom]) -> bool {
    let core: Vec<&str> = atoms
        .iter()
        .filter(|a| !is_punct_atom(a))
        .map(|a| a.surface.as_str())
        .collect();
    core.len() >= 4
        && core[core.len() - 3..] == ["な", "の", "は"]
        && atoms
            .iter()
            .find(|a| !is_punct_atom(a))
            .is_some_and(|a| !a.is_locked && a.upos == Some(Upos::Adj))
}

fn clause_contains_dewa_naku(atoms: &[Atom]) -> bool {
    let core: Vec<&str> = atoms
        .iter()
        .filter(|a| !is_punct_atom(a))
        .map(|a| a.surface.as_str())
        .collect();
    (core.len() >= 3 && core[core.len() - 3..] == ["で", "は", "なく"])
        || (core.len() >= 2 && core[core.len() - 2..] == ["じゃ", "なく"])
}

fn try_render_proportional_clause(atoms: &[Atom]) -> Option<String> {
    let core: Vec<&Atom> = atoms.iter().filter(|a| !is_punct_atom(a)).collect();
    if core.len() < 3 {
        return None;
    }
    let ba_idx = core
        .iter()
        .position(|a| !a.is_locked && a.surface == "ば")?;
    if ba_idx == 0 {
        return None;
    }
    let after_ba = &core[ba_idx + 1..];
    let valid_hodo = (after_ba.len() == 1
        && !after_ba[0].is_locked
        && after_ba[0].surface == "なるほど"
        && matches!(core[ba_idx - 1].surface.as_str(), "なれ" | "なる"))
        || (after_ba.len() >= 2
            && !after_ba.last()?.is_locked
            && after_ba.last()?.surface == "ほど");
    if !valid_hodo {
        return None;
    }

    let head_slice = &core[..ba_idx];
    let pred = head_slice.last()?;
    let pred_lemma = pred.lemma.as_deref().unwrap_or(pred.surface.as_str());
    let pre_pred = &head_slice[..head_slice.len() - 1];

    // Pattern `[Subj, "が"|"は", Noun, "すれ"] ば ... ほど` (`技術が進歩すればするほど`)
    // or `[Subj?, Adv/Adj, "に", "なれ"] ば ... ほど` (`便利になればなるほど`).
    let last_is_ni = pre_pred
        .last()
        .is_some_and(|a| !a.is_locked && a.surface == "に");
    if last_is_ni
        && matches!(pred_lemma, "なる" | "なる_adj")
        && pre_pred.len() >= 2
        && !pre_pred[pre_pred.len() - 2].is_locked
        && matches!(
            pre_pred[pre_pred.len() - 2].upos,
            Some(Upos::Adj | Upos::Noun)
        )
    {
        let adj_atom = &pre_pred[pre_pred.len() - 2];
        let adj_zh =
            translate_single_ja_term(&adj_atom.surface, adj_atom.lemma.as_deref(), adj_atom.upos);
        let rest = &pre_pred[..pre_pred.len() - 2];
        let mut subj_zh = String::new();
        for a in rest {
            if !a.is_locked && matches!(a.surface.as_str(), "が" | "は") {
                continue;
            }
            subj_zh.push_str(&translate_single_ja_term(
                &a.surface,
                a.lemma.as_deref(),
                a.upos,
            ));
        }
        return Some(format!("{subj_zh}越{adj_zh}"));
    }

    let (subj_slice, main_pred_zh) = if matches!(pred.surface.as_str(), "すれ" | "する")
        && pre_pred.last().is_some_and(|x| x.upos == Some(Upos::Noun))
    {
        let vn = pre_pred.last()?;
        let s = &pre_pred[..pre_pred.len() - 1];
        (
            s,
            translate_single_ja_term(&vn.surface, vn.lemma.as_deref(), Some(Upos::Verb)),
        )
    } else if pred.surface != "ば" {
        (
            pre_pred,
            translate_single_ja_term(&pred.surface, pred.lemma.as_deref(), Some(Upos::Verb)),
        )
    } else {
        return None;
    };

    let mut subj_zh = String::new();
    for (idx, a) in subj_slice.iter().enumerate() {
        if !a.is_locked && matches!(a.surface.as_str(), "が" | "は") {
            continue;
        }
        let next_ga = subj_slice
            .get(idx + 1)
            .is_some_and(|nx| !nx.is_locked && nx.surface == "が");
        if a.surface == "技術" && next_ga && subj_slice.len() == 2 {
            subj_zh.push_str("科技");
        } else {
            subj_zh.push_str(&translate_single_ja_term(
                &a.surface,
                a.lemma.as_deref(),
                a.upos,
            ));
        }
    }
    Some(format!("{subj_zh}越{main_pred_zh}"))
}

fn try_extract_passive_cognitive_gachi(atoms: &[Atom]) -> Option<&[Atom]> {
    let core_len = atoms
        .iter()
        .rposition(|a| !is_punct_atom(a))
        .map(|idx| idx + 1)?;
    let core = &atoms[..core_len];
    if core.len() < 7 {
        return None;
    }
    let surfs: Vec<&str> = core.iter().map(|a| a.surface.as_str()).collect();
    let tail = &surfs[surfs.len() - 5..];
    let has_cog_gachi = tail[0] == "と"
        && matches!(tail[1], "考え" | "思わ" | "見ら" | "言わ" | "みら")
        && matches!(tail[2], "られ" | "れ")
        && tail[3] == "がち"
        && matches!(tail[4], "だ" | "です");
    if has_cog_gachi {
        Some(&core[..core.len() - 5])
    } else {
        None
    }
}

fn translate_proportional_cognitive_inner(
    inner: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    target: TargetVariant,
) -> String {
    // Strip redundant `人間 の` when the matrix frame already introduced `人們往往認為`.
    let trimmed = if inner.len() >= 3
        && !inner[0].is_locked
        && inner[0].surface == "人間"
        && !inner[1].is_locked
        && inner[1].surface == "の"
    {
        &inner[2..]
    } else {
        inner
    };

    // Pattern `[Subj..., "が"|"は", Adv/Adj ("楽に"|"楽"+"に"), "なる"]` -> `{subj}就越{adj}`
    if trimmed.len() >= 4
        && !trimmed[trimmed.len() - 1].is_locked
        && matches!(trimmed[trimmed.len() - 1].surface.as_str(), "なる" | "なり")
    {
        let adv = &trimmed[trimmed.len() - 2];
        let mut pred_zh = translate_single_ja_term(&adv.surface, adv.lemma.as_deref(), adv.upos);
        let subj_part = &trimmed[..trimmed.len() - 2];
        let subj_end = if subj_part
            .last()
            .is_some_and(|a| !a.is_locked && matches!(a.surface.as_str(), "が" | "は" | "に"))
        {
            if subj_part.last().is_some_and(|a| a.surface == "に") {
                pred_zh = translate_single_ja_term(
                    &trimmed[trimmed.len() - 3].surface,
                    trimmed[trimmed.len() - 3].lemma.as_deref(),
                    trimmed[trimmed.len() - 3].upos,
                );
                subj_part.len() - 2
            } else {
                subj_part.len() - 1
            }
        } else {
            subj_part.len()
        };
        let mut subj_zh = String::new();
        for a in &subj_part[..subj_end] {
            if !a.is_locked && matches!(a.surface.as_str(), "が" | "は") {
                continue;
            }
            if !a.is_locked && a.surface == "の" {
                subj_zh.push('的');
            } else {
                subj_zh.push_str(&translate_single_ja_term(
                    &a.surface,
                    a.lemma.as_deref(),
                    a.upos,
                ));
            }
        }
        return format!("{subj_zh}就越{pred_zh}");
    }

    translate_clause_atoms(
        trimmed,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    )
    .0
}

fn is_punct_atom(a: &Atom) -> bool {
    !a.is_locked
        && (a.upos == Some(Upos::Punct)
            || matches!(
                a.surface.as_str(),
                "、" | "，"
                    | "。"
                    | "！"
                    | "？"
                    | "!"
                    | "?"
                    | "…"
                    | "……"
                    | "「"
                    | "」"
                    | "『"
                    | "』"
            ))
}

fn is_clause_split_sconj(s: &str) -> bool {
    matches!(
        s,
        "から"
            | "ので"
            | "けれど"
            | "けれども"
            | "けど"
            | "のに"
            | "たら"
            | "なら"
            | "ば"
            | "ながら"
            | "ても"
            | "ものの"
    )
}

fn localize_punct(p: &str) -> String {
    match p {
        "、" | "," => "，".to_owned(),
        "!" => "！".to_owned(),
        "?" => "？".to_owned(),
        other => other.to_owned(),
    }
}

fn nominal_receiving_request_position(
    segments: &[Segment],
    seg_idx: usize,
    doc: &Document,
    sentence_idx: usize,
) -> bool {
    let [prefix @ .., Segment::Clause { trailing_punct, .. }] = segments else {
        return false;
    };
    if seg_idx != prefix.len()
        || !matches!(trailing_punct.as_deref(), None | Some("。" | "？" | "?"))
    {
        return false;
    }
    // A preceding sentence in the same paragraph can supply a restriction
    // or a third-person receiver even when this question has no subject.
    if sentence_idx > 0
        && doc.segmentation.sentences[sentence_idx - 1].paragraph
            == doc.segmentation.sentences[sentence_idx].paragraph
    {
        return false;
    }
    match prefix {
        [] => true,
        [
            Segment::Clause {
                atoms,
                trailing_punct: Some(comma),
                ..
            },
        ] if matches!(comma.as_str(), "、" | "，" | ",") => match atoms.as_slice() {
            [time] => direct_quote_time(time, doc).is_some(),
            [date, no, period] => {
                !date.is_locked
                    && !no.is_locked
                    && !period.is_locked
                    && matches!(date.surface.as_str(), "今日" | "明日" | "昨日")
                    && no.surface == "の"
                    && matches!(
                        period.surface.as_str(),
                        "朝" | "午前" | "午後" | "昼" | "夜"
                    )
            }
            _ => false,
        },
        _ => false,
    }
}

#[derive(Clone, Copy)]
enum NominalReceivingAmount<'a> {
    Approximate(&'a str),
    Count { numeral: &'a str, unit: &'a str },
}

fn nominal_receiving_amount(
    atoms: &[Atom],
    predicate: usize,
) -> Option<NominalReceivingAmount<'_>> {
    let (numeral, unit) = match atoms {
        [atom]
            if atom.head == Some(predicate)
                && atom.deprel == "advmod"
                && atom.upos == Some(Upos::Adv) =>
        {
            if matches!(atom.surface.as_str(), "少し" | "ちょっと") {
                return lookup_curated_ja_word(&atom.surface)
                    .map(NominalReceivingAmount::Approximate);
            }
            // The analyzer may give an undivided counter such as 一杯.
            let (index, _) = atom.surface.char_indices().next_back()?;
            atom.surface.split_at(index)
        }
        [numeral, unit]
            if numeral.upos == Some(Upos::Num)
                && numeral.deprel == "nummod"
                && numeral.head == Some(unit.doc_tok_idx?)
                && numeral.span.end_char == unit.span.start_char
                && unit.upos == Some(Upos::Noun)
                && unit.deprel == "obl"
                && unit.head == Some(predicate) =>
        {
            (numeral.surface.as_str(), unit.surface.as_str())
        }
        _ => return None,
    };
    let explicit_number = !numeral.is_empty()
        && (numeral
            .chars()
            .all(|ch| matches!(ch, '0'..='9' | '０'..='９'))
            || numeral
                .chars()
                .all(|ch| "〇零一二三四五六七八九十百千万萬億".contains(ch)));
    explicit_number.then_some(NominalReceivingAmount::Count { numeral, unit })
}

fn render_nominal_receiving_count(
    object: &mut Bunsetsu,
    noun: &Atom,
    numeral: &str,
    unit: &str,
) -> Option<String> {
    let zh = translate_single_ja_term(&noun.surface, noun.lemma.as_deref(), noun.upos);
    let counter = counted_object::classifier(unit, &zh)?;
    let mut quantity = String::new();
    if object.atoms.iter().any(|atom| atom.upos == Some(Upos::Det)) {
        // The explicit counter takes the place of the nominal renderer's
        // default classifier after a demonstrative attached to the object.
        let [det, head, _] = object.atoms.as_slice() else {
            return None;
        };
        quantity.push(attached_demonstrative(det, head)?);
        object.atoms.remove(0);
    }
    counted_object::push_quantity(&mut quantity, numeral, counter);
    Some(quantity)
}

fn try_translate_nominal_receiving_request(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
) -> Option<String> {
    let [args @ .., verb, masu, neg, question] = atoms else {
        return None;
    };
    if atoms.iter().any(|atom| atom.is_locked)
        || ![verb, masu, neg, question]
            .windows(2)
            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
        || verb.upos != Some(Upos::Verb)
        || verb.surface != "いただけ"
        || verb.lemma.as_deref() != Some("いただける")
        || verb.deprel != "root"
        || masu.surface != "ませ"
        || neg.surface != "ん"
        || question.surface != "か"
        || [masu, neg].iter().any(|a| a.upos != Some(Upos::Aux))
        || question.upos != Some(Upos::Part)
    {
        return None;
    }
    let predicate = verb.doc_tok_idx?;
    if [masu, neg, question]
        .iter()
        .any(|a| a.head != Some(predicate))
    {
        return None;
    }

    let case_index = args
        .iter()
        .position(|a| a.surface == "を" && a.upos == Some(Upos::Adp) && a.deprel == "case")?;
    let (phrase, after) = args.split_at(case_index + 1);
    let [.., noun, particle] = phrase else {
        return None;
    };
    if !matches!(noun.upos, Some(Upos::Noun | Upos::Propn))
        || noun.head != Some(predicate)
        || noun.deprel != "obj"
        || particle.head != Some(noun.doc_tok_idx?)
    {
        return None;
    }

    let (phrase, amount) = if let [first, rest @ ..] = phrase
        && let Some(amount) = nominal_receiving_amount(std::slice::from_ref(first), predicate)
    {
        (rest, Some(amount))
    } else {
        (phrase, None)
    };
    let amount = match (after, amount) {
        ([], amount) => amount,
        (measures, None) => Some(nominal_receiving_amount(measures, predicate)?),
        _ => return None,
    };

    // Keep overt participants, restrictions and verbal complements on their
    // existing path. Render only a plain object and its nominal modifiers.
    let [modifiers @ .., _, _] = phrase else {
        return None;
    };
    if !modifiers.iter().all(|a| match a.upos {
        Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::Det | Upos::Num) => {
            matches!(a.deprel.as_str(), "compound" | "nmod" | "det" | "nummod")
        }
        Some(Upos::Adp) => a.surface == "の" && a.deprel == "case",
        Some(Upos::Intj) => a.surface == "お",
        _ => false,
    }) {
        return None;
    }
    let normalized = normalize_ja_atoms(&apply_phrase_matches(phrase));
    let mut object = classify_bunsetsu(normalized, doc, sentence_idx);
    if object.role != BunsetsuRole::Obj || object.case_particle.as_deref() != Some("を") {
        return None;
    }
    let amount_text = match amount {
        Some(NominalReceivingAmount::Count { numeral, unit }) => {
            render_nominal_receiving_count(&mut object, noun, numeral, unit)?
        }
        Some(NominalReceivingAmount::Approximate(text)) => text.to_owned(),
        None => String::new(),
    };
    let object = render_bunsetsu_np(&object, TargetVariant::ZhHant);
    if object.is_empty() || has_kana_no_hangul(&object) {
        return None;
    }
    if object == "時間" {
        if matches!(amount, Some(NominalReceivingAmount::Count { .. })) {
            return None;
        }
        let quantity = if amount_text.is_empty() {
            "的"
        } else {
            &amount_text
        };
        return Some(format!("可以占用您{quantity}{object}嗎"));
    }
    Some(format!("可以給我{amount_text}{object}嗎"))
}

/// Core clause translator:
/// 1. Matches multi-token idioms/collocations if the clause is a known idiom or contains one.
/// 2. Normalizes passive `さ+れ`, adj-`そう`, `そんなに`, and adverbial `-く` forms.
/// 3. Groups tokens into bunsetsus, merges adnominal modifiers (`の`, prenominal `Adj`, `acl`),
///    and reorders SOV -> SVO with Layer 6 pro-drop & passive patient promotion.
///
/// Returns `(rendered_zh, had_overt_subj_no_pred)`.
fn translate_clause_atoms(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    conditioning: Option<&SentenceConditioning>,
    allow_pro_drop_inject: bool,
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> (String, bool) {
    if atoms.is_empty() {
        return (String::new(), false);
    }

    if target == TargetVariant::ZhHant
        && let Some(text) = payment_potential::render(core_non_punct(atoms), target, false)
    {
        return (text, false);
    }

    if let Some(request) = te_help_request(core_non_punct(atoms), doc)
        && let Some(rendered) = render_te_help_request(request, doc, sentence_idx, ctx, target)
    {
        return (rendered, false);
    }

    // Fast path: bare `正直` clause opener -> `老實說`.
    let non_punct: Vec<&Atom> = atoms.iter().filter(|a| !is_punct_atom(a)).collect();
    if non_punct.len() == 1 && !non_punct[0].is_locked && non_punct[0].surface == "正直" {
        return ("老實說".to_owned(), false);
    }

    // Fast path: check if entire clause (excluding trailing SFP) matches a curated idiom.
    let raw_clause: String = atoms.iter().map(|a| a.surface.as_str()).collect();
    if let Some(idiom) = lookup_curated_phrase(&raw_clause) {
        return (idiom.to_owned(), false);
    }

    if let Some(res) = try_translate_keigo_or_mirative(atoms, ctx) {
        return res;
    }

    if let Some(res) = try_translate_polite_request_closing(atoms) {
        return res;
    }

    if let Some(res) = try_translate_tsumori_clause(atoms, doc, sentence_idx, target) {
        return res;
    }

    if let Some(res) = try_translate_betsu_ni_wake_janai(atoms, doc, sentence_idx, target) {
        return res;
    }

    if let Some(res) = try_translate_benefactive_reproach(atoms, doc, sentence_idx, target) {
        return res;
    }

    if let Some(res) = try_translate_focus_jikan_dake(atoms, doc, sentence_idx, target) {
        return res;
    }

    if let Some(res) = try_translate_toki_clause(atoms, doc, sentence_idx, target) {
        return res;
    }

    if let Some(res) = try_translate_spoken_quotative(atoms, doc, sentence_idx, ctx, target) {
        return res;
    }

    if let Some(res) = try_translate_epistemic_towa(atoms, doc, sentence_idx, ctx, target) {
        return res;
    }

    if let Some(res) = try_translate_dakara_to_itte_potential(atoms, doc, sentence_idx, target) {
        return res;
    }

    if let Some(res) = try_translate_koto_mo_dekinai(atoms, doc, sentence_idx, ctx, target) {
        return res;
    }

    if let Some(res) = try_translate_gnomic_monoda(atoms, doc, sentence_idx, target) {
        return res;
    }

    if let Some(res) = try_translate_litotes_koto_mo(atoms, doc, sentence_idx, ctx, target) {
        return res;
    }

    if let Some(res) = try_translate_purposive_tame(atoms, doc, sentence_idx, ctx, target) {
        return res;
    }

    if let Some(res) = try_translate_motomerareru_bamen(atoms, doc, sentence_idx, target) {
        return res;
    }

    if let Some(res) = try_translate_cleft_and_medial_n1(atoms, doc, sentence_idx, ctx, target) {
        return res;
    }

    if let Some(res) = try_translate_indirect_directive(
        atoms,
        doc,
        sentence_idx,
        conditioning,
        allow_pro_drop_inject,
        target,
    ) {
        return res;
    }

    if let Some(res) = try_translate_cognitive_matrix(atoms, doc, sentence_idx, ctx, target) {
        return res;
    }

    if let Some(res) = try_translate_to_ieba_or_uso(atoms, doc, sentence_idx, target) {
        return res;
    }

    if let Some(res) = try_translate_kagiranai(atoms, doc, sentence_idx, ctx, target) {
        return res;
    }

    let proposal = polite_volitional(atoms);
    // In a proposal question, か asks whether the suggestion is welcome.
    // Leave the original verb and arguments on the ordinary assembly path.
    let body = if proposal.is_some_and(|(_, question)| question) {
        &atoms[..atoms.len() - 1]
    } else {
        atoms
    };
    // Group the complement and its modifiers before attaching the conditional copula.
    let (body, condition_tail) = match nominal_conditional_tail(body) {
        Some((body, tail)) => (body, Some(tail)),
        None => (body, None),
    };

    // Multi-token phrase & collocation replacement over atoms first, then normalize remaining atoms.
    let trip = body
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a.surface.as_str(), "て" | "で"))
        .find_map(|(i, _)| return_trip_at(body, i, doc).map(|trip| (i, trip)));
    let original_body = body;
    let trip_source = trip.map(|(link, trip)| {
        body.iter()
            .enumerate()
            .filter(|(i, _)| *i != link - 1 && *i != link)
            .map(|(_, a)| {
                let mut a = a.clone();
                if a.head == Some(trip.manner) {
                    a.head = Some(trip.returning);
                }
                a
            })
            .collect::<Vec<_>>()
    });
    let body = trip_source.as_deref().unwrap_or(body);
    let counted = counted_object::bind(body);
    let merged_atoms = apply_phrase_matches(counted.as_deref().unwrap_or(body));
    let mut norm_atoms = normalize_building_floors(normalize_ja_atoms(&merged_atoms), doc);
    if target == TargetVariant::ZhHant {
        payment_potential::localize_money_verb(&mut norm_atoms, body, target);
    }
    calendar::normalize(&mut norm_atoms, ctx.contrasts_recurrence);
    recover_route_predicate(&mut norm_atoms, body);
    let motion_go = motion_purpose_governor(&norm_atoms, ctx.inside_quote);
    if let Some(go) = motion_go {
        recover_split_purpose_stems(&mut norm_atoms, go);
    }

    // Group into bunsetsus, then fold adnominal modifiers (`の`, prenominal `Adj`, `acl`).
    let mut grouped = group_into_bunsetsus(&norm_atoms, doc, sentence_idx);
    mark_return_arguments(&mut grouped, original_body, doc);
    for group in &mut grouped {
        if deadline_governor(group, doc).is_some_and(|head| has_detached_voice_negation(body, head))
        {
            group.role = BunsetsuRole::DeadlineWithDetachedNegation;
        }
    }
    fold_nominalized_deadline_ability(&mut grouped, body, doc, target);
    if let Some(go) = motion_go {
        mark_motion_purpose(&mut grouped, &norm_atoms, go, doc, sentence_idx);
    }
    mark_residence_place(&mut grouped, body);
    mark_joint_question(&mut grouped, &norm_atoms, body);
    let mut grouped = compose_relative_daypart(grouped);
    if let Some((predicate, _)) = proposal {
        localize_proposal_arguments(&mut grouped, predicate, body, target);
    }
    let mut bunsetsus = merge_adnominal_bunsetsus(grouped, body, doc, target);
    if target == TargetVariant::ZhHant
        && condition_tail.is_none()
        && let Some(homework) = try_translate_homework_transfer(&bunsetsus, body, doc, target)
    {
        return (homework, false);
    }
    if let Some(tail) = condition_tail
        && let Some(complement) = bunsetsus.last_mut()
    {
        complement.atoms.extend(tail);
        complement.role = BunsetsuRole::Predicate;
        complement.sconj_tail = Some("ば".to_owned());
    }

    let trip_ctx = trip.map(|(_, trip)| ClauseDiscourseCtx {
        return_trip: Some(trip),
        ..ctx.clone()
    });
    let (mut text, subject_only) = translate_clause_bunsetsus(
        bunsetsus,
        trip_source.as_deref().unwrap_or(atoms),
        &raw_clause,
        conditioning,
        allow_pro_drop_inject,
        trip_ctx.as_ref().unwrap_or(ctx),
        target,
    );
    if let Some((predicate, question)) = proposal {
        if !question {
            text.push('吧');
        } else if body.iter().enumerate().any(|(i, a)| {
            !a.is_locked
                && proposal_attachment(a, predicate, body)
                && matches!(
                    a.surface.as_str(),
                    "何" | "何時" | "誰" | "いつ" | "どこ" | "どれ" | "どう" | "どちら"
                )
                && !body
                    .get(i..i + 2)
                    .is_some_and(|pair| indefinite_argument(pair, predicate).is_some())
        }) {
            text.push_str("好呢");
        } else if matches!(body, [verb, _, _] if verb.lemma.as_deref() == Some("手伝う")) {
            text = format!("需要{text}嗎");
        } else {
            text.push_str("，好嗎");
        }
    }
    (text, subject_only)
}

fn polite_volitional(atoms: &[Atom]) -> Option<(usize, bool)> {
    let (body, question) = match atoms {
        [body @ .., ka]
            if !ka.is_locked
                && ka.surface == "か"
                && ka.upos == Some(Upos::Part)
                && ka.deprel == "discourse" =>
        {
            (body, Some(ka))
        }
        _ => (atoms, None),
    };
    let [.., verb, polite, volition] = body else {
        return None;
    };
    let pred = verb.doc_tok_idx?;
    if verb.is_locked
        || verb.upos != Some(Upos::Verb)
        || polite.is_locked
        || polite.upos != Some(Upos::Aux)
        || polite.surface != "ましょ"
        || polite.lemma.as_deref() != Some("ます")
        || polite.deprel != "aux"
        || volition.is_locked
        || volition.upos != Some(Upos::Aux)
        || volition.surface != "う"
        || volition.deprel != "aux"
        || polite.head != Some(pred)
        || volition.head != Some(pred)
        || verb.span.end_char != polite.span.start_char
        || polite.span.end_char != volition.span.start_char
        || question
            .is_some_and(|ka| ka.head != Some(pred) || volition.span.end_char != ka.span.start_char)
    {
        return None;
    }
    Some((pred, question.is_some()))
}

// The CPU parser can attach a manner phrase to an intervening adjectival
// modifier, or a question word to the noun it precedes. Neither can carry
// its own finite mood. Stop before crossing a second verbal predicate.
fn proposal_attachment(atom: &Atom, predicate: usize, source: &[Atom]) -> bool {
    atom.head == Some(predicate)
        || atom.head.is_some_and(|head| {
            source.iter().any(|parent| {
                parent.doc_tok_idx == Some(head)
                    && parent.head == Some(predicate)
                    && !parent.is_locked
                    && ((atom.deprel == "compound"
                        && matches!(parent.upos, Some(Upos::Noun | Upos::Propn)))
                        || (parent.deprel == "advcl" && parent.upos == Some(Upos::Adj)))
            })
        })
}

fn proposal_numeral(atom: &Atom) -> bool {
    !atom.is_locked
        && !atom.surface.is_empty()
        && atom.surface.chars().all(|ch| matches!(ch,
            '0'..='9' | '０'..='９' | '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十' | '百' | '千' | '万' | '零' | '〇'))
}

fn localize_proposal_arguments(
    groups: &mut Vec<Bunsetsu>,
    predicate: usize,
    source: &[Atom],
    target: TargetVariant,
) {
    let Some(action) = source.iter().find(|a| a.doc_tok_idx == Some(predicate)) else {
        return;
    };
    localize_indefinite_arguments(groups, action, source);
    let companion = place_companion(source, predicate, None);
    let speaker_offers = source.iter().any(|a| {
        !a.is_locked
            && a.head == Some(predicate)
            && a.deprel == "nsubj"
            && matches!(
                a.surface.as_str(),
                "私" | "わたし" | "僕" | "ぼく" | "俺" | "おれ"
            )
    });
    // Keep modifiers separate before the general noun/coordination fold.
    let mut previous: Vec<Bunsetsu> = Vec::with_capacity(groups.len());
    for mut group in std::mem::take(groups) {
        if companion.is_some()
            && matches!(group.atoms.as_slice(), [.., noun, case] if noun.doc_tok_idx == companion
                && !case.is_locked && case.surface == "と" && case.head == companion)
        {
            group.role = BunsetsuRole::ProposalCompanion;
        }
        if let [prefix @ .., noun, case] = group.atoms.as_slice()
            && !noun.is_locked
            && !case.is_locked
            && noun.deprel == "obl"
            && proposal_attachment(noun, predicate, source)
            && noun.doc_tok_idx.is_some()
            && case.head == noun.doc_tok_idx
            && case.deprel == "case"
            && noun.span.end_char == case.span.start_char
        {
            let mut first = group.atoms[0].span.start_char;
            let text = match (noun.surface.as_str(), case.surface.as_str()) {
                ("みんな" | "皆", "で") if prefix.is_empty() => Some("大家一起".to_owned()),
                ("一緒", "に") if prefix.is_empty() => {
                    if previous.last().is_some_and(|g| {
                        matches!(g.atoms.as_slice(), [.., partner, to]
                            if !to.is_locked && to.surface == "と" && to.deprel == "case"
                                && to.head == partner.doc_tok_idx && to.head.is_some()
                                && partner.head == Some(predicate)
                                && to.span.end_char == first)
                    }) {
                        let partner = previous.pop().unwrap();
                        first = partner.atoms[0].span.start_char;
                        Some(format!("和{}一起", render_bunsetsu_np(&partner, target)))
                    } else {
                        Some("一起".to_owned())
                    }
                }
                ("何時", "に") if prefix.is_empty() => Some("幾點".to_owned()),
                ("日", "に")
                    if matches!(prefix, [weekday]
                        if !weekday.is_locked && weekday.surface == "曜" && weekday.head == noun.doc_tok_idx
                            && weekday.span.end_char == noun.span.start_char)
                        && previous.last().is_some_and(|g| {
                            matches!(g.atoms.as_slice(), [wh]
                            if !wh.is_locked && wh.surface == "何" && wh.deprel == "compound"
                                && wh.head == noun.doc_tok_idx && wh.span.end_char == first)
                        }) =>
                {
                    first = previous.pop().unwrap().atoms[0].span.start_char;
                    Some("星期幾".to_owned())
                }
                ("人", "で")
                    if !prefix.is_empty()
                        && prefix.iter().all(proposal_numeral)
                        && group
                            .atoms
                            .windows(2)
                            .all(|pair| pair[0].span.end_char == pair[1].span.start_char) =>
                {
                    let mut count: String = prefix.iter().map(|a| a.surface.as_str()).collect();
                    while previous.last().is_some_and(|g| {
                        g.atoms.iter().all(proposal_numeral)
                            && g.atoms.last().is_some_and(|a| a.span.end_char == first)
                    }) {
                        let prior = previous.pop().unwrap();
                        first = prior.atoms[0].span.start_char;
                        let digits: String =
                            prior.atoms.iter().map(|a| a.surface.as_str()).collect();
                        count.insert_str(0, &digits);
                    }
                    match count.as_str() {
                        "一" | "1" | "１" => Some("一個人".to_owned()),
                        "二" | "2" | "２" => Some("兩個人一起".to_owned()),
                        _ => Some(format!("{count}個人一起")),
                    }
                }
                _ => None,
            };
            if let Some(text) = text {
                let mut adverb = noun.clone();
                adverb.surface = text;
                adverb.span.start_char = first;
                adverb.span.end_char = case.span.end_char;
                adverb.is_locked = true;
                group.atoms = vec![adverb];
                group.role = BunsetsuRole::ProposalModifier;
                group.case_particle = None;
            }
        }
        if speaker_offers
            && group.role == BunsetsuRole::Predicate
            && group.atoms.iter().any(|a| {
                !a.is_locked
                    && a.doc_tok_idx == Some(predicate)
                    && a.lemma.as_deref() != Some("来る")
            })
        {
            prepend_locked_atom(&mut group, "來".to_owned());
        }
        previous.push(group);
    }
    *groups = previous;
}

// Only match at a clause boundary. In obligation and proportional patterns,
// the existing splitter keeps the continuation after ば in the same clause.
fn nominal_conditional_tail(atoms: &[Atom]) -> Option<(&[Atom], [Atom; 2])> {
    let [body @ .., de, inflection, ba] = atoms else {
        return None;
    };
    let nominal = body.last()?;
    if [nominal, de, inflection, ba].iter().any(|a| a.is_locked)
        || !matches!(
            nominal.upos,
            Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::Num)
        )
        || de.surface != "で"
        || !matches!(de.upos, Some(Upos::Adp | Upos::Aux))
        || ba.surface != "ば"
        || ba.upos != Some(Upos::Sconj)
        || ![nominal, de, inflection, ba]
            .windows(2)
            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
    {
        return None;
    }

    // ADP で is ambiguous. Only a recognized person or membership noun
    // licenses the qualification reading. Leave other ADP complements to
    // the existing oblique path, even when the instrument is unknown.
    if de.upos == Some(Upos::Adp)
        && !(nominal.upos == Some(Upos::Noun)
            && (matches!(nominal.surface.as_str(), "委員" | "会員")
                || is_demonstrative_person_noun(nominal)))
    {
        return None;
    }

    let copula = match (inflection.surface.as_str(), inflection.upos) {
        ("あれ", Some(Upos::Verb | Upos::Aux)) => "だ",
        ("なけれ", Some(Upos::Adj | Upos::Aux)) => "ではない",
        _ => return None,
    };
    let mut predicate = inflection.clone();
    predicate.span.start_char = de.span.start_char;
    predicate.surface = copula.to_owned();
    predicate.lemma = Some(copula.to_owned());
    predicate.upos = Some(Upos::Aux);
    Some((body, [predicate, ba.clone()]))
}

enum ClassroomRole {
    Student,
    Teacher,
}

fn classroom_role(modifiers: &[Atom], atom: &Atom) -> Option<(ClassroomRole, &'static str)> {
    if atom.upos != Some(Upos::Noun) {
        return None;
    }
    let name = atom.lemma.as_deref().unwrap_or(&atom.surface);
    match (modifiers, name) {
        ([], "生徒" | "学生") => Some((ClassroomRole::Student, "學生")),
        ([], "先生" | "教師") => Some((ClassroomRole::Teacher, "老師")),
        ([], "大学生") => Some((ClassroomRole::Student, "大學生")),
        ([school], "生")
            if school.upos == Some(Upos::Noun)
                && school.surface == "大学"
                && school.deprel == "compound"
                && school.head == atom.doc_tok_idx
                && school.span.end_char == atom.span.start_char =>
        {
            Some((ClassroomRole::Student, "大學生"))
        }
        _ => None,
    }
}

fn dasu_case_link(noun: &Atom, particle: &Atom) -> bool {
    !particle.is_locked
        && particle.upos == Some(Upos::Adp)
        && particle.deprel == "case"
        && noun.doc_tok_idx.is_some()
        && particle.head == noun.doc_tok_idx
        && noun.span.end_char == particle.span.start_char
}

fn try_translate_homework_transfer(
    bunsetsus: &[Bunsetsu],
    source: &[Atom],
    doc: &Document,
    target: TargetVariant,
) -> Option<String> {
    // Inspect the source before nominal groups render protected atoms.
    // A normalized clock can pass if it has no glossary placeholder.
    if source.iter().any(|atom| {
        atom.is_locked
            && (atom.lemma.as_deref() != Some(CLOCK_ATOM) || atom.surface.contains("[[PH"))
    }) {
        return None;
    }
    let mut predicates = bunsetsus
        .iter()
        .filter(|bunsetsu| bunsetsu.role == BunsetsuRole::Predicate);
    let predicate = predicates.next()?;
    let (verb, endings) = predicate.atoms.split_first()?;
    let head = Some(verb.doc_tok_idx?);
    let terminal_ka = source.last().filter(|ka| {
        !ka.is_locked
            && ka.surface == "か"
            && ka.upos == Some(Upos::Part)
            && ka.deprel == "discourse"
            && ka.head == head
    });
    let is_terminal_ka = |atom: &Atom| {
        terminal_ka.is_some_and(|ka| ka.doc_tok_idx.is_some() && atom.doc_tok_idx == ka.doc_tok_idx)
    };
    if predicates.next().is_some()
        || verb.upos != Some(Upos::Verb)
        || verb.lemma.as_deref().unwrap_or(&verb.surface) != "出す"
        || !endings.iter().all(|atom| {
            matches!(
                atom.surface.as_str(),
                "ます"
                    | "まし"
                    | "ました"
                    | "た"
                    | "ない"
                    | "なかっ"
                    | "なかった"
                    | "ません"
                    | "ませ"
                    | "ん"
                    | "でし"
                    | "でした"
            ) || is_terminal_ka(atom)
        })
    {
        return None;
    }
    let mut deadline = None;
    let mut continuation_end = None;
    let mut has_clock_modifier = false;
    let mut subject = None;
    let mut recipient = None;
    let mut homework = None;
    let mut adjuncts = Vec::new();
    for bunsetsu in bunsetsus {
        if bunsetsu.sconj_tail.is_some()
            || (bunsetsu.sfp_tail.is_some() && !bunsetsu.atoms.last().is_some_and(is_terminal_ka))
        {
            return None;
        }
        if bunsetsu.case_particle.as_deref() == Some("までに") {
            if deadline.is_some() || deadline_governor(bunsetsu, doc) != head {
                return None;
            }
            deadline = Some((adjuncts.len(), bunsetsu));
            continue;
        }
        if let Some((clock, marker)) = explicit_clock_case(bunsetsu) {
            if clock_case_governor(clock, marker, doc) != head {
                return None;
            }
            has_clock_modifier = true;
            match marker.surface.as_str() {
                "まで" => {
                    if continuation_end.is_some() {
                        return None;
                    }
                    continuation_end = Some((adjuncts.len(), temporal_made_time(bunsetsu)?));
                }
                "から" => {
                    adjuncts.push(format!(
                        "{}起",
                        render_bunsetsu_obl(bunsetsu, target, false)
                    ));
                }
                _ => adjuncts.push(render_bunsetsu_obl(bunsetsu, target, false)),
            }
            continue;
        }
        if bunsetsu.role != BunsetsuRole::Obj && bunsetsu.atoms.iter().any(|atom| atom.is_locked) {
            return None;
        }
        if bunsetsu.role == BunsetsuRole::Predicate
            || bunsetsu.role == BunsetsuRole::Punct
            || matches!(bunsetsu.atoms.as_slice(), [ka] if is_terminal_ka(ka))
        {
            continue;
        }
        if is_event_daypart(bunsetsu, bunsetsus, source)
            || (matches!(bunsetsu.role, BunsetsuRole::Adv | BunsetsuRole::Other)
                && bunsetsu.case_particle.is_none()
                && bunsetsu
                    .atoms
                    .iter()
                    .any(|atom| atom.head == head && atom.deprel == "advmod"))
        {
            adjuncts.push(render_bunsetsu_np(bunsetsu, target));
            continue;
        }
        let [modifiers @ .., noun, marker] = bunsetsu.atoms.as_slice() else {
            return None;
        };
        if noun.head != head || !dasu_case_link(noun, marker) {
            return None;
        }
        match (noun.deprel.as_str(), marker.surface.as_str()) {
            ("nsubj", "は" | "が") if subject.is_none() => {
                subject = Some(classroom_role(modifiers, noun)?);
            }
            ("obl" | "iobj", "に") if recipient.is_none() => {
                recipient = Some(classroom_role(modifiers, noun)?);
            }
            ("obl", "で")
                if noun.surface == "教室"
                    || is_building_floor(noun)
                    || is_existing_spatial_head(noun)
                    || faberline_nlp::lexicon::lookup_ja(&noun.surface)
                        .is_some_and(|hit| hit.sem_class == Some(SemanticClass::Location)) =>
            {
                adjuncts.push(render_bunsetsu_obl(bunsetsu, target, false));
            }
            ("obj", "を")
                if noun.upos == Some(Upos::Noun)
                    && noun.surface == "宿題"
                    && homework.is_none() =>
            {
                homework = Some(render_bunsetsu_np(bunsetsu, target));
            }
            _ => return None,
        }
    }
    let ((subject_role, subject), (recipient_role, recipient)) = (subject?, recipient?);
    let homework = homework?;
    let action = match (subject_role, recipient_role) {
        (ClassroomRole::Student, ClassroomRole::Teacher) => format!("把{homework}交給{recipient}"),
        (ClassroomRole::Teacher, ClassroomRole::Student) => format!("出{homework}給{recipient}"),
        _ => return None,
    };
    let (before_verb, verb, after_verb, passive) =
        synthesize_predicate_complex(std::slice::from_ref(predicate), true, target, None);
    if verb != "拿出" || passive {
        return None;
    }
    let mut continuation_tail = None;
    if let Some((position, group)) = deadline {
        if continuation_end.is_some() {
            return None;
        }
        let time = render_bunsetsu_np(group, target);
        // Negation applies to completing the transfer by the deadline.
        let cutoff = match before_verb.as_slice() {
            [] => format!("最晚在{time}"),
            [word] if word == "沒有" => format!("沒有在{time}前"),
            [word] if word == "不" && terminal_ka.is_none() => format!("不會在{time}前"),
            // A negative nonpast question can invite the listener to act.
            _ => return None,
        };
        adjuncts.insert(position, cutoff);
    } else {
        if let Some((position, time)) = continuation_end {
            if before_verb
                .iter()
                .any(|word| matches!(word.as_str(), "不" | "沒有"))
            {
                adjuncts.insert(position, format!("到{time}為止"));
            } else {
                continuation_tail = Some(format!("到{time}"));
            }
        }
        adjuncts.extend(before_verb);
    }
    let mut parts = vec![subject.to_owned()];
    parts.extend(adjuncts);
    parts.push(action);
    parts.extend(
        after_verb
            .into_iter()
            .filter(|word| continuation_tail.is_none() || word != "了"),
    );
    parts.extend(continuation_tail);
    if terminal_ka.is_some() {
        if deadline.is_none() && !has_clock_modifier {
            return None;
        }
        let asks_deadline_time = deadline.is_some_and(|(_, group)| {
            group
                .atoms
                .iter()
                .any(|atom| deadline_time_question(atom).is_some())
        });
        if !asks_deadline_time {
            parts.push("嗎".to_owned());
        }
    }
    Some(join_zh_parts(&parts))
}

fn core_non_punct(atoms: &[Atom]) -> &[Atom] {
    let end = atoms
        .iter()
        .rposition(|a| !is_punct_atom(a))
        .map_or(0, |idx| idx + 1);
    &atoms[..end]
}

fn try_translate_keigo_or_mirative(
    atoms: &[Atom],
    ctx: &ClauseDiscourseCtx,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.is_empty() {
        return None;
    }
    if core.len() == 1
        && !core[0].is_locked
        && matches!(core[0].surface.as_str(), "へえ" | "へぇ" | "ほう")
    {
        return Some(("喔".to_owned(), false));
    }
    // `恐れ入り(ます)(が)` -> `不好意思`
    if !core[0].is_locked
        && matches!(
            core[0].surface.as_str(),
            "恐れ入り" | "恐れ入ります" | "恐縮"
        )
        && core[1..].iter().all(|a| {
            !a.is_locked
                && matches!(
                    a.surface.as_str(),
                    "ます" | "です" | "が" | "けれど" | "けど"
                )
        })
    {
        return Some(("不好意思".to_owned(), false));
    }
    // Mirative `へえ、そんなに [Adj] んだ。` -> `原來你這麼{Adj}啊`
    if ctx.mirative_opening
        && core.len() >= 3
        && !core[0].is_locked
        && core[0].surface == "そんな"
        && !core[1].is_locked
        && core[1].surface == "に"
    {
        let mut idx = 2;
        if let Some(adj) = core.get(idx)
            && !adj.is_locked
            && matches!(adj.upos, Some(Upos::Adj | Upos::Noun))
        {
            idx += 1;
            let tail_ok = core[idx..].iter().all(|a| {
                !a.is_locked && matches!(a.surface.as_str(), "ん" | "の" | "だ" | "です" | "な")
            });
            if tail_ok {
                let mut adj_zh =
                    translate_single_ja_term(&adj.surface, adj.lemma.as_deref(), adj.upos);
                if adj_zh == "忙碌" {
                    adj_zh = "忙".to_owned();
                }
                return Some((format!("原來你這麼{adj_zh}啊"), false));
            }
        }
    }
    None
}

fn try_translate_polite_request_closing(atoms: &[Atom]) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 4 {
        return None;
    }
    // Match `... いただけ(ます)(と)? 幸い (です)?` or `... くださ(れば/ると) 幸い (です)?`
    let saiwai_idx = core
        .iter()
        .rposition(|a| !a.is_locked && matches!(a.surface.as_str(), "幸い" | "幸いです"))?;
    if saiwai_idx == 0 {
        return None;
    }
    if core[saiwai_idx + 1..]
        .iter()
        .any(|a| !matches!(a.surface.as_str(), "です" | "に" | "存じ" | "ます"))
    {
        return None;
    }
    let ita_idx = core[..saiwai_idx].iter().rposition(|a| {
        !a.is_locked && (a.surface.starts_with("いただけ") || a.surface.starts_with("くださ"))
    })?;
    if !core[ita_idx + 1..saiwai_idx].iter().all(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "ます" | "れば" | "ると" | "る" | "と" | "たら" | "ましたら"
            )
    }) {
        return None;
    }

    let req_part = &core[..ita_idx];
    let mut parts = Vec::new();
    for a in req_part {
        if !a.is_locked && matches!(a.surface.as_str(), "ご" | "お" | "て" | "で") {
            continue;
        }
        parts.push(translate_single_ja_term(
            &a.surface,
            a.lemma.as_deref(),
            a.upos,
        ));
    }
    let body = parts.concat();
    if body.is_empty() {
        return None;
    }
    Some((format!("若能請您{body}，將不勝感激"), false))
}

fn render_te_help_request(
    request: TeHelpRequest<'_>,
    doc: &Document,
    sentence_idx: usize,
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> Option<String> {
    let mut repeat = false;
    let action_head = request.action.last().and_then(|atom| atom.doc_tok_idx);
    let participants = request
        .manner
        .as_ref()
        .and_then(|manner| manner.count)
        .map(render_joint_people);
    let performer_count = participants.as_deref().filter(|_| {
        request
            .person
            .as_ref()
            .is_some_and(|person| person.role == TeHelpPersonRole::Performer)
    });
    let mut action: Vec<_> = request
        .action
        .iter()
        .enumerate()
        .filter_map(|(index, atom)| {
            if request
                .person
                .as_ref()
                .is_some_and(|person| (person.start..person.end).contains(&index))
                || request
                    .letter_addressee
                    .as_ref()
                    .is_some_and(|person| (person.start..person.end).contains(&index))
                || request
                    .letter_writers
                    .as_ref()
                    .is_some_and(|(writers, _)| (writers.start..writers.end).contains(&index))
                || request.child_suffix == Some(index)
                || is_punct_atom(atom)
            {
                return None;
            }
            if let Some(manner) = &request.manner
                && (manner.start..manner.end).contains(&index)
            {
                return (index == manner.start && performer_count.is_none()).then(|| Atom {
                    span: Span {
                        start_char: atom.span.start_char,
                        end_char: request.action[manner.end - 1].span.end_char,
                    },
                    surface: "一起".to_owned(),
                    lemma: None,
                    upos: Some(Upos::Adv),
                    doc_tok_idx: None,
                    head: action_head,
                    deprel: "advmod".to_owned(),
                    is_locked: false,
                });
            }
            if !atom.is_locked && atom.surface == "もう一度" && atom.deprel == "advmod" {
                repeat = true;
                return None;
            }
            let mut localized = atom.clone();
            if request.child_suffix == Some(index + 1) {
                localized.surface = "子供".to_owned();
                localized.lemma = None;
                localized.span.end_char = request.action[index + 1].span.end_char;
            }
            Some(localized)
        })
        .collect();
    if let Some(index) = action.iter().position(|atom| {
        !atom.is_locked
            && atom.upos == Some(Upos::Noun)
            && atom.deprel == "obj"
            && atom.head == action_head
            && matches!(atom.surface.as_str(), "荷物" | "書類")
    }) {
        let object = &action[index];
        let (noun, classifier) = if object.surface == "荷物" {
            ("行李", "件")
        } else {
            ("文件", "份")
        };
        let mut localized = object.clone();
        let mut start = index;
        localized.surface = noun.to_owned();
        if let Some(det_index) = index.checked_sub(1)
            && let Some(demonstrative) = attached_demonstrative(&action[det_index], object)
        {
            start = det_index;
            localized.span.start_char = action[start].span.start_char;
            localized.surface = format!("{demonstrative}{classifier}{noun}");
        }
        localized.lemma = None;
        localized.is_locked = true;
        action.splice(start..=index, [localized]);
    }
    if let Some((verb, arguments)) = action.split_last_mut()
        && !verb.is_locked
        && verb.lemma.as_deref() == Some("描く")
        && let Some(action_idx) = verb.doc_tok_idx
        && arguments
            .iter()
            .any(|atom| atom.deprel == "obj" && atom.head == Some(action_idx))
    {
        // 畫畫 is intransitive in Mandarin. The explicit drawing object
        // licenses 畫 in this request.
        verb.surface = "畫".to_owned();
        verb.lemma = None;
    }
    let (body, _) = translate_clause_atoms(&action, doc, sentence_idx, None, false, ctx, target);
    let with_repeat = |text: &str| {
        if repeat {
            format!("再{text}一次")
        } else {
            text.to_owned()
        }
    };
    let render_name = |phrase: &[Atom]| {
        render_bunsetsu_np(
            &Bunsetsu {
                atoms: phrase.to_vec(),
                role: BunsetsuRole::Other,
                case_particle: None,
                sconj_tail: None,
                sfp_tail: None,
            },
            target,
        )
    };
    let render_person = |person: &TeHelpPerson<'_>| render_name(person.phrase);
    let polite = if request.negative_form {
        "可以麻煩"
    } else {
        "可以請"
    };
    let fallback_listener = if request.humble || request.negative_form {
        "您"
    } else {
        "你"
    };

    let addressee = if let Some(address) = &request.letter_addressee {
        let names = address
            .members
            .iter()
            .map(|member| render_name(member))
            .collect::<Vec<_>>();
        if address.illustrative {
            Some(format!("{}等人", names.join("、")))
        } else {
            let mut names = names.join("和");
            if let Some(count) = address.count {
                names.push_str(&render_joint_people(count));
            }
            Some(names)
        }
    } else {
        // Only the wording chooses an addressee for an unresolved dative.
        request
            .person
            .as_ref()
            .filter(|person| person.role == TeHelpPersonRole::UnresolvedLetterParticipant)
            .map(render_person)
    };
    if let Some(addressee) = addressee {
        let counted_listener = |count: &str| format!("{}位", count.trim_end_matches('人'));
        let mut joint = "";
        let listener = if let Some((writers, number)) = &request.letter_writers {
            format!(
                "{}{}",
                counted_listener(&render_joint_people(number)),
                render_person(writers)
            )
        } else if let Some(writer) = request
            .person
            .as_ref()
            .filter(|person| person.role == TeHelpPersonRole::Performer)
        {
            let name = render_person(writer);
            if let Some(count) = performer_count {
                joint = "一起";
                format!("包括{name}在內的{}", counted_listener(count))
            } else {
                name
            }
        } else if let Some(companion) = request.manner.as_ref().and_then(|bound| bound.companion) {
            format!(
                "{fallback_listener}和{}兩人",
                render_name(std::slice::from_ref(companion))
            )
        } else {
            participants
                .as_deref()
                .map(counted_listener)
                .unwrap_or_else(|| fallback_listener.to_owned())
        };
        return Some(format!(
            "{polite}{listener}{joint}{}嗎",
            with_repeat(&format!("{body}給{addressee}"))
        ));
    }

    if let Some(person) = request.person {
        let name = render_person(&person);
        if person.role == TeHelpPersonRole::Performer {
            if let Some(count) = performer_count {
                return Some(format!(
                    "{polite}包括{name}在內的{count}一起幫我{}嗎",
                    with_repeat(&body)
                ));
            }
            return Some(format!("{polite}{name}幫我{}嗎", with_repeat(&body)));
        }
        // Use the bare transfer verb so the linked recipient follows one 給.
        // If rendering did not keep the verb and object, keep the original clause.
        let verb = request.action.last()?;
        let transfer = translate_single_ja_term(&verb.surface, verb.lemma.as_deref(), verb.upos);
        let verb = purpose_transfer_action(&transfer)?;
        let (modifiers, object) = body
            .split_once(&transfer)
            .filter(|(_, object)| !object.is_empty())?;
        let listener = participants.as_deref().unwrap_or(fallback_listener);
        let action_body = format!("{modifiers}把{object}{verb}給{name}");
        return Some(format!("{polite}{listener}{}嗎", with_repeat(&action_body)));
    }
    let opening = match (
        participants.is_some(),
        request.humble,
        request.negative_form,
    ) {
        (true, _, false) => "可以請",
        (true, _, true) => "可以麻煩",
        (false, true, false) => "可以請您幫我",
        (false, true, true) => "可以麻煩您幫我",
        (false, false, false) => "可以幫我",
        (false, false, true) => "可以麻煩你幫我",
    };
    let participants = participants.as_deref().unwrap_or_default();
    Some(format!("{opening}{participants}{}嗎", with_repeat(&body)))
}

fn try_translate_tsumori_clause(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 3 {
        return None;
    }
    let tsu_idx = core
        .iter()
        .rposition(|a| !a.is_locked && a.surface == "つもり")?;
    if tsu_idx == 0 {
        return None;
    }
    let tail_ok = core[tsu_idx + 1..].iter().all(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "だ" | "です"
                    | "だった"
                    | "でした"
                    | "のに"
                    | "けど"
                    | "けれど"
                    | "けれども"
                    | "が"
            )
    });
    if !tail_ok {
        return None;
    }
    let inner = &core[..tsu_idx];
    let (mut inner_zh, _) = translate_clause_atoms(
        inner,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    );
    while inner_zh.ends_with('了') {
        inner_zh.pop();
    }
    if let Some(stripped) = inner_zh.strip_suffix("早點回去")
        && !inner
            .iter()
            .any(|a| !a.is_locked && matches!(a.surface.as_str(), "に" | "へ"))
    {
        inner_zh = format!("{stripped}早點回家");
    }
    if inner_zh.is_empty() {
        return None;
    }
    Some((format!("本來打算{inner_zh}"), false))
}

fn try_translate_betsu_ni_wake_janai(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 4 || core[0].is_locked || core[0].surface != "別に" {
        return None;
    }
    let wake_idx = core
        .iter()
        .rposition(|a| !a.is_locked && matches!(a.surface.as_str(), "わけ" | "わけではない"))?;
    if wake_idx <= 1 {
        return None;
    }
    let tail = &core[wake_idx..];
    let tail_ok = tail.iter().all(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "わけ"
                    | "わけではない"
                    | "じゃ"
                    | "で"
                    | "は"
                    | "ない"
                    | "あり"
                    | "ませ"
                    | "ん"
                    | "けど"
                    | "けれど"
                    | "けれども"
                    | "が"
            )
    }) && tail
        .iter()
        .any(|a| matches!(a.surface.as_str(), "ない" | "ん" | "わけではない"));
    if !tail_ok {
        return None;
    }
    let inner = &core[1..wake_idx];
    let (mut inner_zh, _) = translate_clause_atoms(
        inner,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    );
    if let Some(stripped) = inner_zh.strip_prefix("正在") {
        inner_zh = format!("在{stripped}");
    }
    let has_contrast = tail.last().is_some_and(|a| {
        !a.is_locked && matches!(a.surface.as_str(), "けど" | "けれど" | "けれども" | "が")
    });
    Some((
        if has_contrast {
            format!("我也不是{inner_zh}，只是")
        } else {
            format!("我也不是{inner_zh}")
        },
        false,
    ))
}

fn try_translate_benefactive_reproach(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 5 {
        return None;
    }
    let (has_semete, start) = if !core[0].is_locked && core[0].surface == "せめて" {
        (true, 1usize)
    } else {
        (false, 0usize)
    };
    let kure_idx = (start + 1..core.len()).rfind(|&idx| {
        !core[idx].is_locked
            && matches!(core[idx].surface.as_str(), "くれ" | "くれる")
            && idx > start
            && !core[idx - 1].is_locked
            && matches!(core[idx - 1].surface.as_str(), "て" | "で")
    })?;
    let tail_surfs: Vec<&str> = core[kure_idx + 1..]
        .iter()
        .map(|a| a.surface.as_str())
        .collect();
    let matches_reproach_tail = matches!(
        tail_surfs.as_slice(),
        ["て", "も", "よかっ", "た", "んじゃない"]
            | ["て", "も", "よかっ", "た", "んじゃないか"]
            | ["ても", "よかっ", "た", "んじゃない"]
            | ["ても", "よかった", "んじゃない"]
            | ["て", "も", "よかっ", "た", "ん", "じゃ", "ない"]
            | ["て", "も", "いい", "んじゃない"]
            | ["ても", "いい", "んじゃない"]
            | ["て", "も", "良かっ", "た", "んじゃない"]
    );
    if !matches_reproach_tail {
        return None;
    }
    let inner = &core[start..kure_idx - 1];
    if inner.is_empty() {
        return None;
    }

    let inner_surfs: Vec<&str> = inner.iter().map(|a| a.surface.as_str()).collect();
    let vp_zh = if inner_surfs.contains(&"一言")
        && inner_surfs.iter().any(|s| s.starts_with("相談"))
    {
        "先跟我商量一下".to_owned()
    } else {
        let mut rendered = translate_clause_atoms(
            inner,
            doc,
            sentence_idx,
            None,
            false,
            &ClauseDiscourseCtx::default(),
            target,
        )
        .0;
        let is_comm = inner.iter().any(|a| {
            matches!(
                a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                "相談" | "相談する" | "連絡" | "連絡する" | "報告" | "伝える" | "教える" | "見せる"
            ) || a.surface.starts_with("相談")
                || a.surface.starts_with("連絡")
        });
        if is_comm {
            let v_word = if let Some(pos) = inner
                .iter()
                .position(|a| a.surface.starts_with("相談") || a.surface.starts_with("連絡"))
            {
                let prefix_parts: String = inner[..pos]
                    .iter()
                    .map(|a| translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos))
                    .collect();
                let v_tr = translate_single_ja_term(
                    &inner[pos].surface,
                    inner[pos].lemma.as_deref(),
                    Some(Upos::Verb),
                );
                format!("{prefix_parts}跟我{v_tr}")
            } else {
                format!("跟我{rendered}")
            };
            rendered = v_word;
        } else {
            rendered = format!("幫我{rendered}");
        }
        rendered
    };

    Some((
        format!("你{}可以{vp_zh}吧", if has_semete { "至少" } else { "" }),
        false,
    ))
}

fn try_translate_focus_jikan_dake(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 6 {
        return None;
    }
    let jikan_idx = core.windows(4).position(|w| {
        !w[0].is_locked
            && w[0].surface == "時間"
            && !w[1].is_locked
            && w[1].surface == "だけ"
            && !w[2].is_locked
            && w[2].surface == "は"
            && !w[3].is_locked
            && w[3].surface == "ある"
    })?;
    let tail_atoms = &core[jikan_idx + 4..];
    if !tail_atoms.iter().all(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "ん" | "の" | "だ" | "です" | "ね" | "よね" | "よ"
            )
    }) {
        return None;
    }
    let sfp = tail_atoms.last().and_then(|a| match a.surface.as_str() {
        "ね" | "よね" => Some("呢"),
        "よ" => Some("喔"),
        _ => None,
    });

    let mut idx = 0usize;
    let mut conn = String::new();
    if !core[0].is_locked && matches!(core[0].surface.as_str(), "それなのに" | "でも" | "なのに")
    {
        conn = translate_single_ja_term(&core[0].surface, core[0].lemma.as_deref(), core[0].upos);
        idx = 1;
    }
    let mut time_str = String::new();
    while idx < jikan_idx {
        let a = &core[idx];
        if !a.is_locked
            && matches!(
                a.surface.as_str(),
                "毎日" | "今日" | "いつも" | "普段" | "最近" | "週末"
            )
        {
            time_str.push_str(&translate_single_ja_term(
                &a.surface,
                a.lemma.as_deref(),
                a.upos,
            ));
            idx += 1;
        } else {
            break;
        }
    }
    let vp_atoms = &core[idx..jikan_idx];
    let vp_zh = if vp_atoms.len() == 2
        && !vp_atoms[0].is_locked
        && vp_atoms[0].surface == "ゲーム"
        && !vp_atoms[1].is_locked
        && vp_atoms[1].surface == "する"
    {
        "玩遊戲".to_owned()
    } else {
        translate_clause_atoms(
            vp_atoms,
            doc,
            sentence_idx,
            None,
            false,
            &ClauseDiscourseCtx::default(),
            target,
        )
        .0
    };
    Some((
        format!("{conn}{time_str}倒是都有時間{vp_zh}{}", sfp.unwrap_or("")),
        false,
    ))
}

fn try_translate_toki_clause(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 2 {
        return None;
    }
    let last = core.last()?;
    if last.is_locked {
        return None;
    }
    let is_tokiniwa = last.surface == "ときには";
    let is_toki = last.surface == "とき" || last.surface == "時";
    if !is_tokiniwa && !is_toki {
        return None;
    }
    let inner = &core[..core.len() - 1];
    // Require inner to end with a predicate or past auxiliary (`た` / Verb / Adj), not `Noun + の`.
    let ends_pred = inner.last().is_some_and(|a| {
        !a.is_locked
            && (matches!(a.upos, Some(Upos::Verb | Upos::Adj | Upos::Aux))
                || matches!(a.surface.as_str(), "た" | "だ"))
    });
    if !ends_pred {
        return None;
    }

    if is_tokiniwa
        && inner.iter().all(|a| {
            !a.is_locked
                && (a.surface == "た"
                    || a.lemma.as_deref() == Some("気づく")
                    || a.surface.starts_with("気づ")
                    || a.surface.starts_with("気付"))
        })
    {
        return Some(("等到察覺時".to_owned(), false));
    }

    let (mut inner_zh, _) = translate_clause_atoms(
        inner,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    );
    while inner_zh.ends_with('了') {
        inner_zh.pop();
    }
    if is_tokiniwa {
        Some((format!("等到{inner_zh}時"), false))
    } else {
        Some((format!("{inner_zh}時"), false))
    }
}

fn try_translate_spoken_quotative(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 5 {
        return None;
    }
    let q_idx = (1..core.len() - 1).rfind(|&idx| {
        !core[idx].is_locked
            && core[idx].surface == "って"
            && !core[idx + 1].is_locked
            && core[idx + 1].surface.starts_with("言")
    })?;
    let tail = &core[q_idx + 2..];
    let has_kara = tail
        .last()
        .is_some_and(|a| !a.is_locked && matches!(a.surface.as_str(), "から" | "ので"));
    let verb_tail = if has_kara {
        &tail[..tail.len() - 1]
    } else {
        tail
    };
    if !verb_tail.iter().all(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "て" | "い" | "いる" | "いた" | "た" | "ます" | "まし" | "ました"
            )
    }) {
        return None;
    }
    let inner = &core[..q_idx];
    let (inner_zh, _) = translate_clause_atoms(
        inner,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    );
    if inner_zh.is_empty() {
        return None;
    }
    let speaker = if let Some(tp) = ctx.sentence_topic_pronoun {
        format!("{tp}說")
    } else if ctx.had_comm_event_in_sentence {
        "對方說".to_owned()
    } else {
        "他說".to_owned()
    };
    Some((
        format!(
            "{speaker}{inner_zh}{}",
            if has_kara { "，所以" } else { "" }
        ),
        false,
    ))
}

fn try_translate_epistemic_towa(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 6 {
        return None;
    }
    let towa_idx = (1..core.len() - 2).rfind(|&idx| {
        !core[idx].is_locked
            && core[idx].surface == "と"
            && !core[idx + 1].is_locked
            && core[idx + 1].surface == "は"
    })?;
    let tail = &core[towa_idx + 2..];
    let tail_surfs: Vec<&str> = tail.iter().map(|a| a.surface.as_str()).collect();
    enum EpistemicKind {
        HardToAssert(bool), // true = has trailing `が`/`けど`
        HardToBelieve,
    }
    let kind = match tail_surfs.as_slice() {
        ["言い切れ", "ない", "が"]
        | ["言い切れ", "ない", "けど"]
        | ["言い切れ", "ませ", "ん", "が"]
        | ["言え", "ない", "が"] => EpistemicKind::HardToAssert(true),
        ["言い切れ", "ない"] | ["言い切れ", "ませ", "ん"] | ["言え", "ない"] => {
            EpistemicKind::HardToAssert(false)
        }
        ["考え", "にくい", "ものの"]
        | ["考え", "にくい", "が"]
        | ["考え", "にくい", "けど"]
        | ["考え", "にくい"] => EpistemicKind::HardToBelieve,
        _ => return None,
    };

    let inner = &core[..towa_idx];
    // Idiom: `[Subj, "が"|"は", "何", "も", "知ら", "なかっ"|"ない", ...]` -> `{Subj}對此一無所知`
    let inner_zh = if inner.len() >= 6
        && !inner[inner.len() - 5].is_locked
        && inner[inner.len() - 5].surface == "何"
        && !inner[inner.len() - 4].is_locked
        && inner[inner.len() - 4].surface == "も"
        && !inner[inner.len() - 3].is_locked
        && inner[inner.len() - 3].surface == "知ら"
        && !inner[inner.len() - 2].is_locked
        && matches!(inner[inner.len() - 2].surface.as_str(), "なかっ" | "ない")
    {
        let subj_part = &inner[..inner.len() - 5];
        let mut subj_zh = String::new();
        for a in subj_part {
            if !a.is_locked && matches!(a.surface.as_str(), "が" | "は") {
                continue;
            }
            subj_zh.push_str(&translate_single_ja_term(
                &a.surface,
                a.lemma.as_deref(),
                a.upos,
            ));
        }
        if subj_zh.is_empty()
            && let Some(tp) = ctx.sentence_topic_pronoun
        {
            subj_zh = tp.to_owned();
        }
        format!("{subj_zh}對此一無所知")
    } else {
        let mut s = translate_clause_atoms(
            inner,
            doc,
            sentence_idx,
            None,
            false,
            &ClauseDiscourseCtx::default(),
            target,
        )
        .0;
        while s.ends_with('了') {
            s.pop();
        }
        s
    };

    if inner_zh.is_empty() {
        return None;
    }
    match kind {
        EpistemicKind::HardToAssert(has_ga) => Some((
            format!("不能斷言{inner_zh}{}", if has_ga { "，但" } else { "" }),
            false,
        )),
        EpistemicKind::HardToBelieve => Some((format!("雖然很難相信{inner_zh}"), false)),
    }
}

fn try_translate_dakara_to_itte_potential(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 4 || core[0].is_locked || core[0].surface != "だからといって" {
        return None;
    }
    // Check ends with `できる` + (`わけでもない` | `わけ` `で` `も` `ない` | `わけ` `でも` `ない`)
    let end_surfs: Vec<&str> = core[1..].iter().map(|a| a.surface.as_str()).collect();
    let suffix_len = if end_surfs.ends_with(&["できる", "わけ", "で", "も", "ない"]) {
        5usize
    } else if end_surfs.ends_with(&["できる", "わけ", "でも", "ない"]) {
        4usize
    } else if end_surfs.ends_with(&["できる", "わけでもない"]) {
        2usize
    } else {
        return None;
    };
    let mid = &core[1..core.len() - suffix_len];
    if mid.is_empty() {
        return None;
    }
    // Guard overt subject (`自分も` etc.): only trigger when subjectless or no nominative particle.
    let has_overt_subj = mid.iter().any(|a| {
        !a.is_locked
            && (matches!(a.upos, Some(Upos::Pron)) || matches!(a.surface.as_str(), "が" | "は"))
    });
    if has_overt_subj {
        return None;
    }
    let norm_mid = normalize_ja_atoms(mid);
    let mid_zh = translate_clause_atoms(
        &norm_mid,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    )
    .0;
    if mid_zh.is_empty() {
        return None;
    }
    Some((format!("但也不代表我能{mid_zh}"), false))
}

fn try_translate_koto_mo_dekinai(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 7 {
        return None;
    }
    let (has_dakara, start) = if !core[0].is_locked && core[0].surface == "だからといって" {
        (true, 1usize)
    } else {
        (false, 0usize)
    };
    let tail_surfs: Vec<&str> = core[start..].iter().map(|a| a.surface.as_str()).collect();
    let koto_tail_len = if tail_surfs.ends_with(&["こと", "も", "でき", "ない"])
        || tail_surfs.ends_with(&["こと", "も", "でき", "ません"])
        || tail_surfs.ends_with(&["こと", "は", "でき", "ない"])
    {
        4usize
    } else {
        return None;
    };
    let matrix_end = core.len() - koto_tail_len;
    // Find `と` before the matrix verb (`断定` / `断定する` / `判断する` / etc.)
    let to_idx =
        (start + 1..matrix_end).rfind(|&idx| !core[idx].is_locked && core[idx].surface == "と")?;
    let matrix_atoms = &core[to_idx + 1..matrix_end];
    if matrix_atoms.is_empty()
        || !matrix_atoms
            .iter()
            .any(|a| !a.is_locked && matches!(a.upos, Some(Upos::Verb | Upos::Noun)))
    {
        return None;
    }
    let mut matrix_zh = String::new();
    for a in matrix_atoms {
        if !a.is_locked && matches!(a.surface.as_str(), "する" | "し") && !matrix_zh.is_empty() {
            continue;
        }
        matrix_zh.push_str(&translate_single_ja_term(
            &a.surface,
            a.lemma.as_deref(),
            a.upos,
        ));
    }
    if matrix_zh.is_empty() {
        return None;
    }

    let inner = &core[start..to_idx];
    let norm_inner = normalize_ja_atoms(&apply_phrase_matches(inner));
    let mut inner_zh =
        translate_clause_atoms(&norm_inner, doc, sentence_idx, None, false, ctx, target).0;
    while inner_zh.ends_with('了') {
        inner_zh.pop();
    }
    inner_zh = inner_zh.replace("正在", "");
    let has_inner_subj = inner.iter().any(|a| {
        !a.is_locked
            && (matches!(a.upos, Some(Upos::Pron)) || matches!(a.surface.as_str(), "が" | "は"))
    });
    if !has_inner_subj
        && let Some(tp) = ctx.sentence_topic_pronoun
        && !inner_zh.starts_with(tp)
    {
        inner_zh = format!("{tp}{inner_zh}");
    }
    let prefix = if has_dakara {
        "但也不能因此"
    } else {
        "也不能"
    };
    Some((format!("{prefix}{matrix_zh}{inner_zh}"), false))
}

fn try_translate_gnomic_monoda(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 7 {
        return None;
    }
    let mono_idx = (3..core.len() - 1).rfind(|&idx| {
        !core[idx].is_locked
            && core[idx].surface == "もの"
            && !core[idx + 1].is_locked
            && matches!(core[idx + 1].surface.as_str(), "だ" | "です")
    })?;
    if !core[mono_idx + 2..].iter().all(|a| {
        !a.is_locked && matches!(a.surface.as_str(), "が" | "けど" | "けれど" | "けれども")
    }) {
        return None;
    }
    let has_contrast = core.len() > mono_idx + 2;
    let body = &core[..mono_idx];
    let first_idx = (1..body.len() - 1).find(|&idx| {
        !body[idx].is_locked
            && body[idx].surface == "初めて"
            && !body[idx - 1].is_locked
            && matches!(body[idx - 1].surface.as_str(), "て" | "で")
    })?;

    let prefix_slice = &body[..first_idx - 1];
    let rest_slice = &body[first_idx + 1..];
    if rest_slice.is_empty() {
        return None;
    }

    // Extract `[Subj, "は"|"が"]` from `prefix_slice` if present.
    let (subj_zh, v1_slice) = if let Some(wa_pos) = prefix_slice
        .iter()
        .position(|a| !a.is_locked && matches!(a.surface.as_str(), "は" | "が"))
        && wa_pos > 0
    {
        let s: String = prefix_slice[..wa_pos]
            .iter()
            .map(|a| translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos))
            .collect();
        (s, &prefix_slice[wa_pos + 1..])
    } else {
        (String::new(), prefix_slice)
    };
    if v1_slice.is_empty() {
        return None;
    }

    let v1_zh = translate_clause_atoms(
        v1_slice,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    )
    .0;
    let mut rest_zh = translate_clause_atoms(
        rest_slice,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    )
    .0;
    if rest_slice.iter().any(|a| {
        !a.is_locked
            && (a.lemma.as_deref() == Some("気づく")
                || a.surface.starts_with("気づ")
                || a.surface.starts_with("気付"))
    }) {
        rest_zh = rest_zh.replace("發現", "明白");
    }
    Some((
        format!(
            "{subj_zh}往往直到{v1_zh}之後才{rest_zh}{}",
            if has_contrast { "，但" } else { "" }
        ),
        false,
    ))
}

fn try_translate_litotes_koto_mo(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 5 {
        return None;
    }
    let tail_surfs: Vec<&str> = core[core.len() - 4..]
        .iter()
        .map(|a| a.surface.as_str())
        .collect();
    if !matches!(
        tail_surfs.as_slice(),
        ["こと", "も", "少なく", "ない"] | ["こと", "は", "少なく", "ない"]
    ) {
        return None;
    }
    let inner = &core[..core.len() - 4];
    if inner.is_empty() {
        return None;
    }
    let inner_zh = translate_clause_atoms(inner, doc, sentence_idx, None, false, ctx, target).0;
    if inner_zh.is_empty() {
        return None;
    }
    Some((format!("很多時候{inner_zh}"), false))
}

fn try_translate_purposive_tame(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 3 {
        return None;
    }
    let last = core.last()?;
    if last.is_locked || last.surface != "ため" {
        return None;
    }
    let (prefix_zh, body_start) = if !core[0].is_locked && core[0].surface == "実際には" {
        ("實際上，", 1usize)
    } else if core.len() >= 4
        && !core[0].is_locked
        && core[0].surface == "実際"
        && core[1].surface == "に"
        && core[2].surface == "は"
    {
        ("實際上，", 3usize)
    } else {
        ("", 0usize)
    };
    let body = &core[body_start..core.len() - 1];
    if body.is_empty()
        || !body
            .iter()
            .any(|a| !a.is_locked && a.upos == Some(Upos::Verb))
    {
        return None;
    }
    let vp_zh = translate_clause_atoms(body, doc, sentence_idx, None, false, ctx, target).0;
    if vp_zh.is_empty() {
        return None;
    }
    Some((format!("{prefix_zh}為了{vp_zh}"), false))
}

fn try_translate_motomerareru_bamen(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.len() < 9 {
        return None;
    }
    let s7: Vec<&str> = core[core.len() - 7..]
        .iter()
        .map(|a| a.surface.as_str())
        .collect();
    if !(s7[0].starts_with("求め")
        && matches!(s7[1], "られる" | "られ")
        && matches!(s7[2], "場面" | "状況" | "ケース" | "機会")
        && matches!(s7[3], "が" | "も")
        && s7[4] == "増え"
        && s7[5] == "て"
        && matches!(s7[6], "いる" | "います" | "いた"))
    {
        return None;
    }
    let obj_end = core.len() - 7;
    if obj_end == 0 || core[obj_end - 1].is_locked || core[obj_end - 1].surface != "を" {
        return None;
    }
    let obj_atoms = &core[..obj_end - 1];
    let mut obj_zh = translate_clause_atoms(
        obj_atoms,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    )
    .0;
    if obj_zh.contains("比") && obj_zh.ends_with("的判斷") {
        obj_zh = obj_zh.replacen("的判斷", "判斷", 1);
    }
    let need_verb =
        if obj_zh.ends_with("判斷") || obj_zh.ends_with("決定") || obj_zh.ends_with("選擇") {
            "需要做出"
        } else {
            "需要"
        };
    Some((format!("{need_verb}{obj_zh}的情況也越來越多"), false))
}

fn try_translate_cleft_and_medial_n1(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core = core_non_punct(atoms);
    if core.is_empty() {
        return None;
    }
    let surfs: Vec<&str> = core.iter().map(|a| a.surface.as_str()).collect();

    // 1. Cleft topic: `[Adj, "な", "の", "は"]` (`重要なのは` / `大切なのは` -> `重要的`)
    if core.len() == 4
        && !core[0].is_locked
        && core[0].upos == Some(Upos::Adj)
        && surfs[1..] == ["な", "の", "は"]
    {
        let mut adj_zh =
            translate_single_ja_term(&core[0].surface, core[0].lemma.as_deref(), core[0].upos);
        if core[0].surface == "大切" {
            adj_zh = "重要".to_owned();
        }
        return Some((format!("{adj_zh}的"), false));
    }

    // 2. Negated cleft: `[NP..., "で", "は", "なく"]` (`技術そのものの優劣ではなく` -> `不是技術本身孰優孰劣，而是`)
    if core.len() >= 4 && surfs[core.len() - 3..] == ["で", "は", "なく"] {
        let np_atoms = &core[..core.len() - 3];
        let np_zh = translate_clause_atoms(np_atoms, doc, sentence_idx, None, false, ctx, target).0;
        if !np_zh.is_empty() {
            return Some((format!("不是{np_zh}，而是"), false));
        }
    }

    // 3. Medial `それをどのように [Verb-stem]` (`それをどのように活用し` / `それをどのように使い`)
    if core.len() >= 4
        && !core[0].is_locked
        && matches!(core[0].surface.as_str(), "それ" | "これ")
        && !core[1].is_locked
        && core[1].surface == "を"
        && !core[2].is_locked
        && matches!(core[2].surface.as_str(), "どのように" | "どうやって")
    {
        let v_part = &core[3..];
        let is_renyoukei_end = v_part.last().is_some_and(|a| {
            !a.is_locked
                && (a.surface == "し"
                    || (a.upos == Some(Upos::Verb)
                        && !a.surface.ends_with('る')
                        && !a.surface.ends_with('う'))
                    || matches!(a.surface.as_str(), "使い"))
        });
        if is_renyoukei_end {
            let v_zh = if !v_part.is_empty() && v_part[0].surface == "使い" {
                "使用".to_owned()
            } else {
                translate_single_ja_term(
                    &v_part[0].surface,
                    v_part[0].lemma.as_deref(),
                    Some(Upos::Verb),
                )
            };
            let ante = ctx.sonomono_antecedent.as_deref().unwrap_or("它");
            return Some((format!("如何{v_zh}{ante}，以及"), false));
        }
    }

    // 4. Embedded indirect question matrix:
    // `どこまで [Subj] が/は [VP] べき か を 見極める/判断する こと だ`
    if core.len() >= 10
        && !core[0].is_locked
        && matches!(core[0].surface.as_str(), "どこまで" | "どれだけ")
    {
        let tail_start = if surfs.ends_with(&["べき", "か", "を", "見極める", "こと", "だ"])
            || surfs.ends_with(&["べき", "か", "を", "見極める", "こと", "です"])
            || surfs.ends_with(&["べき", "か", "を", "判断する", "こと", "だ"])
        {
            core.len() - 6
        } else {
            return None;
        };
        let mid = &core[1..tail_start];
        let ga_pos = mid
            .iter()
            .position(|a| !a.is_locked && matches!(a.surface.as_str(), "が" | "は"))?;
        let subj_zh: String = mid[..ga_pos]
            .iter()
            .map(|a| translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos))
            .collect();
        let vp_slice = &mid[ga_pos + 1..];
        let _ = (doc, sentence_idx, target);
        // Pattern `[Noun, "を", "持つ"|"負う"|"担う"]` (`責任を持つ`) -> `承擔多少{Noun}`
        if vp_slice.len() == 3
            && !vp_slice[1].is_locked
            && vp_slice[1].surface == "を"
            && !vp_slice[2].is_locked
            && matches!(vp_slice[2].surface.as_str(), "持つ" | "負う" | "担う")
        {
            let obj_zh = translate_single_ja_term(
                &vp_slice[0].surface,
                vp_slice[0].lemma.as_deref(),
                vp_slice[0].upos,
            );
            return Some((format!("如何判斷{subj_zh}應當承擔多少{obj_zh}"), false));
        }
        let vp_zh = translate_clause_atoms(
            vp_slice,
            doc,
            sentence_idx,
            None,
            false,
            &ClauseDiscourseCtx::default(),
            target,
        )
        .0;
        if !vp_zh.is_empty() {
            return Some((format!("如何判斷{subj_zh}應當{vp_zh}到什麼程度"), false));
        }
    }

    None
}

fn try_translate_to_ieba_or_uso(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core_len = atoms
        .iter()
        .rposition(|a| !is_punct_atom(a))
        .map(|idx| idx + 1)?;
    let core = &atoms[..core_len];

    // `[Inner] と 言え ば` / `[Inner] と 言っ たら` -> `要說{Inner}`
    if core.len() > 3 {
        let n = core.len();
        let tail = &core[n - 3..];
        if !tail.iter().any(|a| a.is_locked)
            && tail[0].surface == "と"
            && ((tail[1].surface == "言え" && tail[2].surface == "ば")
                || (tail[1].surface == "言っ" && tail[2].surface == "たら"))
        {
            let (inner_zh, _) = translate_clause_atoms(
                &core[..n - 3],
                doc,
                sentence_idx,
                None,
                false,
                &ClauseDiscourseCtx::default(),
                target,
            );
            if !inner_zh.is_empty() {
                return Some((format!("要說{inner_zh}"), false));
            }
        }
    }

    // `嘘になる` (+ `けど` / `けれど` / `けれども`) -> `那是騙人的` (+ `，但是`)
    let (body, contrast_tail) = if let Some(last) = core.last()
        && !last.is_locked
        && matches!(last.surface.as_str(), "けど" | "けれど" | "けれども")
    {
        (&core[..core.len() - 1], true)
    } else {
        (core, false)
    };
    if body.len() == 3
        && !body.iter().any(|a| a.is_locked)
        && matches!(body[0].surface.as_str(), "嘘" | "うそ")
        && body[1].surface == "に"
        && matches!(body[2].surface.as_str(), "なる" | "なり" | "なっ")
    {
        return Some((
            if contrast_tail {
                "那是騙人的，但是".to_owned()
            } else {
                "那是騙人的".to_owned()
            },
            false,
        ));
    }

    None
}

fn try_translate_kagiranai(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    _ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core_len = atoms
        .iter()
        .rposition(|a| !is_punct_atom(a))
        .map(|idx| idx + 1)?;
    let mut core = &atoms[..core_len];
    let has_shi = core
        .last()
        .is_some_and(|a| !a.is_locked && a.surface == "し");
    if has_shi {
        core = &core[..core.len() - 1];
    }

    let (suffix_len, suffix_has_kanarazushimo) = if core.len() >= 6 {
        let s = &core[core.len() - 6..];
        if !s.iter().any(|a| a.is_locked)
            && s[0].surface == "と"
            && s[1].surface == "は"
            && s[2].surface == "必ずしも"
            && matches!(s[3].surface.as_str(), "限り" | "かぎり")
            && s[4].surface == "ませ"
            && s[5].surface == "ん"
        {
            Some((6usize, true))
        } else {
            None
        }
    } else {
        None
    }
    .or_else(|| {
        if core.len() >= 5 {
            let s = &core[core.len() - 5..];
            if !s.iter().any(|a| a.is_locked)
                && s[0].surface == "と"
                && s[1].surface == "は"
                && s[2].surface == "必ずしも"
                && matches!(s[3].surface.as_str(), "限ら" | "かぎら")
                && s[4].surface == "ない"
            {
                return Some((5usize, true));
            }
            if !s.iter().any(|a| a.is_locked)
                && s[0].surface == "と"
                && s[1].surface == "は"
                && matches!(s[2].surface.as_str(), "限り" | "かぎり")
                && s[3].surface == "ませ"
                && s[4].surface == "ん"
            {
                return Some((5usize, false));
            }
        }
        None
    })
    .or_else(|| {
        if core.len() >= 4 {
            let s = &core[core.len() - 4..];
            if !s.iter().any(|a| a.is_locked)
                && s[0].surface == "と"
                && s[1].surface == "は"
                && matches!(s[2].surface.as_str(), "限ら" | "かぎら")
                && s[3].surface == "ない"
            {
                return Some((4usize, false));
            }
        }
        None
    })?;

    let mut inner = &core[..core.len() - suffix_len];
    if inner.is_empty() {
        return None;
    }

    let mut has_kesshite = false;
    if let Some(first) = inner.first()
        && !first.is_locked
    {
        if first.surface == "決して" {
            has_kesshite = true;
            inner = &inner[1..];
        } else if first.surface == "必ずしも" {
            inner = &inner[1..];
        }
    }
    if inner.is_empty() {
        return None;
    }

    let is_clausal_or_i_adj = inner.iter().any(|a| {
        (!a.is_locked && a.upos == Some(Upos::Verb))
            || (!a.is_locked
                && a.upos == Some(Upos::Adj)
                && (a.surface.ends_with('い') || a.surface.ends_with("かっ")))
    });

    let has_tsunagaru = inner.iter().any(|a| {
        matches!(
            a.lemma.as_deref().unwrap_or(a.surface.as_str()),
            "つながる" | "繋がる"
        ) || matches!(a.surface.as_str(), "つながる" | "繋がる")
    });
    let mut inner_vec = inner.to_vec();
    if suffix_has_kanarazushimo
        && inner_vec.len() >= 2
        && !inner_vec[0].is_locked
        && inner_vec[0].surface == "それ"
        && !inner_vec[1].is_locked
        && inner_vec[1].surface == "が"
    {
        inner_vec[0].surface = "它".to_owned();
        inner_vec[0].lemma = Some("它".to_owned());
        inner_vec[0].upos = Some(Upos::Pron);
        inner_vec[0].is_locked = true;
    }

    let (mut inner_zh, _) = translate_clause_atoms(
        &inner_vec,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    );
    if inner_zh.is_empty() {
        return None;
    }
    if suffix_has_kanarazushimo
        && has_tsunagaru
        && inner_zh.contains("帶來")
        && !inner_zh.contains("能帶來")
    {
        inner_zh = inner_zh.replacen("帶來", "能帶來", 1);
    }

    let prefix = if suffix_has_kanarazushimo {
        "也不一定代表"
    } else if is_clausal_or_i_adj {
        if has_kesshite {
            "絕不見得"
        } else {
            "未必"
        }
    } else if has_kesshite {
        "絕不見得是"
    } else {
        "未必是"
    };
    let mut rendered = format!("{prefix}{inner_zh}");
    if has_shi {
        rendered.push_str("，而且");
    }
    Some((rendered, false))
}

fn try_translate_cognitive_matrix(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core_len = atoms
        .iter()
        .rposition(|a| !is_punct_atom(a))
        .map(|idx| idx + 1)?;
    let core = &atoms[..core_len];
    if core.len() < 4 {
        return None;
    }

    let mut v_idx = core.len() - 1;
    while v_idx > 0 {
        let a = &core[v_idx];
        if !a.is_locked
            && matches!(
                a.surface.as_str(),
                "た" | "だ"
                    | "て"
                    | "い"
                    | "いる"
                    | "いた"
                    | "ます"
                    | "まし"
                    | "ました"
                    | "ません"
                    | "ませ"
                    | "ん"
                    | "ない"
                    | "なかった"
                    | "なかっ"
                    | "う"
                    | "よう"
                    | "よ"
                    | "ね"
                    | "よね"
                    | "か"
                    | "けど"
                    | "けれど"
                    | "けれども"
                    | "が"
                    | "し"
                    | "の"
            )
        {
            v_idx -= 1;
        } else {
            break;
        }
    }

    if v_idx < 2 {
        return None;
    }
    let v_atom = &core[v_idx];
    if v_atom.is_locked || core[v_idx - 1].is_locked || core[v_idx - 1].surface != "と" {
        return None;
    }

    let tail = &core[v_idx + 1..];
    if tail.iter().any(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "れる" | "られる" | "れ" | "られ" | "され" | "される" | "された"
            )
    }) {
        return None;
    }

    let v_sym = v_atom.lemma.as_deref().unwrap_or(v_atom.surface.as_str());
    let (pos_zh, neg_zh) = if v_sym == "思う" || v_atom.surface.starts_with("思") {
        ("覺得", "不覺得")
    } else if v_sym == "考える" || v_atom.surface.starts_with("考え") {
        ("認為", "不認為")
    } else if v_sym == "感じる" || v_atom.surface.starts_with("感じ") {
        ("感到", "不感到")
    } else if v_sym == "信じる" || v_atom.surface.starts_with("信じ") {
        ("相信", "不相信")
    } else {
        return None;
    };

    let mut pre_to = &core[..v_idx - 1];
    if pre_to.len() >= 3
        && !pre_to[pre_to.len() - 2].is_locked
        && !pre_to[pre_to.len() - 1].is_locked
        && matches!(pre_to[pre_to.len() - 2].surface.as_str(), "もの" | "こと")
        && matches!(pre_to[pre_to.len() - 1].surface.as_str(), "だ" | "です")
        && pre_to[..pre_to.len() - 2].iter().any(|a| {
            !a.is_locked
                && (matches!(a.upos, Some(Upos::Verb | Upos::Adj))
                    || matches!(a.surface.as_str(), "いる" | "ある" | "だ" | "です"))
        })
    {
        pre_to = &pre_to[..pre_to.len() - 2];
    }
    let has_tekkiri = pre_to
        .first()
        .is_some_and(|a| !a.is_locked && a.surface == "てっきり");
    if has_tekkiri {
        pre_to = &pre_to[1..];
    }
    if pre_to.is_empty() {
        return None;
    }

    let (subj_zh, inner_atoms) = if pre_to.len() >= 3
        && !pre_to[0].is_locked
        && matches!(
            pre_to[0].surface.as_str(),
            "私" | "わたし" | "あたし" | "僕" | "ぼく" | "俺" | "おれ"
        )
        && !pre_to[1].is_locked
        && pre_to[1].surface == "は"
        && pre_to[2..].iter().any(|a| {
            !a.is_locked
                && (matches!(a.upos, Some(Upos::Verb | Upos::Adj))
                    || matches!(a.surface.as_str(), "だ" | "です" | "だった" | "でした"))
        }) {
        ("我", &pre_to[2..])
    } else {
        ("我", pre_to)
    };

    let (mut inner_zh, _) = translate_clause_atoms(
        inner_atoms,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    );
    if inner_zh.is_empty() {
        return None;
    }
    let has_overt_subj = inner_atoms
        .iter()
        .any(|a| !a.is_locked && matches!(a.surface.as_str(), "が" | "は"));
    if has_tekkiri
        && ctx.had_comm_event_in_sentence
        && !has_overt_subj
        && inner_zh.starts_with('在')
    {
        inner_zh = format!("他人{inner_zh}");
    }

    let is_neg = tail.iter().any(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "ない" | "なかった" | "ません" | "ませ" | "ん"
            )
    });
    let verb_zh = if has_tekkiri && !is_neg && (v_sym == "思う" || v_atom.surface.starts_with("思"))
    {
        "一直以為"
    } else if is_neg {
        neg_zh
    } else {
        pos_zh
    };
    let mut rendered = format!("{subj_zh}{verb_zh}{inner_zh}");
    if tail.last().is_some_and(|a| {
        !a.is_locked && matches!(a.surface.as_str(), "けど" | "けれど" | "けれども")
    }) {
        rendered.push_str("，但是");
    } else if tail
        .last()
        .is_some_and(|a| !a.is_locked && a.surface == "し")
    {
        rendered.push_str("，而且");
    }
    Some((rendered, false))
}

fn try_translate_indirect_directive(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    conditioning: Option<&SentenceConditioning>,
    allow_pro_drop_inject: bool,
    target: TargetVariant,
) -> Option<(String, bool)> {
    let core_len = atoms
        .iter()
        .rposition(|a| !is_punct_atom(a))
        .map(|idx| idx + 1)?;
    let core = &atoms[..core_len];
    if core.len() < 4 {
        return None;
    }

    let you_idx = (1..core.len().saturating_sub(2)).rfind(|&i| {
        !core[i].is_locked
            && core[i].surface == "よう"
            && !core[i + 1].is_locked
            && core[i + 1].surface == "に"
            && !core[i - 1].is_locked
            && matches!(core[i - 1].upos, Some(Upos::Verb | Upos::Aux | Upos::Adj))
    })?;

    let pred_atoms = &core[you_idx + 2..];
    if pred_atoms.iter().any(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "れ" | "られ" | "れる" | "られる" | "され" | "される"
            )
    }) {
        return None;
    }
    let head_verb = pred_atoms.iter().find(|a| {
        !a.is_locked
            && (a.upos == Some(Upos::Verb)
                || matches!(
                    a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                    "言う" | "伝える" | "頼む" | "求める" | "命じる" | "指示する"
                ))
    })?;
    let v_lemma = head_verb
        .lemma
        .as_deref()
        .unwrap_or(head_verb.surface.as_str());
    let is_directive_verb = matches!(
        v_lemma,
        "言う" | "伝える" | "頼む" | "求める" | "命じる" | "指示する"
    ) || head_verb.surface.starts_with("言")
        || head_verb.surface.starts_with("伝え")
        || head_verb.surface.starts_with("頼")
        || head_verb.surface.starts_with("求め")
        || head_verb.surface.starts_with("命じ");
    if !is_directive_verb {
        return None;
    }

    let prefix_bunsetsus = group_into_bunsetsus(&core[..you_idx], doc, sentence_idx);
    let mut matrix_subjs: Vec<String> = Vec::new();
    let mut matrix_recips: Vec<String> = Vec::new();
    let mut consumed_atoms = 0usize;

    for b in &prefix_bunsetsus {
        let is_topic_subj = b.role == BunsetsuRole::Subj
            && b.case_particle
                .as_deref()
                .is_some_and(|cp| matches!(cp, "は" | "が"));
        let is_recip_ni = b.role == BunsetsuRole::IObj && b.case_particle.as_deref() == Some("に");
        if (is_topic_subj && matrix_subjs.is_empty() && matrix_recips.is_empty())
            || (is_recip_ni && matrix_recips.is_empty())
        {
            let np = render_bunsetsu_np(b, target);
            if !np.is_empty() {
                if is_topic_subj {
                    matrix_subjs.push(np);
                } else {
                    matrix_recips.push(np);
                }
            }
            consumed_atoms += b.atoms.len();
        } else {
            break;
        }
    }

    let sub_atoms = &core[consumed_atoms..you_idx];
    if sub_atoms.is_empty() {
        return None;
    }
    let (sub_zh, _) = translate_clause_atoms(
        sub_atoms,
        doc,
        sentence_idx,
        None,
        false,
        &ClauseDiscourseCtx::default(),
        target,
    );
    if sub_zh.is_empty() {
        return None;
    }

    let pred_is_neg = pred_atoms.iter().any(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "ない" | "なかった" | "ません" | "ませんでした" | "ず" | "ぬ"
            )
    });
    let pred_is_past = pred_atoms.iter().any(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "た" | "だ" | "ました" | "だった" | "でした" | "なかった" | "ませんでした"
            )
    });
    if matrix_subjs.is_empty()
        && allow_pro_drop_inject
        && let Some(cond) = conditioning
    {
        if let Some(hint) = cond.pronoun_hint.as_deref()
            && let Some(zh_pro) = zh_pronoun_for_hint(hint)
        {
            matrix_subjs.push(zh_pro.to_owned());
        } else if let Some(pro_drop) = cond.resolved_pro_drop.as_deref()
            && !pro_drop.is_empty()
        {
            matrix_subjs.push(translate_single_ja_term(pro_drop, None, Some(Upos::Propn)));
        }
    }

    let mut out = String::new();
    for s in &matrix_subjs {
        out.push_str(s);
    }
    if pred_is_neg {
        out.push_str(if pred_is_past { "沒" } else { "不" });
    }
    let recip = matrix_recips.join("");
    let verb_zh = if v_lemma == "言う" || head_verb.surface.starts_with("言") {
        if recip.is_empty() {
            "說要"
        } else if pred_is_neg {
            "叫"
        } else {
            "要"
        }
    } else if v_lemma == "伝える" || head_verb.surface.starts_with("伝え") {
        "轉告"
    } else if v_lemma == "頼む" || head_verb.surface.starts_with("頼") {
        "請"
    } else if v_lemma == "求める" || head_verb.surface.starts_with("求め") {
        "要求"
    } else if v_lemma == "命じる" || head_verb.surface.starts_with("命じ") {
        "命令"
    } else {
        "指示"
    };
    out.push_str(verb_zh);
    if !recip.is_empty() {
        out.push_str(&recip);
        if verb_zh == "轉告" {
            out.push('要');
        }
    }
    out.push_str(&sub_zh);

    Some((out, false))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GivingReceivingKind {
    Give,
    GiveTowardSpeaker,
    Receive,
}

#[derive(Clone, Copy)]
enum BenefactiveCore {
    Generic,
    Show,
    Lend,
}

#[derive(Clone, Copy)]
struct GivingReceivingPredicate {
    kind: GivingReceivingKind,
    atom_index: usize,
    auxiliary: bool,
    benefactive_core: BenefactiveCore,
}

fn giving_receiving_kind(atom: &Atom) -> Option<GivingReceivingKind> {
    if atom.is_locked {
        return None;
    }
    match atom.lemma.as_deref().unwrap_or(&atom.surface) {
        "あげる" => Some(GivingReceivingKind::Give),
        "くれる" | "くださる" => Some(GivingReceivingKind::GiveTowardSpeaker),
        "もらう" | "いただく" => Some(GivingReceivingKind::Receive),
        _ => None,
    }
}

fn giving_receiving_predicate(
    predicates: &[Bunsetsu],
    has_recipient: bool,
) -> Option<GivingReceivingPredicate> {
    let atoms: Vec<&Atom> = predicates.iter().flat_map(|b| &b.atoms).collect();
    let mut candidates = atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| giving_receiving_kind(a).map(|kind| (i, kind)));
    let (atom_index, kind) = candidates.next()?;
    if candidates.next().is_some()
        || atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.surface.as_str(),
                    "れる" | "られる" | "れ" | "られ" | "せる" | "させる" | "せ" | "させ"
                )
        })
    {
        return None;
    }
    let auxiliary = atom_index > 0
        && !atoms[atom_index - 1].is_locked
        && matches!(atoms[atom_index - 1].surface.as_str(), "て" | "で")
        && atoms[..atom_index - 1]
            .iter()
            .any(|a| a.upos == Some(Upos::Verb));
    // Honorific lexical verbs are ambiguous outside a te-form construction.
    // Bare `てください` remains a request.
    if matches!(
        atoms[atom_index]
            .lemma
            .as_deref()
            .unwrap_or(&atoms[atom_index].surface),
        "いただく" | "くださる"
    ) && (!auxiliary
        || atom_index < 2
        || !is_te_form_link(
            atoms[atom_index - 2],
            atoms[atom_index - 1],
            atoms[atom_index],
        )
        || matches!(
            atoms[atom_index].surface.as_str(),
            "ください" | "くださいませ"
        ))
    {
        return None;
    }
    // Kana `あげる` also means raising something. Require an explicit recipient
    // before interpreting a standalone occurrence as giving.
    if !auxiliary
        && !has_recipient
        && atoms[atom_index]
            .lemma
            .as_deref()
            .unwrap_or(&atoms[atom_index].surface)
            == "あげる"
    {
        return None;
    }
    // A desire belongs to the receiving subject, not to the person performing the action.
    // Keep control constructions on their existing path until that outer role is represented.
    if auxiliary
        && kind == GivingReceivingKind::Receive
        && atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.surface.as_str(),
                    "たい" | "たく" | "たかっ" | "たかった" | "べき" | "はず"
                )
        })
    {
        return None;
    }
    // Bare `てくれ` is a request, rather than an assertion of a benefit received.
    if auxiliary && atoms[atom_index].surface == "くれ" && atom_index + 1 == atoms.len() {
        return None;
    }
    if !auxiliary
        && atoms[..atom_index]
            .iter()
            .any(|a| a.upos == Some(Upos::Verb))
    {
        return None;
    }
    // Show and lend take the beneficiary as their own recipient. Do not turn
    // `lend to me` into `help me lend`. Extra cores stay on the generic path.
    let simple_core = auxiliary
        && atoms.iter().all(|a| {
            !a.is_locked
                && matches!(
                    a.upos,
                    Some(Upos::Verb | Upos::Aux | Upos::Sconj | Upos::Part | Upos::Adp)
                )
        })
        && atoms[..atom_index - 1]
            .iter()
            .filter(|a| a.upos == Some(Upos::Verb))
            .count()
            == 1;
    let benefactive_core = if simple_core {
        match atoms[..atom_index - 1]
            .iter()
            .find(|a| a.upos == Some(Upos::Verb))
            .map(|a| a.lemma.as_deref().unwrap_or(&a.surface))
        {
            Some("見せる") => BenefactiveCore::Show,
            Some("貸す") => BenefactiveCore::Lend,
            _ => BenefactiveCore::Generic,
        }
    } else {
        BenefactiveCore::Generic
    };
    Some(GivingReceivingPredicate {
        kind,
        atom_index,
        auxiliary,
        benefactive_core,
    })
}

// Only remove a trailing marker that clause assembly explicitly handles.
fn finite_predicate_atoms(b: &Bunsetsu) -> Vec<&Atom> {
    let mut atoms: Vec<&Atom> = b.atoms.iter().filter(|a| !is_punct_atom(a)).collect();
    while let Some(last) = atoms.last() {
        let handled_sconj = b.sconj_tail.as_deref() == Some(last.surface.as_str())
            && matches!(
                last.surface.as_str(),
                "てから"
                    | "から"
                    | "ので"
                    | "けれど"
                    | "けれども"
                    | "けど"
                    | "のに"
                    | "だけど"
                    | "し"
                    | "たら"
                    | "なら"
                    | "ば"
                    | "ながら"
                    | "つつ"
                    | "ても"
                    | "でも"
                    | "としても"
                    | "くせに"
                    | "が"
                    | "と"
            );
        let handled_sfp = b.sfp_tail.as_deref() == Some(last.surface.as_str())
            && matches!(
                last.surface.as_str(),
                "か" | "かな" | "よ" | "ね" | "よね" | "わ" | "ぞ" | "ぜ"
            );
        if last.is_locked || !(handled_sconj || handled_sfp) {
            break;
        }
        atoms.pop();
    }
    atoms
}

// Admit only complete, nonpast standalone forms. Unknown auxiliaries fall back.
fn standalone_iru_nonpast(predicates: &[Bunsetsu]) -> Option<bool> {
    if predicates.len() != 1 {
        return None;
    }
    let atoms = finite_predicate_atoms(&predicates[0]);
    let first = *atoms.first()?;
    if first.is_locked
        || first.upos != Some(Upos::Verb)
        || first.lemma.as_deref() != Some("いる")
        || atoms
            .iter()
            .skip(1)
            .any(|a| a.is_locked || a.upos != Some(Upos::Aux))
        || atoms
            .windows(2)
            .any(|p| p[0].span.end_char != p[1].span.start_char)
    {
        return None;
    }
    let form: String = atoms.iter().map(|a| a.surface.as_str()).collect();
    match form.as_str() {
        "いる" | "います" => Some(false),
        "いない" | "いません" => Some(true),
        _ => None,
    }
}

fn existential_location_link(b: &Bunsetsu, predicate_idx: usize) -> bool {
    if b.case_particle.as_deref() != Some("に")
        || b.atoms.iter().any(is_clock_atom)
        || (is_time_or_place_bunsetsu(&b.atoms) && !is_place_location_bunsetsu(&b.atoms))
    {
        return false;
    }
    let Some(particle) = b.atoms.iter().find(|a| !a.is_locked && a.surface == "に") else {
        return false;
    };
    let Some(head) = particle.head else {
        return false;
    };
    b.atoms.iter().any(|a| {
        a.doc_tok_idx == Some(head)
            && a.head == Some(predicate_idx)
            && matches!(a.deprel.as_str(), "obl" | "advmod")
    })
}

// Bare negative past can express inanimate absence. Reject other predicates
// and all unknown morphology so this never admits animate いる past.
fn bare_absence_past(predicates: &[Bunsetsu]) -> bool {
    if predicates.len() != 1 {
        return false;
    }
    let atoms = finite_predicate_atoms(&predicates[0]);
    if atoms.is_empty()
        || atoms.iter().any(|a| a.is_locked)
        || atoms[0].lemma.as_deref() != Some("ない")
        || atoms.iter().skip(1).any(|a| a.upos != Some(Upos::Aux))
        || atoms
            .windows(2)
            .any(|p| p[0].span.end_char != p[1].span.start_char)
    {
        return false;
    }
    let form: String = atoms.iter().map(|a| a.surface.as_str()).collect();
    form == "なかった"
}

// Grammatical attachment alone cannot distinguish temporal 前/後 from place.
fn fused_relational_location(pair: &[Bunsetsu], predicate_idx: Option<usize>) -> bool {
    let [owner, location] = pair else {
        return false;
    };
    let Some(spatial) = location.atoms.first() else {
        return false;
    };
    let Some(genitive) = owner.atoms.last() else {
        return false;
    };
    let Some(owner_idx) = genitive.head else {
        return false;
    };
    ends_in_genitive_no(owner)
        && !genitive.is_locked
        && owner.atoms.iter().any(|a| a.doc_tok_idx == Some(owner_idx))
        && owner.atoms.iter().all(|a| a.upos != Some(Upos::Verb))
        && genitive.span.end_char == spatial.span.start_char
        && location.role == BunsetsuRole::Adv
        && location.atoms.len() == 1
        && !spatial.is_locked
        && matches!(
            spatial.surface.as_str(),
            "中に" | "上に" | "下に" | "外に" | "横に"
        )
        && spatial.head.is_some()
        && spatial.head == predicate_idx
}

const BUILDING_FLOOR: &str = "__ja_zh_building_floor";

fn is_building_floor(atom: &Atom) -> bool {
    !atom.is_locked && atom.lemma.as_deref() == Some(BUILDING_FLOOR)
}

fn is_japanese_numeral(surface: &str) -> bool {
    !surface.is_empty()
        && surface.chars().all(|ch| {
            matches!(
                ch,
                '0'..='9' | '０'..='９' | '〇' | '零' | '一' | '二' | '三' | '四' | '五'
                    | '六' | '七' | '八' | '九' | '十' | '百' | '千' | '万' | '萬'
            )
        })
}

fn is_floor_facility(atom: &Atom, doc: &Document) -> bool {
    if atom.is_locked || !matches!(atom.upos, Some(Upos::Noun | Upos::Propn)) {
        return false;
    }
    let source = atom.lemma.as_deref().unwrap_or(&atom.surface);
    // The lexicon does not tag all facilities. Productive room, building,
    // store and service-location compounds also give place evidence.
    atom.doc_tok_idx
        .and_then(|index| doc.tokens.get(index))
        .is_some_and(|token| token.sem_class == Some(SemanticClass::Location))
        || is_place_location_bunsetsu(std::slice::from_ref(atom))
        || matches!(
            source,
            "事務所" | "受付" | "食堂" | "台所" | "トイレ" | "玄関" | "売り場" | "店舗"
        )
        || (source.ends_with(['室', '館', '棟', '店']) && !matches!(source, "王室" | "皇室"))
        || source.ends_with("薬局")
        || source.ends_with("郵便局")
}

fn has_place_genitive(atoms: &[Atom], head: usize, doc: &Document) -> bool {
    atoms.windows(2).any(|modifier| {
        let place = &modifier[0];
        let particle = &modifier[1];
        place.head == Some(head)
            && place.deprel == "nmod"
            && !particle.is_locked
            && particle.surface == "の"
            && particle.head.is_some()
            && particle.head == place.doc_tok_idx
            && is_floor_facility(place, doc)
    })
}

fn is_floor_service_space(atom: &Atom, atoms: &[Atom], doc: &Document) -> bool {
    const SERVICE_SPACES: [&str; 4] = ["窓口", "案内所", "コーナー", "カウンター"];

    // A service counter or section can also be an abstract contact point.
    // Require a source modifier that links it to a real place.
    let Some(head) = atom.doc_tok_idx else {
        return false;
    };
    if atom.is_locked || atom.upos != Some(Upos::Noun) || !has_place_genitive(atoms, head, doc) {
        return false;
    }
    if SERVICE_SPACES
        .into_iter()
        .any(|suffix| atom.surface.ends_with(suffix))
    {
        return true;
    }

    // Recognize a known service compound without relabeling its bare noun
    // head or changing the token spans and noun rendering.
    let modifier = atoms.windows(2).find_map(|pair| {
        let dependent = &pair[0];
        let noun = &pair[1];
        (noun.doc_tok_idx == Some(head)
            && noun.span == atom.span
            && dependent.head == Some(head)
            && dependent.deprel == "compound"
            && dependent.span.end_char == atom.span.start_char
            && is_compound_noun_pair(dependent, Some(atom)))
        .then_some(dependent.surface.as_str())
    });
    modifier.is_some_and(|modifier| {
        SERVICE_SPACES.into_iter().any(|name| {
            name.strip_suffix(atom.surface.as_str())
                .is_some_and(|stem| modifier.ends_with(stem))
        })
    })
}

fn has_floor_order_context(atoms: &[Atom], floor: usize, predicate: usize) -> bool {
    atoms.iter().any(|atom| {
        if atom.is_locked
            || !(atom.doc_tok_idx == Some(predicate)
                || (atom.head == Some(predicate)
                    && matches!(atom.deprel.as_str(), "nsubj" | "obj" | "compound"))
                || (atom.head == Some(floor)
                    && matches!(atom.deprel.as_str(), "nmod" | "compound")))
        {
            return false;
        }

        // Mathematical operators and expressions can use 階 for order. Read
        // the attached stem or noun, not an unrelated word in the clause.
        let source = atom.lemma.as_deref().unwrap_or(&atom.surface);
        let term = source.strip_suffix("する").unwrap_or(source);
        matches!(
            term,
            "数学" | "式" | "数式" | "公式" | "次数" | "階数" | "行列" | "演算" | "計算" | "近似"
        ) || ["微分", "積分", "関数", "函数", "方程式", "多項式", "級数"]
            .into_iter()
            .any(|suffix| term.ends_with(suffix))
    })
}

// Identify the complete unlocked floor label in the source.
fn building_floor_label_start(atoms: &[Atom], end: usize) -> Option<usize> {
    let floor = &atoms[end];
    if floor.is_locked || floor.upos != Some(Upos::Noun) {
        return None;
    }
    let prefix = floor.surface.strip_suffix('階')?;
    let mut start = end;
    if prefix.is_empty() {
        while let Some(number) = start.checked_sub(1).and_then(|i| atoms.get(i))
            && !number.is_locked
            && matches!(number.upos, Some(Upos::Num | Upos::Noun))
            && is_japanese_numeral(&number.surface)
            && number.span.end_char == atoms[start].span.start_char
        {
            start -= 1;
        }
        if start == end {
            return None;
        }
    } else if !is_japanese_numeral(prefix.strip_prefix("地下").unwrap_or(prefix)) {
        return None;
    }
    if !prefix.starts_with("地下") && start > 0 {
        let before = &atoms[start - 1];
        if !before.is_locked
            && before.span.end_char == atoms[start].span.start_char
            && before.surface == "地下"
        {
            start -= 1;
        } else if start > 1
            && !before.is_locked
            && before.span.end_char == atoms[start].span.start_char
            && before.surface == "下"
            && !atoms[start - 2].is_locked
            && atoms[start - 2].surface == "地"
            && atoms[start - 2].span.end_char == before.span.start_char
        {
            start -= 2;
        }
    }
    if start > 0 {
        let before = &atoms[start - 1];
        if before.span.end_char == atoms[start].span.start_char
            && (before.is_locked
                || matches!(before.upos, Some(Upos::Noun | Upos::Propn | Upos::Num))
                || before.surface == "第")
        {
            return None;
        }
    }
    Some(start)
}

fn has_floor_number_link(label: &[Atom], floor: &Atom, floor_idx: usize) -> bool {
    floor.surface != "階"
        || label.iter().any(|number| {
            number.head == Some(floor_idx)
                && number.deprel == "nummod"
                && is_japanese_numeral(&number.surface)
        })
}

fn building_floor_start(atoms: &[Atom], end: usize, doc: &Document) -> Option<usize> {
    let start = building_floor_label_start(atoms, end)?;
    let floor = &atoms[end];
    let floor_idx = floor.doc_tok_idx?;
    let tail = atoms.get(end + 1)?;
    if tail.is_locked || floor.span.end_char != tail.span.start_char {
        return None;
    }
    let negative_copula = matches!(
        tail.surface.as_str(),
        "じゃない" | "ではない" | "ではありません"
    );
    if tail.upos == Some(Upos::Aux)
        && (matches!(tail.surface.as_str(), "だ" | "です") || negative_copula)
    {
        // Polite negative ではありません can attach both nouns to the
        // original ある. The normalized negative copula retains that head.
        let shared_negative_head = floor
            .head
            .filter(|head| negative_copula && floor.deprel == "nsubj" && tail.head == Some(*head));
        return atoms[..start]
            .iter()
            .any(|subject| {
                subject.deprel == "nsubj"
                    && (subject.head == Some(floor_idx)
                        || shared_negative_head.is_some_and(|head| subject.head == Some(head)))
                    && (is_floor_facility(subject, doc)
                        || is_floor_service_space(subject, &atoms[..start], doc))
            })
            .then_some(start);
    }

    let governor = floor.head?;
    if tail.upos != Some(Upos::Adp) || tail.head != Some(floor_idx) {
        return None;
    }
    let predicate = atoms.iter().find(|predicate| {
        !predicate.is_locked
            && predicate.doc_tok_idx == Some(governor)
            && predicate.upos == Some(Upos::Verb)
    })?;
    let verb = predicate.lemma.as_deref().unwrap_or(&predicate.surface);
    let motion = matches!(
        verb,
        "行く" | "来る" | "帰る" | "戻る" | "着く" | "向かう" | "入る"
    );
    let spatial_case = tail.deprel == "case" && floor.deprel == "obl";
    let is_spatial = match tail.surface.as_str() {
        "に" => motion || matches!(verb, "ある" | "いる"),
        "へ" => motion,
        "から" => spatial_case && (motion || is_made_motion_verb(verb)),
        "まで" => spatial_case && is_made_motion_verb(verb),
        "で" => {
            // A numbered floor linked to an event by で normally gives its
            // venue. An explicit building modifier resolves math contexts.
            spatial_case
                && has_floor_number_link(&atoms[start..end], floor, floor_idx)
                && (has_place_genitive(&atoms[..start], floor_idx, doc)
                    || !has_floor_order_context(atoms, floor_idx, governor))
        }
        _ => false,
    };
    is_spatial.then_some(start)
}

fn building_floor_atom(label: &[Atom], floor: &Atom) -> Atom {
    // Keep the source head of 階 and the complete numeral and suffix span.
    let mut floor = floor.clone();
    floor.span.start_char = label[0].span.start_char;
    floor.surface = label.iter().map(|a| a.surface.as_str()).collect();
    floor.lemma = Some(BUILDING_FLOOR.to_owned());
    floor
}

fn normalize_building_floors(mut atoms: Vec<Atom>, doc: &Document) -> Vec<Atom> {
    let mut end = atoms.len();
    while end > 0 {
        end -= 1;
        let Some(start) = building_floor_start(&atoms, end, doc) else {
            continue;
        };
        let floor = building_floor_atom(&atoms[start..=end], &atoms[end]);
        atoms.splice(start..=end, [floor]);
        end = start;
    }
    atoms
}

const LOCKED_PLACE_ATOM: &str = "__ja_locked_glossary_place";

const LOCKED_RESIDENT_ATOM: &str = "__ja_locked_glossary_resident";

/// A focus particle can precede or follow the case particle. Both must belong
/// to the same noun. The focus of 妹だけが differs from that of 京都だけに.
fn residence_case(group: &Bunsetsu) -> Option<(&Atom, &Atom, Option<&Atom>)> {
    let (noun, case, focus) = match group.atoms.as_slice() {
        [.., noun, focus, case] if focus.surface == "だけ" => (noun, case, Some(focus)),
        [.., noun, case, focus]
            if matches!(focus.surface.as_str(), "は" | "も" | "だけ")
                && case.upos == Some(Upos::Adp) =>
        {
            (noun, case, Some(focus))
        }
        [.., noun, case] => (noun, case, None),
        _ => return None,
    };
    let contiguous = focus.map_or(noun.span.end_char == case.span.start_char, |f| {
        !f.is_locked
            && f.head == noun.doc_tok_idx
            && (f.surface == "だけ" && f.deprel == "discourse" || f.deprel == "case")
            && (noun.span.end_char == f.span.start_char && f.span.end_char == case.span.start_char
                || noun.span.end_char == case.span.start_char
                    && case.span.end_char == f.span.start_char)
    });
    (contiguous
        && !case.is_locked
        && case.upos == Some(Upos::Adp)
        && case.deprel == "case"
        && noun.doc_tok_idx.is_some()
        && case.head == noun.doc_tok_idx)
        .then_some((noun, case, focus))
}

fn mark_residence_place(groups: &mut Vec<Bunsetsu>, source: &[Atom]) {
    use faberline_nlp::lexicon::{SemanticClass, lookup_ja};

    let Some(action_index) = source.iter().position(|a| {
        !a.is_locked
            && a.upos == Some(Upos::Verb)
            && a.lemma.as_deref().unwrap_or(&a.surface) == "住む"
    }) else {
        return;
    };
    let action = &source[action_index];
    let Some(id) = action.doc_tok_idx else { return };
    let (form, finite, tail) = match &source[action_index + 1..] {
        [te, iru, tail @ ..]
            if action.deprel == "advcl"
                && action.head == iru.doc_tok_idx
                && !te.is_locked
                && !iru.is_locked
                && te.deprel == "mark"
                && te.head == Some(id)
                && matches!(te.surface.as_str(), "て" | "で")
                && iru.upos == Some(Upos::Verb)
                && iru.lemma.as_deref().unwrap_or(&iru.surface) == "いる"
                && iru.deprel == "root" =>
        {
            (ResidenceForm::State, iru, tail)
        }
        [plan, copula, tail @ ..]
            if action.surface == "住む"
                && action.deprel == "advcl"
                && action.head == plan.doc_tok_idx
                && !plan.is_locked
                && plan.surface == "予定"
                && plan.deprel == "root"
                && !copula.is_locked
                && copula.surface == "です"
                && copula.deprel == "aux"
                && copula.head == plan.doc_tok_idx =>
        {
            (ResidenceForm::Plan, plan, tail)
        }
        tail if action.deprel == "root" => (ResidenceForm::Simple, action, tail),
        _ => return,
    };
    if tail.iter().any(|a| {
        a.is_locked
            || a.head != finite.doc_tok_idx
            || !(a.upos == Some(Upos::Aux)
                && matches!(
                    a.surface.as_str(),
                    "た" | "だ" | "ます" | "まし" | "ませ" | "ん" | "でし" | "ない" | "なかっ"
                )
                || a.deprel == "discourse"
                || is_punct_atom(a))
    }) {
        return;
    }
    let mut places = groups.iter().enumerate().filter_map(|(i, group)| {
        let (noun, ni, focus) = residence_case(group)?;
        let protected_place = noun.is_locked && noun.lemma.as_deref() == Some(LOCKED_PLACE_ATOM);
        if noun.is_locked && !protected_place
            || ni.is_locked
            || !(noun.deprel == "obl"
                || focus.is_some_and(|f| matches!(f.surface.as_str(), "は" | "も"))
                    && noun.deprel == "nsubj")
            || noun.head != Some(id)
            || ni.surface != "に"
        {
            return None;
        }
        // The dictionary tags proper place names. Ordinary housing nouns do
        // not all carry that tag. Check the head, never a place in its possessor.
        let wh = noun.surface == "どこ";
        if wh
            && focus.is_some_and(|mo| mo.surface == "も")
            && !tail.iter().any(|a| {
                a.upos == Some(Upos::Aux) && matches!(a.surface.as_str(), "ん" | "ない" | "なかっ")
            })
        {
            return None;
        }
        let place = wh
            || protected_place
            || is_place_location_bunsetsu(std::slice::from_ref(noun))
            || lookup_ja(&noun.surface)
                .is_some_and(|e| e.sem_class == Some(SemanticClass::Location))
            || noun.upos == Some(Upos::Noun)
                && matches!(
                    noun.surface.as_str(),
                    "アパート"
                        | "マンション"
                        | "住宅"
                        | "寮"
                        | "村"
                        | "町"
                        | "市"
                        | "島"
                        | "田舎"
                        | "都会"
                );
        place.then_some((
            i,
            noun.doc_tok_idx?,
            focus.filter(|f| f.surface == "だけ").map(|f| f.span),
        ))
    });
    let Some((i, place, only_place)) = places.next() else {
        return;
    };
    // にも repeated on independently marked places applies the same
    // predicate to each. Two plain に arguments do not prove coordination.
    let extra_places: Vec<_> = places.map(|(index, _, _)| index).collect();
    if !extra_places.is_empty()
        && !std::iter::once(&i).chain(&extra_places).all(|&index| {
            residence_case(&groups[index]).is_some_and(|(noun, _, focus)| {
                noun.surface != "どこ" && focus.is_some_and(|f| f.surface == "も")
            })
        })
    {
        return;
    }
    for index in extra_places {
        groups[index].role = BunsetsuRole::AdditionalResidencePlace;
    }
    let wh = source.windows(2).enumerate().any(|(index, w)| {
        let (word, case) = (&w[0], &w[1]);
        !word.is_locked
            && word.head == Some(id)
            && matches!(word.deprel.as_str(), "nsubj" | "obl")
            && matches!(word.surface.as_str(), "誰" | "だれ" | "どこ")
            && !(word.surface == "どこ"
                && case.surface == "に"
                && source.get(index + 2).is_some_and(|mo| {
                    !mo.is_locked
                        && mo.surface == "も"
                        && mo.deprel == "case"
                        && mo.head == word.doc_tok_idx
                        && mo.span.start_char == case.span.end_char
                }))
            && !case.is_locked
            && case.upos == Some(Upos::Adp)
            && case.deprel == "case"
            && case.head == word.doc_tok_idx
            && word.span.end_char == case.span.start_char
    });
    groups[i].role = BunsetsuRole::ResidencePlace {
        form,
        past: tail.iter().any(|a| a.surface == "た"),
        wh,
        only: only_place.is_some(),
    };
    if let Some(span) = only_place {
        groups[i].atoms.retain(|a| a.span != span);
    }
    // A final subject marker applies to the entire coordinated noun phrase.
    let mut coordinated = Vec::new();
    for subject in 0..groups.len() {
        let Some((noun, case, focus)) = residence_case(&groups[subject]) else {
            continue;
        };
        if noun.head != Some(id)
            || noun.deprel != "nsubj"
            || !matches!(case.surface.as_str(), "は" | "が")
        {
            continue;
        }
        let only = focus.filter(|a| a.surface == "だけ").map(|a| a.span);
        let mut first = subject;
        while first > 0
            && residence_case(&groups[first - 1]).is_some_and(|(noun, case, _)| {
                noun.head == Some(id)
                    && case.surface == "と"
                    && case.span.end_char == groups[first].atoms[0].span.start_char
            })
        {
            first -= 1;
        }
        coordinated.extend(first..subject);
        if let Some(span) = only {
            groups[subject].atoms.retain(|a| a.span != span);
            prepend_locked_atom(&mut groups[first], "只有".to_owned());
        }
    }
    let together = groups.iter().position(|g| {
        residence_case(g).is_some_and(|(noun, case, _)| {
            !noun.is_locked
                && noun.surface == "一緒"
                && noun.head == Some(id)
                && case.surface == "に"
        })
    });
    if let Some(index) = together {
        groups[index].role = BunsetsuRole::ResidenceTogether;
    }
    let person = place_companion(source, id, Some(place));
    for (index, group) in groups.iter_mut().enumerate() {
        let Some((noun, case, _)) = residence_case(group) else {
            continue;
        };
        if case.surface == "で"
            && noun.head == Some(id)
            && noun.surface == "人"
            && matches!(group.atoms.as_slice(), [number, _, _] if !number.is_locked
                && number.upos == Some(Upos::Num) && number.deprel == "nummod"
                && number.head == noun.doc_tok_idx && number.span.end_char == noun.span.start_char)
        {
            group.role = BunsetsuRole::ResidenceCount;
        } else if case.surface == "と" && noun.head == Some(id) && !coordinated.contains(&index) {
            // Xと一緒に explicitly marks a companion. Without 一緒, the
            // dictionary's complete kinship noun can establish this reading.
            let kin_or_pet = !noun.is_locked
                && noun.upos == Some(Upos::Noun)
                && (matches!(noun.surface.as_str(), "犬" | "猫")
                    || lookup_ja(&noun.surface).is_some_and(|entry| {
                        entry.upos_candidates.as_slice() == [Upos::Noun]
                            && matches!(
                                entry.counterpart.as_deref(),
                                Some(
                                    "家人"
                                        | "父母"
                                        | "親族"
                                        | "親子"
                                        | "兄弟"
                                        | "姊妹"
                                        | "夫婦"
                                        | "丈夫"
                                        | "妻子"
                                )
                            )
                    }));
            let explicit_companion = together == Some(index + 1)
                && source.iter().any(|a| {
                    a.span.start_char == case.span.end_char
                        && !a.is_locked
                        && a.surface == "一緒"
                        && a.head == Some(id)
                });
            if noun.doc_tok_idx == person
                || kin_or_pet
                || explicit_companion
                || noun.is_locked
                    && (noun.lemma.as_deref() == Some(LOCKED_RESIDENT_ATOM)
                        || noun.lemma.as_deref() != Some(LOCKED_PLACE_ATOM)
                            && (together.is_some() || i < index))
            {
                group.role = BunsetsuRole::JointCompanion;
            } else if noun.is_locked && noun.lemma.as_deref() != Some(LOCKED_PLACE_ATOM) {
                // Opaque と before a place can mark a companion or a second place.
                // Preserve the source particle along with the protected surface.
                group.role = BunsetsuRole::UnresolvedResidenceArgument;
            }
        }
    }
    localize_indefinite_arguments(groups, action, source);
    // A bare number and a time unit next to 住む give a duration. 年 can be
    // analysed as a separate adverb, even when the two source words touch.
    if let [.., number, unit] = &source[..action_index]
        && !number.is_locked
        && !unit.is_locked
        && number.upos == Some(Upos::Num)
        && number.span.end_char == unit.span.start_char
        && unit.span.end_char == action.span.start_char
        && matches!(
            unit.surface.as_str(),
            "年" | "年間" | "週間" | "日間" | "か月" | "ヶ月"
        )
        && unit.head == Some(id)
        && matches!(unit.deprel.as_str(), "obl" | "advmod")
        && (number.head == unit.doc_tok_idx && number.deprel == "nummod"
            || number.head == Some(id) && number.deprel == "obl")
        && let Some(start) = groups
            .iter()
            .position(|g| g.atoms.first().is_some_and(|a| a.span == number.span))
        && let Some(end) = groups
            .iter()
            .position(|g| g.atoms.last().is_some_and(|a| a.span == unit.span))
        && start <= end
    {
        for group in &mut groups[start..=end] {
            group.role = BunsetsuRole::ResidenceDuration;
        }
    }
}

fn translate_clause_bunsetsus(
    mut bunsetsus: Vec<Bunsetsu>,
    source: &[Atom],
    raw_clause: &str,
    conditioning: Option<&SentenceConditioning>,
    allow_pro_drop_inject: bool,
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> (String, bool) {
    #[derive(Clone, Copy)]
    enum DeadlineEndpoint {
        Calendar,
        ClockOrDaypart,
        Question { calendar: bool },
    }

    // Track extra bunsetsu-level metadata for `に`/`で` localization.
    #[derive(Clone)]
    struct PlacedArg {
        text: String,
        case_particle: Option<String>,
        is_instrument_de: bool,
        is_spatial_made: bool,
        temporal_made_time: Option<String>,
        is_event_time: bool,
        is_building_floor: bool,
        deadline_endpoint: DeadlineEndpoint,
        deadline_has_detached_negation: bool,
        is_event_time_ni: bool,
        transfer_dative: bool,
        physical_goal_head: Option<usize>,
    }

    struct PlacedSubj {
        text: String,
        has_mo: bool,
        had_ga: bool,
        is_hou: bool,
        is_person_like: bool,
    }

    let mut connectors: Vec<String> = Vec::new();
    let mut subj_entries: Vec<PlacedSubj> = Vec::new();
    let mut subjects: Vec<String> = Vec::new();
    let mut obliques: Vec<PlacedArg> = Vec::new();
    let mut temporal_others: Vec<String> = Vec::new();
    let mut event_counts = Vec::new();
    let mut counted_arrival = None;
    let arrival_head = bunsetsus
        .iter()
        .flat_map(|b| &b.atoms)
        .find(|a| calendar::is_counted_subject(a))
        .and_then(|a| a.head);
    let mut count_modifier = None;
    let mut joint_manner = None;
    let mut joint_participants = None;
    let mut joint_companions = Vec::new();
    let mut travel_companions = Vec::new();
    let mut residence = None;
    let mut residence_additive = false;
    let mut additional_residences = Vec::new();
    let mut residence_duration = String::new();
    let mut residence_manners = Vec::new();
    let mut adverbs: Vec<String> = Vec::new();
    let mut iobjs: Vec<PlacedArg> = Vec::new();
    let mut objs: Vec<String> = Vec::new();
    let mut route = None;
    let mut purpose = None;
    let mut purpose_partners = Vec::new();
    let mut purpose_recipients = Vec::new();
    let wh_argument = bunsetsus.iter().find(|b| b.role == BunsetsuRole::MotionPurpose)
        .is_some_and(|purpose| {
            let stem = &purpose.atoms[0];
            bunsetsus.iter().any(|b| matches!(b.atoms.as_slice(), [wh, case]
                if !wh.is_locked && !case.is_locked && wh.doc_tok_idx.is_some() && wh.head.is_some()
                && (wh.head == stem.doc_tok_idx || wh.head == stem.head)
                && matches!(wh.surface.as_str(), "何" | "誰" | "どれ" | "どこ" | "いつ")
                && matches!(wh.deprel.as_str(), "nsubj" | "obj" | "obl" | "iobj" | "advmod")
                && case.upos == Some(Upos::Adp) && case.deprel == "case"
                && matches!(case.surface.as_str(), "が" | "を" | "に" | "へ" | "で" | "と" | "から" | "まで")
                && case.head == wh.doc_tok_idx && wh.span.end_char == case.span.start_char))
        });
    let mut go_together = Vec::new();
    let mut to_complements: Vec<(String, bool)> = Vec::new();
    let mut predicates: Vec<Bunsetsu> = Vec::new();
    let mut others: Vec<String> = Vec::new();
    let mut sconj_prefix: Option<&'static str> = None;
    let mut sconj_suffix: Option<&'static str> = None;
    let mut sfp_suffix: Option<&'static str> = None;
    let mut saw_to_sconj = false;
    let mut is_toshitemo_clause = false;
    let mut has_competing_transfer_recipient = false;

    let mut candidates: Vec<Bunsetsu> = bunsetsus
        .iter()
        .filter(|b| b.role == BunsetsuRole::Predicate)
        .cloned()
        .collect();
    let motion = governed_route_motion(&candidates, source);
    let motion_head = motion.map(|m| m.head);
    let mut traversal_object = false;
    // Interpret a source-linked terminal が on a clone first. Do not mutate
    // clause metadata until every standalone existential guard has passed.
    let mut candidate_contrast = false;
    if candidates.len() == 1
        && bunsetsus
            .last()
            .is_some_and(|b| b.role == BunsetsuRole::Predicate)
    {
        let b = &mut candidates[0];
        let predicate_idx = b
            .atoms
            .iter()
            .find(|a| {
                !a.is_locked && a.lemma.as_deref() == Some("いる") && a.upos == Some(Upos::Verb)
            })
            .and_then(|a| a.doc_tok_idx);
        candidate_contrast = matches!(b.sconj_tail.as_deref(), None | Some("が"))
            && b.atoms.last().is_some_and(|a| {
                !a.is_locked
                    && a.surface == "が"
                    && predicate_idx.is_some()
                    && a.head == predicate_idx
            });
        if candidate_contrast {
            b.sconj_tail = Some("が".to_owned());
        }
    }
    let predicate_idx = candidates
        .first()
        .and_then(|b| {
            b.atoms
                .iter()
                .find(|a| !a.is_locked && a.lemma.as_deref() == Some("いる"))
        })
        .and_then(|a| a.doc_tok_idx);
    let linked = predicate_idx
        .map(|idx| {
            bunsetsus
                .iter()
                .filter(|b| existential_location_link(b, idx))
                .count()
        })
        .unwrap_or(0);
    let ni_count = bunsetsus
        .iter()
        .filter(|b| b.case_particle.as_deref() == Some("に"))
        .count();
    let subject_count = bunsetsus
        .iter()
        .filter(|b| b.role == BunsetsuRole::Subj)
        .count();
    let animate_existential = standalone_iru_nonpast(&candidates).filter(|_| {
        (candidates[0].sconj_tail.as_deref() != Some("が") || candidate_contrast)
            && linked == 1
            && ni_count == 1
            && subject_count == 1
            && !bunsetsus.iter().any(|b| b.role == BunsetsuRole::Obj)
            && bunsetsus
                .iter()
                .filter(|b| b.role == BunsetsuRole::Subj)
                .all(|b| matches!(b.case_particle.as_deref(), Some("が" | "は")))
            && !bunsetsus
                .iter()
                .filter(|b| b.role != BunsetsuRole::Predicate)
                .flat_map(|b| &b.atoms)
                .any(|a| !a.is_locked && a.upos == Some(Upos::Verb))
    });
    let animate_contrast = animate_existential.is_some() && candidate_contrast;
    if animate_contrast && let Some(b) = bunsetsus.last_mut() {
        b.sconj_tail = Some("が".to_owned());
        if b.case_particle.as_deref() == Some("が") {
            b.case_particle = None;
        }
    }

    let absence_predicate_idx = candidates.first().and_then(|b| {
        b.atoms
            .iter()
            .find(|a| !a.is_locked && a.lemma.as_deref() == Some("ない"))
            .and_then(|a| a.doc_tok_idx)
    });
    let absence_locations = absence_predicate_idx
        .map(|idx| {
            bunsetsus
                .iter()
                .filter(|b| existential_location_link(b, idx))
                .count()
        })
        .unwrap_or(0);
    let fused_pairs: Vec<usize> = bunsetsus
        .windows(2)
        .enumerate()
        .filter_map(|(idx, pair)| {
            fused_relational_location(pair, absence_predicate_idx).then_some(idx)
        })
        .collect();
    let unique_absence_location =
        (absence_locations == 1 && ni_count == 1 && fused_pairs.is_empty())
            || (absence_locations == 0 && ni_count == 0 && fused_pairs.len() == 1);
    let inanimate_past = bare_absence_past(&candidates)
        && unique_absence_location
        && subject_count == 1
        && bunsetsus
            .iter()
            .filter(|b| b.role == BunsetsuRole::Subj)
            .all(|b| b.case_particle.as_deref() == Some("が"))
        && !bunsetsus.iter().any(|b| b.role == BunsetsuRole::Obj)
        && !bunsetsus
            .iter()
            .filter(|b| b.role != BunsetsuRole::Predicate)
            .flat_map(|b| &b.atoms)
            .any(|a| !a.is_locked && a.upos == Some(Upos::Verb));
    // A fused spatial adverb can leave N の outside the normal noun merge.
    // Only join the adjacent source constituents for bare past absence.
    if inanimate_past && let Some(&idx) = fused_pairs.first() {
        let mut owner = bunsetsus[idx].clone();
        let spatial = &bunsetsus[idx + 1];
        owner.atoms.pop();
        let owner = render_bunsetsu_np(&owner, target);
        let stem = spatial.atoms[0].surface.strip_suffix('に').unwrap_or("");
        let relation = translate_single_ja_term(stem, None, Some(Upos::Noun));
        if !owner.is_empty() && !relation.is_empty() {
            bunsetsus.remove(idx);
            bunsetsus.remove(idx);
            subjects.push(format!("{owner}{relation}"));
        }
    }

    let asks_event_time = source.last().is_some_and(|ka| {
        !ka.is_locked
            && ka.surface == "か"
            && ka.upos == Some(Upos::Part)
            && ka.deprel == "discourse"
            && ka.head.is_some()
            && bunsetsus.iter().any(|group| {
                group.atoms.iter().any(|time| {
                    time.head == ka.head
                        && (calendar::asks_interval(time)
                            || match group.case_particle.as_deref() {
                                Some("までに") => deadline_time_question(time).is_some(),
                                Some("に") => calendar::question(time).is_some(),
                                _ => false,
                            })
                })
            })
    });
    let explicit_deadline = bunsetsus.iter().find_map(|b| match b.role {
        BunsetsuRole::ExplicitDeadline { modal, additive } => Some((modal, additive)),
        _ => None,
    });
    let deadline_action = if explicit_deadline.is_none() {
        deadline_ability_action(&bunsetsus)
    } else {
        None
    };
    // Resolve time and result roles while the other arguments of the predicate are available.
    let timed_result = timed_result_frame(&bunsetsus, source);
    let mut trip_destination = None;
    let returning = ctx.return_trip.map(|trip| trip.returning).or_else(|| {
        if !bunsetsus
            .iter()
            .any(|b| b.role == BunsetsuRole::TravelCompanion)
        {
            return None;
        }
        // A separate return keeps its own companion and a single destination.
        let root = source
            .iter()
            .find(|a| {
                !a.is_locked
                    && a.deprel == "root"
                    && matches!(a.lemma.as_deref().unwrap_or(&a.surface), "帰る" | "戻る")
            })?
            .doc_tok_idx?;
        (bunsetsus.iter().filter(|b| return_goal(b, root)).count() == 1).then_some(root)
    });
    let mut affected_object = None;
    let mut result_value = None;
    let mut setting_origin = None;
    let daypart_groups: Vec<_> = bunsetsus
        .iter()
        .enumerate()
        .filter_map(|(idx, group)| is_event_daypart(group, &bunsetsus, source).then_some(idx))
        .collect();
    for (idx, b) in bunsetsus.drain(..).enumerate() {
        if arrival_head.is_some()
            && b.role == BunsetsuRole::Subj
            && b.case_particle.as_deref() == Some("が")
            && b.atoms
                .iter()
                .any(|a| a.deprel == "nsubj" && a.head == arrival_head)
        {
            counted_arrival = Some(calendar::arrival_noun(&b, target));
            continue;
        }
        if let [atom] = b.atoms.as_slice() {
            if calendar::individual_arrivals(atom) {
                adverbs.push(atom.surface.clone());
                continue;
            }
            if calendar::is_interval(atom) || calendar::is_habit(atom) {
                temporal_others.push(atom.surface.clone());
                continue;
            }
            if let Some(modifier) = calendar::count_modifier(atom) {
                if !modifier.is_empty() {
                    count_modifier = Some(modifier);
                }
                event_counts.push(atom.surface.clone());
                continue;
            }
        }
        if let Some(trip) = ctx.return_trip
            && let Some(label) = trip_relative_daypart(&b, source, trip.returning)
        {
            temporal_others.push(label);
            continue;
        }
        if returning.is_some_and(|root| return_goal(&b, root)) {
            trip_destination = Some(render_bunsetsu_np(&b, target));
            continue;
        }
        if timed_result.is_some_and(|f| f.object == idx) {
            affected_object = Some(render_bunsetsu_np(&b, target));
            continue;
        }
        if let Some(frame) = timed_result.filter(|f| f.value == idx) {
            result_value = Some(if matches!(frame.time, SettingTime::From(_)) {
                setting_time_label(&b, target)
            } else {
                frame
                    .site_name
                    .map_or_else(|| render_bunsetsu_np(&b, target), str::to_owned)
            });
            continue;
        }
        if timed_result.is_some_and(|f| f.time == SettingTime::From(idx)) {
            setting_origin = Some(format!("從{}", setting_time_label(&b, target)));
            continue;
        }
        if daypart_groups.contains(&idx)
            || timed_result.is_some_and(|f| f.time == SettingTime::At(idx))
        {
            temporal_others.push(render_daypart_np(&b, target));
            continue;
        }
        if b.case_particle.as_deref() == Some("へ") && !is_time_or_place_bunsetsu(&b.atoms) {
            has_competing_transfer_recipient = true;
        }
        if let Some(sc) = b.sconj_tail.as_deref() {
            match sc {
                "てから" => sconj_suffix = Some("後"),
                "から" | "ので" => sconj_prefix = Some("因為"),
                "からといって" => sconj_prefix = Some("即便因為"),
                "が" if animate_contrast => sconj_prefix = Some("雖然"),
                "けれど" | "けれども" | "けど" | "のに" => sconj_suffix = Some("，但是"),
                "だけど" => sconj_suffix = Some("，不過"),
                "し" => sconj_suffix = Some("，而且"),
                "たら" | "なら" | "ば" => sconj_prefix = Some("如果"),
                "ながら" | "つつ" => sconj_prefix = Some("一邊"),
                "ても" | "でも" => sconj_prefix = Some("即使"),
                "としても" => {
                    sconj_prefix = Some(if raw_clause.contains("仮に") {
                        "即使"
                    } else {
                        "就算"
                    });
                    is_toshitemo_clause = true;
                }
                "くせに" => sconj_prefix = Some("明明"),
                "と" => {
                    saw_to_sconj = true;
                    sconj_prefix = Some("當");
                    sconj_suffix = Some("時");
                }
                _ => {}
            }
        }
        if let Some(sfp) = b.sfp_tail.as_deref() {
            sfp_suffix = match sfp {
                "か" if asks_event_time => None,
                "か" | "かな" => Some("嗎"),
                "よ" => Some("喔"),
                "ね" | "よね" => Some("呢"),
                "わ" => Some("呀"),
                "ぞ" | "ぜ" => Some("啊"),
                _ => sfp_suffix,
            };
        }

        match b.role {
            BunsetsuRole::ResidencePlace {
                form,
                past,
                wh,
                only,
            } => {
                residence_additive = b.atoms.last().is_some_and(|a| a.surface == "も");
                // Negative どこにも means nowhere. It does not ask which place.
                let anywhere = residence_additive
                    && matches!(b.atoms.as_slice(),
                    [where_, _, _] if !where_.is_locked && where_.surface == "どこ");
                residence = Some((
                    render_bunsetsu_np(&b, target),
                    form,
                    past,
                    wh,
                    only,
                    anywhere,
                ));
            }
            BunsetsuRole::AdditionalResidencePlace => {
                additional_residences.push(render_bunsetsu_np(&b, target))
            }
            BunsetsuRole::ResidenceTogether => residence_manners.push("一起".to_owned()),
            BunsetsuRole::ResidenceCount => {
                if let Some(number) = b
                    .atoms
                    .iter()
                    .find(|a| a.upos == Some(Upos::Num) && !a.is_locked)
                {
                    let number = normalize_decimal_digits(&number.surface);
                    let count = match number.as_str() {
                        "一" | "1" => "一個人".to_owned(),
                        "二" | "2" => "兩個人".to_owned(),
                        other => format!("{other}個人"),
                    };
                    residence_manners.push(count);
                }
            }
            BunsetsuRole::UnresolvedResidenceArgument => {
                joint_companions.push(format!("{}と", render_bunsetsu_np(&b, target)));
            }
            BunsetsuRole::ResidenceDuration => {
                for atom in &b.atoms {
                    let text = match atom.surface.as_str() {
                        "年間" => "年",
                        "週間" => "週",
                        "日間" => "天",
                        "か月" | "ヶ月" => "個月",
                        "二" => "兩",
                        other => other,
                    };
                    residence_duration.push_str(text);
                }
            }
            BunsetsuRole::JointManner {
                manner,
                invitation,
                wh_question,
            } => {
                joint_participants = match manner {
                    JointManner::Companion => {
                        joint_companions.push(format!("和{}", render_bunsetsu_np(&b, target)));
                        None
                    }
                    JointManner::Together => None,
                    JointManner::Everyone => Some("大家".to_owned()),
                    JointManner::Counted => Some(render_joint_people(&b.atoms[0].surface)),
                };
                joint_manner = Some((manner, invitation, wh_question));
            }
            BunsetsuRole::JointCompanion => {
                joint_companions.push(format!("和{}", render_bunsetsu_np(&b, target)));
            }
            BunsetsuRole::TravelCompanion => {
                travel_companions.push(format!("跟{}", render_bunsetsu_np(&b, target)));
            }
            BunsetsuRole::MotionPurpose => purpose = b.atoms[0].lemma.clone(),
            BunsetsuRole::PurposeModifier if b.atoms[0].surface == "一緒" => {
                go_together.push(render_bunsetsu_np(&b, target));
            }
            BunsetsuRole::PurposeCompanion => {
                go_together.push(format!("和{}", render_bunsetsu_np(&b, target)));
            }
            BunsetsuRole::PurposeModifier | BunsetsuRole::ProposalModifier => {
                temporal_others.push(render_bunsetsu_np(&b, target));
            }
            BunsetsuRole::ProposalCompanion => {
                temporal_others.push(format!("和{}", render_bunsetsu_np(&b, target)));
            }
            BunsetsuRole::PurposePartner(marker) => {
                purpose_partners.push(format!("{marker}{}", render_bunsetsu_np(&b, target)));
            }
            BunsetsuRole::PurposeRecipient => {
                purpose_recipients.push(format!("給{}", render_bunsetsu_np(&b, target)));
            }
            BunsetsuRole::Connector => {
                let s = render_bunsetsu_np(&b, target);
                if !s.is_empty() {
                    connectors.push(s);
                }
            }
            BunsetsuRole::Subj => {
                let is_wh_mo = b.case_particle.as_deref() == Some("も")
                    && b.atoms.iter().any(|a| {
                        !a.is_locked
                            && matches!(
                                a.surface.as_str(),
                                "何" | "なに" | "なん" | "誰" | "どこ" | "いつ"
                            )
                    });
                let is_datte = b.case_particle.as_deref() == Some("だって");
                let has_mo = b.case_particle.as_deref() == Some("も");
                let had_ga = b.case_particle.as_deref() == Some("が");
                let is_hou = had_ga && b.atoms.iter().any(|a| !a.is_locked && a.surface == "ほう");
                let is_person_like = is_person_like_bunsetsu(&b);
                let s = render_bunsetsu_np(&b, target);
                if !s.is_empty() {
                    if is_wh_mo {
                        adverbs.push(format!("{s}都"));
                    } else if is_datte {
                        subj_entries.push(PlacedSubj {
                            text: format!("就算是{s}"),
                            has_mo: false,
                            had_ga: false,
                            is_hou: false,
                            is_person_like,
                        });
                    } else {
                        subj_entries.push(PlacedSubj {
                            text: s,
                            has_mo,
                            had_ga,
                            is_hou,
                            is_person_like,
                        });
                    }
                }
            }
            BunsetsuRole::Obl
            | BunsetsuRole::ExplicitDeadline { .. }
            | BunsetsuRole::DeadlineWithDetachedNegation => {
                let is_inst = b.case_particle.as_deref() == Some("で")
                    && ((!is_place_location_bunsetsu(&b.atoms)
                        && is_instrument_bunsetsu(&b.atoms))
                        || is_payment_instrument(&b.atoms, &candidates));
                let s = render_bunsetsu_obl(&b, target, is_inst);
                if !s.is_empty() {
                    obliques.push(PlacedArg {
                        text: s,
                        case_particle: b.case_particle.clone(),
                        is_instrument_de: is_inst,
                        is_spatial_made: is_spatial_made_bunsetsu(&b),
                        temporal_made_time: temporal_made_time(&b),
                        is_event_time: b
                            .atoms
                            .iter()
                            .any(|a| is_clock_atom(a) || calendar::is_date(a)),
                        is_building_floor: b.atoms.iter().any(is_building_floor),
                        deadline_endpoint: if b
                            .atoms
                            .iter()
                            .any(|a| calendar::question(a).is_some())
                        {
                            DeadlineEndpoint::Question { calendar: true }
                        } else if b.atoms.iter().any(|a| deadline_time_question(a).is_some()) {
                            DeadlineEndpoint::Question { calendar: false }
                        } else if b.atoms.iter().any(|a| {
                            is_clock_atom(a) || !a.is_locked && is_daypart_surface(&a.surface)
                        }) {
                            DeadlineEndpoint::ClockOrDaypart
                        } else {
                            DeadlineEndpoint::Calendar
                        },
                        deadline_has_detached_negation: b.role
                            == BunsetsuRole::DeadlineWithDetachedNegation,
                        is_event_time_ni: is_plain_clock_ni(&b)
                            || (b.case_particle.as_deref() == Some("に")
                                && b.atoms.iter().any(calendar::is_date)),
                        transfer_dative: b.case_particle.as_deref() == Some("に")
                            && !is_time_or_place_bunsetsu(&b.atoms),
                        physical_goal_head: physical_goal_head(&b),
                    });
                }
            }
            BunsetsuRole::Adv => {
                let s = render_bunsetsu_np(&b, target);
                if !s.is_empty() {
                    if (b.atoms.len() == 1
                        && b.atoms[0].lemma.as_deref() == Some(RELATIVE_DAYPART_ATOM))
                        || matches!(
                            s.as_str(),
                            "每天"
                                | "今天"
                                | "明天"
                                | "昨天"
                                | "現在"
                                | "早上"
                                | "夜晚"
                                | "今年"
                                | "明年"
                                | "去年"
                                | "今後"
                                | "久違地"
                        )
                    {
                        temporal_others.push(s);
                    } else {
                        adverbs.push(s);
                    }
                }
            }
            BunsetsuRole::IObj => {
                let s = render_bunsetsu_iobj(&b, target);
                if !s.is_empty() {
                    iobjs.push(PlacedArg {
                        text: s,
                        case_particle: b.case_particle.clone(),
                        is_instrument_de: false,
                        is_spatial_made: false,
                        temporal_made_time: None,
                        is_event_time: false,
                        is_building_floor: b.atoms.iter().any(is_building_floor),
                        deadline_endpoint: DeadlineEndpoint::Calendar,
                        deadline_has_detached_negation: false,
                        is_event_time_ni: false,
                        transfer_dative: b.case_particle.as_deref() == Some("に")
                            && !is_time_or_place_bunsetsu(&b.atoms),
                        physical_goal_head: physical_goal_head(&b),
                    });
                }
            }
            BunsetsuRole::Obj => {
                let text = render_bunsetsu_np(&b, target);
                if let Some(marker) = motion
                    .filter(|m| !m.expressed_by_verb)
                    .and_then(|m| route_marker(&b, source, m.head))
                {
                    route = Some(format!("{marker}{text}"));
                } else if !text.is_empty() {
                    traversal_object |= motion.is_some_and(|m| {
                        m.expressed_by_verb && governs_traversed_space(source, m.head)
                    });
                    objs.push(text);
                }
            }
            BunsetsuRole::Predicate => {
                predicates.push(b);
            }
            BunsetsuRole::Punct => {}
            BunsetsuRole::Other => {
                let s = render_bunsetsu_np(&b, target);
                if !s.is_empty() {
                    if matches!(b.case_particle.as_deref(), Some("と") | Some("とも")) {
                        let has_mo = b.case_particle.as_deref() == Some("とも")
                            || b.atoms.iter().any(|a| {
                                !a.is_locked && matches!(a.surface.as_str(), "も" | "とも")
                            });
                        to_complements.push((s, has_mo));
                    } else if matches!(
                        s.as_str(),
                        "每天" | "今天" | "明天" | "昨天" | "現在" | "早上" | "夜晚"
                    ) {
                        temporal_others.push(s);
                    } else {
                        others.push(s);
                    }
                }
            }
        }
    }

    if saw_to_sconj && (raw_clause.ends_with("とすぐ") || raw_clause.ends_with("とすぐに")) {
        if let Some(pos) = adverbs.iter().rposition(|a| a == "馬上") {
            adverbs.remove(pos);
        }
        sconj_prefix = Some("一");
        sconj_suffix = Some("，就立刻");
    }
    if sconj_prefix == Some("即使")
        && adverbs
            .iter()
            .any(|a| a.contains("無論") || a.contains("不管"))
    {
        sconj_prefix = None;
    }

    if is_toshitemo_clause {
        adverbs.retain(|a| !matches!(a.as_str(), "如果" | "就算" | "萬一" | "即使"));
        connectors.retain(|c| !matches!(c.as_str(), "如果" | "就算" | "萬一" | "即使"));
    }

    let had_overt_subj_no_pred = !subj_entries.is_empty() && predicates.is_empty();
    let pred_is_imperative = predicates.iter().any(predicate_is_imperative);
    let pred_looks_passive = predicates_has_passive(&predicates);

    // Layer 6 Pro-Drop resolution: if no overt subject and narrative/coref resolved pro-drop,
    // inject pronoun when appropriate (skip when passive clause has a promotable patient in `objs`).
    if subj_entries.is_empty()
        && (!pred_looks_passive || objs.is_empty())
        && allow_pro_drop_inject
        && !pred_is_imperative
        && !predicates.is_empty()
        && let Some(cond) = conditioning
    {
        if let Some(hint) = cond.pronoun_hint.as_deref()
            && let Some(zh_pro) = zh_pronoun_for_hint(hint)
        {
            subjects.push(zh_pro.to_owned());
        } else if let Some(pro_drop) = cond.resolved_pro_drop.as_deref()
            && !pro_drop.is_empty()
            && !raw_clause.contains(pro_drop)
        {
            let translated = translate_single_ja_term(pro_drop, None, Some(Upos::Propn));
            subjects.push(translated);
        }
    }

    // Bind the transfer argument before motion and other verb rules consume `に`.
    let dative_count = iobjs
        .iter()
        .chain(&obliques)
        .filter(|p| p.transfer_dative)
        .count();
    let from_count = obliques
        .iter()
        .filter(|p| p.case_particle.as_deref() == Some("から"))
        .count();
    let transfer = giving_receiving_predicate(&predicates, dative_count == 1).filter(|_| {
        dative_count + from_count <= 1
            && subj_entries.len() + subjects.len() <= 1
            && !has_competing_transfer_recipient
    });
    let transfer_counterpart = transfer.and_then(|transfer| {
        if let Some(i) = iobjs.iter().position(|p| p.transfer_dative) {
            let arg = iobjs.remove(i);
            Some(arg.text.strip_prefix('對').unwrap_or(&arg.text).to_owned())
        } else if let Some(i) = obliques.iter().position(|p| p.transfer_dative) {
            let arg = obliques.remove(i);
            Some(
                arg.text
                    .strip_prefix('在')
                    .or_else(|| arg.text.strip_prefix('對'))
                    .unwrap_or(&arg.text)
                    .to_owned(),
            )
        } else if transfer.kind == GivingReceivingKind::Receive
            && let Some(i) = obliques
                .iter()
                .position(|p| p.case_particle.as_deref() == Some("から"))
        {
            let arg = obliques.remove(i);
            Some(arg.text.strip_prefix('從').unwrap_or(&arg.text).to_owned())
        } else if transfer.kind == GivingReceivingKind::GiveTowardSpeaker && from_count == 0 {
            // `くれる` points toward the speaker, including in attributed direct speech.
            Some("我".to_owned())
        } else {
            None
        }
    });

    // Synthesize predicate complex.
    let (mut pre_verb_mods, mut verb_core, mut post_verb_mods, passive_agent_needed) =
        synthesize_predicate_complex(&predicates, !objs.is_empty(), target, transfer);
    if let Some(trip) = ctx.return_trip {
        verb_core = if let Some(place) = trip_destination {
            format!("{}回{}", trip.action, place)
        } else {
            format!("{}{verb_core}", trip.action)
        };
    } else if let Some(place) = trip_destination {
        verb_core = format!("回到{place}");
    }
    if let Some((place, form, past, wh, only, anywhere)) = residence {
        joint_companions.append(&mut residence_manners);
        let mut elapsed_aspect = None;
        pre_verb_mods.retain(|m| m != "正在");
        if form == ResidenceForm::Plan {
            pre_verb_mods.insert(0, "打算".to_owned());
        }
        let negative = pre_verb_mods
            .iter()
            .any(|m| matches!(m.as_str(), "不" | "沒有"));
        if residence_duration.is_empty() {
            if past && (form == ResidenceForm::State || !negative) {
                post_verb_mods.retain(|m| m != "了");
                // The past negative already has 沒有. An overt time such as
                // 当時 belongs in its original modifier or noun phrase.
                if !negative {
                    pre_verb_mods.insert(0, "曾".to_owned());
                }
            }
        } else {
            post_verb_mods.retain(|m| m != "了");
            if negative {
                if form == ResidenceForm::State {
                    if !past {
                        temporal_others.push("已經".to_owned());
                    }
                    for m in &mut pre_verb_mods {
                        if m == "不" {
                            *m = "沒有".to_owned();
                        }
                    }
                }
                temporal_others.push(residence_duration.clone());
            } else {
                let aspect = if past || form == ResidenceForm::State {
                    "了"
                } else {
                    ""
                };
                if form == ResidenceForm::State && !past {
                    pre_verb_mods.push("已經".to_owned());
                }
                elapsed_aspect = Some(aspect);
            }
        }
        if only {
            pre_verb_mods.push("只".to_owned());
        }
        let render_place = |location: &str| match elapsed_aspect {
            Some(aspect) => format!("在{location}住{aspect}{residence_duration}"),
            None => format!("住在{location}"),
        };
        verb_core = render_place(&place);
        for other in additional_residences {
            verb_core.push_str("，也");
            for modifier in &pre_verb_mods {
                verb_core.push_str(modifier);
            }
            verb_core.push_str(&render_place(&other));
        }
        let nowhere = anywhere && negative;
        if nowhere {
            residence_additive = false;
            verb_core = "住".to_owned();
            if let Some(index) = pre_verb_mods
                .iter()
                .position(|m| matches!(m.as_str(), "不" | "沒有"))
            {
                if form == ResidenceForm::State {
                    pre_verb_mods[index] = "沒有".to_owned();
                }
                pre_verb_mods.insert(index, "哪裡都".to_owned());
            }
        }
        if wh && sfp_suffix == Some("嗎") {
            sfp_suffix = Some("？");
        }
    }
    if let Some(frame) = timed_result {
        verb_core = frame.action.to_owned();
    }

    if inanimate_past {
        pre_verb_mods.insert(0, "當時".to_owned());
    }

    if is_toshitemo_clause {
        post_verb_mods.retain(|m| m != "了");
    }

    if let Some(negative) = animate_existential
        && subj_entries.len() == 1
        && ((iobjs.len() == 1 && obliques.is_empty()) || (obliques.len() == 1 && iobjs.is_empty()))
    {
        let location = if iobjs.len() == 1 {
            iobjs.remove(0).text
        } else {
            obliques.remove(0).text
        };
        let location = location
            .strip_prefix('給')
            .or_else(|| location.strip_prefix('對'))
            .or_else(|| location.strip_prefix('在'))
            .unwrap_or(&location)
            .to_owned();
        let participant = subj_entries.remove(0);
        pre_verb_mods.retain(|m| m != "不" && m != "沒有");
        if participant.had_ga {
            subjects.push(location);
            objs.push(participant.text);
            verb_core = if negative { "沒有" } else { "有" }.to_owned();
        } else {
            subjects.push(participant.text);
            verb_core = if negative { "不在" } else { "在" }.to_owned();
            objs.push(location);
        }
        post_verb_mods.clear();
    }

    // Comparative preference `[X] ほうが [好き / いい / 良い]`
    if let Some(h_idx) = subj_entries.iter().position(|s| s.is_hou) {
        let raw_comp = subj_entries.remove(h_idx).text;
        let comp_target = raw_comp
            .strip_suffix("的方")
            .or_else(|| raw_comp.strip_suffix('方'))
            .unwrap_or(&raw_comp)
            .trim_end_matches('的')
            .to_owned();
        if verb_core == "喜歡" {
            let pref = if ctx.in_toshitemo_follow || ctx.in_concessive_follow {
                "也比較"
            } else {
                "比較"
            };
            pre_verb_mods.push(pref.to_owned());
            objs.push(comp_target);
            post_verb_mods.retain(|m| m != "了");
        } else if matches!(verb_core.as_str(), "好" | "很好") {
            subjects.push(comp_target);
            verb_core = "比較好".to_owned();
            post_verb_mods.retain(|m| m != "了");
        } else {
            subjects.push(comp_target);
        }
    }

    // Existential emphatic `[X も] + ない`: `休む暇もない` -> `連休息的空閒都沒有`
    if verb_core == "沒有"
        && objs.is_empty()
        && let Some(mo_idx) = subj_entries.iter().position(|s| s.has_mo)
    {
        let mo_subj = subj_entries.remove(mo_idx).text;
        subjects.push(format!("連{mo_subj}都沒有"));
        verb_core.clear();
        post_verb_mods.retain(|m| m != "了");
    }

    // Existential / possessive `ある`: `[X に] + [Y が] + ある` -> `[X] 有 [Y]`; `[Y が] + ある` -> `有 [Y]`
    let pred_is_aru = predicates.iter().any(|b| {
        b.atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                    "ある" | "あり" | "あっ" | "あった"
                )
        })
    });
    let pred_is_bare_nai = predicates.iter().flat_map(|b| &b.atoms).any(|a| {
        !a.is_locked
            && matches!(
                a.lemma.as_deref().unwrap_or(&a.surface),
                "ない" | "なかっ" | "なかった"
            )
    }) && predicates.iter().flat_map(|b| &b.atoms).all(|a| {
        !a.is_locked
            && (matches!(a.upos, Some(Upos::Aux | Upos::Part | Upos::Sconj))
                || (a.upos == Some(Upos::Adj)
                    && matches!(
                        a.lemma.as_deref().unwrap_or(&a.surface),
                        "ない" | "なかっ" | "なかった"
                    )))
    });
    let negative_existence = verb_core == "沒有" && (pred_is_aru || pred_is_bare_nai);
    if (pred_is_aru && verb_core == "有") || negative_existence {
        // Put a single nominal item after 有 / 沒有 in both polarities.
        // Keep people and any separate possessor topic as subjects.
        let move_item_topic = subj_entries.len() == 1 && !subj_entries[0].is_person_like;
        let has_ni = iobjs
            .iter()
            .chain(obliques.iter())
            .any(|p| p.case_particle.as_deref() == Some("に"));
        if has_ni {
            let mut kept_iobjs = Vec::new();
            for io in iobjs.drain(..) {
                if io.case_particle.as_deref() == Some("に") {
                    let raw_np = io
                        .text
                        .strip_prefix('對')
                        .or_else(|| io.text.strip_prefix('給'))
                        .or_else(|| io.text.strip_prefix('在'))
                        .unwrap_or(&io.text);
                    subjects.push(raw_np.to_owned());
                } else {
                    kept_iobjs.push(io);
                }
            }
            iobjs = kept_iobjs;
            let mut kept_obls = Vec::new();
            for o in obliques.drain(..) {
                if o.case_particle.as_deref() == Some("に") {
                    let raw_np = o
                        .text
                        .strip_prefix('在')
                        .or_else(|| o.text.strip_prefix('對'))
                        .or_else(|| o.text.strip_prefix('給'))
                        .unwrap_or(&o.text);
                    subjects.push(raw_np.to_owned());
                } else {
                    kept_obls.push(o);
                }
            }
            obliques = kept_obls;
            let mut rem_subjs = Vec::new();
            for s in subj_entries.drain(..) {
                if s.had_ga || move_item_topic {
                    objs.push(s.text);
                } else {
                    rem_subjs.push(s);
                }
            }
            subj_entries = rem_subjs;
        } else if objs.is_empty()
            && !subj_entries.is_empty()
            && (move_item_topic || subj_entries.iter().any(|s| s.had_ga))
        {
            subj_entries.retain(|s| {
                if s.had_ga || move_item_topic {
                    objs.push(s.text.clone());
                    false
                } else {
                    true
                }
            });
        }
    }

    // Emit remaining subjects and inclusive `も` (`也`).
    let mut added_mo_adv = residence_additive;
    if residence_additive {
        adverbs.push("也".to_owned());
    }
    for s in subj_entries.drain(..) {
        if s.has_mo && !ctx.in_concessive_follow && !added_mo_adv {
            adverbs.insert(0, "也".to_owned());
            added_mo_adv = true;
        }
        subjects.push(s.text);
    }
    if added_mo_adv {
        for m in &mut pre_verb_mods {
            if m == "也並不是" {
                *m = "並不是".to_owned();
            }
        }
    }

    let pred_has_naru = predicates.iter().any(|b| {
        b.atoms.iter().any(|a| {
            !a.is_locked
                && (a.lemma.as_deref() == Some("なる")
                    || a.surface == "なる"
                    || a.surface.starts_with("なっ"))
        })
    });
    if saw_to_sconj
        && pred_has_naru
        && subjects.is_empty()
        && objs.is_empty()
        && iobjs.len() + obliques.len() == 1
    {
        let single_ni = iobjs
            .iter()
            .chain(obliques.iter())
            .find(|p| p.case_particle.as_deref() == Some("に"))
            .map(|p| {
                p.text
                    .strip_prefix('對')
                    .or_else(|| p.text.strip_prefix('給'))
                    .or_else(|| p.text.strip_prefix('在'))
                    .unwrap_or(&p.text)
                    .to_owned()
            });
        if let Some(time_np) = single_ni
            && (matches!(
                time_np.as_str(),
                "週末"
                    | "春"
                    | "夏"
                    | "秋"
                    | "冬"
                    | "秋天"
                    | "春天"
                    | "夏天"
                    | "冬天"
                    | "早上"
                    | "中午"
                    | "夜晚"
            ) || time_np
                .ends_with(['末', '時', '天', '年', '月', '日', '週', '期', '季', '夜']))
        {
            iobjs.clear();
            obliques.clear();
            sconj_prefix = Some("一到");
            sconj_suffix = None;
            verb_core.clear();
            pre_verb_mods.clear();
            post_verb_mods.clear();
            others.push(time_np);
        }
    }

    if !to_complements.is_empty() {
        if pred_has_naru && matches!(verb_core.as_str(), "成為" | "變成") {
            for (np, has_mo) in to_complements.drain(..) {
                if has_mo {
                    objs.push(format!("也成為了{np}"));
                } else {
                    objs.push(format!("已成為{np}"));
                }
            }
            verb_core.clear();
            pre_verb_mods.retain(|m| m != "正在");
            post_verb_mods.retain(|m| m != "了");
        } else {
            for (np, _) in to_complements.drain(..) {
                obliques.push(PlacedArg {
                    text: format!("跟{np}"),
                    case_particle: Some("と".to_owned()),
                    is_instrument_de: false,
                    is_spatial_made: false,
                    temporal_made_time: None,
                    is_event_time: false,
                    is_building_floor: false,
                    deadline_endpoint: DeadlineEndpoint::Calendar,
                    deadline_has_detached_negation: false,
                    is_event_time_ni: false,
                    transfer_dative: false,
                    physical_goal_head: None,
                });
            }
        }
    }

    // Weather-subject collapse: `雨が` + `下` -> verb `下雨` (keeps `正在`).
    if let Some(last_subj) = subjects.last()
        && matches!(last_subj.as_str(), "雨" | "雪")
        && verb_core == "下"
    {
        let weather = subjects.pop().unwrap_or_default();
        verb_core = format!("下{weather}");
    }

    // Light-verb `する` (`做`) + object promotion:
    if verb_core == "做" && objs.len() == 1 {
        if objs[0] == "親吻" {
            verb_core = "親吻".to_owned();
            objs.clear();
        } else if objs[0].ends_with("親吻") {
            verb_core = "獻上".to_owned();
        } else if matches!(
            objs[0].as_str(),
            "閱讀" | "學習" | "道歉" | "研究" | "調查" | "練習" | "運動" | "工作" | "散步"
        ) {
            verb_core = objs.remove(0);
        }
    }

    // Move numeric measure phrases (`100公尺`) into post-verbal object slot for motion verbs.
    if objs.is_empty()
        && others.len() == 1
        && others[0].chars().next().is_some_and(|c| c.is_ascii_digit())
        && predicates.iter().any(|b| {
            b.atoms.iter().any(|a| {
                !a.is_locked
                    && matches!(
                        a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                        "泳ぐ" | "泳げる" | "走る" | "歩く" | "飛ぶ"
                    )
            })
        })
    {
        objs.push(others.remove(0));
    }

    // Idiomatic `生活/日々/人生 + を + 送る` -> `過`
    if predicates.iter().any(|b| {
        b.atoms.iter().any(|a| {
            !a.is_locked && (a.lemma.as_deref() == Some("送る") || a.surface.starts_with("送"))
        })
    }) && objs.iter().any(|o| {
        o.ends_with("生活") || o.ends_with("日子") || o.ends_with("每天") || o.ends_with("人生")
    }) {
        verb_core = "過".to_owned();
    }

    // Subjectless emotional reaction `驚いた` -> `感到驚訝`
    if verb_core == "驚訝" && subjects.is_empty() && !pre_verb_mods.iter().any(|m| m == "感到")
    {
        verb_core = "感到驚訝".to_owned();
    }

    // Passive patient promotion: when passive has no overt subject and has an `を` object, promote `objs` before `被`,
    // except for ditransitive / assignment verbs (`仕事を頼まれる` -> `被交代工作`, `会議を入れられる` -> `被安排會議`).
    let is_ditransitive_assign = predicates.iter().any(|b| {
        b.atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                    "頼む" | "入れる" | "任せる" | "命じる" | "指示する" | "求める"
                )
        })
    });
    if passive_agent_needed && !is_ditransitive_assign && subjects.is_empty() && !objs.is_empty() {
        subjects.append(&mut objs);
    }
    if passive_agent_needed {
        let has_tanomu = predicates.iter().any(|b| {
            b.atoms.iter().any(|a| {
                !a.is_locked && (a.lemma.as_deref() == Some("頼む") || a.surface.starts_with("頼"))
            })
        });
        let has_ireru = predicates.iter().any(|b| {
            b.atoms.iter().any(|a| {
                !a.is_locked
                    && (a.lemma.as_deref() == Some("入れる") || a.surface.starts_with("入れ"))
            })
        });
        if has_tanomu
            && objs
                .iter()
                .any(|o| o.contains("工作") || o.contains("業務") || o.contains("事"))
        {
            verb_core = "交代".to_owned();
        } else if has_ireru
            && objs.iter().any(|o| {
                o.contains("會議") || o.contains("預定") || o.contains("行程") || o.contains("班")
            })
        {
            verb_core = "安排".to_owned();
        }
    }
    if verb_core == "進展"
        && subjects
            .iter()
            .any(|s| s.contains("計畫") || s.contains("計画"))
    {
        verb_core = "進行".to_owned();
    }

    // Purpose-of-motion `[ActionNoun に] + [出かける / 行く / 来る]` -> `出門旅行` / `去購物`
    if predicates.iter().any(|b| {
        b.atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                    "出かける" | "出掛ける" | "行く" | "来る"
                )
        })
    }) {
        let is_action_purpose = |np: &str| {
            matches!(
                np,
                "旅行"
                    | "購物"
                    | "買東西"
                    | "散步"
                    | "吃飯"
                    | "用餐"
                    | "工作"
                    | "玩"
                    | "觀光"
                    | "探望"
                    | "迎接"
            )
        };
        let mut kept_iobjs = Vec::new();
        for io in iobjs.drain(..) {
            if io.case_particle.as_deref() == Some("に") {
                let raw_np = io
                    .text
                    .strip_prefix('對')
                    .or_else(|| io.text.strip_prefix('給'))
                    .or_else(|| io.text.strip_prefix('在'))
                    .unwrap_or(&io.text);
                if is_action_purpose(raw_np) {
                    objs.push(raw_np.to_owned());
                    continue;
                }
            }
            kept_iobjs.push(io);
        }
        iobjs = kept_iobjs;

        let mut kept_obls = Vec::new();
        for o in obliques.drain(..) {
            if o.case_particle.as_deref() == Some("に") && !o.is_event_time {
                let raw_np = o
                    .text
                    .strip_prefix('在')
                    .or_else(|| o.text.strip_prefix('對'))
                    .or_else(|| o.text.strip_prefix('給'))
                    .unwrap_or(&o.text);
                if is_action_purpose(raw_np) {
                    objs.push(raw_np.to_owned());
                    continue;
                }
            }
            kept_obls.push(o);
        }
        obliques = kept_obls;
    }

    let is_goal_verb = predicates_is_goal_verb(&predicates);
    if is_goal_verb {
        for o in &mut obliques {
            if o.is_instrument_de
                && let Some(veh) = o.text.strip_prefix('用')
                && is_transport_term(veh)
            {
                o.text = format!("坐{veh}");
            }
        }
    }
    let is_cause_verb = predicates_is_cause_verb(&predicates);
    let has_causative = pre_verb_mods.iter().any(|m| m == "讓");

    // Only a single directly linked physical goal of a directly negated 行く
    // can use Chinese 去 + place order. Leave competing に / まで untouched.
    let mut lexical_verbs = predicates
        .iter()
        .flat_map(|b| &b.atoms)
        .filter(|a| !a.is_locked && a.upos == Some(Upos::Verb));
    let direct_go = lexical_verbs
        .next()
        .filter(|a| a.lemma.as_deref().unwrap_or(&a.surface) == "行く");
    let is_negative_go = verb_core == "去"
        && pre_verb_mods
            .iter()
            .any(|m| matches!(m.as_str(), "不" | "沒有"));
    let negative_go_head = if lexical_verbs.next().is_none()
        && is_negative_go
        && !has_causative
        && !passive_agent_needed
        && objs.is_empty()
        && !predicates
            .iter()
            .flat_map(|b| &b.atoms)
            .any(|a| a.is_locked)
    {
        direct_go.and_then(|a| a.doc_tok_idx)
    } else {
        None
    };
    let mut marked_goals = obliques
        .iter()
        .chain(&iobjs)
        .filter(|o| matches!(o.case_particle.as_deref(), Some("に" | "へ")) && !o.is_event_time_ni);
    let only_goal = marked_goals.next();
    let direct_negative_physical_go = negative_go_head.is_some()
        && only_goal.is_some_and(|o| o.physical_goal_head == negative_go_head)
        && marked_goals.next().is_none()
        && !obliques
            .iter()
            .any(|o| o.case_particle.as_deref() == Some("まで"));
    let format_directional_goal = |arg: &PlacedArg, raw_np: &str| {
        if direct_negative_physical_go && arg.physical_goal_head == negative_go_head {
            raw_np.to_owned()
        } else {
            format_goal_tail(&verb_core, raw_np, arg.is_building_floor && !is_negative_go)
        }
    };

    // Extract directional `に` goals (from `obliques` or `iobjs`) BEFORE passive agent consumes `iobjs`!
    let mut goal_tails: Vec<String> = Vec::new();
    let mut cause_prefixes: Vec<String> = Vec::new();
    let mut pre_iobjs: Vec<String> = Vec::new();

    if is_goal_verb {
        let mut kept_obls = Vec::new();
        for o in obliques.drain(..) {
            if o.case_particle.as_deref() == Some("に") && !o.is_event_time {
                let raw_np = o
                    .text
                    .strip_prefix('在')
                    .or_else(|| o.text.strip_prefix('對'))
                    .or_else(|| o.text.strip_prefix('給'))
                    .unwrap_or(&o.text);
                goal_tails.push(format_directional_goal(&o, raw_np));
            } else {
                kept_obls.push(o);
            }
        }
        obliques = kept_obls;

        let mut kept_iobjs = Vec::new();
        for io in iobjs.drain(..) {
            if io.case_particle.as_deref() == Some("に") {
                let raw_np = io
                    .text
                    .strip_prefix('對')
                    .or_else(|| io.text.strip_prefix('給'))
                    .or_else(|| io.text.strip_prefix('在'))
                    .unwrap_or(&io.text);
                goal_tails.push(format_directional_goal(&io, raw_np));
            } else {
                kept_iobjs.push(io);
            }
        }
        iobjs = kept_iobjs;
    }

    if !goal_tails.is_empty() && verb_core == "回去" {
        verb_core = "回".to_owned();
    }

    if is_cause_verb && !passive_agent_needed {
        let mut kept_iobjs = Vec::new();
        for io in iobjs.drain(..) {
            if io.case_particle.as_deref() == Some("に") {
                let raw_np = io
                    .text
                    .strip_prefix('對')
                    .or_else(|| io.text.strip_prefix('給'))
                    .unwrap_or(&io.text);
                cause_prefixes.push(format!("因{raw_np}而"));
            } else {
                kept_iobjs.push(io);
            }
        }
        iobjs = kept_iobjs;
    }

    // Promote `に` complements to direct objects for verbs like `気づく` (`發現`), `かかる` (`取決於`), `拍車をかける` (`加劇`), `つながる` (`帶來`), `対応` (`應對`).
    let promotes_ni_to_obj = matches!(verb_core.as_str(), "取決於" | "加劇" | "帶來" | "應對")
        || predicates.iter().any(|b| {
            b.atoms.iter().any(|a| {
                !a.is_locked
                    && (matches!(
                        a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                        "気づく" | "気付く" | "つながる" | "繋がる" | "対応" | "対応する"
                    ) || a.surface.starts_with("気づ")
                        || a.surface.starts_with("気付")
                        || a.surface.starts_with("つなが")
                        || a.surface.starts_with("繋が")
                        || a.surface.starts_with("対応"))
            })
        });
    if promotes_ni_to_obj && !passive_agent_needed {
        let mut kept_iobjs = Vec::new();
        for io in iobjs.drain(..) {
            if io.case_particle.as_deref() == Some("に") {
                let raw_np = io
                    .text
                    .strip_prefix('對')
                    .or_else(|| io.text.strip_prefix('給'))
                    .or_else(|| io.text.strip_prefix('在'))
                    .unwrap_or(&io.text);
                objs.push(raw_np.to_owned());
            } else {
                kept_iobjs.push(io);
            }
        }
        iobjs = kept_iobjs;

        let mut kept_obls = Vec::new();
        for o in obliques.drain(..) {
            if o.case_particle.as_deref() == Some("に") {
                let raw_np = o
                    .text
                    .strip_prefix('在')
                    .or_else(|| o.text.strip_prefix('對'))
                    .or_else(|| o.text.strip_prefix('給'))
                    .unwrap_or(&o.text);
                objs.push(raw_np.to_owned());
            } else {
                kept_obls.push(o);
            }
        }
        obliques = kept_obls;
    }

    // Separate top-level pre-verbal modals (`請不要` / `不要` / `請` / `讓` / `聽說`) so modals precede adverbs.
    let mut front_modals: Vec<String> = Vec::new();
    let mut inner_pre_mods: Vec<String> = Vec::new();
    for m in pre_verb_mods.drain(..) {
        if matches!(m.as_str(), "請不要" | "不要" | "請" | "讓" | "聽說") {
            front_modals.push(m);
        } else {
            inner_pre_mods.push(m);
        }
    }
    if ctx.in_nagara_follow {
        inner_pre_mods.retain(|m| m != "正在");
        inner_pre_mods.insert(0, "一邊".to_owned());
    }
    if adverbs
        .iter()
        .any(|a| matches!(a.as_str(), "一定" | "總是" | "經常"))
        || temporal_others.iter().any(|t| t == "每天")
    {
        inner_pre_mods.retain(|m| m != "正在");
        if front_modals.iter().any(|m| m == "聽說")
            && !post_verb_mods.iter().any(|m| m == "了")
            && !verb_core.is_empty()
        {
            for a in &mut adverbs {
                if a == "一定" {
                    *a = "一定會".to_owned();
                }
            }
        }
    }

    if obliques
        .iter()
        .any(|o| o.case_particle.as_deref() == Some("より"))
        && (verb_core.starts_with("變得") || verb_core.starts_with('變'))
    {
        let stem = verb_core
            .strip_prefix("變得")
            .or_else(|| verb_core.strip_prefix('變'))
            .unwrap_or(&verb_core);
        if stem == "忙碌" && obliques.iter().any(|o| o.text == "比平常") {
            verb_core = "更忙".to_owned();
        } else {
            verb_core = format!("更{stem}");
        }
        if ctx.sentence_has_work_return && verb_core == "更晚" {
            verb_core = "更晚下班".to_owned();
        }
    }
    if ctx.had_noni_in_sentence {
        for adv in &mut adverbs {
            if adv == "結果" {
                *adv = "結果反而".to_owned();
            }
        }
        for conn in &mut connectors {
            if conn == "結果" {
                *conn = "結果反而".to_owned();
            }
        }
    }
    if ctx.in_proportional_follow && verb_core.starts_with("容易") {
        verb_core = format!("就越{verb_core}");
    }

    let mut transfer_prefix = None;
    let mut transfer_recipient = None;
    if let Some(transfer) = transfer
        && subjects.len() <= 1
    {
        if transfer.auxiliary {
            let beneficiary = if transfer.kind == GivingReceivingKind::Receive {
                let receiver = subjects.pop();
                if let Some(actor) = transfer_counterpart {
                    subjects.push(actor);
                } else if receiver.is_some() {
                    subjects.push("有人".to_owned());
                }
                receiver
            } else {
                transfer_counterpart
            };
            match transfer.benefactive_core {
                BenefactiveCore::Show => {
                    if let Some(viewer) = beneficiary {
                        verb_core = "看".to_owned();
                        transfer_prefix = Some(format!("給{viewer}"));
                    } else {
                        verb_core = "展示".to_owned();
                    }
                }
                BenefactiveCore::Lend => {
                    if let Some(borrower) = beneficiary {
                        verb_core = "借給".to_owned();
                        transfer_recipient = Some(borrower);
                    } else {
                        verb_core = "出借".to_owned();
                    }
                }
                BenefactiveCore::Generic => {
                    transfer_prefix =
                        Some(beneficiary.map_or_else(|| "幫忙".to_owned(), |np| format!("幫{np}")));
                }
            }
        } else if transfer.kind == GivingReceivingKind::Receive {
            transfer_prefix = transfer_counterpart.map(|np| format!("從{np}"));
        } else {
            transfer_recipient = transfer_counterpart;
        }
    }

    // Ability takes the timed action as its complement. In particular,
    // failing to meet a deadline does not imply inability throughout the interval.
    let deadline_modal = deadline_action.and_then(|action| {
        let modal = match inner_pre_mods.as_slice() {
            [neg] if neg == "不" => "沒辦法",
            [neg] if neg == "沒有" => "沒能",
            [] if post_verb_mods.is_empty() => "可以",
            _ => return None,
        };
        inner_pre_mods.clear();
        verb_core = action;
        Some(modal)
    });

    // Assemble SVO order:
    let implicit_subject = subjects.is_empty();
    let mut parts: Vec<String> = Vec::new();
    if matches!(sconj_prefix, Some("一邊" | "一" | "明明")) {
        parts.extend(connectors);
        parts.extend(subjects);
        if let Some(sp) = sconj_prefix {
            parts.push(sp.to_owned());
        }
    } else {
        if let Some(sp) = sconj_prefix {
            parts.push(sp.to_owned());
        }
        parts.extend(connectors);
        parts.extend(subjects);
    }

    if (ctx.in_kuseni_follow || ctx.in_noni_follow) && sconj_prefix.is_none() {
        parts.push("卻".to_owned());
    }

    // This form can be an invitation or a negative question. Preserve both.
    let polite_negative_question = sfp_suffix == Some("嗎")
        && raw_clause
            .trim_end_matches(['。', '？', '?'])
            .ends_with("ませんか")
        && predicates
            .iter()
            .flat_map(|b| &b.atoms)
            .all(|a| !a.is_locked)
        && inner_pre_mods.iter().any(|m| m == "不");
    let negative_go = verb_core == "去"
        && inner_pre_mods
            .iter()
            .any(|m| matches!(m.as_str(), "不" | "沒有"));
    let offer = polite_negative_question
        && front_modals.is_empty()
        && joint_manner.is_some_and(|(_, invitation, _)| invitation);
    if offer {
        inner_pre_mods.retain(|m| m != "不");
        sfp_suffix = Some("？");
        if verb_core == "跑" && goal_tails.is_empty() && obliques.is_empty() && objs.is_empty() {
            verb_core = "跑步".to_owned();
        } else if verb_core == "哼唱" {
            verb_core = if objs.is_empty() { "唱歌" } else { "唱" }.to_owned();
        }
    } else if polite_negative_question
        && joint_manner.is_some_and(|(_, _, wh_question)| wh_question)
    {
        sfp_suffix = Some("？");
    } else if purpose.is_some() && sfp_suffix == Some("嗎") {
        if wh_argument {
            sfp_suffix = Some("？");
        } else if implicit_subject && polite_negative_question && negative_go {
            if let Some(negative) = inner_pre_mods.iter_mut().find(|m| *m == "不") {
                *negative = "要不要".to_owned();
            }
            sfp_suffix = Some("？");
        } else {
            sfp_suffix = Some("嗎？");
        }
    }
    // In 私は先生と二人で, the count includes both the subject and the
    // companion. Keep them together before any time, negation or ability.
    let counted_companion =
        matches!(joint_manner, Some((JointManner::Counted, ..))) && !joint_companions.is_empty();
    if counted_companion && !offer && !implicit_subject {
        parts.append(&mut joint_companions);
        parts.extend(joint_participants.take());
    }
    // Limit the added time ordering to the same endpoint rule as the motion.
    let has_spatial_made = (!negative_go || polite_negative_question)
        && !has_causative
        && !passive_agent_needed
        && objs.is_empty()
        && predicates_is_made_motion_verb(&predicates)
        && obliques.iter().any(|o| o.is_spatial_made);
    if has_spatial_made {
        let mut remaining = Vec::new();
        for adv in adverbs.drain(..) {
            if matches!(adv.as_str(), "每天早上" | "每天晚上" | "每晚") {
                temporal_others.push(adv);
            } else {
                remaining.push(adv);
            }
        }
        adverbs = remaining;
    }
    let has_front_modal = !front_modals.is_empty();
    let mut manner_adverbs = Vec::new();
    if !has_front_modal {
        parts.append(&mut temporal_others);
        if offer {
            parts.push("要不要".to_owned());
            if added_mo_adv && adverbs.first().is_some_and(|a| a == "也") {
                parts.push(adverbs.remove(0));
            }
            // 大家 names the subject. A number counts the whole group,
            // including the person in 誰かと二人で.
            if matches!(joint_manner, Some((JointManner::Everyone, ..))) {
                parts.extend(joint_participants.take());
            }
            parts.append(&mut joint_companions);
            parts.extend(joint_participants.take());
            parts.push("一起".to_owned());
            joint_manner.take();
        }
        for adv in adverbs {
            if matches!(
                adv.as_str(),
                "突然" | "結果" | "結果反而" | "終究" | "最後" | "總算"
            ) {
                parts.push(adv);
            } else {
                manner_adverbs.push(adv);
            }
        }
    } else {
        manner_adverbs = adverbs;
    }

    if passive_agent_needed && !iobjs.is_empty() {
        for io in iobjs.drain(..) {
            let trimmed = io
                .text
                .strip_prefix('對')
                .or_else(|| io.text.strip_prefix('給'))
                .unwrap_or(&io.text);
            parts.push(format!("被{trimmed}"));
        }
    } else if passive_agent_needed {
        parts.push("被".to_owned());
    }

    // Place front modals (`請不要` / `不要` / `請` / `讓`), attaching causative causee right after `讓`.
    parts.extend(front_modals);
    if has_causative && !passive_agent_needed && !iobjs.is_empty() {
        for io in iobjs.drain(..) {
            let trimmed = io
                .text
                .strip_prefix('對')
                .or_else(|| io.text.strip_prefix('給'))
                .unwrap_or(&io.text);
            parts.push(trimmed.to_owned());
        }
    } else if !passive_agent_needed && !iobjs.is_empty() {
        for io in iobjs.drain(..) {
            if verb_core == "不可或缺" {
                pre_iobjs.push(io.text);
            } else if verb_core.contains("道歉")
                || verb_core.contains("謝罪")
                || verb_core.contains("報告")
                || verb_core.contains("相談")
            {
                let rest = io.text.strip_prefix('對').unwrap_or(&io.text);
                pre_iobjs.push(format!("向{rest}"));
            } else if let Some(rest) = io.text.strip_prefix('對') {
                pre_iobjs.push(format!("給{rest}"));
            } else {
                pre_iobjs.push(io.text);
            }
        }
    }

    if has_front_modal {
        parts.extend(temporal_others);
    }
    parts.extend(travel_companions);
    // A place marked by まで is the endpoint of a motion event. Keep this
    // local to まで: the global goal flag also changes recipients and vehicles.
    let mut made_endpoints = Vec::new();
    let mut route_endpoints = Vec::new();
    let mut route_directions = Vec::new();
    let mut temporal_made_tails = Vec::new();
    let predicate_is_negative = inner_pre_mods
        .iter()
        .any(|m| matches!(m.as_str(), "不" | "沒有"));
    // A direction with へ renders as 往. Keep 前 and 往 from reading as 前往.
    let has_direction_he = obliques
        .iter()
        .any(|o| o.case_particle.as_deref() == Some("へ"));
    for o in obliques {
        let _ = o.is_instrument_de;
        if let Some(time) = o.temporal_made_time {
            if predicate_is_negative {
                // Put an "until" frame before the negated event and retain its negation.
                parts.push(format!("到{time}為止"));
            } else {
                // The endpoint follows the full activity, including a direct object.
                temporal_made_tails.push(format!("到{time}"));
            }
        } else if o.case_particle.as_deref() == Some("までに") {
            // Earliest completion is unrestricted. The stated time itself is
            // allowed. Keep tense, mood and polarity on the original predicate.
            if let Some((modal, additive)) =
                explicit_deadline.or(deadline_modal.map(|m| (m, false)))
            {
                let also = if additive { "也" } else { "" };
                if matches!(o.deadline_endpoint, DeadlineEndpoint::Question { .. })
                    && matches!(modal, "可以" | "得以")
                {
                    // 幾點前 followed by 往 can read as the departure time (前往).
                    parts.push(format!("最晚{}{also}{modal}", o.text));
                } else {
                    // 明天截止前 includes the deadline day; 明天前 can mean today.
                    let cutoff = match o.deadline_endpoint {
                        DeadlineEndpoint::Calendar if has_direction_he => "截止之前",
                        DeadlineEndpoint::Calendar => "截止前",
                        DeadlineEndpoint::ClockOrDaypart if has_direction_he => "之前",
                        DeadlineEndpoint::ClockOrDaypart => "前",
                        DeadlineEndpoint::Question { .. } => "之前",
                    };
                    parts.push(format!("{also}{modal}在{}{cutoff}", o.text));
                }
            } else if predicate_is_negative && !o.deadline_has_detached_negation {
                parts.push(format!("到{}為止", o.text));
            } else if matches!(
                o.deadline_endpoint,
                DeadlineEndpoint::Question { calendar: true }
            ) {
                parts.push(format!("最晚{}", o.text));
            } else {
                parts.push(format!("最晚在{}", o.text));
            }
        } else if route.is_some()
            && o.case_particle.as_deref() == Some("へ")
            && o.physical_goal_head.is_some()
            && o.physical_goal_head == motion_head
        {
            route_directions.push(o.text);
        } else if has_spatial_made && o.is_spatial_made {
            made_endpoints.push(o.text);
        } else if (route.is_some() || traversal_object)
            && o.case_particle.as_deref() == Some("まで")
            && o.physical_goal_head.is_some()
            && o.physical_goal_head == motion_head
        {
            route_endpoints.push(o.text);
        } else if o.case_particle.as_deref() == Some("まで") && !is_goal_verb {
            let raw_np = o.text.strip_prefix('到').unwrap_or(&o.text);
            parts.push(format!("連{raw_np}都"));
        } else if direct_negative_physical_go
            && o.case_particle.as_deref() == Some("へ")
            && o.physical_goal_head == negative_go_head
        {
            if let Some(place) = o.text.strip_prefix('往') {
                goal_tails.push(place.to_owned());
            } else {
                parts.push(o.text);
            }
        } else {
            parts.push(o.text);
        }
    }
    parts.extend(cause_prefixes);
    if let Some(modifier) = count_modifier {
        parts.push(modifier.to_owned());
    }
    if ctx.in_concessive_follow
        && !manner_adverbs.iter().any(|x| x == "都")
        && !inner_pre_mods.iter().any(|x| x == "都")
    {
        parts.push("都".to_owned());
    }
    parts.extend(manner_adverbs);
    parts.extend(pre_iobjs);
    parts.extend(others);
    let preserve_go = polite_negative_question || inner_pre_mods.iter().any(|m| m == "想");
    // A head count or みんな is the participant group. Negation scopes
    // over its activity, so keep the group before the negative or modal.
    if !counted_companion {
        parts.extend(joint_participants.take());
    }
    parts.extend(inner_pre_mods);
    // 遊べる and 飲める already carry ability in their Chinese lexical gloss.
    // Keep 不能 together, then put the shared activity before the action.
    if joint_manner.is_some() && !offer {
        for modal in ["可以", "能"] {
            if let Some(action) = verb_core.strip_prefix(modal)
                && !action.is_empty()
            {
                parts.push(modal.to_owned());
                verb_core = action.to_owned();
                break;
            }
        }
    }
    parts.extend(joint_companions);
    parts.extend(joint_participants);
    if joint_manner.is_some() {
        parts.push("一起".to_owned());
    }
    parts.extend(go_together);
    if route.is_some()
        && (verb_core == "進展"
            || (verb_core == "去" && goal_tails.is_empty() && made_endpoints.is_empty()))
    {
        verb_core = "前進".to_owned();
    }
    parts.extend(route);
    parts.extend(route_directions);
    parts.extend(transfer_prefix);
    if let Some(object) = affected_object {
        parts.push(format!("把{object}"));
    }
    parts.extend(setting_origin);
    // 到 already expresses the destination relation of bare 去. Other motion
    // verbs retain their manner: 走到, 跑到.
    let keep_go = has_spatial_made && preserve_go && verb_core == "去";
    if has_spatial_made && verb_core == "去" && !preserve_go {
        verb_core.clear();
    }
    if !verb_core.is_empty() {
        parts.push(verb_core);
    }
    if keep_go {
        for endpoint in &mut made_endpoints {
            if let Some(place) = endpoint.strip_prefix('到') {
                *endpoint = place.to_owned();
            }
        }
    }
    parts.extend(made_endpoints);
    parts.extend(result_value);
    let is_change_of_state_you =
        raw_clause.contains("ようになり") || raw_clause.contains("ようになった");
    let trailing_change_le =
        if (is_change_of_state_you || purpose.is_some() || !route_endpoints.is_empty())
            && post_verb_mods.last().is_some_and(|m| m == "了")
        {
            post_verb_mods.pop()
        } else {
            None
        };
    // A temporal continuation endpoint already bounds the activity. Omit the
    // ordinary perfective marker rather than break up the activity and endpoint.
    if !temporal_made_tails.is_empty() {
        post_verb_mods.retain(|modifier| modifier != "了");
    }
    if !goal_tails.is_empty() {
        parts.extend(goal_tails);
        parts.append(&mut post_verb_mods);
    } else {
        parts.extend(post_verb_mods);
    }
    parts.extend(purpose_partners);
    if let Some(lemma) = purpose {
        // Bare transitive purposes use the ordinary Mandarin verb before the object.
        let action = match (lemma.as_str(), objs.is_empty()) {
            ("借りる", false) => "借".to_owned(),
            ("探す", false) => "找".to_owned(),
            ("迎える", false) => "迎接".to_owned(),
            _ => {
                let action = translate_single_ja_term(&lemma, Some(&lemma), Some(Upos::Verb));
                if objs.is_empty() {
                    action
                } else {
                    purpose_transfer_action(&action)
                        .unwrap_or(&action)
                        .to_owned()
                }
            }
        };
        parts.push(action);
    }
    parts.extend(transfer_recipient);
    parts.extend(event_counts);
    parts.extend(counted_arrival);
    parts.extend(objs);
    parts.extend(purpose_recipients);
    parts.extend(route_endpoints);
    parts.extend(temporal_made_tails);
    if let Some(le) = trailing_change_le {
        parts.push(le);
    }
    if let Some(sfp) = sfp_suffix {
        parts.push(sfp.to_owned());
    }
    if let Some(ss) = sconj_suffix {
        parts.push(ss.to_owned());
    }

    (join_zh_parts(&parts), had_overt_subj_no_pred)
}

fn format_goal_tail(verb_core: &str, raw_np: &str, is_building_floor: bool) -> String {
    if verb_core == "進入" || raw_np.starts_with('到') || (is_building_floor && verb_core == "去")
    {
        raw_np.to_owned()
    } else {
        format!("到{raw_np}")
    }
}

/// Recover a kana noun whose internal か was tagged as a discourse particle.
/// Both forms require a case marker governed by the noun and a clear left
/// boundary. A predicate followed by an embedded か, or two nouns linked by
/// a choice particle, has different dependencies.
fn recover_split_kana_noun(atoms: &[Atom], start: usize) -> Option<(Atom, usize)> {
    let first = &atoms[start];
    let is_ka = |a: &Atom| a.upos == Some(Upos::Part) && a.surface == "か";
    let (head, parts, case) = match &atoms[start..] {
        [ka, noun, case, ..] if is_ka(ka) && noun.upos == Some(Upos::Noun) => (noun, 2, case),
        [noun, ka, suffix, case, ..]
            if is_ka(ka)
                && matches!(suffix.upos, Some(Upos::Aux | Upos::Part))
                && ka.head == noun.doc_tok_idx
                && suffix.head == noun.doc_tok_idx =>
        {
            (noun, 3, case)
        }
        _ => return None,
    };
    if case.is_locked
        || case.upos != Some(Upos::Adp)
        || case.deprel != "case"
        || head.doc_tok_idx.is_none()
        || head.doc_tok_idx != case.head
        || !matches!(head.deprel.as_str(), "obj" | "nsubj" | "obl" | "nmod")
        || start.checked_sub(1).is_some_and(|i| {
            let before = &atoms[i];
            before.span.end_char == first.span.start_char
                && (before.is_locked || !(before.deprel == "case" || is_punct_atom(before)))
        })
    {
        return None;
    }
    let span = &atoms[start..start + parts];
    if span
        .iter()
        .any(|a| a.is_locked || !a.surface.chars().all(|ch| matches!(ch, 'ぁ'..='ゖ')))
        || atoms[start..=start + parts]
            .windows(2)
            .any(|p| p[0].span.end_char != p[1].span.start_char)
    {
        return None;
    }
    // Keep the Japanese spelling and original offsets. Let normal noun
    // rendering use the dictionary or retain the entire unresolved word.
    let mut noun = head.clone();
    noun.surface = span.iter().map(|a| a.surface.as_str()).collect();
    noun.lemma = None;
    noun.upos = Some(Upos::Noun);
    noun.span.start_char = first.span.start_char;
    noun.span.end_char = case.span.start_char;
    Some((noun, parts))
}

fn normalize_ja_atoms(atoms: &[Atom]) -> Vec<Atom> {
    let mut out = Vec::with_capacity(atoms.len());
    let mut i = 0usize;
    while i < atoms.len() {
        let a = &atoms[i];

        if let Some((noun, used)) = recover_split_kana_noun(atoms, i) {
            out.push(noun);
            i += used;
            continue;
        }

        // A. Chained coordinate `・` between nouns/proper nouns (`政府・日銀`, `A・B・C`).
        if !a.is_locked
            && matches!(a.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
            && atoms
                .get(i + 1)
                .is_some_and(|nx| !nx.is_locked && nx.surface == "・")
            && atoms.get(i + 2).is_some_and(|nx2| {
                !nx2.is_locked && matches!(nx2.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
            })
        {
            let mut end_idx = i;
            let mut parts = vec![translate_single_ja_term(
                &a.surface,
                a.lemma.as_deref(),
                a.upos,
            )];
            while atoms
                .get(end_idx + 1)
                .is_some_and(|nx| !nx.is_locked && nx.surface == "・")
                && let Some(nx2) = atoms.get(end_idx + 2)
                && !nx2.is_locked
                && matches!(nx2.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
            {
                parts.push(translate_single_ja_term(
                    &nx2.surface,
                    nx2.lemma.as_deref(),
                    nx2.upos,
                ));
                end_idx += 2;
            }
            let last = &atoms[end_idx];
            let zh = parts.join("、");
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: last.span.end_char,
                },
                surface: zh.clone(),
                lemma: Some(zh),
                upos: Some(Upos::Noun),
                doc_tok_idx: last.doc_tok_idx,
                head: last.head,
                deprel: last.deprel.clone(),
                is_locked: true,
            });
            i = end_idx + 1;
            continue;
        }

        // B. Apposition `[Noun|Propn|Pron, "すなわち"|"つまり", Noun|Propn|Pron]`.
        if !a.is_locked
            && matches!(a.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
            && let Some(mid) = atoms.get(i + 1)
            && !mid.is_locked
            && matches!(mid.surface.as_str(), "すなわち" | "つまり")
            && let Some(right) = atoms.get(i + 2)
            && !right.is_locked
            && matches!(right.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
        {
            let l_zh = translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos);
            let r_zh = translate_single_ja_term(&right.surface, right.lemma.as_deref(), right.upos);
            let zh = format!("{l_zh}亦即{r_zh}");
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: right.span.end_char,
                },
                surface: zh.clone(),
                lemma: Some(zh),
                upos: Some(Upos::Noun),
                doc_tok_idx: right.doc_tok_idx,
                head: right.head,
                deprel: right.deprel.clone(),
                is_locked: true,
            });
            i += 3;
            continue;
        }

        // C. `["これまで", "の"]` -> locked Det `"以往的"`.
        if !a.is_locked
            && a.surface == "これまで"
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
            && next.surface == "の"
        {
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: next.span.end_char,
                },
                surface: "以往的".to_owned(),
                lemma: Some("以往的".to_owned()),
                upos: Some(Upos::Det),
                doc_tok_idx: next.doc_tok_idx,
                head: next.head,
                deprel: next.deprel.clone(),
                is_locked: true,
            });
            i += 2;
            continue;
        }

        // D. `["欠か", "せ", "ず"|"ない"|"なかった"]` / `["欠かせ", "ず"|"ない"|"なかった"]` -> `"不可或缺"`.
        if !a.is_locked
            && a.surface == "欠か"
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
            && next.surface == "せ"
            && let Some(third) = atoms.get(i + 2)
            && !third.is_locked
            && matches!(third.surface.as_str(), "ず" | "ない" | "なかった")
        {
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: third.span.end_char,
                },
                surface: "不可或缺".to_owned(),
                lemma: Some("不可或缺".to_owned()),
                upos: Some(Upos::Adj),
                doc_tok_idx: third.doc_tok_idx,
                head: third.head,
                deprel: third.deprel.clone(),
                is_locked: true,
            });
            if third.surface == "なかった" {
                out.push(Atom {
                    span: third.span,
                    surface: "た".to_owned(),
                    lemma: Some("た".to_owned()),
                    upos: Some(Upos::Aux),
                    doc_tok_idx: None,
                    head: None,
                    deprel: String::new(),
                    is_locked: false,
                });
            }
            i += 3;
            continue;
        }
        if !a.is_locked
            && a.surface == "欠かせ"
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
            && matches!(next.surface.as_str(), "ず" | "ない" | "なかった")
        {
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: next.span.end_char,
                },
                surface: "不可或缺".to_owned(),
                lemma: Some("不可或缺".to_owned()),
                upos: Some(Upos::Adj),
                doc_tok_idx: next.doc_tok_idx,
                head: next.head,
                deprel: next.deprel.clone(),
                is_locked: true,
            });
            if next.surface == "なかった" {
                out.push(Atom {
                    span: next.span,
                    surface: "た".to_owned(),
                    lemma: Some("た".to_owned()),
                    upos: Some(Upos::Aux),
                    doc_tok_idx: None,
                    head: None,
                    deprel: String::new(),
                    is_locked: false,
                });
            }
            i += 2;
            continue;
        }

        // E. Defensive `["に", "よっ", "て"]` -> ADP `"によって"` and `["に", "よる"]` -> ADP `"による"`.
        if !a.is_locked
            && a.surface == "に"
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
        {
            if next.surface == "よっ"
                && let Some(third) = atoms.get(i + 2)
                && !third.is_locked
                && third.surface == "て"
            {
                out.push(Atom {
                    span: Span {
                        start_char: a.span.start_char,
                        end_char: third.span.end_char,
                    },
                    surface: "によって".to_owned(),
                    lemma: Some("によって".to_owned()),
                    upos: Some(Upos::Adp),
                    doc_tok_idx: third.doc_tok_idx,
                    head: third.head,
                    deprel: third.deprel.clone(),
                    is_locked: false,
                });
                i += 3;
                continue;
            }
            if next.surface == "よる" {
                out.push(Atom {
                    span: Span {
                        start_char: a.span.start_char,
                        end_char: next.span.end_char,
                    },
                    surface: "による".to_owned(),
                    lemma: Some("による".to_owned()),
                    upos: Some(Upos::Adp),
                    doc_tok_idx: next.doc_tok_idx,
                    head: next.head,
                    deprel: next.deprel.clone(),
                    is_locked: false,
                });
                i += 2;
                continue;
            }
        }

        // 1. Defensive `[Noun/Verb] + さ + れ/れる/れた`: avoid `さ` -> `啊` leakage.
        if !a.is_locked
            && a.surface == "さ"
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
            && matches!(next.surface.as_str(), "れ" | "れる" | "れた" | "れない")
        {
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: next.span.end_char,
                },
                surface: format!("さ{}", next.surface),
                lemma: Some("される".to_owned()),
                upos: Some(Upos::Aux),
                doc_tok_idx: next.doc_tok_idx,
                head: next.head,
                deprel: next.deprel.clone(),
                is_locked: false,
            });
            i += 2;
            continue;
        }

        // 1b. Trailing explanatory `["の"|"ん", "だ"|"です"|"だった"|"でした"]` after predicate,
        // or `["の"|"ん", "だけど"|"だから"|"なのに"|"なら"]`.
        if !a.is_locked
            && matches!(a.surface.as_str(), "の" | "ん")
            && out.last().is_some_and(|prev| {
                prev.is_locked || matches!(prev.upos, Some(Upos::Verb | Upos::Adj | Upos::Aux))
            })
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
        {
            if matches!(
                next.surface.as_str(),
                "だけど" | "だから" | "なのに" | "なら"
            ) {
                i += 1;
                continue;
            }
            if matches!(next.surface.as_str(), "だ" | "です" | "だった" | "でした")
                && (i + 2 >= atoms.len()
                    || atoms[i + 2..].iter().all(|later| {
                        is_punct_atom(later)
                            || (!later.is_locked
                                && matches!(
                                    later.surface.as_str(),
                                    "よ" | "ね" | "よね" | "か" | "が" | "けど"
                                ))
                    }))
            {
                if matches!(next.surface.as_str(), "だった" | "でした") {
                    out.push(Atom {
                        span: next.span,
                        surface: "た".to_owned(),
                        lemma: Some("た".to_owned()),
                        upos: Some(Upos::Aux),
                        doc_tok_idx: None,
                        head: None,
                        deprel: String::new(),
                        is_locked: false,
                    });
                }
                i += 2;
                continue;
            }
        }

        // 1b-ext. Post-predicate `だけど` stays attached to predicate bunsetsu as SCONJ.
        if !a.is_locked
            && a.surface == "だけど"
            && out.last().is_some_and(|prev| {
                prev.is_locked || matches!(prev.upos, Some(Upos::Verb | Upos::Adj | Upos::Aux))
            })
        {
            let mut sconj_atom = a.clone();
            sconj_atom.upos = Some(Upos::Sconj);
            out.push(sconj_atom);
            i += 1;
            continue;
        }

        // 1c. Partial negation tail `わけではない` / `わけでもない` after predicate.
        if !a.is_locked
            && a.surface == "わけ"
            && out.last().is_some_and(|prev| {
                prev.is_locked || matches!(prev.upos, Some(Upos::Verb | Upos::Adj | Upos::Aux))
            })
        {
            let tail_matches = |pat: &[&str]| -> bool {
                atoms.len() >= i + pat.len()
                    && atoms[i..i + pat.len()]
                        .iter()
                        .zip(pat.iter())
                        .all(|(at, p)| !at.is_locked && at.surface == *p)
            };
            let matched = if tail_matches(&["わけ", "で", "は", "あり", "ませ", "ん"]) {
                Some(("わけではない", 6usize))
            } else if tail_matches(&["わけ", "で", "も", "あり", "ませ", "ん"]) {
                Some(("わけでもない", 6usize))
            } else if tail_matches(&["わけ", "でも", "あり", "ませ", "ん"]) {
                Some(("わけでもない", 5usize))
            } else if tail_matches(&["わけ", "で", "は", "ありません"])
                || tail_matches(&["わけ", "で", "は", "ない"])
            {
                Some(("わけではない", 4usize))
            } else if tail_matches(&["わけ", "で", "も", "ありません"])
                || tail_matches(&["わけ", "で", "も", "ない"])
            {
                Some(("わけでもない", 4usize))
            } else if tail_matches(&["わけ", "じゃ", "ない"]) {
                Some(("わけではない", 3usize))
            } else if tail_matches(&["わけ", "でも", "ありません"])
                || tail_matches(&["わけ", "でも", "ない"])
            {
                Some(("わけでもない", 3usize))
            } else {
                None
            };
            if let Some((surf, count)) = matched {
                let last = &atoms[i + count - 1];
                out.push(Atom {
                    span: Span {
                        start_char: a.span.start_char,
                        end_char: last.span.end_char,
                    },
                    surface: surf.to_owned(),
                    lemma: Some(surf.to_owned()),
                    upos: Some(Upos::Aux),
                    doc_tok_idx: last.doc_tok_idx,
                    head: last.head,
                    deprel: last.deprel.clone(),
                    is_locked: false,
                });
                i += count;
                continue;
            }
        }

        // A complete negative copula is an auxiliary, not locative `で + は`.
        // Leave case particles intact unless the negative tail is adjacent.
        let copula_prefix = if !a.is_locked
            && a.surface == "で"
            && a.upos == Some(Upos::Adp)
            && atoms
                .get(i + 1)
                .is_some_and(|ha| !ha.is_locked && ha.surface == "は" && ha.upos == Some(Upos::Adp))
        {
            2
        } else if !a.is_locked && a.surface == "じゃ" && a.upos == Some(Upos::Sconj) {
            1
        } else {
            0
        };
        if copula_prefix > 0 {
            let start = i + copula_prefix;
            let negative_len = if atoms.get(start).is_some_and(|neg| {
                !neg.is_locked
                    && neg.surface == "ない"
                    && matches!(neg.upos, Some(Upos::Adj | Upos::Aux))
            }) {
                1
            } else if atoms.get(start..start + 3).is_some_and(|tail| {
                tail.iter().all(|a| !a.is_locked)
                    && tail[0].upos == Some(Upos::Verb)
                    && tail[0].lemma.as_deref().unwrap_or(&tail[0].surface) == "ある"
                    && tail[1].upos == Some(Upos::Aux)
                    && tail[1].surface == "ませ"
                    && tail[2].upos == Some(Upos::Aux)
                    && tail[2].surface == "ん"
            }) {
                3
            } else {
                0
            };
            let mut consumed = copula_prefix + negative_len;
            let past_start = i + consumed;
            if negative_len == 3
                && atoms.get(past_start..past_start + 2).is_some_and(|past| {
                    !past[0].is_locked
                        && past[0].surface == "でし"
                        && past[0].upos == Some(Upos::Aux)
                        && !past[1].is_locked
                        && past[1].surface == "た"
                        && past[1].upos == Some(Upos::Aux)
                })
            {
                consumed += 2;
            }
            if negative_len > 0
                && atoms[i..i + consumed]
                    .windows(2)
                    .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
            {
                let last = &atoms[i + consumed - 1];
                out.push(Atom {
                    span: Span {
                        start_char: a.span.start_char,
                        end_char: last.span.end_char,
                    },
                    surface: "ではない".to_owned(),
                    lemma: Some("ではない".to_owned()),
                    upos: Some(Upos::Aux),
                    doc_tok_idx: last.doc_tok_idx,
                    head: last.head,
                    deprel: last.deprel.clone(),
                    is_locked: false,
                });
                i += consumed;
                continue;
            }
        }

        // 1d. Post-predicate concessive `くせに` and hypothetical concessive `としても`.
        if !a.is_locked
            && a.surface == "くせ"
            && out.last().is_some_and(|prev| {
                prev.is_locked || matches!(prev.upos, Some(Upos::Verb | Upos::Adj | Upos::Aux))
            })
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
            && next.surface == "に"
        {
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: next.span.end_char,
                },
                surface: "くせに".to_owned(),
                lemma: Some("くせに".to_owned()),
                upos: Some(Upos::Sconj),
                doc_tok_idx: next.doc_tok_idx,
                head: next.head,
                deprel: next.deprel.clone(),
                is_locked: false,
            });
            i += 2;
            continue;
        }
        if !a.is_locked
            && a.surface == "として"
            && out.last().is_some_and(|prev| {
                prev.is_locked || matches!(prev.upos, Some(Upos::Verb | Upos::Adj | Upos::Aux))
            })
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
            && next.surface == "も"
        {
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: next.span.end_char,
                },
                surface: "としても".to_owned(),
                lemma: Some("としても".to_owned()),
                upos: Some(Upos::Sconj),
                doc_tok_idx: next.doc_tok_idx,
                head: next.head,
                deprel: next.deprel.clone(),
                is_locked: false,
            });
            i += 2;
            continue;
        }

        // 1e. Polite `お` prefix fallback before an unlisted noun.
        if !a.is_locked
            && a.surface == "お"
            && a.upos == Some(Upos::Intj)
            && atoms
                .get(i + 1)
                .is_some_and(|nx| !nx.is_locked && nx.upos == Some(Upos::Noun))
        {
            i += 1;
            continue;
        }

        // 1f. `[na-Adj, "に", "なれる"|"なる"|"なり"|"なっ"|"なった"]` -> `能變得{adj}` / `變得{adj}`.
        if !a.is_locked
            && a.upos == Some(Upos::Adj)
            && !a.surface.ends_with('い')
            && !a.surface.ends_with('く')
            && let Some(ni) = atoms.get(i + 1)
            && !ni.is_locked
            && ni.surface == "に"
            && let Some(naru) = atoms.get(i + 2)
            && !naru.is_locked
            && matches!(
                naru.surface.as_str(),
                "なれる" | "なれ" | "なる" | "なり" | "なっ" | "なった"
            )
        {
            let adj_zh = translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos);
            let zh = if matches!(naru.surface.as_str(), "なれる" | "なれ") {
                format!("能變得{adj_zh}")
            } else {
                format!("變得{adj_zh}")
            };
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: naru.span.end_char,
                },
                surface: zh,
                lemma: Some("なる_adj".to_owned()),
                upos: Some(Upos::Verb),
                doc_tok_idx: naru.doc_tok_idx,
                head: naru.head,
                deprel: naru.deprel.clone(),
                is_locked: true,
            });
            if naru.surface == "なった" {
                out.push(Atom {
                    span: naru.span,
                    surface: "た".to_owned(),
                    lemma: Some("た".to_owned()),
                    upos: Some(Upos::Aux),
                    doc_tok_idx: None,
                    head: None,
                    deprel: String::new(),
                    is_locked: false,
                });
            }
            i += 3;
            continue;
        }

        // 2. `[Adj stem] + そう (+ に / な)`: `恥ずかし + そう + に` -> `害羞地`.
        if !a.is_locked
            && a.upos == Some(Upos::Adj)
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
            && next.surface == "そう"
        {
            let dict_form = a
                .lemma
                .clone()
                .filter(|l| l.ends_with('い'))
                .unwrap_or_else(|| format!("{}い", a.surface));
            let adj_zh = translate_single_ja_term(&dict_form, Some(&dict_form), Some(Upos::Adj));
            if let Some(third) = atoms.get(i + 2)
                && !third.is_locked
                && matches!(third.surface.as_str(), "に" | "な")
            {
                let (zh, upos) = if third.surface == "に" {
                    (format!("{adj_zh}地"), Some(Upos::Adv))
                } else {
                    (format!("{adj_zh}的"), Some(Upos::Adj))
                };
                out.push(Atom {
                    span: Span {
                        start_char: a.span.start_char,
                        end_char: third.span.end_char,
                    },
                    surface: zh.clone(),
                    lemma: Some(zh),
                    upos,
                    doc_tok_idx: third.doc_tok_idx,
                    head: third.head,
                    deprel: third.deprel.clone(),
                    is_locked: true,
                });
                i += 3;
                continue;
            }
            let zh = format!("看起來很{adj_zh}");
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: next.span.end_char,
                },
                surface: zh.clone(),
                lemma: Some(zh),
                upos: Some(Upos::Adj),
                doc_tok_idx: next.doc_tok_idx,
                head: next.head,
                deprel: next.deprel.clone(),
                is_locked: true,
            });
            i += 2;
            continue;
        }

        // 3. `"そんな" + "に"` -> Adv `"那樣"`.
        if !a.is_locked
            && a.surface == "そんな"
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
            && next.surface == "に"
        {
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: next.span.end_char,
                },
                surface: "那樣".to_owned(),
                lemma: Some("那樣".to_owned()),
                upos: Some(Upos::Adv),
                doc_tok_idx: next.doc_tok_idx,
                head: next.head,
                deprel: next.deprel.clone(),
                is_locked: true,
            });
            i += 2;
            continue;
        }

        // 3b. `[Verb] + ["よう", "に", "なる"|"なり"|"なっ"|"なった"]`: skip `ようになり` wrapper.
        if !a.is_locked
            && a.surface == "よう"
            && out
                .last()
                .is_some_and(|prev| matches!(prev.upos, Some(Upos::Verb | Upos::Aux)))
            && let Some(ni) = atoms.get(i + 1)
            && !ni.is_locked
            && ni.surface == "に"
            && let Some(naru) = atoms.get(i + 2)
            && !naru.is_locked
            && matches!(naru.surface.as_str(), "なる" | "なり" | "なっ" | "なった")
        {
            if naru.surface == "なった" {
                out.push(Atom {
                    span: naru.span,
                    surface: "た".to_owned(),
                    lemma: Some("た".to_owned()),
                    upos: Some(Upos::Aux),
                    doc_tok_idx: None,
                    head: None,
                    deprel: String::new(),
                    is_locked: false,
                });
            }
            i += 3;
            continue;
        }

        // 3c. `[Adj-く, "なる"|"なり"|"なっ"|"なった"]` -> locked Verb `"變得{adj_zh}"` (`lemma = "なる_adj"`).
        if !a.is_locked
            && a.upos == Some(Upos::Adj)
            && a.surface.ends_with('く')
            && a.surface != "いく"
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
            && matches!(next.surface.as_str(), "なる" | "なり" | "なっ" | "なった")
        {
            let dict_form = a.lemma.clone().unwrap_or_else(|| {
                let stem = a.surface.strip_suffix('く').unwrap_or(&a.surface);
                format!("{stem}い")
            });
            let adj_zh = if dict_form == "遅い"
                && atoms.iter().any(|x| {
                    matches!(
                        x.surface.as_str(),
                        "いつも" | "普段" | "より" | "時間" | "今日"
                    ) || x.surface.starts_with('帰')
                }) {
                "晚".to_owned()
            } else {
                translate_single_ja_term(&dict_form, Some(&dict_form), Some(Upos::Adj))
            };
            let zh = if adj_zh.chars().count() == 1 {
                format!("變{adj_zh}")
            } else {
                format!("變得{adj_zh}")
            };
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: next.span.end_char,
                },
                surface: zh,
                lemma: Some("なる_adj".to_owned()),
                upos: Some(Upos::Verb),
                doc_tok_idx: next.doc_tok_idx,
                head: next.head,
                deprel: next.deprel.clone(),
                is_locked: true,
            });
            if next.surface == "なった" {
                out.push(Atom {
                    span: next.span,
                    surface: "た".to_owned(),
                    lemma: Some("た".to_owned()),
                    upos: Some(Upos::Aux),
                    doc_tok_idx: None,
                    head: None,
                    deprel: String::new(),
                    is_locked: false,
                });
            }
            i += 2;
            continue;
        }

        // 4. Adverbial `-く` Adj (`優しく`, `深く`, `激しく`) before a predicate.
        let next_blocks_adv = atoms.get(i + 1).is_some_and(|nx| {
            !nx.is_locked
                && matches!(
                    nx.surface.as_str(),
                    "ない"
                        | "なかっ"
                        | "なかった"
                        | "て"
                        | "ても"
                        | "なる"
                        | "なり"
                        | "なっ"
                        | "なった"
                        | "する"
                        | "し"
                        | "した"
                )
        });
        let pred_follows = atoms[i + 1..]
            .iter()
            .any(|later| !later.is_locked && matches!(later.upos, Some(Upos::Verb | Upos::Adj)));
        if !a.is_locked
            && a.upos == Some(Upos::Adj)
            && a.surface.ends_with('く')
            && a.surface != "いく"
            && !next_blocks_adv
            && pred_follows
        {
            let dict_form = a.lemma.clone().unwrap_or_else(|| {
                let stem = a.surface.strip_suffix('く').unwrap_or(&a.surface);
                format!("{stem}い")
            });
            let early_time_verb = dict_form == "早い"
                && atoms[i + 1..].iter().any(|later| {
                    !later.is_locked
                        && (matches!(
                            later.lemma.as_deref(),
                            Some("来る" | "行く" | "帰る" | "起きる" | "寝る")
                        ) || matches!(
                            later.surface.as_str(),
                            "来る"
                                | "来"
                                | "行く"
                                | "行っ"
                                | "帰る"
                                | "帰り"
                                | "起きる"
                                | "起き"
                                | "寝る"
                                | "寝"
                        ))
                });
            let adv_zh = if early_time_verb {
                "早點".to_owned()
            } else {
                let adj_zh =
                    translate_single_ja_term(&dict_form, Some(&dict_form), Some(Upos::Adj));
                if adj_zh.chars().count() == 1 {
                    format!("{adj_zh}{adj_zh}地")
                } else if adj_zh.ends_with('地') {
                    adj_zh
                } else {
                    format!("{adj_zh}地")
                }
            };
            out.push(Atom {
                span: a.span,
                surface: adv_zh.clone(),
                lemma: Some(adv_zh),
                upos: Some(Upos::Adv),
                doc_tok_idx: a.doc_tok_idx,
                head: a.head,
                deprel: a.deprel.clone(),
                is_locked: true,
            });
            i += 1;
            continue;
        }

        // 5a. Keigo humble `["ご"|"お", VN, "いたし"|"いたす"|"致し"|"致す", ("まし", "た" | "た")?]` -> `"向您{vn_zh}"` (+ `"た"`).
        if !a.is_locked
            && matches!(a.surface.as_str(), "ご" | "お")
            && let Some(vn) = atoms.get(i + 1)
            && !vn.is_locked
            && matches!(vn.upos, Some(Upos::Noun | Upos::Verb))
            && let Some(ita) = atoms.get(i + 2)
            && !ita.is_locked
            && matches!(ita.surface.as_str(), "いたし" | "いたす" | "致し" | "致す")
        {
            let vn_zh =
                translate_single_ja_term(&vn.surface, vn.lemma.as_deref(), Some(Upos::Noun));
            let (consumed, emit_past) = if atoms
                .get(i + 3)
                .is_some_and(|x| !x.is_locked && x.surface == "まし")
                && atoms
                    .get(i + 4)
                    .is_some_and(|x| !x.is_locked && x.surface == "た")
            {
                (5usize, true)
            } else if atoms
                .get(i + 3)
                .is_some_and(|x| !x.is_locked && x.surface == "た")
            {
                (4usize, true)
            } else {
                (3usize, false)
            };
            let v_zh = format!("向您{vn_zh}");
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: ita.span.end_char,
                },
                surface: v_zh.clone(),
                lemma: Some(v_zh),
                upos: Some(Upos::Verb),
                doc_tok_idx: ita.doc_tok_idx,
                head: ita.head,
                deprel: ita.deprel.clone(),
                is_locked: true,
            });
            if emit_past {
                let ta_span = atoms[i + consumed - 1].span;
                out.push(Atom {
                    span: ta_span,
                    surface: "た".to_owned(),
                    lemma: Some("た".to_owned()),
                    upos: Some(Upos::Aux),
                    doc_tok_idx: None,
                    head: None,
                    deprel: String::new(),
                    is_locked: false,
                });
            }
            i += consumed;
            continue;
        }

        // 5b. `[Noun, "そのもの", "の", "優劣"]` and `[Noun, "そのもの"]`.
        if !a.is_locked
            && a.upos == Some(Upos::Noun)
            && let Some(s1) = atoms.get(i + 1)
            && !s1.is_locked
            && s1.surface == "そのもの"
        {
            let n_zh = translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos);
            if let Some(s2) = atoms.get(i + 2)
                && !s2.is_locked
                && s2.surface == "の"
                && let Some(s3) = atoms.get(i + 3)
                && !s3.is_locked
                && s3.surface == "優劣"
            {
                let zh = format!("{n_zh}本身孰優孰劣");
                out.push(Atom {
                    span: Span {
                        start_char: a.span.start_char,
                        end_char: s3.span.end_char,
                    },
                    surface: zh.clone(),
                    lemma: Some(zh),
                    upos: Some(Upos::Noun),
                    doc_tok_idx: a.doc_tok_idx,
                    head: a.head,
                    deprel: a.deprel.clone(),
                    is_locked: true,
                });
                i += 4;
                continue;
            }
            let zh = format!("{n_zh}本身");
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: s1.span.end_char,
                },
                surface: zh.clone(),
                lemma: Some(zh),
                upos: Some(Upos::Noun),
                doc_tok_idx: a.doc_tok_idx,
                head: a.head,
                deprel: a.deprel.clone(),
                is_locked: true,
            });
            i += 2;
            continue;
        }

        // 5c. `[Noun|Adv, "的", "に"|"な"]` (`全面的に`->`完全`, `意図的に`->`刻意`, `長期的な`->`長期的`).
        if !a.is_locked
            && matches!(a.upos, Some(Upos::Noun | Upos::Adv))
            && !matches!(a.surface.as_str(), "私" | "彼" | "彼女" | "あなた")
            && let Some(teki) = atoms.get(i + 1)
            && !teki.is_locked
            && teki.surface == "的"
            && let Some(tail) = atoms.get(i + 2)
            && !tail.is_locked
            && matches!(tail.surface.as_str(), "に" | "な")
        {
            let base_zh = translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos);
            if tail.surface == "に" {
                let zh = if a.surface == "全面" {
                    "完全".to_owned()
                } else if a.surface == "意図" {
                    "刻意".to_owned()
                } else if base_zh.ends_with('地') {
                    base_zh
                } else {
                    format!("{base_zh}地")
                };
                out.push(Atom {
                    span: Span {
                        start_char: a.span.start_char,
                        end_char: tail.span.end_char,
                    },
                    surface: zh.clone(),
                    lemma: Some(zh),
                    upos: Some(Upos::Adv),
                    doc_tok_idx: a.doc_tok_idx,
                    head: a.head,
                    deprel: a.deprel.clone(),
                    is_locked: true,
                });
            } else {
                let zh = if base_zh.ends_with('的') {
                    base_zh
                } else {
                    format!("{base_zh}的")
                };
                out.push(Atom {
                    span: Span {
                        start_char: a.span.start_char,
                        end_char: tail.span.end_char,
                    },
                    surface: zh.clone(),
                    lemma: Some(zh),
                    upos: Some(Upos::Adj),
                    doc_tok_idx: a.doc_tok_idx,
                    head: a.head,
                    deprel: a.deprel.clone(),
                    is_locked: true,
                });
            }
            i += 3;
            continue;
        }

        // 5d. `[Noun, "どおり"|"通り"]` (`予定どおり`->`如期`, else `按照{N}`).
        if !a.is_locked
            && a.upos == Some(Upos::Noun)
            && let Some(next) = atoms.get(i + 1)
            && !next.is_locked
            && matches!(next.surface.as_str(), "どおり" | "通り")
        {
            let zh = if a.surface == "予定" {
                "如期".to_owned()
            } else {
                let n_zh = translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos);
                format!("按照{n_zh}")
            };
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: next.span.end_char,
                },
                surface: zh.clone(),
                lemma: Some(zh),
                upos: Some(Upos::Adv),
                doc_tok_idx: a.doc_tok_idx,
                head: a.head,
                deprel: a.deprel.clone(),
                is_locked: true,
            });
            i += 2;
            continue;
        }

        // 5e. `["その", Adj, "さ"]` -> `"其珍貴"`/`"其{adj}"`, and `[Adj, "さ"]` -> `"珍貴"`/`"{adj}"`.
        if !a.is_locked
            && a.surface == "その"
            && let Some(adj) = atoms.get(i + 1)
            && !adj.is_locked
            && adj.upos == Some(Upos::Adj)
            && let Some(sa) = atoms.get(i + 2)
            && !sa.is_locked
            && sa.surface == "さ"
        {
            let zh = if adj.surface == "大切" {
                "其珍貴".to_owned()
            } else {
                let adj_zh =
                    translate_single_ja_term(&adj.surface, adj.lemma.as_deref(), Some(Upos::Adj));
                format!("其{adj_zh}")
            };
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: sa.span.end_char,
                },
                surface: zh.clone(),
                lemma: Some(zh),
                upos: Some(Upos::Noun),
                doc_tok_idx: adj.doc_tok_idx,
                head: adj.head,
                deprel: adj.deprel.clone(),
                is_locked: true,
            });
            i += 3;
            continue;
        }
        if !a.is_locked
            && a.upos == Some(Upos::Adj)
            && let Some(sa) = atoms.get(i + 1)
            && !sa.is_locked
            && sa.surface == "さ"
        {
            let zh = if a.surface == "大切" {
                "珍貴".to_owned()
            } else {
                translate_single_ja_term(&a.surface, a.lemma.as_deref(), Some(Upos::Adj))
            };
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: sa.span.end_char,
                },
                surface: zh.clone(),
                lemma: Some(zh),
                upos: Some(Upos::Noun),
                doc_tok_idx: a.doc_tok_idx,
                head: a.head,
                deprel: a.deprel.clone(),
                is_locked: true,
            });
            i += 2;
            continue;
        }

        // 5f. Contextual `"件"` (preceded by Verb/Aux/Det/`の`, not Num) -> `"事項"`.
        if !a.is_locked && a.surface == "件" && i > 0 {
            let prev = &atoms[i - 1];
            let preceded_by_num = prev.upos == Some(Upos::Num)
                || (!prev.is_locked && prev.surface.chars().all(|c| c.is_ascii_digit()));
            let preceded_by_clause_or_det =
                matches!(prev.upos, Some(Upos::Verb | Upos::Aux | Upos::Det))
                    || matches!(
                        prev.surface.as_str(),
                        "の" | "た" | "だ" | "この" | "その" | "あの" | "例"
                    )
                    || out
                        .last()
                        .is_some_and(|x| matches!(x.upos, Some(Upos::Verb | Upos::Aux)));
            if !preceded_by_num && preceded_by_clause_or_det {
                out.push(Atom {
                    span: a.span,
                    surface: "事項".to_owned(),
                    lemma: Some("事項".to_owned()),
                    upos: Some(Upos::Noun),
                    doc_tok_idx: a.doc_tok_idx,
                    head: a.head,
                    deprel: a.deprel.clone(),
                    is_locked: true,
                });
                i += 1;
                continue;
            }
        }

        // 5g. `["手遅れ", "に", "なっ", "て", "いる"|"いた"]` -> locked Verb `"太遲了"`.
        if i + 4 < atoms.len()
            && !atoms[i..i + 5].iter().any(|x| x.is_locked)
            && atoms[i].surface == "手遅れ"
            && atoms[i + 1].surface == "に"
            && atoms[i + 2].surface == "なっ"
            && atoms[i + 3].surface == "て"
            && matches!(atoms[i + 4].surface.as_str(), "いる" | "いた")
        {
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: atoms[i + 4].span.end_char,
                },
                surface: "太遲了".to_owned(),
                lemma: Some("太遲".to_owned()),
                upos: Some(Upos::Verb),
                doc_tok_idx: atoms[i + 2].doc_tok_idx,
                head: atoms[i + 2].head,
                deprel: atoms[i + 2].deprel.clone(),
                is_locked: true,
            });
            i += 5;
            continue;
        }

        // 5h. `[Verb, "がち", "だ"|"です"]` -> locked Verb `"容易{v_zh}"`.
        if !a.is_locked
            && a.upos == Some(Upos::Verb)
            && let Some(gachi) = atoms.get(i + 1)
            && !gachi.is_locked
            && gachi.surface == "がち"
            && let Some(cop) = atoms.get(i + 2)
            && !cop.is_locked
            && matches!(cop.surface.as_str(), "だ" | "です")
        {
            let v_zh = if a.surface == "やめ"
                || matches!(a.lemma.as_deref(), Some("やめる" | "やむ"))
            {
                "停止".to_owned()
            } else {
                translate_single_ja_term(&a.surface, a.lemma.as_deref(), Some(Upos::Verb))
            };
            let zh = format!("容易{v_zh}");
            out.push(Atom {
                span: Span {
                    start_char: a.span.start_char,
                    end_char: cop.span.end_char,
                },
                surface: zh.clone(),
                lemma: Some(zh),
                upos: Some(Upos::Verb),
                doc_tok_idx: a.doc_tok_idx,
                head: a.head,
                deprel: a.deprel.clone(),
                is_locked: true,
            });
            i += 3;
            continue;
        }

        out.push(a.clone());
        i += 1;
    }
    out
}

fn join_zh_parts(parts: &[String]) -> String {
    let mut out = String::new();
    for p in parts {
        if p.is_empty() {
            continue;
        }
        // Avoid double `的的` at boundaries.
        if out.ends_with('的') && p.starts_with('的') {
            out.push_str(&p['的'.len_utf8()..]);
        } else {
            out.push_str(p);
        }
    }
    out
}

fn apply_phrase_matches(atoms: &[Atom]) -> Vec<Atom> {
    let mut out = Vec::with_capacity(atoms.len());
    let mut i = 0usize;
    while i < atoms.len() {
        if atoms[i].is_locked {
            out.push(atoms[i].clone());
            i += 1;
            continue;
        }
        let preceded_by_no = i > 0 && !atoms[i - 1].is_locked && atoms[i - 1].surface == "の";
        let receiving_auxiliary = i >= 2
            && is_te_form_link(&atoms[i - 2], &atoms[i - 1], &atoms[i])
            && atoms[i].lemma.as_deref().unwrap_or(&atoms[i].surface) == "いただく";
        let mut matched_len = 0usize;
        let mut matched_zh: Option<&'static str> = None;
        let max_win = (atoms.len() - i).min(6);
        for len in (2..=max_win).rev() {
            let slice = &atoms[i..i + len];
            if slice.iter().any(|a| a.is_locked) {
                continue;
            }
            if preceded_by_no
                && slice.len() >= 3
                && matches!(
                    slice[1].surface.as_str(),
                    "に" | "で" | "へ" | "と" | "から" | "が" | "を"
                )
            {
                continue;
            }
            let concat: String = slice.iter().map(|a| a.surface.as_str()).collect();
            if receiving_auxiliary && concat == "いただきます" {
                continue;
            }
            if let Some(zh) = lookup_curated_phrase(&concat) {
                matched_len = len;
                matched_zh = Some(zh);
                break;
            }
        }
        if let Some(zh) = matched_zh {
            let last = &atoms[i + matched_len - 1];
            let matched_upos = if matches!(zh, "累積努力" | "加劇") {
                Some(Upos::Verb)
            } else {
                last.upos
            };
            out.push(Atom {
                span: Span {
                    start_char: atoms[i].span.start_char,
                    end_char: last.span.end_char,
                },
                surface: zh.to_owned(),
                lemma: Some(zh.to_owned()),
                upos: matched_upos,
                doc_tok_idx: last.doc_tok_idx,
                head: last.head,
                deprel: last.deprel.clone(),
                is_locked: true, // already translated to Chinese, treat as atomic!
            });
            i += matched_len;
        } else {
            out.push(atoms[i].clone());
            i += 1;
        }
    }
    out
}

const RELATIVE_DAYPART_ATOM: &str = "__ja_relative_daypart_atom";

// Combine only an independent calendar-day/daypart modifier.  This must run
// before adnominal folding and before the two groups enter separate buckets.
fn compose_relative_daypart(groups: Vec<Bunsetsu>) -> Vec<Bunsetsu> {
    let mut result = Vec::with_capacity(groups.len());
    let mut iter = groups.into_iter().peekable();
    while let Some(day) = iter.next() {
        let paired = iter.peek().is_some_and(|part| {
            let [date, no] = day.atoms.as_slice() else {
                return false;
            };
            let [period] = part.atoms.as_slice() else {
                return false;
            };
            !date.is_locked
                && !no.is_locked
                && !period.is_locked
                && matches!(date.surface.as_str(), "今日" | "明日" | "昨日")
                && no.surface == "の"
                && matches!(
                    period.surface.as_str(),
                    "朝" | "午前" | "午後" | "昼" | "夜"
                )
                && date.span.end_char == no.span.start_char
                && no.span.end_char == period.span.start_char
                && !is_clock_atom(period)
                && matches!(day.case_particle.as_deref(), None | Some("の"))
                && part.case_particle.is_none()
                && day.sconj_tail.is_none()
                && part.sconj_tail.is_none()
                && day.sfp_tail.is_none()
                && part.sfp_tail.is_none()
                && matches!(day.role, BunsetsuRole::Other | BunsetsuRole::Adv)
                && matches!(part.role, BunsetsuRole::Other | BunsetsuRole::Adv)
        });
        if paired {
            let part = iter.next().expect("peeked daypart");
            // A following unmarked nominal may make this a modifier of that
            // noun. Leave that ambiguous structure on the existing path.
            let attached_nominal = iter.peek().is_some_and(|next| {
                matches!(next.role, BunsetsuRole::Other | BunsetsuRole::Obj)
                    && next
                        .atoms
                        .first()
                        .is_some_and(|a| matches!(a.upos, Some(Upos::Noun | Upos::Propn)))
            });
            if attached_nominal {
                result.push(day);
                result.push(part);
                continue;
            }
            let date = &day.atoms[0];
            let period = &part.atoms[0];
            let mut combined = date.clone();
            combined.span.end_char = period.span.end_char;
            combined.surface = format!(
                "{}{}",
                translate_single_ja_term(&date.surface, date.lemma.as_deref(), date.upos),
                translate_single_ja_term(&period.surface, period.lemma.as_deref(), period.upos)
            );
            combined.lemma = Some(RELATIVE_DAYPART_ATOM.to_owned());
            combined.is_locked = true;
            result.push(Bunsetsu {
                atoms: vec![combined],
                role: BunsetsuRole::Adv,
                case_particle: None,
                sconj_tail: None,
                sfp_tail: None,
            });
        } else {
            result.push(day);
        }
    }
    result
}

fn linked_joint_case(noun: &Atom, tail: &Atom) -> bool {
    !noun.is_locked
        && !tail.is_locked
        && matches!(noun.deprel.as_str(), "obl" | "advmod")
        && noun.doc_tok_idx.is_some()
        && tail.head == noun.doc_tok_idx
        && tail.upos == Some(Upos::Adp)
        && tail.deprel == "case"
        && noun.span.end_char == tail.span.start_char
}

fn linked_joint_people(number: &Atom, people: &Atom, de: &Atom) -> bool {
    !number.is_locked
        && ((number.upos == Some(Upos::Num) && number.deprel == "nummod")
            || (number.upos == Some(Upos::Noun)
                && number.deprel == "compound"
                && number.surface.chars().all(|ch| matches!(ch, '０'..='９'))))
        && number.head == people.doc_tok_idx
        && number.span.end_char == people.span.start_char
        && linked_joint_case(people, de)
        && people.surface == "人"
        && de.surface == "で"
        && (normalize_decimal_digits(&number.surface)
            .parse::<u32>()
            .is_ok_and(|n| n > 1)
            || number
                .surface
                .chars()
                .all(|c| "一二三四五六七八九十百千萬万".contains(c))
                && number.surface.chars().any(|c| c != '一'))
}

fn render_joint_people(number: &str) -> String {
    let number = normalize_decimal_digits(number);
    let number = if matches!(number.as_str(), "二" | "2") {
        "兩"
    } else {
        &number
    };
    format!("{number}人")
}

// An overt joint participant or a joint manner can invite the listener.
// With an overt third party, the same finite form asks a negative question.
fn mark_joint_question(groups: &mut Vec<Bunsetsu>, source: &[Atom], original: &[Atom]) {
    let [body @ .., verb, polite, neg, ka] = source else {
        return;
    };
    let Some(head) = verb.doc_tok_idx else { return };
    // 勉強し can be split into a bound verbal noun and the finite light verb.
    // The former owns its objects and manner, the latter owns ませんか.
    let action = body
        .last()
        .filter(|word| {
            verb.lemma.as_deref() == Some("する")
                && !word.is_locked
                && word.upos == Some(Upos::Verb)
                && word.deprel == "advcl"
                && word.head == Some(head)
                && word.span.end_char == verb.span.start_char
                && faberline_nlp::lexicon::lookup_ja(&word.surface).is_some_and(|hit| {
                    hit.upos_candidates.contains(&Upos::Noun)
                        && hit.upos_candidates.contains(&Upos::Verb)
                })
        })
        .unwrap_or(verb);
    let lemma = action.lemma.as_deref().unwrap_or(&action.surface);
    let governs = |atom: &Atom| {
        atom.head == Some(head) || action.doc_tok_idx.is_some() && atom.head == action.doc_tok_idx
    };
    if verb.is_locked
        || verb.upos != Some(Upos::Verb)
        || verb.deprel != "root"
        || !matches!(
            (
                polite.surface.as_str(),
                neg.surface.as_str(),
                ka.surface.as_str()
            ),
            ("ませ", "ん", "か")
        )
        || [polite, neg, ka]
            .iter()
            .any(|a| a.is_locked || a.head != Some(head))
        || polite.deprel != "aux"
        || neg.deprel != "aux"
        || ka.deprel != "discourse"
        || verb.span.end_char != polite.span.start_char
        || polite.span.end_char != neg.span.start_char
        || neg.span.end_char != ka.span.start_char
        || body.iter().any(|a| {
            !a.is_locked && a.upos == Some(Upos::Verb) && a.doc_tok_idx != action.doc_tok_idx
        })
        || !faberline_nlp::lexicon::lookup_ja(lemma)
            .is_some_and(|hit| hit.upos_candidates.contains(&Upos::Verb))
    {
        return;
    }
    // Additive も can explicitly invite the addressee to join. A subject
    // such as a teacher, bus or dog keeps the ordinary negative question.
    let listeners_only = body
        .iter()
        .filter(|a| a.deprel == "nsubj" && governs(a))
        .all(|a| {
            !a.is_locked
                && a.upos == Some(Upos::Pron)
                && matches!(a.surface.as_str(), "あなた" | "君")
                && body.iter().any(|mo| {
                    !mo.is_locked
                        && mo.surface == "も"
                        && mo.deprel == "case"
                        && mo.head == a.doc_tok_idx
                        && a.span.end_char == mo.span.start_char
                })
        });
    let actionable_gloss = faberline_nlp::lexicon::ja::lookup_ja_gloss(lemma, Some(Upos::Verb))
        .is_some_and(|value| !value.starts_with('能') && !value.starts_with("可以"));
    let wh_question = body.windows(2).any(|pair| {
        let (word, case) = (&pair[0], &pair[1]);
        !word.is_locked
            && !case.is_locked
            && governs(word)
            && word.doc_tok_idx.is_some()
            && case.head == word.doc_tok_idx
            && case.upos == Some(Upos::Adp)
            && case.deprel == "case"
            && word.span.end_char == case.span.start_char
            && matches!(
                word.surface.as_str(),
                "誰" | "何" | "どれ" | "どこ" | "いつ"
            )
    });
    let intentional_action = actionable_gloss
        && !matches!(
            lemma,
            "ある"
                | "いる"
                | "なる"
                | "分かる"
                | "わかる"
                | "できる"
                | "出来る"
                | "見える"
                | "聞こえる"
                | "くれる"
                | "くださる"
                | "もらえる"
                | "いただく"
                | "いただける"
        );
    let can_offer = listeners_only && intentional_action && !wh_question;
    let attached_case = |noun: &Atom, tail: &Atom| governs(noun) && linked_joint_case(noun, tail);
    let mut markers = groups.iter().enumerate().filter_map(|(i, group)| {
        let text = match group.atoms.as_slice() {
            [joint, case] if attached_case(joint, case) => {
                match (joint.surface.as_str(), case.surface.as_str()) {
                    ("一緒", "に") => JointManner::Together,
                    ("みんな" | "皆", "で") => JointManner::Everyone,
                    _ => return None,
                }
            }
            [joint]
                if !joint.is_locked
                    && matches!(joint.surface.as_str(), "共に" | "ともに")
                    && joint.upos == Some(Upos::Adv)
                    && joint.deprel == "advmod"
                    && governs(joint) =>
            {
                JointManner::Together
            }
            [number, people, de] if governs(people) && linked_joint_people(number, people, de) => {
                JointManner::Counted
            }
            _ => return None,
        };
        Some((i, text))
    });
    let explicit = markers.next();
    if markers.next().is_some() || (explicit.is_none() && !intentional_action) {
        return;
    }
    // Use the source case heads before floor normalization and noun folding.
    // A companion can identify the joint activity without an explicit 一緒.
    let companion = action
        .doc_tok_idx
        .and_then(|id| place_companion(original, id, None));
    let companion_group = companion.and_then(|id| {
        groups.iter().position(|group| {
            group.atoms.iter().any(|atom| atom.doc_tok_idx == Some(id))
                && group.atoms.last().is_some_and(|atom| atom.surface == "と")
        })
    });
    let Some((i, text)) = explicit.or_else(|| companion_group.map(|i| (i, JointManner::Companion)))
    else {
        return;
    };
    if let Some(partner) = companion_group {
        groups[partner].role = BunsetsuRole::JointCompanion;
    }
    groups[i].role = BunsetsuRole::JointManner {
        manner: text,
        invitation: can_offer,
        wh_question,
    };
    if let Some(before) = i.checked_sub(1)
        && let [.., noun, to] = groups[before].atoms.as_slice()
        && attached_case(noun, to)
        && to.surface == "と"
        && to.span.end_char == groups[i].atoms[0].span.start_char
    {
        groups[before].role = BunsetsuRole::JointCompanion;
    }
    let has_object = body.iter().any(|a| a.deprel == "obj" && governs(a));
    for group in groups.iter_mut() {
        let Some(word) = group.atoms.first().filter(|word| governs(word)) else {
            continue;
        };
        let Some((mut noun, case)) = word
            .head
            .and_then(|predicate| indefinite_argument(&group.atoms, predicate))
        else {
            continue;
        };
        let marker = case.as_ref().map(|case| case.surface.as_str());
        let role = if can_offer
            && indefinite_content_object(
                word,
                case.as_ref(),
                noun.span.end_char,
                action,
                has_object,
            ) {
            noun.surface = "點什麼".to_owned();
            BunsetsuRole::Obj
        } else {
            match (word.surface.as_str(), word.deprel.as_str(), marker) {
                ("誰", "obl", Some("と"))
                    if matches!(group.role, BunsetsuRole::JointManner { .. }) =>
                {
                    group.role
                }
                ("誰", "obl", Some("と")) => BunsetsuRole::JointCompanion,
                ("誰" | "何", "nsubj", Some("が" | "は" | "も")) => BunsetsuRole::Subj,
                ("どこ", "obl", Some("へ" | "に" | "で")) => group.role,
                _ => continue,
            }
        };
        group.atoms.clear();
        group.atoms.push(noun);
        group.atoms.extend(case);
        group.role = role;
    }
    // In 何か作業をする, 何か modifies 作業. Keep the entire noun phrase
    // in the original object's position.
    for i in (1..groups.len()).rev() {
        let modifier = groups[i]
            .atoms
            .first()
            .filter(|object| governs(object))
            .and_then(|object| object.head)
            .and_then(|predicate| {
                indefinite_object_modifier(&groups[i - 1], &groups[i], predicate)
            });
        if let Some(modifier) = modifier {
            groups[i].atoms.insert(0, modifier);
            groups.remove(i - 1);
        }
    }
}

// The polite stem is also the continuative used before a motion-purpose に.
// Ask the first-party inflection table to analyze exactly that stem and ます.
// A place name followed by the same suffix does not yield a lexical verb.
fn purpose_stem_lemma(stem: &Atom) -> Option<String> {
    let surface: Vec<_> = stem.surface.chars().chain("ます".chars()).collect();
    let analyzed = faberline_nlp::lexicon::ja::segment_ja(&surface, 0);
    let [verb, polite] = analyzed.as_slice() else {
        return None;
    };
    let len = stem.span.end_char - stem.span.start_char;
    if verb.decided_upos != Some(Upos::Verb)
        || verb.span.start_char != 0
        || verb.span.end_char != len
        || polite.decided_upos != Some(Upos::Aux)
        || polite.span.start_char != len
        || polite.span.end_char != surface.len()
    {
        return None;
    }
    let lemma = verb.lemma.as_deref()?;
    if lookup_curated_ja_word(lemma).is_none()
        && lookup_ja_lexicon_gloss(lemma, None, Some(Upos::Verb)).is_none()
    {
        return None;
    }
    Some(lemma.to_owned())
}

// Some continuatives appear as a nominal base and kana fragments. Only join the
// full span before に when the inflection table recognizes it and the source
// boundary and dependency graph identify the same outer 行く.
fn recover_split_purpose_stems(atoms: &mut Vec<Atom>, go: usize) {
    let mut i = 0;
    while i < atoms.len() {
        if !atoms[i].is_locked
            && atoms[i].surface == "に"
            && let Some((start, stem)) =
                split_purpose_stem(&atoms[..i], &atoms[i], &atoms[i + 1..], go)
        {
            atoms.splice(start..i, [stem]);
            i = start + 1;
        }
        i += 1;
    }
}

fn split_purpose_stem(
    before: &[Atom],
    ni: &Atom,
    after: &[Atom],
    go: usize,
) -> Option<(usize, Atom)> {
    if ni.is_locked || ni.upos != Some(Upos::Adp) || ni.deprel != "case" {
        return None;
    }
    let is_kana = |p: &Atom| !p.is_locked && p.surface.chars().all(|c| ('ぁ'..='ゖ').contains(&c));
    let start = before.iter().rposition(|a| !is_kana(a))?;
    let (left, pieces) = before.split_at(start);
    let [first, tail @ ..] = pieces else {
        return None;
    };
    let last = pieces.last()?;
    let anchor = pieces
        .iter()
        .find(|p| p.doc_tok_idx.is_some() && p.doc_tok_idx == ni.head)?;
    if tail.is_empty()
        || first.is_locked
        || first.upos != Some(Upos::Noun)
        || anchor.deprel != "obl"
        || last.span.end_char != ni.span.start_char
        || pieces
            .windows(2)
            .any(|p| p[0].span.end_char != p[1].span.start_char)
        || pieces.iter().any(|p| {
            p.is_locked
                || p.doc_tok_idx.is_none()
                || !(p.head == anchor.doc_tok_idx || (p.deprel == "obl" && p.head == anchor.head))
        })
    {
        return None;
    }
    if anchor.head != Some(go) || !after.iter().any(|a| a.doc_tok_idx == Some(go)) {
        return None;
    }
    let mut stem = first.clone();
    stem.doc_tok_idx = ni.head;
    stem.head = anchor.head;
    stem.deprel = anchor.deprel.clone();
    stem.surface = pieces.iter().map(|p| p.surface.as_str()).collect();
    stem.span.end_char = last.span.end_char;
    stem.lemma = Some(purpose_stem_lemma(&stem)?);
    let explicit_object = matches!(left, [.., obj, wo]
        if obj.deprel == "obj" && obj.head == anchor.head && obj.doc_tok_idx.is_some()
        && !wo.is_locked && wo.surface == "を" && wo.upos == Some(Upos::Adp)
        && wo.deprel == "case" && wo.head == obj.doc_tok_idx
        && obj.span.end_char == wo.span.start_char && wo.span.end_char == first.span.start_char);
    if !explicit_object {
        // Without an accusative, require an independent left boundary and no
        // competing nominal entry for the entire reconstructed surface.
        let independent = match left {
            [] => true,
            [.., modifier]
                if !modifier.is_locked
                    && modifier.deprel == "advmod"
                    && modifier.head == anchor.head =>
            {
                true
            }
            [.., noun, case] => {
                !case.is_locked
                    && case.deprel == "case"
                    && matches!(
                        case.surface.as_str(),
                        "は" | "が" | "も" | "で" | "に" | "へ"
                    )
                    && noun.doc_tok_idx.is_some()
                    && case.head == noun.doc_tok_idx
                    && noun.head == anchor.head
            }
            _ => false,
        };
        if !independent
            || faberline_nlp::lexicon::lookup_ja(&stem.surface).is_some_and(|e| {
                e.upos_candidates
                    .iter()
                    .any(|pos| matches!(pos, Upos::Noun | Upos::Propn))
            })
        {
            return None;
        }
        stem.upos = Some(Upos::Verb);
    }
    Some((start, stem))
}

// These dictionary glosses encode a transfer. Mandarin places the item after
// the bare action and the recipient after 給.
fn purpose_transfer_action(gloss: &str) -> Option<&str> {
    match gloss {
        "寄送" => Some("寄"),
        "贈送" | "送達" => Some("送"),
        "出借" => Some("借"),
        "付錢" => Some("付"),
        _ => gloss.strip_suffix('給').filter(|action| !action.is_empty()),
    }
}

// A report can be the root of the full sentence even though 行く is the
// finite predicate within the quoted clause. Its governor must be outside
// that clause, and no independent predicate can follow it inside the quote.
fn motion_purpose_governor(source: &[Atom], inside_quote: bool) -> Option<usize> {
    let mut roots = source.iter().enumerate().filter(|(i, go)| {
        !go.is_locked
            && go.upos == Some(Upos::Verb)
            && go.doc_tok_idx.is_some()
            && go.lemma.as_deref().unwrap_or(&go.surface) == "行く"
            && (go.deprel == "root"
                || (inside_quote
                    && matches!(go.deprel.as_str(), "advcl" | "ccomp")
                    && go.head.is_some()
                    && source.iter().all(|a| a.doc_tok_idx != go.head)
                    && source[i + 1..].iter().all(|a| {
                        !a.is_locked && matches!(a.upos, Some(Upos::Aux | Upos::Part | Upos::Punct))
                    })))
    });
    let (_, go) = roots.next()?;
    let id = go.doc_tok_idx?;
    roots.next().is_none().then_some(id)
}

fn purpose_floor_goal(
    group: &Bunsetsu,
    source: &[Atom],
    purpose_id: usize,
    purpose_sets_value: bool,
) -> Option<(usize, usize)> {
    let [label @ .., floor, case] = group.atoms.as_slice() else {
        return None;
    };
    let floor_idx = floor.doc_tok_idx?;
    if is_building_floor(floor)
        || floor.head != Some(purpose_id)
        || floor.deprel != "obl"
        || case.is_locked
        || case.upos != Some(Upos::Adp)
        || !matches!(case.surface.as_str(), "に" | "へ")
        || case.deprel != "case"
        || case.head != Some(floor_idx)
        || floor.span.end_char != case.span.start_char
    {
        return None;
    }
    let end = source
        .iter()
        .position(|a| a.doc_tok_idx == Some(floor_idx))?;
    let start = building_floor_label_start(source, end)?;
    let group_start = end.checked_sub(label.len())?;
    if !has_floor_number_link(label, floor, floor_idx) || group_start < start {
        return None;
    }
    // The only part of a floor label that can come from preceding groups
    // here is a basement prefix. Every part must attach to this purpose
    // or to the floor itself, as well as pass the label's span checks.
    if group_start > start {
        let basement_end = match &source[start..end] {
            [atom, ..] if atom.surface == "地下" => start + 1,
            [ground, below, ..] if ground.surface == "地" && below.surface == "下" => start + 2,
            _ => return None,
        };
        if group_start > basement_end
            || source[start..basement_end].iter().any(|atom| {
                !(atom.head == Some(purpose_id) && matches!(atom.deprel.as_str(), "obl" | "advmod")
                    || atom.head == Some(floor_idx)
                        && matches!(atom.deprel.as_str(), "compound" | "nmod"))
            })
        {
            return None;
        }
    }
    // A change verb with mathematical context can take a new order in に.
    if case.surface == "に"
        && purpose_sets_value
        && has_floor_order_context(source, floor_idx, purpose_id)
    {
        return None;
    }
    Some((start, end))
}

fn mark_motion_purpose(
    groups: &mut Vec<Bunsetsu>,
    source: &[Atom],
    motion_go: usize,
    doc: &Document,
    sentence_idx: usize,
) {
    use faberline_nlp::lexicon::{SemanticClass, lookup_ja};
    let mut found = None;
    for (index, group) in groups.iter().enumerate() {
        let [stem, ni] = group.atoms.as_slice() else {
            continue;
        };
        let Some(id) = stem.doc_tok_idx else { continue };
        let Some(go) = source
            .iter()
            .find(|a| a.doc_tok_idx.is_some() && a.doc_tok_idx == stem.head)
        else {
            continue;
        };
        if stem.is_locked
            || ni.is_locked
            || go.is_locked
            || !matches!(stem.upos, Some(Upos::Verb | Upos::Noun))
            || stem.deprel != "obl"
            || ni.upos != Some(Upos::Adp)
            || ni.surface != "に"
            || ni.deprel != "case"
            || ni.head != Some(id)
            || stem.span.end_char != ni.span.start_char
            || go.upos != Some(Upos::Verb)
            || go.lemma.as_deref().unwrap_or(&go.surface) != "行く"
            || go.doc_tok_idx != Some(motion_go)
            || ni.span.end_char > go.span.start_char
        {
            continue;
        }
        if stem.upos == Some(Upos::Noun) {
            // With a nominal analysis the CPU parser attaches the preceding
            // accusative to 行く. Verify the adjacent, explicit を and its object.
            let has_object = source.windows(2).any(|pair| {
                let (obj, wo) = (&pair[0], &pair[1]);
                obj.doc_tok_idx.is_some()
                    && obj.head == go.doc_tok_idx
                    && obj.deprel == "obj"
                    && !wo.is_locked
                    && wo.surface == "を"
                    && wo.deprel == "case"
                    && wo.head == obj.doc_tok_idx
                    && obj.span.end_char == wo.span.start_char
                    && wo.span.end_char == stem.span.start_char
            });
            if !has_object {
                continue;
            }
        }
        if let Some(lemma) = purpose_stem_lemma(stem) {
            if found.is_some() {
                return;
            }
            found = Some((index, id, go.doc_tok_idx, lemma));
        }
    }
    let Some((mut index, id, go, lemma)) = found else {
        return;
    };
    let partner_marker = match lemma.as_str() {
        "会う" => Some("和"),
        "借りる" => Some("向"),
        _ => None,
    };
    let nominal_stem = groups[index].atoms[0].upos == Some(Upos::Noun);
    let direct_object = source
        .iter()
        .any(|a| a.deprel == "obj" && (a.head == Some(id) || (nominal_stem && a.head == go)));
    let purpose_sets_value = direct_object && matches!(lemma.as_str(), "する" | "戻す" | "変える");
    let transfer = direct_object
        && purpose_transfer_action(&translate_single_ja_term(
            &lemma,
            Some(&lemma),
            Some(Upos::Verb),
        ))
        .is_some();

    // Normalize proven floor spans before regrouping. A split 地 or 下
    // can have its own group even when it belongs to the same floor.
    let mut with_floors = None;
    for group in groups.iter().rev() {
        if let Some((start, end)) = purpose_floor_goal(group, source, id, purpose_sets_value) {
            let atoms = with_floors.get_or_insert_with(|| source.to_vec());
            let floor = building_floor_atom(&source[start..=end], &source[end]);
            atoms.splice(start..=end, [floor]);
        }
    }
    if let Some(atoms) = with_floors {
        let regrouped = group_into_bunsetsus(&atoms, doc, sentence_idx);
        if let Some(new_index) = regrouped.iter().position(
            |group| matches!(group.atoms.as_slice(), [stem, _] if stem.doc_tok_idx == Some(id)),
        ) {
            *groups = regrouped;
            index = new_index;
        }
    }

    groups[index].role = BunsetsuRole::MotionPurpose;
    groups[index].atoms[0].lemma = Some(lemma);
    for group in groups {
        let [noun, ni] = group.atoms.as_slice() else {
            continue;
        };
        if group.role == BunsetsuRole::MotionPurpose
            || is_building_floor(noun)
            || noun.is_locked
            || ni.is_locked
            || ni.upos != Some(Upos::Adp)
            || ni.deprel != "case"
            || noun.doc_tok_idx.is_none()
            || ni.head != noun.doc_tok_idx
            || noun.span.end_char != ni.span.start_char
        {
            continue;
        }
        // Nと immediately before 一緒に marks a companion, not the
        // first noun of a coordinated subject. Both modify this action.
        if ni.surface == "と"
            && noun.deprel == "obl"
            && (noun.head == Some(id) || noun.head == go)
            && source.windows(2).any(|w| {
                let (together, case) = (&w[0], &w[1]);
                !together.is_locked
                    && together.surface == "一緒"
                    && together.head == noun.head
                    && ni.span.end_char == together.span.start_char
                    && together.doc_tok_idx.is_some()
                    && !case.is_locked
                    && case.surface == "に"
                    && case.deprel == "case"
                    && case.head == together.doc_tok_idx
                    && together.span.end_char == case.span.start_char
            })
        {
            group.role = BunsetsuRole::PurposeCompanion;
            continue;
        }
        if ni.surface != "に" {
            continue;
        }
        if noun.deprel == "obl"
            && noun.head == Some(id)
            && is_person_like_atom(noun)
            && let Some(marker) = partner_marker
        {
            group.role = BunsetsuRole::PurposePartner(marker);
        } else if transfer
            && matches!(noun.deprel.as_str(), "obl" | "iobj")
            && (noun.head == Some(id) || (nominal_stem && noun.head == go))
            && is_person_like_atom(noun)
        {
            group.role = BunsetsuRole::PurposeRecipient;
        } else if matches!(noun.deprel.as_str(), "obl" | "advmod")
            && (noun.head == go || noun.head == Some(id))
            && (matches!(noun.surface.as_str(), "週末" | "月末" | "年末" | "一緒")
                || lookup_ja(&noun.surface)
                    .is_some_and(|hit| hit.sem_class == Some(SemanticClass::Time)))
        {
            group.role = BunsetsuRole::PurposeModifier;
        }
    }
}

fn group_into_bunsetsus(atoms: &[Atom], doc: &Document, sentence_idx: usize) -> Vec<Bunsetsu> {
    let mut bunsetsus: Vec<Bunsetsu> = Vec::new();
    let mut cur: Vec<Atom> = Vec::new();

    for (idx, atom) in atoms.iter().enumerate() {
        cur.push(atom.clone());
        let is_func = is_bunsetsu_tail_atom(atom);
        let next = atoms.get(idx + 1);
        let next_is_func = next.is_some_and(is_bunsetsu_tail_atom);
        let next_is_aux = next.is_some_and(|n| {
            !n.is_locked
                && matches!(
                    n.upos,
                    Some(Upos::Aux | Upos::Part | Upos::Adp | Upos::Sconj)
                )
        });
        let keep_te_subsidiary = !atom.is_locked
            && atom.upos == Some(Upos::Sconj)
            && matches!(atom.surface.as_str(), "て" | "で")
            && next.is_some_and(is_te_subsidiary_atom);

        // 明日 and 先生と can modify the same activity independently.
        // A time attached to the companion itself stays inside its noun phrase.
        let activity_adverb = atom.deprel == "advmod"
            && !atom.is_locked
            && atom.head.is_some()
            && atom.upos == Some(Upos::Adv)
            && matches!((next, atoms.get(idx + 2)), (Some(noun), Some(to))
                if !noun.is_locked && !to.is_locked && noun.head == atom.head
                    && noun.deprel == "obl" && noun.doc_tok_idx.is_some()
                    && to.head == noun.doc_tok_idx && to.deprel == "case"
                    && to.surface == "と" && to.span.start_char == noun.span.end_char);
        // Keep `A の B`, `Verb+て+いる/しまう/ください`, or `A さんの` together up to the trailing particles.
        if !next_is_func
            && !next_is_aux
            && !keep_te_subsidiary
            && (is_func || activity_adverb || !is_compound_noun_pair(atom, next))
        {
            bunsetsus.push(classify_bunsetsu(
                std::mem::take(&mut cur),
                doc,
                sentence_idx,
            ));
        }
    }
    if !cur.is_empty() {
        bunsetsus.push(classify_bunsetsu(cur, doc, sentence_idx));
    }
    bunsetsus
}

fn ends_in_genitive_no(b: &Bunsetsu) -> bool {
    !b.atoms.is_empty()
        && b.atoms
            .last()
            .is_some_and(|a| !a.is_locked && a.surface == "の")
}

fn ends_in_coord_ya(b: &Bunsetsu) -> bool {
    !b.atoms.is_empty()
        && b.atoms
            .last()
            .is_some_and(|a| !a.is_locked && a.surface == "や")
}

fn ends_in_coord_to(b: &Bunsetsu) -> bool {
    !matches!(
        b.role,
        BunsetsuRole::JointCompanion
            | BunsetsuRole::TravelCompanion
            | BunsetsuRole::UnresolvedResidenceArgument
            | BunsetsuRole::PurposeCompanion
            | BunsetsuRole::JointManner {
                manner: JointManner::Companion,
                ..
            }
    ) && is_noun_bunsetsu(b)
        && b.atoms
            .last()
            .is_some_and(|a| !a.is_locked && a.surface == "と")
}

fn is_person_like_bunsetsu(b: &Bunsetsu) -> bool {
    b.atoms.iter().any(is_person_like_atom)
}

fn is_person_like_atom(a: &Atom) -> bool {
    matches!(a.upos, Some(Upos::Pron | Upos::Propn))
        || matches!(
            a.surface.as_str(),
            "友達"
                | "先生"
                | "先輩"
                | "人"
                | "彼"
                | "彼女"
                | "私"
                | "父"
                | "母"
                | "兄"
                | "姉"
                | "弟"
                | "妹"
                | "子供"
                | "学生"
                | "お父さん"
                | "お母さん"
                | "お兄さん"
                | "お姉さん"
                | "おじさん"
                | "おばさん"
        )
}

fn is_noun_bunsetsu(b: &Bunsetsu) -> bool {
    if ends_in_genitive_no(b)
        || matches!(
            b.role,
            BunsetsuRole::Predicate
                | BunsetsuRole::Punct
                | BunsetsuRole::Connector
                | BunsetsuRole::ProposalModifier
                | BunsetsuRole::ProposalCompanion
                | BunsetsuRole::Adv
        )
    {
        return false;
    }
    b.atoms.iter().any(|a| {
        a.is_locked
            || matches!(
                a.upos,
                Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::Num)
            )
    })
}

fn is_prenominal_pred(b: &Bunsetsu) -> bool {
    if b.sconj_tail.is_some()
        || b.sfp_tail.is_some()
        || b.atoms
            .windows(2)
            .any(|p| attached_demonstrative(&p[0], &p[1]).is_some())
    {
        // A noun carrying its own demonstrative is not a modifier of the next noun.
        return false;
    }
    let has_pred_or_det = b.role == BunsetsuRole::Predicate
        || b.atoms.iter().any(|a| {
            (!a.is_locked || matches!(a.upos, Some(Upos::Adj | Upos::Det)))
                && matches!(a.upos, Some(Upos::Adj | Upos::Det))
        });
    if !has_pred_or_det {
        return false;
    }
    let has_polite_or_copula_or_te = b.atoms.iter().any(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "ます" | "まし" | "ません" | "です" | "でした" | "だ" | "だった" | "じゃ"
            )
    }) || b
        .atoms
        .last()
        .is_some_and(|a| !a.is_locked && matches!(a.surface.as_str(), "て" | "で"));
    !has_polite_or_copula_or_te
}

/// A separately analysed demonstrative points to its adjacent noun. A standalone
/// pronoun, an interjection or a protected phrase has no such source attachment.
fn attached_demonstrative(det: &Atom, noun: &Atom) -> Option<char> {
    attached_demonstrative_at(det, noun, noun.span.start_char)
}

fn attached_demonstrative_at(det: &Atom, noun: &Atom, nominal_start: usize) -> Option<char> {
    if det.is_locked
        || noun.is_locked
        || det.upos != Some(Upos::Det)
        || noun.upos != Some(Upos::Noun)
        || det.span.end_char != nominal_start
        || det.deprel != "det"
        || det.doc_tok_idx.is_none()
        || noun.doc_tok_idx.is_none()
        || det.head != noun.doc_tok_idx
    {
        return None;
    }
    match det.surface.as_str() {
        "この" => Some('這'),
        "その" | "あの" => Some('那'),
        _ => None,
    }
}

/// A demonstrative already has an attached noun here. Use complete person nouns,
/// including occupations whose dictionary entries have no semantic tag.
fn is_demonstrative_person_noun(noun: &Atom) -> bool {
    is_person_like_atom(noun)
        || matches!(
            noun.surface.as_str(),
            "医師"
                | "医者"
                | "教師"
                | "教員"
                | "教授"
                | "店員"
                | "職員"
                | "社員"
                | "社長"
                | "部長"
                | "課長"
                | "作家"
                | "友人"
                | "生徒"
        )
        || faberline_nlp::lexicon::lookup_ja(&noun.surface).is_some_and(|hit| {
            hit.sem_class == Some(faberline_nlp::lexicon::SemanticClass::Person)
                && hit.upos_candidates.as_slice() == [Upos::Noun]
        })
}

fn is_embedded_acl_arg(b: &Bunsetsu) -> bool {
    if b.atoms.is_empty() {
        return false;
    }
    if b.case_particle.as_deref().is_some_and(|cp| {
        matches!(
            cp,
            "が" | "を" | "で" | "に" | "へ" | "から" | "まで" | "より"
        )
    }) {
        return true;
    }
    if b.case_particle.as_deref() == Some("も")
        && b.atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.surface.as_str(),
                    "何" | "なに" | "なん" | "誰" | "どこ" | "いつ"
                )
        })
    {
        return true;
    }
    b.role == BunsetsuRole::Adv && !b.atoms.iter().any(|a| is_discourse_connector(&a.surface))
}

fn is_spatial_head_bunsetsu(b: &Bunsetsu) -> bool {
    b.atoms.first().is_some_and(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "上" | "下" | "中" | "内" | "外" | "前" | "後" | "隣" | "横" | "そば"
            )
    })
}

fn starts_with_num_counter(b: &Bunsetsu) -> bool {
    b.atoms.first().is_some_and(|a| {
        a.upos == Some(Upos::Num) || (!a.is_locked && a.surface.chars().all(|c| c.is_ascii_digit()))
    })
}

fn prepend_locked_atom(head: &mut Bunsetsu, text: String) {
    if text.is_empty() {
        return;
    }
    let span = head.atoms.first().map(|a| a.span).unwrap_or(Span {
        start_char: 0,
        end_char: 0,
    });
    head.atoms.insert(
        0,
        Atom {
            span,
            surface: text.clone(),
            lemma: Some(text),
            upos: None,
            doc_tok_idx: None,
            head: None,
            deprel: String::new(),
            is_locked: true,
        },
    );
}

// A deadline can precede the topic of an embedded action. Inspect the
// source dependency even when the time and verb are in separate groups.
fn take_adnominal_arguments(
    remaining: &mut Vec<Bunsetsu>,
    predicate: Bunsetsu,
    doc: &Document,
    allow_bare_duration: bool,
) -> Vec<Bunsetsu> {
    let mut deadlines: Vec<_> = remaining
        .extract_if(.., |arg| {
            deadline_governor(arg, doc)
                .or_else(|| match arg.atoms.as_slice() {
                    [a] if calendar::is_interval(a)
                        || calendar::count_modifier(a).is_some()
                        || calendar::individual_arrivals(a) =>
                    {
                        a.head
                    }
                    [date, marker @ ..]
                        if calendar::is_date(date)
                            && marker.iter().all(|m| {
                                !m.is_locked && m.deprel == "case" && m.head == date.doc_tok_idx
                            }) =>
                    {
                        date.head
                    }
                    _ => None,
                })
                .is_some_and(|head| {
                    predicate
                        .atoms
                        .iter()
                        .any(|a| !a.is_locked && a.doc_tok_idx == Some(head))
                })
        })
        .collect();
    let mut arguments = Vec::new();
    while let Some(cand) = remaining.last() {
        let duration = allow_bare_duration
            && matches!(cand.role, BunsetsuRole::Other | BunsetsuRole::Adv)
            && cand.case_particle.is_none()
            && cand.atoms.iter().any(|a| {
                !a.is_locked
                    && matches!(
                        a.surface.as_str(),
                        "長年" | "長期間" | "長い間" | "数年" | "数ヶ月" | "数日" | "何年"
                    )
            });
        if !is_embedded_acl_arg(cand) && !duration {
            break;
        }
        arguments.push(remaining.pop().unwrap());
    }
    arguments.reverse();
    if !deadlines.is_empty() {
        arguments.append(&mut deadlines);
        arguments.sort_by_key(|arg| arg.atoms.first().map(|a| a.span.start_char));
    }
    arguments.push(predicate);
    arguments
}

fn merge_adnominal_bunsetsus(
    bunsetsus: Vec<Bunsetsu>,
    source: &[Atom],
    doc: &Document,
    target: TargetVariant,
) -> Vec<Bunsetsu> {
    let mut res: Vec<Bunsetsu> = Vec::with_capacity(bunsetsus.len());

    for mut b in bunsetsus {
        if (is_noun_bunsetsu(&b) || ends_in_genitive_no(&b))
            && let Some(noun) = b.atoms.first()
            && let Some(modifier) = res.pop_if(|prev| {
                matches!(prev.atoms.as_slice(), [det] if attached_demonstrative(det, noun).is_some())
            })
        {
            // Keep the original atoms, syntactic role, and any following の or case particle.
            b.atoms.splice(0..0, modifier.atoms);
        }
        if is_noun_bunsetsu(&b) {
            let is_sue_ni = b.case_particle.as_deref() == Some("に")
                && b.atoms
                    .first()
                    .is_some_and(|a| !a.is_locked && a.surface == "末")
                && res.last().is_some_and(|prev| {
                    is_prenominal_pred(prev)
                        && prev
                            .atoms
                            .iter()
                            .any(|a| !a.surface.starts_with("[[PH") && a.upos == Some(Upos::Verb))
                });
            if is_sue_ni {
                let pred_b = res.pop().expect("checked last");
                let acl_bunsetsus = take_adnominal_arguments(&mut res, pred_b, doc, true);
                let (mut inner, _) = translate_clause_bunsetsus(
                    acl_bunsetsus,
                    source,
                    "",
                    None,
                    false,
                    &ClauseDiscourseCtx::default(),
                    target,
                );
                while inner.ends_with('了') {
                    inner.pop();
                }
                if !inner.is_empty() {
                    let span = b.atoms[0].span;
                    b.atoms = vec![Atom {
                        span,
                        surface: format!("經過{inner}之後"),
                        lemma: Some(inner),
                        upos: Some(Upos::Adv),
                        doc_tok_idx: None,
                        head: None,
                        deprel: String::new(),
                        is_locked: true,
                    }];
                    b.case_particle = None;
                    b.role = BunsetsuRole::Adv;
                }
                res.push(b);
                continue;
            }

            loop {
                if let Some(mut mod_b) = res.pop_if(|prev| ends_in_genitive_no(prev)) {
                    let has_pred_inside = mod_b.atoms.iter().any(|a| {
                        !a.surface.starts_with("[[PH")
                            && matches!(a.upos, Some(Upos::Verb | Upos::Adj))
                    });
                    if has_pred_inside {
                        let has_hodo_no = mod_b.atoms.len() >= 2
                            && mod_b.atoms[mod_b.atoms.len() - 2].surface == "ほど"
                            && mod_b.atoms[mod_b.atoms.len() - 1].surface == "の";
                        if has_hodo_no {
                            mod_b.atoms.pop();
                            mod_b.atoms.pop();
                        } else {
                            mod_b.atoms.pop();
                        }
                        mod_b.case_particle = None;
                        mod_b.role = BunsetsuRole::Predicate;

                        let acl_bunsetsus = take_adnominal_arguments(&mut res, mod_b, doc, false);
                        let (mut inner, _) = translate_clause_bunsetsus(
                            acl_bunsetsus,
                            source,
                            "",
                            None,
                            false,
                            &ClauseDiscourseCtx::default(),
                            target,
                        );
                        while inner.ends_with('了') {
                            inner.pop();
                        }
                        if has_hodo_no {
                            if !inner.ends_with("般的") {
                                inner.push_str("般的");
                            }
                        } else if !inner.ends_with('的') {
                            inner.push('的');
                        }
                        prepend_locked_atom(&mut b, inner);
                        continue;
                    }

                    if merge_temporal_made_anchor(&mut b, &mod_b.atoms) {
                        continue;
                    }

                    // Pure `N の` modifier (e.g. `美咲の`, `机の`, `3冊の`, `今回の`).
                    mod_b.atoms.pop();
                    let omit_de = starts_with_num_counter(&mod_b)
                        || is_spatial_head_bunsetsu(&b)
                        || mod_b.atoms.first().is_some_and(|a| a.surface == "今回");
                    let mut mod_str = render_bunsetsu_np(&mod_b, target);
                    if !mod_str.is_empty() && !omit_de && !mod_str.ends_with('的') {
                        mod_str.push('的');
                    }
                    prepend_locked_atom(&mut b, mod_str);
                    continue;
                }

                if let Some(mut ya_b) = res.pop_if(|prev| ends_in_coord_ya(prev)) {
                    ya_b.atoms.pop();
                    let ya_str = render_bunsetsu_np(&ya_b, target);
                    if !ya_str.is_empty() {
                        prepend_locked_atom(&mut b, format!("{ya_str}與"));
                    }
                    continue;
                }

                if let Some(mut to_b) = res.pop_if(|prev| {
                    ends_in_coord_to(prev)
                        && !matches!(
                            b.role,
                            BunsetsuRole::ResidenceTogether | BunsetsuRole::ResidenceCount
                        )
                        && !(b.case_particle.as_deref() == Some("を")
                            && is_person_like_bunsetsu(prev)
                            && !is_person_like_bunsetsu(&b))
                }) {
                    to_b.atoms.pop();
                    let to_str = render_bunsetsu_np(&to_b, target);
                    if !to_str.is_empty() {
                        prepend_locked_atom(&mut b, format!("{to_str}和"));
                    }
                    continue;
                }

                if let Some(pred_b) = res.pop_if(|prev| is_prenominal_pred(prev)) {
                    let has_verb = pred_b
                        .atoms
                        .iter()
                        .any(|a| !a.surface.starts_with("[[PH") && a.upos == Some(Upos::Verb));
                    if !has_verb {
                        // Simple prenominal adjective / determiner (`優しいキス`, `甘い吐息`, `これまでの`).
                        let mut adj_str = if pred_b.atoms.len() == 1
                            && !pred_b.atoms[0].is_locked
                            && pred_b.atoms[0].surface == "新しい"
                        {
                            "新".to_owned()
                        } else {
                            render_bunsetsu_np(&pred_b, target)
                        };
                        if adj_str == "以往的"
                            && b.atoms.first().is_some_and(|a| {
                                a.is_locked
                                    && (a.upos == Some(Upos::Adj) || a.surface.ends_with('的'))
                            })
                        {
                            adj_str = "以往".to_owned();
                        } else if pred_b.atoms.len() == 1
                            && !pred_b.atoms[0].is_locked
                            && pred_b.atoms[0].surface == "新しい"
                        {
                            // Keep concise `新` before noun (`新技術`).
                        } else if !adj_str.is_empty() && !adj_str.ends_with('的') {
                            adj_str.push('的');
                        }
                        prepend_locked_atom(&mut b, adj_str);
                        continue;
                    }

                    // Relative clause (`彼は` stops, `彼が書いた` merges into `手紙`).
                    let acl_bunsetsus = take_adnominal_arguments(&mut res, pred_b, doc, false);
                    let (mut inner, _) = translate_clause_bunsetsus(
                        acl_bunsetsus,
                        source,
                        "",
                        None,
                        false,
                        &ClauseDiscourseCtx::default(),
                        target,
                    );
                    while inner.ends_with('了') {
                        inner.pop();
                    }
                    if b.atoms.len() >= 2
                        && !b.atoms[0].is_locked
                        && b.atoms[0].surface == "こと"
                        && matches!(b.case_particle.as_deref(), Some("を" | "が" | "は" | "も"))
                    {
                        b.atoms[0] = Atom {
                            span: b.atoms[0].span,
                            surface: inner.clone(),
                            lemma: Some(inner),
                            upos: Some(Upos::Noun),
                            doc_tok_idx: None,
                            head: None,
                            deprel: String::new(),
                            is_locked: true,
                        };
                        continue;
                    }
                    if !inner.is_empty() && !inner.ends_with('的') {
                        inner.push('的');
                    }
                    prepend_locked_atom(&mut b, inner);
                    continue;
                }

                break;
            }
        }
        res.push(b);
    }

    res
}

fn is_bunsetsu_tail_atom(a: &Atom) -> bool {
    !a.is_locked
        && matches!(
            a.upos,
            Some(Upos::Adp | Upos::Aux | Upos::Part | Upos::Sconj)
        )
}

fn is_compound_noun_pair(a: &Atom, b: Option<&Atom>) -> bool {
    let Some(b) = b else {
        return false;
    };
    if a.is_locked || b.is_locked {
        return false;
    }
    // Keep honorific suffixes (`さん`, `先生`, `先輩`, `ちゃん`, `くん`, `様`) attached to the name!
    if is_honorific_suffix(&b.surface) {
        return true;
    }
    // Keep サ変 `Noun + する` (`左右する`, `加速する`) attached as one predicate bunsetsu.
    if a.upos == Some(Upos::Noun)
        && b.upos == Some(Upos::Verb)
        && matches!(
            b.surface.as_str(),
            "する" | "し" | "さ" | "せ" | "すれ" | "しろ" | "せよ"
        )
        && !matches!(
            a.surface.as_str(),
            "今日"
                | "明日"
                | "昨日"
                | "毎日"
                | "今"
                | "今年"
                | "来年"
                | "去年"
                | "朝"
                | "昼"
                | "夜"
                | "何"
                | "どう"
                | "そう"
                | "こう"
                | "ああ"
        )
    {
        return true;
    }
    // Keep noun-noun compounds or `Verb + Aux` attached.
    matches!(
        (a.upos, b.upos),
        (Some(Upos::Noun | Upos::Propn | Upos::Num), Some(Upos::Noun))
            | (Some(Upos::Verb | Upos::Adj), Some(Upos::Aux))
    )
}

fn is_honorific_suffix(s: &str) -> bool {
    matches!(
        s,
        "さん" | "先生" | "先輩" | "ちゃん" | "くん" | "様" | "さま"
    )
}

fn deadline_weekday(surface: &str) -> Option<&'static str> {
    Some(match surface {
        "月曜日" => "星期一",
        "火曜日" => "星期二",
        "水曜日" => "星期三",
        "木曜日" => "星期四",
        "金曜日" => "星期五",
        "土曜日" => "星期六",
        "日曜日" => "星期日",
        _ => return None,
    })
}

// A time question can fill the endpoint slot just like a specified date or clock.
fn deadline_time_question(atom: &Atom) -> Option<&str> {
    if let Some(day) = calendar::question(atom) {
        return Some(day);
    }
    if atom.is_locked
        || atom.upos != Some(Upos::Pron)
        || !matches!(atom.deprel.as_str(), "obl" | "iobj")
        || atom.head.is_none()
    {
        return None;
    }
    match atom.surface.as_str() {
        "何時" => Some("幾點"),
        "いつ" => Some("什麼時候"),
        _ => None,
    }
}

fn has_temporal_deadline(atoms: &[Atom], doc: &Document) -> bool {
    let [prefix @ .., time, made, ni] = atoms else {
        return false;
    };
    if made.is_locked
        || ni.is_locked
        || made.upos != Some(Upos::Adp)
        || ni.upos != Some(Upos::Adp)
        || made.surface != "まで"
        || ni.surface != "に"
        || made.deprel != "case"
        || ni.deprel != "case"
        || made.head != ni.head
        || time.span.end_char != made.span.start_char
        || made.span.end_char != ni.span.start_char
    {
        return false;
    }
    // Normalized clocks have no token index. The governor of both particles
    // must still fall inside the source span of the recognized time.
    let Some(head) = made.head.and_then(|idx| doc.tokens.get(idx)) else {
        return false;
    };
    if head.span.start_char < time.span.start_char || head.span.end_char > time.span.end_char {
        return false;
    }
    // The segmenter emits 六月 as one noun, but 12月 as a number and 月.
    let month_number = time.surface.strip_suffix('月').and_then(|n| {
        if n.is_empty() {
            prefix
                .last()
                .filter(|a| {
                    !a.is_locked
                        && a.head == time.doc_tok_idx
                        && matches!(a.deprel.as_str(), "nummod" | "compound")
                        && a.span.end_char == time.span.start_char
                })
                .map(|a| a.surface.as_str())
        } else {
            Some(n)
        }
    });
    is_clock_atom(time)
        || calendar::is_date(time)
        || deadline_time_question(time).is_some()
        || (!time.is_locked
            && (time.upos == Some(Upos::Noun)
                && month_number.is_some_and(|n| {
                    n.chars().count() <= 3
                        && crate::opaque::parse_cjk_or_ascii_num(n)
                            .is_some_and(|m| (1..=12).contains(&m))
                })
                || is_daypart_surface(&time.surface)
                || deadline_weekday(&time.surface).is_some()
                || matches!(
                    time.surface.as_str(),
                    "今" | "昨日"
                        | "今日"
                        | "明日"
                        | "明後日"
                        | "先週"
                        | "今週"
                        | "来週"
                        | "週末"
                        | "先月"
                        | "今月"
                        | "来月"
                        | "月末"
                        | "去年"
                        | "今年"
                        | "来年"
                        | "年末"
                )))
}

// A split voice form can put られる on the timed verb but attach the negative
// auxiliary to a second verb. That negation does not establish an interval
// throughout which the timed event fails. Keep the affirmative cutoff marker.
fn has_detached_voice_negation(source: &[Atom], timed: usize) -> bool {
    let Some(other) = source
        .iter()
        .find(|a| {
            a.doc_tok_idx == Some(timed)
                && !a.is_locked
                && a.upos == Some(Upos::Verb)
                && a.deprel == "advcl"
        })
        .and_then(|a| a.head)
    else {
        return false;
    };
    source.iter().any(|a| {
        !a.is_locked
            && a.head == Some(timed)
            && a.upos == Some(Upos::Aux)
            && a.deprel == "aux"
            && matches!(a.lemma.as_deref(), Some("れる" | "られる"))
    }) && source
        .iter()
        .any(|a| !a.is_locked && a.doc_tok_idx == Some(other) && a.upos == Some(Upos::Verb))
        && source.iter().any(|a| {
            !a.is_locked
                && a.head == Some(other)
                && a.upos == Some(Upos::Aux)
                && a.deprel == "aux"
                && matches!(a.lemma.as_deref().unwrap_or(&a.surface), "ない" | "ん")
        })
}

fn deadline_governor(group: &Bunsetsu, doc: &Document) -> Option<usize> {
    if group.case_particle.as_deref() != Some("までに") {
        return None;
    }
    deadline_case_governor(group.atoms.last()?, doc)
}

fn deadline_case_governor(case: &Atom, doc: &Document) -> Option<usize> {
    // Even on a normalized clock, the case marker retains the original time token.
    let time = case.head?;
    doc.syntax.iter().find(|node| node.token == time)?.head
}

// The deadline belongs to the action in [V こと] が できる. Keep all its
// arguments on the normal assembly path and move only the outer ability.
fn fold_nominalized_deadline_ability(
    groups: &mut Vec<Bunsetsu>,
    source: &[Atom],
    doc: &Document,
    target: TargetVariant,
) {
    let [arguments @ .., action_group, nominal, matrix] = groups.as_slice() else {
        return;
    };
    let ([compound @ .., action], [koto, case], [ability, tail @ ..]) = (
        action_group.atoms.as_slice(),
        nominal.atoms.as_slice(),
        matrix.atoms.as_slice(),
    ) else {
        return;
    };
    let Some(action_idx) = action.doc_tok_idx else {
        return;
    };
    if compound.iter().any(|a| {
        a.is_locked
            || a.upos != Some(Upos::Noun)
            || a.deprel != "compound"
            || a.head != Some(action_idx)
    }) || [action, koto, case, ability].iter().any(|a| a.is_locked)
        || action.upos != Some(Upos::Verb)
        || action.deprel != "acl"
        || koto.surface != "こと"
        || koto.upos != Some(Upos::Noun)
        || koto.doc_tok_idx.is_none()
        || action.head != koto.doc_tok_idx
        || koto.deprel != "nsubj"
        || !matches!(case.surface.as_str(), "が" | "は" | "も")
        || case.upos != Some(Upos::Adp)
        || case.deprel != "case"
        || case.head != koto.doc_tok_idx
        || ability.deprel != "root"
        || ability.upos != Some(Upos::Verb)
        || ability.doc_tok_idx.is_none()
        || !matches!(
            ability.lemma.as_deref().unwrap_or(&ability.surface),
            "できる" | "出来る"
        )
        || koto.head != ability.doc_tok_idx
        || [action, koto, case, ability]
            .windows(2)
            .any(|p| p[0].span.end_char != p[1].span.start_char)
        || matrix
            .atoms
            .windows(2)
            .any(|p| p[0].span.end_char != p[1].span.start_char)
        || tail.iter().any(|a| {
            a.is_locked
                || a.head != ability.doc_tok_idx
                || !(a.upos == Some(Upos::Aux) && a.deprel == "aux"
                    || a.upos == Some(Upos::Part)
                        && a.deprel == "discourse"
                        && matrix.sfp_tail.as_deref() == Some(a.surface.as_str()))
        })
    {
        return;
    }
    let mut deadlines = arguments
        .iter()
        .enumerate()
        .filter(|(_, b)| b.case_particle.as_deref() == Some("までに"));
    let Some((deadline_idx, deadline)) = deadlines.next() else {
        return;
    };
    if deadlines.next().is_some() || deadline_governor(deadline, doc) != Some(action_idx) {
        return;
    }
    let (before, _, after, passive) =
        synthesize_predicate_complex(std::slice::from_ref(matrix), false, target, None);
    let modal = match (before.as_slice(), after.as_slice(), passive) {
        ([], [], false) => "可以",
        ([], [past], false) if past == "了" => "得以",
        ([negative], [], false) if negative == "不" => "沒辦法",
        ([negative], [], false) if negative == "沒有" => "沒能",
        _ => return,
    };
    let additive = case.surface == "も";
    let wh = source.iter().enumerate().any(|(i, a)| {
        !a.is_locked
            && a.upos == Some(Upos::Pron)
            && a.head == Some(action_idx)
            && (matches!(a.surface.as_str(), "誰" | "何" | "どこ" | "どれ")
                || deadline.atoms.iter().any(|time| {
                    time.doc_tok_idx == a.doc_tok_idx && deadline_time_question(time).is_some()
                }))
            && !source
                .get(i..i + 2)
                .is_some_and(|pair| indefinite_argument(pair, action_idx).is_some())
    });
    let particle = matrix.sfp_tail.clone().filter(|s| s != "か" || !wh);
    let action = action.clone();
    groups.truncate(groups.len() - 2);
    groups.last_mut().unwrap().sfp_tail = particle;
    groups[deadline_idx].role = BunsetsuRole::ExplicitDeadline { modal, additive };
    localize_indefinite_arguments(groups, &action, source);
}

fn deadline_ability_action(groups: &[Bunsetsu]) -> Option<String> {
    // Only move a modal when the deadline and finite predicate have one reading.
    if groups
        .iter()
        .filter(|b| b.case_particle.as_deref() == Some("までに"))
        .count()
        != 1
    {
        return None;
    }
    let mut predicates = groups.iter().filter(|b| b.role == BunsetsuRole::Predicate);
    let finite = finite_predicate_atoms(predicates.next()?);
    let (verb, tail) = finite.split_first()?;
    if predicates.next().is_some()
        || verb.upos != Some(Upos::Verb)
        || verb.is_locked
        || tail
            .iter()
            .any(|a| a.is_locked || a.upos != Some(Upos::Aux))
        || finite
            .windows(2)
            .any(|p| p[0].span.end_char != p[1].span.start_char)
    {
        return None;
    }
    if verb.lemma.as_deref() == Some("できる") {
        // In 提出できる the action noun already occupies its own phrase.
        // Standalone できる can also mean that something comes into existence.
        return groups
            .iter()
            .flat_map(|b| &b.atoms)
            .any(|noun| {
                !noun.is_locked
                    && noun.upos == Some(Upos::Noun)
                    && noun.deprel == "compound"
                    && noun.head.is_some()
                    && noun.head == verb.doc_tok_idx
                    && noun.span.end_char == verb.span.start_char
            })
            .then(String::new);
    }
    let action = translate_single_ja_term(&verb.surface, verb.lemma.as_deref(), verb.upos);
    // The parser also expresses a plain negative potential (e.g. 書けない)
    // as the base verb plus a negative auxiliary carrying the lemma れる.
    if tail
        .first()
        .is_some_and(|a| a.surface == "ない" && a.lemma.as_deref() == Some("れる"))
    {
        return Some(action);
    }
    action
        .strip_prefix("能夠")
        .or_else(|| action.strip_prefix('能'))
        .filter(|rest| !rest.is_empty())
        .map(str::to_owned)
}

fn classify_bunsetsu(atoms: Vec<Atom>, doc: &Document, sentence_idx: usize) -> Bunsetsu {
    let mut case_particle = None;
    let mut sconj_tail = None;
    let mut sfp_tail = None;
    let mut head_doc_idx = None;
    let mut has_verb_or_adj = false;
    let mut has_copula = false;
    let mut is_connector = false;
    let mut is_adv = false;
    let mut is_det = false;
    let mut deprel_hint = "";

    for a in &atoms {
        if a.doc_tok_idx.is_some() {
            head_doc_idx = a.doc_tok_idx;
        }
        if !a.deprel.is_empty() {
            deprel_hint = a.deprel.as_str();
        }
        if a.is_locked {
            if !a.surface.starts_with("[[PH") && matches!(a.upos, Some(Upos::Verb | Upos::Adj)) {
                has_verb_or_adj = true;
            }
            continue;
        }
        match a.upos {
            Some(Upos::Adp) => {
                if matches!(a.surface.as_str(), "は" | "も")
                    && matches!(
                        case_particle.as_deref(),
                        Some("に" | "で" | "へ" | "から" | "まで" | "と" | "より")
                    )
                {
                    // Preserve semantic case marker when focused by topic/inclusive particle (`日本では`, `場とも`, `以前よりも`).
                } else if matches!(
                    a.surface.as_str(),
                    "が" | "は"
                        | "を"
                        | "に"
                        | "で"
                        | "と"
                        | "とも"
                        | "から"
                        | "まで"
                        | "へ"
                        | "より"
                        | "の"
                        | "も"
                        | "によって"
                        | "につきまして"
                        | "につきましては"
                        | "について"
                        | "に関して"
                ) {
                    case_particle = Some(a.surface.clone());
                }
            }
            Some(Upos::Sconj) => {
                sconj_tail = Some(a.surface.clone());
            }
            Some(Upos::Part) => {
                if a.surface == "だって" {
                    case_particle = Some("だって".to_owned());
                } else if matches!(
                    a.surface.as_str(),
                    "か" | "よ" | "ね" | "わ" | "ぞ" | "ぜ" | "かな" | "よね"
                ) {
                    sfp_tail = Some(a.surface.clone());
                }
            }
            Some(Upos::Verb) => has_verb_or_adj = true,
            Some(Upos::Adj) => has_verb_or_adj = true,
            Some(Upos::Det) => is_det = true,
            Some(Upos::Aux) => {
                if matches!(
                    a.surface.as_str(),
                    "だ" | "です" | "でした" | "だった" | "じゃない" | "ではない"
                ) {
                    has_copula = true;
                }
            }
            Some(Upos::Cconj) => is_connector = true,
            Some(Upos::Adv) => is_adv = true,
            _ => {}
        }
        if is_discourse_connector(&a.surface)
            && !matches!(a.upos, Some(Upos::Adp | Upos::Part | Upos::Sconj))
        {
            is_connector = true;
        }
    }

    // Check Layer 5 `doc.frames` for ARG0 / ARG1 / ARGM hints on `head_doc_idx`.
    let mut frame_role: Option<FrameRole> = None;
    if let Some(hidx) = head_doc_idx {
        for fr in &doc.frames {
            if fr.sentence == sentence_idx {
                if fr.predicate == hidx {
                    has_verb_or_adj = true;
                }
                for arg in &fr.args {
                    if arg.head == hidx {
                        frame_role = Some(arg.role);
                    }
                }
            }
        }
    }

    // Reclassify post-predicate `から`/`ので`/`けど`/... or non-past `と` as `sconj_tail` even if tagged `Adp`.
    let last_non_punct = atoms.iter().rev().find(|a| {
        !is_punct_atom(a)
            && (a.is_locked
                || !matches!(
                    a.surface.as_str(),
                    "か" | "よ" | "ね" | "な" | "わ" | "ぞ" | "ぜ" | "かな" | "よね"
                ))
    });
    let ends_with_no = last_non_punct.is_some_and(|a| !a.is_locked && a.surface == "の");
    if has_temporal_te_kara_tail(&atoms) {
        sconj_tail = Some("てから".to_owned());
        case_particle = None;
    } else if (has_verb_or_adj || has_copula)
        && !ends_with_no
        && let Some(tail) = last_non_punct
        && !tail.is_locked
    {
        let ts = tail.surface.as_str();
        let has_past_in_b = atoms.iter().any(|x| {
            !x.is_locked
                && matches!(
                    x.surface.as_str(),
                    "た" | "だ" | "ました" | "だった" | "でした"
                )
        });
        if is_clause_split_sconj(ts)
            || matches!(
                ts,
                "ば" | "つつ" | "でも" | "としても" | "くせに" | "だけど" | "し"
            )
        {
            sconj_tail = Some(tail.surface.clone());
            if case_particle.as_deref() == Some(ts) {
                case_particle = None;
            }
        } else if ts == "と" && !has_past_in_b {
            sconj_tail = Some("と".to_owned());
            if case_particle.as_deref() == Some("と") {
                case_particle = None;
            }
        }
    }

    let role = if is_connector
        && case_particle.is_none()
        && !has_verb_or_adj
        && !has_copula
        && atoms.len() <= 2
    {
        BunsetsuRole::Connector
    } else if (has_verb_or_adj || has_copula || deprel_hint == "root") && !ends_with_no {
        BunsetsuRole::Predicate
    } else if case_particle.as_deref() == Some("に") && has_temporal_deadline(&atoms, doc) {
        case_particle = Some("までに".to_owned());
        BunsetsuRole::Obl
    } else if let Some(cp) = case_particle.as_deref() {
        match cp {
            "が" | "は" | "も" | "だって" => BunsetsuRole::Subj,
            "を" => BunsetsuRole::Obj,
            "に" => {
                // Time/location vs indirect object: check if noun is time/place or animate/frame ARG1.
                if is_time_or_place_bunsetsu(&atoms)
                    || frame_role == Some(FrameRole::Argm)
                    || atoms
                        .iter()
                        .any(|a| a.lemma.as_deref() == Some(OPAQUE_EVENT_OBLIQUE))
                {
                    BunsetsuRole::Obl
                } else {
                    BunsetsuRole::IObj
                }
            }
            "で"
            | "から"
            | "まで"
            | "へ"
            | "より"
            | "によって"
            | "につきまして"
            | "につきましては"
            | "について"
            | "に関して" => BunsetsuRole::Obl,
            "と" | "とも" => BunsetsuRole::Other,
            _ => BunsetsuRole::Other,
        }
    } else if atoms.iter().any(is_clock_atom) {
        BunsetsuRole::Adv
    } else if is_det {
        BunsetsuRole::Other
    } else if frame_role == Some(FrameRole::Arg0) || deprel_hint == "nsubj" {
        BunsetsuRole::Subj
    } else if frame_role == Some(FrameRole::Arg1) || deprel_hint == "obj" {
        BunsetsuRole::Obj
    } else if is_adv || deprel_hint == "advmod" {
        BunsetsuRole::Adv
    } else if atoms.iter().all(is_punct_atom) {
        BunsetsuRole::Punct
    } else {
        BunsetsuRole::Other
    };

    let joined_b: String = atoms.iter().map(|a| a.surface.as_str()).collect();
    if joined_b.contains("ねばなら")
        || joined_b.contains("なければなら")
        || joined_b.contains("なくてはなら")
        || joined_b.contains("なきゃなら")
    {
        sconj_tail = None;
        if sfp_tail.as_deref() == Some("ね") {
            sfp_tail = None;
        }
    }

    Bunsetsu {
        atoms,
        role,
        case_particle,
        sconj_tail,
        sfp_tail,
    }
}

fn is_place_location_bunsetsu(atoms: &[Atom]) -> bool {
    atoms.iter().any(|a| {
        if is_building_floor(a) {
            return true;
        }
        matches!(
            a.surface.as_str(),
            "部屋"
                | "学校"
                | "図書館"
                | "公園"
                | "駅"
                | "東京駅"
                | "東京"
                | "家"
                | "会社"
                | "病院"
                | "中"
                | "上"
                | "下"
                | "前"
                | "後"
                | "奥"
                | "ここ"
                | "そこ"
                | "あそこ"
                | "ベッド"
                | "椅子"
                | "机"
                | "風呂"
        )
    })
}

fn is_instrument_bunsetsu(atoms: &[Atom]) -> bool {
    atoms.iter().any(|a| {
        matches!(
            a.surface.as_str(),
            "指先"
                | "舌先"
                | "指"
                | "手"
                | "両手"
                | "片手"
                | "舌"
                | "唇"
                | "口"
                | "歯"
                | "足"
                | "声"
                | "目"
                | "陰茎"
                | "ペニス"
                | "亀頭"
                | "ペン"
                | "鉛筆"
                | "箸"
                | "フォーク"
                | "スプーン"
                | "ナイフ"
                | "車"
                | "電車"
                | "バス"
                | "携帯"
                | "スマホ"
                | "電話"
                | "パソコン"
                | "バイブ"
                | "ローター"
                | "全身"
                | "心"
        )
    })
}

fn is_payment_instrument(atoms: &[Atom], predicates: &[Bunsetsu]) -> bool {
    // The head of で must name a payment method and modify the payment
    // predicate itself. A shop or a counter can modify the same predicate.
    atoms.windows(2).enumerate().any(|(idx, pair)| {
        let method = &pair[0];
        let particle = &pair[1];
        if method.is_locked
            || method.upos != Some(Upos::Noun)
            || method.deprel != "obl"
            || !is_payment_medium(method, atoms[..idx].last())
            || particle.is_locked
            || particle.surface != "で"
            || particle.upos != Some(Upos::Adp)
            || particle.deprel != "case"
            || particle.head.is_none()
            || particle.head != method.doc_tok_idx
        {
            return false;
        }
        let Some(predicate_idx) = method.head else {
            return false;
        };
        predicates.iter().flat_map(|b| &b.atoms).any(|verb| {
            verb.doc_tok_idx == Some(predicate_idx)
                && !verb.is_locked
                && verb.upos == Some(Upos::Verb)
                && matches!(
                    verb.lemma.as_deref().unwrap_or(&verb.surface),
                    "支払う" | "払う"
                )
        })
    })
}

fn is_payment_medium(method: &Atom, modifier: Option<&Atom>) -> bool {
    let word = method.lemma.as_deref().unwrap_or(&method.surface);
    matches!(
        word,
        "現金"
            | "カード"
            | "クレジットカード"
            | "デビットカード"
            | "プリペイドカード"
            | "小銭"
            | "電子マネー"
    ) || modifier.is_some_and(|m| {
        !m.is_locked
            && m.upos == Some(Upos::Noun)
            && m.deprel == "compound"
            && m.head == method.doc_tok_idx
            && m.span.end_char == method.span.start_char
            && matches!(
                (m.lemma.as_deref().unwrap_or(&m.surface), word),
                ("小", "銭") | ("電子", "マネー")
            )
    })
}

fn is_time_or_place_bunsetsu(atoms: &[Atom]) -> bool {
    atoms
        .iter()
        .any(|a| is_clock_atom(a) || calendar::is_date(a))
        || is_place_location_bunsetsu(atoms)
        || atoms.iter().any(|a| {
            matches!(
                a.surface.as_str(),
                "今日"
                    | "明日"
                    | "昨日"
                    | "今"
                    | "朝"
                    | "昼"
                    | "夜"
                    | "子宮"
                    | "時代"
                    | "世紀"
                    | "年代"
                    | "時期"
                    | "期間"
                    | "現代"
                    | "現在"
                    | "今後"
                    | "将来"
                    | "過去"
                    | "当時"
                    | "世界"
                    | "社会"
            )
        })
}

fn is_daypart_surface(surface: &str) -> bool {
    matches!(
        surface.strip_suffix("過ぎ").unwrap_or(surface),
        "早朝"
            | "明け方"
            | "朝"
            | "午前"
            | "正午"
            | "昼"
            | "昼間"
            | "午後"
            | "夕方"
            | "夜"
            | "夜中"
            | "夜間"
            | "深夜"
    )
}

// These predicates can select the time itself. The analyzer uses the same obl
// relation for the selected value and for an ordinary event-time modifier.
fn selects_time_value(lemma: &str) -> bool {
    let stem = lemma.strip_suffix("する").unwrap_or(lemma);
    if time_adjustment_action(stem).is_some() {
        return true;
    }
    // Result and selection, e.g. 会議を夕方に決める.
    const CHOOSES_VALUE: &[&str] = &[
        "なる",
        "成る",
        "変わる",
        "決まる",
        "定まる",
        "選ぶ",
        "選択",
        "予定",
        "予約",
    ];
    // Rescheduling a source time, including verbal nouns followed by する.
    const RESCHEDULES: &[&str] = &[
        "合う",
        "ずれる",
        "ずれ込む",
        "延びる",
        "遅れる",
        "早まる",
        "繰り上がる",
        "繰り下がる",
    ];
    // The time can be a reference or endpoint rather than when one acts.
    const TAKES_REFERENCE: &[&str] = &[
        "似る",
        "比べる",
        "備える",
        "慣れる",
        "間に合う",
        "近づく",
        "及ぶ",
        "至る",
    ];
    CHOOSES_VALUE.contains(&stem) || RESCHEDULES.contains(&stem) || TAKES_REFERENCE.contains(&stem)
}

fn has_verb_reading(surface: &str) -> bool {
    faberline_nlp::lexicon::lookup_ja(surface)
        .is_some_and(|hit| hit.upos_candidates.contains(&Upos::Verb))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingTime {
    At(usize),
    From(usize),
}

// The ordinary noun fold can join 昨日の夕方 as an attributive phrase.
// Its final に still belongs to the travel time when the return governs it.
fn trip_relative_daypart(group: &Bunsetsu, source: &[Atom], returning: usize) -> Option<String> {
    let [prefix, head, marker] = group.atoms.as_slice() else {
        return None;
    };
    if group.case_particle.as_deref() != Some("に")
        || !prefix.is_locked
        || marker.doc_tok_idx.is_none()
    {
        return None;
    }
    let [date, no, time, ni] = source
        .windows(4)
        .find(|w| w[2].doc_tok_idx == head.doc_tok_idx && w[3].doc_tok_idx == marker.doc_tok_idx)?
    else {
        return None;
    };
    if [date, no, time, ni].iter().any(|a| a.is_locked)
        || no.surface != "の"
        || ni.surface != "に"
        || !is_daypart_surface(&time.surface)
        || date.deprel != "nmod"
        || date.doc_tok_idx.is_none()
        || time.doc_tok_idx.is_none()
        || date.head != time.doc_tok_idx
        || source.iter().any(|a| {
            a.head == date.doc_tok_idx && matches!(a.deprel.as_str(), "nmod" | "acl" | "compound")
        })
        || no.head != date.doc_tok_idx
        || no.deprel != "case"
        || ni.head != time.doc_tok_idx
        || ni.deprel != "case"
        || time.head != Some(returning)
        || !matches!(time.deprel.as_str(), "obl" | "iobj" | "advmod")
    {
        return None;
    }
    Some(format!(
        "{}{}",
        render_temporal_made_atom(date)?,
        render_temporal_made_atom(time)?
    ))
}

fn render_daypart_np(group: &Bunsetsu, target: TargetVariant) -> String {
    if let [t, _] = group.atoms.as_slice()
        && let Some(label) = after_daypart(t)
    {
        return label;
    }
    render_bunsetsu_np(group, target)
}

fn after_daypart(time: &Atom) -> Option<String> {
    if time.is_locked || !matches!(time.upos, Some(Upos::Noun | Upos::Adv)) {
        return None;
    }
    let daypart = time.surface.strip_suffix("過ぎ")?;
    is_daypart_surface(daypart)
        .then(|| format!("{}過後", translate_single_ja_term(daypart, None, time.upos)))
}

fn setting_time_label(group: &Bunsetsu, target: TargetVariant) -> String {
    // Within a pair of readings for one clock or schedule, 一時 is one o'clock.
    if matches!(group.atoms.as_slice(), [t, _] if !t.is_locked && t.surface == "一時") {
        return "一點".to_owned();
    }
    render_daypart_np(group, target)
}

#[derive(Clone, Copy)]
struct TimedResult {
    time: SettingTime,
    object: usize,
    value: usize,
    action: &'static str,
    site_name: Option<&'static str>,
}

// The transitive members of the time-value predicates can set a new value.
// A comparison, an intransitive change, or a plain temporal adjunct cannot.
fn time_adjustment_action(stem: &str) -> Option<&'static str> {
    match stem {
        "戻す" => Some("改回"),
        "変える" | "変更" => Some("改到"),
        "合わせる" | "調整" => Some("調到"),
        "ずらす" | "移す" => Some("移到"),
        "決める" | "定める" | "決定" | "設定" | "指定" => Some("定在"),
        "延期" | "延ばす" | "遅らせる" | "繰り下げる" | "先送り" => Some("延到"),
        "早める" | "繰り上げる" | "前倒し" => Some("提前到"),
        _ => None,
    }
}

// Tに [setting を] Vに [change] has two time arguments. Their positions
// around the affected object and attachment to the same predicate identify
// the new value even when the two source times are exchanged.
fn timed_result_frame(groups: &[Bunsetsu], source: &[Atom]) -> Option<TimedResult> {
    let covers = |b: &Bunsetsu, a: &Atom| {
        b.atoms
            .iter()
            .any(|p| p.span.start_char <= a.span.start_char && a.span.end_char <= p.span.end_char)
    };
    let predicates = groups.iter().filter(|b| b.role == BunsetsuRole::Predicate);
    let root = source.iter().find(|a| {
        !a.is_locked
            && a.upos == Some(Upos::Verb)
            && a.deprel == "root"
            && predicates.clone().any(|p| covers(p, a))
    })?;
    // 変更する can be a single verb. For 延期 + する the lexical verb
    // carries all three arguments and directly adjoins the light verb.
    let verb = if root.lemma.as_deref() == Some("する") {
        source.iter().find(|a| {
            !a.is_locked
                && a.upos == Some(Upos::Verb)
                && a.deprel == "advcl"
                && root.doc_tok_idx.is_some()
                && a.head == root.doc_tok_idx
                && a.span.end_char == root.span.start_char
        })?
    } else {
        root
    };
    if predicates.clone().any(|p| {
        (!covers(p, root) && !covers(p, verb)) || predicates_has_passive(std::slice::from_ref(p))
    }) {
        return None;
    }
    let lemma = verb.lemma.as_deref().unwrap_or(&verb.surface);
    let action = time_adjustment_action(lemma.strip_suffix("する").unwrap_or(lemma))?;
    let id = verb.doc_tok_idx?;
    let mut objects = groups
        .iter()
        .enumerate()
        .filter(|(_, b)| b.role == BunsetsuRole::Obj && b.case_particle.as_deref() == Some("を"));
    let (object, group) = objects.next()?;
    if objects.next().is_some() {
        return None;
    }
    let noun = group
        .atoms
        .iter()
        .find(|a| a.head == Some(id) && a.deprel == "obj" && !a.is_locked)?;
    let wo = group.atoms.last()?;
    if wo.is_locked
        || wo.surface != "を"
        || wo.deprel != "case"
        || noun.doc_tok_idx.is_none()
        || wo.head != noun.doc_tok_idx
        || noun.span.end_char != wo.span.start_char
    {
        return None;
    }
    // A normalized clock can span several source tokens and no longer has
    // one governor index. Only accept that clock in a single-predicate clause
    // moving an identified time setting between two explicit readings.
    let setting_range = lemma == "戻す"
        && source_time_head(noun)
        && source
            .iter()
            .any(|a| !a.is_locked && a.surface == "から" && a.deprel == "case")
        && source
            .iter()
            .filter(|a| a.upos == Some(Upos::Verb))
            .all(|a| a.doc_tok_idx == Some(id));
    let args = groups.iter().enumerate().filter_map(|(i, b)| {
        let [.., t, ni] = b.atoms.as_slice() else {
            return None;
        };
        let attached = !t.is_locked
            && matches!(t.deprel.as_str(), "obl" | "advmod")
            && t.head == Some(id)
            && t.doc_tok_idx.is_some()
            && ni.head == t.doc_tok_idx
            && ni.deprel == "case";
        let clock = setting_range
            && t.is_locked
            && t.lemma.as_deref() == Some(CLOCK_ATOM)
            && !t.surface.contains("[[PH")
            && b.atoms.len() == 2;
        (matches!(b.case_particle.as_deref(), Some("に" | "から"))
            && (attached || clock)
            && !ni.is_locked
            && matches!(
                t.upos,
                Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::Adv)
            )
            && b.case_particle.as_deref() == Some(ni.surface.as_str())
            && ni.upos == Some(Upos::Adp)
            && t.span.end_char == ni.span.start_char)
            .then_some((i, t, ni))
    });
    let mut values = args.clone().filter(|(_, _, p)| p.surface == "に");
    let first_value = values.next()?;
    let second_value = values.next();
    if values.next().is_some() {
        return None;
    }
    if second_value.is_none() && setting_range {
        let mut origins = args.filter(|(_, _, p)| p.surface == "から");
        let origin = origins.next()?;
        let value = first_value;
        let reading = |(i, t, _): (usize, &Atom, &Atom)| {
            groups[i].atoms.len() == 2
                && (t.lemma.as_deref() == Some(CLOCK_ATOM)
                    || is_daypart_surface(&t.surface)
                    || t.surface == "一時"
                    || render_clock_surface(&t.surface).is_some())
        };
        if origins.next().is_some() || !reading(origin) || !reading(value) {
            return None;
        }
        let action = if matches!(noun.surface.as_str(), "針" | "時針" | "分針" | "秒針") {
            "撥回"
        } else {
            action
        };
        return Some(TimedResult {
            time: SettingTime::From(origin.0),
            object,
            value: value.0,
            action,
            site_name: None,
        });
    }
    let (a, b) = (first_value, second_value?);
    let a_time = groups[a.0].atoms.len() == 2 && is_daypart_surface(&a.1.surface);
    let b_time = groups[b.0].atoms.len() == 2 && is_daypart_surface(&b.1.surface);
    if a_time && b_time {
        return (source_time_head(noun)
            && a.0 < object
            && object < b.0
            && a.2.span.end_char <= noun.span.start_char
            && wo.span.end_char <= b.1.span.start_char
            && b.2.span.end_char == verb.span.start_char)
            .then_some(TimedResult {
                time: SettingTime::At(a.0),
                object,
                value: b.0,
                action,
                site_name: None,
            });
    }
    // With one time adjunct and a different に complement, 戻す returns
    // the object to that complement. The attachment, not a known place
    // name, identifies when the return happened in either word order.
    if lemma != "戻す" || a_time == b_time {
        return None;
    }
    let (when, site) = if a_time { (a.0, b) } else { (b.0, a) };
    let atoms = &groups[site.0].atoms;
    if atoms.iter().any(|p| p.is_locked) || source_time_head(site.1) {
        return None;
    }
    // The ordinary noun renderer supplies the destination. Use a more
    // specific Chinese locative only when the whole Japanese noun supports it.
    let name: String = atoms[..atoms.len() - 1]
        .iter()
        .map(|p| p.surface.as_str())
        .collect();
    let site_name = match name.as_str() {
        "棚" => Some("架上"),
        "本棚" | "書棚" => Some("書架上"),
        "机" => Some("桌上"),
        "箱" => Some("箱子裡"),
        "引き出し" => Some("抽屜裡"),
        "車庫" => Some("車庫"),
        _ => None,
    };
    use faberline_nlp::lexicon::{SemanticClass, lookup_ja};
    let person = match site.1.upos {
        Some(Upos::Pron | Upos::Propn) => is_motion_actor(site.1),
        Some(Upos::Noun) => is_demonstrative_person_noun(site.1),
        _ => false,
    };
    let action = if person {
        "還給"
    } else if site_name.is_some() {
        "放回"
    } else if is_existing_spatial_head(site.1)
        || lookup_ja(&site.1.surface).is_some_and(|e| e.sem_class == Some(SemanticClass::Location))
    {
        "帶回"
    } else if has_verb_reading(&noun.surface) {
        "改回"
    } else {
        "放回"
    };
    Some(TimedResult {
        time: SettingTime::At(when),
        object,
        value: site.0,
        action,
        site_name,
    })
}

fn is_transport_term(term: &str) -> bool {
    matches!(
        term,
        "電車" | "公車" | "巴士" | "計程車" | "新幹線" | "飛機" | "船" | "列車" | "地鐵"
    )
}

fn is_motion_actor(a: &Atom) -> bool {
    use faberline_nlp::lexicon::{SemanticClass, lookup_ja};
    if a.is_locked {
        return false;
    }
    match a.upos {
        Some(Upos::Pron) => {
            let base = ["たち", "達", "ら", "等"]
                .iter()
                .find_map(|suffix| a.surface.strip_suffix(suffix))
                .unwrap_or(&a.surface);
            matches!(base, "私" | "僕" | "俺" | "彼" | "彼女" | "あなた" | "君")
        }
        Some(Upos::Noun) => {
            is_demonstrative_person_noun(a)
                || lookup_ja(&a.surface).is_some_and(|hit| {
                    hit.upos_candidates.as_slice() == [Upos::Noun]
                        && hit.counterpart.as_deref().is_some_and(is_transport_term)
                })
        }
        Some(Upos::Propn) => lookup_ja(&a.surface).is_some_and(|hit| {
            hit.sem_class == Some(SemanticClass::Person)
                && hit.upos_candidates.as_slice() == [Upos::Propn]
        }),
        _ => false,
    }
}

// A source nominal can itself have a scheduled time or display a reading.
// Only inspect its own head and its direct compound; a person or clock in
// a possessor phrase does not change the identity of the subject.
fn source_time_head(a: &Atom) -> bool {
    use faberline_nlp::lexicon::{SemanticClass, lookup_ja};
    is_clock_atom(a)
        || (!a.is_locked
            && (is_daypart_surface(&a.surface)
                || matches!(
                    a.surface.as_str(),
                    "時間"
                        | "時刻"
                        | "日時"
                        | "日程"
                        | "日付"
                        | "時計"
                        | "針"
                        | "時針"
                        | "分針"
                        | "秒針"
                )
                || lookup_ja(&a.surface)
                    .is_some_and(|hit| hit.sem_class == Some(SemanticClass::Time))))
}

fn is_time_setting(a: &Atom, source: &[Atom]) -> bool {
    if a.is_locked {
        return is_clock_atom(a);
    }
    if source_time_head(a) {
        return true;
    }
    if a.upos == Some(Upos::Noun) && has_verb_reading(&a.surface) {
        return true;
    }
    matches!(
        a.surface.as_str(),
        "枠" | "帯" | "表" | "欄" | "盤" | "目盛り"
    ) && a.doc_tok_idx.is_some()
        && source.iter().any(|part| {
            part.head == a.doc_tok_idx
                && part.deprel == "compound"
                && !part.is_locked
                && (is_daypart_surface(&part.surface)
                    || has_verb_reading(&part.surface)
                    || matches!(part.surface.as_str(), "時間" | "時刻" | "時計"))
        })
}

// For 戻る and other verbs of movement or change, a subject can be the
// mover or the thing restored. An independent location or a human subject
// supports actual movement. An unrelated subject's modifier is insufficient.
fn has_motion_reading(clause: &[Bunsetsu], source: &[Atom], head: usize) -> bool {
    use faberline_nlp::lexicon::{SemanticClass, lookup_ja};
    let spatial_path = clause.iter().any(|b| {
        matches!(
            b.case_particle.as_deref(),
            Some("から" | "に" | "へ" | "まで")
        ) && b.atoms.iter().any(|a| {
            !a.is_locked
                && a.head == Some(head)
                && a.deprel == "obl"
                && (is_place_location_bunsetsu(std::slice::from_ref(a))
                    || lookup_ja(&a.surface)
                        .is_some_and(|hit| hit.sem_class == Some(SemanticClass::Location)))
        })
    });
    if spatial_path {
        return true;
    }
    source
        .iter()
        .filter(|a| a.head == Some(head) && a.deprel == "nsubj")
        .all(|a| {
            is_motion_actor(a)
                || (!a.is_locked
                    && a.upos == Some(Upos::Noun)
                    && !matches!(a.surface.as_str(), "たち" | "達" | "ら" | "等")
                    && !is_time_setting(a, source))
                || (!a.is_locked
                    && a.upos == Some(Upos::Noun)
                    && matches!(a.surface.as_str(), "たち" | "達" | "ら" | "等")
                    && a.doc_tok_idx.is_some()
                    && source.iter().any(|stem| {
                        stem.head == a.doc_tok_idx
                            && stem.deprel == "compound"
                            && stem.span.end_char == a.span.start_char
                            && is_motion_actor(stem)
                    }))
        })
}

// 夕方に働く, 夕方にする and 夕方に戻る can share the same case relation.
// Keep the time only when its own governor and that governor's other
// arguments support an event-time reading.
fn is_event_daypart(group: &Bunsetsu, clause: &[Bunsetsu], source: &[Atom]) -> bool {
    let [time, ni] = group.atoms.as_slice() else {
        return false;
    };
    let (Some(time_id), Some(head)) = (time.doc_tok_idx, time.head) else {
        return false;
    };
    if group.case_particle.as_deref() != Some("に")
        || !matches!(group.role, BunsetsuRole::IObj | BunsetsuRole::Obl)
        || group.sconj_tail.is_some()
        || group.sfp_tail.is_some()
        || time.is_locked
        || ni.is_locked
        || !is_daypart_surface(&time.surface)
        || !matches!(time.upos, Some(Upos::Noun | Upos::Adv))
        || !matches!(time.deprel.as_str(), "obl" | "advmod" | "iobj")
        || ni.surface != "に"
        || ni.upos != Some(Upos::Adp)
        || ni.deprel != "case"
        || ni.doc_tok_idx.is_none()
        || ni.head != Some(time_id)
        || time.span.end_char != ni.span.start_char
    {
        return false;
    }
    // Phrase rendering can join a lexical governor and する into one translated
    // atom. Read its original dependency and verify its span in this clause.
    let Some(verb) = source
        .iter()
        .find(|a| a.doc_tok_idx == Some(head) && !a.is_locked && a.upos == Some(Upos::Verb))
    else {
        return false;
    };
    if !clause
        .iter()
        .flat_map(|b| &b.atoms)
        .any(|a| a.span.start_char <= verb.span.start_char && verb.span.end_char <= a.span.end_char)
    {
        return false;
    }
    let lemma = verb.lemma.as_deref().unwrap_or(&verb.surface);
    if !has_verb_reading(lemma) || selects_time_value(lemma) {
        return false;
    }
    // With a transfer predicate the analyzer labels every に as iobj. The
    // separately attached theme or recipient supplies the transfer argument.
    if time.deprel == "iobj"
        && !source.iter().any(|a| {
            a.doc_tok_idx != Some(time_id)
                && a.head == Some(head)
                && matches!(a.deprel.as_str(), "obj" | "iobj")
        })
    {
        return false;
    }
    if clause.iter().any(|b| {
        b.case_particle.as_deref() == Some("から")
            && b.atoms.iter().any(|a| {
                is_clock_atom(a)
                    || (!a.is_locked
                        && a.head == Some(head)
                        && (is_daypart_surface(&a.surface)
                            || a.surface
                                .strip_suffix('時')
                                .and_then(clock_number)
                                .is_some_and(|h| h <= 24)
                            || faberline_nlp::lexicon::lookup_ja(&a.surface).is_some_and(|hit| {
                                hit.sem_class == Some(faberline_nlp::lexicon::SemanticClass::Time)
                            })))
            })
    }) {
        return false;
    }
    // 帰る expresses a return home. 戻る can instead restore a setting,
    // and 移る / 動く can move a scheduled value.
    if matches!(lemma, "戻る" | "移る" | "動く") && !has_motion_reading(clause, source, head)
    {
        return false;
    }
    // A replacement can take both an affected object and the new value,
    // as in Xを夕方に書き換える. Ordinary observation or writing has no
    // replacement argument, regardless of which noun is being observed.
    if matches!(lemma, "書き換える" | "置き換える" | "切り替える")
        && ni.span.end_char == verb.span.start_char
        && source.iter().any(|a| {
            a.head == Some(head) && a.deprel == "obj" && a.span.end_char <= time.span.start_char
        })
    {
        return false;
    }

    if verb.deprel == "root" {
        if lemma != "する" {
            return true;
        }
        // A separate する needs a following action noun: 夕方に買い物をする.
        // In 買い物を夕方にする the time can instead be the selected value.
        return clause.iter().any(|b| {
            let [action, wo] = b.atoms.as_slice() else {
                return false;
            };
            !action.is_locked
                && !wo.is_locked
                && action.deprel == "obj"
                && action.head == Some(head)
                && action.span.start_char >= ni.span.end_char
                && action.span.end_char == wo.span.start_char
                && wo.span.end_char == verb.span.start_char
                && wo.surface == "を"
                && wo.deprel == "case"
                && action.doc_tok_idx.is_some()
                && wo.head == action.doc_tok_idx
                && has_verb_reading(&action.surface)
                && !selects_time_value(&action.surface)
        });
    }
    let Some(aux_head) = verb.head.filter(|_| verb.deprel == "advcl") else {
        return false;
    };
    let Some(aux) = source.iter().find(|a| {
        a.doc_tok_idx == Some(aux_head)
            && !a.is_locked
            && a.upos == Some(Upos::Verb)
            && a.deprel == "root"
    }) else {
        return false;
    };
    match aux.lemma.as_deref() {
        // A bound verbal noun and its light verb form one lexical event.
        Some("する") => verb.span.end_char == aux.span.start_char && lemma != "する",
        // A linked て／で + いる expresses the aspect of the lexical event.
        Some("いる") => {
            lemma != "する"
                && source.iter().any(|a| {
                    !a.is_locked
                        && matches!(a.surface.as_str(), "て" | "で")
                        && a.deprel == "mark"
                        && a.head == Some(head)
                        && verb.span.end_char == a.span.start_char
                        && a.span.end_char == aux.span.start_char
                })
        }
        _ => false,
    }
}

fn predicates_has_passive(predicates: &[Bunsetsu]) -> bool {
    predicates.iter().any(|b| {
        b.atoms.iter().any(|a| {
            !a.is_locked
                && (matches!(
                    a.surface.as_str(),
                    "れる" | "られる" | "れ" | "られ" | "され" | "される" | "された"
                ) || a.surface.ends_with("され")
                    || a.surface.ends_with("される")
                    || a.surface.ends_with("された"))
        })
    })
}

#[cfg(test)]
mod kana_noun_rule_tests {
    use super::*;
    use crate::{TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

    fn atom(
        word: &str,
        start: usize,
        pos: Upos,
        index: usize,
        head: usize,
        relation: &str,
    ) -> Atom {
        Atom {
            surface: word.to_owned(),
            lemma: None,
            upos: Some(pos),
            span: Span {
                start_char: start,
                end_char: start + word.chars().count(),
            },
            doc_tok_idx: Some(index),
            head: Some(head),
            deprel: relation.to_owned(),
            is_locked: false,
        }
    }

    #[test]
    fn recovered_fragments_keep_the_whole_source_and_the_governor() {
        let head = atom("み", 4, Upos::Noun, 1, 7, "obj");
        let ka = atom("か", 5, Upos::Part, 2, 1, "discourse");
        let tail = atom("ん", 6, Upos::Aux, 3, 1, "aux");
        let case = atom("を", 7, Upos::Adp, 4, 1, "case");
        let mut fragments = [head, ka, tail, case];
        let (word, used) = recover_split_kana_noun(&fragments, 0).unwrap();
        assert_eq!((word.surface.as_str(), used), ("みかん", 3));
        assert_eq!((word.span.start_char, word.span.end_char), (4, 7));
        assert_eq!(
            (word.doc_tok_idx, word.head, word.deprel.as_str()),
            (Some(1), Some(7), "obj")
        );
        fragments[2].is_locked = true;
        assert!(recover_split_kana_noun(&fragments, 0).is_none());
        fragments[2].is_locked = false;
        fragments[3].head = Some(7);
        assert!(recover_split_kana_noun(&fragments, 0).is_none());

        let mut initial = [
            atom("は", 1, Upos::Adp, 1, 0, "case"),
            atom("か", 2, Upos::Part, 2, 0, "discourse"),
            atom("ばん", 3, Upos::Noun, 3, 5, "obj"),
            atom("を", 5, Upos::Adp, 4, 3, "case"),
        ];
        let (word, used) = recover_split_kana_noun(&initial, 1).unwrap();
        assert_eq!((word.surface.as_str(), used), ("かばん", 2));
        assert_eq!((word.span.start_char, word.span.end_char), (2, 5));
        // The same middle particle after an alternative nominal has no left boundary.
        initial[0] = atom("本", 1, Upos::Noun, 1, 3, "nmod");
        assert!(recover_split_kana_noun(&initial, 1).is_none());
        initial[0] = atom("は", 1, Upos::Adp, 1, 0, "case");
        initial[2].span.start_char += 1;
        assert!(recover_split_kana_noun(&initial, 1).is_none());
    }

    fn translate(source: &str) -> String {
        analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap()
        .target_text
    }

    #[test]
    fn only_the_real_final_question_changes_the_clause_mood() {
        for source in [
            "姉はかばんを持っています。",
            "みかんを三つ買いました。",
            "かばんを買います。",
        ] {
            assert!(!translate(source).contains('嗎'), "{source}");
        }
        for source in ["姉はかばんを持っていますか。", "ここに座ってもいいですか。"]
        {
            assert!(translate(source).ends_with("嗎？"), "{source}");
        }
    }

    #[test]
    fn question_mark_belongs_to_the_terminated_clause() {
        let quoted = translate("彼は「行きますか。」と言った。");
        assert!(
            quoted.contains("嗎？」") && quoted.ends_with('。'),
            "{quoted}"
        );
        let outside = translate("彼は「行きます。」と聞きますか。");
        assert!(
            outside.contains("。」") && outside.ends_with("嗎？"),
            "{outside}"
        );
        for source in [
            "本か雑誌を買います。",
            "誰が来るか分かりません。",
            "「行きますか」と聞いた。",
        ] {
            assert!(!translate(source).contains('？'), "{source}");
        }
        assert!(translate("駅に行きますか！").ends_with("嗎！"));
        assert!(translate("駅に行きますか？").ends_with("嗎？"));
    }
}

#[cfg(test)]
mod clock_rule_tests {
    use super::{
        Atom, CLOCK_ATOM, OPAQUE_PERIOD_CLOCK, clock_number, is_clock_atom, normalize_clock_atoms,
        render_clock_surface,
    };
    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
    };
    use faberline_nlp::doc::model::Backend;
    use faberline_nlp::{Span, doc::Upos};
    use std::collections::BTreeMap;

    fn atom(surface: &str, start: usize, locked: bool) -> Atom {
        Atom {
            span: Span {
                start_char: start,
                end_char: start + surface.chars().count(),
            },
            surface: surface.to_owned(),
            lemma: None,
            upos: Some(Upos::Noun),
            doc_tok_idx: None,
            head: None,
            deprel: String::new(),
            is_locked: locked,
        }
    }

    #[test]
    fn clock_numbers_and_half_are_bounded() {
        assert_eq!(clock_number("二十三"), Some(23));
        assert_eq!(clock_number("２４"), Some(24));
        assert_eq!(clock_number("二十五"), None);
        assert_eq!(
            render_clock_surface("午後五時半に"),
            Some(("下午五點半".to_owned(), true))
        );
        assert_eq!(
            render_clock_surface("二十四時"),
            Some(("二十四點".to_owned(), false))
        );
        assert_eq!(render_clock_surface("二十四時半"), None);
        for non_clock in ["二時間半", "二時限", "半分", "先生に", "五時半分"] {
            assert_eq!(render_clock_surface(non_clock), None, "{non_clock}");
        }
    }

    #[test]
    fn noon_midnight_and_uncertain_twelve_half_remain_distinct() {
        assert_eq!(
            render_clock_surface("午前十二時"),
            Some(("中午十二點".to_owned(), false))
        );
        assert_eq!(
            render_clock_surface("午後十二時"),
            Some(("午夜十二點".to_owned(), false))
        );
        assert_eq!(
            render_clock_surface("午前零時"),
            Some(("午夜零點".to_owned(), false))
        );
        assert_eq!(
            render_clock_surface("午後零時"),
            Some(("中午十二點".to_owned(), false))
        );
        let ambiguous = render_clock_surface("午前十二時半").unwrap().0;
        assert!(ambiguous.contains("中午") && ambiguous.contains("凌晨"));
    }

    #[test]
    fn split_hour_and_particle_become_one_clock_without_swallowing_recipient() {
        let input = vec![
            atom("二", 0, false),
            atom("十", 1, false),
            atom("三時", 2, false),
            atom("に", 4, false),
            atom("先生", 5, false),
        ];
        let got = normalize_clock_atoms(&input);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].surface, "二十三點");
        assert_eq!(got[0].lemma.as_deref(), Some(CLOCK_ATOM));
        assert_eq!(got[1].surface, "に");
        assert_eq!(got[2].surface, "先生");
        let near = vec![atom("二", 0, false), atom("時限", 1, false)];
        assert_eq!(normalize_clock_atoms(&near).len(), 2);
    }

    #[test]
    fn protected_hour_needs_visible_period_and_keeps_unknown_content() {
        let mut protected = atom("[[PH0]]", 2, true);
        protected.span.end_char = 3; // Span is in the source, not placeholder bytes.
        let got = normalize_clock_atoms(&[
            atom("午後", 0, false),
            protected.clone(),
            atom("時半", 3, false),
            atom("に", 5, false),
        ]);
        assert_eq!(got[1].surface, "[[PH0]]時半");
        assert_eq!(got[1].lemma.as_deref(), Some(OPAQUE_PERIOD_CLOCK));
        assert_eq!(got[2].surface, "に");
        let split = normalize_clock_atoms(&[
            atom("午後", 0, false),
            protected.clone(),
            atom("時", 3, false),
            atom("半", 4, false),
        ]);
        assert_eq!(split[1].surface, "[[PH0]]時半");
        assert_eq!(split[1].lemma.as_deref(), Some(OPAQUE_PERIOD_CLOCK));
        let untouched = normalize_clock_atoms(&[protected.clone(), atom("時半", 3, false)]);
        assert_eq!(untouched[0].surface, "[[PH0]]");
        assert_eq!(untouched[0].lemma, None);
        assert!(untouched.iter().all(|a| !is_clock_atom(a)));
        let split_unknown =
            normalize_clock_atoms(&[protected, atom("時", 3, false), atom("半", 4, false)]);
        assert!(split_unknown.iter().all(|a| !is_clock_atom(a)));
    }

    #[test]
    fn bare_kanji_ichiji_keeps_ambiguity_but_clear_clocks_still_work() {
        for ambiguous in ["一時", "一時に"] {
            assert_eq!(render_clock_surface(ambiguous), None);
        }
        for atoms in [
            vec![atom("一時", 0, false)],
            vec![atom("一", 0, false), atom("時", 1, false)],
        ] {
            let output = normalize_clock_atoms(&atoms);
            assert!(output.iter().all(|a| !is_clock_atom(a)));
            assert_eq!(
                output
                    .iter()
                    .map(|a| a.surface.as_str())
                    .collect::<String>(),
                "一時"
            );
        }
        assert_eq!(
            render_clock_surface("午後一時に"),
            Some(("下午一點".to_owned(), true))
        );
        assert_eq!(
            render_clock_surface("一時半"),
            Some(("一點半".to_owned(), false))
        );
        assert_eq!(
            render_clock_surface("二時"),
            Some(("兩點".to_owned(), false))
        );
        let qualified = normalize_clock_atoms(&[
            atom("午後", 0, false),
            atom("一", 2, false),
            atom("時", 3, false),
        ]);
        assert_eq!(qualified.len(), 1);
        assert_eq!(qualified[0].surface, "下午一點");
        assert!(is_clock_atom(&qualified[0]));
    }

    #[test]
    fn api_keeps_ichiji_ambiguous_and_supports_explicit_clock() {
        let glossary = TranslationGlossary::new();
        let temporary = public_translate("一時停止します。", &glossary);
        assert!(
            !temporary.target_text.contains("一點"),
            "{}",
            temporary.target_text
        );
        let clock = public_translate("会議は午後一時に始まります。", &glossary);
        assert!(
            clock.target_text.contains("下午一點"),
            "{}",
            clock.target_text
        );
        let unknown = public_translate("会議は`予定`時半に始まります。", &glossary);
        assert_eq!(unknown.opaque_spans.len(), 1);
        assert!(
            unknown.target_text.contains("`予定`"),
            "{}",
            unknown.target_text
        );
        assert!(
            !unknown.target_text.contains("點"),
            "{}",
            unknown.target_text
        );
        let qualified_unknown = public_translate("会議は午後`予定`時半に始まります。", &glossary);
        assert_eq!(qualified_unknown.opaque_spans.len(), 1);
        assert!(
            qualified_unknown.target_text.contains("午後`予定`時半"),
            "{}",
            qualified_unknown.target_text
        );
        assert!(
            !qualified_unknown.target_text.contains("點"),
            "{}",
            qualified_unknown.target_text
        );
    }

    #[test]
    fn split_suffixes_do_not_turn_nonclocks_into_clock_prefixes() {
        for suffix in ["間", "限", "半分", "３０分"] {
            let input = vec![atom("二時", 0, false), atom(suffix, 2, false)];
            let result = normalize_clock_atoms(&input);
            assert!(
                result
                    .iter()
                    .all(|a| a.lemma.as_deref() != Some(CLOCK_ATOM)),
                "{suffix}"
            );
            assert_eq!(
                result
                    .iter()
                    .map(|a| a.surface.as_str())
                    .collect::<String>(),
                format!("二時{suffix}")
            );
        }
    }

    fn public_translate(source: &str, glossary: &TranslationGlossary) -> crate::TranslatedDocument {
        analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            glossary,
            None,
            Backend::Onnx,
        )
        .unwrap()
    }

    #[test]
    fn clock_is_not_a_motion_destination() {
        let output =
            public_translate("午後二時に教室に行きます。", &TranslationGlossary::new()).target_text;
        assert!(
            output.contains("下午兩點") && output.contains("教室"),
            "{output}"
        );
        assert!(!output.contains("到下午兩點"), "{output}");
    }

    #[test]
    fn calendar_days_preserve_deadlines_instants_and_question_scope() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("何日までに本を読みますか。", "最晚幾號讀書？"),
            ("何曜日に本を読みますか。", "星期幾讀書？"),
            ("五月五日までに本を読みますか。", "最晚在五月五日讀書嗎？"),
            (
                "十二月二十五日に本を読みますか。",
                "在十二月二十五日讀書嗎？",
            ),
            ("５月２５日に本を読みます。", "在5月25日讀書。"),
            ("五日間、毎日本を読みます。", "五天，每天讀書。"),
            (
                "二〇二四年四月一日に入社した。",
                "在二〇二四年四月一日進入公司了。",
            ),
            (
                "私は四月五日に駅に行きませんでした。",
                "我在四月五日沒有去車站。",
            ),
            (
                "四月五日までに手紙を書きませんでした。",
                "到四月五日為止沒有寫信。",
            ),
            (
                "何曜日までに駅に着くことを決めましたか。",
                "決定了最晚星期幾抵達到車站嗎？",
            ),
        ] {
            assert_eq!(
                public_translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn calendar_days_with_a_count_distinguish_a_rate_from_one_dated_event() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "毎日ではなく、二日に一度だけ薬を飲みます。",
                "不是每天，而是每兩天只喝一次藥。",
            ),
            ("三日に一回だけ走ります。", "每三天只跑一次。"),
            ("一日に三回薬を飲みました。", "每天喝了三次藥。"),
            ("何日に一度だけ駅へ行きますか。", "每隔幾天往車站只去一次？"),
            ("二日に一度だけ駅へ行きますか。", "每兩天往車站只去一次嗎？"),
            ("二日に一度、郵便局へ行きます。", "每兩天一次，往郵局去。"),
            (
                "六月二日に一度だけ手紙を書きました。",
                "在六月二日只寫了一次信。",
            ),
            (
                "五月の二日に一度だけ駅へ行きます。",
                "在五月的二號往車站只去一次。",
            ),
            ("二日間、家で休みました。", "兩天，在家休息了。"),
        ] {
            assert_eq!(
                public_translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn counted_rates_keep_the_amount_with_the_book_or_arriving_person() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "毎日ではなく、二日に一冊だけ本を読みます。",
                "不是每天，而是每兩天只讀一本書。",
            ),
            (
                "毎日ではなく、二日に本を一冊だけ読みます。",
                "不是每天，而是每兩天只讀一本書。",
            ),
            (
                "毎日ではなく、二日に一人ずつお客さんが来ます。",
                "不是每天，而是每兩天來一位客人。",
            ),
            (
                "毎朝ではなく、三日に一人ずつ学生が来ます。",
                "不是每天早上，而是每三天來一位學生。",
            ),
            ("三日に二枚ずつ写真を撮ります。", "每三天拍攝兩張照片。"),
            (
                "六月二日に一冊だけ本を読みました。",
                "在六月二日只讀了一本書。",
            ),
            (
                "今月の二日に一冊だけ本を読みました。",
                "在這個月的二號只讀了一本書。",
            ),
            (
                "六月二日に一人だけお客さんが来ました。",
                "在六月二日只來了一位客人。",
            ),
        ] {
            assert_eq!(
                public_translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
        for source in [
            "二日に一冊だけ本を読みました。",
            "何日に一人のお客さんが来ますか。",
            "二日に一冊ずつ二人が本を読みます。",
        ] {
            let output = public_translate(source, &glossary).target_text;
            assert!(
                !output.contains('號') && !output.contains("每兩天"),
                "{source}: {output}"
            );
        }
    }

    #[test]
    fn opaque_count_keeps_the_day_ambiguous_unless_a_month_is_present() {
        let mut glossary = TranslationGlossary::new();
        glossary.domain_terms.push(GlossaryEntry {
            canonical: "一度".to_owned(),
            surfaces: vec!["一度".to_owned()],
            targets: BTreeMap::from([(TargetVariant::ZhHant, "保留次數".to_owned())]),
            ..GlossaryEntry::default()
        });
        for (source, dated) in [
            ("二日に一度だけ薬を飲みます。", false),
            ("六月二日に一度だけ薬を飲みます。", true),
        ] {
            let output = public_translate(source, &glossary);
            assert_eq!(output.glossary_hits.len(), 1, "{source}");
            assert_eq!(output.target_text.matches("保留次數").count(), 1);
            assert_eq!(output.target_text.contains("在六月二日"), dated);
            assert!(
                !output.target_text.contains("二號") && !output.target_text.contains("每兩天"),
                "{}",
                output.target_text
            );
        }
    }

    #[test]
    fn counted_arrivals_modify_the_relative_noun_and_keep_the_matrix_subject() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "六月二日と三日に一人ずつ学生が来る教室を先生が選びました。",
                "老師選擇了在六月二日和三日各來一位學生的教室。",
            ),
            (
                "七月三日と五日に二人ずつ患者が来る病院を見学しました。",
                "參觀了在七月三日和五日各來兩位患者的醫院。",
            ),
            (
                "六月二日と三日に一人ずつ学生が来ました。",
                "在六月二日和三日各來了一位學生。",
            ),
            (
                "六月二日と三日にそれぞれ一冊ずつ本を読んだ学生が来ました。",
                "在六月二日和三日各讀了一本書的學生來了。",
            ),
        ] {
            assert_eq!(
                public_translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn reported_calendar_events_keep_the_speaker_separate_from_their_subject() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "六月二日と三日に一人ずつ学生が来たと先生が答えました。",
                "老師回答，在六月二日和三日各來了一位學生。",
            ),
            (
                "先生は六月二日と三日に一人ずつ学生が来たと答えました。",
                "老師回答，在六月二日和三日各來了一位學生。",
            ),
            (
                "毎日ではなく、三日に一冊ずつ本を借りたと先生が答えました。",
                "老師回答，不是每天，而是每三天借用了一本書。",
            ),
            (
                "学生は、毎日ではなく二日に一冊だけ本を読んだと答えました。",
                "學生回答，不是每天，而是每兩天只讀了一本書。",
            ),
            (
                "二日に一冊ずつ本を借りたと先生は答えませんでした。",
                "老師沒有回答說每兩天借用了一本書。",
            ),
            (
                "二日に一冊ずつ本を借りたと先生が答えましたか。",
                "老師是否回答說每兩天借用了一本書？",
            ),
        ] {
            assert_eq!(
                public_translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
        let conditional = public_translate(
            "六月二日と三日に一人ずつ学生が来ると先生が喜びました。",
            &glossary,
        );
        assert!(
            conditional.target_text.starts_with("當")
                && conditional.target_text.ends_with("老師高興了。")
        );
    }

    #[test]
    fn coordinated_dates_distribute_one_event_on_each_date() {
        let glossary = TranslationGlossary::new();
        for (jp, zh) in [
            (
                "六月二日と三日に一人ずつ学生が来ました。",
                "在六月二日和三日各來了一位學生。",
            ),
            (
                "六月二日と七月三日に一人ずつ学生が来ました。",
                "在六月二日和七月三日各來了一位學生。",
            ),
            (
                "七月三日と五日に一冊ずつ本を借りました。",
                "在七月三日和五日各借用了一本書。",
            ),
            (
                "今月の二日と三日に一冊ずつ本を買いました。",
                "在這個月二日和三日各買了一本書。",
            ),
            (
                "六月二日と三日、それぞれ一冊ずつ本を借りました。",
                "在六月二日和三日各借用了一本書。",
            ),
            (
                "毎日ではなく、三日に一人ずつ学生が来ました。",
                "不是每天，而是每三天來了一位學生。",
            ),
        ] {
            assert_eq!(public_translate(jp, &glossary).target_text, zh, "{jp}");
        }
    }

    #[test]
    fn habitual_quantities_stay_with_their_own_predicate() {
        let glossary = TranslationGlossary::new();
        for jp in [
            "普段は三日に一杯コーヒーを飲みます。",
            "ふだんは三日に一杯コーヒーを飲みます。",
        ] {
            assert_eq!(
                public_translate(jp, &glossary).target_text,
                "平常每三天喝一杯咖啡。"
            );
        }
        assert_eq!(
            public_translate(
                "毎日ではなく、二日に一冊だけ本を返した学生が来ました。",
                &glossary
            )
            .target_text,
            "不是每天，而是每兩天只還給了一本書的學生來了。"
        );
        assert_eq!(
            public_translate(
                "毎日ではなく、三日に二冊だけ本を読んだ学生が来ました。",
                &glossary
            )
            .target_text,
            "不是每天，而是每三天只讀了兩本書的學生來了。"
        );
        let ambiguous = public_translate("三日に一杯コーヒーを飲みます。", &glossary);
        assert!(
            !ambiguous.target_text.contains("每三天") && !ambiguous.target_text.contains("三號")
        );
    }

    #[test]
    fn a_single_day_restricts_only_the_matching_day_in_the_next_sentence() {
        let glossary = TranslationGlossary::new();
        let jp = "試験は六月二日だけです。二日に学生が一人ずつ来ます。";
        assert_eq!(
            public_translate(jp, &glossary).target_text,
            "試驗只在六月二日。學生在六月二日一位接一位地來。"
        );
        for jp in [
            "試験は六月二日だけです。三日に学生が一人ずつ来ます。",
            "試験は六月二日だけです。二日に一度だけ薬を飲みます。",
            "試験は六月二日だけです。\n\n二日に学生が一人ずつ来ます。",
        ] {
            let text = public_translate(jp, &glossary).target_text;
            assert_eq!(text.matches("六月二日").count(), 1, "{jp}: {text}");
        }
        let mut glossary = TranslationGlossary::new();
        glossary.domain_terms.push(GlossaryEntry {
            canonical: "六月二日".to_owned(),
            surfaces: vec!["六月二日".to_owned()],
            targets: BTreeMap::from([(TargetVariant::ZhHant, "特別日".to_owned())]),
            ..GlossaryEntry::default()
        });
        let locked = public_translate(jp, &glossary);
        assert_eq!(locked.target_text.matches("特別日").count(), 1);
        assert!(!locked.target_text.contains("六月二日"));
        let coordinated = public_translate("六月二日と三日に一人ずつ学生が来ました。", &glossary);
        assert_eq!(coordinated.target_text.matches("特別日").count(), 1);
        assert!(!coordinated.target_text.contains("每三天"));
    }

    #[test]
    fn duration_and_selected_values_do_not_become_calendar_dates() {
        let glossary = TranslationGlossary::new();
        for source in [
            "何日に一回走りますか。",
            "二日に一回走ります。",
            "一日に三人来ます。",
            "何日で本を読みますか。",
            "五日後に行きます。",
            "彼は何日に設定しましたか。",
        ] {
            let text = public_translate(source, &glossary).target_text;
            assert!(
                !text.contains('號') && !text.contains("最晚"),
                "{source}: {text}"
            );
        }
    }

    #[test]
    fn deadline_keeps_the_time_separate_from_recipients_and_destinations() {
        let glossary = TranslationGlossary::new();
        let translate = |s| public_translate(s, &glossary).target_text;
        assert_eq!(
            translate("私は六時までに手紙を書きます。"),
            "我最晚在六點寫信。"
        );
        assert_eq!(
            translate("金曜日までに先生に手紙を書きます。"),
            "最晚在星期五給老師寫信。"
        );
        assert_eq!(translate("週末までに本を読みます。"), "最晚在週末讀書。");
        assert_eq!(
            translate("六時までに駅まで走ります。"),
            "最晚在六點跑到車站。"
        );
        assert_eq!(
            translate("午後三時までに手紙を書きませんでした。"),
            "到下午三點為止沒有寫信。"
        );
    }

    #[test]
    fn deadline_requires_two_case_particles_on_the_same_time_expression() {
        let glossary = TranslationGlossary::new();
        let translate = |s| public_translate(s, &glossary).target_text;
        assert_eq!(translate("私は六時に手紙を書きます。"), "我在六點寫信。");
        assert_eq!(translate("私は先生に手紙を書きます。"), "我給老師寫信。");
        assert_eq!(
            translate("私は毎朝駅までバスで行きます。"),
            "我每天早上坐公車到車站。"
        );
        for source in [
            "私は六時まで日本語を勉強します。",
            "学校までに公園があります。",
        ] {
            let result = translate(source);
            assert!(!result.contains("最晚"), "{source}: {result}");
        }
    }

    #[test]
    fn daypart_deadlines_keep_the_endpoint_and_modal_scope() {
        let glossary = TranslationGlossary::new();
        let translate = |s| public_translate(s, &glossary).target_text;
        for (source, expected) in [
            ("正午までに本を読みます。", "最晚在正午讀書。"),
            ("夕方までに駅に帰ります。", "最晚在傍晚回到車站。"),
            ("私は夕方までに手紙を書けません。", "我沒辦法在傍晚前寫信。"),
            ("深夜までに手紙を書きませんでした。", "到深夜為止沒有寫信。"),
            ("六時までに手紙を書けますか。", "可以在六點前寫信嗎？"),
            ("六時まで本を読みます。", "讀書到六點。"),
            ("六時までに本を読みます。", "最晚在六點讀書。"),
            ("夕方まで本を読みます。", "讀書到傍晚。"),
            ("六時まで手紙を書けません。", "到六點為止不能寫信。"),
            ("夕方に本を読みます。", "傍晚讀書。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn ability_scopes_over_meeting_the_deadline() {
        let glossary = TranslationGlossary::new();
        let translate = |s| public_translate(s, &glossary).target_text;
        for (source, expected) in [
            ("私は六時までに手紙を書けません。", "我沒辦法在六點前寫信。"),
            (
                "私は金曜日までに手紙を書けない。",
                "我沒辦法在星期五截止前寫信。",
            ),
            ("六時までに手紙を書けませんでした。", "沒能在六點前寫信。"),
            (
                "私は明日までに書類を提出できません。",
                "我沒辦法在明天截止前提出書類。",
            ),
            ("六時までに手紙を書けます。", "可以在六點前寫信。"),
            ("六時に手紙を書けません。", "在六點不能寫信。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn explicit_ability_keeps_the_deadline_inside_the_action() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "私は六時までに会場に来ることができません。",
                "我沒辦法在六點前來到會場。",
            ),
            (
                "六時までに会場に来ることができません。",
                "沒辦法在六點前來到會場。",
            ),
            (
                "誰が六時までに会場に来ることができますか。",
                "誰可以在六點前來到會場？",
            ),
            (
                "ハルコは明日までに帰ることができません。",
                "ハルコ沒辦法在明天截止前回去。",
            ),
            (
                "明日までに手紙を書くことができませんでした。",
                "沒能在明天截止前寫信。",
            ),
            (
                "明日までに手紙を書くことができますか。",
                "可以在明天截止前寫信嗎？",
            ),
            (
                "明日までに漢字を書くことが出来ません。",
                "沒辦法在明天截止前寫漢字。",
            ),
            (
                "六月までに書類を提出することはできません。",
                "沒辦法在六月截止前提出書類。",
            ),
            (
                "12月までに帰ることができません。",
                "沒辦法在12月截止前回去。",
            ),
            (
                "六時までに手紙を書くこともできません。",
                "也沒辦法在六點前寫信。",
            ),
        ] {
            assert_eq!(
                public_translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
        for source in [
            "六時に来ることができません。",
            "六時まで会社に来ることができません。",
            "私は六時までに会場に来ることを決めました。",
            "明日までに雨が降ることがあります。",
            "私は六時までに会場に来られません。",
            "明日までに言うことができた人はいますか。",
        ] {
            let text = public_translate(source, &glossary).target_text;
            assert!(
                !text.contains("沒辦法在") && !text.contains("得以在"),
                "{source}: {text}"
            );
        }
    }

    #[test]
    fn indefinite_arguments_keep_the_deadline_statement_or_question() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "誰かが六時までに会場に来ることができません。",
                "某個人沒辦法在六點前來到會場。",
            ),
            (
                "誰かは六時までに会場に来ることができますか。",
                "某個人可以在六點前來到會場嗎？",
            ),
            (
                "彼は六時までに何か読むことができますか。",
                "他可以在六點前讀點什麼嗎？",
            ),
            (
                "明日までに何か作業をすることができます。",
                "可以在明天截止前做一些作業。",
            ),
        ] {
            assert_eq!(
                public_translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn an_open_deadline_asks_for_the_time_of_its_own_action() {
        let glossary = TranslationGlossary::new();
        let translate = |source: &str| public_translate(source, &glossary).target_text;
        for (source, expected) in [
            ("いつまでに手紙を書きますか。", "最晚在什麼時候寫信？"),
            ("何時までに書きましょうか。", "最晚在幾點寫好呢？"),
            (
                "いつまでに先生に手紙を渡しますか。",
                "最晚在什麼時候給老師遞給信？",
            ),
            ("何時までに駅に着きますか。", "最晚在幾點抵達到車站？"),
            ("何時までに手紙を書けますか。", "最晚幾點可以寫信？"),
            (
                "私は六時までに駅へ行くことができます。",
                "我可以在六點之前往車站去。",
            ),
            (
                "六時までに公園へ行くことができませんか。",
                "沒辦法在六點之前往公園去嗎？",
            ),
            (
                "何時までに駅へ行くことができますか。",
                "最晚幾點可以往車站去？",
            ),
            (
                "何時までに公園へ行くことができますか。",
                "最晚幾點可以往公園去？",
            ),
            (
                "何時までに帰ることができませんでしたか。",
                "沒能在幾點之前回去？",
            ),
            (
                "いつまでに手紙を書くことができますか。",
                "最晚什麼時候可以寫信？",
            ),
            // The か asks whether a decision occurred. The deadline is on reading.
            (
                "何時までに本を読むことを決めましたか。",
                "決定了最晚在幾點讀書嗎？",
            ),
            ("六時までに本を読みますか。", "最晚在六點讀書嗎？"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        for source in [
            "何時に本を読みますか。",
            "いつまで本を読みますか。",
            "いつかまでに帰りますか。",
            "駅までに銀行がありますか。",
        ] {
            assert!(!translate(source).contains("最晚"), "{source}");
        }
    }

    #[test]
    fn a_deadline_stays_with_the_verb_that_governs_its_time() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "明日までに手紙を書くことを決めました。",
                "決定了最晚在明天寫信。",
            ),
            (
                "手紙を明日までに書くことを決めました。",
                "決定了最晚在明天寫信。",
            ),
            (
                "手紙を書くことを明日までに決めました。",
                "最晚在明天決定了寫信。",
            ),
            (
                "明日までに私は手紙を書くことを決めました。",
                "我決定了最晚在明天寫信。",
            ),
            (
                "明日までに手紙を書くことを五時までに決めました。",
                "最晚在五點決定了最晚在明天寫信。",
            ),
            (
                "明日までに手紙を書く人を探しています。",
                "正在尋找最晚在明天寫信的人。",
            ),
            (
                "手紙を書く人を明日までに探しています。",
                "最晚在明天正在尋找寫信的人。",
            ),
        ] {
            assert_eq!(
                public_translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn a_detached_negative_does_not_turn_the_cutoff_into_an_interval() {
        let glossary = TranslationGlossary::new();
        for (source, cutoff) in [
            ("六時までに駅に来られません。", "最晚在六點"),
            ("明日までに映画を見られません。", "最晚在明天"),
        ] {
            let text = public_translate(source, &glossary).target_text;
            assert!(
                text.contains(cutoff) && !text.contains("為止") && !text.contains("沒辦法"),
                "{source}: {text}"
            );
        }
        assert_eq!(
            public_translate("六時までに会場に来ませんでした。", &glossary).target_text,
            "到六點為止沒有來到會場。"
        );
    }

    #[test]
    fn glossary_clock_components_keep_literal_and_order() {
        let source = "会議は午後二時半に始まります。";
        for (surface, target) in [
            ("午後二時半", "下午兩點半"),
            ("二", "二"),
            ("時", "時"),
            ("半", "半"),
            ("午後", "午後"),
        ] {
            let mut glossary = TranslationGlossary::new();
            glossary.domain_terms.push(GlossaryEntry {
                canonical: surface.to_owned(),
                surfaces: vec![surface.to_owned()],
                targets: BTreeMap::from([(TargetVariant::ZhHant, target.to_owned())]),
                ..GlossaryEntry::default()
            });
            let doc = public_translate(source, &glossary);
            let text = &doc.target_text;
            assert_eq!(doc.glossary_hits.len(), 1, "{surface}: {text}");
            assert_eq!(text.matches(target).count(), 1, "{surface}: {text}");
            assert!(text.find(target) < text.find("開始"), "{surface}: {text}");
            assert!(!text.contains("給"), "{surface}: {text}");
            if surface == "二" {
                assert!(text.contains("二點半"), "{text}");
            }
            if surface == "時" {
                assert!(text.contains("兩時半"), "{text}");
            }
            if surface == "半" {
                assert!(text.contains("兩點半"), "{text}");
            }
        }
    }

    #[test]
    fn opaque_clock_position_and_opaque_recipient_are_distinct() {
        let glossary = TranslationGlossary::new();
        let full = public_translate("会議は`午後二時半`に始まります。", &glossary);
        assert_eq!(full.opaque_spans.len(), 1);
        assert!(full.target_text.contains("`午後二時半`"));
        assert!(
            !full.target_text.contains("給`午後二時半`"),
            "{}",
            full.target_text
        );
        let partial = public_translate("会議は午後`二`時半に始まります。", &glossary);
        assert!(
            partial.target_text.contains("午後`二`時半"),
            "{}",
            partial.target_text
        );
        assert!(
            partial.target_text.find("午後`二`時半") < partial.target_text.find("開始"),
            "{}",
            partial.target_text
        );
        let period = public_translate("会議は`午後`二時半に始まります。", &glossary);
        assert!(
            period.target_text.contains("`午後`兩點半"),
            "{}",
            period.target_text
        );
        assert!(
            period.target_text.find("`午後`兩點半") < period.target_text.find("開始"),
            "{}",
            period.target_text
        );
        let topic = public_translate("`会議は`午後二時半に始まります。", &glossary);
        assert!(
            topic.target_text.find("`会議は`") < topic.target_text.find("下午兩點半"),
            "{}",
            topic.target_text
        );
        let recipient = public_translate("`先生`に手紙を書きます。", &glossary);
        assert!(
            recipient.target_text.contains("給`先生`"),
            "{}",
            recipient.target_text
        );
    }
}

#[cfg(test)]
mod motion_endpoint_tests {
    use super::{
        Atom, Bunsetsu, BunsetsuRole, CLOCK_ATOM, OPAQUE_PERIOD_CLOCK, is_plain_clock_ni,
        physical_goal_head,
    };
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;
    use faberline_nlp::{Span, doc::Upos};

    fn translate(source: &str) -> String {
        let mut options = TranslateOptions::new(TargetVariant::ZhHant);
        options.source_hint = Some(TargetVariant::Ja);
        analyze_and_translate(
            source,
            &options,
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .expect("CPU rules translate the control sentence")
        .target_text
    }

    #[test]
    fn dayparts_mark_event_time_without_displacing_destinations_or_recipients() {
        for (source, expected) in [
            ("私は夕方に帰ります。", "我傍晚回去。"),
            ("お母さんは夕方に家に帰ります。", "母親傍晚回到家。"),
            ("夕方に図書館で勉強します。", "傍晚在圖書館學習。"),
            ("私は夕方に帰りません。", "我傍晚不回去。"),
            ("私は早朝に家に帰ります。", "我清晨回到家。"),
            ("私は夕方に本を読みます。", "我傍晚讀書。"),
            ("私は夕方に働きます。", "我傍晚工作。"),
            ("夕方にご飯を食べます。", "傍晚吃飯。"),
            ("夕方に働きませんでした。", "傍晚沒有工作。"),
            ("私は夕方に先生に電話します。", "我傍晚給老師打電話。"),
            ("夕方に散歩します。", "傍晚散步。"),
            ("会議が夕方に始まります。", "會議傍晚開始。"),
            ("私は夕方に家に戻りました。", "我傍晚回來到家了。"),
            ("私は夕方を待ちます。", "我等待傍晚。"),
            ("夕方から図書館で勉強します。", "從傍晚在圖書館學習。"),
            ("私は先生に手紙を書きます。", "我給老師寫信。"),
            ("私は`夕方`に帰ります。", "我回到`夕方`。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn restoring_a_schedule_keeps_its_time_endpoint() {
        let interval = translate("集合時間が夜から夕方に戻りました。");
        let (before, after) = interval.split_once("從夜晚").unwrap();
        assert!(
            before.contains("集合時間") && after.contains("到傍晚"),
            "{interval}"
        );
        let restored = translate("会議の時間が夕方に戻りました。");
        assert!(
            restored.contains("會議的時間") && restored.contains("到傍晚"),
            "{restored}"
        );
    }

    #[test]
    fn a_direct_object_does_not_change_when_one_checks_or_writes_it() {
        for (source, expected) in [
            ("私は夕方に名前を確認しました。", "我傍晚確認了名字。"),
            ("私は夕方に時間を確認しました。", "我傍晚確認了時間。"),
            (
                "私は夕方に会議の日程を確認しました。",
                "我傍晚確認了會議的日程。",
            ),
            ("私は夕方に名前を書きました。", "我傍晚寫了名字。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        let date = translate("私は夕方に日付を書きました。");
        assert!(
            date.starts_with("我傍晚寫了") && (date.contains("日期") || date.contains("日付")),
            "{date}"
        );
        let transfer = translate("私は夕方に友達に本を渡します。");
        assert!(
            transfer.starts_with("我傍晚")
                && transfer.contains("朋友")
                && transfer.contains("書")
                && transfer.contains("遞給"),
            "{transfer}"
        );
    }

    #[test]
    fn the_same_return_verb_preserves_physical_and_time_endpoints() {
        for (source, endpoint) in [
            ("時計の針が五時から正午に戻りました。", "從五點回來到正午"),
            (
                "午後一時の時計を一時間戻すと、針が正午に戻りました。",
                "針回來到正午",
            ),
            ("夜の放送枠が夕方に戻りました。", "回來到傍晚"),
        ] {
            let text = translate(source);
            assert!(text.contains(endpoint), "{source}: {text}");
        }
        assert_eq!(
            translate("兄は夕方に病院から戻りました。"),
            "哥哥傍晚從醫院回來了。"
        );
    }

    #[test]
    fn personal_plurals_and_vehicles_can_return_at_a_time() {
        for (source, expected) in [
            ("私たちは夕方に戻りました。", "我們傍晚回來了。"),
            ("先生たちが夕方に戻りました。", "老師們傍晚回來了。"),
            ("バスは夕方に戻りました。", "公車傍晚回來了。"),
            ("両親は正午に帰りました。", "父母正午回去了。"),
            ("犬は夕方に帰りました。", "狗傍晚回去了。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        for source in [
            "私たちの時計の針が正午に戻りました。",
            "時計たちが正午に戻りました。",
            "時計の針が五時から正午に戻りました。",
        ] {
            let text = translate(source);
            assert!(text.contains("到正午"), "{source}: {text}");
        }
    }

    #[test]
    fn manner_and_return_share_their_time_and_route() {
        for (source, expected) in [
            (
                "学生は昨日の夕方に歩いて帰りました。",
                "學生昨天傍晚走回去了。",
            ),
            (
                "姉は夕方に図書館から駅まで歩いて帰りました。",
                "姊姊傍晚從圖書館走回車站了。",
            ),
            (
                "妹は昼過ぎに走って学校へ戻りました。",
                "妹妹中午過後跑回學校了。",
            ),
            (
                "父は朝に走って公園へ戻りませんでした。",
                "父親早上沒有跑回公園。",
            ),
            ("母は泳いで島へ戻りました。", "母親游回島了。"),
            ("私は歩いて店に戻りました。", "我走回商店了。"),
            ("「夕方に歩いて戻ります。」", "「傍晚走回來。」"),
            ("今朝走って戻りました。", "今朝跑回來了。"),
            ("妹は走って今朝公園へ戻りました。", "妹妹今朝跑回公園了。"),
            ("母は昨日夕方に歩いて帰りました。", "母親昨天傍晚走回去了。"),
            (
                "父は夕方に歩いてゆっくり家に帰りました。",
                "父親傍晚慢慢地走回家了。",
            ),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn return_does_not_absorb_a_distinct_activity_or_endpoint() {
        for (source, first, second) in [
            ("朝に歩いて夜に駅へ戻りました。", "早上走", "夜晚"),
            ("八時に歩いて九時に家へ戻りました。", "八點走", "九點"),
            ("駅から走って公園から家に戻りました。", "從車站跑", "從公園"),
            ("駅では歩いて家へ戻りました。", "在車站走", "家"),
            ("駅まで走って家に戻りました。", "跑到車站", "家"),
            ("六時まで歩いて家に帰りました。", "走到六點", "家"),
            ("彼は公園で走って家へ戻りました。", "在公園跑", "家"),
            ("私は夕方に歩いて、妹が家に戻りました。", "走", "妹妹"),
            ("私は歩いてから駅へ戻りました。", "走後", "車站"),
            ("私は`昼`に歩いて家に戻りました。", "`昼`", "家"),
            ("妹は昨日走って今朝家に戻りました。", "昨天跑", "今朝"),
            ("父は昨夜歩いて明朝病院へ戻ります。", "昨夜走", "明朝"),
            ("今朝走って今夜戻りました。", "今朝跑", "今夜"),
            ("昨日は走って今朝家に戻りました。", "昨天跑", "今朝"),
            (
                "妹は昨夜から走って今日公園に戻りました。",
                "從昨夜跑",
                "今天",
            ),
            ("兄は八時に歩いて今朝病院へ戻りました。", "八點走", "今朝"),
            (
                "父は昨日ゆっくり歩いて今朝家に帰りました。",
                "昨天慢慢地走",
                "今朝",
            ),
        ] {
            let text = translate(source);
            let (a, b) = text
                .split_once('，')
                .unwrap_or_else(|| panic!("{source}: {text}"));
            assert!(a.contains(first) && b.contains(second), "{source}: {text}");
        }
    }

    #[test]
    fn return_modifiers_describe_their_action_or_their_noun() {
        for (source, expected) in [
            (
                "父は今朝早く走って家へ戻りました。",
                "父親今朝早早地跑回家了。",
            ),
            (
                "先生は楽しく歩いて公園へ戻りました。",
                "老師快樂地走回公園了。",
            ),
            ("父は古い家へ走って戻りました。", "父親跑回舊的家了。"),
            (
                "今朝早く走って今夜家に戻りました。",
                "今朝早早地跑，今夜回來到家了。",
            ),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn return_companions_keep_their_action_and_destination() {
        for (source, expected) in [
            ("私は妹と歩いて駅へ戻りました。", "我跟妹妹走回車站了。"),
            (
                "私は妹と歩いて父と駅へ戻りました。",
                "我跟妹妹走，跟父親回到車站了。",
            ),
            (
                "先生は弟と走って、母と公園に帰りました。",
                "老師跟弟弟跑，跟母親回到公園了。",
            ),
            (
                "私は母と妹と歩いて病院へ戻りました。",
                "我跟母親和妹妹走回醫院了。",
            ),
            ("私と妹は歩いて駅へ戻りました。", "我和妹妹走回車站了。"),
            (
                "私は妹と父の家に歩いて戻りました。",
                "我跟妹妹走回父親的家了。",
            ),
            (
                "私は妹と一緒に歩いて駅へ戻りました。",
                "我跟妹妹一起走回車站了。",
            ),
            ("誰かと歩いて駅へ戻りました。", "跟某個人走回車站了。"),
            ("犬と走って家へ戻りました。", "跟狗跑回家了。"),
            (
                "私は妹と歩いて父と駅へ戻りましたか。",
                "我跟妹妹走，跟父親回到車站了嗎？",
            ),
            (
                "二人は歩いて駅と公園に戻りました。",
                "二人走回車站和公園了。",
            ),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn time_range_of_a_setting_stays_distinct_from_event_duration() {
        assert_eq!(
            translate("友達は昼過ぎまで公園で走りました。"),
            "朋友在公園跑到中午過後。"
        );
        for (source, reading) in [
            ("時計の針を一時から正午に戻しました。", "從一點撥回正午"),
            ("時計の針を三時から正午に戻しました。", "從三點撥回正午"),
            ("会議の時間を夕方から朝に戻しました。", "從傍晚改回早上"),
        ] {
            let text = translate(source);
            assert!(
                text.starts_with('把') && text.contains(reading),
                "{source}: {text}"
            );
        }
        let storage = translate("本を棚から机に戻しました。");
        assert!(
            storage.contains("放回") && !storage.contains("改回"),
            "{storage}"
        );
    }

    #[test]
    fn ordinary_returning_subjects_and_time_holders_are_distinct() {
        for (s, expected) in [
            ("家族が夕方に戻りました。", "家人傍晚回來了。"),
            ("同僚が夕方に戻りました。", "同僚傍晚回來了。"),
            ("客は正午に戻りました。", "客正午回來了。"),
            ("犬が夕方に戻りました。", "狗傍晚回來了。"),
            ("彼の車は夕方に戻りました。", "他的車傍晚回來了。"),
        ] {
            assert_eq!(translate(s), expected, "{s}");
        }
    }

    #[test]
    fn a_restoration_keeps_the_event_time_and_the_new_clock_value() {
        for (s, before, after) in [
            (
                "私は夕方に会議の開始時刻を正午に戻しました。",
                "我傍晚把",
                "改回正午",
            ),
            (
                "私は正午に会議の開始時刻を夕方に戻しました。",
                "我正午把",
                "改回傍晚",
            ),
            (
                "朝に予約の時間を夜に戻しませんでした。",
                "早上沒有把",
                "改回夜晚",
            ),
        ] {
            let text = translate(s);
            assert!(
                text.starts_with(before) && text.contains(after),
                "{s}: {text}"
            );
            assert!(
                text.contains("時刻") || text.contains("時間"),
                "{s}: {text}"
            );
        }
        let physical = translate("私は夕方に本を棚に戻しました。");
        assert!(
            !physical.contains("改回") && physical.contains("書"),
            "{physical}"
        );
    }

    #[test]
    fn polite_volition_changes_the_suggestion_but_keeps_the_arguments() {
        for (source, expected) in [
            ("電車で行きましょう。", "坐電車去吧。"),
            ("今日図書館に行きましょう。", "今天去到圖書館吧。"),
            ("みんなで行きましょう！", "大家一起去吧！"),
            ("一緒に駅まで歩きましょう。", "一起走到車站吧。"),
            ("ハルコを待ちましょう。", "等待ハルコ吧。"),
            ("先生を呼びましょうか。", "叫老師，好嗎？"),
            ("どこで会いましょうか。", "在哪裡見面好呢？"),
            ("何を読みましょうか。", "讀什麼好呢？"),
            ("お手伝いしましょうか。", "需要協助嗎？"),
            ("私が手伝いましょうか。", "我來協助，好嗎？"),
            ("僕が本を読みましょうか。", "我來讀書，好嗎？"),
            ("私がここに来ましょうか。", "我來到這裡，好嗎？"),
            ("「行きましょうか。」と聞きました。", "「去，好嗎？」聽了。"),
            ("「行きましょうか」と聞きました。", "「去，好嗎」聽了。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn proposal_questions_put_interrogative_time_before_the_activity() {
        for (source, expected) in [
            ("何時に駅へ行きましょうか。", "幾點往車站去好呢？"),
            ("何時に出ましょうか。", "幾點出去好呢？"),
            ("何曜日に本を読みましょうか。", "星期幾讀書好呢？"),
            ("いつ駅に行きましょうか。", "什麼時候去到車站好呢？"),
            ("先生に手紙を書きましょうか。", "給老師寫信，好嗎？"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn proposals_distinguish_indefinite_arguments_from_open_questions() {
        for (source, expected) in [
            ("誰かと一緒に歌いましょう。", "和某個人一起哼唱吧。"),
            ("誰かと一緒に歩きましょうか。", "和某個人一起走，好嗎？"),
            ("誰と一緒に歌いましょうか。", "和誰一起哼唱好呢？"),
            ("誰かと何時に会いましょうか。", "幾點跟某個人見面好呢？"),
            ("何か読みましょうか。", "讀點什麼，好嗎？"),
            ("何かを食べましょう。", "吃點什麼吧。"),
            ("何か作業をしましょうか。", "做一些作業，好嗎？"),
            ("どこかで会いましょうか。", "在某個地方見面，好嗎？"),
            ("どこで会いましょうか。", "在哪裡見面好呢？"),
            ("誰かが手伝いましょう。", "某個人協助吧。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn proposals_distinguish_companions_from_coordinated_destinations() {
        for (source, expected) in [
            ("誰かと図書館へ行きましょう。", "和某個人往圖書館去吧。"),
            ("誰と図書館へ行きましょうか。", "和誰往圖書館去好呢？"),
            (
                "先生と図書館の二階に行きましょう。",
                "和老師去圖書館的二樓吧。",
            ),
            ("友達と駅まで歩きましょう。", "和朋友走到車站吧。"),
            ("図書館へ先生と行きましょう。", "和老師往圖書館去吧。"),
            ("図書館と公園へ行きましょう。", "往圖書館和公園去吧。"),
            ("東京と大阪に行きましょう。", "去到東京和大阪吧。"),
            ("先生と友達に手紙を書きましょう。", "給老師和朋友寫信吧。"),
            ("私と先生が図書館に行きましょう。", "我和老師去到圖書館吧。"),
            ("友達と先生を駅で待ちましょう。", "在車站等待朋友和老師吧。"),
            (
                "先生と二人で図書館へ行きましょう。",
                "和老師兩個人一起往圖書館去吧。",
            ),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn proposals_keep_the_number_and_location_of_the_participants() {
        for (source, expected) in [
            ("二人で本を読みましょう。", "兩個人一起讀書吧。"),
            ("一人で図書館へ行きましょう。", "一個人往圖書館去吧。"),
            ("十二人で本を読みましょう。", "十二個人一起讀書吧。"),
            ("先生と一緒に駅へ行きましょう。", "和老師一起往車站去吧。"),
            ("二階で本を読みましょう。", "在二樓讀書吧。"),
            ("二人に本を読みましょう。", "給二人讀書吧。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        let path = translate("一緒に川の近くを歩きましょう。");
        assert!(
            path.starts_with("一起走") && path.contains('川') && path.ends_with("吧。"),
            "{path}"
        );
    }

    #[test]
    fn ordinary_and_negative_polite_forms_do_not_make_volitional_suggestions() {
        let statement = translate("電車で行きます。");
        assert_eq!(statement, "坐電車去。");
        for (source, expected) in [
            ("私が荷物を持ちますか。", "我拿著荷物嗎"),
            ("駅まで歩きませんか。", "不走到車站嗎"),
        ] {
            assert_eq!(
                translate(source).trim_end_matches(['。', '？']),
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn wishes_and_negative_questions_keep_the_direction() {
        assert_eq!(translate("駅まで行きたいです。"), "想去車站。");
        assert_eq!(translate("公園まで行きたいです。"), "想去公園。");
        assert_eq!(translate("公園まで走りたいです。"), "想跑到公園。");
        assert_eq!(translate("駅まで行きませんか。"), "不去車站嗎？");
        assert_eq!(translate("公園まで行きませんか。"), "不去公園嗎？");
        assert_eq!(translate("公園まで走りませんか。"), "不跑到公園嗎？");
    }

    #[test]
    fn joint_activities_preserve_the_offer_and_its_arguments() {
        for (source, expected) in [
            ("一緒に走りませんか。", "要不要一起跑步？"),
            ("一緒にお茶を飲みませんか。", "要不要一起喝茶？"),
            ("一緒に公園へ行きませんか。", "要不要一起去公園？"),
            ("一緒に歌いませんか。", "要不要一起唱歌？"),
            ("一緒に勉強しませんか。", "要不要一起學習？"),
            ("２人で歌いませんか。", "要不要兩人一起唱歌？"),
            ("３人で歌いませんか。", "要不要3人一起唱歌？"),
            ("一緒に見学しませんか。", "要不要一起參觀？"),
            ("本を一緒に読みませんか。", "要不要一起讀書？"),
            ("一緒に駅まで歩きませんか。", "要不要一起走到車站？"),
            (
                "一緒に図書館で本を読みませんか。",
                "要不要一起在圖書館讀書？",
            ),
            ("友達と一緒に映画を見ませんか。", "要不要和朋友一起看電影？"),
            ("誰かと一緒に歌いませんか。", "要不要和某個人一起唱歌？"),
            (
                "誰かと図書館へ行きませんか。",
                "要不要和某個人一起去圖書館？",
            ),
            (
                "先生と図書館の二階に行きませんか。",
                "要不要和老師一起去圖書館的二樓？",
            ),
            (
                "友達と図書館で勉強しませんか。",
                "要不要和朋友一起在圖書館學習？",
            ),
            ("一緒に誰かと歩きませんか。", "要不要和某個人一起走？"),
            (
                "誰かと何かを一緒に読みませんか。",
                "要不要和某個人一起讀點什麼？",
            ),
            ("みんなで何か読みませんか。", "要不要大家一起讀點什麼？"),
            (
                "一緒にどこかで何か食べませんか。",
                "要不要一起在某個地方吃點什麼？",
            ),
            ("誰かと二人で歌いませんか。", "要不要和某個人兩人一起唱歌？"),
            ("私たちと一緒に歩きませんか。", "要不要和我們一起走？"),
            ("あなたも一緒に歌いませんか。", "你要不要也一起唱歌？"),
            ("みんなで歌いませんか。", "要不要大家一起唱歌？"),
            ("二人で歌いませんか。", "要不要兩人一起唱歌？"),
            ("三人で歌いませんか。", "要不要三人一起唱歌？"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn participant_counts_include_the_subject_and_companion() {
        for (source, expected) in [
            (
                "私は先生と二人で公園まで歩きませんか。",
                "我和老師兩人不一起走到公園嗎？",
            ),
            (
                "先生は友達と三人で駅まで歩きませんか。",
                "老師和朋友三人不一起走到車站嗎？",
            ),
            (
                "先生と二人で公園まで歩きませんか。",
                "要不要和老師兩人一起走到公園？",
            ),
            (
                "私と先生が二人で公園まで歩きませんか。",
                "我和老師兩人不一起走到公園嗎？",
            ),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn activity_time_and_genitive_time_have_separate_attachments() {
        for source in [
            "私は明日先生と二人で公園まで歩きませんか。",
            "私は先生と二人で明日公園まで歩きませんか。",
        ] {
            assert_eq!(
                translate(source),
                "我和老師兩人明天不一起走到公園嗎？",
                "{source}"
            );
        }
        assert_eq!(
            translate("私は明日の先生と二人で公園まで歩きませんか。"),
            "我和明天的老師兩人不一起走到公園嗎？",
        );
    }

    #[test]
    fn reported_subjects_and_other_speech_acts_keep_their_scope() {
        assert_eq!(translate("この犬は泳ぎませんか。"), "這狗不游泳嗎？");
        assert_eq!(
            translate("このバスは空港へ行きませんか。"),
            "這輛公車不去機場嗎？"
        );
        assert_eq!(translate("行きませんか。"), "不去嗎？");
        assert_eq!(
            translate("駅と図書館に行きませんか。"),
            "不去到車站和圖書館嗎？"
        );
        assert_eq!(
            translate("先生は友達と図書館に行きませんか。"),
            "老師不和朋友一起去圖書館嗎？"
        );
        assert_eq!(
            translate("私と先生が図書館に行きませんか。"),
            "我和老師不去圖書館嗎？"
        );
        assert_eq!(
            translate("先生と私の友達が図書館に行きませんか。"),
            "老師和我的朋友不去圖書館嗎？"
        );
        assert_eq!(
            translate("先生と学生を駅で待ちませんか。"),
            "在車站不等待老師和學生嗎？"
        );
        assert_eq!(translate("一緒に遊べませんか。"), "不能一起玩嗎？");
        assert_eq!(
            translate("みんなで何を読みませんか。"),
            "大家不一起讀什麼？"
        );
        assert_eq!(translate("みんなで遊べませんか。"), "大家不能一起玩嗎？");
        assert_eq!(
            translate("誰かと一緒に何を読みませんか。"),
            "不和某個人一起讀什麼？"
        );
        assert_eq!(
            translate("先生は誰かと一緒に歌いませんか。"),
            "老師不和某個人一起哼唱嗎？"
        );
        assert_eq!(translate("一緒に誰かが来ませんか。"), "某個人不一起來嗎？");
        assert!(!translate("一緒に何か起きませんか。").contains("點什麼"));
        assert_eq!(
            translate("みんなで何か作業をしませんか。"),
            "要不要大家一起做一些作業？"
        );
        assert_eq!(translate("二人で何を読みませんか。"), "兩人不一起讀什麼？");
        assert_eq!(translate("一緒に飲めませんか。"), "不能一起喝嗎？");
        assert_eq!(
            translate("先生は一緒に歌いませんか。"),
            "老師不一起哼唱嗎？"
        );
        for source in [
            "この犬は一緒に走りませんか。",
            "一緒に眠れませんか。",
            "一緒に飲めませんか。",
            "一人で歌いませんか。",
            "１人で歌いませんか。",
            "一緒に勉強してくれませんか。",
            "何を一緒に読みませんか。",
            "一緒に歌いましょうか。",
            "一緒に公園へ行きません。",
            "一緒に来ていただけませんか。",
            "一緒に本を読んでくれませんか。",
            "歌いませんか。",
            "一緒に走りませんでしたか。",
            "`一緒`に歌いませんか。",
        ] {
            let out = translate(source);
            assert!(!out.contains("要不要"), "{source}: {out}");
        }
    }

    #[test]
    fn motion_purpose_keeps_its_action_object_and_outer_polarity() {
        for (source, expected) in [
            ("私は本を借りに行きます。", "我去借書。"),
            ("私は傘を借りに行きます。", "我去借傘。"),
            ("私は友達を探しに行きます。", "我去找朋友。"),
            ("私は鍵を探しに行きます。", "我去找鑰匙。"),
            ("本を買いに行きます。", "去買書。"),
            ("私はパンを食べに行きます。", "我去吃麵包。"),
            ("魚を食べに行きます。", "去吃魚。"),
            ("友達と一緒に本を借りに行きます。", "和朋友一起去借書。"),
            (
                "私は友達と一緒に本を借りに行きます。",
                "我和朋友一起去借書。",
            ),
            ("友達と妹が本を借りに行きます。", "朋友和妹妹去借書。"),
            ("誰が本を借りに行きますか。", "誰去借書？"),
            ("誰と一緒に本を借りに行きますか。", "和誰一起去借書？"),
            ("誰に本を返しに行きますか。", "去還書給誰？"),
            ("友達が本を借りに行きますか。", "朋友去借書嗎？"),
            ("私は食べに行きます。", "我去吃。"),
            (
                "「本を借りに行きます」と先生が言いました。",
                "「去借書」老師說了。",
            ),
            (
                "先生は「本を借りに行きます」と言いました。",
                "老師說「去借書」。",
            ),
            (
                "「パンを食べに行きます」と先生が言いました。",
                "「去吃麵包」老師說了。",
            ),
            (
                "「本を買いに行きません」と先生が言いました。",
                "「不去買書」老師說了。",
            ),
            (
                "先生は学生に「明日は家に帰ります。本を借りに行きます」と伝えた。",
                "老師告訴學生：「明天回到家。去借書。」",
            ),
            (
                "「明日は家に帰ります。本を借りに行きます」と先生が言いました。",
                "「明天回到家。去借書」老師說了。",
            ),
            ("食べに行きませんか。", "要不要去吃？"),
            ("材料を混ぜに行きます。", "去攪拌材料。"),
            ("窓を閉めに行きます。", "去關上窗戶。"),
            ("父を迎えに行きます。", "去迎接父親。"),
            ("私は先生に本を返しに行きます。", "我去還書給老師。"),
            ("私は先生に本を借りに行きます。", "我去向老師借書。"),
            ("先生に何を返しに行きますか。", "去還什麼給老師？"),
            ("先生に花を渡しに行きます。", "去遞花給老師。"),
            ("図書館に本を返しに行きます。", "去到圖書館還書。"),
            ("本を読みに行きません。", "不去讀書。"),
            ("音楽を聞きに行った。", "去聽音樂了。"),
            ("花を見に行きませんか。", "要不要去看花？"),
            ("一緒に花を見に行きませんか。", "要不要一起去看花？"),
            ("週末に映画を見に行きませんか。", "週末要不要去看電影？"),
            ("友達に会いに行きます。", "去和朋友見面。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn motion_purpose_keeps_a_source_linked_floor_as_the_destination() {
        for (source, expected) in [
            ("私は三階に本を返しに行きます。", "我去三樓還書。"),
            ("私は二階に友達を探しに行きます。", "我去二樓找朋友。"),
            ("三階へ本を借りに行きますか。", "往三樓去借書嗎？"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn motion_purpose_keeps_the_entire_basement_floor_label() {
        for (source, expected) in [
            ("地下二階へ本を借りに行きますか。", "往地下二樓去借書嗎？"),
            ("地下二階に本を返しに行きます。", "去地下二樓還書。"),
            (
                "学校の地下二階に本を借りに行きます。",
                "去學校的地下二樓借書。",
            ),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        for (source, locked) in [
            ("`地下二階`に本を返しに行きます。", "`地下二階`"),
            ("`地下`二階に本を返しに行きます。", "`地下`"),
        ] {
            let output = translate(source);
            assert!(
                output.matches(locked).count() == 1 && !output.contains('樓'),
                "{source}: {output}"
            );
        }
    }

    #[test]
    fn motion_purpose_floor_distinguishes_math_order_from_a_place_to_write() {
        for source in [
            "私は式を二階にしに行きます。",
            "私は式を二階に戻しに行きます。",
        ] {
            let output = translate(source);
            assert!(
                output.contains("二階") && output.contains('式') && !output.contains('樓'),
                "{source}: {output}"
            );
        }
        let source = "私は三階に数式を書きに行きます。";
        let output = translate(source);
        assert!(
            output.contains("去三樓寫") && output.contains('式'),
            "{source}: {output}"
        );
    }

    #[test]
    fn motion_purpose_floor_keeps_other_roles_and_locks() {
        let source = "私は三階で本を返します。";
        let output = translate(source);
        assert!(
            output.contains("在三樓") && output.contains('書') && !output.contains('去'),
            "{source}: {output}"
        );
        for (source, kept) in [
            ("この微分方程式は二階です。", "二階"),
            ("私は二階堂さんに本を返しに行きます。", "二階堂"),
            ("私は`三階`に本を返しに行きます。", "`三階`"),
            ("私は三階の図書室に本を返しに行きます。", "三階"),
        ] {
            let output = translate(source);
            assert!(
                output.contains(kept) && !output.contains('樓'),
                "{source}: {output}"
            );
        }
    }

    #[test]
    fn nominal_places_and_other_source_relations_keep_their_roles() {
        for (source, expected) in [
            ("教室に行きます。", "去到教室。"),
            ("午後二時に教室に行きます。", "在下午兩點去到教室。"),
            ("私は買い物に行きます。", "我去購物。"),
            ("私は映画を`見`に行きます。", "我去到`見`電影。"),
            ("本を借りて行きます。", "借用書，去。"),
            ("彼は帰りに駅に行きました。", "他去到車站到帰り了。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn stray_closing_quote_does_not_supply_a_motion_governor() {
        let nested = translate("先生は「今日は『休み』です。本を借りに行きます」と言いました。");
        assert!(nested.ends_with("去借書」說了。"), "{nested}");
        let indefinite = translate("誰かが本を借りに行きますか。");
        assert!(indefinite.ends_with("嗎？"), "{indefinite}");
        for source in [
            "本を借りに行きます」と先生が言いました。",
            "「明日、帰ります。」本を借りに行きます」と先生が言いました。",
            "挨拶は`「`です。本を借りに行きます」と先生が言いました。",
            "先生は「今日は『休み」です。本を借りに行きます」と言いました。",
        ] {
            let output = translate(source);
            assert!(!output.contains("去借書"), "{source}: {output}");
        }
    }

    #[test]
    fn changing_a_schedule_keeps_the_event_time_and_result_value_distinct() {
        for (source, expected) in [
            (
                "父は朝に会議の時刻を夕方に延期しました。",
                "父親早上把會議的時刻延到傍晚了。",
            ),
            (
                "父は夕方に会議の時刻を正午に調整しました。",
                "父親傍晚把會議的時刻調到正午了。",
            ),
            (
                "先生は朝に出発の時刻を夕方に変更しました。",
                "老師早上把出發的時刻改到傍晚了。",
            ),
            (
                "先生は夕方に出発の時刻を朝に変更しました。",
                "老師傍晚把出發的時刻改到早上了。",
            ),
            (
                "朝に集合時刻を夕方に変更しませんでした。",
                "早上沒有把集合時刻改到傍晚。",
            ),
            (
                "母は正午に時計を夕方に合わせました。",
                "母親正午把手錶調到傍晚了。",
            ),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        for source in [
            "母は正午に先生に時計を返しました。",
            "朝に車を車庫に戻しました。",
            "生徒は先生に時計を友達に渡しました。",
            "私は正午に会議の時間を朝と比べました。",
        ] {
            let text = translate(source);
            assert!(
                !text.contains("改回") && !text.contains("改到") && !text.contains("調到"),
                "{source}: {text}"
            );
        }
    }

    #[test]
    fn physical_return_keeps_the_time_object_and_storage_site() {
        for (source, expected) in [
            ("私は夕方に本を棚に戻しました。", "我傍晚把書放回架上了。"),
            ("私は本を棚に夕方に戻しました。", "我傍晚把書放回架上了。"),
            (
                "私は朝に本を棚に戻しませんでした。",
                "我早上沒有把書放回架上。",
            ),
            (
                "私は夕方に鍵を箱に戻しました。",
                "我傍晚把鑰匙放回箱子裡了。",
            ),
            (
                "私は夕方に財布を鞄に戻しました。",
                "我傍晚把錢包放回包包了。",
            ),
            ("私は夕方に本を教室に戻しました。", "我傍晚把書放回教室了。"),
            ("私は本を先生に夕方に戻しました。", "我傍晚把書還給老師了。"),
            ("私は夕方に本を彼に戻しました。", "我傍晚把書還給他了。"),
            (
                "私は夕方に設定を標準に戻しました。",
                "我傍晚把設定改回標準了。",
            ),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        let recipient = translate("私は夕方に本を先生に返しました。");
        assert!(
            recipient.contains("老師") && recipient.contains('書') && !recipient.contains("放回")
        );
    }

    #[test]
    fn motion_routes_keep_their_manner_and_polarity() {
        for (source, expected) in [
            ("犬は道を走ります。", "狗沿著道路跑。"),
            ("道を走っています。", "正在沿著道路跑。"),
            ("犬が道を走っていません。", "狗沒有沿著道路跑。"),
            ("列車はトンネルを通ります。", "列車經過隧道。"),
            ("列車は駅までトンネルを通ります。", "列車經過隧道到車站。"),
            ("駅まで橋を渡りました。", "渡過橋到車站了。"),
            ("公園を通って家に帰ります。", "經過公園，回到家。"),
            ("公園を通っています。", "正在經過公園。"),
            ("私は公園の中を歩きました。", "我在公園裡面走了。"),
            ("国道を走ります。", "沿著國道跑。"),
            ("道を学校へ走ります。", "沿著道路往學校跑。"),
            ("道路を進みます。", "沿著道路前進。"),
            ("予想通りです。", "是按照予想。"),
            ("犬は道を走りません。", "狗不沿著道路跑。"),
            ("私は道路を歩きます。", "我沿著道路走。"),
            ("子供は橋を走りました。", "孩子沿著橋跑了。"),
            ("犬は道を歩きませんでした。", "狗沒有沿著道路走。"),
            ("道を駅まで走ります。", "沿著道路跑到車站。"),
            ("私は二階まで道を走ります。", "我沿著道路跑到二樓。"),
            ("公園を散歩します。", "在公園散步。"),
            ("犬は公園まで走ります。", "狗跑到公園。"),
            ("私は道を選びます。", "我選擇道路。"),
            ("私は橋を渡ります。", "我渡過橋。"),
            ("私は橋を運びます。", "我搬運橋。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        for source in [
            "道を走らせます。",
            "車を走らせます。",
            "一時間を走ります。",
            "私は駅を出ます。",
        ] {
            assert!(!translate(source).contains("沿著"), "{source}");
        }
        let commute = translate("学校に通っています。");
        assert!(commute.contains("學校") && commute.contains("上學") && !commute.contains("經過"));
        let corridor = translate("私は三階から廊下を二階まで歩きます。");
        assert!(corridor.starts_with("我從三樓沿著") && corridor.ends_with("走到二樓。"));
    }

    #[test]
    fn negative_questions_keep_nonhuman_subjects() {
        assert_eq!(
            translate("このバスは駅まで行きませんか。"),
            "這輛公車不去車站嗎？"
        );
        assert_eq!(
            translate("この電車は駅まで行きませんか。"),
            "這輛電車不去車站嗎？"
        );
    }

    #[test]
    fn negative_go_keeps_physical_destination_after_go() {
        for source in ["図書館に行かなかった。", "図書館へ行かなかった。"] {
            let output = translate(source);
            assert!(output.contains("沒有去圖書館"), "{source}: {output}");
        }
        for source in ["空港に行かなかった。", "空港へ行かなかった。"] {
            let output = translate(source);
            assert!(output.contains("沒有去機場"), "{source}: {output}");
        }
        let positive = translate("図書館へ行きました。");
        assert!(positive.contains("往圖書館"), "{positive}");
        let question = translate("図書館に行きませんか。");
        assert!(question.contains('嗎'), "{question}");
        let endpoint = translate("図書館まで行かなかった。");
        assert!(
            endpoint.contains("圖書館") && endpoint.contains("沒有"),
            "{endpoint}"
        );
    }

    #[test]
    fn negative_go_keeps_clock_and_direct_destination() {
        for source in [
            "午前九時に学校へ行きませんでした。",
            "午前九時に学校に行きませんでした。",
        ] {
            let output = translate(source);
            assert!(
                output.contains("上午九點") && output.contains("沒有去學校"),
                "{source}: {output}"
            );
            assert!(
                !output.contains("去到學校") && !output.contains("往學校"),
                "{source}: {output}"
            );
        }
        let positive = translate("午前九時に学校へ行きました。");
        assert!(
            positive.contains("上午九點") && positive.contains("往學校"),
            "{positive}"
        );
        let endpoint = translate("午前九時に学校まで行きませんでした。");
        assert!(
            endpoint.contains("上午九點") && endpoint.contains("學校") && endpoint.contains("沒有"),
            "{endpoint}"
        );
    }

    #[test]
    fn negative_go_keeps_clock_and_direct_building_floor() {
        for source in [
            "私は午前九時に二階に行きませんでした。",
            "私は午前九時に二階へ行きませんでした。",
        ] {
            let output = translate(source);
            assert!(output.contains("上午九點沒有去二樓"), "{source}: {output}");
        }
    }

    #[test]
    fn clock_exclusion_requires_plain_contiguous_clock_particle() {
        let clock = Atom {
            span: Span {
                start_char: 0,
                end_char: 4,
            },
            surface: "上午九點".to_owned(),
            lemma: Some(CLOCK_ATOM.to_owned()),
            upos: Some(Upos::Noun),
            doc_tok_idx: None,
            head: None,
            deprel: String::new(),
            is_locked: true,
        };
        let particle = Atom {
            span: Span {
                start_char: 4,
                end_char: 5,
            },
            surface: "に".to_owned(),
            lemma: None,
            upos: Some(Upos::Adp),
            doc_tok_idx: None,
            head: None,
            deprel: "case".to_owned(),
            is_locked: false,
        };
        let mut b = Bunsetsu {
            atoms: vec![clock, particle],
            role: BunsetsuRole::Obl,
            case_particle: Some("に".to_owned()),
            sconj_tail: None,
            sfp_tail: None,
        };
        assert!(is_plain_clock_ni(&b));
        b.atoms[0].lemma = Some(OPAQUE_PERIOD_CLOCK.to_owned());
        assert!(!is_plain_clock_ni(&b));
        b.atoms[0].lemma = Some(CLOCK_ATOM.to_owned());
        b.atoms[0].surface = "[[PH0]]".to_owned();
        assert!(!is_plain_clock_ni(&b));
        b.atoms[0].surface = "上午九點".to_owned();
        b.atoms[1].span.start_char += 1;
        assert!(!is_plain_clock_ni(&b));
        b.atoms[1].span.start_char -= 1;
        b.atoms[1].is_locked = true;
        assert!(!is_plain_clock_ni(&b));
        b.atoms[1].is_locked = false;
        b.role = BunsetsuRole::IObj;
        assert!(!is_plain_clock_ni(&b));
        b.role = BunsetsuRole::Obl;
        b.atoms[1].surface = "へ".to_owned();
        b.case_particle = Some("へ".to_owned());
        assert!(!is_plain_clock_ni(&b));
        b.atoms[1].surface = "に".to_owned();
        b.case_particle = Some("に".to_owned());
        b.atoms.push(b.atoms[1].clone());
        assert!(!is_plain_clock_ni(&b));
    }

    #[test]
    fn negative_go_goal_guard_requires_source_link_and_unlocked_place() {
        fn atom(surface: &str, start: usize, locked: bool) -> Atom {
            Atom {
                span: Span {
                    start_char: start,
                    end_char: start + surface.chars().count(),
                },
                surface: surface.to_owned(),
                lemma: None,
                upos: Some(Upos::Noun),
                doc_tok_idx: None,
                head: None,
                deprel: String::new(),
                is_locked: locked,
            }
        }
        let mut place = atom("図書館", 0, false);
        place.head = Some(8);
        let particle = Atom {
            surface: "に".to_owned(),
            upos: Some(Upos::Adp),
            ..atom("に", 3, false)
        };
        let mut b = Bunsetsu {
            atoms: vec![place, particle],
            role: BunsetsuRole::Obl,
            case_particle: Some("に".to_owned()),
            sconj_tail: None,
            sfp_tail: None,
        };
        assert_eq!(physical_goal_head(&b), Some(8));
        b.atoms[0].is_locked = true;
        assert_eq!(physical_goal_head(&b), None);
        b.atoms[0].is_locked = false;
        b.atoms[0].head = None;
        assert_eq!(physical_goal_head(&b), None);
        b.atoms[0].surface = "夕方".to_owned();
        b.atoms[0].head = Some(8);
        assert_eq!(physical_goal_head(&b), None);
        b.atoms[0].surface = "空港".to_owned();
        assert_eq!(physical_goal_head(&b), Some(8));
        b.atoms[0].surface = "大阪".to_owned();
        b.atoms[0].upos = Some(Upos::Propn);
        assert_eq!(physical_goal_head(&b), Some(8));
        b.atoms[0].upos = Some(Upos::Noun);
        assert_eq!(physical_goal_head(&b), None);
        b.atoms[0].surface = "空港".to_owned();
        b.atoms[0].head = Some(8);
        b.atoms[0].doc_tok_idx = Some(4);
        b.atoms[0].span = Span {
            start_char: 2,
            end_char: 4,
        };
        b.atoms[1].span = Span {
            start_char: 4,
            end_char: 5,
        };
        let mut modifier = atom("新宿", 0, false);
        modifier.upos = Some(Upos::Propn);
        modifier.head = Some(4);
        modifier.deprel = "compound".to_owned();
        b.atoms.insert(0, modifier);
        assert_eq!(physical_goal_head(&b), Some(8));
        b.atoms[0].head = Some(7);
        assert_eq!(physical_goal_head(&b), None);
        b.atoms[0].head = Some(4);
        b.atoms[0].is_locked = true;
        assert_eq!(physical_goal_head(&b), None);
    }

    #[test]
    fn near_negative_controls_keep_their_existing_meaning() {
        assert_eq!(translate("駅まで行きませんでしたか。"), "到車站沒有去嗎？");
        assert_eq!(translate("公園まで描きませんか。"), "連公園都不畫畫嗎？");
        assert_eq!(translate("駅まで行かなかった。"), "到車站沒有去。");
    }

    #[test]
    fn walking_and_running_keep_the_place_as_endpoint() {
        for (source, phrase) in [
            ("公園まで歩いた。", "走到公園"),
            ("学校まで走った。", "跑到學校"),
        ] {
            let output = translate(source);
            assert!(output.contains(phrase), "{source}: {output}");
            assert!(!output.contains('連'), "{source}: {output}");
        }
    }

    #[test]
    fn transport_and_time_precede_the_motion_endpoint() {
        assert_eq!(
            translate("私は毎朝駅までバスで行きます。"),
            "我每天早上坐公車到車站。"
        );
        let reversed_source_order = translate("バスで駅まで行った。");
        assert!(reversed_source_order.contains("坐公車到車站"));
    }

    #[test]
    fn time_ordering_remains_bounded_to_spatial_motion() {
        assert_eq!(translate("毎朝新聞を読む。"), "每天早上讀報紙。");
        assert_eq!(translate("毎晩本を読む。"), "每晚讀書。");
    }

    #[test]
    fn people_and_non_motion_keep_additive_focus() {
        let person = translate("先生まで歩いた。");
        assert!(person.contains("連老師都"), "{person}");
        assert!(!person.contains("到老師"), "{person}");
        let non_motion = translate("公園まで描いた。");
        assert!(non_motion.contains("連公園都"), "{non_motion}");
    }

    #[test]
    fn residence_has_a_place_and_keeps_its_own_tense_and_polarity() {
        for (source, expected) in [
            ("私は東京に住んでいます。", "我住在東京。"),
            ("私は大阪に住んでいました。", "我曾住在大阪。"),
            ("私は京都に住んでいません。", "我不住在京都。"),
            ("私は東京に住んでいませんでした。", "我沒有住在東京。"),
            ("先生はアパートに住んでいます。", "老師住在公寓。"),
            ("友達は町に住んでいる。", "朋友住在城鎮。"),
            ("京都には三年住んでいました。", "在京都住了三年。"),
            ("友達と東京に住んでいます。", "和朋友住在東京。"),
            ("どこに住んでいますか。", "住在哪裡？"),
            ("私は東京から来ました。", "我從東京來了。"),
            ("私は京都に行きませんでした。", "我沒有去京都。"),
            ("東京の友達に電話します。", "給東京的朋友打電話。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        let handoff = translate("私は大阪の先生に本を渡しました。");
        assert!(
            handoff.contains("大阪的老師") && handoff.contains("書") && !handoff.contains("住在")
        );
    }

    #[test]
    fn residence_keeps_the_source_scope_of_past_time_and_additive_places() {
        for (source, expected) in [
            (
                "父は当時東京に住んでいませんでした。",
                "父親當時沒有住在東京。",
            ),
            (
                "当時、父は町に住んでいませんでした。",
                "當時，父親沒有住在城鎮。",
            ),
            (
                "父は当時の家に住んでいませんでした。",
                "父親沒有住在當時的家。",
            ),
            (
                "母は去年大阪に住んでいませんでした。",
                "母親去年沒有住在大阪。",
            ),
            ("兄は京都にも住んでいました。", "哥哥也曾住在京都。"),
            (
                "妹は当時町にも住んでいませんでした。",
                "妹妹當時也沒有住在城鎮。",
            ),
            ("姉も東京にも住んでいます。", "姊姊也住在東京。"),
            (
                "私は東京にも大阪にも住んでいませんでした。",
                "我也沒有住在東京，也沒有住在大阪。",
            ),
            ("私は東京と大阪に住んでいました。", "我曾住在東京和大阪。"),
            ("どこにも住んでいませんか。", "哪裡都沒有住嗎？"),
            ("どこにも住みません。", "哪裡都不住。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn residence_duration_keeps_aspect_and_plans() {
        for (source, expected) in [
            ("京都に三年住みました。", "在京都住了三年。"),
            ("京都に三年住んでいました。", "在京都住了三年。"),
            ("京都に三年住んでいます。", "已經在京都住了三年。"),
            ("京都に三年住む予定です。", "打算在京都住三年。"),
            ("京都に三年住みます。", "在京都住三年。"),
            ("京都に三年住んでいません。", "已經三年沒有住在京都。"),
            ("京都と大阪に住みました。", "曾住在京都和大阪。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn residence_assigns_focus_and_questions_to_the_attached_argument() {
        for (source, expected) in [
            ("兄だけが京都に住んでいます。", "只有哥哥住在京都。"),
            ("私と妹だけが東京に住んでいます。", "只有我和妹妹住在東京。"),
            ("父だけが大阪に住んでいました。", "只有父親曾住在大阪。"),
            ("兄は京都だけに住んでいます。", "哥哥只住在京都。"),
            ("私は京都にだけ住んでいます。", "我只住在京都。"),
            ("誰が京都に住んでいますか。", "誰住在京都？"),
            ("誰と京都に住んでいますか。", "和誰住在京都？"),
            ("誰かが京都に住んでいますか。", "某個人住在京都嗎？"),
            ("京都に住んでいますか。", "住在京都嗎？"),
            (
                "私は妹と一緒に東京に住んでいます。",
                "我和妹妹一起住在東京。",
            ),
            (
                "私は京都に一緒に妹と住んでいます。",
                "我和妹妹一起住在京都。",
            ),
            ("姉が一人で京都に住んでいます。", "姊姊一個人住在京都。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn residents_have_companions_and_places_can_share_a_case() {
        for (jp, tw) in [
            ("家族と東京に住んでいました。", "曾和家人住在東京。"),
            (
                "私は家族と一緒に大阪に住んでいました。",
                "我曾和家人一起住在大阪。",
            ),
            ("両親と東京に住んでいました。", "曾和父母住在東京。"),
            ("犬と東京に住んでいました。", "曾和狗住在東京。"),
            (
                "家族と妹だけが東京に住んでいました。",
                "只有家人和妹妹曾住在東京。",
            ),
            ("家族の家と東京に住んでいました。", "曾住在家人的家和東京。"),
            (
                "東京と大阪に一緒に住んでいました。",
                "曾一起住在東京和大阪。",
            ),
        ] {
            assert_eq!(translate(jp), tw, "{jp}");
        }
    }

    #[test]
    fn origin_and_recipient_cases_keep_their_relations() {
        let origin = translate("駅から歩いた。");
        assert!(origin.contains("從車站"), "{origin}");
        assert!(!origin.contains("到車站"), "{origin}");
        let recipient = translate("友達に手紙を書いた。");
        assert!(recipient.contains("給朋友"), "{recipient}");
        assert!(!recipient.contains("到朋友"), "{recipient}");
    }

    #[test]
    fn a_place_modifier_does_not_make_a_teacher_a_destination() {
        let output = translate("学校の先生まで歩いた。");
        assert!(output.contains('連'), "{output}");
        assert!(output.contains("老師都"), "{output}");
        assert!(!output.contains("到學校"), "{output}");
    }

    #[test]
    fn temporal_made_follows_the_activity_and_preserves_the_time_reading() {
        assert_eq!(
            translate("私は六時まで日本語を勉強します。"),
            "我學習日本語到六點。"
        );
        assert_eq!(
            translate("私は午後六時まで日本語を勉強します。"),
            "我學習日本語到下午六點。"
        );
        assert_eq!(
            translate("私は日曜日まで日本語を勉強します。"),
            "我學習日本語到星期日。"
        );
        assert_eq!(
            translate("図書館で夕方まで本を読みます。"),
            "在圖書館讀書到傍晚。"
        );
    }

    #[test]
    fn temporal_made_places_the_past_activity_and_location_before_the_endpoint() {
        for (source, expected) in [
            ("私は六時まで働きます。", "我工作到六點。"),
            ("私は六時まで働きました。", "我工作到六點。"),
            ("私は日曜日まで家で働きました。", "我在家工作到星期日。"),
            ("子供は夕方まで公園で遊びました。", "孩子在公園玩到傍晚。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        // Author controls derived from TMD-N5-01 and the work example above.
        assert_eq!(
            translate("私は六時まで日本語を勉強しました。"),
            "我學習日本語到六點。"
        );
        assert_eq!(
            translate("私は六時まで働きませんでした。"),
            "我到六點為止沒有工作。"
        );
    }

    #[test]
    fn compound_temporal_made_keeps_both_time_parts_after_the_activity() {
        for (source, expected) in [
            (
                "私は明日の朝まで日本語を勉強します。",
                "我學習日本語到明天早上。",
            ),
            (
                "私は来週の月曜日まで日本語を勉強します。",
                "我學習日本語到下週一。",
            ),
            (
                "私は明日の午前六時まで日本語を勉強します。",
                "我學習日本語到明天上午六點。",
            ),
            ("私は来週の金曜日まで働きます。", "我工作到下週五。"),
            ("私は明日の朝まで働きません。", "我到明天早上為止不工作。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn compound_temporal_made_leaves_other_genitive_roles_unchanged() {
        for (source, expected) in [
            ("明日の朝までに手紙を書きます。", "最晚在明天的早上寫信。"),
            ("私は明日の朝に手紙を書きます。", "我在明天的早上寫信。"),
            ("朝の電車まで走りました。", "連早上的電車都跑了。"),
            ("私は学校の先生まで歩いた。", "我連學校的老師都走了。"),
            ("学校の図書館まで歩いた。", "走到學校的圖書館了。"),
            (
                "来週の会議まで日本語を勉強します。",
                "連下週的會議都學習日本語。",
            ),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }

    #[test]
    fn temporal_made_leaves_deadlines_and_other_endpoints_alone() {
        assert_eq!(
            translate("私は六時までに手紙を書きます。"),
            "我最晚在六點寫信。"
        );
        assert_eq!(translate("私は六時に手紙を書きます。"), "我在六點寫信。");
        assert_eq!(translate("先生まで歩いた。"), "連老師都走了。");
        assert_eq!(translate("駅まで歩いた。"), "走到車站了。");
    }

    #[test]
    fn negative_motion_and_a_following_clause_keep_their_scope() {
        let negative = translate("駅まで歩かなかった。");
        assert!(negative.contains("沒有走到車站"), "{negative}");
        let sequence = translate("公園まで歩いて、友達に会った。");
        assert!(sequence.contains("走到公園"), "{sequence}");
        assert!(sequence.contains("朋友"), "{sequence}");
        assert!(sequence.contains("見面"), "{sequence}");
        assert!(!sequence.contains("連公園"), "{sequence}");
    }
}

// A recognized, self-contained numeric clock followed directly by its time
// particle is not a competing destination. Opaque/glossary placeholders and
// mixed groups remain on the conservative path.
fn is_plain_clock_ni(b: &Bunsetsu) -> bool {
    b.case_particle.as_deref() == Some("に") && explicit_clock_case(b).is_some()
}

fn explicit_clock_case(b: &Bunsetsu) -> Option<(&Atom, &Atom)> {
    let [clock, particle] = b.atoms.as_slice() else {
        return None;
    };
    (b.role == BunsetsuRole::Obl
        && matches!(b.case_particle.as_deref(), Some("に" | "まで" | "から"))
        && clock.lemma.as_deref() == Some(CLOCK_ATOM)
        && clock.is_locked
        && !clock.surface.contains("[[PH")
        && !particle.is_locked
        && particle.upos == Some(Upos::Adp)
        && b.case_particle.as_deref() == Some(particle.surface.as_str())
        && clock.span.end_char == particle.span.start_char)
        .then_some((clock, particle))
}

fn clock_case_governor(clock: &Atom, marker: &Atom, doc: &Document) -> Option<usize> {
    if marker.deprel != "case" {
        return None;
    }
    let marker_idx = marker.doc_tok_idx?;
    let original_marker = &doc.tokens.get(marker_idx)?.span;
    if *original_marker == marker.span {
        let time = &doc.tokens.get(marker.head?)?.span;
        if time.start_char >= clock.span.start_char && time.end_char <= clock.span.end_char {
            return deadline_case_governor(marker, doc);
        }
        return None;
    }

    // A token such as 時に can supply both the end of the clock and its case.
    // Its source node must attach as a temporal modifier on its own.
    if marker.surface != "に"
        || original_marker.start_char < clock.span.start_char
        || original_marker.start_char >= clock.span.end_char
        || original_marker.end_char != marker.span.end_char
    {
        return None;
    }
    let node = doc.syntax.iter().find(|node| node.token == marker_idx)?;
    if matches!(node.deprel.as_str(), "advmod" | "obl") && node.head == marker.head {
        node.head
    } else {
        None
    }
}

fn physical_goal_head(b: &Bunsetsu) -> Option<usize> {
    let (particle, content) = b.atoms.split_last()?;
    if !matches!(b.case_particle.as_deref(), Some("に" | "へ" | "まで"))
        || Some(particle.surface.as_str()) != b.case_particle.as_deref()
        || particle.upos != Some(Upos::Adp)
        || particle.is_locked
    {
        return None;
    }
    let noun = content.last()?;
    if noun.span.end_char != particle.span.start_char
        || !matches!(noun.upos, Some(Upos::Noun | Upos::Propn))
    {
        return None;
    }
    if is_building_floor(noun) {
        // The floor marker proves the source attachment before adnominal
        // merging. A synthetic genitive prefix may carry a real placeholder.
        let floor_idx = noun.doc_tok_idx?;
        if particle.head != Some(floor_idx)
            || content
                .iter()
                .any(|a| a.is_locked && (a.doc_tok_idx.is_some() || a.surface.contains("[[PH")))
        {
            return None;
        }
        return noun.head;
    }
    if content.iter().any(|a| a.is_locked) {
        return None;
    }
    let lexical = faberline_nlp::lexicon::lookup_ja(&noun.surface);
    let compatible = lexical.as_ref().is_some_and(|entry| {
        entry.decided_upos == noun.upos
            && matches!(entry.decided_upos, Some(Upos::Noun | Upos::Propn))
    });
    let tagged_place = compatible
        && lexical.as_ref().is_some_and(|entry| {
            entry.sem_class == Some(faberline_nlp::lexicon::SemanticClass::Location)
        });
    // 空港 is a productive, physically specific venue head. Require an exact
    // unambiguous noun lexicon entry, never accept an unknown suffix alone.
    let airport = compatible && noun.upos == Some(Upos::Noun) && noun.surface.ends_with("空港");
    if !(tagged_place
        || airport
        || (compatible && content.len() == 1 && is_existing_spatial_head(noun)))
    {
        return None;
    }
    if content.len() > 1 {
        let head_idx = noun.doc_tok_idx?;
        for pair in content.windows(2) {
            if pair[0].span.end_char != pair[1].span.start_char
                || pair[0].head != Some(head_idx)
                || !matches!(pair[0].deprel.as_str(), "compound" | "nmod")
                || !matches!(pair[0].upos, Some(Upos::Noun | Upos::Propn))
            {
                return None;
            }
        }
    }
    if particle.surface == "まで"
        && (noun.deprel != "obl" || particle.deprel != "case" || particle.head != noun.doc_tok_idx)
    {
        return None;
    }
    noun.head
}

// Compose time-on-time の before the generic adnominal merge renders it as 的
// and loses the source type. Keep non-time modifiers on their original path.
fn merge_temporal_made_anchor(b: &mut Bunsetsu, modifier: &[Atom]) -> bool {
    let [anchor, no] = modifier else {
        return false;
    };
    if no.is_locked
        || no.upos != Some(Upos::Adp)
        || no.surface != "の"
        || anchor.span.end_char != no.span.start_char
        || b.atoms
            .first()
            .is_none_or(|head| no.span.end_char != head.span.start_char)
    {
        return false;
    }
    let Some(time) = temporal_made_time(b) else {
        return false;
    };
    let Some(anchor_text) = render_temporal_made_atom(anchor) else {
        return false;
    };

    // A weekday within a named week has the compact reading 下週一, 上週五, etc.
    let tail = if matches!(anchor.surface.as_str(), "先週" | "今週" | "来週") {
        time.strip_prefix("星期").unwrap_or(&time)
    } else {
        &time
    };
    let head = &mut b.atoms[0];
    head.span.start_char = anchor.span.start_char;
    head.surface = format!("{anchor_text}{tail}");
    head.lemma = Some(TEMPORAL_MADE_COMPOUND_ATOM.to_owned());
    head.is_locked = true;
    true
}

// Only a typed time with adjacent, bare まで licenses the continuation
// reading. Leave までに, opaque groups and other nouns alone.
fn temporal_made_time(b: &Bunsetsu) -> Option<String> {
    let [time, particle] = b.atoms.as_slice() else {
        return None;
    };
    if b.case_particle.as_deref() != Some("まで")
        || particle.is_locked
        || particle.upos != Some(Upos::Adp)
        || particle.surface != "まで"
        || time.span.end_char != particle.span.start_char
    {
        return None;
    }
    render_temporal_made_atom(time)
}

fn render_temporal_made_atom(time: &Atom) -> Option<String> {
    if matches!(
        time.lemma.as_deref(),
        Some(CLOCK_ATOM | TEMPORAL_MADE_COMPOUND_ATOM)
    ) && !time.surface.contains("[[PH")
    {
        return Some(time.surface.clone());
    }
    if time.is_locked || !matches!(time.upos, Some(Upos::Noun | Upos::Adv)) {
        return None;
    }

    // The existing weekday glosses are indexed by the abbreviated form.
    let weekday = time.surface.strip_suffix('日').unwrap_or(&time.surface);
    if matches!(
        weekday,
        "月曜" | "火曜" | "水曜" | "木曜" | "金曜" | "土曜" | "日曜"
    ) {
        return Some(translate_single_ja_term(weekday, None, time.upos));
    }
    // These date and daypart heads are unambiguous as standalone time words.
    let is_time = is_daypart_surface(&time.surface)
        || matches!(
            time.surface.as_str(),
            "今" | "昨日"
                | "今日"
                | "明日"
                | "明後日"
                | "翌日"
                | "先週"
                | "今週"
                | "来週"
                | "週末"
                | "先月"
                | "今月"
                | "来月"
                | "月末"
                | "去年"
                | "今年"
                | "来年"
                | "年末"
        )
        || faberline_nlp::lexicon::lookup_ja(&time.surface).is_some_and(|entry| {
            entry.sem_class == Some(faberline_nlp::lexicon::SemanticClass::Time)
        });
    is_time.then(|| {
        after_daypart(time).unwrap_or_else(|| {
            translate_single_ja_term(&time.surface, time.lemma.as_deref(), time.upos)
        })
    })
}

fn is_spatial_made_bunsetsu(b: &Bunsetsu) -> bool {
    if b.case_particle.as_deref() != Some("まで") {
        return false;
    }
    // Use the noun phrase head, not a place inside a modifier such as
    // 学校の先生. Relational nouns such as 前/後 can also be temporal.
    b.atoms
        .iter()
        .rev()
        .find(|a| {
            !matches!(a.upos, Some(Upos::Adp | Upos::Part | Upos::Punct))
                && !is_case_or_sfp_to_strip(&a.surface)
        })
        .is_some_and(|a| is_building_floor(a) || is_existing_spatial_head(a))
}

fn is_existing_spatial_head(a: &Atom) -> bool {
    !a.is_locked
        && matches!(
            a.surface.as_str(),
            "部屋"
                | "学校"
                | "図書館"
                | "公園"
                | "駅"
                | "東京駅"
                | "東京"
                | "家"
                | "会社"
                | "病院"
                | "ここ"
                | "そこ"
                | "あそこ"
                | "ベッド"
                | "椅子"
                | "机"
                | "風呂"
        )
}

// Intransitive manner verbs can take a traversed space with を. Verbs such
// as 渡る and 通る already express that relation in their Chinese gloss.
#[derive(Clone, Copy)]
struct RouteMotion {
    head: usize,
    expressed_by_verb: bool,
}

fn governed_route_motion(predicates: &[Bunsetsu], source: &[Atom]) -> Option<RouteMotion> {
    let mut verbs = predicates
        .iter()
        .flat_map(|p| &p.atoms)
        .filter(|a| !a.is_locked && a.upos == Some(Upos::Verb));
    let first = verbs.next()?;
    let second = verbs.next();
    if verbs.next().is_some() {
        return None;
    }
    // The grouped predicate may contain just the aspect verb or both verbs.
    let verb = if let Some(aspect) =
        second.or_else(|| (first.lemma.as_deref() == Some("いる")).then_some(first))
    {
        let [motion, _, _] = source.windows(3).find(|w| {
            let (motion, te, iru) = (&w[0], &w[1], &w[2]);
            iru.lemma.as_deref() == Some("いる")
                && iru.doc_tok_idx == aspect.doc_tok_idx
                && iru.doc_tok_idx.is_some()
                && motion.head == iru.doc_tok_idx
                && motion.deprel == "advcl"
                && te.head == motion.doc_tok_idx
                && te.deprel == "mark"
                && is_te_form_link(motion, te, iru)
                && (first.doc_tok_idx == motion.doc_tok_idx || first.doc_tok_idx == iru.doc_tok_idx)
        })?
        else {
            return None;
        };
        motion
    } else {
        first
    };
    let head = verb.doc_tok_idx?;
    let lemma = verb.lemma.as_deref().unwrap_or(&verb.surface);
    let expressed_by_verb = matches!(lemma, "通る" | "渡る" | "くぐる" | "横切る");
    if !(expressed_by_verb
        || matches!(
            lemma,
            "走る"
                | "歩く"
                | "駆ける"
                | "歩き回る"
                | "散歩する"
                | "泳ぐ"
                | "飛ぶ"
                | "進む"
                | "行く"
        ))
        || source.iter().any(|a| {
            a.head == Some(head)
                && a.upos == Some(Upos::Aux)
                && matches!(
                    a.lemma.as_deref().unwrap_or(&a.surface),
                    "せる" | "させる" | "れる" | "られる"
                )
        })
        || source
            .iter()
            .filter(|a| a.head == Some(head) && a.deprel == "obj")
            .count()
            != 1
    {
        return None;
    }
    Some(RouteMotion {
        head,
        expressed_by_verb,
    })
}

fn route_marker(group: &Bunsetsu, source: &[Atom], motion: usize) -> Option<&'static str> {
    let case = group.atoms.last()?;
    if case.is_locked
        || case.upos != Some(Upos::Adp)
        || case.surface != "を"
        || case.deprel != "case"
    {
        return None;
    }
    let head = case.head?;
    let place = source.iter().find(|a| a.doc_tok_idx == Some(head))?;
    if place.is_locked
        || place.upos != Some(Upos::Noun)
        || place.deprel != "obj"
        || place.head != Some(motion)
        || place.span.end_char != case.span.start_char
    {
        return None;
    }
    if let Some(marker) = spatial_route_marker(place) {
        return Some(marker);
    }
    // 中 can also describe time. Verify the place in 公園の中 before moving
    // the complete phrase in front of its governing verb.
    if !matches!(
        place.surface.as_str(),
        "中" | "外" | "上" | "下" | "奥" | "前" | "後ろ"
    ) {
        return None;
    }
    source
        .windows(2)
        .any(|pair| {
            let (owner, no) = (&pair[0], &pair[1]);
            !owner.is_locked
                && !no.is_locked
                && owner.head == Some(head)
                && owner.deprel == "nmod"
                && owner.doc_tok_idx.is_some()
                && no.head == owner.doc_tok_idx
                && no.surface == "の"
                && no.deprel == "case"
                && no.upos == Some(Upos::Adp)
                && owner.span.end_char == no.span.start_char
                && no.span.end_char == place.span.start_char
                && (spatial_route_marker(owner).is_some()
                    || is_existing_spatial_head(owner)
                    || matches!(owner.surface.as_str(), "庭" | "教室" | "建物" | "店"))
        })
        .then_some("在")
}

fn spatial_route_marker(place: &Atom) -> Option<&'static str> {
    match place.surface.as_str() {
        "道" | "道路" | "歩道" | "山道" | "坂道" | "車道" | "遊歩道" | "街道" | "小道" | "通路"
        | "廊下" | "路地" | "橋" | "海岸" | "河岸" | "川岸" | "国道" | "県道" | "林道" | "農道" => {
            Some("沿著")
        }
        "公園" | "広場" | "校庭" | "空" | "海" | "川" | "河" | "トンネル" => {
            Some("在")
        }
        _ => None,
    }
}

fn governs_traversed_space(source: &[Atom], head: usize) -> bool {
    source.windows(2).any(|pair| {
        let (place, wo) = (&pair[0], &pair[1]);
        !place.is_locked
            && place.upos == Some(Upos::Noun)
            && place.deprel == "obj"
            && place.head == Some(head)
            && place.doc_tok_idx.is_some()
            && (spatial_route_marker(place).is_some() || is_existing_spatial_head(place))
            && !wo.is_locked
            && wo.upos == Some(Upos::Adp)
            && wo.surface == "を"
            && wo.deprel == "case"
            && wo.head == place.doc_tok_idx
            && place.span.end_char == wo.span.start_char
    })
}

fn recover_route_predicate(groups: &mut [Atom], source: &[Atom]) {
    // 通って can be either 通る or 通う. A traversed space is marked を.
    // In 学校に通っている, the commuter's destination is marked に instead.
    for stem in groups.iter_mut().filter(|a| {
        !a.is_locked
            && a.surface == "通っ"
            && a.upos == Some(Upos::Verb)
            && a.lemma.as_deref() == Some("通う")
    }) {
        if let Some(head) = stem.doc_tok_idx
            && governs_traversed_space(source, head)
            && source.iter().any(|a| {
                !a.is_locked
                    && a.surface == "て"
                    && a.upos == Some(Upos::Sconj)
                    && a.deprel == "mark"
                    && a.head == Some(head)
                    && a.span.start_char == stem.span.end_char
            })
        {
            stem.lemma = Some("通る".to_owned());
        }
    }
    // 通り can be tagged as a noun before ます. The other polite inflection
    // supplies an independently recognized verb from the same stem.
    let Some(stem) = groups
        .iter_mut()
        .find(|a| !a.is_locked && a.upos == Some(Upos::Noun) && a.deprel == "root")
    else {
        return;
    };
    let Some(head) = stem.doc_tok_idx else { return };
    let polite = source.iter().any(|a| {
        !a.is_locked
            && a.surface == "ます"
            && a.upos == Some(Upos::Aux)
            && a.head == Some(head)
            && a.deprel == "aux"
            && stem.span.end_char == a.span.start_char
    });
    if !polite || !governs_traversed_space(source, head) {
        return;
    }
    let surface: Vec<_> = stem.surface.chars().chain("ません".chars()).collect();
    let segments = faberline_nlp::lexicon::ja::segment_ja(&surface, 0);
    let [verb, polite, negative] = segments.as_slice() else {
        return;
    };
    let len = stem.span.end_char - stem.span.start_char;
    if verb.span.start_char == 0
        && verb.span.end_char == len
        && verb.decided_upos == Some(Upos::Verb)
        && polite.span.start_char == len
        && polite.span.end_char == len + 2
        && polite.decided_upos == Some(Upos::Aux)
        && negative.span.start_char == len + 2
        && negative.span.end_char == surface.len()
        && negative.decided_upos == Some(Upos::Aux)
        && matches!(
            verb.lemma.as_deref(),
            Some("通る" | "渡る" | "くぐる" | "横切る")
        )
    {
        stem.upos = Some(Upos::Verb);
        stem.lemma = verb.lemma.clone();
    }
}

fn is_made_motion_verb(lemma: &str) -> bool {
    matches!(lemma, "行く" | "歩く" | "走る")
}

fn predicates_is_made_motion_verb(predicates: &[Bunsetsu]) -> bool {
    let mut verbs = predicates
        .iter()
        .flat_map(|b| &b.atoms)
        .filter(|a| !a.is_locked && a.upos == Some(Upos::Verb));
    let Some(verb) = verbs.next() else {
        return false;
    };
    // Do not move an endpoint across another lexical predicate. Clauses with
    // multiple verbs need an explicit attachment relation before reordering.
    verbs.next().is_none() && is_made_motion_verb(verb.lemma.as_deref().unwrap_or(&verb.surface))
}

fn predicates_is_goal_verb(predicates: &[Bunsetsu]) -> bool {
    predicates.iter().any(|b| {
        b.atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                    "注ぎ込む"
                        | "注ぐ"
                        | "入れる"
                        | "挿れる"
                        | "届く"
                        | "届ける"
                        | "入る"
                        | "行く"
                        | "来る"
                        | "帰る"
                        | "戻る"
                        | "出かける"
                        | "出掛ける"
                        | "着く"
                        | "向かう"
                        | "乗る"
                        | "座る"
                        | "落ちる"
                        | "落ち込む"
                        | "沈む"
                        | "埋める"
                        | "押し込む"
                        | "突き込む"
                        | "達する"
                )
        })
    })
}

fn predicates_is_cause_verb(predicates: &[Bunsetsu]) -> bool {
    predicates.iter().any(|b| {
        b.atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                    "震える"
                        | "驚く"
                        | "泣く"
                        | "苦しむ"
                        | "酔う"
                        | "溺れる"
                        | "濡れる"
                        | "喘ぐ"
                        | "乱れる"
                        | "悶える"
                        | "喜ぶ"
                        | "悲しむ"
                        | "怒る"
                        | "悩む"
                        | "困る"
                        | "疲れる"
                )
        })
    })
}

// In 誰かと, the first か belongs to the pronoun, not to the finite verb.
// Preserve the original case particle so callers can place this argument.
fn indefinite_argument(atoms: &[Atom], predicate: usize) -> Option<(Atom, Option<Atom>)> {
    let [word, ka, rest @ ..] = atoms else {
        return None;
    };
    let noun_idx = word.doc_tok_idx?;
    if word.is_locked
        || ka.is_locked
        || word.upos != Some(Upos::Pron)
        || word.head != Some(predicate)
        || ka.upos != Some(Upos::Part)
        || ka.surface != "か"
        || ka.deprel != "discourse"
        || ka.head != Some(noun_idx)
        || word.span.end_char != ka.span.start_char
    {
        return None;
    }
    let case = match rest {
        [] => None,
        [case]
            if !case.is_locked
                && case.upos == Some(Upos::Adp)
                && case.deprel == "case"
                && case.head == Some(noun_idx)
                && ka.span.end_char == case.span.start_char =>
        {
            Some(case.clone())
        }
        _ => return None,
    };
    let text = match word.surface.as_str() {
        "誰" => "某個人",
        "何" => "某樣東西",
        "どこ" => "某個地方",
        _ => return None,
    };
    let mut noun = word.clone();
    noun.surface = text.to_owned();
    noun.lemma = None;
    noun.span.end_char = ka.span.end_char;
    noun.is_locked = true;
    Some((noun, case))
}

// A content object may omit を immediately before a transitive activity.
// A separate object or a non-transitive predicate leaves that reading open.
fn indefinite_content_object(
    word: &Atom,
    case: Option<&Atom>,
    end: usize,
    action: &Atom,
    has_object: bool,
) -> bool {
    if word.surface != "何" {
        return false;
    }
    match (word.deprel.as_str(), case) {
        ("obj", Some(case)) => case.surface == "を",
        ("obl", None) => {
            !has_object
                && end == action.span.start_char
                && matches!(
                    action.lemma.as_deref().unwrap_or(&action.surface),
                    "読む"
                        | "飲む"
                        | "食べる"
                        | "書く"
                        | "買う"
                        | "作る"
                        | "見る"
                        | "聞く"
                        | "選ぶ"
                )
        }
        _ => false,
    }
}

// In 何か作業を the pronoun modifies the following object noun. The
// noun supplies the object, and the two-token modifier contributes 一些.
fn indefinite_object_modifier(
    prior: &Bunsetsu,
    argument: &Bunsetsu,
    predicate: usize,
) -> Option<Atom> {
    let object = argument.atoms.first()?;
    let word = prior.atoms.first()?;
    if argument.role != BunsetsuRole::Obj
        || object.is_locked
        || object.deprel != "obj"
        || object.head != Some(predicate)
        || word.surface != "何"
        || word.deprel != "nmod"
    {
        return None;
    }
    let (mut modifier, None) = indefinite_argument(&prior.atoms, object.doc_tok_idx?)? else {
        return None;
    };
    if modifier.span.end_char != object.span.start_char {
        return None;
    }
    modifier.surface = "一些".to_owned();
    Some(modifier)
}

// Consume only a か attached to its own pronoun. Keep the original case for
// clause assembly so an unknown actor does not introduce a sentence question.
fn localize_indefinite_arguments(groups: &mut Vec<Bunsetsu>, action: &Atom, source: &[Atom]) {
    let Some(predicate) = action.doc_tok_idx else {
        return;
    };
    let has_object = source
        .iter()
        .any(|a| a.head == Some(predicate) && a.deprel == "obj");
    let mut previous = Vec::with_capacity(groups.len());
    for mut group in std::mem::take(groups) {
        if let Some(modifier) = previous
            .last()
            .and_then(|prior| indefinite_object_modifier(prior, &group, predicate))
        {
            previous.pop();
            group.atoms.insert(0, modifier);
        }
        if let Some((mut noun, case)) = indefinite_argument(&group.atoms, predicate) {
            if indefinite_content_object(
                &group.atoms[0],
                case.as_ref(),
                noun.span.end_char,
                action,
                has_object,
            ) {
                noun.surface = "點什麼".to_owned();
                group.role = BunsetsuRole::Obj;
            }
            group.sfp_tail = None;
            group.atoms = vec![noun];
            group.atoms.extend(case);
        }
        previous.push(group);
    }
    *groups = previous;
}

fn is_companion_person(noun: &Atom) -> bool {
    use faberline_nlp::lexicon::SemanticClass;
    match noun.upos {
        Some(Upos::Pron) => matches!(
            noun.surface.as_str(),
            "誰" | "だれ" | "私" | "僕" | "彼" | "彼女" | "君" | "あなた" | "みんな" | "皆"
        ),
        Some(Upos::Propn) => {
            is_honorific_suffix(&noun.surface)
                || faberline_nlp::lexicon::lookup_ja(&noun.surface)
                    .is_some_and(|entry| entry.sem_class == Some(SemanticClass::Person))
        }
        Some(Upos::Noun) => is_demonstrative_person_noun(noun),
        _ => false,
    }
}

// In 先生と図書館に the person and the place have separate case particles.
// Two places share their direction instead (東京と大阪に).
fn place_companion(source: &[Atom], predicate: usize, known_place: Option<usize>) -> Option<usize> {
    use faberline_nlp::lexicon::SemanticClass;
    let mut person = None;
    let mut has_place = false;
    let mut after_companion = false;
    for (i, marker) in source.iter().enumerate() {
        if marker.is_locked || marker.upos != Some(Upos::Adp) || marker.deprel != "case" {
            continue;
        }
        let prefix = &source[..i];
        let Some(last) = prefix.last() else { continue };
        let noun = if last.surface == "か"
            && i >= 2
            && indefinite_argument(&source[i - 2..=i], predicate).is_some()
        {
            &source[i - 2]
        } else {
            last
        };
        if noun.is_locked
            || noun.doc_tok_idx.is_none()
            || noun.doc_tok_idx != marker.head
            || noun.head != Some(predicate)
            || !matches!(noun.deprel.as_str(), "obl" | "nsubj" | "obj" | "iobj")
            || last.span.end_char != marker.span.start_char
        {
            continue;
        }
        let class = faberline_nlp::lexicon::lookup_ja(&noun.surface).and_then(|hit| hit.sem_class);
        let human = is_companion_person(noun);
        // 私と先生が and 先生と学生を can share the second noun's case.
        // A later location does not establish a separate companion there.
        // In 先生と二人で the count describes how the two people participate.
        let counted_manner = marker.surface == "で"
            && noun.surface == "人"
            && prefix.iter().any(|n| {
                !n.is_locked
                    && n.head == noun.doc_tok_idx
                    && n.span.end_char == noun.span.start_char
                    && (n.upos == Some(Upos::Num) && n.deprel == "nummod"
                        || n.deprel == "compound"
                            && n.surface
                                .chars()
                                .all(|c| matches!(c, '０'..='９' | '0'..='9')))
            });
        if after_companion && human && !counted_manner {
            return None;
        }
        after_companion = false;
        if noun.deprel != "obl" {
            continue;
        }
        if marker.surface == "と" {
            if human && person.replace(noun.doc_tok_idx?).is_some() {
                return None;
            }
            after_companion = human;
        } else if matches!(marker.surface.as_str(), "へ" | "に" | "で" | "まで") {
            let place = known_place.is_some_and(|id| noun.doc_tok_idx == Some(id))
                || is_place_location_bunsetsu(std::slice::from_ref(noun))
                || class == Some(SemanticClass::Location)
                || noun.surface == "どこ"
                || noun.upos == Some(Upos::Noun)
                    && (noun.surface.ends_with('園')
                        || matches!(noun.surface.as_str(), "階" | "屋")
                            && source.iter().any(|part| {
                                !part.is_locked
                                    && part.head == noun.doc_tok_idx
                                    && part.span.end_char == noun.span.start_char
                                    && matches!(part.deprel.as_str(), "nummod" | "compound")
                            }));
            has_place |= place;
        }
    }
    person.filter(|_| has_place)
}

fn render_bunsetsu_np(b: &Bunsetsu, _target: TargetVariant) -> String {
    let mut out = String::new();
    let mut i = 0usize;
    while i < b.atoms.len() {
        let a = &b.atoms[i];
        if b.case_particle.as_deref() == Some("までに")
            && !a.is_locked
            && let Some(time) = deadline_time_question(a).or_else(|| deadline_weekday(&a.surface))
        {
            out.push_str(time);
            i += 1;
            continue;
        }
        if let Some(noun) = b.atoms.get(i + 1)
            && let Some(demonstrative) = attached_demonstrative(a, noun)
        {
            out.push(demonstrative);
            // A component of a multi-token compound does not establish a classifier
            // for the whole phrase. Render its noun atoms normally in that case.
            if b.atoms.get(i + 2).is_some_and(|next| {
                noun.span.end_char == next.span.start_char
                    && (next.is_locked || matches!(next.upos, Some(Upos::Noun | Upos::Propn)))
            }) {
                i += 1;
                continue;
            }
            let word = translate_single_ja_term(&noun.surface, noun.lemma.as_deref(), noun.upos);
            let classifier = match word.as_str() {
                "書" | "小說" | "雜誌" | "字典" => "本",
                "車" | "公車" | "電車" | "汽車" | "卡車" | "腳踏車" | "自行車" => {
                    "輛"
                }
                "資料" | "文件" | "報告" | "報紙" => "份",
                "商店" | "餐廳" | "公司" => "家",
                "寺" | "寺廟" | "橋" => "座",
                "箱" | "人" | "答案" | "條件" | "蛋糕" | "城鎮" | "藥" => "個",
                _ if is_demonstrative_person_noun(noun) => "個",
                // With no established unit or count reading, keep the deictic
                // without guessing an amount, e.g. 這水 rather than a cup of water.
                _ => "",
            };
            out.push_str(classifier);
            out.push_str(if word == "箱" { "箱子" } else { &word });
            i += 2;
            continue;
        }
        if !a.is_locked
            && (is_case_or_sfp_to_strip(&a.surface)
                || (b.sconj_tail.as_deref() == Some(a.surface.as_str())))
        {
            i += 1;
            continue;
        }
        // Honorific localization: `<Name> + さん/先生/先輩/ちゃん/くん/様`
        if !a.is_locked
            && let Some(next) = b.atoms.get(i + 1)
            && !next.is_locked
            && is_honorific_suffix(&next.surface)
        {
            let name = if a.is_locked {
                a.surface.clone()
            } else {
                translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos)
            };
            let localized = localize_honorific(&name, &next.surface);
            out.push_str(&localized);
            i += 2;
            continue;
        }
        if a.is_locked {
            // Check if next atom is honorific suffix!
            if let Some(next) = b.atoms.get(i + 1)
                && !next.is_locked
                && is_honorific_suffix(&next.surface)
            {
                out.push_str(&localize_honorific(&a.surface, &next.surface));
                i += 2;
                continue;
            }
            out.push_str(&a.surface);
        } else if a.surface == "の" {
            if !out.ends_with('的') {
                out.push('的');
            }
        } else {
            out.push_str(&translate_single_ja_term(
                &a.surface,
                a.lemma.as_deref(),
                a.upos,
            ));
        }
        i += 1;
    }
    out
}

fn render_bunsetsu_obl(b: &Bunsetsu, target: TargetVariant, is_instrument: bool) -> String {
    let core = render_bunsetsu_np(b, target);
    if core.is_empty() {
        return core;
    }
    match b.case_particle.as_deref() {
        Some("で") if is_instrument => {
            if core.starts_with('用') {
                core
            } else {
                format!("用{core}")
            }
        }
        Some("に") if b.atoms.iter().any(|a| calendar::question(a).is_some()) => core,
        Some("に") | Some("で") => {
            if core.starts_with('在')
                || (core.starts_with("[[PH") && b.atoms.iter().any(is_clock_atom))
            {
                core
            } else {
                format!("在{core}")
            }
        }
        Some("から") => format!("從{core}"),
        Some("まで") => format!("到{core}"),
        Some("へ") => format!("往{core}"),
        Some("より") => {
            let has_mo = b
                .atoms
                .iter()
                .any(|a| !a.is_locked && matches!(a.surface.as_str(), "も" | "よりも"));
            if matches!(core.as_str(), "總是" | "平常") {
                "比平常".to_owned()
            } else if has_mo {
                format!("比{core}更加")
            } else {
                format!("比{core}")
            }
        }
        Some("によって") => format!("在{core}下"),
        Some("につきまして" | "につきましては" | "について" | "に関して") => {
            format!("關於{core}")
        }
        _ => core,
    }
}

fn render_bunsetsu_iobj(b: &Bunsetsu, target: TargetVariant) -> String {
    let core = render_bunsetsu_np(b, target);
    if core.is_empty() {
        return core;
    }
    format!("對{core}")
}

fn is_case_or_sfp_to_strip(s: &str) -> bool {
    matches!(
        s,
        "が" | "は"
            | "を"
            | "に"
            | "で"
            | "と"
            | "とも"
            | "や"
            | "によって"
            | "につきまして"
            | "につきましては"
            | "について"
            | "に関して"
            | "から"
            | "まで"
            | "へ"
            | "より"
            | "も"
            | "だって"
            | "か"
            | "よ"
            | "ね"
            | "な"
            | "わ"
            | "ぞ"
            | "ぜ"
            | "かな"
            | "よね"
    )
}

fn localize_honorific(name: &str, suffix: &str) -> String {
    match suffix {
        "さん" => format!("{name}小姐"),
        "先生" => format!("{name}老師"),
        "先輩" => format!("{name}前輩"),
        "ちゃん" => format!("小{name}"),
        "くん" => format!("{name}君"),
        "様" | "さま" => format!("{name}大人"),
        _ => format!("{name}{suffix}"),
    }
}

fn predicate_is_imperative(b: &Bunsetsu) -> bool {
    b.atoms
        .iter()
        .any(|a| !a.is_locked && matches!(a.surface.as_str(), "ください" | "なさい" | "な"))
        || b.atoms.windows(2).any(|w| {
            !w[0].is_locked && w[0].surface == "ない" && !w[1].is_locked && w[1].surface == "で"
        })
        || b.atoms
            .iter()
            .any(|a| !a.is_locked && a.surface == "ないで")
}

/// Synthesize predicate complex from predicate bunsetsu(s).
/// Returns `(pre_verb_mods, verb_core, post_verb_mods, needs_passive_agent)`.
fn synthesize_predicate_complex(
    predicates: &[Bunsetsu],
    _has_obj: bool,
    target: TargetVariant,
    transfer: Option<GivingReceivingPredicate>,
) -> (Vec<String>, String, Vec<String>, bool) {
    if predicates.is_empty() {
        return (Vec::new(), String::new(), Vec::new(), false);
    }

    let mut pre: Vec<String> = Vec::new();
    let mut cores: Vec<String> = Vec::new();
    let mut durative_gloss_index = None;
    let mut post: Vec<String> = Vec::new();
    let mut passive_agent = false;

    let all_atoms: Vec<&Atom> = predicates.iter().flat_map(|b| b.atoms.iter()).collect();
    let surfaces: Vec<&str> = all_atoms.iter().map(|a| a.surface.as_str()).collect();
    let joined = surfaces.join("");

    // Check multi-token predicate endings first.
    let has_naide_kudasai = joined.contains("ないでください");
    let has_naide =
        !has_naide_kudasai && (joined.ends_with("ないで") || joined.contains("ないでくれ"));
    let has_kudasai =
        !has_naide_kudasai && (joined.contains("てください") || joined.contains("ください"));
    let has_te_shimau = joined.contains("てしまう")
        || joined.contains("てしまった")
        || joined.contains("ちゃう")
        || joined.contains("ちゃった")
        || joined.contains("じゃう")
        || joined.contains("じゃった");
    // `ていただいた` contains the letters `ていた`, but is not progressive.
    // Keep a boundary at the humble verb while checking the other aspect atoms.
    let aspect_joined: String = all_atoms
        .iter()
        .map(|a| {
            if !a.is_locked && a.lemma.as_deref().unwrap_or(&a.surface) == "いただく" {
                "|"
            } else {
                a.surface.as_str()
            }
        })
        .collect();
    let has_te_iru = aspect_joined.contains("ている")
        || aspect_joined.contains("ていた")
        || aspect_joined.contains("てる")
        || aspect_joined.contains("てた")
        || aspect_joined.contains("ています")
        || aspect_joined.contains("ていました")
        || aspect_joined.contains("ていない")
        || aspect_joined.contains("ていなく")
        || aspect_joined.contains("ていなかった")
        || aspect_joined.contains("ていません")
        || aspect_joined.contains("ていませんでした");
    let has_sugiru =
        joined.contains("すぎる") || joined.contains("すぎた") || joined.contains("過ぎる");
    let has_tai = surfaces
        .iter()
        .any(|&s| matches!(s, "たい" | "たかっ" | "たかった"));
    let has_obligation_zaru = joined.contains("ざるを得な");
    let has_obligation = has_obligation_zaru
        || joined.contains("ねばなら")
        || joined.contains("なければなら")
        || joined.contains("なくてはなら")
        || joined.contains("なきゃなら");
    let has_wake_demo_nai = surfaces.contains(&"わけでもない");
    let has_wake_dewa_nai = !has_wake_demo_nai && surfaces.contains(&"わけではない");
    let has_rashii = surfaces.contains(&"らしい");
    let has_neg_past =
        !has_obligation && (joined.contains("なかった") || joined.contains("ませんでした"));
    let has_split_polite_neg = predicates.iter().any(|b| {
        b.atoms.windows(2).any(|pair| {
            !pair[0].is_locked
                && !pair[1].is_locked
                && pair[0].upos == Some(Upos::Aux)
                && pair[1].upos == Some(Upos::Aux)
                && pair[0].surface == "ませ"
                && pair[1].surface == "ん"
                && pair[0].span.end_char == pair[1].span.start_char
        })
    });
    let has_neg = !has_neg_past
        && !has_naide
        && !has_naide_kudasai
        && !has_obligation
        && (has_split_polite_neg
            || all_atoms.iter().enumerate().any(|(idx, a)| {
                if a.is_locked {
                    return false;
                }
                match a.surface.as_str() {
                    "ない" | "ません" | "ず" | "ぬ" => true,
                    "なく" => !all_atoms.get(idx + 1).is_some_and(|nx| {
                        !nx.is_locked
                            && matches!(nx.surface.as_str(), "なる" | "なり" | "なっ" | "なった")
                    }),
                    "ん" => {
                        idx > 0
                            && !all_atoms[idx - 1].is_locked
                            && all_atoms[idx - 1].upos == Some(Upos::Verb)
                            && !matches!(all_atoms[idx - 1].surface.as_str(), "た" | "だ")
                            && !all_atoms.get(idx + 1).is_some_and(|nx| {
                                !nx.is_locked
                                    && matches!(
                                        nx.surface.as_str(),
                                        "だ" | "です" | "だけど" | "だから" | "なのに" | "なら"
                                    )
                            })
                    }
                    _ => false,
                }
            }));
    let has_past = all_atoms.iter().enumerate().any(|(idx, a)| {
        if a.is_locked {
            return false;
        }
        match a.surface.as_str() {
            "た" | "ました" | "だった" | "でした" => true,
            "だ" => {
                idx > 0
                    && !all_atoms[idx - 1].is_locked
                    && all_atoms[idx - 1].upos == Some(Upos::Verb)
                    && all_atoms[idx - 1]
                        .surface
                        .chars()
                        .last()
                        .is_some_and(|c| matches!(c, 'ん' | 'い'))
            }
            _ => false,
        }
    });
    let has_passive = surfaces.iter().any(|&s| {
        matches!(
            s,
            "れる" | "られる" | "れ" | "られ" | "され" | "される" | "された"
        ) || s.ends_with("され")
            || s.ends_with("される")
            || s.ends_with("された")
    });
    let has_causative = surfaces
        .iter()
        .any(|&s| matches!(s, "せる" | "させる" | "せ" | "させ"));
    let is_copula_neg = !has_wake_dewa_nai
        && !has_wake_demo_nai
        && (joined.contains("じゃない")
            || joined.contains("ではない")
            || joined.contains("ではありません"));
    let is_copula_pos = !is_copula_neg && surfaces.iter().any(|&s| matches!(s, "だ" | "です"));
    let has_locative_floor = all_atoms.iter().any(|a| is_building_floor(a));
    let is_stative_te_iru = has_te_iru
        && all_atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                    "残る"
                        | "知る"
                        | "持つ"
                        | "住む"
                        | "着る"
                        | "似る"
                        | "結婚する"
                        | "間違う"
                        | "間違える"
                )
        });
    let is_stative_cogn_neg_past = has_neg_past
        && all_atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                    "知る" | "わかる" | "分かる"
                )
        });
    let is_progress_suppress_te_iru = has_te_iru
        && all_atoms.iter().any(|a| {
            let sym = a.lemma.as_deref().unwrap_or(a.surface.as_str());
            matches!(sym, "なる" | "なる_adj" | "かかる" | "登場")
                || (!a.is_locked
                    && (a.surface.starts_with("なっ")
                        || a.surface.starts_with("かかっ")
                        || a.surface == "登場"))
        });
    let is_toujou_te_iru = has_te_iru
        && all_atoms
            .iter()
            .any(|a| a.surface == "登場" || a.lemma.as_deref() == Some("登場"));

    if has_rashii {
        pre.push("聽說".to_owned());
    }
    if has_naide_kudasai {
        pre.push("請不要".to_owned());
    } else if has_naide {
        pre.push("不要".to_owned());
    } else if has_kudasai {
        pre.push("請".to_owned());
    }
    if has_causative {
        pre.push("讓".to_owned());
    }
    if has_passive {
        passive_agent = true;
    }
    if has_wake_demo_nai {
        pre.push("也並不是".to_owned());
    } else if has_wake_dewa_nai {
        pre.push("並不是".to_owned());
    }
    if is_copula_neg {
        pre.push(
            if has_locative_floor {
                "不在"
            } else {
                "不是"
            }
            .to_owned(),
        );
    } else if is_stative_cogn_neg_past {
        pre.push("不".to_owned());
    } else if has_neg_past || (has_te_iru && has_neg) {
        pre.push("沒有".to_owned());
    } else if has_neg {
        pre.push("不".to_owned());
    }
    if has_obligation {
        pre.push(if has_obligation_zaru {
            "不得不".to_owned()
        } else {
            "必須".to_owned()
        });
    }
    if has_tai {
        pre.push("想".to_owned());
    }
    if is_toujou_te_iru {
        pre.push("已".to_owned());
    } else if has_te_iru
        && !is_stative_te_iru
        && !is_progress_suppress_te_iru
        && !has_neg
        && !has_neg_past
    {
        pre.push("正在".to_owned());
    }
    if has_sugiru {
        pre.push("太".to_owned());
    }

    let obl_start_idx = if has_obligation {
        (0..all_atoms.len()).find(|&idx| {
            let tail = surfaces[idx..].join("");
            tail.starts_with("ねばなら")
                || tail.starts_with("なければなら")
                || tail.starts_with("なくてはなら")
                || tail.starts_with("なきゃなら")
                || tail.starts_with("ざるを得な")
        })
    } else {
        None
    };

    for (idx, a) in all_atoms.iter().enumerate() {
        if let Some(obl_start) = obl_start_idx
            && idx >= obl_start
        {
            continue;
        }
        if a.is_locked {
            cores.push(a.surface.clone());
            continue;
        }
        if let Some(transfer) = transfer
            && idx == transfer.atom_index
        {
            if !transfer.auxiliary {
                cores.push(match transfer.kind {
                    GivingReceivingKind::Give | GivingReceivingKind::GiveTowardSpeaker => {
                        "給".to_owned()
                    }
                    GivingReceivingKind::Receive => "收到".to_owned(),
                });
            }
            continue;
        }
        if is_predicate_aux_or_tail(&a.surface) {
            continue;
        }
        if !cores.is_empty() && matches!(a.surface.as_str(), "する" | "し" | "さ" | "せ" | "すれ")
        {
            continue;
        }
        if a.surface == "い"
            && (a.lemma.as_deref() == Some("いる")
                || (idx > 0 && matches!(surfaces[idx - 1], "て" | "で")))
        {
            continue;
        }
        if a.upos == Some(Upos::Sconj) || a.upos == Some(Upos::Part) {
            continue;
        }
        let surf_clean = a
            .surface
            .strip_suffix("され")
            .or_else(|| a.surface.strip_suffix("される"))
            .or_else(|| a.surface.strip_suffix("された"))
            .unwrap_or(&a.surface);
        let tr = translate_single_ja_term(surf_clean, a.lemma.as_deref(), a.upos);
        if !tr.is_empty() {
            if a.upos == Some(Upos::Verb) && tr.ends_with(['著', '着']) {
                durative_gloss_index = Some(cores.len());
            }
            cores.push(tr);
        }
    }

    if is_stative_te_iru && let Some(last) = cores.last_mut() {
        if last == "留下" {
            *last = "留".to_owned();
        } else if all_atoms.iter().any(|a| {
            !a.is_locked
                && (a.lemma.as_deref() == Some("間違う")
                    || (a.surface.starts_with("間違っ") && a.lemma.as_deref() != Some("間違える")))
        }) {
            *last = "有錯".to_owned();
        }
    }

    let has_single_aru_core = cores.len() == 1
        && cores[0] == "有"
        && all_atoms.iter().all(|a| !a.is_locked)
        && all_atoms
            .iter()
            .filter(|a| a.upos == Some(Upos::Verb))
            .count()
            == 1
        && all_atoms.iter().any(|a| {
            a.upos == Some(Upos::Verb) && a.lemma.as_deref().unwrap_or(&a.surface) == "ある"
        });
    if has_single_aru_core && (has_neg || has_neg_past) && !is_copula_neg {
        pre.retain(|m| m != "不" && m != "沒有");
        cores[0] = "沒有".to_owned();
    }

    let is_sole_existential_neg = cores.is_empty()
        && !is_copula_pos
        && !is_copula_neg
        && !has_wake_demo_nai
        && !has_wake_dewa_nai
        && all_atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.surface.as_str(),
                    "ない" | "なかっ" | "なかった" | "ありません" | "ませんでした"
                )
        });
    if is_sole_existential_neg {
        pre.retain(|m| m != "不" && m != "沒有");
        cores.push("沒有".to_owned());
    } else if is_copula_pos && cores.is_empty() {
        cores.push("是".to_owned());
    } else if is_copula_pos
        && !cores.is_empty()
        && !all_atoms
            .iter()
            .any(|a| a.upos == Some(Upos::Verb) || a.upos == Some(Upos::Adj))
    {
        let copula = if all_atoms
            .iter()
            .any(|a| calendar::is_date(a) && a.deprel == "root")
            && all_atoms
                .iter()
                .any(|a| !a.is_locked && a.surface == "だけ")
        {
            "只在"
        } else if has_locative_floor {
            "在"
        } else {
            "是"
        };
        pre.insert(0, copula.to_owned());
    }

    if has_te_shimau {
        if all_atoms
            .iter()
            .any(|a| matches!(a.lemma.as_deref(), Some("なる" | "なる_adj")))
        {
            post.push("了".to_owned());
        } else {
            post.push("掉了".to_owned());
        }
    } else if is_stative_te_iru
        && !has_neg
        && !has_neg_past
        && !matches!(
            cores.last().map(String::as_str),
            Some("知道" | "認識" | "了解" | "像" | "相似" | "結婚" | "有錯" | "錯" | "弄錯")
        )
    {
        let marker = if target == TargetVariant::ZhHans {
            '着'
        } else {
            '著'
        };
        if let Some(index) = durative_gloss_index
            && index + 1 == cores.len()
        {
            // The final verb gloss already carries the aspect, e.g. 拿著.
            cores[index].pop();
            cores[index].push(marker);
        } else {
            post.push(marker.to_string());
        }
    } else if has_past
        && !has_neg_past
        && !is_sole_existential_neg
        && !has_naide_kudasai
        && !has_naide
        && !has_te_iru
        && !has_obligation
    {
        post.push("了".to_owned());
    }

    (pre, cores.join(""), post, passive_agent)
}

fn is_predicate_aux_or_tail(s: &str) -> bool {
    matches!(
        s,
        "た" | "だ"
            | "だっ"
            | "です"
            | "でし"
            | "ます"
            | "まし"
            | "ました"
            | "ません"
            | "ませ"
            | "ましょ"
            | "ませんでした"
            | "だった"
            | "でした"
            | "ない"
            | "なかっ"
            | "なかった"
            | "なく"
            | "なけれ"
            | "なきゃ"
            | "なくちゃ"
            | "ないで"
            | "ず"
            | "ぬ"
            | "ん"
            | "たい"
            | "たく"
            | "たかっ"
            | "たかった"
            | "れる"
            | "られる"
            | "れ"
            | "られ"
            | "され"
            | "される"
            | "された"
            | "せる"
            | "させる"
            | "せ"
            | "させ"
            | "て"
            | "で"
            | "いる"
            | "いた"
            | "います"
            | "いました"
            | "てる"
            | "てた"
            | "しまう"
            | "しまった"
            | "しまっ"
            | "ちゃう"
            | "ちゃっ"
            | "ちゃった"
            | "じゃう"
            | "じゃっ"
            | "じゃった"
            | "ちまう"
            | "ちまっ"
            | "ちまった"
            | "とく"
            | "とい"
            | "といた"
            | "といて"
            | "くれる"
            | "くれ"
            | "くれた"
            | "もらう"
            | "もらっ"
            | "もらった"
            | "あげる"
            | "あげ"
            | "あげた"
            | "ください"
            | "くださ"
            | "くださる"
            | "なさい"
            | "すぎる"
            | "すぎ"
            | "すぎた"
            | "過ぎる"
            | "やすい"
            | "にくい"
            | "じゃない"
            | "ではない"
            | "ではありません"
            | "てい"
            | "てお"
            | "う"
            | "よう"
            | "から"
            | "ので"
            | "けど"
            | "けれど"
            | "けれども"
            | "のに"
            | "たら"
            | "なら"
            | "ば"
            | "ながら"
            | "つつ"
            | "ても"
            | "でも"
            | "と"
            | "ほど"
            | "の"
            | "わけではない"
            | "わけでもない"
            | "らしい"
            | "くせに"
            | "としても"
            | "だけど"
    )
}

fn is_discourse_connector(s: &str) -> bool {
    matches!(
        s,
        "しかし"
            | "だが"
            | "けれど"
            | "けれども"
            | "けど"
            | "ところが"
            | "だから"
            | "それなのに"
            | "そのため"
            | "したがって"
            | "ゆえに"
            | "そして"
            | "それに"
            | "しかも"
            | "それから"
            | "その後"
            | "また"
            | "もし"
            | "なぜなら"
            | "でも"
            | "だって"
            | "すると"
            | "そのまま"
            | "つまり"
            | "たとえば"
            | "例えば"
            | "とにかく"
            | "やはり"
            | "やっぱり"
            | "もちろん"
            | "勿論"
    )
}

fn lookup_curated_phrase(s: &str) -> Option<&'static str> {
    Some(match s {
        // Adult / erotic / light-novel idioms
        "気持ちいい" | "気持ち良い" | "きもちいい" | "キモチイイ" => "好舒服",
        "頭が真っ白になる" | "頭が真っ白" => "腦中一片空白",
        "息が荒くなる" | "息が荒くなり" | "息が荒く" | "息が荒い" => {
            "呼吸變得急促"
        }
        "声を漏らす" | "声が漏れる" => "漏出聲音",
        "奥まで届く" | "奥まで届き" | "奥まで届い" | "奥まで" => "頂到最深處",
        "熱く濡れる" | "熱く濡れ" => "濡濕發燙",
        "快感に震える" | "快感に震え" => "因快感而顫抖",
        "我慢できない" | "我慢できなく" => "忍不住了",
        "中に出す" | "中にだす" => "射在裡面",
        "いっちゃう" | "イッちゃう" | "イキそう" => "要去了",
        "腰を振る" => "擺動腰部",
        "腰が抜ける" => "雙腿發軟",
        "力が抜ける" => "全身無力",
        "顔が赤くなる" => "臉紅了",
        "頬を染める" => "臉頰染上紅暈",
        "目が潤む" => "眼眶濕潤",
        "舌を絡める" => "舌頭交纏",
        "抱きしめる" => "緊緊抱住",
        "押し倒す" => "推倒",
        "雨が降る" | "雨が降り" | "雨が降っ" => "下雨",
        "意識が飛ぶ" | "意識が飛び" | "意識が飛ん" => "神智飛散",
        // Formal / literary collocations
        "努力を重ねる" | "努力を重ね" => "累積努力",
        "拍車をかける" | "拍車をかけ" => "加劇",
        // Spoken & analytical idioms
        "嘘をつく" | "嘘をつき" | "嘘をつい" => "說謊",
        "電話する" | "電話し" => "打電話",
        "ゲームする" | "ゲームし" | "ゲームをする" => "玩遊戲",
        "責任を持つ" | "責任を持っ" | "責任を持ち" => "承擔責任",
        // Everyday greetings & idioms
        "おはようございます" => "早安",
        "おはよう" => "早安",
        "こんにちは" => "你好",
        "こんばんは" => "晚安",
        "ありがとうございます" | "ありがとう" => "謝謝",
        "すみません" => "不好意思",
        "ごめんなさい" => "對不起",
        "いただきます" => "我要開動了",
        "ごちそうさまでした" => "謝謝款待",
        "よろしくお願いします" => "請多指教",
        "おやすみなさい" => "晚安",
        "大丈夫" => "沒關係",
        "どうしたの" | "どうしたんですか" => "怎麼了",
        "なんでもない" | "何でもない" => "沒什麼",
        "仕方がない" | "仕方ない" => "沒辦法",
        "かもしれない" => "也許",
        "に違いない" => "一定是",
        "わけがない" | "はずがない" => "不可能",
        "愛してる" => "我愛你",
        "大好き" => "最喜歡",
        _ => return None,
    })
}

fn translate_single_ja_term(surface: &str, lemma: Option<&str>, upos: Option<Upos>) -> String {
    if lemma == Some(BUILDING_FLOOR)
        && let Some(label) = surface.strip_suffix('階')
    {
        return format!("{label}樓");
    }
    // 1. Curated Japanese -> zh-Hant dictionary (covers false friends, pronouns, connectors,
    //    core verbs/adjectives/nouns, katakana, onomatopoeia, and adult/erotic terms).
    if let Some(cur) = lookup_curated_ja_word(surface) {
        return cur.to_owned();
    }
    if let Some(lem) = lemma
        && lem != surface
        && let Some(cur) = lookup_curated_ja_word(lem)
    {
        return cur.to_owned();
    }

    // 2. Embedded lexicon lookup (`lookup_ja_gloss` with UPOS disambiguation).
    if let Some(gloss) = lookup_ja_lexicon_gloss(surface, lemma, upos) {
        return gloss;
    }

    // 2b. Inflected Adj-`く` stem fallback when `lemma` is absent (`羨ましく` -> `羨ましい`).
    if upos == Some(Upos::Adj)
        && lemma.is_none()
        && let Some(stem) = surface.strip_suffix('く')
        && !stem.is_empty()
    {
        let dict_form = format!("{stem}い");
        if let Some(cur) = lookup_curated_ja_word(&dict_form) {
            return cur.to_owned();
        }
        if let Some(gloss) = lookup_ja_lexicon_gloss(&dict_form, Some(&dict_form), Some(Upos::Adj))
        {
            return gloss;
        }
    }

    // 3. Pure Han kanji compound fallback: map Japanese Shinjitai -> Traditional Han, then
    //    convert to zh-Hant.
    if !surface.is_empty() && surface.chars().all(is_han) {
        let mapped = faberline_nlp::lexicon::ja::map_ja_shinjitai(surface);
        return convert_zh_text_to_target(&mapped, TargetVariant::ZhHant);
    }

    // 4. Fallback: preserve surface.
    surface.to_owned()
}

fn lookup_curated_ja_word(s: &str) -> Option<&'static str> {
    Some(match s {
        // Discourse connectors & adverbs
        "しかし" | "だが" | "けれど" | "けれども" | "けど" | "ところが" => "但是",
        "だから" | "そのため" | "したがって" | "ゆえに" => "所以",
        "そして" | "それに" | "しかも" => "而且",
        "それから" | "その後" => "然後",
        "また" => "此外",
        "もし" => "如果",
        "だからといって" => "但話雖如此",
        "とはいえ" => "話雖如此",
        "からといって" => "即便因為",
        "必ずしも" => "未必",
        "なぜなら" | "だって" => "因為",
        "でも" => "不過",
        "すると" => "於是",
        "そのまま" => "就這樣",
        "つまり" => "也就是說",
        "たとえば" | "例えば" => "例如",
        "とにかく" => "總之",
        "やはり" | "やっぱり" => "果然",
        "もちろん" | "勿論" => "當然",
        "まだ" => "還",
        "もう" => "已經",
        "すぐ" | "すぐに" => "馬上",
        "やっと" | "とうとう" => "終於",
        "いきなり" | "突然" => "突然",
        "ゆっくり" => "慢慢地",
        "しっかり" => "好好地",
        "のんびり" => "悠閒地",
        "びっくり" => "嚇一跳",
        "そっと" => "輕輕地",
        "じっと" => "靜靜地",
        "ぐっと" => "猛地",
        "ぎゅっと" => "緊緊地",
        "びくっと" | "ビクッと" => "猛地一顫",
        "とろっと" => "融化般地",
        "とても" | "すごく" => "非常",
        "本当に" => "真的",
        "少し" | "ちょっと" => "一點",
        "たくさん" => "很多",
        "全部" => "全部",
        "いつも" => "總是",
        "よく" => "經常",
        "ときどき" | "時々" => "有時候",
        "全然" => "完全",
        "決して" => "絕",
        "あまり" => "不太",
        "もっと" => "更加",
        "お願い" => "拜託",
        "久しぶり" => "久違",
        "久しぶりに" => "久違地",
        "同じ" => "同樣",
        // Pronouns & demonstratives
        "私" | "わたし" | "俺" | "おれ" | "僕" | "ぼく" | "あたし" => "我",
        "あなた" | "君" | "きみ" | "お前" | "おまえ" => "你",
        "彼" => "他",
        "彼女" => "她",
        "私たち" | "僕たち" | "俺たち" => "我們",
        "彼ら" => "他們",
        "彼女ら" => "她們",
        "これ" => "這個",
        "それ" | "あれ" => "那個",
        "ここ" => "這裡",
        "そこ" | "あそこ" => "那裡",
        "誰" => "誰",
        "何" | "なに" | "なん" => "什麼",
        "どこ" => "哪裡",
        "いつ" => "什麼時候",
        "なぜ" | "どうして" => "為什麼",
        "どう" => "怎麼",
        // False friends & core nouns
        "手紙" => "信",
        "勉強" => "學習",
        "娘" => "女兒",
        "愛人" => "情婦",
        "老婆" => "老太婆",
        "湯" => "熱水",
        "新聞" => "報紙",
        "汽車" => "火車",
        "電車" => "電車",
        "机" => "書桌",
        "本" => "書",
        "ご飯" => "飯",
        "水" => "水",
        "お茶" => "茶",
        "コーヒー" => "咖啡",
        "パン" => "麵包",
        "りんご" | "リンゴ" => "蘋果",
        "猫" => "貓",
        "犬" => "狗",
        "雨" => "雨",
        "雪" => "雪",
        "風" => "風",
        "花" => "花",
        "空" => "天空",
        "海" => "大海",
        "山" => "山",
        "人" => "人",
        "男" => "男人",
        "女" => "女人",
        "子供" => "孩子",
        "女の子" => "女孩",
        "男の子" => "男孩",
        "友達" => "朋友",
        "先生" => "老師",
        "学生" => "學生",
        "部屋" => "房間",
        "学校" => "學校",
        "図書館" => "圖書館",
        "公園" => "公園",
        "駅" => "車站",
        "家" => "家",
        "今日" => "今天",
        "明日" => "明天",
        "昨日" => "昨天",
        "今" => "現在",
        "毎日" => "每天",
        "朝" => "早上",
        "昼" => "中午",
        "夜" => "夜晚",
        "昔" => "以前",
        "時間" => "時間",
        "お金" => "金錢",
        "余裕" => "餘裕",
        "暇" => "空閒",
        // Core verbs
        "走る" => "跑",
        "歩く" => "走",
        "行く" => "去",
        "来る" => "來",
        "帰る" => "回去",
        "出かける" | "出掛ける" => "出門",
        "過ごす" => "度過",
        "なれる" => "能變得",
        "食べる" => "吃",
        "飲む" => "喝",
        "見る" => "看",
        "聞く" => "聽",
        "話す" => "說",
        "言う" => "說",
        "読む" => "讀",
        "書く" => "寫",
        "買う" => "買",
        "売る" => "賣",
        "作る" => "做",
        "使う" => "使用",
        "知る" => "知道",
        "思う" => "想",
        "考える" => "思考",
        "忘れる" => "忘記",
        "覚える" => "記住",
        "諦める" => "放棄",
        "待つ" => "等待",
        "会う" => "見面",
        "遊ぶ" => "玩",
        "働く" => "工作",
        "休む" => "休息",
        "寝る" => "睡覺",
        "起きる" => "起床",
        "立つ" => "站",
        "座る" => "坐",
        "入る" => "進入",
        "出る" => "出去",
        "開ける" => "打開",
        "閉める" => "關上",
        "始める" => "開始",
        "終わる" => "結束",
        "笑う" => "笑",
        "泣く" => "哭",
        "怒る" => "生氣",
        "喜ぶ" => "高興",
        "愛する" => "愛",
        "する" => "做",
        "ある" => "有",
        "いる" => "在",
        "なる" => "變成",
        "できる" => "能夠",
        "わかる" | "分かる" => "明白",
        // Adjectives
        "大きい" => "大",
        "小さい" => "小",
        "新しい" => "新",
        "古い" => "舊",
        "良い" | "いい" => "好",
        "悪い" => "壞",
        "高い" => "高",
        "安い" => "便宜",
        "長い" => "長",
        "短い" => "短",
        "早い" | "速い" => "快",
        "遅い" => "慢",
        "美しい" | "きれい" | "綺麗" => "美麗",
        "可愛い" | "かわいい" => "可愛",
        "嬉しい" => "開心",
        "悲しい" => "悲傷",
        "楽しい" => "快樂",
        "恥ずかしい" => "害羞",
        "怖い" => "害怕",
        "痛い" => "痛",
        "冷たい" => "冰冷",
        "暖かい" => "溫暖",
        "暗い" => "昏暗",
        "明るい" => "明亮",
        "静か" => "安靜",
        "賑やか" => "熱鬧",
        "元気" => "有精神",
        "大切" => "重要",
        "簡単" => "簡單",
        "難しい" => "困難",
        "美味しい" => "好吃",
        "羨ましい" | "うらやましい" => "羨慕",
        "無駄" => "浪費",
        "贅沢" => "奢侈",
        "凄い" | "すごい" => "好厲害",
        // Literary, anatomical, and adult/erotic vocabulary (content-neutral, uncensored)
        "触る" | "触れる" => "觸摸",
        "舐める" => "舔",
        "吸う" => "吮吸",
        "噛む" => "咬",
        "抱く" => "抱住",
        "脱ぐ" => "脫下",
        "着る" => "穿上",
        "入れる" => "插入",
        "抜く" => "抽出",
        "突く" => "頂入",
        "揺れる" | "揺らす" => "搖晃",
        "震える" => "顫抖",
        "漏れる" | "漏らす" => "漏出",
        "零れる" => "溢出",
        "残る" => "留下",
        "溢れる" => "溢出",
        "溶ける" => "融化",
        "乱れる" => "凌亂",
        "感じる" => "感覺到",
        "囁く" => "低聲呢喃",
        "喘ぐ" => "喘息",
        "見つめる" => "凝視",
        "濡れる" => "濕潤",
        "包む" | "包ま" => "包圍",
        "イく" | "イク" => "高潮",
        "射精" => "射精",
        "絶頂" => "絕頂",
        "快感" => "快感",
        "嬌声" => "嬌喘",
        "喘ぎ" => "喘息",
        "乳首" => "乳頭",
        "おっぱい" => "雙乳",
        "胸" => "胸部",
        "腰" => "腰",
        "尻" | "お尻" => "臀部",
        "太もも" => "大腿",
        "内側" => "內側",
        "膣" => "陰道",
        "陰茎" | "ペニス" | "ちんぽ" | "ちんこ" => "陰莖",
        "亀頭" => "龜頭",
        "クリトリス" => "陰蒂",
        "まんこ" | "オマンコ" | "アソコ" => "小穴",
        "アナル" => "後庭",
        "精液" => "精液",
        "愛液" => "愛液",
        "中出し" => "內射",
        "顔射" => "顏射",
        "緊縛" => "緊縛",
        "調教" => "調教",
        "痴漢" => "癡漢",
        "触手" => "觸手",
        "媚薬" => "媚藥",
        "発情" => "發情",
        "余韻" => "餘韻",
        "痙攣" => "痙攣",
        "愛撫" => "愛撫",
        "フェラ" | "フェラチオ" => "口交",
        "パイズリ" => "乳交",
        "セックス" => "做愛",
        "エッチ" => "親熱",
        "ダメ" => "不行",
        "やめて" => "不要",
        "激しい" => "激烈",
        "熱い" => "滾燙",
        "濡れた" => "濕潤的",
        "固い" | "硬い" => "堅硬",
        "柔らかい" => "柔軟",
        "気持ち" => "感覺",
        "声" => "聲音",
        "息" => "呼吸",
        "体" | "身体" => "身體",
        "顔" => "臉",
        "頬" => "臉頰",
        "唇" => "嘴唇",
        "舌" => "舌頭",
        "首" => "脖子",
        "肩" => "肩膀",
        "腕" => "手臂",
        "手" => "手",
        "指" => "手指",
        "足" | "脚" => "腿",
        "髪" => "頭髮",
        "目" | "瞳" => "眼睛",
        "耳" => "耳朵",
        "鼻" => "鼻子",
        "口" => "嘴",
        "心" => "心",
        "奥" => "深處",
        "中" => "裡面",
        "場" => "場所",
        _ => return None,
    })
}

#[cfg(test)]
mod payment_instrument_tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

    #[test]
    fn payment_method_follows_the_payment_predicate() {
        let mut options = TranslateOptions::new(TargetVariant::ZhHant);
        options.source_hint = Some(TargetVariant::Ja);
        let glossary = TranslationGlossary::new();

        for (source, expected) in [
            ("カードで支払います。", "用卡片結帳。"),
            ("現金で払います。", "用現金付錢。"),
            ("店でカードで支払います。", "在商店用卡片結帳。"),
            ("会社のカードで払います。", "用公司的卡片付錢。"),
            ("私は小銭で払いました。", "我用小錢付錢了。"),
            ("受付で電子マネーで支払います。", "在受付用電子マネー結帳。"),
            ("私は小 銭で払いました。", "我在小錢付錢了。"),
            ("電子マネーの窓口で払います。", "在電子マネー的窓口付錢。"),
            ("電子マネーで困っています。", "在電子マネー正在傷腦筋。"),
            ("現金の窓口で支払います。", "在現金的窓口結帳。"),
            ("二人で支払います。", "在二人結帳。"),
            (
                "生徒は教室で教師に宿題を出しました。",
                "學生在教室把作業交給老師了。",
            ),
            (
                "先生は二階で学生に宿題を出します。",
                "老師在二樓出作業給學生。",
            ),
            (
                "私は店で支払い、友達はカードで遊びます。",
                "我在商店結帳，朋友在卡片遊びます。",
            ),
        ] {
            let output = analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                .expect("the CPU translator handles the payment instrument control");
            assert_eq!(output.target_text, expected, "{source}");
        }
    }
}

#[cfg(test)]
mod existence_rule_tests {
    use super::*;
    use crate::{GlossaryEntry, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;
    use std::collections::BTreeMap;

    fn translate(source: &str, glossary: &TranslationGlossary) -> crate::TranslatedDocument {
        analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            glossary,
            None,
            Backend::Onnx,
        )
        .unwrap()
    }

    fn atom(surface: &str, lemma: Option<&str>, upos: Upos, start: usize) -> Atom {
        Atom {
            span: Span {
                start_char: start,
                end_char: start + surface.chars().count(),
            },
            surface: surface.to_owned(),
            lemma: lemma.map(str::to_owned),
            upos: Some(upos),
            doc_tok_idx: None,
            head: None,
            deprel: String::new(),
            is_locked: false,
        }
    }

    #[test]
    fn residence_uses_the_glossary_role_and_preserves_an_ambiguous_protected_argument() {
        let entry = GlossaryEntry {
            canonical: "青葉".to_owned(),
            surfaces: vec!["青葉".to_owned()],
            targets: BTreeMap::from([(TargetVariant::ZhHant, "青葉".to_owned())]),
            ..GlossaryEntry::default()
        };
        let source = "私は青葉と京都に住んでいます。";
        let mut glossary = TranslationGlossary::new();
        glossary.actors.push(entry.clone());
        let actor = translate(source, &glossary);
        assert_eq!(actor.target_text, "我和青葉住在京都。");
        assert_eq!(actor.glossary_hits.len(), 1);
        glossary.actors.clear();
        glossary.places.push(entry);
        let place = translate(source, &glossary);
        assert_eq!(place.target_text, "我住在青葉和京都。");
        assert_eq!(place.glossary_hits.len(), 1);

        let plain = TranslationGlossary::new();
        for (jp, zh) in [
            ("私は`Aki`と京都に住んでいます。", "我`Aki`と住在京都。"),
            (
                "私は`Aki`と一緒に京都に住んでいます。",
                "我和`Aki`一起住在京都。",
            ),
        ] {
            let doc = translate(jp, &plain);
            assert_eq!(doc.target_text, zh);
            assert_eq!(doc.target_text.matches("`Aki`").count(), 1);
        }
    }

    #[test]
    fn residence_preserves_a_glossary_place_once() {
        let mut glossary = TranslationGlossary::new();
        glossary.places.push(GlossaryEntry {
            canonical: "青葉町".to_owned(),
            surfaces: vec!["青葉町".to_owned()],
            targets: BTreeMap::from([(TargetVariant::ZhHant, "青葉鎮".to_owned())]),
            ..GlossaryEntry::default()
        });
        let doc = translate("私は青葉町に住んでいます。", &glossary);
        assert_eq!(doc.target_text, "我住在青葉鎮。");
        assert_eq!(doc.glossary_hits.len(), 1);
        let span = doc.glossary_hits[0].target_span;
        assert_eq!(
            doc.target_text
                .chars()
                .skip(span.start_char)
                .take(span.end_char - span.start_char)
                .collect::<String>(),
            "青葉鎮"
        );
    }

    #[test]
    fn building_floor_keeps_location_destination_and_polarity() {
        for (source, expected) in [
            ("私の部屋は七階です。", "我的房間在七樓。"),
            ("私の部屋は七階ではありません。", "我的房間不在七樓。"),
            ("私は学校の二階に行きます。", "我去學校的二樓。"),
            ("私は学校の二階に行きません。", "我不去學校的二樓。"),
            ("学校の地下二階に行きます。", "去學校的地下二樓。"),
        ] {
            assert_eq!(
                translate(source, &TranslationGlossary::new()).target_text,
                expected
            );
        }
        let direction = translate("私は四階へ行きます。", &TranslationGlossary::new());
        assert_eq!(direction.target_text, "我往四樓去。");
        let fullwidth = translate("学校の１２階に行きます。", &TranslationGlossary::new());
        assert!(
            fullwidth.target_text.contains("１２樓"),
            "{}",
            fullwidth.target_text
        );
    }

    #[test]
    fn building_floor_uses_facility_subjects_and_spatial_roles() {
        for (source, expected) in [
            ("受付は二階です。", "受付在二樓。"),
            ("保健室は二階です。", "保健室在二樓。"),
            ("学校の図書館は二階です。", "學校的圖書館在二樓。"),
            ("私は三階から二階に行きます。", "我從三樓去二樓。"),
            ("私は三階で本を読みます。", "我在三樓讀書。"),
            ("私は三階で勉強します。", "我在三樓學習。"),
            ("学校の三階で準備します。", "在學校的三樓準備。"),
            ("私は三階で準備します。", "我在三樓準備。"),
        ] {
            assert_eq!(
                translate(source, &TranslationGlossary::new()).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn building_floor_keeps_walking_and_running_endpoints_spatial() {
        for (source, expected) in [
            ("私は三階から二階まで歩きます。", "我從三樓走到二樓。"),
            ("私は三階から歩きます。", "我從三樓走。"),
            ("私は三階まで走ります。", "我跑到三樓。"),
            ("私は三階まで歩きません。", "我不走到三樓。"),
            ("私は夕方まで走ります。", "我跑到傍晚。"),
        ] {
            assert_eq!(
                translate(source, &TranslationGlossary::new()).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn building_floor_uses_service_facilities_and_media_activities() {
        for (source, expected) in [
            ("病院の薬局は三階です。", "醫院的藥局在三樓。"),
            ("薬局は三階です。", "藥局在三樓。"),
            ("三階で写真を撮ります。", "在三樓拍攝照片。"),
            ("三階で撮影します。", "在三樓拍攝。"),
            ("三階で手紙を書きます。", "在三樓寫信。"),
        ] {
            assert_eq!(
                translate(source, &TranslationGlossary::new()).target_text,
                expected,
                "{source}"
            );
        }
        let source = "病院の窓口は三階です。";
        let output = translate(source, &TranslationGlossary::new()).target_text;
        assert!(output.contains("在三樓"), "{source}: {output}");
    }

    #[test]
    fn building_floor_uses_source_relations_for_ordinary_venues() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("三階で休みます。", "在三樓休息。"),
            ("病院の二階で式を書きます。", "在醫院的二樓寫式。"),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }

        let form = translate("四階で申込書に名前を書きました。", &glossary);
        assert!(form.target_text.contains("在四樓"), "{}", form.target_text);
        let meeting = translate("三階で二階堂さんに会います。", &glossary);
        assert!(
            meeting.target_text.contains("在三樓") && meeting.target_text.contains("二階堂"),
            "{}",
            meeting.target_text
        );
        for source in ["第三段階で休みます。", "第三階で休みます。"] {
            let text = translate(source, &glossary).target_text;
            assert!(
                text.contains('階') && !text.contains('樓'),
                "{source}: {text}"
            );
        }
    }

    #[test]
    fn building_floor_uses_linked_service_compound_without_relabeling_bare_counter() {
        let glossary = TranslationGlossary::new();
        for (source, floor) in [
            ("市民会館の案内所は四階です。", "在四樓"),
            ("病院の案内所は二階です。", "在二樓"),
        ] {
            let output = translate(source, &glossary).target_text;
            assert!(output.contains(floor), "{source}: {output}");
        }

        let bare = translate("市民会館の所は四階です。", &glossary).target_text;
        assert!(bare.contains("四階") && !bare.contains("四樓"), "{bare}");
        let count = translate("病院の入口は二か所です。", &glossary).target_text;
        assert!(count.contains('二') && !count.contains('樓'), "{count}");

        let source = "市民会館の`案内所`は四階です。";
        let locked = translate(source, &glossary).target_text;
        assert!(
            locked.matches("`案内所`").count() == 1
                && locked.contains("四階")
                && !locked.contains("四樓"),
            "{locked}"
        );
    }

    #[test]
    fn building_floor_leaves_compounds_names_and_locks_alone() {
        for (source, keep) in [
            ("この計画は三段階です。", "三段階"),
            ("二階微分を計算します。", "二階微分"),
            ("この微分は二階です。", "二階"),
            ("二階で微分します。", "二階"),
            ("二階で式を書きます。", "二階"),
            ("病院の数学は二階です。", "二階"),
            ("階段を上ります。", "樓梯"),
            ("階級を確認します。", "階級"),
            ("二階堂さんは来ました。", "二階堂"),
        ] {
            let result = translate(source, &TranslationGlossary::new());
            assert!(result.target_text.contains(keep), "{}", result.target_text);
        }
        let source = "私は`二階`に行きます。";
        let result = translate(source, &TranslationGlossary::new());
        let [record] = result.opaque_spans.as_slice() else {
            panic!("expected one protected span");
        };
        let target_span = record.target_span.expect("protected target span");
        let slice: String = result
            .target_text
            .chars()
            .skip(target_span.start_char)
            .take(target_span.end_char - target_span.start_char)
            .collect();
        assert_eq!(slice, "`二階`");
        assert_eq!(result.target_text.matches("`二階`").count(), 1);
        assert_eq!(
            record.source_span,
            Span {
                start_char: 2,
                end_char: 6
            }
        );
    }

    #[test]
    fn stative_te_iru_marks_the_verb_once_and_preserves_the_object() {
        for (source, target, expected) in [
            (
                "妹は傘を持っていますか。",
                TargetVariant::ZhHant,
                "妹妹拿著傘嗎？",
            ),
            (
                "妹は傘を持っていますか。",
                TargetVariant::ZhHans,
                "妹妹拿着伞吗？",
            ),
            ("ご飯が残っています。", TargetVariant::ZhHant, "飯留著。"),
            (
                "姉は`著著`を持っています。",
                TargetVariant::ZhHant,
                "姊姊拿著`著著`。",
            ),
            (
                "かばんを持っていません。",
                TargetVariant::ZhHant,
                "沒有拿著包包。",
            ),
        ] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(target),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(doc.target_text, expected, "{source}");
        }
    }

    #[test]
    fn te_help_request_preserves_action_object_and_politeness() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "この荷物を持っていただけますか。",
                "可以請您幫我拿著這件行李嗎？",
            ),
            (
                "この荷物を持っていただけませんか。",
                "可以麻煩您幫我拿著這件行李嗎？",
            ),
            ("写真を撮ってもらえますか。", "可以幫我拍攝照片嗎？"),
            (
                "この書類を確認していただけますか。",
                "可以請您幫我確認這份文件嗎？",
            ),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }

        let repeat = translate("もう一度、ゆっくり話していただけませんか。", &glossary);
        assert!(
            repeat.target_text.starts_with("可以麻煩您幫我再")
                && repeat.target_text.contains("慢慢")
                && repeat.target_text.ends_with("一次嗎？"),
            "{}",
            repeat.target_text
        );
    }

    #[test]
    fn te_help_request_binds_recipient_and_performer_separately() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "この本を先生に返していただけますか。",
                "可以請您把這本書還給老師嗎？",
            ),
            (
                "先生にこの本を返してもらえませんか。",
                "可以麻煩您把這本書還給老師嗎？",
            ),
            (
                "先生に地図を描いていただけますか。",
                "可以請老師幫我畫地圖嗎？",
            ),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }

        let ambiguous = translate("先生に本を読んでいただけますか。", &glossary).target_text;
        assert!(
            ambiguous.contains("老師")
                && ambiguous.contains("讀")
                && ambiguous.ends_with("嗎？")
                && !ambiguous.contains("關於"),
            "{ambiguous}"
        );
        let statement = translate("私は先生に写真を撮ってもらいました。", &glossary).target_text;
        assert!(
            statement.starts_with("老師幫我") && !statement.contains('嗎'),
            "{statement}"
        );
    }

    #[test]
    fn te_help_request_keeps_the_linked_recipient_of_a_delivery() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "友達に手紙を送ってもらえますか。",
                "可以請你把信寄給朋友嗎？",
            ),
            (
                "手紙を友達に送ってもらえますか。",
                "可以請你把信寄給朋友嗎？",
            ),
            (
                "先生にメールを送っていただけますか。",
                "可以請您把電子郵件寄給老師嗎？",
            ),
            (
                "友達にこの本を渡していただけますか。",
                "可以請您把這本書遞給朋友嗎？",
            ),
            ("友達に手紙を書いてもらえますか。", "給朋友寫信，もらえ嗎？"),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }

        for source in [
            "先生に子供を送っていただけますか。",
            "先生に幼児を送っていただけますか。",
        ] {
            let escort = translate(source, &glossary).target_text;
            assert!(
                escort.contains("老師") && !escort.contains("寄給") && !escort.contains("關於"),
                "{source}: {escort}"
            );
        }
        let source = "手紙を送ってもらえますか。";
        let implicit = translate(source, &glossary).target_text;
        assert!(
            implicit.contains("幫我")
                && implicit.contains('寄')
                && implicit.contains('信')
                && !implicit.contains("寄給"),
            "{source}: {implicit}"
        );
    }

    #[test]
    fn te_help_letter_request_uses_one_reading_for_an_unresolved_participant() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "先生に二人で手紙を書いていただけますか。",
                "可以請兩位一起寫信給老師嗎？",
            ),
            (
                "二人で先生に手紙を書いていただけますか。",
                "可以請兩位一起寫信給老師嗎？",
            ),
            (
                "先生に三人で手紙を書いていただけませんか。",
                "可以麻煩三位一起寫信給老師嗎？",
            ),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn te_help_letter_request_keeps_explicit_writers_and_addressees() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "先生宛ての手紙を二人で書いていただけますか。",
                "可以請兩位一起寫信給老師嗎？",
            ),
            (
                "先生宛てに二人で手紙を書いていただけますか。",
                "可以請兩位一起寫信給老師嗎？",
            ),
            (
                "先生方お二人に、父宛ての手紙を書いていただけますか。",
                "可以請兩位老師寫信給父親嗎？",
            ),
            (
                "父宛の手紙を先生方お三人に書いていただけませんか。",
                "可以麻煩三位老師寫信給父親嗎？",
            ),
            (
                "父に先生宛ての手紙を書いていただけますか。",
                "可以請父親寫信給老師嗎？",
            ),
            (
                "先生に二人で父宛ての手紙を書いていただけますか。",
                "可以請包括老師在內的兩位一起寫信給父親嗎？",
            ),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn te_help_letter_request_keeps_conjoined_recipients_and_counts() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "父や母宛ての手紙を書いていただけますか。",
                "可以請您寫信給父親、母親等人嗎？",
            ),
            (
                "父と母の二人宛ての手紙を書いていただけますか。",
                "可以請您寫信給父親和母親兩人嗎？",
            ),
            (
                "先生方お二人に、父と母宛ての手紙を書いていただけますか。",
                "可以請兩位老師寫信給父親和母親嗎？",
            ),
            (
                "先生方お二人に、校長宛ての手紙を書いていただけますか。",
                "可以請兩位老師寫信給校長嗎？",
            ),
            (
                "先生方お二人に、明日、父宛ての手紙を書いていただけますか。",
                "可以請兩位老師明天寫信給父親嗎？",
            ),
            (
                "先生宛ての手紙を明日お二人で書いていただけますか。",
                "可以請兩位明天一起寫信給老師嗎？",
            ),
            (
                "明日、母と二人で先生宛ての手紙を書いていただけますか。",
                "可以請您和母親兩人明天一起寫信給老師嗎？",
            ),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn te_help_letter_rule_preserves_statements_and_unspecified_roles() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("先生に手紙を書きました。", "給老師寫了信。"),
            (
                "先生に二人で手紙を書きませんか。",
                "要不要兩人一起給老師寫信？",
            ),
            ("先生に二人で手紙を書けますか。", "在二人給老師能寫信嗎？"),
            ("手紙を書いていただけますか。", "可以請您幫我寫信嗎？"),
            (
                "二人で手紙を書いていただけますか。",
                "可以請兩人一起寫信嗎？",
            ),
            (
                "父と母宛ての手紙を書いていただけますか。",
                "父親和母親宛，寫信，能得到嗎？",
            ),
            (
                "姉と母宛ての手紙を二人で書いていただけますか。",
                "姊姊和母親宛，在二人寫信，能得到嗎？",
            ),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn te_help_request_keeps_joint_manner_and_deadlines_on_the_action() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "一緒にこの荷物を持っていただけませんか。",
                "可以麻煩您幫我一起拿著這件行李嗎？",
            ),
            (
                "この荷物を一緒に持っていただけませんか。",
                "可以麻煩您幫我一起拿著這件行李嗎？",
            ),
            (
                "二人で友達に手紙を送っていただけますか。",
                "可以請兩人一起把信寄給朋友嗎？",
            ),
            (
                "五時までに友達に手紙を送っていただけますか。",
                "可以請您最晚在五點把信寄給朋友嗎？",
            ),
            (
                "友達に手紙を五時までに送っていただけますか。",
                "可以請您最晚在五點把信寄給朋友嗎？",
            ),
            (
                "二人で五時までに友達に手紙を送っていただけますか。",
                "可以請兩人最晚在五點一起把信寄給朋友嗎？",
            ),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn te_help_request_keeps_the_named_performer_in_the_group() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "先生に二人で地図を描いていただけますか。",
                "可以請包括老師在內的兩人一起幫我畫地圖嗎？",
            ),
            (
                "二人で先生に地図を描いていただけますか。",
                "可以請包括老師在內的兩人一起幫我畫地圖嗎？",
            ),
            (
                "先生に三人で地図を描いていただけませんか。",
                "可以麻煩包括老師在內的三人一起幫我畫地圖嗎？",
            ),
            (
                "友達に二人で絵を描いてもらえますか。",
                "可以請包括朋友在內的兩人一起幫我畫畫嗎？",
            ),
            (
                "先生に地図を描いていただけますか。",
                "可以請老師幫我畫地圖嗎？",
            ),
            (
                "二人で地図を描いていただけますか。",
                "可以請兩人一起畫地圖嗎？",
            ),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn te_help_request_draws_the_linked_object_once() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("二人で絵を描いてもらえますか。", "可以請兩人一起畫畫嗎？"),
            ("絵を描いてもらえますか。", "可以幫我畫畫嗎？"),
            ("子供の絵を描いてもらえますか。", "可以幫我畫孩子的畫嗎？"),
            (
                "もう一度絵を描いていただけませんか。",
                "可以麻煩您幫我再畫畫一次嗎？",
            ),
            (
                "二人で`原圖`を描いてもらえますか。",
                "可以請兩人一起畫`原圖`嗎？",
            ),
            ("描いてもらえますか。", "可以幫我畫畫嗎？"),
            ("二人で描いてもらえますか。", "可以請兩人一起畫畫嗎？"),
            ("絵を撮ってもらえますか。", "可以幫我拍攝畫嗎？"),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn te_help_request_preserves_other_cases_and_the_invitation() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("一緒にこの荷物を持ちませんか。", "要不要一起拿著這荷物？"),
            (
                "公園で写真を撮ってもらえますか。",
                "可以幫我在公園拍攝照片嗎？",
            ),
            (
                "`一緒`にこの荷物を持っていただけませんか。",
                "給`一緒`拿著這荷物，不能得到嗎？",
            ),
            (
                "五時に友達に手紙を送っていただけますか。",
                "在五點給朋友寄送信，能得到嗎？",
            ),
            (
                "私たちと一緒に本を読んでいただけますか。",
                "給我們和一起讀書，能得到嗎？",
            ),
            (
                "一緒に友達とこの荷物を持っていただけませんか。",
                "跟朋友給一起拿著這荷物，不能得到嗎？",
            ),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn te_help_request_keeps_linked_child_possessive_inside_the_object() {
        let glossary = TranslationGlossary::new();
        for (source, owner) in [
            ("子どもの写真を撮ってもらえますか。", "孩子"),
            ("犬の写真を撮ってもらえますか。", "狗"),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                format!("可以幫我拍攝{owner}的照片嗎？"),
                "{source}"
            );
        }
    }

    #[test]
    fn te_help_request_leaves_other_roles_and_protected_text_intact() {
        let glossary = TranslationGlossary::new();
        for source in [
            "この資料をいただけませんか。",
            "田中さんご本人に、母のためにこの書類を確認していただけませんか。",
            "私が内容を確認してから、窓口で控えを一部いただけますか。",
            "写真を撮って送ってもらえますか。",
        ] {
            let output = translate(source, &glossary).target_text;
            assert!(!output.contains("幫我"), "{source}: {output}");
        }
        for (source, ability) in [("泳げますか。", "能游泳"), ("泳げませんか。", "不能游泳")]
        {
            let output = translate(source, &glossary).target_text;
            assert!(
                output.contains(ability) && !output.contains("幫我"),
                "{source}: {output}"
            );
        }

        let locked = translate("この`資料A`を確認していただけませんか。", &glossary);
        assert!(locked.target_text.starts_with("可以麻煩您幫我確認"));
        assert!(locked.target_text.ends_with("嗎？"));
        assert_eq!(locked.target_text.matches("`資料A`").count(), 1);
        let [record] = locked.opaque_spans.as_slice() else {
            panic!("expected one protected object");
        };
        let span = record.target_span.expect("restored object");
        let slice: String = locked
            .target_text
            .chars()
            .take(span.end_char)
            .skip(span.start_char)
            .collect();
        assert_eq!(slice, "`資料A`");
    }

    #[test]
    fn nominal_conditions_keep_polarity_in_their_own_clause() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (
                "彼が学生であれば、この本を読みますか。",
                "如果他是學生，讀這本書嗎？",
            ),
            (
                "彼が学生であればこの本を読みませんか。",
                "如果他是學生，不讀這本書嗎？",
            ),
            (
                "彼が学生でなければ、この本を読みますか。",
                "如果他不是學生，讀這本書嗎？",
            ),
            (
                "彼が学生でなければこの本を読みませんか。",
                "如果他不是學生，不讀這本書嗎？",
            ),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
        let qualified = translate("彼がこの学校の先生であれば、説明します。", &glossary);
        assert_eq!(qualified.target_text, "如果他是這學校的老師，說明。");
        let omitted = translate("医師であれば、説明します。", &glossary);
        assert_eq!(omitted.target_text, "如果是醫師，說明。");
        let owner = translate("車の所有者が会員であれば、登録します。", &glossary);
        assert_eq!(owner.target_text, "如果車的所有者是會員，註冊。");
    }

    #[test]
    fn nominal_conditions_leave_locative_obligation_and_proportion_alone() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("学校で本を読めば、分かります。", "如果在學校讀書，明白。"),
            ("便利になればなるほど、使います。", "越方便，使用。"),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
        let obligation = translate("学生でなければならない。", &glossary);
        assert!(obligation.target_text.contains("必須"));
        assert!(!obligation.target_text.contains("如果"));
    }

    #[test]
    fn nominal_conditions_preserve_ambiguous_instrumental_de() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("私は車であれば通勤できます。", "如果我用車那個，通勤能夠。"),
            (
                "私は車でなければ通勤できません。",
                "如果我用車，通勤不能夠。",
            ),
            (
                "私が自転車であれば、通勤できます。",
                "如果我在自轉車那個，通勤能夠。",
            ),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn demonstrative_nouns_keep_attachment_case_and_protected_text() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("私はこの雑誌を読みます。", "我讀這本雜誌。"),
            ("私はその本を読みません。", "我不讀那本書。"),
            ("私はあの本を読みます。", "我讀那本書。"),
            ("私はこの本の名前を読みます。", "我讀這本書的名字。"),
            ("私は新しい本を読みます。", "我讀新書。"),
            ("私は甘いケーキを食べます。", "我吃甜的蛋糕。"),
        ] {
            let doc = translate(source, &glossary);
            assert_eq!(doc.target_text, expected, "{source}");
        }
        let source = "私はその本を読む、彼は`この車`を見る。";
        let doc = translate(source, &glossary);
        assert!(doc.target_text.contains("那本書"));
        let [lock] = doc.opaque_spans.as_slice() else {
            panic!("{doc:?}")
        };
        let slice = |s: &str, span: Span| {
            s.chars()
                .skip(span.start_char)
                .take(span.end_char - span.start_char)
                .collect::<String>()
        };
        assert_eq!(slice(source, lock.source_span), "`この車`");
        assert_eq!(
            slice(&doc.target_text, lock.target_span.unwrap()),
            "`この車`"
        );
        assert_eq!(doc.target_text.matches("`この車`").count(), 1);

        let glossary = TranslationGlossary {
            objects: vec![GlossaryEntry {
                canonical: "その本".into(),
                surfaces: vec!["その本".into()],
                targets: BTreeMap::from([(TargetVariant::ZhHant, "藏書甲".into())]),
                ..GlossaryEntry::default()
            }],
            ..TranslationGlossary::new()
        };
        let source = "私はその本を読みます。";
        let doc = translate(source, &glossary);
        let [hit] = doc.glossary_hits.as_slice() else {
            panic!("{doc:?}")
        };
        assert_eq!(slice(source, hit.source_span), "その本");
        assert_eq!(slice(&doc.target_text, hit.target_span), "藏書甲");
        assert_eq!(doc.target_text.matches("藏書甲").count(), 1);
    }

    #[test]
    fn demonstrative_classifiers_preserve_people_and_mass_nouns() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("その先生は来ます。", "那個老師來。"),
            ("その先生が来ます。", "那個老師來。"),
            ("あの先生は忙しいです。", "那個老師忙碌。"),
            ("私はこの先生を見ます。", "我看這個老師。"),
            ("その医師は来ます。", "那個醫師來。"),
            ("私はその友人を見ます。", "我看那個朋友。"),
            ("私はその後輩を見ます。", "我看那個後輩。"),
            ("私はこの水を飲みます。", "我喝這水。"),
            ("私はその砂糖を買います。", "我買那砂糖。"),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn pointing_at_medicine_does_not_supply_a_kind_or_dosage() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("その薬は床に落ちました。", "那個藥掉落到地板了。"),
            ("その薬を拾います。", "拾起那個藥。"),
            ("私はこの薬を飲みます。", "我喝這個藥。"),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
        let explicit = translate("その薬の種類を調べます。", &glossary);
        assert!(
            explicit.target_text.contains("那個藥的種類"),
            "{explicit:?}"
        );
    }

    #[test]
    fn demonstrative_requires_a_real_adjacent_noun_dependency() {
        let mut det = atom("その", None, Upos::Det, 0);
        let mut noun = atom("本", None, Upos::Noun, 2);
        det.doc_tok_idx = Some(0);
        noun.doc_tok_idx = Some(1);
        det.deprel = "det".into();
        det.head = Some(1);
        assert_eq!(attached_demonstrative(&det, &noun), Some('那'));
        let mut locked = noun.clone();
        locked.is_locked = true;
        assert_eq!(attached_demonstrative(&det, &locked), None);
        let mut detached = noun.clone();
        detached.span.start_char += 1;
        assert_eq!(attached_demonstrative(&det, &detached), None);
        let mut pronoun = det.clone();
        pronoun.upos = Some(Upos::Pron);
        assert_eq!(attached_demonstrative(&pronoun, &noun), None);
        det.head = Some(2);
        assert_eq!(attached_demonstrative(&det, &noun), None);
    }

    #[test]
    fn relative_daypart_is_bounded_by_source_and_nominal_context() {
        fn group(atoms: Vec<Atom>, role: BunsetsuRole, case: Option<&str>) -> Bunsetsu {
            Bunsetsu {
                atoms,
                role,
                case_particle: case.map(str::to_owned),
                sconj_tail: None,
                sfp_tail: None,
            }
        }
        let date = group(
            vec![
                atom("明日", None, Upos::Noun, 0),
                atom("の", None, Upos::Adp, 2),
            ],
            BunsetsuRole::Other,
            Some("の"),
        );
        let afternoon = group(
            vec![atom("午後", None, Upos::Noun, 3)],
            BunsetsuRole::Adv,
            None,
        );
        let morning = group(
            vec![atom("朝", None, Upos::Noun, 3)],
            BunsetsuRole::Adv,
            None,
        );
        let composed = compose_relative_daypart(vec![date.clone(), afternoon.clone()]);
        assert_eq!(composed.len(), 1);
        assert_eq!(
            composed[0].atoms[0].lemma.as_deref(),
            Some(RELATIVE_DAYPART_ATOM)
        );
        assert_eq!(
            composed[0].atoms[0].surface,
            format!(
                "{}{}",
                translate_single_ja_term("明日", None, Some(Upos::Noun)),
                translate_single_ja_term("午後", None, Some(Upos::Noun))
            )
        );
        assert_eq!(
            compose_relative_daypart(vec![date.clone(), morning]).len(),
            1
        );
        let noun = group(
            vec![atom("会議", None, Upos::Noun, 5)],
            BunsetsuRole::Other,
            None,
        );
        assert_eq!(
            compose_relative_daypart(vec![date.clone(), afternoon.clone(), noun]).len(),
            3
        );
        let mut locked = afternoon.clone();
        locked.atoms[0].is_locked = true;
        assert_eq!(
            compose_relative_daypart(vec![date.clone(), locked]).len(),
            2
        );
        let clock = group(
            vec![atom("午後二時", Some(CLOCK_ATOM), Upos::Noun, 3)],
            BunsetsuRole::Adv,
            None,
        );
        assert_eq!(compose_relative_daypart(vec![date.clone(), clock]).len(), 2);
        let possessive = group(
            vec![
                atom("私", None, Upos::Pron, 0),
                atom("の", None, Upos::Adp, 1),
            ],
            BunsetsuRole::Other,
            Some("の"),
        );
        assert_eq!(
            compose_relative_daypart(vec![possessive, afternoon.clone()]).len(),
            2
        );
        assert_eq!(compose_relative_daypart(vec![afternoon]).len(), 1);
    }

    #[test]
    fn relative_daypart_stays_ordered_in_adopted_event() {
        for (source, expected) in [
            ("明日の朝、私は家で本を読みます。", "明天早上"),
            ("明日の午後、私は家で本を読みます。", "明天下午"),
        ] {
            let result = translate(source, &TranslationGlossary::new());
            assert!(
                result.target_text.contains(expected),
                "{}",
                result.target_text
            );
            assert!(
                !result.target_text.contains("明天的"),
                "{}",
                result.target_text
            );
        }
        let text = translate(
            "明日の朝私は公園で本を読みます。",
            &TranslationGlossary::new(),
        )
        .target_text;
        let time = text.find("明天早上").expect("combined time");
        let place = text.find("公園").expect("place");
        assert!(time < place, "{text}");
    }

    fn predicate(atoms: Vec<Atom>) -> Bunsetsu {
        Bunsetsu {
            atoms,
            role: BunsetsuRole::Predicate,
            case_particle: None,
            sconj_tail: None,
            sfp_tail: None,
        }
    }

    #[test]
    fn animate_nonpast_accepts_adopted_polite_forms_and_rejects_past_and_action() {
        let base = atom("い", Some("いる"), Upos::Verb, 0);
        let positive = predicate(vec![base.clone(), atom("ます", None, Upos::Aux, 1)]);
        let negative = predicate(vec![
            base.clone(),
            atom("ませ", None, Upos::Aux, 1),
            atom("ん", None, Upos::Aux, 3),
        ]);
        assert_eq!(standalone_iru_nonpast(&[positive]), Some(false));
        assert_eq!(standalone_iru_nonpast(&[negative]), Some(true));
        let past = predicate(vec![base.clone(), atom("ました", None, Upos::Aux, 1)]);
        assert_eq!(standalone_iru_nonpast(&[past]), None);
        let unknown = predicate(vec![
            base.clone(),
            atom("ます", None, Upos::Aux, 1),
            atom("だけ", None, Upos::Part, 3),
        ]);
        assert_eq!(standalone_iru_nonpast(&[unknown]), None);
        let mut contrast = predicate(vec![
            base.clone(),
            atom("ます", None, Upos::Aux, 1),
            atom("が", None, Upos::Adp, 3),
        ]);
        assert_eq!(standalone_iru_nonpast(&[contrast.clone()]), None);
        contrast.sconj_tail = Some("が".to_owned());
        assert_eq!(standalone_iru_nonpast(&[contrast]), Some(false));
        let action = predicate(vec![
            atom("走っ", Some("走る"), Upos::Verb, 0),
            atom("て", None, Upos::Sconj, 2),
            atom("い", Some("いる"), Upos::Verb, 3),
            atom("ます", None, Upos::Aux, 4),
        ]);
        assert_eq!(standalone_iru_nonpast(&[action]), None);
    }

    #[test]
    fn location_requires_particle_and_predicate_source_link() {
        let mut place = atom("庭", None, Upos::Noun, 0);
        place.doc_tok_idx = Some(0);
        place.head = Some(2);
        place.deprel = "obl".to_owned();
        let mut particle = atom("に", None, Upos::Adp, 1);
        particle.head = Some(0);
        let b = Bunsetsu {
            atoms: vec![place, particle],
            role: BunsetsuRole::IObj,
            case_particle: Some("に".to_owned()),
            sconj_tail: None,
            sfp_tail: None,
        };
        assert!(existential_location_link(&b, 2));
        assert!(!existential_location_link(&b, 3));
        let mut clock = b.clone();
        // normalize_clock_atoms emits CLOCK_ATOM for a recognized clock.
        clock.atoms[0] = atom("二時", Some(CLOCK_ATOM), Upos::Noun, 0);
        clock.atoms[0].doc_tok_idx = Some(0);
        clock.atoms[0].head = Some(2);
        clock.atoms[0].deprel = "obl".to_owned();
        assert!(!existential_location_link(&clock, 2));
    }

    #[test]
    fn fused_location_requires_unambiguous_relation_and_source_attachment() {
        // The token structure follows the adopted jz155 source observation.
        let mut owner = atom("財布", None, Upos::Noun, 0);
        owner.doc_tok_idx = Some(0);
        let mut genitive = atom("の", None, Upos::Adp, 2);
        genitive.head = Some(0);
        let mut spatial = atom("中に", None, Upos::Adv, 3);
        spatial.head = Some(5);
        let owner_b = Bunsetsu {
            atoms: vec![owner, genitive],
            role: BunsetsuRole::Other,
            case_particle: Some("の".to_owned()),
            sconj_tail: None,
            sfp_tail: None,
        };
        let spatial_b = Bunsetsu {
            atoms: vec![spatial],
            role: BunsetsuRole::Adv,
            case_particle: None,
            sconj_tail: None,
            sfp_tail: None,
        };
        let mut pair = vec![owner_b, spatial_b];
        assert!(fused_relational_location(&pair, Some(5)));
        assert!(!fused_relational_location(&pair, Some(6)));
        for ambiguous in ["前に", "後に"] {
            pair[1].atoms[0].surface = ambiguous.to_owned();
            assert!(!fused_relational_location(&pair, Some(5)));
        }
    }

    #[test]
    fn adopted_existentials_keep_polarity_and_action_guards() {
        let glossary = TranslationGlossary::new();
        for (source, negative) in [("庭に猫がいます。", false), ("庭に猫がいません。", true)]
        {
            let text = translate(source, &glossary).target_text;
            assert!(text.contains('貓') && text.contains('庭'), "{text}");
            assert!(
                if negative {
                    text.contains("沒有")
                } else {
                    text.contains('有')
                },
                "{text}"
            );
            assert!(!text.contains('給'), "{text}");
        }
        let action = translate("棚の上に箱を置いた。", &glossary).target_text;
        assert!(
            action.contains('箱') && action.contains('棚') && action.contains('放'),
            "{action}"
        );
        let recipient = translate("私は`QX`に手紙を書きます。", &glossary).target_text;
        assert!(
            recipient.contains("`QX`") && recipient.contains('給') && recipient.contains('信'),
            "{recipient}"
        );
    }

    #[test]
    fn adopted_past_relations_keep_owner_entity_and_time() {
        let glossary = TranslationGlossary::new();
        for (source, owner, entity) in [
            ("財布の中にお金がなかった。", "錢包", "金錢"),
            ("棚の上に箱がなかった。", "棚", "箱"),
        ] {
            let text = translate(source, &glossary).target_text;
            assert!(
                text.contains(owner)
                    && text.contains(entity)
                    && text.contains("沒有")
                    && text.contains("當時"),
                "{text}"
            );
        }
    }

    #[test]
    fn adopted_glossary_and_opaque_content_stay_in_roles() {
        let mut glossary = TranslationGlossary::new();
        glossary.places.push(GlossaryEntry {
            canonical: "青葉公園".to_owned(),
            surfaces: vec!["青葉公園".to_owned()],
            targets: BTreeMap::from([(TargetVariant::ZhHant, "青葉公園".to_owned())]),
            ..GlossaryEntry::default()
        });
        let doc = translate("青葉公園に犬がいます。", &glossary);
        assert_eq!(doc.glossary_hits.len(), 1);
        assert!(
            doc.target_text.contains("青葉公園")
                && doc.target_text.contains('狗')
                && doc.target_text.contains('有')
                && !doc.target_text.contains('給'),
            "{}",
            doc.target_text
        );
        let opaque = translate("庭に`QX`がいません。", &TranslationGlossary::new());
        assert_eq!(opaque.opaque_spans.len(), 1);
        assert!(
            opaque.target_text.contains("`QX`")
                && opaque.target_text.contains("沒有")
                && !opaque.target_text.contains('給'),
            "{}",
            opaque.target_text
        );
        let target_span = opaque.opaque_spans[0].target_span.unwrap();
        let protected: String = opaque
            .target_text
            .chars()
            .skip(target_span.start_char)
            .take(target_span.end_char - target_span.start_char)
            .collect();
        assert_eq!(protected, "`QX`");
        let mut actor = TranslationGlossary::new();
        actor.actors.push(GlossaryEntry {
            canonical: "青木さん".to_owned(),
            surfaces: vec!["青木さん".to_owned()],
            targets: BTreeMap::from([(TargetVariant::ZhHant, "青木".to_owned())]),
            ..GlossaryEntry::default()
        });
        let recipient = translate("私は青木さんに手紙を書きます。", &actor);
        assert_eq!(recipient.glossary_hits.len(), 1);
        assert!(
            recipient.target_text.contains("給青木") && recipient.target_text.contains('信'),
            "{}",
            recipient.target_text
        );
    }

    #[test]
    fn adopted_opposed_clauses_keep_polarity_local() {
        let glossary = TranslationGlossary::new();
        for (source, first_negative) in [
            ("庭に猫がいますが、犬は道を走りません。", false),
            ("庭に猫がいませんが、犬は道を走ります。", true),
        ] {
            let text = translate(source, &glossary).target_text;
            let clauses: Vec<&str> = text.split('，').collect();
            assert_eq!(clauses.len(), 2, "{text}");
            assert!(
                clauses[0].contains("雖然") || clauses[1].contains('但'),
                "{text}"
            );
            assert!(
                clauses[0].contains('貓') && clauses[0].contains('庭'),
                "{text}"
            );
            assert_eq!(clauses[0].contains("沒有"), first_negative, "{text}");
            assert!(!clauses[0].contains('給'), "{text}");
            if first_negative {
                assert!(clauses[0].contains("沒有"), "{text}");
            } else {
                assert!(
                    clauses[0].contains('有') || clauses[0].contains('在'),
                    "{text}"
                );
            }
            assert!(
                clauses[1].contains('狗')
                    && clauses[1].contains('跑')
                    && clauses[1].contains("道路"),
                "{text}"
            );
            assert_eq!(clauses[1].contains('不'), !first_negative, "{text}");
        }
    }
}

#[cfg(test)]
mod shinjitai_fallback_tests {
    use super::*;

    #[test]
    fn unglossed_han_terms_use_shared_glyph_forms() {
        assert_eq!(
            translate_single_ja_term("製糸", None, Some(Upos::Noun)),
            "製絲"
        );
        assert_eq!(
            translate_single_ja_term("𠀀糸", None, Some(Upos::Noun)),
            "𠀀絲"
        );
    }

    #[test]
    fn glyph_fallback_keeps_word_glosses_and_non_han_surfaces() {
        assert_eq!(
            translate_single_ja_term("駅", None, Some(Upos::Noun)),
            "車站"
        );
        assert_eq!(
            translate_single_ja_term("未登録かな", Some("読む"), Some(Upos::Verb)),
            "讀"
        );
        for surface in ["糸かな", "糸A", "糸한", "𠀀", ""] {
            assert_eq!(translate_single_ja_term(surface, None, None), surface);
        }
    }
}
