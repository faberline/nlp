//! Multi-layer deterministic `ja -> zh-Hant / zh-Hans` translation engine.
//!
//! Consumes all `nlp.doc.v1` layers (`segmentation`, `tokens`, `entities`, `syntax`, `frames`,
//! `coref` / `zero_mentions`, `relations`) plus `SentenceConditioning` and `SpanLockTable` to
//! reorder Japanese SOV clauses into Chinese SVO, synthesize predicates, resolve pro-drop, and
//! render uncensored, idiomatic `zh-Hant` or `zh-Hans` prose and dialogue.

use std::collections::HashMap;

use faberline_nlp::Span;
use faberline_nlp::doc::{Document, FrameRole, Token, Upos};
use faberline_nlp::lexicon::SemanticClass;

use crate::conditioning::zh_pronoun_for_hint;
use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::{ClauseAlignment, SentenceConditioning, TargetVariant};
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
enum BunsetsuRole {
    Connector,
    Subj,
    Obl,
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

/// Translate a single Japanese (or Ja-dominant Mixed) sentence into `target` (`zh-Hant` / `zh-Hans`),
/// returning the masked target string (with `[[PHk]]` placeholders intact for `SpanLockTable::restore`)
/// and `ClauseAlignment` entries relative to sentence start (0-based within the sentence).
pub(crate) fn translate_ja_sentence(
    sentence_idx: usize,
    chars: &[char],
    doc: &Document,
    sent_tokens: &[&Token],
    lock_table: &SpanLockTable,
    conditioning: &SentenceConditioning,
    target: TargetVariant,
) -> (String, Vec<ClauseAlignment>) {
    let sent_span = doc.segmentation.sentences[sentence_idx].span;
    let atoms = build_sentence_atoms(sentence_idx, chars, doc, sent_tokens, lock_table);
    let atoms = normalize_keigo_atoms(&atoms);
    let atoms = normalize_glossary_clocks(&atoms, chars, lock_table);
    let atoms = normalize_clock_atoms(&atoms);
    let atoms = preserve_clock_adjacent_locks(&atoms, doc, sentence_idx);

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

    // Split atoms into clauses at `、` / `，` / quoted spans / SCONJ tails.
    let segments = split_atoms_into_segments(&atoms, chars, sent_span);

    let mut out = String::new();
    let mut seg_target_spans: Vec<(Span, Span)> = Vec::with_capacity(segments.len());
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

    for (seg_idx, seg) in segments.iter().enumerate() {
        if skip_next_seg {
            skip_next_seg = false;
            continue;
        }
        let tgt_start = out.chars().count();
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
                let inner_segs = split_atoms_into_segments(inner_atoms, chars, *span);
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
                            &ClauseDiscourseCtx::default(),
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
                out.push(close_zh);
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
                out.push(localize_open_quote(*open_q, target));
                let inner_segs = split_atoms_into_segments(inner_atoms, chars, *span);
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
                            &ClauseDiscourseCtx::default(),
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
                out.push(localize_close_quote(*close_q, target));
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
                let (mut rendered, had_subj_only) = if editorial_shikashi {
                    ("然而".to_owned(), false)
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
    for (si, sent) in doc.segmentation.sentences.iter().enumerate() {
        if sent.span.start_char > cursor {
            for &c in &chars[cursor..sent.span.start_char] {
                out.push(c);
            }
        }
        let sent_tokens: Vec<&Token> = doc.tokens.iter().filter(|t| t.sentence == si).collect();
        let lock_table = SpanLockTable::build(Vec::new());
        let cond = SentenceConditioning::default();
        let (tr, _) =
            translate_ja_sentence(si, &chars, &doc, &sent_tokens, &lock_table, &cond, target);
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

fn clock_number(text: &str) -> Option<u32> {
    if text.is_empty() {
        return None;
    }
    let normalized: String = text
        .chars()
        .map(|ch| match ch {
            '０'..='９' => char::from_u32(ch as u32 - '０' as u32 + '0' as u32).unwrap_or(ch),
            _ => ch,
        })
        .collect();
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
                atoms.push(Atom {
                    span: lk.source_span,
                    surface: lk.placeholder.clone(),
                    lemma: None,
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

fn split_atoms_into_segments(atoms: &[Atom], _chars: &[char], parent_span: Span) -> Vec<Segment> {
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

    while i < atoms.len() {
        let a = &atoms[i];
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
            flush_clause(&mut current, Some(a.surface.clone()), &mut segments);
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

#[derive(Clone, Default)]
struct ClauseDiscourseCtx {
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

    // Multi-token phrase & collocation replacement over atoms first, then normalize remaining atoms.
    let merged_atoms = apply_phrase_matches(atoms);
    let norm_atoms = normalize_building_floors(normalize_ja_atoms(&merged_atoms), doc);

    // Group into bunsetsus, then fold adnominal modifiers (`の`, prenominal `Adj`, `acl`).
    let grouped = group_into_bunsetsus(&norm_atoms, doc, sentence_idx);
    let grouped = compose_relative_daypart(grouped);
    let bunsetsus = merge_adnominal_bunsetsus(grouped, doc, sentence_idx, target);

    translate_clause_bunsetsus(
        bunsetsus,
        &raw_clause,
        conditioning,
        allow_pro_drop_inject,
        ctx,
        target,
    )
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

fn is_floor_number(surface: &str) -> bool {
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
    // The lexicon does not tag all facilities. Room, hall and building
    // compounds give place evidence even without a semantic tag.
    atom.doc_tok_idx
        .and_then(|index| doc.tokens.get(index))
        .is_some_and(|token| token.sem_class == Some(SemanticClass::Location))
        || is_place_location_bunsetsu(std::slice::from_ref(atom))
        || matches!(
            source,
            "事務所" | "受付" | "食堂" | "台所" | "トイレ" | "玄関" | "売り場" | "店舗"
        )
        || (source.ends_with(['室', '館', '棟']) && !matches!(source, "王室" | "皇室"))
}

// Identify the complete floor label in the source before changing how it is
// rendered. Keep the head of 階 and the span of the entire numeral and suffix.
fn building_floor_start(atoms: &[Atom], end: usize, doc: &Document) -> Option<usize> {
    let floor = &atoms[end];
    if floor.is_locked || floor.upos != Some(Upos::Noun) {
        return None;
    }
    let floor_idx = floor.doc_tok_idx?;
    let prefix = floor.surface.strip_suffix('階')?;
    let mut start = end;
    if prefix.is_empty() {
        while let Some(number) = start.checked_sub(1).and_then(|i| atoms.get(i))
            && !number.is_locked
            && matches!(number.upos, Some(Upos::Num | Upos::Noun))
            && is_floor_number(&number.surface)
            && number.span.end_char == atoms[start].span.start_char
        {
            start -= 1;
        }
        if start == end {
            return None;
        }
    } else if !is_floor_number(prefix.strip_prefix("地下").unwrap_or(prefix)) {
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
                    && is_floor_facility(subject, doc)
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
            spatial_case
                && (matches!(
                    verb,
                    "待つ"
                        | "働く"
                        | "食べる"
                        | "寝る"
                        | "読む"
                        | "会う"
                        | "学ぶ"
                        | "勉強"
                        | "勉強する"
                        | "学習"
                        | "学習する"
                        | "準備"
                        | "準備する"
                ) || atoms[..start].windows(2).any(|modifier| {
                    let place = &modifier[0];
                    let particle = &modifier[1];
                    place.head == Some(floor_idx)
                        && place.deprel == "nmod"
                        && !particle.is_locked
                        && particle.surface == "の"
                        && particle.head.is_some()
                        && particle.head == place.doc_tok_idx
                        && is_floor_facility(place, doc)
                }))
        }
        _ => false,
    };
    is_spatial.then_some(start)
}

fn normalize_building_floors(mut atoms: Vec<Atom>, doc: &Document) -> Vec<Atom> {
    let mut end = atoms.len();
    while end > 0 {
        end -= 1;
        let Some(start) = building_floor_start(&atoms, end, doc) else {
            continue;
        };
        let mut floor = atoms[end].clone();
        floor.span.start_char = atoms[start].span.start_char;
        floor.surface = atoms[start..=end]
            .iter()
            .map(|a| a.surface.as_str())
            .collect();
        floor.lemma = Some(BUILDING_FLOOR.to_owned());
        atoms.splice(start..=end, [floor]);
        end = start;
    }
    atoms
}

fn translate_clause_bunsetsus(
    mut bunsetsus: Vec<Bunsetsu>,
    raw_clause: &str,
    conditioning: Option<&SentenceConditioning>,
    allow_pro_drop_inject: bool,
    ctx: &ClauseDiscourseCtx,
    target: TargetVariant,
) -> (String, bool) {
    // Track extra bunsetsu-level metadata for `に`/`で` localization.
    #[derive(Clone)]
    struct PlacedArg {
        text: String,
        case_particle: Option<String>,
        is_instrument_de: bool,
        is_spatial_made: bool,
        is_clock: bool,
        is_building_floor: bool,
        transfer_dative: bool,
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
    let mut adverbs: Vec<String> = Vec::new();
    let mut iobjs: Vec<PlacedArg> = Vec::new();
    let mut objs: Vec<String> = Vec::new();
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

    for b in bunsetsus.drain(..) {
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
                "か" | "かな" => Some("嗎"),
                "よ" => Some("喔"),
                "ね" | "よね" => Some("呢"),
                "わ" => Some("呀"),
                "ぞ" | "ぜ" => Some("啊"),
                _ => sfp_suffix,
            };
        }

        match b.role {
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
            BunsetsuRole::Obl => {
                let is_inst = b.case_particle.as_deref() == Some("で")
                    && !is_place_location_bunsetsu(&b.atoms)
                    && is_instrument_bunsetsu(&b.atoms);
                let s = render_bunsetsu_obl(&b, target);
                if !s.is_empty() {
                    obliques.push(PlacedArg {
                        text: s,
                        case_particle: b.case_particle.clone(),
                        is_instrument_de: is_inst,
                        is_spatial_made: is_spatial_made_bunsetsu(&b),
                        is_clock: b.atoms.iter().any(is_clock_atom),
                        is_building_floor: b.atoms.iter().any(is_building_floor),
                        transfer_dative: b.case_particle.as_deref() == Some("に")
                            && !is_time_or_place_bunsetsu(&b.atoms),
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
                        is_clock: false,
                        is_building_floor: b.atoms.iter().any(is_building_floor),
                        transfer_dative: b.case_particle.as_deref() == Some("に")
                            && !is_time_or_place_bunsetsu(&b.atoms),
                    });
                }
            }
            BunsetsuRole::Obj => {
                let s = render_bunsetsu_np(&b, target);
                if !s.is_empty() {
                    objs.push(s);
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
    let mut added_mo_adv = false;
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
                    is_clock: false,
                    is_building_floor: false,
                    transfer_dative: false,
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
            if o.case_particle.as_deref() == Some("に") && !o.is_clock {
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
                && matches!(
                    veh,
                    "電車"
                        | "公車"
                        | "巴士"
                        | "計程車"
                        | "新幹線"
                        | "飛機"
                        | "船"
                        | "列車"
                        | "地鐵"
                )
            {
                o.text = format!("坐{veh}");
            }
        }
    }
    let is_cause_verb = predicates_is_cause_verb(&predicates);
    let has_causative = pre_verb_mods.iter().any(|m| m == "讓");

    // Extract directional `に` goals (from `obliques` or `iobjs`) BEFORE passive agent consumes `iobjs`!
    let mut goal_tails: Vec<String> = Vec::new();
    let mut cause_prefixes: Vec<String> = Vec::new();
    let mut pre_iobjs: Vec<String> = Vec::new();

    if is_goal_verb {
        let mut kept_obls = Vec::new();
        for o in obliques.drain(..) {
            if o.case_particle.as_deref() == Some("に") && !o.is_clock {
                let raw_np = o
                    .text
                    .strip_prefix('在')
                    .or_else(|| o.text.strip_prefix('對'))
                    .or_else(|| o.text.strip_prefix('給'))
                    .unwrap_or(&o.text);
                goal_tails.push(format_goal_tail(&verb_core, raw_np, o.is_building_floor));
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
                goal_tails.push(format_goal_tail(&verb_core, raw_np, io.is_building_floor));
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

    // Assemble SVO order:
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
    // A place marked by まで is the endpoint of a motion event. Keep this
    // local to まで: the global goal flag also changes recipients and vehicles.
    let mut made_endpoints = Vec::new();
    for o in obliques {
        let _ = o.is_instrument_de;
        if has_spatial_made && o.is_spatial_made {
            made_endpoints.push(o.text);
        } else if o.case_particle.as_deref() == Some("まで") && !is_goal_verb {
            let raw_np = o.text.strip_prefix('到').unwrap_or(&o.text);
            parts.push(format!("連{raw_np}都"));
        } else {
            parts.push(o.text);
        }
    }
    parts.extend(cause_prefixes);
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
    parts.extend(inner_pre_mods);
    parts.extend(transfer_prefix);
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
    let is_change_of_state_you =
        raw_clause.contains("ようになり") || raw_clause.contains("ようになった");
    let trailing_change_le =
        if is_change_of_state_you && post_verb_mods.last().is_some_and(|m| m == "了") {
            post_verb_mods.pop()
        } else {
            None
        };
    if !goal_tails.is_empty() {
        parts.extend(goal_tails);
        parts.append(&mut post_verb_mods);
    } else {
        parts.extend(post_verb_mods);
    }
    parts.extend(transfer_recipient);
    parts.extend(objs);
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

fn normalize_ja_atoms(atoms: &[Atom]) -> Vec<Atom> {
    let mut out = Vec::with_capacity(atoms.len());
    let mut i = 0usize;
    while i < atoms.len() {
        let a = &atoms[i];

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

        // Keep `A の B`, `Verb+て+いる/しまう/ください`, or `A さんの` together up to the trailing particles.
        if !next_is_func
            && !next_is_aux
            && !keep_te_subsidiary
            && (is_func || !is_compound_noun_pair(atom, next))
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
    is_noun_bunsetsu(b)
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
    if det.is_locked
        || noun.is_locked
        || det.upos != Some(Upos::Det)
        || noun.upos != Some(Upos::Noun)
        || det.span.end_char != noun.span.start_char
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

fn merge_adnominal_bunsetsus(
    bunsetsus: Vec<Bunsetsu>,
    _doc: &Document,
    _sentence_idx: usize,
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
                let mut acl_bunsetsus: Vec<Bunsetsu> = Vec::new();
                while let Some(cand) = res.last() {
                    let is_bare_temporal =
                        matches!(cand.role, BunsetsuRole::Other | BunsetsuRole::Adv)
                            && cand.case_particle.is_none()
                            && cand.atoms.iter().any(|a| {
                                !a.is_locked
                                    && matches!(
                                        a.surface.as_str(),
                                        "長年"
                                            | "長期間"
                                            | "長い間"
                                            | "数年"
                                            | "数ヶ月"
                                            | "数日"
                                            | "何年"
                                    )
                            });
                    if is_embedded_acl_arg(cand) || is_bare_temporal {
                        acl_bunsetsus.push(res.pop().expect("checked last"));
                    } else {
                        break;
                    }
                }
                acl_bunsetsus.reverse();
                acl_bunsetsus.push(pred_b);
                let (mut inner, _) = translate_clause_bunsetsus(
                    acl_bunsetsus,
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

                        let mut acl_bunsetsus: Vec<Bunsetsu> = Vec::new();
                        while let Some(cand) = res.last() {
                            if is_embedded_acl_arg(cand) {
                                acl_bunsetsus.push(res.pop().expect("checked last"));
                            } else {
                                break;
                            }
                        }
                        acl_bunsetsus.reverse();
                        acl_bunsetsus.push(mod_b);
                        let (mut inner, _) = translate_clause_bunsetsus(
                            acl_bunsetsus,
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
                    let mut acl_bunsetsus: Vec<Bunsetsu> = Vec::new();
                    while let Some(cand) = res.last() {
                        if is_embedded_acl_arg(cand) {
                            acl_bunsetsus.push(res.pop().expect("checked last"));
                        } else {
                            break;
                        }
                    }
                    acl_bunsetsus.reverse();
                    acl_bunsetsus.push(pred_b);
                    let (mut inner, _) = translate_clause_bunsetsus(
                        acl_bunsetsus,
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

fn is_time_or_place_bunsetsu(atoms: &[Atom]) -> bool {
    atoms.iter().any(is_clock_atom)
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
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

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
    fn wishes_and_negative_questions_keep_the_direction() {
        assert_eq!(translate("駅まで行きたいです。"), "想去車站。");
        assert_eq!(translate("公園まで行きたいです。"), "想去公園。");
        assert_eq!(translate("公園まで走りたいです。"), "想跑到公園。");
        assert_eq!(translate("駅まで行きませんか。"), "不去車站嗎。");
        assert_eq!(translate("公園まで行きませんか。"), "不去公園嗎。");
        assert_eq!(translate("公園まで走りませんか。"), "不跑到公園嗎。");
    }

    #[test]
    fn negative_questions_keep_nonhuman_subjects() {
        assert_eq!(
            translate("このバスは駅まで行きませんか。"),
            "這輛公車不去車站嗎。"
        );
        assert_eq!(
            translate("この電車は駅まで行きませんか。"),
            "這輛電車不去車站嗎。"
        );
    }

    #[test]
    fn near_negative_controls_keep_their_existing_meaning() {
        assert_eq!(translate("駅まで行きませんでしたか。"), "到車站沒有去嗎。");
        assert_eq!(translate("公園まで描きませんか。"), "連公園都不畫畫嗎。");
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
        .is_some_and(|a| {
            if is_building_floor(a) {
                return true;
            }
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
        })
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

fn render_bunsetsu_np(b: &Bunsetsu, _target: TargetVariant) -> String {
    let mut out = String::new();
    let mut i = 0usize;
    while i < b.atoms.len() {
        let a = &b.atoms[i];
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

fn render_bunsetsu_obl(b: &Bunsetsu, target: TargetVariant) -> String {
    let core = render_bunsetsu_np(b, target);
    if core.is_empty() {
        return core;
    }
    match b.case_particle.as_deref() {
        Some("で") if !is_place_location_bunsetsu(&b.atoms) && is_instrument_bunsetsu(&b.atoms) => {
            if core.starts_with('用') {
                core
            } else {
                format!("用{core}")
            }
        }
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
        pre.insert(0, if has_locative_floor { "在" } else { "是" }.to_owned());
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
        post.push(if target == TargetVariant::ZhHans {
            "着".to_owned()
        } else {
            "著".to_owned()
        });
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
            ("私は夕方まで走ります。", "我連傍晚都跑。"),
        ] {
            assert_eq!(
                translate(source, &TranslationGlossary::new()).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn building_floor_leaves_compounds_names_and_locks_alone() {
        for (source, keep) in [
            ("この計画は三段階です。", "三段階"),
            ("二階微分を計算します。", "二階微分"),
            ("この微分は二階です。", "二階"),
            ("二階で微分します。", "二階"),
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
            ("明日の午後、私は家で本を読みます。", "明天午後"),
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
