//! Multi-layer deterministic `ja -> zh-Hant / zh-Hans` translation engine.
//!
//! Consumes all `nlp.doc.v1` layers (`segmentation`, `tokens`, `entities`, `syntax`, `frames`,
//! `coref` / `zero_mentions`, `relations`) plus `SentenceConditioning` and `SpanLockTable` to
//! reorder Japanese SOV clauses into Chinese SVO, synthesize predicates, resolve pro-drop, and
//! render uncensored, idiomatic `zh-Hant` or `zh-Hans` prose and dialogue.

use std::collections::HashMap;

use faberline_nlp::Span;
use faberline_nlp::doc::{Document, FrameRole, Token, Upos};

use crate::conditioning::zh_pronoun_for_hint;
use crate::lock::SpanLockTable;
use crate::schema::{ClauseAlignment, SentenceConditioning, TargetVariant};
use crate::variant::convert_zh_text_to_target;

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

    for (seg_idx, seg) in segments.iter().enumerate() {
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
                            ClauseDiscourseCtx::default(),
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
                            ClauseDiscourseCtx::default(),
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
                let cur_is_subordinate = !cur_ends_kuseni
                    && trimmed_atoms.last().is_some_and(|a| {
                        !a.is_locked
                            && (is_clause_split_sconj(&a.surface)
                                || matches!(a.surface.as_str(), "と" | "でも" | "としても"))
                    });

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
                };
                let (mut rendered, had_subj_only) = translate_clause_atoms(
                    trimmed_atoms,
                    doc,
                    sentence_idx,
                    cond_opt,
                    allow_pro_drop,
                    ctx,
                    target,
                );
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
                in_toshitemo_follow = cur_ends_toshitemo;

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
                                )
                        }),
                        _ => false,
                    });
                if next_starts_with_contrast {
                    if let Some(stripped) = rendered.strip_suffix("，但是") {
                        rendered = stripped.to_owned();
                    } else if let Some(stripped) = rendered.strip_suffix("，不過") {
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
                if let Some(p) = trailing_punct {
                    if !((p == "、" || p == ",")
                        && (rendered.ends_with("，就立刻")
                            || rendered.ends_with("，但是")
                            || rendered.ends_with("，不過")
                            || rendered.ends_with("，而且")
                            || rendered.ends_with('，')))
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
        convert_zh_text_to_target(&out, target)
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
                    (first.1.start_char, last.1.end_char.min(final_chars))
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
        && matches!(
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
        )
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

        // SCONJ clause boundary: `から`, `ので`, `けれど`, `けれども`, `けど`, `のに`, `たら`, `なら`, `ば`, `ながら`, `ても`
        // Guard obligation `ねばならない` / `なければならない` (`なら` before `ない`, `ば` before `なら`) and `と言っても過言`.
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
        let is_kagon_sconj = matches!(a.surface.as_str(), "ても" | "も")
            && atoms[i + 1..]
                .iter()
                .any(|later| !later.is_locked && later.surface == "過言");
        if !a.is_locked
            && !is_obligation_nara
            && !is_kagon_sconj
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
                    "すぐ" | "すぐに" | "は" | "も" | "する" | "し" | "なる" | "なっ" | "いう"
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

        // Concessive `[て] + [も]` (e.g. `忙しくても`): keep `[て, も]` together instead of splitting at `て`.
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

        // Conjunctive `て`/`で` clause boundary (preserving `ないで`, `なくては`/`なくても`, `によって`, and `て + subsidiary verb`).
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

#[derive(Clone, Copy, Default)]
struct ClauseDiscourseCtx {
    in_nagara_follow: bool,
    in_concessive_follow: bool,
    in_kuseni_follow: bool,
    in_toshitemo_follow: bool,
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
    ctx: ClauseDiscourseCtx,
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

    if let Some(res) = try_translate_cognitive_matrix(atoms, doc, sentence_idx, target) {
        return res;
    }

    if let Some(res) = try_translate_to_ieba_or_uso(atoms, doc, sentence_idx, target) {
        return res;
    }

    if let Some(res) = try_translate_kagiranai(atoms, doc, sentence_idx, target) {
        return res;
    }

    // Multi-token phrase & collocation replacement over atoms first, then normalize remaining atoms.
    let merged_atoms = apply_phrase_matches(atoms);
    let norm_atoms = normalize_ja_atoms(&merged_atoms);

    // Group into bunsetsus, then fold adnominal modifiers (`の`, prenominal `Adj`, `acl`).
    let grouped = group_into_bunsetsus(&norm_atoms, doc, sentence_idx);
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
                ClauseDiscourseCtx::default(),
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

    let suffix_len = if core.len() >= 5 {
        let s = &core[core.len() - 5..];
        if !s.iter().any(|a| a.is_locked)
            && s[0].surface == "と"
            && s[1].surface == "は"
            && matches!(s[2].surface.as_str(), "限り" | "かぎり")
            && s[3].surface == "ませ"
            && s[4].surface == "ん"
        {
            Some(5usize)
        } else {
            None
        }
    } else {
        None
    }
    .or_else(|| {
        if core.len() >= 4 {
            let s = &core[core.len() - 4..];
            if !s.iter().any(|a| a.is_locked)
                && s[0].surface == "と"
                && s[1].surface == "は"
                && matches!(s[2].surface.as_str(), "限ら" | "かぎら")
                && s[3].surface == "ない"
            {
                return Some(4usize);
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

    let (inner_zh, _) = translate_clause_atoms(
        inner,
        doc,
        sentence_idx,
        None,
        false,
        ClauseDiscourseCtx::default(),
        target,
    );
    if inner_zh.is_empty() {
        return None;
    }

    let prefix = if is_clausal_or_i_adj {
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

    let pre_to = &core[..v_idx - 1];
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

    let (inner_zh, _) = translate_clause_atoms(
        inner_atoms,
        doc,
        sentence_idx,
        None,
        false,
        ClauseDiscourseCtx::default(),
        target,
    );
    if inner_zh.is_empty() {
        return None;
    }

    let is_neg = tail.iter().any(|a| {
        !a.is_locked
            && matches!(
                a.surface.as_str(),
                "ない" | "なかった" | "ません" | "ませ" | "ん"
            )
    });
    let verb_zh = if is_neg { neg_zh } else { pos_zh };
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
        ClauseDiscourseCtx::default(),
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

fn translate_clause_bunsetsus(
    mut bunsetsus: Vec<Bunsetsu>,
    raw_clause: &str,
    conditioning: Option<&SentenceConditioning>,
    allow_pro_drop_inject: bool,
    ctx: ClauseDiscourseCtx,
    target: TargetVariant,
) -> (String, bool) {
    // Track extra bunsetsu-level metadata for `に`/`で` localization.
    #[derive(Clone)]
    struct PlacedArg {
        text: String,
        case_particle: Option<String>,
        is_instrument_de: bool,
    }

    struct PlacedSubj {
        text: String,
        has_mo: bool,
        had_ga: bool,
        is_hou: bool,
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

    for b in bunsetsus.drain(..) {
        if let Some(sc) = b.sconj_tail.as_deref() {
            match sc {
                "から" | "ので" => sconj_prefix = Some("因為"),
                "からといって" => sconj_prefix = Some("即便因為"),
                "けれど" | "けれども" | "けど" | "のに" => sconj_suffix = Some("，但是"),
                "だけど" => sconj_suffix = Some("，不過"),
                "し" => sconj_suffix = Some("，而且"),
                "たら" | "なら" | "ば" => sconj_prefix = Some("如果"),
                "ながら" | "つつ" => sconj_prefix = Some("一邊"),
                "ても" | "でも" => sconj_prefix = Some("即使"),
                "としても" => {
                    sconj_prefix = Some("就算");
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
                        });
                    } else {
                        subj_entries.push(PlacedSubj {
                            text: s,
                            has_mo,
                            had_ga,
                            is_hou,
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
                    });
                }
            }
            BunsetsuRole::Adv => {
                let s = render_bunsetsu_np(&b, target);
                if !s.is_empty() {
                    if matches!(
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
                    ) {
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
        adverbs.retain(|a| !matches!(a.as_str(), "如果" | "就算" | "萬一"));
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

    // Synthesize predicate complex.
    let (mut pre_verb_mods, mut verb_core, mut post_verb_mods, passive_agent_needed) =
        synthesize_predicate_complex(&predicates, !objs.is_empty(), target);

    if is_toshitemo_clause {
        post_verb_mods.retain(|m| m != "了");
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
    if pred_is_aru && verb_core == "有" {
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
                if s.had_ga {
                    objs.push(s.text);
                } else {
                    rem_subjs.push(s);
                }
            }
            subj_entries = rem_subjs;
        } else if objs.is_empty()
            && !subj_entries.is_empty()
            && subj_entries.iter().all(|s| s.had_ga)
        {
            for s in subj_entries.drain(..) {
                objs.push(s.text);
            }
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

    // Passive patient promotion: when passive has no overt subject and has an `を` object, promote `objs` before `被`.
    if passive_agent_needed && subjects.is_empty() && !objs.is_empty() {
        subjects.append(&mut objs);
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
            if o.case_particle.as_deref() == Some("に") {
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
            if o.case_particle.as_deref() == Some("に") {
                let raw_np = o
                    .text
                    .strip_prefix('在')
                    .or_else(|| o.text.strip_prefix('對'))
                    .or_else(|| o.text.strip_prefix('給'))
                    .unwrap_or(&o.text);
                goal_tails.push(format_goal_tail(&verb_core, raw_np));
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
                goal_tails.push(format_goal_tail(&verb_core, raw_np));
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

    // Promote `に` complements to direct objects for verbs like `気づく` (`發現`), `かかる` (`取決於`), `拍車をかける` (`加劇`).
    let promotes_ni_to_obj = matches!(verb_core.as_str(), "取決於" | "加劇")
        || predicates.iter().any(|b| {
            b.atoms.iter().any(|a| {
                !a.is_locked
                    && (matches!(
                        a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                        "気づく" | "気付く"
                    ) || a.surface.starts_with("気づ")
                        || a.surface.starts_with("気付"))
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

    if ctx.in_kuseni_follow && sconj_prefix.is_none() {
        parts.push("卻".to_owned());
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
    let has_front_modal = !front_modals.is_empty();
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

    let _ = has_front_modal;
    parts.extend(temporal_others);
    for o in obliques {
        let _ = o.is_instrument_de;
        if o.case_particle.as_deref() == Some("まで") && !is_goal_verb {
            let raw_np = o.text.strip_prefix('到').unwrap_or(&o.text);
            parts.push(format!("連{raw_np}都"));
        } else {
            parts.push(o.text);
        }
    }
    parts.extend(cause_prefixes);
    if ctx.in_concessive_follow
        && !adverbs.iter().any(|x| x == "都")
        && !inner_pre_mods.iter().any(|x| x == "都")
    {
        parts.push("都".to_owned());
    }
    parts.extend(adverbs);
    parts.extend(pre_iobjs);
    parts.extend(others);
    parts.extend(inner_pre_mods);
    if !verb_core.is_empty() {
        parts.push(verb_core);
    }
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

fn format_goal_tail(verb_core: &str, raw_np: &str) -> String {
    if verb_core == "進入" || raw_np.starts_with('到') {
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
            let adj_zh = translate_single_ja_term(&dict_form, Some(&dict_form), Some(Upos::Adj));
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
    b.atoms.iter().any(|a| {
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
    })
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
    if b.sconj_tail.is_some() || b.sfp_tail.is_some() {
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

fn is_embedded_acl_arg(b: &Bunsetsu) -> bool {
    if b.atoms.is_empty() {
        return false;
    }
    if b.case_particle
        .as_deref()
        .is_some_and(|cp| matches!(cp, "が" | "を" | "で" | "に" | "へ" | "から" | "まで"))
    {
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
                    ClauseDiscourseCtx::default(),
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
                            ClauseDiscourseCtx::default(),
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

                    // Pure `N の` modifier (e.g. `美咲の`, `机の`, `3冊の`).
                    mod_b.atoms.pop();
                    let omit_de = starts_with_num_counter(&mod_b) || is_spatial_head_bunsetsu(&b);
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
                        let mut adj_str = render_bunsetsu_np(&pred_b, target);
                        if adj_str == "以往的"
                            && b.atoms.first().is_some_and(|a| {
                                a.is_locked
                                    && (a.upos == Some(Upos::Adj) || a.surface.ends_with('的'))
                            })
                        {
                            adj_str = "以往".to_owned();
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
                        ClauseDiscourseCtx::default(),
                        target,
                    );
                    while inner.ends_with('了') {
                        inner.pop();
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
                        Some("に" | "で" | "へ" | "から" | "まで" | "と")
                    )
                {
                    // Preserve semantic case marker when focused by topic/inclusive particle (`日本では`, `場とも`).
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
    if (has_verb_or_adj || has_copula)
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
                if is_time_or_place_bunsetsu(&atoms) || frame_role == Some(FrameRole::Argm) {
                    BunsetsuRole::Obl
                } else {
                    BunsetsuRole::IObj
                }
            }
            "で" | "から" | "まで" | "へ" | "より" | "によって" => BunsetsuRole::Obl,
            "と" | "とも" => BunsetsuRole::Other,
            _ => BunsetsuRole::Other,
        }
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
    is_place_location_bunsetsu(atoms)
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
            if core.starts_with('在') {
                core
            } else {
                format!("在{core}")
            }
        }
        Some("から") => format!("從{core}"),
        Some("まで") => format!("到{core}"),
        Some("へ") => format!("往{core}"),
        Some("より") => format!("比{core}"),
        Some("によって") => format!("在{core}下"),
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
    let has_te_kureru = joined.contains("てくれる") || joined.contains("てくれた");
    let has_te_morau = joined.contains("てもらう") || joined.contains("てもらった");
    let has_te_shimau = joined.contains("てしまう")
        || joined.contains("てしまった")
        || joined.contains("ちゃう")
        || joined.contains("ちゃった")
        || joined.contains("じゃう")
        || joined.contains("じゃった");
    let has_te_iru = joined.contains("ている")
        || joined.contains("ていた")
        || joined.contains("てる")
        || joined.contains("てた")
        || joined.contains("ています")
        || joined.contains("ていました")
        || joined.contains("ていない")
        || joined.contains("ていなく")
        || joined.contains("ていなかった")
        || joined.contains("ていません")
        || joined.contains("ていませんでした");
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
    let has_neg = !has_neg_past
        && !has_naide
        && !has_naide_kudasai
        && !has_obligation
        && all_atoms.iter().enumerate().any(|(idx, a)| {
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
        });
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
    let is_stative_te_iru = has_te_iru
        && all_atoms.iter().any(|a| {
            !a.is_locked
                && matches!(
                    a.lemma.as_deref().unwrap_or(a.surface.as_str()),
                    "残る" | "知る" | "持つ" | "住む" | "着る" | "似る" | "結婚する"
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
    if has_te_kureru {
        pre.push("幫我".to_owned());
    }
    if has_te_morau {
        pre.push("請對方幫忙".to_owned());
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
        pre.push("不是".to_owned());
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

    if is_stative_te_iru
        && let Some(last) = cores.last_mut()
        && last == "留下"
    {
        *last = "留".to_owned();
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
                    "ない" | "なかった" | "ありません" | "ませんでした"
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
        pre.insert(0, "是".to_owned());
    }

    if has_te_shimau {
        post.push("掉了".to_owned());
    } else if is_stative_te_iru
        && !has_neg
        && !has_neg_past
        && !matches!(
            cores.last().map(String::as_str),
            Some("知道" | "認識" | "了解" | "像" | "相似" | "結婚")
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
    if surface.chars().all(is_han) {
        let mapped: String = surface.chars().map(map_ja_shinjitai_char).collect();
        return convert_zh_text_to_target(&mapped, TargetVariant::ZhHant);
    }

    // 4. Fallback: preserve surface.
    surface.to_owned()
}

fn map_ja_shinjitai_char(c: char) -> char {
    match c {
        '図' => '圖',
        '読' => '讀',
        '国' => '國',
        '学' => '學',
        '駅' => '站',
        '気' => '氣',
        '声' => '聲',
        '体' => '體',
        '髪' => '髮',
        '黒' => '黑',
        '広' => '廣',
        '楽' => '樂',
        '変' => '變',
        '恋' => '戀',
        '悪' => '惡',
        '帰' => '歸',
        '絵' => '繪',
        '売' => '賣',
        '昼' => '晝',
        '豊' => '豐',
        '静' => '靜',
        '触' => '觸',
        '届' => '屆',
        '絶' => '絕',
        '続' => '續',
        '残' => '殘',
        '遅' => '遲',
        '辺' => '邊',
        '抜' => '拔',
        '揺' => '搖',
        '乱' => '亂',
        '余' => '餘',
        '亀' => '龜',
        '茎' => '莖',
        '顔' => '顏',
        '薬' => '藥',
        '痴' => '癡',
        '沢' => '澤',
        '実' => '實',
        '宝' => '寶',
        '剣' => '劍',
        '戦' => '戰',
        '竜' => '龍',
        '桜' => '櫻',
        _ => c,
    }
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
