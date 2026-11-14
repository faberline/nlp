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

    for seg in &segments {
        let tgt_start = out.chars().count();
        match seg {
            Segment::Quote {
                open_q,
                inner_atoms,
                close_q,
                span,
            } => {
                let open_zh = if target == TargetVariant::ZhHans {
                    if *open_q == '『' { '‘' } else { '“' }
                } else if *open_q == '『' {
                    '『'
                } else {
                    '「'
                };
                let close_zh = if target == TargetVariant::ZhHans {
                    if *close_q == '』' { '’' } else { '”' }
                } else if *close_q == '』' {
                    '』'
                } else {
                    '」'
                };
                out.push(open_zh);
                let inner_segs = split_atoms_into_segments(inner_atoms, chars, *span);
                for iseg in &inner_segs {
                    if let Segment::Clause {
                        atoms: catoms,
                        trailing_punct,
                        ..
                    } = iseg
                    {
                        let rendered =
                            translate_clause_atoms(catoms, doc, sentence_idx, None, false, target);
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
            Segment::Clause {
                atoms: catoms,
                trailing_punct,
                span,
            } => {
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
                let cond_opt = if is_first_non_quote_clause {
                    is_first_non_quote_clause = false;
                    Some(conditioning)
                } else {
                    None
                };
                let rendered = translate_clause_atoms(
                    trimmed_atoms,
                    doc,
                    sentence_idx,
                    cond_opt,
                    cond_opt.is_some(),
                    target,
                );
                out.push_str(&rendered);
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

    let final_zh = convert_zh_text_to_target(&out, target);
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
    Clause {
        atoms: Vec<Atom>,
        trailing_punct: Option<String>,
        span: Span,
    },
    Punct(String, Span),
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
            while j < atoms.len() {
                if !atoms[j].is_locked && atoms[j].surface == close_target {
                    q_end = atoms[j].span.end_char;
                    j += 1;
                    break;
                }
                inner.push(atoms[j].clone());
                q_end = atoms[j].span.end_char;
                j += 1;
            }
            segments.push(Segment::Quote {
                open_q,
                inner_atoms: inner,
                close_q: close_target.chars().next().unwrap_or('」'),
                span: Span {
                    start_char: q_start,
                    end_char: q_end,
                },
            });
            i = j;
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

        // SCONJ clause boundary: `から`, `ので`, `けれど`, `けれども`, `けど`, `のに`, `たら`, `なら`, `ながら`, `ても`
        // Split after SCONJ if there are more non-punct tokens after it.
        if !a.is_locked
            && is_clause_split_sconj(&a.surface)
            && i + 1 < atoms.len()
            && !is_punct_atom(&atoms[i + 1])
        {
            flush_clause(&mut current, Some("，".to_owned()), &mut segments);
        }
        i += 1;
    }

    flush_clause(&mut current, None, &mut segments);
    segments
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
/// 2. Groups tokens into bunsetsus using `doc.syntax` & `doc.frames`.
/// 3. Reorders SOV -> SVO, synthesizes predicate complex, resolves `の` / relative clauses,
///    and applies Layer 6 pro-drop resolution when subject is omitted.
fn translate_clause_atoms(
    atoms: &[Atom],
    doc: &Document,
    sentence_idx: usize,
    conditioning: Option<&SentenceConditioning>,
    allow_pro_drop_inject: bool,
    target: TargetVariant,
) -> String {
    if atoms.is_empty() {
        return String::new();
    }

    // Fast path: check if entire clause (excluding trailing SFP) matches a curated idiom.
    let raw_clause: String = atoms.iter().map(|a| a.surface.as_str()).collect();
    if let Some(idiom) = lookup_curated_phrase(&raw_clause) {
        return idiom.to_owned();
    }

    // Multi-token phrase & collocation replacement over atoms.
    let merged_atoms = apply_phrase_matches(atoms);

    // Group into bunsetsus.
    let mut bunsetsus = group_into_bunsetsus(&merged_atoms, doc, sentence_idx);

    // Separate connector, subject, obliques, adverbs, iobj, obj, predicate, and others.
    let mut connectors: Vec<String> = Vec::new();
    let mut subjects: Vec<String> = Vec::new();
    let mut obliques: Vec<String> = Vec::new();
    let mut adverbs: Vec<String> = Vec::new();
    let mut iobjs: Vec<String> = Vec::new();
    let mut objs: Vec<String> = Vec::new();
    let mut predicates: Vec<Bunsetsu> = Vec::new();
    let mut others: Vec<String> = Vec::new();
    let mut sconj_prefix: Option<&'static str> = None;
    let mut sconj_suffix: Option<&'static str> = None;
    let mut sfp_suffix: Option<&'static str> = None;

    for b in bunsetsus.drain(..) {
        if let Some(sc) = b.sconj_tail.as_deref() {
            match sc {
                "から" | "ので" => sconj_prefix = Some("因為"),
                "けれど" | "けれども" | "けど" | "のに" => sconj_suffix = Some("，但是"),
                "たら" | "なら" | "ば" => sconj_prefix = Some("如果"),
                "ながら" | "つつ" => sconj_prefix = Some("一邊"),
                "ても" | "でも" => sconj_prefix = Some("即使"),
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
                let s = render_bunsetsu_np(&b, target);
                if !s.is_empty() {
                    subjects.push(s);
                }
            }
            BunsetsuRole::Obl => {
                let s = render_bunsetsu_obl(&b, target);
                if !s.is_empty() {
                    obliques.push(s);
                }
            }
            BunsetsuRole::Adv => {
                let s = render_bunsetsu_np(&b, target);
                if !s.is_empty() {
                    adverbs.push(s);
                }
            }
            BunsetsuRole::IObj => {
                let s = render_bunsetsu_iobj(&b, target);
                if !s.is_empty() {
                    iobjs.push(s);
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
                    others.push(s);
                }
            }
        }
    }

    // Layer 6 Pro-Drop resolution: if no overt subject and narrative/coref resolved pro-drop,
    // inject pronoun when appropriate (and not an imperative or greeting).
    let pred_is_imperative = predicates.iter().any(predicate_is_imperative);
    if subjects.is_empty()
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
    let (pre_verb_mods, verb_core, post_verb_mods, passive_agent_needed) =
        synthesize_predicate_complex(&predicates, !objs.is_empty());

    // Assemble SVO order:
    // [SCONJ_PREFIX] + [Connector] + [Subj] + [Time/Place OBL] + [Adverbs] + [Others]
    // + [Modal/Neg/Passive] + [Verb] + [Post-Verb Aspect] + [Obj] + [IObj (給...)] + [SFP] + [SCONJ_SUFFIX]
    let mut parts: Vec<String> = Vec::new();
    if let Some(sp) = sconj_prefix {
        parts.push(sp.to_owned());
    }
    parts.extend(connectors);
    parts.extend(subjects);
    if passive_agent_needed && !iobjs.is_empty() {
        // Passive `に` agent: `に` -> `被[Agent]` placed before verb.
        for io in iobjs.drain(..) {
            let trimmed = io
                .strip_prefix('對')
                .or_else(|| io.strip_prefix('給'))
                .unwrap_or(&io);
            parts.push(format!("被{trimmed}"));
        }
    } else if passive_agent_needed {
        parts.push("被".to_owned());
    }
    parts.extend(obliques);
    parts.extend(adverbs);
    parts.extend(others);
    parts.extend(pre_verb_mods);
    if !verb_core.is_empty() {
        parts.push(verb_core);
    }
    parts.extend(post_verb_mods);
    parts.extend(objs);
    for io in iobjs {
        if let Some(rest) = io.strip_prefix('對') {
            parts.push(format!("給{rest}"));
        } else {
            parts.push(io);
        }
    }
    if let Some(sfp) = sfp_suffix {
        parts.push(sfp.to_owned());
    }
    if let Some(ss) = sconj_suffix {
        parts.push(ss.to_owned());
    }

    join_zh_parts(&parts)
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
        let mut matched_len = 0usize;
        let mut matched_zh: Option<&'static str> = None;
        let max_win = (atoms.len() - i).min(6);
        for len in (2..=max_win).rev() {
            let slice = &atoms[i..i + len];
            if slice.iter().any(|a| a.is_locked) {
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
            out.push(Atom {
                span: Span {
                    start_char: atoms[i].span.start_char,
                    end_char: last.span.end_char,
                },
                surface: zh.to_owned(),
                lemma: Some(zh.to_owned()),
                upos: last.upos,
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
        let next_is_func = atoms.get(idx + 1).is_some_and(is_bunsetsu_tail_atom);
        let next_is_aux = atoms.get(idx + 1).is_some_and(|n| {
            !n.is_locked
                && matches!(
                    n.upos,
                    Some(Upos::Aux | Upos::Part | Upos::Adp | Upos::Sconj)
                )
        });

        // Keep `A の B` or `A さんの` together only up to the particle, then split after particle/aux
        // when the next token is a new content word.
        if !next_is_func
            && !next_is_aux
            && (is_func || !is_compound_noun_pair(atom, atoms.get(idx + 1)))
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
    let mut deprel_hint = "";

    for a in &atoms {
        if a.doc_tok_idx.is_some() {
            head_doc_idx = a.doc_tok_idx;
        }
        if !a.deprel.is_empty() {
            deprel_hint = a.deprel.as_str();
        }
        if a.is_locked {
            continue;
        }
        match a.upos {
            Some(Upos::Adp) => {
                if matches!(
                    a.surface.as_str(),
                    "が" | "は"
                        | "を"
                        | "に"
                        | "で"
                        | "と"
                        | "から"
                        | "まで"
                        | "へ"
                        | "より"
                        | "の"
                        | "も"
                ) {
                    case_particle = Some(a.surface.clone());
                }
            }
            Some(Upos::Sconj) => {
                sconj_tail = Some(a.surface.clone());
            }
            Some(Upos::Part) => {
                if matches!(
                    a.surface.as_str(),
                    "か" | "よ" | "ね" | "な" | "わ" | "ぞ" | "ぜ" | "かな" | "よね"
                ) {
                    sfp_tail = Some(a.surface.clone());
                }
            }
            Some(Upos::Verb) => has_verb_or_adj = true,
            Some(Upos::Adj) => has_verb_or_adj = true,
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
        if is_discourse_connector(&a.surface) {
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

    let role = if is_connector && atoms.len() <= 2 {
        BunsetsuRole::Connector
    } else if has_verb_or_adj || has_copula || deprel_hint == "root" {
        BunsetsuRole::Predicate
    } else if let Some(cp) = case_particle.as_deref() {
        match cp {
            "が" | "は" | "も" => BunsetsuRole::Subj,
            "を" => BunsetsuRole::Obj,
            "に" => {
                // Time/location vs indirect object: check if noun is time/place or animate/frame ARG1.
                if is_time_or_place_bunsetsu(&atoms) || frame_role == Some(FrameRole::Argm) {
                    BunsetsuRole::Obl
                } else {
                    BunsetsuRole::IObj
                }
            }
            "で" | "から" | "まで" | "へ" | "より" => BunsetsuRole::Obl,
            "と" => BunsetsuRole::Other,
            _ => BunsetsuRole::Other,
        }
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

    Bunsetsu {
        atoms,
        role,
        case_particle,
        sconj_tail,
        sfp_tail,
    }
}

fn is_time_or_place_bunsetsu(atoms: &[Atom]) -> bool {
    atoms.iter().any(|a| {
        matches!(
            a.surface.as_str(),
            "今日"
                | "明日"
                | "昨日"
                | "今"
                | "朝"
                | "昼"
                | "夜"
                | "部屋"
                | "学校"
                | "図書館"
                | "公園"
                | "駅"
                | "家"
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
            | "から"
            | "まで"
            | "へ"
            | "より"
            | "も"
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
}

/// Synthesize predicate complex from predicate bunsetsu(s).
/// Returns `(pre_verb_mods, verb_core, post_verb_mods, needs_passive_agent)`.
fn synthesize_predicate_complex(
    predicates: &[Bunsetsu],
    _has_obj: bool,
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
    let has_kudasai =
        !has_naide_kudasai && joined.contains("てください") || joined.contains("ください");
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
        || joined.contains("ていました");
    let has_sugiru =
        joined.contains("すぎる") || joined.contains("すぎた") || joined.contains("過ぎる");
    let has_tai = surfaces
        .iter()
        .any(|&s| matches!(s, "たい" | "たかっ" | "たかった"));
    let has_neg_past = surfaces
        .iter()
        .any(|&s| matches!(s, "なかった" | "ませんでした"));
    let has_neg = !has_neg_past
        && surfaces
            .iter()
            .any(|&s| matches!(s, "ない" | "ません" | "ず" | "ぬ" | "ん"));
    let has_past = surfaces
        .iter()
        .any(|&s| matches!(s, "た" | "だ" | "ました" | "だった" | "でした"));
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
    let is_copula_neg = joined.contains("じゃない")
        || joined.contains("ではない")
        || joined.contains("ではありません");
    let is_copula_pos = !is_copula_neg && surfaces.iter().any(|&s| matches!(s, "だ" | "です"));

    if has_naide_kudasai {
        pre.push("請不要".to_owned());
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
    if is_copula_neg {
        pre.push("不是".to_owned());
    } else if has_neg_past {
        pre.push("沒有".to_owned());
    } else if has_neg && !has_naide_kudasai {
        pre.push("不".to_owned());
    }
    if has_tai {
        pre.push("想".to_owned());
    }
    if has_te_iru {
        pre.push("正在".to_owned());
    }
    if has_sugiru {
        pre.push("太".to_owned());
    }

    for a in &all_atoms {
        if a.is_locked {
            cores.push(a.surface.clone());
            continue;
        }
        if is_predicate_aux_or_tail(&a.surface) {
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

    if is_copula_pos && cores.is_empty() {
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
    } else if has_past && !has_neg_past && !has_naide_kudasai {
        post.push("了".to_owned());
    }

    (pre, cores.join(""), post, passive_agent)
}

fn is_predicate_aux_or_tail(s: &str) -> bool {
    matches!(
        s,
        "た" | "だ"
            | "です"
            | "ます"
            | "ました"
            | "ません"
            | "ませんでした"
            | "だった"
            | "でした"
            | "ない"
            | "なかった"
            | "ず"
            | "ぬ"
            | "ん"
            | "たい"
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
            | "ちゃう"
            | "ちゃった"
            | "じゃう"
            | "じゃった"
            | "くれる"
            | "くれた"
            | "もらう"
            | "もらった"
            | "ください"
            | "すぎる"
            | "すぎた"
            | "過ぎる"
            | "じゃない"
            | "ではない"
            | "ではありません"
            | "てい"
            | "てお"
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
        "全然" => "完全不",
        "あまり" => "不太",
        "もっと" => "更多",
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
        "時間" => "時間",
        // Core verbs
        "走る" => "跑",
        "歩く" => "走",
        "行く" => "去",
        "来る" => "來",
        "帰る" => "回去",
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
        _ => return None,
    })
}
