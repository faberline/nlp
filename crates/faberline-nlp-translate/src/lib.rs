//! Post-NLP offline deterministic and Seq2Seq translation (`nlp.translate.v1`) over
//! `faberline_nlp::doc::Document` (`nlp.doc.v1`), `faberline_nlp::lexicon`, and optional
//! `faberline_nlp_narrative::ChapterNarrative` (`nlp.narrative.v1`).

pub mod batch;
pub mod conditioning;
pub mod glossary;
pub mod ja_zh;
pub mod lock;
pub mod model;
pub mod opaque;
pub mod postprocess;
mod quote_source;
pub mod render;
pub mod schema;
pub mod variant;
mod verb_glossary;

use std::{collections::HashMap, path::Path};

use faberline_nlp::Span;
use faberline_nlp::doc::model::{Backend, ModelError};
use faberline_nlp::doc::{
    Document, Lang, Opaque, Options as DocOptions, Reason, Token, analyze_document,
};
#[cfg(feature = "narrative")]
use faberline_nlp_narrative::{ChapterNarrative, analyze_chapter_narrative};

use crate::conditioning::{
    build_sentence_conditioning, format_control_prefix, partition_clause_windows,
};
use crate::glossary::{CompiledGlossary, format_entity_for_target};
use crate::lock::{LockOrigin, SpanLockTable};
use crate::model::{Seq2SeqModel, VerifiedSeq2Seq, load_verified_seq2seq, verify_seq2seq};
use crate::opaque::{
    classify_opaque_policy, is_block_opaque, render_heading, unwrap_heading, unwrap_list_item,
};
use crate::postprocess::{
    apply_cjk_latin_spacing, cleanup_cjk_translation_spacing, fix_english_articles_and_case,
    fix_korean_particles, localize_quotation_marks,
};
pub use crate::render::{
    OutputFormat, render_bilingual_jsonl, render_bilingual_markdown, render_output,
};
pub use crate::schema::{
    ClauseAlignment, DEFAULT_TRANSLATE_MODEL_ID, GlossaryEntry, GlossaryHit, GlossaryKind,
    OpaquePolicy, OpaqueSpanRecord, PronounClass, SEQ2SEQ_MANIFEST_SCHEMA, Score,
    SentenceConditioning, SpeechRegister, TRANSLATE_BATCH_RECEIPT_SCHEMA,
    TRANSLATE_GLOSSARY_SCHEMA, TRANSLATE_SCHEMA, TargetVariant, TranslateManifestReceipt,
    TranslateOptions, TranslateUnknown, TranslateUnknownKind, TranslatedDocument,
    TranslatedParagraph, TranslatedSentence, TranslationGlossary, TranslationTrack, sha256_hex,
};
use crate::variant::{HintedJapaneseTerm, convert_fragment_variant, convert_sentence_variant};

/// Reusable loaded Seq2Seq translation model + tokenizer bundle for multi-document / batch runs.
pub struct LoadedTranslateModel {
    model: Box<dyn Seq2SeqModel>,
    #[cfg(any(feature = "onnx", feature = "coreml"))]
    tokenizer: crate::model::Seq2SeqTokenizer,
}

impl LoadedTranslateModel {
    pub fn load(models_dir: &Path, model_id: &str, backend: Backend) -> Result<Self, ModelError> {
        if !crate::model::backend_compiled_in(backend) {
            return Err(ModelError::NoBackend);
        }
        let verified = verify_seq2seq(models_dir, model_id)?;
        Self::from_verified(&verified, backend)
    }

    pub fn from_verified(v: &VerifiedSeq2Seq, backend: Backend) -> Result<Self, ModelError> {
        let model = load_verified_seq2seq(v, backend)?;
        #[cfg(any(feature = "onnx", feature = "coreml"))]
        {
            let tokenizer = crate::model::Seq2SeqTokenizer::from_verified(v)?;
            Ok(Self { model, tokenizer })
        }
        #[cfg(not(any(feature = "onnx", feature = "coreml")))]
        {
            let _ = model;
            Err(ModelError::NoBackend)
        }
    }

    pub fn info(&self) -> &crate::model::Seq2SeqModelInfo {
        self.model.info()
    }
}

/// Build recommended `faberline_nlp::doc::Options` for downstream translation.
pub fn doc_options_for_translate(
    options: &TranslateOptions,
    glossary: &TranslationGlossary,
) -> DocOptions {
    let (names, lexicon) = glossary.surface_terms();
    DocOptions {
        names,
        mask_editorial: options.mask_editorial,
        normalize_traditional: true,
        lexicon,
        enrich_morphology: true,
        first_party_zh_segmenter: true,
        cpu_syntax: true,
        ..DocOptions::default()
    }
}

/// High-level convenience entrypoint: runs `nlp.doc.v1` (+ optional `nlp.narrative.v1`) and
/// translates `text` offline using rules/lexicon (Track A) and optional `models_dir` (Track B).
pub fn analyze_and_translate(
    text: &str,
    options: &TranslateOptions,
    glossary: &TranslationGlossary,
    models_dir: Option<&Path>,
    backend: Backend,
) -> anyhow::Result<TranslatedDocument> {
    let doc_opts = doc_options_for_translate(options, glossary);
    let doc = if let Some(mdir) = models_dir {
        // Attempt syntax model enrichment if `<mdir>` also holds `nlp.models.v1` (`doc-syntax-v1`);
        // otherwise run deterministic `analyze_document`.
        faberline_nlp::doc::analyze_document_with_model(text, &doc_opts, mdir, backend)
            .unwrap_or_else(|_| analyze_document(text, &doc_opts))
    } else {
        analyze_document(text, &doc_opts)
    };

    #[cfg(feature = "narrative")]
    let narrative_opt = {
        let vocab = glossary.to_work_vocabulary();
        analyze_chapter_narrative(1, None, text, &doc, &vocab).ok()
    };

    let loaded_seq2seq = match models_dir {
        Some(mdir) => match LoadedTranslateModel::load(mdir, &options.model_id, backend) {
            Ok(m) => Some(m),
            Err(e) if e.into_reason().is_some() => None,
            Err(e) => return Err(anyhow::anyhow!("translate model error: {e}")),
        },
        None => None,
    };

    translate_document(
        text,
        &doc,
        #[cfg(feature = "narrative")]
        narrative_opt.as_ref(),
        options,
        glossary,
        loaded_seq2seq.as_ref(),
    )
}

/// Translate `text` given its pre-computed `nlp.doc.v1` [`Document`] (and optional
/// [`ChapterNarrative`]), enforcing SHA-256 provenance, `opaque` non-interference, glossary
/// locking, and exact bidirectional `Span` alignment.
pub fn translate_document(
    text: &str,
    doc: &Document,
    #[cfg(feature = "narrative")] narrative: Option<&ChapterNarrative>,
    options: &TranslateOptions,
    glossary: &TranslationGlossary,
    seq2seq: Option<&LoadedTranslateModel>,
) -> anyhow::Result<TranslatedDocument> {
    #[cfg(not(feature = "narrative"))]
    let narrative: Option<&()> = None;

    let chars: Vec<char> = text.chars().collect();
    let source_chars = chars.len();
    if doc.text_chars != source_chars {
        anyhow::bail!(
            "document text_chars ({}) does not match text scalar count ({source_chars})",
            doc.text_chars
        );
    }
    let source_sha256 = sha256_hex(text);
    if doc.text_sha256 != source_sha256 {
        anyhow::bail!(
            "document text_sha256 ({}) does not match source SHA-256 ({source_sha256})",
            doc.text_sha256
        );
    }
    #[cfg(feature = "narrative")]
    if let Some(nar) = narrative
        && (nar.text_chars != source_chars || nar.text_sha256 != source_sha256)
    {
        anyhow::bail!("narrative SHA-256 or scalar length does not match source text");
    }

    let compiled_glossary = CompiledGlossary::compile(glossary, options.target, |frag, tgt| {
        convert_fragment_variant(frag, tgt)
    });
    let verb_grammar = if options.target == TargetVariant::ZhHant && seq2seq.is_none() {
        verb_glossary::structural_document(text, &chars, doc, glossary)
    } else {
        None
    };

    let sentence_spans: Vec<Span> = doc.segmentation.sentences.iter().map(|s| s.span).collect();

    // Partition `doc.segmentation.opaque` into standalone block opaques vs sentence-internal opaques.
    let mut block_opaques: Vec<&Opaque> = Vec::new();
    let mut inline_opaques: Vec<&Opaque> = Vec::new();
    for op in &doc.segmentation.opaque {
        if is_block_opaque(op, &sentence_spans) {
            block_opaques.push(op);
        } else {
            inline_opaques.push(op);
        }
    }

    // Merge block opaques and sentences into one non-overlapping top-level timeline sorted by `start_char`.
    #[derive(Clone, Copy)]
    enum TopItem<'a> {
        BlockOpaque(&'a Opaque),
        Sentence(usize),
    }

    let mut items: Vec<(Span, TopItem<'_>)> =
        Vec::with_capacity(block_opaques.len() + doc.segmentation.sentences.len());
    for op in block_opaques {
        items.push((op.span, TopItem::BlockOpaque(op)));
    }
    for (si, s) in doc.segmentation.sentences.iter().enumerate() {
        items.push((s.span, TopItem::Sentence(si)));
    }
    items.sort_by_key(|(sp, _)| (sp.start_char, sp.end_char));

    // Pre-index tokens and entities by sentence for O(1) slice access.
    let sent_count = doc.segmentation.sentences.len();
    let mut tokens_by_sent: Vec<Vec<&Token>> = vec![Vec::new(); sent_count];
    for tok in &doc.tokens {
        if tok.sentence < sent_count {
            tokens_by_sent[tok.sentence].push(tok);
        }
    }

    let written_quotes = if options.target.is_zh() && seq2seq.is_none() {
        quote_source::plan_written_form_quotes(&chars, doc)
    } else {
        Vec::new()
    };
    let direct_quote_placements = if options.target == TargetVariant::ZhHant && seq2seq.is_none() {
        plan_cross_sentence_direct_quotes(
            &chars,
            doc,
            &tokens_by_sent,
            &inline_opaques,
            options,
            &compiled_glossary,
            &written_quotes,
        )
    } else {
        HashMap::new()
    };

    let mut target_text = String::with_capacity(text.len() + 32);
    let mut target_char_pos = 0usize;
    let mut cursor = 0usize;
    let mut skip_single_leading_newline = false;

    let mut out_sentences: Vec<TranslatedSentence> = Vec::with_capacity(sent_count);
    let mut japanese_term_blocks = 0;
    let mut out_clauses: Vec<ClauseAlignment> = Vec::new();
    let mut out_glossary_hits: Vec<GlossaryHit> = Vec::new();
    let mut out_opaque_records: Vec<OpaqueSpanRecord> =
        Vec::with_capacity(doc.segmentation.opaque.len());
    let mut out_unknown: Vec<TranslateUnknown> = Vec::new();
    let mut consumed_direct_reports = HashMap::new();
    let mut dates = ja_zh::DateContext::default();
    let mut honorifics = ja_zh::HonorificContext::default();
    let glossary_source = std::cell::OnceCell::new();

    for (span, item) in items {
        if span.start_char > cursor {
            let mut gap_start = cursor;
            if skip_single_leading_newline && chars.get(gap_start) == Some(&'\n') {
                gap_start += 1;
            }
            skip_single_leading_newline = false;
            if gap_start < span.start_char {
                for &ch in &chars[gap_start..span.start_char] {
                    target_text.push(ch);
                }
                target_char_pos += span.start_char - gap_start;
            }
        } else {
            skip_single_leading_newline = false;
        }

        match item {
            TopItem::BlockOpaque(op) => {
                let raw_slice: String =
                    chars[op.span.start_char..op.span.end_char].iter().collect();
                let policy = classify_opaque_policy(op, &raw_slice, options.omit_masked_editorial);

                match policy {
                    OpaquePolicy::OmitEditorial => {
                        out_opaque_records.push(OpaqueSpanRecord {
                            kind: op.kind,
                            policy,
                            source_span: op.span,
                            target_span: None,
                        });
                        skip_single_leading_newline = true;
                    }
                    OpaquePolicy::VerbatimPassthrough => {
                        let start = target_char_pos;
                        let len = op.span.end_char - op.span.start_char;
                        target_text.push_str(&raw_slice);
                        target_char_pos += len;
                        out_opaque_records.push(OpaqueSpanRecord {
                            kind: op.kind,
                            policy,
                            source_span: op.span,
                            target_span: Some(Span {
                                start_char: start,
                                end_char: target_char_pos,
                            }),
                        });
                    }
                    OpaquePolicy::StructuralUnwrap => {
                        let start = target_char_pos;
                        let block = translate_structural_block(
                            op,
                            &raw_slice,
                            &chars,
                            options,
                            &compiled_glossary,
                            seq2seq,
                        );
                        japanese_term_blocks += usize::from(block.used_japanese_term);
                        let r_chars = block.text.chars().count();
                        for mut h in block.hits {
                            h.target_span.start_char += start;
                            h.target_span.end_char += start;
                            out_glossary_hits.push(h);
                        }
                        out_unknown.extend(block.unknowns);
                        target_text.push_str(&block.text);
                        target_char_pos += r_chars;
                        out_opaque_records.push(OpaqueSpanRecord {
                            kind: op.kind,
                            policy,
                            source_span: op.span,
                            target_span: Some(Span {
                                start_char: start,
                                end_char: target_char_pos,
                            }),
                        });
                    }
                }
            }
            TopItem::Sentence(si) => {
                let sent = &doc.segmentation.sentences[si];
                let sent_inline_opaques = sentence_opaques(&inline_opaques, sent.span);

                let hinted_term = if sent.lang == Lang::Zh && seq2seq.is_none() {
                    HintedJapaneseTerm::lookup(
                        &chars,
                        sent.span,
                        options.target,
                        options.source_hint,
                    )
                } else {
                    None
                };
                let outcome = translate_single_sentence(
                    si,
                    &chars,
                    doc,
                    verb_grammar.as_ref(),
                    narrative,
                    &tokens_by_sent[si],
                    &sent_inline_opaques,
                    hinted_term,
                    options,
                    glossary,
                    &compiled_glossary,
                    seq2seq,
                    direct_quote_placements.get(&si),
                    &written_quotes,
                    &mut dates,
                    &mut honorifics,
                    &glossary_source,
                );

                let start = target_char_pos;
                let s_chars = outcome.target_text.chars().count();
                let end = start + s_chars;
                if s_chars == 0
                    && outcome.glossary_hits.is_empty()
                    && outcome.opaque_records.is_empty()
                    && outcome.unknown.is_empty()
                    && let Some(first) = direct_quote_placements
                        .get(&si)
                        .and_then(|quote| quote.detached_from)
                {
                    consumed_direct_reports.insert(first, si);
                }
                target_text.push_str(&outcome.target_text);
                target_char_pos = end;

                for mut h in outcome.glossary_hits {
                    h.target_span.start_char += start;
                    h.target_span.end_char += start;
                    out_glossary_hits.push(h);
                }
                for mut op_rec in outcome.opaque_records {
                    if let Some(ts) = &mut op_rec.target_span {
                        ts.start_char += start;
                        ts.end_char += start;
                    }
                    out_opaque_records.push(op_rec);
                }
                for mut ca in outcome.clause_alignments {
                    ca.target_span.start_char += start;
                    ca.target_span.end_char += start;
                    out_clauses.push(ca);
                }
                out_unknown.extend(outcome.unknown);

                out_sentences.push(TranslatedSentence {
                    sentence: si,
                    paragraph: sent.paragraph,
                    source_lang: if hinted_term.is_some() {
                        Lang::Ja
                    } else {
                        sent.lang
                    },
                    track: outcome.track,
                    source_span: sent.span,
                    target_span: Span {
                        start_char: start,
                        end_char: end,
                    },
                    target_text: outcome.target_text,
                    confidence: outcome.confidence,
                    conditioning: outcome.conditioning,
                });
            }
        }

        cursor = span.end_char;
    }

    if cursor < source_chars {
        let mut gap_start = cursor;
        if skip_single_leading_newline && chars.get(gap_start) == Some(&'\n') {
            gap_start += 1;
        }
        if gap_start < source_chars {
            for &ch in &chars[gap_start..source_chars] {
                target_text.push(ch);
            }
            target_char_pos += source_chars - gap_start;
        }
    }

    if let Some(sentence_ids) =
        pair_consumed_direct_reports(&mut out_sentences, &consumed_direct_reports, &target_text)
    {
        for clause in &mut out_clauses {
            clause.sentence = sentence_ids[clause.sentence];
        }
        for sentence in out_glossary_hits
            .iter_mut()
            .filter_map(|hit| hit.sentence.as_mut())
            .chain(
                out_unknown
                    .iter_mut()
                    .filter_map(|item| item.sentence.as_mut()),
            )
        {
            *sentence = sentence_ids[*sentence];
        }
    }

    // Build paragraph alignments from their constituent translated sentences.
    let mut out_paragraphs: Vec<TranslatedParagraph> =
        Vec::with_capacity(doc.segmentation.paragraphs.len());
    for (pi, para) in doc.segmentation.paragraphs.iter().enumerate() {
        let mut p_sents = out_sentences.iter().filter(|s| s.paragraph == pi);
        if let Some(first) = p_sents.next() {
            let last = p_sents.next_back().unwrap_or(first);
            out_paragraphs.push(TranslatedParagraph {
                paragraph: pi,
                source_span: para.span,
                target_span: Span {
                    start_char: first.target_span.start_char,
                    end_char: last.target_span.end_char,
                },
            });
        }
    }

    out_glossary_hits.sort_by_key(|h| (h.source_span.start_char, h.source_span.end_char));
    out_opaque_records.sort_by_key(|o| (o.source_span.start_char, o.source_span.end_char));
    out_unknown.sort_by_key(|u| (u.source_span.start_char, u.source_span.end_char, u.kind));

    let doc_lang = dominant_doc_lang(&out_sentences, japanese_term_blocks);
    let doc_track = summarize_doc_track(
        &out_sentences,
        options.target,
        doc_lang,
        japanese_term_blocks > 0,
    );
    let target_sha256 = sha256_hex(&target_text);

    debug_assert_eq!(target_char_pos, target_text.chars().count());

    Ok(TranslatedDocument {
        schema: TRANSLATE_SCHEMA.to_owned(),
        source_chars,
        source_sha256,
        target_chars: target_char_pos,
        target_sha256,
        source_lang: doc_lang,
        target_variant: options.target,
        track: doc_track,
        target_text,
        paragraphs: out_paragraphs,
        sentences: out_sentences,
        clause_alignments: out_clauses,
        glossary_hits: out_glossary_hits,
        opaque_spans: out_opaque_records,
        unknown: out_unknown,
    })
}

struct SentenceOutcome {
    target_text: String,
    track: TranslationTrack,
    confidence: Option<Score>,
    conditioning: SentenceConditioning,
    glossary_hits: Vec<GlossaryHit>,
    opaque_records: Vec<OpaqueSpanRecord>,
    clause_alignments: Vec<ClauseAlignment>,
    unknown: Vec<TranslateUnknown>,
}

// A consumed standalone report still belongs to the source of the translated
// quote. Group only that empty tail and its quote; the Japanese analysis is unchanged.
fn pair_consumed_direct_reports(
    sentences: &mut Vec<TranslatedSentence>,
    consumed: &HashMap<usize, usize>,
    target_text: &str,
) -> Option<Vec<usize>> {
    if consumed.is_empty()
        || !sentences
            .iter()
            .enumerate()
            .all(|(i, sentence)| sentence.sentence == i)
    {
        return None;
    }
    let target: Vec<char> = target_text.chars().collect();
    let mut ids = vec![0; sentences.len()];
    let mut aligned = Vec::with_capacity(sentences.len());
    let mut old = std::mem::take(sentences).into_iter();
    while let Some(mut sentence) = old.next() {
        let index = sentence.sentence;
        let new_id = aligned.len();
        sentence.sentence = new_id;
        ids[index] = new_id;
        if let Some(&last) = consumed.get(&index) {
            for following in old.by_ref().take(last - index) {
                ids[following.sentence] = new_id;
                sentence.source_span.end_char = following.source_span.end_char;
                sentence.target_span.end_char = following.target_span.end_char;
            }
            sentence.target_text = target
                [sentence.target_span.start_char..sentence.target_span.end_char]
                .iter()
                .collect();
        }
        aligned.push(sentence);
    }
    *sentences = aligned;
    Some(ids)
}

fn sentence_opaques<'a>(inline_opaques: &[&'a Opaque], sentence: Span) -> Vec<&'a Opaque> {
    inline_opaques
        .iter()
        .copied()
        .filter(|op| {
            op.span.start_char >= sentence.start_char && op.span.end_char <= sentence.end_char
        })
        .collect()
}

fn is_unsupported_sentence(doc: &Document, span: Span) -> bool {
    doc.unknown.iter().any(|u| {
        u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage && u.span == span
    })
}

fn plan_cross_sentence_direct_quotes(
    chars: &[char],
    doc: &Document,
    tokens_by_sent: &[Vec<&Token>],
    inline_opaques: &[&Opaque],
    options: &TranslateOptions,
    compiled_glossary: &CompiledGlossary,
    written_quotes: &[Vec<quote_source::WrittenForm>],
) -> HashMap<usize, ja_zh::DirectQuotePlacement> {
    let sentences = &doc.segmentation.sentences;
    let mut placements: HashMap<usize, ja_zh::DirectQuotePlacement> = HashMap::new();
    let mut opening: Option<(usize, ja_zh::DirectQuoteOpening)> = None;
    for (index, sent) in sentences.iter().enumerate() {
        let raw = &chars[sent.span.start_char..sent.span.end_char];
        if let Some(previous) = index.checked_sub(1).and_then(|i| sentences.get(i))
            && opening.is_some()
            && (previous.paragraph != sent.paragraph
                || previous.span.end_char > sent.span.start_char
                || !chars[previous.span.end_char..sent.span.start_char]
                    .iter()
                    .all(|c| c.is_whitespace()))
        {
            opening = None;
        }
        let has_open = raw.iter().any(|c| matches!(c, '「' | '『'));
        let has_close = raw.iter().any(|c| matches!(c, '」' | '』'));
        if opening.is_none() && !has_open {
            continue;
        }
        let raw_sent: String = raw.iter().collect();
        if !(sent.lang == Lang::Ja
            || (sent.lang == Lang::Mixed && ja_zh::has_kana_no_hangul(&raw_sent)))
            || is_unsupported_sentence(doc, sent.span)
        {
            opening = None;
            continue;
        }
        let follows_a_terminal_quote = opening
            .as_ref()
            .is_some_and(|(_, frame)| frame.close.is_none());
        if !has_open && !has_close && !follows_a_terminal_quote {
            continue;
        }

        let locks = build_sentence_locks(
            index,
            chars,
            doc,
            &sentence_opaques(inline_opaques, sent.span),
            &raw_sent,
            options,
            compiled_glossary,
            written_quotes.get(index).map_or(&[], Vec::as_slice),
        );
        let tokens = &tokens_by_sent[index];
        if let Some((first, mut frame)) = opening.take() {
            match ja_zh::direct_quote_end(index, chars, doc, tokens, &locks, &frame) {
                ja_zh::DirectQuoteEnd::Continues => {
                    opening = Some((first, frame));
                    continue;
                }
                ja_zh::DirectQuoteEnd::Closed => {
                    frame.close = None;
                    opening = Some((first, frame));
                    continue;
                }
                ja_zh::DirectQuoteEnd::Reports(report, prefix) => {
                    let detached = frame.close.is_none();
                    placements.entry(first).or_default().opening = Some(prefix);
                    let ending = placements.entry(index).or_default();
                    ending.report = Some(report);
                    if detached {
                        ending.detached_from = Some(first);
                    }
                }
                ja_zh::DirectQuoteEnd::Other => {}
            }
        }
        if has_open
            && let Some(frame) = ja_zh::direct_quote_opening(index, chars, doc, tokens, &locks)
        {
            opening = Some((index, frame));
        }
    }
    placements
}

#[allow(clippy::too_many_arguments)]
fn build_sentence_locks(
    sentence_idx: usize,
    chars: &[char],
    doc: &Document,
    sent_inline_opaques: &[&Opaque],
    raw_sent: &str,
    options: &TranslateOptions,
    compiled_glossary: &CompiledGlossary,
    written_quotes: &[quote_source::WrittenForm],
) -> SpanLockTable {
    let sent_span = doc.segmentation.sentences[sentence_idx].span;
    // Build `SpanLockTable` over inline opaques (priority 0), glossary hits (priority 1), and
    // optional localized entities (priority 2).
    let mut lock_candidates: Vec<(u8, Span, String, LockOrigin, bool)> = Vec::new();
    let mut blocked_for_glossary: Vec<Span> = Vec::with_capacity(sent_inline_opaques.len());

    for op in sent_inline_opaques {
        let slice: String = chars[op.span.start_char..op.span.end_char].iter().collect();
        blocked_for_glossary.push(op.span);
        lock_candidates.push((0, op.span, slice, LockOrigin::InlineOpaque(op.kind), true));
    }

    // Also protect `EntityKind::Email` spans verbatim.
    for ent in &doc.entities {
        if ent.sentence == sentence_idx && ent.kind == faberline_nlp::doc::EntityKind::Email {
            let slice: String = chars[ent.span.start_char..ent.span.end_char]
                .iter()
                .collect();
            blocked_for_glossary.push(ent.span);
            lock_candidates.push((0, ent.span, slice, LockOrigin::Entity, true));
        }
    }

    for &quote_source::WrittenForm { span, .. } in written_quotes {
        if blocked_for_glossary
            .iter()
            .any(|blocked| span.start_char < blocked.end_char && blocked.start_char < span.end_char)
        {
            continue;
        }
        let source = chars[span.start_char..span.end_char].iter().collect();
        blocked_for_glossary.push(span);
        lock_candidates.push((0, span, source, LockOrigin::QuotedSource, true));
    }

    for gm in
        compiled_glossary.find_matches(raw_sent, chars, sent_span.start_char, &blocked_for_glossary)
    {
        lock_candidates.push((
            1,
            gm.span,
            gm.target_text,
            LockOrigin::Glossary {
                canonical: gm.canonical,
                kind: gm.kind,
            },
            true,
        ));
    }

    if options.localize_entities {
        for ent in &doc.entities {
            if ent.sentence == sentence_idx
                && let Some(formatted) = format_entity_for_target(ent, options.target)
            {
                lock_candidates.push((2, ent.span, formatted, LockOrigin::Entity, false));
            }
        }
    }

    SpanLockTable::build(lock_candidates)
}

#[allow(clippy::too_many_arguments)]
fn translate_single_sentence(
    sentence_idx: usize,
    chars: &[char],
    doc: &Document,
    verb_grammar: Option<&Document>,
    #[cfg(feature = "narrative")] narrative: Option<&ChapterNarrative>,
    #[cfg(not(feature = "narrative"))] narrative: Option<&()>,
    sent_tokens: &[&Token],
    sent_inline_opaques: &[&Opaque],
    hinted_term: Option<HintedJapaneseTerm>,
    options: &TranslateOptions,
    glossary: &TranslationGlossary,
    compiled_glossary: &CompiledGlossary,
    seq2seq: Option<&LoadedTranslateModel>,
    direct_quote: Option<&ja_zh::DirectQuotePlacement>,
    written_quotes_by_sentence: &[Vec<quote_source::WrittenForm>],
    dates: &mut ja_zh::DateContext,
    honorifics: &mut ja_zh::HonorificContext,
    glossary_source: &std::cell::OnceCell<Option<Document>>,
) -> SentenceOutcome {
    let sent = &doc.segmentation.sentences[sentence_idx];
    let sent_span = sent.span;
    let written_quotes: &[quote_source::WrittenForm] = written_quotes_by_sentence
        .get(sentence_idx)
        .map_or(&[], Vec::as_slice);
    let raw_sent: String = chars[sent_span.start_char..sent_span.end_char]
        .iter()
        .collect();

    // Check if Layer 1 flagged this sentence as `unsupported_language` (e.g. Cyrillic/Arabic).
    if is_unsupported_sentence(doc, sent_span) {
        let opaque_records = sent_inline_opaques
            .iter()
            .map(|op| OpaqueSpanRecord {
                kind: op.kind,
                policy: OpaquePolicy::VerbatimPassthrough,
                source_span: op.span,
                target_span: Some(Span {
                    start_char: op.span.start_char - sent_span.start_char,
                    end_char: op.span.end_char - sent_span.start_char,
                }),
            })
            .collect();
        return SentenceOutcome {
            target_text: raw_sent,
            track: TranslationTrack::Passthrough,
            confidence: None,
            conditioning: SentenceConditioning::default(),
            glossary_hits: Vec::new(),
            opaque_records,
            clause_alignments: Vec::new(),
            unknown: vec![TranslateUnknown {
                kind: TranslateUnknownKind::UnsupportedLanguage,
                source_span: sent_span,
                sentence: Some(sentence_idx),
                detail: Some("unsupported script in Layer 1 segmentation".to_owned()),
            }],
        };
    }

    let is_ja_to_zh = seq2seq.is_none()
        && options.target.is_zh()
        && (sent.lang == Lang::Ja
            || (sent.lang == Lang::Mixed && ja_zh::has_kana_no_hangul(&raw_sent)));
    let lock_table = build_sentence_locks(
        sentence_idx,
        chars,
        doc,
        sent_inline_opaques,
        &raw_sent,
        options,
        compiled_glossary,
        written_quotes,
    );

    // Determine routing track for this sentence.
    let same_zh = sent.lang == Lang::Zh && options.target.is_zh();
    let same_en = sent.lang == Lang::En && options.target.is_en();
    let same_ja = sent.lang == Lang::Ja && options.target == TargetVariant::Ja;
    let same_ko = sent.lang == Lang::Ko && options.target == TargetVariant::Ko;

    if sent.lang == Lang::Und || same_ja || same_ko {
        let masked = lock_table.mask_span(chars, sent_span);
        let report = lock_table.restore(&masked);
        return finalize_sentence_outcome(
            sentence_idx,
            chars,
            doc,
            sent_span,
            report,
            TranslationTrack::Passthrough,
            Some(Score(1000)),
            SentenceConditioning::default(),
            options,
            Vec::new(),
            Vec::new(),
        );
    }

    if same_zh || same_en {
        let converted = convert_sentence_with_locks(
            chars,
            sent_span,
            sent_tokens,
            &lock_table,
            options,
            hinted_term,
        );
        let report = lock_table.restore(&converted);
        let clause_alignments = build_same_family_clause_alignments(
            sentence_idx,
            chars,
            doc,
            sent_tokens,
            &lock_table,
            options,
            hinted_term,
        );
        let mut outcome = finalize_sentence_outcome(
            sentence_idx,
            chars,
            doc,
            sent_span,
            report,
            TranslationTrack::VariantRule,
            Some(Score(1000)),
            SentenceConditioning::default(),
            options,
            Vec::new(),
            Vec::new(),
        );
        outcome.clause_alignments = clause_alignments;
        return outcome;
    }

    // Cross-language track (Track B: NLP-Conditioned Seq2Seq, or deterministic Ja->Zh when no
    // neural model is provided).
    let mut conditioning = build_sentence_conditioning(
        sentence_idx,
        sent_span,
        doc,
        narrative,
        glossary,
        options.target,
    );

    if is_ja_to_zh {
        let prior_letter_distribution = sentence_idx.checked_sub(1).and_then(|idx| {
            let prior = &doc.segmentation.sentences[idx];
            if !raw_sent.contains("ずつ")
                || !sent_tokens
                    .iter()
                    .any(|token| token.lemma.as_deref() == Some("書く"))
                || prior.paragraph != sent.paragraph
                || prior.lang != Lang::Ja
                || prior.span.end_char > sent_span.start_char
                || !chars[prior.span.end_char..sent_span.start_char]
                    .iter()
                    .all(|ch| ch.is_whitespace())
                || doc.segmentation.opaque.iter().any(|opaque| {
                    opaque.span.start_char < prior.span.end_char
                        && prior.span.start_char < opaque.span.end_char
                })
            {
                return None;
            }
            let raw_prior: String = chars[prior.span.start_char..prior.span.end_char]
                .iter()
                .collect();
            let locks = build_sentence_locks(
                idx,
                chars,
                doc,
                &[],
                &raw_prior,
                options,
                compiled_glossary,
                written_quotes_by_sentence
                    .get(idx)
                    .map_or(&[], Vec::as_slice),
            );
            ja_zh::prior_letter_distribution(idx, chars, doc, &locks)
        });
        let repaired = if options.target == TargetVariant::ZhHant
            && direct_quote.is_none()
            && written_quotes.is_empty()
            && ja_zh::may_restore_glossary_syntax(&raw_sent, chars, &lock_table)
        {
            let source = glossary_source.get_or_init(|| {
                let text: String = chars.iter().collect();
                let source = analyze_document(
                    &text,
                    &DocOptions {
                        names: Vec::new(),
                        lexicon: Vec::new(),
                        enrich_morphology: true,
                        cpu_syntax: true,
                        ..doc.options.clone()
                    },
                );
                (doc.segmentation.sentences == source.segmentation.sentences
                    && doc.segmentation.clauses == source.segmentation.clauses)
                    .then_some(source)
            });
            source.as_ref().and_then(|source| {
                let first = source
                    .tokens
                    .partition_point(|token| token.sentence < sentence_idx);
                let last = source
                    .tokens
                    .partition_point(|token| token.sentence <= sentence_idx);
                let tokens = source.tokens[first..last].iter().collect::<Vec<_>>();
                ja_zh::glossary_render_locks(sentence_idx, source, &tokens, chars, &lock_table)
                    .map(|(locks, glossary_syntax)| (source, tokens, locks, glossary_syntax))
            })
        } else {
            None
        };
        let default_syntax = ja_zh::GlossarySyntax::default();
        let (render_doc, render_tokens, render_locks, glossary_syntax) = match &repaired {
            Some((source, tokens, locks, glossary_syntax)) => {
                (*source, tokens.as_slice(), locks, glossary_syntax)
            }
            None => (doc, sent_tokens, &lock_table, &default_syntax),
        };
        let (masked_zh, mut clause_alignments) = ja_zh::translate_ja_sentence(
            sentence_idx,
            chars,
            render_doc,
            if repaired.is_some() {
                None
            } else {
                verb_grammar
            },
            render_tokens,
            render_locks,
            glossary_syntax,
            &conditioning,
            options.target,
            direct_quote,
            written_quotes
                .first()
                .filter(|quote| quote.continuation)
                .map(|quote| quote.span),
            dates,
            prior_letter_distribution,
            honorifics,
        );
        let mut clause_spans: Vec<Span> = clause_alignments
            .iter()
            .map(|clause| clause.target_span)
            .collect();
        let report = lock_table.restore_with_spans(&masked_zh, &mut clause_spans);
        for (clause, span) in clause_alignments.iter_mut().zip(clause_spans) {
            clause.target_span = span;
        }
        return finalize_sentence_outcome(
            sentence_idx,
            chars,
            doc,
            sent_span,
            report,
            TranslationTrack::VariantRule,
            Some(Score(900)),
            conditioning,
            options,
            std::mem::take(&mut honorifics.unknown),
            clause_alignments,
        );
    }

    if let Some(loaded) = seq2seq {
        let max_win_chars = (loaded.info().src_max_len / 2).max(32);
        let windows = partition_clause_windows(
            chars,
            sentence_idx,
            sent_span,
            &doc.segmentation.clauses,
            max_win_chars,
        );
        conditioning.window_count = windows.len();
        let ctrl_prefix = format_control_prefix(options.target, &conditioning);

        let mut window_outputs: Vec<(ClauseAlignmentPending, String)> =
            Vec::with_capacity(windows.len());
        let mut conf_acc = 0u32;
        let mut extra_unknown = Vec::new();

        for win in &windows {
            let masked_win = lock_table.mask_span(chars, win.span);
            let model_input = format!("{ctrl_prefix}{masked_win}");
            match run_seq2seq_window(loaded, &model_input, options.beam_size) {
                Ok((decoded_str, conf)) => {
                    conf_acc += u32::from(conf.milli());
                    window_outputs.push((
                        ClauseAlignmentPending {
                            clause_indices: win.clause_indices.clone(),
                            source_span: win.span,
                            sep: win.trailing_sep.clone(),
                        },
                        decoded_str,
                    ));
                }
                Err(err) => {
                    extra_unknown.push(TranslateUnknown {
                        kind: TranslateUnknownKind::LowConfidence,
                        source_span: win.span,
                        sentence: Some(sentence_idx),
                        detail: Some(err.to_string()),
                    });
                    window_outputs.push((
                        ClauseAlignmentPending {
                            clause_indices: win.clause_indices.clone(),
                            source_span: win.span,
                            sep: win.trailing_sep.clone(),
                        },
                        masked_win,
                    ));
                }
            }
        }

        let avg_conf = if windows.is_empty() {
            Score(1000)
        } else {
            Score((conf_acc / (windows.len() as u32)) as u16)
        };
        if avg_conf.milli() < 350 {
            extra_unknown.push(TranslateUnknown {
                kind: TranslateUnknownKind::LowConfidence,
                source_span: sent_span,
                sentence: Some(sentence_idx),
                detail: None,
            });
        }

        let mut combined_decoded = String::new();
        for (idx, (meta, w_out)) in window_outputs.iter().enumerate() {
            combined_decoded.push_str(w_out.trim());
            if idx + 1 < window_outputs.len() {
                if !meta.sep.is_empty() {
                    combined_decoded.push_str(&localize_window_sep(&meta.sep, options.target));
                } else if options.target.is_en() || options.target == TargetVariant::Ko {
                    combined_decoded.push(' ');
                }
            }
        }

        let report = lock_table.restore(&combined_decoded);
        return finalize_sentence_outcome(
            sentence_idx,
            chars,
            doc,
            sent_span,
            report,
            TranslationTrack::NeuralSeq2Seq,
            Some(avg_conf),
            conditioning,
            options,
            extra_unknown,
            Vec::new(),
        );
    }

    // No neural model provided (`seq2seq.is_none()`): preserve all locked opaques & glossary
    // terms, apply deterministic variant conversion on matching scripts in `Mixed` text, and
    // emit `TranslateUnknownKind::NoModel`.
    let fallback_masked =
        if sent.lang == Lang::Mixed && (options.target.is_zh() || options.target.is_en()) {
            convert_sentence_with_locks(chars, sent_span, sent_tokens, &lock_table, options, None)
        } else {
            lock_table.mask_span(chars, sent_span)
        };
    let report = lock_table.restore(&fallback_masked);
    finalize_sentence_outcome(
        sentence_idx,
        chars,
        doc,
        sent_span,
        report,
        TranslationTrack::Passthrough,
        None,
        conditioning,
        options,
        vec![TranslateUnknown {
            kind: TranslateUnknownKind::NoModel,
            source_span: sent_span,
            sentence: Some(sentence_idx),
            detail: None,
        }],
        Vec::new(),
    )
}

struct ClauseAlignmentPending {
    #[allow(dead_code)]
    clause_indices: Vec<usize>,
    #[allow(dead_code)]
    source_span: Span,
    sep: String,
}

#[allow(clippy::too_many_arguments)]
fn finalize_sentence_outcome(
    sentence_idx: usize,
    chars: &[char],
    _doc: &Document,
    sent_span: Span,
    report: crate::lock::RestoreReport,
    track: TranslationTrack,
    confidence: Option<Score>,
    conditioning: SentenceConditioning,
    options: &TranslateOptions,
    mut unknown: Vec<TranslateUnknown>,
    mut clause_alignments: Vec<ClauseAlignment>,
) -> SentenceOutcome {
    for (kind, sp, detail) in report.issues {
        unknown.push(TranslateUnknown {
            kind,
            source_span: sp,
            sentence: Some(sentence_idx),
            detail: Some(detail),
        });
    }

    // Track every restored span (`0..k`) through morphological and spacing shifts!
    let mut tracked_spans: Vec<Span> = report.restored.iter().map(|r| r.target_span).collect();
    let protected_indices: Vec<usize> = (0..report.restored.len()).collect();
    tracked_spans.extend(clause_alignments.iter().map(|clause| clause.target_span));

    let mut text = report.text;

    if options.target == TargetVariant::Ko {
        let prot: Vec<Span> = protected_indices
            .iter()
            .map(|&i| tracked_spans[i])
            .collect();
        text = fix_korean_particles(&text, &prot, &mut tracked_spans);
    } else if options.target.is_en() {
        let prot: Vec<Span> = protected_indices
            .iter()
            .map(|&i| tracked_spans[i])
            .collect();
        let src_starts_upper_or_cjk = chars
            .get(sent_span.start_char)
            .is_some_and(|c| c.is_uppercase() || !c.is_ascii());
        let cap_init = track == TranslationTrack::NeuralSeq2Seq && src_starts_upper_or_cjk;
        text = fix_english_articles_and_case(&text, &prot, &mut tracked_spans, cap_init);
    }

    if options.localize_quotes {
        let verbatim_prot: Vec<Span> = report
            .restored
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match r.origin {
                LockOrigin::InlineOpaque(_) | LockOrigin::QuotedSource => Some(tracked_spans[i]),
                _ => None,
            })
            .collect();
        text = localize_quotation_marks(&text, options.target, &verbatim_prot);
    }

    if options.target.is_zh() {
        let verbatim_prot: Vec<Span> = report
            .restored
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match r.origin {
                LockOrigin::InlineOpaque(_) | LockOrigin::QuotedSource => Some(tracked_spans[i]),
                _ => None,
            })
            .collect();
        text = cleanup_cjk_translation_spacing(&text, &verbatim_prot, &mut tracked_spans);
    }

    if options.cjk_latin_spacing {
        let verbatim_prot: Vec<Span> = report
            .restored
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match r.origin {
                LockOrigin::InlineOpaque(_) | LockOrigin::QuotedSource => Some(tracked_spans[i]),
                _ => None,
            })
            .collect();
        text = apply_cjk_latin_spacing(&text, &verbatim_prot, &mut tracked_spans);
    }

    for (clause, span) in clause_alignments
        .iter_mut()
        .zip(&tracked_spans[report.restored.len()..])
    {
        clause.target_span = *span;
    }

    let mut glossary_hits = Vec::new();
    let mut opaque_records = Vec::new();
    for (r, final_target_span) in report.restored.into_iter().zip(tracked_spans) {
        match r.origin {
            LockOrigin::InlineOpaque(kind) => {
                opaque_records.push(OpaqueSpanRecord {
                    kind,
                    policy: OpaquePolicy::VerbatimPassthrough,
                    source_span: r.source_span,
                    target_span: Some(final_target_span),
                });
            }
            LockOrigin::Glossary { canonical, kind } => {
                glossary_hits.push(GlossaryHit {
                    canonical,
                    kind,
                    sentence: Some(sentence_idx),
                    source_span: r.source_span,
                    target_span: final_target_span,
                });
            }
            LockOrigin::Entity | LockOrigin::QuotedSource => {}
        }
    }

    SentenceOutcome {
        target_text: text,
        track,
        confidence,
        conditioning,
        glossary_hits,
        opaque_records,
        clause_alignments,
        unknown,
    }
}

fn convert_sentence_with_locks(
    chars: &[char],
    sent_span: Span,
    sent_tokens: &[&Token],
    lock_table: &SpanLockTable,
    options: &TranslateOptions,
    hinted_term: Option<HintedJapaneseTerm>,
) -> String {
    let convert = |span, tokens: &[&Token]| {
        hinted_term
            .and_then(|term| term.render(chars, span))
            .unwrap_or_else(|| {
                convert_sentence_variant(chars, span, tokens, options.target, options.source_hint)
            })
    };
    if lock_table.is_empty() {
        return convert(sent_span, sent_tokens);
    }

    let mut out = String::new();
    let mut cursor = sent_span.start_char;
    for lock in &lock_table.locks {
        if lock.source_span.end_char <= sent_span.start_char
            || lock.source_span.start_char >= sent_span.end_char
        {
            continue;
        }
        let s = lock.source_span.start_char.max(cursor);
        let e = lock.source_span.end_char.min(sent_span.end_char);
        if s > cursor {
            let sub_span = Span {
                start_char: cursor,
                end_char: s,
            };
            let sub_tokens: Vec<&Token> = sent_tokens
                .iter()
                .copied()
                .filter(|t| t.span.start_char >= cursor && t.span.end_char <= s)
                .collect();
            out.push_str(&convert(sub_span, &sub_tokens));
        }
        out.push_str(&lock.placeholder);
        cursor = e;
    }
    if cursor < sent_span.end_char {
        let sub_span = Span {
            start_char: cursor,
            end_char: sent_span.end_char,
        };
        let sub_tokens: Vec<&Token> = sent_tokens
            .iter()
            .copied()
            .filter(|t| t.span.start_char >= cursor && t.span.end_char <= sent_span.end_char)
            .collect();
        out.push_str(&convert(sub_span, &sub_tokens));
    }
    out
}

fn build_same_family_clause_alignments(
    sentence_idx: usize,
    chars: &[char],
    doc: &Document,
    sent_tokens: &[&Token],
    lock_table: &SpanLockTable,
    options: &TranslateOptions,
    hinted_term: Option<HintedJapaneseTerm>,
) -> Vec<ClauseAlignment> {
    let sent_span = doc.segmentation.sentences[sentence_idx].span;
    let clauses: Vec<(usize, &faberline_nlp::doc::Clause)> = doc
        .segmentation
        .clauses
        .iter()
        .enumerate()
        .filter(|(_, c)| c.sentence == sentence_idx)
        .collect();
    if clauses.is_empty() {
        return Vec::new();
    }

    let mut alignments = Vec::with_capacity(clauses.len());
    let mut t_cursor = 0usize;
    let mut s_cursor = sent_span.start_char;

    for (ci, clause) in clauses {
        if clause.span.start_char > s_cursor {
            t_cursor += clause.span.start_char - s_cursor;
        }
        let c_tokens: Vec<&Token> = sent_tokens
            .iter()
            .copied()
            .filter(|t| {
                t.span.start_char >= clause.span.start_char
                    && t.span.end_char <= clause.span.end_char
            })
            .collect();
        let masked_c = convert_sentence_with_locks(
            chars,
            clause.span,
            &c_tokens,
            lock_table,
            options,
            hinted_term,
        );
        let restored_c = restore_existing_only(lock_table, &masked_c);
        let mut final_c = restored_c;
        if options.localize_quotes {
            final_c = localize_quotation_marks(&final_c, options.target, &[]);
        }
        let len = final_c.chars().count();
        alignments.push(ClauseAlignment {
            clause: ci,
            sentence: sentence_idx,
            source_span: clause.span,
            target_span: Span {
                start_char: t_cursor,
                end_char: t_cursor + len,
            },
        });
        t_cursor += len;
        s_cursor = clause.span.end_char;
    }
    alignments
}

fn restore_existing_only(lock_table: &SpanLockTable, text: &str) -> String {
    let mut out = text.to_owned();
    for lock in &lock_table.locks {
        if out.contains(&lock.placeholder) {
            out = out.replace(&lock.placeholder, &lock.replacement);
        }
    }
    out
}

#[derive(Default)]
struct TranslatedBlock {
    text: String,
    hits: Vec<GlossaryHit>,
    unknowns: Vec<TranslateUnknown>,
    used_japanese_term: bool,
}

fn translate_structural_block(
    op: &Opaque,
    raw_slice: &str,
    chars: &[char],
    options: &TranslateOptions,
    compiled_glossary: &CompiledGlossary,
    seq2seq: Option<&LoadedTranslateModel>,
) -> TranslatedBlock {
    match op.kind {
        faberline_nlp::doc::OpaqueKind::Heading => {
            let unwrapped = unwrap_heading(raw_slice);
            let body = unwrapped.body();
            if body.is_empty() {
                return TranslatedBlock {
                    text: render_heading(&unwrapped, "", options.target),
                    ..Default::default()
                };
            }
            let body_offset = raw_slice
                .trim_end()
                .strip_suffix(body)
                .expect("unwrapped heading body is its source suffix")
                .chars()
                .count();
            let body_span = Span {
                start_char: op.span.start_char + body_offset,
                end_char: op.span.start_char + body_offset + body.chars().count(),
            };
            let mut fragment = translate_inline_fragment(
                body,
                body_span,
                chars,
                options,
                compiled_glossary,
                seq2seq,
            );
            let full = render_heading(&unwrapped, &fragment.text, options.target);
            let prefix_chars = if !fragment.text.trim().is_empty() {
                full.rfind(fragment.text.trim())
                    .map(|b| full[..b].chars().count())
                    .unwrap_or(0)
            } else {
                0
            };
            for h in &mut fragment.hits {
                h.target_span.start_char += prefix_chars;
                h.target_span.end_char += prefix_chars;
            }
            fragment.text = full;
            fragment
        }
        faberline_nlp::doc::OpaqueKind::ListItem => {
            let (prefix, body) = unwrap_list_item(raw_slice);
            if body.trim().is_empty() {
                return TranslatedBlock {
                    text: raw_slice.to_owned(),
                    ..Default::default()
                };
            }
            let prefix_chars = prefix.chars().count();
            let body_span = Span {
                start_char: op.span.start_char + prefix_chars,
                end_char: op.span.end_char,
            };
            let mut fragment = translate_inline_fragment(
                body,
                body_span,
                chars,
                options,
                compiled_glossary,
                seq2seq,
            );
            for h in &mut fragment.hits {
                h.target_span.start_char += prefix_chars;
                h.target_span.end_char += prefix_chars;
            }
            fragment.text.insert_str(0, prefix);
            fragment
        }
        _ => TranslatedBlock {
            text: raw_slice.to_owned(),
            ..Default::default()
        },
    }
}

fn translate_inline_fragment(
    body: &str,
    body_span: Span,
    chars: &[char],
    options: &TranslateOptions,
    compiled_glossary: &CompiledGlossary,
    seq2seq: Option<&LoadedTranslateModel>,
) -> TranslatedBlock {
    // Detect inline opaques (`code`, `formula`, `url`, `emoji`, `citation`) and emails inside
    // unwrapped heading / list-item bodies so backticks, `$...$`, and URLs remain untouched.
    let sub_doc = analyze_document(body, &DocOptions::default());
    let mut candidates: Vec<(u8, Span, String, LockOrigin, bool)> = Vec::new();
    let mut blocked: Vec<Span> = Vec::new();
    for op in &sub_doc.segmentation.opaque {
        let abs = Span {
            start_char: body_span.start_char + op.span.start_char,
            end_char: body_span.start_char + op.span.end_char,
        };
        let slice: String = chars[abs.start_char..abs.end_char].iter().collect();
        blocked.push(abs);
        candidates.push((0, abs, slice, LockOrigin::InlineOpaque(op.kind), true));
    }
    for ent in &sub_doc.entities {
        if ent.kind == faberline_nlp::doc::EntityKind::Email {
            let abs = Span {
                start_char: body_span.start_char + ent.span.start_char,
                end_char: body_span.start_char + ent.span.end_char,
            };
            let slice: String = chars[abs.start_char..abs.end_char].iter().collect();
            blocked.push(abs);
            candidates.push((0, abs, slice, LockOrigin::Entity, true));
        }
    }

    let matches = compiled_glossary.find_matches(body, chars, body_span.start_char, &blocked);
    for gm in matches {
        candidates.push((
            1,
            gm.span,
            gm.target_text,
            LockOrigin::Glossary {
                canonical: gm.canonical,
                kind: gm.kind,
            },
            true,
        ));
    }
    let lock_table = SpanLockTable::build(candidates);
    let masked = lock_table.mask_span(chars, body_span);

    let has_han = body.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c));
    let ascii_only = body.is_ascii();
    let mut unks = Vec::new();

    if seq2seq.is_none()
        && lock_table.is_empty()
        && let Some(term) =
            HintedJapaneseTerm::lookup(chars, body_span, options.target, options.source_hint)
        && let Some(rendered) = term.render(chars, body_span)
    {
        return TranslatedBlock {
            text: rendered,
            used_japanese_term: true,
            ..Default::default()
        };
    }
    let converted =
        if seq2seq.is_none() && options.target.is_zh() && ja_zh::has_kana_no_hangul(body) {
            let (translated, fragment_unknown) =
                ja_zh::translate_ja_fragment(&masked, options.target);
            for mut entry in fragment_unknown {
                entry.source_span = fragment_source_span(entry.source_span, body_span, &lock_table);
                entry.sentence = None;
                unks.push(entry);
            }
            translated
        } else if (has_han && options.target.is_zh()) || (ascii_only && options.target.is_en()) {
            convert_fragment_preserving_placeholders(&masked, options.target)
        } else if let Some(loaded) = seq2seq {
            let inp = format!("{} {masked}", options.target.control_tag());
            match run_seq2seq_window(loaded, &inp, options.beam_size) {
                Ok((dec, _)) => dec,
                Err(e) => {
                    unks.push(TranslateUnknown {
                        kind: TranslateUnknownKind::LowConfidence,
                        source_span: body_span,
                        sentence: None,
                        detail: Some(e.to_string()),
                    });
                    masked
                }
            }
        } else {
            unks.push(TranslateUnknown {
                kind: TranslateUnknownKind::NoModel,
                source_span: body_span,
                sentence: None,
                detail: None,
            });
            convert_fragment_preserving_placeholders(&masked, options.target)
        };

    let report = lock_table.restore(&converted);
    let hits = report
        .restored
        .into_iter()
        .filter_map(|r| match r.origin {
            LockOrigin::Glossary { canonical, kind } => Some(GlossaryHit {
                canonical,
                kind,
                sentence: None,
                source_span: r.source_span,
                target_span: r.target_span,
            }),
            _ => None,
        })
        .collect();
    TranslatedBlock {
        text: report.text,
        hits,
        unknowns: unks,
        used_japanese_term: false,
    }
}

/// Map a masked fragment's scalar positions back to the original document.
fn fragment_source_span(masked: Span, body: Span, locks: &SpanLockTable) -> Span {
    let project = |position: usize, end: bool| {
        let mut source_cursor = body.start_char;
        let mut masked_cursor = 0;
        for lock in &locks.locks {
            let marker_start = masked_cursor + (lock.source_span.start_char - source_cursor);
            if position <= marker_start {
                break;
            }
            let marker_end = marker_start + lock.placeholder.chars().count();
            if position < marker_end {
                return if end {
                    lock.source_span.end_char
                } else {
                    lock.source_span.start_char
                };
            }
            masked_cursor = marker_end;
            source_cursor = lock.source_span.end_char;
        }
        source_cursor + position - masked_cursor
    };
    Span {
        start_char: project(masked.start_char, false),
        end_char: project(masked.end_char, true),
    }
}

fn convert_fragment_preserving_placeholders(masked: &str, target: TargetVariant) -> String {
    let mut out = String::with_capacity(masked.len());
    let mut rest = masked;
    while let Some(open) = rest.find("[[PH") {
        out.push_str(&convert_fragment_variant(&rest[..open], target));
        let after = &rest[open + 4..];
        if let Some(close) = after.find("]]") {
            out.push_str(&rest[open..open + 4 + close + 2]);
            rest = &after[close + 2..];
        } else {
            out.push_str("[[PH");
            rest = after;
        }
    }
    out.push_str(&convert_fragment_variant(rest, target));
    out
}

fn run_seq2seq_window(
    loaded: &LoadedTranslateModel,
    input_text: &str,
    beam_size: usize,
) -> Result<(String, Score), ModelError> {
    #[cfg(any(feature = "onnx", feature = "coreml"))]
    {
        let mut ids = loaded.tokenizer.encode_ids(input_text)?;
        if ids.len() > loaded.info().src_max_len {
            ids.truncate(loaded.info().src_max_len);
        }
        let decoded = crate::model::decode::decode_sequence(&*loaded.model, &ids, beam_size, None)?;
        let text = loaded.tokenizer.decode_ids(&decoded.token_ids)?;
        Ok((text, decoded.confidence))
    }
    #[cfg(not(any(feature = "onnx", feature = "coreml")))]
    {
        let _ = (loaded, input_text, beam_size);
        Err(ModelError::NoBackend)
    }
}

fn localize_window_sep(sep: &str, target: TargetVariant) -> String {
    if target.is_en() {
        sep.replace('，', ", ")
            .replace('；', "; ")
            .replace('：', ": ")
    } else if target.is_zh() || target == TargetVariant::Ja {
        sep.replace(", ", "，")
            .replace("; ", "；")
            .replace(": ", "：")
    } else {
        sep.replace('，', ", ").replace('；', "; ")
    }
}

fn dominant_doc_lang(sentences: &[TranslatedSentence], japanese_term_blocks: usize) -> Lang {
    let mut counts = [0usize; 6];
    counts[2] = japanese_term_blocks;
    for s in sentences {
        match s.source_lang {
            Lang::Zh => counts[0] += 1,
            Lang::En => counts[1] += 1,
            Lang::Ja => counts[2] += 1,
            Lang::Ko => counts[3] += 1,
            Lang::Mixed => counts[4] += 1,
            Lang::Und => counts[5] += 1,
        }
    }
    let best = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, c)| *c)
        .map(|(i, &c)| if c == 0 { 5 } else { i })
        .unwrap_or(5);
    match best {
        0 => Lang::Zh,
        1 => Lang::En,
        2 => Lang::Ja,
        3 => Lang::Ko,
        4 => Lang::Mixed,
        _ => Lang::Und,
    }
}

fn summarize_doc_track(
    sentences: &[TranslatedSentence],
    target: TargetVariant,
    doc_lang: Lang,
    has_japanese_term_block: bool,
) -> TranslationTrack {
    if sentences.is_empty() && !has_japanese_term_block {
        if (doc_lang == Lang::Zh && target.is_zh()) || (doc_lang == Lang::En && target.is_en()) {
            return TranslationTrack::VariantRule;
        }
        return TranslationTrack::Passthrough;
    }
    let has_var = has_japanese_term_block
        || sentences
            .iter()
            .any(|s| s.track == TranslationTrack::VariantRule);
    let has_nmt = sentences
        .iter()
        .any(|s| s.track == TranslationTrack::NeuralSeq2Seq);
    match (has_var, has_nmt) {
        (true, true) => TranslationTrack::Hybrid,
        (true, false) => TranslationTrack::VariantRule,
        (false, true) => TranslationTrack::NeuralSeq2Seq,
        (false, false) => TranslationTrack::Passthrough,
    }
}
