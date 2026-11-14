//! Resumable, deterministic multi-worker batch book/corpus translation pipeline
//! (`nlp_translate_batch`) and `translate_manifest_receipt.json` structural audit.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use faberline_nlp::doc::analyze_document;
use faberline_nlp::doc::model::Backend;
use serde::Deserialize;

use crate::schema::{
    ChapterTranslateReceipt, GlossaryEntry, OpaquePolicy, TRANSLATE_BATCH_RECEIPT_SCHEMA,
    TranslateManifestReceipt, TranslateOptions, TranslateUnknownKind, TranslatedDocument,
    TranslationGlossary, TranslationTrack, sha256_hex,
};
use crate::{
    LoadedTranslateModel, doc_options_for_translate, render_bilingual_markdown, translate_document,
};

#[derive(Deserialize)]
struct DiscoveredLexiconFile {
    #[serde(default)]
    terms: Vec<DiscoveredTermEntry>,
    #[serde(default)]
    compounds: Vec<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DiscoveredTermEntry {
    Plain(String),
    Record { surface: String },
}

#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub options: TranslateOptions,
    pub glossary: TranslationGlossary,
    pub models_dir: Option<PathBuf>,
    pub backend: Backend,
    pub workers: usize,
    pub resume: bool,
    pub write_bilingual: bool,
}

/// Run resumable multi-worker translation over all `.txt` and `.md` files in `cfg.input_dir`,
/// writing per-chapter JSON, target text, optional bilingual Markdown, and
/// `<output_dir>/translate_manifest_receipt.json`.
pub fn run_translate_batch(cfg: &BatchConfig) -> anyhow::Result<TranslateManifestReceipt> {
    let mut entries: Vec<PathBuf> = fs::read_dir(&cfg.input_dir)?
        .filter_map(|e| e.ok().map(|d| d.path()))
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|x| x.to_str())
                    .is_some_and(|ext| matches!(ext, "txt" | "md"))
        })
        .collect();
    entries.sort();

    let json_dir = cfg.output_dir.join("json");
    let text_dir = cfg.output_dir.join("text");
    let bi_dir = cfg.output_dir.join("bilingual");
    fs::create_dir_all(&json_dir)?;
    fs::create_dir_all(&text_dir)?;
    if cfg.write_bilingual {
        fs::create_dir_all(&bi_dir)?;
    }

    // Merge optional `<input_dir>/discovered_lexicon.json` if present.
    let mut merged_glossary = cfg.glossary.clone();
    let discovered_path = cfg.input_dir.join("discovered_lexicon.json");
    if discovered_path.is_file()
        && let Ok(raw) = fs::read_to_string(&discovered_path)
        && let Ok(parsed) = serde_json::from_str::<DiscoveredLexiconFile>(&raw)
    {
        let existing: std::collections::HashSet<String> =
            merged_glossary.surface_terms().1.into_iter().collect();
        let iter_surfaces = parsed
            .compounds
            .into_iter()
            .chain(parsed.terms.into_iter().map(|t| match t {
                DiscoveredTermEntry::Plain(s) => s,
                DiscoveredTermEntry::Record { surface } => surface,
            }));
        for surf in iter_surfaces {
            if !surf.is_empty() && !existing.contains(&surf) {
                merged_glossary.domain_terms.push(GlossaryEntry {
                    canonical: surf.clone(),
                    surfaces: vec![surf],
                    ..GlossaryEntry::default()
                });
            }
        }
    }

    let loaded_seq2seq = match &cfg.models_dir {
        Some(mdir) => match LoadedTranslateModel::load(mdir, &cfg.options.model_id, cfg.backend) {
            Ok(m) => Some(m),
            Err(e) if e.into_reason().is_some() => None,
            Err(e) => return Err(anyhow::anyhow!("translate batch model load failed: {e}")),
        },
        None => None,
    };

    let worker_count = cfg.workers.max(1).min(entries.len().max(1));
    let next_idx = AtomicUsize::new(0);
    let mut slot_results: Vec<Option<anyhow::Result<(TranslatedDocument, bool, String)>>> =
        (0..entries.len()).map(|_| None).collect();

    let slots_mutex = std::sync::Mutex::new(&mut slot_results);
    std::thread::scope(|s| {
        for _ in 0..worker_count {
            let entries_ref = &entries;
            let glossary_ref = &merged_glossary;
            let model_ref = loaded_seq2seq.as_ref();
            let json_dir_ref = &json_dir;
            let text_dir_ref = &text_dir;
            let bi_dir_ref = &bi_dir;
            let next_ref = &next_idx;
            let slots_ref = &slots_mutex;

            s.spawn(move || {
                loop {
                    let i = next_ref.fetch_add(1, Ordering::SeqCst);
                    if i >= entries_ref.len() {
                        break;
                    }
                    let path = &entries_ref[i];
                    let res = process_one_chapter(
                        (i + 1) as u32,
                        path,
                        json_dir_ref,
                        text_dir_ref,
                        bi_dir_ref,
                        cfg,
                        glossary_ref,
                        model_ref,
                    );
                    if let Ok(mut guard) = slots_ref.lock() {
                        guard[i] = Some(res);
                    }
                }
            });
        }
    });

    let mut chapters_translated = 0usize;
    let mut chapters_resumed_cached = 0usize;
    let mut total_source_chars = 0usize;
    let mut total_target_chars = 0usize;
    let mut total_sentences = 0usize;
    let mut variant_rule_sentences = 0usize;
    let mut neural_seq2seq_sentences = 0usize;
    let mut passthrough_sentences = 0usize;
    let mut glossary_hits = 0usize;
    let mut verbatim_opaque_spans = 0usize;
    let mut verbatim_opaque_byte_identical = 0usize;
    let mut placeholder_repairs = 0usize;
    let mut unknown_count = 0usize;
    let mut chapter_receipts = Vec::with_capacity(entries.len());

    for (i, slot) in slot_results.into_iter().enumerate() {
        let (tr_doc, cached, src_text) =
            slot.ok_or_else(|| anyhow::anyhow!("missing worker result for slot {i}"))??;
        if cached {
            chapters_resumed_cached += 1;
        } else {
            chapters_translated += 1;
        }
        total_source_chars += tr_doc.source_chars;
        total_target_chars += tr_doc.target_chars;
        total_sentences += tr_doc.sentences.len();
        for s in &tr_doc.sentences {
            match s.track {
                TranslationTrack::VariantRule => variant_rule_sentences += 1,
                TranslationTrack::NeuralSeq2Seq => neural_seq2seq_sentences += 1,
                TranslationTrack::Passthrough | TranslationTrack::Hybrid => {
                    passthrough_sentences += 1;
                }
            }
        }
        glossary_hits += tr_doc.glossary_hits.len();

        let src_chars: Vec<char> = src_text.chars().collect();
        let tgt_chars: Vec<char> = tr_doc.target_text.chars().collect();
        for op in &tr_doc.opaque_spans {
            if op.policy == OpaquePolicy::VerbatimPassthrough
                && let Some(ts) = op.target_span
            {
                verbatim_opaque_spans += 1;
                let s_sub: String = src_chars[op.source_span.start_char..op.source_span.end_char]
                    .iter()
                    .collect();
                let t_sub: String = tgt_chars[ts.start_char..ts.end_char].iter().collect();
                if s_sub == t_sub {
                    verbatim_opaque_byte_identical += 1;
                }
            }
        }

        placeholder_repairs += tr_doc
            .unknown
            .iter()
            .filter(|u| u.kind == TranslateUnknownKind::PlaceholderRepaired)
            .count();
        unknown_count += tr_doc.unknown.len();

        let fname = entries[i]
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_owned();
        chapter_receipts.push(ChapterTranslateReceipt {
            chapter: (i + 1) as u32,
            file: fname,
            source_sha256: tr_doc.source_sha256,
            target_sha256: tr_doc.target_sha256,
            sentences: tr_doc.sentences.len(),
            glossary_hits: tr_doc.glossary_hits.len(),
            unknown_count: tr_doc.unknown.len(),
        });
    }

    let receipt = TranslateManifestReceipt {
        schema: TRANSLATE_BATCH_RECEIPT_SCHEMA.to_owned(),
        target_variant: cfg.options.target,
        chapters_total: entries.len(),
        chapters_translated,
        chapters_resumed_cached,
        total_source_chars,
        total_target_chars,
        total_sentences,
        variant_rule_sentences,
        neural_seq2seq_sentences,
        passthrough_sentences,
        glossary_hits,
        verbatim_opaque_spans,
        verbatim_opaque_byte_identical,
        placeholder_repairs,
        unknown_count,
        chapters: chapter_receipts,
    };

    fs::write(
        cfg.output_dir.join("translate_manifest_receipt.json"),
        serde_json::to_string_pretty(&receipt)? + "\n",
    )?;
    Ok(receipt)
}

#[allow(clippy::too_many_arguments)]
fn process_one_chapter(
    chapter_num: u32,
    input_path: &Path,
    json_dir: &Path,
    text_dir: &Path,
    bi_dir: &Path,
    cfg: &BatchConfig,
    glossary: &TranslationGlossary,
    seq2seq: Option<&LoadedTranslateModel>,
) -> anyhow::Result<(TranslatedDocument, bool, String)> {
    let src_text = fs::read_to_string(input_path)?;
    let src_sha = sha256_hex(&src_text);
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("chapter");
    let ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("txt");

    let json_path = json_dir.join(format!("{stem}.translate.json"));
    let text_path = text_dir.join(format!("{stem}.{ext}"));
    let bi_path = bi_dir.join(format!("{stem}.bilingual.md"));

    if cfg.resume
        && json_path.is_file()
        && text_path.is_file()
        && let Ok(existing_raw) = fs::read_to_string(&json_path)
        && let Ok(existing_doc) = serde_json::from_str::<TranslatedDocument>(&existing_raw)
        && existing_doc.source_sha256 == src_sha
        && existing_doc.target_variant == cfg.options.target
    {
        return Ok((existing_doc, true, src_text));
    }

    let doc_opts = doc_options_for_translate(&cfg.options, glossary);
    let doc = analyze_document(&src_text, &doc_opts);

    #[cfg(feature = "narrative")]
    let nar_opt = {
        let vocab = glossary.to_work_vocabulary();
        faberline_nlp_narrative::analyze_chapter_narrative(
            chapter_num,
            Some(stem.to_owned()),
            &src_text,
            &doc,
            &vocab,
        )
        .ok()
    };
    #[cfg(not(feature = "narrative"))]
    let _ = chapter_num;

    let tr_doc = translate_document(
        &src_text,
        &doc,
        #[cfg(feature = "narrative")]
        nar_opt.as_ref(),
        &cfg.options,
        glossary,
        seq2seq,
    )?;

    fs::write(&json_path, serde_json::to_string_pretty(&tr_doc)? + "\n")?;
    fs::write(&text_path, &tr_doc.target_text)?;
    if cfg.write_bilingual {
        let bi = render_bilingual_markdown(&tr_doc, &src_text);
        fs::write(&bi_path, bi)?;
    }

    Ok((tr_doc, false, src_text))
}
