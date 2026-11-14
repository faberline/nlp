use anyhow::{Context, Result, bail};
use clap::Parser;
use faberline_nlp::doc::Document;
use faberline_nlp_narrative::{
    BOOK_RECEIPT_SCHEMA, CHAPTER_NARRATIVE_SCHEMA, ChapterNarrative, ContextZone, EdgeStatus,
    Factuality, NarrativeManifestReceipt, WorkVocabulary, analyze_chapter_narrative,
    build_chapter_dag_from_narrative, text_sha256_hex,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const COMPLETED_NARRATIVE_INDEX_SCHEMA: &str = "nlp.narrative.completed_index.v1";

#[derive(Debug, Parser)]
#[command(
    name = "nlp_narrative_batch",
    about = "Build source-linked narrative event and timeline graphs over precomputed nlp_novel_batch chapters"
)]
struct Args {
    /// Directory containing nlp_novel_batch outputs (`chapters/*.json`, optional `completed_index.jsonl`, optional `discovered_lexicon.json`).
    #[arg(long, value_name = "DIR")]
    analysis_dir: PathBuf,

    /// Directory containing original UTF-8 bundle .txt files referenced by chapter source_file/source_span.
    #[arg(long, value_name = "DIR")]
    bundles_dir: PathBuf,

    /// Output directory for per-chapter ChapterNarrative JSONs, corpus JSONL streams, and receipt.
    #[arg(long, value_name = "DIR")]
    out_dir: PathBuf,

    /// Optional WorkVocabulary JSON file (`actors`, `places`, `objects`, `domain_terms`).
    #[arg(long, value_name = "PATH")]
    vocab: Option<PathBuf>,

    /// Number of parallel OS worker threads (defaults to available logical CPU cores).
    #[arg(long, value_name = "N")]
    workers: Option<usize>,

    /// Recompute all chapters from scratch instead of resuming completed chapters in --out-dir.
    #[arg(long)]
    no_resume: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
struct SourceSpan {
    start_char: usize,
    end_char: usize,
    start_byte: usize,
    end_byte: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct InputIndexEntry {
    order: usize,
    source_file: String,
    source_sha256: String,
    source_span: SourceSpan,
    chapter_sha256: String,
    #[serde(default)]
    output_file: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct InputChapterHeader {
    order: usize,
    source_file: String,
    source_sha256: String,
    source_span: SourceSpan,
    chapter_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct InputChapterFile {
    order: usize,
    title: String,
    source_file: String,
    source_sha256: String,
    source_span: SourceSpan,
    chapter_sha256: String,
    doc: Document,
}

#[derive(Clone, Debug, Deserialize)]
struct DiscoveredLexiconFile {
    #[serde(default)]
    terms: Vec<DiscoveredTermEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct DiscoveredTermEntry {
    term: String,
}

#[derive(Clone, Debug)]
struct ChapterDescriptor {
    chapter: u32,
    source_file: String,
    source_sha256: String,
    source_span: SourceSpan,
    chapter_sha256: String,
    input_rel_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct CompletedNarrativeIndexEntry {
    schema: String,
    chapter: u32,
    text_chars: usize,
    text_sha256: String,
    vocab_sha256: String,
    output_file: String,
    output_sha256: String,
    file_len: u64,
}

#[derive(Default)]
struct ChapterRollup {
    text_chars: usize,
    dialogues: usize,
    events: usize,
    actual_narrative_events: usize,
    mentions: usize,
    edges: usize,
    supported_edges: usize,
    non_chronological_edges: usize,
    conflict_cycles_broken: usize,
    residual_dag_cycles: usize,
    unknown_by_kind: BTreeMap<String, usize>,
}

struct ChapterWorkResult {
    index_entry: CompletedNarrativeIndexEntry,
    resumed: bool,
    rollup: ChapterRollup,
    events_jsonl: String,
    edges_jsonl: String,
    mentions_jsonl: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn default_workers() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

fn atomic_write_bytes(target: &Path, bytes: &[u8], tag: &str) -> Result<()> {
    let parent = target
        .parent()
        .ok_or_else(|| anyhow::anyhow!("target path {} has no parent", target.display()))?;
    let file_name = target
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let pid = std::process::id();
    let tid = format!("{:?}", std::thread::current().id());
    let sanitized_tid: String = tid.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    let tmp_path = parent.join(format!(".tmp-{file_name}-{tag}-{pid}-{sanitized_tid}"));

    {
        let file = File::create(&tmp_path)
            .with_context(|| format!("failed to create temp file {}", tmp_path.display()))?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(bytes)
            .with_context(|| format!("failed to write temp file {}", tmp_path.display()))?;
        writer
            .flush()
            .with_context(|| format!("failed to flush temp file {}", tmp_path.display()))?;
    }

    fs::rename(&tmp_path, target).with_context(|| {
        let _ = fs::remove_file(&tmp_path);
        format!(
            "failed to rename {} to {}",
            tmp_path.display(),
            target.display()
        )
    })?;
    Ok(())
}

fn load_and_merge_vocab(
    vocab_path: Option<&Path>,
    analysis_dir: &Path,
) -> Result<(WorkVocabulary, usize, bool)> {
    let mut vocab = match vocab_path {
        Some(path) => {
            let bytes = fs::read(path)
                .with_context(|| format!("failed to read vocabulary file {}", path.display()))?;
            serde_json::from_slice::<WorkVocabulary>(&bytes)
                .with_context(|| format!("failed to parse vocabulary JSON {}", path.display()))?
        }
        None => WorkVocabulary::default(),
    };

    let mut seen_terms: HashSet<String> = vocab.domain_terms.iter().cloned().collect();
    let mut discovered_count = 0usize;
    let discovered_path = analysis_dir.join("discovered_lexicon.json");
    if discovered_path.is_file() {
        let bytes = fs::read(&discovered_path).with_context(|| {
            format!(
                "failed to read discovered lexicon {}",
                discovered_path.display()
            )
        })?;
        let parsed: DiscoveredLexiconFile = serde_json::from_slice(&bytes).with_context(|| {
            format!(
                "failed to parse discovered lexicon {}",
                discovered_path.display()
            )
        })?;
        for entry in parsed.terms {
            if !entry.term.is_empty() && seen_terms.insert(entry.term.clone()) {
                vocab.domain_terms.push(entry.term);
                discovered_count += 1;
            }
        }
    }

    Ok((vocab, discovered_count, vocab_path.is_some()))
}

fn load_chapter_descriptors(analysis_dir: &Path) -> Result<Vec<ChapterDescriptor>> {
    let index_path = analysis_dir.join("completed_index.jsonl");
    let mut descriptors = Vec::new();

    if index_path.is_file() {
        let file = File::open(&index_path)
            .with_context(|| format!("failed to open {}", index_path.display()))?;
        let reader = BufReader::new(file);
        for (line_no, line) in reader.lines().enumerate() {
            let line = line.with_context(|| {
                format!(
                    "failed reading {} line {}",
                    index_path.display(),
                    line_no + 1
                )
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: InputIndexEntry = serde_json::from_str(trimmed).with_context(|| {
                format!(
                    "failed parsing {} line {}",
                    index_path.display(),
                    line_no + 1
                )
            })?;
            let chapter = u32::try_from(entry.order)
                .with_context(|| format!("chapter order {} exceeds u32", entry.order))?;
            let input_rel_path = entry
                .output_file
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| format!("chapters/{:04}.json", entry.order));
            descriptors.push(ChapterDescriptor {
                chapter,
                source_file: entry.source_file,
                source_sha256: entry.source_sha256,
                source_span: entry.source_span,
                chapter_sha256: entry.chapter_sha256,
                input_rel_path,
            });
        }
    } else {
        let chapters_dir = analysis_dir.join("chapters");
        if !chapters_dir.is_dir() {
            bail!(
                "neither {} nor {} exists",
                index_path.display(),
                chapters_dir.display()
            );
        }
        for entry in fs::read_dir(&chapters_dir)
            .with_context(|| format!("failed reading {}", chapters_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let file_name = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| anyhow::anyhow!("non-UTF-8 filename in {}", path.display()))?;
            if file_name.starts_with('.') {
                continue;
            }
            let bytes = fs::read(&path)
                .with_context(|| format!("failed reading chapter JSON {}", path.display()))?;
            let hdr: InputChapterHeader = serde_json::from_slice(&bytes)
                .with_context(|| format!("failed parsing chapter header {}", path.display()))?;
            let chapter = u32::try_from(hdr.order)
                .with_context(|| format!("chapter order {} exceeds u32", hdr.order))?;
            descriptors.push(ChapterDescriptor {
                chapter,
                source_file: hdr.source_file,
                source_sha256: hdr.source_sha256,
                source_span: hdr.source_span,
                chapter_sha256: hdr.chapter_sha256,
                input_rel_path: format!("chapters/{file_name}"),
            });
        }
    }

    if descriptors.is_empty() {
        bail!("no chapters found in {}", analysis_dir.display());
    }

    descriptors.sort_by_key(|d| d.chapter);
    for pair in descriptors.windows(2) {
        if pair[0].chapter == pair[1].chapter {
            bail!("duplicate chapter order {} found in input", pair[0].chapter);
        }
    }

    Ok(descriptors)
}

fn load_and_verify_bundles(
    bundles_dir: &Path,
    descriptors: &[ChapterDescriptor],
) -> Result<HashMap<String, Vec<u8>>> {
    let mut expected_hashes: BTreeMap<String, String> = BTreeMap::new();
    for desc in descriptors {
        match expected_hashes.get(&desc.source_file) {
            Some(prev) if prev != &desc.source_sha256 => {
                bail!(
                    "conflicting source_sha256 for bundle {}: {} vs {}",
                    desc.source_file,
                    prev,
                    desc.source_sha256
                );
            }
            Some(_) => {}
            None => {
                expected_hashes.insert(desc.source_file.clone(), desc.source_sha256.clone());
            }
        }
    }

    let mut bundles = HashMap::with_capacity(expected_hashes.len());
    for (source_file, expected_sha) in expected_hashes {
        let bundle_path = bundles_dir.join(&source_file);
        let bytes = fs::read(&bundle_path)
            .with_context(|| format!("failed to read bundle file {}", bundle_path.display()))?;
        let actual_sha = sha256_hex(&bytes);
        if actual_sha != expected_sha {
            bail!(
                "bundle SHA-256 mismatch for {}: expected {}, got {}",
                bundle_path.display(),
                expected_sha,
                actual_sha
            );
        }
        bundles.insert(source_file, bytes);
    }
    Ok(bundles)
}

fn load_resume_index(index_path: &Path) -> HashMap<u32, CompletedNarrativeIndexEntry> {
    let mut map = HashMap::new();
    let Ok(file) = File::open(index_path) else {
        return map;
    };
    let reader = BufReader::new(file);
    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<CompletedNarrativeIndexEntry>(trimmed) {
            map.insert(entry.chapter, entry);
        }
    }
    map
}

fn build_rollup_and_streams(
    ch: &ChapterNarrative,
) -> Result<(ChapterRollup, String, String, String)> {
    let dag = build_chapter_dag_from_narrative(ch);
    let residual_dag_cycles = usize::from(dag.has_cycle());

    let mut rollup = ChapterRollup {
        text_chars: ch.text_chars,
        dialogues: ch.dialogues.len(),
        events: ch.events.len(),
        actual_narrative_events: ch
            .events
            .iter()
            .filter(|e| e.zone == ContextZone::Narrative && e.factuality == Factuality::Actual)
            .count(),
        mentions: ch.mentions.len(),
        edges: ch.edges.len(),
        supported_edges: ch
            .edges
            .iter()
            .filter(|e| e.status == EdgeStatus::Supported)
            .count(),
        non_chronological_edges: ch
            .edges
            .iter()
            .filter(|e| e.status == EdgeStatus::NonChronologicalFactuality)
            .count(),
        conflict_cycles_broken: ch.conflict_cycles_broken,
        residual_dag_cycles,
        unknown_by_kind: BTreeMap::new(),
    };
    for u in &ch.unknown {
        *rollup
            .unknown_by_kind
            .entry(u.kind.as_str().to_string())
            .or_insert(0) += 1;
    }

    let mut events_jsonl = String::new();
    for ev in &ch.events {
        events_jsonl.push_str(&serde_json::to_string(ev)?);
        events_jsonl.push('\n');
    }

    let mut edges_jsonl = String::new();
    for edge in &ch.edges {
        edges_jsonl.push_str(&serde_json::to_string(edge)?);
        edges_jsonl.push('\n');
    }

    let mut mentions_jsonl = String::new();
    for m in &ch.mentions {
        mentions_jsonl.push_str(&serde_json::to_string(m)?);
        mentions_jsonl.push('\n');
    }

    Ok((rollup, events_jsonl, edges_jsonl, mentions_jsonl))
}

fn try_resume_chapter(
    desc: &ChapterDescriptor,
    out_dir: &Path,
    effective_vocab_sha256: &str,
    resume_index: &HashMap<u32, CompletedNarrativeIndexEntry>,
) -> Option<(CompletedNarrativeIndexEntry, ChapterNarrative)> {
    let entry = resume_index.get(&desc.chapter)?;
    if entry.text_sha256 != desc.chapter_sha256 || entry.vocab_sha256 != effective_vocab_sha256 {
        return None;
    }
    let out_path = out_dir.join(&entry.output_file);
    let bytes = fs::read(&out_path).ok()?;
    if bytes.len() as u64 != entry.file_len {
        return None;
    }
    if sha256_hex(&bytes) != entry.output_sha256 {
        return None;
    }
    let ch: ChapterNarrative = serde_json::from_slice(&bytes).ok()?;
    if ch.schema != CHAPTER_NARRATIVE_SCHEMA
        || ch.chapter != desc.chapter
        || ch.text_sha256 != desc.chapter_sha256
    {
        return None;
    }
    Some((entry.clone(), ch))
}

struct BatchContext {
    analysis_dir: PathBuf,
    out_dir: PathBuf,
    bundles: HashMap<String, Vec<u8>>,
    vocab: WorkVocabulary,
    effective_vocab_sha256: String,
    resume_index: HashMap<u32, CompletedNarrativeIndexEntry>,
    no_resume: bool,
}

fn process_one_chapter(desc: &ChapterDescriptor, ctx: &BatchContext) -> Result<ChapterWorkResult> {
    if !ctx.no_resume
        && let Some((index_entry, narrative)) = try_resume_chapter(
            desc,
            &ctx.out_dir,
            &ctx.effective_vocab_sha256,
            &ctx.resume_index,
        )
    {
        let (rollup, events_jsonl, edges_jsonl, mentions_jsonl) =
            build_rollup_and_streams(&narrative)?;
        if rollup.residual_dag_cycles > 0 {
            bail!(
                "resumed chapter {} contains residual DAG cycle",
                desc.chapter
            );
        }
        return Ok(ChapterWorkResult {
            index_entry,
            resumed: true,
            rollup,
            events_jsonl,
            edges_jsonl,
            mentions_jsonl,
        });
    }

    let input_path = ctx.analysis_dir.join(&desc.input_rel_path);
    let input_bytes = fs::read(&input_path)
        .with_context(|| format!("failed reading input chapter {}", input_path.display()))?;
    let input_ch: InputChapterFile = serde_json::from_slice(&input_bytes)
        .with_context(|| format!("failed parsing input chapter {}", input_path.display()))?;

    if input_ch.order as u32 != desc.chapter {
        bail!(
            "chapter order mismatch in {}: expected {}, got {}",
            input_path.display(),
            desc.chapter,
            input_ch.order
        );
    }
    if input_ch.source_file != desc.source_file || input_ch.source_sha256 != desc.source_sha256 {
        bail!(
            "source bundle metadata mismatch in {}",
            input_path.display()
        );
    }
    if input_ch.source_span != desc.source_span {
        bail!("source_span mismatch in {}", input_path.display());
    }

    let bundle = ctx
        .bundles
        .get(&desc.source_file)
        .ok_or_else(|| anyhow::anyhow!("missing loaded bundle {}", desc.source_file))?;
    let span = input_ch.source_span;
    if span.start_byte > span.end_byte || span.end_byte > bundle.len() {
        bail!(
            "chapter {} byte span {}..{} out of bounds for bundle {} (len {})",
            desc.chapter,
            span.start_byte,
            span.end_byte,
            desc.source_file,
            bundle.len()
        );
    }
    let slice = &bundle[span.start_byte..span.end_byte];
    let text = std::str::from_utf8(slice).with_context(|| {
        format!(
            "chapter {} slice {}..{} in {} is not valid UTF-8",
            desc.chapter, span.start_byte, span.end_byte, desc.source_file
        )
    })?;

    let actual_text_sha = text_sha256_hex(text);
    if actual_text_sha != input_ch.chapter_sha256 || input_ch.chapter_sha256 != desc.chapter_sha256
    {
        bail!(
            "chapter {} SHA-256 mismatch: text={}, file={}, index={}",
            desc.chapter,
            actual_text_sha,
            input_ch.chapter_sha256,
            desc.chapter_sha256
        );
    }
    if input_ch.doc.text_sha256 != actual_text_sha {
        bail!(
            "chapter {} doc.text_sha256 ({}) != chapter text SHA-256 ({})",
            desc.chapter,
            input_ch.doc.text_sha256,
            actual_text_sha
        );
    }

    let title = (!input_ch.title.is_empty()).then_some(input_ch.title);
    let narrative =
        analyze_chapter_narrative(desc.chapter, title, text, &input_ch.doc, &ctx.vocab)?;
    let (rollup, events_jsonl, edges_jsonl, mentions_jsonl) = build_rollup_and_streams(&narrative)?;
    if rollup.residual_dag_cycles > 0 {
        bail!("chapter {} has residual DAG cycle", desc.chapter);
    }

    let out_rel = format!("chapters/{:04}.json", desc.chapter);
    let out_path = ctx.out_dir.join(&out_rel);
    let mut out_bytes = serde_json::to_vec(&narrative)
        .with_context(|| format!("failed serializing chapter {}", desc.chapter))?;
    out_bytes.push(b'\n');
    let output_sha256 = sha256_hex(&out_bytes);
    let file_len = out_bytes.len() as u64;
    atomic_write_bytes(&out_path, &out_bytes, "chapter")?;

    let index_entry = CompletedNarrativeIndexEntry {
        schema: COMPLETED_NARRATIVE_INDEX_SCHEMA.to_string(),
        chapter: desc.chapter,
        text_chars: narrative.text_chars,
        text_sha256: narrative.text_sha256,
        vocab_sha256: ctx.effective_vocab_sha256.clone(),
        output_file: out_rel,
        output_sha256,
        file_len,
    };

    Ok(ChapterWorkResult {
        index_entry,
        resumed: false,
        rollup,
        events_jsonl,
        edges_jsonl,
        mentions_jsonl,
    })
}

fn compute_chapter_outputs_sha256(entries: &[CompletedNarrativeIndexEntry]) -> String {
    let mut hasher = Sha256::new();
    for e in entries {
        hasher.update(format!("{:04}:{}\n", e.chapter, e.output_sha256).as_bytes());
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn run() -> Result<()> {
    let started = Instant::now();
    let args = Args::parse();
    let workers = args.workers.unwrap_or_else(default_workers).max(1);

    let (vocab, discovered_lexicon_terms, vocab_file_provided) =
        load_and_merge_vocab(args.vocab.as_deref(), &args.analysis_dir)?;
    let vocab_bytes = serde_json::to_vec(&vocab).context("failed to serialize vocabulary")?;
    let effective_vocab_sha256 = sha256_hex(&vocab_bytes);
    let receipt_vocab_sha256 =
        (vocab_file_provided || discovered_lexicon_terms > 0 || !vocab.is_empty())
            .then(|| effective_vocab_sha256.clone());

    let descriptors = load_chapter_descriptors(&args.analysis_dir)?;
    let bundles = load_and_verify_bundles(&args.bundles_dir, &descriptors)?;

    fs::create_dir_all(args.out_dir.join("chapters")).with_context(|| {
        format!(
            "failed to create output directory {}",
            args.out_dir.join("chapters").display()
        )
    })?;

    let index_path = args.out_dir.join("completed_narrative_index.jsonl");
    let resume_index = if args.no_resume {
        HashMap::new()
    } else {
        load_resume_index(&index_path)
    };

    let descriptors = Arc::new(descriptors);
    let ctx = Arc::new(BatchContext {
        analysis_dir: args.analysis_dir.clone(),
        out_dir: args.out_dir.clone(),
        bundles,
        vocab,
        effective_vocab_sha256,
        resume_index,
        no_resume: args.no_resume,
    });

    let total_chapters = descriptors.len();
    let next_idx = Arc::new(AtomicUsize::new(0));
    let slots: Arc<Vec<std::sync::Mutex<Option<ChapterWorkResult>>>> = Arc::new(
        (0..total_chapters)
            .map(|_| std::sync::Mutex::new(None))
            .collect(),
    );

    let thread_count = workers.min(total_chapters);
    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(thread_count);
        for _ in 0..thread_count {
            let descriptors = Arc::clone(&descriptors);
            let ctx = Arc::clone(&ctx);
            let next_idx = Arc::clone(&next_idx);
            let slots = Arc::clone(&slots);

            handles.push(scope.spawn(move || -> Result<()> {
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= descriptors.len() {
                        break;
                    }
                    let res = process_one_chapter(&descriptors[idx], &ctx)?;
                    let mut guard = slots[idx]
                        .lock()
                        .map_err(|_| anyhow::anyhow!("worker slot lock poisoned"))?;
                    *guard = Some(res);
                }
                Ok(())
            }));
        }

        for handle in handles {
            let thread_res = handle
                .join()
                .map_err(|_| anyhow::anyhow!("worker thread panicked"))?;
            thread_res?;
        }
        Ok(())
    })?;

    let mut index_entries = Vec::with_capacity(total_chapters);
    let mut index_bytes = Vec::new();
    let mut events_bytes = Vec::new();
    let mut edges_bytes = Vec::new();
    let mut mentions_bytes = Vec::new();

    let mut resumed_chapters = 0usize;
    let mut processed_chapters = 0usize;
    let mut total_chars = 0usize;
    let mut total_dialogues = 0usize;
    let mut total_events = 0usize;
    let mut actual_narrative_events = 0usize;
    let mut total_mentions = 0usize;
    let mut total_edges = 0usize;
    let mut supported_edges = 0usize;
    let mut non_chronological_edges = 0usize;
    let mut conflict_cycles_broken = 0usize;
    let mut residual_dag_cycles = 0usize;
    let mut unknown_by_kind: BTreeMap<String, usize> = BTreeMap::new();

    for (idx, slot_mutex) in slots.iter().enumerate() {
        let res = slot_mutex
            .lock()
            .map_err(|_| anyhow::anyhow!("slot lock poisoned"))?
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing chapter result at index {idx}"))?;

        if res.resumed {
            resumed_chapters += 1;
        } else {
            processed_chapters += 1;
        }

        total_chars += res.rollup.text_chars;
        total_dialogues += res.rollup.dialogues;
        total_events += res.rollup.events;
        actual_narrative_events += res.rollup.actual_narrative_events;
        total_mentions += res.rollup.mentions;
        total_edges += res.rollup.edges;
        supported_edges += res.rollup.supported_edges;
        non_chronological_edges += res.rollup.non_chronological_edges;
        conflict_cycles_broken += res.rollup.conflict_cycles_broken;
        residual_dag_cycles += res.rollup.residual_dag_cycles;
        for (k, v) in res.rollup.unknown_by_kind {
            *unknown_by_kind.entry(k).or_insert(0) += v;
        }

        let line = serde_json::to_string(&res.index_entry)?;
        index_bytes.extend_from_slice(line.as_bytes());
        index_bytes.push(b'\n');
        index_entries.push(res.index_entry);

        events_bytes.extend_from_slice(res.events_jsonl.as_bytes());
        edges_bytes.extend_from_slice(res.edges_jsonl.as_bytes());
        mentions_bytes.extend_from_slice(res.mentions_jsonl.as_bytes());
    }

    if residual_dag_cycles != 0 {
        bail!("corpus contains {residual_dag_cycles} residual DAG cycles (invariant requires 0)");
    }

    atomic_write_bytes(&index_path, &index_bytes, "index")?;

    let events_stream_path = args.out_dir.join("events_stream.jsonl");
    let timeline_edges_path = args.out_dir.join("timeline_edges.jsonl");
    let actor_mentions_path = args.out_dir.join("actor_mentions.jsonl");

    atomic_write_bytes(&events_stream_path, &events_bytes, "events")?;
    atomic_write_bytes(&timeline_edges_path, &edges_bytes, "edges")?;
    atomic_write_bytes(&actor_mentions_path, &mentions_bytes, "mentions")?;

    let chapter_outputs_sha256 = compute_chapter_outputs_sha256(&index_entries);
    let events_stream_sha256 = sha256_hex(&events_bytes);
    let timeline_edges_sha256 = sha256_hex(&edges_bytes);
    let actor_mentions_sha256 = sha256_hex(&mentions_bytes);

    let receipt = NarrativeManifestReceipt {
        schema: BOOK_RECEIPT_SCHEMA.to_string(),
        input_mode: "nlp_novel_batch+bundles".to_string(),
        workers,
        vocab_sha256: receipt_vocab_sha256,
        discovered_lexicon_terms,
        total_chapters,
        resumed_chapters,
        processed_chapters,
        total_chars,
        elapsed_secs: started.elapsed().as_secs_f64(),
        total_dialogues,
        total_events,
        actual_narrative_events,
        total_mentions,
        total_edges,
        supported_edges,
        non_chronological_edges,
        conflict_cycles_broken,
        residual_dag_cycles,
        unknown_by_kind,
        chapter_outputs_sha256,
        events_stream_sha256,
        timeline_edges_sha256,
        actor_mentions_sha256,
    };

    let mut receipt_bytes = serde_json::to_vec_pretty(&receipt)?;
    receipt_bytes.push(b'\n');
    let receipt_path = args.out_dir.join("narrative_manifest_receipt.json");
    atomic_write_bytes(&receipt_path, &receipt_bytes, "receipt")?;

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("nlp_narrative_batch: {err:#}");
        std::process::exit(1);
    }
}
