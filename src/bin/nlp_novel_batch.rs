//! Multi-core resumable full-novel batch analyzer (`nlp.doc.v1` + book-level rollups).
//!
//! Supports both directory-of-bundles (`--bundles-dir`) and `manifest.json` (`--manifest`) inputs,
//! preserving exact UTF-8 byte and Unicode scalar provenance spans, atomic per-chapter outputs,
//! SHA-256 resume verification, lock-free per-worker neural syntax inference, and deterministic
//! book-level audit and timeline-cue artifacts.

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use faberline_nlp::Span;
#[cfg(any(feature = "onnx", feature = "coreml"))]
use faberline_nlp::doc::analyze_document_with_loaded_model;
use faberline_nlp::doc::model::Verified;
#[cfg(any(feature = "onnx", feature = "coreml"))]
use faberline_nlp::doc::model::{Backend, SubwordTokenizer};
use faberline_nlp::doc::{
    Document, EntityKind, Options, Reason, RelationKind, RelationLevel, SYNTAX_MODEL_ID,
    analyze_document,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

pub const CHAPTER_SCHEMA: &str = "nlp.novel_chapter.v1";
pub const INDEX_SCHEMA: &str = "nlp.novel_completed_index.v1";
pub const RECEIPT_SCHEMA: &str = "nlp.novel_book_receipt.v1";
pub const CUE_SCHEMA: &str = "nlp.novel_timeline_cue.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum BackendMode {
    /// Core ML with all Apple Silicon compute units (CPU + GPU + Neural Engine).
    #[value(name = "coreml-fast")]
    CoremlFast,
    /// Core ML restricted to CPU-only for strict cross-backend logit parity.
    #[value(name = "coreml")]
    Coreml,
    /// Tract ONNX CPU runtime.
    #[value(name = "onnx")]
    Onnx,
    /// Rule layers only (Layers 1, 1b, 2, 3, 7; no neural model).
    #[value(name = "none")]
    None,
}

impl BackendMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::CoremlFast => "coreml-fast",
            Self::Coreml => "coreml",
            Self::Onnx => "onnx",
            Self::None => "none",
        }
    }

    #[cfg(any(feature = "onnx", feature = "coreml"))]
    fn to_doc_backend(self) -> Option<Backend> {
        match self {
            Self::CoremlFast => Some(Backend::CoremlFast),
            Self::Coreml => Some(Backend::Coreml),
            Self::Onnx => Some(Backend::Onnx),
            Self::None => None,
        }
    }

    fn is_compiled(self) -> bool {
        match self {
            Self::CoremlFast | Self::Coreml => cfg!(all(feature = "coreml", target_os = "macos")),
            Self::Onnx => cfg!(feature = "onnx"),
            Self::None => true,
        }
    }
}

fn default_backend_for_models() -> Result<BackendMode> {
    if cfg!(all(feature = "coreml", target_os = "macos")) {
        Ok(BackendMode::CoremlFast)
    } else if cfg!(feature = "onnx") {
        Ok(BackendMode::Onnx)
    } else {
        bail!(
            "--models was supplied, but no neural backend is compiled in (rebuild with --features coreml or --features onnx, or pass --backend none)"
        )
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "nlp_novel_batch",
    about = "Multi-core resumable novel batch analyzer producing nlp.doc.v1 chapters and book rollups"
)]
struct Args {
    /// Directory of multi-chapter UTF-8 `.txt` bundle files, processed in sorted filename order.
    #[arg(long, value_name = "DIR", conflicts_with = "manifest")]
    bundles_dir: Option<PathBuf>,

    /// Path to `manifest.json` listing ordered chapter `.txt` files.
    #[arg(long, value_name = "PATH", conflicts_with = "bundles_dir")]
    manifest: Option<PathBuf>,

    /// Output directory for `chapters/*.json`, `completed_index.jsonl`, `book_manifest_receipt.json`, and `entities_timeline_cues.jsonl`.
    #[arg(long, value_name = "DIR")]
    out_dir: PathBuf,

    /// Directory containing `manifest.json` and model subdirectories for Layer 4 syntax & contextual UPOS.
    #[arg(long, value_name = "DIR")]
    models: Option<PathBuf>,

    /// Model identifier inside `--models` manifest.
    #[arg(long, default_value = SYNTAX_MODEL_ID)]
    model_id: String,

    /// Execution backend (`coreml-fast`, `coreml`, `onnx`, or `none`).
    #[arg(long, value_enum)]
    backend: Option<BackendMode>,

    /// Number of parallel worker OS threads (default: available CPU cores).
    #[arg(long, value_name = "N")]
    workers: Option<usize>,

    /// Verify and skip already-completed chapters matching `chapter_sha256` and `model_sha256` (enabled by default).
    #[arg(long, overrides_with = "no_resume")]
    resume: bool,

    /// Disable resume and recompute all chapters from scratch.
    #[arg(long, overrides_with = "resume")]
    no_resume: bool,

    /// Optional exact surface names to pass to `nlp.doc.v1` Layer 3 name extraction.
    #[arg(long = "name", value_name = "SURFACE")]
    names: Vec<String>,
}

/// Exact Unicode scalar (`start_char..end_char`) and UTF-8 byte (`start_byte..end_byte`) span within `source_file`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start_char: usize,
    pub end_char: usize,
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChapterOutput {
    pub schema: String,
    pub order: usize,
    pub title: String,
    pub source_file: String,
    pub source_sha256: String,
    pub source_span: SourceSpan,
    pub chapter_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
    pub backend: String,
    pub doc: Document,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct CompletedIndexEntry {
    schema: String,
    order: usize,
    title: String,
    source_file: String,
    source_sha256: String,
    source_span: SourceSpan,
    chapter_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_sha256: Option<String>,
    backend: String,
    output_file: String,
    output_sha256: String,
    file_len: u64,
    stats: ChapterStats,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct ChapterStats {
    text_chars: usize,
    paragraphs: usize,
    sentences: usize,
    supported_sentences: usize,
    clauses: usize,
    tokens: usize,
    decided_upos_tokens: usize,
    syntax_tokens: usize,
    entities: usize,
    relations: usize,
    timeline_cues: usize,
    unknown_by_layer_reason: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct UnknownBreakdownEntry {
    layer: String,
    reason: String,
    count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BookManifestReceipt {
    schema: String,
    input_mode: String,
    backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_sha256: Option<String>,
    workers: usize,
    total_chapters: usize,
    resumed_chapters: usize,
    processed_chapters: usize,
    total_chars: usize,
    total_paragraphs: usize,
    total_sentences: usize,
    supported_sentences: usize,
    total_clauses: usize,
    total_tokens: usize,
    decided_upos_tokens: usize,
    decided_upos_rate: f64,
    syntax_tokens: usize,
    syntax_token_coverage_rate: f64,
    total_entities: usize,
    total_relations: usize,
    total_timeline_cues: usize,
    unknown_total: usize,
    unknown_by_layer_reason: BTreeMap<String, usize>,
    unknown_breakdown: Vec<UnknownBreakdownEntry>,
    elapsed_secs: f64,
    chapters_per_sec: f64,
    sentences_per_sec: f64,
    processed_sentences_per_sec: f64,
    chapter_outputs_sha256: String,
    timeline_cues_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct TimelineCueRecord {
    schema: String,
    chapter: usize,
    chapter_title: String,
    source_file: String,
    cue_type: String,
    kind: String,
    span: Span,
    bundle_span: Span,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sentence: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    level: Option<RelationLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    connector: Option<Span>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    connector_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    correlate: Option<Span>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    correlate_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    arg1: Option<Span>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    arg2: Option<Span>,
}

#[derive(Clone, Debug)]
struct InputChapter {
    order: usize,
    title: String,
    source_file: String,
    source_sha256: String,
    source_span: SourceSpan,
    chapter_sha256: String,
    text: String,
}

#[derive(Deserialize)]
struct InputManifest {
    chapters: Vec<InputManifestChapter>,
}

#[derive(Deserialize)]
struct InputManifestChapter {
    order: usize,
    #[serde(default)]
    title: Option<String>,
    file: String,
    #[serde(default)]
    status: Option<String>,
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

fn round6(val: f64) -> f64 {
    if !val.is_finite() {
        0.0
    } else {
        (val * 1_000_000.0).round() / 1_000_000.0
    }
}

fn compute_model_sha256(verified: &Verified) -> String {
    let mut hasher = Sha256::new();
    hasher.update(verified.info.id.as_bytes());
    hasher.update(b"\n");
    let mut files: Vec<(&str, &str)> = verified
        .info
        .files
        .iter()
        .map(|f| (f.path.as_str(), f.sha256.as_str()))
        .collect();
    files.sort_unstable();
    for (path, sha) in files {
        hasher.update(path.as_bytes());
        hasher.update(b":");
        hasher.update(sha.as_bytes());
        hasher.update(b"\n");
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn atomic_write_bytes(target: &Path, bytes: &[u8], tag: &str) -> Result<()> {
    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("target path has no parent directory: {}", target.display()))?;
    let file_name = target
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("artifact");
    let pid = std::process::id();
    let tmp_path = parent.join(format!(".{file_name}.tmp.{pid}.{tag}"));
    {
        let mut f = File::create(&tmp_path)
            .with_context(|| format!("create temp file {}", tmp_path.display()))?;
        f.write_all(bytes)
            .with_context(|| format!("write temp file {}", tmp_path.display()))?;
        f.flush()
            .with_context(|| format!("flush temp file {}", tmp_path.display()))?;
    }
    fs::rename(&tmp_path, target).with_context(|| {
        format!(
            "atomic rename {} -> {}",
            tmp_path.display(),
            target.display()
        )
    })?;
    Ok(())
}

// ============================================================================
// Generic Multi-Chapter Bundle Splitter (Chinese & English, Exact Provenance)
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct RawLine<'a> {
    raw: &'a str,
    start_char: usize,
    end_char: usize,
    start_byte: usize,
    end_byte: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DetectedHeading {
    title: String,
    number: Option<usize>,
    norm_sub: String,
    is_prefixed_xa0: bool,
}

fn is_heading_ws(c: char) -> bool {
    c.is_whitespace() || c == '\u{00a0}' || c == '\u{3000}' || c == '\u{feff}'
}

fn is_blank_line(s: &str) -> bool {
    s.chars().all(is_heading_ws)
}

fn zh_digit_value(c: char) -> Option<usize> {
    match c {
        '零' | '〇' => Some(0),
        '一' | '壹' => Some(1),
        '二' | '兩' | '两' | '貳' | '贰' => Some(2),
        '三' | '參' | '叁' => Some(3),
        '四' | '肆' => Some(4),
        '五' | '伍' => Some(5),
        '六' | '陸' | '陆' => Some(6),
        '七' | '柒' => Some(7),
        '八' | '捌' => Some(8),
        '九' | '玖' => Some(9),
        _ => None,
    }
}

fn zh_small_unit_value(c: char) -> Option<usize> {
    match c {
        '十' | '拾' => Some(10),
        '百' | '佰' => Some(100),
        '千' | '仟' => Some(1000),
        _ => None,
    }
}

fn is_strict_zh_or_ascii_numeral_char(c: char) -> bool {
    c.is_ascii_digit()
        || zh_digit_value(c).is_some()
        || zh_small_unit_value(c).is_some()
        || matches!(c, '萬' | '万' | '億' | '亿')
}

/// Parses an Arabic or Chinese chapter integer, optionally tolerating at most one inner OCR/IME
/// homophone unit character flanked by digits (such as `一<typo>八十三` -> `183`).
fn parse_chapter_number(s: &str, allow_single_inner_typo: bool) -> Option<usize> {
    let chars: Vec<char> = s.chars().filter(|c| !is_heading_ws(*c)).collect();
    if chars.is_empty() || chars.len() > 12 {
        return None;
    }
    if chars.iter().all(char::is_ascii_digit) {
        let ascii: String = chars.into_iter().collect();
        return ascii.parse::<usize>().ok();
    }

    let mut total = 0usize;
    let mut section = 0usize;
    let mut cur = 0usize;
    let mut seen = false;
    let mut typo_used = false;

    for (idx, &c) in chars.iter().enumerate() {
        if let Some(d) = zh_digit_value(c) {
            cur = d;
            seen = true;
        } else if let Some(u) = zh_small_unit_value(c) {
            let mult = if cur == 0 { 1 } else { cur };
            section = section.checked_add(mult.checked_mul(u)?)?;
            cur = 0;
            seen = true;
        } else if matches!(c, '萬' | '万') {
            section = section.checked_add(cur)?;
            let mult = if section == 0 { 1 } else { section };
            total = total.checked_add(mult.checked_mul(10_000)?)?;
            section = 0;
            cur = 0;
            seen = true;
        } else if matches!(c, '億' | '亿') {
            section = section.checked_add(cur)?;
            let mult = if section == 0 { 1 } else { section };
            total = total.checked_add(mult.checked_mul(100_000_000)?)?;
            section = 0;
            cur = 0;
            seen = true;
        } else if allow_single_inner_typo
            && !typo_used
            && idx > 0
            && idx + 1 < chars.len()
            && zh_digit_value(chars[idx - 1]).is_some()
            && zh_digit_value(chars[idx + 1]).is_some()
        {
            let max_rem_unit = chars[idx + 1..]
                .iter()
                .filter_map(|&ch| zh_small_unit_value(ch))
                .max()
                .unwrap_or(1);
            let inferred_u = (max_rem_unit * 10).max(10);
            if !matches!(inferred_u, 10 | 100 | 1000) {
                return None;
            }
            let mult = if cur == 0 { 1 } else { cur };
            section = section.checked_add(mult.checked_mul(inferred_u)?)?;
            cur = 0;
            seen = true;
            typo_used = true;
        } else {
            return None;
        }
    }

    seen.then(|| total.saturating_add(section).saturating_add(cur))
}

fn parse_roman_numeral(s: &str) -> Option<usize> {
    if s.is_empty() || s.len() > 15 {
        return None;
    }
    let val = |b: u8| -> Option<usize> {
        match b.to_ascii_uppercase() {
            b'I' => Some(1),
            b'V' => Some(5),
            b'X' => Some(10),
            b'L' => Some(50),
            b'C' => Some(100),
            b'D' => Some(500),
            b'M' => Some(1000),
            _ => None,
        }
    };
    let bytes = s.as_bytes();
    let mut total = 0usize;
    for i in 0..bytes.len() {
        let cur = val(bytes[i])?;
        let next = if i + 1 < bytes.len() {
            val(bytes[i + 1])?
        } else {
            0
        };
        if cur < next {
            total = total.saturating_sub(cur);
        } else {
            total = total.saturating_add(cur);
        }
    }
    (total > 0).then_some(total)
}

fn parse_english_word_number(s: &str) -> Option<usize> {
    match s.to_ascii_lowercase().as_str() {
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "thirteen" => Some(13),
        "fourteen" => Some(14),
        "fifteen" => Some(15),
        "sixteen" => Some(16),
        "seventeen" => Some(17),
        "eighteen" => Some(18),
        "nineteen" => Some(19),
        "twenty" => Some(20),
        _ => None,
    }
}

fn normalize_subtitle(s: &str) -> String {
    let trimmed = s.trim_matches(|c: char| is_heading_ws(c) || matches!(c, ':' | '：' | '-' | '—'));
    trimmed.chars().filter(|c| !is_heading_ws(*c)).collect()
}

/// Strips an optional leading Chinese volume prefix like `第六卷 卷名 ...` or `六卷 卷名 ...`.
fn strip_optional_zh_volume_prefix(s: &str) -> &str {
    let rest = s
        .strip_prefix('第')
        .map_or(s, |r| r.trim_start_matches(is_heading_ws));
    let mut byte_after_num = 0usize;
    let mut num_chars = 0usize;
    for (b_idx, ch) in rest.char_indices() {
        if is_strict_zh_or_ascii_numeral_char(ch) || is_heading_ws(ch) {
            if !is_heading_ws(ch) {
                num_chars += 1;
            }
            byte_after_num = b_idx + ch.len_utf8();
        } else {
            break;
        }
    }
    if num_chars == 0 || num_chars > 10 {
        return s;
    }
    let after_num = &rest[byte_after_num..];
    let Some(first_ch) = after_num.chars().next() else {
        return s;
    };
    if !matches!(first_ch, '卷' | '部' | '集') {
        return s;
    }
    let after_vol_marker = &after_num[first_ch.len_utf8()..];
    let Some(ws1_ch) = after_vol_marker.chars().next() else {
        return s;
    };
    if !is_heading_ws(ws1_ch) {
        return s;
    }
    let after_ws1 = after_vol_marker.trim_start_matches(is_heading_ws);
    // Skip volume name token up to next whitespace.
    let mut vol_title_end = 0usize;
    for (b_idx, ch) in after_ws1.char_indices() {
        if is_heading_ws(ch) {
            vol_title_end = b_idx;
            break;
        }
    }
    if vol_title_end == 0 {
        return s;
    }
    let candidate = after_ws1[vol_title_end..].trim_start_matches(is_heading_ws);
    if candidate.is_empty() { s } else { candidate }
}

/// Strips an optional leading English `Book N` / `Volume N` / `Part N` prefix before `Chapter ...`.
fn strip_optional_en_volume_prefix(s: &str) -> &str {
    let lower = s.to_ascii_lowercase();
    let has_vol =
        lower.starts_with("book ") || lower.starts_with("volume ") || lower.starts_with("part ");
    if !has_vol {
        return s;
    }
    if let Some(pos) = lower.find("chapter ") {
        s.get(pos..).unwrap_or(s)
    } else {
        s
    }
}

fn match_heading_candidate(
    cand: &str,
    full_title: &str,
    is_prefixed_xa0: bool,
) -> Option<DetectedHeading> {
    let rest = strip_optional_en_volume_prefix(strip_optional_zh_volume_prefix(cand));

    // Form 1: `第 <numeral (tolerating <= 1 inner homophone)> [章節回节] [subtitle]`
    if let Some(after_di) = rest.strip_prefix('第') {
        let after_di_trim = after_di.trim_start_matches(is_heading_ws);
        // Find first chapter marker within first 14 chars.
        let mut marker_pos: Option<(usize, char)> = None;
        for (char_i, (b_idx, ch)) in after_di_trim.char_indices().enumerate() {
            if char_i > 14 {
                break;
            }
            if matches!(ch, '章' | '節' | '节' | '回') {
                marker_pos = Some((b_idx, ch));
                break;
            }
        }
        if let Some((m_byte, m_ch)) = marker_pos {
            let num_slice = &after_di_trim[..m_byte];
            if !num_slice.contains('，')
                && !num_slice.contains('。')
                && !num_slice.contains('！')
                && !num_slice.contains('？')
                && let Some(num) = parse_chapter_number(num_slice, true)
            {
                let after_marker = &after_di_trim[m_byte + m_ch.len_utf8()..];
                // Guard against narrative prose starting with e.g. "第一章的內容..."
                let prose_like = after_marker.chars().next().is_some_and(|c| {
                    !is_heading_ws(c)
                        && matches!(
                            c,
                            '的' | '中'
                                | '裡'
                                | '裏'
                                | '里'
                                | '內'
                                | '内'
                                | '是'
                                | '在'
                                | '有'
                                | '上'
                                | '下'
                                | '前'
                                | '後'
                                | '后'
                        )
                });
                if !prose_like {
                    return Some(DetectedHeading {
                        title: full_title.to_owned(),
                        number: Some(num),
                        norm_sub: normalize_subtitle(after_marker),
                        is_prefixed_xa0,
                    });
                }
            }
        }

        // Form 3: `第 <strict numeral (2..=12 chars)> <ws> <subtitle>` (missing `章` marker)
        let mut num_end_byte = 0usize;
        let mut strict_count = 0usize;
        for (b_idx, ch) in after_di_trim.char_indices() {
            if is_strict_zh_or_ascii_numeral_char(ch) {
                strict_count += 1;
                num_end_byte = b_idx + ch.len_utf8();
            } else {
                break;
            }
        }
        if (2..=12).contains(&strict_count) {
            let rem = &after_di_trim[num_end_byte..];
            if rem.chars().next().is_some_and(is_heading_ws) {
                let sub = rem.trim_start_matches(is_heading_ws);
                if !sub.is_empty()
                    && !sub.contains('，')
                    && let Some(num) = parse_chapter_number(&after_di_trim[..num_end_byte], false)
                {
                    return Some(DetectedHeading {
                        title: full_title.to_owned(),
                        number: Some(num),
                        norm_sub: normalize_subtitle(sub),
                        is_prefixed_xa0,
                    });
                }
            }
        }
    }

    // Form 2: `<strict numeral (2..=12 chars)> 章 (<ws + subtitle> | $)` (missing leading `第`)
    {
        let mut num_end_byte = 0usize;
        let mut strict_count = 0usize;
        for (b_idx, ch) in rest.char_indices() {
            if is_strict_zh_or_ascii_numeral_char(ch) || is_heading_ws(ch) {
                if !is_heading_ws(ch) {
                    strict_count += 1;
                }
                num_end_byte = b_idx + ch.len_utf8();
            } else {
                break;
            }
        }
        if (2..=12).contains(&strict_count) {
            let rem = &rest[num_end_byte..];
            if let Some(after_zhang) = rem.strip_prefix('章') {
                let valid_tail = after_zhang.is_empty()
                    || (after_zhang.chars().next().is_some_and(is_heading_ws)
                        && !after_zhang.trim_matches(is_heading_ws).is_empty());
                if valid_tail && let Some(num) = parse_chapter_number(&rest[..num_end_byte], false)
                {
                    return Some(DetectedHeading {
                        title: full_title.to_owned(),
                        number: Some(num),
                        norm_sub: normalize_subtitle(after_zhang),
                        is_prefixed_xa0,
                    });
                }
            }
        }
    }

    // Form 4: English `Chapter <digits | roman | word> [delimiter subtitle]`
    if let Some(after_kw) = rest.get(7..)
        && rest[..7].eq_ignore_ascii_case("chapter")
        && after_kw.chars().next().is_some_and(is_heading_ws)
    {
        let after_ws = after_kw.trim_start_matches(is_heading_ws);
        let mut tok_end = 0usize;
        for (b_idx, ch) in after_ws.char_indices() {
            if ch.is_ascii_alphanumeric() || ch == '-' {
                tok_end = b_idx + ch.len_utf8();
            } else {
                break;
            }
        }
        if tok_end > 0 {
            let tok = &after_ws[..tok_end];
            let tail = &after_ws[tok_end..];
            let valid_boundary = tail.is_empty()
                || tail.chars().next().is_some_and(|c| {
                    is_heading_ws(c) || matches!(c, ':' | '.' | '-' | '—' | ')' | '(')
                });
            let num = if tok.bytes().all(|b| b.is_ascii_digit()) {
                tok.parse::<usize>().ok()
            } else {
                parse_english_word_number(tok).or_else(|| parse_roman_numeral(tok))
            };
            if valid_boundary && num.is_some() {
                return Some(DetectedHeading {
                    title: full_title.to_owned(),
                    number: num,
                    norm_sub: normalize_subtitle(tail).to_ascii_lowercase(),
                    is_prefixed_xa0,
                });
            }
        }
    }

    None
}

fn detect_chapter_heading(raw_line: &str) -> Option<DetectedHeading> {
    let trimmed = raw_line.trim_matches(is_heading_ws);
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().count() > 80 {
        return None;
    }
    if trimmed.contains('。') || trimmed.contains('；') || trimmed.ends_with('.') {
        return None;
    }
    // Markdown ATX heading (`# Title`, `## Chapter 1`, etc.)
    if let Some(after_hashes) = trimmed.strip_prefix('#') {
        let hashes_stripped = after_hashes.trim_start_matches('#');
        let hash_count = 1 + (after_hashes.len() - hashes_stripped.len());
        if hash_count <= 6 && hashes_stripped.chars().next().is_some_and(is_heading_ws) {
            let md_title = hashes_stripped.trim_matches(is_heading_ws);
            if !md_title.is_empty() {
                if let Some(mut hd) = match_heading_candidate(md_title, trimmed, false) {
                    hd.title = md_title.to_owned();
                    return Some(hd);
                }
                return Some(DetectedHeading {
                    title: md_title.to_owned(),
                    number: None,
                    norm_sub: normalize_subtitle(md_title),
                    is_prefixed_xa0: false,
                });
            }
        }
    }
    if let Some(hd) = match_heading_candidate(trimmed, trimmed, false) {
        return Some(hd);
    }
    if let Some((_prefix, after_xa0)) = trimmed.split_once('\u{00a0}') {
        let after_trimmed = after_xa0.trim_matches(is_heading_ws);
        if !after_trimmed.is_empty()
            && let Some(hd) = match_heading_candidate(after_trimmed, trimmed, true)
        {
            return Some(hd);
        }
    }
    None
}

/// Detects a 4-line web-novel chapter header block:
/// `line[idx] = <title>` (at SOF or after a blank line), `line[idx + 1]` blank, and
/// `line[idx + 2]` equal to `<non-empty book prefix>[\u{00a0}\u{3000}]+<title>`.
fn detect_paired_banner_heading(lines: &[RawLine<'_>], idx: usize) -> Option<DetectedHeading> {
    if idx > 0 && !is_blank_line(lines[idx - 1].raw) {
        return None;
    }
    let line0 = lines[idx].raw.trim_matches(is_heading_ws);
    if line0.is_empty() || line0.chars().count() > 80 || line0.contains('。') {
        return None;
    }
    let line1 = lines.get(idx + 1)?.raw;
    if !is_blank_line(line1) {
        return None;
    }
    let line2 = lines.get(idx + 2)?.raw.trim_matches(is_heading_ws);
    let prefix = line2.strip_suffix(line0)?;
    let has_nbsp_sep = prefix
        .chars()
        .last()
        .is_some_and(|c| matches!(c, '\u{00a0}' | '\u{3000}'));
    let book_prefix = prefix.trim_end_matches(|c: char| is_heading_ws(c));
    if !has_nbsp_sep || book_prefix.is_empty() {
        return None;
    }
    if let Some(mut hd) = detect_chapter_heading(lines[idx].raw) {
        hd.is_prefixed_xa0 = false;
        return Some(hd);
    }
    Some(DetectedHeading {
        title: line0.to_owned(),
        number: None,
        norm_sub: normalize_subtitle(line0),
        is_prefixed_xa0: false,
    })
}

fn scan_lines(text: &str) -> Vec<RawLine<'_>> {
    let mut out = Vec::new();
    let mut start_byte = 0usize;
    let mut start_char = 0usize;
    let mut cur_char = 0usize;

    for (b_idx, ch) in text.char_indices() {
        cur_char += 1;
        if ch == '\n' {
            let end_byte = b_idx + 1;
            out.push(RawLine {
                raw: &text[start_byte..end_byte],
                start_char,
                end_char: cur_char,
                start_byte,
                end_byte,
            });
            start_byte = end_byte;
            start_char = cur_char;
        }
    }
    if start_byte < text.len() {
        out.push(RawLine {
            raw: &text[start_byte..],
            start_char,
            end_char: cur_char,
            start_byte,
            end_byte: text.len(),
        });
    }
    // Strip leading UTF-8 BOM from line 0 if present so chapter slices start cleanly after BOM.
    if let Some(first) = out.first_mut()
        && let Some(stripped) = first.raw.strip_prefix('\u{feff}')
    {
        first.raw = stripped;
        first.start_char += 1;
        first.start_byte += '\u{feff}'.len_utf8();
    }
    out
}

struct OpenChapter {
    title: String,
    number: Option<usize>,
    norm_sub: String,
    start_line: usize,
    has_body: bool,
    body_chars: usize,
}

fn split_bundle_text(
    source_file: &str,
    bundle_text: &str,
    source_sha256: &str,
    next_order: &mut usize,
) -> Vec<InputChapter> {
    let lines = scan_lines(bundle_text);
    if lines.is_empty() {
        return Vec::new();
    }

    let mut raw_chapters: Vec<(String, usize, usize)> = Vec::new();
    let mut cur: Option<OpenChapter> = None;

    for (idx, line) in lines.iter().enumerate() {
        let paired_banner = detect_paired_banner_heading(&lines, idx);
        let is_paired = paired_banner.is_some();
        if let Some(hd) = paired_banner.or_else(|| detect_chapter_heading(line.raw)) {
            match &mut cur {
                None => {
                    cur = Some(OpenChapter {
                        title: hd.title,
                        number: hd.number,
                        norm_sub: hd.norm_sub,
                        start_line: idx,
                        has_body: false,
                        body_chars: 0,
                    });
                }
                Some(open) => {
                    let same_num = hd.number.is_some() && open.number == hd.number;
                    let is_duplicate_banner = !is_paired
                        && (!open.has_body
                            || hd.is_prefixed_xa0
                            || (same_num
                                && (open.body_chars < 200
                                    || hd.norm_sub.is_empty()
                                    || hd.norm_sub == open.norm_sub)));
                    if !is_duplicate_banner {
                        raw_chapters.push((open.title.clone(), open.start_line, idx));
                        cur = Some(OpenChapter {
                            title: hd.title,
                            number: hd.number,
                            norm_sub: hd.norm_sub,
                            start_line: idx,
                            has_body: false,
                            body_chars: 0,
                        });
                    }
                }
            }
        } else if let Some(open) = &mut cur {
            let trimmed = line.raw.trim_matches(is_heading_ws);
            if !trimmed.is_empty() {
                open.has_body = true;
                open.body_chars += trimmed.chars().count();
            }
        }
    }

    if let Some(open) = cur {
        raw_chapters.push((open.title, open.start_line, lines.len()));
    }

    // Fallback: if a non-empty file has no numbered chapter headings, treat the whole file as one chapter.
    if raw_chapters.is_empty()
        && let Some(first_non_blank) = lines.iter().position(|l| !is_blank_line(l.raw))
    {
        let title = lines[first_non_blank]
            .raw
            .trim_matches(is_heading_ws)
            .to_owned();
        raw_chapters.push((title, first_non_blank, lines.len()));
    }

    let mut out = Vec::with_capacity(raw_chapters.len());
    for (title, start_line, mut end_line) in raw_chapters {
        while end_line > start_line + 1 && is_blank_line(lines[end_line - 1].raw) {
            end_line -= 1;
        }
        let start_char = lines[start_line].start_char;
        let end_char = lines[end_line - 1].end_char;
        let start_byte = lines[start_line].start_byte;
        let end_byte = lines[end_line - 1].end_byte;
        let slice = &bundle_text[start_byte..end_byte];
        debug_assert_eq!(slice.chars().count(), end_char - start_char);

        let order = *next_order;
        *next_order += 1;
        out.push(InputChapter {
            order,
            title,
            source_file: source_file.to_owned(),
            source_sha256: source_sha256.to_owned(),
            source_span: SourceSpan {
                start_char,
                end_char,
                start_byte,
                end_byte,
            },
            chapter_sha256: sha256_hex(slice.as_bytes()),
            text: slice.to_owned(),
        });
    }
    out
}

fn load_inputs_from_bundles_dir(bundles_dir: &Path) -> Result<Vec<InputChapter>> {
    let mut entries: Vec<(String, PathBuf)> = Vec::new();
    for entry in fs::read_dir(bundles_dir)
        .with_context(|| format!("read bundles directory {}", bundles_dir.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_file() {
            continue;
        }
        let file_name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow!("non-UTF-8 filename in {}", bundles_dir.display()))?;
        if file_name.starts_with('.') {
            continue;
        }
        let is_txt = Path::new(&file_name)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("txt"));
        if is_txt {
            entries.push((file_name, entry.path()));
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    if entries.is_empty() {
        bail!("no .txt bundle files found in {}", bundles_dir.display());
    }

    let mut chapters = Vec::new();
    let mut next_order = 1usize;
    for (file_name, path) in entries {
        let bytes = fs::read(&path).with_context(|| format!("read bundle {}", path.display()))?;
        let text = std::str::from_utf8(&bytes)
            .with_context(|| format!("bundle {} is not valid UTF-8", path.display()))?;
        let bundle_sha = sha256_hex(&bytes);
        let extracted = split_bundle_text(&file_name, text, &bundle_sha, &mut next_order);
        chapters.extend(extracted);
    }
    if chapters.is_empty() {
        bail!(
            "no non-empty chapters extracted from {}",
            bundles_dir.display()
        );
    }
    Ok(chapters)
}

fn load_inputs_from_manifest(manifest_path: &Path) -> Result<Vec<InputChapter>> {
    let raw = fs::read(manifest_path)
        .with_context(|| format!("read manifest {}", manifest_path.display()))?;
    let parsed: InputManifest = serde_json::from_slice(&raw)
        .with_context(|| format!("parse manifest JSON {}", manifest_path.display()))?;
    let base_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));

    let mut entries = parsed.chapters;
    entries.sort_by_key(|c| c.order);

    let mut seen_orders = HashSet::with_capacity(entries.len());
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some(status) = &entry.status
            && !status.eq_ignore_ascii_case("ok")
        {
            eprintln!(
                "nlp_novel_batch: skipping order={} ({}) with status={status}",
                entry.order, entry.file
            );
            continue;
        }
        if entry.order == 0 {
            bail!(
                "manifest chapter order must be >= 1 (found 0 for {})",
                entry.file
            );
        }
        if !seen_orders.insert(entry.order) {
            bail!("duplicate chapter order {} in manifest", entry.order);
        }

        let file_path = base_dir.join(&entry.file);
        let bytes = fs::read(&file_path)
            .with_context(|| format!("read chapter file {}", file_path.display()))?;
        let mut text = String::from_utf8(bytes)
            .with_context(|| format!("chapter file {} is not valid UTF-8", file_path.display()))?;
        if text.starts_with('\u{feff}') {
            text.remove(0);
        }
        let sha = sha256_hex(text.as_bytes());
        let char_count = text.chars().count();
        let byte_count = text.len();
        let title = match entry.title {
            Some(t) if !t.trim().is_empty() => t,
            _ => text
                .lines()
                .map(|l| l.trim_matches(is_heading_ws))
                .find(|l| !l.is_empty())
                .unwrap_or("")
                .to_owned(),
        };

        out.push(InputChapter {
            order: entry.order,
            title,
            source_file: entry.file,
            source_sha256: sha.clone(),
            source_span: SourceSpan {
                start_char: 0,
                end_char: char_count,
                start_byte: 0,
                end_byte: byte_count,
            },
            chapter_sha256: sha,
            text,
        });
    }
    if out.is_empty() {
        bail!("no active chapters found in {}", manifest_path.display());
    }
    Ok(out)
}

// ============================================================================
// Stats & Timeline Cue Extraction
// ============================================================================

fn reason_snake_str(r: Reason) -> &'static str {
    match r {
        Reason::NoModel => "no_model",
        Reason::Ambiguous => "ambiguous",
        Reason::LowConfidence => "low_confidence",
        Reason::UnsupportedLanguage => "unsupported_language",
        Reason::NotApplicable => "not_applicable",
    }
}

fn entity_kind_snake_str(k: EntityKind) -> Option<&'static str> {
    match k {
        EntityKind::Date => Some("date"),
        EntityKind::RelativeDate => Some("relative_date"),
        EntityKind::Time => Some("time"),
        EntityKind::Quantity => Some("quantity"),
        EntityKind::Number => Some("number"),
        EntityKind::Url | EntityKind::Email | EntityKind::Name => None,
    }
}

fn relation_kind_snake_str(k: RelationKind) -> &'static str {
    match k {
        RelationKind::Sequence => "sequence",
        RelationKind::Cause => "cause",
        RelationKind::Contrast => "contrast",
        RelationKind::Parallel => "parallel",
        RelationKind::Condition => "condition",
    }
}

fn char_to_byte_offsets(text: &str) -> Vec<usize> {
    let mut table = Vec::with_capacity(text.chars().count() + 1);
    for (b, _) in text.char_indices() {
        table.push(b);
    }
    table.push(text.len());
    table
}

fn slice_span<'a>(text: &'a str, char_to_byte: &[usize], span: Span) -> Option<&'a str> {
    let start = *char_to_byte.get(span.start_char)?;
    let end = *char_to_byte.get(span.end_char)?;
    (start <= end && end <= text.len()).then(|| &text[start..end])
}

fn extract_timeline_cues(
    order: usize,
    title: &str,
    source_file: &str,
    source_start_char: usize,
    chapter_text: Option<&str>,
    doc: &Document,
) -> Vec<TimelineCueRecord> {
    let char_table = chapter_text.map(char_to_byte_offsets);
    let text_for = |span: Span| -> Option<String> {
        let t = chapter_text?;
        let tbl = char_table.as_deref()?;
        slice_span(t, tbl, span).map(str::to_owned)
    };
    let to_bundle = |s: Span| Span {
        start_char: source_start_char + s.start_char,
        end_char: source_start_char + s.end_char,
    };

    let mut cues = Vec::new();
    for ent in &doc.entities {
        let Some(kind_str) = entity_kind_snake_str(ent.kind) else {
            continue;
        };
        cues.push(TimelineCueRecord {
            schema: CUE_SCHEMA.to_owned(),
            chapter: order,
            chapter_title: title.to_owned(),
            source_file: source_file.to_owned(),
            cue_type: "entity".to_owned(),
            kind: kind_str.to_owned(),
            span: ent.span,
            bundle_span: to_bundle(ent.span),
            sentence: Some(ent.sentence),
            level: None,
            text: text_for(ent.span),
            value: ent.value.clone(),
            unit: ent.unit.clone(),
            connector: None,
            connector_text: None,
            correlate: None,
            correlate_text: None,
            arg1: None,
            arg2: None,
        });
    }

    for rel in &doc.relations {
        let start_char = rel
            .arg1
            .start_char
            .min(rel.arg2.start_char)
            .min(rel.connector.start_char);
        let end_char = rel
            .arg1
            .end_char
            .max(rel.arg2.end_char)
            .max(rel.connector.end_char);
        let span = Span {
            start_char,
            end_char,
        };
        cues.push(TimelineCueRecord {
            schema: CUE_SCHEMA.to_owned(),
            chapter: order,
            chapter_title: title.to_owned(),
            source_file: source_file.to_owned(),
            cue_type: "relation".to_owned(),
            kind: relation_kind_snake_str(rel.kind).to_owned(),
            span,
            bundle_span: to_bundle(span),
            sentence: None,
            level: Some(rel.level),
            text: None,
            value: None,
            unit: None,
            connector: Some(rel.connector),
            connector_text: text_for(rel.connector),
            correlate: rel.correlate,
            correlate_text: rel.correlate.and_then(&text_for),
            arg1: Some(rel.arg1),
            arg2: Some(rel.arg2),
        });
    }

    cues.sort_by(|a, b| {
        (
            a.span.start_char,
            a.span.end_char,
            a.cue_type.as_str(),
            a.kind.as_str(),
        )
            .cmp(&(
                b.span.start_char,
                b.span.end_char,
                b.cue_type.as_str(),
                b.kind.as_str(),
            ))
    });
    cues
}

fn summarize_document(doc: &Document, timeline_cues: usize) -> ChapterStats {
    let unsupported_sents = doc
        .unknown
        .iter()
        .filter(|u| u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage)
        .count();
    let decided_upos_tokens = doc.tokens.iter().filter(|t| t.upos.is_some()).count();
    let mut unknown_by_layer_reason = BTreeMap::new();
    for u in &doc.unknown {
        let key = format!("{}:{}", u.layer, reason_snake_str(u.reason));
        *unknown_by_layer_reason.entry(key).or_insert(0) += 1;
    }

    ChapterStats {
        text_chars: doc.text_chars,
        paragraphs: doc.segmentation.paragraphs.len(),
        sentences: doc.segmentation.sentences.len(),
        supported_sentences: doc
            .segmentation
            .sentences
            .len()
            .saturating_sub(unsupported_sents),
        clauses: doc.segmentation.clauses.len(),
        tokens: doc.tokens.len(),
        decided_upos_tokens,
        syntax_tokens: doc.syntax.len(),
        entities: doc.entities.len(),
        relations: doc.relations.len(),
        timeline_cues,
        unknown_by_layer_reason,
    }
}

fn chapter_rel_output_path(order: usize) -> String {
    format!("chapters/{order:04}.json")
}

fn index_entry_matches_chapter(
    entry: &CompletedIndexEntry,
    ch: &InputChapter,
    expected_model_id: Option<&str>,
    expected_model_sha256: Option<&str>,
    expected_backend: &str,
) -> bool {
    entry.schema == INDEX_SCHEMA
        && entry.order == ch.order
        && entry.title == ch.title
        && entry.source_file == ch.source_file
        && entry.source_sha256 == ch.source_sha256
        && entry.source_span == ch.source_span
        && entry.chapter_sha256 == ch.chapter_sha256
        && entry.model_id.as_deref() == expected_model_id
        && entry.model_sha256.as_deref() == expected_model_sha256
        && entry.backend == expected_backend
}

fn chapter_output_matches_chapter(
    out: &ChapterOutput,
    ch: &InputChapter,
    expected_model_id: Option<&str>,
    expected_model_sha256: Option<&str>,
    expected_backend: &str,
) -> bool {
    out.schema == CHAPTER_SCHEMA
        && out.order == ch.order
        && out.title == ch.title
        && out.source_file == ch.source_file
        && out.source_sha256 == ch.source_sha256
        && out.source_span == ch.source_span
        && out.chapter_sha256 == ch.chapter_sha256
        && out.doc.text_sha256 == ch.chapter_sha256
        && out.model_id.as_deref() == expected_model_id
        && out.model_sha256.as_deref() == expected_model_sha256
        && out.backend == expected_backend
}

fn load_completed_index(index_path: &Path) -> HashMap<usize, CompletedIndexEntry> {
    let Ok(file) = File::open(index_path) else {
        return HashMap::new();
    };
    let reader = BufReader::new(file);
    let mut map = HashMap::new();
    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<CompletedIndexEntry>(trimmed) {
            map.insert(entry.order, entry);
        }
    }
    map
}

fn compute_chapter_outputs_sha256(entries: &[CompletedIndexEntry]) -> String {
    let mut hasher = Sha256::new();
    for e in entries {
        hasher.update(format!("{:04}:{}\n", e.order, e.output_sha256).as_bytes());
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
fn with_platform_pool<T, F: FnOnce() -> T>(f: F) -> T {
    objc2::rc::autoreleasepool(|_| f())
}

#[cfg(not(all(feature = "coreml", target_os = "macos")))]
fn with_platform_pool<T, F: FnOnce() -> T>(f: F) -> T {
    f()
}

fn run() -> Result<()> {
    let start_instant = Instant::now();
    let args = Args::parse();

    let (input_mode, chapters) = match (&args.bundles_dir, &args.manifest) {
        (Some(dir), None) => ("bundles_dir", load_inputs_from_bundles_dir(dir)?),
        (None, Some(manifest)) => ("manifest", load_inputs_from_manifest(manifest)?),
        (None, None) => bail!("must specify either --bundles-dir <DIR> or --manifest <PATH>"),
        (Some(_), Some(_)) => bail!("--bundles-dir and --manifest are mutually exclusive"),
    };

    let backend_mode = match (args.backend, &args.models) {
        (Some(BackendMode::None), _) => BackendMode::None,
        (Some(mode), Some(_)) => mode,
        (Some(mode), None) => {
            bail!(
                "--backend {} requires --models <DIR> (or pass --backend none)",
                mode.as_str()
            );
        }
        (None, Some(_)) => default_backend_for_models()?,
        (None, None) => BackendMode::None,
    };

    if !backend_mode.is_compiled() {
        bail!(
            "backend '{}' is not compiled into this binary (rebuild with the corresponding Cargo feature)",
            backend_mode.as_str()
        );
    }

    let requested_workers = match args.workers {
        Some(0) => bail!("--workers must be >= 1"),
        Some(n) => n,
        None => std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .max(1),
    };
    let resume_enabled = !args.no_resume;
    let doc_options = Options {
        names: args.names.clone(),
    };

    // Verify model bytes ONCE up front when neural backend is active.
    let verified_opt: Option<Verified> = if backend_mode != BackendMode::None {
        let models_dir = args
            .models
            .as_deref()
            .expect("--models checked present for active backend");
        let v = faberline_nlp::doc::model::verify(models_dir, &args.model_id).map_err(|e| {
            anyhow!(
                "verify model '{}' in {}: {e}",
                args.model_id,
                models_dir.display()
            )
        })?;
        Some(v)
    } else {
        None
    };

    let expected_model_id: Option<String> = verified_opt.as_ref().map(|v| v.info.id.clone());
    let expected_model_sha256: Option<String> = verified_opt.as_ref().map(compute_model_sha256);
    let backend_str = backend_mode.as_str();

    let chapters_out_dir = args.out_dir.join("chapters");
    fs::create_dir_all(&chapters_out_dir)
        .with_context(|| format!("create output directory {}", chapters_out_dir.display()))?;
    let index_path = args.out_dir.join("completed_index.jsonl");
    let cues_path = args.out_dir.join("entities_timeline_cues.jsonl");
    let receipt_path = args.out_dir.join("book_manifest_receipt.json");

    let total_chapters = chapters.len();
    let mut completed_slots: Vec<Option<CompletedIndexEntry>> = vec![None; total_chapters];
    let mut cues_slots: Vec<Option<Vec<TimelineCueRecord>>> = vec![None; total_chapters];
    let mut disk_check_indices: Vec<usize> = Vec::new();
    let mut pending_indices: Vec<usize> = Vec::new();

    if resume_enabled {
        let cached_index = load_completed_index(&index_path);
        for (idx, ch) in chapters.iter().enumerate() {
            let rel_path = chapter_rel_output_path(ch.order);
            let full_path = args.out_dir.join(&rel_path);
            let Ok(meta) = fs::metadata(&full_path) else {
                pending_indices.push(idx);
                continue;
            };
            if !meta.is_file() {
                pending_indices.push(idx);
                continue;
            }
            if let Some(entry) = cached_index.get(&ch.order)
                && entry.file_len == meta.len()
                && index_entry_matches_chapter(
                    entry,
                    ch,
                    expected_model_id.as_deref(),
                    expected_model_sha256.as_deref(),
                    backend_str,
                )
            {
                completed_slots[idx] = Some(entry.clone());
            } else {
                disk_check_indices.push(idx);
            }
        }

        // Verify any unindexed existing chapter JSONs on disk in parallel.
        if !disk_check_indices.is_empty() {
            let check_workers = requested_workers.min(disk_check_indices.len()).max(1);
            let next_disk = AtomicUsize::new(0);
            type DiskVerifyResult =
                Vec<(usize, Option<(CompletedIndexEntry, Vec<TimelineCueRecord>)>)>;
            let verified_from_disk: Mutex<DiskVerifyResult> = Mutex::new(Vec::new());

            std::thread::scope(|s| {
                for _ in 0..check_workers {
                    s.spawn(|| {
                        let mut local = Vec::new();
                        loop {
                            let pos = next_disk.fetch_add(1, Ordering::Relaxed);
                            if pos >= disk_check_indices.len() {
                                break;
                            }
                            let idx = disk_check_indices[pos];
                            let ch = &chapters[idx];
                            let rel_path = chapter_rel_output_path(ch.order);
                            let full_path = args.out_dir.join(&rel_path);
                            let parsed_ok = fs::read(&full_path).ok().and_then(|bytes| {
                                let out: ChapterOutput = serde_json::from_slice(&bytes).ok()?;
                                if !chapter_output_matches_chapter(
                                    &out,
                                    ch,
                                    expected_model_id.as_deref(),
                                    expected_model_sha256.as_deref(),
                                    backend_str,
                                ) {
                                    return None;
                                }
                                let cues = extract_timeline_cues(
                                    ch.order,
                                    &ch.title,
                                    &ch.source_file,
                                    ch.source_span.start_char,
                                    Some(&ch.text),
                                    &out.doc,
                                );
                                let stats = summarize_document(&out.doc, cues.len());
                                let entry = CompletedIndexEntry {
                                    schema: INDEX_SCHEMA.to_owned(),
                                    order: ch.order,
                                    title: ch.title.clone(),
                                    source_file: ch.source_file.clone(),
                                    source_sha256: ch.source_sha256.clone(),
                                    source_span: ch.source_span,
                                    chapter_sha256: ch.chapter_sha256.clone(),
                                    model_id: out.model_id,
                                    model_sha256: out.model_sha256,
                                    backend: out.backend,
                                    output_file: rel_path,
                                    output_sha256: sha256_hex(&bytes),
                                    file_len: bytes.len() as u64,
                                    stats,
                                };
                                Some((entry, cues))
                            });
                            local.push((idx, parsed_ok));
                        }
                        verified_from_disk.lock().unwrap().extend(local);
                    });
                }
            });

            let mut recovered = verified_from_disk.into_inner().unwrap();
            recovered.sort_by_key(|(idx, _)| *idx);
            for (idx, maybe_hit) in recovered {
                match maybe_hit {
                    Some((entry, cues)) => {
                        completed_slots[idx] = Some(entry);
                        cues_slots[idx] = Some(cues);
                    }
                    None => pending_indices.push(idx),
                }
            }
            pending_indices.sort_unstable();
        }
    } else {
        pending_indices.extend(0..total_chapters);
    }

    let resumed_chapters = total_chapters - pending_indices.len();
    let processed_chapters = pending_indices.len();
    let effective_workers = if pending_indices.is_empty() {
        requested_workers
    } else {
        requested_workers.min(pending_indices.len()).max(1)
    };

    if resume_enabled && resumed_chapters > 0 {
        eprintln!(
            "[resume] skipped {resumed_chapters}/{total_chapters} verified chapters, {processed_chapters} pending (workers={effective_workers}, backend={backend_str})"
        );
    }

    let mut newly_processed_sentences = 0usize;

    if !pending_indices.is_empty() {
        if verified_opt.is_some() {
            eprintln!(
                "[init] initializing {effective_workers} independent {backend_str} worker instances for {processed_chapters} chapters..."
            );
        }
        let journal_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&index_path)
            .with_context(|| format!("open journal {}", index_path.display()))?;
        let journal = Mutex::new(BufWriter::new(journal_file));

        let init_gate = Mutex::new(());
        let next_pending = AtomicUsize::new(0);
        let done_pending = AtomicUsize::new(0);
        let done_sents = AtomicUsize::new(0);
        let worker_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);
        type WorkerBatch = Vec<(usize, CompletedIndexEntry, Vec<TimelineCueRecord>)>;
        let collected: Mutex<WorkerBatch> = Mutex::new(Vec::with_capacity(pending_indices.len()));
        let worker_phase_start = Instant::now();

        std::thread::scope(|s| {
            for worker_id in 0..effective_workers {
                let verified_ref = verified_opt.as_ref();
                let doc_options_ref = &doc_options;
                let expected_model_id_ref = expected_model_id.as_deref();
                let expected_model_sha_ref = expected_model_sha256.as_deref();
                let chapters_ref = &chapters;
                let pending_ref = &pending_indices;
                let out_dir_ref = &args.out_dir;
                let journal_ref = &journal;
                let init_gate_ref = &init_gate;
                let next_pending_ref = &next_pending;
                let done_pending_ref = &done_pending;
                let done_sents_ref = &done_sents;
                let worker_err_ref = &worker_err;
                let collected_ref = &collected;

                s.spawn(move || {
                    let record_err = |err: anyhow::Error| {
                        let mut guard = worker_err_ref.lock().unwrap();
                        if guard.is_none() {
                            *guard = Some(err);
                        }
                    };

                    #[cfg(any(feature = "onnx", feature = "coreml"))]
                    let mut thread_runtime: Option<(
                        Box<dyn faberline_nlp::doc::model::Model>,
                        SubwordTokenizer,
                    )> = None;

                    #[cfg(any(feature = "onnx", feature = "coreml"))]
                    if let (Some(verified), Some(doc_backend)) =
                        (verified_ref, backend_mode.to_doc_backend())
                    {
                        let init_res = {
                            let _guard = init_gate_ref.lock().unwrap();
                            with_platform_pool(|| -> Result<_> {
                                let model =
                                    faberline_nlp::doc::model::load_verified(verified, doc_backend)
                                        .map_err(|e| anyhow!("worker {worker_id} load model: {e}"))?;
                                let tok = SubwordTokenizer::from_verified(verified)
                                    .map_err(|e| anyhow!("worker {worker_id} load tokenizer: {e}"))?;
                                Ok((model, tok))
                            })
                        };
                        match init_res {
                            Ok(pair) => thread_runtime = Some(pair),
                            Err(err) => {
                                record_err(err);
                                return;
                            }
                        }
                    }

                    #[cfg(not(any(feature = "onnx", feature = "coreml")))]
                    {
                        let _ = (verified_ref, init_gate_ref);
                    }

                    let mut local_out = Vec::new();
                    loop {
                        if worker_err_ref.lock().unwrap().is_some() {
                            break;
                        }
                        let slot = next_pending_ref.fetch_add(1, Ordering::Relaxed);
                        if slot >= pending_ref.len() {
                            break;
                        }
                        let chap_idx = pending_ref[slot];
                        let ch = &chapters_ref[chap_idx];

                        let step_res = with_platform_pool(|| -> Result<(CompletedIndexEntry, Vec<TimelineCueRecord>)> {
                            let doc = {
                                #[cfg(any(feature = "onnx", feature = "coreml"))]
                                {
                                    if let Some((model, tokenizer)) = &thread_runtime {
                                        analyze_document_with_loaded_model(
                                            &ch.text,
                                            doc_options_ref,
                                            &**model,
                                            tokenizer,
                                        )
                                        .map_err(|e| {
                                            anyhow!(
                                                "chapter {} ({}) syntax inference failed: {e}",
                                                ch.order,
                                                ch.title
                                            )
                                        })?
                                    } else {
                                        analyze_document(&ch.text, doc_options_ref)
                                    }
                                }
                                #[cfg(not(any(feature = "onnx", feature = "coreml")))]
                                {
                                    analyze_document(&ch.text, doc_options_ref)
                                }
                            };

                            let cues = extract_timeline_cues(
                                ch.order,
                                &ch.title,
                                &ch.source_file,
                                ch.source_span.start_char,
                                Some(&ch.text),
                                &doc,
                            );
                            let stats = summarize_document(&doc, cues.len());

                            let chapter_output = ChapterOutput {
                                schema: CHAPTER_SCHEMA.to_owned(),
                                order: ch.order,
                                title: ch.title.clone(),
                                source_file: ch.source_file.clone(),
                                source_sha256: ch.source_sha256.clone(),
                                source_span: ch.source_span,
                                chapter_sha256: ch.chapter_sha256.clone(),
                                model_id: expected_model_id_ref.map(str::to_owned),
                                model_sha256: expected_model_sha_ref.map(str::to_owned),
                                backend: backend_str.to_owned(),
                                doc,
                            };

                            let rel_path = chapter_rel_output_path(ch.order);
                            let full_path = out_dir_ref.join(&rel_path);
                            let mut json_bytes = serde_json::to_vec(&chapter_output)?;
                            json_bytes.push(b'\n');
                            let output_sha256 = sha256_hex(&json_bytes);
                            let tag = format!("w{worker_id}.c{}", ch.order);
                            atomic_write_bytes(&full_path, &json_bytes, &tag)?;

                            let entry = CompletedIndexEntry {
                                schema: INDEX_SCHEMA.to_owned(),
                                order: ch.order,
                                title: ch.title.clone(),
                                source_file: ch.source_file.clone(),
                                source_sha256: ch.source_sha256.clone(),
                                source_span: ch.source_span,
                                chapter_sha256: ch.chapter_sha256.clone(),
                                model_id: expected_model_id_ref.map(str::to_owned),
                                model_sha256: expected_model_sha_ref.map(str::to_owned),
                                backend: backend_str.to_owned(),
                                output_file: rel_path,
                                output_sha256,
                                file_len: json_bytes.len() as u64,
                                stats,
                            };
                            Ok((entry, cues))
                        });

                        match step_res {
                            Ok((entry, cues)) => {
                                let sent_count = entry.stats.sentences;
                                if let Ok(line) = serde_json::to_string(&entry)
                                    && let Ok(mut j) = journal_ref.lock()
                                {
                                    let _ = writeln!(j, "{line}");
                                    let _ = j.flush();
                                }

                                let total_sents_done =
                                    done_sents_ref.fetch_add(sent_count, Ordering::Relaxed)
                                        + sent_count;
                                let newly_done =
                                    done_pending_ref.fetch_add(1, Ordering::Relaxed) + 1;
                                let overall_done = resumed_chapters + newly_done;

                                if overall_done.is_multiple_of(25)
                                    || overall_done == total_chapters
                                {
                                    let elapsed = worker_phase_start.elapsed().as_secs_f64().max(1e-6);
                                    let ch_per_sec = newly_done as f64 / elapsed;
                                    let sents_per_sec = total_sents_done as f64 / elapsed;
                                    let remaining = pending_ref.len().saturating_sub(newly_done);
                                    let eta = if ch_per_sec > 0.0 {
                                        remaining as f64 / ch_per_sec
                                    } else {
                                        0.0
                                    };
                                    eprintln!(
                                        "[{overall_done}/{total_chapters}] ch={:04} elapsed={elapsed:.1}s chapters/sec={ch_per_sec:.2} sents/sec={sents_per_sec:.1} ETA={eta:.1}s",
                                        ch.order
                                    );
                                }

                                local_out.push((chap_idx, entry, cues));
                            }
                            Err(err) => {
                                record_err(err);
                                break;
                            }
                        }
                    }

                    collected_ref.lock().unwrap().extend(local_out);
                });
            }
        });

        if let Some(err) = worker_err.into_inner().unwrap() {
            return Err(err);
        }

        newly_processed_sentences = done_sents.load(Ordering::Relaxed);
        for (chap_idx, entry, cues) in collected.into_inner().unwrap() {
            completed_slots[chap_idx] = Some(entry);
            cues_slots[chap_idx] = Some(cues);
        }
    }

    let ordered_entries: Vec<CompletedIndexEntry> = completed_slots
        .into_iter()
        .enumerate()
        .map(|(i, slot)| {
            slot.ok_or_else(|| anyhow!("missing completed index slot for chapter index {i}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let chapter_outputs_sha256 = compute_chapter_outputs_sha256(&ordered_entries);

    // Fast-path check for 100% resumed run where `entities_timeline_cues.jsonl` is already intact.
    let reusable_cues_sha256: Option<String> =
        if processed_chapters == 0 && cues_path.is_file() && receipt_path.is_file() {
            let prev_receipt = fs::read(&receipt_path)
                .ok()
                .and_then(|b| serde_json::from_slice::<BookManifestReceipt>(&b).ok());
            if let Some(prev) = prev_receipt
                && prev.chapter_outputs_sha256 == chapter_outputs_sha256
                && let Ok(cues_bytes) = fs::read(&cues_path)
            {
                let actual_cues_sha = sha256_hex(&cues_bytes);
                (actual_cues_sha == prev.timeline_cues_sha256).then_some(actual_cues_sha)
            } else {
                None
            }
        } else {
            None
        };

    let timeline_cues_sha256 = if let Some(sha) = reusable_cues_sha256 {
        sha
    } else {
        // Populate any missing `cues_slots` from existing `chapters/*.json` in parallel.
        let missing_cue_indices: Vec<usize> = cues_slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.is_none().then_some(idx))
            .collect();

        if !missing_cue_indices.is_empty() {
            let load_workers = requested_workers.min(missing_cue_indices.len()).max(1);
            let next_miss = AtomicUsize::new(0);
            let loaded_cues: Mutex<Vec<(usize, Vec<TimelineCueRecord>)>> =
                Mutex::new(Vec::with_capacity(missing_cue_indices.len()));
            let cue_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);

            std::thread::scope(|s| {
                for _ in 0..load_workers {
                    s.spawn(|| {
                        let mut local = Vec::new();
                        loop {
                            if cue_err.lock().unwrap().is_some() {
                                break;
                            }
                            let pos = next_miss.fetch_add(1, Ordering::Relaxed);
                            if pos >= missing_cue_indices.len() {
                                break;
                            }
                            let idx = missing_cue_indices[pos];
                            let ch = &chapters[idx];
                            let full_path = args.out_dir.join(chapter_rel_output_path(ch.order));
                            let res = (|| -> Result<Vec<TimelineCueRecord>> {
                                let bytes = fs::read(&full_path).with_context(|| {
                                    format!("read chapter JSON {}", full_path.display())
                                })?;
                                let parsed: ChapterOutput = serde_json::from_slice(&bytes)
                                    .with_context(|| {
                                        format!("parse chapter JSON {}", full_path.display())
                                    })?;
                                Ok(extract_timeline_cues(
                                    ch.order,
                                    &ch.title,
                                    &ch.source_file,
                                    ch.source_span.start_char,
                                    Some(&ch.text),
                                    &parsed.doc,
                                ))
                            })();
                            match res {
                                Ok(cues) => local.push((idx, cues)),
                                Err(e) => {
                                    let mut guard = cue_err.lock().unwrap();
                                    if guard.is_none() {
                                        *guard = Some(e);
                                    }
                                    break;
                                }
                            }
                        }
                        loaded_cues.lock().unwrap().extend(local);
                    });
                }
            });

            if let Some(err) = cue_err.into_inner().unwrap() {
                return Err(err);
            }
            for (idx, cues) in loaded_cues.into_inner().unwrap() {
                cues_slots[idx] = Some(cues);
            }
        }

        let mut cues_buf = Vec::new();
        for slot in &cues_slots {
            let cues = slot
                .as_ref()
                .ok_or_else(|| anyhow!("missing timeline cues slot"))?;
            for cue in cues {
                serde_json::to_writer(&mut cues_buf, cue)?;
                cues_buf.push(b'\n');
            }
        }
        let sha = sha256_hex(&cues_buf);
        atomic_write_bytes(&cues_path, &cues_buf, "cues")?;
        sha
    };

    // Rewrite canonical `completed_index.jsonl` sorted strictly by `order`.
    if processed_chapters > 0 || !index_path.is_file() {
        let mut idx_buf = Vec::new();
        for entry in &ordered_entries {
            serde_json::to_writer(&mut idx_buf, entry)?;
            idx_buf.push(b'\n');
        }
        atomic_write_bytes(&index_path, &idx_buf, "index")?;
    }

    // Aggregate deterministic book-level statistics.
    let mut total_chars = 0usize;
    let mut total_paragraphs = 0usize;
    let mut total_sentences = 0usize;
    let mut supported_sentences = 0usize;
    let mut total_clauses = 0usize;
    let mut total_tokens = 0usize;
    let mut decided_upos_tokens = 0usize;
    let mut syntax_tokens = 0usize;
    let mut total_entities = 0usize;
    let mut total_relations = 0usize;
    let mut total_timeline_cues = 0usize;
    let mut unknown_by_layer_reason: BTreeMap<String, usize> = BTreeMap::new();

    for entry in &ordered_entries {
        let st = &entry.stats;
        total_chars += st.text_chars;
        total_paragraphs += st.paragraphs;
        total_sentences += st.sentences;
        supported_sentences += st.supported_sentences;
        total_clauses += st.clauses;
        total_tokens += st.tokens;
        decided_upos_tokens += st.decided_upos_tokens;
        syntax_tokens += st.syntax_tokens;
        total_entities += st.entities;
        total_relations += st.relations;
        total_timeline_cues += st.timeline_cues;
        for (k, v) in &st.unknown_by_layer_reason {
            *unknown_by_layer_reason.entry(k.clone()).or_insert(0) += *v;
        }
    }

    let unknown_total: usize = unknown_by_layer_reason.values().sum();
    let unknown_breakdown: Vec<UnknownBreakdownEntry> = unknown_by_layer_reason
        .iter()
        .map(|(k, &count)| {
            let (layer, reason) = k.split_once(':').unwrap_or((k.as_str(), "unknown"));
            UnknownBreakdownEntry {
                layer: layer.to_owned(),
                reason: reason.to_owned(),
                count,
            }
        })
        .collect();

    let decided_upos_rate = if total_tokens == 0 {
        1.0
    } else {
        round6(decided_upos_tokens as f64 / total_tokens as f64)
    };
    let syntax_token_coverage_rate = if total_tokens == 0 {
        1.0
    } else {
        round6(syntax_tokens as f64 / total_tokens as f64)
    };

    let elapsed_secs = round6(start_instant.elapsed().as_secs_f64());
    let denom = elapsed_secs.max(1e-6);
    let chapters_per_sec = round6(total_chapters as f64 / denom);
    let sentences_per_sec = round6(total_sentences as f64 / denom);
    let processed_sentences_per_sec = round6(newly_processed_sentences as f64 / denom);

    let receipt = BookManifestReceipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        input_mode: input_mode.to_owned(),
        backend: backend_str.to_owned(),
        model_id: expected_model_id,
        model_sha256: expected_model_sha256,
        workers: effective_workers,
        total_chapters,
        resumed_chapters,
        processed_chapters,
        total_chars,
        total_paragraphs,
        total_sentences,
        supported_sentences,
        total_clauses,
        total_tokens,
        decided_upos_tokens,
        decided_upos_rate,
        syntax_tokens,
        syntax_token_coverage_rate,
        total_entities,
        total_relations,
        total_timeline_cues,
        unknown_total,
        unknown_by_layer_reason,
        unknown_breakdown,
        elapsed_secs,
        chapters_per_sec,
        sentences_per_sec,
        processed_sentences_per_sec,
        chapter_outputs_sha256,
        timeline_cues_sha256,
    };

    let mut receipt_bytes = serde_json::to_vec_pretty(&receipt)?;
    receipt_bytes.push(b'\n');
    atomic_write_bytes(&receipt_path, &receipt_bytes, "receipt")?;

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("nlp_novel_batch: {err:#}");
        std::process::exit(1);
    }
}
