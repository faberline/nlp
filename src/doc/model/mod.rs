//! Model runtime for layers 4 to 6 of `nlp.doc.v1`. See `docs/doc-v1.md`, "Model runtime".
//!
//! A model lives in `<models_dir>/<dir>/` and is listed in `<models_dir>/manifest.json`.
//! [`verify`] checks every listed file against its sha256 before any file is parsed. [`load`]
//! verifies too, then loads the model. Without a compiled backend feature, [`load`] returns
//! `NoBackend` at once, so a layer reports `unknown` with reason `no_model`.

#[cfg(all(feature = "coreml", target_os = "macos"))]
mod coreml;
#[cfg(feature = "onnx")]
mod onnx;

use crate::Span;
use crate::doc::{Reason, Unknown};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Component, Path};

pub const MANIFEST_SCHEMA: &str = "nlp.models.v1";

/// The largest `max_len` that a manifest may give.
pub const MAX_LEN_CAP: usize = 4096;

/// A dense f32 tensor in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// One id sequence per item. Each sequence is at most `max_len` long.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Batch {
    pub ids: Vec<Vec<i64>>,
}

/// One tensor per item, shape `[len, labels]`, with the first `len` positions only.
#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub logits: Vec<Tensor>,
}

pub trait Model: Send + Sync {
    fn info(&self) -> &ModelInfo;
    fn run(&self, batch: &Batch) -> Result<Output, ModelError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub layers: Vec<String>,
    pub licence: String,
    pub max_len: usize,
    pub pad_id: i64,
    pub files: Vec<FileInfo>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileInfo {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// The feature for this backend is not compiled in.
    NoBackend,
    UnknownModel(String),
    /// A file listed in the manifest is absent.
    Missing(String),
    HashMismatch {
        file: String,
        expected: String,
        actual: String,
    },
    Manifest(String),
    /// A model or tokenizer file exists and matches its hash, but the backend cannot load it.
    Load(String),
    TooLong {
        len: usize,
        max: usize,
    },
    Run(String),
}

impl ModelError {
    /// The `unknown` reason a layer reports for this error. Only the cases where no model
    /// is available map to a reason. Integrity and runtime errors are not `unknown`.
    pub fn into_reason(&self) -> Option<Reason> {
        match self {
            Self::NoBackend | Self::UnknownModel(_) | Self::Missing(_) => Some(Reason::NoModel),
            _ => None,
        }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBackend => write!(f, "no model backend is compiled in"),
            Self::UnknownModel(id) => write!(f, "unknown model: {id}"),
            Self::Missing(path) => write!(f, "missing model file: {path}"),
            Self::HashMismatch {
                file,
                expected,
                actual,
            } => write!(
                f,
                "hash mismatch for {file}: expected {expected}, got {actual}"
            ),
            Self::Manifest(msg) => write!(f, "invalid manifest: {msg}"),
            Self::Load(msg) => write!(f, "cannot load model: {msg}"),
            Self::TooLong { len, max } => write!(f, "sequence of {len} ids exceeds max_len {max}"),
            Self::Run(msg) => write!(f, "model run failed: {msg}"),
        }
    }
}

impl std::error::Error for ModelError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Onnx,
    /// Core ML on the CPU. Compiled only with the `coreml` feature on macOS.
    Coreml,
}

/// A manifest entry whose files were read once, hashed, and found to match. The parsers read
/// these bytes, never the files on disk again, so a file replaced after `verify` has no effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verified {
    pub info: ModelInfo,
    /// The verified bytes of the model file.
    pub model: Vec<u8>,
    /// The verified bytes of the tokenizer file.
    pub tokenizer: Vec<u8>,
    /// The verified bytes of the expected-output file (for tests and checks).
    pub expected: Vec<u8>,
    /// The Core ML model, when the manifest lists one.
    pub coreml: Option<CoremlFile>,
}

/// The verified bytes of a Core ML model file and the size of its embedding table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoremlFile {
    pub model: Vec<u8>,
    pub vocab: usize,
}

#[derive(Deserialize)]
struct Manifest {
    schema: String,
    models: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    id: String,
    layers: Vec<String>,
    licence: String,
    dir: String,
    max_len: usize,
    pad_id: i64,
    model: String,
    tokenizer: String,
    expected: String,
    #[serde(default)]
    coreml: Option<CoremlEntry>,
    files: Vec<FileEntry>,
}

#[derive(Deserialize)]
struct CoremlEntry {
    model: String,
    vocab: usize,
}

#[derive(Deserialize)]
struct FileEntry {
    path: String,
    sha256: String,
    #[serde(default)]
    bytes: Option<u64>,
}

/// Reads the manifest, checks that the model's files exist and match their sha256 and byte
/// counts. No file is parsed here.
pub fn verify(models_dir: &Path, model_id: &str) -> Result<Verified, ModelError> {
    let bytes = read_file(&models_dir.join("manifest.json"), "manifest.json", None)?;
    let manifest: Manifest =
        serde_json::from_slice(&bytes).map_err(|e| ModelError::Manifest(e.to_string()))?;
    if manifest.schema != MANIFEST_SCHEMA {
        return Err(ModelError::Manifest(format!(
            "schema is {}, expected {MANIFEST_SCHEMA}",
            manifest.schema
        )));
    }
    let entry = manifest
        .models
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| ModelError::UnknownModel(model_id.to_owned()))?;

    check_relative(&entry.dir)?;
    // Resolves a path that must be listed in `files` and returns its index there.
    let listed = |path: &str| -> Result<usize, ModelError> {
        check_relative(path)?;
        entry
            .files
            .iter()
            .position(|f| f.path == path)
            .ok_or_else(|| ModelError::Manifest(format!("{path} is not listed in files")))
    };
    let model_ix = listed(&entry.model)?;
    let tokenizer_ix = listed(&entry.tokenizer)?;
    let expected_ix = listed(&entry.expected)?;
    if model_ix == tokenizer_ix {
        return Err(ModelError::Manifest(format!(
            "model and tokenizer name the same file: {}",
            entry.model
        )));
    }
    if expected_ix == model_ix || expected_ix == tokenizer_ix {
        return Err(ModelError::Manifest(format!(
            "expected names the same file as model or tokenizer: {}",
            entry.expected
        )));
    }
    let coreml_ix = match &entry.coreml {
        Some(c) => {
            if c.vocab == 0 {
                return Err(ModelError::Manifest(
                    "coreml vocab must be positive".to_owned(),
                ));
            }
            let ix = listed(&c.model)?;
            if ix == model_ix || ix == tokenizer_ix || ix == expected_ix {
                return Err(ModelError::Manifest(format!(
                    "coreml names the same file as model, tokenizer or expected: {}",
                    c.model
                )));
            }
            Some(ix)
        }
        None => None,
    };
    for f in &entry.files {
        check_relative(&f.path)?;
    }
    if entry.max_len == 0 || entry.max_len > MAX_LEN_CAP {
        return Err(ModelError::Manifest(format!(
            "max_len must be from 1 to {MAX_LEN_CAP}"
        )));
    }

    let dir = models_dir.join(&entry.dir);
    let mut contents: Vec<Vec<u8>> = Vec::with_capacity(entry.files.len());
    for f in &entry.files {
        contents.push(read_file(&dir.join(&f.path), &f.path, f.bytes)?);
    }

    let mut files = Vec::with_capacity(entry.files.len());
    for (f, bytes) in entry.files.iter().zip(&contents) {
        let actual = sha256_hex(bytes);
        if actual != f.sha256 {
            return Err(ModelError::HashMismatch {
                file: f.path.clone(),
                expected: f.sha256.clone(),
                actual,
            });
        }
        files.push(FileInfo {
            path: f.path.clone(),
            sha256: f.sha256.clone(),
            bytes: bytes.len() as u64,
        });
    }

    let coreml = match (&entry.coreml, coreml_ix) {
        (Some(c), Some(ix)) => Some(CoremlFile {
            model: std::mem::take(&mut contents[ix]),
            vocab: c.vocab,
        }),
        _ => None,
    };

    Ok(Verified {
        info: ModelInfo {
            id: entry.id,
            layers: entry.layers,
            licence: entry.licence,
            max_len: entry.max_len,
            pad_id: entry.pad_id,
            files,
        },
        model: std::mem::take(&mut contents[model_ix]),
        tokenizer: std::mem::take(&mut contents[tokenizer_ix]),
        expected: std::mem::take(&mut contents[expected_ix]),
        coreml,
    })
}

/// Whether `backend` is compiled into this build.
pub(crate) fn compiled(backend: Backend) -> bool {
    match backend {
        Backend::Onnx => cfg!(feature = "onnx"),
        Backend::Coreml => cfg!(all(feature = "coreml", target_os = "macos")),
    }
}

/// Verifies the model and loads it with `backend`. When that backend is not compiled into this
/// build, this returns `NoBackend` at once, and no file is read.
pub fn load(
    models_dir: &Path,
    model_id: &str,
    backend: Backend,
) -> Result<Box<dyn Model>, ModelError> {
    if !compiled(backend) {
        return Err(ModelError::NoBackend);
    }
    load_verified(&verify(models_dir, model_id)?, backend)
}

/// Loads a model from bytes that `verify` has already checked.
pub fn load_verified(verified: &Verified, backend: Backend) -> Result<Box<dyn Model>, ModelError> {
    match backend {
        Backend::Onnx => {
            #[cfg(feature = "onnx")]
            {
                onnx::load(verified).map(|m| Box::new(m) as Box<dyn Model>)
            }
            #[cfg(not(feature = "onnx"))]
            {
                let _ = verified;
                Err(ModelError::NoBackend)
            }
        }
        Backend::Coreml => {
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            {
                coreml::load(verified).map(|m| Box::new(m) as Box<dyn Model>)
            }
            #[cfg(not(all(feature = "coreml", target_os = "macos")))]
            {
                let _ = verified;
                Err(ModelError::NoBackend)
            }
        }
    }
}

/// The `unknown` entry a layer reports when no model is available.
pub fn no_model_unknown(layer: &str, span: Span) -> Unknown {
    Unknown {
        layer: layer.to_owned(),
        span,
        reason: Reason::NoModel,
    }
}

/// Index of the largest value on the last axis of `t`. Ties go to the lowest index, and NaN
/// counts as smaller than any number.
pub fn argmax_labels(t: &Tensor) -> Vec<usize> {
    let Some(&labels) = t.shape.last() else {
        return Vec::new();
    };
    if labels == 0 {
        return Vec::new();
    }
    t.data
        .chunks(labels)
        .map(|row| {
            let mut best = 0;
            for (i, &v) in row.iter().enumerate().skip(1) {
                let best_v = row[best];
                if v > best_v || (best_v.is_nan() && !v.is_nan()) {
                    best = i;
                }
            }
            best
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subwords {
    pub ids: Vec<i64>,
    /// One span per id, in Unicode scalar offsets. Special tokens have an empty span.
    pub spans: Vec<Span>,
}

#[cfg(any(feature = "onnx", feature = "coreml"))]
pub struct SubwordTokenizer {
    inner: tokenizers::Tokenizer,
}

#[cfg(any(feature = "onnx", feature = "coreml"))]
impl SubwordTokenizer {
    /// Parses the tokenizer from the bytes that `verify` checked.
    pub fn from_verified(verified: &Verified) -> Result<Self, ModelError> {
        tokenizers::Tokenizer::from_bytes(&verified.tokenizer)
            .map(|inner| Self { inner })
            .map_err(|e| ModelError::Load(format!("tokenizer: {e}")))
    }

    pub fn encode(&self, text: &str) -> Result<Subwords, ModelError> {
        let enc = self
            .inner
            .encode(text, true)
            .map_err(|e| ModelError::Run(format!("tokenize: {e}")))?;
        let table = char_index_table(text);
        let to_char = |b: usize| {
            table
                .get(b)
                .copied()
                .ok_or_else(|| ModelError::Run(format!("offset {b} outside the text")))
        };
        let mut spans = Vec::with_capacity(enc.get_offsets().len());
        for &(s, e) in enc.get_offsets() {
            if s > e {
                return Err(ModelError::Run(format!("offset {s} > {e}")));
            }
            let start = to_char(s)?;
            // An end inside a character belongs to that character, so it rounds up.
            let end = if s == e || text.is_char_boundary(e) {
                to_char(e)?
            } else {
                to_char(e)? + 1
            };
            spans.push(Span {
                start_char: start,
                end_char: end,
            });
        }
        Ok(Subwords {
            ids: enc.get_ids().iter().map(|&i| i64::from(i)).collect(),
            spans,
        })
    }
}

/// Maps each byte index to the index of the character that contains it. The index at
/// `text.len()` is the character count.
#[cfg(any(feature = "onnx", feature = "coreml"))]
fn char_index_table(text: &str) -> Vec<usize> {
    let mut table = vec![0; text.len() + 1];
    let mut index = 0;
    for (byte, ch) in text.char_indices() {
        for slot in &mut table[byte..byte + ch.len_utf8()] {
            *slot = index;
        }
        index += 1;
    }
    table[text.len()] = index;
    table
}

/// Reads a regular file. Symlinks and other non-files are refused. The size is compared with
/// the manifest's `bytes` before the file is read, and the opened handle is checked again.
fn read_file(path: &Path, shown: &str, bytes: Option<u64>) -> Result<Vec<u8>, ModelError> {
    let io_err = |e: io::Error| match e.kind() {
        io::ErrorKind::NotFound => ModelError::Missing(shown.to_owned()),
        _ => ModelError::Load(format!("{shown}: {e}")),
    };
    let meta = fs::symlink_metadata(path).map_err(io_err)?;
    if meta.file_type().is_symlink() {
        return Err(ModelError::Load(format!("{shown}: symlinks are refused")));
    }
    if !meta.file_type().is_file() {
        return Err(ModelError::Load(format!("{shown}: not a regular file")));
    }
    let size_mismatch = |actual: u64| ModelError::HashMismatch {
        file: format!("{shown} (bytes)"),
        expected: bytes.map(|n| n.to_string()).unwrap_or_default(),
        actual: actual.to_string(),
    };
    if bytes.is_some_and(|n| meta.len() != n) {
        return Err(size_mismatch(meta.len()));
    }
    let mut file = fs::File::open(path).map_err(io_err)?;
    let opened = file.metadata().map_err(io_err)?;
    if !opened.is_file() {
        return Err(ModelError::Load(format!("{shown}: not a regular file")));
    }
    let mut buf = Vec::with_capacity(opened.len() as usize);
    file.read_to_end(&mut buf).map_err(io_err)?;
    if bytes.is_some_and(|n| buf.len() as u64 != n) {
        return Err(size_mismatch(buf.len() as u64));
    }
    Ok(buf)
}

fn check_relative(path: &str) -> Result<(), ModelError> {
    let p = Path::new(path);
    let plain = !path.is_empty()
        && p.components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
    if plain {
        Ok(())
    } else {
        Err(ModelError::Manifest(format!(
            "path must be relative and inside the model directory: {path}"
        )))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
