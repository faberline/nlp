//! Verified offline Seq2Seq model runtime (`nlp.models.seq2seq.v1`) on `tract-onnx` and
//! `objc2-core-ml`, mirroring the zero-network, single-read SHA-256 verification guarantees of
//! `faberline_nlp::doc::model`.

#[cfg(all(feature = "coreml", target_os = "macos"))]
mod coreml;
pub mod decode;
#[cfg(feature = "onnx")]
mod onnx;

#[cfg(any(feature = "onnx", feature = "coreml"))]
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Component, Path};

pub use faberline_nlp::doc::model::{Backend, FileInfo, MAX_LEN_CAP, ModelError};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::schema::{SEQ2SEQ_MANIFEST_SCHEMA, TargetVariant};

/// Metadata of a verified offline Seq2Seq translation model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seq2SeqModelInfo {
    pub id: String,
    pub licence: String,
    pub src_max_len: usize,
    pub tgt_max_len: usize,
    pub d_model: usize,
    pub vocab: usize,
    pub pad_id: i64,
    pub bos_id: i64,
    pub eos_id: i64,
    pub supported_variants: Vec<TargetVariant>,
    pub files: Vec<FileInfo>,
}

/// Encoded source representation passed from the encoder to each decoder step, with padded
/// positions (`src_len..src_max_len`) deterministically zero-masked.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderState {
    pub src_len: usize,
    pub src_max_len: usize,
    pub d_model: usize,
    /// Flat `[1, src_max_len, d_model]` row-major buffer with `src_len..src_max_len` zeroed.
    pub hidden: Vec<f32>,
}

pub trait Seq2SeqModel: Send + Sync {
    fn info(&self) -> &Seq2SeqModelInfo;
    fn encode(&self, input_ids: &[i64]) -> Result<EncoderState, ModelError>;
    fn decode_step(
        &self,
        encoder: &EncoderState,
        decoder_ids: &[i64],
    ) -> Result<Vec<f32>, ModelError>;
}

/// In-memory verified bytes of all model artifacts; parsers read exclusively from these buffers
/// so replacing a file on disk after `verify_seq2seq` has zero effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedSeq2Seq {
    pub info: Seq2SeqModelInfo,
    pub encoder_onnx: Vec<u8>,
    pub decoder_onnx: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub expected: Vec<u8>,
    pub coreml: Option<VerifiedCoremlSeq2Seq>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedCoremlSeq2Seq {
    pub encoder: Vec<u8>,
    pub decoder: Vec<u8>,
}

#[derive(Deserialize)]
struct Seq2SeqManifest {
    schema: String,
    models: Vec<Seq2SeqEntry>,
}

#[derive(Deserialize)]
struct Seq2SeqEntry {
    id: String,
    licence: String,
    dir: String,
    src_max_len: usize,
    tgt_max_len: usize,
    d_model: usize,
    vocab: usize,
    pad_id: i64,
    bos_id: i64,
    eos_id: i64,
    #[serde(default)]
    supported_variants: Vec<TargetVariant>,
    encoder_onnx: String,
    decoder_onnx: String,
    tokenizer: String,
    expected: String,
    #[serde(default)]
    coreml: Option<CoremlSeq2SeqEntry>,
    files: Vec<FileEntry>,
}

#[derive(Deserialize)]
struct CoremlSeq2SeqEntry {
    encoder: String,
    decoder: String,
}

#[derive(Deserialize)]
struct FileEntry {
    path: String,
    sha256: String,
    #[serde(default)]
    bytes: Option<u64>,
}

/// Wrapper around `tokenizers::Tokenizer` built strictly from verified bytes.
#[cfg(any(feature = "onnx", feature = "coreml"))]
pub struct Seq2SeqTokenizer {
    inner: tokenizers::Tokenizer,
}

#[cfg(any(feature = "onnx", feature = "coreml"))]
impl fmt::Debug for Seq2SeqTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Seq2SeqTokenizer").finish_non_exhaustive()
    }
}

#[cfg(any(feature = "onnx", feature = "coreml"))]
impl Seq2SeqTokenizer {
    pub fn from_verified(v: &VerifiedSeq2Seq) -> Result<Self, ModelError> {
        let inner = tokenizers::Tokenizer::from_bytes(&v.tokenizer)
            .map_err(|e| ModelError::Load(format!("tokenizer: {e}")))?;
        Ok(Self { inner })
    }

    pub fn encode_ids(&self, text: &str) -> Result<Vec<i64>, ModelError> {
        let enc = self
            .inner
            .encode(text, true)
            .map_err(|e| ModelError::Run(format!("encode: {e}")))?;
        Ok(enc.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    pub fn encode_raw_ids(&self, text: &str) -> Result<Vec<i64>, ModelError> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| ModelError::Run(format!("encode_raw: {e}")))?;
        Ok(enc.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    pub fn decode_ids(&self, ids: &[i64]) -> Result<String, ModelError> {
        let u32_ids: Vec<u32> = ids
            .iter()
            .map(|&id| u32::try_from(id).map_err(|_| ModelError::Run(format!("bad id {id}"))))
            .collect::<Result<Vec<_>, _>>()?;
        self.inner
            .decode(&u32_ids, true)
            .map_err(|e| ModelError::Run(format!("decode: {e}")))
    }
}

/// Verify `<models_dir>/manifest.json` (`schema: "nlp.models.seq2seq.v1"`) and read + SHA-256
/// check every file listed for `model_id` into memory before any model parser runs.
pub fn verify_seq2seq(models_dir: &Path, model_id: &str) -> Result<VerifiedSeq2Seq, ModelError> {
    let bytes = read_regular_file(&models_dir.join("manifest.json"), "manifest.json", None)?;
    let manifest: Seq2SeqManifest =
        serde_json::from_slice(&bytes).map_err(|e| ModelError::Manifest(e.to_string()))?;
    if manifest.schema != SEQ2SEQ_MANIFEST_SCHEMA {
        return Err(ModelError::Manifest(format!(
            "schema is {}, expected {SEQ2SEQ_MANIFEST_SCHEMA}",
            manifest.schema
        )));
    }
    let entry = manifest
        .models
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| ModelError::UnknownModel(model_id.to_owned()))?;

    check_relative(&entry.dir)?;
    if entry.src_max_len == 0 || entry.src_max_len > MAX_LEN_CAP {
        return Err(ModelError::Manifest(format!(
            "src_max_len {} outside 1..={MAX_LEN_CAP}",
            entry.src_max_len
        )));
    }
    if entry.tgt_max_len == 0 || entry.tgt_max_len > MAX_LEN_CAP {
        return Err(ModelError::Manifest(format!(
            "tgt_max_len {} outside 1..={MAX_LEN_CAP}",
            entry.tgt_max_len
        )));
    }
    if entry.d_model == 0 || entry.d_model > MAX_LEN_CAP {
        return Err(ModelError::Manifest(format!(
            "d_model {} outside 1..={MAX_LEN_CAP}",
            entry.d_model
        )));
    }
    if entry.vocab == 0 {
        return Err(ModelError::Manifest("vocab must be positive".to_owned()));
    }
    for (label, id) in [
        ("pad_id", entry.pad_id),
        ("bos_id", entry.bos_id),
        ("eos_id", entry.eos_id),
    ] {
        if usize::try_from(id).map_or(true, |u| u >= entry.vocab) {
            return Err(ModelError::Manifest(format!(
                "{label} {id} outside 0..{}",
                entry.vocab
            )));
        }
    }
    if entry.bos_id == entry.eos_id {
        return Err(ModelError::Manifest(
            "bos_id and eos_id must differ".to_owned(),
        ));
    }

    let listed = |path: &str| -> Result<usize, ModelError> {
        check_relative(path)?;
        entry
            .files
            .iter()
            .position(|f| f.path == path)
            .ok_or_else(|| ModelError::Manifest(format!("{path} is not listed in files")))
    };

    let enc_ix = listed(&entry.encoder_onnx)?;
    let dec_ix = listed(&entry.decoder_onnx)?;
    let tok_ix = listed(&entry.tokenizer)?;
    let exp_ix = listed(&entry.expected)?;
    let mut seen_ix = vec![enc_ix, dec_ix, tok_ix, exp_ix];

    let coreml_ix = if let Some(cm) = &entry.coreml {
        let ce = listed(&cm.encoder)?;
        let cd = listed(&cm.decoder)?;
        seen_ix.push(ce);
        seen_ix.push(cd);
        Some((ce, cd))
    } else {
        None
    };

    seen_ix.sort_unstable();
    for pair in seen_ix.windows(2) {
        if pair[0] == pair[1] {
            return Err(ModelError::Manifest(
                "distinct model roles must reference distinct files".to_owned(),
            ));
        }
    }

    let dir_path = models_dir.join(&entry.dir);
    let mut raw_files: Vec<Vec<u8>> = Vec::with_capacity(entry.files.len());
    let mut infos: Vec<FileInfo> = Vec::with_capacity(entry.files.len());

    for f in &entry.files {
        check_relative(&f.path)?;
        let data = read_regular_file(&dir_path.join(&f.path), &f.path, f.bytes)?;
        let actual = sha256_bytes_hex(&data);
        if actual != f.sha256 {
            return Err(ModelError::HashMismatch {
                file: f.path.clone(),
                expected: f.sha256.clone(),
                actual,
            });
        }
        infos.push(FileInfo {
            path: f.path.clone(),
            sha256: actual,
            bytes: data.len() as u64,
        });
        raw_files.push(data);
    }

    let coreml = coreml_ix.map(|(ce, cd)| VerifiedCoremlSeq2Seq {
        encoder: raw_files[ce].clone(),
        decoder: raw_files[cd].clone(),
    });

    Ok(VerifiedSeq2Seq {
        info: Seq2SeqModelInfo {
            id: entry.id,
            licence: entry.licence,
            src_max_len: entry.src_max_len,
            tgt_max_len: entry.tgt_max_len,
            d_model: entry.d_model,
            vocab: entry.vocab,
            pad_id: entry.pad_id,
            bos_id: entry.bos_id,
            eos_id: entry.eos_id,
            supported_variants: entry.supported_variants,
            files: infos,
        },
        encoder_onnx: raw_files[enc_ix].clone(),
        decoder_onnx: raw_files[dec_ix].clone(),
        tokenizer: raw_files[tok_ix].clone(),
        expected: raw_files[exp_ix].clone(),
        coreml,
    })
}

/// Load a verified Seq2Seq model on the requested `Backend`.
pub fn load_verified_seq2seq(
    v: &VerifiedSeq2Seq,
    backend: Backend,
) -> Result<Box<dyn Seq2SeqModel>, ModelError> {
    match backend {
        Backend::Onnx => {
            #[cfg(feature = "onnx")]
            {
                Ok(Box::new(onnx::load(v)?))
            }
            #[cfg(not(feature = "onnx"))]
            {
                let _ = v;
                Err(ModelError::NoBackend)
            }
        }
        Backend::Coreml => {
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            {
                Ok(Box::new(coreml::load_cpu(v)?))
            }
            #[cfg(not(all(feature = "coreml", target_os = "macos")))]
            {
                let _ = v;
                Err(ModelError::NoBackend)
            }
        }
        Backend::CoremlFast => {
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            {
                Ok(Box::new(coreml::load_fast(v)?))
            }
            #[cfg(not(all(feature = "coreml", target_os = "macos")))]
            {
                let _ = v;
                Err(ModelError::NoBackend)
            }
        }
    }
}

/// Verify and load `model_id` from `models_dir` on `backend`.
/// Note: mirroring `faberline_nlp::doc::model::load`, if the requested backend feature is not
/// compiled in, returns `ModelError::NoBackend` immediately without reading disk.
pub fn load_seq2seq(
    models_dir: &Path,
    model_id: &str,
    backend: Backend,
) -> Result<Box<dyn Seq2SeqModel>, ModelError> {
    if !backend_compiled_in(backend) {
        return Err(ModelError::NoBackend);
    }
    let verified = verify_seq2seq(models_dir, model_id)?;
    load_verified_seq2seq(&verified, backend)
}

pub fn backend_compiled_in(backend: Backend) -> bool {
    match backend {
        Backend::Onnx => cfg!(feature = "onnx"),
        Backend::Coreml | Backend::CoremlFast => {
            cfg!(all(feature = "coreml", target_os = "macos"))
        }
    }
}

fn check_relative(path: &str) -> Result<(), ModelError> {
    if path.is_empty() {
        return Err(ModelError::Manifest("empty path".to_owned()));
    }
    for comp in Path::new(path).components() {
        match comp {
            Component::Normal(_) => {}
            _ => {
                return Err(ModelError::Manifest(format!(
                    "non-relative path component in '{path}'"
                )));
            }
        }
    }
    Ok(())
}

fn read_regular_file(
    full_path: &Path,
    display_name: &str,
    expected_bytes: Option<u64>,
) -> Result<Vec<u8>, ModelError> {
    let meta = match fs::symlink_metadata(full_path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(ModelError::Missing(display_name.to_owned()));
        }
        Err(e) => return Err(ModelError::Load(format!("{display_name}: {e}"))),
    };
    if meta.file_type().is_symlink() || !meta.file_type().is_file() {
        return Err(ModelError::Load(format!(
            "{display_name}: refused non-regular file"
        )));
    }
    if let Some(exp) = expected_bytes
        && meta.len() != exp
    {
        return Err(ModelError::HashMismatch {
            file: format!("{display_name} (bytes)"),
            expected: exp.to_string(),
            actual: meta.len().to_string(),
        });
    }
    let mut f =
        fs::File::open(full_path).map_err(|e| ModelError::Load(format!("{display_name}: {e}")))?;
    let mut buf = Vec::with_capacity(meta.len() as usize);
    f.read_to_end(&mut buf)
        .map_err(|e| ModelError::Load(format!("{display_name}: {e}")))?;
    Ok(buf)
}

fn sha256_bytes_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
