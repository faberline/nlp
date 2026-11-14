use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use faberline_nlp::doc::model::Backend;
use faberline_nlp::doc::{Document, analyze_document, analyze_document_with_model};
use faberline_nlp_narrative::{WorkVocabulary, analyze_chapter_narrative, text_sha256_hex};
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
    Coreml,
    CoremlFast,
    Onnx,
}

impl From<BackendArg> for Backend {
    fn from(arg: BackendArg) -> Self {
        match arg {
            BackendArg::Coreml => Self::Coreml,
            BackendArg::CoremlFast => Self::CoremlFast,
            BackendArg::Onnx => Self::Onnx,
        }
    }
}

fn default_backend() -> Backend {
    if cfg!(all(feature = "coreml", target_os = "macos")) {
        Backend::Coreml
    } else {
        Backend::Onnx
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "nlp_narrative_doc",
    about = "Analyze a single UTF-8 document into nlp.narrative.chapter.v1 JSON"
)]
struct Args {
    /// 1-based chapter index.
    #[arg(long, default_value_t = 1, value_name = "U32")]
    chapter: u32,

    /// Optional chapter title.
    #[arg(long, value_name = "STR")]
    title: Option<String>,

    /// Optional WorkVocabulary JSON file.
    #[arg(long, value_name = "PATH")]
    vocab: Option<PathBuf>,

    /// Optional precomputed nlp.doc.v1 Document JSON (or batch wrapper with `doc` field).
    #[arg(long, value_name = "PATH", conflicts_with = "models")]
    doc_json: Option<PathBuf>,

    /// Directory containing manifest.json and model subdirectories for Layer 4 syntax decoding.
    #[arg(long, value_name = "DIR")]
    models: Option<PathBuf>,

    /// Model runtime backend to use when --models is supplied.
    #[arg(long, value_enum, requires = "models")]
    backend: Option<BackendArg>,
}

fn load_vocab(path: Option<&PathBuf>) -> Result<WorkVocabulary> {
    let Some(path) = path else {
        return Ok(WorkVocabulary::default());
    };
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read vocabulary file {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse vocabulary JSON {}", path.display()))
}

fn load_precomputed_doc(path: &PathBuf) -> Result<Document> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read doc JSON file {}", path.display()))?;
    let val: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse JSON in {}", path.display()))?;
    if let Some(doc_val) = val.get("doc") {
        serde_json::from_value::<Document>(doc_val.clone()).with_context(|| {
            format!(
                "failed to deserialize wrapped `doc` field in {}",
                path.display()
            )
        })
    } else {
        serde_json::from_value::<Document>(val)
            .with_context(|| format!("failed to deserialize Document JSON in {}", path.display()))
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    let mut text = String::new();
    io::stdin()
        .read_to_string(&mut text)
        .context("failed to read UTF-8 text from stdin")?;

    let vocab = load_vocab(args.vocab.as_ref())?;
    let doc = match (&args.doc_json, &args.models) {
        (Some(doc_path), _) => {
            let doc = load_precomputed_doc(doc_path)?;
            let actual_sha = text_sha256_hex(&text);
            if doc.text_sha256 != actual_sha {
                bail!(
                    "doc.text_sha256 ({}) does not match stdin text SHA-256 ({actual_sha})",
                    doc.text_sha256
                );
            }
            doc
        }
        (None, Some(models_dir)) => {
            let backend = args.backend.map_or_else(default_backend, Backend::from);
            if !faberline_nlp::doc::model::compiled(backend) {
                bail!(
                    "--models requires model runtime support compiled in for backend {:?} (enable `--features onnx` or `--features coreml`)",
                    backend
                );
            }
            let options = vocab.to_doc_options();
            analyze_document_with_model(&text, &options, models_dir, backend)
                .map_err(|err| anyhow::anyhow!("model inference failed: {err}"))?
        }
        (None, None) => {
            let options = vocab.to_doc_options();
            analyze_document(&text, &options)
        }
    };

    let narrative = analyze_chapter_narrative(args.chapter, args.title, &text, &doc, &vocab)?;
    let out = serde_json::to_string(&narrative).context("failed to serialize ChapterNarrative")?;
    println!("{out}");
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("nlp_narrative_doc: {err:#}");
        std::process::exit(1);
    }
}
