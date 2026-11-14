use std::fs;
use std::path::PathBuf;
use std::process;

use clap::Parser;
use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::batch::{BatchConfig, run_translate_batch};
use faberline_nlp_translate::{TargetVariant, TranslateOptions, TranslationGlossary};

#[derive(Parser)]
#[command(
    name = "nlp_translate_batch",
    about = "Resumable multi-worker corpus/book translator (nlp.translate.v1)"
)]
struct Cli {
    /// Directory containing `.txt` or `.md` chapter files.
    #[arg(short, long)]
    input_dir: PathBuf,

    /// Output directory for `json/`, `text/`, `bilingual/`, and `translate_manifest_receipt.json`.
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Target language variant: zh-Hant, zh-Hans, en-US, en-GB, ja, or ko.
    #[arg(short, long)]
    target: TargetVariant,

    /// Optional JSON `TranslationGlossary` file.
    #[arg(short, long)]
    glossary: Option<PathBuf>,

    /// Optional offline Seq2Seq models directory (`nlp.models.seq2seq.v1`).
    #[arg(long)]
    models: Option<PathBuf>,

    /// Model ID inside `<models>/manifest.json`.
    #[arg(long, default_value = "doc-translate-v1")]
    model_id: String,

    /// Backend: coreml-fast, coreml, or onnx.
    #[arg(long)]
    backend: Option<String>,

    /// Number of worker threads.
    #[arg(short, long, default_value_t = 4)]
    workers: usize,

    /// Disable resuming from matching `source_sha256` chapter outputs.
    #[arg(long)]
    no_resume: bool,

    /// Also emit `bilingual/<stem>.bilingual.md` side-by-side chapter files.
    #[arg(long)]
    bilingual: bool,

    /// Mask editorial/scraper boilerplate lines (enabled by default for batch novel corpora).
    #[arg(long, default_value_t = true)]
    mask_editorial: bool,

    /// Omit masked editorial blocks from target text.
    #[arg(long)]
    omit_editorial: bool,
}

fn parse_backend(s: Option<&str>) -> Result<Backend, String> {
    match s {
        None => {
            if cfg!(all(feature = "coreml", target_os = "macos")) {
                Ok(Backend::CoremlFast)
            } else {
                Ok(Backend::Onnx)
            }
        }
        Some("coreml") => Ok(Backend::Coreml),
        Some("coreml-fast") => Ok(Backend::CoremlFast),
        Some("onnx") => Ok(Backend::Onnx),
        Some(other) => Err(format!(
            "unknown backend '{other}' (expected coreml, coreml-fast, or onnx)"
        )),
    }
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let backend = parse_backend(cli.backend.as_deref()).map_err(anyhow::Error::msg)?;

    let glossary = match &cli.glossary {
        Some(p) => {
            let raw = fs::read_to_string(p)?;
            serde_json::from_str::<TranslationGlossary>(&raw)?
        }
        None => TranslationGlossary::new(),
    };

    let options = TranslateOptions {
        target: cli.target,
        source_hint: None,
        mask_editorial: cli.mask_editorial,
        omit_masked_editorial: cli.omit_editorial,
        localize_quotes: true,
        localize_entities: false,
        cjk_latin_spacing: false,
        beam_size: 1,
        model_id: cli.model_id,
    };

    let cfg = BatchConfig {
        input_dir: cli.input_dir,
        output_dir: cli.output_dir,
        options,
        glossary,
        models_dir: cli.models,
        backend,
        workers: cli.workers,
        resume: !cli.no_resume,
        write_bilingual: cli.bilingual,
    };

    let receipt = run_translate_batch(&cfg)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("nlp_translate_batch: {err}");
        process::exit(1);
    }
}
