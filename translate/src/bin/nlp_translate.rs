use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process;

use clap::Parser;
use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    OutputFormat, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
    render_output,
};

#[derive(Parser)]
#[command(
    name = "nlp_translate",
    about = "Offline post-NLP translator (nlp.translate.v1)"
)]
struct Cli {
    /// Target language variant: zh-Hant, zh-Hans, en-US, en-GB, ja, or ko.
    #[arg(short, long)]
    target: TargetVariant,

    /// Optional source variant hint (zh-Hant, zh-Hans, en-US, en-GB, ja, ko).
    #[arg(long)]
    source_hint: Option<TargetVariant>,

    /// Output format: json, json-pretty, text, bilingual-md, or bilingual-jsonl.
    #[arg(short, long, default_value = "json")]
    format: OutputFormat,

    /// Path to a JSON `TranslationGlossary` file.
    #[arg(short, long)]
    glossary: Option<PathBuf>,

    /// Path to a verified offline Seq2Seq models directory (`nlp.models.seq2seq.v1`).
    #[arg(long)]
    models: Option<PathBuf>,

    /// Model ID inside `<models>/manifest.json`.
    #[arg(long, default_value = "doc-translate-v1")]
    model_id: String,

    /// Model backend: coreml, coreml-fast, or onnx.
    #[arg(long)]
    backend: Option<String>,

    /// Beam width (1..=4).
    #[arg(long, default_value_t = 1)]
    beam_size: usize,

    /// Mask editorial/scraper boilerplate lines.
    #[arg(long)]
    mask_editorial: bool,

    /// Omit masked editorial blocks from rendered target text.
    #[arg(long)]
    omit_editorial: bool,

    /// Disable target-locale quotation mark localization.
    #[arg(long)]
    no_localize_quotes: bool,

    /// Format Layer-3 entities (dates, numbers, quantities) into target locale conventions.
    #[arg(long)]
    localize_entities: bool,

    /// Insert spaces at CJK–ASCII word boundaries.
    #[arg(long)]
    cjk_latin_spacing: bool,
}

fn parse_backend(s: Option<&str>) -> Result<Backend, String> {
    match s {
        None => {
            if cfg!(all(feature = "coreml", target_os = "macos")) {
                Ok(Backend::Coreml)
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

    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    let glossary = match &cli.glossary {
        Some(p) => {
            let raw = fs::read_to_string(p)?;
            serde_json::from_str::<TranslationGlossary>(&raw)?
        }
        None => TranslationGlossary::new(),
    };

    let options = TranslateOptions {
        target: cli.target,
        source_hint: cli.source_hint,
        mask_editorial: cli.mask_editorial,
        omit_masked_editorial: cli.omit_editorial,
        localize_quotes: !cli.no_localize_quotes,
        localize_entities: cli.localize_entities,
        cjk_latin_spacing: cli.cjk_latin_spacing,
        beam_size: cli.beam_size.clamp(1, 4),
        model_id: cli.model_id,
    };

    let doc = analyze_and_translate(&input, &options, &glossary, cli.models.as_deref(), backend)?;
    let rendered = render_output(&doc, &input, cli.format)?;
    if matches!(cli.format, OutputFormat::Text) {
        print!("{rendered}");
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("nlp_translate: {err}");
        process::exit(1);
    }
}
