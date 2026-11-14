use clap::{Parser, ValueEnum};
use faberline_nlp::doc::model::Backend;
use faberline_nlp::doc::{Options, analyze_document, analyze_document_with_model};
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
#[command(name = "nlp_doc", about = "Analyze UTF-8 text into nlp.doc.v1 JSON")]
struct Args {
    /// Directory containing manifest.json and model subdirectories for Layer 4+.
    #[arg(long, value_name = "DIR")]
    models: Option<PathBuf>,

    /// Model runtime backend to use when --models is supplied.
    #[arg(long, value_enum, requires = "models")]
    backend: Option<BackendArg>,

    /// Exact surface names to match in Layer 3 (`options.names`, repeatable).
    #[arg(long = "name", value_name = "SURFACE")]
    names: Vec<String>,

    /// Extra domain compounds to protect during Chinese word segmentation (`options.lexicon`, repeatable).
    #[arg(long = "lexicon", value_name = "SURFACE")]
    lexicon: Vec<String>,

    /// Mask generic web-novel editorial postscripts and scraper watermarks (`options.mask_editorial`).
    #[arg(long)]
    mask_editorial: bool,

    /// Project Traditional Han characters 1:1 onto Simplified equivalents during dictionary lookup (`options.normalize_traditional`).
    #[arg(long)]
    normalize_traditional: bool,

    /// Enrich tokens with morphology and semantic fields (`options.enrich_morphology`).
    #[arg(long)]
    morphology: bool,

    /// Use the first-party `zh-Hant` + `zh-Hans` lexicon segmenter for Chinese runs (`options.first_party_zh_segmenter`).
    #[arg(long)]
    first_party_dict: bool,
}

fn main() {
    let args = Args::parse();
    let mut text = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut text) {
        eprintln!("nlp_doc: {err}");
        std::process::exit(1);
    }
    let options = Options {
        names: args.names,
        mask_editorial: args.mask_editorial,
        normalize_traditional: args.normalize_traditional,
        lexicon: args.lexicon,
        enrich_morphology: args.morphology,
        first_party_zh_segmenter: args.first_party_dict,
    };
    let doc = match &args.models {
        Some(models_dir) => {
            let backend = args.backend.map_or_else(default_backend, Backend::from);
            match analyze_document_with_model(&text, &options, models_dir, backend) {
                Ok(doc) => doc,
                Err(err) => {
                    eprintln!("nlp_doc: {err}");
                    std::process::exit(1);
                }
            }
        }
        None => analyze_document(&text, &options),
    };
    println!(
        "{}",
        serde_json::to_string(&doc).expect("document serializes to JSON")
    );
}
