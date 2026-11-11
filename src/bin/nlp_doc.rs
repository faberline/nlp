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
}

fn main() {
    let args = Args::parse();
    let mut text = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut text) {
        eprintln!("nlp_doc: {err}");
        std::process::exit(1);
    }
    let options = Options::default();
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
