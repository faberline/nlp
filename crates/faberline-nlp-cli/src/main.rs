use std::ffi::OsString;

#[path = "../../faberline-nlp/src/bin/nlp_doc.rs"]
mod doc;
#[path = "../../faberline-nlp-narrative/src/bin/nlp_narrative_batch.rs"]
mod narrative_batch;
#[path = "../../faberline-nlp-narrative/src/bin/nlp_narrative_doc.rs"]
mod narrative_doc;
#[path = "../../faberline-nlp/src/bin/nlp_novel_batch.rs"]
mod novel_batch;
#[path = "../../faberline-nlp/src/bin/nlp_source_stream.rs"]
mod source_stream;
#[path = "../../faberline-nlp-translate/src/bin/nlp_translate.rs"]
mod translate;
#[path = "../../faberline-nlp-translate/src/bin/nlp_translate_batch.rs"]
mod translate_batch;

fn main() {
    let mut args = std::env::args_os();
    let _ = args.next();
    let command = args.next();
    let Some(command) = command else {
        eprintln!(
            "Usage: nlp <COMMAND> [OPTIONS]\n\nCommands: doc, novel-batch, source-stream, narrative-doc, narrative-batch, translate, translate-batch"
        );
        std::process::exit(2);
    };
    if command == "--help" || command == "-h" {
        println!(
            "Usage: nlp <COMMAND> [OPTIONS]\n\nCommands: doc, novel-batch, source-stream, narrative-doc, narrative-batch, translate, translate-batch"
        );
        return;
    }
    if command == "--version" || command == "-V" {
        println!("nlp {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    let Some(name) = command.to_str() else {
        eprintln!("nlp: invalid command");
        std::process::exit(2);
    };
    let handler: fn(Vec<OsString>) = match name {
        "doc" => doc::dispatch,
        "novel-batch" => novel_batch::dispatch,
        "source-stream" => |args| {
            if let Err(err) = source_stream::dispatch(args) {
                eprintln!("Error: {err:?}");
                std::process::exit(1);
            }
        },
        "narrative-doc" => narrative_doc::dispatch,
        "narrative-batch" => narrative_batch::dispatch,
        "translate" => translate::dispatch,
        "translate-batch" => translate_batch::dispatch,
        _ => {
            eprintln!("nlp: unknown command: {name}");
            std::process::exit(2);
        }
    };
    handler(
        std::iter::once(OsString::from(format!("nlp {name}")))
            .chain(args)
            .collect(),
    );
}
