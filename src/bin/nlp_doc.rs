use faberline_nlp::doc::{Options, analyze_document};
use std::io::{self, Read};

fn main() {
    let mut text = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut text) {
        eprintln!("nlp_doc: {err}");
        std::process::exit(1);
    }
    let doc = analyze_document(&text, &Options::default());
    println!(
        "{}",
        serde_json::to_string(&doc).expect("document serializes to JSON")
    );
}
