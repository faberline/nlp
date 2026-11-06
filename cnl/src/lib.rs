//! Syntax parser for English SOM CNL v1, with source ranges for each rule.

use std::ops::Range;

use pest::{Parser as _, iterators::Pair};
use pest_derive::Parser;

mod document;
pub use document::*;
pub use service::{SERVICE_SYNTAX_API_VERSION, parse_service_document};

/// Sources used by this crate, for consumers that bind builds to source bytes.
/// Paths are relative to this crate's manifest directory and sorted.
pub const SOURCE_FILES: &[(&str, &[u8])] = &[
    ("Cargo.toml", include_bytes!("../Cargo.toml")),
    ("src/document.rs", include_bytes!("document.rs")),
    (
        "src/document/diagnostics.rs",
        include_bytes!("document/diagnostics.rs"),
    ),
    ("src/document/flow.rs", include_bytes!("document/flow.rs")),
    (
        "src/document/request.rs",
        include_bytes!("document/request.rs"),
    ),
    (
        "src/document/resources.rs",
        include_bytes!("document/resources.rs"),
    ),
    ("src/document/scope.rs", include_bytes!("document/scope.rs")),
    (
        "src/document/service.rs",
        include_bytes!("document/service.rs"),
    ),
    (
        "src/document/service/types.rs",
        include_bytes!("document/service/types.rs"),
    ),
    ("src/document/types.rs", include_bytes!("document/types.rs")),
    ("src/lib.rs", include_bytes!("lib.rs")),
    ("src/som_cnl.pest", include_bytes!("som_cnl.pest")),
];

#[derive(Parser)]
#[grammar = "som_cnl.pest"]
struct EnglishCnl;

/// A matched grammar rule. Its byte range indexes the original UTF-8 source.
#[derive(Debug, PartialEq, Eq)]
pub struct Node<'source> {
    pub rule: Rule,
    pub text: &'source str,
    pub bytes: Range<usize>,
    pub children: Vec<Node<'source>>,
}

/// A syntax error. Pest reports the line, column, and byte position in the source.
pub type ParseError = pest::error::Error<Rule>;

/// Parse the entire document. SOM still owns prompt validation and code generation.
pub fn parse(source: &str) -> Result<Node<'_>, Box<ParseError>> {
    let document = EnglishCnl::parse(Rule::document, source)?
        .next()
        .expect("document is a visible grammar rule");
    Ok(node(document))
}

/// Check the same grammar without building the public syntax tree.
pub fn check_syntax(source: &str) -> Result<(), Box<ParseError>> {
    EnglishCnl::parse(Rule::document, source)
        .map(|_| ())
        .map_err(Box::new)
}

fn node(pair: Pair<'_, Rule>) -> Node<'_> {
    let span = pair.as_span();
    Node {
        rule: pair.as_rule(),
        text: pair.as_str(),
        bytes: span.start()..span.end(),
        children: pair
            .into_inner()
            .filter(|child| child.as_rule() != Rule::EOI)
            .map(node)
            .collect(),
    }
}
