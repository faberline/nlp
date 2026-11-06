//! Syntax parser for English SOM CNL v1, with source ranges for each rule.

use std::ops::Range;

use pest::{Parser as _, iterators::Pair};
use pest_derive::Parser;

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
