//! Line-based syntax scanner for the SOM CNL v1 contract.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

mod diagnostics;
mod flow;
mod request;
mod resources;
mod scope;
pub mod service;
mod types;

pub use diagnostics::{AnchorKind, BaseIssue, RequestIssue, SyntaxFact, SyntaxRule};
pub use flow::{Comparison, FlowIssue, FlowNameKind, Predicate, ValueKind, ValueSyntax};
pub use resources::{QuerySelector, ResourceIssue};
pub use scope::{ScopePart, SyntaxSlot};
pub use types::*;

/// Version of the event and diagnostic interface.
pub const SYNTAX_API_VERSION: u32 = 2;

/// Check the entire input lexically before scanning the selected grammar.
/// Stop at the first structural issue and retain the preceding events.
pub fn parse_document(source: &str) -> SyntaxReport {
    let mut scanner = Scanner {
        lines: Vec::new(),
        eof: end_of_source(source),
        at: 0,
        version: 0,
        map: BTreeMap::new(),
        events: Vec::new(),
        parents: Vec::new(),
    };
    let mut issue = match lex(source, &mut scanner.lines) {
        Ok(()) => scanner.document().err().map(|issue| *issue),
        Err(mut issue) => {
            issue.phase = Phase::Lexical;
            Some(*issue)
        }
    };
    if let Some(issue) = &mut issue {
        issue.fact.block = scanner.parents.last().copied();
    }
    SyntaxReport {
        api_version: SYNTAX_API_VERSION,
        lines: scanner.lines,
        eof: scanner.eof,
        events: scanner.events,
        issue,
    }
}

fn end_of_source(source: &str) -> SourceSpan {
    let line = source.bytes().filter(|&byte| byte == b'\n').count() + 1;
    let column = source.rsplit('\n').next().unwrap_or("").chars().count() + 1;
    SourceSpan {
        line,
        column,
        end_line: line,
        end_column: column,
        byte_start: source.len(),
        byte_end: source.len(),
    }
}

fn error(
    code: &str,
    span: SourceSpan,
    pointer: Option<&str>,
    cause: impl Into<String>,
    allowed_form: impl Into<String>,
    anchor: AnchorKind,
    rule: SyntaxRule,
) -> Box<SyntaxIssue> {
    let kind = match code {
        "CNL_SYNTAX" => IssueKind::Syntax,
        "CNL_MISSING" => IssueKind::Missing,
        "CNL_DUPLICATE" => IssueKind::Duplicate,
        "CNL_BLOCK_MISMATCH" => IssueKind::BlockMismatch,
        "UNSUPPORTED_VERSION" => IssueKind::UnsupportedVersion,
        _ => unreachable!("unrecognized syntax issue {code}"),
    };
    let scope = scope::issue_scope(pointer);
    debug_assert_eq!(
        scope.is_some(),
        pointer.is_some(),
        "unknown diagnostic scope"
    );
    Box::new(SyntaxIssue {
        phase: Phase::Structural,
        kind,
        span,
        fact: SyntaxFact {
            rule,
            scope,
            anchor,
            block: None,
        },
        pointer: pointer.map(str::to_owned),
        cause: cause.into(),
        allowed_form: allowed_form.into(),
    })
}

struct Scanner {
    lines: Vec<Line>,
    eof: SourceSpan,
    at: usize,
    version: usize,
    // Track consumed scalar slots to recognize a repeated declaration.
    map: BTreeMap<String, SourceSpan>,
    events: Vec<SyntaxEvent>,
    parents: Vec<usize>,
}

impl Scanner {
    fn event(&mut self, line: &Line, kind: EventKind) {
        self.events.push(SyntaxEvent {
            span: line.span.clone(),
            parent: self.parents.last().copied(),
            kind,
        });
    }

    fn begin(&mut self, line: &Line, block: Block) {
        self.event(line, EventKind::Begin(block));
        self.parents.push(self.events.len() - 1);
    }

    fn close(&mut self, line: &Line, block: BlockKind) {
        self.event(line, EventKind::End(block));
        self.parents.pop().expect("an open block");
    }

    fn clause(&mut self, line: &Line, clause: Clause) {
        self.event(line, EventKind::Clause(clause));
    }
}

fn joined(a: &SourceSpan, b: &SourceSpan) -> SourceSpan {
    SourceSpan {
        line: a.line,
        column: a.column,
        end_line: b.end_line,
        end_column: b.end_column,
        byte_start: a.byte_start,
        byte_end: b.byte_end,
    }
}
fn lex(source: &str, lines: &mut Vec<Line>) -> Result<(), Box<SyntaxIssue>> {
    let mut offset = 0;
    for (i, raw) in source.split_inclusive('\n').enumerate() {
        let lf = raw.ends_with('\n');
        let mut text = if lf { &raw[..raw.len() - 1] } else { raw };
        if lf && text.ends_with('\r') {
            text = &text[..text.len() - 1];
        }
        let columns: Vec<_> = text
            .char_indices()
            .map(|(at, _)| at)
            .chain([text.len()])
            .collect();
        let column = |at| {
            columns
                .binary_search(&at)
                .expect("token boundaries are UTF-8 boundaries")
                + 1
        };
        let sp = |a: usize, b: usize| SourceSpan {
            line: i + 1,
            column: column(a),
            end_line: i + 1,
            end_column: column(b),
            byte_start: offset + a,
            byte_end: offset + b,
        };
        if let Some(p) = text.find('\r').or_else(|| {
            if offset == 0 && text.starts_with('\u{feff}') {
                Some(0)
            } else {
                None
            }
        }) {
            let actual = text[p..].chars().next().unwrap();
            return Err(error(
                "CNL_SYNTAX",
                sp(p, p + actual.len_utf8()),
                None,
                "Bare CR and BOM are not allowed.",
                "Use LF or CRLF and no BOM.",
                AnchorKind::Token,
                SyntaxRule::Base(BaseIssue::BareCrOrBom { actual }),
            ));
        }
        let trim = text.trim_matches([' ', '\t']);
        let start = text.len() - text.trim_start_matches([' ', '\t']).len();
        if !trim.is_empty() {
            let end = start + trim.len();
            let whole = sp(start, end);
            if !trim.ends_with('.') {
                return Err(error(
                    "CNL_SYNTAX",
                    whole,
                    None,
                    "A sentence needs its final period.",
                    "End the complete sentence with a period.",
                    AnchorKind::Statement,
                    SyntaxRule::Base(BaseIssue::MissingPeriod {
                        actual: trim.into(),
                    }),
                ));
            }
            let body_end = end - 1;
            let mut p = start;
            let mut tokens = vec![];
            while p < body_end {
                while p < body_end && matches!(text.as_bytes()[p], b' ' | b'\t') {
                    p += 1;
                }
                if p >= body_end {
                    break;
                }
                let begin = p;
                if text.as_bytes()[p] == b'"' {
                    p += 1;
                    let mut escaped = false;
                    let mut closed = false;
                    while p < body_end {
                        let ch = text[p..].chars().next().unwrap();
                        p += ch.len_utf8();
                        if escaped {
                            escaped = false;
                        } else if ch == '\\' {
                            escaped = true;
                        } else if ch == '"' {
                            closed = true;
                            break;
                        }
                    }
                    if !closed || serde_json::from_str::<String>(&text[begin..p]).is_err() {
                        return Err(error(
                            "CNL_SYNTAX",
                            sp(begin, p),
                            None,
                            "Invalid JSON string literal.",
                            "Use a quoted JSON string with valid escapes.",
                            AnchorKind::Token,
                            SyntaxRule::Base(BaseIssue::InvalidJsonString {
                                actual: text[begin..p].into(),
                            }),
                        ));
                    }
                    if p < body_end && !matches!(text.as_bytes()[p], b' ' | b'\t') {
                        return Err(error(
                            "CNL_SYNTAX",
                            sp(begin, p),
                            None,
                            "A JSON string needs a token boundary.",
                            "Separate grammar tokens with spaces or tabs.",
                            AnchorKind::Token,
                            SyntaxRule::Base(BaseIssue::JsonStringBoundary {
                                actual: text[p..].chars().next().unwrap(),
                            }),
                        ));
                    }
                } else {
                    while p < body_end && !matches!(text.as_bytes()[p], b' ' | b'\t') {
                        p += text[p..].chars().next().unwrap().len_utf8();
                    }
                }
                tokens.push(Token {
                    text: text[begin..p].into(),
                    span: sp(begin, p),
                });
            }
            let normal = tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
                + ".";
            lines.push(Line {
                tokens,
                span: whole,
                normal,
            });
        }
        offset += raw.len();
    }
    Ok(())
}
struct Cursor {
    line: Line,
    at: usize,
}
impl Cursor {
    fn token(&mut self) -> Result<Token, Box<SyntaxIssue>> {
        let t = self.line.tokens.get(self.at).cloned().ok_or_else(|| {
            error(
                "CNL_SYNTAX",
                self.line.span.clone(),
                None,
                "Incomplete sentence.",
                "Use the complete fixed sentence form.",
                AnchorKind::Statement,
                SyntaxRule::Base(BaseIssue::IncompleteSentence),
            )
        })?;
        self.at += 1;
        Ok(t)
    }
    fn word(&mut self, w: &str) -> Result<(), Box<SyntaxIssue>> {
        let t = self.token()?;
        if t.text != w {
            return Err(error(
                "CNL_SYNTAX",
                t.span,
                None,
                format!("Expected {w}."),
                format!("Use the keyword {w}."),
                AnchorKind::Token,
                SyntaxRule::Base(BaseIssue::ExpectedKeyword {
                    expected: w.into(),
                    actual: t.text,
                }),
            ));
        }
        Ok(())
    }
    fn words(&mut self, w: &[&str]) -> Result<(), Box<SyntaxIssue>> {
        for x in w {
            self.word(x)?;
        }
        Ok(())
    }
    fn end(&self) -> Result<(), Box<SyntaxIssue>> {
        if self.at < self.line.tokens.len() {
            Err(error(
                "CNL_SYNTAX",
                self.line.tokens[self.at].span.clone(),
                None,
                "Unexpected trailing token.",
                "Use only the fixed sentence form.",
                AnchorKind::Token,
                SyntaxRule::Base(BaseIssue::TrailingToken {
                    actual: self.line.tokens[self.at].text.clone(),
                }),
            ))
        } else {
            Ok(())
        }
    }
    fn ident(&mut self) -> Result<Token, Box<SyntaxIssue>> {
        let t = self.token()?;
        if !identifier(&t.text) {
            return Err(error(
                "CNL_SYNTAX",
                t.span,
                None,
                "Invalid identifier token.",
                "Use an ASCII Python identifier.",
                AnchorKind::Token,
                SyntaxRule::Base(BaseIssue::InvalidIdentifier { actual: t.text }),
            ));
        }
        Ok(t)
    }
    fn json(&mut self) -> Result<(Value, Token), Box<SyntaxIssue>> {
        let t = self.token()?;
        let v = serde_json::from_str::<Value>(&t.text).map_err(|_| {
            error(
                "CNL_SYNTAX",
                t.span.clone(),
                None,
                "Invalid JSON scalar.",
                "Use one JSON scalar literal.",
                AnchorKind::Token,
                SyntaxRule::Base(BaseIssue::InvalidJsonScalar {
                    actual: t.text.clone(),
                }),
            )
        })?;
        if v.is_object() || v.is_array() {
            return Err(error(
                "CNL_SYNTAX",
                t.span,
                None,
                "Objects and arrays are not scalar literals.",
                "Use a JSON scalar or declared typed input.",
                AnchorKind::Token,
                SyntaxRule::Base(BaseIssue::NonScalarJson { actual: t.text }),
            ));
        }
        Ok((v, t))
    }
}
fn identifier(s: &str) -> bool {
    let mut it = s.bytes();
    it.next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && it.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

impl Scanner {
    fn peek(&self) -> Option<&Line> {
        self.lines.get(self.at)
    }
    fn normal(&self) -> &str {
        self.peek().map(|l| l.normal.as_str()).unwrap_or("")
    }
    fn take(&mut self) -> Result<Cursor, Box<SyntaxIssue>> {
        let line = self.peek().cloned().ok_or_else(|| {
            error(
                "CNL_MISSING",
                self.eof.clone(),
                None,
                "Unexpected end of input.",
                "Supply the required closing sentence.",
                AnchorKind::Eof,
                SyntaxRule::Base(BaseIssue::UnexpectedEof),
            )
        })?;
        self.at += 1;
        Ok(Cursor { line, at: 0 })
    }
    fn repeated_scalar(&self, context: &str) -> Option<Box<SyntaxIssue>> {
        let l = self.peek()?;
        let n = l.normal.as_str();
        let direct = [
            ("Use package ", "/package"),
            ("Use prompt version ", "/version"),
            ("Bind contract ", "/contract/id"),
            ("Use migration revision ", "/migration/revision"),
            (
                "Use previous migration revision ",
                "/migration/down_revision",
            ),
            (
                "Use no previous migration revision.",
                "/migration/down_revision",
            ),
            ("Use database ", "/capabilities/database"),
            ("Use authentication ", "/capabilities/auth"),
            ("Use object storage ", "/capabilities/storage"),
            ("Use workflow value field ", "/capabilities/value_field"),
            ("Use no workflow value field.", "/capabilities/value_field"),
            ("Use workflow group field ", "/capabilities/group_field"),
            ("Use no workflow group field.", "/capabilities/group_field"),
            ("Use workflow ", "/capabilities/workflow"),
        ];
        let mut ptr = direct
            .into_iter()
            .find(|(prefix, _)| n.starts_with(prefix))
            .map(|(_, ptr)| ptr.to_owned());
        for (prefix, key) in [
            ("Use resource name ", "name"),
            ("Use table ", "table"),
            ("Use path ", "path"),
        ] {
            if n.starts_with(prefix) {
                let role = if context.starts_with("/resources/parent") {
                    Some("/resources/parent")
                } else if context.starts_with("/resources/child") {
                    Some("/resources/child")
                } else if context.starts_with("/resource/") {
                    Some("/resource")
                } else {
                    None
                };
                ptr = role.map(|r| format!("{r}/{key}")).or_else(|| {
                    self.map
                        .iter()
                        .filter(|(p, _)| {
                            p.ends_with(&format!("/{key}"))
                                && (p.starts_with("/resource/") || p.starts_with("/resources/"))
                        })
                        .max_by_key(|(_, s)| s.byte_start)
                        .map(|(p, _)| p.clone())
                });
            }
        }
        let ptr = ptr?;
        if self.map.contains_key(&ptr) {
            Some(self.duplicate(&ptr, l, "scalar clause"))
        } else {
            None
        }
    }
    fn missing_scalar(&self, p: &str, form: &str) -> Box<SyntaxIssue> {
        let end = if p.starts_with("/resources/parent/") {
            "End parent resource."
        } else if p.starts_with("/resources/child/") {
            "End child resource."
        } else if p.starts_with("/resource/") {
            "End resource."
        } else if p.starts_with("/capabilities/") {
            "End capabilities."
        } else {
            "End request."
        };
        let close = self.lines[self.at..].iter().find(|l| l.normal == end);
        let (span, anchor) = close.map_or_else(
            || (self.eof.clone(), AnchorKind::Eof),
            |line| (line.span.clone(), AnchorKind::Closing),
        );
        error(
            "CNL_MISSING",
            span,
            Some(p),
            "Required scalar clause is missing.",
            form,
            anchor,
            SyntaxRule::Base(BaseIssue::MissingScalar { form: form.into() }),
        )
    }
    fn expected(&mut self, s: &str) -> Result<Line, Box<SyntaxIssue>> {
        if self.normal() == s {
            Ok(self.take()?.line)
        } else {
            if let Some(e) = self.repeated_scalar("") {
                return Err(e);
            }
            let span = self
                .peek()
                .map(|l| l.span.clone())
                .unwrap_or_else(|| self.eof.clone());
            let (code, anchor) = if self.peek().is_none() {
                ("CNL_MISSING", AnchorKind::Eof)
            } else if self.normal().starts_with("End ") {
                ("CNL_BLOCK_MISMATCH", AnchorKind::Closing)
            } else {
                ("CNL_SYNTAX", AnchorKind::Statement)
            };
            Err(error(
                code,
                span,
                None,
                format!("Expected {s}"),
                s,
                anchor,
                SyntaxRule::Base(BaseIssue::ExpectedSentence {
                    expected: s.into(),
                    actual: self.peek().map(|line| line.normal.clone()),
                }),
            ))
        }
    }
    fn locate(&mut self, p: &str, t: &Token) {
        self.map.insert(p.into(), t.span.clone());
    }

    fn span(&mut self, p: &str, l: &Line) {
        self.map.insert(p.into(), l.span.clone());
    }

    fn scalar(
        &mut self,
        w: &[&str],
        p: &str,
        string: bool,
    ) -> Result<(Token, Line), Box<SyntaxIssue>> {
        let prefix = w.join(" ") + " ";
        if !self.normal().starts_with(&prefix) {
            if let Some(e) = self.repeated_scalar(p) {
                return Err(e);
            }
            if self.peek().is_none()
                || self.normal().starts_with("End ")
                || self.normal().starts_with("Use ")
                || self.normal().starts_with("Begin ")
                || self.normal().starts_with("Field ")
            {
                return Err(self.missing_scalar(
                    p,
                    &format!(
                        "{} {}.",
                        w.join(" "),
                        if string { "JSON_STRING" } else { "IDENT" }
                    ),
                ));
            }
        }
        let mut c = self.take()?;
        c.words(w)?;
        let t = c.token()?;
        c.end()?;
        if string {
            serde_json::from_str::<String>(&t.text).map_err(|_| {
                error(
                    "CNL_SYNTAX",
                    t.span.clone(),
                    None,
                    "Expected JSON string.",
                    "Use a quoted JSON string.",
                    AnchorKind::Token,
                    SyntaxRule::Base(BaseIssue::ExpectedJsonString {
                        actual: t.text.clone(),
                    }),
                )
            })?;
        } else if !identifier(&t.text) {
            return Err(error(
                "CNL_SYNTAX",
                t.span,
                None,
                "Invalid identifier token.",
                "Use an ASCII Python identifier.",
                AnchorKind::Token,
                SyntaxRule::Base(BaseIssue::InvalidIdentifier { actual: t.text }),
            ));
        }
        self.locate(p, &t);
        Ok((t, c.line))
    }

    fn duplicate(&self, p: &str, l: &Line, what: &str) -> Box<SyntaxIssue> {
        error(
            "CNL_DUPLICATE",
            l.span.clone(),
            Some(p),
            format!("Duplicate {what}."),
            format!("Declare {what} exactly once."),
            AnchorKind::Statement,
            SyntaxRule::Base(BaseIssue::Duplicate { item: what.into() }),
        )
    }
    fn missing(
        &self,
        p: &str,
        l: &Line,
        cause: &str,
        allowed: &str,
        anchor: AnchorKind,
        rule: SyntaxRule,
    ) -> Box<SyntaxIssue> {
        error(
            "CNL_MISSING",
            l.span.clone(),
            Some(p),
            cause,
            allowed,
            anchor,
            rule,
        )
    }
}
