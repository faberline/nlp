//! Syntax only for the separate SOM service CNL grammar, versions 1 and 2.

mod types;

pub use types::*;

use super::{
    BaseIssue, Cursor, IssueKind, Line, Phase, SourceSpan, SyntaxIssue, SyntaxRule, Token,
};
use super::{end_of_source, lex};

pub const SERVICE_SYNTAX_API_VERSION: u32 = 1;

const STATEMENTS: [ServiceSlot; 7] = [
    ServiceSlot::Header,
    ServiceSlot::Package,
    ServiceSlot::Target,
    ServiceSlot::ServiceBinding,
    ServiceSlot::OperationBinding,
    ServiceSlot::Input(0),
    ServiceSlot::Input(1),
];

/// Scan the full document before reporting a version or statement error.
pub fn parse_service_document(source: &str) -> ServiceSyntaxReport {
    let mut report = ServiceSyntaxReport {
        api_version: SERVICE_SYNTAX_API_VERSION,
        eof: end_of_source(source),
        events: Vec::new(),
        issue: None,
    };
    let mut lines = Vec::new();
    report.issue = match lex(source, &mut lines) {
        Ok(()) => scan(lines, &report.eof, &mut report.events).err(),
        Err(issue) => Some(base_issue(*issue, Phase::Lexical, None)),
    };
    report
}

fn scan(
    lines: Vec<Line>,
    eof: &SourceSpan,
    events: &mut Vec<ServiceEvent>,
) -> Result<(), ServiceSyntaxIssue> {
    let mut lines = lines.into_iter();
    for (index, slot) in STATEMENTS.into_iter().enumerate() {
        let phase = if slot == ServiceSlot::Header {
            Phase::Dispatch
        } else {
            Phase::Structural
        };
        let line = lines.next().ok_or_else(|| ServiceSyntaxIssue {
            phase,
            kind: IssueKind::Missing,
            span: eof.clone(),
            slot: Some(slot),
            rule: ServiceSyntaxRule::MissingStatement,
        })?;
        if phase == Phase::Structural
            && let Some(observed) = statement_slot(&line)
            && observed != slot
            && !(observed == ServiceSlot::Input(0) && slot == ServiceSlot::Input(1))
        {
            return Err(unexpected(line, Some(observed), slot, &STATEMENTS[..index]));
        }
        let event = Statement {
            cursor: Cursor { line, at: 0 },
            phase,
            slot,
        }
        .parse()?;
        let is_v2 =
            matches!(&event.kind, ServiceEventKind::Header { version } if version.text == "2");
        if is_v2 {
            events.push(event);
            return scan_v2(lines, eof, events);
        }
        if let ServiceEventKind::InputBinding { name, .. } = &event.kind
            && let Some(ServiceEventKind::InputBinding { name: first, .. }) =
                events.last().map(|previous| &previous.kind)
            && name.text == first.text
        {
            return Err(ServiceSyntaxIssue {
                phase,
                kind: IssueKind::Duplicate,
                span: name.span.clone(),
                slot: Some(slot),
                rule: ServiceSyntaxRule::DuplicateInput,
            });
        }
        events.push(event);
    }
    if let Some(line) = lines.next() {
        let observed = statement_slot(&line);
        return Err(unexpected(line, observed, ServiceSlot::End, &STATEMENTS));
    }
    Ok(())
}

/// Version 2: the prelude is followed by one or more operations, each with
/// its own inputs and an optional request body. The header is already pushed.
fn scan_v2(
    mut lines: std::vec::IntoIter<Line>,
    eof: &SourceSpan,
    events: &mut Vec<ServiceEvent>,
) -> Result<(), ServiceSyntaxIssue> {
    use ServiceSlot::*;
    let issue = |kind, span, slot, rule| ServiceSyntaxIssue {
        phase: Phase::Structural,
        kind,
        span,
        slot: Some(slot),
        rule,
    };
    let parse = |line: Line, slot| {
        Statement {
            cursor: Cursor { line, at: 0 },
            phase: Phase::Structural,
            slot,
        }
        .parse()
    };

    let mut consumed = vec![Header];
    for slot in [Package, Target, ServiceBinding] {
        let Some(line) = lines.next() else {
            return Err(issue(
                IssueKind::Missing,
                eof.clone(),
                slot,
                ServiceSyntaxRule::MissingStatement,
            ));
        };
        let observed = statement_slot_v2(&line);
        if observed != Some(slot) {
            return Err(unexpected(line, observed, slot, &consumed));
        }
        events.push(parse(line, slot)?);
        consumed.push(slot);
    }

    let mut aliases: Vec<String> = Vec::new();
    let mut inputs: Vec<String> = Vec::new();
    let mut body = false;
    for line in lines {
        let have_op = !aliases.is_empty();
        let span = line.span.clone();
        match statement_slot_v2(&line) {
            Some(OperationBinding) => {
                let event = parse(line, OperationBinding)?;
                let ServiceEventKind::OperationBinding { alias, .. } = &event.kind else {
                    unreachable!("an operation slot parses to an operation binding");
                };
                if aliases.contains(&alias.text) {
                    return Err(issue(
                        IssueKind::Duplicate,
                        alias.span.clone(),
                        OperationBinding,
                        ServiceSyntaxRule::DuplicateOperation,
                    ));
                }
                aliases.push(alias.text.clone());
                inputs.clear();
                body = false;
                events.push(event);
            }
            Some(Input(_)) => {
                if !have_op {
                    return Err(issue(
                        IssueKind::Syntax,
                        span,
                        Input(0),
                        ServiceSyntaxRule::InputBeforeOperation,
                    ));
                }
                let slot = Input(inputs.len());
                if body {
                    return Err(issue(
                        IssueKind::Syntax,
                        span,
                        slot,
                        ServiceSyntaxRule::UnexpectedStatement,
                    ));
                }
                let event = parse(line, slot)?;
                let ServiceEventKind::InputBinding { name, .. } = &event.kind else {
                    unreachable!("an input slot parses to an input binding");
                };
                if inputs.contains(&name.text) {
                    return Err(issue(
                        IssueKind::Duplicate,
                        name.span.clone(),
                        slot,
                        ServiceSyntaxRule::DuplicateInput,
                    ));
                }
                inputs.push(name.text.clone());
                events.push(event);
            }
            Some(Body) => {
                if !have_op {
                    return Err(issue(
                        IssueKind::Syntax,
                        span,
                        Body,
                        ServiceSyntaxRule::BodyBeforeOperation,
                    ));
                }
                if body {
                    return Err(issue(
                        IssueKind::Duplicate,
                        span,
                        Body,
                        ServiceSyntaxRule::DuplicateBody,
                    ));
                }
                events.push(parse(line, Body)?);
                body = true;
            }
            Some(slot) => {
                return Err(issue(
                    IssueKind::Duplicate,
                    span,
                    slot,
                    ServiceSyntaxRule::DuplicateStatement,
                ));
            }
            None => {
                let slot = if have_op { End } else { OperationBinding };
                return Err(issue(
                    IssueKind::Syntax,
                    span,
                    slot,
                    ServiceSyntaxRule::UnexpectedStatement,
                ));
            }
        }
    }
    if aliases.is_empty() {
        return Err(issue(
            IssueKind::Missing,
            eof.clone(),
            OperationBinding,
            ServiceSyntaxRule::MissingStatement,
        ));
    }
    Ok(())
}

/// Version 2 adds the body statement; version 1 keeps `statement_slot` as is.
fn statement_slot_v2(line: &Line) -> Option<ServiceSlot> {
    let mut words = line.tokens.iter().map(|token| token.text.as_str());
    match (words.next(), words.next()) {
        (Some("Send"), Some("body")) => Some(ServiceSlot::Body),
        _ => statement_slot(line),
    }
}

fn statement_slot(line: &Line) -> Option<ServiceSlot> {
    use ServiceSlot::*;
    let mut words = line.tokens.iter().map(|token| token.text.as_str());
    match (words.next(), words.next()) {
        (Some("Service"), _) => Some(Header),
        (Some("Package"), _) => Some(Package),
        (Some("Target"), _) => Some(Target),
        (Some("Bind"), Some("service")) => Some(ServiceBinding),
        (Some("Bind"), Some("operation")) => Some(OperationBinding),
        (Some("Accept"), _) => Some(Input(0)),
        _ => None,
    }
}

fn unexpected(
    line: Line,
    observed: Option<ServiceSlot>,
    expected: ServiceSlot,
    consumed: &[ServiceSlot],
) -> ServiceSyntaxIssue {
    let (kind, slot, rule) = match observed {
        Some(ServiceSlot::Input(0)) if expected == ServiceSlot::End => (
            IssueKind::Syntax,
            ServiceSlot::Input(2),
            ServiceSyntaxRule::ExtraInput,
        ),
        Some(slot) if consumed.contains(&slot) => (
            IssueKind::Duplicate,
            slot,
            ServiceSyntaxRule::DuplicateStatement,
        ),
        _ => (
            IssueKind::Syntax,
            expected,
            ServiceSyntaxRule::UnexpectedStatement,
        ),
    };
    ServiceSyntaxIssue {
        phase: Phase::Structural,
        kind,
        span: line.span,
        slot: Some(slot),
        rule,
    }
}

struct Statement {
    cursor: Cursor,
    phase: Phase,
    slot: ServiceSlot,
}

impl Statement {
    fn parse(mut self) -> Result<ServiceEvent, ServiceSyntaxIssue> {
        use ServiceEventKind::*;
        let kind = match self.slot {
            ServiceSlot::Header => {
                self.words(&["Service", "prompt", "version"])?;
                Header {
                    version: self.token()?,
                }
            }
            ServiceSlot::Package => {
                self.words(&["Package"])?;
                Package {
                    name: self.raw_word()?,
                }
            }
            ServiceSlot::Target => {
                self.words(&["Target"])?;
                Target {
                    profile: self.raw_word()?,
                }
            }
            ServiceSlot::ServiceBinding => {
                self.words(&["Bind", "service"])?;
                let service = self.raw_word()?;
                self.words(&["using", "contract"])?;
                let contract = self.token()?;
                if !contract.text.starts_with('"') {
                    return Err(self.issue(
                        IssueKind::Syntax,
                        contract.span,
                        ServiceSyntaxRule::Base(BaseIssue::ExpectedJsonString {
                            actual: contract.text,
                        }),
                    ));
                }
                ServiceBinding { service, contract }
            }
            ServiceSlot::OperationBinding => {
                self.words(&["Bind", "operation"])?;
                let alias = self.raw_word()?;
                self.words(&["to"])?;
                OperationBinding {
                    alias,
                    operation: self.raw_word()?,
                }
            }
            ServiceSlot::Body => {
                self.words(&["Send", "body"])?;
                let name = self.raw_word()?;
                self.words(&["as"])?;
                BodyBinding {
                    name,
                    ty: self.raw_word()?,
                }
            }
            ServiceSlot::Input(_) => {
                self.words(&["Accept"])?;
                let name = self.raw_word()?;
                self.words(&["as"])?;
                let ty = self.raw_word()?;
                self.words(&["for"])?;
                InputBinding {
                    name,
                    ty,
                    parameter: self.raw_word()?,
                }
            }
            ServiceSlot::End => unreachable!("end of input is not a statement"),
        };
        self.cursor.end().map_err(|issue| self.base(*issue))?;
        if let Header { version } = &kind
            && version.text != "1"
            && version.text != "2"
        {
            return Err(self.issue(
                IssueKind::UnsupportedVersion,
                version.span.clone(),
                ServiceSyntaxRule::UnsupportedVersion,
            ));
        }
        Ok(ServiceEvent {
            span: self.cursor.line.span,
            kind,
        })
    }

    fn words(&mut self, words: &[&str]) -> Result<(), ServiceSyntaxIssue> {
        self.cursor.words(words).map_err(|issue| self.base(*issue))
    }

    fn token(&mut self) -> Result<Token, ServiceSyntaxIssue> {
        self.cursor.token().map_err(|issue| self.base(*issue))
    }

    fn raw_word(&mut self) -> Result<Token, ServiceSyntaxIssue> {
        let token = self.token()?;
        if token.text.starts_with('"') {
            return Err(self.issue(
                IssueKind::Syntax,
                token.span,
                ServiceSyntaxRule::ExpectedWord,
            ));
        }
        Ok(token)
    }

    fn issue(
        &self,
        kind: IssueKind,
        span: SourceSpan,
        rule: ServiceSyntaxRule,
    ) -> ServiceSyntaxIssue {
        ServiceSyntaxIssue {
            phase: self.phase,
            kind,
            span,
            slot: Some(self.slot),
            rule,
        }
    }

    fn base(&self, issue: SyntaxIssue) -> ServiceSyntaxIssue {
        base_issue(issue, self.phase, Some(self.slot))
    }
}

fn base_issue(issue: SyntaxIssue, phase: Phase, slot: Option<ServiceSlot>) -> ServiceSyntaxIssue {
    let SyntaxRule::Base(rule) = issue.fact.rule else {
        unreachable!("the shared lexer and cursor only produce base rules");
    };
    ServiceSyntaxIssue {
        phase,
        kind: issue.kind,
        span: issue.span,
        slot,
        rule: ServiceSyntaxRule::Base(rule),
    }
}
