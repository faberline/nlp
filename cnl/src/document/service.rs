//! Syntax only for the separate, flat SOM service CNL v1 grammar.

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
