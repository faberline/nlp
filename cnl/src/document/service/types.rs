use super::super::{BaseIssue, IssueKind, Phase, SourceSpan, Token};

/// A versioned syntax report for the flat service CNL grammar.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServiceSyntaxReport {
    pub api_version: u32,
    pub eof: SourceSpan,
    /// Events preceding an issue are for diagnostics only.
    pub events: Vec<ServiceEvent>,
    /// The first issue in lexical, version dispatch, then structural order.
    pub issue: Option<ServiceSyntaxIssue>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServiceEvent {
    pub span: SourceSpan,
    pub kind: ServiceEventKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServiceEventKind {
    Header {
        version: Token,
    },
    Package {
        name: Token,
    },
    Target {
        profile: Token,
    },
    ServiceBinding {
        service: Token,
        contract: Token,
    },
    OperationBinding {
        alias: Token,
        operation: Token,
    },
    InputBinding {
        name: Token,
        ty: Token,
        parameter: Token,
    },
    BodyBinding {
        name: Token,
        ty: Token,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServiceSyntaxIssue {
    pub phase: Phase,
    pub kind: IssueKind,
    pub span: SourceSpan,
    pub slot: Option<ServiceSlot>,
    pub rule: ServiceSyntaxRule,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceSlot {
    Header,
    Package,
    Target,
    ServiceBinding,
    OperationBinding,
    Body,
    /// The zero-based position of an input declaration (within its operation in v2).
    Input(usize),
    End,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServiceSyntaxRule {
    Base(BaseIssue),
    UnsupportedVersion,
    MissingStatement,
    DuplicateStatement,
    DuplicateInput,
    ExtraInput,
    UnexpectedStatement,
    ExpectedWord,
    DuplicateOperation,
    InputBeforeOperation,
    BodyBeforeOperation,
    DuplicateBody,
}
