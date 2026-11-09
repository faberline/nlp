use super::{Predicate, SyntaxFact, ValueSyntax};

/// Positions in the original, decoded UTF-8 document. Ends are exclusive.
/// Lines and columns start at one. Columns count Unicode scalar values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceSpan {
    pub line: usize,
    pub column: usize,
    pub end_line: usize,
    pub end_column: usize,
    pub byte_start: usize,
    pub byte_end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Token {
    pub text: String,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Line {
    pub tokens: Vec<Token>,
    pub span: SourceSpan,
    pub(super) normal: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyntaxReport {
    pub api_version: u32,
    pub lines: Vec<Line>,
    pub eof: SourceSpan,
    pub events: Vec<SyntaxEvent>,
    /// The first issue in lexical, dispatch, then structural check order.
    /// Events preceding an issue are available for diagnosis only.
    pub issue: Option<SyntaxIssue>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Lexical,
    Dispatch,
    Structural,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IssueKind {
    Syntax,
    Missing,
    Duplicate,
    BlockMismatch,
    UnsupportedVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyntaxIssue {
    pub phase: Phase,
    pub kind: IssueKind,
    pub span: SourceSpan,
    pub fact: SyntaxFact,
    /// Compatibility detail from the first version of the syntax report.
    /// Consumers with their own diagnostic format can use `fact` instead.
    pub pointer: Option<String>,
    pub cause: String,
    pub allowed_form: String,
}

/// One event in source order. Parent indexes the opening event in this report.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyntaxEvent {
    pub span: SourceSpan,
    pub parent: Option<usize>,
    pub kind: EventKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
// Keep ordinary clauses inline to avoid one heap allocation per source line.
#[allow(clippy::large_enum_variant)]
pub enum EventKind {
    Begin(Block),
    End(BlockKind),
    Clause(Clause),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResourceRole {
    Single,
    Parent,
    Child,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Block {
    Resource(ResourceRole),
    Query,
    Capabilities,
    Routes,
    Checks,
    Flow(Token),
    Steps,
    Step {
        name: Token,
        operation: Token,
    },
    Transaction {
        name: Token,
        keyword: Token,
    },
    Branch {
        name: Token,
        keyword: Token,
        predicate: Box<Predicate>,
    },
    Then,
    Else,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockKind {
    Resource,
    Query,
    Capabilities,
    Routes,
    Checks,
    Flow,
    Steps,
    Step,
    Transaction,
    Branch,
    Then,
    Else,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Capability {
    Database,
    Authentication,
    Storage,
    Workflow,
    WorkflowValue,
    WorkflowGroup,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum RouteSyntax {
    Legacy(String),
    Scoped { scope: String, sentence: String },
    Selected { scope: String, action: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Clause {
    Header,
    Version {
        number: usize,
        token: Token,
    },
    Package(Token),
    Contract {
        id: Token,
        hash: Token,
    },
    ResourceName(Token),
    Table(Token),
    Path(Token),
    Field {
        name: Token,
        required: bool,
        required_token: Token,
        ty: Token,
        unique: bool,
        unique_token: Token,
        default: Option<Token>,
    },
    ExactFilter {
        field: Option<Token>,
        value_span: SourceSpan,
    },
    DateFilter {
        field: Option<Token>,
        value_span: SourceSpan,
    },
    QueryLimit(Token),
    Relation(Token),
    MigrationRevision(Token),
    PreviousRevision(Option<Token>),
    NoCapabilities,
    Capability {
        kind: Capability,
        value: Option<Token>,
    },
    Route(RouteSyntax),
    Check(String),
    Input {
        name: Token,
        ty: Token,
    },
    NoInputs,
    Argument {
        name: Token,
        value: ValueSyntax,
    },
    Undo {
        mode: Option<Token>,
        value_span: SourceSpan,
    },
    Export {
        name: Token,
        yes: ValueSyntax,
        no: ValueSyntax,
    },
    NoExports,
    Result {
        name: Token,
        value: ValueSyntax,
    },
    NoResults,
    EndRequest,
}
