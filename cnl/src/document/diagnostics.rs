use super::{Capability, FlowIssue, ResourceIssue, ScopePart};

/// The source construct selected for a diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnchorKind {
    Token,
    Statement,
    Closing,
    Eof,
}

/// Parser facts that a consumer can map to its own diagnostic format.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyntaxFact {
    pub rule: SyntaxRule,
    /// A grammar slot, with list indices and declared names kept distinct.
    pub scope: Option<Vec<ScopePart>>,
    pub anchor: AnchorKind,
    /// Index of the innermost open block event, when there is one.
    pub block: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SyntaxRule {
    Base(BaseIssue),
    Request(RequestIssue),
    Resource(ResourceIssue),
    Flow(FlowIssue),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BaseIssue {
    BareCrOrBom {
        actual: char,
    },
    MissingPeriod {
        actual: String,
    },
    InvalidJsonString {
        actual: String,
    },
    JsonStringBoundary {
        actual: char,
    },
    IncompleteSentence,
    ExpectedKeyword {
        expected: String,
        actual: String,
    },
    TrailingToken {
        actual: String,
    },
    InvalidIdentifier {
        actual: String,
    },
    InvalidJsonScalar {
        actual: String,
    },
    NonScalarJson {
        actual: String,
    },
    UnexpectedEof,
    MissingScalar {
        form: String,
    },
    ExpectedSentence {
        expected: String,
        actual: Option<String>,
    },
    ExpectedJsonString {
        actual: String,
    },
    Duplicate {
        item: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RequestIssue {
    InvalidHeader {
        actual: String,
    },
    UnsupportedVersion {
        actual: String,
    },
    NoncanonicalVersion {
        actual: String,
        number: usize,
    },
    TrailingContent {
        actual: String,
    },
    UnknownCapability {
        kind: Capability,
        actual: String,
        expected: Vec<String>,
    },
    UnknownRouteScope {
        actual: String,
    },
    UnknownSelectedOperation {
        action: String,
        scope: String,
    },
    MissingRoute,
}
