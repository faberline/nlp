use super::*;
use std::collections::{HashMap, HashSet};

/// A runtime value in the source grammar, before prompt validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValueSyntax {
    pub span: SourceSpan,
    pub kind: ValueKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValueKind {
    Input(Token),
    Step { name: Token, field: Token },
    Literal { ty: Token, value: Token },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Comparison {
    IsNull,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Predicate {
    pub left: ValueSyntax,
    pub operator: Comparison,
    pub operator_span: SourceSpan,
    pub right: Option<ValueSyntax>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlowNameKind {
    Input,
    Step,
    Transaction,
    Branch,
    Export,
}

impl FlowNameKind {
    fn label(self) -> &'static str {
        match self {
            Self::Input => "Input",
            Self::Step => "Step",
            Self::Transaction => "Transaction",
            Self::Branch => "Branch",
            Self::Export => "Export",
        }
    }
}

/// Structured facts for errors specific to the flow grammar.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FlowIssue {
    DuplicateName {
        current: FlowNameKind,
        prior: FlowNameKind,
        name: String,
    },
    MissingBlockEnd {
        expected: String,
    },
    MismatchedBlockDelimiter {
        expected: String,
        observed: String,
    },
    UnsupportedInputType {
        actual: String,
    },
    MissingInputDeclaration,
    UnknownNodeKind {
        actual: String,
    },
    UnknownOperation {
        actual: String,
    },
    DuplicateArgument {
        name: String,
    },
    UnknownUndoMode {
        actual: String,
    },
    MissingUndoDeclaration,
    MissingBranchExport,
    DuplicateResultName {
        name: String,
    },
    MissingResultDeclaration,
    ResultFieldNotString {
        actual: String,
    },
    UnknownScalarLiteralType {
        actual: String,
    },
    UnknownRuntimeReference {
        actual: String,
    },
    UnknownAtComparison {
        actual: String,
    },
    UnknownComparison {
        actual: String,
    },
}

#[derive(Clone, Copy)]
enum ListPath {
    Steps,
    Nested { container: usize, arm: &'static str },
}

#[derive(Clone, Copy)]
struct NodePath {
    list: ListPath,
    index: usize,
}

/// Keep one parent link per container. Format a path only when returning an issue.
#[derive(Default)]
struct Paths(Vec<NodePath>);

impl Paths {
    fn add(&mut self, node: NodePath) -> usize {
        let id = self.0.len();
        self.0.push(node);
        id
    }

    fn list(&self, mut list: ListPath) -> String {
        use std::fmt::Write;
        let mut segments = Vec::new();
        while let ListPath::Nested { container, arm } = list {
            let node = self.0[container];
            segments.push((node.index, arm));
            list = node.list;
        }
        let mut pointer = String::from("/flow/steps");
        for (index, arm) in segments.into_iter().rev() {
            let _ = write!(pointer, "/{index}/{arm}");
        }
        pointer
    }

    fn node(&self, node: NodePath, suffix: &str) -> String {
        format!("{}/{}{}", self.list(node.list), node.index, suffix)
    }
}

enum Closing {
    Steps,
    Transaction(String),
    Then,
    Else,
}

impl Closing {
    fn matches(&self, line: &str) -> bool {
        match self {
            Self::Steps => line == "End steps.",
            Self::Transaction(name) => line
                .strip_prefix("End transaction ")
                .and_then(|line| line.strip_suffix('.'))
                .is_some_and(|actual| actual == name),
            Self::Then => line == "End then.",
            Self::Else => line == "End else.",
        }
    }

    fn sentence(&self) -> String {
        match self {
            Self::Steps => "End steps.".into(),
            Self::Transaction(name) => format!("End transaction {name}."),
            Self::Then => "End then.".into(),
            Self::Else => "End else.".into(),
        }
    }

    fn block_kind(&self) -> BlockKind {
        match self {
            Self::Steps => BlockKind::Steps,
            Self::Transaction(_) => BlockKind::Transaction,
            Self::Then => BlockKind::Then,
            Self::Else => BlockKind::Else,
        }
    }
}

enum Task {
    List {
        path: ListPath,
        closer: Closing,
        index: usize,
    },
    Then(usize),
    Else(usize),
    Exports {
        node: NodePath,
        name: String,
    },
}

#[derive(Default)]
struct FlowState {
    names: HashMap<String, FlowNameKind>,
    paths: Paths,
    tasks: Vec<Task>,
}

impl FlowState {
    fn declare(
        &mut self,
        name: &Token,
        current: FlowNameKind,
        pointer: impl FnOnce(&Paths) -> String,
    ) -> Result<(), Box<SyntaxIssue>> {
        if let Some(&prior) = self.names.get(&name.text) {
            let label = current.label();
            return Err(error(
                "CNL_DUPLICATE",
                name.span.clone(),
                Some(&pointer(&self.paths)),
                format!(
                    "{label} name {} conflicts with the earlier {} name {}.",
                    name.text,
                    prior.label().to_ascii_lowercase(),
                    name.text
                ),
                "Use unique names across inputs and nodes.",
                AnchorKind::Token,
                SyntaxRule::Flow(FlowIssue::DuplicateName {
                    current,
                    prior,
                    name: name.text.clone(),
                }),
            ));
        }
        self.names.insert(name.text.clone(), current);
        Ok(())
    }

    fn list(&mut self, path: ListPath, closer: Closing) {
        self.tasks.push(Task::List {
            path,
            closer,
            index: 0,
        });
    }
}

impl Scanner {
    fn flow_anchor(&self) -> (SourceSpan, AnchorKind) {
        match self.peek() {
            Some(line) if line.normal.starts_with("End ") => {
                (line.span.clone(), AnchorKind::Closing)
            }
            Some(line) => (line.span.clone(), AnchorKind::Statement),
            None => (self.eof.clone(), AnchorKind::Eof),
        }
    }

    pub(super) fn flow(&mut self) -> Result<(), Box<SyntaxIssue>> {
        let mut state = FlowState::default();
        let mut opener = self.take()?;
        opener.words(&["Begin", "flow"])?;
        let name = opener.ident()?;
        opener.end()?;
        let closer = format!("End flow {}.", name.text);
        self.begin(&opener.line, Block::Flow(name));
        self.flow_inputs(&mut state)?;

        let line = self.expected("Begin steps.")?;
        self.begin(&line, Block::Steps);
        state.list(ListPath::Steps, Closing::Steps);
        while let Some(task) = state.tasks.pop() {
            match task {
                Task::List {
                    path,
                    closer,
                    index,
                } => {
                    if closer.matches(self.normal()) {
                        let line = self.take()?.line;
                        self.close(&line, closer.block_kind());
                        continue;
                    }
                    let line = self.peek().ok_or_else(|| {
                        let expected = closer.sentence();
                        error(
                            "CNL_MISSING",
                            self.eof.clone(),
                            Some(&state.paths.list(path)),
                            "Missing block end.",
                            expected.clone(),
                            AnchorKind::Eof,
                            SyntaxRule::Flow(FlowIssue::MissingBlockEnd { expected }),
                        )
                    })?;
                    if line.normal.starts_with("End ") {
                        let expected = closer.sentence();
                        return Err(error(
                            "CNL_BLOCK_MISMATCH",
                            line.span.clone(),
                            Some(&state.paths.list(path)),
                            "Mismatched block delimiter.",
                            expected.clone(),
                            AnchorKind::Closing,
                            SyntaxRule::Flow(FlowIssue::MismatchedBlockDelimiter {
                                expected,
                                observed: line.normal.clone(),
                            }),
                        ));
                    }
                    state.tasks.push(Task::List {
                        path,
                        closer,
                        index: index + 1,
                    });
                    self.flow_node(&mut state, NodePath { list: path, index })?;
                }
                Task::Then(container) => {
                    let line = self.expected("Begin then.")?;
                    self.begin(&line, Block::Then);
                    state.list(
                        ListPath::Nested {
                            container,
                            arm: "then",
                        },
                        Closing::Then,
                    );
                }
                Task::Else(container) => {
                    let line = self.expected("Begin else.")?;
                    self.begin(&line, Block::Else);
                    state.list(
                        ListPath::Nested {
                            container,
                            arm: "else",
                        },
                        Closing::Else,
                    );
                }
                Task::Exports { node, name } => {
                    self.flow_exports(&mut state, node)?;
                    let line = self.expected(&format!("End branch {name}."))?;
                    self.close(&line, BlockKind::Branch);
                }
            }
        }
        self.flow_results()?;
        let line = self.expected(&closer)?;
        self.close(&line, BlockKind::Flow);
        Ok(())
    }

    fn flow_inputs(&mut self, state: &mut FlowState) -> Result<(), Box<SyntaxIssue>> {
        if self.normal() == "Use no inputs." {
            let line = self.take()?.line;
            self.clause(&line, Clause::NoInputs);
            return Ok(());
        }
        let mut count = 0;
        while self.normal().starts_with("Input ") {
            let mut cursor = self.take()?;
            cursor.word("Input")?;
            let name = cursor.ident()?;
            state.declare(&name, FlowNameKind::Input, |_| {
                format!("/flow/inputs/{count}/name")
            })?;
            cursor.words(&["has", "type"])?;
            let ty = cursor.token()?;
            cursor.end()?;
            if ![
                "string",
                "integer",
                "float",
                "boolean",
                "date",
                "parent_input",
                "child_input",
                "parent_patch",
                "child_patch",
                "children_input",
            ]
            .contains(&ty.text.as_str())
            {
                return Err(error(
                    "CNL_SYNTAX",
                    ty.span,
                    None,
                    "Unsupported input type token.",
                    "Use a type listed by the CNL input grammar.",
                    AnchorKind::Token,
                    SyntaxRule::Flow(FlowIssue::UnsupportedInputType { actual: ty.text }),
                ));
            }
            self.clause(&cursor.line, Clause::Input { name, ty });
            count += 1;
        }
        if count == 0 {
            let (span, anchor) = self.flow_anchor();
            return Err(error(
                "CNL_MISSING",
                span,
                Some("/flow/inputs"),
                "Input declaration is missing.",
                "Declare inputs or Use no inputs.",
                anchor,
                SyntaxRule::Flow(FlowIssue::MissingInputDeclaration),
            ));
        }
        Ok(())
    }

    fn flow_node(&mut self, state: &mut FlowState, node: NodePath) -> Result<(), Box<SyntaxIssue>> {
        let mut cursor = self.take()?;
        cursor.word("Begin")?;
        let kind = cursor.token()?;
        let name = cursor.ident()?;
        let label = match kind.text.as_str() {
            "step" => FlowNameKind::Step,
            "transaction" => FlowNameKind::Transaction,
            "branch" => FlowNameKind::Branch,
            _ => {
                return Err(error(
                    "CNL_SYNTAX",
                    kind.span,
                    None,
                    "Unknown node kind.",
                    "Begin step, transaction, or branch.",
                    AnchorKind::Token,
                    SyntaxRule::Flow(FlowIssue::UnknownNodeKind { actual: kind.text }),
                ));
            }
        };
        state.declare(&name, label, |paths| paths.node(node, "/name"))?;
        match kind.text.as_str() {
            "transaction" => {
                cursor.end()?;
                let closer = Closing::Transaction(name.text.clone());
                self.begin(
                    &cursor.line,
                    Block::Transaction {
                        name,
                        keyword: kind,
                    },
                );
                let container = state.paths.add(node);
                state.list(
                    ListPath::Nested {
                        container,
                        arm: "steps",
                    },
                    closer,
                );
            }
            "branch" => {
                cursor.word("when")?;
                let predicate = predicate(&mut cursor)?;
                cursor.end()?;
                let close_name = name.text.clone();
                self.begin(
                    &cursor.line,
                    Block::Branch {
                        name,
                        keyword: kind,
                        predicate: Box::new(predicate),
                    },
                );
                let container = state.paths.add(node);
                state.tasks.push(Task::Exports {
                    node,
                    name: close_name,
                });
                state.tasks.push(Task::Else(container));
                state.tasks.push(Task::Then(container));
            }
            _ => self.flow_step(cursor, name, &state.paths, node)?,
        }
        Ok(())
    }

    fn flow_step(
        &mut self,
        mut cursor: Cursor,
        name: Token,
        paths: &Paths,
        node: NodePath,
    ) -> Result<(), Box<SyntaxIssue>> {
        cursor.words(&["with", "operation"])?;
        let operation = cursor.token()?;
        cursor.end()?;
        if ![
            "authorize",
            "parent.create",
            "parent.get",
            "parent.list",
            "parent.patch",
            "child.create",
            "child.get",
            "child.list",
            "child.patch",
            "batch.create",
            "workflow.run",
            "workflow.import",
            "object.put",
            "object.get",
        ]
        .contains(&operation.text.as_str())
        {
            return Err(error(
                "CNL_SYNTAX",
                operation.span,
                None,
                "Operation token is not in the CNL operation vocabulary.",
                "Use one operation listed by the grammar.",
                AnchorKind::Token,
                SyntaxRule::Flow(FlowIssue::UnknownOperation {
                    actual: operation.text,
                }),
            ));
        }
        let closer = format!("End step {}.", name.text);
        self.begin(&cursor.line, Block::Step { name, operation });
        let mut args = HashSet::new();
        while self.normal().starts_with("Set argument ") {
            let mut cursor = self.take()?;
            cursor.words(&["Set", "argument"])?;
            let name = cursor.ident()?;
            if !args.insert(name.text.clone()) {
                return Err(duplicate(
                    &paths.node(node, &format!("/args/{}", name.text)),
                    &cursor.line,
                    "argument",
                    FlowIssue::DuplicateArgument { name: name.text },
                ));
            }
            cursor.word("to")?;
            let value = flow_value(&mut cursor)?;
            cursor.end()?;
            self.clause(&cursor.line, Clause::Argument { name, value });
        }
        self.flow_undo(paths, node)?;
        let line = self.expected(&closer)?;
        self.close(&line, BlockKind::Step);
        Ok(())
    }

    fn flow_undo(&mut self, paths: &Paths, node: NodePath) -> Result<(), Box<SyntaxIssue>> {
        if self.normal() == "On failure, use no undo." {
            let line = self.take()?.line;
            let value_span = joined(&line.tokens[3].span, &line.tokens[4].span);
            self.clause(
                &line,
                Clause::Undo {
                    mode: None,
                    value_span,
                },
            );
        } else if self.normal().starts_with("On failure, undo with ") {
            let mut cursor = self.take()?;
            cursor.words(&["On", "failure,", "undo", "with"])?;
            let mode = cursor.token()?;
            cursor.end()?;
            if !["delete_created", "restore_patch"].contains(&mode.text.as_str()) {
                return Err(error(
                    "CNL_SYNTAX",
                    mode.span,
                    None,
                    "Unknown undo mode.",
                    "Use delete_created, restore_patch, or no undo.",
                    AnchorKind::Token,
                    SyntaxRule::Flow(FlowIssue::UnknownUndoMode { actual: mode.text }),
                ));
            }
            let value_span = mode.span.clone();
            self.clause(
                &cursor.line,
                Clause::Undo {
                    mode: Some(mode),
                    value_span,
                },
            );
        } else {
            let (span, anchor) = self.flow_anchor();
            return Err(error(
                "CNL_MISSING",
                span,
                Some(&paths.node(node, "/undo")),
                "Missing undo declaration.",
                "Declare the exact undo or use no undo.",
                anchor,
                SyntaxRule::Flow(FlowIssue::MissingUndoDeclaration),
            ));
        }
        Ok(())
    }

    fn flow_exports(
        &mut self,
        state: &mut FlowState,
        node: NodePath,
    ) -> Result<(), Box<SyntaxIssue>> {
        if self.normal() == "Export no values." {
            let line = self.take()?.line;
            self.clause(&line, Clause::NoExports);
            return Ok(());
        }
        let mut found = false;
        while self.normal().starts_with("Export ") {
            let mut cursor = self.take()?;
            cursor.word("Export")?;
            let name = cursor.ident()?;
            state.declare(&name, FlowNameKind::Export, |paths| {
                paths.node(node, &format!("/exports/{}", name.text))
            })?;
            cursor.words(&["from", "then"])?;
            let yes = flow_value(&mut cursor)?;
            cursor.words(&["and", "from", "else"])?;
            let no = flow_value(&mut cursor)?;
            cursor.end()?;
            self.clause(&cursor.line, Clause::Export { name, yes, no });
            found = true;
        }
        if !found {
            let (span, anchor) = self.flow_anchor();
            return Err(error(
                "CNL_MISSING",
                span,
                Some(&state.paths.node(node, "/exports")),
                "Branch export declaration is missing.",
                "Export no values. or Export IDENT from then VALUE and from else VALUE.",
                anchor,
                SyntaxRule::Flow(FlowIssue::MissingBranchExport),
            ));
        }
        Ok(())
    }

    fn flow_results(&mut self) -> Result<(), Box<SyntaxIssue>> {
        if self.normal() == "Return no values." {
            let line = self.take()?.line;
            self.clause(&line, Clause::NoResults);
            return Ok(());
        }
        let mut names = HashSet::new();
        while self.normal().starts_with("Return ") {
            let mut cursor = self.take()?;
            cursor.word("Return")?;
            let name = cursor.ident()?;
            if !names.insert(name.text.clone()) {
                return Err(duplicate(
                    &format!("/flow/result/{}", name.text),
                    &cursor.line,
                    "result name",
                    FlowIssue::DuplicateResultName { name: name.text },
                ));
            }
            cursor.word("from")?;
            let value = flow_value(&mut cursor)?;
            cursor.end()?;
            self.clause(&cursor.line, Clause::Result { name, value });
        }
        if names.is_empty() {
            let (span, anchor) = self.flow_anchor();
            return Err(error(
                "CNL_MISSING",
                span,
                Some("/flow/result"),
                "Result declaration is missing.",
                "Return no values. or Return IDENT from VALUE.",
                anchor,
                SyntaxRule::Flow(FlowIssue::MissingResultDeclaration),
            ));
        }
        Ok(())
    }
}

fn duplicate(pointer: &str, line: &Line, what: &str, rule: FlowIssue) -> Box<SyntaxIssue> {
    error(
        "CNL_DUPLICATE",
        line.span.clone(),
        Some(pointer),
        format!("Duplicate {what}."),
        format!("Declare {what} exactly once."),
        AnchorKind::Statement,
        SyntaxRule::Flow(rule),
    )
}

fn flow_value(cursor: &mut Cursor) -> Result<ValueSyntax, Box<SyntaxIssue>> {
    let first = cursor.token()?;
    let kind = match first.text.as_str() {
        "input" => ValueKind::Input(cursor.ident()?),
        "step" => {
            let name = cursor.ident()?;
            cursor.word("field")?;
            let (value, field) = cursor.json()?;
            if !value.is_string() {
                return Err(error(
                    "CNL_SYNTAX",
                    field.span,
                    None,
                    "Result field must be a JSON string.",
                    "step IDENT field JSON_STRING",
                    AnchorKind::Token,
                    SyntaxRule::Flow(FlowIssue::ResultFieldNotString { actual: field.text }),
                ));
            }
            ValueKind::Step { name, field }
        }
        "literal" => {
            let ty = cursor.token()?;
            if !["string", "integer", "float", "boolean", "date", "bytes"]
                .contains(&ty.text.as_str())
            {
                return Err(error(
                    "CNL_SYNTAX",
                    ty.span,
                    None,
                    "Unknown scalar literal type.",
                    "Use string, integer, float, boolean, date, or bytes.",
                    AnchorKind::Token,
                    SyntaxRule::Flow(FlowIssue::UnknownScalarLiteralType { actual: ty.text }),
                ));
            }
            let (_, value) = cursor.json()?;
            ValueKind::Literal { ty, value }
        }
        _ => {
            return Err(error(
                "CNL_SYNTAX",
                first.span,
                None,
                "Unknown runtime reference form.",
                "Use input NAME, step NAME field JSON_STRING, or literal TYPE JSON_SCALAR.",
                AnchorKind::Token,
                SyntaxRule::Flow(FlowIssue::UnknownRuntimeReference { actual: first.text }),
            ));
        }
    };
    let last = match &kind {
        ValueKind::Input(name) => &name.span,
        ValueKind::Step { field, .. } => &field.span,
        ValueKind::Literal { value, .. } => &value.span,
    };
    let span = joined(&first.span, last);
    Ok(ValueSyntax { span, kind })
}

fn predicate(cursor: &mut Cursor) -> Result<Predicate, Box<SyntaxIssue>> {
    let left = flow_value(cursor)?;
    cursor.word("is")?;
    let token = cursor.token()?;
    let operator = match token.text.as_str() {
        "null" => Comparison::IsNull,
        "equal" => {
            cursor.word("to")?;
            Comparison::Eq
        }
        "not" => {
            cursor.words(&["equal", "to"])?;
            Comparison::Ne
        }
        "less" => {
            cursor.word("than")?;
            Comparison::Lt
        }
        "greater" => {
            cursor.word("than")?;
            Comparison::Gt
        }
        "at" => {
            let bound = cursor.token()?;
            match bound.text.as_str() {
                "most" => Comparison::Le,
                "least" => Comparison::Ge,
                _ => {
                    return Err(error(
                        "CNL_SYNTAX",
                        bound.span,
                        None,
                        "Unknown comparison.",
                        "Use at most or at least.",
                        AnchorKind::Token,
                        SyntaxRule::Flow(FlowIssue::UnknownAtComparison { actual: bound.text }),
                    ));
                }
            }
        }
        _ => {
            return Err(error(
                "CNL_SYNTAX",
                token.span,
                None,
                "Unknown comparison.",
                "Use a fixed comparison or is null.",
                AnchorKind::Token,
                SyntaxRule::Flow(FlowIssue::UnknownComparison { actual: token.text }),
            ));
        }
    };
    let right = (operator != Comparison::IsNull)
        .then(|| flow_value(cursor))
        .transpose()?;
    Ok(Predicate {
        left,
        operator,
        operator_span: token.span,
        right,
    })
}
