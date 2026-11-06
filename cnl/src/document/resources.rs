use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuerySelector {
    ExactFilter,
    DateFilter,
    Limit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResourceIssue {
    MissingQueryEnd,
    MissingQuerySelector {
        selector: QuerySelector,
    },
    InvalidQueryLimit {
        actual: String,
    },
    UnknownQuerySentence {
        actual: String,
    },
    DateSelectorUnavailable {
        version: usize,
    },
    QuerySelectorsOutOfOrder {
        previous: QuerySelector,
        actual: QuerySelector,
    },
    DuplicateFieldName {
        name: String,
    },
    InvalidFieldRequirement {
        actual: String,
    },
    UnknownFieldType {
        actual: String,
    },
    InvalidFieldUniqueness {
        actual: String,
    },
    MissingResourceFields,
}

impl Scanner {
    pub(super) fn query(&mut self, p: &str) -> Result<(), Box<SyntaxIssue>> {
        let opener = self.expected("Begin query.")?;
        self.begin(&opener, Block::Query);
        let keys = ["filter_field", "date_field", "max_limit"];
        let selectors = [
            QuerySelector::ExactFilter,
            QuerySelector::DateFilter,
            QuerySelector::Limit,
        ];
        let mut seen = [false; 3];
        let mut ordinal = 0;
        loop {
            let line = self.peek().cloned().ok_or_else(|| {
                error(
                    "CNL_MISSING",
                    self.eof.clone(),
                    Some(p),
                    "Missing query end.",
                    "End query.",
                    AnchorKind::Eof,
                    SyntaxRule::Resource(ResourceIssue::MissingQueryEnd),
                )
            })?;
            if line.normal == "End query." {
                for (index, cause, form) in [
                    (
                        0,
                        "The query block has no exact-filter selector clause.",
                        "Use Filter exactly on field IDENT. or Use no exact filter.",
                    ),
                    (
                        1,
                        "The query block has no date-filter selector clause.",
                        "Use Filter before date on field IDENT. or Use no date filter.",
                    ),
                    (
                        2,
                        "The query block has no result limit clause.",
                        "Limit query results to UINT.",
                    ),
                ] {
                    if (index != 1 || self.version >= 4) && !seen[index] {
                        return Err(self.missing(
                            &format!("{p}/{}", keys[index]),
                            &line,
                            cause,
                            form,
                            AnchorKind::Closing,
                            SyntaxRule::Resource(ResourceIssue::MissingQuerySelector {
                                selector: selectors[index],
                            }),
                        ));
                    }
                }
                self.span(p, &line);
                self.take()?;
                self.close(&line, BlockKind::Query);
                return Ok(());
            }

            let mut cursor = self.take()?;
            let (next, token, clause) = if line.normal == "Use no exact filter." {
                let token = cursor.line.tokens[2].clone();
                let value_span = token.span.clone();
                (
                    0,
                    token,
                    Clause::ExactFilter {
                        field: None,
                        value_span,
                    },
                )
            } else if line.normal == "Use no date filter." {
                let token = cursor.line.tokens[2].clone();
                let value_span = token.span.clone();
                (
                    1,
                    token,
                    Clause::DateFilter {
                        field: None,
                        value_span,
                    },
                )
            } else if line.normal.starts_with("Filter exactly on field ") {
                cursor.words(&["Filter", "exactly", "on", "field"])?;
                let token = cursor.ident()?;
                cursor.end()?;
                let clause = Clause::ExactFilter {
                    field: Some(token.clone()),
                    value_span: token.span.clone(),
                };
                (0, token, clause)
            } else if line.normal.starts_with("Filter before date on field ") {
                cursor.words(&["Filter", "before", "date", "on", "field"])?;
                let token = cursor.ident()?;
                cursor.end()?;
                let clause = Clause::DateFilter {
                    field: Some(token.clone()),
                    value_span: token.span.clone(),
                };
                (1, token, clause)
            } else if line.normal.starts_with("Limit query results to ") {
                cursor.words(&["Limit", "query", "results", "to"])?;
                let token = cursor.token()?;
                cursor.end()?;
                if !token.text.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err(error(
                        "CNL_SYNTAX",
                        token.span,
                        None,
                        "Query limit must be an unsigned integer.",
                        "Limit query results to UINT.",
                        AnchorKind::Token,
                        SyntaxRule::Resource(ResourceIssue::InvalidQueryLimit {
                            actual: token.text,
                        }),
                    ));
                }
                // The typed consumer checks the integer range after the complete syntax pass.
                (2, token.clone(), Clause::QueryLimit(token))
            } else {
                let anchor = if line.normal.starts_with("End ") {
                    AnchorKind::Closing
                } else {
                    AnchorKind::Statement
                };
                return Err(error(
                    "CNL_SYNTAX",
                    line.span,
                    None,
                    "Unknown query sentence.",
                    "Use the fixed query selectors and limit.",
                    anchor,
                    SyntaxRule::Resource(ResourceIssue::UnknownQuerySentence {
                        actual: line.normal,
                    }),
                ));
            };
            let key = keys[next];
            let pointer = format!("{p}/{key}");
            if seen[next] {
                return Err(self.duplicate(&pointer, &line, key));
            }
            if next == 1 && self.version < 4 {
                return Err(error(
                    "CNL_SYNTAX",
                    line.span,
                    None,
                    "Date selector is unavailable in this prompt version.",
                    "Use only the exact filter and limit.",
                    AnchorKind::Statement,
                    SyntaxRule::Resource(ResourceIssue::DateSelectorUnavailable {
                        version: self.version,
                    }),
                ));
            }
            if next < ordinal {
                return Err(error(
                    "CNL_SYNTAX",
                    line.span,
                    None,
                    "Query clauses are out of order.",
                    "Declare exact filter, date filter, then limit.",
                    AnchorKind::Statement,
                    SyntaxRule::Resource(ResourceIssue::QuerySelectorsOutOfOrder {
                        previous: selectors[ordinal],
                        actual: selectors[next],
                    }),
                ));
            }
            ordinal = next;
            self.locate(&pointer, &token);
            self.clause(&line, clause);
            seen[next] = true;
        }
    }

    pub(super) fn resource(&mut self, role: ResourceRole, p: &str) -> Result<(), Box<SyntaxIssue>> {
        let label = match role {
            ResourceRole::Single => "resource",
            ResourceRole::Parent => "parent resource",
            ResourceRole::Child => "child resource",
        };
        let opener = self.expected(&format!("Begin {label}."))?;
        self.begin(&opener, Block::Resource(role));
        let (name, line) =
            self.scalar(&["Use", "resource", "name"], &format!("{p}/name"), false)?;
        self.clause(&line, Clause::ResourceName(name));
        let (table, line) = self.scalar(&["Use", "table"], &format!("{p}/table"), false)?;
        self.clause(&line, Clause::Table(table));
        let (path, line) = self.scalar(&["Use", "path"], &format!("{p}/path"), true)?;
        self.clause(&line, Clause::Path(path));
        let mut seen = BTreeSet::new();
        while self.normal().starts_with("Field ") {
            let mut cursor = self.take()?;
            cursor.word("Field")?;
            let name = cursor.ident()?;
            let field = format!("{p}/fields/{}", seen.len());
            if !seen.insert(name.text.clone()) {
                return Err(error(
                    "CNL_DUPLICATE",
                    name.span,
                    Some(&format!("{field}/name")),
                    "Duplicate field name.",
                    "Use unique field names within the resource.",
                    AnchorKind::Token,
                    SyntaxRule::Resource(ResourceIssue::DuplicateFieldName { name: name.text }),
                ));
            }
            self.locate(&format!("{field}/name"), &name);
            cursor.word("is")?;
            let required_token = cursor.token()?;
            let required = match required_token.text.as_str() {
                "required" => true,
                "optional" => false,
                _ => {
                    return Err(error(
                        "CNL_SYNTAX",
                        required_token.span,
                        None,
                        "Expected required or optional.",
                        "Use required or optional.",
                        AnchorKind::Token,
                        SyntaxRule::Resource(ResourceIssue::InvalidFieldRequirement {
                            actual: required_token.text,
                        }),
                    ));
                }
            };
            self.locate(&format!("{field}/required"), &required_token);
            let ty = cursor.token()?;
            if !["str", "int", "float", "bool", "date"].contains(&ty.text.as_str()) {
                return Err(error(
                    "CNL_SYNTAX",
                    ty.span,
                    None,
                    format!("Field token {} is outside the grammar vocabulary.", ty.text),
                    "str, int, float, bool, or date when the prompt version allows date.",
                    AnchorKind::Token,
                    SyntaxRule::Resource(ResourceIssue::UnknownFieldType { actual: ty.text }),
                ));
            }
            self.locate(&format!("{field}/type"), &ty);
            cursor.words(&["and", "is"])?;
            let unique_token = cursor.token()?;
            let unique = match unique_token.text.as_str() {
                "unique" => true,
                "not" => {
                    cursor.word("unique")?;
                    false
                }
                _ => {
                    return Err(error(
                        "CNL_SYNTAX",
                        unique_token.span,
                        None,
                        "Expected unique or not unique.",
                        "Use unique or not unique.",
                        AnchorKind::Token,
                        SyntaxRule::Resource(ResourceIssue::InvalidFieldUniqueness {
                            actual: unique_token.text,
                        }),
                    ));
                }
            };
            self.locate(&format!("{field}/unique"), &unique_token);
            let default = if cursor.at < cursor.line.tokens.len() {
                cursor.words(&["with", "default"])?;
                let (_, token) = cursor.json()?;
                self.locate(&format!("{field}/default"), &token);
                Some(token)
            } else {
                if !required && self.version >= 3 {
                    self.span(&format!("{field}/default"), &cursor.line);
                }
                None
            };
            cursor.end()?;
            self.span(&field, &cursor.line);
            self.clause(
                &cursor.line,
                Clause::Field {
                    name,
                    required,
                    required_token,
                    ty,
                    unique,
                    unique_token,
                    default,
                },
            );
        }
        if role != ResourceRole::Single {
            self.query(&format!("{p}/query"))?;
        }
        if seen.is_empty() {
            let anchor = match self.peek() {
                None => AnchorKind::Eof,
                Some(line) if line.normal.starts_with("End ") => AnchorKind::Closing,
                Some(_) => AnchorKind::Statement,
            };
            let line = self.peek().cloned().unwrap_or_else(|| Line {
                tokens: vec![],
                span: self.eof.clone(),
                normal: String::new(),
            });
            return Err(self.missing(
                &format!("{p}/fields"),
                &line,
                "Resource needs at least one field.",
                "Field IDENT is required|optional TYPE and is unique|not unique.",
                anchor,
                SyntaxRule::Resource(ResourceIssue::MissingResourceFields),
            ));
        }
        let line = self.expected(&format!("End {label}."))?;
        self.span(p, &line);
        self.close(&line, BlockKind::Resource);
        Ok(())
    }
}
