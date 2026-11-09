use super::*;

impl Scanner {
    pub(super) fn document(&mut self) -> Result<(), Box<SyntaxIssue>> {
        let header = self.take()?.line;
        if header.normal != "Use SOM CNL version 1." {
            let span = header
                .tokens
                .last()
                .map(|token| token.span.clone())
                .unwrap_or(header.span);
            return Err(error(
                "CNL_SYNTAX",
                span,
                None,
                "Unsupported CNL language version in header.",
                "Use SOM CNL version 1.",
                if header.tokens.is_empty() {
                    AnchorKind::Statement
                } else {
                    AnchorKind::Token
                },
                SyntaxRule::Request(RequestIssue::InvalidHeader {
                    actual: header.normal,
                }),
            ));
        }
        self.clause(&header, Clause::Header);
        let mut version = self.take()?;
        version.words(&["Use", "prompt", "version"])?;
        let token = version.token()?;
        version.end()?;
        let number = token.text.parse::<usize>().unwrap_or(0);
        if !(1..=7).contains(&number) {
            let mut issue = error(
                "UNSUPPORTED_VERSION",
                token.span,
                Some("/version"),
                "Prompt version is outside the supported set 1 through 7.",
                "Use prompt version 1 through 7.",
                AnchorKind::Token,
                SyntaxRule::Request(RequestIssue::UnsupportedVersion { actual: token.text }),
            );
            issue.phase = Phase::Dispatch;
            return Err(issue);
        }
        if token.text != number.to_string() {
            let mut issue = error(
                "CNL_SYNTAX",
                token.span,
                Some("/version"),
                "Prompt version token is not a fixed grammar choice.",
                "Use prompt version 1 through 7.",
                AnchorKind::Token,
                SyntaxRule::Request(RequestIssue::NoncanonicalVersion {
                    actual: token.text,
                    number,
                }),
            );
            issue.phase = Phase::Dispatch;
            return Err(issue);
        }
        self.version = number;
        self.locate("/version", &token);
        self.clause(&version.line, Clause::Version { number, token });

        let (package, line) = self.scalar(&["Use", "package"], "/package", false)?;
        self.clause(&line, Clause::Package(package));
        if number >= 6 {
            let mut contract = self.take()?;
            contract.words(&["Bind", "contract"])?;
            let id = contract.token()?;
            contract.words(&["with", "SHA-256"])?;
            let (_, hash) = contract.json()?;
            contract.end()?;
            self.locate("/contract/id", &id);
            self.clause(&contract.line, Clause::Contract { id, hash });
        }
        if number <= 4 {
            self.resource(ResourceRole::Single, "/resource")?;
            if number >= 2 {
                self.query("/query")?;
            }
        } else {
            self.resource(ResourceRole::Parent, "/resources/parent")?;
            self.resource(ResourceRole::Child, "/resources/child")?;
            let mut relation = self.take()?;
            relation.words(&["Relate", "child", "to", "parent", "with", "foreign", "key"])?;
            let key = relation.ident()?;
            relation.words(&["and", "restrict", "deletion"])?;
            relation.end()?;
            self.clause(&relation.line, Clause::Relation(key));
        }
        self.migration()?;
        if number >= 5 {
            self.capabilities()?;
        }
        self.routes()?;
        if number <= 5 {
            let begin = self.expected("Begin checks.")?;
            self.begin(&begin, Block::Checks);
            let check = self.take()?.line;
            self.clause(&check, Clause::Check(check.normal.clone()));
            let end = self.expected("End checks.")?;
            self.close(&end, BlockKind::Checks);
        }
        if number == 7 {
            self.flow()?;
        }
        let end = self.expected("End request.")?;
        self.clause(&end, Clause::EndRequest);
        if let Some(line) = self.peek() {
            return Err(error(
                "CNL_SYNTAX",
                line.span.clone(),
                None,
                "Unexpected content after End request.",
                "Supply one request only.",
                AnchorKind::Statement,
                SyntaxRule::Request(RequestIssue::TrailingContent {
                    actual: line.normal.clone(),
                }),
            ));
        }
        Ok(())
    }

    fn migration(&mut self) -> Result<(), Box<SyntaxIssue>> {
        let (revision, line) = self.scalar(
            &["Use", "migration", "revision"],
            "/migration/revision",
            false,
        )?;
        self.clause(&line, Clause::MigrationRevision(revision));
        let (previous, line) = if self.normal() == "Use no previous migration revision." {
            let line = self.take()?.line;
            self.span("/migration/down_revision", &line);
            (None, line)
        } else {
            let (token, line) = self.scalar(
                &["Use", "previous", "migration", "revision"],
                "/migration/down_revision",
                false,
            )?;
            (Some(token), line)
        };
        self.clause(&line, Clause::PreviousRevision(previous));
        Ok(())
    }

    fn capabilities(&mut self) -> Result<(), Box<SyntaxIssue>> {
        if self.version == 5 && self.normal() == "Use no capabilities." {
            let line = self.take()?.line;
            self.clause(&line, Clause::NoCapabilities);
            return Ok(());
        }
        let begin = self.expected("Begin capabilities.")?;
        self.begin(&begin, Block::Capabilities);
        for (kind, key, words, allowed) in [
            (
                Capability::Database,
                "database",
                &["Use", "database"][..],
                &["postgresql", "mongodb"][..],
            ),
            (
                Capability::Authentication,
                "auth",
                &["Use", "authentication"][..],
                &["jwt", "firebase", "google_oauth"][..],
            ),
            (
                Capability::Storage,
                "storage",
                &["Use", "object", "storage"][..],
                &["none", "s3", "gcs"][..],
            ),
            (
                Capability::Workflow,
                "workflow",
                &["Use", "workflow"][..],
                &[
                    "json_export",
                    "gzip_export",
                    "csv",
                    "decimal_summary",
                    "statistics",
                ][..],
            ),
        ] {
            let phrase = words.join(" ");
            let pointer = format!("/capabilities/{key}");
            if !self.normal().starts_with(&format!("{phrase} ")) {
                if let Some(issue) = self.repeated_scalar("/capabilities") {
                    return Err(issue);
                }
                if self.normal().starts_with("Use ")
                    || self.normal() == "End capabilities."
                    || self.peek().is_none()
                {
                    return Err(self.missing_scalar(&pointer, &format!("{phrase} ENUM.")));
                }
            }
            let mut clause = self.take()?;
            clause.words(words)?;
            let token = clause.token()?;
            clause.end()?;
            if !allowed.contains(&token.text.as_str()) {
                return Err(error(
                    "CNL_SYNTAX",
                    token.span,
                    None,
                    format!("Unknown {key} capability token."),
                    allowed.join(", "),
                    AnchorKind::Token,
                    SyntaxRule::Request(RequestIssue::UnknownCapability {
                        kind,
                        actual: token.text,
                        expected: allowed.iter().map(|word| (*word).into()).collect(),
                    }),
                ));
            }
            self.locate(&pointer, &token);
            self.clause(
                &clause.line,
                Clause::Capability {
                    kind,
                    value: Some(token),
                },
            );
        }
        for (kind, key) in [
            (Capability::WorkflowValue, "value"),
            (Capability::WorkflowGroup, "group"),
        ] {
            let pointer = format!("/capabilities/{key}_field");
            let (value, line) = if self.normal() == format!("Use no workflow {key} field.") {
                let line = self.take()?.line;
                self.span(&pointer, &line);
                (None, line)
            } else {
                let (token, line) =
                    self.scalar(&["Use", "workflow", key, "field"], &pointer, false)?;
                (Some(token), line)
            };
            self.clause(&line, Clause::Capability { kind, value });
        }
        let end = self.expected("End capabilities.")?;
        self.close(&end, BlockKind::Capabilities);
        Ok(())
    }

    fn routes(&mut self) -> Result<(), Box<SyntaxIssue>> {
        let begin = self.expected("Begin routes.")?;
        self.begin(&begin, Block::Routes);
        let pointer = if self.version >= 6 {
            "/operations"
        } else {
            "/logic"
        };
        let mut seen = BTreeSet::new();
        while self.normal() != "End routes." {
            if let Some(issue) = self.repeated_scalar("") {
                return Err(issue);
            }
            let mut clause = self.take()?;
            let route = if self.version <= 4 {
                RouteSyntax::Legacy(clause.line.normal.clone())
            } else if self.version == 5 {
                clause.word("For")?;
                let scope = clause.token()?;
                let role = scope.text.strip_suffix(':').unwrap_or("");
                if !["parent", "child", "batch"].contains(&role) {
                    return Err(error(
                        "CNL_SYNTAX",
                        scope.span,
                        None,
                        "Unknown related route scope.",
                        "For parent, child, or batch: the fixed sentence.",
                        AnchorKind::Token,
                        SyntaxRule::Request(RequestIssue::UnknownRouteScope { actual: scope.text }),
                    ));
                }
                let sentence = clause.line.tokens[clause.at..]
                    .iter()
                    .map(|token| token.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
                    + ".";
                RouteSyntax::Scoped {
                    scope: role.into(),
                    sentence,
                }
            } else if clause.line.normal == "Allow batch creation." {
                RouteSyntax::Selected {
                    scope: "batch".into(),
                    action: "batch_create".into(),
                }
            } else {
                clause.word("Allow")?;
                let action = clause.token()?;
                clause.word("on")?;
                let scope = clause.token()?;
                clause.end()?;
                if !["create", "get", "list", "patch", "delete"].contains(&action.text.as_str())
                    || !["parent", "child"].contains(&scope.text.as_str())
                {
                    return Err(error(
                        "CNL_SYNTAX",
                        clause.line.span,
                        None,
                        "Unknown selected operation.",
                        "Allow ACTION on parent|child. or Allow batch creation.",
                        AnchorKind::Statement,
                        SyntaxRule::Request(RequestIssue::UnknownSelectedOperation {
                            action: action.text,
                            scope: scope.text,
                        }),
                    ));
                }
                RouteSyntax::Selected {
                    scope: scope.text,
                    action: action.text,
                }
            };
            let index = seen.len();
            if !seen.insert(route.clone()) {
                return Err(self.duplicate(
                    &format!("{pointer}/{index}"),
                    &clause.line,
                    "route operation",
                ));
            }
            self.clause(&clause.line, Clause::Route(route));
        }
        let end = self.expected("End routes.")?;
        if seen.is_empty() {
            return Err(self.missing(
                pointer,
                &end,
                "The routes block has no operation.",
                "Declare the required fixed route sentences.",
                AnchorKind::Closing,
                SyntaxRule::Request(RequestIssue::MissingRoute),
            ));
        }
        self.close(&end, BlockKind::Routes);
        Ok(())
    }
}
