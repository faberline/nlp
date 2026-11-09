use faberline_nlp_cnl::{
    BaseIssue, IssueKind, Phase, SERVICE_SYNTAX_API_VERSION, SourceSpan, check_syntax,
    parse_document, parse_service_document,
    service::{ServiceEventKind, ServiceSlot, ServiceSyntaxRule},
};

const TASK_STATUS: &str = r#"Service prompt version 1.
Package defer_status_client.
Target rust-service-client.v1.
Bind service defer using contract "defer.http@5f7adac1fe1333b33e3ec69a92b5ca4a634a97d0".
Bind operation read_task_status to task_status.
Accept queue as string for queue.
Accept task_id as string for task_id.
"#;

const CONTRACT: &str = r#""defer.http@5f7adac1fe1333b33e3ec69a92b5ca4a634a97d0""#;
const SECOND_INPUT: &str = "Accept task_id as string for task_id.\n";

fn source_text<'a>(source: &'a str, span: &SourceSpan) -> &'a str {
    &source[span.byte_start..span.byte_end]
}

#[test]
fn emits_the_seven_statements_with_original_token_spans() {
    let report = parse_service_document(TASK_STATUS);
    assert_eq!(report.api_version, SERVICE_SYNTAX_API_VERSION);
    assert_eq!(report.api_version, 1);
    assert_eq!(report.issue, None);
    assert_eq!(report.events.len(), 7);

    let mut line_start = 0;
    for (index, (event, line)) in report.events.iter().zip(TASK_STATUS.lines()).enumerate() {
        let tokens = match (index, &event.kind) {
            (0, ServiceEventKind::Header { version }) => vec![(version, "1")],
            (1, ServiceEventKind::Package { name }) => vec![(name, "defer_status_client")],
            (2, ServiceEventKind::Target { profile }) => vec![(profile, "rust-service-client.v1")],
            (3, ServiceEventKind::ServiceBinding { service, contract }) => {
                vec![(service, "defer"), (contract, CONTRACT)]
            }
            (4, ServiceEventKind::OperationBinding { alias, operation }) => {
                vec![(alias, "read_task_status"), (operation, "task_status")]
            }
            (
                5,
                ServiceEventKind::InputBinding {
                    name,
                    ty,
                    parameter,
                },
            ) => {
                vec![(name, "queue"), (ty, "string"), (parameter, "queue")]
            }
            (
                6,
                ServiceEventKind::InputBinding {
                    name,
                    ty,
                    parameter,
                },
            ) => {
                vec![(name, "task_id"), (ty, "string"), (parameter, "task_id")]
            }
            _ => panic!("unexpected event at index {index}: {event:?}"),
        };

        assert_eq!(
            event.span,
            SourceSpan {
                line: index + 1,
                column: 1,
                end_line: index + 1,
                end_column: line.len() + 1,
                byte_start: line_start,
                byte_end: line_start + line.len(),
            }
        );

        let mut search_start = 0;
        for (token, expected) in tokens {
            let start = search_start + line[search_start..].find(expected).unwrap();
            let end = start + expected.len();
            assert_eq!(token.text, expected);
            assert_eq!(
                token.span,
                SourceSpan {
                    line: index + 1,
                    column: start + 1,
                    end_line: index + 1,
                    end_column: end + 1,
                    byte_start: line_start + start,
                    byte_end: line_start + end,
                }
            );
            search_start = end;
        }
        line_start += line.len() + 1;
    }
}

#[test]
fn keeps_crlf_utf8_and_eof_positions_in_the_original_source() {
    let raw_contract = r#""資料\u0020catalog""#;
    let source = TASK_STATUS
        .replace(CONTRACT, raw_contract)
        .replace('\n', "\r\n");
    let source = format!("{source} \t");
    let report = parse_service_document(&source);
    assert_eq!(report.issue, None);
    let ServiceEventKind::ServiceBinding { contract, .. } = &report.events[3].kind else {
        panic!("expected a service binding");
    };
    assert_eq!(contract.text, raw_contract);
    assert_eq!(source_text(&source, &contract.span), raw_contract);
    assert_eq!(contract.span.byte_start, source.find(raw_contract).unwrap());
    assert_eq!(contract.span.line, 4);
    assert_eq!(
        contract.span.end_column - contract.span.column,
        raw_contract.chars().count()
    );
    assert_eq!(
        report.eof,
        SourceSpan {
            line: 8,
            column: 3,
            end_line: 8,
            end_column: 3,
            byte_start: source.len(),
            byte_end: source.len(),
        }
    );
}

#[test]
fn checks_lexical_errors_before_version_dispatch_and_structure() {
    let bad_version = TASK_STATUS.replace("version 1", "version 2");
    let bad_structure = "Bind operation read_task_status from task_status.";
    for source in [
        bad_version.replace(CONTRACT, r#""\q""#),
        TASK_STATUS
            .replace(
                "Bind operation read_task_status to task_status.",
                bad_structure,
            )
            .replace("for task_id.", "for task_id"),
    ] {
        let report = parse_service_document(&source);
        let issue = report.issue.unwrap();
        assert_eq!(issue.phase, Phase::Lexical);
        assert!(report.events.is_empty());
        match issue.rule {
            ServiceSyntaxRule::Base(BaseIssue::InvalidJsonString { .. }) => {
                assert_eq!(source_text(&source, &issue.span), r#""\q""#);
            }
            ServiceSyntaxRule::Base(BaseIssue::MissingPeriod { .. }) => {
                assert_eq!(
                    source_text(&source, &issue.span),
                    "Accept task_id as string for task_id"
                );
            }
            rule => panic!("unexpected lexical rule: {rule:?}"),
        }
    }

    let issue = parse_service_document(&bad_version).issue.unwrap();
    assert_eq!(issue.phase, Phase::Dispatch);
    assert_eq!(issue.kind, IssueKind::UnsupportedVersion);
    assert_eq!(issue.slot, Some(ServiceSlot::Header));
    assert_eq!(issue.rule, ServiceSyntaxRule::UnsupportedVersion);
    assert_eq!(source_text(&bad_version, &issue.span), "2");

    let source = TASK_STATUS.replace(
        "Bind operation read_task_status to task_status.",
        bad_structure,
    );
    let report = parse_service_document(&source);
    let issue = report.issue.unwrap();
    assert_eq!(issue.phase, Phase::Structural);
    assert_eq!(issue.slot, Some(ServiceSlot::OperationBinding));
    assert_eq!(
        issue.rule,
        ServiceSyntaxRule::Base(BaseIssue::ExpectedKeyword {
            expected: "to".into(),
            actual: "from".into(),
        })
    );
    assert_eq!(source_text(&source, &issue.span), "from");
    assert_eq!(report.events.len(), 4);
}

#[test]
fn reports_missing_and_duplicate_statements() {
    let source = TASK_STATUS.strip_suffix(SECOND_INPUT).unwrap().trim_end();
    let report = parse_service_document(source);
    let issue = report.issue.unwrap();
    assert_eq!(issue.phase, Phase::Structural);
    assert_eq!(issue.kind, IssueKind::Missing);
    assert_eq!(issue.slot, Some(ServiceSlot::Input(1)));
    assert_eq!(issue.span, report.eof);
    assert_eq!(issue.rule, ServiceSyntaxRule::MissingStatement);
    assert_eq!(report.events.len(), 6);

    let source = TASK_STATUS.replace(
        "Package defer_status_client.",
        "Package defer_status_client.\nPackage repeated.",
    );
    let issue = parse_service_document(&source).issue.unwrap();
    assert_eq!(issue.phase, Phase::Structural);
    assert_eq!(issue.kind, IssueKind::Duplicate);
    assert_eq!(issue.rule, ServiceSyntaxRule::DuplicateStatement);

    let source = TASK_STATUS.replace("Accept task_id as", "Accept queue as");
    let issue = parse_service_document(&source).issue.unwrap();
    assert_eq!(issue.phase, Phase::Structural);
    assert_eq!(issue.kind, IssueKind::Duplicate);
    assert_eq!(issue.slot, Some(ServiceSlot::Input(1)));
    assert_eq!(issue.rule, ServiceSyntaxRule::DuplicateInput);
}

#[test]
fn rejects_extra_input_trailing_content_and_wrong_order() {
    let source = format!("{TASK_STATUS}Accept extra as string for extra.\n");
    let issue = parse_service_document(&source).issue.unwrap();
    assert_eq!(issue.phase, Phase::Structural);
    assert_eq!(issue.slot, Some(ServiceSlot::Input(2)));
    assert_eq!(issue.rule, ServiceSyntaxRule::ExtraInput);

    let source = format!("{TASK_STATUS}End service.\n");
    let issue = parse_service_document(&source).issue.unwrap();
    assert_eq!(issue.phase, Phase::Structural);
    assert_eq!(issue.slot, Some(ServiceSlot::End));
    assert_eq!(issue.rule, ServiceSyntaxRule::UnexpectedStatement);

    let source = TASK_STATUS.replace(
        "Package defer_status_client.\nTarget rust-service-client.v1.",
        "Target rust-service-client.v1.\nPackage defer_status_client.",
    );
    let issue = parse_service_document(&source).issue.unwrap();
    assert_eq!(issue.phase, Phase::Structural);
    assert_eq!(issue.slot, Some(ServiceSlot::Package));
    assert_eq!(issue.rule, ServiceSyntaxRule::UnexpectedStatement);
}

#[test]
fn keeps_catalog_names_and_parameter_bindings_for_the_consumer() {
    let source = TASK_STATUS
        .replace("rust-service-client.v1", "another-target.v9")
        .replace("Bind service defer", "Bind service unknown")
        .replace("to task_status.", "to task_cancel.")
        .replace(
            "task_id as string for task_id.",
            "task_id as SomeType for queue.",
        );
    let report = parse_service_document(&source);
    assert_eq!(report.issue, None);
    let ServiceEventKind::Target { profile } = &report.events[2].kind else {
        panic!("expected a target");
    };
    let ServiceEventKind::ServiceBinding { service, .. } = &report.events[3].kind else {
        panic!("expected a service binding");
    };
    let ServiceEventKind::OperationBinding { operation, .. } = &report.events[4].kind else {
        panic!("expected an operation binding");
    };
    let ServiceEventKind::InputBinding {
        name,
        ty,
        parameter,
    } = &report.events[6].kind
    else {
        panic!("expected an input binding");
    };
    assert_eq!(profile.text, "another-target.v9");
    assert_eq!(service.text, "unknown");
    assert_eq!(operation.text, "task_cancel");
    assert_eq!(name.text, "task_id");
    assert_eq!(ty.text, "SomeType");
    assert_eq!(parameter.text, "queue");

    let source = TASK_STATUS.replace("rust-service-client.v1", r#""rust-service-client.v1""#);
    let issue = parse_service_document(&source).issue.unwrap();
    assert_eq!(issue.slot, Some(ServiceSlot::Target));
    assert_eq!(issue.rule, ServiceSyntaxRule::ExpectedWord);
    assert_eq!(
        source_text(&source, &issue.span),
        r#""rust-service-client.v1""#
    );

    let source = TASK_STATUS.replace(CONTRACT, "unquoted-contract");
    let issue = parse_service_document(&source).issue.unwrap();
    assert_eq!(issue.slot, Some(ServiceSlot::ServiceBinding));
    assert_eq!(
        issue.rule,
        ServiceSyntaxRule::Base(BaseIssue::ExpectedJsonString {
            actual: "unquoted-contract".into(),
        })
    );
    assert_eq!(source_text(&source, &issue.span), "unquoted-contract");
}

#[test]
fn old_parser_does_not_admit_service_documents() {
    let report = parse_document(TASK_STATUS);
    assert_eq!(report.api_version, 2);
    assert!(report.issue.is_some());
    assert!(check_syntax(TASK_STATUS).is_err());
}
