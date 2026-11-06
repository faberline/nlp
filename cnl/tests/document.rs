use std::{fs, path::Path};

use faberline_nlp_cnl::{
    AnchorKind, BaseIssue, Block, BlockKind, Clause, EventKind, IssueKind, Phase, SOURCE_FILES,
    SYNTAX_API_VERSION, ScopePart, SyntaxRule, SyntaxSlot, ValueKind, parse_document,
};

const WORKFLOW: &str = r#"Use SOM CNL version 1.
Use prompt version 7.
Use package entries.
Bind contract related-flow.v1 with SHA-256 "example".
Begin parent resource.
Use resource name Parent.
Use table parents.
Use path "/parents".
Field name is required str and is unique.
Begin query.
Use no exact filter.
Use no date filter.
Limit query results to 100.
End query.
End parent resource.
Begin child resource.
Use resource name Entry.
Use table entries.
Use path "/entries".
Field name is required str and is unique.
Begin query.
Use no exact filter.
Use no date filter.
Limit query results to 100.
End query.
End child resource.
Relate child to parent with foreign key parent_id and restrict deletion.
Use migration revision initial.
Use no previous migration revision.
Begin capabilities.
Use database postgresql.
Use authentication jwt.
Use object storage none.
Use workflow json_export.
Use no workflow value field.
Use no workflow group field.
End capabilities.
Begin routes.
Allow get on child.
End routes.
Begin flow read_entry.
Use no inputs.
Begin steps.
Begin step read with operation child.get.
Set argument entry_id to literal integer 1.
On failure, use no undo.
End step read.
End steps.
Return item from step read field "資料".
End flow read_entry.
End request.
"#;

#[test]
fn emits_flat_events_with_original_spans() {
    let source = format!("{} \t", WORKFLOW.replace('\n', "\r\n"));
    let report = parse_document(&source);
    assert_eq!(report.api_version, SYNTAX_API_VERSION);
    assert_eq!(report.issue, None);
    assert_eq!(report.eof.byte_end, source.len());
    let mut opened = Vec::new();
    for (index, event) in report.events.iter().enumerate() {
        assert_eq!(event.parent, opened.last().copied());
        match &event.kind {
            EventKind::Begin(_) => opened.push(index),
            EventKind::End(_) => {
                opened.pop().unwrap();
            }
            EventKind::Clause(_) => {}
        }
    }
    assert!(opened.is_empty());
    let result = report
        .events
        .iter()
        .find_map(|event| match &event.kind {
            EventKind::Clause(Clause::Result { value, .. }) => Some(value),
            _ => None,
        })
        .unwrap();
    let ValueKind::Step { field, .. } = &result.kind else {
        panic!("expected a step reference");
    };
    assert_eq!(
        &source[field.span.byte_start..field.span.byte_end],
        "\"資料\""
    );
    assert_eq!(field.span.end_column - field.span.column, 4);
    assert_eq!(
        &source[result.span.byte_start..result.span.byte_end],
        "step read field \"資料\""
    );
}

#[test]
fn reports_later_structure_errors_with_an_unknown_contract() {
    let unknown = WORKFLOW.replace("related-flow.v1", "not-a-contract.v1");
    assert_eq!(parse_document(&unknown).issue, None);
    for (before, after, kind, expected) in [
        (
            "entry_id to literal",
            "entry_id from literal",
            IssueKind::Syntax,
            "from",
        ),
        (
            "End flow read_entry.",
            "End flow bad_name.",
            IssueKind::BlockMismatch,
            "End flow bad_name.",
        ),
    ] {
        let source = unknown.replace(before, after);
        let report = parse_document(&source);
        let issue = report.issue.unwrap();
        assert_eq!(issue.phase, Phase::Structural);
        assert_eq!(issue.kind, kind);
        assert_eq!(
            &source[issue.span.byte_start..issue.span.byte_end],
            expected
        );
        assert!(
            report
                .events
                .iter()
                .any(|event| matches!(event.kind, EventKind::Clause(Clause::Contract { .. })))
        );
    }
}

#[test]
fn checks_all_lines_before_dispatching_the_version() {
    let unknown = WORKFLOW.replace("prompt version 7", "prompt version 8");
    let dispatch = parse_document(&unknown).issue.unwrap();
    assert_eq!(dispatch.phase, Phase::Dispatch);
    assert_eq!(dispatch.kind, IssueKind::UnsupportedVersion);
    assert_eq!(dispatch.fact.anchor, AnchorKind::Token);
    assert_eq!(
        dispatch.fact.scope,
        Some(vec![ScopePart::Slot(SyntaxSlot::Version)])
    );
    let source = unknown.replace("End request.", "End request");
    let report = parse_document(&source);
    let issue = report.issue.unwrap();
    assert_eq!(issue.phase, Phase::Lexical);
    assert_eq!(
        &source[issue.span.byte_start..issue.span.byte_end],
        "End request"
    );
    assert!(report.events.is_empty());
    assert_eq!(report.eof.byte_start, source.len());
}

#[test]
fn reports_rule_arguments_and_block_context_for_a_bad_clause() {
    let source = WORKFLOW.replace("entry_id to literal", "entry_id from literal");
    let report = parse_document(&source);
    let issue = report.issue.unwrap();
    assert_eq!(issue.fact.anchor, AnchorKind::Token);
    assert_eq!(
        issue.fact.rule,
        SyntaxRule::Base(BaseIssue::ExpectedKeyword {
            expected: "to".into(),
            actual: "from".into(),
        })
    );
    let event = &report.events[issue.fact.block.unwrap()];
    assert!(matches!(event.kind, EventKind::Begin(Block::Step { .. })));
}

#[test]
fn scans_deep_nesting_before_any_business_limit() {
    let depth = 2048;
    let mut nested = String::new();
    use std::fmt::Write;
    for i in 0..depth {
        writeln!(nested, "Begin transaction tx{i}.").unwrap();
    }
    for i in (0..depth).rev() {
        writeln!(nested, "End transaction tx{i}.").unwrap();
    }
    let source = WORKFLOW.replace("Begin steps.\n", &format!("Begin steps.\n{nested}"));
    let report = parse_document(&source);
    assert_eq!(report.issue, None);
    assert_eq!(
        report
            .events
            .iter()
            .filter(|event| matches!(event.kind, EventKind::Begin(Block::Transaction { .. })))
            .count(),
        depth
    );
    assert_eq!(
        report
            .events
            .iter()
            .filter(|event| matches!(event.kind, EventKind::End(BlockKind::Transaction)))
            .count(),
        depth
    );

    let source = source.replace(
        &format!("End transaction tx{}.", depth - 1),
        "End transaction wrong.",
    );
    let issue = parse_document(&source).issue.unwrap();
    assert_eq!(issue.kind, IssueKind::BlockMismatch);
    assert_eq!(
        &source[issue.span.byte_start..issue.span.byte_end],
        "End transaction wrong."
    );
    assert_eq!(
        issue.pointer.unwrap(),
        format!("/flow/steps{}", "/0/steps".repeat(depth))
    );
}

#[test]
fn source_manifest_covers_the_crate_sources_and_matches_their_bytes() {
    fn visit(root: &Path, directory: &Path, paths: &mut Vec<String>) {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(root, &path, paths);
            } else {
                paths.push(
                    path.strip_prefix(root)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .replace('\\', "/"),
                );
            }
        }
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut expected = vec![String::from("Cargo.toml")];
    visit(root, &root.join("src"), &mut expected);
    expected.sort_unstable();
    assert_eq!(
        SOURCE_FILES
            .iter()
            .map(|(path, _)| *path)
            .collect::<Vec<_>>(),
        expected
    );
    for (path, bytes) in SOURCE_FILES {
        assert_eq!(&fs::read(root.join(path)).unwrap(), bytes, "{path}");
    }
}
