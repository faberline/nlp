use serde_json::{Value, json};
use som::{
    cnl, flow, load_blocks_from_bytes,
    planner::{PlannerAssets, plan},
    quality,
};

const DEFER_CNL: &str = include_str!("cnl-dsl/defer-queue-task.cnl");

#[test]
fn defer_task_cnl_and_typed_prompt_produce_the_same_dsl() {
    let typed: Value =
        serde_json::from_str(include_str!("cnl-dsl/defer-queue-task.prompt.json")).unwrap();
    faberline_nlp_cnl::check_syntax(DEFER_CNL).unwrap();
    let from_cnl = cnl::parse(DEFER_CNL).unwrap().prompt;
    assert_eq!(from_cnl, typed);

    let quality_catalog = serde_json::from_str(quality::CATALOG_TEXT).unwrap();
    let assets = PlannerAssets {
        legacy: &load_blocks_from_bytes(include_bytes!(
            "../../som-code-python/data/blocks/v5/blocks.json"
        ))
        .unwrap(),
        quality_blocks: &load_blocks_from_bytes(include_bytes!(
            "../../som-code-python/data/blocks/composed-quality/blocks.json"
        ))
        .unwrap(),
        current: &load_blocks_from_bytes(include_bytes!(
            "../../som-code-python/data/blocks/flow-v7-compact-self/blocks.json"
        ))
        .unwrap(),
        quality_catalog: &quality_catalog,
        compact_catalog: &flow::COMPACT_CATALOG,
    };
    let dsl = plan(&from_cnl, &assets).unwrap();
    assert_eq!(
        flow::canonical(&dsl),
        flow::canonical(&plan(&typed, &assets).unwrap())
    );
    assert_eq!(dsl["version"], "som.dsl.v3");

    let transaction = &dsl["flow"]["steps"][1];
    assert_eq!(transaction["id"], "flow.postgresql.transaction");
    let task = &transaction["steps"][0];
    assert_eq!(task["id"], "flow.postgresql.child.create");
    assert_eq!(task["args"]["parent_id"], json!({"input": "queue_id"}));
    assert_eq!(task["undo"], "flow.postgresql.undo.delete_created");
}

#[test]
fn som_rejects_a_string_queue_id_after_nlp_accepts_its_syntax() {
    let source = DEFER_CNL.replacen(
        "Input queue_id has type integer.",
        "Input queue_id has type string.",
        1,
    );
    faberline_nlp_cnl::check_syntax(&source).unwrap();
    let error = cnl::parse(&source).expect_err("child.create requires an integer parent ID");
    assert_eq!(error.code, "INVALID_REF");
    let diagnostic = &error.diagnostics[0];
    assert_eq!(
        diagnostic.prompt_pointer.as_deref(),
        Some("/flow/steps/1/steps/0/args/parent_id")
    );
    assert_eq!(
        &source[diagnostic.span.byte_start..diagnostic.span.byte_end],
        "input queue_id"
    );
}
