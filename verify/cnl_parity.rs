use serde_json::{Value, json};
use som::{
    compiler::{Compiler, package_assets},
    flow::canonical,
    strict_json,
};
use std::{collections::BTreeMap, env, fs, path::PathBuf};

#[test]
fn write_compiler_parity_report() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let examples = root.join("contracts/compiler-v1/examples");
    let fixture_dir = root.join("tests/compiler-v1");
    let fixtures = |name: &str| -> Value {
        serde_json::from_slice(&fs::read(fixture_dir.join(name)).unwrap()).unwrap()
    };

    let mut cases = Vec::new();
    for path in fs::read_dir(examples)
        .unwrap()
        .map(|entry| entry.unwrap().path())
    {
        if path.extension().is_some_and(|ext| ext == "cnl") {
            cases.push((
                format!("examples/{}", path.file_name().unwrap().to_str().unwrap()),
                fs::read_to_string(path).unwrap(),
                false,
            ));
        }
    }
    let repo_examples = PathBuf::from(env::var_os("SOM_PARITY_REPO_EXAMPLES").unwrap());
    for path in fs::read_dir(repo_examples)
        .unwrap()
        .map(|entry| entry.unwrap().path())
    {
        if path.extension().is_some_and(|ext| ext == "cnl") {
            let name = path.file_stem().unwrap().to_str().unwrap();
            let source = fs::read_to_string(&path).unwrap();
            if name == "defer-queue-task" {
                cases.push((
                    "repo/defer-trailing-padding".into(),
                    format!("{source} \t"),
                    false,
                ));
                cases.push((
                    "repo/defer-string-queue-id".into(),
                    source.replacen(
                        "Input queue_id has type integer.",
                        "Input queue_id has type string.",
                        1,
                    ),
                    true,
                ));
            } else if name == "flaskr-post" {
                cases.push((
                    "repo/flaskr-self-reference".into(),
                    source.replacen(
                        "Set argument child_id to input post_id.",
                        "Set argument child_id to step fetched field \"id\".",
                        1,
                    ),
                    true,
                ));
            }
            cases.push((format!("repo/{name}"), source, false));
        }
    }
    let positives = fixtures("cnl-positive-fixtures.json");
    for case in positives["cases"].as_array().unwrap() {
        cases.push((
            format!("frozen/{}", case["id"].as_str().unwrap()),
            case["cnl"].as_str().unwrap().to_owned(),
            false,
        ));
    }
    let names = fixtures("self-name-fixtures.json");
    for group in ["positive", "accepted", "negative"] {
        for case in names[group].as_array().unwrap() {
            cases.push((
                format!("names/{group}/{}", case["id"].as_str().unwrap()),
                case["cnl"].as_str().unwrap().to_owned(),
                group == "negative",
            ));
        }
    }
    let baseline = fs::read_to_string(root.join("contracts/compiler-v1/examples/v7.cnl")).unwrap();
    let negatives = fixtures("cnl-negative-cases.json");
    for case in negatives["cases"].as_array().unwrap() {
        let mut lines: Vec<String> = baseline.lines().map(str::to_owned).collect();
        if let Some(line) = case["replace_line"].as_u64() {
            lines[line as usize - 1] = case["with"].as_str().unwrap().into();
        }
        if let Some(edits) = case["replace_lines"].as_array() {
            for edit in edits {
                lines[edit["line"].as_u64().unwrap() as usize - 1] =
                    edit["with"].as_str().unwrap().into();
            }
        }
        if let Some(insert) = case.get("insert_before_line") {
            lines.insert(
                insert["line"].as_u64().unwrap() as usize - 1,
                insert["text"].as_str().unwrap().into(),
            );
        }
        cases.push((
            format!("negative/{}", case["id"].as_str().unwrap()),
            lines.join("\n") + "\n",
            true,
        ));
    }

    let assets = PathBuf::from(env::var_os("SOM_PARITY_ASSETS").unwrap());
    let ruff = PathBuf::from(env::var_os("SOM_PARITY_RUFF").unwrap());
    package_assets(&assets, &ruff).unwrap();
    let compiler = Compiler::load(&assets).unwrap();
    let mut reports = BTreeMap::new();

    for (case, source, negative) in cases {
        for target in ["python", "rust", "typescript", "react"] {
            let input = json!({"format": "som.cnl.v1", "source": source, "target": target});
            let (status, response) = som_targets::compile(&compiler, &input.to_string());
            let record = if status == 200 {
                assert!(!negative, "{case}/{target} unexpectedly compiled");
                let bundle = &response["bundle"];
                assert!(bundle["dsl"].is_object(), "{case}/{target} has no DSL");
                let assemble = json!({"bundle": bundle}).to_string();
                let document = strict_json::parse(&assemble).unwrap();
                let (status, response) = if target == "python" {
                    match compiler.assemble_bundle(bundle) {
                        Ok(files) => (200, json!({"files": files})),
                        Err(error) => (
                            som_targets::status(error.code),
                            json!({"code": error.code, "detail": error.detail}),
                        ),
                    }
                } else {
                    som_targets::assemble(&compiler, &document)
                };
                if status == 200 {
                    let files = response["files"].as_object().unwrap();
                    assert!(!files.is_empty(), "{case}/{target} produced no files");
                    json!({
                        "prompt": canonical(&bundle["prompt"]),
                        "dsl": canonical(&bundle["dsl"]),
                        "files": files,
                    })
                } else {
                    json!({
                        "prompt": canonical(&bundle["prompt"]),
                        "dsl": canonical(&bundle["dsl"]),
                        "assemble_error": [status, response],
                    })
                }
            } else {
                json!({"compile_error": [status, response]})
            };
            assert!(reports.insert(format!("{target}/{case}"), record).is_none());
        }
    }
    let out = env::var_os("SOM_PARITY_REPORT").unwrap();
    fs::write(out, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
}
