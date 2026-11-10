//! Cross-backend check for the model runtime: ONNX (tract) and Core ML give the same labels and
//! logits within 1e-4 on the 12 fixtures of `models/test-tiny/expected.json`.
//!
//! `backends_agree` needs macOS and both features. With `NLP_WRITE_RECEIPT=1` it writes
//! `docs/receipts/model-runtime-<date>.json`. `receipt_is_consistent` runs everywhere (Linux
//! CI too): it reads the one receipt and checks it against the manifest and its own numbers.

use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

const MODEL: &str = "test-tiny";
const TOLERANCE: f64 = 1e-4;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn json(path: &std::path::Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

/// Every receipt of this layer. The repository keeps exactly one.
fn receipts() -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = fs::read_dir(root().join("docs/receipts"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("model-runtime-") && n.ends_with(".json"))
        })
        .collect();
    found.sort();
    found
}

#[derive(Deserialize)]
struct Receipt {
    schema: String,
    model: String,
    tolerance: f64,
    labels_equal: bool,
    max_abs_diff: f64,
    fixtures: Vec<ReceiptFixture>,
    files: Vec<ReceiptFile>,
    macos: String,
    rust: String,
}

#[derive(Deserialize)]
struct ReceiptFixture {
    id: String,
    tokens: usize,
    labels_equal: bool,
    max_abs_diff: f64,
}

#[derive(Deserialize)]
struct ReceiptFile {
    path: String,
    sha256: String,
}

#[test]
fn receipt_is_consistent() {
    let found = receipts();
    assert_eq!(found.len(), 1, "receipts found: {found:?}");
    let r: Receipt = serde_json::from_value(json(&found[0])).unwrap();
    assert_eq!(r.schema, "nlp.receipt.model-runtime.v1");
    assert_eq!(r.model, MODEL);
    assert_eq!(r.tolerance, TOLERANCE);
    assert!(!r.macos.is_empty() && r.rust.starts_with("rustc "));

    assert!(r.labels_equal, "labels differ between backends");
    assert!(r.fixtures.iter().all(|f| f.labels_equal));
    assert!(
        r.max_abs_diff <= TOLERANCE,
        "max difference {} is over {TOLERANCE}",
        r.max_abs_diff
    );
    let worst = r
        .fixtures
        .iter()
        .map(|f| f.max_abs_diff)
        .fold(0.0, f64::max);
    assert_eq!(
        worst, r.max_abs_diff,
        "max_abs_diff is not the fixture maximum"
    );

    // The receipt covers exactly the fixtures of the test model.
    let expected = json(&root().join("models").join(MODEL).join("expected.json"));
    let want: Vec<(String, usize)> = expected["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            (
                f["id"].as_str().unwrap().to_owned(),
                f["ids"].as_array().unwrap().len(),
            )
        })
        .collect();
    let got: Vec<(String, usize)> = r
        .fixtures
        .iter()
        .map(|f| (f.id.clone(), f.tokens))
        .collect();
    assert_eq!(got, want);
    assert_eq!(want.len(), 12);

    // The hashes are the manifest's: the receipt describes the files in the repository.
    let manifest = json(&root().join("models/manifest.json"));
    let entry = manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == MODEL)
        .unwrap();
    for name in ["model.onnx", "model.mlmodel"] {
        let listed = entry["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["path"] == name)
            .unwrap_or_else(|| panic!("{name} is not in the manifest"))["sha256"]
            .as_str()
            .unwrap();
        let in_receipt = r
            .files
            .iter()
            .find(|f| f.path == name)
            .unwrap_or_else(|| panic!("{name} is not in the receipt"));
        assert_eq!(in_receipt.sha256, listed, "{name}");
    }
}

#[cfg(all(feature = "onnx", feature = "coreml", target_os = "macos"))]
mod mac {
    use super::*;
    use faberline_nlp::doc::model::{Backend, Batch, argmax_labels, load};
    use std::process::Command;

    fn command_output(program: &str, args: &[&str]) -> String {
        let out = Command::new(program).args(args).output().unwrap();
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    fn num(x: f64) -> Value {
        serde_json::Number::from_f64(x).map(Value::Number).unwrap()
    }

    #[test]
    fn backends_agree() {
        let models = root().join("models");
        let onnx = load(&models, MODEL, Backend::Onnx).unwrap();
        let coreml = load(&models, MODEL, Backend::Coreml).unwrap();
        assert_eq!(onnx.info(), coreml.info());

        let expected = json(&models.join(MODEL).join("expected.json"));
        let mut fixtures = Vec::new();
        let mut max_all = 0f64;
        let mut all_equal = true;
        for fx in expected["fixtures"].as_array().unwrap() {
            let ids: Vec<i64> = fx["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let batch = Batch {
                ids: vec![ids.clone()],
            };
            let a = &onnx.run(&batch).unwrap().logits[0];
            let b = &coreml.run(&batch).unwrap().logits[0];
            assert_eq!(a.shape, b.shape, "{}", fx["id"]);
            assert_eq!(a.data.len(), b.data.len(), "{}", fx["id"]);
            let labels_equal = argmax_labels(a) == argmax_labels(b);
            let diff = a
                .data
                .iter()
                .zip(&b.data)
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
                .fold(0.0, f64::max);
            assert!(labels_equal, "{}: labels differ", fx["id"]);
            assert!(diff <= TOLERANCE, "{}: difference {diff}", fx["id"]);
            max_all = max_all.max(diff);
            all_equal &= labels_equal;
            fixtures.push(serde_json::json!({
                "id": fx["id"],
                "tokens": ids.len(),
                "labels_equal": labels_equal,
                "max_abs_diff": num(diff),
            }));
        }
        assert_eq!(fixtures.len(), 12);
        println!("max absolute logit difference: {max_all:e}");

        if std::env::var("NLP_WRITE_RECEIPT").as_deref() != Ok("1") {
            return;
        }
        let manifest = json(&models.join("manifest.json"));
        let entry = manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == MODEL)
            .unwrap();
        let files: Vec<Value> = ["model.onnx", "model.mlmodel"]
            .iter()
            .map(|name| {
                let f = entry["files"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|f| f["path"] == *name)
                    .unwrap();
                serde_json::json!({ "path": name, "sha256": f["sha256"], "bytes": f["bytes"] })
            })
            .collect();
        let date = command_output("date", &["-u", "+%Y-%m-%d"]);
        let receipt = serde_json::json!({
            "schema": "nlp.receipt.model-runtime.v1",
            "model": MODEL,
            "backends": ["onnx (tract-onnx)", "coreml (objc2-core-ml, CPU only)"],
            "tolerance": TOLERANCE,
            "labels_equal": all_equal,
            "max_abs_diff": num(max_all),
            "fixtures": fixtures,
            "files": files,
            "macos": command_output("sw_vers", &["-productVersion"]),
            "arch": command_output("uname", &["-m"]),
            "rust": command_output("rustc", &["--version"]),
        });
        let path = root().join(format!("docs/receipts/model-runtime-{date}.json"));
        fs::write(path, serde_json::to_string_pretty(&receipt).unwrap() + "\n").unwrap();
    }
}
