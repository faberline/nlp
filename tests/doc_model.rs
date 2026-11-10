use faberline_nlp::Span;
use faberline_nlp::doc::Reason;
#[cfg(feature = "onnx")]
use faberline_nlp::doc::model::Batch;
use faberline_nlp::doc::model::{
    Backend, Model, ModelError, Tensor, argmax_labels, load, no_model_unknown,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const MODEL: &str = "test-tiny";
const LABELS: usize = 5;
const MAX_LEN: usize = 64;

const FIXTURES: &[(&str, &str)] = &[
    ("fx01", "她推開門，屋裡很安靜。"),
    ("fx02", "She opened the door and the room was quiet."),
    ("fx03", "她說 hello，然後笑了。"),
    ("fx04", "本文提出一個簡單的方法。"),
    ("fx05", "We propose a simple method."),
    ("fx06", "我們使用 BERT 模型。"),
    ("fx07", "哈哈，今天好累😋"),
    ("fx08", "lol that was so good!"),
    ("fx09", "明天 meeting 幾點？"),
    ("fx10", ""),
    (
        "fx11",
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab",
    ),
    ("fx12", "好"),
];

fn models_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/models"))
}

#[derive(Deserialize)]
struct Expected {
    schema: String,
    model: String,
    max_len: usize,
    labels: usize,
    fixtures: Vec<Fixture>,
}

#[derive(Deserialize)]
#[allow(dead_code)] // the logit fields are read only with the onnx feature
struct Fixture {
    id: String,
    text: String,
    ids: Vec<i64>,
    spans: Vec<[usize; 2]>,
    logits: Vec<Vec<f32>>,
    argmax: Vec<usize>,
}

fn expected() -> Expected {
    let text = fs::read_to_string(models_dir().join(MODEL).join("expected.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn err_of(r: Result<Box<dyn Model>, ModelError>) -> ModelError {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    }
}

#[cfg(feature = "onnx")]
fn verified() -> faberline_nlp::doc::model::Verified {
    faberline_nlp::doc::model::verify(&models_dir(), MODEL).unwrap()
}

/// Copies `models/` to a temporary directory, rewrites the manifest with `edit`, and loads
/// the model from there.
fn load_edited(edit: impl FnOnce(&mut serde_json::Value)) -> ModelError {
    let tmp = copied_models();
    let mut m = manifest_value(tmp.path());
    edit(&mut m);
    write_manifest(tmp.path(), &m);
    err_of(load(tmp.path(), MODEL, Backend::Onnx))
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let dst = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &dst);
        } else {
            fs::copy(entry.path(), dst).unwrap();
        }
    }
}

fn copied_models() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    copy_dir(&models_dir(), tmp.path());
    tmp
}

fn manifest_value(dir: &Path) -> serde_json::Value {
    let text = fs::read_to_string(dir.join("manifest.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn write_manifest(dir: &Path, value: &serde_json::Value) {
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(value).unwrap(),
    )
    .unwrap();
}

#[cfg(feature = "onnx")]
fn bits(t: &Tensor) -> (String, Vec<usize>, Vec<u32>) {
    (
        t.name.clone(),
        t.shape.clone(),
        t.data.iter().map(|x| x.to_bits()).collect(),
    )
}

#[test]
fn manifest_parses_and_lists_hashes() {
    let m = manifest_value(&models_dir());
    assert_eq!(m["schema"], "nlp.models.v1");
    let entry = m["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["id"] == MODEL)
        .expect("test-tiny entry");
    let dir = models_dir().join(entry["dir"].as_str().unwrap());
    let files = entry["files"].as_array().unwrap();
    assert_eq!(files.len(), 3);
    let mut total = 0u64;
    for f in files {
        let path = f["path"].as_str().unwrap();
        let bytes = fs::read(dir.join(path)).unwrap();
        assert_eq!(f["sha256"], sha256_hex(&bytes).as_str(), "{path}");
        assert_eq!(f["bytes"].as_u64().unwrap(), bytes.len() as u64, "{path}");
        total += bytes.len() as u64;
    }
    assert!(total < 100_000, "total {total} bytes");
}

#[test]
fn fixture_list_matches_expected_json() {
    let exp = expected();
    assert_eq!(exp.schema, "nlp.models.expected.v1");
    assert_eq!(exp.model, MODEL);
    assert_eq!(exp.max_len, MAX_LEN);
    assert_eq!(exp.labels, LABELS);
    let got: Vec<(&str, &str)> = exp
        .fixtures
        .iter()
        .map(|f| (f.id.as_str(), f.text.as_str()))
        .collect();
    assert_eq!(got, FIXTURES);
}

#[test]
fn hash_mismatch_is_refused() {
    let tmp = copied_models();
    let path = tmp.path().join(MODEL).join("model.onnx");
    let mut bytes = fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&path, bytes).unwrap();
    match err_of(load(tmp.path(), MODEL, Backend::Onnx)) {
        ModelError::HashMismatch { file, .. } => assert_eq!(file, "model.onnx"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn missing_file_is_refused() {
    let tmp = copied_models();
    fs::remove_file(tmp.path().join(MODEL).join("tokenizer.json")).unwrap();
    match err_of(load(tmp.path(), MODEL, Backend::Onnx)) {
        ModelError::Missing(path) => assert_eq!(path, "tokenizer.json"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn unknown_model_id_is_refused() {
    match err_of(load(&models_dir(), "no-such-model", Backend::Onnx)) {
        ModelError::UnknownModel(id) => assert_eq!(id, "no-such-model"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn manifest_path_escape_is_refused() {
    let tmp = copied_models();
    let mut m = manifest_value(tmp.path());
    m["models"][0]["model"] = serde_json::json!("../model.onnx");
    write_manifest(tmp.path(), &m);
    assert!(matches!(
        err_of(load(tmp.path(), MODEL, Backend::Onnx)),
        ModelError::Manifest(_)
    ));
}

#[test]
fn absolute_file_path_is_refused() {
    let e = load_edited(|m| {
        m["models"][0]["files"][1]["path"] = serde_json::json!("/etc/hosts");
    });
    assert!(matches!(e, ModelError::Manifest(_)), "{e:?}");
}

#[test]
fn dir_with_parent_component_is_refused() {
    let e = load_edited(|m| m["models"][0]["dir"] = serde_json::json!("../test-tiny"));
    assert!(matches!(e, ModelError::Manifest(_)), "{e:?}");
}

#[test]
fn model_not_listed_in_files_is_refused() {
    let e = load_edited(|m| {
        let files = m["models"][0]["files"].as_array_mut().unwrap();
        files.retain(|f| f["path"] != "model.onnx");
    });
    match e {
        ModelError::Manifest(msg) => assert!(msg.contains("model.onnx"), "{msg}"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn truncated_file_is_refused_by_size() {
    let tmp = copied_models();
    let path = tmp.path().join(MODEL).join("model.onnx");
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
    match err_of(load(tmp.path(), MODEL, Backend::Onnx)) {
        ModelError::HashMismatch { file, .. } => assert_eq!(file, "model.onnx (bytes)"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn appended_file_is_refused_by_size() {
    let tmp = copied_models();
    let path = tmp.path().join(MODEL).join("tokenizer.json");
    let mut bytes = fs::read(&path).unwrap();
    bytes.extend_from_slice(b" ");
    fs::write(&path, bytes).unwrap();
    match err_of(load(tmp.path(), MODEL, Backend::Onnx)) {
        ModelError::HashMismatch { file, .. } => assert_eq!(file, "tokenizer.json (bytes)"),
        other => panic!("unexpected {other:?}"),
    }
}

#[cfg(unix)]
#[test]
fn symlinked_file_is_refused() {
    let tmp = copied_models();
    let dir = tmp.path().join(MODEL);
    let real = tmp.path().join("elsewhere.json");
    fs::copy(dir.join("tokenizer.json"), &real).unwrap();
    fs::remove_file(dir.join("tokenizer.json")).unwrap();
    std::os::unix::fs::symlink(&real, dir.join("tokenizer.json")).unwrap();
    match err_of(load(tmp.path(), MODEL, Backend::Onnx)) {
        ModelError::Load(msg) => assert!(msg.contains("tokenizer.json"), "{msg}"),
        other => panic!("unexpected {other:?}"),
    }
}

#[cfg(not(feature = "onnx"))]
#[test]
fn no_backend_without_feature() {
    assert!(matches!(
        err_of(load(&models_dir(), MODEL, Backend::Onnx)),
        ModelError::NoBackend
    ));
}

#[test]
fn no_model_unknown_shape() {
    let u = no_model_unknown(
        "syntax",
        Span {
            start_char: 3,
            end_char: 9,
        },
    );
    assert_eq!(u.layer, "syntax");
    assert_eq!(u.reason, Reason::NoModel);
    assert_eq!(
        serde_json::to_value(&u).unwrap(),
        serde_json::json!({
            "layer": "syntax",
            "span": {"start_char": 3, "end_char": 9},
            "reason": "no_model"
        })
    );
}

#[test]
fn argmax_labels_ties_and_nan() {
    let t = Tensor {
        name: "logits".to_owned(),
        shape: vec![3, 3],
        data: vec![1.0, 1.0, 0.0, 0.0, 2.0, 2.0, f32::NAN, 1.0, 0.0],
    };
    assert_eq!(argmax_labels(&t), vec![0, 1, 1]);
}

#[test]
fn into_reason_maps_no_model_errors_only() {
    assert_eq!(ModelError::NoBackend.into_reason(), Some(Reason::NoModel));
    assert_eq!(
        ModelError::UnknownModel("x".to_owned()).into_reason(),
        Some(Reason::NoModel)
    );
    assert_eq!(
        ModelError::Missing("x".to_owned()).into_reason(),
        Some(Reason::NoModel)
    );
    assert_eq!(ModelError::TooLong { len: 17, max: 16 }.into_reason(), None);
    assert_eq!(
        ModelError::HashMismatch {
            file: "model.onnx".to_owned(),
            expected: "a".to_owned(),
            actual: "b".to_owned(),
        }
        .into_reason(),
        None
    );
}

#[cfg(feature = "onnx")]
#[test]
fn encode_offsets_map_to_chars() {
    use faberline_nlp::doc::model::SubwordTokenizer;

    let tok = SubwordTokenizer::from_verified(&verified()).unwrap();
    for fx in &expected().fixtures {
        let sub = tok.encode(&fx.text).unwrap();
        let n = fx.text.chars().count();
        assert_eq!(sub.ids, fx.ids, "{}", fx.id);
        assert_eq!(sub.spans.len(), sub.ids.len(), "{}", fx.id);
        let pairs: Vec<[usize; 2]> = sub
            .spans
            .iter()
            .map(|s| [s.start_char, s.end_char])
            .collect();
        assert_eq!(pairs, fx.spans, "{}", fx.id);
        let mut covered = vec![false; n];
        for s in &sub.spans {
            assert!(s.start_char <= s.end_char && s.end_char <= n, "{}", fx.id);
            for flag in &mut covered[s.start_char..s.end_char] {
                *flag = true;
            }
        }
        for (i, c) in fx.text.chars().enumerate() {
            assert!(
                c.is_whitespace() || covered[i],
                "{}: char {i} uncovered",
                fx.id
            );
        }
    }
    for text in ["好", "😋", "她"] {
        let sub = tok.encode(text).unwrap();
        assert!(sub.ids.len() > 1, "{text}");
        for s in &sub.spans {
            assert_eq!(
                *s,
                Span {
                    start_char: 0,
                    end_char: 1
                },
                "{text}"
            );
        }
    }
}

#[cfg(feature = "onnx")]
#[test]
fn logits_match_numpy() {
    let m = load(&models_dir(), MODEL, Backend::Onnx).unwrap();
    assert_eq!(m.info().id, MODEL);
    assert_eq!(m.info().max_len, MAX_LEN);
    for fx in &expected().fixtures {
        let out = m
            .run(&Batch {
                ids: vec![fx.ids.clone()],
            })
            .unwrap();
        let t = &out.logits[0];
        assert_eq!(t.shape, vec![fx.ids.len(), LABELS], "{}", fx.id);
        assert_eq!(t.data.len(), fx.ids.len() * LABELS, "{}", fx.id);
        for (row, want) in t.data.chunks(LABELS).zip(&fx.logits) {
            for (got, w) in row.iter().zip(want) {
                assert!((got - w).abs() <= 1e-5, "{}: {got} vs {w}", fx.id);
            }
        }
        assert_eq!(argmax_labels(t), fx.argmax, "{}", fx.id);
    }
}

#[cfg(feature = "onnx")]
#[test]
fn batch_equals_single_runs() {
    let m = load(&models_dir(), MODEL, Backend::Onnx).unwrap();
    let ids: Vec<Vec<i64>> = expected().fixtures.iter().map(|f| f.ids.clone()).collect();
    let batch = m.run(&Batch { ids: ids.clone() }).unwrap();
    assert_eq!(batch.logits.len(), ids.len());
    for (i, id) in ids.iter().enumerate() {
        let single = m
            .run(&Batch {
                ids: vec![id.clone()],
            })
            .unwrap();
        assert_eq!(bits(&batch.logits[i]), bits(&single.logits[0]), "item {i}");
    }
}

#[cfg(feature = "onnx")]
#[test]
fn run_is_deterministic() {
    let a = load(&models_dir(), MODEL, Backend::Onnx).unwrap();
    let b = load(&models_dir(), MODEL, Backend::Onnx).unwrap();
    assert_eq!(a.info(), b.info());
    let batch = Batch {
        ids: expected().fixtures.iter().map(|f| f.ids.clone()).collect(),
    };
    let first = a.run(&batch).unwrap();
    let second = a.run(&batch).unwrap();
    let third = b.run(&batch).unwrap();
    for i in 0..batch.ids.len() {
        assert_eq!(bits(&first.logits[i]), bits(&second.logits[i]));
        assert_eq!(bits(&first.logits[i]), bits(&third.logits[i]));
    }
}

#[cfg(feature = "onnx")]
#[test]
fn too_long_is_an_error() {
    let m = load(&models_dir(), MODEL, Backend::Onnx).unwrap();
    let r = m.run(&Batch {
        ids: vec![vec![2, 3], vec![2; MAX_LEN + 1]],
    });
    match r {
        Err(ModelError::TooLong { len, max }) => {
            assert_eq!(len, MAX_LEN + 1);
            assert_eq!(max, MAX_LEN);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[cfg(feature = "onnx")]
#[test]
fn parsers_use_verified_bytes_not_the_disk() {
    use faberline_nlp::doc::model::{SubwordTokenizer, load_verified};

    let tmp = copied_models();
    let dir = tmp.path().join(MODEL);
    let v = faberline_nlp::doc::model::verify(tmp.path(), MODEL).unwrap();
    // Replace both files after the check. Parsing must still see the verified bytes.
    fs::write(dir.join("model.onnx"), b"not an onnx model").unwrap();
    fs::write(dir.join("tokenizer.json"), b"{ garbage").unwrap();

    let tok = SubwordTokenizer::from_verified(&v).unwrap();
    let fx = &expected().fixtures[0];
    assert_eq!(tok.encode(&fx.text).unwrap().ids, fx.ids);

    let m = load_verified(&v, Backend::Onnx).unwrap();
    let out = m
        .run(&Batch {
            ids: vec![fx.ids.clone()],
        })
        .unwrap();
    for (row, want) in out.logits[0].data.chunks(LABELS).zip(&fx.logits) {
        for (got, w) in row.iter().zip(want) {
            assert!((got - w).abs() <= 1e-5, "{got} vs {w}");
        }
    }
    assert_eq!(v.model.len() as u64, v.info.files[0].bytes);
}

#[cfg(feature = "onnx")]
#[test]
fn ids_outside_vocabulary_are_errors() {
    let m = load(&models_dir(), MODEL, Backend::Onnx).unwrap();
    for bad in [258i64, 100_000, -1] {
        let r = m.run(&Batch {
            ids: vec![vec![2, 3], vec![2, bad]],
        });
        assert!(matches!(r, Err(ModelError::Run(_))), "id {bad}: {r:?}");
    }
    // The largest valid id (257) still runs.
    assert!(
        m.run(&Batch {
            ids: vec![vec![257]]
        })
        .is_ok()
    );
}
