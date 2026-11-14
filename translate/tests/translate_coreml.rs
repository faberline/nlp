//! Verification, tamper-refusal, and cross-backend parity suite for the offline Seq2Seq
//! runtime (`nlp.models.seq2seq.v1`): ONNX (`tract-onnx`) and Core ML (`objc2-core-ml`,
//! `CPUOnly`) must agree on encoder hidden states, decoder logits (within `1e-4`), and
//! greedy/beam decoded token IDs across all 12 fixtures of
//! `translate/models/translate-test-tiny/expected.json`.
//!
//! With `NLP_WRITE_TRANSLATE_RECEIPT=1`, `backends_agree` writes
//! `docs/receipts/translate-model-runtime-<date>.json`. `receipt_is_consistent` runs on all
//! platforms (including Linux CI without Core ML) and checks the committed receipt against
//! `translate/models/manifest.json` and `expected.json`.

use std::fs;
use std::path::{Path, PathBuf};

#[cfg(any(feature = "onnx", all(feature = "coreml", target_os = "macos")))]
use faberline_nlp::doc::model::Backend;
use faberline_nlp::doc::model::ModelError;
#[cfg(feature = "onnx")]
use faberline_nlp_translate::model::decode::decode_sequence;
use faberline_nlp_translate::model::verify_seq2seq;
#[cfg(any(feature = "onnx", all(feature = "coreml", target_os = "macos")))]
use faberline_nlp_translate::model::{Seq2SeqTokenizer, load_seq2seq};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn sha256_bytes_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

const MODEL: &str = "translate-test-tiny";
const SRC_MAX_LEN: usize = 80;
const TGT_MAX_LEN: usize = 32;
const D_MODEL: usize = 8;
const VOCAB: usize = 260;
const TOLERANCE: f64 = 1e-4;

const FIXTURES: &[(&str, &str)] = &[
    (
        "fx01",
        "<2en-US> [SUBJ=lin_yuan:she] [REG=polite] 她推開門。",
    ),
    ("fx02", "<2zh-Hant> She opened the door quietly."),
    ("fx03", "<2ja> [SUBJ=lin_yuan:she] 今天天氣很好。"),
    ("fx04", "<2ko> [[PH0]]走了進來。"),
    ("fx05", "<2zh-Hant> 彼女は静かにドアを開けた。"),
    ("fx06", "<2en-US> 그녀는 조용히 문을 열었다."),
    ("fx07", "<2en-GB> 我們使用 [[PH0]] 模型。"),
    ("fx08", "<2en-US> [REG=casual] 哈哈，今天好累😋"),
    ("fx09", "<2en-US> [FACT=negated] 他沒有去學校。"),
    ("fx10", ""),
    (
        "fx11",
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqr",
    ),
    ("fx12", "好"),
];

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_root() -> PathBuf {
    crate_dir().parent().unwrap().to_path_buf()
}

fn models_dir() -> PathBuf {
    crate_dir().join("models")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[derive(Deserialize)]
struct Expected {
    schema: String,
    model: String,
    src_max_len: usize,
    tgt_max_len: usize,
    d_model: usize,
    vocab: usize,
    fixtures: Vec<ExpectedFixture>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ExpectedFixture {
    id: String,
    text: String,
    src_ids: Vec<i64>,
    enc_pos0: Vec<f32>,
    step0_logits_head: Vec<f32>,
    greedy_ids: Vec<i64>,
    greedy_text: String,
}

fn expected() -> Expected {
    let raw = fs::read_to_string(models_dir().join(MODEL).join("expected.json")).unwrap();
    serde_json::from_str(&raw).unwrap()
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

fn refused(dir: &Path, id: &str) -> ModelError {
    let err = match verify_seq2seq(dir, id) {
        Ok(_) => panic!("verify_seq2seq unexpectedly accepted"),
        Err(e) => e,
    };
    #[cfg(feature = "onnx")]
    {
        let l = match load_seq2seq(dir, id, Backend::Onnx) {
            Ok(_) => panic!("load_seq2seq(Onnx) unexpectedly accepted"),
            Err(e) => e,
        };
        assert_eq!(l, err, "load_seq2seq(Onnx) and verify_seq2seq disagree");
    }
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        let l = match load_seq2seq(dir, id, Backend::Coreml) {
            Ok(_) => panic!("load_seq2seq(Coreml) unexpectedly accepted"),
            Err(e) => e,
        };
        assert_eq!(l, err, "load_seq2seq(Coreml) and verify_seq2seq disagree");
    }
    err
}

#[test]
fn manifest_parses_and_verifies_hashes() {
    let m = read_json(&models_dir().join("manifest.json"));
    assert_eq!(m["schema"], "nlp.models.seq2seq.v1");
    let entry = m["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["id"] == MODEL)
        .expect("translate-test-tiny entry");
    let dir = models_dir().join(entry["dir"].as_str().unwrap());
    let files = entry["files"].as_array().unwrap();
    assert_eq!(files.len(), 6);
    assert_eq!(entry["coreml"]["encoder"], "encoder.mlmodel");
    assert_eq!(entry["coreml"]["decoder"], "decoder_step.mlmodel");
    let mut total = 0u64;
    for f in files {
        let rel = f["path"].as_str().unwrap();
        let bytes = fs::read(dir.join(rel)).unwrap();
        assert_eq!(f["sha256"], sha256_bytes_hex(&bytes), "{rel}");
        assert_eq!(f["bytes"].as_u64().unwrap(), bytes.len() as u64, "{rel}");
        total += bytes.len() as u64;
    }
    assert!(total < 120_000, "total {total} bytes exceeds 120 KB budget");

    let verified = verify_seq2seq(&models_dir(), MODEL).unwrap();
    assert_eq!(verified.info.id, MODEL);
    assert_eq!(verified.info.src_max_len, SRC_MAX_LEN);
    assert_eq!(verified.info.tgt_max_len, TGT_MAX_LEN);
    assert_eq!(verified.info.d_model, D_MODEL);
    assert_eq!(verified.info.vocab, VOCAB);
}

#[test]
fn fixture_list_matches_expected_json() {
    let exp = expected();
    assert_eq!(exp.schema, "nlp.models.seq2seq.expected.v1");
    assert_eq!(exp.model, MODEL);
    assert_eq!(exp.src_max_len, SRC_MAX_LEN);
    assert_eq!(exp.tgt_max_len, TGT_MAX_LEN);
    assert_eq!(exp.d_model, D_MODEL);
    assert_eq!(exp.vocab, VOCAB);
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
    let path = tmp.path().join(MODEL).join("decoder_step.onnx");
    let mut bytes = fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&path, bytes).unwrap();
    match refused(tmp.path(), MODEL) {
        ModelError::HashMismatch { file, .. } => assert_eq!(file, "decoder_step.onnx"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn missing_file_is_refused() {
    let tmp = copied_models();
    fs::remove_file(tmp.path().join(MODEL).join("tokenizer.json")).unwrap();
    match refused(tmp.path(), MODEL) {
        ModelError::Missing(path) => assert_eq!(path, "tokenizer.json"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn unknown_model_id_is_refused() {
    match refused(&models_dir(), "nonexistent-seq2seq") {
        ModelError::UnknownModel(id) => assert_eq!(id, "nonexistent-seq2seq"),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn path_traversal_and_absolute_paths_are_refused() {
    let tmp = copied_models();
    let mut m = read_json(&tmp.path().join("manifest.json"));
    m["models"][0]["encoder_onnx"] = serde_json::json!("../encoder.onnx");
    fs::write(
        tmp.path().join("manifest.json"),
        serde_json::to_string_pretty(&m).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        refused(tmp.path(), MODEL),
        ModelError::Manifest(_)
    ));

    let tmp2 = copied_models();
    let mut m2 = read_json(&tmp2.path().join("manifest.json"));
    m2["models"][0]["files"][0]["path"] = serde_json::json!("/etc/hosts");
    fs::write(
        tmp2.path().join("manifest.json"),
        serde_json::to_string_pretty(&m2).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        refused(tmp2.path(), MODEL),
        ModelError::Manifest(_)
    ));
}

#[cfg(unix)]
#[test]
fn symlinked_model_file_is_refused() {
    use std::os::unix::fs::symlink;
    let tmp = copied_models();
    let target = tmp.path().join(MODEL).join("encoder.onnx");
    let backup = tmp.path().join(MODEL).join("encoder.onnx.real");
    fs::rename(&target, &backup).unwrap();
    symlink(&backup, &target).unwrap();
    assert!(matches!(refused(tmp.path(), MODEL), ModelError::Load(_)));
}

#[cfg(feature = "onnx")]
#[test]
fn onnx_matches_expected_and_enforces_bounds() {
    let verified = verify_seq2seq(&models_dir(), MODEL).unwrap();
    let tok = Seq2SeqTokenizer::from_verified(&verified).unwrap();
    let model = load_seq2seq(&models_dir(), MODEL, Backend::Onnx).unwrap();
    let exp = expected();

    for fx in &exp.fixtures {
        let ids = tok.encode_ids(&fx.text).unwrap();
        assert_eq!(ids, fx.src_ids, "{}: tokenizer IDs mismatch", fx.id);

        let enc = model.encode(&ids).unwrap();
        // Verify padded positions are strictly zero-masked.
        assert!(
            enc.hidden[ids.len() * D_MODEL..].iter().all(|&v| v == 0.0),
            "{}: padded encoder positions must be 0.0",
            fx.id
        );
        if !ids.is_empty() {
            for (got, want) in enc.hidden[..D_MODEL].iter().zip(&fx.enc_pos0) {
                assert!(
                    (*got - *want).abs() <= 1e-4,
                    "{}: enc_pos0 got {got} want {want}",
                    fx.id
                );
            }
        }

        let step0 = model.decode_step(&enc, &[model.info().bos_id]).unwrap();
        assert_eq!(step0.len(), VOCAB);
        for (got, want) in step0[..8].iter().zip(&fx.step0_logits_head) {
            assert!(
                (*got - *want).abs() <= 1e-4,
                "{}: step0 logit got {got} want {want}",
                fx.id
            );
        }

        let g = decode_sequence(model.as_ref(), &ids, 1, None).unwrap();
        assert_eq!(g.token_ids, fx.greedy_ids, "{}: greedy IDs mismatch", fx.id);
        let decoded = tok.decode_ids(&g.token_ids).unwrap();
        assert_eq!(decoded, fx.greedy_text, "{}: greedy text mismatch", fx.id);

        // Beam decoding (beam=3) must also be deterministic across repeated calls.
        let b1 = decode_sequence(model.as_ref(), &ids, 3, None).unwrap();
        let b2 = decode_sequence(model.as_ref(), &ids, 3, None).unwrap();
        assert_eq!(b1, b2, "{}: beam decode non-deterministic", fx.id);
    }

    // Over-length and out-of-vocab checks.
    let too_long = vec![4i64; SRC_MAX_LEN + 1];
    assert_eq!(
        model.encode(&too_long).unwrap_err(),
        ModelError::TooLong {
            len: SRC_MAX_LEN + 1,
            max: SRC_MAX_LEN
        }
    );
    assert!(matches!(
        model.encode(&[VOCAB as i64]).unwrap_err(),
        ModelError::Run(_)
    ));
}

// ---------------------------------------------------------------------------
// Cross-backend receipt (`docs/receipts/translate-model-runtime-<date>.json`)
// ---------------------------------------------------------------------------

fn translate_receipts() -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = fs::read_dir(workspace_root().join("docs/receipts"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("translate-model-runtime-") && n.ends_with(".json"))
        })
        .collect();
    found.sort();
    found
}

#[derive(Deserialize)]
struct TranslateRuntimeReceipt {
    schema: String,
    model: String,
    tolerance: f64,
    tokens_equal: bool,
    max_encoder_abs_diff: f64,
    max_logit_abs_diff: f64,
    fixtures: Vec<ReceiptFixture>,
    files: Vec<ReceiptFile>,
    macos: String,
    rust: String,
}

#[derive(Deserialize)]
struct ReceiptFixture {
    id: String,
    src_tokens: usize,
    decoded_tokens: usize,
    tokens_equal: bool,
    max_encoder_abs_diff: f64,
    max_logit_abs_diff: f64,
}

#[derive(Deserialize)]
struct ReceiptFile {
    path: String,
    sha256: String,
}

#[test]
fn receipt_is_consistent() {
    let found = translate_receipts();
    assert_eq!(found.len(), 1, "translate receipts found: {found:?}");
    let r: TranslateRuntimeReceipt = serde_json::from_value(read_json(&found[0])).unwrap();
    assert_eq!(r.schema, "nlp.receipt.translate-model-runtime.v1");
    assert_eq!(r.model, MODEL);
    assert_eq!(r.tolerance, TOLERANCE);
    assert!(!r.macos.is_empty() && r.rust.starts_with("rustc "));

    assert!(r.tokens_equal, "decoded tokens differ between backends");
    assert!(r.fixtures.iter().all(|f| f.tokens_equal));
    assert!(
        r.max_encoder_abs_diff <= TOLERANCE,
        "max encoder diff {} > {TOLERANCE}",
        r.max_encoder_abs_diff
    );
    assert!(
        r.max_logit_abs_diff <= TOLERANCE,
        "max logit diff {} > {TOLERANCE}",
        r.max_logit_abs_diff
    );
    let worst_enc = r
        .fixtures
        .iter()
        .map(|f| f.max_encoder_abs_diff)
        .fold(0.0, f64::max);
    assert_eq!(worst_enc, r.max_encoder_abs_diff);
    let worst_logit = r
        .fixtures
        .iter()
        .map(|f| f.max_logit_abs_diff)
        .fold(0.0, f64::max);
    assert_eq!(worst_logit, r.max_logit_abs_diff);

    let exp = expected();
    let want: Vec<(String, usize, usize)> = exp
        .fixtures
        .iter()
        .map(|f| (f.id.clone(), f.src_ids.len(), f.greedy_ids.len()))
        .collect();
    let got: Vec<(String, usize, usize)> = r
        .fixtures
        .iter()
        .map(|f| (f.id.clone(), f.src_tokens, f.decoded_tokens))
        .collect();
    assert_eq!(got, want);
    assert_eq!(want.len(), 12);

    let manifest = read_json(&models_dir().join("manifest.json"));
    let entry = manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == MODEL)
        .unwrap();
    for name in [
        "encoder.onnx",
        "decoder_step.onnx",
        "encoder.mlmodel",
        "decoder_step.mlmodel",
    ] {
        let listed = entry["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["path"] == name)
            .unwrap_or_else(|| panic!("{name} not in manifest"))["sha256"]
            .as_str()
            .unwrap();
        let in_receipt = r
            .files
            .iter()
            .find(|f| f.path == name)
            .unwrap_or_else(|| panic!("{name} not in receipt"));
        assert_eq!(in_receipt.sha256, listed, "{name}");
    }
}

#[cfg(all(feature = "onnx", feature = "coreml", target_os = "macos"))]
mod mac {
    use super::*;
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
        let mdir = models_dir();
        let verified = verify_seq2seq(&mdir, MODEL).unwrap();
        let tok = Seq2SeqTokenizer::from_verified(&verified).unwrap();
        let onnx = load_seq2seq(&mdir, MODEL, Backend::Onnx).unwrap();
        let coreml = load_seq2seq(&mdir, MODEL, Backend::Coreml).unwrap();
        let coreml_fast = load_seq2seq(&mdir, MODEL, Backend::CoremlFast).unwrap();
        assert_eq!(onnx.info(), coreml.info());
        assert_eq!(onnx.info(), coreml_fast.info());

        let exp = expected();
        let mut fixtures = Vec::new();
        let mut max_enc_all = 0.0f64;
        let mut max_logit_all = 0.0f64;
        let mut all_equal = true;

        for fx in &exp.fixtures {
            let ids = tok.encode_ids(&fx.text).unwrap();
            let enc_a = onnx.encode(&ids).unwrap();
            let enc_b = coreml.encode(&ids).unwrap();
            assert_eq!(enc_a.hidden.len(), enc_b.hidden.len(), "{}", fx.id);

            let enc_diff = enc_a
                .hidden
                .iter()
                .zip(&enc_b.hidden)
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
                .fold(0.0, f64::max);
            assert!(
                enc_diff <= TOLERANCE,
                "{}: encoder diff {enc_diff} > {TOLERANCE}",
                fx.id
            );

            // Step-by-step logit parity along the entire greedy trajectory.
            let mut dec_prefix = vec![onnx.info().bos_id];
            let mut max_fx_logit_diff = 0.0f64;
            for &next_id in &fx.greedy_ids {
                let la = onnx.decode_step(&enc_a, &dec_prefix).unwrap();
                let lb = coreml.decode_step(&enc_b, &dec_prefix).unwrap();
                let d = la
                    .iter()
                    .zip(&lb)
                    .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
                    .fold(0.0, f64::max);
                max_fx_logit_diff = max_fx_logit_diff.max(d);
                dec_prefix.push(next_id);
            }
            // Final EOS-emitting step.
            {
                let la = onnx.decode_step(&enc_a, &dec_prefix).unwrap();
                let lb = coreml.decode_step(&enc_b, &dec_prefix).unwrap();
                let d = la
                    .iter()
                    .zip(&lb)
                    .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
                    .fold(0.0, f64::max);
                max_fx_logit_diff = max_fx_logit_diff.max(d);
            }
            assert!(
                max_fx_logit_diff <= TOLERANCE,
                "{}: logit diff {max_fx_logit_diff} > {TOLERANCE}",
                fx.id
            );

            let ga = decode_sequence(onnx.as_ref(), &ids, 1, None).unwrap();
            let gb = decode_sequence(coreml.as_ref(), &ids, 1, None).unwrap();
            let gf = decode_sequence(coreml_fast.as_ref(), &ids, 1, None).unwrap();
            let ba = decode_sequence(onnx.as_ref(), &ids, 3, None).unwrap();
            let bb = decode_sequence(coreml.as_ref(), &ids, 3, None).unwrap();

            let tokens_equal = ga.token_ids == gb.token_ids
                && ga.token_ids == fx.greedy_ids
                && ba.token_ids == bb.token_ids;
            assert!(
                tokens_equal,
                "{}: decoded tokens differ across backends",
                fx.id
            );
            // CoremlFast (CPU+GPU+ANE fp16) must also produce valid bounded tokens.
            assert_eq!(
                gf.token_ids, ga.token_ids,
                "{}: CoremlFast greedy tokens differ",
                fx.id
            );

            max_enc_all = max_enc_all.max(enc_diff);
            max_logit_all = max_logit_all.max(max_fx_logit_diff);
            all_equal &= tokens_equal;

            fixtures.push(serde_json::json!({
                "id": fx.id,
                "src_tokens": ids.len(),
                "decoded_tokens": ga.token_ids.len(),
                "tokens_equal": tokens_equal,
                "max_encoder_abs_diff": num(enc_diff),
                "max_logit_abs_diff": num(max_fx_logit_diff),
            }));
        }
        assert_eq!(fixtures.len(), 12);

        if std::env::var("NLP_WRITE_TRANSLATE_RECEIPT").as_deref() != Ok("1") {
            return;
        }
        let manifest = read_json(&mdir.join("manifest.json"));
        let entry = manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == MODEL)
            .unwrap();
        let files: Vec<Value> = [
            "encoder.onnx",
            "decoder_step.onnx",
            "encoder.mlmodel",
            "decoder_step.mlmodel",
        ]
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
            "schema": "nlp.receipt.translate-model-runtime.v1",
            "model": MODEL,
            "backends": ["onnx (tract-onnx)", "coreml (objc2-core-ml, CPU only)"],
            "tolerance": TOLERANCE,
            "tokens_equal": all_equal,
            "max_encoder_abs_diff": num(max_enc_all),
            "max_logit_abs_diff": num(max_logit_all),
            "fixtures": fixtures,
            "files": files,
            "macos": command_output("sw_vers", &["-productVersion"]),
            "arch": command_output("uname", &["-m"]),
            "rust": command_output("rustc", &["--version"]),
        });
        let path =
            workspace_root().join(format!("docs/receipts/translate-model-runtime-{date}.json"));
        fs::write(path, serde_json::to_string_pretty(&receipt).unwrap() + "\n").unwrap();
    }
}
