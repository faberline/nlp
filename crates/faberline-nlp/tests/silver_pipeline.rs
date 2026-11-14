//! Issue 27: offline CI test for the silver-label dry-run fixture, structural validator,
//! two-pass agreement filter, and resumable manifest contract.
//! Pure Rust: runs no Python, no network, and no model.

use faberline_nlp::doc::{Options, analyze_document};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

const DRYRUN: &str = "training/silver/sample_dryrun.jsonl";
const PROBE_TEXTS: &str = "training/probe/probe_texts.jsonl";
const SILVER_SCHEMA: &str = "nlp.silver.v1";

const DEPRELS: &[&str] = &[
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
];

const ROLES: &[&str] = &["ARG0", "ARG1", "ARGM"];
const FLAGS: &[&str] = &["layer2_token_error"];

#[derive(Debug, PartialEq, Eq)]
enum SilverOutcome {
    Kept(Value),
    Dropped(String),
    Invalid(String),
}

fn read_jsonl(path: &str) -> Vec<Value> {
    fs::read_to_string(workspace_root().join(path))
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{path}: bad JSON: {e}")))
        .collect()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn strings(v: &Value, key: &str) -> Result<Vec<String>, String> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} missing"))?
        .iter()
        .map(|x| {
            x.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{key}: not a string"))
        })
        .collect()
}

fn ints(v: &Value, key: &str) -> Result<Vec<usize>, String> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} missing"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| format!("{key}: not an integer"))
        })
        .collect()
}

fn sent_ranges(v: &Value) -> Result<Vec<(usize, usize)>, String> {
    v.get("sentences")
        .and_then(Value::as_array)
        .ok_or_else(|| "sentences missing".to_owned())?
        .iter()
        .map(|s| {
            let arr = s
                .as_array()
                .ok_or_else(|| "sentences: invalid range".to_owned())?;
            if arr.len() != 2 {
                return Err("sentences: invalid range".to_owned());
            }
            let b = arr[0]
                .as_u64()
                .ok_or_else(|| "sentences: invalid range".to_owned())? as usize;
            let e = arr[1]
                .as_u64()
                .ok_or_else(|| "sentences: invalid range".to_owned())? as usize;
            Ok((b, e))
        })
        .collect()
}

fn check_pass(
    ann: &Value,
    ref_tokens: &[String],
    ref_sents: &[(usize, usize)],
) -> Result<(), String> {
    let tokens = strings(ann, "tokens")?;
    if tokens != ref_tokens {
        return Err("tokens differ from layer 1/2 output".into());
    }
    let n = tokens.len();
    if n == 0 {
        return Err("tokens empty".into());
    }

    let sents = sent_ranges(ann)?;
    if sents != ref_sents {
        return Err("sentence ranges differ from layer 1/2 output".into());
    }
    if sents.first().map(|s| s.0) != Some(0) || sents.last().map(|s| s.1) != Some(n) {
        return Err("sentences do not cover the tokens".into());
    }
    for (idx, &(b, e)) in sents.iter().enumerate() {
        if b >= e || (idx > 0 && sents[idx - 1].1 != b) {
            return Err("sentences do not cover the tokens".into());
        }
    }

    let heads = ints(ann, "heads")?;
    let rels = strings(ann, "deprels")?;
    if heads.len() != n || rels.len() != n {
        return Err("tokens, heads, deprels differ in length".into());
    }

    for &(b, e) in &sents {
        let mut roots = 0usize;
        for i in b..e {
            if !DEPRELS.contains(&rels[i].as_str()) {
                return Err(format!(
                    "token {}: label {:?} not in the scheme",
                    i + 1,
                    rels[i]
                ));
            }
            if (heads[i] == 0) != (rels[i] == "root") {
                return Err(format!("token {}: root label and head 0 disagree", i + 1));
            }
            if heads[i] == 0 {
                roots += 1;
            } else if heads[i] == i + 1 || heads[i] < b + 1 || heads[i] > e {
                return Err(format!(
                    "token {}: head {} outside the sentence",
                    i + 1,
                    heads[i]
                ));
            }
        }
        if roots != 1 {
            return Err(format!("sentence has {roots} roots"));
        }
        for i in b..e {
            let (mut cur, mut steps) = (i + 1, 0usize);
            while cur != 0 {
                cur = heads[cur - 1];
                steps += 1;
                if steps > n {
                    return Err(format!("cycle through token {}", i + 1));
                }
            }
        }
    }

    if let Some(flags) = ann.get("flags").filter(|v| !v.is_null()) {
        for f in flags.as_array().ok_or("flags not a list")? {
            let s = f.as_str().unwrap_or("");
            if !FLAGS.contains(&s) {
                return Err(format!("unknown flag {f}"));
            }
        }
    }

    for f in ann
        .get("frames")
        .and_then(Value::as_array)
        .ok_or("frames missing")?
    {
        let p = f
            .get("predicate")
            .and_then(Value::as_u64)
            .ok_or("predicate missing")? as usize;
        if p < 1 || p > n {
            return Err(format!("predicate {p} out of range"));
        }
        let &(sb, se) = sents.iter().find(|&&(b, e)| p > b && p <= e).unwrap();
        for g in f
            .get("args")
            .and_then(Value::as_array)
            .ok_or("args missing")?
        {
            let h = g
                .get("head")
                .and_then(Value::as_u64)
                .ok_or("argument head missing")? as usize;
            if h < 1 || h > n || h == p {
                return Err(format!("argument head {h} invalid for predicate {p}"));
            }
            if h <= sb || h > se {
                return Err(format!(
                    "argument head {h} is not in the sentence of predicate {p}"
                ));
            }
            let role = g.get("role").and_then(Value::as_str).unwrap_or("");
            if !ROLES.contains(&role) {
                return Err("unknown role".into());
            }
        }
    }

    for cluster in ann
        .get("coref")
        .and_then(Value::as_array)
        .ok_or("coref missing")?
    {
        let ms = cluster.as_array().ok_or("cluster not a list")?;
        if ms.len() < 2 {
            return Err("cluster with fewer than two mentions".into());
        }
        let mut seen = BTreeSet::new();
        for m in ms {
            let arr = m.as_array().ok_or("invalid mention span")?;
            if arr.len() != 2 {
                return Err("invalid mention span".into());
            }
            let s = arr[0].as_u64().ok_or("invalid mention span")? as usize;
            let e = arr[1].as_u64().ok_or("invalid mention span")? as usize;
            if s < 1 || s > e || e > n {
                return Err(format!("mention [{s},{e}] out of range"));
            }
            if !seen.insert((s, e)) {
                return Err(format!("duplicate mention [{s},{e}] in cluster"));
            }
        }
    }

    Ok(())
}

type CanonicalFrames = Vec<(usize, Vec<(usize, String)>)>;
type CanonicalCoref = Vec<Vec<(usize, usize)>>;

fn canonical_frames(ann: &Value) -> CanonicalFrames {
    let mut out = Vec::new();
    for f in ann["frames"].as_array().unwrap() {
        let p = f["predicate"].as_u64().unwrap() as usize;
        let mut args: Vec<(usize, String)> = f["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| {
                (
                    g["head"].as_u64().unwrap() as usize,
                    g["role"].as_str().unwrap().to_owned(),
                )
            })
            .collect();
        args.sort();
        args.dedup();
        out.push((p, args));
    }
    out.sort_by_key(|(p, _)| *p);
    out
}

fn canonical_coref(ann: &Value) -> CanonicalCoref {
    let mut clusters: Vec<Vec<(usize, usize)>> = ann["coref"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let mut spans: Vec<(usize, usize)> = c
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    (
                        m[0].as_u64().unwrap() as usize,
                        m[1].as_u64().unwrap() as usize,
                    )
                })
                .collect();
            spans.sort();
            spans
        })
        .collect();
    clusters.sort_by_key(|c| (c[0].0, c[0].1, c.len()));
    clusters
}

fn evaluate_pair(item: &Value) -> SilverOutcome {
    let ref_tokens = strings(item, "tokens").expect("fixture tokens");
    let ref_sents = sent_ranges(item).expect("fixture sentences");
    let pass_a = &item["pass_a"];
    let pass_b = &item["pass_b"];

    if let Err(e) = check_pass(pass_a, &ref_tokens, &ref_sents) {
        return SilverOutcome::Invalid(format!("pass_a: {e}"));
    }
    if let Err(e) = check_pass(pass_b, &ref_tokens, &ref_sents) {
        return SilverOutcome::Invalid(format!("pass_b: {e}"));
    }

    if pass_a["heads"] != pass_b["heads"] || pass_a["deprels"] != pass_b["deprels"] {
        return SilverOutcome::Dropped("disagreed: syntax".into());
    }

    let flags_of = |v: &Value| -> Vec<String> {
        let mut f: Vec<String> = v
            .get("flags")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        f.sort();
        f.dedup();
        f
    };
    let flags_a = flags_of(pass_a);
    let flags_b = flags_of(pass_b);
    if flags_a != flags_b {
        return SilverOutcome::Dropped("disagreed: flags".into());
    }

    let has_tok_err = flags_a.iter().any(|f| f == "layer2_token_error");
    let frames_a = canonical_frames(pass_a);
    let frames_b = canonical_frames(pass_b);
    let coref_a = canonical_coref(pass_a);
    let coref_b = canonical_coref(pass_b);

    if !has_tok_err {
        if frames_a != frames_b {
            return SilverOutcome::Dropped("disagreed: frames".into());
        }
        if coref_a != coref_b {
            return SilverOutcome::Dropped("disagreed: coref".into());
        }
    }

    let usable = if has_tok_err {
        serde_json::json!(["syntax"])
    } else {
        serde_json::json!(["syntax", "frames", "coref"])
    };
    SilverOutcome::Kept(serde_json::json!({
        "id": item["id"],
        "tokens": ref_tokens,
        "sentences": item["sentences"],
        "heads": pass_a["heads"],
        "deprels": pass_a["deprels"],
        "usable_layers": usable,
    }))
}

#[test]
fn dryrun_tokens_match_layer1_layer2_and_do_not_overlap_probe() {
    let items = read_jsonl(DRYRUN);
    let probe = read_jsonl(PROBE_TEXTS);
    let probe_set: BTreeSet<String> = probe
        .iter()
        .map(|p| p["text"].as_str().unwrap().to_owned())
        .collect();

    assert_eq!(items.len(), 10);
    for item in &items {
        let id = item["id"].as_str().unwrap();
        let text = item["text"].as_str().unwrap();
        assert!(
            !probe_set.contains(text),
            "{id}: dryrun text overlaps probe_texts.jsonl"
        );

        let chars: Vec<char> = text.chars().collect();
        let doc = analyze_document(text, &Options::default());
        let repo_tokens: Vec<String> = doc
            .tokens
            .iter()
            .map(|t| chars[t.span.start_char..t.span.end_char].iter().collect())
            .collect();
        let mut repo_sents: Vec<(usize, usize)> = Vec::new();
        for (i, tok) in doc.tokens.iter().enumerate() {
            match repo_sents.last_mut() {
                Some(last) if i > 0 && doc.tokens[i - 1].sentence == tok.sentence => last.1 = i + 1,
                _ => repo_sents.push((i, i + 1)),
            }
        }

        let got_tokens = strings(item, "tokens").unwrap();
        let got_sents = sent_ranges(item).unwrap();
        assert_eq!(
            repo_tokens, got_tokens,
            "{id}: tokens differ from analyze_document"
        );
        assert_eq!(
            repo_sents, got_sents,
            "{id}: sentence ranges differ from analyze_document"
        );
    }
}

#[test]
fn dryrun_enforces_validation_and_two_pass_agreement_filter() {
    let items = read_jsonl(DRYRUN);
    let (mut kept, mut dropped, mut invalid) = (0usize, 0usize, 0usize);

    for item in &items {
        let id = item["id"].as_str().unwrap();
        let want_status = item["expected_status"].as_str().unwrap();
        let want_reason = item["expected_reason"].as_str().unwrap();
        let ref_tokens = strings(item, "tokens").unwrap();
        let ref_sents = sent_ranges(item).unwrap();

        match evaluate_pair(item) {
            SilverOutcome::Kept(merged) => {
                kept += 1;
                assert_eq!(want_status, "kept", "{id}");
                assert_eq!(want_reason, "agreed", "{id}");
                assert_eq!(
                    merged["usable_layers"],
                    serde_json::json!(["syntax", "frames", "coref"])
                );
            }
            SilverOutcome::Dropped(reason) => {
                dropped += 1;
                assert_eq!(want_status, "dropped", "{id}: {reason}");
                assert!(
                    reason.contains(want_reason),
                    "{id}: {reason} vs {want_reason}"
                );
                // Both passes must be individually valid so the drop truly tests disagreement.
                assert_eq!(
                    check_pass(&item["pass_a"], &ref_tokens, &ref_sents),
                    Ok(()),
                    "{id} pass_a"
                );
                assert_eq!(
                    check_pass(&item["pass_b"], &ref_tokens, &ref_sents),
                    Ok(()),
                    "{id} pass_b"
                );
            }
            SilverOutcome::Invalid(reason) => {
                invalid += 1;
                assert_eq!(want_status, "invalid", "{id}: {reason}");
                assert!(
                    reason.contains(want_reason),
                    "{id}: {reason} vs {want_reason}"
                );
            }
        }
    }

    assert_eq!(kept, 4);
    assert_eq!(dropped, 3);
    assert_eq!(invalid, 3);
}

#[test]
fn layer2_token_error_flag_excludes_frames_and_coref_from_merged_layers() {
    let mut item = read_jsonl(DRYRUN)[0].clone();
    item["pass_a"]["flags"] = serde_json::json!(["layer2_token_error"]);
    item["pass_b"]["flags"] = serde_json::json!(["layer2_token_error"]);
    // Even if Pass A and Pass B differ on frames when layer2_token_error is flagged, syntax is kept
    // and frames/coref are excluded from usable_layers.
    item["pass_b"]["frames"] = serde_json::json!([]);
    match evaluate_pair(&item) {
        SilverOutcome::Kept(merged) => {
            assert_eq!(merged["usable_layers"], serde_json::json!(["syntax"]));
        }
        other => panic!("expected Kept with syntax-only usable_layers, got {other:?}"),
    }
}

#[test]
fn resumable_manifest_checks_source_hash_and_batch_states() {
    let raw = fs::read(workspace_root().join(DRYRUN)).unwrap();
    let src_sha = sha256_bytes(&raw);

    let verify_manifest = |m: &Value, expected_source_sha: &str| -> Result<(), String> {
        if m["schema"].as_str() != Some(SILVER_SCHEMA) {
            return Err("bad schema".into());
        }
        if m["model_id"].as_str() != Some("claude-sonnet-5-5") {
            return Err("unexpected model_id".into());
        }
        if m["prompt_version"].as_str().is_none_or(str::is_empty) {
            return Err("missing prompt_version".into());
        }
        if m["source_sha256"].as_str() != Some(expected_source_sha) {
            return Err("source_sha256 mismatch on resume".into());
        }
        for b in m["batches"].as_array().ok_or("batches missing")? {
            let st = b["status"].as_str().unwrap_or("");
            if !["pending", "partial", "ready_to_merge", "merged"].contains(&st) {
                return Err(format!("unknown batch status {st}"));
            }
        }
        Ok(())
    };

    let manifest = serde_json::json!({
        "schema": SILVER_SCHEMA,
        "source_sha256": src_sha,
        "model_id": "claude-sonnet-5-5",
        "prompt_version": "silver-v1",
        "batch_size": 5,
        "batches": [
            {"batch_id": "batch_0000", "status": "merged", "counts": {"total": 5, "kept": 4, "invalid": 0, "dropped": 1}},
            {"batch_id": "batch_0001", "status": "pending", "counts": null}
        ]
    });
    assert_eq!(verify_manifest(&manifest, &src_sha), Ok(()));
    assert_eq!(
        verify_manifest(
            &manifest,
            "0000000000000000000000000000000000000000000000000000000000000000"
        ),
        Err("source_sha256 mismatch on resume".into())
    );
}

#[test]
fn models_manifest_pins_doc_syntax_v1_metadata() {
    let raw =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("models/manifest.json"))
            .expect("models/manifest.json");
    let m: Value = serde_json::from_str(&raw).expect("valid JSON");
    assert_eq!(m["schema"], "nlp.models.v1");
    let models = m["models"].as_array().expect("models list");
    assert_eq!(models[0]["id"], "test-tiny");
    let syn = models
        .iter()
        .find(|e| e["id"] == "doc-syntax-v1")
        .expect("doc-syntax-v1 pinned in models/manifest.json");
    assert_eq!(syn["licence"], "MIT");
    assert_eq!(syn["max_len"], 256);
    assert_eq!(syn["pad_id"], 0);
    assert_eq!(syn["coreml"]["model"], "model.mlmodel");
    assert_eq!(syn["coreml"]["vocab"], 32768);
    let files = syn["files"].as_array().expect("files array");
    assert_eq!(files.len(), 4);
    for f in files {
        let sha = f["sha256"].as_str().expect("sha256 string");
        assert_eq!(sha.len(), 64);
        assert!(f["bytes"].as_u64().unwrap_or(0) > 0);
    }
}

#[cfg(all(feature = "onnx", feature = "coreml", target_os = "macos"))]
#[test]
fn staged_doc_syntax_v1_tract_and_coreml_agree_within_1e4() {
    use faberline_nlp::doc::model::{Backend, Batch, SubwordTokenizer, load_verified, verify};
    use std::path::PathBuf;

    let stage = std::env::var("NLP_MODELS_STAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/nlp-models-stage"));
    if !stage.join("doc-syntax-v1/model.onnx").is_file() {
        // Offline / clean-worktree CI without staged binaries: skip runtime check gracefully.
        return;
    }

    let repo_m: Value = serde_json::from_str(
        &fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("models/manifest.json"))
            .unwrap(),
    )
    .unwrap();
    let stage_m: Value =
        serde_json::from_str(&fs::read_to_string(stage.join("manifest.json")).unwrap()).unwrap();
    let repo_entry = repo_m["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["id"] == "doc-syntax-v1")
        .unwrap();
    let stage_entry = stage_m["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["id"] == "doc-syntax-v1")
        .unwrap();
    assert_eq!(
        repo_entry, stage_entry,
        "repo manifest and staged manifest must match"
    );

    let verified = verify(&stage, "doc-syntax-v1").expect("verify staged doc-syntax-v1");
    let onnx = load_verified(&verified, Backend::Onnx).expect("tract-onnx load doc-syntax-v1");
    let coreml =
        load_verified(&verified, Backend::Coreml).expect("objc2-core-ml load doc-syntax-v1");
    let tok = SubwordTokenizer::from_verified(&verified).expect("load pruned SubwordTokenizer");
    assert_eq!(onnx.info(), coreml.info());

    let expected: Value = serde_json::from_slice(&verified.expected).expect("expected.json");
    assert_eq!(expected["labels"], 421);
    let fixtures = expected["fixtures"].as_array().expect("fixtures array");
    assert!(fixtures.len() >= 16);

    let mut worst_diff = 0.0f32;
    for fx in fixtures {
        let fx_id = fx["id"].as_str().unwrap();
        let text = fx["text"].as_str().unwrap();
        let sub = tok.encode(text).unwrap_or_else(|e| panic!("{fx_id}: {e}"));
        let want_ids: Vec<i64> = fx["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(sub.ids, want_ids, "{fx_id}: subword ids");

        let batch = Batch {
            ids: vec![sub.ids.clone()],
        };
        let out_o = onnx
            .run(&batch)
            .unwrap_or_else(|e| panic!("{fx_id} onnx: {e}"));
        let out_c = coreml
            .run(&batch)
            .unwrap_or_else(|e| panic!("{fx_id} coreml: {e}"));
        let to = &out_o.logits[0];
        let tc = &out_c.logits[0];
        assert_eq!(to.shape, vec![sub.ids.len(), 421], "{fx_id} shape");
        assert_eq!(to.shape, tc.shape, "{fx_id} shape parity");

        for (a, b) in to.data.iter().zip(&tc.data) {
            assert!(a.is_finite() && b.is_finite(), "{fx_id}: non-finite logit");
            let d = (a - b).abs();
            if d > worst_diff {
                worst_diff = d;
            }
            assert!(
                d <= 1e-4,
                "{fx_id}: tract-onnx ({a}) vs objc2-core-ml ({b}) diff {d} > 1e-4"
            );
        }
    }
    assert!(worst_diff <= 1e-4, "worst_diff {worst_diff} > 1e-4");
}
