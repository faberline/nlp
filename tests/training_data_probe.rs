//! Issue 26: the 30-sentence probe annotated by the labeler (Claude Sonnet).
//! Checks that tokens are the real layer 1/2 tokens and that every label is structurally valid.
//! Reads files only: no Python, no network, no model.

use faberline_nlp::doc::{Options, analyze_document};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;

const TEXTS: &str = "training/probe/probe_texts.jsonl";
const ANNOTATIONS: &str = "training/probe/annotations.jsonl";
const DOC: &str = "docs/training-data.md";

/// The 37 Universal Dependencies relation names (no subtypes).
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

fn lines(path: &str) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{path}: bad JSON: {e}")))
        .collect()
}

fn ints(v: &Value, key: &str) -> Result<Vec<usize>, String> {
    v[key]
        .as_array()
        .ok_or(format!("{key} missing"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|n| n as usize)
                .ok_or(format!("{key}: not an integer"))
        })
        .collect()
}

/// Structural check of one annotation. `Ok(())` when valid, else the first problem.
/// Token ids are 1-based; head 0 is the root of a sentence.
fn check(a: &Value) -> Result<(), String> {
    let n = a["tokens"].as_array().ok_or("tokens missing")?.len();
    let heads = ints(a, "heads")?;
    let rels: Vec<&str> = a["deprels"]
        .as_array()
        .ok_or("deprels missing")?
        .iter()
        .map(|x| x.as_str().unwrap_or(""))
        .collect();
    if n == 0 || heads.len() != n || rels.len() != n {
        return Err("tokens, heads, deprels differ in length".into());
    }
    let sents: Vec<(usize, usize)> = a["sentences"]
        .as_array()
        .ok_or("sentences missing")?
        .iter()
        .map(|s| {
            (
                s[0].as_u64().unwrap_or(0) as usize,
                s[1].as_u64().unwrap_or(0) as usize,
            )
        })
        .collect();
    if sents.first().map(|s| s.0) != Some(0) || sents.last().map(|s| s.1) != Some(n) {
        return Err("sentences do not cover the tokens".into());
    }
    for (b, e) in &sents {
        let mut roots = 0;
        for i in *b..*e {
            if !DEPRELS.contains(&rels[i]) {
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
            } else if heads[i] == i + 1 || heads[i] < b + 1 || heads[i] > *e {
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
        for i in *b..*e {
            let (mut cur, mut steps) = (i + 1, 0);
            while cur != 0 {
                cur = heads[cur - 1];
                steps += 1;
                if steps > n {
                    return Err(format!("cycle through token {}", i + 1));
                }
            }
        }
    }
    for f in a["frames"].as_array().ok_or("frames missing")? {
        let p = f["predicate"].as_u64().unwrap_or(0) as usize;
        if p < 1 || p > n {
            return Err(format!("predicate {p} out of range"));
        }
        for g in f["args"].as_array().ok_or("args missing")? {
            let h = g["head"].as_u64().unwrap_or(0) as usize;
            if h < 1 || h > n || h == p {
                return Err(format!("argument head {h} invalid for predicate {p}"));
            }
            if !ROLES.contains(&g["role"].as_str().unwrap_or("")) {
                return Err("unknown role".into());
            }
        }
    }
    for c in a["coref"].as_array().ok_or("coref missing")? {
        let ms = c.as_array().ok_or("cluster not a list")?;
        if ms.len() < 2 {
            return Err("cluster with fewer than two mentions".into());
        }
        for m in ms {
            let (s, e) = (
                m[0].as_u64().unwrap_or(0) as usize,
                m[1].as_u64().unwrap_or(0) as usize,
            );
            if s < 1 || s > e || e > n {
                return Err(format!("mention [{s},{e}] out of range"));
            }
        }
    }
    Ok(())
}

fn sha(path: &str) -> String {
    let mut h = Sha256::new();
    h.update(fs::read(path).unwrap());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn probe_annotations_are_valid_and_use_repo_tokens() {
    let texts = lines(TEXTS);
    let anns = lines(ANNOTATIONS);
    assert_eq!(texts.len(), 30);
    assert_eq!(anns.len(), 30);
    let mut valid = 0;
    for (t, a) in texts.iter().zip(&anns) {
        assert_eq!(t["id"], a["id"]);
        let text = t["text"].as_str().unwrap();
        let chars: Vec<char> = text.chars().collect();
        let doc = analyze_document(text, &Options::default());
        let toks: Vec<String> = doc
            .tokens
            .iter()
            .map(|k| chars[k.span.start_char..k.span.end_char].iter().collect())
            .collect();
        let want: Vec<String> = a["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            toks, want,
            "{}: tokens differ from layer 1/2 output",
            t["id"]
        );
        match check(a) {
            Ok(()) => valid += 1,
            Err(e) => panic!("{}: {e}", t["id"]),
        }
    }
    assert_eq!(valid, 30);
}

#[test]
fn probe_covers_genres_and_languages() {
    let texts = lines(TEXTS);
    for g in ["novel", "paper", "casual"] {
        let of: Vec<&Value> = texts.iter().filter(|t| t["genre"] == g).collect();
        assert_eq!(of.len(), 10, "{g}");
        for (l, n) in [("zh", 4), ("en", 4), ("mixed", 2)] {
            assert_eq!(of.iter().filter(|t| t["lang"] == l).count(), n, "{g} {l}");
        }
    }
}

#[test]
fn checker_rejects_bad_annotations() {
    let good = lines(ANNOTATIONS)[9].clone(); // p10, one sentence
    assert_eq!(check(&good), Ok(()));
    let mut cycle = good.clone();
    cycle["heads"] = serde_json::json!([2, 1, 0, 6, 6, 3, 3]);
    assert!(check(&cycle).unwrap_err().contains("cycle"));
    let mut two_roots = good.clone();
    two_roots["heads"] = serde_json::json!([2, 3, 0, 6, 6, 0, 3]);
    two_roots["deprels"][5] = serde_json::json!("root");
    assert!(check(&two_roots).unwrap_err().contains("2 roots"));
    let mut bad_label = good.clone();
    bad_label["deprels"][0] = serde_json::json!("nsubj:pass");
    assert!(check(&bad_label).unwrap_err().contains("not in the scheme"));
    let mut bad_head = good.clone();
    bad_head["heads"][0] = serde_json::json!(99);
    assert!(check(&bad_head).is_err());
    let mut bad_frame = good.clone();
    bad_frame["frames"] =
        serde_json::json!([{"predicate": 3, "args": [{"role": "ARG9", "head": 2}]}]);
    assert!(check(&bad_frame).is_err());
    let mut bad_coref = good;
    bad_coref["coref"] = serde_json::json!([[[1, 2], [6, 99]]]);
    assert!(check(&bad_coref).is_err());
}

#[test]
fn doc_records_decision_hashes_and_counts() {
    let doc = fs::read_to_string(DOC).unwrap();
    let mut last = 0;
    for h in [
        "# Training data",
        "## Decision",
        "## Labeling model",
        "## Text sources",
        "## Probe",
        "## To confirm with the user",
        "## Changes issue 27 needs",
        "## Limits",
    ] {
        let at = doc.find(h).unwrap_or_else(|| panic!("heading {h} missing"));
        assert!(at >= last, "heading {h} out of order");
        last = at;
    }
    assert!(doc.contains("Claude Sonnet"));
    assert!(
        doc.contains(&sha(TEXTS)),
        "probe_texts.jsonl hash out of date"
    );
    assert!(
        doc.contains(&sha(ANNOTATIONS)),
        "annotations.jsonl hash out of date"
    );
    assert!(doc.contains("30 of 30"), "validity count missing");
}

#[test]
fn source_table_has_no_forbidden_licences() {
    let doc = fs::read_to_string(DOC).unwrap();
    let section = doc
        .split("## Text sources")
        .nth(1)
        .unwrap()
        .split("\n## ")
        .next()
        .unwrap();
    let header = "| genre | lang | source | licence | licence text | fetch | size | attribution |";
    assert!(section.contains(header));
    let rows: Vec<Vec<&str>> = section
        .lines()
        .filter(|l| l.starts_with("| ") && !l.contains("---") && !l.starts_with("| genre"))
        .map(|l| l.trim_matches('|').split(" | ").map(str::trim).collect())
        .collect();
    for (g, l) in [
        ("novel", "en"),
        ("novel", "zh"),
        ("paper", "en"),
        ("paper", "zh"),
        ("casual", "en"),
        ("casual", "zh"),
    ] {
        assert!(
            rows.iter().any(|r| r.len() == 8
                && r[0] == g
                && r[1] == l
                && !r[3].starts_with("rejected")
                && r[4].starts_with("https://")),
            "no qualified source for {g}/{l}"
        );
    }
    for r in rows.iter().filter(|r| !r[3].starts_with("rejected")) {
        for bad in ["BY-SA", "BY-NC", "BY-ND", "GPL"] {
            assert!(!r[3].contains(bad), "{} uses {bad}", r[2]);
        }
    }
}
