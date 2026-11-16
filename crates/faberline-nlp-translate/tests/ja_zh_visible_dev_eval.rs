//! Report-only similarity for two adopted, development-visible banks.
#[path = "support/ja_zh_chrf.rs"]
mod ja_zh_chrf;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate, sha256_hex,
};
use serde_json::Value;

struct Spec<'a> {
    name: &'a str,
    file: &'a str,
    schema: &'a str,
    sha256: &'a str,
    count: usize,
    level_field: &'a str,
}

const BANKS: [Spec<'static>; 2] = [
    Spec {
        name: "nlevels_dev_v1",
        file: "nlevels_dev_v1.json",
        schema: "faberline.nlp.ja_zh.nlevels_dev.v1",
        sha256: "a24d91502b3d361eec1ff2bf97381d0accbf27d7516ba2188e77b3b4e96695c0",
        count: 20,
        level_field: "level",
    },
    Spec {
        name: "honorific_san_dev_v1",
        file: "honorific_san_dev_v1.json",
        schema: "faberline.nlp.ja_zh.honorific_san_dev.v1",
        sha256: "870e9e759b65d9f75c277517792315f1cbd0b63ae147c5685690f0eab4f62177",
        count: 12,
        level_field: "level_approx",
    },
];

struct Row {
    id: String,
    pair: String,
    level: String,
    source: String,
    reference: String,
}

fn nonempty<'a>(value: &'a Value, key: &str) -> Result<&'a str, &'static str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or("missing or empty field")
}

fn level_label(s: &str) -> Result<&str, &'static str> {
    let (level, suffix) = s.split_once(' ').unwrap_or((s, ""));
    if matches!(level, "N1" | "N2" | "N3" | "N4" | "N5")
        && matches!(suffix, "" | "approx" | "approximate")
    {
        Ok(level)
    } else {
        Err("invalid level")
    }
}

fn validate_bank(text: &str, spec: &Spec<'_>) -> Result<Vec<Row>, &'static str> {
    if sha256_hex(text) != spec.sha256 {
        return Err("bank hash mismatch");
    }
    let root: Value = serde_json::from_str(text).map_err(|_| "invalid JSON")?;
    if root.get("schema").and_then(Value::as_str) != Some(spec.schema) {
        return Err("bank schema mismatch");
    }
    let items = root
        .get("items")
        .and_then(Value::as_array)
        .ok_or("missing items")?;
    if items.len() != spec.count {
        return Err("bank count mismatch");
    }
    // Validate all visibility markers before inspecting sentence or reference fields.
    if items.iter().any(|r| {
        r.get("visibility").and_then(Value::as_str) != Some("development-visible")
            || r.get("heldout").and_then(Value::as_bool) != Some(false)
    }) {
        return Err("bank contains non-development data");
    }
    let mut rows = Vec::with_capacity(items.len());
    let mut ids = HashSet::new();
    for item in items {
        let id = nonempty(item, "id")?;
        if !ids.insert(id.to_owned()) {
            return Err("duplicate id");
        }
        let variants = item
            .get("allowed_variants")
            .and_then(Value::as_array)
            .ok_or("invalid variants")?;
        if variants
            .iter()
            .any(|v| v.as_str().is_none_or(|s| s.trim().is_empty()))
        {
            return Err("invalid variants");
        }
        rows.push(Row {
            id: id.to_owned(),
            pair: nonempty(item, "pair")?.to_owned(),
            level: level_label(nonempty(item, spec.level_field)?)?.to_owned(),
            source: nonempty(item, "jp")?.to_owned(),
            reference: nonempty(item, "zh_hant")?.to_owned(),
        });
    }
    let by_id: HashMap<&str, &Row> = rows.iter().map(|r| (r.id.as_str(), r)).collect();
    for row in &rows {
        let partner = by_id.get(row.pair.as_str()).ok_or("missing pair")?;
        if partner.id == row.id || partner.pair != row.id || partner.level != row.level {
            return Err("invalid reciprocal pair");
        }
    }
    Ok(rows)
}

#[derive(Default)]
struct Aggregate {
    count: usize,
    sum: f64,
    exact: usize,
}

impl Aggregate {
    fn add(&mut self, score: f64, exact: bool) {
        self.count += 1;
        self.sum += score;
        self.exact += usize::from(exact);
    }
    fn print(&self, key: &str) {
        let mean = self.sum / self.count as f64;
        assert!(mean.is_finite(), "nonfinite aggregate");
        println!(
            "report-only/{key}\tn={}\tchrf={mean:.2}\texact={}",
            self.count, self.exact
        );
    }
}

#[test]
fn adopted_banks_report_only() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ja_zh");
    let mut loaded = Vec::new();
    for spec in &BANKS {
        let text = std::fs::read_to_string(dir.join(spec.file)).expect("read adopted bank");
        let rows = validate_bank(&text, spec).expect("validate adopted bank");
        loaded.push((spec, rows));
    }
    let rev = Command::new("git")
        .args(["rev-parse", "--verify", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("read engine revision");
    assert!(rev.status.success(), "cannot determine engine revision");
    let rev = String::from_utf8(rev.stdout).expect("revision is UTF-8");
    let rev = rev.trim();
    assert!(
        rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid engine revision"
    );
    println!(
        "report-only/status\tprovisional\tengine_git_rev={rev}\tengine_dirty_state=not_checked"
    );
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let glossary = TranslationGlossary::new();
    for (spec, rows) in loaded {
        println!(
            "report-only/{}/metadata\tbank_sha256={}\treference=primary-only",
            spec.name, spec.sha256
        );
        let mut total = Aggregate::default();
        let mut levels: BTreeMap<String, Aggregate> = BTreeMap::new();
        for row in rows {
            let result =
                analyze_and_translate(&row.source, &options, &glossary, None, Backend::Onnx)
                    .unwrap_or_else(|_| panic!("translation failed in {}", spec.name));
            let output = result.target_text.trim();
            assert!(!output.is_empty(), "empty output in {}", spec.name);
            let score = ja_zh_chrf::chrf(output, &row.reference);
            assert!(score.is_finite(), "nonfinite score in {}", spec.name);
            let exact = output == row.reference.trim();
            total.add(score, exact);
            levels.entry(row.level).or_default().add(score, exact);
        }
        total.print(&format!("{}/all", spec.name));
        for (level, aggregate) in levels {
            aggregate.print(&format!("{}/{level}", spec.name));
        }
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        json!({"schema":"toy", "items":[
            {"id":"a","pair":"b","level":"N5 approx","jp":"a","zh_hant":"a","allowed_variants":[],"visibility":"development-visible","heldout":false},
            {"id":"b","pair":"a","level":"N5 approx","jp":"b","zh_hant":"b","allowed_variants":[],"visibility":"development-visible","heldout":false}
        ]})
    }
    fn check(v: &Value, expected_count: usize, schema: &str, expected_hash: Option<&str>) -> bool {
        let text = v.to_string();
        let hash = expected_hash.map_or_else(|| sha256_hex(&text), str::to_owned);
        let spec = Spec {
            name: "toy",
            file: "toy",
            schema,
            sha256: &hash,
            count: expected_count,
            level_field: "level",
        };
        validate_bank(&text, &spec).is_ok()
    }
    #[test]
    fn rejects_invalid_metadata_and_content() {
        let base = fixture();
        assert!(check(&base, 2, "toy", None));
        assert!(!check(&base, 2, "toy", Some("wrong")));
        assert!(!check(&base, 2, "other", None));
        assert!(!check(&base, 3, "toy", None));
        for (field, value) in [
            ("heldout", json!(true)),
            ("visibility", json!("test")),
            ("zh_hant", json!("")),
            ("id", json!("b")),
            ("pair", json!("missing")),
            ("level", json!("N6")),
        ] {
            let mut modified = base.clone();
            modified["items"][0][field] = value;
            assert!(!check(&modified, 2, "toy", None));
        }
        let mut missing_reference = base.clone();
        missing_reference["items"][0]
            .as_object_mut()
            .unwrap()
            .remove("zh_hant");
        assert!(!check(&missing_reference, 2, "toy", None));
        let mut different_level = base.clone();
        different_level["items"][1]["level"] = json!("N4");
        assert!(!check(&different_level, 2, "toy", None));
    }
}
