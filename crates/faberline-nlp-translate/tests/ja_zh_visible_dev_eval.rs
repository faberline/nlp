//! Report-only similarity by bank, level, and adopted development phenomenon.
#[path = "support/ja_zh_chrf.rs"]
mod ja_zh_chrf;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate, sha256_hex,
};
use serde::Deserialize;
use serde_json::Value;

struct Spec<'a> {
    name: &'a str,
    file: &'a str,
    schema: &'a str,
    sha256: &'a str,
    count: usize,
    level_field: &'a str,
    expected: fn() -> BTreeMap<String, String>,
    row_provenance: bool,
}

const BANKS: [Spec<'static>; 3] = [
    Spec {
        name: "nlevels_dev_v1",
        file: "nlevels_dev_v1.json",
        schema: "faberline.nlp.ja_zh.nlevels_dev.v1",
        sha256: "a24d91502b3d361eec1ff2bf97381d0accbf27d7516ba2188e77b3b4e96695c0",
        count: 20,
        level_field: "level",
        expected: original_membership,
        row_provenance: true,
    },
    Spec {
        name: "honorific_san_dev_v1",
        file: "honorific_san_dev_v1.json",
        schema: "faberline.nlp.ja_zh.honorific_san_dev.v1",
        sha256: "280da5014f9069994bce1f23e8cdc10c3cad3200450278c9b4e3618231134229",
        count: 12,
        level_field: "level_approx",
        expected: honorific_membership,
        row_provenance: false,
    },
    Spec {
        name: "nlevels_expansion_dev_v1",
        file: "nlevels_expansion_dev_v1.json",
        schema: "faberline.nlp.ja_zh.nlevels_expansion_dev.v1",
        sha256: "22d0753fd384ae5f07a118c6417c53b213cc66c1637a8369b8c0315e69ab4fcc",
        count: 50,
        level_field: "level_approx",
        expected: expansion_membership,
        row_provenance: true,
    },
];

const PHENOMENON_MAP_FILE: &str = "phenomena/dev_groups_v1.json";
const PHENOMENON_MAP_SHA256: &str =
    "53e1f5361480f81b315ed94885a9bc61932e9b1e68b61f14046f72c3fd122499";
const PHENOMENON_MAP_SOURCE: &str = "2c89250b31dba56ac8f02cd55629bf6037d3b179";
const PHENOMENON_BANK_FILES: [&str; 2] = ["nlevels_dev_v1.json", "honorific_san_dev_v1.json"];
const PHENOMENON_GROUP_COUNT: usize = 13;
const PHENOMENON_MEMBER_COUNT: usize = 32;

#[derive(Deserialize)]
struct PhenomenonMap {
    schema: String,
    version: String,
    source_revision: String,
    banks: Vec<PhenomenonBank>,
    groups: Vec<PhenomenonGroup>,
}

#[derive(Deserialize)]
struct PhenomenonBank {
    file: String,
    sha256: String,
    rows: usize,
}

#[derive(Deserialize)]
struct PhenomenonGroup {
    id: String,
    members: Vec<PhenomenonMember>,
}

#[derive(Deserialize)]
struct PhenomenonMember {
    bank: String,
    id: String,
}

fn original_membership() -> BTreeMap<String, String> {
    (1..=5)
        .flat_map(|level| (1..=4).map(move |n| (format!("L{level}-{n:02}"), format!("N{level}"))))
        .collect()
}

fn honorific_membership() -> BTreeMap<String, String> {
    (1..=12)
        .map(|n| {
            (
                format!("HS-{n:02}"),
                if (3..=8).contains(&n) { "N4" } else { "N5" }.to_owned(),
            )
        })
        .collect()
}

fn expansion_membership() -> BTreeMap<String, String> {
    (1..=5)
        .flat_map(|level| {
            (1..=10).map(move |n| {
                let id = if level <= 2 {
                    format!("N{level}-D{n:02}")
                } else {
                    format!("N{level}X-{n:02}")
                };
                (id, format!("N{level}"))
            })
        })
        .collect()
}

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
    if root
        .pointer("/provenance/data_license")
        .and_then(Value::as_str)
        != Some("first_party_mit")
    {
        return Err("invalid bank licence or provenance");
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
        if spec.row_provenance && !nonempty(item, "provenance")?.contains("first_party_mit") {
            return Err("invalid row provenance");
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
    let actual: BTreeMap<String, String> = rows
        .iter()
        .map(|r| (r.id.clone(), r.level.clone()))
        .collect();
    if actual != (spec.expected)() {
        return Err("bank ID or level membership mismatch");
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

fn load_banks<'a>(
    specs: &'a [Spec<'a>],
    mut read: impl FnMut(&str) -> Result<String, &'static str>,
) -> Result<Vec<(&'a Spec<'a>, Vec<Row>)>, &'static str> {
    let mut loaded = Vec::new();
    let mut ids = HashSet::new();
    for spec in specs {
        let rows = validate_bank(&read(spec.file)?, spec)?;
        for row in &rows {
            if !ids.insert(row.id.clone()) {
                return Err("duplicate ID across banks");
            }
        }
        loaded.push((spec, rows));
    }
    Ok(loaded)
}

fn validate_phenomenon_map(
    text: &str,
    expected_sha256: &str,
) -> Result<HashMap<String, String>, &'static str> {
    if sha256_hex(text) != expected_sha256 {
        return Err("phenomenon map hash mismatch");
    }
    let map: PhenomenonMap =
        serde_json::from_str(text).map_err(|_| "invalid phenomenon map JSON")?;
    if map.schema != "faberline.nlp.ja_zh.dev_phenomenon_map.v1"
        || map.version != "1.0.0"
        || map.source_revision != PHENOMENON_MAP_SOURCE
    {
        return Err("phenomenon map identity mismatch");
    }
    if map.groups.len() != PHENOMENON_GROUP_COUNT {
        return Err("phenomenon group count mismatch");
    }

    // The report validates the loaded banks against the same specs before measuring any row.
    let mut expected_banks: HashMap<_, _> = BANKS
        .iter()
        .filter(|spec| PHENOMENON_BANK_FILES.contains(&spec.file))
        .map(|spec| (spec.file, spec))
        .collect();
    let mut expected_members: HashMap<_, _> = expected_banks
        .values()
        .flat_map(|spec| (spec.expected)().into_keys().map(move |id| (id, spec.file)))
        .collect();
    if expected_banks.len() != PHENOMENON_BANK_FILES.len()
        || expected_members.len() != PHENOMENON_MEMBER_COUNT
    {
        return Err("pinned phenomenon bank membership mismatch");
    }
    for bank in &map.banks {
        let spec = expected_banks
            .remove(bank.file.as_str())
            .ok_or("unexpected or duplicate phenomenon bank")?;
        if bank.sha256 != spec.sha256 || bank.rows != spec.count {
            return Err("phenomenon bank hash or count mismatch");
        }
    }
    if !expected_banks.is_empty() {
        return Err("missing phenomenon bank");
    }

    let mut group_ids = HashSet::new();
    let mut groups_by_member = HashMap::new();
    for group in &map.groups {
        if group.id.trim().is_empty() || !group_ids.insert(group.id.as_str()) {
            return Err("empty or duplicate phenomenon group ID");
        }
        if group.members.is_empty() {
            return Err("empty phenomenon group");
        }
        for member in &group.members {
            let bank = expected_members
                .remove(member.id.as_str())
                .ok_or("unexpected or duplicate phenomenon member")?;
            if member.bank != bank {
                return Err("phenomenon member bank mismatch");
            }
            groups_by_member.insert(member.id.clone(), group.id.clone());
        }
    }
    if !expected_members.is_empty() {
        return Err("missing phenomenon members");
    }
    Ok(groups_by_member)
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
    let loaded = load_banks(&BANKS, |file| {
        std::fs::read_to_string(dir.join(file)).map_err(|_| "cannot read adopted bank")
    })
    .expect("load all adopted banks");
    let map = std::fs::read_to_string(dir.join(PHENOMENON_MAP_FILE))
        .expect("read adopted phenomenon map");
    let group_by_member = validate_phenomenon_map(&map, PHENOMENON_MAP_SHA256)
        .expect("validate adopted phenomenon map");
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
    let mut executed = HashSet::new();
    let mut overall: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut phenomena: BTreeMap<&str, Aggregate> = BTreeMap::new();
    for (spec, rows) in loaded {
        println!(
            "report-only/{}/metadata\tbank_sha256={}\treference=primary-only",
            spec.name, spec.sha256
        );
        let mut total = Aggregate::default();
        let mut levels: BTreeMap<String, Aggregate> = BTreeMap::new();
        let mut members: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for row in rows {
            assert!(
                executed.insert(row.id.clone()),
                "row evaluated more than once"
            );
            members
                .entry(row.level.clone())
                .or_default()
                .push(row.id.clone());
            overall
                .entry(row.level.clone())
                .or_default()
                .push(row.id.clone());
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
            if let Some(group) = group_by_member.get(&row.id) {
                phenomena
                    .entry(group.as_str())
                    .or_default()
                    .add(score, exact);
            }
        }
        for (level, ids) in &mut members {
            ids.sort();
            println!(
                "report-only/{}/{level}/members\tn={}\tids={}",
                spec.name,
                ids.len(),
                ids.join(",")
            );
        }
        total.print(&format!("{}/all", spec.name));
        for (level, aggregate) in levels {
            aggregate.print(&format!("{}/{level}", spec.name));
        }
    }
    assert_eq!(executed.len(), BANKS.iter().map(|s| s.count).sum::<usize>());
    for (level, ids) in &mut overall {
        ids.sort();
        println!(
            "report-only/all/{level}/members\tn={}\tids={}",
            ids.len(),
            ids.join(",")
        );
    }
    println!("report-only/all/count\tn={}", executed.len());
    let mapped_count = phenomena
        .values()
        .map(|aggregate| aggregate.count)
        .sum::<usize>();
    assert_eq!(mapped_count, PHENOMENON_MEMBER_COUNT);
    assert_eq!(phenomena.len(), PHENOMENON_GROUP_COUNT);
    println!(
        "report-only/phenomena/metadata\tmap_sha256={PHENOMENON_MAP_SHA256}\treference=primary-only\tgroups={}\tn={mapped_count}",
        phenomena.len()
    );
    for (group, aggregate) in phenomena {
        aggregate.print(&format!("phenomena/{group}"));
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        json!({"schema":"toy", "provenance":{"data_license":"first_party_mit"}, "items":[
            {"id":"a","pair":"b","level":"N5 approx","jp":"a","zh_hant":"a","allowed_variants":[],"visibility":"development-visible","heldout":false},
            {"id":"b","pair":"a","level":"N5 approx","jp":"b","zh_hant":"b","allowed_variants":[],"visibility":"development-visible","heldout":false}
        ]})
    }
    fn toy_membership() -> BTreeMap<String, String> {
        [
            ("a".to_owned(), "N5".to_owned()),
            ("b".to_owned(), "N5".to_owned()),
        ]
        .into_iter()
        .collect()
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
            expected: toy_membership,
            row_provenance: false,
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
        let mut wrong_id = base.clone();
        wrong_id["items"][0]["id"] = json!("c");
        wrong_id["items"][1]["pair"] = json!("c");
        assert!(!check(&wrong_id, 2, "toy", None));
        let mut wrong_licence = base.clone();
        wrong_licence["provenance"]["data_license"] = json!("unknown");
        assert!(!check(&wrong_licence, 2, "toy", None));
        let mut missing_reference = base.clone();
        missing_reference["items"][0]
            .as_object_mut()
            .unwrap()
            .remove("zh_hant");
        assert!(!check(&missing_reference, 2, "toy", None));
        let mut different_level = base.clone();
        different_level["items"][0]["level"] = json!("N4");
        different_level["items"][1]["level"] = json!("N4");
        assert!(!check(&different_level, 2, "toy", None));
    }
    #[test]
    fn missing_bank_and_cross_bank_collision_fail_closed() {
        let text = fixture().to_string();
        let hash = sha256_hex(&text);
        let spec = || Spec {
            name: "toy",
            file: "toy",
            schema: "toy",
            sha256: &hash,
            count: 2,
            level_field: "level",
            expected: toy_membership,
            row_provenance: false,
        };
        let specs = [spec(), spec()];
        assert!(load_banks(&specs, |_| Err("missing")).is_err());
        assert!(load_banks(&specs, |_| Ok(text.clone())).is_err());
    }

    #[test]
    fn row_provenance_is_required_when_declared() {
        let mut v = fixture();
        let text = v.to_string();
        let hash = sha256_hex(&text);
        let spec = Spec {
            name: "toy",
            file: "toy",
            schema: "toy",
            sha256: &hash,
            count: 2,
            level_field: "level",
            expected: toy_membership,
            row_provenance: true,
        };
        assert!(validate_bank(&text, &spec).is_err());
        v["items"][0]["provenance"] = json!("first_party_mit");
        v["items"][1]["provenance"] = json!("first_party_mit");
        let text = v.to_string();
        let hash = sha256_hex(&text);
        let spec = Spec {
            sha256: &hash,
            ..spec
        };
        assert!(validate_bank(&text, &spec).is_ok());
    }

    #[test]
    fn rejects_invalid_phenomenon_maps() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ja_zh");
        let text = std::fs::read_to_string(dir.join(PHENOMENON_MAP_FILE)).unwrap();
        assert_eq!(
            validate_phenomenon_map(&text, "wrong"),
            Err("phenomenon map hash mismatch")
        );
        assert_eq!(
            validate_phenomenon_map("{", &sha256_hex("{")),
            Err("invalid phenomenon map JSON")
        );
        let base: Value = serde_json::from_str(&text).unwrap();
        for (pointer, replacement, expected_error) in [
            (
                "/schema",
                json!("other"),
                "phenomenon map identity mismatch",
            ),
            (
                "/version",
                json!("2.0.0"),
                "phenomenon map identity mismatch",
            ),
            (
                "/source_revision",
                json!("other"),
                "phenomenon map identity mismatch",
            ),
            ("/banks", json!([]), "missing phenomenon bank"),
            (
                "/banks/0/sha256",
                json!("wrong"),
                "phenomenon bank hash or count mismatch",
            ),
            (
                "/banks/0/rows",
                json!(19),
                "phenomenon bank hash or count mismatch",
            ),
            (
                "/banks/0/file",
                json!("nlevels_expansion_dev_v1.json"),
                "unexpected or duplicate phenomenon bank",
            ),
            (
                "/banks/1",
                base["banks"][0].clone(),
                "unexpected or duplicate phenomenon bank",
            ),
            ("/groups", json!([]), "phenomenon group count mismatch"),
            (
                "/groups/0/id",
                base["groups"][1]["id"].clone(),
                "empty or duplicate phenomenon group ID",
            ),
            ("/groups/0/members", json!([]), "empty phenomenon group"),
            (
                "/groups/0/members/0/id",
                json!("N5X-01"),
                "unexpected or duplicate phenomenon member",
            ),
            (
                "/groups/0/members/0",
                base["groups"][1]["members"][0].clone(),
                "unexpected or duplicate phenomenon member",
            ),
            (
                "/groups/0/members/0/bank",
                json!("honorific_san_dev_v1.json"),
                "phenomenon member bank mismatch",
            ),
            (
                "/groups/0/members",
                json!([base["groups"][0]["members"][0].clone()]),
                "missing phenomenon members",
            ),
        ] {
            let mut modified = base.clone();
            *modified.pointer_mut(pointer).unwrap() = replacement;
            let text = modified.to_string();
            assert_eq!(
                validate_phenomenon_map(&text, &sha256_hex(&text)),
                Err(expected_error),
                "{pointer}",
            );
        }
    }
}
