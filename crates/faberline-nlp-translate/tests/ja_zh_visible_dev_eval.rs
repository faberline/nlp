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
const PHENOMENON_BASELINE_FILE: &str = "phenomena/baseline.tsv";
const PHENOMENON_BASELINE_SCHEMA: &str = "faberline.nlp.ja_zh.dev_phenomenon_baseline.v1";
const PHENOMENON_BASELINE_HEADER: &str = "\ngroup_id\tn\tchrf\texact\n";
const PHENOMENON_BASELINE_ENGINE_REV: &str = "90c97a5db6d41aaa41d04a5546c7d51b04dd769f";
const PHENOMENON_BASELINE_REPORT_SHA256: &str =
    "9e1402fefb1f8b8ec32e8de813d351de6be29de73f0fdd55f5f4d00678e1b257";
const PHENOMENON_REFERENCE: &str = "primary-only";

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

fn phenomenon_report_metadata() -> String {
    format!(
        "report-only/phenomena/metadata\tmap_sha256={PHENOMENON_MAP_SHA256}\treference={PHENOMENON_REFERENCE}\tgroups={PHENOMENON_GROUP_COUNT}\tn={PHENOMENON_MEMBER_COUNT}"
    )
}

fn report_score_line(key: &str, count: usize, chrf: f64, exact: usize) -> String {
    format!("report-only/{key}\tn={count}\tchrf={chrf:.2}\texact={exact}")
}

fn validate_baseline_metadata<'a>(
    text: &'a str,
    group_by_member: &HashMap<String, String>,
) -> Result<&'a str, String> {
    let mut fields = HashMap::new();
    let mut banks = HashMap::new();
    for line in text.lines() {
        let comment = line
            .strip_prefix("# ")
            .ok_or("invalid phenomenon baseline metadata line")?;
        let Some((key, value)) = comment.split_once('\t') else {
            continue;
        };
        if key == "bank" {
            let (file, binding) = value
                .split_once('\t')
                .ok_or("invalid phenomenon baseline bank record")?;
            if banks.insert(file, binding).is_some() {
                return Err(format!("duplicate phenomenon baseline bank {file}"));
            }
        } else if fields.insert(key, value).is_some() {
            return Err(format!("duplicate phenomenon baseline field {key}"));
        }
    }
    // #97 adopted this historical source. The check also works with a shallow checkout.
    let source = fields
        .remove("engine_rev")
        .ok_or("missing phenomenon baseline engine_rev")?;
    if source != PHENOMENON_BASELINE_ENGINE_REV {
        return Err("phenomenon baseline historical revision mismatch".to_owned());
    }
    let report_sha256 = fields
        .remove("report_sha256")
        .ok_or("missing phenomenon baseline report_sha256")?;
    let report_count = BANKS.iter().map(|spec| spec.count).sum::<usize>();
    for (key, expected) in [
        ("schema", PHENOMENON_BASELINE_SCHEMA.to_owned()),
        ("map_sha256", PHENOMENON_MAP_SHA256.to_owned()),
        ("reference", PHENOMENON_REFERENCE.to_owned()),
        ("report_count", report_count.to_string()),
        ("mapped_count", PHENOMENON_MEMBER_COUNT.to_string()),
        ("group_count", PHENOMENON_GROUP_COUNT.to_string()),
    ] {
        let actual = fields
            .remove(key)
            .ok_or_else(|| format!("missing phenomenon baseline field {key}"))?;
        if actual != expected {
            return Err(format!("phenomenon baseline {key} mismatch"));
        }
    }
    if !fields.is_empty() {
        return Err("unknown phenomenon baseline metadata field".to_owned());
    }
    for spec in &BANKS {
        let mapped_count = (spec.expected)()
            .into_keys()
            .filter(|id| group_by_member.contains_key(id))
            .count();
        let expected = format!("{}\t{}\t{mapped_count}", spec.sha256, spec.count);
        let actual = banks
            .remove(spec.file)
            .ok_or_else(|| format!("missing phenomenon baseline bank {}", spec.file))?;
        if actual != expected {
            return Err(format!(
                "phenomenon baseline bank mismatch for {}",
                spec.file
            ));
        }
    }
    if !banks.is_empty() {
        return Err("unknown phenomenon baseline bank".to_owned());
    }
    Ok(report_sha256)
}

fn validate_phenomenon_baseline(
    text: &str,
    group_by_member: &HashMap<String, String>,
) -> Result<(), String> {
    let (metadata, rows) = text
        .split_once(PHENOMENON_BASELINE_HEADER)
        .ok_or("missing or invalid phenomenon baseline header")?;
    let expected_report_sha256 = validate_baseline_metadata(metadata, group_by_member)?;
    let mut expected_groups = BTreeMap::new();
    for group in group_by_member.values() {
        *expected_groups.entry(group.as_str()).or_insert(0) += 1;
    }
    let mut historical_scores = BTreeMap::new();
    for row in rows.lines() {
        let columns: Vec<&str> = row.split('\t').collect();
        let [group, count, chrf, exact] = columns.as_slice() else {
            return Err("invalid phenomenon baseline score row".to_owned());
        };
        let expected_count = expected_groups
            .remove(*group)
            .ok_or_else(|| format!("unexpected or duplicate phenomenon baseline group {group}"))?;
        if count.parse::<usize>().ok() != Some(expected_count) {
            return Err(format!("phenomenon baseline count mismatch for {group}"));
        }
        let score = chrf
            .parse::<f64>()
            .map_err(|_| format!("invalid phenomenon baseline chrF for {group}"))?;
        if !(0.0..=100.0).contains(&score) || format!("{score:.2}") != *chrf {
            return Err(format!("invalid phenomenon baseline chrF for {group}"));
        }
        let exact = exact
            .parse::<usize>()
            .map_err(|_| format!("invalid phenomenon baseline exact count for {group}"))?;
        if exact > expected_count {
            return Err(format!(
                "invalid phenomenon baseline exact count for {group}"
            ));
        }
        historical_scores.insert(
            *group,
            report_score_line(&format!("phenomena/{group}"), expected_count, score, exact),
        );
    }
    if !expected_groups.is_empty() {
        let missing: Vec<_> = expected_groups.into_keys().collect();
        return Err(format!(
            "missing phenomenon baseline groups: {}",
            missing.join(", ")
        ));
    }

    // Rebuild the historical report from stored scores. Live scores may improve independently.
    let historical_report = std::iter::once(phenomenon_report_metadata())
        .chain(historical_scores.into_values())
        .collect::<Vec<_>>()
        .join("\n");
    let historical_sha256 = sha256_hex(&historical_report);
    if historical_sha256 != expected_report_sha256 {
        return Err("phenomenon baseline historical report hash mismatch".to_owned());
    }
    if historical_sha256 != PHENOMENON_BASELINE_REPORT_SHA256 {
        return Err(
            "phenomenon baseline report differs from the adopted historical report".to_owned(),
        );
    }
    Ok(())
}

fn require_phenomenon_baseline(
    path: &Path,
    group_by_member: &HashMap<String, String>,
) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read required phenomenon baseline: {error}"))?;
    validate_phenomenon_baseline(&text, group_by_member)
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
        println!("{}", report_score_line(key, self.count, mean, self.exact));
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
    require_phenomenon_baseline(&dir.join(PHENOMENON_BASELINE_FILE), &group_by_member)
        .expect("validate required phenomenon baseline");
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
    println!("{}", phenomenon_report_metadata());
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

    fn pinned_baseline_fixture() -> (String, HashMap<String, String>) {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ja_zh");
        let map = std::fs::read_to_string(dir.join(PHENOMENON_MAP_FILE)).unwrap();
        let groups = validate_phenomenon_map(&map, PHENOMENON_MAP_SHA256).unwrap();
        let baseline = std::fs::read_to_string(dir.join(PHENOMENON_BASELINE_FILE)).unwrap();
        (baseline, groups)
    }

    #[test]
    fn required_phenomenon_baseline_accepts_valid_file_and_rejects_bad_files() {
        let (baseline, groups) = pinned_baseline_fixture();
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("baseline.tsv");
        let missing = require_phenomenon_baseline(&path, &groups).unwrap_err();
        assert!(missing.starts_with("cannot read required phenomenon baseline"));

        std::fs::write(&path, &baseline).unwrap();
        require_phenomenon_baseline(&path, &groups).unwrap();

        let missing_group = baseline
            .lines()
            .filter(|line| !line.starts_with("weather_cause_and_intention\t"))
            .collect::<Vec<_>>()
            .join("\n");
        let missing_reference = baseline.replacen("# reference\tprimary-only\n", "", 1);
        for (contents, expected) in [
            (
                "corrupt baseline",
                "missing or invalid phenomenon baseline header",
            ),
            (
                &missing_group,
                "missing phenomenon baseline groups: weather_cause_and_intention",
            ),
            (
                &missing_reference,
                "missing phenomenon baseline field reference",
            ),
        ] {
            std::fs::write(&path, contents).unwrap();
            assert_eq!(
                require_phenomenon_baseline(&path, &groups),
                Err(expected.to_owned())
            );
        }
    }

    #[test]
    fn rejects_phenomenon_baseline_binding_and_score_errors() {
        let (baseline, groups) = pinned_baseline_fixture();
        let source_line = baseline
            .lines()
            .find(|line| line.starts_with("# engine_rev\t"))
            .unwrap();
        let group = "weather_cause_and_intention";
        let row = baseline
            .lines()
            .find(|line| line.starts_with(group))
            .unwrap();
        let columns: Vec<_> = row.split('\t').collect();
        let [_, count, score, exact] = columns.as_slice() else {
            panic!("invalid baseline fixture row");
        };
        let score_row = |score: &str| format!("{group}\t{count}\t{score}\t{exact}");
        let changed_score = if *score == "100.00" {
            "99.99"
        } else {
            "100.00"
        };
        for (from, to, expected) in [
            (
                PHENOMENON_BASELINE_SCHEMA,
                "other".to_owned(),
                "phenomenon baseline schema mismatch",
            ),
            (
                PHENOMENON_MAP_SHA256,
                "wrong".to_owned(),
                "phenomenon baseline map_sha256 mismatch",
            ),
            (
                source_line,
                "# engine_rev\twrong".to_owned(),
                "phenomenon baseline historical revision mismatch",
            ),
            (
                source_line,
                "# engine_rev\t314181e27ec0fb85413d271ff446aae3607cef80".to_owned(),
                "phenomenon baseline historical revision mismatch",
            ),
            (
                "# reference\tprimary-only",
                "# reference\tbest-variant".to_owned(),
                "phenomenon baseline reference mismatch",
            ),
            (
                "# report_count\t82",
                "# report_count\t81".to_owned(),
                "phenomenon baseline report_count mismatch",
            ),
            (
                "# mapped_count\t32",
                "# mapped_count\t31".to_owned(),
                "phenomenon baseline mapped_count mismatch",
            ),
            (
                "# group_count\t13",
                "# group_count\t12".to_owned(),
                "phenomenon baseline group_count mismatch",
            ),
            (
                BANKS[0].sha256,
                "wrong".to_owned(),
                "phenomenon baseline bank mismatch for nlevels_dev_v1.json",
            ),
            (
                "\t50\t0\n",
                "\t50\t1\n".to_owned(),
                "phenomenon baseline bank mismatch for nlevels_expansion_dev_v1.json",
            ),
            (
                row,
                format!("{group}\t0\t{score}\t{exact}"),
                "phenomenon baseline count mismatch for weather_cause_and_intention",
            ),
            (
                row,
                score_row("NaN"),
                "invalid phenomenon baseline chrF for weather_cause_and_intention",
            ),
            (
                row,
                score_row("100.01"),
                "invalid phenomenon baseline chrF for weather_cause_and_intention",
            ),
            (
                row,
                score_row("50.000"),
                "invalid phenomenon baseline chrF for weather_cause_and_intention",
            ),
            (
                row,
                format!("{group}\t{count}\t{score}\t999"),
                "invalid phenomenon baseline exact count for weather_cause_and_intention",
            ),
            (
                row,
                score_row(changed_score),
                "phenomenon baseline historical report hash mismatch",
            ),
            (
                row,
                row.replacen(group, "unknown_group", 1),
                "unexpected or duplicate phenomenon baseline group unknown_group",
            ),
            (
                row,
                row.replacen(group, "cause_concession_and_outcome", 1),
                "unexpected or duplicate phenomenon baseline group cause_concession_and_outcome",
            ),
        ] {
            assert!(baseline.contains(from), "fixture is missing {from:?}");
            let changed = baseline.replacen(from, &to, 1);
            assert_eq!(
                validate_phenomenon_baseline(&changed, &groups),
                Err(expected.to_owned()),
                "{from:?} -> {to:?}",
            );
        }

        let changed = baseline.replacen(row, &score_row(changed_score), 1);
        let (_, rows) = changed.split_once(PHENOMENON_BASELINE_HEADER).unwrap();
        let mut report = vec![phenomenon_report_metadata()];
        for line in rows.lines() {
            let columns: Vec<_> = line.split('\t').collect();
            let [group, count, score, exact] = columns.as_slice() else {
                panic!("invalid baseline fixture row");
            };
            report.push(format!(
                "report-only/phenomena/{group}\tn={count}\tchrf={score}\texact={exact}"
            ));
        }
        let changed_sha256 = sha256_hex(&report.join("\n"));
        assert_ne!(changed_sha256, PHENOMENON_BASELINE_REPORT_SHA256);
        let rehashed = changed.replacen(PHENOMENON_BASELINE_REPORT_SHA256, &changed_sha256, 1);
        assert_eq!(
            validate_phenomenon_baseline(&rehashed, &groups),
            Err("phenomenon baseline report differs from the adopted historical report".to_owned())
        );
    }
}
