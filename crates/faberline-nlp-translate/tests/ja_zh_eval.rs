//! `ja -> zh-Hant` evaluation over `tests/data/ja_zh/challenge.tsv` with a chrF ratchet.
//!
//! See `tests/data/ja_zh/README.md` for the data format, the dev/test rule and how to
//! update `baseline.tsv`.
//!
//! Environment variables:
//! - `JA_ZH_EVAL_WRITE_BASELINE=1`: write `baseline.tsv` from this run instead of checking it.
//! - `JA_ZH_EVAL_VERBOSE=1`: print per-row details for `dev` rows (never for `test` rows).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::{
    TargetVariant, TranslateOptions, TranslationGlossary, TranslationTrack, analyze_and_translate,
    sha256_hex,
};

/// Allowed chrF drop (in chrF points, 0-100 scale) before the ratchet fails.
const CHRF_TOLERANCE: f64 = 0.5;
const MAX_NGRAM: usize = 6;
const BETA: f64 = 2.0;
const HEADER: &str = "id\tset\tgenre\tphenomenon\tja\tzh_hant\tzh_hant_alt";

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ja_zh")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Split {
    Dev,
    Test,
}

struct Row {
    id: String,
    set: Split,
    phenomenon: String,
    ja: String,
    refs: Vec<String>,
}

fn is_blank(s: &str) -> bool {
    s.chars().all(char::is_whitespace)
}

fn parse_challenge(text: &str) -> Result<Vec<Row>, String> {
    let mut lines = text.lines();
    if lines.next() != Some(HEADER) {
        return Err("challenge.tsv header mismatch".to_owned());
    }
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    for (idx, line) in lines.enumerate() {
        let lineno = idx + 2;
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 6 && cols.len() != 7 {
            return Err(format!(
                "challenge.tsv line {lineno}: expected 7 columns, got {}",
                cols.len()
            ));
        }
        let set = match cols[1] {
            "dev" => Split::Dev,
            "test" => Split::Test,
            other => return Err(format!("challenge.tsv line {lineno}: bad set {other:?}")),
        };
        if !seen.insert(cols[0].to_owned()) {
            return Err(format!(
                "challenge.tsv line {lineno}: duplicate id {}",
                cols[0]
            ));
        }
        if is_blank(cols[4]) {
            return Err(format!("challenge.tsv line {lineno}: empty source"));
        }
        // An empty primary reference would make chrF score an empty output as 100.
        if is_blank(cols[5]) {
            return Err(format!(
                "challenge.tsv line {lineno}: empty primary reference"
            ));
        }
        let mut refs = vec![cols[5].to_owned()];
        if let Some(alt) = cols.get(6).filter(|s| !is_blank(s)) {
            refs.push((*alt).to_owned());
        }
        rows.push(Row {
            id: cols[0].to_owned(),
            set,
            phenomenon: cols[3].to_owned(),
            ja: cols[4].to_owned(),
            refs,
        });
    }
    Ok(rows)
}

fn ngram_counts(chars: &[char], n: usize) -> HashMap<&[char], usize> {
    let mut counts = HashMap::new();
    if chars.len() >= n {
        for w in chars.windows(n) {
            *counts.entry(w).or_insert(0) += 1;
        }
    }
    counts
}

/// Sentence-level chrF (character n-grams 1..=6, beta 2), 0-100. Whitespace is ignored.
/// Precision and recall are averaged over the orders the reference has, then combined.
fn chrf(hyp: &str, reference: &str) -> f64 {
    let h: Vec<char> = hyp.chars().filter(|c| !c.is_whitespace()).collect();
    let r: Vec<char> = reference.chars().filter(|c| !c.is_whitespace()).collect();
    let (mut p_sum, mut r_sum, mut orders) = (0.0, 0.0, 0usize);
    for n in 1..=MAX_NGRAM {
        if r.len() < n {
            break;
        }
        let hc = ngram_counts(&h, n);
        let rc = ngram_counts(&r, n);
        let matched: usize = rc
            .iter()
            .map(|(g, &c)| c.min(hc.get(g).copied().unwrap_or(0)))
            .sum();
        let h_total = h.len().saturating_sub(n - 1);
        let r_total = r.len() - (n - 1);
        p_sum += if h_total == 0 {
            0.0
        } else {
            matched as f64 / h_total as f64
        };
        r_sum += matched as f64 / r_total as f64;
        orders += 1;
    }
    if orders == 0 {
        return if h.is_empty() { 100.0 } else { 0.0 };
    }
    let p = p_sum / orders as f64;
    let rc = r_sum / orders as f64;
    if p + rc == 0.0 {
        return 0.0;
    }
    let b2 = BETA * BETA;
    100.0 * (1.0 + b2) * p * rc / (b2 * p + rc)
}

fn has_kana(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c,
            '\u{3041}'..='\u{3096}' // hiragana
            | '\u{30A1}'..='\u{30FA}' // katakana (without `・` and `ー`)
            | '\u{FF66}'..='\u{FF9F}' // halfwidth katakana
        )
    })
}

#[derive(Default, Clone)]
struct Agg {
    n: usize,
    chrf_sum: f64,
    exact: usize,
    kana_left: usize,
    passthrough: usize,
    errors: usize,
}

impl Agg {
    fn add(&mut self, s: &Scored) {
        self.n += 1;
        self.chrf_sum += s.chrf;
        self.exact += usize::from(s.exact);
        self.kana_left += usize::from(s.kana_left);
        self.passthrough += usize::from(s.passthrough);
        self.errors += usize::from(s.error);
    }
    fn chrf(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.chrf_sum / self.n as f64
        }
    }
    fn exact_pct(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            100.0 * self.exact as f64 / self.n as f64
        }
    }
}

struct Scored {
    chrf: f64,
    exact: bool,
    kana_left: bool,
    passthrough: bool,
    error: bool,
    output: String,
}

fn score_row(row: &Row, glossary: &TranslationGlossary) -> Scored {
    let options = TranslateOptions::new(TargetVariant::ZhHant);
    let result = analyze_and_translate(&row.ja, &options, glossary, None, Backend::Onnx);
    let (output, track, error) = match result {
        Ok(doc) => (doc.target_text, Some(doc.track), false),
        Err(_) => (String::new(), None, true),
    };
    let out = output.trim();
    let chrf = row
        .refs
        .iter()
        .map(|r| chrf(out, r))
        .fold(0.0_f64, f64::max);
    Scored {
        chrf,
        exact: row.refs.iter().any(|r| r.trim() == out),
        kana_left: has_kana(out),
        passthrough: out == row.ja.trim() || track == Some(TranslationTrack::Passthrough),
        error,
        output,
    }
}

/// Ordered metric keys and their aggregates. Per-phenomenon keys cover `dev` rows only,
/// so no near-per-row `test` numbers are ever printed or stored.
fn aggregate(rows: &[Row], scored: &[Scored]) -> Vec<(String, Agg)> {
    let mut overall = Agg::default();
    let mut dev = Agg::default();
    let mut test = Agg::default();
    let mut per_ph: BTreeMap<String, Agg> = BTreeMap::new();
    for (row, s) in rows.iter().zip(scored) {
        overall.add(s);
        match row.set {
            Split::Dev => {
                dev.add(s);
                per_ph.entry(row.phenomenon.clone()).or_default().add(s);
            }
            Split::Test => test.add(s),
        }
    }
    let mut out = vec![
        ("overall".to_owned(), overall),
        ("set/dev".to_owned(), dev),
        ("set/test".to_owned(), test),
    ];
    out.extend(per_ph.into_iter().map(|(k, v)| (format!("dev/{k}"), v)));
    out
}

/// Repo-relative path prefixes whose content decides the engine output. A baseline is
/// written only when none of them has uncommitted changes, so `engine_rev` names the exact
/// engine that produced it. Evaluation files (`tests/`) are not engine inputs.
const ENGINE_INPUTS: &[&str] = &[
    "crates/faberline-nlp-translate/src/",
    "crates/faberline-nlp/src/",
    "crates/faberline-nlp/lexicon/",
    "crates/faberline-nlp-narrative/src/",
];

/// Engine-input paths listed in `git status --porcelain=v1` output.
fn dirty_engine_inputs(porcelain: &str) -> Vec<String> {
    porcelain
        .lines()
        .filter(|l| l.len() > 3)
        .flat_map(|l| {
            let paths = &l[3..];
            // A rename lists `old -> new`; both sides matter.
            paths
                .split(" -> ")
                .map(|p| p.trim_matches('"').to_owned())
                .collect::<Vec<_>>()
        })
        .filter(|p| ENGINE_INPUTS.iter().any(|prefix| p.starts_with(prefix)))
        .collect()
}

fn git(args: &[&str]) -> Result<String, String> {
    let out = std::process::Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map_err(|e| format!("git {args:?}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The commit of a clean engine, or an error when the engine revision cannot be named.
fn clean_engine_revision() -> Result<String, String> {
    let root = git(&["rev-parse", "--show-toplevel"])?;
    let porcelain = std::process::Command::new("git")
        .args(["status", "--porcelain=v1", "--untracked-files=all"])
        .current_dir(root.trim())
        .output()
        .map_err(|e| format!("git status: {e}"))?;
    if !porcelain.status.success() {
        return Err("git status failed".to_owned());
    }
    let dirty = dirty_engine_inputs(&String::from_utf8_lossy(&porcelain.stdout));
    if !dirty.is_empty() {
        return Err(format!(
            "engine inputs have uncommitted changes; commit them before writing a baseline:\n{}",
            dirty.join("\n")
        ));
    }
    let rev = git(&["rev-parse", "HEAD"])?.trim().to_owned();
    if rev.is_empty() {
        return Err("git rev-parse HEAD returned nothing".to_owned());
    }
    Ok(rev)
}

/// Phenomenon left out of the report-only dev number: with an empty glossary the engine
/// cannot transliterate names, so this group mostly measures the glossary, not the rules.
const REPORT_EXCLUDED_PHENOMENON: &str = "katakana_names";

/// Report-only lines (tab separated, starting with `report`). They are printed and written
/// into `baseline.tsv` as comments; the ratchet does not read them.
fn report_lines(rows: &[Row], scored: &[Scored]) -> Vec<String> {
    let mut dev_without = Agg::default();
    let mut dev = Agg::default();
    let mut overall = Agg::default();
    for (row, s) in rows.iter().zip(scored) {
        overall.add(s);
        if row.set == Split::Dev {
            dev.add(s);
            if row.phenomenon != REPORT_EXCLUDED_PHENOMENON {
                dev_without.add(s);
            }
        }
    }
    vec![
        format!(
            "report\tdev_without_{REPORT_EXCLUDED_PHENOMENON}\tn={}\tchrf={:.2}",
            dev_without.n,
            dev_without.chrf()
        ),
        format!("report\tkana_left_dev\t{}/{}", dev.kana_left, dev.n),
        format!(
            "report\tkana_left_overall\t{}/{}",
            overall.kana_left, overall.n
        ),
    ]
}

fn render_baseline(rev: &str, challenge_sha: &str, aggs: &[(String, Agg)]) -> String {
    let mut s = String::new();
    writeln!(
        s,
        "# ja -> zh-Hant evaluation baseline. Written by tests/ja_zh_eval.rs."
    )
    .unwrap();
    writeln!(s, "# engine_rev\t{rev}").unwrap();
    writeln!(s, "# challenge_sha256\t{challenge_sha}").unwrap();
    writeln!(s, "key\tn\tchrf\texact_pct\tkana_left\tpassthrough\terrors").unwrap();
    for (k, a) in aggs {
        writeln!(
            s,
            "{k}\t{}\t{:.2}\t{:.2}\t{}\t{}\t{}",
            a.n,
            a.chrf(),
            a.exact_pct(),
            a.kana_left,
            a.passthrough,
            a.errors
        )
        .unwrap();
    }
    s
}

struct Baseline {
    challenge_sha: String,
    chrf: BTreeMap<String, f64>,
}

/// Every aggregate key a run over `rows` produces: overall, both sets, each dev phenomenon.
fn expected_keys(rows: &[Row]) -> Vec<String> {
    let mut keys = vec![
        "overall".to_owned(),
        "set/dev".to_owned(),
        "set/test".to_owned(),
    ];
    let phenomena: std::collections::BTreeSet<&str> = rows
        .iter()
        .filter(|r| r.set == Split::Dev)
        .map(|r| r.phenomenon.as_str())
        .collect();
    keys.extend(phenomena.into_iter().map(|p| format!("dev/{p}")));
    keys
}

/// Parses `baseline.tsv` and requires exactly the `expected` keys, each once, each finite.
fn parse_baseline(text: &str, expected: &[String]) -> Result<Baseline, String> {
    let mut challenge_sha = String::new();
    let mut chrf = BTreeMap::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# challenge_sha256\t") {
            challenge_sha = rest.trim().to_owned();
            continue;
        }
        if line.starts_with('#') || line.starts_with("key\t") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let value: f64 = cols
            .get(2)
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| format!("bad baseline line: {line:?}"))?;
        if !value.is_finite() {
            return Err(format!("non-finite chrF in baseline line: {line:?}"));
        }
        if chrf.insert(cols[0].to_owned(), value).is_some() {
            return Err(format!("duplicate baseline key {:?}", cols[0]));
        }
    }
    if challenge_sha.is_empty() {
        return Err("baseline has no challenge_sha256 line".to_owned());
    }
    let missing: Vec<&str> = expected
        .iter()
        .filter(|k| !chrf.contains_key(k.as_str()))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        return Err(format!("baseline is missing keys: {}", missing.join(", ")));
    }
    let unknown: Vec<&str> = chrf
        .keys()
        .filter(|k| !expected.contains(k))
        .map(String::as_str)
        .collect();
    if !unknown.is_empty() {
        return Err(format!("baseline has unknown keys: {}", unknown.join(", ")));
    }
    Ok(Baseline {
        challenge_sha,
        chrf,
    })
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

#[test]
fn chrf_metric_sanity() {
    assert!((chrf("我去。", "我去。") - 100.0).abs() < 1e-9);
    assert_eq!(chrf("", "我去。"), 0.0);
    assert_eq!(chrf("天氣", "我去。"), 0.0);
    let partial = chrf("我去了。", "我去。");
    assert!(partial > 0.0 && partial < 100.0);
    // Recall is weighted more than precision (beta = 2).
    assert!(chrf("我去", "我去。") < chrf("我去。。", "我去。"));
}

#[test]
fn kana_detection_covers_halfwidth_katakana() {
    assert!(has_kana("東京へ"));
    assert!(has_kana("コーヒー"));
    assert!(has_kana("ｺｰﾋｰ"));
    assert!(has_kana("咖啡ｺ"));
    assert!(!has_kana("東京ー"));
    assert!(!has_kana("咖啡・紅茶"));
}

const TEST_HEADER_ROW: &str = "jz1\tdev\tdaily\tnegation\t行かない。\t不去。\t";

fn challenge_with(row: &str) -> String {
    format!("{HEADER}\n{row}\n")
}

#[test]
fn challenge_rejects_blank_primary_reference() {
    assert!(parse_challenge(&challenge_with(TEST_HEADER_ROW)).is_ok());
    for primary in ["", "   ", "\u{3000}"] {
        let row = format!("jz1\tdev\tdaily\tnegation\t行かない。\t{primary}\t不去。");
        let err = parse_challenge(&challenge_with(&row)).err();
        assert!(
            err.as_deref()
                .is_some_and(|e| e.contains("empty primary reference")),
            "primary {primary:?}: {err:?}"
        );
    }
    // An empty or blank alt reference is allowed and ignored.
    for alt in ["", " "] {
        let row = format!("jz1\tdev\tdaily\tnegation\t行かない。\t不去。\t{alt}");
        let rows = parse_challenge(&challenge_with(&row)).unwrap();
        assert_eq!(rows[0].refs, vec!["不去。".to_owned()]);
    }
}

fn sample_rows() -> Vec<Row> {
    let text =
        format!("{HEADER}\n{TEST_HEADER_ROW}\njz2\ttest\tdaily\tpassive\t盗まれた。\t被偷了。\t\n");
    parse_challenge(&text).unwrap()
}

#[test]
fn baseline_requires_every_key_once_and_finite() {
    let rows = sample_rows();
    let expected = expected_keys(&rows);
    assert_eq!(expected, ["overall", "set/dev", "set/test", "dev/negation"]);
    let head = "# challenge_sha256\tabc\nkey\tn\tchrf\texact_pct\tkana_left\tpassthrough\terrors\n";
    let good = format!(
        "{head}overall\t2\t10.00\t0\t0\t0\t0\nset/dev\t1\t10.00\t0\t0\t0\t0\n\
         set/test\t1\t10.00\t0\t0\t0\t0\ndev/negation\t1\t10.00\t0\t0\t0\t0\n"
    );
    assert!(parse_baseline(&good, &expected).is_ok());

    // Right hash, no rows.
    assert!(parse_baseline(head, &expected).is_err());
    // A missing key.
    let missing = good.replace("dev/negation\t1\t10.00\t0\t0\t0\t0\n", "");
    assert!(parse_baseline(&missing, &expected).is_err());
    // A duplicate key.
    let dup = format!("{good}overall\t2\t10.00\t0\t0\t0\t0\n");
    assert!(parse_baseline(&dup, &expected).is_err());
    // An unknown key.
    let unknown = format!("{good}dev/idioms\t1\t10.00\t0\t0\t0\t0\n");
    assert!(parse_baseline(&unknown, &expected).is_err());
    // Non-finite scores.
    for bad in ["NaN", "inf", "-inf"] {
        let text = good.replacen("overall\t2\t10.00", &format!("overall\t2\t{bad}"), 1);
        assert!(parse_baseline(&text, &expected).is_err(), "{bad}");
    }
    // No hash line.
    let no_hash = good.replace("# challenge_sha256\tabc\n", "");
    assert!(parse_baseline(&no_hash, &expected).is_err());
}

fn scored(chrf: f64, kana_left: bool) -> Scored {
    Scored {
        chrf,
        exact: false,
        kana_left,
        passthrough: false,
        error: false,
        output: String::new(),
    }
}

#[test]
fn report_lines_exclude_katakana_names_and_count_kana() {
    let text = format!(
        "{HEADER}\n{TEST_HEADER_ROW}\n\
         jz2\tdev\tdaily\tkatakana_names\tマリア。\t瑪麗亞。\t\n\
         jz3\ttest\tdaily\tpassive\t盗まれた。\t被偷了。\t\n"
    );
    let rows = parse_challenge(&text).unwrap();
    let s = [scored(40.0, false), scored(0.0, true), scored(10.0, true)];
    assert_eq!(
        report_lines(&rows, &s),
        [
            "report\tdev_without_katakana_names\tn=1\tchrf=40.00",
            "report\tkana_left_dev\t1/2",
            "report\tkana_left_overall\t2/3",
        ]
    );
    // Report lines are stored as comments, so the baseline parser ignores them.
    let expected = expected_keys(&rows);
    let mut baseline = render_baseline("rev", "abc", &aggregate(&rows, &s));
    for line in report_lines(&rows, &s) {
        writeln!(baseline, "# {line}").unwrap();
    }
    assert_eq!(
        parse_baseline(&baseline, &expected).unwrap().chrf.len(),
        expected.len()
    );
}

#[test]
fn dirty_engine_inputs_block_baseline_but_eval_files_do_not() {
    let porcelain = concat!(
        " M crates/faberline-nlp-translate/tests/ja_zh_eval.rs\n",
        "?? crates/faberline-nlp-translate/tests/data/ja_zh/baseline.tsv\n",
        " M README.md\n",
    );
    assert!(dirty_engine_inputs(porcelain).is_empty());
    let porcelain = concat!(
        " M crates/faberline-nlp-translate/src/lib.rs\n",
        "?? crates/faberline-nlp/lexicon/data/new.tsv\n",
        "R  crates/faberline-nlp/src/a.rs -> crates/faberline-nlp/src/b.rs\n",
        " M crates/faberline-nlp-translate/tests/ja_zh_eval.rs\n",
    );
    assert_eq!(
        dirty_engine_inputs(porcelain),
        [
            "crates/faberline-nlp-translate/src/lib.rs",
            "crates/faberline-nlp/lexicon/data/new.tsv",
            "crates/faberline-nlp/src/a.rs",
            "crates/faberline-nlp/src/b.rs",
        ]
    );
}

#[test]
fn ja_zh_hant_challenge_ratchet() {
    let dir = data_dir();
    let challenge = std::fs::read_to_string(dir.join("challenge.tsv")).expect("read challenge.tsv");
    let challenge_sha = sha256_hex(&challenge);
    let rows = parse_challenge(&challenge).unwrap_or_else(|e| panic!("{e}"));
    let glossary = TranslationGlossary::new();
    let verbose = env_flag("JA_ZH_EVAL_VERBOSE");

    let scored: Vec<Scored> = rows.iter().map(|r| score_row(r, &glossary)).collect();

    if verbose {
        for (row, s) in rows.iter().zip(&scored) {
            // Never print test sentences, outputs or per-row scores.
            if row.set == Split::Dev {
                println!(
                    "{}\t{}\tchrF={:.1}{}{}\t{} => {}",
                    row.id,
                    row.phenomenon,
                    s.chrf,
                    if s.kana_left { " kana" } else { "" },
                    if s.passthrough { " passthrough" } else { "" },
                    row.ja,
                    s.output.trim()
                );
            }
        }
    }

    let aggs = aggregate(&rows, &scored);
    println!("key\tn\tchrF\texact%\tkana_left\tpassthrough\terrors");
    for (k, a) in &aggs {
        println!(
            "{k}\t{}\t{:.2}\t{:.1}\t{}\t{}\t{}",
            a.n,
            a.chrf(),
            a.exact_pct(),
            a.kana_left,
            a.passthrough,
            a.errors
        );
    }

    // Report-only numbers: printed and stored as comments, never part of the ratchet.
    let report = report_lines(&rows, &scored);
    for line in &report {
        println!("{line}");
    }

    let baseline_path = dir.join("baseline.tsv");
    if env_flag("JA_ZH_EVAL_WRITE_BASELINE") {
        let rev = clean_engine_revision().unwrap_or_else(|e| panic!("{e}"));
        let mut text = render_baseline(&rev, &challenge_sha, &aggs);
        for line in &report {
            writeln!(text, "# {line}").unwrap();
        }
        std::fs::write(&baseline_path, text).expect("write baseline.tsv");
        println!("wrote {}", baseline_path.display());
        return;
    }

    let expected = expected_keys(&rows);
    let baseline = parse_baseline(
        &std::fs::read_to_string(&baseline_path)
            .expect("read baseline.tsv (run with JA_ZH_EVAL_WRITE_BASELINE=1 to create it)"),
        &expected,
    )
    .unwrap_or_else(|e| panic!("baseline.tsv: {e}"));
    assert_eq!(
        baseline.challenge_sha, challenge_sha,
        "challenge.tsv changed since baseline.tsv was written; \
         re-run with JA_ZH_EVAL_WRITE_BASELINE=1 and commit both files"
    );
    let current: BTreeMap<&str, f64> = aggs.iter().map(|(k, a)| (k.as_str(), a.chrf())).collect();
    let mut failures = Vec::new();
    for key in &expected {
        let base = baseline.chrf[key];
        match current.get(key.as_str()) {
            None => failures.push(format!("{key}: missing from this run")),
            Some(&now) if !now.is_finite() || now + CHRF_TOLERANCE < base => {
                failures.push(format!("{key}: chrF {now:.2} < baseline {base:.2}"));
            }
            Some(_) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "chrF ratchet failed (tolerance {CHRF_TOLERANCE}):\n{}",
        failures.join("\n")
    );
}
