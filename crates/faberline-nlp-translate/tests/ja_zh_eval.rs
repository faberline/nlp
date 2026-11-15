//! `ja -> zh-Hant` evaluation over `tests/data/ja_zh/challenge.tsv` with a chrF ratchet.
//!
//! See `tests/data/ja_zh/README.md` for the data format, the dev/test rule and how to
//! update `baseline.tsv`.
//!
//! Environment variables:
//! - `JA_ZH_EVAL_WRITE_BASELINE=1`: write `baseline.tsv` from this run instead of checking it.
//! - `JA_ZH_EVAL_VERBOSE=1`: print per-row details for `dev` rows (never for `test` rows).

use std::collections::{BTreeMap, HashMap};
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

fn parse_challenge(text: &str) -> Vec<Row> {
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(HEADER), "challenge.tsv header mismatch");
    let mut rows = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (idx, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert!(
            cols.len() == 6 || cols.len() == 7,
            "challenge.tsv line {}: expected 7 columns, got {}",
            idx + 2,
            cols.len()
        );
        let set = match cols[1] {
            "dev" => Split::Dev,
            "test" => Split::Test,
            other => panic!("challenge.tsv line {}: bad set {other:?}", idx + 2),
        };
        assert!(seen.insert(cols[0].to_owned()), "duplicate id {}", cols[0]);
        assert!(!cols[5].is_empty(), "empty reference for {}", cols[0]);
        let mut refs = vec![cols[5].to_owned()];
        if let Some(alt) = cols.get(6).filter(|s| !s.is_empty()) {
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
    rows
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
    text.chars()
        .any(|c| matches!(c, '\u{3041}'..='\u{3096}' | '\u{30A1}'..='\u{30FA}'))
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

fn engine_revision() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
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

fn parse_baseline(text: &str) -> Baseline {
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
            .unwrap_or_else(|| panic!("bad baseline line: {line:?}"));
        chrf.insert(cols[0].to_owned(), value);
    }
    Baseline {
        challenge_sha,
        chrf,
    }
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
    assert!(has_kana("東京へ"));
    assert!(!has_kana("東京ー"));
}

#[test]
fn ja_zh_hant_challenge_ratchet() {
    let dir = data_dir();
    let challenge = std::fs::read_to_string(dir.join("challenge.tsv")).expect("read challenge.tsv");
    let challenge_sha = sha256_hex(&challenge);
    let rows = parse_challenge(&challenge);
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

    let baseline_path = dir.join("baseline.tsv");
    if env_flag("JA_ZH_EVAL_WRITE_BASELINE") {
        let text = render_baseline(&engine_revision(), &challenge_sha, &aggs);
        std::fs::write(&baseline_path, text).expect("write baseline.tsv");
        println!("wrote {}", baseline_path.display());
        return;
    }

    let baseline = parse_baseline(
        &std::fs::read_to_string(&baseline_path)
            .expect("read baseline.tsv (run with JA_ZH_EVAL_WRITE_BASELINE=1 to create it)"),
    );
    assert_eq!(
        baseline.challenge_sha, challenge_sha,
        "challenge.tsv changed since baseline.tsv was written; \
         re-run with JA_ZH_EVAL_WRITE_BASELINE=1 and commit both files"
    );
    let current: BTreeMap<&str, f64> = aggs.iter().map(|(k, a)| (k.as_str(), a.chrf())).collect();
    let mut failures = Vec::new();
    for (key, &base) in &baseline.chrf {
        match current.get(key.as_str()) {
            None => failures.push(format!("{key}: missing from this run")),
            Some(&now) if now + CHRF_TOLERANCE < base => {
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
