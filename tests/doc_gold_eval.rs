//! Issue 28: frozen 300-item / 450-sentence gold evaluation set (`gold_layers4_6_v1.jsonl`),
//! annotation guide (`docs/annotation-guide.md`), and validator + reference scorer checks.
//! Offline only: reads repository files and calls `analyze_document` with no Python, no network,
//! and no external model files.

use faberline_nlp::doc::{Lang, Options, Upos, analyze_document};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;

const GOLD_PATH: &str = "tests/data/gold_layers4_6_v1.jsonl";
const GUIDE_PATH: &str = "docs/annotation-guide.md";
const PROBE_TEXTS_PATH: &str = "training/probe/probe_texts.jsonl";
const POS_GOLD_PATH: &str = "tests/data/pos_gold.json";

/// Pinned SHA-256 of `tests/data/gold_layers4_6_v1.jsonl`. Changing the frozen gold set requires
/// updating both this constant and `docs/annotation-guide.md`.
const GOLD_SHA256: &str = "cd78c88a1ebfbbcaf7113df0b5ef104402560a177d26d4fea5987850c8e3f149";

/// All 17 UD v2 Universal POS tags.
const UPOS_TAGS: &[&str] = &[
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
    "PUNCT", "SCONJ", "SYM", "VERB", "X",
];

/// All 37 UD v2 base dependency relation labels (no subtypes).
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
const ZERO_ROLES: &[&str] = &["nsubj", "obj"];
const GENRES: &[&str] = &["novel", "paper", "casual"];
const LANGS: &[(&str, usize)] = &[("zh", 40), ("en", 40), ("mixed", 20)];
const VARIANTS: &[(&str, &str, usize, usize)] = &[
    // (variant, expected_lang, 1-sent count per genre, 2-sent count per genre)
    ("zh-TW", "zh", 10, 10),
    ("zh-CN", "zh", 10, 10),
    ("en", "en", 20, 20),
    ("mixed-TW", "mixed", 5, 5),
    ("mixed-CN", "mixed", 5, 5),
];

/// Disjoint Traditional-only vs Simplified-only diagnostic character sets to prevent mechanical
/// or accidental cross-script contamination between `*-TW` and `*-CN` items.
const TRAD_ONLY_CHARS: &str = "這們個為與會從開關門車馬鳥魚風雲電書長見貝買賣讀寫聽說請讓給對進遠還邊過運選連週醫雙離難響頁題顆飯館體點齊";
const SIMP_ONLY_CHARS: &str = "这们个为与会从开关门车马鸟鱼风云电书长见贝买卖读写听说请让给对进远还边过运选连医双离难响页题颗饭馆体点齐";

fn read_jsonl(path: &str) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(idx, l)| {
            serde_json::from_str(l).unwrap_or_else(|e| panic!("{path}:{}: bad JSON: {e}", idx + 1))
        })
        .collect()
}

fn sha256_hex(path: &str) -> String {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn norm_text(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn lang_str(l: Lang) -> &'static str {
    match l {
        Lang::Zh => "zh",
        Lang::En => "en",
        Lang::Mixed => "mixed",
        Lang::Und => "und",
    }
}

fn upos_str(u: Upos) -> &'static str {
    match u {
        Upos::Adj => "ADJ",
        Upos::Adp => "ADP",
        Upos::Adv => "ADV",
        Upos::Aux => "AUX",
        Upos::Cconj => "CCONJ",
        Upos::Det => "DET",
        Upos::Intj => "INTJ",
        Upos::Noun => "NOUN",
        Upos::Num => "NUM",
        Upos::Part => "PART",
        Upos::Pron => "PRON",
        Upos::Propn => "PROPN",
        Upos::Punct => "PUNCT",
        Upos::Sconj => "SCONJ",
        Upos::Sym => "SYM",
        Upos::Verb => "VERB",
        Upos::X => "X",
    }
}

fn ints(v: &Value, key: &str) -> Result<Vec<usize>, String> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} missing"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| format!("{key}: not a non-negative integer"))
        })
        .collect()
}

fn strs<'a>(v: &'a Value, key: &str) -> Result<Vec<&'a str>, String> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} missing"))?
        .iter()
        .map(|x| x.as_str().ok_or_else(|| format!("{key}: not a string")))
        .collect()
}

/// Checks that 1-based inclusive span `[s, e]` lies within one sentence and has a single syntactic
/// head inside `s..=e` whose governor lies outside `s..=e`.
fn check_constituent_span(
    s: usize,
    e: usize,
    n: usize,
    sents: &[(usize, usize)],
    heads: &[usize],
    label: &str,
) -> Result<(), String> {
    if s < 1 || s > e || e > n {
        return Err(format!("{label} [{s},{e}] out of range"));
    }
    if !sents.iter().any(|(b, se)| s > *b && e <= *se) {
        return Err(format!("{label} [{s},{e}] crosses sentence boundary"));
    }
    let external_heads = (s..=e)
        .filter(|&tok| {
            let h = heads[tok - 1];
            h < s || h > e
        })
        .count();
    if external_heads != 1 {
        return Err(format!(
            "{label} [{s},{e}] is not a single-headed syntactic constituent"
        ));
    }
    Ok(())
}

/// Full structural & linguistic check of one gold item across Layers 4, 5, and 6.
fn check_gold_item(a: &Value) -> Result<(), String> {
    let toks = strs(a, "tokens")?;
    let n = toks.len();
    let upos = strs(a, "upos")?;
    let heads = ints(a, "heads")?;
    let rels = strs(a, "deprels")?;
    if n == 0 || upos.len() != n || heads.len() != n || rels.len() != n {
        return Err("tokens, upos, heads, deprels differ in length".into());
    }

    let raw_sents = a
        .get("sentences")
        .and_then(Value::as_array)
        .ok_or_else(|| "sentences missing".to_string())?;
    let mut sents = Vec::with_capacity(raw_sents.len());
    for s in raw_sents {
        let pair = s.as_array().ok_or("sentence range not a 2-element array")?;
        if pair.len() != 2 {
            return Err("sentence range not a 2-element array".into());
        }
        let b = pair[0].as_u64().ok_or("sentence start not an integer")? as usize;
        let e = pair[1].as_u64().ok_or("sentence end not an integer")? as usize;
        if b >= e {
            return Err("empty or inverted sentence range".into());
        }
        sents.push((b, e));
    }
    if sents.first().map(|s| s.0) != Some(0) || sents.last().map(|s| s.1) != Some(n) {
        return Err("sentences do not cover the tokens".into());
    }
    for w in sents.windows(2) {
        if w[0].1 != w[1].0 {
            return Err("sentences are not contiguous".into());
        }
    }

    for (i, &u) in upos.iter().enumerate() {
        if !UPOS_TAGS.contains(&u) {
            return Err(format!("token {}: invalid upos {u:?}", i + 1));
        }
        if u == "PUNCT" && rels[i] != "punct" {
            return Err(format!(
                "token {}: PUNCT token has deprel {:?}",
                i + 1,
                rels[i]
            ));
        }
    }
    for (dep_idx, &h) in heads.iter().enumerate() {
        if h >= 1 && h <= n && upos[h - 1] == "PUNCT" {
            return Err(format!("token {}: attached to PUNCT head {h}", dep_idx + 1));
        }
    }

    for &(b, e) in &sents {
        let mut roots = 0;
        for i in b..e {
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
            let (mut cur, mut steps) = (i + 1, 0);
            while cur != 0 {
                cur = heads[cur - 1];
                steps += 1;
                if steps > e - b {
                    return Err(format!("cycle through token {}", i + 1));
                }
            }
        }
    }

    let frames = a
        .get("frames")
        .and_then(Value::as_array)
        .ok_or_else(|| "frames missing".to_string())?;
    let mut seen_preds = HashSet::new();
    for f in frames {
        let p = f.get("predicate").and_then(Value::as_u64).unwrap_or(0) as usize;
        if p < 1 || p > n {
            return Err(format!("predicate {p} out of range"));
        }
        if !seen_preds.insert(p) {
            return Err(format!("duplicate frame for predicate {p}"));
        }
        let &(sb, se) = sents
            .iter()
            .find(|(b, e)| p > *b && p <= *e)
            .ok_or_else(|| format!("predicate {p} not in any sentence"))?;
        let args = f
            .get("args")
            .and_then(Value::as_array)
            .ok_or_else(|| "args missing".to_string())?;
        if args.is_empty() {
            return Err(format!("predicate {p} has empty args"));
        }
        let mut seen_args = HashSet::new();
        for g in args {
            let h = g.get("head").and_then(Value::as_u64).unwrap_or(0) as usize;
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
            if !seen_args.insert((role, h)) {
                return Err(format!("duplicate argument ({role}, {h}) on predicate {p}"));
            }
        }
    }

    let coref = a
        .get("coref")
        .and_then(Value::as_array)
        .ok_or_else(|| "coref missing".to_string())?;
    for c in coref {
        let ms = c.as_array().ok_or("cluster not a list")?;
        if ms.len() < 2 {
            return Err("cluster with fewer than two mentions".into());
        }
        let mut seen_mentions = HashSet::new();
        for m in ms {
            let pair = m.as_array().ok_or("mention not a 2-element list")?;
            if pair.len() != 2 {
                return Err("mention not a 2-element list".into());
            }
            let s = pair[0].as_u64().unwrap_or(0) as usize;
            let e = pair[1].as_u64().unwrap_or(0) as usize;
            check_constituent_span(s, e, n, &sents, &heads, "mention")?;
            if !seen_mentions.insert((s, e)) {
                return Err(format!("duplicate mention [{s},{e}] in cluster"));
            }
        }
    }

    if let Some(zeros_val) = a.get("zero_mentions") {
        let zeros = zeros_val
            .as_array()
            .ok_or_else(|| "zero_mentions not a list".to_string())?;
        let mut seen_zeros = HashSet::new();
        for z in zeros {
            let za = z.get("zero_at").and_then(Value::as_u64).unwrap_or(0) as usize;
            if za < 1 || za > n {
                return Err(format!("zero_at {za} out of range"));
            }
            if !matches!(upos[za - 1], "VERB" | "ADJ") {
                return Err(format!("zero_at {za} is not a VERB/ADJ predicate"));
            }
            let role = z.get("role").and_then(Value::as_str).unwrap_or("");
            if !ZERO_ROLES.contains(&role) {
                return Err(format!("invalid zero_mentions role {role:?}"));
            }
            if !seen_zeros.insert((za, role)) {
                return Err(format!("duplicate zero_mention ({za}, {role})"));
            }
            let &(sb, se) = sents.iter().find(|(b, e)| za > *b && za <= *e).unwrap();
            for tok in (sb + 1)..=se {
                if heads[tok - 1] == za && rels[tok - 1] == role {
                    return Err(format!(
                        "zero_mention ({za}, {role}) conflicts with overt {role} at token {tok}"
                    ));
                }
            }
            let ant = z
                .get("antecedent")
                .and_then(Value::as_array)
                .ok_or_else(|| "zero_mention antecedent missing".to_string())?;
            if ant.len() != 2 {
                return Err("zero_mention antecedent not a 2-element span".into());
            }
            let s = ant[0].as_u64().unwrap_or(0) as usize;
            let e = ant[1].as_u64().unwrap_or(0) as usize;
            check_constituent_span(s, e, n, &sents, &heads, "zero antecedent")?;
        }
    }

    Ok(())
}

#[derive(Debug, PartialEq)]
struct EvalMetrics {
    uas: f64,
    las: f64,
    upos_acc: f64,
    frame_f1: f64,
    b3_f1: f64,
}

fn prf1(tp: f64, p_den: f64, r_den: f64) -> f64 {
    if p_den == 0.0 && r_den == 0.0 {
        return 1.0;
    }
    if p_den == 0.0 || r_den == 0.0 || tp == 0.0 {
        return 0.0;
    }
    let p = tp / p_den;
    let r = tp / r_den;
    2.0 * p * r / (p + r)
}

/// Reference Rust scorer over annotation JSON values (mirrors `training/eval_gold.py`).
fn score_predictions(gold: &[Value], pred: &[Value]) -> EvalMetrics {
    assert_eq!(gold.len(), pred.len());
    let mut tok_total = 0usize;
    let mut uas_ok = 0usize;
    let mut las_ok = 0usize;
    let mut upos_ok = 0usize;

    let mut g_frames: HashSet<(usize, usize, String, usize)> = HashSet::new();
    let mut p_frames: HashSet<(usize, usize, String, usize)> = HashSet::new();

    let mut b3_r_num = 0.0_f64;
    let mut b3_r_den = 0.0_f64;
    let mut b3_p_num = 0.0_f64;
    let mut b3_p_den = 0.0_f64;

    for (idx, (g, p)) in gold.iter().zip(pred.iter()).enumerate() {
        let g_u = strs(g, "upos").unwrap();
        let g_h = ints(g, "heads").unwrap();
        let g_r = strs(g, "deprels").unwrap();
        let p_u = strs(p, "upos").unwrap();
        let p_h = ints(p, "heads").unwrap();
        let p_r = strs(p, "deprels").unwrap();
        assert_eq!(g_u.len(), p_u.len());

        for i in 0..g_u.len() {
            tok_total += 1;
            if g_u[i] == p_u[i] {
                upos_ok += 1;
            }
            if g_h[i] == p_h[i] {
                uas_ok += 1;
                if g_r[i] == p_r[i] {
                    las_ok += 1;
                }
            }
        }

        for (src, dst) in [(&g["frames"], &mut g_frames), (&p["frames"], &mut p_frames)] {
            if let Some(arr) = src.as_array() {
                for f in arr {
                    let pred_tok = f["predicate"].as_u64().unwrap() as usize;
                    for a in f["args"].as_array().unwrap() {
                        dst.insert((
                            idx,
                            pred_tok,
                            a["role"].as_str().unwrap().to_owned(),
                            a["head"].as_u64().unwrap() as usize,
                        ));
                    }
                }
            }
        }

        let parse_clusters = |v: &Value| -> Vec<HashSet<(usize, usize)>> {
            v.as_array()
                .map(|cls| {
                    cls.iter()
                        .map(|cl| {
                            cl.as_array()
                                .unwrap()
                                .iter()
                                .map(|m| {
                                    (
                                        m[0].as_u64().unwrap() as usize,
                                        m[1].as_u64().unwrap() as usize,
                                    )
                                })
                                .collect::<HashSet<_>>()
                        })
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .unwrap_or_default()
        };
        let kc = parse_clusters(&g["coref"]);
        let rc = parse_clusters(&p["coref"]);
        for k in &kc {
            b3_r_den += k.len() as f64;
            for r in &rc {
                let inter = k.intersection(r).count() as f64;
                if inter > 0.0 {
                    b3_r_num += (inter * inter) / (k.len() as f64);
                }
            }
        }
        for r in &rc {
            b3_p_den += r.len() as f64;
            for k in &kc {
                let inter = r.intersection(k).count() as f64;
                if inter > 0.0 {
                    b3_p_num += (inter * inter) / (r.len() as f64);
                }
            }
        }
    }

    let frame_tp = g_frames.intersection(&p_frames).count() as f64;
    let frame_f1 = prf1(frame_tp, p_frames.len() as f64, g_frames.len() as f64);
    let b3_f1 = if b3_p_den == 0.0 && b3_r_den == 0.0 {
        1.0
    } else if b3_p_den == 0.0 || b3_r_den == 0.0 || b3_p_num == 0.0 || b3_r_num == 0.0 {
        0.0
    } else {
        let p = b3_p_num / b3_p_den;
        let r = b3_r_num / b3_r_den;
        2.0 * p * r / (p + r)
    };

    EvalMetrics {
        uas: (uas_ok as f64) / (tok_total as f64),
        las: (las_ok as f64) / (tok_total as f64),
        upos_acc: (upos_ok as f64) / (tok_total as f64),
        frame_f1,
        b3_f1,
    }
}

#[test]
fn gold_sha256_and_guide_headings_are_frozen() {
    assert_eq!(
        sha256_hex(GOLD_PATH),
        GOLD_SHA256,
        "tests/data/gold_layers4_6_v1.jsonl changed without updating GOLD_SHA256"
    );

    let guide = fs::read_to_string(GUIDE_PATH).unwrap();
    assert!(
        guide.contains(GOLD_SHA256),
        "docs/annotation-guide.md does not record current GOLD_SHA256"
    );

    let mut last_pos = 0;
    for heading in [
        "# Annotation Guide (Layers 4–6)",
        "## Corpus & Provenance",
        "## JSONL Schema & Layer 1–2 Alignment",
        "## Layer 4: UPOS & Dependency Syntax",
        "## Layer 5: Semantic Frames & Issue #14 Role Mapping",
        "## Layer 6: Coreference Clusters & Pro-Drop Zero Mentions",
        "## Second-Pass Audit & Resolved Disagreements",
        "## Evaluation Scorer (`training/eval_gold.py`)",
    ] {
        let pos = guide
            .find(heading)
            .unwrap_or_else(|| panic!("missing heading {heading:?} in {GUIDE_PATH}"));
        assert!(pos >= last_pos, "heading {heading:?} out of order");
        last_pos = pos;
    }

    for tag in UPOS_TAGS {
        assert!(
            guide.contains(&format!("`{tag}`")),
            "annotation guide missing UPOS tag {tag}"
        );
    }
    for rel in DEPRELS {
        assert!(
            guide.contains(&format!("`{rel}`")),
            "annotation guide missing deprel {rel}"
        );
    }
    for role in ROLES {
        assert!(
            guide.contains(role),
            "annotation guide missing frame role {role}"
        );
    }
    for fine_role in [
        "agent", "patient", "target", "time", "place", "manner", "quantity",
    ] {
        assert!(
            guide.contains(&format!("`{fine_role}`")),
            "annotation guide missing Issue #14 role `{fine_role}`"
        );
    }
}

#[test]
fn gold_corpus_balance_variants_and_disjointness() {
    let gold = read_jsonl(GOLD_PATH);
    assert_eq!(gold.len(), 300, "expected 300 gold items");

    let total_sents: usize = gold
        .iter()
        .map(|g| g["sentences"].as_array().unwrap().len())
        .sum();
    assert!(
        total_sents >= 300,
        "expected >= 300 gold sentences, got {total_sents}"
    );
    assert_eq!(total_sents, 450, "expected exact 450 sentences");

    // Verify IDs g001..g300 in canonical order.
    for (idx, item) in gold.iter().enumerate() {
        let want_id = format!("g{:03}", idx + 1);
        assert_eq!(item["id"].as_str(), Some(want_id.as_str()));
    }

    // Verify balanced 3 genres x (40 zh, 40 en, 20 mixed) and 5 regional variants per genre.
    for &genre in GENRES {
        let genre_items: Vec<&Value> = gold.iter().filter(|g| g["genre"] == genre).collect();
        assert_eq!(genre_items.len(), 100, "genre {genre} item count");
        let genre_sents: usize = genre_items
            .iter()
            .map(|g| g["sentences"].as_array().unwrap().len())
            .sum();
        assert_eq!(genre_sents, 150, "genre {genre} sentence count");

        for &(lang, want_n) in LANGS {
            let cnt = genre_items.iter().filter(|g| g["lang"] == lang).count();
            assert_eq!(cnt, want_n, "{genre}/{lang}");
        }

        for &(variant, want_lang, want_1s, want_2s) in VARIANTS {
            let var_items: Vec<&&Value> = genre_items
                .iter()
                .filter(|g| g["variant"] == variant)
                .collect();
            assert_eq!(
                var_items.len(),
                want_1s + want_2s,
                "{genre}/{variant} total"
            );
            for v in &var_items {
                assert_eq!(
                    v["lang"].as_str(),
                    Some(want_lang),
                    "{genre}/{variant} lang"
                );
            }
            let n_1s = var_items
                .iter()
                .filter(|v| v["sentences"].as_array().unwrap().len() == 1)
                .count();
            let n_2s = var_items
                .iter()
                .filter(|v| v["sentences"].as_array().unwrap().len() == 2)
                .count();
            assert_eq!(n_1s, want_1s, "{genre}/{variant} 1-sentence count");
            assert_eq!(n_2s, want_2s, "{genre}/{variant} 2-sentence count");
        }
    }

    // Verify Traditional vs Simplified script separation across *-TW and *-CN variants.
    let trad_only: HashSet<char> = TRAD_ONLY_CHARS.chars().collect();
    let simp_only: HashSet<char> = SIMP_ONLY_CHARS.chars().collect();
    assert!(trad_only.is_disjoint(&simp_only));
    let mut tw_trad_hits = 0usize;
    let mut cn_simp_hits = 0usize;
    for item in &gold {
        let id = item["id"].as_str().unwrap();
        let variant = item["variant"].as_str().unwrap();
        let text = item["text"].as_str().unwrap();
        match variant {
            "zh-TW" | "mixed-TW" => {
                for ch in text.chars() {
                    assert!(
                        !simp_only.contains(&ch),
                        "{id} ({variant}) contains Simplified-only char {ch:?}"
                    );
                    if trad_only.contains(&ch) {
                        tw_trad_hits += 1;
                    }
                }
            }
            "zh-CN" | "mixed-CN" => {
                for ch in text.chars() {
                    assert!(
                        !trad_only.contains(&ch),
                        "{id} ({variant}) contains Traditional-only char {ch:?}"
                    );
                    if simp_only.contains(&ch) {
                        cn_simp_hits += 1;
                    }
                }
            }
            "en" => {}
            other => panic!("{id}: unknown variant {other}"),
        }
    }
    assert!(
        tw_trad_hits >= 80,
        "expected rich Traditional usage in *-TW"
    );
    assert!(cn_simp_hits >= 80, "expected rich Simplified usage in *-CN");

    // Verify strict text disjointness from probe_texts.jsonl and pos_gold.json, and internal uniqueness.
    let mut forbidden = HashSet::new();
    for p in read_jsonl(PROBE_TEXTS_PATH) {
        forbidden.insert(norm_text(p["text"].as_str().unwrap()));
    }
    let pos_gold: Value =
        serde_json::from_str(&fs::read_to_string(POS_GOLD_PATH).unwrap()).unwrap();
    if let Some(arr) = pos_gold
        .get("cases")
        .and_then(Value::as_array)
        .or_else(|| pos_gold.as_array())
    {
        for entry in arr {
            if let Some(t) = entry.get("text").and_then(Value::as_str) {
                forbidden.insert(norm_text(t));
            }
        }
    }

    let mut seen_gold_texts = HashSet::new();
    for item in &gold {
        let id = item["id"].as_str().unwrap();
        let nt = norm_text(item["text"].as_str().unwrap());
        assert!(
            !forbidden.contains(&nt),
            "{id} overlaps probe_texts.jsonl or pos_gold.json"
        );
        assert!(
            seen_gold_texts.insert(nt),
            "{id} duplicates another gold text"
        );
    }
}

#[test]
fn gold_items_match_analyze_document_and_pass_validator() {
    let gold = read_jsonl(GOLD_PATH);
    let opts = Options::default();
    let mut failures = Vec::new();
    let mut total_frames = 0usize;
    let mut total_clusters = 0usize;
    let mut total_zeros = 0usize;

    for item in &gold {
        let id = item["id"].as_str().unwrap();
        let text = item["text"].as_str().unwrap();
        let chars: Vec<char> = text.chars().collect();
        let doc = analyze_document(text, &opts);

        assert_eq!(
            doc.segmentation.paragraphs.len(),
            1,
            "{id}: expected 1 paragraph"
        );
        assert_eq!(
            lang_str(doc.segmentation.paragraphs[0].lang),
            item["lang"].as_str().unwrap(),
            "{id}: paragraph lang differs from analyze_document"
        );

        let doc_tokens: Vec<String> = doc
            .tokens
            .iter()
            .map(|t| chars[t.span.start_char..t.span.end_char].iter().collect())
            .collect();
        let gold_tokens: Vec<String> = strs(item, "tokens")
            .unwrap()
            .into_iter()
            .map(str::to_owned)
            .collect();
        assert_eq!(
            doc_tokens, gold_tokens,
            "{id}: tokens differ from Layer 1/2 analyze_document"
        );

        let mut repo_sents: Vec<(usize, usize)> = Vec::new();
        for (i, tok) in doc.tokens.iter().enumerate() {
            match repo_sents.last_mut() {
                Some(last) if i > 0 && doc.tokens[i - 1].sentence == tok.sentence => {
                    last.1 = i + 1;
                }
                _ => repo_sents.push((i, i + 1)),
            }
        }
        let gold_sents: Vec<(usize, usize)> = item["sentences"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                (
                    s[0].as_u64().unwrap() as usize,
                    s[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        assert_eq!(
            repo_sents, gold_sents,
            "{id}: sentence ranges differ from Layer 1/2 analyze_document"
        );

        let gold_upos = strs(item, "upos").unwrap();
        for (tok_idx, (doc_tok, &g_upos)) in doc.tokens.iter().zip(&gold_upos).enumerate() {
            if let Some(decided) = doc_tok.upos {
                assert_eq!(
                    upos_str(decided),
                    g_upos,
                    "{id} token {} ({:?}): gold upos {g_upos} disagrees with Layer-2 decided {:?}",
                    tok_idx + 1,
                    doc_tokens[tok_idx],
                    decided
                );
            }
        }

        if let Err(err) = check_gold_item(item) {
            failures.push(format!("{id}: {err}"));
        }
        total_frames += item["frames"].as_array().unwrap().len();
        total_clusters += item["coref"].as_array().unwrap().len();
        total_zeros += item["zero_mentions"].as_array().unwrap().len();
    }

    assert!(
        failures.is_empty(),
        "gold validation failures ({}): {failures:#?}",
        failures.len()
    );
    assert!(
        total_frames >= 500,
        "expected rich frame coverage, got {total_frames}"
    );
    assert!(
        total_clusters >= 120,
        "expected rich coreference coverage, got {total_clusters}"
    );
    assert!(
        total_zeros >= 20,
        "expected pro-drop zero-mention coverage, got {total_zeros}"
    );
}

#[test]
fn validator_and_scorer_reject_bad_inputs_and_score_perturbations() {
    let gold = read_jsonl(GOLD_PATH);

    // 1. Reference scorer achieves 1.0 on gold-vs-gold and drops predictably under perturbations.
    let perfect = score_predictions(&gold, &gold);
    assert_eq!(perfect.uas, 1.0);
    assert_eq!(perfect.las, 1.0);
    assert_eq!(perfect.upos_acc, 1.0);
    assert_eq!(perfect.frame_f1, 1.0);
    assert_eq!(perfect.b3_f1, 1.0);

    let mut rel_perturbed = gold.clone();
    for item in rel_perturbed.iter_mut().step_by(2) {
        for rel in item["deprels"].as_array_mut().unwrap() {
            if rel.as_str() != Some("root") {
                *rel = serde_json::json!("dep");
            }
        }
    }
    let rel_scores = score_predictions(&gold, &rel_perturbed);
    assert_eq!(
        rel_scores.uas, 1.0,
        "deprel-only perturbation must keep UAS=1.0"
    );
    assert!(
        rel_scores.las < 0.70,
        "LAS must drop under deprel perturbation"
    );
    assert_eq!(rel_scores.upos_acc, 1.0);

    let mut all_perturbed = gold.clone();
    for item in all_perturbed.iter_mut().step_by(2) {
        let n = item["tokens"].as_array().unwrap().len();
        item["upos"] = serde_json::json!(vec!["NOUN"; n]);
        let mut flat_heads = vec![1usize; n];
        flat_heads[0] = 0;
        item["heads"] = serde_json::json!(flat_heads);
        let mut flat_rels = vec!["dep"; n];
        flat_rels[0] = "root";
        item["deprels"] = serde_json::json!(flat_rels);
        item["frames"] = serde_json::json!([]);
        item["coref"] = serde_json::json!([]);
    }
    let pert_scores = score_predictions(&gold, &all_perturbed);
    assert!(pert_scores.uas > 0.4 && pert_scores.uas < 0.9);
    assert!(pert_scores.las < pert_scores.uas);
    assert!(pert_scores.upos_acc > 0.4 && pert_scores.upos_acc < 0.75);
    assert!(pert_scores.frame_f1 > 0.3 && pert_scores.frame_f1 < 0.8);
    assert!(pert_scores.b3_f1 > 0.3 && pert_scores.b3_f1 < 0.8);

    // 2. Negative validator tests on single-sentence (g001) and two-sentence (g026) gold items.
    let g001 = gold[0].clone();
    assert_eq!(check_gold_item(&g001), Ok(()));

    let mut bad_upos = g001.clone();
    bad_upos["upos"][0] = serde_json::json!("NN");
    assert_eq!(
        check_gold_item(&bad_upos).unwrap_err(),
        "token 1: invalid upos \"NN\""
    );

    let mut subtype_rel = g001.clone();
    subtype_rel["deprels"][1] = serde_json::json!("nsubj:pass");
    assert_eq!(
        check_gold_item(&subtype_rel).unwrap_err(),
        "token 2: label \"nsubj:pass\" not in the scheme"
    );

    let mut punct_head = g001.clone();
    punct_head["heads"][0] = serde_json::json!(10); // token 10 is "。" (PUNCT)
    assert_eq!(
        check_gold_item(&punct_head).unwrap_err(),
        "token 1: attached to PUNCT head 10"
    );

    let mut root_disagree = g001.clone();
    root_disagree["deprels"][6] = serde_json::json!("dep"); // head is 0
    assert_eq!(
        check_gold_item(&root_disagree).unwrap_err(),
        "token 7: root label and head 0 disagree"
    );

    let mut two_roots = g001.clone();
    two_roots["heads"][1] = serde_json::json!(0);
    two_roots["deprels"][1] = serde_json::json!("root");
    assert_eq!(
        check_gold_item(&two_roots).unwrap_err(),
        "sentence has 2 roots"
    );

    let mut cycle = g001.clone();
    cycle["heads"][0] = serde_json::json!(2);
    cycle["heads"][1] = serde_json::json!(1);
    assert!(check_gold_item(&cycle).unwrap_err().contains("cycle"));

    let mut out_of_sent = g001.clone();
    out_of_sent["heads"][0] = serde_json::json!(99);
    assert_eq!(
        check_gold_item(&out_of_sent).unwrap_err(),
        "token 1: head 99 outside the sentence"
    );

    let mut bad_role = g001.clone();
    bad_role["frames"] =
        serde_json::json!([{"predicate": 7, "args": [{"role": "ARG9", "head": 2}]}]);
    assert_eq!(check_gold_item(&bad_role).unwrap_err(), "unknown role");

    let g026 = gold[25].clone(); // two sentences, has coref and zero_mentions
    assert_eq!(check_gold_item(&g026), Ok(()));

    let mut cross_sent_arg = g026.clone();
    cross_sent_arg["frames"] =
        serde_json::json!([{"predicate": 3, "args": [{"role": "ARG0", "head": 9}]}]);
    assert_eq!(
        check_gold_item(&cross_sent_arg).unwrap_err(),
        "argument head 9 is not in the sentence of predicate 3"
    );

    let mut singleton_cluster = g026.clone();
    singleton_cluster["coref"] = serde_json::json!([[[1, 2]]]);
    assert_eq!(
        check_gold_item(&singleton_cluster).unwrap_err(),
        "cluster with fewer than two mentions"
    );

    let mut bad_span = g026.clone();
    bad_span["coref"] = serde_json::json!([[[1, 2], [9, 99]]]);
    assert_eq!(
        check_gold_item(&bad_span).unwrap_err(),
        "mention [9,99] out of range"
    );

    let mut non_constituent = g026.clone();
    // Tokens [2, 3] = ["小姐", "借"] includes token 1 ("周" -> head 2) outside, wait:
    // 2->3 and 3->0, so only 3 has external head 0; whereas [1, 4] = ["周","小姐","借","了"] without obj 7 is constituent too;
    // [5, 6] = ["三本"(->7), "歷史"(->7)] has TWO external heads (both -> 7)!
    non_constituent["coref"] = serde_json::json!([[[5, 6], [9, 9]]]);
    assert_eq!(
        check_gold_item(&non_constituent).unwrap_err(),
        "mention [5,6] is not a single-headed syntactic constituent"
    );

    let mut bad_zero_role = g026.clone();
    bad_zero_role["zero_mentions"] =
        serde_json::json!([{"zero_at": 13, "role": "iobj", "antecedent": [5, 7]}]);
    assert_eq!(
        check_gold_item(&bad_zero_role).unwrap_err(),
        "invalid zero_mentions role \"iobj\""
    );

    let mut conflict_zero = g026;
    // Sentence 1 predicate 3 ("借") already has overt obj at token 7 ("書").
    conflict_zero["zero_mentions"] =
        serde_json::json!([{"zero_at": 3, "role": "obj", "antecedent": [5, 7]}]);
    assert_eq!(
        check_gold_item(&conflict_zero).unwrap_err(),
        "zero_mention (3, obj) conflicts with overt obj at token 7"
    );
}
