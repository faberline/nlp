use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};

use faberline_nlp::doc::{
    Document, FrameRole, Options, SyntaxToken, ZeroRole, analyze_document,
    analyze_document_all_layers,
};
use serde_json::Value;

fn assert_valid_syntax_trees(doc: &Document, label: &str) {
    assert!(
        !doc.syntax.is_empty(),
        "{label}: syntax should not be empty"
    );
    let n_toks = doc.tokens.len();
    let mut by_sent: BTreeMap<usize, Vec<&SyntaxToken>> = BTreeMap::new();
    for st in &doc.syntax {
        assert!(st.token < n_toks, "{label}: token index out of bounds");
        by_sent.entry(st.sentence).or_default().push(st);
    }

    for (sent_idx, sent) in doc.segmentation.sentences.iter().enumerate() {
        let sent_toks: Vec<usize> = doc
            .tokens
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (t.sentence == sent_idx).then_some(i))
            .collect();
        if sent_toks.is_empty() {
            continue;
        }
        let st_list = by_sent.get(&sent_idx).unwrap_or_else(|| {
            panic!(
                "{label}: sentence {sent_idx} ({:?}) has 0 syntax tokens",
                sent.lang
            )
        });
        let mut head_map: HashMap<usize, Option<usize>> = HashMap::new();
        let mut roots = 0usize;
        for st in st_list {
            assert_eq!(st.sentence, sent_idx, "{label}: mismatched sentence index");
            assert!(
                doc.tokens[st.token].upos.is_some(),
                "{label}: token {} has no upos after all-layers",
                st.token
            );
            if st.head.is_none() {
                assert_eq!(
                    st.deprel, "root",
                    "{label}: head=None must have deprel=root"
                );
                roots += 1;
            }
            head_map.insert(st.token, st.head);
        }
        assert_eq!(
            roots, 1,
            "{label}: sentence {sent_idx} has {roots} roots (expected 1)"
        );

        // No orphan arcs and no cycles
        for st in st_list {
            let mut cur = st.head;
            let mut steps = 0usize;
            while let Some(h) = cur {
                assert!(
                    head_map.contains_key(&h),
                    "{label}: orphan arc {} -> {} in sentence {sent_idx}",
                    st.token,
                    h
                );
                assert_ne!(h, st.token, "{label}: self-loop on {}", st.token);
                steps += 1;
                assert!(
                    steps <= st_list.len(),
                    "{label}: cycle detected in sentence {sent_idx}"
                );
                cur = head_map[&h];
            }
        }
    }
}

#[test]
fn backwards_compat_default_options_omits_layers_4_to_6() {
    let opts_json = serde_json::to_string(&Options::default()).unwrap();
    assert_eq!(opts_json, "{}");

    let text = "老木匠把桌上的木屑掃進竹籃裡。然後他打開了窗戶。";
    let doc = analyze_document(text, &Options::default());
    assert!(doc.syntax.is_empty());
    assert!(doc.frames.is_empty());
    assert!(doc.coref.is_empty());
    assert!(doc.zero_mentions.is_empty());

    let val = serde_json::to_value(&doc).unwrap();
    let obj = val.as_object().unwrap();
    assert!(!obj.contains_key("syntax"));
    assert!(!obj.contains_key("frames"));
    assert!(!obj.contains_key("coref"));
    assert!(!obj.contains_key("zero_mentions"));
}

#[test]
fn all_layers_multilingual_and_cli_roundtrip() {
    let cases: &[(&str, &str, bool, bool)] = &[
        (
            "zh-Hant",
            "因為林默走進房間，所以他打開了燈。接著林默拿起桌上的鑰匙，出門了。",
            true,
            true,
        ),
        (
            "zh-Hans",
            "李明走进书房，打开了台灯。因为外面下雨，所以他拿了一把伞，出门了。",
            true,
            true,
        ),
        (
            "en",
            "Alice opened the wooden door because she heard a noise. Then Alice walked into the garden.",
            true,
            false,
        ),
        ("ja", "太郎が部屋に入った。そして本を読んだ。", false, true),
        (
            "ko",
            "민수가 방에 들어갔다. 그리고 책을 읽었다.",
            false,
            true,
        ),
        (
            "mixed-zh-en",
            "Alice把報告交給了主管，然後她打開了laptop。",
            true,
            false,
        ),
        (
            "mixed-ja-en",
            "太郎は新しいPythonの本を買った。",
            false,
            false,
        ),
        ("mixed-ko-en", "민수는 새로운 Rust 책을 샀다.", false, false),
    ];

    for &(label, text, expect_coref, expect_zero) in cases {
        let opts = Options {
            names: vec!["林默".to_owned(), "李明".to_owned(), "Alice".to_owned()],
            ..Options::default()
        };
        let doc = analyze_document_all_layers(text, &opts);
        assert_valid_syntax_trees(&doc, label);
        assert!(!doc.frames.is_empty(), "{label}: expected non-empty frames");
        if expect_coref {
            assert!(
                !doc.coref.is_empty(),
                "{label}: expected non-empty coref clusters"
            );
        }
        if expect_zero {
            assert!(
                !doc.zero_mentions.is_empty(),
                "{label}: expected non-empty zero_mentions for CJK pro-drop"
            );
        }
        if label.starts_with("zh-") || label == "en" {
            assert!(
                !doc.relations.is_empty(),
                "{label}: expected non-empty discourse relations"
            );
        }
        if label.starts_with("zh-") || label == "en" || label == "mixed-zh-en" {
            assert!(
                !doc.entities.is_empty(),
                "{label}: expected non-empty entities"
            );
        }

        // JSON serde round-trip
        let json = serde_json::to_string(&doc).unwrap();
        let decoded: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(doc, decoded, "{label}: JSON round-trip mismatch");
    }

    // Verify `nlp_doc --all-layers` CLI binary
    let cli_text = "因為林默走進房間，所以他打開了燈。接著拿起桌上的鑰匙。";
    let mut child = Command::new(env!("CARGO_BIN_EXE_nlp_doc"))
        .args(["--all-layers", "--name", "林默"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn nlp_doc");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(cli_text.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "nlp_doc --all-layers failed");
    let cli_doc: Document = serde_json::from_slice(&out.stdout).unwrap();
    assert_valid_syntax_trees(&cli_doc, "nlp_doc --all-layers");
    assert!(!cli_doc.frames.is_empty());
    assert!(!cli_doc.coref.is_empty());
    assert!(!cli_doc.zero_mentions.is_empty());
    assert!(!cli_doc.relations.is_empty());
    assert!(!cli_doc.entities.is_empty());
}

fn greedy_ceafe(gold: &[BTreeSet<(usize, usize)>], pred: &[BTreeSet<(usize, usize)>]) -> f64 {
    if gold.is_empty() || pred.is_empty() {
        return 0.0;
    }
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (gi, g) in gold.iter().enumerate() {
        for (pi, p) in pred.iter().enumerate() {
            let inter = g.intersection(p).count() as f64;
            if inter > 0.0 {
                let phi = 2.0 * inter / (g.len() + p.len()) as f64;
                pairs.push((phi, gi, pi));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut used_g = vec![false; gold.len()];
    let mut used_p = vec![false; pred.len()];
    let mut total = 0.0;
    for (phi, gi, pi) in pairs {
        if !used_g[gi] && !used_p[pi] {
            used_g[gi] = true;
            used_p[pi] = true;
            total += phi;
        }
    }
    total
}

#[test]
fn end_to_end_cpu_eval_gold_300() {
    let raw = fs::read_to_string("tests/data/gold_layers4_6_v1.jsonl").unwrap();

    let mut total_tok = 0usize;
    let mut upos_ok = 0usize;
    let mut uas_ok = 0usize;
    let mut las_ok = 0usize;

    let mut frame_tp = 0usize;
    let mut frame_pred = 0usize;
    let mut frame_gold = 0usize;

    let mut muc_r_num = 0.0f64;
    let mut muc_r_den = 0.0f64;
    let mut muc_p_num = 0.0f64;
    let mut muc_p_den = 0.0f64;
    let mut b3_r_num = 0.0f64;
    let mut b3_r_den = 0.0f64;
    let mut b3_p_num = 0.0f64;
    let mut b3_p_den = 0.0f64;
    let mut ceaf_num = 0.0f64;
    let mut ceaf_r_den = 0.0f64;
    let mut ceaf_p_den = 0.0f64;

    let mut zero_tp = 0usize;
    let mut zero_pred = 0usize;
    let mut zero_gold = 0usize;

    let mut item_count = 0usize;
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        item_count += 1;
        let v: Value = serde_json::from_str(line).unwrap();
        let id = v["id"].as_str().unwrap();
        let text = v["text"].as_str().unwrap();

        let doc = analyze_document_all_layers(text, &Options::default());
        assert_valid_syntax_trees(&doc, id);

        let gold_upos: Vec<&str> = v["upos"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();
        let gold_heads: Vec<usize> = v["heads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let gold_deprels: Vec<&str> = v["deprels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();

        assert_eq!(
            doc.tokens.len(),
            gold_upos.len(),
            "{id}: token count mismatch"
        );
        assert_eq!(
            doc.syntax.len(),
            gold_heads.len(),
            "{id}: syntax count mismatch"
        );

        for i in 0..gold_upos.len() {
            total_tok += 1;
            let pred_u = serde_json::to_value(doc.tokens[i].upos.unwrap())
                .unwrap()
                .as_str()
                .unwrap()
                .to_owned();
            if pred_u == gold_upos[i] {
                upos_ok += 1;
            }
            let st = &doc.syntax[i];
            let pred_h = st.head.map_or(0, |h| h + 1);
            if pred_h == gold_heads[i] {
                uas_ok += 1;
                if st.deprel == gold_deprels[i] {
                    las_ok += 1;
                }
            }
        }

        // Layer 5: Frames (gold uses 1-based token indices)
        let mut g_frames: HashSet<(usize, &'static str, usize)> = HashSet::new();
        for gf in v["frames"].as_array().unwrap() {
            let pred = gf["predicate"].as_u64().unwrap() as usize;
            for ga in gf["args"].as_array().unwrap() {
                let role = match ga["role"].as_str().unwrap() {
                    "ARG0" => "ARG0",
                    "ARG1" => "ARG1",
                    _ => "ARGM",
                };
                let head = ga["head"].as_u64().unwrap() as usize;
                g_frames.insert((pred, role, head));
            }
        }
        let mut p_frames: HashSet<(usize, &'static str, usize)> = HashSet::new();
        for pf in &doc.frames {
            for pa in &pf.args {
                let role = match pa.role {
                    FrameRole::Arg0 => "ARG0",
                    FrameRole::Arg1 => "ARG1",
                    FrameRole::Argm => "ARGM",
                };
                p_frames.insert((pf.predicate + 1, role, pa.head + 1));
            }
        }
        frame_gold += g_frames.len();
        frame_pred += p_frames.len();
        frame_tp += g_frames.intersection(&p_frames).count();

        // Layer 6: Coref (gold uses 1-based inclusive [start, end])
        let gold_clusters: Vec<BTreeSet<(usize, usize)>> = v["coref"]
            .as_array()
            .unwrap()
            .iter()
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
                    .collect()
            })
            .collect();
        let pred_clusters: Vec<BTreeSet<(usize, usize)>> = doc
            .coref
            .iter()
            .map(|cl| {
                cl.iter()
                    .map(|m| (m.start_token + 1, m.end_token + 1))
                    .collect()
            })
            .collect();

        let mut g_map: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (gi, g) in gold_clusters.iter().enumerate() {
            for &m in g {
                g_map.insert(m, gi);
            }
        }
        let mut p_map: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (pi, p) in pred_clusters.iter().enumerate() {
            for &m in p {
                p_map.insert(m, pi);
            }
        }

        for g in &gold_clusters {
            muc_r_den += (g.len() - 1) as f64;
            let mut seen_p = HashSet::new();
            let mut unaligned = 0usize;
            for m in g {
                if let Some(&pi) = p_map.get(m) {
                    seen_p.insert(pi);
                } else {
                    unaligned += 1;
                }
            }
            muc_r_num += (g.len() - (seen_p.len() + unaligned)) as f64;
        }
        for p in &pred_clusters {
            muc_p_den += (p.len() - 1) as f64;
            let mut seen_g = HashSet::new();
            let mut unaligned = 0usize;
            for m in p {
                if let Some(&gi) = g_map.get(m) {
                    seen_g.insert(gi);
                } else {
                    unaligned += 1;
                }
            }
            muc_p_num += (p.len() - (seen_g.len() + unaligned)) as f64;
        }
        for g in &gold_clusters {
            for m in g {
                b3_r_den += 1.0;
                if let Some(&pi) = p_map.get(m) {
                    let inter = g.intersection(&pred_clusters[pi]).count() as f64;
                    b3_r_num += inter / g.len() as f64;
                }
            }
        }
        for p in &pred_clusters {
            for m in p {
                b3_p_den += 1.0;
                if let Some(&gi) = g_map.get(m) {
                    let inter = p.intersection(&gold_clusters[gi]).count() as f64;
                    b3_p_num += inter / p.len() as f64;
                }
            }
        }
        ceaf_r_den += gold_clusters.len() as f64;
        ceaf_p_den += pred_clusters.len() as f64;
        ceaf_num += greedy_ceafe(&gold_clusters, &pred_clusters);

        // Layer 6: Zero mentions
        let mut g_zeros: HashSet<(usize, &'static str, usize, usize)> = HashSet::new();
        for gz in v["zero_mentions"].as_array().unwrap() {
            let zero_at = gz["zero_at"].as_u64().unwrap() as usize;
            let role = if gz["role"].as_str().unwrap() == "nsubj" {
                "nsubj"
            } else {
                "obj"
            };
            let ant = gz["antecedent"].as_array().unwrap();
            let ant_s = ant[0].as_u64().unwrap() as usize;
            let ant_e = ant[1].as_u64().unwrap() as usize;
            g_zeros.insert((zero_at, role, ant_s, ant_e));
        }
        let mut p_zeros: HashSet<(usize, &'static str, usize, usize)> = HashSet::new();
        for pz in &doc.zero_mentions {
            let role = match pz.role {
                ZeroRole::Nsubj => "nsubj",
                ZeroRole::Obj => "obj",
            };
            p_zeros.insert((
                pz.zero_at + 1,
                role,
                pz.antecedent.start_token + 1,
                pz.antecedent.end_token + 1,
            ));
        }
        zero_gold += g_zeros.len();
        zero_pred += p_zeros.len();
        zero_tp += g_zeros.intersection(&p_zeros).count();
    }

    assert_eq!(item_count, 300);

    let upos_acc = upos_ok as f64 / total_tok as f64;
    let uas = uas_ok as f64 / total_tok as f64;
    let las = las_ok as f64 / total_tok as f64;

    let frame_p = frame_tp as f64 / frame_pred.max(1) as f64;
    let frame_r = frame_tp as f64 / frame_gold.max(1) as f64;
    let frame_f1 = if frame_p + frame_r > 0.0 {
        2.0 * frame_p * frame_r / (frame_p + frame_r)
    } else {
        0.0
    };

    let f1_of = |p: f64, r: f64| {
        if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        }
    };
    let muc_f1 = f1_of(
        muc_p_num / muc_p_den.max(1.0),
        muc_r_num / muc_r_den.max(1.0),
    );
    let b3_f1 = f1_of(b3_p_num / b3_p_den.max(1.0), b3_r_num / b3_r_den.max(1.0));
    let ceaf_f1 = f1_of(
        ceaf_num / ceaf_p_den.max(1.0),
        ceaf_num / ceaf_r_den.max(1.0),
    );
    let coref_f1 = (muc_f1 + b3_f1 + ceaf_f1) / 3.0;

    let zero_p = zero_tp as f64 / zero_pred.max(1) as f64;
    let zero_r = zero_tp as f64 / zero_gold.max(1) as f64;
    let zero_f1 = f1_of(zero_p, zero_r);

    eprintln!(
        "[CPU E2E gold_300] UPOS={upos_acc:.4} UAS={uas:.4} LAS={las:.4} | \
         FrameF1={frame_f1:.4} (tp={frame_tp}, pred={frame_pred}, gold={frame_gold}) | \
         CorefF1={coref_f1:.4} (MUC={muc_f1:.4}, B3={b3_f1:.4}, CEAFe={ceaf_f1:.4}) | \
         ZeroMentionF1={zero_f1:.4} (tp={zero_tp}, pred={zero_pred}, gold={zero_gold})"
    );

    assert!(upos_acc >= 0.92, "UPOS too low: {upos_acc:.4}");
    assert!(uas >= 0.75, "UAS too low: {uas:.4}");
    assert!(las >= 0.70, "LAS too low: {las:.4}");
    assert!(frame_f1 >= 0.60, "FrameF1 too low: {frame_f1:.4}");
    assert!(coref_f1 >= 0.75, "CorefF1 too low: {coref_f1:.4}");
    assert!(zero_f1 >= 0.25, "ZeroMentionF1 too low: {zero_f1:.4}");
}
