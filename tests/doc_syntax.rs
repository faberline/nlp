//! Layer 4 (`doc-syntax-v1`) unit and synthetic-model integration tests.
//! Covers default `analyze_document` invariance, `Score(u16)` quantization & JSON round-trip,
//! single-root Chu-Liu/Edmonds (including cyclic/multi-root traps and brute-force tree oracle),
//! packed `[len, 17 + max_len + 148]` decoding, UPOS upgrade, `low_confidence`, `not_applicable`,
//! `no_model`, standalone `SubwordTokenizer` under `coreml` or `onnx`, and the `nlp_doc` CLI.

use faberline_nlp::Span;
use faberline_nlp::doc::model::{
    Backend, Batch, Model, ModelError, ModelInfo, Output, Subwords, Tensor,
};
use faberline_nlp::doc::syntax::{
    DEPRELS, ROOT_DEPREL_INDEX, SYNTAX_MODEL_ID, SYNTAX_THRESHOLD_MILLI, Score, SyntaxToken,
    UPOS_TAGS, annotate_with_encoder, chu_liu_edmonds,
};
use faberline_nlp::doc::{
    Document, Options, Reason, Unknown, Upos, analyze_document, analyze_document_with_model,
};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const GENRE_LANG_SAMPLES: &[(&str, &str)] = &[
    ("novel_zh", "小甲推開木門。阿乙坐在窗邊看書。"),
    (
        "novel_en",
        "Alice opened the wooden door. Bob waited quietly.",
    ),
    ("novel_mixed", "小甲打開 laptop，然後寫了一封 message。"),
    ("paper_zh", "本研究量測兩組樣本。結果顯示誤差穩定。"),
    ("paper_en", "We evaluate the parser on three datasets."),
    ("paper_mixed", "我們訓練 transformer 模型，並比較 latency。"),
    ("casual_zh", "今天天氣超好！我們去買咖啡吧。"),
    ("casual_en", "grabbed coffee this morning, feeling great!"),
    ("casual_mixed", "剛剛跑完 benchmark，結果超 fast！"),
];

fn assert_valid_single_root_tree(heads: &[usize]) {
    let m = heads.len();
    assert!(m >= 1);
    let roots = heads.iter().filter(|&&h| h == 0).count();
    assert_eq!(roots, 1, "expected exactly 1 root in {heads:?}");
    for (i, &h) in heads.iter().enumerate() {
        let u = i + 1;
        assert!(h <= m, "head {h} out of bounds 0..={m}");
        assert_ne!(h, u, "self-loop at token {u}: {heads:?}");
        let mut cur = u;
        let mut steps = 0;
        while cur != 0 {
            cur = heads[cur - 1];
            steps += 1;
            assert!(
                steps <= m,
                "cycle detected starting from token {u}: {heads:?}"
            );
        }
    }
}

fn is_valid_single_root_tree(heads: &[usize]) -> bool {
    let m = heads.len();
    if heads.iter().filter(|&&h| h == 0).count() != 1 {
        return false;
    }
    for (i, &h) in heads.iter().enumerate() {
        let u = i + 1;
        if h > m || h == u {
            return false;
        }
        let mut cur = u;
        let mut steps = 0;
        while cur != 0 {
            cur = heads[cur - 1];
            steps += 1;
            if steps > m {
                return false;
            }
        }
    }
    true
}

fn tree_weight(arc_scores: &[Vec<f32>], heads: &[usize]) -> f64 {
    heads
        .iter()
        .enumerate()
        .map(|(i, &h)| f64::from(arc_scores[i][h]))
        .sum()
}

/// Enumerates every valid single-root tree over `1..=m` and returns the maximum total score.
fn brute_force_best_weight(arc_scores: &[Vec<f32>]) -> f64 {
    let m = arc_scores.len();
    let total = (m + 1).pow(m as u32);
    let mut best = f64::NEG_INFINITY;
    let mut cand = vec![0usize; m];
    for mut code in 0..total {
        for slot in &mut cand {
            *slot = code % (m + 1);
            code /= m + 1;
        }
        if is_valid_single_root_tree(&cand) {
            best = best.max(tree_weight(arc_scores, &cand));
        }
    }
    best
}

struct Xorshift64(u64);

impl Xorshift64 {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        let unit = ((x >> 40) as f32) / ((1u32 << 24) as f32);
        (unit - 0.5) * 20.0
    }
}

#[test]
fn constants_match_contract() {
    assert_eq!(SYNTAX_MODEL_ID, "doc-syntax-v1");
    assert_eq!(UPOS_TAGS.len(), 17);
    assert_eq!(DEPRELS.len(), 37);
    assert_eq!(DEPRELS[ROOT_DEPREL_INDEX], "root");
    for w in DEPRELS.windows(2) {
        assert!(w[0] < w[1], "DEPRELS not strictly alphabetical: {w:?}");
    }
}

#[test]
fn score_preserves_eq_and_round_trips_all_thousandths() {
    for milli in 0u16..=1000 {
        let s = Score(milli);
        let json = serde_json::to_string(&s).expect("serialize");
        let back: Score = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, s, "round-trip failed for milli={milli} ({json})");
    }
    assert_eq!(serde_json::to_string(&Score(0)).unwrap(), "0.0");
    assert_eq!(serde_json::to_string(&Score(350)).unwrap(), "0.35");
    assert_eq!(serde_json::to_string(&Score(875)).unwrap(), "0.875");
    assert_eq!(serde_json::to_string(&Score(1000)).unwrap(), "1.0");
    assert!(serde_json::from_str::<Score>("-0.01").is_err());
    assert!(serde_json::from_str::<Score>("1.001").is_err());
}

#[test]
fn default_analyze_document_omits_syntax_and_keeps_exact_schema() {
    for &(name, text) in GENRE_LANG_SAMPLES {
        let doc = analyze_document(text, &Options::default());
        assert!(doc.syntax.is_empty(), "{name}");
        assert!(
            !doc.unknown.iter().any(|u| u.layer == "syntax"),
            "{name}: unexpected syntax unknown"
        );
        let raw = serde_json::to_string(&doc).expect("json");
        assert!(
            !raw.contains("\"syntax\""),
            "{name}: syntax must be omitted"
        );
        let back: Document = serde_json::from_str(&raw).expect("round-trip");
        assert_eq!(back, doc, "{name}");
    }
}

#[test]
fn chu_liu_edmonds_breaks_cycles_and_enforces_single_root() {
    assert_eq!(chu_liu_edmonds(&[]), Vec::<usize>::new());
    assert_eq!(chu_liu_edmonds(&[vec![-5.0, 99.0]]), vec![0]);

    // Classic 2-cycle trap:
    // Greedy picks 2->1 (10.0) and 1->2 (10.0) -> cycle {1, 2} with 0 roots!
    // Root options:
    // - Tree A: 0->1 (4.0) + 1->2 (10.0) = 14.0
    // - Tree B: 0->2 (5.0) + 2->1 (10.0) = 15.0  <-- true optimum!
    let two_cycle = vec![
        vec![4.0, 0.0, 10.0], // u = 1: [0->1: 4, 1->1: _, 2->1: 10]
        vec![5.0, 10.0, 0.0], // u = 2: [0->2: 5, 1->2: 10, 2->2: _]
    ];
    assert_eq!(chu_liu_edmonds(&two_cycle), vec![2, 0]);

    // 3-cycle + multi-root temptation:
    // Greedy without single-root constraint or cycle breaking would either form 1->2->3->1
    // or attach multiple nodes to root 0 (both 0->1 = 20 and 0->2 = 19 are huge!).
    let multi_root_and_cycle = vec![
        vec![20.0, 0.0, 2.0, 9.0], // u = 1: 0->1=20, 2->1=2, 3->1=9
        vec![19.0, 8.0, 0.0, 1.0], // u = 2: 0->2=19, 1->2=8, 3->2=1
        vec![18.0, 1.0, 8.0, 0.0], // u = 3: 0->3=18, 1->3=1, 2->3=8
    ];
    let heads = chu_liu_edmonds(&multi_root_and_cycle);
    assert_valid_single_root_tree(&heads);
    // Best single-root tree: 0->1 (20) + 1->2 (8) + 2->3 (8) = 36!
    assert_eq!(heads, vec![0, 1, 2]);

    // Extreme values, NaN, infinities, and all-zero ties must still produce a valid 1-root tree.
    let weird = vec![
        vec![0.0, f32::NAN, 0.0, 0.0],
        vec![0.0, 0.0, f32::INFINITY, 0.0],
        vec![0.0, 0.0, 0.0, f32::NEG_INFINITY],
    ];
    let weird_heads = chu_liu_edmonds(&weird);
    assert_valid_single_root_tree(&weird_heads);
}

#[test]
fn chu_liu_edmonds_matches_brute_force_cayley_oracle() {
    let mut rng = Xorshift64(0xDEC0_DED5_2026_1111);
    for m in 2..=5 {
        let trials = if m <= 4 { 60 } else { 25 };
        for trial in 0..trials {
            let mut mat = vec![vec![0.0f32; m + 1]; m];
            for row in &mut mat {
                for slot in row.iter_mut() {
                    *slot = rng.next_f32();
                }
                // Deliberately boost root score half the time to stress single-root constraint.
                if trial % 2 == 0 {
                    row[0] += 15.0;
                }
            }
            let got = chu_liu_edmonds(&mat);
            assert_valid_single_root_tree(&got);
            let got_w = tree_weight(&mat, &got);
            let best_w = brute_force_best_weight(&mat);
            assert!(
                (got_w - best_w).abs() <= 1e-4,
                "m={m} trial={trial}: got {got:?} weight {got_w}, best {best_w}"
            );
        }
    }
}

type FillFn = dyn Fn(&[i64], usize, &mut [f32]) + Send + Sync;

/// Synthetic model that builds deterministic packed `[seq_len, 17 + max_len + 148]` logits.
struct SyntheticSyntaxModel {
    info: ModelInfo,
    /// Configured per-call scripted builder: given `seq_len`, fills `seq_len * cols`.
    fill: Box<FillFn>,
}

impl SyntheticSyntaxModel {
    fn new(
        max_len: usize,
        fill: impl Fn(&[i64], usize, &mut [f32]) + Send + Sync + 'static,
    ) -> Self {
        Self {
            info: ModelInfo {
                id: SYNTAX_MODEL_ID.to_owned(),
                layers: vec!["syntax".to_owned(), "pos".to_owned()],
                licence: "CC0-1.0".to_owned(),
                max_len,
                pad_id: 0,
                files: Vec::new(),
            },
            fill: Box::new(fill),
        }
    }
}

impl Model for SyntheticSyntaxModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn run(&self, batch: &Batch) -> Result<Output, ModelError> {
        let max = self.info.max_len;
        let cols = 17 + max + 4 * 37;
        let mut logits = Vec::with_capacity(batch.ids.len());
        for ids in &batch.ids {
            if ids.len() > max {
                return Err(ModelError::TooLong {
                    len: ids.len(),
                    max,
                });
            }
            let mut data = vec![0.0f32; ids.len() * cols];
            (self.fill)(ids, cols, &mut data);
            logits.push(Tensor {
                name: "logits".to_owned(),
                shape: vec![ids.len(), cols],
                data,
            });
        }
        Ok(Output { logits })
    }
}

/// Character-level encoder with `[CLS]` (id 2, empty span) and `[SEP]` (id 3, empty span).
fn char_subword_encode(sentence: &str) -> Result<Subwords, ModelError> {
    let mut ids = vec![2i64];
    let mut spans = vec![Span {
        start_char: 0,
        end_char: 0,
    }];
    for (idx, ch) in sentence.chars().enumerate() {
        if !ch.is_whitespace() {
            ids.push(4 + i64::from(ch as u32 % 200));
            spans.push(Span {
                start_char: idx,
                end_char: idx + 1,
            });
        }
    }
    ids.push(3);
    spans.push(Span {
        start_char: 0,
        end_char: 0,
    });
    Ok(Subwords { ids, spans })
}

#[test]
fn decodes_packed_421_and_229_tensors_across_genres_and_languages() {
    for max_len in [256usize, 64usize] {
        // Confident chain tree: token 1 is root (Verb), every subsequent token attaches to token 1
        // with "obj" (index 28) — even though we set "root" (index 34) relation logit to +50.0 to
        // prove `v > 0` excludes index 34!
        let model = SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
            let rel_off = 17 + max_len;
            for s in 1..ids.len().saturating_sub(1) {
                let row = &mut data[s * cols..(s + 1) * cols];
                // Confident UPOS = Noun (index 7), except s == 1 -> Verb (index 15).
                let upos_ix = if s == 1 { 15 } else { 7 };
                row[upos_ix] = 12.0;
                // Arc logits: s == 1 attaches to [CLS] (0); s > 1 attaches to subword 1.
                if s == 1 {
                    row[17] = 14.0;
                } else {
                    row[17 + 1] = 14.0;
                }
                // Relation logits: make "root" (34) huge (+50) and "obj" (28) second (+12) via
                // both additive and biaffine terms so we test the full formula!
                row[rel_off + 34] = 50.0;
                row[rel_off + 28] = 6.0; // r_dep[28] = 6
                row[rel_off + 37 + 28] = 2.0; // r_head[28] = 2
                row[rel_off + 74 + 28] = 2.0; // r_bi_dep[28] = 2
                row[rel_off + 111 + 28] = 2.0; // r_bi_head[28] = 2 -> 6 + 2 + 4 = 12.0
            }
        });

        for &(name, text) in GENRE_LANG_SAMPLES {
            let mut doc = analyze_document(text, &Options::default());
            annotate_with_encoder(&mut doc, text, &model, char_subword_encode).expect("annotate");

            assert_eq!(
                doc.syntax.len(),
                doc.tokens.len(),
                "{name} (max_len={max_len})"
            );
            // Every token now has a decided UPOS, and zero "pos" unknowns remain!
            assert!(
                doc.tokens.iter().all(|t| t.upos.is_some()),
                "{name}: undecided UPOS left"
            );
            assert!(
                !doc.unknown.iter().any(|u| u.layer == "pos"),
                "{name}: pos unknown not cleared"
            );

            // Check per-sentence root count, global token indexing, and exclusion of "root" on v>0.
            for sent_idx in 0..doc.segmentation.sentences.len() {
                let sent_syn: Vec<&SyntaxToken> = doc
                    .syntax
                    .iter()
                    .filter(|s| s.sentence == sent_idx)
                    .collect();
                assert!(!sent_syn.is_empty(), "{name} sent {sent_idx}");
                let roots: Vec<&&SyntaxToken> =
                    sent_syn.iter().filter(|s| s.head.is_none()).collect();
                assert_eq!(roots.len(), 1, "{name} sent {sent_idx}");
                assert_eq!(roots[0].deprel, "root");
                assert!(roots[0].score.milli() >= SYNTAX_THRESHOLD_MILLI);

                let root_global = roots[0].token;
                for st in &sent_syn {
                    assert_eq!(st.span, doc.tokens[st.token].span);
                    if st.token != root_global {
                        assert_eq!(st.head, Some(root_global));
                        assert_eq!(st.deprel, "obj");
                    }
                }
            }

            // Check JSON field ordering: "entities" < "syntax" < "relations" < "unknown".
            let json = serde_json::to_string(&doc).expect("json");
            let p_ent = json.find("\"entities\"").unwrap();
            let p_syn = json.find("\"syntax\"").unwrap();
            let p_rel = json.find("\"relations\"").unwrap();
            let p_unk = json.find("\"unknown\"").unwrap();
            assert!(p_ent < p_syn && p_syn < p_rel && p_rel < p_unk, "{json}");

            let round_trip: Document = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(round_trip, doc);
        }
    }
}

#[test]
fn low_confidence_syntax_and_upos_emit_low_confidence_unknowns() {
    let max_len = 64;
    // All-zero logits:
    // - For UPOS (17 classes): max prob = 1/17 = 0.0588 < 0.50 -> undecided tokens get "pos" LowConfidence!
    // - Layer-2 decided tokens (like "。" PUNCT or "他" PRON) MUST keep their Layer-2 UPOS!
    // - For syntax (m >= 2 tokens): p_arc = 1/(m+1) <= 1/3 = 0.333 < 0.35 -> "syntax" LowConfidence!
    let zero_model = SyntheticSyntaxModel::new(max_len, |_ids, _cols, _data| {});
    let text = "他推開門。";
    let mut doc = analyze_document(text, &Options::default());
    // In Layer 2, "他" is PRON and "。" is PUNCT; content words have upos == None.
    assert_eq!(doc.tokens[0].upos, Some(Upos::Pron));
    assert_eq!(doc.tokens.last().unwrap().upos, Some(Upos::Punct));
    assert!(doc.tokens.iter().any(|t| t.upos.is_none()));

    annotate_with_encoder(&mut doc, text, &zero_model, char_subword_encode).expect("annotate");

    assert!(
        doc.syntax.is_empty(),
        "all-zero model should be low_confidence"
    );
    assert_eq!(doc.tokens[0].upos, Some(Upos::Pron));
    assert_eq!(doc.tokens.last().unwrap().upos, Some(Upos::Punct));

    for tok in &doc.tokens {
        assert!(
            doc.unknown.contains(&Unknown {
                layer: "syntax".to_owned(),
                span: tok.span,
                reason: Reason::LowConfidence,
            }),
            "missing syntax LowConfidence for {:?}",
            tok.span
        );
        if tok.upos.is_none() {
            assert!(
                doc.unknown.contains(&Unknown {
                    layer: "pos".to_owned(),
                    span: tok.span,
                    reason: Reason::LowConfidence,
                }),
                "missing pos LowConfidence for {:?}",
                tok.span
            );
        }
    }
}

#[test]
fn too_long_sentence_emits_not_applicable_and_keeps_short_sentence() {
    let max_len = 8; // Very small cap so sentence 0 overflows while sentence 1 fits!
    let model = SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            row[15] = 10.0; // Verb
            if s == 1 {
                row[17] = 12.0;
            } else {
                row[17 + 1] = 12.0;
            }
            row[17 + max_len + 26] = 12.0; // "nsubj"
        }
    });

    let text = "小甲今天早上走進木屋看書。好！";
    let mut doc = analyze_document(text, &Options::default());
    assert_eq!(doc.segmentation.sentences.len(), 2);

    annotate_with_encoder(&mut doc, text, &model, char_subword_encode).expect("annotate");

    // Sentence 0 exceeded max_len=8 -> single NotApplicable unknown over sentence 0's span.
    assert!(doc.unknown.contains(&Unknown {
        layer: "syntax".to_owned(),
        span: doc.segmentation.sentences[0].span,
        reason: Reason::NotApplicable,
    }));
    // Sentence 1 ("好！", 2 chars + [CLS] + [SEP] = 4 <= 8) succeeded!
    assert!(doc.syntax.iter().all(|s| s.sentence == 1));
    assert_eq!(doc.syntax.len(), 2);
}

#[test]
fn unsupported_language_sentence_is_skipped_cleanly() {
    let max_len = 64;
    let model = SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            row[7] = 10.0;
            if s == 1 {
                row[17] = 12.0;
            } else {
                row[17 + 1] = 12.0;
            }
            row[17 + max_len + 32] = 12.0; // "punct"
        }
    });

    let text = "안녕하세요 감사합니다.\n\n小甲走了。";
    let mut doc = analyze_document(text, &Options::default());
    annotate_with_encoder(&mut doc, text, &model, char_subword_encode).expect("annotate");

    assert!(doc.syntax.iter().all(|s| s.sentence == 1));
    assert!(!doc.syntax.is_empty());
    assert!(
        !doc.unknown
            .iter()
            .any(|u| u.layer == "syntax" && u.span == doc.segmentation.sentences[0].span)
    );
}

#[test]
fn analyze_document_with_model_reports_no_model_and_propagates_integrity_errors() {
    let models_dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/models"));
    let text = "小甲走進木屋。We measured the data.\n\n안녕하세요.";
    // `models/` has `test-tiny`, not `doc-syntax-v1` (or `NoBackend` without backend feature).
    for backend in [Backend::Onnx, Backend::Coreml] {
        let doc =
            analyze_document_with_model(text, &Options::default(), &models_dir, backend).unwrap();
        assert!(doc.syntax.is_empty());
        // Supported sentences 0 and 1 get `no_model` for "syntax"; unsupported sentence 2 does not.
        assert!(doc.unknown.contains(&Unknown {
            layer: "syntax".to_owned(),
            span: doc.segmentation.sentences[0].span,
            reason: Reason::NoModel,
        }));
        assert!(doc.unknown.contains(&Unknown {
            layer: "syntax".to_owned(),
            span: doc.segmentation.sentences[1].span,
            reason: Reason::NoModel,
        }));
        assert!(
            !doc.unknown
                .iter()
                .any(|u| u.layer == "syntax" && u.span == doc.segmentation.sentences[2].span)
        );
    }

    // Missing directory also maps to `no_model`.
    let tmp = tempfile::tempdir().unwrap();
    let missing_doc = analyze_document_with_model(
        "好。",
        &Options::default(),
        &tmp.path().join("absent"),
        Backend::Onnx,
    )
    .unwrap();
    assert!(
        missing_doc
            .unknown
            .iter()
            .any(|u| { u.layer == "syntax" && u.reason == Reason::NoModel })
    );

    // Bad manifest JSON MUST return Err(ModelError::Manifest(_)) whenever the backend is compiled.
    std::fs::write(tmp.path().join("manifest.json"), "{ invalid json").unwrap();
    #[cfg(feature = "onnx")]
    {
        let err =
            analyze_document_with_model("好。", &Options::default(), tmp.path(), Backend::Onnx)
                .unwrap_err();
        assert!(matches!(err, ModelError::Manifest(_)), "{err:?}");
    }
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        let err =
            analyze_document_with_model("好。", &Options::default(), tmp.path(), Backend::Coreml)
                .unwrap_err();
        assert!(matches!(err, ModelError::Manifest(_)), "{err:?}");
    }
}

#[cfg(any(feature = "onnx", feature = "coreml"))]
mod real_subword_tokenizer_tests {
    use super::*;
    use faberline_nlp::doc::analyze_document_with_loaded_model;
    use faberline_nlp::doc::model::{SubwordTokenizer, Verified};

    /// Builds an in-memory HuggingFace `Tokenizer` JSON adding `[CLS]` (id 2) and `[SEP]` (id 3)
    /// on top of the repository's `models/test-tiny/tokenizer.json` byte-level BPE vocabulary.
    fn syntax_subword_tokenizer() -> SubwordTokenizer {
        let path = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/test-tiny/tokenizer.json"
        ));
        let mut tok_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

        // Shift byte symbols up by 2 to reserve 0=[PAD], 1=[UNK], 2=[CLS], 3=[SEP].
        let vocab = tok_json["model"]["vocab"].as_object_mut().unwrap();
        for (k, v) in vocab.iter_mut() {
            if k != "[PAD]" && k != "[UNK]" {
                *v = serde_json::json!(v.as_u64().unwrap() + 2);
            }
        }
        vocab.insert("[CLS]".to_owned(), serde_json::json!(2));
        vocab.insert("[SEP]".to_owned(), serde_json::json!(3));
        tok_json["added_tokens"] = serde_json::json!([
            {"id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 1, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 2, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 3, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ]);
        tok_json["post_processor"] = serde_json::json!({
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
            ],
            "pair": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
            ],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}
            }
        });

        let verified = Verified {
            info: ModelInfo {
                id: SYNTAX_MODEL_ID.to_owned(),
                layers: vec!["syntax".to_owned()],
                licence: "CC0-1.0".to_owned(),
                max_len: 256,
                pad_id: 0,
                files: Vec::new(),
            },
            model: Vec::new(),
            tokenizer: serde_json::to_vec(&tok_json).unwrap(),
            expected: Vec::new(),
            coreml: None,
        };
        SubwordTokenizer::from_verified(&verified).expect("SubwordTokenizer builds without onnx")
    }

    #[test]
    fn subword_tokenizer_aligns_cls_sep_and_emits_byte_identical_json_under_jitter() {
        let tokenizer = syntax_subword_tokenizer();
        let encoded = tokenizer.encode("她說 hello！").unwrap();
        assert_eq!(encoded.ids.first().copied(), Some(2));
        assert_eq!(encoded.ids.last().copied(), Some(3));
        assert_eq!(
            encoded.spans[0],
            Span {
                start_char: 0,
                end_char: 0
            }
        );
        assert_eq!(
            *encoded.spans.last().unwrap(),
            Span {
                start_char: 0,
                end_char: 0
            }
        );

        let make_model = |jitter: f32| {
            SyntheticSyntaxModel::new(256, move |ids, cols, data| {
                let rel_off = 17 + 256;
                for s in 1..ids.len().saturating_sub(1) {
                    let row = &mut data[s * cols..(s + 1) * cols];
                    row[15] = 9.0 + jitter;
                    // Subword 1 is always the first overlapping subword of the sentence's first
                    // token: attach subword 1 to [CLS] (0), and every later subword to subword 1.
                    if s == 1 {
                        row[17] = 10.0 + jitter;
                    } else {
                        row[17 + 1] = 10.0 + jitter;
                    }
                    row[rel_off + 26] = 10.0 + jitter; // "nsubj"
                }
            })
        };

        let model_a = make_model(0.0);
        let model_b = make_model(5e-6); // Simulated Core ML vs ONNX float32 delta (< 1e-4).

        for &(_name, text) in GENRE_LANG_SAMPLES {
            let doc_a =
                analyze_document_with_loaded_model(text, &Options::default(), &model_a, &tokenizer)
                    .unwrap();
            let doc_b =
                analyze_document_with_loaded_model(text, &Options::default(), &model_b, &tokenizer)
                    .unwrap();
            assert_eq!(doc_a, doc_b);
            assert_eq!(
                serde_json::to_string(&doc_a).unwrap(),
                serde_json::to_string(&doc_b).unwrap()
            );
            assert_eq!(doc_a.syntax.len(), doc_a.tokens.len());
        }
    }
}

#[test]
fn nlp_doc_cli_zero_flags_unchanged_and_model_flags_work() {
    let bin = env!("CARGO_BIN_EXE_nlp_doc");
    let sample = "小甲走進木屋。We measured 3 kg.\n";

    // 1. Zero flags: byte-identical to `analyze_document`.
    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nlp_doc");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(sample.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let want =
        serde_json::to_string(&analyze_document(sample, &Options::default())).unwrap() + "\n";
    assert_eq!(String::from_utf8(out.stdout).unwrap(), want);

    // 2. With `--models <absent> --backend onnx`: exits 0 and emits `no_model` for "syntax".
    let tmp = tempfile::tempdir().unwrap();
    let absent = tmp.path().join("no-models");
    let mut child = Command::new(bin)
        .args(["--models", absent.to_str().unwrap(), "--backend", "onnx"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nlp_doc --models");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(sample.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let parsed: Document = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        parsed
            .unknown
            .iter()
            .filter(|u| u.layer == "syntax" && u.reason == Reason::NoModel)
            .count(),
        2
    );
}
