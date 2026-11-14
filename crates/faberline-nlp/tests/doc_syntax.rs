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
    ("novel_zh_tw", "小甲搭計程車到捷運站。阿乙在車廂裡看影片。"),
    (
        "novel_zh_cn",
        "小甲坐出租车到地铁站。阿乙已经从房间里出来了。",
    ),
    (
        "novel_en",
        "Alice opened the wooden door. Bob waited quietly.",
    ),
    (
        "novel_mixed_tw",
        "小甲打開筆電測試演算法，然後傳了一封 message。",
    ),
    (
        "novel_mixed_cn",
        "他们刚到酒店就打开 laptop，马上回了一封 email。",
    ),
    (
        "paper_zh_tw",
        "本專案透過新演算法最佳化資料庫與記憶體配置。",
    ),
    (
        "paper_zh_cn",
        "因为该软件通过新算法优化数据库与内存占用，服务器响应稳定。",
    ),
    ("paper_en", "We evaluate the parser on three datasets."),
    (
        "paper_mixed_tw",
        "我們透過 transformer 模型壓低伺服器 latency。",
    ),
    (
        "paper_mixed_cn",
        "我们通过 transformer 模型降低服务器 latency。",
    ),
    ("casual_zh_tw", "今天搭捷運上班超順！晚上一起去看部影片吧。"),
    (
        "casual_zh_cn",
        "今天坐地铁上班真快，这软件的视频加载马上就好了！",
    ),
    ("casual_en", "grabbed coffee this morning, feeling great!"),
    (
        "casual_mixed_tw",
        "剛剛更新軟體跑完 benchmark，結果超 fast！",
    ),
    (
        "casual_mixed_cn",
        "刚才重启服务器跑完 benchmark，结果特别 fast！",
    ),
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
    let make_model = |max_len: usize| {
        SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
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
        })
    };

    let text = "小甲今天早上走進木屋看書。好！";

    // 1. With `max_len = 8` (`max_len - 2 = 6`), sentence 0 exceeds `max_len` as a whole (15
    //    subwords > 8), but every token has <= 2 subwords (`<= 6`), so mega-clause token-boundary
    //    windowing decodes all tokens cleanly with zero `NotApplicable` unknowns.
    let mut windowed = analyze_document(text, &Options::default());
    assert_eq!(windowed.segmentation.sentences.len(), 2);
    annotate_with_encoder(&mut windowed, text, &make_model(8), char_subword_encode)
        .expect("annotate windowed");
    assert_eq!(windowed.syntax.len(), windowed.tokens.len());
    assert!(windowed.tokens.iter().all(|t| t.upos.is_some()));
    assert!(
        !windowed
            .unknown
            .iter()
            .any(|u| u.layer == "syntax" && u.reason == Reason::NotApplicable)
    );

    // 2. With `max_len = 3` (`max_len - 2 = 1`), sentence 0's 2-char token `"小甲"` alone exceeds
    //    `max_len - 2`, emitting a single `NotApplicable` unknown on sentence 0 while sentence 1
    //    (`"好！"`, 1-char tokens) still decodes via windowing!
    let mut doc = analyze_document(text, &Options::default());
    assert_eq!(doc.segmentation.sentences.len(), 2);
    annotate_with_encoder(&mut doc, text, &make_model(3), char_subword_encode).expect("annotate");
    assert!(doc.unknown.contains(&Unknown {
        layer: "syntax".to_owned(),
        span: doc.segmentation.sentences[0].span,
        reason: Reason::NotApplicable,
    }));
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

    for (text, expect_seg_unknown) in [
        ("안녕하세요 감사합니다.\n\n小甲走了。", false),
        ("東京駅に着いた。\n\n小甲走了。", false),
        ("Привет, мир.\n\n小甲走了。", true),
    ] {
        let mut doc = analyze_document(text, &Options::default());
        annotate_with_encoder(&mut doc, text, &model, char_subword_encode).expect("annotate");

        assert!(doc.syntax.iter().all(|s| s.sentence == 1), "{text}");
        assert!(!doc.syntax.is_empty(), "{text}");
        assert!(
            !doc.unknown.iter().any(|u| {
                u.layer == "syntax"
                    && u.span.end_char <= doc.segmentation.sentences[0].span.end_char
            }),
            "{text}: {:?}",
            doc.unknown
        );
        if expect_seg_unknown {
            assert!(
                doc.unknown
                    .iter()
                    .any(|u| u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage),
                "{text}"
            );
        } else {
            assert!(
                !doc.unknown.iter().any(|u| u.layer == "segmentation"),
                "{text}"
            );
            assert!(
                !doc.unknown
                    .iter()
                    .any(|u| u.reason == Reason::UnsupportedLanguage),
                "{text}"
            );
            assert!(doc.tokens.iter().any(|t| t.upos.is_some()), "{text}");
        }
    }
}

#[test]
fn zh_cn_mainland_tokens_upgrade_contextual_upos_and_decode_syntax() {
    let text = "因为我们通过软件查询了数据库，内存已经很稳定。";
    let chars: Vec<char> = text.chars().collect();
    let base = analyze_document(text, &Options::default());

    let surface = |span: Span| -> String { chars[span.start_char..span.end_char].iter().collect() };
    let tok_surfaces: Vec<String> = base.tokens.iter().map(|t| surface(t.span)).collect();
    for mainland_word in ["通过", "软件", "数据库", "内存"] {
        let idx = tok_surfaces
            .iter()
            .position(|w| w == mainland_word)
            .unwrap_or_else(|| {
                panic!("expected Mainland token {mainland_word:?} in {tok_surfaces:?}")
            });
        assert_eq!(
            base.tokens[idx].upos, None,
            "Layer 2 must leave open-class Mainland {mainland_word:?} undecided"
        );
        assert!(
            base.unknown.contains(&Unknown {
                layer: "pos".to_owned(),
                span: base.tokens[idx].span,
                reason: Reason::NoModel,
            }),
            "Layer 2 must emit pos NoModel for {mainland_word:?}"
        );
    }
    let le_idx = tok_surfaces.iter().position(|w| w == "了").unwrap();
    let hen_idx = tok_surfaces.iter().position(|w| w == "很").unwrap();
    // Unambiguous closed-class characters/words are already decided in Layer 2.
    for (decided_word, expected_upos) in [
        ("因为", Upos::Sconj),
        ("我们", Upos::Pron),
        ("了", Upos::Part),
        ("已经", Upos::Adv),
        ("很", Upos::Adv),
    ] {
        let idx = tok_surfaces.iter().position(|w| w == decided_word).unwrap();
        assert_eq!(base.tokens[idx].upos, Some(expected_upos));
    }

    // Map each Mainland token's first subword index (`s_u` under `char_subword_encode`) to its
    // true contextual UD UPOS tag, while deliberately predicting `Verb` (15) on "了" and "很" to
    // verify Layer-2 decided tags are never overwritten!
    let enc = char_subword_encode(text).unwrap();
    let subword_of = |span: Span| -> usize {
        enc.spans
            .iter()
            .enumerate()
            .skip(1)
            .find_map(|(s_idx, sub)| {
                (sub.start_char < sub.end_char
                    && sub.start_char < span.end_char
                    && span.start_char < sub.end_char)
                    .then_some(s_idx)
            })
            .unwrap()
    };

    let wanted_upos: Vec<(&str, usize, Upos)> = vec![
        ("因为", 13, Upos::Sconj),
        ("我们", 10, Upos::Pron),
        ("通过", 1, Upos::Adp),
        ("软件", 7, Upos::Noun),
        ("数据库", 7, Upos::Noun),
        ("内存", 7, Upos::Noun),
        ("已经", 2, Upos::Adv),
    ];
    let mut subword_upos_plan = Vec::new();
    for &(word, upos_col, _) in &wanted_upos {
        let idx = tok_surfaces.iter().position(|w| w == word).unwrap();
        subword_upos_plan.push((subword_of(base.tokens[idx].span), upos_col));
    }
    let query_idx = tok_surfaces
        .iter()
        .position(|w| w.starts_with('查'))
        .unwrap();
    let root_sub = subword_of(base.tokens[query_idx].span);

    let max_len = 128;
    let model = SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
        let rel_off = 17 + max_len;
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            // Default confident Verb (15), overridden per Mainland token plan.
            let upos_col = subword_upos_plan
                .iter()
                .find_map(|&(target_s, col)| (target_s == s).then_some(col))
                .unwrap_or(15);
            row[upos_col] = 11.0;

            if s == root_sub {
                row[17] = 13.0; // Attach main predicate to [CLS] root
            } else {
                row[17 + root_sub] = 13.0;
                row[rel_off + 29] = 11.0; // "obl"
            }
        }
    });

    let mut doc = base;
    annotate_with_encoder(&mut doc, text, &model, char_subword_encode).expect("annotate zh-CN");

    for &(word, _, expected_tag) in &wanted_upos {
        let idx = tok_surfaces.iter().position(|w| w == word).unwrap();
        assert_eq!(
            doc.tokens[idx].upos,
            Some(expected_tag),
            "Mainland token {word:?} should upgrade to {expected_tag:?}"
        );
    }
    // Shared Layer-2 decided tags stay intact even though model logits favored Verb.
    assert_eq!(doc.tokens[le_idx].upos, Some(Upos::Part));
    assert_eq!(doc.tokens[hen_idx].upos, Some(Upos::Adv));
    assert!(!doc.unknown.iter().any(|u| u.layer == "pos"));
    assert_eq!(doc.syntax.len(), doc.tokens.len());
    let roots: Vec<_> = doc.syntax.iter().filter(|s| s.head.is_none()).collect();
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0].token, query_idx);
    assert_eq!(roots[0].deprel, "root");
}

#[test]
fn analyze_document_with_model_reports_no_model_and_propagates_integrity_errors() {
    let repo_models = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/models"));
    let unstage_tmp = tempfile::tempdir().unwrap();
    let models_dir = if repo_models.join("doc-syntax-v1").exists() {
        std::fs::copy(
            repo_models.join("manifest.json"),
            unstage_tmp.path().join("manifest.json"),
        )
        .unwrap();
        unstage_tmp.path().to_path_buf()
    } else {
        repo_models
    };
    let text = "小甲走進木屋。We measured the data.\n\n안녕하세요.\n\nこんにちは。\n\nПривет.";
    // Without gitignored `models/doc-syntax-v1/` weights, `manifest.json` cleanly maps to `no_model`.
    for backend in [Backend::Onnx, Backend::Coreml] {
        let doc =
            analyze_document_with_model(text, &Options::default(), &models_dir, backend).unwrap();
        assert!(doc.syntax.is_empty());
        // Supported sentences 0 and 1 get `no_model` for "syntax"; JA/KO/unsupported sentences 2..=4 do not.
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
        for skipped_idx in 2..=4 {
            assert!(!doc.unknown.iter().any(|u| {
                u.layer == "syntax" && u.span == doc.segmentation.sentences[skipped_idx].span
            }));
        }
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

#[test]
fn ws3_multi_subword_upos_mean_pooling_and_shenme_disambiguation() {
    let max_len = 64;
    // Subword 1 (`一`) strongly prefers `NUM` (index 8 = 7.0), while subwords 2..=3 (`會`, `兒`)
    // strongly prefer `NOUN` (index 7 = 8.0). First-subword lookup would mis-tag `一會兒` as `NUM`,
    // whereas arithmetic mean pooling across all overlapping subwords (`(7, 8, 8)`) selects `NOUN`!
    let pooling_model = SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            if s == 1 {
                row[8] = 7.0; // NUM on leading numeral character
                row[7] = 1.0;
            } else {
                row[7] = 8.0; // NOUN on remaining characters
            }
            row[17] = if s == 1 { 10.0 } else { 0.0 };
            row[17 + 1] = if s == 1 { 0.0 } else { 10.0 };
            row[17 + max_len + 29] = 10.0; // "obl"
        }
    });

    let opts: Options = serde_json::from_value(serde_json::json!({
        "names": ["一會兒", "一時間", "一清二楚", "五顏六色"]
    }))
    .unwrap();
    for phrase in ["一會兒。", "一時間。", "一清二楚。", "五顏六色。"] {
        let mut doc = analyze_document(phrase, &opts);
        assert_eq!(doc.tokens[0].upos, None);
        annotate_with_encoder(&mut doc, phrase, &pooling_model, char_subword_encode)
            .expect("mean-pooled UPOS annotation");
        assert_eq!(
            doc.tokens[0].upos,
            Some(Upos::Noun),
            "mean pooling across subwords should override leading numeral subword on {phrase}"
        );
    }

    // `什麼` / `什么`: when Layer 2 leaves `upos == None` and the model predicts `PART` or `< 0.50`,
    // upgrade to `DET` immediately before `NOUN`/`PROPN` in the same sentence, else `PRON`.
    let shenme_model = SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            // Emit `PART` (index 9) on early subwords and `NOUN` (index 7) on later subwords (`書卷`/`地方`).
            if s <= 4 {
                row[9] = 8.0; // PART
            } else {
                row[7] = 8.0; // NOUN
            }
            if s == 1 {
                row[17] = 10.0;
            } else {
                row[17 + 1] = 10.0;
            }
            row[17 + max_len + 28] = 10.0; // "obj"
        }
    });

    let det_opts: Options = serde_json::from_value(serde_json::json!({
        "names": ["什麼", "什么", "書卷", "地方"]
    }))
    .unwrap();

    for (text, target, expected_upos) in [
        ("他買了什麼書卷？", "什麼", Upos::Det),
        ("去了什么地方？", "什么", Upos::Det),
        ("你想說什麼？", "什麼", Upos::Pron),
        ("他在看什么？", "什么", Upos::Pron),
    ] {
        let mut doc = analyze_document(text, &det_opts);
        annotate_with_encoder(&mut doc, text, &shenme_model, char_subword_encode)
            .expect("annotate shenme");
        let tok = doc
            .tokens
            .iter()
            .find(|t| {
                let s: String = text
                    .chars()
                    .skip(t.span.start_char)
                    .take(t.span.end_char - t.span.start_char)
                    .collect();
                s == target
            })
            .expect("target token present");
        assert_eq!(tok.upos, Some(expected_upos), "failed on {text}");
        assert!(
            !doc.unknown
                .iter()
                .any(|u| u.layer == "pos" && u.span == tok.span),
            "pos unknown should be cleared for {target} in {text}"
        );
    }
}

#[test]
fn ws5_soft_clause_windowing_and_ancestor_tree_retention() {
    use std::sync::Mutex;

    // 1. Soft clause-group windowing: >= 4 clauses and > 64 tokens, even when subwords <= max_len (256).
    let clauses = [
        "清晨的薄霧籠罩著山谷裡的青石小徑與竹林木屋",
        "旅人背起行囊推開木門沿著溪流緩緩向東邊行走",
        "微風吹過樹梢帶來遠處渡口與村落的陣陣鐘聲",
        "陽光穿過雲層照亮了河岸兩側整齊排列的石碑",
        "他停下腳步在石橋邊取出地圖仔細核對沿途方向",
        "隨後沿著寬闊的官道繼續前往山腳下的古老驛站。",
    ];
    let long_paratactic = clauses.join("，");
    let mut doc = analyze_document(&long_paratactic, &Options::default());
    assert_eq!(doc.segmentation.sentences.len(), 1);
    assert!(doc.segmentation.clauses.len() >= 6);
    assert!(
        doc.tokens.len() > 64,
        "expected > 64 tokens, got {}",
        doc.tokens.len()
    );
    let full_subwords = char_subword_encode(&long_paratactic).unwrap();
    assert!(
        full_subwords.ids.len() < 256,
        "subwords ({}) must be < 256 to exercise soft windowing",
        full_subwords.ids.len()
    );

    let window_lengths: std::sync::Arc<Mutex<Vec<usize>>> =
        std::sync::Arc::new(Mutex::new(Vec::new()));
    let recorded = std::sync::Arc::clone(&window_lengths);
    let max_len = 256;
    let soft_model = SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
        recorded.lock().unwrap().push(ids.len());
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            row[7] = 10.0; // NOUN
            if s == 1 {
                row[17] = 12.0; // window root -> [CLS]
            } else {
                row[17 + 1] = 12.0;
            }
            row[17 + max_len + 26] = 12.0; // "nsubj"
        }
    });

    annotate_with_encoder(&mut doc, &long_paratactic, &soft_model, char_subword_encode)
        .expect("soft windowed syntax");
    let lengths = window_lengths.lock().unwrap().clone();
    assert!(
        lengths.len() >= 2,
        "expected multiple soft clause windows, got {lengths:?}"
    );
    assert!(
        lengths.iter().all(|&len| len <= 96),
        "every soft window should respect <= 96 subword budget: {lengths:?}"
    );
    assert_eq!(doc.syntax.len(), doc.tokens.len());
    let roots: Vec<_> = doc.syntax.iter().filter(|s| s.head.is_none()).collect();
    assert_eq!(
        roots.len(),
        1,
        "stitched sentence must have a single global root"
    );
    let root_tok = roots[0].token;
    let parataxis_count = doc
        .syntax
        .iter()
        .filter(|s| s.head == Some(root_tok) && s.deprel == "parataxis")
        .count();
    assert_eq!(parataxis_count, lengths.len() - 1);

    // 2. Tree-consistent root & ancestor retention:
    //    Sentence tokens: 4 single-char tokens ("甲 乙 丙 。" via lexicon/text).
    //    - Token 0 (root): arc prob ~0.24 (< 0.35), rescued because child connects to it!
    //    - Token 1 (parent A -> token 0): score ~0.25 (>= 0.20, < 0.35), rescued by token 2!
    //    - Token 2 (child A -> token 1): score ~0.92 (>= 0.35), retains 2 -> 1 -> 0!
    //    - Token 3 (parent B -> token 0): score ~0.12 (< 0.20), NOT rescued!
    //    - Token 4 (child B -> token 3): score ~0.92 (>= 0.35), DROPPED because parent B < 0.20!
    let tree_text = "甲 乙 丙 丁。";
    let mut tree_doc = analyze_document(tree_text, &Options::default());
    assert_eq!(tree_doc.tokens.len(), 5);
    let tok_subs: Vec<usize> = {
        let sub = char_subword_encode(tree_text).unwrap();
        tree_doc
            .tokens
            .iter()
            .map(|t| {
                sub.spans
                    .iter()
                    .enumerate()
                    .skip(1)
                    .find_map(|(i, sp)| {
                        (sp.start_char < t.span.end_char && t.span.start_char < sp.end_char)
                            .then_some(i)
                    })
                    .unwrap()
            })
            .collect()
    };

    let m_max = 32;
    let retention_model = SyntheticSyntaxModel::new(m_max, move |_ids, cols, data| {
        let [s0, s1, s2, s3, s4] = [
            tok_subs[0],
            tok_subs[1],
            tok_subs[2],
            tok_subs[3],
            tok_subs[4],
        ];
        // Token 0 (root): attaches to [CLS] (s=0) with moderate logit so p_arc ~ 0.26 (< 0.35).
        {
            let r1 = &mut data[s0 * cols..(s0 + 1) * cols];
            r1[7] = 10.0;
            r1[17] = 0.35; // exp(0.35) / (exp(0.35) + 4*1.0) = 1.419 / 5.419 = 0.262 -> Score(262)
        }
        // Token 1: attaches to Token 0 with p_arc * p_rel ~ 0.31 (>= 0.20, < 0.35).
        {
            let r2 = &mut data[s1 * cols..(s1 + 1) * cols];
            r2[7] = 10.0;
            r2[17] = -20.0;
            r2[17 + s0] = 0.30;
            r2[17 + m_max + 26] = 10.0;
        }
        // Token 2: attaches to Token 1 with high confidence ~ 0.99 (>= 0.35).
        {
            let r3 = &mut data[s2 * cols..(s2 + 1) * cols];
            r3[7] = 10.0;
            r3[17] = -20.0;
            r3[17 + s1] = 12.0;
            r3[17 + m_max + 28] = 12.0;
        }
        // Token 3: attaches to Token 0 with low structural arc confidence ~ 0.174 (< 0.20).
        {
            let r4 = &mut data[s3 * cols..(s3 + 1) * cols];
            r4[7] = 10.0;
            r4[17 + s0] = 0.05;
        }
        // Token 4: attaches to Token 3 with high confidence ~ 0.99 (>= 0.35), dropped via Token 3!
        {
            let r5 = &mut data[s4 * cols..(s4 + 1) * cols];
            r5[7] = 10.0;
            r5[17] = -20.0;
            r5[17 + s3] = 12.0;
            r5[17 + m_max + 32] = 12.0;
        }
    });

    annotate_with_encoder(
        &mut tree_doc,
        tree_text,
        &retention_model,
        char_subword_encode,
    )
    .expect("retention syntax");

    let surviving_tokens: Vec<usize> = tree_doc.syntax.iter().map(|s| s.token).collect();
    assert_eq!(
        surviving_tokens,
        vec![0, 1, 2],
        "root (0) and >=0.20 ancestor (1) must be retained for child (2), while broken branch (3, 4) drops"
    );
    assert_eq!(
        tree_doc.syntax.iter().filter(|s| s.head.is_none()).count(),
        1
    );
    for syn in &tree_doc.syntax {
        if let Some(h) = syn.head {
            assert!(
                surviving_tokens.contains(&h),
                "orphan head {h} in {:?}",
                tree_doc.syntax
            );
        }
    }
    for dropped_tok in [3usize, 4usize] {
        assert!(tree_doc.unknown.contains(&Unknown {
            layer: "syntax".to_owned(),
            span: tree_doc.tokens[dropped_tok].span,
            reason: Reason::LowConfidence,
        }));
    }
}

#[cfg(any(feature = "onnx", feature = "coreml"))]
#[test]
fn ws3_ws5_real_model_gold_and_numeral_phrases_when_staged() {
    let Some(models_dir) = std::env::var_os("NLP_SYNTAX_MODELS_DIR").map(PathBuf::from) else {
        return;
    };
    if !models_dir.join(SYNTAX_MODEL_ID).is_dir() {
        return;
    }
    let backend = if faberline_nlp::doc::model::compiled(Backend::Onnx) {
        Backend::Onnx
    } else if faberline_nlp::doc::model::compiled(Backend::Coreml) {
        Backend::Coreml
    } else {
        return;
    };

    let verified = faberline_nlp::doc::model::verify(&models_dir, SYNTAX_MODEL_ID)
        .expect("verify staged doc-syntax-v1");
    let model =
        faberline_nlp::doc::model::load_verified(&verified, backend).expect("load doc-syntax-v1");
    let tokenizer = faberline_nlp::doc::model::SubwordTokenizer::from_verified(&verified)
        .expect("load subword tokenizer");

    // 1. Check multi-char numeral-prefix phrases and `什麼`/`什么` with real `doc-syntax-v1`.
    let phrase_opts: Options = serde_json::from_value(serde_json::json!({
        "names": ["一會兒", "一時間", "一清二楚", "五顏六色"]
    }))
    .unwrap();
    for (sentence, target) in [
        ("過了一會兒，門開了。", "一會兒"),
        ("一時間眾人都沉默了。", "一時間"),
        ("他看得一清二楚。", "一清二楚"),
        ("花園裡開滿了五顏六色的花。", "五顏六色"),
    ] {
        let doc = faberline_nlp::doc::syntax::analyze_document_with_loaded_model(
            sentence,
            &phrase_opts,
            &*model,
            &tokenizer,
        )
        .expect("decode numeral-prefix sentence");
        let tok = doc
            .tokens
            .iter()
            .find(|t| {
                let s: String = sentence
                    .chars()
                    .skip(t.span.start_char)
                    .take(t.span.end_char - t.span.start_char)
                    .collect();
                s == target
            })
            .expect("target token found");
        assert_ne!(
            tok.upos,
            Some(Upos::Num),
            "{target:?} in {sentence:?} must not be mis-tagged NUM"
        );
    }

    for (sentence, target, expected) in [
        ("他買了什麼書？", "什麼", Upos::Det),
        ("你想說什麼？", "什麼", Upos::Pron),
        ("他买了什么书？", "什么", Upos::Det),
        ("你想说什么？", "什么", Upos::Pron),
    ] {
        let doc = faberline_nlp::doc::syntax::analyze_document_with_loaded_model(
            sentence,
            &Options::default(),
            &*model,
            &tokenizer,
        )
        .expect("decode shenme sentence");
        let tok = doc
            .tokens
            .iter()
            .find(|t| {
                let s: String = sentence
                    .chars()
                    .skip(t.span.start_char)
                    .take(t.span.end_char - t.span.start_char)
                    .collect();
                s == target
            })
            .expect("shenme token found");
        assert_eq!(tok.upos, Some(expected), "{sentence}");
    }

    // 1b. WS3-v3 & WS5-v3 real-model checks (`二話不說`/`一動不動`, disposal `將`, lexicon prior,
    //     and multi-clause subordinate root guard).
    let v3_opts: Options = serde_json::from_value(serde_json::json!({
        "normalize_traditional": true,
        "lexicon": ["二話不說", "二话不说", "一動不動", "一动不动", "玄冰玉匣", "段天涯"],
        "names": ["段天涯"]
    }))
    .unwrap();
    for (sentence, target) in [
        ("老者二話不說，轉身走入石室。", "二話不說"),
        ("青年二话不说，转身走入石室。", "二话不说"),
        ("石像立在殿內，一動不動。", "一動不動"),
        ("守卫站在门前，一动不动。", "一动不动"),
    ] {
        let doc = faberline_nlp::doc::syntax::analyze_document_with_loaded_model(
            sentence, &v3_opts, &*model, &tokenizer,
        )
        .expect("decode v3 numeral idiom");
        let tok = doc
            .tokens
            .iter()
            .find(|t| {
                sentence
                    .chars()
                    .skip(t.span.start_char)
                    .take(t.span.end_char - t.span.start_char)
                    .collect::<String>()
                    == target
            })
            .expect("idiom token found");
        assert!(
            tok.upos.is_some() && tok.upos != Some(Upos::Num),
            "{target:?} in {sentence:?} should be decided non-NUM, got {:?}",
            tok.upos
        );
    }

    {
        let sentence = "段天涯將玄冰玉匣收起。";
        let doc = faberline_nlp::doc::syntax::analyze_document_with_loaded_model(
            sentence, &v3_opts, &*model, &tokenizer,
        )
        .expect("decode disposal + lexicon sentence");
        let find_upos = |surface: &str| {
            doc.tokens
                .iter()
                .find(|t| {
                    sentence
                        .chars()
                        .skip(t.span.start_char)
                        .take(t.span.end_char - t.span.start_char)
                        .collect::<String>()
                        == surface
                })
                .expect("token present")
                .upos
        };
        assert_eq!(find_upos("段天涯"), Some(Upos::Propn));
        assert_eq!(find_upos("將"), Some(Upos::Adp));
        assert_eq!(find_upos("玄冰玉匣"), Some(Upos::Noun));
    }

    {
        let multiclause = "雖然山路十分崎嶇，旅人還是推開了古老的木門。";
        let doc = faberline_nlp::doc::syntax::analyze_document_with_loaded_model(
            multiclause,
            &v3_opts,
            &*model,
            &tokenizer,
        )
        .expect("decode multiclause sentence");
        let root_syn = doc
            .syntax
            .iter()
            .find(|s| s.head.is_none())
            .expect("single root emitted");
        let root_text: String = multiclause
            .chars()
            .skip(root_syn.span.start_char)
            .take(root_syn.span.end_char - root_syn.span.start_char)
            .collect();
        assert_ne!(
            root_text, "崎嶇",
            "concessive subordinate predicate must not steal matrix root"
        );
    }

    // 2. Score all 300 items of `tests/data/gold_layers4_6_v1.jsonl`.
    let gold_raw = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/gold_layers4_6_v1.jsonl"),
    )
    .expect("read gold_layers4_6_v1.jsonl");
    let mut total_toks = 0usize;
    let mut upos_ok = 0usize;
    let mut uas_ok = 0usize;
    let mut las_ok = 0usize;
    let mut emitted_toks = 0usize;

    for line in gold_raw.lines().filter(|l| !l.trim().is_empty()) {
        let val: serde_json::Value = serde_json::from_str(line).unwrap();
        let text = val["text"].as_str().unwrap();
        let gold_upos: Vec<&str> = val["upos"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let gold_heads: Vec<usize> = val["heads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let gold_deprels: Vec<&str> = val["deprels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();

        let doc = faberline_nlp::doc::syntax::analyze_document_with_loaded_model(
            text,
            &Options::default(),
            &*model,
            &tokenizer,
        )
        .expect("decode gold item");
        assert_eq!(doc.tokens.len(), gold_upos.len());

        // Check sentence tree invariants on every sentence of every Gold item.
        for s_idx in 0..doc.segmentation.sentences.len() {
            let sent_syn: Vec<_> = doc.syntax.iter().filter(|s| s.sentence == s_idx).collect();
            if !sent_syn.is_empty() {
                assert_eq!(sent_syn.iter().filter(|s| s.head.is_none()).count(), 1);
                for syn in &sent_syn {
                    if let Some(h) = syn.head {
                        assert!(sent_syn.iter().any(|p| p.token == h));
                    }
                }
            }
        }

        // Convert `gold_heads` (1-based within each sentence or within item? Check `eval_gold.py` line 85: `0 if raw_head is None else int(raw_head) + 1`, so 1-based global within item!).
        let mut pred_head_1based = vec![None; doc.tokens.len()];
        let mut pred_deprel = vec![""; doc.tokens.len()];
        for syn in &doc.syntax {
            emitted_toks += 1;
            pred_head_1based[syn.token] = Some(syn.head.map_or(0, |h| h + 1));
            pred_deprel[syn.token] = syn.deprel.as_str();
        }

        for i in 0..doc.tokens.len() {
            total_toks += 1;
            let pred_u = doc.tokens[i].upos.map(|u| {
                serde_json::to_value(u)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_owned()
            });
            if pred_u.as_deref() == Some(gold_upos[i]) {
                upos_ok += 1;
            }
            if pred_head_1based[i] == Some(gold_heads[i]) {
                uas_ok += 1;
                if pred_deprel[i] == gold_deprels[i] {
                    las_ok += 1;
                }
            }
        }
    }

    let upos_acc = upos_ok as f64 / total_toks as f64;
    let uas = uas_ok as f64 / total_toks as f64;
    let las = las_ok as f64 / total_toks as f64;
    let emitted_rate = emitted_toks as f64 / total_toks as f64;
    eprintln!(
        "[gold_layers4_6_v1] backend={backend:?} total={total_toks} emitted={emitted_toks} ({emitted_rate:.4}) UPOS={upos_ok}/{total_toks} ({upos_acc:.4}) UAS={uas_ok}/{total_toks} ({uas:.4}) LAS={las_ok}/{total_toks} ({las:.4})"
    );
    assert!(
        emitted_rate >= 0.9287,
        "emitted_rate {emitted_rate:.4} regressed below Phase 1 baseline 0.9287"
    );
    assert!(
        upos_acc >= 0.8883,
        "UPOS {upos_acc:.4} regressed below Phase 1 baseline 0.8883"
    );
    assert!(
        uas >= 0.7496,
        "UAS {uas:.4} regressed below Phase 1 baseline 0.7496"
    );
    assert!(
        las >= 0.6888,
        "LAS {las:.4} regressed below Phase 1 baseline 0.6888"
    );
}

#[test]
fn ws3_v3_scraper_infix_subword_cleaning_preserves_token_spans() {
    use std::sync::Mutex;

    let text = "那名少丅女向長堊老行禮後啟動傳丅送。";
    let opts: Options = serde_json::from_value(serde_json::json!({
        "normalize_traditional": true,
        "lexicon": ["少丅女", "長堊老", "傳丅送"]
    }))
    .unwrap();
    let mut doc = analyze_document(text, &opts);
    let max_len = 64;
    let model = SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            row[7] = 9.0; // NOUN
            row[17] = if s == 1 { 10.0 } else { 0.0 };
            row[17 + 1] = if s == 1 { 0.0 } else { 10.0 };
            row[17 + max_len + 28] = 10.0;
        }
    });

    let seen_inputs: Mutex<Vec<String>> = Mutex::new(Vec::new());
    annotate_with_encoder(&mut doc, text, &model, |slice| {
        seen_inputs.lock().unwrap().push(slice.to_owned());
        char_subword_encode(slice)
    })
    .expect("annotate scraper-infixed sentence");

    let recorded = seen_inputs.lock().unwrap().clone();
    assert_eq!(
        recorded,
        vec!["那名少女向長老行禮後啟動傳送。".to_owned()],
        "encoder must see clean Han compounds without sandwiched scraper infixes"
    );

    // Every `Token.span` and `SyntaxToken.span` must remain exact on the original un-mutated text!
    let slice_chars = |sp: Span| -> String {
        text.chars()
            .skip(sp.start_char)
            .take(sp.end_char - sp.start_char)
            .collect()
    };
    for expected_raw in ["少丅女", "長堊老", "傳丅送"] {
        let tok_idx = doc
            .tokens
            .iter()
            .position(|t| slice_chars(t.span) == expected_raw)
            .unwrap_or_else(|| panic!("expected original span for {expected_raw}"));
        let syn = doc
            .syntax
            .iter()
            .find(|s| s.token == tok_idx)
            .expect("infixed token mapped cleanly into syntax without unmapped unknown");
        assert_eq!(slice_chars(syn.span), expected_raw);
    }
    assert!(
        !doc.unknown
            .iter()
            .any(|u| u.layer == "syntax" && u.reason == Reason::NotApplicable),
        "no token may become NotApplicable after infix stripping: {:?}",
        doc.unknown
    );

    // When `normalize_traditional` is false, `encode` sees the raw text verbatim.
    let raw_opts: Options = serde_json::from_value(serde_json::json!({
        "normalize_traditional": false,
        "lexicon": ["少丅女", "長堊老", "傳丅送"]
    }))
    .unwrap();
    let mut raw_doc = analyze_document(text, &raw_opts);
    let raw_seen: Mutex<Vec<String>> = Mutex::new(Vec::new());
    annotate_with_encoder(&mut raw_doc, text, &model, |slice| {
        raw_seen.lock().unwrap().push(slice.to_owned());
        char_subword_encode(slice)
    })
    .expect("annotate raw sentence");
    assert_eq!(raw_seen.lock().unwrap().as_slice(), &[text.to_owned()]);
}

#[test]
fn ws3_v3_disposal_coverb_jiang_lexicon_prior_and_four_char_idioms() {
    let max_len = 64;

    // 1. Disposal coverb `將`/`将` disambiguation + lexicon nominal/proper-noun prior.
    let opts: Options = serde_json::from_value(serde_json::json!({
        "normalize_traditional": true,
        "lexicon": ["玄冰玉匣", "段天涯", "青石鎮", "一清二楚"],
        "names": ["段天涯", "青石鎮"]
    }))
    .unwrap();

    // Model mis-predicts `NOUN` on `將`, `VERB` on lexicon `玄冰玉匣`, `ADV` on `段天涯`, and
    // `VERB` on real predicates (`收起`, `擊退`, `率領`, `前往`, `尋找`, `離開`).
    let make_coverb_model = |text: &'static str, verbs: &'static [&'static str]| {
        let sub = char_subword_encode(text).unwrap();
        let verb_first_subs: Vec<usize> = verbs
            .iter()
            .filter_map(|v| {
                let char_pos = text
                    .char_indices()
                    .enumerate()
                    .find_map(|(ci, (bi, _))| text[bi..].starts_with(v).then_some(ci))?;
                sub.spans.iter().enumerate().skip(1).find_map(|(si, sp)| {
                    (sp.start_char <= char_pos && char_pos < sp.end_char).then_some(si)
                })
            })
            .collect();
        SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
            for s in 1..ids.len().saturating_sub(1) {
                let row = &mut data[s * cols..(s + 1) * cols];
                if verb_first_subs.contains(&s) {
                    row[15] = 9.0; // VERB
                } else if s == 1 {
                    row[2] = 9.0; // ADV (tests `段天涯` lexicon+names -> PROPN override)
                } else {
                    row[7] = 9.0; // NOUN (default on `將` and nominals)
                }
                row[17] = if s == 1 { 10.0 } else { 0.0 };
                row[17 + 1] = if s == 1 { 0.0 } else { 10.0 };
                row[17 + max_len + 28] = 10.0;
            }
        })
    };

    {
        let s1 = "段天涯將玄冰玉匣收起。";
        let mut d1 = analyze_document(s1, &opts);
        // Force model to predict `VERB` on `玄冰玉匣` too so we verify `VERB -> NOUN` lexicon override!
        let m1 = make_coverb_model(s1, &["玄冰玉匣", "收起"]);
        annotate_with_encoder(&mut d1, s1, &m1, char_subword_encode).unwrap();
        let upos_of = |w: &str| {
            d1.tokens
                .iter()
                .find(|t| {
                    s1.chars()
                        .skip(t.span.start_char)
                        .take(t.span.end_char - t.span.start_char)
                        .collect::<String>()
                        == w
                })
                .unwrap()
                .upos
        };
        assert_eq!(upos_of("段天涯"), Some(Upos::Propn));
        assert_eq!(upos_of("將"), Some(Upos::Adp));
        assert_eq!(upos_of("玄冰玉匣"), Some(Upos::Noun));
        assert_eq!(upos_of("收起"), Some(Upos::Verb));
    }

    {
        let s2 = "青年将那名老者擊退。";
        let mut d2 = analyze_document(s2, &opts);
        let m2 = make_coverb_model(s2, &["擊退"]);
        annotate_with_encoder(&mut d2, s2, &m2, char_subword_encode).unwrap();
        let jiang = d2
            .tokens
            .iter()
            .find(|t| {
                s2.chars()
                    .skip(t.span.start_char)
                    .take(t.span.end_char - t.span.start_char)
                    .collect::<String>()
                    == "将"
            })
            .unwrap();
        assert_eq!(jiang.upos, Some(Upos::Adp));
    }

    // Negative guards on `將`: noun `"猛將"` (`Preceded by 猛`) and future adverb `"將前往"` / `"將要"` must NOT become `ADP`.
    for (neg_text, verbs) in [
        ("兩員猛將率領眾人出城。", &["率領", "出城"][..]),
        ("青年將前往青石鎮尋找線索。", &["前往", "尋找"][..]),
        ("青年將要離開山谷。", &["離開"][..]),
    ] {
        let mut dn = analyze_document(neg_text, &opts);
        let mn = make_coverb_model(neg_text, verbs);
        annotate_with_encoder(&mut dn, neg_text, &mn, char_subword_encode).unwrap();
        for tok in &dn.tokens {
            let surface: String = neg_text
                .chars()
                .skip(tok.span.start_char)
                .take(tok.span.end_char - tok.span.start_char)
                .collect();
            if surface == "將" {
                assert_ne!(
                    tok.upos,
                    Some(Upos::Adp),
                    "`將` must not become ADP in {neg_text}"
                );
            }
        }
    }

    // 2. Diluted 4-char numeral idioms (`二話不說`, `一動不動`, `四面八方`, `一般無二`, `一五一十`).
    //    Simulate heterogeneous per-character subword peaks (`NUM`, `NOUN`, `ADV`, `VERB`) whose
    //    arithmetic mean softmax is ~0.24 (< 0.50 `UPOS_THRESHOLD`).
    let idiom_opts: Options = serde_json::from_value(serde_json::json!({
        "lexicon": ["二話不說", "一動不動", "四面八方", "一般無二", "一五一十"]
    }))
    .unwrap();
    let diluted_model = SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
        let cycle = [8usize, 7, 2, 15]; // NUM, NOUN, ADV, VERB -> diluted arithmetic mean!
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            if s <= 4 {
                row[cycle[(s - 1) % 4]] = 5.0;
            } else {
                row[15] = 9.0; // VERB on following clause predicate
            }
            row[17] = if s == 1 { 10.0 } else { 0.0 };
            row[17 + 1] = if s == 1 { 0.0 } else { 10.0 };
            row[17 + max_len + 2] = 10.0;
        }
    });

    for (sent, target, expected) in [
        ("二話不說，轉身離去。", "二話不說", Upos::Adv),
        ("一動不動。", "一動不動", Upos::Verb),
        ("四面八方。", "四面八方", Upos::Noun),
        ("一般無二。", "一般無二", Upos::Adj),
        ("一五一十，告知眾人。", "一五一十", Upos::Adv),
    ] {
        let mut di = analyze_document(sent, &idiom_opts);
        annotate_with_encoder(&mut di, sent, &diluted_model, char_subword_encode).unwrap();
        let tok = &di.tokens[0];
        let surface: String = sent
            .chars()
            .skip(tok.span.start_char)
            .take(tok.span.end_char - tok.span.start_char)
            .collect();
        assert_eq!(surface, target);
        assert_eq!(tok.upos, Some(expected), "failed idiom recovery on {sent}");
        assert!(
            !di.unknown
                .iter()
                .any(|u| u.layer == "pos" && u.span == tok.span),
            "pos unknown should be cleared on {target}"
        );
    }
}

#[test]
fn ws5_v3_multiclause_root_penalty_and_clause_local_argument_guard() {
    let max_len = 64;

    // 1. Subordinate clause (`雖然...`), trailing temporal (`...之後`), prenominal relative
    //    (`擊退的對手`), and non-predicative (`終於`) root-stealing guard.
    let s_root = "雖然山路崎嶇，旅人終於擊敗了剛才擊退的對手。";
    let opts: Options = serde_json::from_value(serde_json::json!({
        "normalize_traditional": true,
        "lexicon": ["山路", "崎嶇", "旅人", "終於", "擊敗", "剛才", "擊退", "對手"]
    }))
    .unwrap();
    let mut doc_root = analyze_document(s_root, &opts);
    let sub_root = char_subword_encode(s_root).unwrap();
    let first_sub_of = |word: &str, text: &str, sub: &Subwords| -> usize {
        let char_pos = text
            .char_indices()
            .enumerate()
            .find_map(|(ci, (bi, _))| text[bi..].starts_with(word).then_some(ci))
            .unwrap();
        sub.spans
            .iter()
            .enumerate()
            .skip(1)
            .find_map(|(si, sp)| {
                (sp.start_char <= char_pos && char_pos < sp.end_char).then_some(si)
            })
            .unwrap()
    };
    let s_qiqu = first_sub_of("崎嶇", s_root, &sub_root);
    let s_zhongyu = first_sub_of("終於", s_root, &sub_root);
    let s_jibai = first_sub_of("擊敗", s_root, &sub_root);
    let s_jitui = first_sub_of("擊退", s_root, &sub_root);

    let root_model = SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            if s == s_qiqu {
                row[0] = 9.0; // ADJ inside concessive `雖然` clause
                row[17] = 4.2; // raw root logit higher than `擊敗` (3.0), but penalized by -2.5 -> 1.7!
                row[17 + s_jibai] = 3.5;
            } else if s == s_zhongyu {
                row[2] = 9.0; // ADV (`終於`)
                row[17] = 4.5; // raw root logit 4.5, penalized by -2.5 -> 2.0!
                row[17 + s_jibai] = 3.5;
            } else if s == s_jitui {
                row[15] = 9.0; // VERB immediately before `的對手` (prenominal relative clause)
                row[17] = 4.3; // raw root logit 4.3, penalized by -2.5 -> 1.8!
                row[17 + s_jibai] = 3.5;
            } else if s == s_jibai {
                row[15] = 9.0; // Main-clause matrix VERB `擊敗`
                row[17] = 3.0; // Wins cleanly (`3.0 > 2.0, 1.8, 1.7`)!
            } else {
                row[7] = 9.0; // NOUN
                row[17 + s_jibai] = 8.0;
            }
            row[17 + max_len + 1] = 9.0; // "advcl"
        }
    });

    annotate_with_encoder(&mut doc_root, s_root, &root_model, char_subword_encode).unwrap();
    let root_syn = doc_root
        .syntax
        .iter()
        .find(|s| s.head.is_none())
        .expect("single matrix root");
    let root_surface: String = s_root
        .chars()
        .skip(root_syn.span.start_char)
        .take(root_syn.span.end_char - root_syn.span.start_char)
        .collect();
    assert_eq!(
        root_surface, "擊敗",
        "matrix verb `擊敗` must beat subordinate `崎嶇`, adverb `終於`, and relative `擊退`"
    );

    // 2. Clause-local re-attachment (`prob >= 0.22`) and cross-clause `< 0.35` fallback relabeling
    //    (`obj -> dep` on nominals, `compound -> parataxis` on predicates), preserving tree invariants.
    let s_flow = "老者取出木盒，少年收好信件，守衛推開鐵門。";
    let flow_opts: Options = serde_json::from_value(serde_json::json!({
        "normalize_traditional": true,
        "lexicon": ["老者", "取出", "木盒", "少年", "收好", "信件", "守衛", "推開", "鐵門"]
    }))
    .unwrap();
    let mut doc_flow = analyze_document(s_flow, &flow_opts);
    let sub_flow = char_subword_encode(s_flow).unwrap();
    let s_laozhe = first_sub_of("老者", s_flow, &sub_flow);
    let s_quchu = first_sub_of("取出", s_flow, &sub_flow);
    let s_muhe = first_sub_of("木盒", s_flow, &sub_flow);
    let s_shaonian = first_sub_of("少年", s_flow, &sub_flow);
    let s_shouhao = first_sub_of("收好", s_flow, &sub_flow);
    let s_xinjian = first_sub_of("信件", s_flow, &sub_flow);
    let s_tuikai = first_sub_of("推開", s_flow, &sub_flow);

    let flow_model = SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            row[17] = -20.0;
            if s == s_tuikai {
                row[15] = 10.0; // VERB (matrix root in Clause 2)
                row[17] = 12.0;
            } else if s == s_quchu {
                row[15] = 10.0; // VERB in Clause 0 (emitted with high confidence -> `推開`)
                row[17 + s_tuikai] = 10.0;
                row[17 + max_len + 31] = 10.0; // "parataxis"
            } else if s == s_muhe {
                // `木盒` in Clause 0: Chu-Liu/Edmonds initially picks `推開` (Clause 2) with
                // low confidence ~0.31 (< 0.35), while local Clause-0 verb `取出` has ~0.27 (>= 0.22)!
                // Plus high-confidence child `老者 -> 木盒` rescues `木盒` via ancestor closure!
                row[7] = 10.0; // NOUN
                row[17 + s_tuikai] = 1.65;
                row[17 + s_quchu] = 1.50;
                row[17 + max_len + 28] = 10.0; // "obj"
            } else if s == s_laozhe {
                row[7] = 10.0; // High-confidence child of `木盒` so `木盒` is retained via closure
                row[17 + s_muhe] = 12.0;
                row[17 + max_len + 25] = 10.0; // "nmod"
            } else if s == s_shouhao {
                // `收好` (VERB in Clause 1): low-confidence (< 0.35) cross-clause `"compound"` -> `推開`,
                // rescued by `信件`/`少年`; must be relabeled `"compound" -> "parataxis"`!
                row[15] = 10.0; // VERB
                row[17 + s_tuikai] = 1.55;
                row[17 + max_len + 10] = 10.0; // "compound"
            } else if s == s_xinjian {
                // `信件` (NOUN in Clause 1): low-confidence (< 0.35) cross-clause `"obj"` -> `推開`
                // AND local `收好` prob < 0.22 (-5.0); rescued by `少年 -> 信件`; must become `"dep"`!
                row[7] = 10.0; // NOUN
                row[17 + s_tuikai] = 1.55;
                row[17 + s_shouhao] = -5.0;
                row[17 + max_len + 28] = 10.0; // "obj"
            } else if s == s_shaonian {
                // High-confidence child of `信件` AND we also attach `鐵門` etc. to `推開`.
                row[7] = 10.0;
                row[17 + s_xinjian] = 12.0;
                row[17 + max_len + 25] = 10.0;
            } else {
                // Remaining tokens (e.g. `守衛` -> `收好` high conf to retain `收好`, `鐵門`/punct -> `推開`).
                row[7] = 10.0;
                if s == first_sub_of("守衛", s_flow, &sub_flow) {
                    row[17 + s_shouhao] = 12.0;
                } else {
                    row[17 + s_tuikai] = 12.0;
                }
                row[17 + max_len + 28] = 10.0;
            }
        }
    });

    annotate_with_encoder(&mut doc_flow, s_flow, &flow_model, char_subword_encode).unwrap();
    let tok_idx_of = |w: &str| -> usize {
        doc_flow
            .tokens
            .iter()
            .position(|t| {
                s_flow
                    .chars()
                    .skip(t.span.start_char)
                    .take(t.span.end_char - t.span.start_char)
                    .collect::<String>()
                    == w
            })
            .unwrap()
    };
    let syn_of = |w: &str| -> &SyntaxToken {
        let ti = tok_idx_of(w);
        doc_flow.syntax.iter().find(|s| s.token == ti).unwrap()
    };

    // `木盒` re-attached locally to Clause-0 verb `取出` (`prob >= 0.22`)!
    assert_eq!(
        syn_of("木盒").head,
        Some(tok_idx_of("取出")),
        "`木盒` should re-attach locally to `取出` in Clause 0"
    );
    assert_eq!(syn_of("木盒").deprel, "obj");

    // `信件` (`< 0.35` cross-clause nominal without `>= 0.22` local verb) relabeled `"obj" -> "dep"`!
    assert_eq!(syn_of("信件").head, Some(tok_idx_of("推開")));
    assert_eq!(
        syn_of("信件").deprel,
        "dep",
        "cross-clause <0.35 nominal closure arc must not keep `obj`"
    );

    // `收好` (`< 0.35` cross-clause VERB) relabeled `"compound" -> "parataxis"`!
    assert_eq!(syn_of("收好").head, Some(tok_idx_of("推開")));
    assert_eq!(
        syn_of("收好").deprel,
        "parataxis",
        "cross-clause <0.35 verb closure arc must become `parataxis`"
    );
}
