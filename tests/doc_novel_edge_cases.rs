use faberline_nlp::Span;
use faberline_nlp::doc::model::{
    Backend, Batch, Model, ModelError, ModelInfo, Output, Subwords, Tensor, compiled, load,
};
use faberline_nlp::doc::syntax::{DEPRELS, annotate_with_encoder};
use faberline_nlp::doc::{
    Document, EntityKind, OpaqueKind, Options, Reason, RelationKind, RelationLevel, Unknown, Upos,
    analyze_document, analyze_document_with_model,
};
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn slice(text: &str, s: Span) -> String {
    text.chars()
        .skip(s.start_char)
        .take(s.end_char - s.start_char)
        .collect()
}

struct SyntheticSyntaxModel<F> {
    info: ModelInfo,
    fill: F,
}

impl<F> SyntheticSyntaxModel<F>
where
    F: Fn(&[i64], usize, &mut [f32]) + Send + Sync,
{
    fn new(max_len: usize, fill: F) -> Self {
        Self {
            info: ModelInfo {
                id: "doc-syntax-v1".to_owned(),
                layers: vec!["pos".to_owned(), "syntax".to_owned()],
                licence: "MIT".to_owned(),
                max_len,
                pad_id: 1,
                files: Vec::new(),
            },
            fill,
        }
    }
}

impl<F> Model for SyntheticSyntaxModel<F>
where
    F: Fn(&[i64], usize, &mut [f32]) + Send + Sync,
{
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn run(&self, batch: &Batch) -> Result<Output, ModelError> {
        let max = self.info.max_len;
        let cols = 17 + max + 4 * DEPRELS.len();
        let mut logits = Vec::with_capacity(batch.ids.len());
        for seq in &batch.ids {
            if seq.len() > max {
                return Err(ModelError::TooLong {
                    len: seq.len(),
                    max,
                });
            }
            let mut data = vec![0.0_f32; seq.len() * cols];
            (self.fill)(seq, cols, &mut data);
            logits.push(Tensor {
                name: "logits".to_owned(),
                shape: vec![seq.len(), cols],
                data,
            });
        }
        Ok(Output { logits })
    }
}

/// Encodes each scalar as 1 subword, except `U+007F` and `U+FFFD` which normalize to empty spans
/// (reproducing SentencePiece normalization of control/replacement characters).
fn novel_subword_encode(sentence_text: &str) -> Result<Subwords, ModelError> {
    let mut ids = vec![0_i64]; // [CLS]
    let mut spans = vec![Span {
        start_char: 0,
        end_char: 0,
    }];
    for (idx, ch) in sentence_text.chars().enumerate() {
        if matches!(ch, '\u{7f}' | '\u{fffd}') {
            continue;
        }
        ids.push((ch as i64 % 200) + 3);
        spans.push(Span {
            start_char: idx,
            end_char: idx + 1,
        });
    }
    let total = sentence_text.chars().count();
    ids.push(2); // [SEP]
    spans.push(Span {
        start_char: total,
        end_char: total,
    });
    Ok(Subwords { ids, spans })
}

fn make_chain_model(max_len: usize) -> SyntheticSyntaxModel<impl Fn(&[i64], usize, &mut [f32])> {
    SyntheticSyntaxModel::new(max_len, move |ids, cols, data| {
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            row[7] = 10.0; // NOUN
            if s == 1 {
                row[17] = 12.0; // attach first token of window to [CLS] root
            } else {
                row[17 + 1] = 12.0; // attach remaining tokens in window to token 1
            }
            row[17 + max_len + 26] = 12.0; // "nsubj"
        }
    })
}

fn assert_valid_acyclic_ud_forest(doc: &Document) {
    let n_tokens = doc.tokens.len();
    let valid_deprels: HashSet<&str> = DEPRELS.iter().copied().collect();
    let mut by_tok = vec![None; n_tokens];
    for syn in &doc.syntax {
        assert!(syn.token < n_tokens, "token index out of bounds");
        assert_eq!(syn.sentence, doc.tokens[syn.token].sentence);
        assert_eq!(syn.span, doc.tokens[syn.token].span);
        assert!(
            valid_deprels.contains(syn.deprel.as_str()),
            "invalid UD deprel: {}",
            syn.deprel
        );
        match syn.head {
            None => assert_eq!(syn.deprel, "root"),
            Some(h) => {
                assert!(h < n_tokens, "head index out of bounds");
                assert_ne!(h, syn.token, "self-loop on token {}", syn.token);
                assert_eq!(
                    doc.tokens[h].sentence, syn.sentence,
                    "cross-sentence head edge"
                );
                assert_ne!(syn.deprel, "root");
            }
        }
        assert!(
            by_tok[syn.token].replace(syn).is_none(),
            "duplicate syntax entry for token {}",
            syn.token
        );
    }

    // Walk every head chain to guarantee zero directed cycles.
    for start in 0..n_tokens {
        let mut seen = HashSet::new();
        let mut cur = start;
        while let Some(syn) = by_tok[cur] {
            assert!(
                seen.insert(cur),
                "cycle detected in UD graph starting at {start}"
            );
            match syn.head {
                Some(next) => cur = next,
                None => break,
            }
        }
    }
}

#[test]
fn coreml_fast_backend_compiled_and_cli_accepted() {
    assert_eq!(
        compiled(Backend::CoremlFast),
        cfg!(all(feature = "coreml", target_os = "macos"))
    );
    assert_eq!(compiled(Backend::CoremlFast), compiled(Backend::Coreml));

    let repo_models = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    if !compiled(Backend::CoremlFast) {
        assert_eq!(
            load(&repo_models, "test-tiny", Backend::CoremlFast).err(),
            Some(ModelError::NoBackend)
        );
    }

    // Validate missing `doc-syntax-v1` weights cleanly degrade to `no_model` (even when
    // `models/doc-syntax-v1` is staged locally for benchmarks), and also validate staged run.
    let unstage_tmp = tempfile::tempdir().unwrap();
    std::fs::copy(
        repo_models.join("manifest.json"),
        unstage_tmp.path().join("manifest.json"),
    )
    .expect("copy manifest");
    let doc = analyze_document_with_model(
        "小甲走進木屋。",
        &Options::default(),
        unstage_tmp.path(),
        Backend::CoremlFast,
    )
    .expect("missing doc-syntax-v1 weights degrade to no_model");
    assert!(doc.syntax.is_empty());
    assert!(
        doc.unknown
            .iter()
            .any(|u| u.layer == "syntax" && u.reason == Reason::NoModel)
    );
    if compiled(Backend::CoremlFast) && repo_models.join("doc-syntax-v1").exists() {
        let staged_doc = analyze_document_with_model(
            "小甲走進木屋。",
            &Options::default(),
            &repo_models,
            Backend::CoremlFast,
        )
        .expect("staged doc-syntax-v1 runs on CoremlFast");
        assert!(!staged_doc.syntax.is_empty());
    }

    let bin = env!("CARGO_BIN_EXE_nlp_doc");
    let mut child = Command::new(bin)
        .args([
            "--models",
            repo_models.to_str().unwrap(),
            "--backend",
            "coreml-fast",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nlp_doc");
    child
        .stdin
        .take()
        .unwrap()
        .write_all("小甲走進木屋。".as_bytes())
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait nlp_doc");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
#[test]
fn coreml_fast_matches_coreml_on_test_tiny_and_real_model_when_staged() {
    use faberline_nlp::doc::analyze_document_with_loaded_model;
    use faberline_nlp::doc::model::{SubwordTokenizer, argmax_labels, load_verified, verify};
    use std::time::Instant;

    let repo_models = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    let verified = verify(&repo_models, "test-tiny").expect("verify test-tiny");
    let cpu_model = load_verified(&verified, Backend::Coreml).expect("load Coreml");
    let fast_model = load_verified(&verified, Backend::CoremlFast).expect("load CoremlFast");
    let tok = SubwordTokenizer::from_verified(&verified).expect("tokenizer");

    for sample in [
        "Mountain fog drifted across the stone bridge.",
        "旅人沿著溪谷緩步前行，遠處燈火微明。",
        "Mixed run: 3個時辰 and 20 meters.",
    ] {
        let sub = tok.encode(sample).expect("encode");
        let batch = Batch { ids: vec![sub.ids] };
        let out_cpu = cpu_model.run(&batch).expect("run cpu");
        let out_fast = fast_model.run(&batch).expect("run fast");
        assert_eq!(out_cpu.logits[0].shape, out_fast.logits[0].shape);
        assert_eq!(
            argmax_labels(&out_cpu.logits[0]),
            argmax_labels(&out_fast.logits[0])
        );
    }

    // Exercise multi-worker thread safety and Objective-C autoreleasepool stability.
    std::thread::scope(|s| {
        for _ in 0..4 {
            let m = &fast_model;
            s.spawn(move || {
                let batch = Batch {
                    ids: vec![vec![5, 9, 13, 21, 34, 55]],
                };
                for _ in 0..25 {
                    let out = m.run(&batch).expect("threaded CoremlFast run");
                    assert_eq!(out.logits[0].shape, vec![6, 5]);
                }
            });
        }
    });

    // Optional staged `doc-syntax-v1` parity & throughput check when `NLP_SYNTAX_MODELS_DIR` is set.
    if let Ok(staged_dir) = std::env::var("NLP_SYNTAX_MODELS_DIR") {
        let staged_path = PathBuf::from(staged_dir);
        if let Ok(v_syn) = verify(&staged_path, "doc-syntax-v1") {
            let syn_cpu = load_verified(&v_syn, Backend::Coreml).expect("load doc-syntax-v1 cpu");
            let syn_fast =
                load_verified(&v_syn, Backend::CoremlFast).expect("load doc-syntax-v1 fast");
            let syn_tok = SubwordTokenizer::from_verified(&v_syn).expect("load syntax tokenizer");

            let corpus = [
                "晨霧尚未散盡，旅人背起行囊穿過石橋，沿著蜿蜒山徑往北邊的驛站前進。",
                "“這么晚了，他心里還是記掛著家里的事，并不是不想休息。”同伴輕聲說道，然后添了幾根木柴。",
                "若是明日天色放晴，那么觀測隊便在兩個時辰內啟程，先越過三十丈寬的淺灘，接著在山腰紮營。",
                "清風吹過松林，溪水拍打著青石，遠處鐘聲緩緩迴盪，旅人停下腳步，抬頭望向層層疊疊的雲海，\
                 隨即取出地圖仔細比對方向，確認前方的岔路通往山谷深處的舊木屋，然後收起行囊，整理好衣袖，\
                 邁步踏上濕潤的石階，一路穿過竹林與溪橋，終於在日落之前抵達了目的地。",
            ];

            let t0 = Instant::now();
            let mut cpu_docs = Vec::new();
            for text in &corpus {
                for _ in 0..5 {
                    cpu_docs.push(
                        analyze_document_with_loaded_model(
                            text,
                            &Options::default(),
                            syn_cpu.as_ref(),
                            &syn_tok,
                        )
                        .expect("cpu doc"),
                    );
                }
            }
            let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let mut fast_docs = Vec::new();
            for text in &corpus {
                for _ in 0..5 {
                    fast_docs.push(
                        analyze_document_with_loaded_model(
                            text,
                            &Options::default(),
                            syn_fast.as_ref(),
                            &syn_tok,
                        )
                        .expect("fast doc"),
                    );
                }
            }
            let fast_ms = t1.elapsed().as_secs_f64() * 1000.0;

            let (mut total_tok, mut upos_eq, mut total_syn, mut syn_eq) =
                (0usize, 0usize, 0usize, 0usize);
            for (dc, df) in cpu_docs.iter().zip(&fast_docs) {
                assert_valid_acyclic_ud_forest(df);
                for (tc, tf) in dc.tokens.iter().zip(&df.tokens) {
                    total_tok += 1;
                    if tc.upos == tf.upos {
                        upos_eq += 1;
                    }
                }
                let fast_map: std::collections::HashMap<usize, (&Option<usize>, &str)> = df
                    .syntax
                    .iter()
                    .map(|s| (s.token, (&s.head, s.deprel.as_str())))
                    .collect();
                for sc in &dc.syntax {
                    total_syn += 1;
                    if fast_map.get(&sc.token) == Some(&(&sc.head, sc.deprel.as_str())) {
                        syn_eq += 1;
                    }
                }
            }
            let upos_rate = upos_eq as f64 / total_tok.max(1) as f64;
            let syn_rate = syn_eq as f64 / total_syn.max(1) as f64;
            eprintln!(
                "[coreml-fast bench] 20 docs: Coreml={cpu_ms:.1}ms, CoremlFast={fast_ms:.1}ms ({:.2}x), UPOS agree={:.2}%, Syntax agree={:.2}%",
                cpu_ms / fast_ms.max(0.001),
                upos_rate * 100.0,
                syn_rate * 100.0
            );
            assert!(upos_rate >= 0.99, "UPOS agreement too low: {upos_rate}");
            assert!(syn_rate >= 0.99, "Syntax agreement too low: {syn_rate}");
        }
    }
}

#[test]
fn long_multi_clause_novel_sentence_windows_along_clause_boundaries() {
    // 22 comma-separated clauses (>320 scalar chars -> >320 subwords > 256 max_len).
    let clauses = [
        "清晨的山谷還籠罩在薄霧之中",
        "旅人背著沉重的行囊走出木屋",
        "沿著蜿蜒曲折的石階緩步下山",
        "穿過一片寂靜無聲的松樹林",
        "聽見溪水拍打岩石的低沉聲響",
        "遠處的山嶺逐漸露出明亮的輪廓",
        "微涼的晨風吹散了林間的水氣",
        "幾隻飛鳥從枝頭驚起掠過天際",
        "他在石橋旁邊停下腳步稍作歇息",
        "取出隨身攜帶的水壺喝了兩口清水",
        "又打開舊地圖仔細核對前方的岔路",
        "確認北面的古道仍然可以順利通行",
        "隨後重新整理好肩上的繩索與包裹",
        "邁開步伐踏上鋪滿落葉的山間小徑",
        "一路翻過兩座地勢平緩的矮丘",
        "穿過荒廢已久的舊驛站與石亭",
        "沿途紀錄下三處清澈的山泉位置",
        "避開了兩段鬆動濕滑的碎石陡坡",
        "繞過橫跨溪流的古老木製水車",
        "聽見遠處村落傳來的陣陣鐘聲",
        "望見遠方小鎮升起的縷縷炊煙",
        "終於在正午之前抵達了河岸邊的渡口。",
    ];
    let long_sentence = clauses.join("，");
    assert!(long_sentence.chars().count() > 300);

    for max_len in [256usize, 48usize] {
        let mut doc = analyze_document(&long_sentence, &Options::default());
        assert_eq!(doc.segmentation.sentences.len(), 1);
        assert_eq!(doc.segmentation.clauses.len(), clauses.len());
        annotate_with_encoder(
            &mut doc,
            &long_sentence,
            &make_chain_model(max_len),
            novel_subword_encode,
        )
        .expect("windowed syntax annotation");

        assert_eq!(
            doc.syntax.len(),
            doc.tokens.len(),
            "all tokens must decode for max_len={max_len}"
        );
        assert!(doc.tokens.iter().all(|t| t.upos.is_some()));
        assert!(
            !doc.unknown
                .iter()
                .any(|u| u.layer == "syntax" && u.reason == Reason::NotApplicable),
            "unexpected NotApplicable for max_len={max_len}"
        );
        assert_valid_acyclic_ud_forest(&doc);
    }
}

#[test]
fn mega_clause_splits_on_token_boundaries_and_monster_token_emits_not_applicable() {
    // Half-width ASCII commas with no spaces stay a single Layer-1 clause; token-boundary
    // fallback must still partition >256 subwords cleanly without emitting NotApplicable.
    let segments = [
        "旅人推開沉重的木門走進寬敞的石室",
        "石室中央擺放著一張斑駁的長木桌",
        "桌面整齊排列著十幾冊泛黃的舊卷宗",
        "牆角立著兩排高大的木製書架",
        "微弱的燈火在夜風裡輕輕搖晃",
        "他在木椅上坐下翻開第一冊筆記",
        "仔細閱讀前人留下的觀測紀錄",
        "逐行比對山川地形與水文變化",
        "將關鍵的河道改道年份抄入紙頁",
        "又核對了三張手繪的峽谷剖面圖",
        "把缺漏的里程數字逐一補充完整",
        "重新歸類散落在木匣裡的舊信件",
        "直到窗外漸漸露出魚肚白的天色",
        "才合上書冊起身推開北側的木窗",
        "迎著清晨的涼風深深吸了一口氣",
        "隨即收拾行裝準備繼續前往下一站",
        "沿著河谷一路向東記錄沿途的里程",
        "在傍晚時分順利抵達山腳下的村落",
        "向守門的老人詢問通往港口的道路",
        "最後在客棧安頓下來整理當日見聞。",
    ];
    let ascii_comma_sentence = segments.join(",");
    assert!(ascii_comma_sentence.chars().count() > 280);

    let mut doc = analyze_document(&ascii_comma_sentence, &Options::default());
    assert_eq!(doc.segmentation.sentences.len(), 1);
    assert_eq!(doc.segmentation.clauses.len(), 1);
    annotate_with_encoder(
        &mut doc,
        &ascii_comma_sentence,
        &make_chain_model(256),
        novel_subword_encode,
    )
    .expect("mega-clause token-boundary split");
    assert_eq!(doc.syntax.len(), doc.tokens.len());
    assert!(
        !doc.unknown
            .iter()
            .any(|u| u.layer == "syntax" && u.reason == Reason::NotApplicable)
    );
    assert_valid_acyclic_ud_forest(&doc);

    // Single monster token exceeding `max_len - 2` (20-char ASCII word with `max_len = 12`).
    let monster = "Supercalifragilisticexpialidocious walks. 小甲走了。";
    let mut doc2 = analyze_document(monster, &Options::default());
    assert_eq!(doc2.segmentation.sentences.len(), 2);
    annotate_with_encoder(
        &mut doc2,
        monster,
        &make_chain_model(12),
        novel_subword_encode,
    )
    .expect("monster token handled cleanly");
    assert!(doc2.unknown.contains(&Unknown {
        layer: "syntax".to_owned(),
        span: doc2.segmentation.sentences[0].span,
        reason: Reason::NotApplicable,
    }));
    assert!(doc2.syntax.iter().all(|s| s.sentence == 1));
    assert!(!doc2.syntax.is_empty());
}

#[test]
fn unmapped_control_and_replacement_char_tokens_preserve_sentence_syntax() {
    let text = "旅人推開木門\u{fffd}走進石室\u{7f}點燃油燈。";
    let mut doc = analyze_document(text, &Options::default());
    assert_eq!(doc.segmentation.sentences.len(), 1);

    annotate_with_encoder(&mut doc, text, &make_chain_model(256), novel_subword_encode)
        .expect("annotate sentence with U+FFFD and U+007F");

    let unmapped_spans: Vec<String> = doc
        .unknown
        .iter()
        .filter(|u| u.layer == "syntax" && u.reason == Reason::NotApplicable)
        .map(|u| slice(text, u.span))
        .collect();
    assert_eq!(unmapped_spans, vec!["\u{fffd}", "\u{7f}"]);
    assert_eq!(doc.syntax.len(), doc.tokens.len() - 2);
    assert_valid_acyclic_ud_forest(&doc);
}

#[test]
fn chinese_headings_cover_thousands_myriads_nbsp_book_prefixes_and_volume_headers() {
    let text = [
        "第一千二百三十四章 山谷晨霧",
        "正文第一段。",
        "第兩千四百四十六章 遠行(終篇)",
        "正文第二段。",
        "第两千零五章 夜雨",
        "第一萬章 新紀元",
        "第兩百二十六 章 石橋",
        "某長篇故事\u{00a0}第一千零八章 古道",
        "某長篇故事\u{00a0}第一千八百八十三章古道（二）",
        "第五卷 遠山行 第七百三十六章 渡河",
        "一百三十三章 舊友",
        "最後一段正文。",
    ]
    .join("\n\n");

    let doc = analyze_document(&text, &Options::default());
    let headings: Vec<String> = doc
        .segmentation
        .opaque
        .iter()
        .filter(|o| o.kind == OpaqueKind::Heading)
        .map(|o| slice(&text, o.span))
        .collect();
    assert_eq!(
        headings,
        vec![
            "第一千二百三十四章 山谷晨霧",
            "第兩千四百四十六章 遠行(終篇)",
            "第两千零五章 夜雨",
            "第一萬章 新紀元",
            "第兩百二十六 章 石橋",
            "某長篇故事\u{00a0}第一千零八章 古道",
            "某長篇故事\u{00a0}第一千八百八十三章古道（二）",
            "第五卷 遠山行 第七百三十六章 渡河",
            "一百三十三章 舊友",
        ]
    );
    let sentences: Vec<String> = doc
        .segmentation
        .sentences
        .iter()
        .map(|s| slice(&text, s.span))
        .collect();
    assert_eq!(
        sentences,
        vec!["正文第一段。", "正文第二段。", "最後一段正文。"]
    );

    // Prose lines that mention chapters/sections must never become headings.
    for prose in [
        "第一千二百章講述了資料整理方法。",
        "第三章的內容",
        "請看 第三章的說明",
        "一章一節都要仔細校對",
    ] {
        let d = analyze_document(prose, &Options::default());
        assert!(
            d.segmentation.opaque.is_empty(),
            "prose falsely matched as heading: {prose}"
        );
    }
}

#[test]
fn hybrid_traditional_simplified_and_novel_entities_relations() {
    let text = "“這么晚了，他心里還是記掛著家里的事，并不是不想休息。”\
旅人嘆了口氣，然后轉身離開。若是明日放晴，那么我們便在三個時辰內出發，走過三十丈寬的石橋，喝一盞茶再走三尺路。";
    let mut doc = analyze_document(text, &Options::default());
    assert_eq!(doc.segmentation.sentences.len(), 3);

    let tok_texts: Vec<String> = doc.tokens.iter().map(|t| slice(text, t.span)).collect();
    for expected_word in ["這么晚", "心里", "家里", "然后", "那么"] {
        assert!(
            tok_texts.iter().any(|t| t == expected_word),
            "missing hybrid token {expected_word:?} in {tok_texts:?}"
        );
    }

    // Relations: `然后` (Sequence) and `若是 ... 那么` (Condition with correlate `那么`).
    let rels: Vec<(RelationKind, RelationLevel, String, Option<String>)> = doc
        .relations
        .iter()
        .map(|r| {
            (
                r.kind,
                r.level,
                slice(text, r.connector),
                r.correlate.map(|c| slice(text, c)),
            )
        })
        .collect();
    assert_eq!(
        rels,
        vec![
            (
                RelationKind::Sequence,
                RelationLevel::Clause,
                "然后".to_owned(),
                None
            ),
            (
                RelationKind::Condition,
                RelationLevel::Clause,
                "若是".to_owned(),
                Some("那么".to_owned())
            ),
        ]
    );

    // Classical & Simplified quantities.
    let quantities: Vec<(String, Option<String>, Option<String>)> = doc
        .entities
        .iter()
        .filter(|e| e.kind == EntityKind::Quantity)
        .map(|e| (slice(text, e.span), e.value.clone(), e.unit.clone()))
        .collect();
    assert_eq!(
        quantities,
        vec![
            (
                "三個時辰".to_owned(),
                Some("3".to_owned()),
                Some("個時辰".to_owned())
            ),
            (
                "三十丈".to_owned(),
                Some("30".to_owned()),
                Some("丈".to_owned())
            ),
            (
                "一盞茶".to_owned(),
                Some("1".to_owned()),
                Some("盞茶".to_owned())
            ),
            (
                "三尺".to_owned(),
                Some("3".to_owned()),
                Some("尺".to_owned())
            ),
        ]
    );

    // Ensure `若干` is still rejected as a non-connector while `因为`, `接着`, `随后`, `并且`, `同时` work.
    let extra = "因为山路濕滑，所以隊伍放慢腳步，接着穿過竹林，随后抵達營地，并且生起篝火。同时，哨兵留守路口。若干問題仍待確認。";
    let extra_doc = analyze_document(extra, &Options::default());
    let extra_conns: Vec<String> = extra_doc
        .relations
        .iter()
        .map(|r| slice(extra, r.connector))
        .collect();
    assert_eq!(extra_conns, vec!["因为", "接着", "随后", "并且"]);
    let ambig: Vec<String> = extra_doc
        .unknown
        .iter()
        .filter(|u| u.layer == "relations" && u.reason == Reason::Ambiguous)
        .map(|u| slice(extra, u.span))
        .collect();
    assert_eq!(ambig, vec!["同时"]);

    // Layer 4 contextual UPOS upgrade on hybrid Traditional+Simplified sentence.
    annotate_with_encoder(&mut doc, text, &make_chain_model(256), novel_subword_encode)
        .expect("annotate hybrid doc");
    assert!(doc.tokens.iter().all(|t| t.upos.is_some()));
}

#[test]
fn ws1_ws2_ws3_opaque_t2s_degluing_and_classical_pronouns_synthetic() {
    // 1. WS1: HTML `<P`, `&qut;`, PUA run, box-drawing banner, missing-`章` heading, and `mask_editorial`.
    let raw_ch = concat!(
        "┏━━━━━━━━━━┓\n",
        "┃ 合成測試橫幅 ┃\n",
        "┗━━━━━━━━━━┛\n\n",
        "第一千六百九十六 古道夜行\n\n",
        "<P在下與妾身帶著築基丹走進石室，&qut;老夫已等候多時&qut;。\u{e101}\u{e102}\u{e103}\n\n",
        "他將那青雲石盤放在木桌上，對此毫不在意，當石門緩緩合上時，外頭還在下雨。\n\n",
        "（月底最後一天，求下月票與推薦票支持！）\n"
    );
    let opts = Options {
        mask_editorial: true,
        normalize_traditional: true,
        lexicon: vec!["青雲石盤".to_owned()],
        ..Options::default()
    };
    let doc = analyze_document(raw_ch, &opts);

    // Heading `第一千六百九十六 古道夜行` masked as Heading; box banner, `<P`, `&qut;`, PUA, and vote plea masked!
    assert!(
        doc.segmentation
            .opaque
            .iter()
            .any(|o| o.kind == OpaqueKind::Heading
                && slice(raw_ch, o.span) == "第一千六百九十六 古道夜行")
    );
    let tok_surfaces: Vec<String> = doc.tokens.iter().map(|t| slice(raw_ch, t.span)).collect();
    assert!(
        !tok_surfaces
            .iter()
            .any(|t| t.contains("月票") || t.contains("橫幅")),
        "box banner and editorial postscript must not leak into tokens: {tok_surfaces:?}"
    );
    // `<P` and `&qut;` are inline opaque `Code` tokens (`upos == Some(Upos::X)`), never glued to `在下` or `老夫`.
    for expected in [
        "在下",
        "妾身",
        "築基丹",
        "老夫",
        "將",
        "那",
        "青雲石盤",
        "對",
        "此",
        "當",
        "在",
        "下雨",
    ] {
        assert!(
            tok_surfaces.iter().any(|t| t == expected),
            "expected token {expected:?} in {tok_surfaces:?}"
        );
    }

    // Classical pronouns (`在下`, `妾身`, `老夫`) get `Upos::Pron` in Layer 2.
    for pron in ["在下", "妾身", "老夫"] {
        let tok = doc
            .tokens
            .iter()
            .find(|t| slice(raw_ch, t.span) == pron)
            .unwrap_or_else(|| panic!("missing pronoun token {pron}"));
        assert_eq!(tok.upos, Some(Upos::Pron), "{pron} should be Upos::Pron");
    }

    // Verify `nlp_doc` CLI exposes `--mask-editorial`, `--normalize-traditional`, `--lexicon`, and `--name`.
    let bin = env!("CARGO_BIN_EXE_nlp_doc");
    let mut child = Command::new(bin)
        .args([
            "--mask-editorial",
            "--normalize-traditional",
            "--lexicon",
            "青雲石盤",
            "--name",
            "沈星河",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nlp_doc");
    child
        .stdin
        .take()
        .unwrap()
        .write_all("沈星河將青雲石盤收起。\n（求下月票！）\n".as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let cli_doc: Document = serde_json::from_slice(&out.stdout).unwrap();
    assert!(cli_doc.options.mask_editorial);
    assert!(cli_doc.options.normalize_traditional);
    assert_eq!(cli_doc.entities.len(), 1);
    assert_eq!(cli_doc.entities[0].kind, EntityKind::Name);
}

#[test]
fn ws4_ws6_novel_entities_and_narrative_relations_synthetic() {
    // WS4: Chinese numeral durations, fractional `半`, approximate `數`/`余` quantities,
    // relative duration anchors (`四年后`), moment cues (`片刻`), and calendar `2026年` preservation.
    let ent_text = "四年后，隊伍在山谷停留了四年與數日，走了百余丈山路，等候半個時辰與一炷香，片刻便在2026年春啟程。";
    let ent_doc = analyze_document(ent_text, &Options::default());
    let ents: Vec<(EntityKind, String, Option<String>, Option<String>)> = ent_doc
        .entities
        .iter()
        .map(|e| {
            (
                e.kind,
                slice(ent_text, e.span),
                e.value.clone(),
                e.unit.clone(),
            )
        })
        .collect();
    assert_eq!(
        ents,
        vec![
            (EntityKind::RelativeDate, "四年后".to_owned(), None, None),
            (
                EntityKind::Quantity,
                "四年".to_owned(),
                Some("4".to_owned()),
                Some("年".to_owned()),
            ),
            (
                EntityKind::Quantity,
                "數日".to_owned(),
                None,
                Some("日".to_owned()),
            ),
            (
                EntityKind::Quantity,
                "百余丈".to_owned(),
                None,
                Some("丈".to_owned()),
            ),
            (
                EntityKind::Quantity,
                "半個時辰".to_owned(),
                Some("0.5".to_owned()),
                Some("個時辰".to_owned()),
            ),
            (
                EntityKind::Quantity,
                "一炷香".to_owned(),
                Some("1".to_owned()),
                Some("炷香".to_owned()),
            ),
            (EntityKind::RelativeDate, "片刻".to_owned(), None, None),
            (
                EntityKind::Date,
                "2026年".to_owned(),
                Some("2026".to_owned()),
                None,
            ),
        ]
    );

    // WS6: `隨即`, `與此同時`, `不過` (while `不過如此` is not a connector), `否則`,
    // `先...旋即...`, and post-positional temporal frames (`...之後，...`, `...之際，...`).
    let rel_text = concat!(
        "旅人收起地圖，隨即邁步下山，與此同時同伴點亮火把，不過風勢漸大，否則早已抵達渡口。",
        "那座石橋不過如此。",
        "先檢查繩索，旋即跨過溪流。",
        "整理好行裝之後，眾人踏上古道。",
        "燈火搖曳之際，門外傳來腳步聲。"
    );
    let rel_doc = analyze_document(rel_text, &Options::default());
    let rels: Vec<(RelationKind, String, Option<String>)> = rel_doc
        .relations
        .iter()
        .map(|r| {
            (
                r.kind,
                slice(rel_text, r.connector),
                r.correlate.map(|c| slice(rel_text, c)),
            )
        })
        .collect();
    assert_eq!(
        rels,
        vec![
            (RelationKind::Sequence, "隨即".to_owned(), None),
            (RelationKind::Parallel, "與此同時".to_owned(), None),
            (RelationKind::Contrast, "不過".to_owned(), None),
            (RelationKind::Condition, "否則".to_owned(), None),
            (
                RelationKind::Sequence,
                "先".to_owned(),
                Some("旋即".to_owned()),
            ),
            (RelationKind::Sequence, "之後".to_owned(), None),
            (RelationKind::Sequence, "之際".to_owned(), None),
        ]
    );
}

#[test]
fn ws3_pooling_and_ws5_soft_windowing_and_tree_retention_synthetic() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    // 1. WS3-Pooling + `什麼` disambiguation:
    //    `一清二楚` (4 subwords) has subword #0 biased toward `NUM` (6.0), while subwords #1..#3
    //    strongly favor `ADV` (9.0) -> mean-pooling yields `ADV` instead of `NUM`!
    //    `什麼` before `辦法` (`NOUN`) becomes `DET`, while `什麼` before `。` becomes `PRON`.
    let pool_text = "他把路線看得一清二楚，想出什麼辦法，卻沒說什麼。";
    let mut pool_doc = analyze_document(
        pool_text,
        &Options {
            lexicon: vec!["一清二楚".to_owned()],
            ..Options::default()
        },
    );
    let sent_chars: Vec<char> = pool_text.chars().collect();
    let pool_model = SyntheticSyntaxModel::new(256, move |ids, cols, data| {
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            let ch = sent_chars.get(s - 1).copied().unwrap_or(' ');
            if ch == '一' {
                row[8] = 6.0; // NUM bias on leading `一` subword
                row[2] = 1.0; // ADV
            } else if matches!(ch, '清' | '二' | '楚') {
                row[2] = 10.0; // ADV on remaining subwords of `一清二楚`
            } else if matches!(ch, '麼') || ch == '什' {
                row[9] = 9.0; // PART -> triggers `什麼` contextual disambiguation
            } else {
                row[7] = 9.0; // NOUN (including `辦法`)
            }
            if s == 1 {
                row[17] = 10.0;
            } else {
                row[17 + 1] = 10.0;
            }
            row[17 + 256 + 26] = 10.0;
        }
    });
    annotate_with_encoder(&mut pool_doc, pool_text, &pool_model, novel_subword_encode)
        .expect("annotate pooling doc");

    let idiom_tok = pool_doc
        .tokens
        .iter()
        .find(|t| slice(pool_text, t.span) == "一清二楚")
        .expect("idiom token `一清二楚`");
    assert_eq!(idiom_tok.upos, Some(Upos::Adv));

    let shenme_toks: Vec<_> = pool_doc
        .tokens
        .iter()
        .filter(|t| slice(pool_text, t.span) == "什麼")
        .collect();
    assert_eq!(shenme_toks.len(), 2);
    assert_eq!(shenme_toks[0].upos, Some(Upos::Det));
    assert_eq!(shenme_toks[1].upos, Some(Upos::Pron));

    // 2. WS5 Soft clause-group windowing (`clauses >= 4`, `tokens > 64`, `subwords <= 256`):
    //    10 clauses (~130 chars < 256 max_len, ~75 tokens > 64) must soft-window into >= 2 calls
    //    and attach window `k >= 1` local roots to window `0` root via `deprel = "parataxis"`.
    let paratactic_clauses = [
        "清晨的山風吹散了溪谷薄霧",
        "旅人背起行囊踏上青石古道",
        "沿著蜿蜒山脊一路往北前行",
        "穿過寂靜無聲的松林與竹海",
        "在石亭旁邊停下腳步喝口水",
        "打開地圖仔細核對前方岔路",
        "確認通往山腰驛站的舊標記",
        "重新收好水壺與隨身筆記本",
        "繞過兩處濕滑陡峭的碎石坡",
        "終於在午時之前抵達了木屋。",
    ];
    let paratactic_sent = paratactic_clauses.join("，");
    assert!(paratactic_sent.chars().count() < 240);
    let mut soft_doc = analyze_document(&paratactic_sent, &Options::default());
    assert!(soft_doc.segmentation.clauses.len() >= 4);
    assert!(soft_doc.tokens.len() > 64);

    let calls = AtomicUsize::new(0);
    let counting_model = SyntheticSyntaxModel::new(256, |ids, cols, data| {
        calls.fetch_add(1, Ordering::SeqCst);
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            row[7] = 10.0;
            if s == 1 {
                row[17] = 12.0;
            } else {
                row[17 + 1] = 12.0;
            }
            row[17 + 256 + 26] = 12.0;
        }
    });
    annotate_with_encoder(
        &mut soft_doc,
        &paratactic_sent,
        &counting_model,
        novel_subword_encode,
    )
    .expect("soft-windowed syntax");
    assert!(
        calls.load(Ordering::SeqCst) >= 2,
        "expected soft windowing to invoke >= 2 windows, got {}",
        calls.load(Ordering::SeqCst)
    );
    assert_valid_acyclic_ud_forest(&soft_doc);
    let roots: Vec<_> = soft_doc
        .syntax
        .iter()
        .filter(|s| s.head.is_none())
        .collect();
    assert_eq!(roots.len(), 1, "sentence must have strictly one root");
    assert!(
        soft_doc
            .syntax
            .iter()
            .any(|s| s.deprel == "parataxis" && s.head == Some(roots[0].token)),
        "subsequent window roots must attach to window 0 root with parataxis"
    );

    // 3. WS5 Tree-consistent root & ancestor retention:
    //    `short_sent = "旅人推開門。"` -> tokens: 0="旅人" (s_0=1), 1="推開" (s_0=3), 2="門" (s_0=5), 3="。" (s_0=6).
    //    Token 0 is the decoded root with low confidence (logit margin small -> score < 0.35),
    //    Token 1 attaches to Token 0 (`s_v = 1`) with moderate arc prob (~0.31, clearing ancestor
    //    floor 0.12 but below 0.35), and Token 2 attaches to Token 1 (`s_v = 3`) with high confidence (> 0.95).
    //    Retention must keep {Token 0 (root), Token 1 (ancestor), Token 2 (high-conf child)} ->
    //    0 zero-root sentences and 0 orphan arcs!
    let short_sent = "旅人推開門。";
    let mut tree_doc = analyze_document(short_sent, &Options::default());
    let low_root_model = SyntheticSyntaxModel::new(256, |ids, cols, data| {
        for s in 1..ids.len().saturating_sub(1) {
            let row = &mut data[s * cols..(s + 1) * cols];
            row[15] = 9.0; // VERB
            if s == 1 {
                // Token 0 ("旅人"): slightly prefers [CLS] root (`s_v = 0`), p_arc ~ 0.24 < 0.35.
                row[17] = 0.25;
            } else if s == 3 {
                // Token 1 ("推開"): prefers head Token 0 (`s_v = 1`) with p_arc ~ 0.31 (< 0.35, >= 0.12 floor).
                row[17] = -5.0;
                row[17 + 1] = 0.30;
            } else if s == 5 {
                // Token 2 ("門"): strongly attaches to Token 1 (`s_v = 3`) with confidence > 0.99!
                row[17] = -5.0;
                row[17 + 3] = 12.0;
            } else {
                // Token 3 ("。") and non-initial subwords: uniform low confidence (< 0.35) -> omitted cleanly.
                row[17] = -5.0;
            }
            row[17 + 256 + 29] = 12.0; // high p_rel on "obj"
        }
    });
    annotate_with_encoder(
        &mut tree_doc,
        short_sent,
        &low_root_model,
        novel_subword_encode,
    )
    .expect("tree retention decode");
    assert!(!tree_doc.syntax.is_empty());
    assert_valid_acyclic_ud_forest(&tree_doc);
    let emitted_tokens: HashSet<usize> = tree_doc.syntax.iter().map(|s| s.token).collect();
    assert_eq!(
        tree_doc.syntax.iter().filter(|s| s.head.is_none()).count(),
        1,
        "must retain global root (0 zero-root sentences)"
    );
    for syn in &tree_doc.syntax {
        if let Some(h) = syn.head {
            assert!(
                emitted_tokens.contains(&h),
                "orphan syntax head {h} on token {}",
                syn.token
            );
        }
    }
}

#[test]
fn phase4_v3_end_to_end_ws1_to_ws6_synthetic() {
    // End-to-end synthetic chapter exercising Phase 4 (`v3`) WS1–WS6 together:
    // - WS1-v3: tight heading `第一千零八章古道試煉`, unclosed `</P` & `<!go`, mixed-bracket editorial
    //   postscript `(月底最後一更，求下月票支持！）`, and pirate footer line (`silent_noise` emits 0 `Upos::X`
    //   tokens and 0 intra-line double-space splits; night-watch `三更半夜` stays normal prose).
    // - WS2-v3: scraper infixes `少丅女` & `長堊老` stay intact 3-char tokens on original offsets;
    //   `其見` and `中含` de-glue cleanly.
    // - WS3-v3: `終於` -> `ADV`, `一旦` -> `SCONJ`, `一齊` -> `ADV`, `除了` -> `ADP`; Layer-4 contextual
    //   `將` coverb -> `ADP` and `>= 3`-char `options.lexicon` compound `玄雲半天鏡` -> `NOUN` with
    //   scraper-infix subword projection.
    // - WS4-v3: interior `半天` inside `玄雲半天鏡`, `九天之上`, `退避三尺`, and bare `眨眼` emit 0 false
    //   entities, while `眨眼工夫` (`RelativeDate`) and `七八丈` (`Quantity` with `value = "7-8"`) fire.
    // - WS5-v3: multi-clause `雖然...` subordinate root-stealing guard promotes main-clause predicate
    //   `取出` to `"root"` over subordinate `看見`, preserving an acyclic single-root UD tree.
    // - WS6-v3: spatial `站在石柱之後` and kinship `名門之後` emit 0 false `Sequence` relations, while
    //   temporal `收好地圖之後` (`Sequence`) and clause + sentence `與此同時` (`Parallel`) fire cleanly.
    let raw_text = concat!(
        "第一千零八章古道試煉\n\n",
        "</P雖然少丅女看見了迷霧，長堊老終於將玄雲半天鏡取出。(月底最後一更，求下月票支持！）\n\n",
        "<!go三更半夜裡，除了風聲，一旦機關啟動眾人便一齊退避三尺，其見名門之後站在石柱之後眨眼望向九天之上，石匣中含著古圖。\n\n",
        "收好地圖之後，隊伍眨眼工夫走了七八丈山路，與此同時前哨點亮火把。與此同時，後隊守住山口。\n\n",
        "一一小說 ddyueshu 最新全本，瀏覽貼子與轉貼次數。\n"
    );

    let opts = Options {
        mask_editorial: true,
        normalize_traditional: true,
        lexicon: vec!["玄雲半天鏡".to_owned()],
        ..Options::default()
    };
    let mut doc = analyze_document(raw_text, &opts);

    // 1. WS1-v3: tight heading matched; editorial/HTML/pirate noise masked with zero `Upos::X` noise tokens.
    assert!(doc.segmentation.opaque.iter().any(
        |o| o.kind == OpaqueKind::Heading && slice(raw_text, o.span) == "第一千零八章古道試煉"
    ));
    let tok_surfaces: Vec<String> = doc.tokens.iter().map(|t| slice(raw_text, t.span)).collect();
    for leaked in ["月票", "一一小說", "ddyueshu", "</P", "<!go"] {
        assert!(
            !tok_surfaces.iter().any(|t| t.contains(leaked)),
            "editorial/HTML noise `{leaked}` must be silently masked, got: {tok_surfaces:?}"
        );
    }
    assert!(
        !doc.tokens.iter().any(|t| t.upos == Some(Upos::X)),
        "silent_noise inline masks must emit zero Upos::X tokens, got: {:?}",
        doc.tokens
    );
    assert!(
        tok_surfaces
            .iter()
            .any(|t| t.contains("三更") || t == "三更半夜"),
        "night-watch prose `三更半夜` must remain unmasked: {tok_surfaces:?}"
    );

    // 2. WS2-v3 & WS3-v3 (Layer 2): scraper infixes stay intact on raw spans; `其見`/`中含` de-glue; closed-class UPOS decided.
    for expected_tok in [
        "少丅女",
        "長堊老",
        "玄雲半天鏡",
        "其",
        "見",
        "中",
        "含",
        "終於",
        "除了",
        "一旦",
        "一齊",
    ] {
        assert!(
            tok_surfaces.iter().any(|t| t == expected_tok),
            "expected token {expected_tok:?} in {tok_surfaces:?}"
        );
    }
    let upos_of = |d: &Document, surface: &str| -> Option<Upos> {
        d.tokens
            .iter()
            .find(|t| slice(raw_text, t.span) == surface)
            .and_then(|t| t.upos)
    };
    assert_eq!(upos_of(&doc, "終於"), Some(Upos::Adv));
    assert_eq!(upos_of(&doc, "除了"), Some(Upos::Adp));
    assert_eq!(upos_of(&doc, "一旦"), Some(Upos::Sconj));
    assert_eq!(upos_of(&doc, "一齊"), Some(Upos::Adv));

    // 3. WS4-v3: interior `半天` inside `玄雲半天鏡`, `九天` before `之上`, `退避三尺`, and bare `眨眼` suppressed;
    //    `眨眼工夫` (`RelativeDate`) and `七八丈` (`Quantity`, `"7-8"`) emitted.
    let ents: Vec<(EntityKind, String, Option<String>, Option<String>)> = doc
        .entities
        .iter()
        .map(|e| {
            (
                e.kind,
                slice(raw_text, e.span),
                e.value.clone(),
                e.unit.clone(),
            )
        })
        .collect();
    for forbidden_ent in ["半天", "九天", "三尺", "眨眼"] {
        assert!(
            !ents.iter().any(|(_, s, _, _)| s == forbidden_ent),
            "guarded false-positive entity `{forbidden_ent}` must not fire: {ents:?}"
        );
    }
    assert!(
        ents.contains(&(EntityKind::RelativeDate, "眨眼工夫".to_owned(), None, None)),
        "expected `眨眼工夫` RelativeDate in {ents:?}"
    );
    assert!(
        ents.contains(&(
            EntityKind::Quantity,
            "七八丈".to_owned(),
            Some("7-8".to_owned()),
            Some("丈".to_owned()),
        )),
        "expected `七八丈` Quantity with value `7-8` in {ents:?}"
    );

    // 4. WS6-v3: kinship `名門之後` and spatial `站在石柱之後` do not emit `Sequence`;
    //    temporal `收好地圖之後` (`Sequence`) and clause + sentence `與此同時` (`Parallel`) fire.
    let rels: Vec<(RelationKind, RelationLevel, String, String)> = doc
        .relations
        .iter()
        .map(|r| {
            (
                r.kind,
                r.level,
                slice(raw_text, r.connector),
                slice(raw_text, r.arg1),
            )
        })
        .collect();
    assert!(
        !rels
            .iter()
            .any(|(_, _, conn, a1)| conn == "之後" && (a1.contains("名門") || a1.contains("石柱"))),
        "kinship/spatial `之後` must be suppressed: {rels:?}"
    );
    assert!(
        rels.iter().any(|(k, lvl, conn, a1)| {
            *k == RelationKind::Sequence
                && *lvl == RelationLevel::Clause
                && conn == "之後"
                && a1.contains("收好地圖")
        }),
        "expected temporal `之後` Sequence relation in {rels:?}"
    );
    assert!(
        rels.iter().any(|(k, lvl, conn, _)| {
            *k == RelationKind::Parallel && *lvl == RelationLevel::Clause && conn == "與此同時"
        }),
        "expected clause-level `與此同時` Parallel relation in {rels:?}"
    );
    assert!(
        rels.iter().any(|(k, lvl, conn, _)| {
            *k == RelationKind::Parallel && *lvl == RelationLevel::Sentence && conn == "與此同時"
        }),
        "expected sentence-level `與此同時` Parallel relation in {rels:?}"
    );

    // 5. WS3-v3 & WS5-v3 (Layer 4): scraper infix subword projection, `將` -> `ADP`, lexicon `玄雲半天鏡` -> `NOUN`,
    //    and multi-clause `雖然...` root-stealing guard (`看見` penalized `-2.5`, so main-clause `取出` wins `"root"`).
    let exact_char_encode = |sentence_text: &str| -> Result<Subwords, ModelError> {
        assert!(
            !sentence_text.contains('丅') && !sentence_text.contains('堊'),
            "scraper infixes must be stripped from temporary projected encoder string: {sentence_text:?}"
        );
        let mut ids = vec![0_i64];
        let mut spans = vec![Span {
            start_char: 0,
            end_char: 0,
        }];
        for (idx, ch) in sentence_text.chars().enumerate() {
            ids.push(i64::from(ch as u32) + 10);
            spans.push(Span {
                start_char: idx,
                end_char: idx + 1,
            });
        }
        let total = sentence_text.chars().count();
        ids.push(2);
        spans.push(Span {
            start_char: total,
            end_char: total,
        });
        Ok(Subwords { ids, spans })
    };

    let v3_model = SyntheticSyntaxModel::new(256, |ids, cols, data| {
        let chars: Vec<char> = ids[1..ids.len().saturating_sub(1)]
            .iter()
            .map(|&id| char::from_u32((id - 10) as u32).unwrap_or(' '))
            .collect();
        let main_verb_s = chars
            .iter()
            .position(|&c| c == '取')
            .map(|i| i + 1)
            .unwrap_or(1);

        for (idx, &ch) in chars.iter().enumerate() {
            let s = idx + 1;
            let row = &mut data[s * cols..(s + 1) * cols];
            match ch {
                '看' | '見' => {
                    row[15] = 10.0; // VERB in subordinate `雖然` clause
                    row[17] = 9.0; // higher raw root logit than `取出` (7.8), must be penalized by `-2.5` -> 6.5!
                    row[17 + main_verb_s] = 7.2;
                    row[17 + 256 + 1] = 10.0; // advcl
                }
                '取' | '出' => {
                    row[15] = 10.0; // VERB in main clause
                    row[17] = 7.8; // wins root after `看見` receives `-2.5` subordinate penalty!
                    row[17 + 256 + 34] = 12.0; // root
                }
                '將' => {
                    row[15] = 9.0; // raw VERB -> WS3-v3 coverb prior must disambiguate to ADP
                    row[17] = -5.0;
                    row[17 + main_verb_s] = 10.0;
                    row[17 + 256 + 6] = 10.0; // case
                }
                '玄' | '雲' | '半' | '天' | '鏡' => {
                    row[0] = 9.0; // raw ADJ -> WS3-v3 lexicon compound prior must override to NOUN
                    row[17] = -5.0;
                    row[17 + main_verb_s] = 10.0;
                    row[17 + 256 + 29] = 10.0; // obj
                }
                '，' | '。' => {
                    row[12] = 10.0; // PUNCT
                    row[17] = -5.0;
                    row[17 + main_verb_s] = 10.0;
                    row[17 + 256 + 32] = 10.0; // punct
                }
                _ => {
                    if s == 1 && main_verb_s == 1 {
                        row[15] = 10.0; // VERB root for remaining sentences
                        row[17] = 10.0;
                    } else {
                        row[7] = 10.0; // NOUN
                        row[17] = -5.0;
                        row[17 + main_verb_s] = 10.0;
                    }
                    row[17 + 256 + 26] = 10.0; // nsubj
                }
            }
        }
    });

    annotate_with_encoder(&mut doc, raw_text, &v3_model, exact_char_encode)
        .expect("annotate v3 end-to-end doc");

    assert_valid_acyclic_ud_forest(&doc);
    assert_eq!(
        upos_of(&doc, "將"),
        Some(Upos::Adp),
        "disposal coverb `將` before NP+VP must upgrade to ADP"
    );
    assert_eq!(
        upos_of(&doc, "玄雲半天鏡"),
        Some(Upos::Noun),
        ">=3-char lexicon compound `玄雲半天鏡` must receive NOUN prior"
    );

    // Verify sentence 0's decoded root is main-clause `取出` (not subordinate `看見`) and scraper-infix
    // tokens `少丅女` / `長堊老` keep exact 3-char spans on `raw_text`.
    let sent0_root = doc
        .syntax
        .iter()
        .find(|s| s.sentence == 0 && s.head.is_none())
        .expect("sentence 0 must have a single root");
    assert_eq!(
        slice(raw_text, sent0_root.span),
        "取出",
        "main-clause predicate `取出` must beat subordinate `雖然...看見` for sentence 0 root"
    );
    for infix_surface in ["少丅女", "長堊老"] {
        let syn = doc
            .syntax
            .iter()
            .find(|s| s.sentence == 0 && slice(raw_text, s.span) == infix_surface)
            .unwrap_or_else(|| panic!("missing syntax token for {infix_surface}"));
        assert_eq!(syn.span.end_char - syn.span.start_char, 3);
    }
}
