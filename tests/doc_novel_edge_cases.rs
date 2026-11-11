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

    // `models/` in git holds only `test-tiny`, so `doc-syntax-v1` cleanly reports `no_model`
    // while validating `--backend coreml-fast` CLI flag parsing.
    let doc = analyze_document_with_model(
        "小甲走進木屋。",
        &Options::default(),
        &repo_models,
        Backend::CoremlFast,
    )
    .expect("missing doc-syntax-v1 degrades to no_model");
    assert!(doc.syntax.is_empty());
    assert!(
        doc.unknown
            .iter()
            .any(|u| u.layer == "syntax" && u.reason == Reason::NoModel)
    );

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
