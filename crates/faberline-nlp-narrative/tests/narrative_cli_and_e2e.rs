use faberline_nlp::doc::{Document, analyze_document};
use faberline_nlp_narrative::{
    ActorMention, AliasEntry, BOOK_RECEIPT_SCHEMA, BookEventRef, BookTimelineGraph,
    CHAPTER_NARRATIVE_SCHEMA, CandidateEvent, ChapterNarrative, ChronologyOrder, ContextZone,
    Factuality, NarrativeManifestReceipt, TimelineEdge, WorkVocabulary, analyze_chapter_narrative,
    build_chapter_dag_from_narrative, text_sha256_hex,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use tempfile::tempdir;

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn synthetic_vocab() -> WorkVocabulary {
    WorkVocabulary {
        actors: vec![
            AliasEntry {
                surface: "林遠".to_string(),
                canonical: "lin_yuan".to_string(),
            },
            AliasEntry {
                surface: "陳伯".to_string(),
                canonical: "chen_bo".to_string(),
            },
            AliasEntry {
                surface: "蘇晴".to_string(),
                canonical: "su_qing".to_string(),
            },
        ],
        places: vec![AliasEntry {
            surface: "青石鎮".to_string(),
            canonical: "qingshi_town".to_string(),
        }],
        objects: vec![AliasEntry {
            surface: "古銅鏡".to_string(),
            canonical: "bronze_mirror".to_string(),
        }],
        domain_terms: vec!["青石鎮".to_string(), "古銅鏡".to_string()],
    }
}

#[test]
fn analyze_chapter_narrative_validates_hash_and_produces_acyclic_narrative() {
    let vocab = synthetic_vocab();
    let text = "林遠推開木門，將古銅鏡放入木匣。陳伯低聲說道：「明日動身。」";
    let doc = analyze_document(text, &vocab.to_doc_options());

    let ch = analyze_chapter_narrative(1, Some("第一章 山門".to_string()), text, &doc, &vocab)
        .expect("chapter narrative analysis succeeds");

    assert_eq!(ch.schema, CHAPTER_NARRATIVE_SCHEMA);
    assert_eq!(ch.chapter, 1);
    assert_eq!(ch.title.as_deref(), Some("第一章 山門"));
    assert_eq!(ch.text_chars, text.chars().count());
    assert_eq!(ch.text_sha256, text_sha256_hex(text));
    assert_eq!(ch.event_dag_layers.len(), ch.events.len());
    assert!(!ch.dialogues.is_empty());
    assert!(!ch.events.is_empty());
    assert!(!ch.mentions.is_empty());

    let dag = build_chapter_dag_from_narrative(&ch);
    assert!(!dag.has_cycle());

    // Hash mismatch must bail.
    let mut bad_sha_doc = doc.clone();
    bad_sha_doc.text_sha256 = "0".repeat(64);
    let err = analyze_chapter_narrative(1, None, text, &bad_sha_doc, &vocab)
        .expect_err("hash mismatch should fail");
    assert!(err.to_string().contains("text_sha256"));

    // Char count mismatch must bail.
    let mut bad_chars_doc = doc;
    bad_chars_doc.text_chars += 1;
    let err = analyze_chapter_narrative(1, None, text, &bad_chars_doc, &vocab)
        .expect_err("text_chars mismatch should fail");
    assert!(err.to_string().contains("text_chars"));
}

#[test]
fn book_timeline_graph_across_synthetic_chapters() {
    let vocab = synthetic_vocab();
    let opts = vocab.to_doc_options();

    let ch1_text = "林遠清晨離開青石鎮，隨後抵達山門。";
    let ch2_text = "次日正午，蘇晴在客棧整理書信，並未外出。";
    let ch3_text = "三年前，陳伯曾來過青石鎮。";

    let ch1 = analyze_chapter_narrative(
        1,
        Some("第一章".to_string()),
        ch1_text,
        &analyze_document(ch1_text, &opts),
        &vocab,
    )
    .expect("ch1");
    let ch2 = analyze_chapter_narrative(
        2,
        Some("第二章".to_string()),
        ch2_text,
        &analyze_document(ch2_text, &opts),
        &vocab,
    )
    .expect("ch2");
    let ch3 = analyze_chapter_narrative(
        3,
        Some("第三章 往事回憶".to_string()),
        ch3_text,
        &analyze_document(ch3_text, &opts),
        &vocab,
    )
    .expect("ch3");

    let ch1_actual = ch1
        .events
        .iter()
        .position(|e| e.zone == ContextZone::Narrative && e.factuality == Factuality::Actual)
        .expect("ch1 actual narrative event");
    let ch2_actual = ch2
        .events
        .iter()
        .position(|e| e.zone == ContextZone::Narrative && e.factuality == Factuality::Actual)
        .expect("ch2 actual narrative event");
    let ch3_flashback = ch3
        .events
        .iter()
        .position(|e| e.zone == ContextZone::Narrative && e.factuality == Factuality::Actual)
        .expect("ch3 actual narrative event");

    let book = BookTimelineGraph::from_chapters(&[ch1.clone(), ch2.clone(), ch3.clone()]);

    assert_eq!(
        book.query_order(
            BookEventRef {
                chapter: 1,
                event_index: ch1_actual,
            },
            BookEventRef {
                chapter: 2,
                event_index: ch2_actual,
            },
        ),
        ChronologyOrder::Before
    );
    assert_eq!(
        book.query_order(
            BookEventRef {
                chapter: 2,
                event_index: ch2_actual,
            },
            BookEventRef {
                chapter: 1,
                event_index: ch1_actual,
            },
        ),
        ChronologyOrder::After
    );

    // Chapter 3 opens with a flashback (`三年前`), so cross-chapter boundary must stay Incomparable.
    assert_eq!(
        book.query_order(
            BookEventRef {
                chapter: 2,
                event_index: ch2_actual,
            },
            BookEventRef {
                chapter: 3,
                event_index: ch3_flashback,
            },
        ),
        ChronologyOrder::Incomparable
    );

    // Negated event in Chapter 2 must also be Incomparable.
    if let Some(neg_idx) = ch2
        .events
        .iter()
        .position(|e| e.factuality == Factuality::Negated)
    {
        assert_eq!(
            book.query_order(
                BookEventRef {
                    chapter: 1,
                    event_index: ch1_actual,
                },
                BookEventRef {
                    chapter: 2,
                    event_index: neg_idx,
                },
            ),
            ChronologyOrder::Incomparable
        );
    }
}

#[test]
fn nlp_narrative_doc_raw_stdin_precomputed_doc_json_and_hash_mismatch() {
    let bin = env!("CARGO_BIN_EXE_nlp_narrative_doc");
    let dir = tempdir().expect("tempdir");
    let vocab = synthetic_vocab();
    let vocab_path = dir.path().join("vocab.json");
    fs::write(&vocab_path, serde_json::to_vec_pretty(&vocab).unwrap()).unwrap();

    let text = "林遠走進庭院，將古銅鏡遞給蘇晴。蘇晴點頭說道：「我已收到。」";

    // 1. Raw stdin run
    let mut child = Command::new(bin)
        .args([
            "--chapter",
            "3",
            "--title",
            "第三章 庭院",
            "--vocab",
            vocab_path.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nlp_narrative_doc");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "raw stdin failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let raw_ch: ChapterNarrative =
        serde_json::from_slice(&output.stdout).expect("parse raw stdout");
    assert_eq!(raw_ch.schema, CHAPTER_NARRATIVE_SCHEMA);
    assert_eq!(raw_ch.chapter, 3);
    assert_eq!(raw_ch.title.as_deref(), Some("第三章 庭院"));
    assert_eq!(raw_ch.text_sha256, text_sha256_hex(text));

    // 2. Precomputed bare Document JSON
    let doc = analyze_document(text, &vocab.to_doc_options());
    let doc_json_path = dir.path().join("doc.json");
    fs::write(&doc_json_path, serde_json::to_vec(&doc).unwrap()).unwrap();

    let mut child = Command::new(bin)
        .args([
            "--chapter",
            "3",
            "--title",
            "第三章 庭院",
            "--vocab",
            vocab_path.to_str().unwrap(),
            "--doc-json",
            doc_json_path.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nlp_narrative_doc --doc-json");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "--doc-json failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let precomputed_ch: ChapterNarrative =
        serde_json::from_slice(&output.stdout).expect("parse precomputed stdout");
    assert_eq!(precomputed_ch, raw_ch);

    // 3. Batch wrapper `{ "doc": Document }` JSON
    #[derive(Serialize)]
    struct Wrapper<'a> {
        order: u32,
        doc: &'a Document,
    }
    let wrapper_path = dir.path().join("wrapped_doc.json");
    fs::write(
        &wrapper_path,
        serde_json::to_vec(&Wrapper {
            order: 3,
            doc: &doc,
        })
        .unwrap(),
    )
    .unwrap();
    let mut child = Command::new(bin)
        .args([
            "--chapter",
            "3",
            "--title",
            "第三章 庭院",
            "--vocab",
            vocab_path.to_str().unwrap(),
            "--doc-json",
            wrapper_path.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nlp_narrative_doc wrapped --doc-json");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "wrapped --doc-json failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let wrapped_ch: ChapterNarrative =
        serde_json::from_slice(&output.stdout).expect("parse wrapped stdout");
    assert_eq!(wrapped_ch, raw_ch);

    // 4. Hash mismatch must exit 1 with `nlp_narrative_doc:` error
    let mut child = Command::new(bin)
        .args(["--doc-json", doc_json_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn nlp_narrative_doc mismatch");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all("完全不同的文本。".as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.starts_with("nlp_narrative_doc:"),
        "unexpected stderr: {stderr}"
    );
}

#[derive(Serialize)]
struct SynthSourceSpan {
    start_char: usize,
    end_char: usize,
    start_byte: usize,
    end_byte: usize,
}

#[derive(Serialize)]
struct SynthChapterFile {
    schema: &'static str,
    order: usize,
    title: String,
    source_file: String,
    source_sha256: String,
    source_span: SynthSourceSpan,
    chapter_sha256: String,
    backend: &'static str,
    doc: Document,
}

#[derive(Serialize)]
struct SynthIndexRow {
    schema: &'static str,
    order: usize,
    title: String,
    source_file: String,
    source_sha256: String,
    source_span: SynthSourceSpan,
    chapter_sha256: String,
    output_file: String,
}

#[derive(Serialize)]
struct SynthDiscoveredLexicon {
    schema: &'static str,
    terms: Vec<SynthLexiconTerm>,
}

#[derive(Serialize)]
struct SynthLexiconTerm {
    term: &'static str,
}

fn setup_synthetic_batch_fixture(
    root: &Path,
    vocab: &WorkVocabulary,
) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let analysis_dir = root.join("analysis");
    let bundles_dir = root.join("bundles");
    let chapters_dir = analysis_dir.join("chapters");
    fs::create_dir_all(&chapters_dir).unwrap();
    fs::create_dir_all(&bundles_dir).unwrap();

    let vocab_path = root.join("vocab.json");
    fs::write(&vocab_path, serde_json::to_vec_pretty(vocab).unwrap()).unwrap();

    // Write discovered_lexicon.json with an extra domain term not in vocab.json
    let discovered = SynthDiscoveredLexicon {
        schema: "nlp.novel_discovered_lexicon.v1",
        terms: vec![
            SynthLexiconTerm { term: "古銅鏡" },
            SynthLexiconTerm { term: "玄鐵劍" },
        ],
    };
    fs::write(
        analysis_dir.join("discovered_lexicon.json"),
        serde_json::to_vec_pretty(&discovered).unwrap(),
    )
    .unwrap();

    let ch1_text = "林遠推開木門，將古銅鏡放入木匣。陳伯低聲說道：「明日動身。」";
    let ch2_text = "次日清晨，林遠帶著玄鐵劍離開青石鎮。與此同時，蘇晴在客棧等待消息。";
    let ch3_text = "蘇晴收到書信後趕往山門，林遠卻早已出發。";

    let bundle1_name = "bundle_01.txt";
    let bundle1_content = format!("{ch1_text}\n{ch2_text}");
    let bundle1_bytes = bundle1_content.as_bytes();
    let bundle1_sha = sha256_hex(bundle1_bytes);
    fs::write(bundles_dir.join(bundle1_name), bundle1_bytes).unwrap();

    let bundle2_name = "bundle_02.txt";
    let bundle2_bytes = ch3_text.as_bytes();
    let bundle2_sha = sha256_hex(bundle2_bytes);
    fs::write(bundles_dir.join(bundle2_name), bundle2_bytes).unwrap();

    let ch1_start_byte = 0usize;
    let ch1_end_byte = ch1_text.len();
    let ch1_start_char = 0usize;
    let ch1_end_char = ch1_text.chars().count();

    let ch2_start_byte = ch1_end_byte + 1;
    let ch2_end_byte = ch2_start_byte + ch2_text.len();
    let ch2_start_char = ch1_end_char + 1;
    let ch2_end_char = ch2_start_char + ch2_text.chars().count();

    let ch3_start_byte = 0usize;
    let ch3_end_byte = ch3_text.len();
    let ch3_start_char = 0usize;
    let ch3_end_char = ch3_text.chars().count();

    let opts = vocab.to_doc_options();
    let specs = vec![
        (
            1usize,
            "第一章 山門",
            ch1_text,
            bundle1_name,
            &bundle1_sha,
            SynthSourceSpan {
                start_char: ch1_start_char,
                end_char: ch1_end_char,
                start_byte: ch1_start_byte,
                end_byte: ch1_end_byte,
            },
        ),
        (
            2usize,
            "第二章 啟程",
            ch2_text,
            bundle1_name,
            &bundle1_sha,
            SynthSourceSpan {
                start_char: ch2_start_char,
                end_char: ch2_end_char,
                start_byte: ch2_start_byte,
                end_byte: ch2_end_byte,
            },
        ),
        (
            3usize,
            "第三章 會合",
            ch3_text,
            bundle2_name,
            &bundle2_sha,
            SynthSourceSpan {
                start_char: ch3_start_char,
                end_char: ch3_end_char,
                start_byte: ch3_start_byte,
                end_byte: ch3_end_byte,
            },
        ),
    ];

    let mut index_lines = String::new();
    for (order, title, text, src_file, src_sha, span) in specs {
        let doc = analyze_document(text, &opts);
        let chapter_sha = text_sha256_hex(text);
        let rel = format!("chapters/{order:04}.json");
        let ch_file = SynthChapterFile {
            schema: "nlp.novel_chapter.v1",
            order,
            title: title.to_string(),
            source_file: src_file.to_string(),
            source_sha256: src_sha.clone(),
            source_span: SynthSourceSpan { ..span },
            chapter_sha256: chapter_sha.clone(),
            backend: "rule",
            doc,
        };
        fs::write(
            analysis_dir.join(&rel),
            serde_json::to_vec(&ch_file).unwrap(),
        )
        .unwrap();

        let idx_row = SynthIndexRow {
            schema: "nlp.novel_completed_index.v1",
            order,
            title: title.to_string(),
            source_file: src_file.to_string(),
            source_sha256: src_sha.clone(),
            source_span: span,
            chapter_sha256: chapter_sha,
            output_file: rel,
        };
        index_lines.push_str(&serde_json::to_string(&idx_row).unwrap());
        index_lines.push('\n');
    }

    fs::write(analysis_dir.join("completed_index.jsonl"), index_lines).unwrap();
    (analysis_dir, bundles_dir, vocab_path)
}

#[test]
fn nlp_narrative_batch_e2e_resume_no_resume_and_sha256_verification() {
    let bin = env!("CARGO_BIN_EXE_nlp_narrative_batch");
    let dir = tempdir().expect("tempdir");
    let vocab = synthetic_vocab();
    let (analysis_dir, bundles_dir, vocab_path) = setup_synthetic_batch_fixture(dir.path(), &vocab);
    let out_dir = dir.path().join("narrative_out");

    // 1. Initial batch run
    let status = Command::new(bin)
        .args([
            "--analysis-dir",
            analysis_dir.to_str().unwrap(),
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--vocab",
            vocab_path.to_str().unwrap(),
            "--workers",
            "2",
        ])
        .status()
        .expect("run nlp_narrative_batch initial");
    assert!(status.success());

    let receipt1: NarrativeManifestReceipt =
        serde_json::from_slice(&fs::read(out_dir.join("narrative_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt1.schema, BOOK_RECEIPT_SCHEMA);
    assert_eq!(receipt1.total_chapters, 3);
    assert_eq!(receipt1.resumed_chapters, 0);
    assert_eq!(receipt1.processed_chapters, 3);
    assert_eq!(receipt1.residual_dag_cycles, 0);
    assert_eq!(receipt1.discovered_lexicon_terms, 1); // 玄鐵劍 added on top of 古銅鏡
    assert!(receipt1.vocab_sha256.is_some());
    assert!(receipt1.total_events > 0);
    assert!(receipt1.total_mentions > 0);

    // Verify per-chapter files and stream hashes
    let mut chapter_hasher = Sha256::new();
    for ch_num in 1..=3u32 {
        let bytes = fs::read(out_dir.join(format!("chapters/{ch_num:04}.json"))).unwrap();
        let ch: ChapterNarrative = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(ch.schema, CHAPTER_NARRATIVE_SCHEMA);
        assert_eq!(ch.chapter, ch_num);
        assert!(!build_chapter_dag_from_narrative(&ch).has_cycle());
        let file_sha = sha256_hex(&bytes);
        chapter_hasher.update(format!("{ch_num:04}:{file_sha}\n").as_bytes());
    }
    let expected_chapters_sha: String = chapter_hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(receipt1.chapter_outputs_sha256, expected_chapters_sha);

    let events_bytes = fs::read(out_dir.join("events_stream.jsonl")).unwrap();
    let edges_bytes = fs::read(out_dir.join("timeline_edges.jsonl")).unwrap();
    let mentions_bytes = fs::read(out_dir.join("actor_mentions.jsonl")).unwrap();
    assert_eq!(receipt1.events_stream_sha256, sha256_hex(&events_bytes));
    assert_eq!(receipt1.timeline_edges_sha256, sha256_hex(&edges_bytes));
    assert_eq!(receipt1.actor_mentions_sha256, sha256_hex(&mentions_bytes));

    let event_lines: Vec<CandidateEvent> = String::from_utf8(events_bytes)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(event_lines.len(), receipt1.total_events);

    let edge_lines: Vec<TimelineEdge> = String::from_utf8(edges_bytes)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(edge_lines.len(), receipt1.total_edges);

    let mention_lines: Vec<ActorMention> = String::from_utf8(mentions_bytes)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(mention_lines.len(), receipt1.total_mentions);

    // 2. Resume run (zero recompute)
    let status = Command::new(bin)
        .args([
            "--analysis-dir",
            analysis_dir.to_str().unwrap(),
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--vocab",
            vocab_path.to_str().unwrap(),
            "--workers",
            "2",
        ])
        .status()
        .expect("run nlp_narrative_batch resume");
    assert!(status.success());

    let receipt2: NarrativeManifestReceipt =
        serde_json::from_slice(&fs::read(out_dir.join("narrative_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt2.total_chapters, 3);
    assert_eq!(receipt2.resumed_chapters, 3);
    assert_eq!(receipt2.processed_chapters, 0);
    assert_eq!(
        receipt2.chapter_outputs_sha256,
        receipt1.chapter_outputs_sha256
    );
    assert_eq!(receipt2.events_stream_sha256, receipt1.events_stream_sha256);
    assert_eq!(
        receipt2.timeline_edges_sha256,
        receipt1.timeline_edges_sha256
    );
    assert_eq!(
        receipt2.actor_mentions_sha256,
        receipt1.actor_mentions_sha256
    );

    // 3. --no-resume run (full recompute, deterministic identical hashes)
    let status = Command::new(bin)
        .args([
            "--analysis-dir",
            analysis_dir.to_str().unwrap(),
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--vocab",
            vocab_path.to_str().unwrap(),
            "--workers",
            "2",
            "--no-resume",
        ])
        .status()
        .expect("run nlp_narrative_batch --no-resume");
    assert!(status.success());

    let receipt3: NarrativeManifestReceipt =
        serde_json::from_slice(&fs::read(out_dir.join("narrative_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt3.total_chapters, 3);
    assert_eq!(receipt3.resumed_chapters, 0);
    assert_eq!(receipt3.processed_chapters, 3);
    assert_eq!(
        receipt3.chapter_outputs_sha256,
        receipt1.chapter_outputs_sha256
    );
    assert_eq!(receipt3.events_stream_sha256, receipt1.events_stream_sha256);
    assert_eq!(
        receipt3.timeline_edges_sha256,
        receipt1.timeline_edges_sha256
    );
    assert_eq!(
        receipt3.actor_mentions_sha256,
        receipt1.actor_mentions_sha256
    );

    // 4. Fallback without completed_index.jsonl
    fs::remove_file(analysis_dir.join("completed_index.jsonl")).unwrap();
    let out_dir_fallback = dir.path().join("narrative_out_fallback");
    let status = Command::new(bin)
        .args([
            "--analysis-dir",
            analysis_dir.to_str().unwrap(),
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_dir_fallback.to_str().unwrap(),
            "--vocab",
            vocab_path.to_str().unwrap(),
            "--workers",
            "2",
        ])
        .status()
        .expect("run nlp_narrative_batch fallback scan");
    assert!(status.success());

    let receipt4: NarrativeManifestReceipt = serde_json::from_slice(
        &fs::read(out_dir_fallback.join("narrative_manifest_receipt.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        receipt4.chapter_outputs_sha256,
        receipt1.chapter_outputs_sha256
    );
    assert_eq!(receipt4.events_stream_sha256, receipt1.events_stream_sha256);
}
