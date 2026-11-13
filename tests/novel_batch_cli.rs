//! Integration tests for the `nlp_novel_batch` multi-core resumable CLI.
//!
//! All test fixtures use 100% neutral, self-authored synthetic Chinese and English narrative text.

use faberline_nlp::doc::{DEPRELS, Document};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
#[cfg(any(feature = "coreml", feature = "onnx"))]
use std::path::PathBuf;
use std::process::Command;

fn bin_path() -> &'static str {
    env!("CARGO_BIN_EXE_nlp_novel_batch")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

/// Discovers a directory containing `manifest.json` + staged `doc-syntax-v1` weights without any
/// hardcoded personal paths (checks env vars, crate `models/`, and git worktree common-dir `models/`).
#[cfg(any(feature = "coreml", feature = "onnx"))]
fn find_staged_models_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    for var in ["NLP_MODELS_DIR", "NLP_MODELS_STAGE"] {
        if let Ok(val) = std::env::var(var) {
            candidates.push(PathBuf::from(val));
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(manifest_dir.join("models"));

    if let Ok(out) = Command::new("git")
        .args(["rev-parse", "--git-common-dir"])
        .current_dir(&manifest_dir)
        .output()
        && out.status.success()
        && let Ok(raw) = String::from_utf8(out.stdout)
    {
        let common = PathBuf::from(raw.trim());
        let resolved = if common.is_absolute() {
            common
        } else {
            manifest_dir.join(common)
        };
        if let Some(repo_root) = resolved.parent() {
            candidates.push(repo_root.join("models"));
        }
    }

    candidates.into_iter().find(|dir| {
        dir.join("manifest.json").is_file() && dir.join("doc-syntax-v1/tokenizer.json").is_file()
    })
}

fn assert_doc_spans_valid(doc: &Document, raw_text: &str) {
    let chars: Vec<char> = raw_text.chars().collect();
    assert_eq!(doc.text_chars, chars.len());
    assert_eq!(doc.text_sha256, sha256_hex(raw_text.as_bytes()));

    let check = |start: usize, end: usize, label: &str| {
        assert!(
            start < end && end <= chars.len(),
            "invalid {label} span {start}..{end} for text_chars={}",
            chars.len()
        );
    };

    for p in &doc.segmentation.paragraphs {
        check(p.span.start_char, p.span.end_char, "paragraph");
    }
    for s in &doc.segmentation.sentences {
        check(s.span.start_char, s.span.end_char, "sentence");
        assert!(s.paragraph < doc.segmentation.paragraphs.len());
    }
    for c in &doc.segmentation.clauses {
        check(c.span.start_char, c.span.end_char, "clause");
        assert!(c.sentence < doc.segmentation.sentences.len());
    }
    for o in &doc.segmentation.opaque {
        check(o.span.start_char, o.span.end_char, "opaque");
    }
    for t in &doc.tokens {
        check(t.span.start_char, t.span.end_char, "token");
        assert!(t.sentence < doc.segmentation.sentences.len());
    }
    for e in &doc.entities {
        check(e.span.start_char, e.span.end_char, "entity");
        assert!(e.sentence < doc.segmentation.sentences.len());
    }
    for r in &doc.relations {
        check(
            r.connector.start_char,
            r.connector.end_char,
            "relation.connector",
        );
        if let Some(corr) = r.correlate {
            check(corr.start_char, corr.end_char, "relation.correlate");
        }
        check(r.arg1.start_char, r.arg1.end_char, "relation.arg1");
        check(r.arg2.start_char, r.arg2.end_char, "relation.arg2");
    }
    for u in &doc.unknown {
        check(u.span.start_char, u.span.end_char, "unknown");
    }

    // Verify Layer 4 syntax indices, UD deprels, no self-loops, and acyclicity.
    let deprel_set: HashSet<&str> = DEPRELS.iter().copied().collect();
    let mut head_by_tok = std::collections::HashMap::new();
    for st in &doc.syntax {
        assert!(st.token < doc.tokens.len());
        assert_eq!(st.span, doc.tokens[st.token].span);
        assert_eq!(st.sentence, doc.tokens[st.token].sentence);
        assert!(
            deprel_set.contains(st.deprel.as_str()),
            "invalid UD deprel: {}",
            st.deprel
        );
        if let Some(h) = st.head {
            assert!(h < doc.tokens.len());
            assert_ne!(h, st.token, "self-loop on token {}", st.token);
            assert_eq!(
                doc.tokens[h].sentence, st.sentence,
                "cross-sentence syntax head"
            );
            assert_ne!(st.deprel, "root");
        } else {
            assert_eq!(st.deprel, "root");
        }
        head_by_tok.insert(st.token, st.head);
    }
    for &tok in head_by_tok.keys() {
        let mut cur = Some(tok);
        let mut steps = 0usize;
        while let Some(node) = cur {
            steps += 1;
            assert!(steps <= doc.tokens.len(), "cycle detected in syntax graph");
            cur = head_by_tok.get(&node).copied().flatten();
        }
    }
}

#[test]
fn bundle_splitting_provenance_offsets_and_edge_headings_are_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let bundles_dir = tmp.path().join("bundles");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&bundles_dir).unwrap();

    // Ignored hidden file in bundles directory.
    fs::write(bundles_dir.join(".DS_Store"), b"ignored").unwrap();

    // Bundle 1 exercises:
    // - Leading metadata header before Chapter 1
    // - Duplicate `\u{00a0}` book-prefixed banner + compact duplicate heading line
    // - Mid-chapter bare chapter-number repeat (`第一千二百三十四章`)
    // - Volume-prefixed chapter (`第六卷 ... 第一千零八十五章 ...`)
    // - Missing `第` (`一百三十三章 ...`), missing `章` (`第四百八十八 ...`), and single OCR unit typo (`第一把八十三章 ...`)
    let b1 = concat!(
        "《星河旅記》整理合集\n",
        "版本說明：合成測試文本\n\n",
        "第一章 晨霧驛站\n\n",
        "星河旅記\u{00a0}第一章 晨霧驛站\n\n",
        "第一章晨霧驛站\n\n",
        "2026年3月5日清晨7點，林舟帶著3公斤乾糧走進石頭驛站。因為山路濕滑，所以他放慢了腳步。雖然寒風刺骨，但是驛站裡爐火溫暖。\n\n\n",
        "第一千二百三十四章 遠山古道?\n\n",
        "星河旅記\u{00a0}第一千二百三十四章 遠山古道?\n\n",
        "翌日中午，商隊沿著古道前進了12公里，隨後在溪邊紮營休息。\n\n",
        "第一千二百三十四章\n\n",
        "如果明天天氣晴朗，商隊就能在傍晚抵達山谷小鎮。\n\n\n",
        "第六卷 北境風雪 第一千零八十五章 石橋夜雨\n\n",
        "星河旅記\u{00a0}第六卷 北境風雪 第一千零八十五章 石橋夜雨\n\n",
        "三天後夜裡，陳默站在石橋上觀察水勢，同時記錄了5次水位變化。\n"
    );
    fs::write(bundles_dir.join("bundle_01.txt"), b1.as_bytes()).unwrap();

    let b2 = concat!(
        "一百三十三章 河畔燈火\n\n",
        "星河旅記\u{00a0}一百三十三章 河畔燈火\n\n",
        "船夫點亮了兩盞油燈，接著把木船推向河心。\n\n\n",
        "第四百八十八 古塔鐘聲\n\n",
        "星河旅記\u{00a0}第四百八十八 古塔鐘聲\n\n",
        "鐘聲響了3下，眾人立刻安靜下來。\n\n\n",
        "第一把八十三章 銅鈴迴響\n\n",
        "星河旅記\u{00a0}第一把八十三章 銅鈴迴響\n\n",
        "微風吹過屋簷，銅鈴發出清脆聲響。\n"
    );
    fs::write(bundles_dir.join("bundle_02.txt"), b2.as_bytes()).unwrap();

    let b3 = concat!(
        "Chapter 1: Harbor Dawn\n\n",
        "On March 8, 2026, Maya walked 5 km along the quiet pier before sunrise. Because the tide was low, the boats stayed anchored.\n\n\n",
        "Chapter II - Distant Lanterns\n\n",
        "Tomorrow the crew will load 12 crates, and then the ship will sail north.\n"
    );
    fs::write(bundles_dir.join("bundle_03.txt"), b3.as_bytes()).unwrap();

    let out = Command::new(bin_path())
        .args([
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--backend",
            "none",
            "--workers",
            "2",
        ])
        .output()
        .expect("run nlp_novel_batch");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let expected_titles = [
        ("bundle_01.txt", "第一章 晨霧驛站"),
        ("bundle_01.txt", "第一千二百三十四章 遠山古道?"),
        ("bundle_01.txt", "第六卷 北境風雪 第一千零八十五章 石橋夜雨"),
        ("bundle_02.txt", "一百三十三章 河畔燈火"),
        ("bundle_02.txt", "第四百八十八 古塔鐘聲"),
        ("bundle_02.txt", "第一把八十三章 銅鈴迴響"),
        ("bundle_03.txt", "Chapter 1: Harbor Dawn"),
        ("bundle_03.txt", "Chapter II - Distant Lanterns"),
    ];

    for (i, (expected_src, expected_title)) in expected_titles.iter().enumerate() {
        let order = i + 1;
        let ch_path = out_dir.join(format!("chapters/{order:04}.json"));
        assert!(ch_path.is_file(), "missing {}", ch_path.display());
        let raw_ch = fs::read(&ch_path).unwrap();
        let ch_json: Value = serde_json::from_slice(&raw_ch).unwrap();

        assert_eq!(ch_json["order"].as_u64(), Some(order as u64));
        assert_eq!(ch_json["title"].as_str(), Some(*expected_title));
        assert_eq!(ch_json["source_file"].as_str(), Some(*expected_src));

        let bundle_bytes = fs::read(bundles_dir.join(expected_src)).unwrap();
        let bundle_text = std::str::from_utf8(&bundle_bytes).unwrap();
        let bundle_chars: Vec<char> = bundle_text.chars().collect();
        assert_eq!(
            ch_json["source_sha256"].as_str().unwrap(),
            sha256_hex(&bundle_bytes)
        );

        let sc = ch_json["source_span"]["start_char"].as_u64().unwrap() as usize;
        let ec = ch_json["source_span"]["end_char"].as_u64().unwrap() as usize;
        let sb = ch_json["source_span"]["start_byte"].as_u64().unwrap() as usize;
        let eb = ch_json["source_span"]["end_byte"].as_u64().unwrap() as usize;

        let by_bytes = std::str::from_utf8(&bundle_bytes[sb..eb]).unwrap();
        let by_chars: String = bundle_chars[sc..ec].iter().collect();
        assert_eq!(
            by_bytes, by_chars,
            "byte span and char span disagree on ch {order}"
        );

        let chap_sha = ch_json["chapter_sha256"].as_str().unwrap();
        assert_eq!(sha256_hex(by_bytes.as_bytes()), chap_sha);

        let doc: Document = serde_json::from_value(ch_json["doc"].clone()).unwrap();
        assert_doc_spans_valid(&doc, by_bytes);
    }
    assert!(!out_dir.join("chapters/0009.json").exists());

    // Verify receipt and timeline cues provenance spans against raw bundle text!
    let receipt: Value =
        serde_json::from_slice(&fs::read(out_dir.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["total_chapters"].as_u64(), Some(8));
    assert_eq!(receipt["processed_chapters"].as_u64(), Some(8));
    assert_eq!(receipt["resumed_chapters"].as_u64(), Some(0));
    assert!(receipt["total_sentences"].as_u64().unwrap() >= 10);
    assert!(receipt["total_tokens"].as_u64().unwrap() >= 50);

    let cues_raw = fs::read(out_dir.join("entities_timeline_cues.jsonl")).unwrap();
    assert_eq!(
        receipt["timeline_cues_sha256"].as_str().unwrap(),
        sha256_hex(&cues_raw)
    );
    let cue_lines: Vec<Value> = std::str::from_utf8(&cues_raw)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        receipt["total_timeline_cues"].as_u64(),
        Some(cue_lines.len() as u64)
    );
    assert!(!cue_lines.is_empty());

    let mut cue_kinds = HashSet::new();
    for cue in &cue_lines {
        cue_kinds.insert(format!(
            "{}:{}",
            cue["cue_type"].as_str().unwrap(),
            cue["kind"].as_str().unwrap()
        ));
        let src_file = cue["source_file"].as_str().unwrap();
        let bundle_text = fs::read_to_string(bundles_dir.join(src_file)).unwrap();
        let bundle_chars: Vec<char> = bundle_text.chars().collect();
        let bsc = cue["bundle_span"]["start_char"].as_u64().unwrap() as usize;
        let bec = cue["bundle_span"]["end_char"].as_u64().unwrap() as usize;
        assert!(bsc < bec && bec <= bundle_chars.len());
        if let Some(txt) = cue["text"].as_str() {
            let sliced: String = bundle_chars[bsc..bec].iter().collect();
            assert_eq!(sliced, txt);
        }
    }
    assert!(cue_kinds.contains("entity:date") || cue_kinds.contains("entity:relative_date"));
    assert!(cue_kinds.contains("entity:quantity"));
    assert!(cue_kinds.contains("relation:cause"));
    assert!(cue_kinds.contains("relation:contrast") || cue_kinds.contains("relation:condition"));
}

#[test]
fn multi_worker_determinism_workers_4_matches_workers_1() {
    let tmp = tempfile::tempdir().unwrap();
    let bundles_dir = tmp.path().join("bundles");
    fs::create_dir_all(&bundles_dir).unwrap();

    for b_idx in 0..3 {
        let mut content = String::new();
        for local in 1..=4 {
            let ch_num = b_idx * 4 + local;
            content.push_str(&format!(
                "第{ch_num}章 觀測記錄{ch_num}\n\n測試合集\u{00a0}第{ch_num}章 觀測記錄{ch_num}\n\n"
            ));
            content.push_str(&format!(
                "第{ch_num}天清晨，研究員在實驗室測量了{}組樣本，隨後記錄了{}毫升溶液的溫度變化。因為數據穩定，所以團隊繼續下一階段測試。\n\n\n",
                ch_num + 2,
                ch_num * 10
            ));
        }
        fs::write(
            bundles_dir.join(format!("part_{b_idx:02}.txt")),
            content.as_bytes(),
        )
        .unwrap();
    }

    let out_w1 = tmp.path().join("out_w1");
    let out_w4 = tmp.path().join("out_w4");

    for (workers, out_dir) in [("1", &out_w1), ("4", &out_w4)] {
        let out = Command::new(bin_path())
            .args([
                "--bundles-dir",
                bundles_dir.to_str().unwrap(),
                "--out-dir",
                out_dir.to_str().unwrap(),
                "--backend",
                "none",
                "--workers",
                workers,
            ])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "workers={workers} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    for order in 1..=12 {
        let rel = format!("chapters/{order:04}.json");
        assert_eq!(
            fs::read(out_w1.join(&rel)).unwrap(),
            fs::read(out_w4.join(&rel)).unwrap(),
            "mismatch on {rel}"
        );
    }
    assert_eq!(
        fs::read(out_w1.join("completed_index.jsonl")).unwrap(),
        fs::read(out_w4.join("completed_index.jsonl")).unwrap()
    );
    assert_eq!(
        fs::read(out_w1.join("entities_timeline_cues.jsonl")).unwrap(),
        fs::read(out_w4.join("entities_timeline_cues.jsonl")).unwrap()
    );

    let r1: Value =
        serde_json::from_slice(&fs::read(out_w1.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    let r4: Value =
        serde_json::from_slice(&fs::read(out_w4.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    for key in [
        "total_chapters",
        "total_chars",
        "total_paragraphs",
        "total_sentences",
        "supported_sentences",
        "total_clauses",
        "total_tokens",
        "decided_upos_tokens",
        "decided_upos_rate",
        "syntax_tokens",
        "syntax_token_coverage_rate",
        "total_entities",
        "total_relations",
        "total_timeline_cues",
        "unknown_total",
        "unknown_by_layer_reason",
        "unknown_breakdown",
        "chapter_outputs_sha256",
        "timeline_cues_sha256",
    ] {
        assert_eq!(r1[key], r4[key], "receipt mismatch on key {key}");
    }
}

#[test]
fn crash_resume_skips_completed_and_reruns_modified_or_corrupted_chapters() {
    let tmp = tempfile::tempdir().unwrap();
    let book_dir = tmp.path().join("book");
    let chaps_in = book_dir.join("chapters");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&chaps_in).unwrap();

    let mut manifest_entries = Vec::new();
    for order in 1..=6 {
        let rel = format!("chapters/{order:04}.txt");
        let title = format!("第{order}章 航行日誌{order}");
        let body = format!(
            "{title}\n\n船隊在第{order}天航行了{}海里，隨後停靠補給。\n",
            order * 15
        );
        fs::write(book_dir.join(&rel), body.as_bytes()).unwrap();
        manifest_entries.push(serde_json::json!({
            "order": order,
            "title": title,
            "file": rel,
            "status": "ok",
        }));
    }
    // Also add a non-ok chapter to verify manifest status filtering.
    manifest_entries.push(serde_json::json!({
        "order": 99,
        "title": "未發佈章節",
        "file": "chapters/0099.txt",
        "status": "missing",
    }));
    let manifest_path = book_dir.join("manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&serde_json::json!({ "chapters": manifest_entries })).unwrap(),
    )
    .unwrap();

    // 1. Initial full run.
    let run1 = Command::new(bin_path())
        .args([
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--backend",
            "none",
            "--workers",
            "3",
        ])
        .output()
        .unwrap();
    assert!(run1.status.success());
    let r1: Value =
        serde_json::from_slice(&fs::read(out_dir.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(r1["total_chapters"].as_u64(), Some(6));
    assert_eq!(r1["processed_chapters"].as_u64(), Some(6));
    assert_eq!(r1["resumed_chapters"].as_u64(), Some(0));

    // 2. Immediate re-run skips all 6 chapters via fast-path resume.
    let run2 = Command::new(bin_path())
        .args([
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--backend",
            "none",
            "--resume",
        ])
        .output()
        .unwrap();
    assert!(run2.status.success());
    let r2: Value =
        serde_json::from_slice(&fs::read(out_dir.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(r2["resumed_chapters"].as_u64(), Some(6));
    assert_eq!(r2["processed_chapters"].as_u64(), Some(0));
    assert_eq!(r2["chapter_outputs_sha256"], r1["chapter_outputs_sha256"]);

    // 3. Simulate:
    //    - chapter 2 source file modified on disk,
    //    - chapter 5 output deleted (interrupted run),
    //    - chapter 6 output truncated/corrupted, and `completed_index.jsonl` deleted.
    //    Chapters 1, 3, 4 must still be recovered & skipped (`resumed == 3`), and 2, 5, 6 re-run (`processed == 3`).
    fs::write(
        book_dir.join("chapters/0002.txt"),
        "第2章 航行日誌2\n\n暴風雨過後，船隊改道向東航行了80海里，因此提前抵達港口。\n",
    )
    .unwrap();
    fs::remove_file(out_dir.join("chapters/0005.json")).unwrap();
    fs::write(out_dir.join("chapters/0006.json"), b"{corrupted_json").unwrap();
    fs::remove_file(out_dir.join("completed_index.jsonl")).unwrap();

    let run3 = Command::new(bin_path())
        .args([
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--backend",
            "none",
            "--workers",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        run3.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&run3.stderr)
    );
    let r3: Value =
        serde_json::from_slice(&fs::read(out_dir.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(r3["resumed_chapters"].as_u64(), Some(3));
    assert_eq!(r3["processed_chapters"].as_u64(), Some(3));
    assert_ne!(r3["chapter_outputs_sha256"], r1["chapter_outputs_sha256"]);

    let ch2: Value =
        serde_json::from_slice(&fs::read(out_dir.join("chapters/0002.json")).unwrap()).unwrap();
    let ch2_src = fs::read(book_dir.join("chapters/0002.txt")).unwrap();
    assert_eq!(
        ch2["chapter_sha256"].as_str().unwrap(),
        sha256_hex(&ch2_src)
    );
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
#[test]
fn coreml_and_coreml_fast_backends_run_multiworker_and_invalidate_none_cache() {
    let Some(models_dir) = find_staged_models_dir() else {
        eprintln!("skipping staged doc-syntax-v1 Core ML CLI test: staged model weights absent");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();
    let bundles_dir = tmp.path().join("bundles");
    fs::create_dir_all(&bundles_dir).unwrap();

    let bundle_a = concat!(
        "第一章 山谷木屋\n\n",
        "合成卷\u{00a0}第一章 山谷木屋\n\n",
        "清晨陽光灑在石階上，少年背起竹簍走進山林。因為昨夜下過一場大雨，溪水比平時湍急許多。\n\n\n",
        "第二章 市集見聞\n\n",
        "合成卷\u{00a0}第二章 市集見聞\n\n",
        "正午時分，小鎮市集聚集了許多商販，大家一邊整理貨物，一邊討論今年的收成。\n"
    );
    let bundle_b = concat!(
        "Chapter 3: Quiet Observatory\n\n",
        "Elena calibrated the telescope at 9 pm and recorded three bright stars above the ridge.\n\n\n",
        "第四章 燈下筆記\n\n",
        "合成卷\u{00a0}第四章 燈下筆記\n\n",
        "雖然夜色已深，學者依然在書桌前核對兩份星圖，隨後把結論寫入日誌。\n"
    );
    fs::write(bundles_dir.join("01.txt"), bundle_a.as_bytes()).unwrap();
    fs::write(bundles_dir.join("02.txt"), bundle_b.as_bytes()).unwrap();

    // 1. Run `--backend none` first in `out_fast_w4` so we also verify `--resume` invalidates `none` cache when `--models` is added!
    let out_fast_w4 = tmp.path().join("out_fast_w4");
    let run_none = Command::new(bin_path())
        .args([
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_fast_w4.to_str().unwrap(),
            "--backend",
            "none",
            "--workers",
            "2",
        ])
        .output()
        .unwrap();
    assert!(run_none.status.success());

    // 2. Re-run on `out_fast_w4` with `--models` + `--backend coreml-fast --workers 2`: must re-process all 4 chapters!
    let run_fast_w4 = Command::new(bin_path())
        .args([
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_fast_w4.to_str().unwrap(),
            "--models",
            models_dir.to_str().unwrap(),
            "--backend",
            "coreml-fast",
            "--workers",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        run_fast_w4.status.success(),
        "coreml-fast stderr: {}",
        String::from_utf8_lossy(&run_fast_w4.stderr)
    );

    let r_fast_w4: Value =
        serde_json::from_slice(&fs::read(out_fast_w4.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(r_fast_w4["backend"].as_str(), Some("coreml-fast"));
    assert_eq!(r_fast_w4["processed_chapters"].as_u64(), Some(4));
    assert_eq!(r_fast_w4["resumed_chapters"].as_u64(), Some(0));
    assert!(r_fast_w4["model_sha256"].as_str().unwrap().len() == 64);
    assert!(r_fast_w4["syntax_token_coverage_rate"].as_f64().unwrap() > 0.75);
    assert!(r_fast_w4["decided_upos_rate"].as_f64().unwrap() > 0.85);

    // 3. Re-running `coreml-fast` immediately skips all 4 completed chapters!
    let run_fast_resume = Command::new(bin_path())
        .args([
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_fast_w4.to_str().unwrap(),
            "--models",
            models_dir.to_str().unwrap(),
            "--backend",
            "coreml-fast",
            "--workers",
            "4",
        ])
        .output()
        .unwrap();
    assert!(run_fast_resume.status.success());
    let r_fast_resume: Value =
        serde_json::from_slice(&fs::read(out_fast_w4.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(r_fast_resume["resumed_chapters"].as_u64(), Some(4));
    assert_eq!(r_fast_resume["processed_chapters"].as_u64(), Some(0));

    // Verify `coreml-fast` multi-worker determinism (`--workers 4` == `--workers 1`).
    let out_fast_w1 = tmp.path().join("out_fast_w1");
    let run_fast_w1 = Command::new(bin_path())
        .args([
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_fast_w1.to_str().unwrap(),
            "--models",
            models_dir.to_str().unwrap(),
            "--backend",
            "coreml-fast",
            "--workers",
            "1",
        ])
        .output()
        .unwrap();
    assert!(run_fast_w1.status.success());
    for order in 1..=4 {
        let rel = format!("chapters/{order:04}.json");
        assert_eq!(
            fs::read(out_fast_w1.join(&rel)).unwrap(),
            fs::read(out_fast_w4.join(&rel)).unwrap(),
            "coreml-fast workers=1 vs workers=4 mismatch on {rel}"
        );
    }

    // 4. Verify `--backend coreml` multi-worker determinism (`--workers 4` == `--workers 1`) and valid UD graphs!
    let out_cpu_w1 = tmp.path().join("out_cpu_w1");
    let out_cpu_w4 = tmp.path().join("out_cpu_w4");
    for (w, out_dir) in [("1", &out_cpu_w1), ("4", &out_cpu_w4)] {
        let run = Command::new(bin_path())
            .args([
                "--bundles-dir",
                bundles_dir.to_str().unwrap(),
                "--out-dir",
                out_dir.to_str().unwrap(),
                "--models",
                models_dir.to_str().unwrap(),
                "--backend",
                "coreml",
                "--workers",
                w,
            ])
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "coreml w={w} stderr: {}",
            String::from_utf8_lossy(&run.stderr)
        );
    }

    for order in 1..=4 {
        let rel = format!("chapters/{order:04}.json");
        let b1 = fs::read(out_cpu_w1.join(&rel)).unwrap();
        let b4 = fs::read(out_cpu_w4.join(&rel)).unwrap();
        assert_eq!(b1, b4, "coreml workers=1 vs workers=4 mismatch on {rel}");

        let ch_json: Value = serde_json::from_slice(&b1).unwrap();
        let src_file = ch_json["source_file"].as_str().unwrap();
        let bundle_bytes = fs::read(bundles_dir.join(src_file)).unwrap();
        let sb = ch_json["source_span"]["start_byte"].as_u64().unwrap() as usize;
        let eb = ch_json["source_span"]["end_byte"].as_u64().unwrap() as usize;
        let doc: Document = serde_json::from_value(ch_json["doc"].clone()).unwrap();
        assert!(
            !doc.syntax.is_empty(),
            "expected non-empty syntax on ch {order}"
        );
        assert_doc_spans_valid(&doc, std::str::from_utf8(&bundle_bytes[sb..eb]).unwrap());
    }
}

#[cfg(feature = "onnx")]
#[test]
fn onnx_backend_runs_multiworker_cleanly() {
    let Some(models_dir) = find_staged_models_dir() else {
        eprintln!("skipping staged doc-syntax-v1 ONNX CLI test: staged model weights absent");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();
    let bundles_dir = tmp.path().join("bundles");
    let out_dir = tmp.path().join("out_onnx");
    fs::create_dir_all(&bundles_dir).unwrap();
    fs::write(
        bundles_dir.join("01.txt"),
        "第一章 實驗記錄\n\n小甲在實驗室測量了3組數據。\n\n\n第二章 結果整理\n\nAlice reviewed 2 charts before noon.\n",
    )
    .unwrap();

    let run = Command::new(bin_path())
        .args([
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--models",
            models_dir.to_str().unwrap(),
            "--backend",
            "onnx",
            "--workers",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "onnx stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );

    let receipt: Value =
        serde_json::from_slice(&fs::read(out_dir.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["backend"].as_str(), Some("onnx"));
    assert_eq!(receipt["total_chapters"].as_u64(), Some(2));
    assert!(receipt["syntax_tokens"].as_u64().unwrap() > 0);
}

#[test]
fn auto_lexicon_and_high_signal_timeline_cues_cli_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let bundles_dir = tmp.path().join("bundles");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&bundles_dir).unwrap();

    let templates = [
        "清晨，沈星河拔出蒼雲鐵劍守住石門。",
        "黃昏時，老匠人擦拭蒼雲鐵劍遞給沈星河。",
        "雷聲中，沈星河舉起蒼雲鐵劍劈開木樁。",
        "院落外，眾學徒圍觀蒼雲鐵劍並向沈星河請教。",
        "燈火下，沈星河收好蒼雲鐵劍走進書房。",
    ];
    let mut bundle_content = String::new();
    for ch_idx in 1..=4 {
        bundle_content.push_str(&format!("第{ch_idx}章 演武紀事{ch_idx}\n\n"));
        for tpl in &templates {
            bundle_content.push_str(tpl);
            bundle_content.push('\n');
        }
        bundle_content
            .push_str("沈星河拿了兩個木盒，走了三公里。「因為山路濕滑，所以我們放慢腳步。」\n");
        bundle_content.push_str("（月底求下月票與推薦票，紫電閣歡迎您！）\n\n\n");
    }
    fs::write(bundles_dir.join("01.txt"), bundle_content.as_bytes()).unwrap();

    // 1. Default run (`--auto-lexicon` enabled, bare numbers filtered).
    let run1 = Command::new(bin_path())
        .args([
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--backend",
            "none",
            "--workers",
            "2",
            "--name",
            "沈星河",
        ])
        .output()
        .unwrap();
    assert!(
        run1.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&run1.stderr)
    );

    let lex_json: Value =
        serde_json::from_slice(&fs::read(out_dir.join("discovered_lexicon.json")).unwrap())
            .unwrap();
    assert_eq!(
        lex_json["schema"].as_str(),
        Some("nlp.novel_discovered_lexicon.v1")
    );
    let terms: Vec<&str> = lex_json["terms"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["term"].as_str())
        .collect();
    assert!(terms.contains(&"蒼雲鐵劍"), "terms={terms:?}");
    assert!(terms.contains(&"沈星河"), "terms={terms:?}");
    assert!(!terms.contains(&"蒼雲"), "terms={terms:?}");
    assert!(!terms.contains(&"紫電閣"), "terms={terms:?}");

    let r1: Value =
        serde_json::from_slice(&fs::read(out_dir.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(r1["auto_lexicon_enabled"].as_bool(), Some(true));
    assert_eq!(
        r1["discovered_lexicon_count"].as_u64(),
        Some(terms.len() as u64)
    );
    assert_eq!(r1["zero_root_sentences"].as_u64(), Some(0));
    assert_eq!(r1["orphan_syntax_arcs"].as_u64(), Some(0));

    // Verify Chapter 1 kept discovered compound `蒼雲鐵劍` as one intact token and masked editorial vote-plea line!
    let ch1_json: Value =
        serde_json::from_slice(&fs::read(out_dir.join("chapters/0001.json")).unwrap()).unwrap();
    let ch1_sb = ch1_json["source_span"]["start_byte"].as_u64().unwrap() as usize;
    let ch1_eb = ch1_json["source_span"]["end_byte"].as_u64().unwrap() as usize;
    let ch1_raw = &bundle_content[ch1_sb..ch1_eb];
    let ch1_chars: Vec<char> = ch1_raw.chars().collect();
    let ch1_doc: Document = serde_json::from_value(ch1_json["doc"].clone()).unwrap();
    assert!(ch1_doc.options.mask_editorial);
    assert!(ch1_doc.options.normalize_traditional);
    assert!(ch1_doc.options.lexicon.iter().any(|w| w == "蒼雲鐵劍"));
    let ch1_toks: Vec<String> = ch1_doc
        .tokens
        .iter()
        .map(|t| {
            ch1_chars[t.span.start_char..t.span.end_char]
                .iter()
                .collect()
        })
        .collect();
    assert!(
        ch1_toks.iter().any(|t| t == "蒼雲鐵劍"),
        "expected discovered compound `蒼雲鐵劍` to stay one token, got: {ch1_toks:?}"
    );
    assert!(
        !ch1_toks.iter().any(|t| t.contains("月票")),
        "editorial postscript line should be masked by default: {ch1_toks:?}"
    );

    let cues1: Vec<Value> = fs::read_to_string(out_dir.join("entities_timeline_cues.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(
        !cues1.iter().any(|c| c["kind"].as_str() == Some("number")),
        "bare numbers must be excluded by default"
    );
    assert!(
        cues1
            .iter()
            .any(|c| c["kind"].as_str() == Some("name") && c["text"].as_str() == Some("沈星河"))
    );
    assert!(
        cues1.iter().any(
            |c| c["kind"].as_str() == Some("quantity") && c["text"].as_str() == Some("三公里")
        )
    );
    let cause_cue = cues1
        .iter()
        .find(|c| c["cue_type"].as_str() == Some("relation") && c["kind"].as_str() == Some("cause"))
        .expect("expected cause relation cue");
    assert_eq!(cause_cue["clean_arg1_text"].as_str(), Some("山路濕滑"));
    assert_eq!(cause_cue["clean_arg2_text"].as_str(), Some("我們放慢腳步"));

    // 2. Re-running with `--include-bare-numbers` refreshes cues to include unitless `number` entities.
    let run2 = Command::new(bin_path())
        .args([
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--backend",
            "none",
            "--workers",
            "2",
            "--name",
            "沈星河",
            "--include-bare-numbers",
        ])
        .output()
        .unwrap();
    assert!(run2.status.success());
    let cues2: Vec<Value> = fs::read_to_string(out_dir.join("entities_timeline_cues.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(
        cues2
            .iter()
            .any(|c| c["kind"].as_str() == Some("number") && c["text"].as_str() == Some("兩"))
    );

    // 3. Re-running with `--no-auto-lexicon` sets `auto_lexicon_enabled = false`.
    let out_no_lex = tmp.path().join("out_no_lex");
    let run3 = Command::new(bin_path())
        .args([
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_no_lex.to_str().unwrap(),
            "--backend",
            "none",
            "--no-auto-lexicon",
        ])
        .output()
        .unwrap();
    assert!(run3.status.success());
    assert!(!out_no_lex.join("discovered_lexicon.json").exists());
    let r3: Value =
        serde_json::from_slice(&fs::read(out_no_lex.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(r3["auto_lexicon_enabled"].as_bool(), Some(false));
    assert_eq!(r3["discovered_lexicon_count"].as_u64(), Some(0));
}

#[test]
fn six_workstreams_end_to_end_cli_synthetic() {
    let tmp = tempfile::tempdir().unwrap();
    let bundles_dir = tmp.path().join("bundles");
    let out_dir = tmp.path().join("out_v2");
    fs::create_dir_all(&bundles_dir).unwrap();

    // Synthetic 4-chapter bundle exercising all 6 Phase 3 workstreams simultaneously:
    // WS1: `<P`, `&qut;`, PUA `\u{e201}`, box-drawing banner, missing-`章` heading, editorial postscript
    // WS2: Traditional->Simplified `jieba` projection, coverb/demonstrative de-gluing (`將那`, `對此`), `--auto-lexicon` (`玄冰玉匣`, `林牧雲`)
    // WS3: `在下`/`妾身` pronouns, numeral-prefix idioms (`一清二楚`, `一會兒`), `什麼` disambiguation
    // WS4: `四年`, `數日`, `半個時辰`, `百余丈`, `四年后`, `片刻`
    // WS5: Paratactic `流水句` (>64 tokens, >=4 clauses) + 0 zero-root / 0 orphan syntax arcs
    // WS6: `隨即`, `與此同時`, `整理好行裝之後，...`, plus `clean_arg1_text` / `clean_arg2_text` and bare-number filtering
    let recurring_lines = [
        "<P清晨，林牧雲打開玄冰玉匣檢查封條，隨即與同伴核對清單。",
        "正午時，妾身將玄冰玉匣交給林牧雲保管，與此同時守衛關上石門。",
        "夜色裡，林牧雲捧著玄冰玉匣走過百余丈長廊，&qut;在下已把路線看得一清二楚&qut;。\u{e201}",
        "風雪中，老匠人替林牧雲加固玄冰玉匣，對此毫不在意。",
        "晨光下，林牧雲收起玄冰玉匣等候半個時辰，片刻便啟程下山。",
    ];
    let paratactic = concat!(
        "四年后，隊伍在山谷停留了四年與數日，",
        "清晨的山風吹散了溪谷薄霧，旅人背起行囊踏上青石古道，",
        "沿著蜿蜒山脊一路往北前行，穿過寂靜無聲的松林與竹海，",
        "在石亭旁邊停下腳步喝了兩口清水，打開地圖仔細核對前方岔路，",
        "確認通往山腰驛站的舊標記，重新收好水壺與隨身筆記本，",
        "繞過三處濕滑陡峭的碎石坡，終於在一會兒之前抵達了木屋。",
        "整理好行裝之後，眾人想出什麼辦法卻沒說什麼。\n"
    );

    let mut bundle = String::from("┏━━━━━━━━━━┓\n┃ 合成測試橫幅 ┃\n┗━━━━━━━━━━┛\n\n");
    for ch in 1..=4 {
        if ch == 2 {
            bundle.push_str("第二百零二 石橋風雪\n\n");
        } else {
            bundle.push_str(&format!("第{ch}章 觀測紀要{ch}\n\n"));
        }
        for line in &recurring_lines {
            bundle.push_str(line);
            bundle.push('\n');
        }
        bundle.push_str(paratactic);
        bundle.push_str("（月底最後一天，求下月票與推薦票！）\n\n\n");
    }
    fs::write(bundles_dir.join("01.txt"), bundle.as_bytes()).unwrap();

    #[cfg(all(feature = "coreml", target_os = "macos"))]
    let staged_models = find_staged_models_dir();
    #[cfg(not(all(feature = "coreml", target_os = "macos")))]
    let staged_models: Option<std::path::PathBuf> = None;

    let mut cmd = Command::new(bin_path());
    cmd.args([
        "--bundles-dir",
        bundles_dir.to_str().unwrap(),
        "--out-dir",
        out_dir.to_str().unwrap(),
        "--workers",
        "2",
    ]);
    if let Some(ref m_dir) = staged_models {
        cmd.args([
            "--models",
            m_dir.to_str().unwrap(),
            "--backend",
            "coreml-fast",
        ]);
    } else {
        cmd.args(["--backend", "none"]);
    }
    let out = cmd
        .output()
        .expect("run six_workstreams_end_to_end_cli_synthetic");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let receipt: Value =
        serde_json::from_slice(&fs::read(out_dir.join("book_manifest_receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["total_chapters"].as_u64(), Some(4));
    assert_eq!(receipt["zero_root_sentences"].as_u64(), Some(0));
    assert_eq!(receipt["orphan_syntax_arcs"].as_u64(), Some(0));
    assert_eq!(receipt["auto_lexicon_enabled"].as_bool(), Some(true));
    assert!(receipt["discovered_lexicon_count"].as_u64().unwrap() >= 2);

    if staged_models.is_some() {
        assert!(receipt["decided_upos_rate"].as_f64().unwrap() > 0.96);
        assert!(receipt["syntax_token_coverage_rate"].as_f64().unwrap() > 0.65);
    }

    let lex_json: Value =
        serde_json::from_slice(&fs::read(out_dir.join("discovered_lexicon.json")).unwrap())
            .unwrap();
    let lex_terms: Vec<&str> = lex_json["terms"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v["term"].as_str())
        .collect();
    assert!(lex_terms.contains(&"玄冰玉匣"), "lex_terms={lex_terms:?}");
    assert!(lex_terms.contains(&"林牧雲"), "lex_terms={lex_terms:?}");

    // Validate every chapter's spans & UD forest, plus WS4 entities and WS6 relations on Chapter 1.
    for order in 1..=4 {
        let ch_json: Value = serde_json::from_slice(
            &fs::read(out_dir.join(format!("chapters/{order:04}.json"))).unwrap(),
        )
        .unwrap();
        let sb = ch_json["source_span"]["start_byte"].as_u64().unwrap() as usize;
        let eb = ch_json["source_span"]["end_byte"].as_u64().unwrap() as usize;
        let ch_doc: Document = serde_json::from_value(ch_json["doc"].clone()).unwrap();
        assert_doc_spans_valid(&ch_doc, &bundle[sb..eb]);
    }

    let cues: Vec<Value> = fs::read_to_string(out_dir.join("entities_timeline_cues.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(!cues.iter().any(|c| c["kind"].as_str() == Some("number")));
    for expected_ent in ["四年后", "四年", "數日", "百余丈", "半個時辰", "片刻"] {
        assert!(
            cues.iter().any(|c| c["cue_type"].as_str() == Some("entity")
                && c["text"].as_str() == Some(expected_ent)),
            "missing entity cue {expected_ent:?}"
        );
    }
    assert!(cues.iter().any(|c| {
        c["cue_type"].as_str() == Some("relation")
            && c["kind"].as_str() == Some("sequence")
            && c["clean_arg1_text"].as_str() == Some("整理好行裝")
    }));
}

#[test]
fn v3_lexicon_pruning_and_clean_relation_cue_invariants_cli() {
    let tmp = tempfile::tempdir().unwrap();
    let bundles_dir = tmp.path().join("bundles");
    let out_dir = tmp.path().join("out_v3");
    fs::create_dir_all(&bundles_dir).unwrap();

    let mut bundle = String::new();
    for ch in 1..=4 {
        bundle.push_str(&format!("第{ch}章 邊塞札記{ch}\n\n"));

        // Mid-frequency true proper names (`12` occurrences across 4 chapters) + 4-char artifact:
        bundle.push_str("清晨，張鐵柱捧起寒星古鏡走進石殿，段天涯站在石門外遠眺。\n");
        bundle.push_str("正午時，老匠人向張鐵柱借來寒星古鏡查驗，眾人隨段天涯巡視營地。\n");
        if ch < 4 {
            bundle.push_str("夜色裡，張鐵柱收好寒星古鏡返回木屋，守衛替段天涯點燃篝火。\n");
        } else {
            bundle.push_str("風雪中，張鐵柱擦拭寒星古鏡放入木匣，學徒請段天涯講解劍譜。\n");
        }

        // Classifier+noun (`株仙草`, `顆寶珠`), left-straddling (`衫老者`, `期修士`, `色光焰`),
        // right-straddling (`青色劍`, `銀色符`, `烏冠老`), and scraper infixes (`少丅女`, `長堊老`):
        bundle
            .push_str("竹簍裡放著三株仙草備用，石桌上擺著一顆寶珠照明，廣場上青衫老者緩步登台。\n");
        bundle
            .push_str("藥圃中長出五株仙草搖曳，錦囊內收好數顆寶珠生輝，閣樓前青衫老者負手佇立。\n");
        bundle
            .push_str("木盒內裝滿幾株仙草清香，掌心裡托起那顆寶珠微熱，庭院外白衫老者撫鬚微笑。\n");
        bundle
            .push_str("石台上擺開這株仙草晾曬，玉盤中滾動兩顆寶珠清脆，走廊邊白衫老者推門而入。\n");

        bundle
            .push_str("大殿內初期修士閉目養神，半空中藍色光焰驟然暴漲，雲層下青色劍光破空疾掠。\n");
        bundle
            .push_str("石室裡初期修士凝神靜氣，山谷間藍色光焰隨風搖晃，山巔處青色劍光橫掃巨石。\n");
        bundle
            .push_str("竹林內中期修士演練劍術，水面上紫色光焰緩緩升騰，峽谷裡青色劍氣縱橫交錯。\n");
        bundle
            .push_str("山門前中期修士拱手行禮，祭壇旁紫色光焰漸漸熄滅，密林間青色劍氣劈開迷霧。\n");

        bundle
            .push_str("石壁上銀色符文閃爍不停，高台旁烏冠老者沉聲開口，松樹下灰衣老者閉目傾聽。\n");
        bundle
            .push_str("陣盤內銀色符文流轉不息，涼亭內烏冠老者拂袖落座，火堆邊灰衣老者添柴取暖。\n");
        bundle
            .push_str("玉簡上銀色符印悄然浮現，石階上烏冠老叟拄杖慢行，溪水畔灰衣老叟垂釣不語。\n");
        bundle
            .push_str("銅鏡背銀色符印漸次亮起，書院前烏冠老叟抬頭望天，渡口處灰衣老叟解纜撐船。\n");

        bundle.push_str("樹蔭下少丅女採摘野果，石亭裡長堊老翻閱古籍。\n");
        bundle.push_str("小溪邊少丅女提籃浣紗，書閣內長堊老講解經義。\n");
        bundle.push_str("花叢間少丅女追逐彩蝶，大堂上長堊老主持議事。\n");
        bundle.push_str("石橋頭少丅女揮手道別，庭院裡長堊老品茗論道。\n");

        // Degenerate trailing-ellipsis & standalone-connector relations + valid lexical relation:
        bundle.push_str("“似乎整座石橋都沒入了河水之中，並且……”\n");
        bundle.push_str("快把木門關緊，否則……\n");
        bundle.push_str("我們必須在天亮前出發。否則。\n");
        bundle.push_str("「……，但是眾人已經抵達山谷。」\n");
        bundle.push_str("因為山路濕滑，所以商隊放慢了腳步。\n");
        bundle.push_str("一一小說 ddyueshu 最新全本 瀏覽貼子 轉貼次數\n\n\n");
    }
    fs::write(bundles_dir.join("01.txt"), bundle.as_bytes()).unwrap();

    let out = Command::new(bin_path())
        .args([
            "--bundles-dir",
            bundles_dir.to_str().unwrap(),
            "--out-dir",
            out_dir.to_str().unwrap(),
            "--backend",
            "none",
            "--workers",
            "2",
        ])
        .output()
        .expect("run v3 cli test");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let lex_json: Value =
        serde_json::from_slice(&fs::read(out_dir.join("discovered_lexicon.json")).unwrap())
            .unwrap();
    let terms: Vec<&str> = lex_json["terms"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v["term"].as_str())
        .collect();

    for kept in ["張鐵柱", "段天涯", "寒星古鏡"] {
        assert!(
            terms.contains(&kept),
            "expected `{kept}` in discovered lexicon, got: {terms:?}"
        );
    }
    for pruned in [
        "株仙草",
        "顆寶珠",
        "衫老者",
        "期修士",
        "色光焰",
        "青色劍",
        "銀色符",
        "烏冠老",
        "少丅女",
        "長堊老",
    ] {
        assert!(
            !terms.contains(&pruned),
            "expected false merge `{pruned}` to be pruned, got: {terms:?}"
        );
    }

    let cues: Vec<Value> = fs::read_to_string(out_dir.join("entities_timeline_cues.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let rel_cues: Vec<&Value> = cues
        .iter()
        .filter(|c| c["cue_type"].as_str() == Some("relation"))
        .collect();
    // Exactly 1 valid relation (`因為山路濕滑，所以商隊放慢了腳步。`) per chapter across 4 chapters = 4 cues!
    assert_eq!(rel_cues.len(), 4, "rel_cues={rel_cues:?}");

    let spans_overlap = |a: &Value, b: &Value| -> bool {
        let a0 = a["start_char"].as_u64().unwrap();
        let a1 = a["end_char"].as_u64().unwrap();
        let b0 = b["start_char"].as_u64().unwrap();
        let b1 = b["end_char"].as_u64().unwrap();
        a0 < b1 && b0 < a1
    };
    for cue in &rel_cues {
        let conn = &cue["connector"];
        let c1 = &cue["clean_arg1_span"];
        let c2 = &cue["clean_arg2_span"];
        assert!(!spans_overlap(c1, conn) && !spans_overlap(c2, conn));
        if !cue["correlate"].is_null() {
            assert!(!spans_overlap(c1, &cue["correlate"]) && !spans_overlap(c2, &cue["correlate"]));
        }
        let t1 = cue["clean_arg1_text"].as_str().unwrap_or("");
        let t2 = cue["clean_arg2_text"].as_str().unwrap_or("");
        assert!(t1.chars().any(char::is_alphanumeric));
        assert!(t2.chars().any(char::is_alphanumeric));
        assert_eq!(t1, "山路濕滑");
        assert_eq!(t2, "商隊放慢了腳步");
    }
}
