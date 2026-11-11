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
