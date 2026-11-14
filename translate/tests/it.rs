//! End-to-end functional, linguistic, structural, batch, and CLI integration tests for
//! `faberline-nlp-translate` (`nlp.translate.v1`).

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use faberline_nlp::Span;
use faberline_nlp::doc::model::Backend;
use faberline_nlp_translate::batch::{BatchConfig, run_translate_batch};
use faberline_nlp_translate::lock::{LockOrigin, SpanLockTable};
use faberline_nlp_translate::postprocess::{
    fix_english_articles_and_case, fix_korean_particles, localize_quotation_marks,
};
use faberline_nlp_translate::{
    GlossaryEntry, GlossaryKind, OpaquePolicy, OutputFormat, PronounClass, SpeechRegister,
    TRANSLATE_BATCH_RECEIPT_SCHEMA, TRANSLATE_SCHEMA, TargetVariant, TranslateManifestReceipt,
    TranslateOptions, TranslateUnknownKind, TranslatedDocument, TranslationGlossary,
    TranslationTrack, analyze_and_translate, render_output, sha256_hex,
};

fn slice_chars(s: &str, span: Span) -> String {
    s.chars()
        .skip(span.start_char)
        .take(span.end_char.saturating_sub(span.start_char))
        .collect()
}

fn assert_valid_doc(src: &str, doc: &TranslatedDocument) {
    assert_eq!(doc.schema, TRANSLATE_SCHEMA);
    assert_eq!(doc.source_chars, src.chars().count());
    assert_eq!(doc.target_chars, doc.target_text.chars().count());
    assert_eq!(doc.source_sha256, sha256_hex(src));
    assert_eq!(doc.target_sha256, sha256_hex(&doc.target_text));

    for p in &doc.paragraphs {
        assert!(p.source_span.start_char <= p.source_span.end_char);
        assert!(p.source_span.end_char <= doc.source_chars);
        assert!(p.target_span.start_char <= p.target_span.end_char);
        assert!(p.target_span.end_char <= doc.target_chars);
    }

    for (idx, s) in doc.sentences.iter().enumerate() {
        assert_eq!(s.sentence, idx);
        assert!(s.source_span.start_char <= s.source_span.end_char);
        assert!(s.source_span.end_char <= doc.source_chars);
        assert!(s.target_span.start_char <= s.target_span.end_char);
        assert!(s.target_span.end_char <= doc.target_chars);
        assert_eq!(slice_chars(&doc.target_text, s.target_span), s.target_text);
    }

    for c in &doc.clause_alignments {
        assert!(c.source_span.start_char <= c.source_span.end_char);
        assert!(c.source_span.end_char <= doc.source_chars);
        assert!(c.target_span.start_char <= c.target_span.end_char);
        assert!(c.target_span.end_char <= doc.target_chars);
    }

    for h in &doc.glossary_hits {
        assert!(h.source_span.start_char <= h.source_span.end_char);
        assert!(h.source_span.end_char <= doc.source_chars);
        assert!(h.target_span.start_char <= h.target_span.end_char);
        assert!(h.target_span.end_char <= doc.target_chars);
        assert!(!slice_chars(src, h.source_span).is_empty());
        assert!(!slice_chars(&doc.target_text, h.target_span).is_empty());
    }

    for op in &doc.opaque_spans {
        assert!(op.source_span.start_char <= op.source_span.end_char);
        assert!(op.source_span.end_char <= doc.source_chars);
        if let Some(ts) = op.target_span {
            assert!(ts.start_char <= ts.end_char);
            assert!(ts.end_char <= doc.target_chars);
            if op.policy == OpaquePolicy::VerbatimPassthrough {
                assert_eq!(
                    slice_chars(src, op.source_span),
                    slice_chars(&doc.target_text, ts),
                    "verbatim opaque span must match byte-for-byte"
                );
            }
        } else {
            assert_eq!(op.policy, OpaquePolicy::OmitEditorial);
        }
    }

    // Roundtrip JSON serialization determinism check.
    let j1 = render_output(doc, src, OutputFormat::Json).unwrap();
    let parsed: TranslatedDocument = serde_json::from_str(&j1).unwrap();
    assert_eq!(&parsed, doc);
}

fn translate_models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

// ---------------------------------------------------------------------------
// 1. Track A: Chinese Variant Conversion (`zh-Hans ↔ zh-Hant`)
// ---------------------------------------------------------------------------

#[test]
fn track_a_zh_hans_to_zh_hant_polysemous_and_cross_strait() {
    let src = concat!(
        "她的头发很长，明天早上出发。\n\n",
        "王后在三天后抵达宫殿。\n\n",
        "房间很干净，他在干活，不要干涉。\n\n",
        "屋里很安静，我们走了三公里。\n\n",
        "他吃了一碗牛肉面，静静看着湖面。\n\n",
        "这没关系，请联系系统管理员。\n\n",
        "这个软件占用了太多内存，需要优化算法和服务器网络。"
    );
    let opts = TranslateOptions::new(TargetVariant::ZhHant);
    let glossary = TranslationGlossary::new();
    let doc = analyze_and_translate(src, &opts, &glossary, None, Backend::Onnx).unwrap();
    assert_valid_doc(src, &doc);
    assert_eq!(doc.track, TranslationTrack::VariantRule);
    assert!(
        doc.unknown.is_empty(),
        "unexpected unknown: {:?}",
        doc.unknown
    );

    let t = &doc.target_text;
    assert!(
        t.contains("頭髮很長") && t.contains("明天早上出發"),
        "got: {t}"
    );
    assert!(t.contains("王后在三天後抵達"), "got: {t}");
    assert!(
        t.contains("房間很乾淨") && t.contains("他在幹活") && t.contains("不要干涉"),
        "got: {t}"
    );
    assert!(
        t.contains("屋裡很安靜") && t.contains("走了三公里"),
        "got: {t}"
    );
    assert!(
        t.contains("吃了一碗牛肉麵") && t.contains("湖面"),
        "got: {t}"
    );
    assert!(
        t.contains("這沒關係") && t.contains("請聯繫系統管理員"),
        "got: {t}"
    );
    assert!(
        t.contains("軟體")
            && t.contains("記憶體")
            && t.contains("演算法")
            && t.contains("伺服器")
            && t.contains("網路"),
        "cross-strait terms missing: {t}"
    );
}

#[test]
fn track_a_zh_hant_to_zh_hans_and_hant_idempotence() {
    let hant = "王后看著他的頭髮與乾糧，走進屋裡吃了一碗牛肉麵。這個軟體佔用了太多記憶體，需要調整演算法與伺服器網路。";

    // Converting already-Traditional text to `zh-Hant` must preserve Traditional distinctions.
    let keep = analyze_and_translate(
        hant,
        &TranslateOptions::new(TargetVariant::ZhHant),
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_doc(hant, &keep);
    assert!(keep.target_text.contains("王后"));
    assert!(keep.target_text.contains("頭髮"));
    assert!(keep.target_text.contains("乾糧"));
    assert!(keep.target_text.contains("屋裡"));
    assert!(keep.target_text.contains("牛肉麵"));

    // Converting `zh-Hant -> zh-Hans` maps both characters and cross-strait vocabulary.
    let hans = analyze_and_translate(
        hant,
        &TranslateOptions::new(TargetVariant::ZhHans),
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_doc(hant, &hans);
    assert_eq!(hans.track, TranslationTrack::VariantRule);
    let s = &hans.target_text;
    assert!(
        s.contains("头发") && s.contains("干粮") && s.contains("屋里") && s.contains("牛肉面"),
        "got: {s}"
    );
    assert!(
        s.contains("软件")
            && s.contains("内存")
            && s.contains("算法")
            && s.contains("服务器")
            && s.contains("网络"),
        "got: {s}"
    );
}

// ---------------------------------------------------------------------------
// 2. Track A: English Variant Conversion (`en-US ↔ en-GB`)
// ---------------------------------------------------------------------------

#[test]
fn track_a_en_us_gb_pos_disambiguation_and_casing() {
    let us_src = "COLORFUL COLORS! Daily practice helps doctors practice medicine at the center. The traveling organizer analyzed the defense.";
    let mut opts_gb = TranslateOptions::new(TargetVariant::EnGb);
    opts_gb.localize_quotes = false;
    let gb_doc = analyze_and_translate(
        us_src,
        &opts_gb,
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_doc(us_src, &gb_doc);
    assert_eq!(gb_doc.track, TranslationTrack::VariantRule);

    let gb = &gb_doc.target_text;
    assert!(gb.starts_with("COLOURFUL COLOURS!"), "ALLCAPS failed: {gb}");
    assert!(
        gb.contains("Daily practice helps doctors practise medicine at the centre."),
        "POS practice/practise failed: {gb}"
    );
    assert!(
        gb.contains("The travelling organiser analysed the defence."),
        "inflected GB conversion failed: {gb}"
    );

    // Reverse GB -> US conversion, including licence (NOUN) vs license (VERB).
    let gb_licence_clean =
        "Her driving licence allows the council to license local theatres and organise colours.";
    let mut opts_us = TranslateOptions::new(TargetVariant::EnUs);
    opts_us.localize_quotes = false;
    let us_doc = analyze_and_translate(
        gb_licence_clean,
        &opts_us,
        &TranslationGlossary::new(),
        None,
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_doc(gb_licence_clean, &us_doc);
    assert_eq!(
        us_doc.target_text,
        "Her driving license allows the council to license local theaters and organize colors."
    );
}

// ---------------------------------------------------------------------------
// 3. Opaque Span Protection, Heading Localization & Editorial Filtering
// ---------------------------------------------------------------------------

#[test]
fn opaque_code_math_urls_headings_and_editorial_policies() {
    let src = concat!(
        "# 第十二章 软件与系统\n\n",
        "- 第一项配置：访问 https://faberline.dev/api?v=1&x=2 或发邮件给 ops@faberline.dev 🚀\n",
        "- 第二项公式：内联 `$E = mc^2$` 与 `fn run_memory_check(软件: &str)` 保持不变。\n\n",
        "```rust\n",
        "fn keep_exact() {\n",
        "    let s = \"软件与内存完全保持原样\";\n",
        "}\n",
        "```\n\n",
        "$$\\sum_{k=1}^{n} \\alpha_k = 1$$\n\n",
        "求月票！求推荐票！本章未完请点击下一页继续阅读！"
    );

    let mut opts = TranslateOptions::new(TargetVariant::ZhHant);
    opts.mask_editorial = true;
    opts.omit_masked_editorial = true;

    let doc = analyze_and_translate(src, &opts, &TranslationGlossary::new(), None, Backend::Onnx)
        .unwrap();
    assert_valid_doc(src, &doc);

    let t = &doc.target_text;
    assert!(t.contains("# 第十二章 軟體與系統"), "heading failed: {t}");
    assert!(
        t.contains("https://faberline.dev/api?v=1&x=2"),
        "url altered: {t}"
    );
    assert!(t.contains("ops@faberline.dev"), "email altered: {t}");
    assert!(t.contains("🚀"), "emoji altered: {t}");
    assert!(
        t.contains("`fn run_memory_check(软件: &str)`"),
        "inline code must not convert simplified chars inside backticks: {t}"
    );
    assert!(
        t.contains("let s = \"软件与内存完全保持原样\";"),
        "fenced code block must remain byte-for-byte identical: {t}"
    );
    assert!(
        t.contains("$$\\sum_{k=1}^{n} \\alpha_k = 1$$"),
        "display math altered: {t}"
    );
    assert!(
        !t.contains("求月票"),
        "editorial footer should be omitted: {t}"
    );
    assert!(
        doc.opaque_spans
            .iter()
            .any(|o| o.policy == OpaquePolicy::OmitEditorial),
        "expected OmitEditorial record in opaque_spans"
    );
}

#[test]
fn cross_language_heading_structural_localization() {
    let mdir = translate_models_dir();
    let src = "# 第十二章 風雪\n\n序章 黎明";

    for (variant, expected_ch, expected_pro) in [
        (TargetVariant::EnUs, "# Chapter 12:", "Prologue:"),
        (TargetVariant::Ja, "# 第12章", "プロローグ"),
        (TargetVariant::Ko, "# 제12장", "프롤로그"),
    ] {
        let mut opts = TranslateOptions::new(variant);
        opts.model_id = "translate-test-tiny".to_owned();
        let doc = analyze_and_translate(
            src,
            &opts,
            &TranslationGlossary::new(),
            Some(&mdir),
            Backend::Onnx,
        )
        .unwrap();
        assert_valid_doc(src, &doc);
        assert!(
            doc.target_text.starts_with(expected_ch),
            "{variant}: expected prefix {expected_ch:?}, got {:?}",
            doc.target_text
        );
        assert!(
            doc.target_text.contains(expected_pro),
            "{variant}: expected prologue {expected_pro:?}, got {:?}",
            doc.target_text
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Glossary Locking, Placeholder Conservation & Morphological Repair
// ---------------------------------------------------------------------------

#[test]
fn korean_josa_harmony_english_articles_and_placeholder_conservation() {
    // 4a. Korean Hangul Batchim Josa harmony (consonant, vowel, and Rieul `ㄹ` Batchim).
    let mut tracked = vec![Span {
        start_char: 0,
        end_char: 3,
    }];
    let ko_raw = "린위안는 첸보은 서울역로 가고 서울으로 갔다.";
    let ko_fixed = fix_korean_particles(ko_raw, &tracked.clone(), &mut tracked);
    assert_eq!(ko_fixed, "린위안은 첸보는 서울역으로 가고 서울로 갔다.");

    // 4b. English indefinite article `a/an` + capitalization while respecting protected spans.
    let en_raw = "a hour ago, an university team bought a 8-core server and `a hour` log.";
    let code_start = en_raw.find("`a hour`").unwrap();
    let prot = vec![Span {
        start_char: code_start,
        end_char: code_start + 8,
    }];
    let mut en_tracked = prot.clone();
    let en_fixed = fix_english_articles_and_case(en_raw, &prot, &mut en_tracked, true);
    assert_eq!(
        en_fixed,
        "An hour ago, a university team bought an 8-core server and `a hour` log."
    );
    assert_eq!(slice_chars(&en_fixed, en_tracked[0]), "`a hour`");

    // 4c. Quotation mark localization.
    let q_hant = localize_quotation_marks("“Hello, ‘world’!”", TargetVariant::ZhHant, &[]);
    assert_eq!(q_hant, "「Hello, 『world』!」");
    let q_en = localize_quotation_marks("他說：「你好，『世界』！」", TargetVariant::EnUs, &[]);
    assert_eq!(q_en, "他說：“你好，‘世界’！”");

    // 4d. 100% Placeholder conservation when a neural hypothesis drops `[[PH0]]`.
    let lock_tbl = SpanLockTable::build(vec![(
        1,
        Span {
            start_char: 0,
            end_char: 2,
        },
        "Lin Yuan".to_owned(),
        LockOrigin::Glossary {
            canonical: "lin_yuan".to_owned(),
            kind: GlossaryKind::Actor,
        },
        true,
    )]);
    assert_eq!(lock_tbl.locks[0].placeholder, "[[PH0]]");
    let dropped_rep = lock_tbl.restore("opened the door quietly.");
    assert_eq!(dropped_rep.issues.len(), 1);
    assert_eq!(
        dropped_rep.issues[0].0,
        TranslateUnknownKind::PlaceholderRepaired
    );
    assert!(dropped_rep.text.contains("Lin Yuan"));
    assert_eq!(
        slice_chars(&dropped_rep.text, dropped_rep.restored[0].target_span),
        "Lin Yuan"
    );
}

// ---------------------------------------------------------------------------
// 5. Narrative Conditioning + End-to-End Track B Seq2Seq (`translate-test-tiny`)
// ---------------------------------------------------------------------------

fn sample_glossary() -> TranslationGlossary {
    let mut g = TranslationGlossary::new();
    g.actors.push(GlossaryEntry {
        canonical: "lin_yuan".to_owned(),
        surfaces: vec!["林淵".to_owned(), "阿淵".to_owned()],
        targets: BTreeMap::from([
            (TargetVariant::EnUs, "Lin Yuan".to_owned()),
            (TargetVariant::Ko, "린위안".to_owned()),
            (TargetVariant::Ja, "リン・ユエン".to_owned()),
        ]),
        pronoun_class: PronounClass::FeminineSingular,
        speech_register: SpeechRegister::Polite,
    });
    g.places.push(GlossaryEntry {
        canonical: "qingyun_tower".to_owned(),
        surfaces: vec!["青雲塔".to_owned()],
        targets: BTreeMap::from([
            (TargetVariant::EnUs, "Azure Cloud Tower".to_owned()),
            (TargetVariant::Ko, "청운탑".to_owned()),
        ]),
        pronoun_class: PronounClass::NeuterSingular,
        speech_register: SpeechRegister::Unspecified,
    });
    g
}

#[test]
fn track_b_graceful_fallback_without_model() {
    let src = concat!(
        "林淵走進青雲塔。推開了沉重的木門。\n\n",
        "林淵輕聲說：「我沒有看見任何人，請查看 `config.toml`。」"
    );
    let glossary = sample_glossary();
    let no_model_opts = TranslateOptions::new(TargetVariant::EnUs);
    let fallback_doc =
        analyze_and_translate(src, &no_model_opts, &glossary, None, Backend::Onnx).unwrap();
    assert_valid_doc(src, &fallback_doc);
    assert!(
        fallback_doc
            .unknown
            .iter()
            .any(|u| u.kind == TranslateUnknownKind::NoModel),
        "expected NoModel unknown, got {:?}",
        fallback_doc.unknown
    );
    assert!(fallback_doc.target_text.contains("Lin Yuan"));
    assert!(fallback_doc.target_text.contains("Azure Cloud Tower"));
    assert!(fallback_doc.target_text.contains("`config.toml`"));
}

#[cfg(feature = "onnx")]
#[test]
fn track_b_narrative_conditioning_and_seq2seq_end_to_end() {
    let src = concat!(
        "林淵走進青雲塔。推開了沉重的木門。\n\n",
        "林淵輕聲說：「我沒有看見任何人，請查看 `config.toml`。」"
    );
    let glossary = sample_glossary();
    let mdir = translate_models_dir();

    // With `translate-test-tiny`: full Track B Seq2Seq + narrative conditioning + lock repair.
    let mut seq_opts = TranslateOptions::new(TargetVariant::EnUs);
    seq_opts.model_id = "translate-test-tiny".to_owned();
    let tr_doc =
        analyze_and_translate(src, &seq_opts, &glossary, Some(&mdir), Backend::Onnx).unwrap();
    assert_valid_doc(src, &tr_doc);
    assert_eq!(tr_doc.track, TranslationTrack::NeuralSeq2Seq);

    // Every glossary occurrence + inline backtick code must survive in target_text!
    assert!(
        tr_doc.target_text.contains("Lin Yuan"),
        "got: {}",
        tr_doc.target_text
    );
    assert!(
        tr_doc.target_text.contains("Azure Cloud Tower"),
        "got: {}",
        tr_doc.target_text
    );
    assert!(
        tr_doc.target_text.contains("`config.toml`"),
        "got: {}",
        tr_doc.target_text
    );
    assert!(
        tr_doc.glossary_hits.len() >= 3,
        "expected >=3 glossary hits, got {:?}",
        tr_doc.glossary_hits
    );

    #[cfg(feature = "narrative")]
    {
        let has_narrative_signal = tr_doc.sentences.iter().any(|s| {
            s.conditioning.speaker.as_deref() == Some("lin_yuan")
                || s.conditioning.resolved_pro_drop.as_deref() == Some("lin_yuan")
                || s.conditioning.speech_register == Some(SpeechRegister::Polite)
        });
        assert!(
            has_narrative_signal,
            "expected narrative conditioning signal in {:?}",
            tr_doc
                .sentences
                .iter()
                .map(|s| &s.conditioning)
                .collect::<Vec<_>>()
        );
    }

    // On macOS with `coreml`, `Backend::Coreml` produces the exact same `target_text`!
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        let cm_doc =
            analyze_and_translate(src, &seq_opts, &glossary, Some(&mdir), Backend::Coreml).unwrap();
        assert_valid_doc(src, &cm_doc);
        assert_eq!(cm_doc.target_text, tr_doc.target_text);
        assert_eq!(cm_doc.target_sha256, tr_doc.target_sha256);
    }
}

#[cfg(feature = "onnx")]
#[test]
fn long_multi_clause_sentence_partitions_cleanly_within_src_max_len() {
    let mdir = translate_models_dir();
    let long_src = "春天的風吹過山谷，夏天的雨落在湖面，秋天的落葉鋪滿石階，冬天的白雪覆蓋了古老的屋頂，旅人依然靜靜地向前走去，直到遠方的燈火再次亮起。";
    let mut opts = TranslateOptions::new(TargetVariant::EnUs);
    opts.model_id = "translate-test-tiny".to_owned();

    let doc = analyze_and_translate(
        long_src,
        &opts,
        &TranslationGlossary::new(),
        Some(&mdir),
        Backend::Onnx,
    )
    .unwrap();
    assert_valid_doc(long_src, &doc);
    assert_eq!(doc.sentences.len(), 1);
    assert!(
        doc.sentences[0].conditioning.window_count >= 2 || doc.clause_alignments.len() >= 2,
        "expected multi-window clause alignment, got cond={:?}, clauses={:?}",
        doc.sentences[0].conditioning,
        doc.clause_alignments
    );
}

// ---------------------------------------------------------------------------
// 6. Renderers, Resumable Multi-Worker Batch & CLI Binaries
// ---------------------------------------------------------------------------

#[test]
fn render_formats_and_resumable_batch_pipeline() {
    let tmp = tempfile::tempdir().unwrap();
    let in_dir = tmp.path().join("chapters");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&in_dir).unwrap();

    fs::write(
        in_dir.join("ch01.md"),
        "# 第一章 软件架构\n\n这个软件优化了内存与网络。\n",
    )
    .unwrap();
    fs::write(in_dir.join("ch02.txt"), "王后在三天后检查了服务器算法。\n").unwrap();

    let cfg = BatchConfig {
        input_dir: in_dir.clone(),
        output_dir: out_dir.clone(),
        options: TranslateOptions::new(TargetVariant::ZhHant),
        glossary: TranslationGlossary::new(),
        models_dir: None,
        backend: Backend::Onnx,
        workers: 2,
        resume: true,
        write_bilingual: true,
    };

    let r1 = run_translate_batch(&cfg).unwrap();
    assert_eq!(r1.schema, TRANSLATE_BATCH_RECEIPT_SCHEMA);
    assert_eq!(r1.chapters_total, 2);
    assert_eq!(r1.chapters_resumed_cached, 0);
    assert!(out_dir.join("text/ch01.md").is_file());
    assert!(out_dir.join("bilingual/ch01.bilingual.md").is_file());
    assert!(out_dir.join("translate_manifest_receipt.json").is_file());

    let ch1_out = fs::read_to_string(out_dir.join("text/ch01.md")).unwrap();
    assert!(ch1_out.contains("# 第一章 軟體架構") && ch1_out.contains("記憶體與網路"));

    // Second run with unchanged inputs must reuse 100% of cached chapters.
    let r2 = run_translate_batch(&cfg).unwrap();
    assert_eq!(r2.chapters_total, 2);
    assert_eq!(r2.chapters_resumed_cached, 2);
    assert_eq!(r1.chapters, r2.chapters);

    // Modifying ch02 invalidates only ch02 while keeping ch01 cached!
    fs::write(
        in_dir.join("ch02.txt"),
        "王后在五天后检查了服务器算法与内存。\n",
    )
    .unwrap();
    let r3 = run_translate_batch(&cfg).unwrap();
    assert_eq!(r3.chapters_resumed_cached, 1);
    let receipt_disk: TranslateManifestReceipt = serde_json::from_str(
        &fs::read_to_string(out_dir.join("translate_manifest_receipt.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt_disk, r3);
}

#[test]
fn cli_binaries_smoke_test() {
    let bin = env!("CARGO_BIN_EXE_nlp_translate");
    let mut child = Command::new(bin)
        .args(["--target", "zh-Hant", "--format", "bilingual-jsonl"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all("这个软件的内存很小。\n".as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["target_text"], "這個軟體的記憶體很小。");

    // Batch CLI smoke test.
    let tmp = tempfile::tempdir().unwrap();
    let in_dir = tmp.path().join("in");
    let out_dir = tmp.path().join("out");
    fs::create_dir_all(&in_dir).unwrap();
    fs::write(in_dir.join("001.txt"), "头发很长，明天出发。\n").unwrap();

    let batch_bin = env!("CARGO_BIN_EXE_nlp_translate_batch");
    let status = Command::new(batch_bin)
        .args([
            "--input-dir",
            in_dir.to_str().unwrap(),
            "--output-dir",
            out_dir.to_str().unwrap(),
            "--target",
            "zh-Hant",
            "--workers",
            "1",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let text_out = fs::read_to_string(out_dir.join("text/001.txt")).unwrap();
    assert_eq!(text_out.trim(), "頭髮很長，明天出發。");
}
