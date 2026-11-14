use faberline_nlp::Span;
use faberline_nlp::doc::{Document, Lang, Options, Reason, Unknown, analyze_document};
use serde_json::json;
use std::io::Write;
use std::process::{Command, Stdio};

fn slice(text: &str, s: faberline_nlp::Span) -> String {
    text.chars()
        .skip(s.start_char)
        .take(s.end_char - s.start_char)
        .collect()
}

fn analyze(text: &str) -> Document {
    analyze_document(text, &Options::default())
}

fn texts(text: &str, spans: &[Span]) -> Vec<String> {
    spans.iter().map(|s| slice(text, *s)).collect()
}

fn paragraph_texts(text: &str, doc: &Document) -> Vec<String> {
    let spans: Vec<Span> = doc.segmentation.paragraphs.iter().map(|p| p.span).collect();
    texts(text, &spans)
}

fn sentence_texts(text: &str, doc: &Document) -> Vec<String> {
    let spans: Vec<Span> = doc.segmentation.sentences.iter().map(|s| s.span).collect();
    texts(text, &spans)
}

fn clause_texts(text: &str, doc: &Document) -> Vec<String> {
    let spans: Vec<Span> = doc.segmentation.clauses.iter().map(|c| c.span).collect();
    texts(text, &spans)
}

#[test]
fn header_hash_and_json() {
    let doc = analyze("abc");
    assert_eq!(doc.schema, "nlp.doc.v1");
    assert_eq!(doc.text_chars, 3);
    assert_eq!(
        doc.text_sha256,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    let v = serde_json::to_value(&doc).unwrap();
    for key in [
        "schema",
        "text_chars",
        "text_sha256",
        "options",
        "segmentation",
        "unknown",
    ] {
        assert!(v.get(key).is_some(), "missing key {key}");
    }
    for key in ["paragraphs", "sentences", "clauses"] {
        assert!(v["segmentation"].get(key).is_some(), "missing key {key}");
    }
    assert_eq!(
        v["segmentation"]["sentences"][0]["span"],
        json!({"start_char": 0, "end_char": 3})
    );
    assert_eq!(v["segmentation"]["sentences"][0]["lang"], "en");

    let empty = analyze("");
    assert_eq!(empty.text_chars, 0);
    assert_eq!(
        empty.text_sha256,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert!(empty.segmentation.paragraphs.is_empty());
    assert!(empty.segmentation.sentences.is_empty());
    assert!(empty.segmentation.clauses.is_empty());
    assert!(empty.unknown.is_empty());
}

#[test]
fn novel_zh() {
    let text = "他走進屋子，看見桌上有一封信。\n\n「你來了。」她說。";
    let doc = analyze(text);
    assert_eq!(
        paragraph_texts(text, &doc),
        ["他走進屋子，看見桌上有一封信。", "「你來了。」她說。"]
    );
    assert_eq!(
        sentence_texts(text, &doc),
        ["他走進屋子，看見桌上有一封信。", "「你來了。」", "她說。"]
    );
    let para: Vec<usize> = doc
        .segmentation
        .sentences
        .iter()
        .map(|s| s.paragraph)
        .collect();
    assert_eq!(para, [0, 1, 1]);
    assert_eq!(
        clause_texts(text, &doc),
        ["他走進屋子", "看見桌上有一封信。", "「你來了。」", "她說。"]
    );
    let sent: Vec<usize> = doc
        .segmentation
        .clauses
        .iter()
        .map(|c| c.sentence)
        .collect();
    assert_eq!(sent, [0, 0, 1, 2]);
    assert!(
        doc.segmentation
            .paragraphs
            .iter()
            .all(|p| p.lang == Lang::Zh)
    );
    assert!(
        doc.segmentation
            .sentences
            .iter()
            .all(|s| s.lang == Lang::Zh)
    );
    assert!(doc.segmentation.clauses.iter().all(|c| c.lang == Lang::Zh));
}

#[test]
fn paper_en() {
    let text = "Smith et al. (2020) show that Dr. Lee's method works. See Fig. 3 for details. It costs 3.5 units, e.g. ten.";
    let doc = analyze(text);
    assert_eq!(
        sentence_texts(text, &doc),
        [
            "Smith et al. (2020) show that Dr. Lee's method works.",
            "See Fig. 3 for details.",
            "It costs 3.5 units, e.g. ten.",
        ]
    );
    let third: Vec<Span> = doc
        .segmentation
        .clauses
        .iter()
        .filter(|c| c.sentence == 2)
        .map(|c| c.span)
        .collect();
    assert_eq!(texts(text, &third), ["It costs 3.5 units", "e.g. ten."]);
    assert!(
        doc.segmentation
            .sentences
            .iter()
            .all(|s| s.lang == Lang::En)
    );
}

#[test]
fn abbreviation_and_number_cases() {
    let t = "Mr. Lee arrived. He sat.";
    assert_eq!(
        sentence_texts(t, &analyze(t)),
        ["Mr. Lee arrived.", "He sat."]
    );

    let t = "Apples, pears, etc. Then we left.";
    assert_eq!(analyze(t).segmentation.sentences.len(), 2);

    let t = "Apples, pears, etc. and more.";
    assert_eq!(analyze(t).segmentation.sentences.len(), 1);

    let t = "At 10:30 it cost 1,000 yen.";
    let doc = analyze(t);
    assert_eq!(doc.segmentation.sentences.len(), 1);
    assert_eq!(doc.segmentation.clauses.len(), 1);

    let t = "Wait... what? Yes.";
    assert_eq!(sentence_texts(t, &analyze(t)), ["Wait... what?", "Yes."]);
}

#[test]
fn abbreviated_month_dot() {
    // A digit after an abbreviated month keeps the sentence whole.
    let t = "Nov. 10, 2026 was fine.";
    assert_eq!(sentence_texts(t, &analyze(t)), [t]);
    let t = "Sept. 3rd was fine.";
    assert_eq!(sentence_texts(t, &analyze(t)), [t]);
    // Any other word after the month dot still ends the sentence.
    let t = "It was Dec. Then we ran.";
    assert_eq!(
        sentence_texts(t, &analyze(t)),
        ["It was Dec.", "Then we ran."]
    );
    // May is not abbreviated, and a dot after a non-month word is unchanged.
    let t = "It was May. 10 was the day.";
    assert_eq!(
        sentence_texts(t, &analyze(t)),
        ["It was May.", "10 was the day."]
    );
}

#[test]
fn casual_lines() {
    let t = "今天好累\n晚上吃什麼\n不知道";
    let doc = analyze(t);
    assert_eq!(doc.segmentation.paragraphs.len(), 1);
    assert_eq!(
        sentence_texts(t, &doc),
        ["今天好累", "晚上吃什麼", "不知道"]
    );
    assert!(
        doc.segmentation
            .sentences
            .iter()
            .all(|s| s.lang == Lang::Zh)
    );

    let t = "omg this is so good\nwe should go again\nlol";
    let doc = analyze(t);
    assert_eq!(doc.segmentation.sentences.len(), 3);
    assert!(
        doc.segmentation
            .sentences
            .iter()
            .all(|s| s.lang == Lang::En)
    );

    let t = "我今天去了 Starbucks\nthe coffee was great";
    let doc = analyze(t);
    assert_eq!(doc.segmentation.sentences.len(), 2);
    let langs: Vec<Lang> = doc.segmentation.sentences.iter().map(|s| s.lang).collect();
    assert_eq!(langs, [Lang::Mixed, Lang::En]);
}

#[test]
fn double_space_boundary() {
    let t = "好啊  走吧";
    assert_eq!(sentence_texts(t, &analyze(t)), ["好啊", "走吧"]);

    let t = "ok  go";
    assert_eq!(sentence_texts(t, &analyze(t)), ["ok", "go"]);

    let t = "ok go";
    assert_eq!(analyze(t).segmentation.sentences.len(), 1);
}

#[test]
fn soft_wrap_vs_short_line() {
    let t = "This sentence was wrapped by an editor at some width\nand it continues here.";
    assert_eq!(analyze(t).segmentation.sentences.len(), 1);

    let t = "Short line\nand more";
    assert_eq!(sentence_texts(t, &analyze(t)), ["Short line", "and more"]);
}

#[test]
fn unsupported_language() {
    let t = "Привет, мир.";
    let doc = analyze(t);
    assert_eq!(doc.segmentation.sentences.len(), 1);
    assert_eq!(doc.segmentation.sentences[0].lang, Lang::Und);
    assert_eq!(
        doc.unknown,
        [Unknown {
            layer: "segmentation".to_owned(),
            span: Span {
                start_char: 0,
                end_char: 12
            },
            reason: Reason::UnsupportedLanguage,
        }]
    );

    let t = "12345";
    let doc = analyze(t);
    assert_eq!(doc.segmentation.sentences[0].lang, Lang::Und);
    assert!(doc.unknown.is_empty());
}

#[test]
fn numeric_offsets() {
    let doc = analyze("你好。Hi!");
    let spans: Vec<(usize, usize)> = doc
        .segmentation
        .sentences
        .iter()
        .map(|s| (s.span.start_char, s.span.end_char))
        .collect();
    assert_eq!(spans, [(0, 3), (3, 6)]);

    let doc = analyze("One.\r\nTwo.\r\n");
    let para: Vec<(usize, usize)> = doc
        .segmentation
        .paragraphs
        .iter()
        .map(|p| (p.span.start_char, p.span.end_char))
        .collect();
    assert_eq!(para, [(0, 10)]);
    let sent: Vec<(usize, usize)> = doc
        .segmentation
        .sentences
        .iter()
        .map(|s| (s.span.start_char, s.span.end_char))
        .collect();
    assert_eq!(sent, [(0, 4), (6, 10)]);
}

#[test]
fn clause_separators() {
    let t = "甲，乙；丙：丁。";
    assert_eq!(clause_texts(t, &analyze(t)), ["甲", "乙", "丙", "丁。"]);

    let t = "a, b; c: d";
    assert_eq!(clause_texts(t, &analyze(t)), ["a", "b", "c", "d"]);

    let t = "a,b";
    assert_eq!(analyze(t).segmentation.clauses.len(), 1);

    let t = "蘋果、香蕉。";
    assert_eq!(analyze(t).segmentation.clauses.len(), 1);
}

#[test]
fn paragraph_rules() {
    let t = "　　第一段。\n　　第二段。";
    assert_eq!(paragraph_texts(t, &analyze(t)), ["第一段。", "第二段。"]);

    let t = "他走了。\n她來了。";
    assert_eq!(analyze(t).segmentation.paragraphs.len(), 2);

    let t = "他走了\n她來了";
    let doc = analyze(t);
    assert_eq!(doc.segmentation.paragraphs.len(), 1);
    assert_eq!(doc.segmentation.sentences.len(), 2);

    let t = "Line one.\nLine two.";
    let doc = analyze(t);
    assert_eq!(doc.segmentation.paragraphs.len(), 1);
    assert_eq!(doc.segmentation.sentences.len(), 2);
}

#[test]
fn invariants_and_determinism() {
    let text = [
        "他走進屋子，看見桌上有一封信。\n\n「你來了。」她說。",
        "Smith et al. (2020) show that Dr. Lee's method works. See Fig. 3 for details. It costs 3.5 units, e.g. ten.",
        "今天好累\n晚上吃什麼\n不知道",
    ]
    .join("\n\n");

    let a = analyze(&text);
    let b = analyze(&text);
    assert_eq!(a, b);
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );

    let seg = &a.segmentation;
    let n_par = seg.paragraphs.len();
    let n_sen = seg.sentences.len();
    let all_spans = seg
        .paragraphs
        .iter()
        .map(|p| p.span)
        .chain(seg.sentences.iter().map(|s| s.span))
        .chain(seg.clauses.iter().map(|c| c.span));
    for s in all_spans {
        assert!(
            s.start_char < s.end_char && s.end_char <= a.text_chars,
            "{s:?}"
        );
    }
    for pair in seg.sentences.windows(2) {
        assert!(pair[0].span.end_char <= pair[1].span.start_char);
    }
    for s in &seg.sentences {
        assert!(s.paragraph < n_par);
        let p = seg.paragraphs[s.paragraph].span;
        assert!(p.start_char <= s.span.start_char && s.span.end_char <= p.end_char);
    }
    for c in &seg.clauses {
        assert!(c.sentence < n_sen);
        let s = seg.sentences[c.sentence].span;
        assert!(s.start_char <= c.span.start_char && c.span.end_char <= s.end_char);
    }
}

#[test]
fn cli_reads_stdin() {
    let spawn = || {
        Command::new(env!("CARGO_BIN_EXE_nlp_doc"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };

    let mut child = spawn();
    child
        .stdin
        .take()
        .unwrap()
        .write_all("你好。Hi!".as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["schema"], "nlp.doc.v1");
    assert_eq!(v["segmentation"]["sentences"].as_array().unwrap().len(), 2);

    let mut child = spawn();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&[0xff, 0xfe])
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).starts_with("nlp_doc:"));
}

#[test]
fn options_serde_backwards_compatibility() {
    let def = Options::default();
    assert_eq!(serde_json::to_value(&def).unwrap(), json!({}));
    assert_eq!(serde_json::from_str::<Options>("{}").unwrap(), def);

    let full = Options {
        names: vec!["白衣老者".to_owned()],
        mask_editorial: true,
        normalize_traditional: true,
        lexicon: vec!["青竹木劍".to_owned()],
        ..Options::default()
    };
    let val = serde_json::to_value(&full).unwrap();
    assert_eq!(
        val,
        json!({
            "names": ["白衣老者"],
            "mask_editorial": true,
            "normalize_traditional": true,
            "lexicon": ["青竹木劍"],
        })
    );
    assert_eq!(serde_json::from_value::<Options>(val).unwrap(), full);
}

#[test]
fn ws1_ws2_v3_silent_noise_and_infix_zero_offset_drift() {
    let text = concat!(
        "第一百零八章古道夜行\n\n",
        "<P那名少丅女與長堊老走上傳丅送台，&qut;需丅要護住肉冇身離開大灬陸，免遭妖獸吞灬噬&qut;。（第一更求月票)</P\n\n",
        "<!over\n"
    );
    let opts = Options {
        mask_editorial: true,
        normalize_traditional: true,
        ..Options::default()
    };
    let doc = analyze_document(text, &opts);
    let seg = &doc.segmentation;
    for s in seg
        .paragraphs
        .iter()
        .map(|p| p.span)
        .chain(seg.sentences.iter().map(|s| s.span))
        .chain(seg.clauses.iter().map(|c| c.span))
        .chain(seg.opaque.iter().map(|o| o.span))
    {
        assert!(s.start_char < s.end_char && s.end_char <= doc.text_chars);
    }
    assert_eq!(doc.segmentation.sentences.len(), 1);
    for tok in &doc.tokens {
        assert!(tok.span.start_char < tok.span.end_char);
        assert!(tok.span.end_char <= doc.text_chars);
        let sent = doc.segmentation.sentences[tok.sentence].span;
        assert!(sent.start_char <= tok.span.start_char && tok.span.end_char <= sent.end_char);
    }
    let surfaces: Vec<String> = doc.tokens.iter().map(|t| slice(text, t.span)).collect();
    for word in [
        "少丅女",
        "長堊老",
        "傳丅送",
        "需丅要",
        "肉冇身",
        "大灬陸",
        "吞灬噬",
    ] {
        assert!(
            surfaces.iter().any(|s| s == word),
            "expected contiguous original-offset token {word:?} in {surfaces:?}"
        );
    }
}

#[test]
fn ja_novel_dialogue_context_and_clauses() {
    // Pure-Kanji dialogue `「約束。」` inside a Japanese paragraph inherits `Lang::Ja`,
    // and Japanese `、` splits clauses while Chinese `、` does not.
    let text = "「約束。」彼は言った。雨が降り、風が吹いた。\n\n「約束。」他說。蘋果、香蕉。";
    let doc = analyze(text);
    assert_eq!(doc.segmentation.paragraphs.len(), 2);
    assert_eq!(doc.segmentation.paragraphs[0].lang, Lang::Ja);
    assert_eq!(doc.segmentation.paragraphs[1].lang, Lang::Zh);
    assert!(!doc.unknown.iter().any(|u| u.layer == "segmentation"));

    assert_eq!(
        sentence_texts(text, &doc),
        [
            "「約束。」",
            "彼は言った。",
            "雨が降り、風が吹いた。",
            "「約束。」",
            "他說。",
            "蘋果、香蕉。",
        ]
    );
    let sent_langs: Vec<Lang> = doc.segmentation.sentences.iter().map(|s| s.lang).collect();
    assert_eq!(
        sent_langs,
        [Lang::Ja, Lang::Ja, Lang::Ja, Lang::Zh, Lang::Zh, Lang::Zh]
    );

    assert_eq!(
        clause_texts(text, &doc),
        [
            "「約束。」",
            "彼は言った。",
            "雨が降り",
            "風が吹いた。",
            "「約束。」",
            "他說。",
            "蘋果、香蕉。",
        ]
    );
    let clause_langs: Vec<Lang> = doc.segmentation.clauses.iter().map(|c| c.lang).collect();
    assert_eq!(
        clause_langs,
        [
            Lang::Ja,
            Lang::Ja,
            Lang::Ja,
            Lang::Ja,
            Lang::Zh,
            Lang::Zh,
            Lang::Zh,
        ]
    );
}

#[test]
fn ja_academic_colloquial_exclusive_kanji_and_paragraph_breaks() {
    // Academic Japanese with `、` clause splits.
    let paper = "本研究では、提案手法の有効性を検証した。その結果、精度が向上した。";
    let doc = analyze(paper);
    assert_eq!(
        sentence_texts(paper, &doc),
        [
            "本研究では、提案手法の有効性を検証した。",
            "その結果、精度が向上した。"
        ]
    );
    assert_eq!(
        clause_texts(paper, &doc),
        [
            "本研究では",
            "提案手法の有効性を検証した。",
            "その結果",
            "精度が向上した。",
        ]
    );
    assert!(
        doc.segmentation
            .sentences
            .iter()
            .all(|s| s.lang == Lang::Ja)
    );
    assert!(!doc.unknown.iter().any(|u| u.layer == "segmentation"));

    // Standalone sentences with JA-exclusive shinjitai/kokuji and no kana.
    for s in ["東京駅発。", "試験続行。", "〆"] {
        let d = analyze(s);
        assert_eq!(d.segmentation.sentences.len(), 1, "{s}");
        assert_eq!(d.segmentation.sentences[0].lang, Lang::Ja, "{s}");
        assert!(!d.unknown.iter().any(|u| u.layer == "segmentation"), "{s}");
    }

    // Colloquial Kana: prolonged sound mark, halfwidth katakana + `｡`, `‥` ellipsis + comma guard,
    // and Kana-heavy single-newline paragraph break on terminators vs non-terminated lines.
    let kana_paras = "そうだね。\nうん、いこう！";
    let d_paras = analyze(kana_paras);
    assert_eq!(
        paragraph_texts(kana_paras, &d_paras),
        ["そうだね。", "うん、いこう！"]
    );
    assert!(
        d_paras
            .segmentation
            .paragraphs
            .iter()
            .all(|p| p.lang == Lang::Ja)
    );

    let bracket_paras = "彼女は【了解】\nつぎのひ、でかけた。";
    assert_eq!(analyze(bracket_paras).segmentation.paragraphs.len(), 2);

    let casual_ja = "きょうはつかれた\nなにたべる\nコーヒーとﾃｽﾄ｡";
    let d_casual = analyze(casual_ja);
    assert_eq!(d_casual.segmentation.paragraphs.len(), 1);
    assert_eq!(
        sentence_texts(casual_ja, &d_casual),
        ["きょうはつかれた", "なにたべる", "コーヒーとﾃｽﾄ｡"]
    );
    assert!(
        d_casual
            .segmentation
            .sentences
            .iter()
            .all(|s| s.lang == Lang::Ja)
    );

    let ellipsis_ja = "「‥‥、やはり行こう。」彼は立ち上がった。";
    assert_eq!(
        sentence_texts(ellipsis_ja, &analyze(ellipsis_ja)),
        ["「‥‥、やはり行こう。」", "彼は立ち上がった。"]
    );
}

#[test]
fn ko_novel_academic_colloquial_and_tight_comma() {
    let novel = "그는 조용히 문을 열었다.\n\n“왔구나.” 그녀가 말했다.";
    let doc = analyze(novel);
    assert_eq!(doc.segmentation.paragraphs.len(), 2);
    assert_eq!(
        sentence_texts(novel, &doc),
        ["그는 조용히 문을 열었다.", "“왔구나.”", "그녀가 말했다."]
    );
    assert!(
        doc.segmentation
            .paragraphs
            .iter()
            .all(|p| p.lang == Lang::Ko)
    );
    assert!(
        doc.segmentation
            .sentences
            .iter()
            .all(|s| s.lang == Lang::Ko)
    );
    assert!(!doc.unknown.iter().any(|u| u.layer == "segmentation"));

    let paper = "본 연구에서는 새로운 분석 방법을 제안한다. 실험 결과는 유의미하였다.";
    let doc = analyze(paper);
    assert_eq!(
        sentence_texts(paper, &doc),
        [
            "본 연구에서는 새로운 분석 방법을 제안한다.",
            "실험 결과는 유의미하였다.",
        ]
    );
    assert!(
        doc.segmentation
            .sentences
            .iter()
            .all(|s| s.lang == Lang::Ko)
    );

    // Compatibility Jamo (`ㅋㅋㅋ`, `ㅎㅎ`) colloquial Korean.
    let jamo = "ㅋㅋㅋ 진짜 웃기네\nㅎㅎ 맞아";
    let doc = analyze(jamo);
    assert_eq!(
        sentence_texts(jamo, &doc),
        ["ㅋㅋㅋ 진짜 웃기네", "ㅎㅎ 맞아"]
    );
    assert!(
        doc.segmentation
            .sentences
            .iter()
            .all(|s| s.lang == Lang::Ko)
    );
    let bare_jamo = analyze("ㅋㅋㅋ");
    assert_eq!(bare_jamo.segmentation.sentences[0].lang, Lang::Ko);
    assert!(!bare_jamo.unknown.iter().any(|u| u.layer == "segmentation"));

    // Tight comma between Hangul syllables splits clauses, while numeric `1,000` stays intact.
    let tight = "먹고,마시고, 1,000원을 냈다.";
    let doc = analyze(tight);
    assert_eq!(doc.segmentation.sentences.len(), 1);
    assert_eq!(
        clause_texts(tight, &doc),
        ["먹고", "마시고", "1,000원을 냈다."]
    );
    assert!(doc.segmentation.clauses.iter().all(|c| c.lang == Lang::Ko));
}

#[test]
fn ko_dotted_date_guard() {
    let t = "2026. 11. 13. 오후 서울에서 회의가 열렸다.";
    let doc = analyze(t);
    assert_eq!(sentence_texts(t, &doc), [t]);
    assert_eq!(doc.segmentation.sentences[0].lang, Lang::Ko);

    let t = "오늘은 11. 13. 금요일이다.";
    assert_eq!(sentence_texts(t, &analyze(t)), [t]);

    let t = "2026. 11. 기준 자료이다.";
    assert_eq!(sentence_texts(t, &analyze(t)), [t]);

    // Single number + dot still splits before a new sentence in both Korean and English.
    let t = "2020. 새해가 밝았다.";
    assert_eq!(sentence_texts(t, &analyze(t)), ["2020.", "새해가 밝았다."]);

    let t = "2020. The year began.";
    assert_eq!(sentence_texts(t, &analyze(t)), ["2020.", "The year began."]);

    let t = "It was May. 10 was the day.";
    assert_eq!(
        sentence_texts(t, &analyze(t)),
        ["It was May.", "10 was the day."]
    );
}

#[test]
fn multilingual_mixed_lang_lattice() {
    // ja + en
    let doc = analyze("東京駅で Rust を学ぶ。");
    assert_eq!(doc.segmentation.sentences[0].lang, Lang::Mixed);

    // ko + en
    let doc = analyze("서울에서 AI 회의가 열렸다.");
    assert_eq!(doc.segmentation.sentences[0].lang, Lang::Mixed);

    // ko + han (Hanja)
    let doc = analyze("漢字와 한글을 함께 쓴다.");
    assert_eq!(doc.segmentation.sentences[0].lang, Lang::Mixed);

    // ja + ko
    let doc = analyze("日本語と한국어を比べる。");
    assert_eq!(doc.segmentation.sentences[0].lang, Lang::Mixed);

    // Fullwidth Latin counts as En.
    let doc = analyze("ＡＢＣ");
    assert_eq!(doc.segmentation.sentences[0].lang, Lang::En);
}

#[test]
fn zh_en_quoted_attribution_ellipsis_and_abbreviations() {
    // English quoted `?`/`!` + lowercase attribution verb stays one sentence.
    let t = "\"Really?\" she asked. \"No!\" he replied.";
    assert_eq!(
        sentence_texts(t, &analyze(t)),
        ["\"Really?\" she asked.", "\"No!\" he replied."]
    );

    let t = "“Wait!” whispered Tom.";
    assert_eq!(sentence_texts(t, &analyze(t)), [t]);

    let t = "'Why?' she wondered.";
    assert_eq!(sentence_texts(t, &analyze(t)), [t]);

    // Capitalized new sentence after closing quote still splits.
    let t = "\"Really?\" She left.";
    assert_eq!(sentence_texts(t, &analyze(t)), ["\"Really?\"", "She left."]);

    // Newline after closing quote still splits.
    let t = "\"Really?\"\nshe asked.";
    assert_eq!(
        sentence_texts(t, &analyze(t)),
        ["\"Really?\"", "she asked."]
    );

    // Ellipsis followed directly by clause separator stays one sentence.
    let t = "「……，但是眾人已經抵達山谷。」";
    let doc = analyze(t);
    assert_eq!(sentence_texts(t, &doc), [t]);
    assert_eq!(clause_texts(t, &doc), ["「……", "但是眾人已經抵達山谷。」"]);

    // Ellipsis followed by closer still ends the quoted sentence.
    let t = "「……」她說。";
    assert_eq!(sentence_texts(t, &analyze(t)), ["「……」", "她說。"]);

    // Capitalized-only title & org abbreviations.
    let t = "Gen. Smith and Sen. Lee met at Univ. of California with Apple Inc. and Sony Ltd. executives.";
    assert_eq!(sentence_texts(t, &analyze(t)), [t]);

    // Lowercase casual homographs (`gen.`) still end sentences.
    let t = "This is next gen. It works.";
    assert_eq!(
        sentence_texts(t, &analyze(t)),
        ["This is next gen.", "It works."]
    );

    // Continuation-guarded dotted/time/degree/Co abbreviations.
    let t = "We met at 9 a.m. sharp and left at 5 p.m. quietly.";
    assert_eq!(sentence_texts(t, &analyze(t)), [t]);

    let t = "We met at 9 a.m. Then we left.";
    assert_eq!(
        sentence_texts(t, &analyze(t)),
        ["We met at 9 a.m.", "Then we left."]
    );

    let t = "She earned a Ph.D. in physics under a U.S. government grant at Acme Co. last year.";
    assert_eq!(sentence_texts(t, &analyze(t)), [t]);

    let t = "He returned to the U.S. Later he left Acme Co. Then he retired.";
    assert_eq!(
        sentence_texts(t, &analyze(t)),
        [
            "He returned to the U.S.",
            "Later he left Acme Co.",
            "Then he retired.",
        ]
    );
}

#[test]
fn real_world_corpora_fixes() {
    // 1. Japanese STEM/NLP academic fullwidth period `．` (U+FF0E) and decimal guard.
    let ja_stem = "本研究では，新しい手法を提案する．実験の結果は良好であった．";
    let doc = analyze(ja_stem);
    assert_eq!(
        sentence_texts(ja_stem, &doc),
        [
            "本研究では，新しい手法を提案する．",
            "実験の結果は良好であった．",
        ]
    );
    assert!(
        doc.segmentation
            .sentences
            .iter()
            .all(|s| s.lang == Lang::Ja)
    );

    let ja_dec = "円周率は３．１４である．値は3．14である．";
    assert_eq!(
        sentence_texts(ja_dec, &analyze(ja_dec)),
        ["円周率は３．１４である．", "値は3．14である．"]
    );

    let ja_closer = "「実験結果である．」次の文．";
    assert_eq!(
        sentence_texts(ja_closer, &analyze(ja_closer)),
        ["「実験結果である．」", "次の文．"]
    );

    // 2. English academic/reference abbreviations & U.S./U.K./E.U./U.N. + Proper Noun.
    let acad = "In Sec. 2, Ch. 3, Vol. 5, Vols. 1-2, pp. 10-12, Smith eds. and Doe trans. analyze the data.";
    assert_eq!(sentence_texts(acad, &analyze(acad)), [acad]);

    let num_abbrev = "See p. 14, ed. 2, art. 5, pt. 3, para. 4, and paras. 6-8 for details.";
    assert_eq!(
        sentence_texts(num_abbrev, &analyze(num_abbrev)),
        [num_abbrev]
    );

    let p_end = "See p. Next sentence.";
    assert_eq!(
        sentence_texts(p_end, &analyze(p_end)),
        ["See p.", "Next sentence."]
    );

    let us_propn = "across 12 corpora from the U.S. National Library of Medicine.";
    assert_eq!(sentence_texts(us_propn, &analyze(us_propn)), [us_propn]);

    let geo_propn = "The U.K. Parliament and U.N. Security Council met with the E.U. Commission.";
    assert_eq!(sentence_texts(geo_propn, &analyze(geo_propn)), [geo_propn]);

    let us_split_he = "She lives in the U.S. He lives in Canada.";
    assert_eq!(
        sentence_texts(us_split_he, &analyze(us_split_he)),
        ["She lives in the U.S.", "He lives in Canada."]
    );

    let us_split_the = "It happened in the U.S. The results were clear.";
    assert_eq!(
        sentence_texts(us_split_the, &analyze(us_split_the)),
        ["It happened in the U.S.", "The results were clear."]
    );

    let geo_newline = "They met at the U.N.\nNext day they left.";
    assert_eq!(
        sentence_texts(geo_newline, &analyze(geo_newline)),
        ["They met at the U.N.", "Next day they left."]
    );

    // 3. Korean quoted speech + quotative verb/particle.
    let ko_q1 = "“네 소원이 무엇이냐?” 하고 하느님이 물으시면";
    assert_eq!(sentence_texts(ko_q1, &analyze(ko_q1)), [ko_q1]);

    let ko_q2 = "“어디 가니?”라고 물었다.";
    assert_eq!(sentence_texts(ko_q2, &analyze(ko_q2)), [ko_q2]);

    let ko_q3 = "“조심하시오.” 하고 말했다.";
    assert_eq!(sentence_texts(ko_q3, &analyze(ko_q3)), [ko_q3]);

    let ko_q4 = "“정말이야?”고 물었다.";
    assert_eq!(sentence_texts(ko_q4, &analyze(ko_q4)), [ko_q4]);

    let ko_q5 = "「괜찮아?」하며 웃었다.";
    assert_eq!(sentence_texts(ko_q5, &analyze(ko_q5)), [ko_q5]);

    let ko_q_split = "“네 소원이 무엇이냐?” 하느님이 물었다.";
    assert_eq!(
        sentence_texts(ko_q_split, &analyze(ko_q_split)),
        ["“네 소원이 무엇이냐?”", "하느님이 물었다."]
    );

    let ko_q_newline = "“네 소원이 무엇이냐?”\n하고 물었다.";
    assert_eq!(
        sentence_texts(ko_q_newline, &analyze(ko_q_newline)),
        ["“네 소원이 무엇이냐?”", "하고 물었다."]
    );

    // 4. Japanese ellipsis + inline parenthetical tag & Ja slang lang detection.
    for tag in [
        "雨…（笑）",
        "雨… (笑)",
        "大変…（泣）",
        "失敗…（汗）",
        "感動…（涙）",
        "すごい…（爆）",
    ] {
        let doc = analyze(tag);
        assert_eq!(sentence_texts(tag, &doc), [tag]);
        assert_eq!(doc.segmentation.sentences[0].lang, Lang::Ja);
    }

    let long_paren = "雨…（とても長い説明文が続く場合）";
    assert_eq!(
        sentence_texts(long_paren, &analyze(long_paren)),
        ["雨…", "（とても長い説明文が続く場合）"]
    );

    // 5. Parenthetical commas & US date commas in clauses_with_lang.
    let parens = "LLM(GPT-3.5, GPT-4, Gemini, EXAONE, LLaMA) 모델을 비교했다.";
    let doc = analyze(parens);
    assert_eq!(sentence_texts(parens, &doc), [parens]);
    assert_eq!(clause_texts(parens, &doc), [parens]);

    let fw_parens = "主要モデル（GPT-4、Gemini、Claude）を比較する。";
    let doc = analyze(fw_parens);
    assert_eq!(clause_texts(fw_parens, &doc), [fw_parens]);

    let smiley = "He smiled :) and, later, left.";
    assert_eq!(
        clause_texts(smiley, &analyze(smiley)),
        ["He smiled :) and", "later", "left."]
    );

    let unclosed = "She said (unclosed, and then left.";
    assert_eq!(
        clause_texts(unclosed, &analyze(unclosed)),
        ["She said (unclosed", "and then left."]
    );

    let us_date = "On Nov. 13, 2026, the report was published.";
    assert_eq!(
        clause_texts(us_date, &analyze(us_date)),
        ["On Nov. 13, 2026", "the report was published."]
    );

    let us_date_full = "November 13, 2026";
    assert_eq!(
        clause_texts(us_date_full, &analyze(us_date_full)),
        [us_date_full]
    );

    let us_date_may = "May 5, 2026, was sunny.";
    assert_eq!(
        clause_texts(us_date_may, &analyze(us_date_may)),
        ["May 5, 2026", "was sunny."]
    );
}
