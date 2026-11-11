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
