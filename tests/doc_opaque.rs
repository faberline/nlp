use faberline_nlp::Span;
use faberline_nlp::doc::{Document, Lang, OpaqueKind, Options, analyze_document};
use serde_json::json;
use std::io::Write;
use std::process::{Command, Stdio};

fn slice(text: &str, s: Span) -> String {
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

fn opaque_pairs(text: &str, doc: &Document) -> Vec<(OpaqueKind, String)> {
    doc.segmentation
        .opaque
        .iter()
        .map(|o| (o.kind, slice(text, o.span)))
        .collect()
}

fn owned(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

#[test]
fn code_block_and_inline_code() {
    let t = "Intro line.\n\n```rust\nlet x = 1.5;\n```\n\nOutro line.";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [(OpaqueKind::Code, "```rust\nlet x = 1.5;\n```".to_owned())]
    );
    assert_eq!(
        paragraph_texts(t, &doc),
        owned(&["Intro line.", "Outro line."])
    );
    assert_eq!(
        sentence_texts(t, &doc),
        owned(&["Intro line.", "Outro line."])
    );

    let t = "Call `foo.bar()` now. Done.";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [(OpaqueKind::Code, "`foo.bar()`".to_owned())]
    );
    assert_eq!(
        sentence_texts(t, &doc),
        owned(&["Call `foo.bar()` now.", "Done."])
    );

    let t = "a ` b.";
    let doc = analyze(t);
    assert!(doc.segmentation.opaque.is_empty());
    assert_eq!(doc.segmentation.sentences.len(), 1);
}

#[test]
fn formulas() {
    let t = "Energy is:\n\n$$\nE = mc^2.\n$$\n\nDone.";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [(OpaqueKind::Formula, "$$\nE = mc^2.\n$$".to_owned())]
    );
    assert_eq!(paragraph_texts(t, &doc), owned(&["Energy is:", "Done."]));

    let t = "Let $x. y$ be given. Next.";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [(OpaqueKind::Formula, "$x. y$".to_owned())]
    );
    assert_eq!(
        sentence_texts(t, &doc),
        owned(&["Let $x. y$ be given.", "Next."])
    );

    let t = "It costs $5 and $10. Next.";
    let doc = analyze(t);
    assert!(doc.segmentation.opaque.is_empty());
    assert_eq!(doc.segmentation.sentences.len(), 2);
}

#[test]
fn headings() {
    let t = "# Title\n\nText here.\n\n第三章 風雨\n\nChapter IV: The Road\n\nCHAPTER 12\n\nBody.";
    let doc = analyze(t);
    let kinds: Vec<OpaqueKind> = doc.segmentation.opaque.iter().map(|o| o.kind).collect();
    assert_eq!(kinds, [OpaqueKind::Heading; 4]);
    assert_eq!(
        opaque_pairs(t, &doc)
            .into_iter()
            .map(|p| p.1)
            .collect::<Vec<_>>(),
        owned(&[
            "# Title",
            "第三章 風雨",
            "Chapter IV: The Road",
            "CHAPTER 12"
        ])
    );
    assert_eq!(paragraph_texts(t, &doc), owned(&["Text here.", "Body."]));

    for t in [
        "第三章講的是方法。",
        "Chapter 3 shows that it works.",
        "#hashtag fun",
    ] {
        assert!(analyze(t).segmentation.opaque.is_empty(), "{t}");
    }
}

#[test]
fn list_items() {
    let t = "- first item\n- second item\n1. Third item\n2) Fourth item\n\nAfter list.";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [
            (OpaqueKind::ListItem, "- first item".to_owned()),
            (OpaqueKind::ListItem, "- second item".to_owned()),
            (OpaqueKind::ListItem, "1. Third item".to_owned()),
            (OpaqueKind::ListItem, "2) Fourth item".to_owned()),
        ]
    );
    assert_eq!(paragraph_texts(t, &doc), owned(&["After list."]));

    let t = "2020. The year began.";
    let doc = analyze(t);
    assert!(doc.segmentation.opaque.is_empty());
    assert_eq!(
        sentence_texts(t, &doc),
        owned(&["2020.", "The year began."])
    );

    let t = "-5 degrees";
    assert!(analyze(t).segmentation.opaque.is_empty());
}

#[test]
fn citations() {
    let t = "Prior work [1, 2] and (Smith & Lee, 2020a) agree. 如（王小明，2020）所述。";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [
            (OpaqueKind::Citation, "[1, 2]".to_owned()),
            (OpaqueKind::Citation, "(Smith & Lee, 2020a)".to_owned()),
            (OpaqueKind::Citation, "（王小明，2020）".to_owned()),
        ]
    );
    assert_eq!(
        sentence_texts(t, &doc),
        owned(&[
            "Prior work [1, 2] and (Smith & Lee, 2020a) agree.",
            "如（王小明，2020）所述。"
        ])
    );
    assert_eq!(
        clause_texts(t, &doc).last().unwrap(),
        "如（王小明，2020）所述。"
    );
    assert_eq!(clause_texts(t, &doc).len(), 2);

    let t = "(Smith, 2019; Lee, 2020)";
    assert_eq!(
        opaque_pairs(t, &analyze(t)),
        [(OpaqueKind::Citation, t.to_owned())]
    );

    let t = "Call me (maybe) [later]. See (see 2020 results).";
    assert!(analyze(t).segmentation.opaque.is_empty());

    let t = "Smith et al. (2020) show it.";
    assert!(analyze(t).segmentation.opaque.is_empty());
}

#[test]
fn urls() {
    let t = "See https://example.com/a.b?x=1. Then www.test.org/page, and done.";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [
            (OpaqueKind::Url, "https://example.com/a.b?x=1".to_owned()),
            (OpaqueKind::Url, "www.test.org/page".to_owned()),
        ]
    );
    assert_eq!(
        sentence_texts(t, &doc),
        owned(&[
            "See https://example.com/a.b?x=1.",
            "Then www.test.org/page, and done."
        ])
    );
    let second = doc.segmentation.sentences[1].span;
    let clauses: Vec<Span> = doc
        .segmentation
        .clauses
        .iter()
        .map(|c| c.span)
        .filter(|c| c.start_char >= second.start_char && c.end_char <= second.end_char)
        .collect();
    assert_eq!(
        texts(t, &clauses),
        owned(&["Then www.test.org/page", "and done."])
    );

    let t = "詳見 https://example.com/中文 的說明。";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [(OpaqueKind::Url, "https://example.com/".to_owned())]
    );
    assert_eq!(doc.segmentation.sentences.len(), 1);

    let t = "http:// alone.";
    assert!(analyze(t).segmentation.opaque.is_empty());
}

#[test]
fn emoji() {
    let t = "好好吃😋 我也要\n晚安👍🏽";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [
            (OpaqueKind::Emoji, "😋".to_owned()),
            (OpaqueKind::Emoji, "👍🏽".to_owned()),
        ]
    );
    assert_eq!(
        sentence_texts(t, &doc),
        owned(&["好好吃😋", "我也要", "晚安👍🏽"])
    );

    let t = "I ❤️ you";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [(OpaqueKind::Emoji, "❤️".to_owned())]
    );
    assert_eq!(doc.segmentation.sentences.len(), 1);

    let t = "Family 👨‍👩‍👧 done";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [(OpaqueKind::Emoji, "👨‍👩‍👧".to_owned())]
    );
    assert_eq!(doc.segmentation.sentences.len(), 1);
}

#[test]
fn carry_over_matrix() {
    let t = "She opened the door and saw the letter on the table.\n\n\"You came,\" he said.";
    let doc = analyze(t);
    assert_eq!(doc.segmentation.paragraphs.len(), 2);
    assert_eq!(
        sentence_texts(t, &doc),
        owned(&[
            "She opened the door and saw the letter on the table.",
            "\"You came,\" he said."
        ])
    );
    let langs: Vec<Lang> = doc.segmentation.sentences.iter().map(|s| s.lang).collect();
    assert_eq!(langs, [Lang::En, Lang::En]);

    let t = "他看了看手機，Slack 上有一則訊息。\n\n「Hello」她說。";
    let doc = analyze(t);
    assert_eq!(
        sentence_texts(t, &doc),
        owned(&["他看了看手機，Slack 上有一則訊息。", "「Hello」她說。"])
    );
    let langs: Vec<Lang> = doc.segmentation.sentences.iter().map(|s| s.lang).collect();
    assert_eq!(langs, [Lang::Mixed, Lang::Mixed]);

    let t = "本研究提出一種新方法，並在三個資料集上進行驗證。結果顯示，準確率提升了百分之五。";
    let doc = analyze(t);
    assert_eq!(sentence_texts(t, &doc).len(), 2);
    assert_eq!(
        clause_texts(t, &doc),
        owned(&[
            "本研究提出一種新方法",
            "並在三個資料集上進行驗證。",
            "結果顯示",
            "準確率提升了百分之五。"
        ])
    );
    assert!(
        doc.segmentation
            .sentences
            .iter()
            .all(|s| s.lang == Lang::Zh)
    );

    let t = "我們使用 BERT 模型進行實驗，結果優於 baseline。";
    let doc = analyze(t);
    assert_eq!(sentence_texts(t, &doc).len(), 1);
    assert_eq!(
        clause_texts(t, &doc),
        owned(&["我們使用 BERT 模型進行實驗", "結果優於 baseline。"])
    );
    assert_eq!(doc.segmentation.sentences[0].lang, Lang::Mixed);
}

#[test]
fn carry_over_closers_and_list_marker() {
    let t = "Done.） Next";
    assert_eq!(sentence_texts(t, &analyze(t)), owned(&["Done.）", "Next"]));

    let t = "1. Item one\n2. Item two";
    let doc = analyze(t);
    assert_eq!(
        opaque_pairs(t, &doc),
        [
            (OpaqueKind::ListItem, "1. Item one".to_owned()),
            (OpaqueKind::ListItem, "2. Item two".to_owned()),
        ]
    );
    assert!(doc.segmentation.sentences.is_empty());
}

#[test]
fn invariants_and_json() {
    let text = [
        "Intro line.\n\n```rust\nlet x = 1.5;\n```\n\nOutro line.",
        "Call `foo.bar()` now. Done.",
        "a ` b.",
        "Energy is:\n\n$$\nE = mc^2.\n$$\n\nDone.",
        "Let $x. y$ be given. Next.",
        "It costs $5 and $10. Next.",
        "# Title\n\nText here.\n\n第三章 風雨\n\nChapter IV: The Road\n\nCHAPTER 12\n\nBody.",
        "第三章講的是方法。",
        "Chapter 3 shows that it works.",
        "#hashtag fun",
        "- first item\n- second item\n1. Third item\n2) Fourth item\n\nAfter list.",
        "2020. The year began.",
        "-5 degrees",
        "Prior work [1, 2] and (Smith & Lee, 2020a) agree. 如（王小明，2020）所述。",
        "(Smith, 2019; Lee, 2020)",
        "Call me (maybe) [later]. See (see 2020 results).",
        "Smith et al. (2020) show it.",
        "See https://example.com/a.b?x=1. Then www.test.org/page, and done.",
        "詳見 https://example.com/中文 的說明。",
        "http:// alone.",
        "好好吃😋 我也要\n晚安👍🏽",
        "I ❤️ you",
        "Family 👨‍👩‍👧 done",
    ]
    .join("\n\n");

    let doc = analyze(&text);
    assert_eq!(doc, analyze(&text));

    let spans: Vec<Span> = doc.segmentation.opaque.iter().map(|o| o.span).collect();
    for (i, s) in spans.iter().enumerate() {
        assert!(s.start_char < s.end_char, "empty span {s:?}");
        assert!(s.end_char <= doc.text_chars, "span out of text {s:?}");
        if i > 0 {
            assert!(
                spans[i - 1].end_char <= s.start_char,
                "overlap or unsorted at {i}"
            );
        }
    }

    let chars: Vec<char> = text.chars().collect();
    let whole_line = |s: Span| {
        (s.start_char == 0 || chars[s.start_char - 1] == '\n')
            && (s.end_char == chars.len() || chars[s.end_char] == '\n')
    };
    let seg = &doc.segmentation;
    for o in &seg.opaque {
        let body = slice(&text, o.span);
        let block = matches!(o.kind, OpaqueKind::Heading | OpaqueKind::ListItem)
            || (matches!(o.kind, OpaqueKind::Code | OpaqueKind::Formula)
                && (body.contains('\n') || whole_line(o.span)));
        if block {
            let others = seg
                .paragraphs
                .iter()
                .map(|p| p.span)
                .chain(seg.sentences.iter().map(|s| s.span))
                .chain(seg.clauses.iter().map(|c| c.span));
            for other in others {
                assert!(
                    other.end_char <= o.span.start_char || o.span.end_char <= other.start_char,
                    "block {o:?} overlaps {other:?}"
                );
            }
        } else {
            assert!(
                seg.sentences.iter().any(|s| {
                    s.span.start_char <= o.span.start_char && o.span.end_char <= s.span.end_char
                }),
                "inline {o:?} crosses a sentence boundary"
            );
        }
    }

    let v = serde_json::to_value(&doc).unwrap();
    let opaque = v["segmentation"]["opaque"].as_array().unwrap();
    assert!(opaque.iter().any(|o| o["kind"] == "list_item"));
    for o in opaque {
        assert!(o["span"]["start_char"].is_u64());
        assert!(o["span"]["end_char"].is_u64());
    }

    let empty = analyze("");
    assert!(empty.segmentation.opaque.is_empty());
    let v = serde_json::to_value(&empty).unwrap();
    assert_eq!(v["segmentation"]["opaque"], json!([]));
}

#[test]
fn cli_has_opaque() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_nlp_doc"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all("see https://a.example/x now".as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let opaque = v["segmentation"]["opaque"].as_array().unwrap();
    assert_eq!(opaque.len(), 1);
    assert_eq!(opaque[0]["kind"], "url");
}
