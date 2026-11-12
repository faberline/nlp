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

#[test]
fn fence_on_one_line_is_not_a_code_block() {
    let text = "```x``` is inline.\n\nNext para.";
    let doc = analyze(text);
    assert!(
        doc.segmentation
            .opaque
            .iter()
            .all(|o| o.kind != OpaqueKind::Code)
            || opaque_pairs(text, &doc).iter().all(|(_, t)| t != text)
    );
    assert_eq!(
        sentence_texts(text, &doc),
        owned(&["```x``` is inline.", "Next para."])
    );

    let text = "~~~好開心~~~\n\n明天見。";
    let doc = analyze(text);
    assert!(doc.segmentation.opaque.is_empty());
    assert_eq!(
        paragraph_texts(text, &doc),
        owned(&["~~~好開心~~~", "明天見。"])
    );
    assert_eq!(sentence_texts(text, &doc).last().unwrap(), "明天見。");

    // A real fence still is one.
    let text = "```\nlet x = 1;\n```\n\nNext para.";
    let doc = analyze(text);
    assert_eq!(opaque_pairs(text, &doc)[0].0, OpaqueKind::Code);
    assert_eq!(sentence_texts(text, &doc), owned(&["Next para."]));
}

#[test]
fn display_formula_does_not_cross_a_blank_line() {
    let text = "$$ x\n\nlong text here.\n\nend $$\n\nAfter.";
    let doc = analyze(text);
    assert!(
        doc.segmentation
            .opaque
            .iter()
            .all(|o| o.kind != OpaqueKind::Formula)
    );
    assert!(sentence_texts(text, &doc).contains(&"long text here.".to_string()));
    assert!(sentence_texts(text, &doc).contains(&"After.".to_string()));
}

#[test]
fn sentence_ending_emoji_does_not_split_paragraph() {
    let text = "晚安👍🏽\n明天見";
    let doc = analyze(text);
    assert_eq!(paragraph_texts(text, &doc), owned(&["晚安👍🏽\n明天見"]));
    assert_eq!(sentence_texts(text, &doc), owned(&["晚安👍🏽", "明天見"]));
}

#[test]
fn web_novel_headings_expanded() {
    let text = [
        "VIP第八卷 遠山紀行 第一千二百章 青石古道",
        "第一段正文。",
        "vip第十卷 風起雲湧",
        "正文第三卷山河志第一百零八章溪橋夜雨",
        "第四卷雲海篇第二百五十六章晨鐘",
        "第一千六百九十六 青竹石室",
        "第五十八 渡口微光（下）",
        "正文第十八卷遠古遺跡探索錄第一千九百九十九章雲霧深處的古老石橋與松林秘境",
        "第二段正文。",
    ]
    .join("\n\n");

    let doc = analyze(&text);
    let headings: Vec<String> = doc
        .segmentation
        .opaque
        .iter()
        .filter(|o| o.kind == OpaqueKind::Heading)
        .map(|o| slice(&text, o.span))
        .collect();
    assert_eq!(
        headings,
        owned(&[
            "VIP第八卷 遠山紀行 第一千二百章 青石古道",
            "vip第十卷 風起雲湧",
            "正文第三卷山河志第一百零八章溪橋夜雨",
            "第四卷雲海篇第二百五十六章晨鐘",
            "第一千六百九十六 青竹石室",
            "第五十八 渡口微光（下）",
            "正文第十八卷遠古遺跡探索錄第一千九百九十九章雲霧深處的古老石橋與松林秘境",
        ])
    );
    assert_eq!(
        sentence_texts(&text, &doc),
        owned(&["第一段正文。", "第二段正文。"])
    );

    // Prose lines and >60-char lines must not match as headings.
    let over_60 = format!("第一千二百章 {}", "遠山雲霧古道石橋松林溪谷".repeat(5));
    assert!(over_60.chars().count() > 60);
    for non_heading in [
        over_60.as_str(),
        "第一 名次已經公佈",
        "第一千二百章講述了資料整理方法。",
        "第三章的內容",
        "正文第一段。",
        "第一千六百九十六 這裡有逗號，所以不是標題",
    ] {
        assert!(
            analyze(non_heading).segmentation.opaque.is_empty(),
            "unexpected heading match for: {non_heading}"
        );
    }
}

#[test]
fn raw_html_tags_entities_pua_and_box_banners() {
    let pua_long = format!("{}9k", "\u{E010}".repeat(12));
    let pua_short = "\u{E001}\u{E002}\u{E003}";
    let box_heavy = "┏━━━━━━━━━━━━┓\n┃ 網站公告欄 ┃\n┗━━━━━━━━━━━━┛";
    let box_light = "┌────────────┐\n│ 閱讀提示欄 │\n└────────────┘";

    let text = format!(
        "{box_heavy}\n\n<P\n\n<br/>\n\n<P 旅人走進木屋。<br>桌上放著&qut;舊書&quot;&nbsp;&amp;&lt;&gt;。\n\n{pua_long}清晨出發，{pua_short}ab繼續前行。\n\n{box_light}\n\n最後一句。"
    );
    let doc = analyze(&text);
    assert_eq!(
        opaque_pairs(&text, &doc),
        vec![
            (OpaqueKind::Code, box_heavy.to_owned()),
            (OpaqueKind::Code, "<P".to_owned()),
            (OpaqueKind::Code, "<br/>".to_owned()),
            (OpaqueKind::Code, "<P".to_owned()),
            (OpaqueKind::Code, "<br>".to_owned()),
            (OpaqueKind::Code, "&qut;".to_owned()),
            (OpaqueKind::Code, "&quot;".to_owned()),
            (OpaqueKind::Code, "&nbsp;".to_owned()),
            (OpaqueKind::Code, "&amp;".to_owned()),
            (OpaqueKind::Code, "&lt;".to_owned()),
            (OpaqueKind::Code, "&gt;".to_owned()),
            (OpaqueKind::Code, pua_long.clone()),
            (OpaqueKind::Code, pua_short.to_owned()),
            (OpaqueKind::Code, box_light.to_owned()),
        ]
    );

    let sents = sentence_texts(&text, &doc);
    assert!(sents.iter().any(|s| s.contains("旅人走進木屋。")));
    assert!(sents.iter().any(|s| s.contains("最後一句。")));
    assert!(!sents.iter().any(|s| s == "<P" || s == "<br/>"));

    // `<path>` and attached `a<p` must not be masked as `<P` tags; 3+ letter ASCII words after
    // PUA runs must not be swallowed.
    let safe = format!("Check <path> and a<p now. {}hello.", "\u{E020}".repeat(10));
    let safe_doc = analyze(&safe);
    assert_eq!(
        opaque_pairs(&safe, &safe_doc),
        vec![(OpaqueKind::Code, "\u{E020}".repeat(10))]
    );
}

#[test]
fn opt_in_editorial_masking_preserves_char_offsets() {
    let text = [
        "旅人推開石門，走入幽靜的庭院。",
        "（月底最後一天，求月票、推薦票鼎立支持！）",
        "今天保底第一更送到，晚上還有加更！",
        "(奇幻)(冒險)(古典)(連載)",
        "老者收起木匣/首/發，緩步走下石階(ab#cd#ef)。（未完待續）",
        "本書收集于網絡，屬轉載作品，請別忘本書，點擊加入書架查看熱門書籍與書友正在讀。",
    ]
    .join("\n\n");

    let default_doc = analyze_document(&text, &Options::default());
    assert!(default_doc.segmentation.opaque.is_empty());

    let masked_doc = analyze_document(
        &text,
        &Options {
            mask_editorial: true,
            ..Options::default()
        },
    );
    assert_eq!(masked_doc.text_chars, default_doc.text_chars);
    assert_eq!(masked_doc.text_sha256, default_doc.text_sha256);

    assert_eq!(
        opaque_pairs(&text, &masked_doc),
        vec![
            (
                OpaqueKind::Code,
                "（月底最後一天，求月票、推薦票鼎立支持！）".to_owned()
            ),
            (
                OpaqueKind::Code,
                "今天保底第一更送到，晚上還有加更！".to_owned()
            ),
            (OpaqueKind::Code, "(奇幻)(冒險)(古典)(連載)".to_owned()),
            (OpaqueKind::Code, "/首/發".to_owned()),
            (OpaqueKind::Code, "(ab#cd#ef)".to_owned()),
            (OpaqueKind::Code, "（未完待續）".to_owned()),
            (
                OpaqueKind::Code,
                "本書收集于網絡，屬轉載作品，請別忘本書，點擊加入書架查看熱門書籍與書友正在讀。"
                    .to_owned()
            ),
        ]
    );

    assert_eq!(
        sentence_texts(&text, &masked_doc),
        owned(&[
            "旅人推開石門，走入幽靜的庭院。",
            "老者收起木匣/首/發，緩步走下石階(ab#cd#ef)。",
        ])
    );
    let masked_toks: Vec<String> = masked_doc
        .tokens
        .iter()
        .map(|t| slice(&text, t.span))
        .collect();
    for suppressed in ["/首/發", "(ab#cd#ef)", "（未完待續）"] {
        assert!(
            !masked_toks.iter().any(|t| t == suppressed),
            "silent noise span {suppressed:?} must not emit tokens: {masked_toks:?}"
        );
    }
    // Verify exact char offsets and block-vs-inline invariants on the masked document.
    let chars: Vec<char> = text.chars().collect();
    let whole_line_or_trailing_note = |s: Span| {
        let whole = (s.start_char == 0 || chars[s.start_char - 1] == '\n')
            && (s.end_char == chars.len() || chars[s.end_char] == '\n');
        let trailing_after_sent = s.start_char > 0
            && chars[s.start_char - 1] == '。'
            && (s.end_char == chars.len() || chars[s.end_char] == '\n');
        whole || trailing_after_sent
    };
    for o in &masked_doc.segmentation.opaque {
        if whole_line_or_trailing_note(o.span) {
            for s in &masked_doc.segmentation.sentences {
                assert!(
                    s.span.end_char <= o.span.start_char || o.span.end_char <= s.span.start_char
                );
            }
        } else {
            assert!(masked_doc.segmentation.sentences.iter().any(|s| {
                s.span.start_char <= o.span.start_char && o.span.end_char <= s.span.end_char
            }));
        }
    }
    for tok in &masked_doc.tokens {
        assert!(tok.span.start_char < tok.span.end_char);
        assert!(tok.span.end_char <= masked_doc.text_chars);
        let s = slice(&text, tok.span);
        assert!(!s.is_empty());
    }
}

#[test]
fn ws1_v3_unclosed_html_sgml_boilerplate_tight_headings_and_silent_noise() {
    let opts = Options {
        mask_editorial: true,
        ..Options::default()
    };

    // 1. Unclosed closing tags `</P`, `</p`, SGML `<!go`/`<!over`, truncated `<quot;0" cellpadding=...>`,
    //    and malformed `&qut;` / `&qu0t;` / `&nbps;`.
    let html_noise = concat!(
        "<!go\n\n",
        "<quot;0\" cellpadding=\"0\" cellspacing=\"0\">\n\n",
        "<P青衫老者推門而入，&qut;諸位請坐&qu0t;&nbps;。</P\n\n",
        "眾人點頭致意。</p\n\n",
        "<!over\n"
    );
    let doc_html = analyze_document(html_noise, &opts);
    assert_eq!(
        sentence_texts(html_noise, &doc_html),
        owned(&[
            "青衫老者推門而入，&qut;諸位請坐&qu0t;&nbps;。",
            "眾人點頭致意。"
        ])
    );
    let html_toks: Vec<String> = doc_html
        .tokens
        .iter()
        .map(|t| slice(html_noise, t.span))
        .collect();
    for noise in [
        "<P", "</P", "</p", "<!go", "<!over", "&qut;", "&qu0t;", "&nbps;",
    ] {
        assert!(
            !html_toks.iter().any(|t| t.contains(noise)),
            "noise {noise:?} leaked into tokens: {html_toks:?}"
        );
    }

    // 2. Mixed brackets, nested parens, night-watch `三更` preservation, and unbracketed postscripts.
    let postscripts = [
        "王天明收起木劍，在(三更半夜)悄然離去。(今天碼字太慢，明天兩更補上）",
        "沈星河走入青雲門石殿。（今日請假一天，明日三更送到)",
        "白衣老者合上古籍。(，如果您喜歡這部作品，歡迎您來書站（example）投推薦票、雙倍月票，請為作者加油！)",
        "山風吹過松林。如果您喜歡這部作品，請投月票與求票支持。",
        "溪水潺潺流淌。另外雙倍月票期間求推薦票！",
        "石門緩緩落下。今天只能一更，明日再補。",
        "爐火漸漸熄滅。不得不向大家請假一日。",
    ]
    .join("\n\n");
    let doc_post = analyze_document(&postscripts, &opts);
    assert_eq!(
        sentence_texts(&postscripts, &doc_post),
        owned(&[
            "王天明收起木劍，在(三更半夜)悄然離去。",
            "沈星河走入青雲門石殿。",
            "白衣老者合上古籍。",
            "山風吹過松林。",
            "溪水潺潺流淌。",
            "石門緩緩落下。",
            "爐火漸漸熄滅。",
        ])
    );

    // 3. Pirate/scraper block boilerplate lines.
    let boiler_lines = [
        "一一小說 www.example.org 最新章節",
        "全本小說吧 724小時不間斷更新",
        "在線書吧 看書窩 56小說 書家文學 回味書庫 理想文學 云軒閣 七星閣",
        "ddyueshu SOSO999 搜/搜/9/9/9 3qzw qiXingge qbxs8",
        "最新全本：青石古道紀行",
        "閱讀提示：按左右鍵快速翻到上下章節",
        "瀏覽貼子 128 轉貼次數 4",
        "手機,電腦.還可以下載電子書",
        "熱門小說標簽 熱門作品推薦 其他書友吐血推薦 本周:古道行",
        "真正的小說正文只有這一句。",
    ]
    .join("\n\n");
    let doc_boiler = analyze_document(&boiler_lines, &opts);
    assert_eq!(
        sentence_texts(&boiler_lines, &doc_boiler),
        owned(&["真正的小說正文只有這一句。"])
    );

    // 4. Tight standalone headings and `»»»` / `·` separators, keeping `第三章的內容` as prose.
    let tight_headings = [
        "第一百二十章青石古道",
        "第35章古道夜行",
        "»»» 第八卷 第一百二十章 山雨欲來",
        "·第一百二十章·松林夜話",
        "正文句子在此。",
    ]
    .join("\n\n");
    let doc_head = analyze(&tight_headings);
    let heads: Vec<String> = doc_head
        .segmentation
        .opaque
        .iter()
        .filter(|o| o.kind == OpaqueKind::Heading)
        .map(|o| slice(&tight_headings, o.span))
        .collect();
    assert_eq!(
        heads,
        owned(&[
            "第一百二十章青石古道",
            "第35章古道夜行",
            "»»» 第八卷 第一百二十章 山雨欲來",
            "·第一百二十章·松林夜話",
        ])
    );
    assert!(analyze("第三章的內容").segmentation.opaque.is_empty());

    // 5. Standard inline opaque kinds (`code_span`, `url`, `citation`, `inline_formula`, `emoji`)
    //    must NEVER be marked `silent_noise` even when `mask_editorial: true`.
    let keep_inline = "請看 `let x = 1` 與 https://example.com [1, 2] 及 $x+1$ 好嗎😋";
    let doc_keep = analyze_document(keep_inline, &opts);
    let keep_toks: Vec<String> = doc_keep
        .tokens
        .iter()
        .map(|t| slice(keep_inline, t.span))
        .collect();
    for must_keep in [
        "`let x = 1`",
        "https://example.com",
        "[1, 2]",
        "$x+1$",
        "😋",
    ] {
        assert!(
            keep_toks.iter().any(|t| t == must_keep),
            "non-noise inline span {must_keep:?} must remain a token: {keep_toks:?}"
        );
    }
}

#[test]
fn inline_pirate_watermarks_preserve_story_paragraphs_and_standalone_footers_stay_block() {
    let opts = Options {
        mask_editorial: true,
        normalize_traditional: true,
        ..Options::default()
    };

    // 1. 120+-char narrative & dialogue paragraphs with inline pirate watermarks and mixed-bracket
    //    author notes must mask ONLY the watermark substring (`silent_noise`, 0 bogus tokens) and
    //    preserve 100% of surrounding story sentences/tokens.
    let p_3qzw = concat!(
        "“這門古老的鍛造技藝流傳至今已經十分罕見，尋常鐵匠鋪根本無人知曉其中的火候奧秘(╰→3qzw）。",
        "倘若我們能夠在入冬之前集齊三塊玄鐵礦石並修復石爐，便可以為全村守衛打造出堅固的護具。",
        "至於剩下的木炭與清水，老夫明日清晨自會派兩名學徒送往山谷工坊。”白鬚匠人神色平靜地向眾人說道。"
    );
    assert!(p_3qzw.chars().count() >= 120);

    let p_empty_parens = concat!(
        "()()(724小時不間斷更新純TXT小說)“二位攔住在下與同伴的去路，莫非是想借閱這卷山川地理圖譜？",
        "我們只是路過此地的商隊護衛，對山嶺深處的古墓傳聞毫無興趣，還請諸位行個方便讓開道路。”青衣護衛手按刀柄眉頭微皺，毫不猶豫地沉聲開口婉拒道。"
    );
    assert!(p_empty_parens.chars().count() >= 120);

    let p_kaomoji = concat!(
        "“區區風雪算得了什麼？沈兄看來還不清楚老夫早年在北境行商的經歷吧。",
        "☆⌒_⌒☆七星閣☆⌒_⌒☆ωωω。qiXingge在下曾在冰原驛站駐守十餘載，對沿途每一處避風岩洞都瞭若指掌。",
        "只要隊伍緊跟著前方的馬車隊列，天黑之前定能平安抵達山腳下的石橋客棧。”灰袍老者笑著拍了拍胸口答道。"
    );
    assert!(p_kaomoji.chars().count() >= 120);

    let p_shujia = concat!(
        "昨夜被狂風吹倒的幾株參天古松橫在溪谷窄道中央，雖然折斷的粗大樹枝暫時堵住了淺灘渡口。書家文學",
        "但是熟悉山路的年輕樵夫還是帶著他（她）們小心翼翼地繞過濕滑的青石斜坡，在濃厚晨霧散去之前順利抵達了半山腰處的木屋營地，並生起溫暖明亮的松木篝火烘乾衣袍。"
    );
    assert!(p_shujia.chars().count() >= 120);

    let p_mixed_notes_and_sites = [
        "(本想晚上把兩章合為一章給大家碼出來的，可沒想到這兩日精神欠佳，等調整好再給大家想辦法補上吧！）旅人沿著蜿蜒的青石古道緩步前行，遠處鐘聲悠揚。",
        "山門外的守衛收起長槍，轉身走入石亭避雨。（手機,電腦.還可以下載電子書TXT,CHM,UMD,JAR電子書）",
        "石橋對面的茶棚燈火通明。(最新最全電子書下載SOSO999)幾名挑夫放下竹簍歇腳。(全部小說超速更新：搜/搜/9/9/9)",
        "老掌櫃笑著端上熱茶。就到一一小說門外風雨漸歇。請：m2.ddyueshu",
        "遠處高塔亮起微光。www.0_3_wx.理想文學書友整理提供",
        "石階盡頭傳來腳步聲。w.０3_x.om理想文學書友整理提供院內燈籠隨風輕晃。理想文學書友整理提供",
        "“這是何處？”青年打量四周石壁。，如欲知后事如何，請登陸.example，章節更多，支持作者，支持正版閱讀！)",
        "兩名劍客在石橋上激戰數十回合，各自後退半步互不示弱！，如欲知后事如何，請登陸.example，章節更多，支持作者，支持正版閱讀！)",
        "爐中器物在黎明時分鍛造完成。［要看最新文字版,全本小說吧］",
    ]
    .join("\n\n");

    for (para, expected_opaque, must_have_words) in [
        (
            p_3qzw,
            vec!["(╰→3qzw）"],
            vec!["鍛造", "奧秘", "鐵礦石", "護具", "工坊", "白鬚", "說道"],
        ),
        (
            p_empty_parens,
            vec!["()()(724小時不間斷更新純TXT小說)"],
            vec!["同伴", "圖譜", "商隊", "護衛", "古墓", "婉拒"],
        ),
        (
            p_kaomoji,
            vec!["☆⌒_⌒☆七星閣☆⌒_⌒☆ωωω。qiXingge"],
            vec!["風雪", "在下", "驛站", "岩洞", "馬車", "客棧"],
        ),
        (
            p_shujia,
            vec!["書家文學"],
            vec!["古松", "渡口", "樵夫", "她", "木屋", "篝火"],
        ),
    ] {
        let doc = analyze_document(para, &opts);
        let op_texts: Vec<String> = doc
            .segmentation
            .opaque
            .iter()
            .map(|o| slice(para, o.span))
            .collect();
        assert_eq!(
            op_texts,
            owned(&expected_opaque),
            "opaque mismatch on {para}"
        );
        assert!(
            doc.segmentation.sentences.len() >= 2,
            "story sentences must not be swallowed: {:?}",
            sentence_texts(para, &doc)
        );
        let tok_texts: Vec<String> = doc.tokens.iter().map(|t| slice(para, t.span)).collect();
        for wm in &expected_opaque {
            assert!(
                !tok_texts.iter().any(|t| t.contains("3qzw")
                    || t.contains("724")
                    || t.contains("qiXingge")
                    || t.contains("七星閣")
                    || t.contains("書家文學")
                    || t == wm),
                "watermark leaked into tokens: {tok_texts:?}"
            );
        }
        for w in must_have_words {
            assert!(
                tok_texts.iter().any(|t| t.contains(w)),
                "expected story word {w:?} preserved in tokens: {tok_texts:?}"
            );
        }
    }

    let doc_mixed = analyze_document(&p_mixed_notes_and_sites, &opts);
    let mixed_sents = sentence_texts(&p_mixed_notes_and_sites, &doc_mixed);
    assert_eq!(
        mixed_sents,
        owned(&[
            "旅人沿著蜿蜒的青石古道緩步前行，遠處鐘聲悠揚。",
            "山門外的守衛收起長槍，轉身走入石亭避雨。",
            "石橋對面的茶棚燈火通明。",
            "幾名挑夫放下竹簍歇腳。",
            "老掌櫃笑著端上熱茶。",
            "門外風雨漸歇。",
            "遠處高塔亮起微光。",
            "石階盡頭傳來腳步聲。",
            "院內燈籠隨風輕晃。",
            "“這是何處？”",
            "青年打量四周石壁。",
            "兩名劍客在石橋上激戰數十回合，各自後退半步互不示弱！",
            "爐中器物在黎明時分鍛造完成。",
        ])
    );
    let mixed_tokens: Vec<String> = doc_mixed
        .tokens
        .iter()
        .map(|t| slice(&p_mixed_notes_and_sites, t.span))
        .collect();
    assert!(
        !mixed_tokens.iter().any(|t| t.contains("理想")
            || t.contains("書友")
            || t.contains("整理")
            || t.contains("ddyueshu")
            || t.contains("SOSO999")),
        "inline watermark leaked into mixed tokens: {mixed_tokens:?}"
    );

    // 2. Standalone pirate footer & mixed-bracket author-note lines still mask as whole-line blocks.
    let standalone_footers = [
        "(大家投票票了！有票票才有動力，沒投的快投一票吧！）",
        "(咱昨天有事少發了一章，心裡記得清楚，會給大家補上的！）",
        "(經書友提醒，發現章節數與章節名字弄錯了，大家見諒一下哦！）",
        "請：m2.ddyueshu",
        "書家文學",
        "一一小說 www.example.org 最新章節",
        "全本小說吧（qbxs8）全部小說為全本小說，不接受連載小說上傳！",
        "（第一更！第二更放到晚上更新了哦！）（，()書友整理上傳更新及。",
        "真正保留下來的故事正文句子。",
    ]
    .join("\n\n");
    let doc_footers = analyze_document(&standalone_footers, &opts);
    assert_eq!(
        sentence_texts(&standalone_footers, &doc_footers),
        owned(&["真正保留下來的故事正文句子。"])
    );
    assert_eq!(doc_footers.segmentation.opaque.len(), 8);
    assert!(
        doc_footers
            .segmentation
            .opaque
            .iter()
            .all(|o| o.kind == OpaqueKind::Code)
    );
}
