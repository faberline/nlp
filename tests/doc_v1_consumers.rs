//! `NarrativeAnalyzer::analyze` and the layer 2 pronoun read use `nlp.doc.v1` for zh-TW, on
//! the lines that pass `doc_applies`. `timeline::build` does not read the document; it is built
//! here from analyzer events only to check that the doc path feeds it.
//! Fixtures are short, neutral and self-written: novel, paper and casual style, each in zh-TW,
//! English and mixed text, plus text in a language that layer 1 does not support.
use faberline_nlp::{
    Lexicon, NarrativeAnalyzer, analyze,
    doc::{Lang, Options, Reason, analyze_document},
    source_cross_chapter::{CrossChapterCandidates, CrossChapterPolicy},
    text::{Analyzer, Locale},
    timeline::{self, TimelineDocument},
};

const ACTORS: &[(&str, &str)] = &[("小甲", "a"), ("阿乙", "b"), ("Alice Lee", "c")];

struct Case {
    name: &'static str,
    text: &'static str,
    lang: Lang,
}

const CASES: &[Case] = &[
    Case {
        name: "novel zh-TW",
        text: "小甲走進木屋。阿乙留在港口。\n小甲說：「我們明天出發。」阿乙點頭。\n",
        lang: Lang::Zh,
    },
    Case {
        name: "novel English",
        text: "Alice Lee walked into the hut. Then she sat down.\nShe said, \"We leave at dawn.\"\n",
        lang: Lang::En,
    },
    Case {
        name: "novel mixed",
        text: "Alice Lee 把信交給小甲。小甲 read it twice, 然後走了。\n",
        lang: Lang::Mixed,
    },
    Case {
        name: "paper zh-TW",
        text: "本文提出一種簡單的方法。我們量測了兩次資料。結果如表 2 所示。\n",
        lang: Lang::Zh,
    },
    Case {
        name: "paper English",
        text: "The method is simple. We measure the data twice. Results follow in Table 2.\n",
        lang: Lang::En,
    },
    Case {
        name: "paper mixed",
        text: "我們使用 transformer 模型。The error is 0.5 percent, 結果如圖 3 所示。\n",
        lang: Lang::Mixed,
    },
    Case {
        name: "casual zh-TW",
        text: "哈哈 今天好累 小甲先走了 我也要回家了\n",
        lang: Lang::Zh,
    },
    Case {
        name: "casual English",
        text: "lol so tired today. alice left early. going home now\n",
        lang: Lang::En,
    },
    Case {
        name: "casual mixed",
        text: "今天 so tired 啦 小甲 already 回家了\n",
        lang: Lang::Mixed,
    },
];

fn chars(text: &str) -> usize {
    text.chars().count()
}

#[test]
fn doc_v1_reads_every_style_and_language() {
    for case in CASES {
        let doc = analyze_document(case.text, &Options::default());
        assert!(!doc.segmentation.sentences.is_empty(), "{}", case.name);
        assert!(
            doc.segmentation
                .sentences
                .iter()
                .any(|s| s.lang == case.lang),
            "{}: no {:?} sentence",
            case.name,
            case.lang
        );
    }
}

#[test]
fn analyze_reads_every_style_without_leaving_the_text() {
    let lexicon = Lexicon {
        actors: ACTORS,
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon).expect("analyzer");
    for case in CASES {
        let result = analyzer.analyze(case.text, "");
        let n = chars(case.text);
        for section in &result.sections {
            assert!(section.evidence.end_char <= n, "{}", case.name);
            for unit in &section.units {
                assert!(unit.evidence.start_char <= unit.evidence.end_char);
                assert!(unit.evidence.end_char <= n, "{}", case.name);
            }
        }
        for event in &result.events {
            for span in &event.evidence {
                assert!(span.end_char <= n, "{}", case.name);
            }
        }
    }
}

#[test]
fn zh_tw_novel_finds_the_actors_and_the_dialogue() {
    let lexicon = Lexicon {
        actors: ACTORS,
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon).expect("analyzer");
    let result = analyzer.analyze(CASES[0].text, "");
    let kinds: Vec<_> = result
        .sections
        .iter()
        .flat_map(|s| &s.units)
        .map(|u| (u.kind.as_str(), u.speaker.as_deref()))
        .collect();
    assert!(kinds.contains(&("dialogue", Some("a"))), "{kinds:?}");
}

#[test]
fn timeline_builds_from_every_style() {
    let lexicon = Lexicon {
        actors: ACTORS,
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon).expect("analyzer");
    let analyses: Vec<_> = CASES
        .iter()
        .map(|case| analyzer.analyze(case.text, ""))
        .collect();
    let documents: Vec<_> = CASES
        .iter()
        .zip(&analyses)
        .enumerate()
        .map(|(i, (case, analysis))| TimelineDocument {
            order: u32::try_from(i + 1).expect("order"),
            text: case.text,
            events: &analysis.events,
        })
        .collect();
    let built = timeline::build(&documents).expect("timeline");
    assert_eq!(built.story_positions.len(), built.events.len());
}

#[test]
fn text_in_an_unsupported_language_is_unknown_and_yields_nothing() {
    let text = "안녕하세요 오늘은 날씨가 좋습니다. 감사합니다.\n";
    let doc = analyze_document(text, &Options::default());
    assert!(
        doc.unknown
            .iter()
            .any(|u| u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage),
        "{:?}",
        doc.unknown
    );
    assert!(doc.tokens.is_empty());
    let lexicon = Lexicon {
        actors: ACTORS,
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon).expect("analyzer");
    let result = analyzer.analyze(text, "");
    assert!(result.events.is_empty());
    let plain = analyze(text, "", &lexicon);
    assert!(plain.events.is_empty());
}

#[test]
fn cross_chapter_push_gives_the_generic_tokenizer_pronouns() {
    const PRONOUN_CHAPTERS: &[&str] = &[
        "他走進木屋。她留在港口。\n",
        "他們說：「我們明天出發。」它們不動。\n",
        "他说她们走了，它也走了。\n",
        "哈哈 他先走了 我也要回家了\n",
        "Alice met him at the 港口. 她笑了。\n",
        "他走了很久……終於到了。\n",
        "他讀了 `a。b` 這段，她也讀了。\n",
        "他看了 https://example.org/a 然後她走了。\n",
        "一行沒有標點的文字\n",
        "\n",
        "- 他走進木屋。\n",
        "# 他走了\n",
        "第一章 他們\n",
        "1. 她走了。\n",
        "把 `他` 換成 `她`。\n",
        "$他$ 她\n",
        "他走了。她留下。\n",
        "他說：「我們走。」她點頭。\n",
        "　　他走進木屋。她留在港口。\n　　他們說：「她們也來了。」\n",
    ];
    const PRONOUN_FORMS: &[&str] = &[
        "他", "她", "它", "他們", "她們", "它們", "他们", "她们", "它们",
    ];

    let lexicon = Lexicon {
        actors: ACTORS,
        ..Lexicon::default()
    };
    let policy = CrossChapterPolicy {
        max_chapter_bytes: 1 << 20,
        max_terms: 8,
        max_term_bytes: 64,
        history_distance: 4,
        max_names_per_chapter: 8,
        max_pronouns_per_chapter: 64,
        max_pairs_per_pronoun: 8,
        max_pairs_per_chapter: 64,
    };
    let generic = Analyzer::new(Locale::ChineseTaiwan).expect("analyzer");
    let chapter1 = "小甲走進木屋。阿乙留在港口。Alice Lee 留下。\n";

    for &chapter2 in PRONOUN_CHAPTERS {
        let mut tw =
            CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, policy).expect("tw");
        let mut cn =
            CrossChapterCandidates::new(Locale::ChineseChina, lexicon, policy).expect("cn");
        tw.push(1, chapter1, "").expect("tw ch1");
        cn.push(1, chapter1, "").expect("cn ch1");
        let got = tw.push(2, chapter2, "").expect("tw ch2");
        let ref_out = cn.push(2, chapter2, "").expect("cn ch2");

        let mut got_json = serde_json::to_value(&got).expect("json");
        let ref_json = serde_json::to_value(&ref_out).expect("json");
        assert_eq!(got_json["locale"], "zh-Hant-TW");
        got_json["locale"] = serde_json::json!("zh-Hans-CN");
        assert_eq!(got_json, ref_json, "push output differs for {chapter2:?}");

        let mut line_base = 0usize;
        let mut want_bytes = Vec::new();
        for line in chapter2.split_inclusive('\n') {
            for tok in generic.analyze(line).into_iter().flat_map(|s| s.tokens) {
                if PRONOUN_FORMS.contains(&tok.text) {
                    want_bytes.push((line_base + tok.bytes.start)..(line_base + tok.bytes.end));
                }
            }
            line_base += line.len();
        }
        let got_bytes: Vec<_> = got
            .pronouns
            .iter()
            .map(|p| p.source.bytes.clone())
            .collect();
        assert_eq!(
            got_bytes, want_bytes,
            "pronoun bytes differ for {chapter2:?}"
        );
    }
}
