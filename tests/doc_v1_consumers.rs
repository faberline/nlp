//! `NarrativeAnalyzer::analyze` and the layer 2 pronoun read use `nlp.doc.v1` for zh-TW, on
//! the lines that pass `doc_applies`. `timeline::build` does not read the document; it is built
//! here from analyzer events only to check that the doc path feeds it.
//! Fixtures are short, neutral and self-written: novel, paper and casual style, each in zh-TW,
//! English and mixed text, plus text in a language that layer 1 does not support.
use faberline_nlp::{
    Lexicon, NarrativeAnalyzer, analyze,
    doc::{Lang, Options, Reason, analyze_document},
    text::Locale,
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
