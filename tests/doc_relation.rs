use faberline_nlp::Span;
use faberline_nlp::doc::{
    Document, Options, Reason, Relation, RelationKind as K, RelationLevel as L, analyze_document,
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const GOLD_PATH: &str = "tests/data/relation_gold.json";
const RECEIPT_PATH: &str = "docs/receipts/relations-2026-11-10.json";

/// (kind, level, connector, correlate, arg1, arg2), each text cut from the input.
type Rel = (K, L, String, Option<String>, String, String);

fn slice(text: &str, s: Span) -> String {
    text.chars()
        .skip(s.start_char)
        .take(s.end_char - s.start_char)
        .collect()
}

fn analyze(text: &str) -> Document {
    analyze_document(text, &Options::default())
}

fn rel(kind: K, level: L, conn: &str, cor: Option<&str>, a1: &str, a2: &str) -> Rel {
    (
        kind,
        level,
        conn.to_owned(),
        cor.map(str::to_owned),
        a1.to_owned(),
        a2.to_owned(),
    )
}

fn cut(text: &str, r: &Relation) -> Rel {
    (
        r.kind,
        r.level,
        slice(text, r.connector),
        r.correlate.map(|s| slice(text, s)),
        slice(text, r.arg1),
        slice(text, r.arg2),
    )
}

fn rels(text: &str) -> Vec<Rel> {
    let doc = analyze(text);
    doc.relations.iter().map(|r| cut(text, r)).collect()
}

/// Connector texts of the `relations` unknowns, each with reason `ambiguous`.
fn rel_unknown(text: &str) -> Vec<String> {
    analyze(text)
        .unknown
        .iter()
        .filter(|u| u.layer == "relations")
        .map(|u| {
            assert_eq!(u.reason, Reason::Ambiguous);
            slice(text, u.span)
        })
        .collect()
}

#[test]
fn zh_sequence_and_cause_pair() {
    assert_eq!(
        rels("他關了燈，然後睡了。"),
        [rel(
            K::Sequence,
            L::Clause,
            "然後",
            None,
            "他關了燈",
            "然後睡了。"
        )]
    );
    assert_eq!(
        rels("因為下雨了，所以比賽取消了。"),
        [rel(
            K::Cause,
            L::Clause,
            "因為",
            Some("所以"),
            "因為下雨了",
            "所以比賽取消了。"
        )]
    );
    assert_eq!(
        rels("我們等了很久，但是他沒有來。"),
        [rel(
            K::Contrast,
            L::Clause,
            "但是",
            None,
            "我們等了很久",
            "但是他沒有來。"
        )]
    );
    assert_eq!(
        rels("如果明天下雨，就取消活動。"),
        [rel(
            K::Condition,
            L::Clause,
            "如果",
            Some("就"),
            "如果明天下雨",
            "就取消活動。"
        )]
    );
}

#[test]
fn zh_sequence_and_parallel_pairs() {
    assert_eq!(
        rels("先洗手，再吃飯。"),
        [rel(
            K::Sequence,
            L::Clause,
            "先",
            Some("再"),
            "先洗手",
            "再吃飯。"
        )]
    );
    assert!(rels("先生來了，我們走。").is_empty());
    assert!(rel_unknown("先生來了，我們走。").is_empty());
    assert_eq!(
        rels("一方面成本下降，另一方面品質上升。"),
        [rel(
            K::Parallel,
            L::Clause,
            "一方面",
            Some("另一方面"),
            "一方面成本下降",
            "另一方面品質上升。"
        )]
    );
    assert_eq!(
        rels("方法A較快，此外也較省記憶體。"),
        [rel(
            K::Parallel,
            L::Clause,
            "此外",
            None,
            "方法A較快",
            "此外也較省記憶體。"
        )]
    );
}

#[test]
fn zh_sentence_level_and_bare_connectors() {
    assert_eq!(
        rels("她推開門。然而，燈沒有亮。"),
        [rel(
            K::Contrast,
            L::Sentence,
            "然而",
            None,
            "她推開門。",
            "然而，燈沒有亮。"
        )]
    );
    assert_eq!(
        rels("他關了燈，然後，睡了。"),
        [rel(
            K::Sequence,
            L::Clause,
            "然後",
            None,
            "他關了燈",
            "睡了。"
        )]
    );
    assert_eq!(
        rels("我今天沒睡，所以好睏。"),
        [rel(
            K::Cause,
            L::Clause,
            "所以",
            None,
            "我今天沒睡",
            "所以好睏。"
        )]
    );
    assert!(rels("我們是為了好玩，但願你也喜歡。").is_empty());
    assert!(rel_unknown("我們是為了好玩，但願你也喜歡。").is_empty());
}

#[test]
fn zh_ambiguous_and_no_argument() {
    assert!(rels("他走了。同時，我留下來。").is_empty());
    assert_eq!(rel_unknown("他走了。同時，我留下來。"), ["同時"]);
    assert!(rels("但是我不同意。").is_empty());
    assert_eq!(rel_unknown("但是我不同意。"), ["但是"]);
}

#[test]
fn en_sequence_cause_contrast_condition() {
    assert_eq!(
        rels("He locked the door, then he went to bed."),
        [rel(
            K::Sequence,
            L::Clause,
            "then",
            None,
            "He locked the door",
            "then he went to bed."
        )]
    );
    assert_eq!(
        rels("Because the road was closed, we turned back."),
        [rel(
            K::Cause,
            L::Clause,
            "Because",
            None,
            "Because the road was closed",
            "we turned back."
        )]
    );
    assert_eq!(
        rels("We turned back, because the road was closed."),
        [rel(
            K::Cause,
            L::Clause,
            "because",
            None,
            "because the road was closed.",
            "We turned back"
        )]
    );
    assert_eq!(
        rels("The test passed, but the build failed."),
        [rel(
            K::Contrast,
            L::Clause,
            "but",
            None,
            "The test passed",
            "but the build failed."
        )]
    );
    assert_eq!(
        rels("If it rains, then we stay home."),
        [rel(
            K::Condition,
            L::Clause,
            "If",
            Some("then"),
            "If it rains",
            "then we stay home."
        )]
    );
    assert_eq!(
        rels("If it rains, we stay home."),
        [rel(
            K::Condition,
            L::Clause,
            "If",
            None,
            "If it rains",
            "we stay home."
        )]
    );
}

#[test]
fn en_sentence_level_and_bare_connectors() {
    assert_eq!(
        rels("The run finished. However, the logs were empty."),
        [rel(
            K::Contrast,
            L::Sentence,
            "However",
            None,
            "The run finished.",
            "However, the logs were empty."
        )]
    );
    assert_eq!(
        rels("He is tall; however, he is slow."),
        [rel(
            K::Contrast,
            L::Clause,
            "however",
            None,
            "He is tall",
            "he is slow."
        )]
    );
    assert_eq!(
        rels("The run finished. So we shipped it."),
        [rel(
            K::Cause,
            L::Sentence,
            "So",
            None,
            "The run finished.",
            "So we shipped it."
        )]
    );
    assert!(rels("The run finished. So, what now?").is_empty());
    assert_eq!(rel_unknown("The run finished. So, what now?"), ["So"]);
    assert_eq!(
        rels("Latency fell. Moreover, memory use fell."),
        [rel(
            K::Parallel,
            L::Sentence,
            "Moreover",
            None,
            "Latency fell.",
            "Moreover, memory use fell."
        )]
    );
    assert_eq!(
        rels("We train the model for ten epochs. Then we evaluate it on the held-out set."),
        [rel(
            K::Sequence,
            L::Sentence,
            "Then",
            None,
            "We train the model for ten epochs.",
            "Then we evaluate it on the held-out set."
        )]
    );
}

#[test]
fn en_ambiguous_and_non_connectors() {
    assert!(rels("It works, while the old one fails.").is_empty());
    assert_eq!(rel_unknown("It works, while the old one fails."), ["while"]);
    assert!(rels("Since the test passed, we merged.").is_empty());
    assert_eq!(rel_unknown("Since the test passed, we merged."), ["Since"]);
    for text in [
        "It is so far so good.",
        "So far, the results hold.",
        "All but one passed.",
        "However hard he tried, he failed.",
        "If only I had known, I would have stayed.",
    ] {
        assert!(rels(text).is_empty(), "{text}");
        assert!(rel_unknown(text).is_empty(), "{text}");
    }
    assert_eq!(
        rels("omg it works, but only on my laptop lol"),
        [rel(
            K::Contrast,
            L::Clause,
            "but",
            None,
            "omg it works",
            "but only on my laptop lol"
        )]
    );
    assert_eq!(
        rels("we tried it. so we left"),
        [rel(
            K::Cause,
            L::Sentence,
            "so",
            None,
            "we tried it.",
            "so we left"
        )]
    );
}

#[test]
fn mixed_zh_en() {
    assert_eq!(
        rels("我們跑了 benchmark，但是 latency 還是太高。"),
        [rel(
            K::Contrast,
            L::Clause,
            "但是",
            None,
            "我們跑了 benchmark",
            "但是 latency 還是太高。"
        )]
    );
    assert_eq!(
        rels("The model is fast, 所以 we keep it."),
        [rel(
            K::Cause,
            L::Clause,
            "所以",
            None,
            "The model is fast",
            "所以 we keep it."
        )]
    );
    assert_eq!(
        rels("lol 好啦，but 我還是不爽"),
        [rel(
            K::Contrast,
            L::Clause,
            "but",
            None,
            "lol 好啦",
            "but 我還是不爽"
        )]
    );
    assert_eq!(
        rels("If the loss drops，我們就停止訓練。"),
        [rel(
            K::Condition,
            L::Clause,
            "If",
            None,
            "If the loss drops",
            "我們就停止訓練。"
        )]
    );
}

#[test]
fn chains_scope_and_opaque() {
    assert_eq!(
        rels("因為下雨了，所以我們回家，然後睡覺。"),
        [
            rel(
                K::Cause,
                L::Clause,
                "因為",
                Some("所以"),
                "因為下雨了",
                "所以我們回家"
            ),
            rel(
                K::Sequence,
                L::Clause,
                "然後",
                None,
                "所以我們回家",
                "然後睡覺。"
            ),
        ]
    );
    assert_eq!(
        rels("如果下雨，但是我們還是去。"),
        [rel(
            K::Contrast,
            L::Clause,
            "但是",
            None,
            "如果下雨",
            "但是我們還是去。"
        )]
    );
    assert_eq!(rel_unknown("如果下雨，但是我們還是去。"), ["如果"]);
    assert!(rels("It works.\n\nBut it is slow.").is_empty());
    assert_eq!(rel_unknown("It works.\n\nBut it is slow."), ["But"]);
    assert!(rels("# Results\n\nBut it works.").is_empty());
    assert_eq!(rel_unknown("# Results\n\nBut it works."), ["But"]);

    let text = "これはテストです。But it works.";
    let doc = analyze(text);
    assert!(
        !doc.unknown
            .iter()
            .any(|u| u.reason == Reason::UnsupportedLanguage)
    );
    assert!(
        doc.tokens
            .iter()
            .any(|t| t.lang == faberline_nlp::doc::Lang::Ja && t.upos.is_some())
    );
    assert_eq!(
        rels(text),
        [rel(
            K::Contrast,
            L::Sentence,
            "But",
            None,
            "これはテストです。",
            "But it works."
        )]
    );
    assert!(rel_unknown(text).is_empty());
    assert!(rels("We ran it. これはテストです。").is_empty());
    assert!(rel_unknown("We ran it. これはテストです。").is_empty());
    let text = "It works. But it is slow. これはテストです。";
    assert_eq!(
        rels(text),
        [rel(
            K::Contrast,
            L::Sentence,
            "But",
            None,
            "It works.",
            "But it is slow."
        )]
    );
    assert!(rel_unknown(text).is_empty());
    let text = "It works. But これはテストです。";
    let doc = analyze(text);
    assert!(
        !doc.unknown
            .iter()
            .any(|u| u.reason == Reason::UnsupportedLanguage)
    );
    assert!(doc.tokens.iter().any(|t| {
        t.lang == faberline_nlp::doc::Lang::Ja
            && t.upos.is_some()
            && t.span.start_char >= doc.segmentation.sentences[1].span.start_char
            && t.span.end_char <= doc.segmentation.sentences[1].span.end_char
    }));
    assert_eq!(
        rels(text),
        [rel(
            K::Contrast,
            L::Sentence,
            "But",
            None,
            "It works.",
            "But これはテストです。"
        )]
    );
    assert!(rel_unknown(text).is_empty());

    // Korean sentences are likewise supported in Layer 7.
    let ko_before = "안녕하세요. But it works.";
    assert_eq!(
        rels(ko_before),
        [rel(
            K::Contrast,
            L::Sentence,
            "But",
            None,
            "안녕하세요.",
            "But it works."
        )]
    );
    assert!(rel_unknown(ko_before).is_empty());
    let ko_mixed = "It works. But 안녕하세요.";
    assert_eq!(
        rels(ko_mixed),
        [rel(
            K::Contrast,
            L::Sentence,
            "But",
            None,
            "It works.",
            "But 안녕하세요."
        )]
    );
    assert!(rel_unknown(ko_mixed).is_empty());

    // Truly unsupported script (Cyrillic) still emits `segmentation` `UnsupportedLanguage`.
    let cyr_before = "Привет, мир. But it works.";
    let doc_cyr_before = analyze(cyr_before);
    let unsupported: Vec<Span> = doc_cyr_before
        .unknown
        .iter()
        .filter(|u| u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage)
        .map(|u| u.span)
        .collect();
    assert_eq!(unsupported, [doc_cyr_before.segmentation.sentences[0].span]);
    assert!(rels(cyr_before).is_empty());
    assert_eq!(rel_unknown(cyr_before), ["But"]);

    let cyr_after = "It works. But Привет.";
    let doc_cyr_after = analyze(cyr_after);
    let unsupported_after: Vec<Span> = doc_cyr_after
        .unknown
        .iter()
        .filter(|u| u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage)
        .map(|u| u.span)
        .collect();
    assert_eq!(
        unsupported_after,
        [doc_cyr_after.segmentation.sentences[1].span]
    );
    assert!(rels(cyr_after).is_empty());
    assert!(rel_unknown(cyr_after).is_empty());

    assert_eq!(
        rels("`then` is a keyword, then we go."),
        [rel(
            K::Sequence,
            L::Clause,
            "then",
            None,
            "`then` is a keyword",
            "then we go."
        )]
    );
}

#[test]
fn because_pairs_with_so_and_therefore() {
    assert_eq!(
        rels("Because the road was closed, so we turned back."),
        [rel(
            K::Cause,
            L::Clause,
            "Because",
            Some("so"),
            "Because the road was closed",
            "so we turned back."
        )]
    );
    assert_eq!(
        rels("Because it rained, therefore we stayed."),
        [rel(
            K::Cause,
            L::Clause,
            "Because",
            Some("therefore"),
            "Because it rained",
            "therefore we stayed."
        )]
    );
    assert_eq!(
        rels("因為下雨，therefore we stayed."),
        [rel(
            K::Cause,
            L::Clause,
            "因為",
            Some("therefore"),
            "因為下雨",
            "therefore we stayed."
        )]
    );
    for text in [
        "Because the road was closed, so we turned back.",
        "Because it rained, therefore we stayed.",
        "因為下雨，therefore we stayed.",
    ] {
        assert!(rel_unknown(text).is_empty(), "{text}");
    }
    // `so far` is not a connector, so it is not a correlate either.
    assert_eq!(
        rels("Because it rained, so far we stayed."),
        [rel(
            K::Cause,
            L::Clause,
            "Because",
            None,
            "Because it rained",
            "so far we stayed."
        )]
    );
}

#[test]
fn then_after_if_is_not_a_sequence() {
    assert!(rels("If, then we go.").is_empty());
    assert_eq!(rel_unknown("If, then we go."), ["If"]);
}

#[test]
fn ja_discourse_relations() {
    assert_eq!(
        rels("雨が降った。しかし、試合は続いた。"),
        [rel(
            K::Contrast,
            L::Sentence,
            "しかし",
            None,
            "雨が降った。",
            "しかし、試合は続いた。"
        )]
    );
    assert_eq!(
        rels("彼は努力した。それにもかかわらず、失敗した。"),
        [rel(
            K::Contrast,
            L::Sentence,
            "それにもかかわらず",
            None,
            "彼は努力した。",
            "それにもかかわらず、失敗した。"
        )]
    );
    assert_eq!(
        rels("雨が降った。したがって、試合は中止になった。"),
        [rel(
            K::Cause,
            L::Sentence,
            "したがって",
            None,
            "雨が降った。",
            "したがって、試合は中止になった。"
        )]
    );
    assert_eq!(
        rels("もし雨が降ったら、家にいます。"),
        [rel(
            K::Condition,
            L::Clause,
            "もし",
            None,
            "もし雨が降ったら",
            "家にいます。"
        )]
    );
    assert_eq!(
        rels("鍵をかけた。それから、寝た。"),
        [rel(
            K::Sequence,
            L::Sentence,
            "それから",
            None,
            "鍵をかけた。",
            "それから、寝た。"
        )]
    );
    assert_eq!(
        rels("速度が上がった。さらに、精度も向上した。"),
        [rel(
            K::Parallel,
            L::Sentence,
            "さらに",
            None,
            "速度が上がった。",
            "さらに、精度も向上した。"
        )]
    );
    // Negative prefix guards: または / またがる / もしもし must NOT trigger.
    for text in [
        "リンゴを買った。または、バナナを買う。",
        "橋がある。またがる問題だ。",
        "電話が鳴った。もしもし、元気ですか。",
    ] {
        assert!(rels(text).is_empty(), "{text}");
        assert!(rel_unknown(text).is_empty(), "{text}");
    }
}

#[test]
fn ko_discourse_relations() {
    assert_eq!(
        rels("비가 왔다. 하지만 경기는 계속되었다."),
        [rel(
            K::Contrast,
            L::Sentence,
            "하지만",
            None,
            "비가 왔다.",
            "하지만 경기는 계속되었다."
        )]
    );
    assert_eq!(
        rels("비가 왔다. 그럼에도 불구하고 출발했다."),
        [rel(
            K::Contrast,
            L::Sentence,
            "그럼에도 불구하고",
            None,
            "비가 왔다.",
            "그럼에도 불구하고 출발했다."
        )]
    );
    assert_eq!(
        rels("비가 왔다. 그러므로 경기는 취소되었다."),
        [rel(
            K::Cause,
            L::Sentence,
            "그러므로",
            None,
            "비가 왔다.",
            "그러므로 경기는 취소되었다."
        )]
    );
    assert_eq!(
        rels("만약 비가 오면, 집에 있겠다."),
        [rel(
            K::Condition,
            L::Clause,
            "만약",
            None,
            "만약 비가 오면",
            "집에 있겠다."
        )]
    );
    assert_eq!(
        rels("문을 잠갔다. 그러고 나서 잠들었다."),
        [rel(
            K::Sequence,
            L::Sentence,
            "그러고 나서",
            None,
            "문을 잠갔다.",
            "그러고 나서 잠들었다."
        )]
    );
    assert_eq!(
        rels("속도가 빨라졌다. 게다가 정확도도 높아졌다."),
        [rel(
            K::Parallel,
            L::Sentence,
            "게다가",
            None,
            "속도가 빨라졌다.",
            "게다가 정확도도 높아졌다."
        )]
    );
    // Negative prefix guards: unspaced Hangul continuation must NOT trigger.
    for text in [
        "비가 왔다. 만약의 경우를 대비하자.",
        "그가 왔다. 그러나저러나 집에 가자.",
    ] {
        assert!(rels(text).is_empty(), "{text}");
        assert!(rel_unknown(text).is_empty(), "{text}");
    }
}

#[test]
fn json_shape_and_plain_text() {
    let text = "他關了燈，然後睡了。";
    let json = serde_json::to_string(&analyze(text)).expect("json");
    let at = |key: &str| json.find(key).unwrap_or_else(|| panic!("no {key}"));
    assert!(at("\"segmentation\"") < at("\"tokens\""));
    assert!(at("\"tokens\"") < at("\"relations\""));
    assert!(at("\"relations\"") < at("\"unknown\""));
    assert!(json.contains("\"kind\":\"sequence\",\"level\":\"clause\""));
    assert!(!json.contains("\"correlate\":null"));

    let plain = serde_json::to_value(analyze("The cat sat.")).expect("json");
    assert_eq!(plain["relations"], json!([]));
}

/// Every text of the numbered cases above, with the bare-connector and pair cases.
const CASES: &[&str] = &[
    // zh-TW
    "他關了燈，然後睡了。",
    "因為下雨了，所以比賽取消了。",
    "我們等了很久，但是他沒有來。",
    "如果明天下雨，就取消活動。",
    "先洗手，再吃飯。",
    "先生來了，我們走。",
    "一方面成本下降，另一方面品質上升。",
    "方法A較快，此外也較省記憶體。",
    "她推開門。然而，燈沒有亮。",
    "他關了燈，然後，睡了。",
    "我今天沒睡，所以好睏。",
    "我們是為了好玩，但願你也喜歡。",
    "他走了。同時，我留下來。",
    "但是我不同意。",
    // English
    "He locked the door, then he went to bed.",
    "Because the road was closed, we turned back.",
    "We turned back, because the road was closed.",
    "The test passed, but the build failed.",
    "If it rains, then we stay home.",
    "If it rains, we stay home.",
    "The run finished. However, the logs were empty.",
    "He is tall; however, he is slow.",
    "He tried, but, he failed.",
    "The run finished. So we shipped it.",
    "The run finished. So, what now?",
    "Latency fell. Moreover, memory use fell.",
    "We train the model for ten epochs. Then we evaluate it on the held-out set.",
    "It works, while the old one fails.",
    "Since the test passed, we merged.",
    "It is so far so good.",
    "So far, the results hold.",
    "All but one passed.",
    "However hard he tried, he failed.",
    "If only I had known, I would have stayed.",
    "omg it works, but only on my laptop lol",
    "we tried it. so we left",
    "Because the road was closed, so we turned back.",
    "Because it rained, therefore we stayed.",
    "Because it rained, so far we stayed.",
    "If, then we go.",
    // mixed
    "我們跑了 benchmark，但是 latency 還是太高。",
    "The model is fast, 所以 we keep it.",
    "lol 好啦，but 我還是不爽",
    "If the loss drops，我們就停止訓練。",
    "因為下雨，therefore we stayed.",
    // chains, scope and opaque
    "因為下雨了，所以我們回家，然後睡覺。",
    "如果下雨，但是我們還是去。",
    "It works.\n\nBut it is slow.",
    "# Results\n\nBut it works.",
    "これはテストです。But it works.",
    "We ran it. これはテストです。",
    "`then` is a keyword, then we go.",
    // review corrections
    "先洗手，然後。",
    "一方面便宜，另一方面。",
    "It works. But it is slow. これはテストです。",
    "It works. But これはテストです。",
    "안녕하세요. But it works.",
    "It works. But 안녕하세요.",
    "Привет, мир. But it works.",
    "It works. But Привет.",
    "We left, because, it rained.",
    "如果，我們回家。",
    "Because, we stayed.",
    "先，然後吃飯。",
    "先洗手，然後，如果餓了，就吃飯。",
    "If it rains.\n\nThen we stay home.",
    "We lost, 😋 because it rained.",
    "如果明天有空，便宜的也可以。",
    "先走吧，再見。",
    "如果明天有空，便來找你。",
    "先走吧，再去找你。",
    "We won. 😋 But we lost.",
    "We ran it, `but` it failed.",
    "We ran it. `But it failed.`",
    "We ran it. (But 2020) says it failed.",
    "If it rains, then, we stay.",
    "If it rains, then.",
    // second review
    "If it rains. Then we stay home.",
    "如果明天下雨。Then we stay home.",
    "先洗手，然後，但是吃飯。",
    // third review
    "If it rains, then, because we stay.",
    "If it rains, we stay. Then we rest.",
    "We go, if it rains. Then we rest.",
    // because alone in its sentence
    "We stayed. Because it rained.",
    "我們留下。因為。",
    "Because.",
    "We stayed. Because, it rained.",
    "We stayed.\n\nBecause it rained.",
    // unknown examples of the docs
    "If it rains.",
    "We left, but.",
    "先洗手。",
    "先洗手，吃飯。",
    "一方面便宜，但是不好。",
];

/// Brackets and quotes that may stand before a connector inside a clause.
const OPENING_MARKS: &str = "「『“‘（([\"'";

/// Chars that the clause start skips before a connector: whitespace, opening marks, and symbols and
/// pictographs (the same ranges as `is_symbol` in relation.rs, where emoji are skipped too).
fn skipped_before_connector(ch: char) -> bool {
    ch.is_whitespace()
        || OPENING_MARKS.contains(ch)
        || matches!(
            ch,
            '\u{A9}' | '\u{AE}' | '\u{2122}' | '\u{2190}'..='\u{2BFF}' | '\u{1F000}'..='\u{1FAFF}'
                | '\u{200D}' | '\u{FE0F}'
        )
}

/// Every returned relation and unknown: offsets are in range, relations are sorted by connector,
/// the arguments are whole layer-1 units and do not overlap, the connector lies in a clause or a
/// sentence, and it overlaps arg1 only when arg1 starts with it. The unknowns of this layer are
/// all `ambiguous`. The result is the same on a second run.
fn check_invariants(text: &str) {
    let doc = analyze(text);
    let n = text.chars().count();
    let chars: Vec<char> = text.chars().collect();
    let in_range = |s: Span| s.start_char <= s.end_char && s.end_char <= n;
    let inside =
        |s: Span, outer: Span| outer.start_char <= s.start_char && s.end_char <= outer.end_char;
    let overlaps = |a: Span, b: Span| a.start_char < b.end_char && b.start_char < a.end_char;
    let clauses: Vec<Span> = doc.segmentation.clauses.iter().map(|c| c.span).collect();
    let sentences: Vec<Span> = doc.segmentation.sentences.iter().map(|s| s.span).collect();
    let units: Vec<Span> = clauses.iter().chain(&sentences).copied().collect();
    assert!(units.iter().all(|&s| in_range(s)), "{text}: layer 1 offset");

    let starts: Vec<usize> = doc
        .relations
        .iter()
        .map(|r| r.connector.start_char)
        .collect();
    assert!(starts.windows(2).all(|w| w[0] <= w[1]), "{text}");
    for r in &doc.relations {
        assert!(
            in_range(r.connector) && in_range(r.arg1) && in_range(r.arg2),
            "{text}"
        );
        assert!(r.correlate.is_none_or(in_range), "{text}");
        let level_units = match r.level {
            L::Clause => &clauses,
            L::Sentence => &sentences,
        };
        assert!(level_units.contains(&r.arg1), "{text}: arg1 is not a unit");
        assert!(level_units.contains(&r.arg2), "{text}: arg2 is not a unit");
        assert!(!overlaps(r.arg1, r.arg2), "{text}: arguments overlap");
        assert!(
            units.iter().any(|&u| inside(r.connector, u)),
            "{text}: connector is outside every clause and sentence"
        );
        if overlaps(r.connector, r.arg1) {
            let before = r.arg1.start_char <= r.connector.start_char
                && chars[r.arg1.start_char..r.connector.start_char]
                    .iter()
                    .all(|ch| skipped_before_connector(*ch));
            assert!(
                before,
                "{text}: connector overlaps arg1 without starting it"
            );
        }
    }

    let unknown: Vec<_> = doc
        .unknown
        .iter()
        .filter(|u| u.layer == "relations")
        .collect();
    for u in &unknown {
        assert_eq!(u.reason, Reason::Ambiguous, "{text}");
        assert!(in_range(u.span), "{text}");
    }
    let again = serde_json::to_string(&analyze(text)).expect("json");
    assert_eq!(serde_json::to_string(&doc).expect("json"), again, "{text}");
}

#[test]
fn invariants() {
    for text in CASES {
        check_invariants(text);
    }
    for g in gold() {
        check_invariants(&g.text);
    }
}

#[derive(Deserialize)]
struct GoldRel {
    kind: K,
    level: L,
    connector: String,
    correlate: Option<String>,
    arg1: String,
    arg2: String,
}

#[derive(Deserialize)]
struct Gold {
    genre: String,
    lang: String,
    text: String,
    relations: Vec<GoldRel>,
    ambiguous: Vec<String>,
}

fn manifest(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn gold() -> Vec<Gold> {
    let raw = fs::read_to_string(manifest(GOLD_PATH)).expect("gold file");
    serde_json::from_str(&raw).expect("gold file parses")
}

#[test]
fn gold_file_is_well_formed() {
    let gold = gold();
    assert_eq!(gold.len(), 45);
    for genre in ["novel", "paper", "casual"] {
        for lang in ["zh", "en", "mixed"] {
            let n = gold
                .iter()
                .filter(|g| g.genre == genre && g.lang == lang)
                .count();
            assert_eq!(n, 5, "{genre} {lang}");
        }
    }
    let none = gold
        .iter()
        .filter(|g| g.relations.is_empty() && g.ambiguous.is_empty())
        .count();
    assert!(none >= 10, "items with no connector: {none}");
    for g in &gold {
        for r in &g.relations {
            assert!(
                g.text.contains(&r.arg1) && g.text.contains(&r.arg2),
                "{}",
                g.text
            );
        }
    }
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// (predicted, matched, gold, per kind as (kind, predicted, gold, matched), ambiguous agreeing,
/// ambiguous gold).
type Score = (
    usize,
    usize,
    usize,
    Vec<(K, usize, usize, usize)>,
    usize,
    usize,
);

/// Predicted relations that match a gold relation, per kind and in total.
fn measure(gold: &[Gold]) -> Score {
    let kinds = [
        K::Sequence,
        K::Cause,
        K::Contrast,
        K::Parallel,
        K::Condition,
    ];
    let (mut predicted, mut matched, mut total_gold) = (0, 0, 0);
    let mut per = Vec::new();
    for kind in kinds {
        per.push((kind, 0usize, 0usize, 0usize)); // (kind, predicted, gold, matched)
    }
    let mut amb_agree = 0;
    let mut amb_gold = 0;
    for g in gold {
        let mut pool: Vec<Rel> = g
            .relations
            .iter()
            .map(|r| {
                (
                    r.kind,
                    r.level,
                    r.connector.clone(),
                    r.correlate.clone(),
                    r.arg1.clone(),
                    r.arg2.clone(),
                )
            })
            .collect();
        for r in &g.relations {
            per.iter_mut().find(|p| p.0 == r.kind).expect("kind").2 += 1;
        }
        total_gold += pool.len();
        let doc = analyze(&g.text);
        let preds: Vec<Rel> = doc.relations.iter().map(|r| cut(&g.text, r)).collect();
        for p in &preds {
            per.iter_mut().find(|x| x.0 == p.0).expect("kind").1 += 1;
            predicted += 1;
            if let Some(i) = pool.iter().position(|q| q == p) {
                pool.remove(i);
                matched += 1;
                per.iter_mut().find(|x| x.0 == p.0).expect("kind").3 += 1;
            }
        }
        let unknowns = rel_unknown(&g.text);
        for a in &g.ambiguous {
            amb_gold += 1;
            if unknowns.contains(a) {
                amb_agree += 1;
            }
        }
    }
    (predicted, matched, total_gold, per, amb_agree, amb_gold)
}

#[test]
fn relation_receipt() {
    let gold = gold();
    let (predicted, matched, total_gold, per, amb_agree, amb_gold) = measure(&gold);
    let precision = matched as f64 / predicted as f64;
    let recall = matched as f64 / total_gold as f64;
    assert!(precision >= 0.9, "precision {precision} is below 0.9");
    assert!(recall >= 0.8, "recall {recall} is below 0.8");

    if std::env::var("NLP_WRITE_RECEIPT").as_deref() != Ok("1") {
        return;
    }
    let mut per_kind = serde_json::Map::new();
    for (kind, p, g, m) in &per {
        let name = serde_json::to_value(kind)
            .expect("kind")
            .as_str()
            .expect("name")
            .to_owned();
        let ratio = |num: usize, den: usize| (den > 0).then(|| round4(num as f64 / den as f64));
        per_kind.insert(
            name,
            json!({
                "gold": g,
                "predicted": p,
                "matched": m,
                "precision": ratio(*m, *p),
                "recall": ratio(*m, *g),
            }),
        );
    }
    let raw = fs::read(manifest(GOLD_PATH)).expect("gold file");
    let receipt = json!({
        "layer": "relations",
        "date": "2026-11-10",
        "gold_file": GOLD_PATH,
        "gold_sha256": Sha256::digest(&raw).iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "gold_items": gold.len(),
        "gold_relations": total_gold,
        "predicted": predicted,
        "matched": matched,
        "precision": round4(precision),
        "recall": round4(recall),
        "per_kind": per_kind,
        "ambiguous_agreement": {
            "matched": amb_agree,
            "gold": amb_gold,
            "rate": round4(amb_agree as f64 / amb_gold as f64),
        },
        "notes": [
            "The gold set was written by the same author as the rules, so it is not independent.",
            "Known misses: connectors in the middle of a clause (for example `He left because it rained`, `A but B` without a comma, and `如果下雨我們就` with 就 mid-clause) are not seen by design.",
            "Only the clause before, or the sentence before in the same paragraph, is the other argument. Implicit relations are not in the gold set.",
            "Precision is matched predicted relations over all predicted; recall is matched gold relations over all gold relations. A match needs kind, level, connector, correlate, arg1 and arg2 to be equal.",
        ],
    });
    let path = manifest(RECEIPT_PATH);
    fs::create_dir_all(path.parent().expect("receipt dir")).expect("create receipt dir");
    let mut text = serde_json::to_string_pretty(&receipt).expect("receipt json");
    text.push('\n');
    fs::write(&path, text).expect("write receipt");
}

#[test]
fn review_correlate_is_a_whole_word() {
    // 便 inside 便宜 and 再 inside 再見 are not correlates.
    assert_eq!(
        rels("如果明天有空，便宜的也可以。"),
        [rel(
            K::Condition,
            L::Clause,
            "如果",
            None,
            "如果明天有空",
            "便宜的也可以。"
        )]
    );
    assert!(rels("先走吧，再見。").is_empty());
    // The real correlates still pair.
    assert_eq!(
        rels("如果明天有空，便來找你。"),
        [rel(
            K::Condition,
            L::Clause,
            "如果",
            Some("便"),
            "如果明天有空",
            "便來找你。"
        )]
    );
    assert_eq!(
        rels("先走吧，再去找你。"),
        [rel(
            K::Sequence,
            L::Clause,
            "先",
            Some("再"),
            "先走吧",
            "再去找你。"
        )]
    );
}

#[test]
fn review_clause_after_an_emoji() {
    let text = "We won. 😋 But we lost.";
    assert_eq!(
        rels(text),
        [rel(
            K::Contrast,
            L::Sentence,
            "But",
            None,
            "We won.",
            "😋 But we lost."
        )]
    );
}

#[test]
fn review_silent_opener_with_a_bare_correlate() {
    // `先` and `一方面` never give an unknown, even when their correlate is bare.
    for text in ["先洗手，然後。", "一方面便宜，另一方面。"] {
        assert!(rels(text).is_empty(), "{text}");
        assert!(rel_unknown(text).is_empty(), "{text}");
    }
}

#[test]
fn review_bare_because_not_first_takes_the_clause_before() {
    assert_eq!(
        rels("We left, because, it rained."),
        [rel(
            K::Cause,
            L::Clause,
            "because",
            None,
            "because",
            "We left"
        )]
    );
}

#[test]
fn review_connector_inside_opaque_is_not_a_connector() {
    let text = "We ran it, `but` it failed.";
    assert!(rels(text).is_empty());
    let text = "We ran it. `But it failed.`";
    assert!(rels(text).is_empty());
    // A citation that holds the word is opaque, so its text is not a connector.
    let text = "We ran it. (But 2020) says it failed.";
    assert!(rels(text).is_empty());
    assert!(rel_unknown(text).is_empty());
}

#[test]
fn review_bare_correlate_takes_the_next_clause() {
    assert_eq!(
        rels("If it rains, then, we stay."),
        [rel(
            K::Condition,
            L::Clause,
            "If",
            Some("then"),
            "If it rains",
            "we stay."
        )]
    );
    // Nothing follows the bare correlate: unknown, no relation.
    let text = "If it rains, then.";
    assert!(rels(text).is_empty());
    assert_eq!(rel_unknown(text), ["If"]);
}

#[test]
fn review_then_after_an_if_sentence_is_not_a_sequence() {
    // `then` that starts the next sentence is the correlate of the `if` before it: no relation.
    // The `if` has no argument, so it is an unknown.
    assert!(rels("If it rains. Then we stay home.").is_empty());
    assert_eq!(rel_unknown("If it rains. Then we stay home."), ["If"]);
    assert!(rels("如果明天下雨。Then we stay home.").is_empty());
    assert_eq!(rel_unknown("如果明天下雨。Then we stay home."), ["如果"]);
    // Only the last clause of the sentence before counts: here it does not open with `if`, so
    // `Then` is a sequence.
    assert_eq!(
        rels("If it rains, we stay. Then we rest."),
        [
            rel(
                K::Condition,
                L::Clause,
                "If",
                None,
                "If it rains",
                "we stay."
            ),
            rel(
                K::Sequence,
                L::Sentence,
                "Then",
                None,
                "If it rains, we stay.",
                "Then we rest."
            ),
        ]
    );
    assert!(rel_unknown("If it rains, we stay. Then we rest.").is_empty());
    // A new paragraph breaks the sentence relation: `Then` has no unit before it.
    assert!(rels("If it rains.\n\nThen we stay home.").is_empty());
    assert_eq!(
        rel_unknown("If it rains.\n\nThen we stay home."),
        ["If", "Then"]
    );
    // The last clause of the sentence before opens with `if`: `Then` gives nothing.
    let text = "We go, if it rains. Then we rest.";
    assert!(rels(text).is_empty());
    assert_eq!(rel_unknown(text), ["if"]);
}

#[test]
fn review_bare_correlate_before_another_connector() {
    // The bare `然後` is arg1 of `但是`; the opener `先` gives nothing.
    assert_eq!(
        rels("先洗手，然後，但是吃飯。"),
        [rel(
            K::Contrast,
            L::Clause,
            "但是",
            None,
            "然後",
            "但是吃飯。"
        )]
    );
    assert!(rel_unknown("先洗手，然後，但是吃飯。").is_empty());
    // The bare `then` is arg2 of `because`; the `If` before it gives an unknown.
    assert_eq!(
        rels("If it rains, then, because we stay."),
        [rel(
            K::Cause,
            L::Clause,
            "because",
            None,
            "because we stay.",
            "then"
        )]
    );
    assert_eq!(rel_unknown("If it rains, then, because we stay."), ["If"]);
}

#[test]
fn review_bare_opener_gives_no_relation() {
    // A bare opener gives no relation; its unknown is the opener itself.
    assert!(rels("如果，我們回家。").is_empty());
    assert_eq!(rel_unknown("如果，我們回家。"), ["如果"]);
    assert!(rels("Because, we stayed.").is_empty());
    assert_eq!(rel_unknown("Because, we stayed."), ["Because"]);
    // A bare `先` gives nothing; the next clause is read on its own.
    assert_eq!(
        rels("先，然後吃飯。"),
        [rel(
            K::Sequence,
            L::Clause,
            "然後",
            None,
            "先",
            "然後吃飯。"
        )]
    );
    assert!(rel_unknown("先，然後吃飯。").is_empty());
    // The opener `如果` in the clause after a bare correlate gives only its own relation.
    assert_eq!(
        rels("先洗手，然後，如果餓了，就吃飯。"),
        [rel(
            K::Condition,
            L::Clause,
            "如果",
            Some("就"),
            "如果餓了",
            "就吃飯。"
        )]
    );
    assert!(rel_unknown("先洗手，然後，如果餓了，就吃飯。").is_empty());
}

#[test]
fn review_because_alone_in_its_sentence() {
    // A `Because` that is the only clause of its sentence takes the sentence before it as arg2.
    assert_eq!(
        rels("We stayed. Because it rained."),
        [rel(
            K::Cause,
            L::Sentence,
            "Because",
            None,
            "Because it rained.",
            "We stayed."
        )]
    );
    assert!(rel_unknown("We stayed. Because it rained.").is_empty());
    // The same when the connector is bare.
    assert_eq!(
        rels("我們留下。因為。"),
        [rel(
            K::Cause,
            L::Sentence,
            "因為",
            None,
            "因為。",
            "我們留下。"
        )]
    );
    assert!(rel_unknown("我們留下。因為。").is_empty());
    // A `Because` with nothing before it or after it gives no relation, only an unknown.
    assert!(rels("Because.").is_empty());
    assert_eq!(rel_unknown("Because."), ["Because"]);
    // The sentence before must be in the same paragraph.
    assert!(rels("We stayed.\n\nBecause it rained.").is_empty());
    assert_eq!(rel_unknown("We stayed.\n\nBecause it rained."), ["Because"]);
    // A bare `Because` with a clause after it is not alone: an unknown, no relation.
    assert!(rels("We stayed. Because, it rained.").is_empty());
    assert_eq!(rel_unknown("We stayed. Because, it rained."), ["Because"]);
}

#[test]
fn review_unknown_examples_of_the_docs() {
    // An opener with no clause after it in its sentence.
    assert!(rels("If it rains.").is_empty());
    assert_eq!(rel_unknown("If it rains."), ["If"]);
    // A bare backward connector that is not first and has no clause after it.
    assert!(rels("We left, but.").is_empty());
    assert_eq!(rel_unknown("We left, but."), ["but"]);
    // `先` and `一方面` give nothing where another opener gives an unknown.
    for text in ["先洗手。", "先洗手，吃飯。", "一方面便宜，但是不好。"] {
        assert!(
            rels(text).iter().all(|r| r.2 != "先" && r.2 != "一方面"),
            "{text}"
        );
        assert!(rel_unknown(text).is_empty(), "{text}");
    }
}

#[test]
fn ws6_narrative_connectors_and_post_temporal_frames() {
    // Unambiguous narrative sequence connectors (`隨即/随即`, `緊接著/紧接着`, `旋即`).
    assert_eq!(
        rels("木門緩緩開啟，隨即走入兩名巡守，緊接著燈火亮起，旋即大廳恢復安靜。"),
        [
            rel(
                K::Sequence,
                L::Clause,
                "隨即",
                None,
                "木門緩緩開啟",
                "隨即走入兩名巡守"
            ),
            rel(
                K::Sequence,
                L::Clause,
                "緊接著",
                None,
                "隨即走入兩名巡守",
                "緊接著燈火亮起"
            ),
            rel(
                K::Sequence,
                L::Clause,
                "旋即",
                None,
                "緊接著燈火亮起",
                "旋即大廳恢復安靜。"
            ),
        ]
    );
    assert_eq!(
        rels("信號發出，随即船帆升起，紧接着船隊離港。"),
        [
            rel(
                K::Sequence,
                L::Clause,
                "随即",
                None,
                "信號發出",
                "随即船帆升起"
            ),
            rel(
                K::Sequence,
                L::Clause,
                "紧接着",
                None,
                "随即船帆升起",
                "紧接着船隊離港。"
            ),
        ]
    );

    // Parallel (`與此同時/与此同时`), Contrast (`不過/不过`), Condition (`否則/否则`).
    assert_eq!(
        rels("北側山谷起霧，與此同時南側渡口也刮起大風。"),
        [rel(
            K::Parallel,
            L::Clause,
            "與此同時",
            None,
            "北側山谷起霧",
            "與此同時南側渡口也刮起大風。"
        )]
    );
    assert_eq!(
        rels("前隊抵達驛站。与此同时後隊仍在山路行進。"),
        [rel(
            K::Parallel,
            L::Sentence,
            "与此同时",
            None,
            "前隊抵達驛站。",
            "与此同时後隊仍在山路行進。"
        )]
    );
    assert_eq!(
        rels("這條山路崎嶇，不過風景十分開闊。"),
        [rel(
            K::Contrast,
            L::Clause,
            "不過",
            None,
            "這條山路崎嶇",
            "不過風景十分開闊。"
        )]
    );
    assert_eq!(
        rels("方案可行，不过還需要補充預算。"),
        [rel(
            K::Contrast,
            L::Clause,
            "不过",
            None,
            "方案可行",
            "不过還需要補充預算。"
        )]
    );
    // Idiomatic `不過如此` / `不過爾爾` must not trigger a contrast connector.
    assert!(rels("大家原以為很難，不過如此而已。").is_empty());
    assert!(rels("傳聞誇大其詞，不過爾爾。").is_empty());
    assert!(rels("传闻夸大其词，不过尔尔。").is_empty());

    assert_eq!(
        rels("必須在日落前趕到渡口，否則船隻就會開走。"),
        [rel(
            K::Condition,
            L::Clause,
            "否則",
            None,
            "必須在日落前趕到渡口",
            "否則船隻就會開走。"
        )]
    );
    assert_eq!(
        rels("請提前備妥通行證。否则無法進入園區。"),
        [rel(
            K::Condition,
            L::Sentence,
            "否则",
            None,
            "請提前備妥通行證。",
            "否则無法進入園區。"
        )]
    );

    // Clause-end post-positional temporal frames (`之後/之后`, `之際/之际`, `之時/之时`).
    for (text, expected_conn, expected_a1, expected_a2) in [
        (
            "片刻之後，旅人推開了木門。",
            "之後",
            "片刻之後",
            "旅人推開了木門。",
        ),
        (
            "商議妥當之后，眾人各自散去。",
            "之后",
            "商議妥當之后",
            "眾人各自散去。",
        ),
        (
            "話音剛落之際，遠處亮起燈火。",
            "之際",
            "話音剛落之際",
            "遠處亮起燈火。",
        ),
        (
            "眾人猶豫之际，北風忽然轉急。",
            "之际",
            "眾人猶豫之际",
            "北風忽然轉急。",
        ),
        (
            "正當危急之時，守衛打開了側門。",
            "之時",
            "正當危急之時",
            "守衛打開了側門。",
        ),
        (
            "天色將明之时，船隊駛出了港灣。",
            "之时",
            "天色將明之时",
            "船隊駛出了港灣。",
        ),
        (
            "「商議妥當之後」，眾人立刻動身。",
            "之後",
            "「商議妥當之後」",
            "眾人立刻動身。",
        ),
    ] {
        assert_eq!(
            rels(text),
            [rel(
                K::Sequence,
                L::Clause,
                expected_conn,
                None,
                expected_a1,
                expected_a2
            )],
            "{text}"
        );
    }

    // Bare `之後` (< 3 chars) is ignored; a conflicting connector on c_{i+1} suppresses duplicate frame emission.
    assert!(rels("之後，旅人離開了。").is_empty());
    assert_eq!(
        rels("數日之後，然而大門依然緊閉。"),
        [rel(
            K::Contrast,
            L::Clause,
            "然而",
            None,
            "數日之後",
            "然而大門依然緊閉。"
        )]
    );
    assert_eq!(
        rels("片刻之後，隨即傳來一陣腳步聲。"),
        [rel(
            K::Sequence,
            L::Clause,
            "隨即",
            None,
            "片刻之後",
            "隨即傳來一陣腳步聲。"
        )]
    );
}

#[test]
fn ws6_v3_kinship_spatial_after_guard_and_parallel_variants() {
    // Kinship ("descendant") `之後/之后` must NOT emit a temporal `Sequence` relation.
    for text in [
        "他本是故人之後，眾人對他頗為禮遇。",
        "此人既是高人之后，見識自然不凡。",
        "身為名門之後，舉止一向穩重。",
        "那少年乃是將門之后，自幼熟讀兵書。",
        "傳聞此女為仙人之後，身懷異稟。",
        "故人之後，理應多加照拂。",
    ] {
        assert!(
            rels(text).is_empty(),
            "kinship sense should not emit Sequence: {text}"
        );
    }

    // Spatial ("behind") `之後/之后` introduced by locative verbs must NOT emit `Sequence`.
    for text in [
        "一道身影出現在了那道光影之後，悄然攔住去路。",
        "兩名護衛出现在車隊之后，警惕地望向四周。",
        "探子緊緊跟在商隊之後，沒有發出半點聲響。",
        "灰衣老者落在眾人之后，緩步踏上石階。",
        "守衛靜靜站在石門之後，手握長矛。",
        "新來的學徒排在隊伍之后，依次登記姓名。",
        "黑影躲在巨石之後，屏住呼吸。",
        "暗哨藏在屏風之后，留意廳內動靜。",
        "一輪圓月懸在山峰之後，灑下清冷輝光。",
        "石像立在殿門之后，高逾數丈。",
        "古碑立於高台之後，字跡斑駁。",
        "古亭立于竹林之后，環境幽靜。",
        "小村位於山嶺之後，極少有人踏足。",
        "驛站位于河灣之后，往來商旅常在此歇腳。",
    ] {
        assert!(
            rels(text).is_empty(),
            "spatial sense should not emit Sequence: {text}"
        );
    }

    // Genuine temporal `之後` after an explicit duration or eventive verb still emits `Sequence`.
    assert_eq!(
        rels("站在門外片刻之後，旅人輕輕叩響了木環。"),
        [rel(
            K::Sequence,
            L::Clause,
            "之後",
            None,
            "站在門外片刻之後",
            "旅人輕輕叩響了木環。"
        )]
    );
    assert_eq!(
        rels("見到故人之後，老者欣然邀其入座。"),
        [rel(
            K::Sequence,
            L::Clause,
            "之後",
            None,
            "見到故人之後",
            "老者欣然邀其入座。"
        )]
    );

    // Clause-level (`，與此同時，…`) and sentence-level (`。與此同時，…`) `Parallel`, including
    // mixed-script (`与此同時`/`與此同时`) and discourse-prefixed (`而與此同時`/`幾乎與此同時`).
    assert_eq!(
        rels("北側山谷起霧，與此同時，南側渡口也刮起大風。"),
        [rel(
            K::Parallel,
            L::Clause,
            "與此同時",
            None,
            "北側山谷起霧",
            "南側渡口也刮起大風。"
        )]
    );
    assert_eq!(
        rels("前隊抵達驛站。與此同時，後隊仍在山路行進。"),
        [rel(
            K::Parallel,
            L::Sentence,
            "與此同時",
            None,
            "前隊抵達驛站。",
            "與此同時，後隊仍在山路行進。"
        )]
    );
    assert_eq!(
        rels("前隊抵達驛站。与此同时，後隊仍在山路行進。"),
        [rel(
            K::Parallel,
            L::Sentence,
            "与此同时",
            None,
            "前隊抵達驛站。",
            "与此同时，後隊仍在山路行進。"
        )]
    );
    assert_eq!(
        rels("東側鐘聲驟響。与此同時，西側烽火也沖天而起。"),
        [rel(
            K::Parallel,
            L::Sentence,
            "与此同時",
            None,
            "東側鐘聲驟響。",
            "与此同時，西側烽火也沖天而起。"
        )]
    );
    assert_eq!(
        rels("前堂賓客散去。而與此同時，後院燈火卻亮了起來。"),
        [rel(
            K::Parallel,
            L::Sentence,
            "而與此同時",
            None,
            "前堂賓客散去。",
            "而與此同時，後院燈火卻亮了起來。"
        )]
    );
    assert_eq!(
        rels("木門轟然合攏，幾乎與此同時，石室四角亮起微光。"),
        [rel(
            K::Parallel,
            L::Clause,
            "幾乎與此同時",
            None,
            "木門轟然合攏",
            "石室四角亮起微光。"
        )]
    );
}

#[test]
fn ja_proper_name_does_not_match_zh_connectors() {
    let text = "若菜貞爾は東京へ向かった。彼は静かに笑った。";
    assert_eq!(rels(text), []);
    assert_eq!(rel_unknown(text), Vec::<String>::new());
}
