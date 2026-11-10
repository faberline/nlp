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
        doc.unknown
            .iter()
            .any(|u| u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage)
    );
    assert!(rels(text).is_empty());
    assert_eq!(rel_unknown(text), ["But"]);
    assert!(rels("We ran it. これはテストです。").is_empty());
    assert!(rel_unknown("We ran it. これはテストです。").is_empty());

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
];

/// Brackets and quotes that may stand before a connector inside a clause.
const OPENING_MARKS: &str = "「『“‘（([\"'";

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
                    .all(|ch| ch.is_whitespace() || OPENING_MARKS.contains(*ch));
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
