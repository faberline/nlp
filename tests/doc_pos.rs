use faberline_nlp::Span;
use faberline_nlp::doc::{Document, Lang, Options, Reason, Upos, analyze_document};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const GOLD_PATH: &str = "tests/data/pos_gold.json";
const RECEIPT_PATH: &str = "docs/receipts/pos-2026-11-10.json";
const POS_SOURCE: &str = include_str!("../src/doc/pos.rs");
const UPOS_NAMES: &[&str] = &[
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
    "PUNCT", "SCONJ", "SYM", "VERB", "X",
];

/// Texts of the rule-based tests. Every one is also checked by `invariants`.
const CASES: &[&str] = &[
    "She didn't open the door.",
    "We propose a well-known method [1] and test it.",
    "lol that's so good 😋",
    "They'll go.",
    "Dr. Lee left.",
    "It cost 1,000 dollars, or 3.5 more.",
    "他沒有回頭，只是把門關上了。",
    "我們提出了新方法，並且測試。",
    "哇，今天真的很好玩！",
    "我們把資料庫放在記憶體裡。",
    "我們 test 了 3 次。",
    "see https://example.com now",
    "三",
    "one two",
    "こんにちは。",
    "# Title here\n\nThe cat sat.",
    "I'm sure I'll go. I've done it, I'd say.",
    "Wait... what?!! No!!!",
    "The U.S. army left.",
    "pages 3-4 and COVID-19 and well\u{2010}known",
    "二〇二六年三月",
    "It\u{2019}s fine, they\u{2019}ll come and we\u{2019}re ok.",
    "Run `x = 1` now.",
];

fn slice(text: &str, s: Span) -> String {
    text.chars()
        .skip(s.start_char)
        .take(s.end_char - s.start_char)
        .collect()
}

fn analyze(text: &str) -> Document {
    analyze_document(text, &Options::default())
}

fn pos_reason(doc: &Document, span: Span) -> Option<Reason> {
    doc.unknown
        .iter()
        .find(|u| u.layer == "pos" && u.span == span)
        .map(|u| u.reason)
}

/// Each token as `text/TAG`: the UPOS, `A` when ambiguous, `M` when no model decides it.
fn show(text: &str) -> Vec<String> {
    let doc = analyze(text);
    doc.tokens
        .iter()
        .map(|t| {
            let tag = match (t.upos, pos_reason(&doc, t.span)) {
                (Some(upos), _) => format!("{upos:?}").to_uppercase(),
                (None, Some(Reason::Ambiguous)) => "A".to_owned(),
                (None, Some(Reason::NoModel)) => "M".to_owned(),
                (None, other) => panic!(
                    "token {:?} has no tag and reason {other:?}",
                    slice(text, t.span)
                ),
            };
            format!("{}/{tag}", slice(text, t.span))
        })
        .collect()
}

/// The `pos` unknown entries as (token text, reason), in order.
fn pos_unknown(text: &str) -> Vec<(String, Reason)> {
    let doc = analyze(text);
    doc.unknown
        .iter()
        .filter(|u| u.layer == "pos")
        .map(|u| (slice(text, u.span), u.reason))
        .collect()
}

#[test]
fn en_novel() {
    let t = "She didn't open the door.";
    assert_eq!(
        show(t),
        [
            "She/PRON", "did/A", "n't/PART", "open/M", "the/DET", "door/M", "./PUNCT"
        ]
    );
    let doc = analyze(t);
    assert_eq!(
        doc.tokens[1].span,
        Span {
            start_char: 4,
            end_char: 7
        }
    );
    assert_eq!(
        doc.tokens[2].span,
        Span {
            start_char: 7,
            end_char: 10
        }
    );
}

#[test]
fn en_paper() {
    assert_eq!(
        show("We propose a well-known method [1] and test it."),
        [
            "We/PRON",
            "propose/M",
            "a/DET",
            "well-known/M",
            "method/M",
            "[1]/X",
            "and/CCONJ",
            "test/M",
            "it/PRON",
            "./PUNCT",
        ]
    );
}

#[test]
fn en_casual() {
    assert_eq!(
        show("lol that's so good 😋"),
        ["lol/M", "that/A", "'s/A", "so/A", "good/M", "😋/SYM"]
    );
}

#[test]
fn en_clitic_abbrev_number() {
    assert_eq!(
        show("They'll go."),
        ["They/PRON", "'ll/AUX", "go/M", "./PUNCT"]
    );
    assert_eq!(
        show("Dr. Lee left."),
        ["Dr./M", "Lee/M", "left/M", "./PUNCT"]
    );
    assert_eq!(
        show("It cost 1,000 dollars, or 3.5 more."),
        [
            "It/PRON",
            "cost/M",
            "1,000/NUM",
            "dollars/M",
            ",/PUNCT",
            "or/CCONJ",
            "3.5/NUM",
            "more/M",
            "./PUNCT",
        ]
    );
}

#[test]
fn zh_novel() {
    assert_eq!(
        show("他沒有回頭，只是把門關上了。"),
        [
            "他/PRON",
            "沒/A",
            "有/VERB",
            "回頭/M",
            "，/PUNCT",
            "只是/M",
            "把/ADP",
            "門關/M",
            "上/M",
            "了/PART",
            "。/PUNCT",
        ]
    );
}

#[test]
fn zh_paper() {
    assert_eq!(
        show("我們提出了新方法，並且測試。"),
        [
            "我們/PRON",
            "提出/M",
            "了/PART",
            "新/M",
            "方法/M",
            "，/PUNCT",
            "並且/CCONJ",
            "測試/M",
            "。/PUNCT",
        ]
    );
}

#[test]
fn zh_casual() {
    assert_eq!(
        show("哇，今天真的很好玩！"),
        [
            "哇/INTJ",
            "，/PUNCT",
            "今天/M",
            "真的/M",
            "很/ADV",
            "好玩/M",
            "！/PUNCT",
        ]
    );
}

#[test]
fn zh_taiwan_terms() {
    assert_eq!(
        show("我們把資料庫放在記憶體裡。"),
        [
            "我們/PRON",
            "把/ADP",
            "資料庫/M",
            "放在/M",
            "記憶體/M",
            "裡/M",
            "。/PUNCT",
        ]
    );
}

#[test]
fn mixed_zh_en() {
    let t = "我們 test 了 3 次。";
    assert_eq!(
        show(t),
        [
            "我們/PRON",
            "test/M",
            "了/PART",
            "3/NUM",
            "次/M",
            "。/PUNCT",
        ]
    );
    let doc = analyze(t);
    assert_eq!(doc.tokens[1].lang, Lang::En);
    assert_eq!(doc.tokens[3].lang, Lang::Und);
    assert_eq!(doc.tokens[0].lang, Lang::Zh);
}

#[test]
fn mixed_with_url() {
    assert_eq!(
        show("see https://example.com now"),
        ["see/M", "https://example.com/X", "now/ADV"]
    );
}

#[test]
fn numerals_and_words() {
    assert_eq!(show("三"), ["三/NUM"]);
    assert_eq!(show("one two"), ["one/NUM", "two/NUM"]);
}

#[test]
fn unknown_entries() {
    let t = "She didn't open the door.";
    assert_eq!(
        pos_unknown(t),
        [
            ("did".to_owned(), Reason::Ambiguous),
            ("open".to_owned(), Reason::NoModel),
            ("door".to_owned(), Reason::NoModel),
        ]
    );
    let doc = analyze(t);
    let keys: Vec<(usize, usize, String)> = doc
        .unknown
        .iter()
        .map(|u| (u.span.start_char, u.span.end_char, u.layer.clone()))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted);
}

#[test]
fn unsupported_sentence_has_no_tokens() {
    let t = "こんにちは。";
    let doc = analyze(t);
    assert!(doc.tokens.is_empty());
    assert!(
        doc.unknown
            .iter()
            .any(|u| u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage)
    );
    assert!(doc.unknown.iter().all(|u| u.layer != "pos"));
}

#[test]
fn block_spans_have_no_tokens() {
    let t = "# Title here\n\nThe cat sat.";
    assert_eq!(show(t), ["The/DET", "cat/M", "sat/M", "./PUNCT"]);
    let doc = analyze(t);
    assert!(doc.tokens.iter().all(|tok| tok.span.start_char >= 14));
}

#[test]
fn clitic_after_one_letter_stem() {
    assert_eq!(
        show("I'm sure I'll go. I've done it, I'd say."),
        [
            "I/PRON", "'m/AUX", "sure/M", "I/PRON", "'ll/AUX", "go/M", "./PUNCT", "I/PRON",
            "'ve/AUX", "done/M", "it/PRON", ",/PUNCT", "I/PRON", "'d/A", "say/M", "./PUNCT",
        ]
    );
}

#[test]
fn repeated_marks_are_one_token() {
    assert_eq!(
        show("Wait... what?!! No!!!"),
        [
            "Wait/M",
            ".../PUNCT",
            "what/A",
            "?/PUNCT",
            "!!/PUNCT",
            "No/A",
            "!!!/PUNCT",
        ]
    );
    // jieba owns the Han words, so only the full-width marks are pinned.
    let shown = show("好！！！真的嗎？？？");
    assert!(shown.contains(&"！！！/PUNCT".to_owned()), "{shown:?}");
    assert!(shown.contains(&"？？？/PUNCT".to_owned()), "{shown:?}");
    assert_eq!(show("a --- b"), ["a/DET", "---/PUNCT", "b/M"]);
}

#[test]
fn abbreviation_with_inner_dot_takes_dot() {
    assert_eq!(
        show("The U.S. army left."),
        ["The/DET", "U.S./M", "army/M", "left/M", "./PUNCT"]
    );
    assert_eq!(
        show("The U.K. army left."),
        ["The/DET", "U.K./M", "army/M", "left/M", "./PUNCT"]
    );
}

#[test]
fn dot_run_after_word_is_not_swallowed() {
    // `No` and `Mr` are listed abbreviations: they take ONE dot, the rest is a run.
    assert_eq!(
        show("No... I don't think so."),
        [
            "No./M", "../PUNCT", "I/PRON", "do/A", "n't/PART", "think/M", "so/A", "./PUNCT"
        ]
    );
    assert_eq!(show("Mr... Smith"), ["Mr./M", "../PUNCT", "Smith/M"]);
    assert_eq!(show("I... go"), ["I./M", "../PUNCT", "go/M"]);
    assert_eq!(
        show("Hello... world"),
        ["Hello/INTJ", ".../PUNCT", "world/M"]
    );
}

#[test]
fn hyphen_needs_a_letter_on_both_sides() {
    assert_eq!(show("pages 3-4"), ["pages/M", "3/NUM", "-/PUNCT", "4/NUM"]);
    assert_eq!(show("COVID-19"), ["COVID/M", "-/PUNCT", "19/NUM"]);
    assert_eq!(show("well\u{2010}known"), ["well\u{2010}known/M"]);
}

#[test]
fn han_set_includes_ideographic_marks() {
    // 〇 (U+3007) and 々 (U+3005) are Han for splitting and for the token language.
    let doc = analyze("二〇二六年");
    let zero = doc
        .tokens
        .iter()
        .find(|t| slice("二〇二六年", t.span).contains('〇'))
        .expect("a token holds 〇");
    assert_eq!(zero.lang, Lang::Zh);
    // The whole Han run goes to jieba, so 〇 gets jieba's cut (pinned on purpose), not the
    // cut the old set gave it when 〇 was a separate run.
    assert_eq!(
        show("二〇二六年三月"),
        ["二/NUM", "〇/NUM", "二/NUM", "六年/M", "三月/M"]
    );
}

#[test]
fn curly_apostrophe_clitics() {
    assert_eq!(
        show("It\u{2019}s fine, they\u{2019}ll come and we\u{2019}re ok."),
        [
            "It/PRON",
            "\u{2019}s/A",
            "fine/M",
            ",/PUNCT",
            "they/PRON",
            "\u{2019}ll/AUX",
            "come/M",
            "and/CCONJ",
            "we/PRON",
            "\u{2019}re/AUX",
            "ok/M",
            "./PUNCT",
        ]
    );
    assert_eq!(
        show("Don\u{2019}t stop."),
        ["Do/A", "n\u{2019}t/PART", "stop/M", "./PUNCT"]
    );
}

#[test]
fn token_lang_follows_the_rule() {
    // A token with a Han char is zh even when it also has a Latin letter.
    let doc = analyze("Run `臺x` now.");
    let code = doc
        .tokens
        .iter()
        .find(|t| slice("Run `臺x` now.", t.span) == "`臺x`")
        .expect("code token");
    assert_eq!(code.lang, Lang::Zh);
    let doc = analyze("3 café");
    assert_eq!(doc.tokens[0].lang, Lang::Und);
    assert_eq!(doc.tokens[1].lang, Lang::En);
}

#[test]
fn inline_code_with_space_is_one_token() {
    assert_eq!(
        show("Run `x = 1` now."),
        ["Run/M", "`x = 1`/X", "now/ADV", "./PUNCT"]
    );
}

#[test]
fn fenced_code_block_is_never_a_token() {
    let t = "```\nlet x = 1;\n```\nThe cat sat.";
    let doc = analyze(t);
    assert!(doc.tokens.iter().all(|tok| tok.span.start_char >= 19));
    assert_eq!(show(t), ["The/DET", "cat/M", "sat/M", "./PUNCT"]);
}

/// Gold sentences of `tests/data/pos_gold.json`.
#[derive(Deserialize)]
struct Gold {
    genre: String,
    lang: String,
    text: String,
    tokens: Vec<(String, String)>,
}

fn manifest(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn gold() -> Vec<Gold> {
    let raw = fs::read_to_string(manifest(GOLD_PATH)).expect("gold file");
    serde_json::from_str(&raw).expect("gold file parses")
}

/// Span of each gold token in its text. Whitespace between tokens is skipped, and every gold
/// token must match the text exactly.
fn align(g: &Gold) -> Vec<Span> {
    let chars: Vec<char> = g.text.chars().collect();
    let mut at = 0;
    let mut out = Vec::new();
    for (tok, _) in &g.tokens {
        while at < chars.len() && chars[at].is_whitespace() {
            at += 1;
        }
        let n = tok.chars().count();
        assert!(
            chars
                .get(at..at + n)
                .is_some_and(|s| s.iter().copied().eq(tok.chars())),
            "gold token {tok:?} is not at char {at} of {:?}",
            g.text
        );
        out.push(Span {
            start_char: at,
            end_char: at + n,
        });
        at += n;
    }
    while at < chars.len() && chars[at].is_whitespace() {
        at += 1;
    }
    assert_eq!(
        at,
        chars.len(),
        "gold text {:?} has untokenized chars",
        g.text
    );
    out
}

#[test]
fn invariants() {
    let gold_texts: Vec<String> = gold().into_iter().map(|g| g.text).collect();
    for t in CASES
        .iter()
        .copied()
        .chain(gold_texts.iter().map(String::as_str))
    {
        check_invariants(t);
    }
}

fn check_invariants(text: &str) {
    let doc = analyze(text);
    assert_eq!(
        serde_json::to_string(&doc).unwrap(),
        serde_json::to_string(&analyze(text)).unwrap(),
        "output differs between two runs for {text:?}"
    );
    let chars: Vec<char> = text.chars().collect();
    let unsupported: Vec<Span> = doc
        .unknown
        .iter()
        .filter(|u| u.reason == Reason::UnsupportedLanguage)
        .map(|u| u.span)
        .collect();

    for w in doc.tokens.windows(2) {
        assert!(
            w[0].span.end_char <= w[1].span.start_char,
            "tokens overlap or are unsorted in {text:?}"
        );
    }
    for t in &doc.tokens {
        let s = doc.segmentation.sentences[t.sentence].span;
        assert!(t.span.start_char >= s.start_char && t.span.end_char <= s.end_char);
        assert!(t.span.end_char > t.span.start_char);
        // An inline opaque span is one token even when it holds whitespace.
        let opaque = doc.segmentation.opaque.iter().any(|o| o.span == t.span);
        assert!(
            opaque || !slice(text, t.span).chars().any(char::is_whitespace),
            "token {:?} holds whitespace in {text:?}",
            slice(text, t.span)
        );
    }
    for (i, s) in doc.segmentation.sentences.iter().enumerate() {
        if unsupported.contains(&s.span) {
            assert!(doc.tokens.iter().all(|t| t.sentence != i));
            continue;
        }
        for (k, ch) in chars
            .iter()
            .enumerate()
            .take(s.span.end_char)
            .skip(s.span.start_char)
        {
            if ch.is_whitespace() {
                continue;
            }
            let n = doc
                .tokens
                .iter()
                .filter(|t| t.sentence == i && t.span.start_char <= k && k < t.span.end_char)
                .count();
            assert_eq!(n, 1, "char {k} of {text:?} is in {n} tokens");
        }
    }
    for o in &doc.segmentation.opaque {
        let inside =
            doc.segmentation.sentences.iter().any(|s| {
                s.span.start_char <= o.span.start_char && o.span.end_char <= s.span.end_char
            });
        if inside {
            assert!(
                doc.tokens.iter().any(|t| t.span == o.span),
                "opaque span {:?} is not one token in {text:?}",
                slice(text, o.span)
            );
        }
    }

    let undecided: Vec<Span> = doc
        .tokens
        .iter()
        .filter(|t| t.upos.is_none())
        .map(|t| t.span)
        .collect();
    let pos_spans: Vec<Span> = doc
        .unknown
        .iter()
        .filter(|u| u.layer == "pos")
        .map(|u| u.span)
        .collect();
    assert_eq!(
        undecided, pos_spans,
        "pos unknowns differ from undecided tokens"
    );

    let keys: Vec<(usize, usize, &str)> = doc
        .unknown
        .iter()
        .map(|u| (u.span.start_char, u.span.end_char, u.layer.as_str()))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "unknown is not sorted in {text:?}");
}

/// Lexicon words of one block of `src/doc/pos.rs`, by block name.
fn lexicon_blocks() -> Vec<(&'static str, Vec<&'static str>)> {
    let mut out: Vec<(&'static str, Vec<&'static str>)> = Vec::new();
    let mut block: Option<&'static str> = None;
    for line in POS_SOURCE.lines() {
        let t = line.trim();
        if t.starts_with("const ") {
            let name = t
                .split_whitespace()
                .nth(1)
                .unwrap_or("")
                .trim_end_matches(':');
            block = ["EN_DECIDED", "ZH_DECIDED", "EN_AMBIGUOUS", "ZH_AMBIGUOUS"]
                .into_iter()
                .find(|n| *n == name);
            if let Some(name) = block {
                out.push((name, Vec::new()));
            }
        } else if t == "];" {
            block = None;
        } else if let Some(name) = block {
            let words = t
                .split('"')
                .skip(1)
                .step_by(2)
                .flat_map(str::split_whitespace);
            out.last_mut()
                .filter(|(n, _)| *n == name)
                .expect("block is open")
                .1
                .extend(words);
        }
    }
    out
}

fn lexicon_entries() -> usize {
    lexicon_blocks().iter().map(|(_, w)| w.len()).sum()
}

#[test]
fn lexicon_is_consistent() {
    let blocks = lexicon_blocks();
    assert_eq!(blocks.len(), 4, "all four lexicon blocks are present");
    let mut decided: Vec<&str> = Vec::new();
    let mut ambiguous: Vec<&str> = Vec::new();
    for (name, words) in &blocks {
        for w in words {
            assert!(!w.is_empty(), "empty key in {name}");
            if name.starts_with("EN_") {
                assert_eq!(*w, w.to_lowercase(), "English key {w:?} is not lowercase");
            }
        }
        if name.ends_with("DECIDED") {
            decided.extend(words);
        } else {
            ambiguous.extend(words);
        }
    }
    let mut seen = decided.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), decided.len(), "a decided word is listed twice");
    let mut seen = ambiguous.clone();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(
        seen.len(),
        ambiguous.len(),
        "an ambiguous word is listed twice"
    );
    for w in &ambiguous {
        assert!(!decided.contains(w), "{w:?} is both decided and ambiguous");
    }
}

fn upos_name(doc: &Document, i: usize) -> Option<String> {
    doc.tokens[i].upos.map(|u| format!("{u:?}").to_uppercase())
}

#[derive(Clone, Copy, Default)]
struct Counts {
    gold: usize,
    matched: usize,
    decided: usize,
    correct: usize,
}

impl Counts {
    fn boundary(&self) -> f64 {
        self.matched as f64 / self.gold as f64
    }
    fn coverage(&self) -> f64 {
        self.decided as f64 / self.matched as f64
    }
    fn precision(&self) -> f64 {
        self.correct as f64 / self.decided as f64
    }
}

/// Counts per language, and in total under the key `all`.
fn measure(gold: &[Gold]) -> HashMap<&'static str, Counts> {
    let mut out: HashMap<&'static str, Counts> = HashMap::new();
    for g in gold {
        let doc = analyze(&g.text);
        let program: HashMap<(usize, usize), usize> = doc
            .tokens
            .iter()
            .enumerate()
            .map(|(i, t)| ((t.span.start_char, t.span.end_char), i))
            .collect();
        let lang: &'static str = match g.lang.as_str() {
            "zh" => "zh",
            "en" => "en",
            "mixed" => "mixed",
            other => panic!("unknown gold lang {other:?}"),
        };
        for (span, (_, tag)) in align(g).into_iter().zip(&g.tokens) {
            for key in [lang, "all"] {
                let c = out.entry(key).or_default();
                c.gold += 1;
                if let Some(&i) = program.get(&(span.start_char, span.end_char)) {
                    c.matched += 1;
                    if let Some(name) = upos_name(&doc, i) {
                        c.decided += 1;
                        if &name == tag {
                            c.correct += 1;
                        }
                    }
                }
            }
        }
    }
    out
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

#[test]
fn gold_file_is_well_formed() {
    let gold = gold();
    assert_eq!(gold.len(), 63);
    for g in &gold {
        assert!(["novel", "paper", "casual"].contains(&g.genre.as_str()));
        assert!(["zh", "en", "mixed"].contains(&g.lang.as_str()));
        for (tok, tag) in &g.tokens {
            assert!(!tok.is_empty());
            assert!(UPOS_NAMES.contains(&tag.as_str()), "bad tag {tag:?}");
        }
        align(g);
    }
}

#[test]
fn pos_receipt() {
    let gold = gold();
    let counts = measure(&gold);
    let total = counts["all"];
    assert!(
        total.precision() >= 0.95,
        "precision {} is below 0.95",
        total.precision()
    );

    if std::env::var("NLP_WRITE_RECEIPT").as_deref() != Ok("1") {
        return;
    }
    let per_lang = |key: &str| {
        let c = counts[key];
        json!({
            "gold_tokens": c.gold,
            "token_boundary_match": round4(c.boundary()),
            "decided": c.decided,
            "coverage": round4(c.coverage()),
            "precision": round4(c.precision()),
        })
    };
    let raw = fs::read(manifest(GOLD_PATH)).expect("gold file");
    let receipt = json!({
        "layer": "pos",
        "date": "2026-11-10",
        "gold_file": GOLD_PATH,
        "gold_sha256": Sha256::digest(&raw).iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "gold_sentences": gold.len(),
        "gold_tokens": total.gold,
        "token_boundary_match": round4(total.boundary()),
        "decided": total.decided,
        "coverage": round4(total.coverage()),
        "precision": round4(total.precision()),
        "per_lang": {
            "zh": per_lang("zh"),
            "en": per_lang("en"),
            "mixed": per_lang("mixed"),
        },
        "lexicon_entries": lexicon_entries(),
        "notes": [
            "The gold set was written by the same author as the lexicon, so it is not independent. The independent gold set comes with issue 28.",
            "UD is used as the label scheme only. No UD treebank text was used, because of its CC BY-SA licence.",
            "Measured tokens are only those whose span equals a gold span. Tokens with no tag are counted in token_boundary_match and coverage, not in precision.",
            "Chinese token_boundary_match is the lowest of the three languages, because jieba word boundaries often differ from the gold segmentation; the Taiwan term list runs before jieba.",
            "jieba boundary errors seen so far: 門關 in 把門關上了 (jieba joins it, but 門 and 關 are separate words) and 臺灣讀 (reported in review; not pinned by a test).",
            "Precision is the share of decided, matched tokens whose tag equals the gold tag."
        ],
    });
    let path = manifest(RECEIPT_PATH);
    fs::create_dir_all(path.parent().expect("receipt dir")).expect("create receipt dir");
    let mut text = serde_json::to_string_pretty(&receipt).expect("receipt json");
    text.push('\n');
    fs::write(&path, text).expect("write receipt");
}

#[test]
fn simplified_closed_class_and_classical_pronouns() {
    assert_eq!(
        show("我们已经从这里出发，因为虽然路远，但是他们应该能够马上到达吗？"),
        [
            "我们/PRON",
            "已经/ADV",
            "从/ADP",
            "这里/PRON",
            "出发/M",
            "，/PUNCT",
            "因为/SCONJ",
            "虽然/SCONJ",
            "路/M",
            "远/M",
            "，/PUNCT",
            "但是/CCONJ",
            "他们/PRON",
            "应该/AUX",
            "能够/AUX",
            "马上/ADV",
            "到达/M",
            "吗/PART",
            "？/PUNCT",
        ]
    );

    assert_eq!(
        show("尽管刚刚根据新方法并且为了大家，你们、她们、它们、咱们和谁在那里曾经必须小心。"),
        [
            "尽管/SCONJ",
            "刚刚/ADV",
            "根据/ADP",
            "新/M",
            "方法/M",
            "并且/CCONJ",
            "为了/ADP",
            "大家/PRON",
            "，/PUNCT",
            "你们/PRON",
            "、/PUNCT",
            "她们/PRON",
            "、/PUNCT",
            "它们/PRON",
            "、/PUNCT",
            "咱们/PRON",
            "和/A",
            "谁/PRON",
            "在/A",
            "那里/PRON",
            "曾经/ADV",
            "必须/AUX",
            "小心/M",
            "。/PUNCT",
        ]
    );

    // Simplified ambiguous closed-class entries emit Reason::Ambiguous (`A`).
    let ambig_doc =
        analyze("这、这个、那个、什么、对、与、给、于是、会、没、没有、还、几、着、过。");
    for expected in [
        "这", "这个", "那个", "什么", "对", "与", "给", "于是", "会", "没", "没有", "还", "几",
        "着", "过",
    ] {
        let tok = ambig_doc
            .tokens
            .iter()
            .find(|t| {
                slice(
                    "这、这个、那个、什么、对、与、给、于是、会、没、没有、还、几、着、过。",
                    t.span,
                ) == expected
            })
            .unwrap_or_else(|| panic!("missing token {expected}"));
        assert_eq!(tok.upos, None, "{expected} should be undecided");
        assert_eq!(
            pos_reason(&ambig_doc, tok.span),
            Some(Reason::Ambiguous),
            "{expected} should be Ambiguous"
        );
    }

    // Classical first-person self-pronouns decide Upos::Pron; `還在下雨` keeps `在` + `下雨`.
    for pron in ["在下", "妾身", "老夫", "老身", "貧道", "贫道", "本座"] {
        let s = format!("{pron}告辭。");
        let shown = show(&s);
        assert_eq!(shown[0], format!("{pron}/PRON"));
    }
    let rain = show("外头还在下雨。");
    assert!(rain.contains(&"在/A".to_owned()), "{rain:?}");
    assert!(rain.contains(&"下雨/M".to_owned()), "{rain:?}");
}

#[test]
fn normalize_traditional_t2s_degluing_and_lexicon() {
    let opts = Options {
        normalize_traditional: true,
        lexicon: vec!["青竹木劍".to_owned(), "白露灵草".to_owned()],
        names: vec!["白衣老者".to_owned()],
        ..Options::default()
    };

    // Coverb/demonstrative de-gluing + Traditional->Simplified projection + lexicon/names protection.
    let t1 =
        "當白衣老者將其與青竹木劍收起，對此沒有多言，向大漢點頭，這讓眾人連忙後退，採摘白露靈草。";
    let doc1 = analyze_document(t1, &opts);
    let toks1: Vec<String> = doc1.tokens.iter().map(|t| slice(t1, t.span)).collect();
    for expected in [
        "當",
        "白衣老者",
        "將",
        "其",
        "與",
        "青竹木劍",
        "收起",
        "對",
        "此",
        "沒有",
        "向",
        "大漢",
        "這",
        "讓",
        "眾人",
        "連忙",
        "後退",
        "白露靈草",
    ] {
        assert!(
            toks1.iter().any(|x| x == expected),
            "expected token {expected:?} in {toks1:?}"
        );
    }

    // Contraction splits (`將此`, `將那`) and lexicalized whitelist preservation.
    let t2 = "將此收好，將那放回。此時將軍與對方在這裡談話，那天從前輩處取回儲物袋與築基丹，神識與靈力如往常一般溫和。";
    let doc2 = analyze_document(t2, &opts);
    let toks2: Vec<String> = doc2.tokens.iter().map(|t| slice(t2, t.span)).collect();
    for expected in [
        "將",
        "此",
        "將",
        "那",
        "此時",
        "將軍",
        "對方",
        "這裡",
        "那天",
        "前輩",
        "儲物袋",
        "築基丹",
        "神識",
        "靈力",
        "往常",
        "溫和",
    ] {
        assert!(
            toks2.iter().any(|x| x == expected),
            "expected token {expected:?} in {toks2:?}"
        );
    }

    // Multi-byte UTF-8 safety: 4-byte CJK Extension B char (`\u{20000}`) alongside 3-byte Traditional chars.
    let t3 = "\u{20000}老者將其放入儲物袋。";
    let doc3 = analyze_document(t3, &opts);
    let toks3: Vec<String> = doc3.tokens.iter().map(|t| slice(t3, t.span)).collect();
    assert_eq!(toks3[0], "\u{20000}");
    assert!(toks3.contains(&"老者".to_owned()));
    assert!(toks3.contains(&"將".to_owned()));
    assert!(toks3.contains(&"其".to_owned()));
    assert!(toks3.contains(&"儲物袋".to_owned()));
}

#[test]
fn ws2_ws3_v3_infix_projection_degluing_and_decided_lexicon() {
    let opts = Options {
        normalize_traditional: true,
        ..Options::default()
    };

    // 1. Scraper anti-censorship infix projection + POS lookup via `strip_scraper_infix`.
    let t_infix = "那名少丅女與長堊老終丅於一丅齊走入傳丅送陣，一丅旦啟程便需丅要護住肉冇身離開大灬陸，免被吞灬噬，老堊身也沒丅有多言。";
    let doc_infix = analyze_document(t_infix, &opts);
    for tok in &doc_infix.tokens {
        assert!(
            tok.span.start_char < tok.span.end_char && tok.span.end_char <= doc_infix.text_chars
        );
    }
    let toks_infix: Vec<(String, Option<Upos>)> = doc_infix
        .tokens
        .iter()
        .map(|t| (slice(t_infix, t.span), t.upos))
        .collect();
    for expected in [
        "少丅女",
        "長堊老",
        "終丅於",
        "一丅齊",
        "傳丅送",
        "一丅旦",
        "需丅要",
        "肉冇身",
        "大灬陸",
        "吞灬噬",
        "老堊身",
        "沒丅有",
    ] {
        assert!(
            toks_infix.iter().any(|(s, _)| s == expected),
            "expected infix-projected token {expected:?} in {toks_infix:?}"
        );
    }
    let find_upos = |surf: &str| {
        toks_infix
            .iter()
            .find(|(s, _)| s == surf)
            .unwrap_or_else(|| panic!("missing {surf}"))
            .1
    };
    assert_eq!(find_upos("終丅於"), Some(Upos::Adv));
    assert_eq!(find_upos("一丅齊"), Some(Upos::Adv));
    assert_eq!(find_upos("一丅旦"), Some(Upos::Sconj));
    assert_eq!(find_upos("老堊身"), Some(Upos::Pron));
    let meiyou_span = doc_infix
        .tokens
        .iter()
        .find(|t| slice(t_infix, t.span) == "沒丅有")
        .unwrap()
        .span;
    assert_eq!(pos_reason(&doc_infix, meiyou_span), Some(Reason::Ambiguous));

    // 2. `其 + verb` and `中 + verb/prep` de-gluing vs lexicalized whitelists.
    let t_deglue = concat!(
        "青衫老者將其放下、將其收起、將其滅去、將其斬斷、將其拿住、將其吞下。",
        "其實其中其他同伴與其它修士其次將其餘木盒在期間期待地放回其間。",
        "他從袋中探出手，在山谷中方才停步，心中狂喜，目中激射精光，在石室中專心打坐，於陣中為眾人護法。",
        "中國中間的中心石台有中年旅人中途拿出中品丹藥，在中期與中階修士於中央中原會合，未曾中毒、中計或中斷，保持中等與中立的中樞秩序。"
    );
    let doc_deglue = analyze_document(t_deglue, &opts);
    for tok in &doc_deglue.tokens {
        assert!(
            tok.span.start_char < tok.span.end_char && tok.span.end_char <= doc_deglue.text_chars
        );
    }
    let surf_deglue: Vec<String> = doc_deglue
        .tokens
        .iter()
        .map(|t| slice(t_deglue, t.span))
        .collect();
    for forbidden in [
        "其放", "其收", "其滅", "其斬", "其拿", "其吞", "中探", "中方", "中狂", "中激", "中專",
        "中為",
    ] {
        assert!(
            !surf_deglue.iter().any(|s| s == forbidden),
            "false merge {forbidden:?} must be split: {surf_deglue:?}"
        );
    }
    for preserved in [
        "其實", "其中", "其他", "其它", "其次", "其餘", "期間", "期待", "其間", "中國", "中間",
        "中心", "中年", "中途", "中品", "中期", "中階", "中央", "中原", "中毒", "中計", "中斷",
        "中等", "中立", "中樞",
    ] {
        assert!(
            surf_deglue.iter().any(|s| s == preserved),
            "lexicalized word {preserved:?} must stay intact: {surf_deglue:?}"
        );
    }

    // 3. New `ZH_DECIDED` closed-class words (`終於`/`终于`/`終于`, `一旦`, `一齊`/`一齐`, `除了`, `老身`).
    let decided_sent = "除了老身之外，眾人終於、终于與終于一齊和一齐退後，一旦石門開啟便進入。";
    let doc_dec = analyze_document(decided_sent, &opts);
    for tok in &doc_dec.tokens {
        assert!(tok.span.start_char < tok.span.end_char && tok.span.end_char <= doc_dec.text_chars);
    }
    check_invariants("除了老身之外，众人终于一齐退后，一旦石门开启便进入。");
    for (word, expected_upos) in [
        ("除了", Upos::Adp),
        ("老身", Upos::Pron),
        ("終於", Upos::Adv),
        ("终于", Upos::Adv),
        ("終于", Upos::Adv),
        ("一齊", Upos::Adv),
        ("一齐", Upos::Adv),
        ("一旦", Upos::Sconj),
    ] {
        let tok = doc_dec
            .tokens
            .iter()
            .find(|t| slice(decided_sent, t.span) == word)
            .unwrap_or_else(|| panic!("missing token {word}"));
        assert_eq!(tok.upos, Some(expected_upos), "{word} upos mismatch");
    }
}
