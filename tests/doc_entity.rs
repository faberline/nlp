//! Layer 3 of `nlp.doc.v1`: entities. See `docs/doc-v1.md`.

use faberline_nlp::Span;
use faberline_nlp::doc::{Document, EntityKind, Options, analyze_document};
use serde_json::json;
use sha2::{Digest, Sha256};

type Row = (EntityKind, String, Option<String>, Option<String>);

fn slice(text: &str, s: Span) -> String {
    text.chars()
        .skip(s.start_char)
        .take(s.end_char - s.start_char)
        .collect()
}

fn analyze(text: &str) -> Document {
    analyze_document(text, &Options::default())
}

fn analyze_names(text: &str, names: &[&str]) -> Document {
    let options = Options {
        names: names.iter().map(|n| n.to_string()).collect(),
        ..Options::default()
    };
    analyze_document(text, &options)
}

/// (kind, span text, value, unit) of every entity.
fn ents(text: &str) -> Vec<Row> {
    rows(text, &analyze(text))
}

fn rows(text: &str, doc: &Document) -> Vec<Row> {
    doc.entities
        .iter()
        .map(|e| (e.kind, slice(text, e.span), e.value.clone(), e.unit.clone()))
        .collect()
}

/// Span text of every ambiguous unknown from layer 3.
fn ent_unknown(text: &str) -> Vec<String> {
    analyze(text)
        .unknown
        .iter()
        .filter(|u| u.layer == "entities")
        .map(|u| slice(text, u.span))
        .collect()
}

fn row(kind: EntityKind, span: &str, value: Option<&str>) -> Row {
    (kind, span.to_owned(), value.map(str::to_owned), None)
}

fn qty(span: &str, value: &str, unit: &str) -> Row {
    (
        EntityKind::Quantity,
        span.to_owned(),
        Some(value.to_owned()),
        Some(unit.to_owned()),
    )
}

fn d(span: &str, value: &str) -> Row {
    row(EntityKind::Date, span, Some(value))
}

fn n(span: &str, value: &str) -> Row {
    row(EntityKind::Number, span, Some(value))
}

fn t(span: &str, value: &str) -> Row {
    row(EntityKind::Time, span, Some(value))
}

fn r(span: &str) -> Row {
    row(EntityKind::RelativeDate, span, None)
}

/// Every text used by the unit tests, for the invariant checks.
const TEXTS: &[&str] = &[
    "The meeting is on Nov 10, 2026.",
    "She left on 10 November 2026.",
    "It was May 2026.",
    "We may go.",
    "Due 2026-11-10.",
    "We go on 5 May.",
    "Sept. 3rd was fine.",
    "會議在2026年11月10日舉行。",
    "民國115年3月5日出生。",
    "二〇二六年十一月十日",
    "十一月十日開會。",
    "明天再說。",
    "他工作了3年。",
    "It cost 1,000 dollars or 3.5 more.",
    "他買了3萬個蘋果。",
    "共三百二十一人。",
    "統一第三。",
    "We saw twenty-one cats and one hundred dogs.",
    "One should go.",
    "Chapter 1st edition",
    "這袋米重5公斤，要價NT$300，折扣50%。",
    "We ran 3 km in 20 minutes.",
    "Add 5m of rope.",
    "Add 5 m of rope.",
    "We meet at 3:30 pm.",
    "會議下午3點半開始。",
    "Meet at 15:30.",
    "Call me at 3:30.",
    "晚上12點見。",
    "see you at noon",
    "see u at 5pm",
    "at midnight",
    "Due 3/4.",
    "Due 13/4/2026.",
    "Due 3/4/2026.",
    "Due 4/13/2026.",
    "Code 2026-13-45.",
    "Mail a.b@example.com or see https://example.com now",
    "王小明 told Alice Lee hello.",
    "Alice Leeds left.",
    "明天下午3點見！",
    "我們 5 點 見 😋",
    "lol see u tmrw 5pm",
    "Use `x = 5` for 2 cats.",
    "See [1] and https://example.com/3 for 2 cats.",
    "# Plan 2026\n\nWe go on 5 May.",
    "こんにちは、3個です。",
    "안녕하세요, 2026년 11월 13일 3개입니다.",
    "Привет мир, 3.",
    "10 Nov. 2026 was fine.",
    "5 Sept. 2026 was fine.",
    "Due 0/5/2026.",
    "Due 2026/2/30.",
    "$",
    "NT$",
    "US$",
    "¥",
    "€",
    "£",
    "百分之",
    "Pay $",
    "It costs NT$",
    "漲了百分之",
    "Pay `x`$",
    "I owe you 10€",
    "I have two three cats.",
    "In nineteen eighty-four it rained.",
    "It was twenty twenty.",
    "They sold two hundred three hundred boxes.",
    "one two three",
    "Five thousand two thousand",
    "Five thousand two hundred thousand",
    "five hundred thousand two hundred thousand",
    "one million two hundred million",
    "Updated 9:45 Jan 2027.",
    "Call at 3:30 Feb 2026 please.",
];

#[test]
fn en_dates() {
    assert_eq!(
        ents("The meeting is on Nov 10, 2026."),
        vec![d("Nov 10, 2026", "2026-11-10")]
    );
    assert_eq!(
        ents("She left on 10 November 2026."),
        vec![d("10 November 2026", "2026-11-10")]
    );
    assert_eq!(ents("It was May 2026."), vec![d("May 2026", "2026-05")]);
    assert_eq!(ents("We may go."), vec![]);
    assert_eq!(ents("Due 2026-11-10."), vec![d("2026-11-10", "2026-11-10")]);
    assert_eq!(ents("We go on 5 May."), vec![d("5 May", "--05-05")]);
    assert_eq!(ents("Sept. 3rd was fine."), vec![d("Sept. 3rd", "--09-03")]);
    // The dot after an abbreviated month is part of the date, with the year.
    assert_eq!(
        ents("10 Nov. 2026 was fine."),
        vec![d("10 Nov. 2026", "2026-11-10")]
    );
    assert_eq!(
        ents("5 Sept. 2026 was fine."),
        vec![d("5 Sept. 2026", "2026-09-05")]
    );
}

#[test]
fn zh_dates() {
    assert_eq!(
        ents("會議在2026年11月10日舉行。"),
        vec![d("2026年11月10日", "2026-11-10")]
    );
    assert_eq!(
        ents("民國115年3月5日出生。"),
        vec![d("民國115年3月5日", "2026-03-05")]
    );
    assert_eq!(
        ents("二〇二六年十一月十日"),
        vec![d("二〇二六年十一月十日", "2026-11-10")]
    );
    assert_eq!(ents("十一月十日開會。"), vec![d("十一月十日", "--11-10")]);
    assert_eq!(ents("明天再說。"), vec![r("明天")]);
    assert_eq!(ents("他工作了3年。"), vec![n("3", "3")]);
}

#[test]
fn numbers() {
    assert_eq!(
        ents("It cost 1,000 dollars or 3.5 more."),
        vec![qty("1,000 dollars", "1000", "dollars"), n("3.5", "3.5"),]
    );
    assert_eq!(ents("他買了3萬個蘋果。"), vec![n("3萬", "30000")]);
    assert_eq!(ents("共三百二十一人。"), vec![n("三百二十一", "321")]);
    assert_eq!(ents("統一第三。"), vec![]);
    assert_eq!(
        ents("We saw twenty-one cats and one hundred dogs."),
        vec![n("twenty-one", "21"), n("one hundred", "100")]
    );
    assert_eq!(ents("One should go."), vec![]);
    assert_eq!(ent_unknown("One should go."), vec!["One"]);
    assert_eq!(ents("Chapter 1st edition"), vec![]);
}

#[test]
fn quantities() {
    assert_eq!(
        ents("這袋米重5公斤，要價NT$300，折扣50%。"),
        vec![
            qty("5公斤", "5", "公斤"),
            qty("NT$300", "300", "nt$"),
            qty("50%", "50", "%"),
        ]
    );
    assert_eq!(
        ents("We ran 3 km in 20 minutes."),
        vec![qty("3 km", "3", "km"), qty("20 minutes", "20", "minutes")]
    );
    assert_eq!(ents("Add 5m of rope."), vec![qty("5m", "5", "m")]);
    assert_eq!(ents("Add 5 m of rope."), vec![n("5", "5")]);
}

#[test]
fn times() {
    assert_eq!(ents("We meet at 3:30 pm."), vec![t("3:30 pm", "15:30")]);
    assert_eq!(ents("會議下午3點半開始。"), vec![t("下午3點半", "15:30")]);
    assert_eq!(ents("Meet at 15:30."), vec![t("15:30", "15:30")]);
    assert_eq!(ents("Call me at 3:30."), vec![]);
    assert_eq!(ent_unknown("Call me at 3:30."), vec!["3:30"]);
    assert_eq!(ents("晚上12點見。"), vec![]);
    assert_eq!(ent_unknown("晚上12點見。"), vec!["晚上12點"]);
    assert_eq!(ents("see you at noon"), vec![t("noon", "12:00")]);
    assert_eq!(ents("see u at 5pm"), vec![t("5pm", "17:00")]);
    assert_eq!(ents("at midnight"), vec![]);
    assert_eq!(ent_unknown("at midnight"), vec!["midnight"]);
}

#[test]
fn zh_number_words() {
    // A multiplier with nothing before it reads as 1.
    assert_eq!(ents("十個"), vec![n("十", "10")]);
    assert_eq!(ents("十五個"), vec![n("十五", "15")]);
    assert_eq!(ents("百個"), vec![n("百", "100")]);
    assert_eq!(ents("十萬個"), vec![n("十萬", "100000")]);
    assert_eq!(ents("一百萬個"), vec![n("一百萬", "1000000")]);
    assert_eq!(ents("三萬五千個"), vec![n("三萬五千", "35000")]);
    // A run that starts with 萬 or 億, or has an empty group before 萬 or 億, is not a number.
    assert_eq!(ents("萬個"), vec![]);
    assert_eq!(ents("億萬個"), vec![]);
    assert_eq!(ents("三億萬個"), vec![]);
    assert_eq!(ents("一億萬個"), vec![]);
    // Tens and units stay one number.
    assert_eq!(ents("五十二個"), vec![n("五十二", "52")]);
    assert_eq!(ents("一百零五個"), vec![n("一百零五", "105")]);
    assert_eq!(ents("一百二十個"), vec![n("一百二十", "120")]);
    // Two plain digits read digit by digit.
    assert_eq!(ents("二三個"), vec![n("二三", "23")]);
    // A second group that would merge into a wrong number is rejected, not merged.
    assert_eq!(ents("十二十個"), vec![]);
    assert_eq!(ents("三十四十個"), vec![]);
    assert_eq!(ents("五百六百個"), vec![]);
    assert_eq!(ents("二三十個"), vec![]);
    assert_eq!(ents("三十個"), vec![n("三十", "30")]);
}

#[test]
fn zh_big_group_order() {
    // 萬 and 億 groups must decrease, so a repeated or rising group is not a number.
    assert_eq!(ents("三萬四萬個"), vec![]);
    assert_eq!(ents("五千萬六千萬個"), vec![]);
    assert_eq!(ents("一萬二億個"), vec![]);
    assert_eq!(ents("三萬億個"), vec![]);
    // Valid decreasing groups still work.
    assert_eq!(ents("三億五千萬個"), vec![n("三億五千萬", "350000000")]);
    assert_eq!(ents("一億零五萬個"), vec![n("一億零五萬", "100050000")]);
    assert_eq!(ents("三萬五千個"), vec![n("三萬五千", "35000")]);
    assert_eq!(ents("十萬個"), vec![n("十萬", "100000")]);
    assert_eq!(ents("一百萬個"), vec![n("一百萬", "1000000")]);
    // 〇 counts as zero after a multiplier, like 零.
    assert_eq!(ents("一百〇五個"), vec![n("一百〇五", "105")]);
}

#[test]
fn zh_clock_and_date_start() {
    // A time or date never starts inside a numeral run or right after 第.
    assert_eq!(ents("指數上漲一百二十點。"), vec![]);
    assert_eq!(ent_unknown("指數上漲一百二十點。"), Vec::<String>::new());
    assert_eq!(ents("他得了三十五點。"), vec![]);
    assert_eq!(ent_unknown("他得了三十五點。"), Vec::<String>::new());
    assert_eq!(ents("第三點"), vec![]);
    assert_eq!(ent_unknown("第三點"), Vec::<String>::new());
    assert_eq!(ents("三十點"), vec![]);
    assert_eq!(ent_unknown("三十點"), Vec::<String>::new());
    assert_eq!(ents("三十一月五日"), vec![]);
    assert_eq!(ent_unknown("三十一月五日"), Vec::<String>::new());
    // Valid starts keep working.
    assert_eq!(ent_unknown("三點見。"), vec!["三點"]);
    assert_eq!(ents("下午三點見。"), vec![t("下午三點", "15:00")]);
    assert_eq!(ent_unknown("十點半見。"), vec!["十點半"]);
    assert_eq!(ents("下午十點半見。"), vec![t("下午十點半", "22:30")]);
    assert_eq!(ents("二十三點十分見。"), vec![t("二十三點十分", "23:10")]);
    assert_eq!(ents("十一月五日開會。"), vec![d("十一月五日", "--11-05")]);
    assert_eq!(ents("民國115年出生。"), vec![d("民國115年", "2026")]);
}

#[test]
fn zh_ordinal_prefix() {
    // 第 marks an ordinal, so the numeral and the numerals inside it are not numbers.
    assert_eq!(ents("第三十個"), vec![]);
    assert_eq!(ents("第一"), vec![]);
    assert_eq!(ents("第三十公斤"), vec![]);
}

#[test]
fn huge_numbers_do_not_panic() {
    let long = "9".repeat(45);
    let texts = [
        "民國18446744073709551615年出生。".to_owned(),
        "民國184467440737095516160年出生。".to_owned(),
        "三百億億億億億個".to_owned(),
        "三百億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億億個"
            .to_owned(),
        "九九九九九九九九九九九九九九九九九九九九九九九九九九九九九九九九萬億億億億億個".to_owned(),
        format!("有{long}個。"),
        format!("{long}萬億億億公斤。"),
        format!("{long}/12/2026 and {long} kg."),
        format!("{long}-11-10 {long}年{long}月{long}日"),
        "one billion billion billion billion billion billion billion billion billion billion"
            .to_owned(),
        "twenty thousand million billion billion hundred thousand".to_owned(),
        "一萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬萬個".to_owned(),
    ];
    for text in &texts {
        let doc = analyze(text);
        for e in &doc.entities {
            assert!(e.span.end_char <= text.chars().count(), "span in {text:?}");
        }
    }
    // The overflowing 民國 year gives no date; its digits are a plain number.
    assert_eq!(
        ents("民國18446744073709551615年出生。"),
        vec![n("18446744073709551615", "18446744073709551615")]
    );
    assert_eq!(ents("三百億億億億億個"), vec![]);
}

#[test]
fn ambiguous_numeric_dates() {
    assert_eq!(ents("Due 3/4."), vec![]);
    assert_eq!(ent_unknown("Due 3/4."), vec!["3/4"]);
    assert_eq!(ents("Due 13/4/2026."), vec![d("13/4/2026", "2026-04-13")]);
    assert_eq!(ents("Due 3/4/2026."), vec![]);
    assert_eq!(ent_unknown("Due 3/4/2026."), vec!["3/4/2026"]);
    assert_eq!(ents("Due 4/13/2026."), vec![d("4/13/2026", "2026-04-13")]);
    assert_eq!(
        ents("Code 2026-13-45."),
        vec![n("2026", "2026"), n("13", "13"), n("45", "45")]
    );
    // Neither reading of A/B/YYYY is a real date: a bad date, so its digits are numbers.
    assert_eq!(
        ents("Due 0/5/2026."),
        vec![n("0", "0"), n("5", "5"), n("2026", "2026")]
    );
    assert!(ent_unknown("Due 0/5/2026.").is_empty());
    assert_eq!(
        ents("Due 5/0/2026."),
        vec![n("5", "5"), n("0", "0"), n("2026", "2026")]
    );
    assert_eq!(
        ents("Due 12/0/2026."),
        vec![n("12", "12"), n("0", "0"), n("2026", "2026")]
    );
    assert!(ent_unknown("Due 12/0/2026.").is_empty());
    assert_eq!(
        ents("Due 0/0/2026."),
        vec![n("0", "0"), n("0", "0"), n("2026", "2026")]
    );
    assert_eq!(
        ents("Due 13/14/2026."),
        vec![n("13", "13"), n("14", "14"), n("2026", "2026")]
    );
    assert!(ent_unknown("Due 13/14/2026.").is_empty());
    // An invalid YYYY/MM/DD is a bad date as a whole: no A/B candidate is read inside it.
    assert_eq!(
        ents("Due 2026/2/30."),
        vec![n("2026", "2026"), n("2", "2"), n("30", "30")]
    );
    assert!(ent_unknown("Due 2026/2/30.").is_empty());
    assert_eq!(
        ents("Due 2026/02/30."),
        vec![n("2026", "2026"), n("02", "02"), n("30", "30")]
    );
    assert_eq!(
        ents("Due 2026/13/10."),
        vec![n("2026", "2026"), n("13", "13"), n("10", "10")]
    );
    assert!(ent_unknown("Due 2026/13/10.").is_empty());
}

#[test]
fn ambiguous_time_before_month() {
    // Only a candidate that lies fully inside a bad date is dropped: the time is an unknown.
    assert_eq!(ent_unknown("Updated 9:45 Jan 2027."), vec!["9:45"]);
    assert_eq!(ents("Updated 9:45 Jan 2027."), vec![n("2027", "2027")]);
    assert_eq!(ent_unknown("Call at 3:30 Feb 2026 please."), vec!["3:30"]);
    // The bad date `2026/2/30` still gives its digits as numbers and no unknown.
    assert_eq!(
        ents("Due 2026/2/30."),
        vec![n("2026", "2026"), n("2", "2"), n("30", "30")]
    );
    assert!(ent_unknown("Due 2026/2/30.").is_empty());
}

#[test]
fn url_email() {
    assert_eq!(
        ents("Mail a.b@example.com or see https://example.com now"),
        vec![
            row(EntityKind::Email, "a.b@example.com", None),
            row(EntityKind::Url, "https://example.com", None),
        ]
    );
}

#[test]
fn names() {
    let text = "王小明 told Alice Lee hello.";
    let doc = analyze_names(text, &["王小明", "Alice Lee"]);
    assert_eq!(
        rows(text, &doc),
        vec![
            row(EntityKind::Name, "王小明", None),
            row(EntityKind::Name, "Alice Lee", None),
        ]
    );
    let near = "Alice Leeds left.";
    assert!(rows(near, &analyze_names(near, &["Alice Lee"])).is_empty());
    assert!(ents(text).iter().all(|e| e.0 != EntityKind::Name));
    assert_eq!(serde_json::to_value(Options::default()).unwrap(), json!({}));
    let with = serde_json::to_value(Options {
        names: vec!["王小明".to_owned()],
        ..Options::default()
    })
    .unwrap();
    assert_eq!(with, json!({ "names": ["王小明"] }));
}

#[test]
fn casual_and_mixed() {
    assert_eq!(
        ents("明天下午3點見！"),
        vec![r("明天"), t("下午3點", "15:00")]
    );
    // The `H點` time form needs the hour attached to 點. A spaced 5 is a number (rule a).
    assert_eq!(ents("我們 5 點 見 😋"), vec![n("5", "5")]);
    assert!(ent_unknown("我們 5 點 見 😋").is_empty());
    assert_eq!(ents("我們15點見"), vec![t("15點", "15:00")]);
    assert_eq!(ents("lol see u tmrw 5pm"), vec![t("5pm", "17:00")]);
}

#[test]
fn opaque_is_skipped() {
    assert_eq!(ents("Use `x = 5` for 2 cats."), vec![n("2", "2")]);
    assert_eq!(
        ents("See [1] and https://example.com/3 for 2 cats."),
        vec![
            row(EntityKind::Url, "https://example.com/3", None),
            n("2", "2"),
        ]
    );
    assert_eq!(
        ents("# Plan 2026\n\nWe go on 5 May."),
        vec![d("5 May", "--05-05")]
    );
}

#[test]
fn unsupported_sentence() {
    use faberline_nlp::doc::{Lang, Reason};

    // Japanese and Korean sentences segment and tokenize in Layers 1–2 (`pos` unsupported on
    // lexical tokens), while Layer 3 skips them so no false-positive entities are emitted.
    for (text, expected_lang) in [
        ("こんにちは、3個です。", Lang::Ja),
        ("안녕하세요, 2026년 11월 13일 3개입니다.", Lang::Ko),
    ] {
        let doc = analyze(text);
        assert_eq!(doc.segmentation.sentences[0].lang, expected_lang);
        assert!(doc.entities.is_empty());
        assert!(
            doc.unknown
                .iter()
                .all(|u| u.layer != "entities" && u.layer != "segmentation")
        );
        assert!(
            doc.unknown
                .iter()
                .any(|u| u.layer == "pos" && u.reason == Reason::UnsupportedLanguage)
        );
    }

    // Truly unsupported script (Cyrillic) emits `segmentation` `UnsupportedLanguage` and no entities.
    let cyr = "Привет мир, 3.";
    let doc_cyr = analyze(cyr);
    assert!(doc_cyr.entities.is_empty());
    assert!(doc_cyr.unknown.iter().all(|u| u.layer != "entities"));
    assert!(
        doc_cyr
            .unknown
            .iter()
            .any(|u| u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage)
    );
}

#[test]
fn invariants() {
    let gold = gold_items();
    let mut texts: Vec<String> = TEXTS.iter().map(|s| s.to_string()).collect();
    texts.extend(gold.iter().map(|g| g.text.clone()));
    for text in &texts {
        let doc = analyze(text);
        let mut last_end = 0;
        for e in &doc.entities {
            assert!(e.span.start_char >= last_end, "overlap in {text:?}");
            last_end = e.span.end_char;
            let sentence = &doc.segmentation.sentences[e.sentence].span;
            assert!(
                sentence.start_char <= e.span.start_char && e.span.end_char <= sentence.end_char,
                "entity crosses a sentence in {text:?}"
            );
            for o in &doc.segmentation.opaque {
                let overlaps =
                    e.span.start_char < o.span.end_char && o.span.start_char < e.span.end_char;
                if overlaps {
                    assert!(
                        e.kind == EntityKind::Url && e.span == o.span,
                        "entity inside opaque span in {text:?}"
                    );
                }
            }
        }
        for u in doc.unknown.iter().filter(|u| u.layer == "entities") {
            assert_eq!(u.reason, faberline_nlp::doc::Reason::Ambiguous);
        }
        let again = analyze(text);
        assert_eq!(
            serde_json::to_string(&doc).unwrap(),
            serde_json::to_string(&again).unwrap()
        );
    }
}

struct Gold {
    text: String,
    entities: Vec<Row>,
    ambiguous: Vec<String>,
}

fn gold_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/entity_gold.json")
}

fn gold_items() -> Vec<Gold> {
    let raw = std::fs::read_to_string(gold_path()).expect("gold file");
    let items: Vec<serde_json::Value> = serde_json::from_str(&raw).expect("gold json");
    items
        .iter()
        .map(|item| {
            let entities = item["entities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| {
                    let kind: EntityKind = serde_json::from_value(e["kind"].clone()).expect("kind");
                    (
                        kind,
                        e["text"].as_str().unwrap().to_owned(),
                        e["value"].as_str().map(str::to_owned),
                        e["unit"].as_str().map(str::to_owned),
                    )
                })
                .collect();
            let ambiguous = item["ambiguous"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap().to_owned())
                .collect();
            Gold {
                text: item["text"].as_str().unwrap().to_owned(),
                entities,
                ambiguous,
            }
        })
        .collect()
}

fn kind_name(kind: EntityKind) -> String {
    serde_json::to_value(kind)
        .unwrap()
        .as_str()
        .unwrap()
        .to_owned()
}

#[test]
fn entity_receipt() {
    let raw = std::fs::read(gold_path()).expect("gold file");
    let gold = gold_items();
    let kinds = [
        EntityKind::Number,
        EntityKind::Date,
        EntityKind::RelativeDate,
        EntityKind::Time,
        EntityKind::Quantity,
        EntityKind::Url,
        EntityKind::Email,
        EntityKind::Name,
    ];
    // Per kind: (gold, predicted, matched).
    let mut stats = vec![(0usize, 0usize, 0usize); kinds.len()];
    let mut notes = vec![
        "The gold was written by the same author as the rules, so it is not an independent check."
            .to_owned(),
        "Regression check only: the gold is self-written, so high scores show the rules did not change, not accuracy on new text."
            .to_owned(),
        "Known false positives: `Call 0912 345 678 now.` gives three numbers (a phone number is not a number entity); `In 2026 we won 3 prizes.` gives 2026 as a number, not a year."
            .to_owned(),
        "Known misses: `The room is 5 m wide.` and `The package weighs 2 g.` give plain numbers, since `g` and `m` need no space; `He paid 5 pounds.` gives a number, since pounds is not a unit; `我买了三十四十個蘋果。` gives nothing, because a run with two tens groups is rejected."
            .to_owned(),
        "Names come only from `options.names`, so `Bob and Alice went home.` gives no name entity without them."
            .to_owned(),
        "Spec test 9: `我們 5 點 見 😋` now yields number \"5\", since a spaced 5 before 點 is a number, not a time."
            .to_owned(),
    ];
    let (mut gold_total, mut predicted_total, mut matched_total) = (0, 0, 0);
    let (mut amb_gold, mut amb_matched) = (0, 0);
    for item in &gold {
        let doc = analyze(&item.text);
        let predicted = rows(&item.text, &doc);
        let mut used = vec![false; predicted.len()];
        for g in &item.entities {
            let k = kinds.iter().position(|k| *k == g.0).unwrap();
            stats[k].0 += 1;
            gold_total += 1;
            let hit = predicted
                .iter()
                .enumerate()
                .position(|(i, p)| !used[i] && p == g);
            match hit {
                Some(i) => {
                    used[i] = true;
                    stats[k].2 += 1;
                    matched_total += 1;
                }
                None => notes.push(format!(
                    "miss: {} {:?} in {:?}",
                    kind_name(g.0),
                    g.1,
                    item.text
                )),
            }
        }
        for (i, p) in predicted.iter().enumerate() {
            let k = kinds.iter().position(|k| *k == p.0).unwrap();
            stats[k].1 += 1;
            predicted_total += 1;
            if !used[i] {
                notes.push(format!(
                    "false positive: {} {:?} in {:?}",
                    kind_name(p.0),
                    p.1,
                    item.text
                ));
            }
        }
        let unknown: Vec<String> = doc
            .unknown
            .iter()
            .filter(|u| u.layer == "entities")
            .map(|u| slice(&item.text, u.span))
            .collect();
        for a in &item.ambiguous {
            amb_gold += 1;
            if unknown.contains(a) {
                amb_matched += 1;
            } else {
                notes.push(format!("ambiguous miss: {a:?} in {:?}", item.text));
            }
        }
    }
    // None (JSON null) when the denominator is zero, so an empty kind is not scored as perfect.
    let rate = |num: usize, den: usize| (den > 0).then(|| num as f64 / den as f64);
    let precision = rate(matched_total, predicted_total);
    let recall = rate(matched_total, gold_total);
    let per_kind: serde_json::Map<String, serde_json::Value> = kinds
        .iter()
        .zip(&stats)
        .map(|(k, (g, p, m))| {
            // A kind with no gold items has no recall, and no precision either.
            let (precision, recall) = if *g == 0 {
                (None, None)
            } else {
                (rate(*m, *p), rate(*m, *g))
            };
            (
                kind_name(*k),
                json!({
                    "gold": g,
                    "predicted": p,
                    "matched": m,
                    "precision": precision,
                    "recall": recall,
                }),
            )
        })
        .collect();
    let receipt = json!({
        "layer": "entities",
        "date": "2026-11-10",
        "gold_file": "tests/data/entity_gold.json",
        "gold_sha256": Sha256::digest(&raw).iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "gold_sentences": gold.len(),
        "gold_entities": gold_total,
        "predicted": predicted_total,
        "precision": precision,
        "recall": recall,
        "per_kind": per_kind,
        "ambiguous_agreement": { "gold": amb_gold, "matched": amb_matched },
        "notes": notes,
    });
    if std::env::var("NLP_WRITE_RECEIPT").as_deref() == Ok("1") {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("docs/receipts/entities-2026-11-10.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&receipt).unwrap() + "\n",
        )
        .unwrap();
    }
    assert!(
        precision.is_none_or(|p| p >= 0.9),
        "precision {precision:?}"
    );
    assert!(recall.is_none_or(|r| r >= 0.8), "recall {recall:?}");
}

#[test]
fn review_liang_is_not_a_digit() {
    assert_eq!(ents("我們要一兩天才能完成。"), vec![]);
    assert_eq!(ents("來了兩三個人。"), vec![]);
    // Alone, 兩 is still 2.
    assert_eq!(ents("來了兩個人。"), vec![n("兩", "2")]);
    // Three or more plain digits stay digit by digit.
    assert_eq!(ents("二〇二三個"), vec![n("二〇二三", "2023")]);
}

#[test]
fn review_mixed_ascii_and_chinese_multiplier() {
    assert_eq!(ents("它值5千元。"), vec![qty("5千元", "5000", "元")]);
    assert_eq!(ents("它值3萬5千元。"), vec![qty("3萬5千元", "35000", "元")]);
    assert_eq!(ents("有3千萬人。"), vec![n("3千萬", "30000000")]);
    assert_eq!(ents("他買了3萬個蘋果。"), vec![n("3萬", "30000")]);
    assert_eq!(
        ents("它值1.5億元。"),
        vec![qty("1.5億元", "150000000", "元")]
    );
    // A run that cannot be read gives no number, and never a bare multiplier.
    assert_eq!(ents("它值3萬5元。"), vec![]);
    assert_eq!(ents("它值3萬4萬元。"), vec![]);
    assert_eq!(ents("它值3五千元。"), vec![]);
}

#[test]
fn review_invalid_full_date_has_no_fallback() {
    assert_eq!(ents("二〇二三年二月二十九日"), vec![]);
    assert_eq!(
        ents("2023年2月29日"),
        vec![n("2023", "2023"), n("2", "2"), n("29", "29")]
    );
    assert!(
        ents("29 Feb 2023 was it.")
            .iter()
            .all(|e| e.0 != EntityKind::Date)
    );
    assert!(
        ents("Feb 29, 2023 was it.")
            .iter()
            .all(|e| e.0 != EntityKind::Date)
    );
    assert_eq!(
        ents("2024年2月29日"),
        vec![d("2024年2月29日", "2024-02-29")]
    );
}

#[test]
fn review_zh_decimal_point() {
    assert_eq!(
        ents("增加了百分之三點五。"),
        vec![qty("百分之三點五", "3.5", "%")]
    );
    assert_eq!(
        ents("重三點五公斤。"),
        vec![qty("三點五公斤", "3.5", "公斤")]
    );
    assert_eq!(ents("三點五個"), vec![n("三點五", "3.5")]);
    // Time readings stay.
    assert_eq!(ent_unknown("三點見。"), vec!["三點"]);
    assert_eq!(ent_unknown("三點半見。"), vec!["三點半"]);
    assert_eq!(ent_unknown("三點五分見。"), vec!["三點五分"]);
    assert_eq!(ents("下午三點五分見。"), vec![t("下午三點五分", "15:05")]);
    // A bit.
    assert_eq!(ents("他有一點累。"), vec![]);
    assert_eq!(ent_unknown("他有一點累。"), Vec::<String>::new());
}

#[test]
fn review_small_rules() {
    assert!(ents("3:30:99 pm").iter().all(|e| e.0 != EntityKind::Time));
    assert_eq!(ents("3:30:59 pm"), vec![t("3:30:59 pm", "15:30")]);
    assert_eq!(ents("hundred dogs"), vec![]);
    assert_eq!(ents("two hundred dogs"), vec![n("two hundred", "200")]);
    // A name with NUL is ignored.
    let text = "Use `x` here.";
    assert!(rows(text, &analyze_names(text, &["\u{0}"])).is_empty());
    assert!(rows(text, &analyze_names(text, &["\u{0}x"])).is_empty());
    let doc = analyze_names("Bob \u{0} left.", &["\u{0}"]);
    assert!(doc.entities.iter().all(|e| e.kind != EntityKind::Name));
}

/// A prefix with no number after it, at the end of the text or of a sentence, must not panic.
#[test]
fn currency_prefix_at_end_does_not_panic() {
    for text in [
        "$",
        "NT$",
        "US$",
        "¥",
        "€",
        "£",
        "百分之",
        "Pay $",
        "It costs NT$",
        "漲了百分之",
        "Pay `x`$",
    ] {
        assert_eq!(ents(text), vec![], "{text:?}");
    }
    assert_eq!(ents("I owe you 10€"), vec![n("10", "10")]);
}

#[test]
fn english_number_order() {
    // A word that breaks the order ends the number before it, and the word starts a new number.
    assert_eq!(
        ents("I have two three cats."),
        vec![n("two", "2"), n("three", "3")]
    );
    assert_eq!(
        ents("In nineteen eighty-four it rained."),
        vec![n("nineteen", "19"), n("eighty-four", "84")]
    );
    assert_eq!(
        ents("It was twenty twenty."),
        vec![n("twenty", "20"), n("twenty", "20")]
    );
    assert_eq!(
        ents("They sold two hundred three hundred boxes."),
        vec![n("two hundred", "200"), n("three hundred", "300")]
    );
    // `one` alone is ambiguous, so it is an unknown and not a number.
    assert_eq!(ents("one two three"), vec![n("two", "2"), n("three", "3")]);
    assert_eq!(ent_unknown("one two three"), vec!["one"]);
    assert_eq!(
        ents("Five thousand two thousand"),
        vec![n("Five thousand", "5000"), n("two thousand", "2000")]
    );
    // When the breaking word is thousand, million or billion, the words after the last such
    // word start the new number.
    assert_eq!(
        ents("Five thousand two hundred thousand"),
        vec![
            n("Five thousand", "5000"),
            n("two hundred thousand", "200000")
        ]
    );
    assert_eq!(
        ents("five hundred thousand two hundred thousand"),
        vec![
            n("five hundred thousand", "500000"),
            n("two hundred thousand", "200000")
        ]
    );
    assert_eq!(
        ents("one million two hundred million"),
        vec![
            n("one million", "1000000"),
            n("two hundred million", "200000000")
        ]
    );
    // These stay one number each.
    assert_eq!(
        ents("three hundred thousand two hundred"),
        vec![n("three hundred thousand two hundred", "300200")]
    );
    assert_eq!(
        ents("two million three hundred thousand"),
        vec![n("two million three hundred thousand", "2300000")]
    );
    // Valid forms keep their values.
    assert_eq!(
        ents("They sold two thousand five hundred people."),
        vec![n("two thousand five hundred", "2500")]
    );
    assert_eq!(
        ents("one hundred twenty-three people"),
        vec![n("one hundred twenty-three", "123")]
    );
    // `and` ends a number, so this one is two numbers.
    assert_eq!(
        ents("one hundred and twenty-three people"),
        vec![n("one hundred", "100"), n("twenty-three", "23")]
    );
    // A unit after hundred is part of the number, but `zero` is not.
    assert_eq!(
        ents("two hundred three people"),
        vec![n("two hundred three", "203")]
    );
    assert_eq!(
        ents("two hundred zero people"),
        vec![n("two hundred", "200"), n("zero", "0")]
    );
}

/// The examples in the docs ("Entities" and "Known limits of entities"), each checked here.
#[test]
fn doc_examples() {
    // Signs and ranges.
    assert_eq!(ents("It was -5 there."), vec![n("5", "5")]);
    assert_eq!(ents("Pages 3-5 are dull."), vec![n("3", "3"), n("5", "5")]);
    // Weekdays, ordinals and time zones.
    assert_eq!(ents("We meet on Monday."), vec![]);
    assert_eq!(ents("Chapter 1st edition"), vec![]);
    assert_eq!(ents("統一第三。"), vec![]);
    assert_eq!(ents("Call at 3pm EST."), vec![t("3pm", "15:00")]);
    // Durations with 年 are numbers.
    assert_eq!(ents("他工作了3年。"), vec![n("3", "3")]);
    // Chinese numerals need a measure word.
    assert_eq!(ents("我有三十五。"), vec![]);
    assert_eq!(ents("我有三十五個。"), vec![n("三十五", "35")]);
    // Chinese date shapes: Y年M月D日 (or 號), Y年M月, M月D日 (or 號), Y年 alone.
    assert_eq!(
        ents("2026年11月10號"),
        vec![d("2026年11月10號", "2026-11-10")]
    );
    assert_eq!(ents("2026年11月"), vec![d("2026年11月", "2026-11")]);
    assert_eq!(ents("11月10日"), vec![d("11月10日", "--11-10")]);
    assert_eq!(ents("2026年"), vec![d("2026年", "2026")]);
    // M月 alone and D日 alone are not dates. ASCII digits are numbers; a Chinese run needs a
    // measure word, so 十一月 alone gives nothing.
    assert_eq!(ents("11月"), vec![n("11", "11")]);
    assert_eq!(ents("10日"), vec![n("10", "10")]);
    assert_eq!(ents("十一月"), vec![]);
    assert_eq!(ents("2026年10日"), vec![d("2026年", "2026"), n("10", "10")]);
    // Ambiguous and bad numeric dates.
    assert_eq!(
        ents("Due 13/14/2026."),
        vec![n("13", "13"), n("14", "14"), n("2026", "2026")]
    );
    assert_eq!(
        ents("Due 2026/2/30."),
        vec![n("2026", "2026"), n("2", "2"), n("30", "30")]
    );
    assert_eq!(
        ents("Due 2/30/2026."),
        vec![n("2", "2"), n("30", "30"), n("2026", "2026")]
    );
    assert_eq!(ents("Due 3/4/2026."), vec![]);
    assert_eq!(ent_unknown("Due 3/4/2026."), vec!["3/4/2026"]);
    assert_eq!(
        ents("Due 0/5/2026."),
        vec![n("0", "0"), n("5", "5"), n("2026", "2026")]
    );
    // An abbreviated month takes a dot before the day.
    assert_eq!(ents("It is Nov. 10 now."), vec![d("Nov. 10", "--11-10")]);
    // The hour is attached to 點 in the H點 form.
    assert_eq!(ents("我們 5 點 見"), vec![n("5", "5")]);
    // Chinese numerals in a full date need a measure word; ASCII digits are numbers.
    assert_eq!(ents("二〇二三年二月二十九日"), vec![]);
    assert_eq!(
        ents("2023年2月29日"),
        vec![n("2023", "2023"), n("2", "2"), n("29", "29")]
    );
}

#[test]
fn ws4_classical_durations_and_approx_quantities() {
    let approx = |span: &str, unit: &str| -> Row {
        (
            EntityKind::Quantity,
            span.to_owned(),
            None,
            Some(unit.to_owned()),
        )
    };

    // Exact classical & temporal durations.
    assert_eq!(ents("他在山中住了四年。"), vec![qty("四年", "4", "年")]);
    assert_eq!(
        ents("古樹已歷三百年與萬年風霜。"),
        vec![qty("三百年", "300", "年"), qty("萬年", "10000", "年")]
    );
    assert_eq!(
        ents("船隊航行了兩日三夜，歷時三載。"),
        vec![
            qty("兩日", "2", "日"),
            qty("三夜", "3", "夜"),
            qty("三載", "3", "載"),
        ]
    );
    assert_eq!(
        ents("他靜坐五载，守到五更，等了一刻鐘與两刻钟。"),
        vec![
            qty("五载", "5", "载"),
            qty("五更", "5", "更"),
            qty("一刻鐘", "1", "刻鐘"),
            qty("两刻钟", "2", "刻钟"),
        ]
    );
    assert_eq!(
        ents("茶水煮了一炷香或两柱香，歷經一甲子與兩個甲子。"),
        vec![
            qty("一炷香", "1", "炷香"),
            qty("两柱香", "2", "柱香"),
            qty("一甲子", "1", "甲子"),
            qty("兩個甲子", "2", "個甲子"),
        ]
    );

    // Fractional `半` quantities (`0.5`).
    assert_eq!(
        ents("他走了半日，休息半個時辰，又等了半盞茶與半炷香。"),
        vec![
            qty("半日", "0.5", "日"),
            qty("半個時辰", "0.5", "個時辰"),
            qty("半盞茶", "0.5", "盞茶"),
            qty("半炷香", "0.5", "炷香"),
        ]
    );
    assert_eq!(
        ents("工程耗時半年、半個月、半天、半小時、半个时辰、半盏茶、半柱香、半小时、半个月。"),
        vec![
            qty("半年", "0.5", "年"),
            qty("半個月", "0.5", "個月"),
            qty("半天", "0.5", "天"),
            qty("半小時", "0.5", "小時"),
            qty("半个时辰", "0.5", "个时辰"),
            qty("半盏茶", "0.5", "盏茶"),
            qty("半柱香", "0.5", "柱香"),
            qty("半小时", "0.5", "小时"),
            qty("半个月", "0.5", "个月"),
        ]
    );

    // Approximate prefixes (`數/数`, `幾/几`, `<十-cardinal>幾`) and infixes (`餘/余/多`).
    assert_eq!(
        ents("數年與数年、數日與数日、數月與数月、數十丈與数十丈、數百年與数百年。"),
        vec![
            approx("數年", "年"),
            approx("数年", "年"),
            approx("數日", "日"),
            approx("数日", "日"),
            approx("數月", "月"),
            approx("数月", "月"),
            approx("數十丈", "丈"),
            approx("数十丈", "丈"),
            approx("數百年", "年"),
            approx("数百年", "年"),
        ]
    );
    assert_eq!(
        ents("石壁高十幾丈或十几丈，他們停留幾年、几年、幾日、几日、幾天、几天、幾個月。"),
        vec![
            approx("十幾丈", "丈"),
            approx("十几丈", "丈"),
            approx("幾年", "年"),
            approx("几年", "年"),
            approx("幾日", "日"),
            approx("几日", "日"),
            approx("幾天", "天"),
            approx("几天", "天"),
            approx("幾個月", "個月"),
        ]
    );
    assert_eq!(
        ents("歷經百餘年與百余年，石塔高百餘丈、百余丈、十餘丈、十余丈，耗時一百多年與三十多天。"),
        vec![
            approx("百餘年", "年"),
            approx("百余年", "年"),
            approx("百餘丈", "丈"),
            approx("百余丈", "丈"),
            approx("十餘丈", "丈"),
            approx("十余丈", "丈"),
            approx("一百多年", "年"),
            approx("三十多天", "天"),
        ]
    );

    // Non-regression guards (`MEASURE` words, calendar years, `半夜`, `一刻也不停`).
    assert_eq!(ents("她走了三十里路，只帶了一包乾糧。"), vec![]);
    assert_eq!(ents("超過一半的人 agreed."), vec![]);
    assert_eq!(ents("半夜聽見風聲，他一刻也不停。"), vec![]);
    assert_eq!(ents("2026年"), vec![d("2026年", "2026")]);
    assert_eq!(ents("二〇二六年"), vec![d("二〇二六年", "2026")]);
}

#[test]
fn ws4_relative_duration_anchors_and_moment_cues() {
    let rd = |span: &str| row(EntityKind::RelativeDate, span, None);

    assert_eq!(
        ents("四年后與四年後、三百年前、兩日后、數月內、半個時辰后、數月前、百餘年後。"),
        vec![
            rd("四年后"),
            rd("四年後"),
            rd("三百年前"),
            rd("兩日后"),
            rd("數月內"),
            rd("半個時辰后"),
            rd("數月前"),
            rd("百餘年後"),
        ]
    );

    assert_eq!(
        ents("片刻、片刻後、片刻后、頃刻、頃刻間、顷刻、顷刻间、剎那、剎那間、刹那、刹那间。"),
        vec![
            rd("片刻"),
            rd("片刻後"),
            rd("片刻后"),
            rd("頃刻"),
            rd("頃刻間"),
            rd("顷刻"),
            rd("顷刻间"),
            rd("剎那"),
            rd("剎那間"),
            rd("刹那"),
            rd("刹那间"),
        ]
    );

    assert_eq!(
        ents(
            "轉眼、轉眼間、转眼、转眼间、眨眼間、眨眼间、眨眼工夫、眨眼功夫、眨眼之間、眨眼之间、瞬息、瞬息間、瞬息间、彈指間、弹指间。"
        ),
        vec![
            rd("轉眼"),
            rd("轉眼間"),
            rd("转眼"),
            rd("转眼间"),
            rd("眨眼間"),
            rd("眨眼间"),
            rd("眨眼工夫"),
            rd("眨眼功夫"),
            rd("眨眼之間"),
            rd("眨眼之间"),
            rd("瞬息"),
            rd("瞬息間"),
            rd("瞬息间"),
            rd("彈指間"),
            rd("弹指间"),
        ]
    );
    assert_eq!(ents("青衫老者眨了眨眼，沒有說話。"), vec![]);

    assert_eq!(
        ents("良久、許久、许久、半晌、一會兒、一会儿。"),
        vec![
            rd("良久"),
            rd("許久"),
            rd("许久"),
            rd("半晌"),
            rd("一會兒"),
            rd("一会儿"),
        ]
    );
}

#[test]
fn ws4_v3_token_boundary_alignment_guard_and_ascending_approx_ranges() {
    let opts = Options {
        normalize_traditional: true,
        lexicon: vec!["萬天星".to_owned(), "玄剎".to_owned()],
        ..Options::default()
    };

    // 1. False-positive candidates cutting inside proper names, idioms, or lexical compounds
    //    must be rejected under `normalize_traditional: true` + context guards.
    let false_pos_text = concat!(
        "萬天星與萬天星光映照石殿，玄剎那道霞光在半天空中化作九天神雷落向九天之上，",
        "這真是千載難逢的機緣，眾人退避三尺，以後天氣轉涼，迎來晴朗的一天。"
    );
    let doc_fp = analyze_document(false_pos_text, &opts);
    let fp_ents: Vec<(EntityKind, String)> = doc_fp
        .entities
        .iter()
        .map(|e| (e.kind, slice(false_pos_text, e.span)))
        .collect();
    assert_eq!(
        fp_ents,
        vec![],
        "expected zero false-positive entities on lexical/idiom compounds, got {fp_ents:?}"
    );

    // 2. Legitimate quantities, relative dates, and composite temporal tokens still pass 100%.
    let valid_text = "王天明在山谷等了半天，三天後又走了半個時辰，苦修三天三夜並採下三株仙草。";
    let doc_valid = analyze_document(valid_text, &opts);
    for e in &doc_valid.entities {
        assert!(e.span.start_char < e.span.end_char && e.span.end_char <= doc_valid.text_chars);
    }
    assert_eq!(
        rows(valid_text, &doc_valid),
        vec![
            qty("半天", "0.5", "天"),
            r("三天後"),
            qty("半個時辰", "0.5", "個時辰"),
            qty("三天", "3", "天"),
            qty("三夜", "3", "夜"),
            n("三", "3"),
        ]
    );

    // 3. Two-digit ascending Chinese approximations normalize to `"a-b"` ranges in `Quantity`.
    let approx_text = "石劍長一二尺、寬二三寸，隊伍行了三四日、越過四五丈溪流、挖出五六寸木盒、停留六七天、退開七八丈、歷時八九年。";
    let doc_approx = analyze_document(approx_text, &opts);
    assert_eq!(
        rows(approx_text, &doc_approx),
        vec![
            qty("一二尺", "1-2", "尺"),
            qty("二三寸", "2-3", "寸"),
            qty("三四日", "3-4", "日"),
            qty("四五丈", "4-5", "丈"),
            qty("五六寸", "5-6", "寸"),
            qty("六七天", "6-7", "天"),
            qty("七八丈", "7-8", "丈"),
            qty("八九年", "8-9", "年"),
        ]
    );
}
