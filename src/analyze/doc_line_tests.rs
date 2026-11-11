//! The `nlp.doc.v1` line path must give the same results as the ICU path (zh-TW).
//!
//! ICU and layer 1 may cut a line at different places, for example ICU keeps an opening quote
//! with the sentence before it. `append_line` joins pieces until quotes close, so the check
//! that counts is the final `Analysis`. Term hits are compared directly. A fixture that
//! differs is an open point for the owner, not a silent change.
use super::{Lexicon, NarrativeAnalyzer};
use crate::text::Locale;

const ACTORS: &[(&str, &str)] = &[("小甲", "a"), ("阿乙", "b"), ("Alice Lee", "c")];
const OBJECTS: &[(&str, &str)] = &[("木劍", "sword"), ("信", "letter")];
const PLACES: &[&str] = &["木屋", "港口"];

/// Chapters of short, neutral, self-written text: novel, paper and casual style, zh-TW,
/// English and mixed.
const CHAPTERS: &[&str] = &[
    "小甲走進木屋。阿乙留在港口。\n小甲說：「我們明天出發。」阿乙點頭。\n",
    "小甲問：「你去哪裡？」阿乙沒有回答。\n「走吧！」小甲說。「好。」阿乙說。\n",
    "小甲把木劍交給阿乙，然後離開了木屋。\n他走了很久……終於到了港口。\n",
    "Alice Lee 把信交給小甲。小甲收下了。\n小甲 met Alice Lee at the 港口. Then they left.\n",
    "The method is simple. We measure the 台灣 data twice. Results follow.\nResults are shown in Table 2. See Fig. 3 for details.\n",
    "小甲看了信（那是昨天寫的。）然後走了。\n阿乙隨後進了木屋。\n",
    "哈哈 今天好累 小甲先走了 我也要回家了\n小甲：你在哪裡？\n阿乙回答！我在港口！\n",
    "一行沒有標點的文字\n\n   \n。\n",
    "小甲「你好」阿乙「再見」\n他說：“It is fine.” 小甲點頭。\n",
    "第一章 開始\n小甲走進木屋。\n- 項目一\n阿乙留在港口。\n",
    "小甲打開了信，看見了網址 https://example.org/a。然後他離開了。\n",
    "# 標題\n小甲走進木屋。\n```\ncode here\n```\n阿乙離開港口。\n",
    "",
    "小甲說：「等一下……」阿乙停下來。\n",
    "小甲說...然後走了。阿乙也走了...\n",
    "你好嗎？！小甲笑了!? 阿乙也笑了。\n",
    "Mr. Lee met Dr. Wu at 3.5 p.m. in the 港口. It was fine.\n",
    "小甲  阿乙  木屋\n",
    "小甲走了😀阿乙留下了。\n",
    "小甲說：\"好的。\"阿乙說：\"走吧!\"\n",
    "小甲走了。」阿乙留下。\n",
    "（小甲走進木屋。）阿乙留下。\n",
    "小甲讀了 `a。b` 這段，然後走了。阿乙留下。\n",
    "小甲引用了文獻 [1]。阿乙也引用了 (Lee, 2020)。\n",
    "小甲算出 $a+b$ 的值。阿乙留下。\n",
    "小甲查了 www.example.org。然後 http://example.org/a.b 也看了。\n",
    "小甲走進木屋；阿乙留在港口：他們都累了。\n",
    "Alice Lee, however, stayed. She read the letter twice; then she left.\n",
    "表 1 列出結果。如圖 2 所示，方法有效。\n",
];

fn analyzers<'a>(lexicon: &Lexicon<'a>) -> (NarrativeAnalyzer<'a>, NarrativeAnalyzer<'a>) {
    let doc = NarrativeAnalyzer::new(Locale::ChineseTaiwan, lexicon).expect("analyzer");
    let mut icu = NarrativeAnalyzer::new(Locale::ChineseTaiwan, lexicon).expect("analyzer");
    assert!(doc.doc_terms.is_some(), "zh-TW reads doc v1");
    icu.doc_terms = None;
    (doc, icu)
}

fn lexicon() -> Lexicon<'static> {
    Lexicon {
        actors: ACTORS,
        objects: OBJECTS,
        places: PLACES,
        ..Lexicon::default()
    }
}

#[test]
fn doc_path_gives_the_icu_analysis_on_neutral_chapters() {
    let lexicon = lexicon();
    let (doc, icu) = analyzers(&lexicon);
    for chapter in CHAPTERS {
        let a = serde_json::to_string(&doc.analyze(chapter, "")).expect("json");
        let b = serde_json::to_string(&icu.analyze(chapter, "")).expect("json");
        assert_eq!(a, b, "analysis differs for {chapter:?}");
        let a = serde_json::to_string(&doc.analyze_source_mentions(chapter, "")).expect("json");
        let b = serde_json::to_string(&icu.analyze_source_mentions(chapter, "")).expect("json");
        assert_eq!(a, b, "source mentions differ for {chapter:?}");
    }
}

#[test]
fn doc_path_gives_the_icu_term_hits_on_every_line() {
    let lexicon = lexicon();
    let (doc, icu) = analyzers(&lexicon);
    for line in CHAPTERS
        .iter()
        .flat_map(|chapter| chapter.split_inclusive('\n'))
    {
        assert_eq!(
            doc.line_sentences(line).1,
            icu.line_sentences(line).1,
            "term hits differ for {line:?}"
        );
    }
}

#[test]
fn the_doc_path_really_serves_plain_zh_tw_lines() {
    let lexicon = lexicon();
    let (doc, _) = analyzers(&lexicon);
    let terms = doc.doc_terms.as_ref().expect("terms");
    let used = CHAPTERS
        .iter()
        .flat_map(|chapter| chapter.split_inclusive('\n'))
        .filter(|line| super::doc_line(line, terms).is_some())
        .count();
    assert!(used >= 15, "only {used} fixture lines use the doc path");
}

/// Open points for the owner: where layer 1 and ICU cut differently, the line stays on the ICU
/// path. Each case here must show a real difference and a fallback. The test fails when either
/// side changes.
#[test]
fn known_differences_stay_on_the_icu_path() {
    let lexicon = lexicon();
    let (doc, icu) = analyzers(&lexicon);
    let terms = doc.doc_terms.as_ref().expect("terms");
    for line in [
        "他走了很久……終於到了港口。\n",
        "小甲說...然後走了。阿乙也走了...\n",
        "Mr. Lee met Dr. Wu in the 港口. It was fine.\n",
        "小甲  阿乙  木屋\n",
        "小甲讀了 `a。b` 這段，然後走了。\n",
        "第一章 開始\n",
        "- 項目一\n",
        "天黑了。「回家吧」，小甲說。\n",
        "小甲走了。、阿乙留下。\n",
        "好？ ！\n",
    ] {
        assert!(super::doc_line(line, terms).is_none(), "{line:?}");
        assert_eq!(
            serde_json::to_string(&doc.analyze(line, "")).expect("json"),
            serde_json::to_string(&icu.analyze(line, "")).expect("json"),
            "the fallback must give the ICU analysis for {line:?}"
        );
        let view = crate::doc::line_view(line, terms);
        let layer1: Vec<_> = view.sentences.iter().map(|r| (r.start, r.end)).collect();
        let old: Vec<_> = icu
            .icu_line(line)
            .0
            .iter()
            .map(|s| (s.start, s.end))
            .collect();
        assert_ne!(layer1, old, "layer 1 now agrees with ICU on {line:?}");
    }
}

#[test]
fn only_the_zh_tw_locale_reads_doc_v1() {
    let lexicon = Lexicon::default();
    for locale in [Locale::English, Locale::ChineseChina] {
        let analyzer = NarrativeAnalyzer::new(locale, &lexicon).expect("analyzer");
        assert!(analyzer.doc_terms.is_none());
    }
}

/// Inputs from the review of pull request 36. Each one made the first doc path differ from ICU.
const REVIEW_INPUTS: &[&str] = &[
    "阿乙說好!小甲走進木屋。\n",
    "小甲笑了!阿乙也笑了。\n",
    "小甲!阿乙!\n",
    "你好?小甲問。阿乙沒回答。\n",
    "小甲問?阿乙答!\n",
    "(小甲走進木屋!)阿乙留下。\n",
    "我們看了《快跑！》這部電影。\n",
    "小甲說：【走吧！】阿乙點頭。\n",
    "小甲說：〈走吧！〉阿乙點頭。\n",
    "小甲說：〔走吧！〕阿乙點頭。\n",
    "小甲說：［走吧！］阿乙點頭。\n",
    "小甲說：[走吧！]阿乙點頭。\n",
    "小甲｡阿乙留下｡\n",
    "小甲走了．阿乙留下。\n",
    "小甲走了\u{2028}阿乙留下。\n",
    "小甲走了。\r\n阿乙留下。\r\n",
    "小甲\t阿乙\n",
    "Alice Lees 走進木屋。\n",
    "Lee Parkway 的小甲走進木屋。\n",
    "小甲走進木屋。阿乙留在港口。\n",
    "小甲說：「我們明天出發。」阿乙點頭。\n",
    "小甲 走進木屋，阿乙　留在港口。\n",
    // Second review: a terminator run followed by a continuation, a terminator or an opening
    // quote, and unbalanced quotes.
    "「好。」，他說。\n",
    "天黑了。「回家吧」，小甲說。\n",
    "小甲笑了。「真的」阿乙說。\n",
    "小甲走了。「好」\n",
    "好！「劍」\n",
    "小甲走了。，阿乙留下。\n",
    "小甲走了。、阿乙留下。\n",
    "好。：好\n",
    "好。；好\n",
    "好？ ！\n",
    "在。 。\n",
    "小甲。」「阿乙。\n",
    "好。」「\n",
    "「好。」，他說。",
    "小甲走了。，阿乙留下。\r\n小甲走了。，阿乙留下。\r\n",
    "小甲。」「阿乙。\n小甲。」「阿乙。\n",
    "　　小甲走進木屋。阿乙留在港口。\n　　「走吧。」小甲說。\n",
];

fn review_lexicon() -> Lexicon<'static> {
    Lexicon {
        actors: &[
            ("小甲", "a"),
            ("阿乙", "b"),
            ("Alice Lee", "c"),
            ("Alice", "d"),
            ("Lee", "e"),
        ],
        objects: &[("Lee Park", "park")],
        places: PLACES,
        ..Lexicon::default()
    }
}

#[test]
fn review_inputs_give_the_icu_analysis_and_hits() {
    let lexicon = review_lexicon();
    let (doc, icu) = analyzers(&lexicon);
    for input in REVIEW_INPUTS {
        let a = serde_json::to_string(&doc.analyze(input, "")).expect("json");
        let b = serde_json::to_string(&icu.analyze(input, "")).expect("json");
        assert_eq!(a, b, "analysis differs for {input:?}");
        let a = serde_json::to_string(&doc.analyze_source_mentions(input, "")).expect("json");
        let b = serde_json::to_string(&icu.analyze_source_mentions(input, "")).expect("json");
        assert_eq!(a, b, "source mentions differ for {input:?}");
        for line in input.split_inclusive('\n') {
            let (ds, dh) = doc.line_sentences(line);
            let (is, ih) = icu.line_sentences(line);
            assert_eq!(dh, ih, "hits differ for {line:?}");
            let ds: Vec<_> = ds.iter().map(|s| (s.start, s.end)).collect();
            let is: Vec<_> = is.iter().map(|s| (s.start, s.end)).collect();
            assert_eq!(ds, is, "sentences differ for {line:?}");
        }
    }
}

#[test]
fn doc_applies_is_an_allow_list_with_a_context_rule() {
    for line in [
        "小甲走進木屋。\n",
        "小甲走進木屋。",
        "小甲 阿乙\n",
        "小甲　阿乙\n",
        "「好。」小甲『是。』阿乙，、；：「」\n",
        "小甲走了。」阿乙留下。\n",
        "小甲說：「走吧！」阿乙點頭。\n",
        "小甲走了。 阿乙留下。\n",
        "　小甲\n",
        "　　小甲走進木屋。\n",
        "\n",
        "",
    ] {
        assert!(super::doc_applies(line), "{line:?}");
    }
    for line in [
        "a\n",
        "小甲1\n",
        "小甲!\n",
        "小甲.\n",
        "小甲…\n",
        "《小甲》\n",
        "【小甲】\n",
        "小甲｡\n",
        "小甲．\n",
        "小甲\u{2028}\n",
        "小甲\t阿乙\n",
        "小甲。\r\n",
        "小甲😀\n",
        "小甲  阿乙\n",
        "小甲 　阿乙\n",
        " 小甲\n",
        "小甲 \n",
        "　\n",
        "　　\n",
        "　 小甲\n",
        " 　小甲\n",
        "「好。」，他說。\n",
        "小甲走了。「好」\n",
        "小甲走了。，阿乙留下。\n",
        "好。：好\n",
        "好。；好\n",
        "好？ ！\n",
        "在。 。\n",
        "好。」「\n",
        "好。」 」\n",
        "好。』『好\n",
        "小甲\n\n",
        "小甲\n阿乙\n",
    ] {
        assert!(!super::doc_applies(line), "{line:?}");
    }
}

#[test]
fn allowed_plain_lines_use_the_doc_path_with_the_same_ranges() {
    let lexicon = review_lexicon();
    let (doc, icu) = analyzers(&lexicon);
    let terms = doc.doc_terms.as_ref().expect("terms");
    for line in [
        "小甲走進木屋。阿乙留在港口。\n",
        "小甲說：「我們明天出發。」阿乙點頭。\n",
        "小甲問：「你去哪裡？」阿乙沒有回答。\n",
    ] {
        let (sentences, hits) = super::doc_line(line, terms).expect("doc path");
        let view = crate::doc::line_view(line, terms);
        let layer1: Vec<_> = view.sentences.iter().map(|r| (r.start, r.end)).collect();
        let got: Vec<_> = sentences.iter().map(|s| (s.start, s.end)).collect();
        assert_eq!(got, layer1, "{line:?}");
        assert_eq!(hits, icu.line_sentences(line).1, "{line:?}");
    }
}

/// A small fixed-seed generator (SplitMix64), so the differential runs are reproducible
/// without a new dependency.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// Han characters of the generated text: the lexicon names, pronouns and common words.
const GEN_HAN: &[char] = &[
    '小', '甲', '阿', '乙', '丙', '老', '王', '他', '她', '們', '說', '走', '了', '笑', '天', '黑',
    '回', '家', '吧', '真', '的', '好', '在', '留', '下', '木', '劍', '信', '屋', '港', '口', '將',
    '軍', '大', '哥', '台', '灣', '是', '不', '也', '來', '去', '問', '我', '你', '它',
];
/// The punctuation of the allow list.
const GEN_PUNCT: &[char] = &[
    '。', '！', '？', '，', '、', '；', '：', '「', '」', '『', '』',
];
/// Characters outside the allow list, for the mixed run.
const GEN_OTHER: &[char] = &[
    'a', 'B', '1', '.', '!', '?', ',', ' ', '\t', '…', '—', '《', '》', '（', '）', '😀', '“', '”',
    '"', '-', '#', '`', '$', '[', ']', '｡', '．', '\u{3000}', '\u{2028}',
];

/// One line from the allow-list alphabet (Han, the listed punctuation, single spaces).
pub(crate) fn gen_allowed_line(rng: &mut Rng) -> String {
    let len = match rng.below(400) {
        0 => 4_000,
        1..=32 => 0,
        _ => 1 + rng.below(30),
    };
    let mut line = String::new();
    for _ in 0..len {
        let ch = match rng.below(20) {
            0..=10 => rng.pick(GEN_HAN),
            11..=18 => rng.pick(GEN_PUNCT),
            _ => rng.pick(&[' ', '\u{3000}']),
        };
        line.push(ch);
    }
    line
}

/// One line of the mixed run: an allow-list line, the same with other characters mixed in,
/// or with a paragraph indent.
pub(crate) fn gen_mixed_line(rng: &mut Rng) -> String {
    let line = gen_allowed_line(rng);
    match rng.below(4) {
        0 => line,
        1 => format!("\u{3000}\u{3000}{line}"),
        2 => format!("{}{line}", rng.pick(&[" ", "\u{3000}", "　 ", "  "])),
        _ => {
            let mut out = String::new();
            for ch in line.chars() {
                if rng.below(8) == 0 {
                    out.push(rng.pick(GEN_OTHER));
                }
                out.push(ch);
            }
            if rng.below(4) == 0 {
                out.push(rng.pick(GEN_OTHER));
            }
            out
        }
    }
}

/// One chapter: 1 to 6 lines, LF or CRLF, empty lines, with or without a final line break.
fn gen_chapter(rng: &mut Rng, line: fn(&mut Rng) -> String) -> String {
    let newline = if rng.below(5) == 0 { "\r\n" } else { "\n" };
    let count = 1 + rng.below(6);
    let mut chapter = String::new();
    for index in 0..count {
        if index > 0 {
            chapter.push_str(newline);
        }
        if rng.below(8) != 0 {
            chapter.push_str(&line(rng));
        }
    }
    if rng.below(3) != 0 {
        chapter.push_str(newline);
    }
    chapter
}

/// The lexicon of the differential runs, with names that cross punctuation and spaces.
fn differential_lexicon() -> Lexicon<'static> {
    Lexicon {
        actors: &[
            ("小甲", "a"),
            ("甲哥", "a"),
            ("阿乙", "b"),
            ("甲走", "c"),
            ("老 王", "d"),
            ("王", "e"),
            ("小甲阿乙", "f"),
            ("「小丙」", "g"),
            ("說。小", "h"),
            ("他", "he"),
            ("台灣", "tw"),
        ],
        objects: &[("木劍", "sword"), ("信", "letter"), ("劍，", "comma")],
        places: &["木屋", "港口", "屋港", "口小"],
        ranks: &[("將軍", "general"), ("大將", "major"), ("軍阿", "odd")],
        ..Lexicon::default()
    }
}

/// Runs `count` generated chapters through both paths and returns the inputs whose full
/// `Analysis` JSON (of `analyze` and `analyze_source_mentions`), line sentence ranges or term
/// hits differ, and the number of lines that used the doc path.
fn differential_run(
    seed: u64,
    count: usize,
    line: fn(&mut Rng) -> String,
) -> (Vec<String>, usize, usize) {
    let lexicon = differential_lexicon();
    let (doc, icu) = analyzers(&lexicon);
    let terms = doc.doc_terms.as_ref().expect("terms");
    let mut rng = Rng(seed);
    let mut differ = Vec::new();
    let (mut lines, mut used) = (0, 0);
    for _ in 0..count {
        let chapter = gen_chapter(&mut rng, line);
        let mut same = serde_json::to_string(&doc.analyze(&chapter, "")).expect("json")
            == serde_json::to_string(&icu.analyze(&chapter, "")).expect("json")
            && serde_json::to_string(&doc.analyze_source_mentions(&chapter, "")).expect("json")
                == serde_json::to_string(&icu.analyze_source_mentions(&chapter, "")).expect("json");
        for line in chapter.split_inclusive('\n') {
            lines += 1;
            if let Some((sentences, hits)) = super::doc_line(line, terms) {
                used += 1;
                let (old, old_hits) = icu.icu_line(line);
                let got: Vec<_> = sentences.iter().map(|s| (s.start, s.end)).collect();
                let old: Vec<_> = old.iter().map(|s| (s.start, s.end)).collect();
                same &= got == old && hits == old_hits;
            }
        }
        if !same {
            differ.push(chapter);
        }
    }
    (differ, lines, used)
}

#[test]
fn generated_allow_list_chapters_give_the_icu_analysis() {
    let (differ, lines, used) = differential_run(0x5EED_0023, 20_001, gen_allowed_line);
    eprintln!(
        "allow-list run: {} differ, {used} of {lines} lines on the doc path",
        differ.len()
    );
    for chapter in differ.iter().take(40) {
        eprintln!("DIFF {chapter:?}");
    }
    assert!(
        used >= 5_000,
        "only {used} of {lines} generated lines use the doc path"
    );
    assert!(
        differ.is_empty(),
        "{} of 20001 generated chapters differ",
        differ.len()
    );
}

#[test]
fn generated_mixed_chapters_give_the_icu_analysis() {
    let (differ, lines, used) = differential_run(0x5EED_1023, 10_000, gen_mixed_line);
    eprintln!(
        "mixed run: {} differ, {used} of {lines} lines on the doc path",
        differ.len()
    );
    for chapter in differ.iter().take(40) {
        eprintln!("DIFF {chapter:?}");
    }
    assert!(
        used >= 5_000,
        "only {used} of {lines} generated lines use the doc path"
    );
    assert!(
        differ.is_empty(),
        "{} of 10000 generated chapters differ",
        differ.len()
    );
}
