//! Sentence and word segmentation with ranges into the original UTF-8 input.

use std::{collections::BTreeMap, ops::Range, sync::OnceLock};

use aho_corasick::AhoCorasick;
use anyhow::{Result, ensure};
use icu_locale_core::{LanguageIdentifier, langid};
use icu_segmenter::{SentenceSegmenter, options::SentenceBreakOptions};
use jieba_rs::Jieba;
use unicode_segmentation::UnicodeSegmentation;

/// Share the existing bundled dictionary between segmentation and probes.
pub(crate) fn default_jieba() -> &'static Jieba {
    static JIEBA: OnceLock<Jieba> = OnceLock::new();
    JIEBA.get_or_init(Jieba::new)
}

const TAIWAN_TERMS: &[&str] = &[
    "臺灣",
    "台灣",
    "人工智慧",
    "機器學習",
    "智慧型手機",
    "積體電路",
    "資料庫",
    "記憶體",
    "計程車",
    "身分證",
    "網際網路",
    "軟體",
    "硬體",
    "行動電源",
    "原始碼",
    "作業系統",
    "隨身碟",
    "印表機",
];
const CHINA_TERMS: &[&str] = &[
    "中国大陆",
    "人工智能",
    "机器学习",
    "智能手机",
    "集成电路",
    "数据库",
    "内存",
    "出租车",
    "身份证",
    "互联网",
    "软件",
    "硬件",
    "移动电源",
    "源代码",
    "操作系统",
    "U盘",
    "打印机",
];

/// The known locale of the text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Locale {
    English,
    ChineseTaiwan,
    ChineseChina,
}

impl Locale {
    /// Return the BCP 47 language tag for this locale.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::ChineseTaiwan => "zh-Hant-TW",
            Self::ChineseChina => "zh-Hans-CN",
        }
    }

    fn language(self) -> LanguageIdentifier {
        match self {
            Self::English => langid!("en"),
            Self::ChineseTaiwan => langid!("zh-Hant-TW"),
            Self::ChineseChina => langid!("zh-Hans-CN"),
        }
    }

    fn terms(self) -> &'static [&'static str] {
        match self {
            Self::English => &[],
            Self::ChineseTaiwan => TAIWAN_TERMS,
            Self::ChineseChina => CHINA_TERMS,
        }
    }
}

/// Distinguishes ordinary words, matched vocabulary, and symbols.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenKind {
    Word,
    Term,
    Symbol,
}

/// A token with a half-open byte range into the original input.
#[derive(Debug, Eq, PartialEq)]
pub struct Token<'a> {
    pub text: &'a str,
    pub bytes: Range<usize>,
    pub kind: TokenKind,
}

/// A sentence with its original punctuation and tokens.
#[derive(Debug, Eq, PartialEq)]
pub struct Sentence<'a> {
    pub text: &'a str,
    pub bytes: Range<usize>,
    pub tokens: Vec<Token<'a>>,
}

/// Reuse this analyzer for texts with the same locale and vocabulary.
pub struct Analyzer {
    locale: Locale,
    sentences: SentenceSegmenter,
    terms: Option<AhoCorasick>,
}

impl Analyzer {
    pub fn new(locale: Locale) -> Result<Self> {
        Self::with_terms(locale, &[])
    }

    /// Add exact, case-sensitive terms for this analyzer. A term may contain
    /// internal spaces. Matches stay within a sentence and prefer the longest
    /// term at the earliest available position.
    pub fn with_terms(locale: Locale, terms: &[&str]) -> Result<Self> {
        ensure!(
            terms.iter().all(|term| {
                !term.is_empty() && term.trim() == *term && !term.contains(['\r', '\n'])
            }),
            "terms must be nonempty and have no surrounding whitespace or line breaks"
        );
        let language = locale.language();
        let mut options = SentenceBreakOptions::default();
        options.content_locale = Some(&language);
        let sentences = SentenceSegmenter::try_new(options)?;
        let regional = locale.terms();
        let terms = (!regional.is_empty() || !terms.is_empty())
            .then(|| AhoCorasick::new(regional.iter().chain(terms).copied()))
            .transpose()?;
        Ok(Self {
            locale,
            sentences,
            terms,
        })
    }

    /// Split the original text into sentences and tokens. Skip blank sentences
    /// and whitespace tokens. All ranges index the whole input.
    pub fn analyze<'a>(&self, source: &'a str) -> Vec<Sentence<'a>> {
        let segmenter = self.sentences.as_borrowed();
        let mut start = 0;
        let mut result = Vec::new();
        for end in segmenter.segment_str(source).skip(1) {
            let part = &source[start..end];
            if self.locale == Locale::English && continues_english_title(part, &source[end..]) {
                continue;
            }
            let text = part.trim();
            if !text.is_empty() {
                let base = end - part.trim_start().len();
                result.push(Sentence {
                    text,
                    bytes: base..base + text.len(),
                    tokens: self.tokens(text, base),
                });
            }
            start = end;
        }
        result
    }

    fn tokens<'a>(&self, text: &'a str, base: usize) -> Vec<Token<'a>> {
        let mut tokens = Vec::new();
        let mut cursor = 0;
        for (start, end) in self.term_ranges(text) {
            if start < cursor {
                continue;
            }
            self.segment(&text[cursor..start], base + cursor, &mut tokens);
            tokens.push(Token {
                text: &text[start..end],
                bytes: base + start..base + end,
                kind: TokenKind::Term,
            });
            cursor = end;
        }
        self.segment(&text[cursor..], base + cursor, &mut tokens);
        tokens
    }

    fn term_ranges(&self, text: &str) -> BTreeMap<usize, usize> {
        let mut ranges = BTreeMap::new();
        let Some(terms) = &self.terms else {
            return ranges;
        };
        let english_boundaries = (self.locale == Locale::English).then(|| {
            text.split_word_bound_indices()
                .map(|(start, _)| start)
                .chain([text.len()])
                .collect::<Vec<_>>()
        });
        for matched in terms.find_overlapping_iter(text) {
            let (start, end) = (matched.start(), matched.end());
            let has_boundaries = match &english_boundaries {
                Some(boundaries) => {
                    boundaries.binary_search(&start).is_ok()
                        && boundaries.binary_search(&end).is_ok()
                }
                None => chinese_term_boundaries(text.as_bytes(), start, end),
            };
            if has_boundaries {
                let longest = ranges.entry(start).or_insert(end);
                *longest = (*longest).max(end);
            }
        }
        ranges
    }

    fn segment<'a>(&self, text: &'a str, base: usize, tokens: &mut Vec<Token<'a>>) {
        let mut push = |start, word: &'a str| {
            if !word.chars().all(char::is_whitespace) {
                tokens.push(Token {
                    text: word,
                    bytes: base + start..base + start + word.len(),
                    kind: if word.chars().any(char::is_alphanumeric) {
                        TokenKind::Word
                    } else {
                        TokenKind::Symbol
                    },
                });
            }
        };
        if self.locale == Locale::English {
            for (start, word) in text.split_word_bound_indices() {
                push(start, word);
            }
        } else if !text.is_empty() {
            for word in default_jieba().cut(text, true) {
                push(word.byte_start, word.word);
            }
        }
    }
}

fn continues_english_title(before: &str, after: &str) -> bool {
    let title = before
        .trim_end_matches([' ', '\t'])
        .rsplit(|character: char| !character.is_ascii_alphabetic() && character != '.')
        .next();
    matches!(
        title,
        Some("Mr." | "Mrs." | "Ms." | "Mx." | "Dr." | "Prof." | "Rev.")
    ) && after
        .trim_start_matches([' ', '\t'])
        .chars()
        .next()
        .is_some_and(char::is_uppercase)
}

fn chinese_term_boundaries(text: &[u8], start: usize, end: usize) -> bool {
    // Chinese terms may touch other Han text. Keep Latin words and numbers intact.
    let splits_ascii = |position: usize| {
        position > 0
            && position < text.len()
            && text[position - 1].is_ascii_alphanumeric()
            && text[position].is_ascii_alphanumeric()
    };
    !splits_ascii(start) && !splits_ascii(end)
}
