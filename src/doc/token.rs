//! Layer 2 tokenization of `nlp.doc.v1`. Tokens stay inside one sentence, and an opaque span
//! inside a sentence is always exactly one token. See `docs/doc-v1.md`.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

use aho_corasick::{AhoCorasick, MatchKind};
use jieba_rs::Jieba;
use unicode_segmentation::UnicodeSegmentation;

use super::{Lang, OpaqueKind, Options, segment, t2s};
use crate::Span;
use crate::text::{TAIWAN_TERMS, default_jieba};

/// One token before tagging. `opaque` is set when the token is an opaque span.
pub(super) struct Piece {
    pub span: Span,
    pub opaque: Option<OpaqueKind>,
}

const CLITICS: &[&str] = &["'s", "'ll", "'re", "'ve", "'m", "'d"];
const REPEATABLE: &[char] = &['.', '…', '-', '—', '~', '!', '?', '！', '？'];
const HYPHENS: &[&str] = &["-", "\u{2010}"];

/// Classical pronouns, address terms, and common genre nouns added to `doc_jieba()`.
/// Frequency `2000` sits above `freq("在")*freq("下")/N` (~1312) so standalone `在下` stays whole,
/// yet far below `freq("在")*freq("下雨")/freq("雨")` (~78680) so `還在下雨` keeps `在` + `下雨`.
const DOC_EXTRA_WORDS: &[&str] = &[
    "在下",
    "妾身",
    "老夫",
    "老身",
    "貧道",
    "贫道",
    "本座",
    "晚輩",
    "晚辈",
    "前輩",
    "前辈",
    "道友",
    "元嬰",
    "元婴",
    "筑基",
    "築基",
    "筑基丹",
    "築基丹",
    "結丹",
    "结丹",
    "化神",
    "煉虛",
    "炼虚",
    "大乘",
    "神識",
    "神识",
    "靈力",
    "灵力",
    "法寶",
    "法宝",
    "靈寶",
    "灵宝",
    "儲物袋",
    "储物袋",
    "丹藥",
    "丹药",
    "功法",
    "口訣",
    "口诀",
    "陣法",
    "阵法",
    "老者",
    "大漢",
    "大汉",
    "婦人",
    "妇人",
];

/// Curated Simplified lexicalized words starting with coverb/preposition or demonstrative chars
/// that must NOT be split by the coverb/demonstrative de-gluing guard.
const LEXICALIZED_PREFIX_WORDS: &[&str] = &[
    // 将
    "将军",
    "将领",
    "将来",
    "将士",
    "将要",
    "将近",
    "将令",
    "将校",
    "将官",
    "将帅",
    "将才",
    "将门",
    "将息",
    "将相",
    "将星",
    "将就",
    "将信将疑",
    "将计就计",
    "将错就错",
    "将心比心",
    "将功赎罪",
    "将功补过",
    // 把
    "把握",
    "把头",
    "把手",
    "把持",
    "把守",
    "把门",
    "把戏",
    "把关",
    "把酒",
    "把子",
    "把柄",
    "把玩",
    "把住",
    "把儿",
    "把脉",
    "把风",
    "把舵",
    "把盏",
    "把兄弟",
    "把持不定",
    "把酒言欢",
    // 被
    "被迫",
    "被捕",
    "被动",
    "被俘",
    "被子",
    "被告",
    "被告人",
    "被窝",
    "被褥",
    "被害",
    "被害人",
    "被害者",
    "被盗",
    "被囚",
    "被单",
    "被套",
    "被服",
    "被控",
    "被动式",
    // 对 (note: `对此` intentionally omitted so `對此`/`对此` splits into `對`+`此`)
    "对于",
    "对方",
    "对象",
    "对外",
    "对手",
    "对付",
    "对待",
    "对抗",
    "对面",
    "对话",
    "对立",
    "对比",
    "对应",
    "对不起",
    "对称",
    "对策",
    "对准",
    "对了",
    "对照",
    "对口",
    "对敌",
    "对峙",
    "对联",
    "对岸",
    "对阵",
    "对内",
    "对得起",
    "对答",
    "对头",
    "对接",
    "对白",
    "对视",
    "对流",
    "对数",
    "对症",
    "对决",
    "对偶",
    "对不住",
    "对立面",
    "对子",
    "对虾",
    "对开",
    "对垒",
    "对冲",
    "对弈",
    "对位",
    "对攻",
    "对饮",
    "对劲",
    "对掌",
    "对招",
    "对望",
    "对眼",
    "对外开放",
    "对症下药",
    "对答如流",
    // 当
    "当时",
    "当然",
    "当年",
    "当地",
    "当下",
    "当即",
    "当选",
    "当前",
    "当真",
    "当中",
    "当局",
    "当初",
    "当日",
    "当作",
    "当今",
    "当代",
    "当天",
    "当场",
    "当事人",
    "当做",
    "当面",
    "当成",
    "当着",
    "当晚",
    "当先",
    "当儿",
    "当兵",
    "当众",
    "当地人",
    "当头",
    "当官",
    "当夜",
    "当家",
    "当值",
    "当政",
    "当心",
    "当铺",
    "当权",
    "当量",
    "当归",
    "当月",
    "当街",
    "当空",
    "当差",
    "当家作主",
    "当务之急",
    "当之无愧",
    "当机立断",
    "当仁不让",
    "当头棒喝",
    // 从
    "从而",
    "从事",
    "从此",
    "从来",
    "从未",
    "从不",
    "从业",
    "从前",
    "从小",
    "从中",
    "从来不",
    "从没",
    "从容",
    "从头",
    "从属",
    "从严",
    "从政",
    "从今",
    "从军",
    "从业者",
    "从轻",
    "从重",
    "从命",
    "从速",
    "从宽",
    "从简",
    "从众",
    "从旁",
    "从容不迫",
    "从未有过",
    "从头到尾",
    "从今以后",
    "从天而降",
    "从头至尾",
    "从长计议",
    "从此以后",
    "从小到大",
    "从早到晚",
    "从无到有",
    "从古到今",
    "从古至今",
    // 向
    "向前",
    "向后",
    "向来",
    "向往",
    "向东",
    "向西",
    "向南",
    "向北",
    "向上",
    "向下",
    "向着",
    "向阳",
    "向导",
    "向量",
    "向日葵",
    "向善",
    "向斜",
    "向心力",
    "向背",
    "向心",
    "向晚",
    "向内",
    "向外",
    "向左",
    "向右",
    // 朝
    "朝廷",
    "朝鲜",
    "朝着",
    "朝阳",
    "朝代",
    "朝政",
    "朝野",
    "朝臣",
    "朝服",
    "朝霞",
    "朝见",
    "朝觐",
    "朝向",
    "朝夕",
    "朝拜",
    "朝外",
    "朝日",
    "朝贡",
    "朝圣",
    "朝里",
    "朝气",
    "朝纲",
    "朝南",
    "朝北",
    "朝东",
    "朝西",
    "朝上",
    "朝下",
    "朝前",
    "朝后",
    "朝内",
    "朝夕相处",
    "朝气蓬勃",
    "朝不保夕",
    // 往
    "往往",
    "往来",
    "往前",
    "往后",
    "往日",
    "往事",
    "往年",
    "往返",
    "往常",
    "往回",
    "往昔",
    "往复",
    "往届",
    "往时",
    "往里",
    "往外",
    "往上",
    "往下",
    "往左",
    "往右",
    "往东",
    "往西",
    "往南",
    "往北",
    // 给
    "给予",
    "给出",
    "给事中",
    "给与",
    "给养",
    "给钱",
    "给定",
    "给付",
    "给水",
    "给力",
    "给面子",
    // 连
    "连续",
    "连接",
    "连忙",
    "连连",
    "连同",
    "连夜",
    "连声",
    "连长",
    "连锁",
    "连年",
    "连队",
    "连带",
    "连日",
    "连任",
    "连环",
    "连续性",
    "连结",
    "连累",
    "连绵",
    "连发",
    "连线",
    "连通",
    "连贯",
    "连珠",
    "连胜",
    "连败",
    "连载",
    "连天",
    "连城",
    "连坐",
    "连战",
    "连片",
    "连杆",
    "连衣裙",
    "连字号",
    "连成一片",
    "连日来",
    "连续不断",
    "连绵不断",
    "连锁店",
    "连续剧",
    "连环画",
    "连锁反应",
    // 和
    "和平",
    "和尚",
    "和谐",
    "和约",
    "和解",
    "和谈",
    "和睦",
    "和善",
    "和声",
    "和议",
    "和蔼",
    "和亲",
    "和暖",
    "和文",
    "和弦",
    "和顺",
    "和缓",
    "和服",
    "和歌",
    "和煦",
    "和气",
    "和风",
    "和易",
    "和稀泥",
    "和平共处",
    "和衷共济",
    "和颜悦色",
    "和蔼可亲",
    "和盘托出",
    "和睦相处",
    // 与
    "与其",
    "与此同时",
    "与会",
    "与否",
    "与会者",
    "与非",
    "与众不同",
    "与世隔绝",
    "与时俱进",
    "与日俱增",
    "与生俱来",
    "与其说",
    "与世无争",
    "与人为善",
    "与世长辞",
    "与虎谋皮",
    // 跟
    "跟随",
    "跟踪",
    "跟着",
    "跟前",
    "跟上",
    "跟进",
    "跟不上",
    "跟头",
    "跟风",
    "跟班",
    "跟斗",
    "跟脚",
    "跟前儿",
    "跟得上",
    // 这 / 那 (2-char & multi-char lexicalized forms)
    "这个",
    "这样",
    "这些",
    "这种",
    "这里",
    "这么",
    "这时",
    "这次",
    "这位",
    "这儿",
    "这般",
    "这话",
    "这天",
    "这边",
    "这点",
    "这回",
    "这下",
    "这一",
    "这方",
    "这时候",
    "这方面",
    "这会儿",
    "这一下",
    "这部分",
    "这样一来",
    "这样的话",
    "这就是说",
    "这一来",
    "这么着",
    "这么些",
    "这么晚",
    "这下子",
    "这点儿",
    "这样儿",
    "那么",
    "那些",
    "那个",
    "那里",
    "那样",
    "那时",
    "那种",
    "那天",
    "那儿",
    "那位",
    "那边",
    "那末",
    "那日",
    "那年",
    "那次",
    "那般",
    "那晚",
    "那时候",
    "那方面",
    "那会儿",
    "那部分",
    "那一下",
    "那阵子",
    "那样的话",
    "那么些",
    "那不勒斯",
    // 此
    "此外",
    "此时",
    "此后",
    "此次",
    "此刻",
    "此事",
    "此处",
    "此前",
    "此间",
    "此举",
    "此地",
    "此类",
    "此言",
    "此话",
    "此物",
    "此等",
    "此番",
    "此行",
    "此生",
    "此际",
    "此时此刻",
    "此起彼伏",
    "此起彼落",
    "此情此景",
    "此消彼长",
];

/// Classifiers/measure/time suffixes allowed in 2-char `这<X>` / `那<X>` determiner tokens.
const DEMONSTRATIVE_CLASSIFIERS: &str = "个些种样么里儿边处时刻天日年夜晩晚次回趟阵番下点段截节集章卷篇部本册页封张幅件套双只条根枝枚颗粒块片滴团股道缕丝声句口盏壶杯碗盘锅瓶盒箱袋匹头峰座栋间家户门派宗路队群批帮伙类等项笔份场出幕台辆艘架尊具把柄支管面方位名般一";

/// The Han set of `nlp.doc.v1` tokens: CJK ideographs, compatibility ideographs, 々 and 〇.
/// Layer 1 keeps its own set (`segment::is_han`).
pub(super) fn is_han(ch: char) -> bool {
    matches!(
        ch,
        '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}' | '\u{3005}' | '\u{3007}'
    )
}

/// Language of one token: zh with a Han char, en with a Latin letter, otherwise und.
pub(super) fn token_lang(text: &str) -> Lang {
    if text.chars().any(is_han) {
        Lang::Zh
    } else if text.chars().any(char::is_alphabetic) {
        Lang::En
    } else {
        Lang::Und
    }
}

fn doc_jieba() -> &'static Jieba {
    static DOC_JIEBA: OnceLock<Jieba> = OnceLock::new();
    DOC_JIEBA.get_or_init(|| {
        let mut j = default_jieba().clone();
        for &w in DOC_EXTRA_WORDS {
            j.add_word(w, Some(2000), None);
        }
        j
    })
}

fn lexicalized_prefix_set() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| LEXICALIZED_PREFIX_WORDS.iter().copied().collect())
}

fn terms() -> &'static AhoCorasick {
    static TERMS: OnceLock<AhoCorasick> = OnceLock::new();
    TERMS.get_or_init(|| {
        AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(TAIWAN_TERMS)
            .expect("the Taiwan term list builds")
    })
}

enum SurfaceTerms {
    Default(&'static AhoCorasick),
    Custom(Arc<AhoCorasick>),
}

impl SurfaceTerms {
    fn as_ac(&self) -> &AhoCorasick {
        match self {
            Self::Default(ac) => ac,
            Self::Custom(ac) => ac,
        }
    }
}

type CachedAutomata = (
    Vec<String>,
    bool,
    Arc<AhoCorasick>,
    Option<Arc<AhoCorasick>>,
);

pub(super) struct TokenMatcher {
    surface: SurfaceTerms,
    simp_extra: Option<Arc<AhoCorasick>>,
    normalize_traditional: bool,
    use_doc_jieba: bool,
}

impl TokenMatcher {
    /// Baseline matcher for `doc::line_tokens` (preserves exact parity with `crate::text::default_jieba`).
    pub(super) fn default_matcher() -> Self {
        Self {
            surface: SurfaceTerms::Default(terms()),
            simp_extra: None,
            normalize_traditional: false,
            use_doc_jieba: false,
        }
    }

    pub(super) fn from_options(options: &Options) -> Self {
        let mut extra: Vec<String> = options
            .lexicon
            .iter()
            .chain(options.names.iter())
            .filter(|w| !w.is_empty() && w.chars().all(is_han))
            .cloned()
            .collect();
        extra.sort();
        extra.dedup();
        if extra.is_empty() {
            return Self {
                surface: SurfaceTerms::Default(terms()),
                simp_extra: None,
                normalize_traditional: options.normalize_traditional,
                use_doc_jieba: true,
            };
        }

        static CACHE: OnceLock<Mutex<Option<CachedAutomata>>> = OnceLock::new();
        let mutex = CACHE.get_or_init(|| Mutex::new(None));
        if let Ok(guard) = mutex.lock()
            && let Some((cached_extra, cached_norm, surf_ac, simp_ac)) = guard.as_ref()
            && *cached_extra == extra
            && *cached_norm == options.normalize_traditional
        {
            return Self {
                surface: SurfaceTerms::Custom(Arc::clone(surf_ac)),
                simp_extra: simp_ac.as_ref().map(Arc::clone),
                normalize_traditional: options.normalize_traditional,
                use_doc_jieba: true,
            };
        }

        let mut patterns: Vec<String> = TAIWAN_TERMS.iter().map(|&s| s.to_owned()).collect();
        for w in &extra {
            if !patterns.contains(w) {
                patterns.push(w.clone());
            }
        }
        let surf_ac = Arc::new(
            AhoCorasick::builder()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&patterns)
                .expect("surface term automaton builds"),
        );
        let simp_ac = if options.normalize_traditional {
            let mut simp_pats: Vec<String> = extra
                .iter()
                .map(|w| w.chars().map(t2s::to_simplified_char).collect())
                .collect();
            simp_pats.sort();
            simp_pats.dedup();
            Some(Arc::new(
                AhoCorasick::builder()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(&simp_pats)
                    .expect("simplified term automaton builds"),
            ))
        } else {
            None
        };

        if let Ok(mut guard) = mutex.lock() {
            *guard = Some((
                extra,
                options.normalize_traditional,
                Arc::clone(&surf_ac),
                simp_ac.as_ref().map(Arc::clone),
            ));
        }

        Self {
            surface: SurfaceTerms::Custom(surf_ac),
            simp_extra: simp_ac,
            normalize_traditional: options.normalize_traditional,
            use_doc_jieba: true,
        }
    }
}

/// Split one sentence into tokens with default options.
pub(super) fn tokenize(c: &[char], sentence: Span, inline: &[(Span, OpaqueKind)]) -> Vec<Piece> {
    tokenize_with_matcher(c, sentence, inline, &TokenMatcher::default_matcher())
}

/// Split one sentence into tokens using caller `TokenMatcher`. `inline` holds the opaque spans
/// of the text; each one inside the sentence becomes a single piece, and the text between them
/// is split by rule.
pub(super) fn tokenize_with_matcher(
    c: &[char],
    sentence: Span,
    inline: &[(Span, OpaqueKind)],
    matcher: &TokenMatcher,
) -> Vec<Piece> {
    let mut inside: Vec<(Span, OpaqueKind)> = inline
        .iter()
        .copied()
        .filter(|(s, _)| s.start_char >= sentence.start_char && s.end_char <= sentence.end_char)
        .collect();
    inside.sort_by_key(|(s, _)| s.start_char);
    let mut out = Vec::new();
    let mut cursor = sentence.start_char;
    for (span, kind) in inside {
        if span.start_char < cursor {
            continue;
        }
        text_pieces(c, cursor, span.start_char, matcher, &mut out);
        out.push(Piece {
            span,
            opaque: Some(kind),
        });
        cursor = span.end_char;
    }
    text_pieces(c, cursor, sentence.end_char, matcher, &mut out);
    out
}

fn text_pieces(c: &[char], start: usize, end: usize, matcher: &TokenMatcher, out: &mut Vec<Piece>) {
    if start >= end {
        return;
    }
    let s: String = c[start..end].iter().collect();
    // Map each byte offset of `s` to its char offset.
    let mut index = vec![0; s.len() + 1];
    let mut n = 0;
    for (b, ch) in s.char_indices() {
        index[b..b + ch.len_utf8()].fill(n);
        n += 1;
    }
    index[s.len()] = n;

    let mut ranges = Vec::new();
    for (a, b, han) in runs(&s) {
        if han {
            han_tokens(&s, a, b, matcher, &mut ranges);
        } else {
            word_ranges(&s, a, b, &mut ranges);
        }
    }
    out.extend(ranges.into_iter().map(|(a, b)| Piece {
        span: Span {
            start_char: start + index[a],
            end_char: start + index[b],
        },
        opaque: None,
    }));
}

/// Maximal runs of Han characters and of everything else, as (start, end, is_han) bytes.
fn runs(s: &str) -> Vec<(usize, usize, bool)> {
    let mut out: Vec<(usize, usize, bool)> = Vec::new();
    for (b, ch) in s.char_indices() {
        let han = is_han(ch);
        let end = b + ch.len_utf8();
        match out.last_mut() {
            Some(last) if last.2 == han => last.1 = end,
            _ => out.push((b, end, han)),
        }
    }
    out
}

/// Han text: Taiwan + caller lexicon/names terms first, then jieba on the stretches between them.
fn han_tokens(s: &str, a: usize, b: usize, matcher: &TokenMatcher, out: &mut Vec<(usize, usize)>) {
    let run = &s[a..b];
    let mut at = 0;
    for m in matcher.surface.as_ac().find_iter(run) {
        han_words(&run[at..m.start()], a + at, matcher, out);
        out.push((a + m.start(), a + m.end()));
        at = m.end();
    }
    han_words(&run[at..], a + at, matcher, out);
}

fn han_words(text: &str, base: usize, matcher: &TokenMatcher, out: &mut Vec<(usize, usize)>) {
    if text.is_empty() {
        return;
    }
    if !matcher.normalize_traditional {
        let jieba = if matcher.use_doc_jieba {
            doc_jieba()
        } else {
            default_jieba()
        };
        for word in jieba.cut(text, true) {
            let start = base + word.byte_start;
            out.push((start, start + word.word.len()));
        }
        return;
    }

    let mut orig_bytes = Vec::with_capacity(text.len() / 3 + 2);
    let mut simp = String::with_capacity(text.len());
    for (b, ch) in text.char_indices() {
        orig_bytes.push(b);
        simp.push(t2s::to_simplified_char(ch));
    }
    orig_bytes.push(text.len());

    if let Some(simp_ac) = &matcher.simp_extra {
        let mut simp_byte_to_char = vec![0usize; simp.len() + 1];
        let mut ci = 0usize;
        for (sb, sch) in simp.char_indices() {
            simp_byte_to_char[sb..sb + sch.len_utf8()].fill(ci);
            ci += 1;
        }
        simp_byte_to_char[simp.len()] = ci;

        let mut at = 0usize;
        for m in simp_ac.find_iter(&simp) {
            emit_jieba_deglued(
                &simp[at..m.start()],
                simp_byte_to_char[at],
                &orig_bytes,
                base,
                out,
            );
            let cs = simp_byte_to_char[m.start()];
            let ce = simp_byte_to_char[m.end()];
            out.push((base + orig_bytes[cs], base + orig_bytes[ce]));
            at = m.end();
        }
        emit_jieba_deglued(&simp[at..], simp_byte_to_char[at], &orig_bytes, base, out);
    } else {
        emit_jieba_deglued(&simp, 0, &orig_bytes, base, out);
    }
}

fn emit_jieba_deglued(
    simp_slice: &str,
    char_offset: usize,
    orig_bytes: &[usize],
    base: usize,
    out: &mut Vec<(usize, usize)>,
) {
    if simp_slice.is_empty() {
        return;
    }
    let mut cur_char = char_offset;
    for word in doc_jieba().cut(simp_slice, true) {
        let w = word.word;
        let w_chars = w.chars().count();
        if should_split_coverb_or_dem(w, w_chars) {
            let first_len = w.chars().next().expect("w_chars >= 2").len_utf8();
            out.push((base + orig_bytes[cur_char], base + orig_bytes[cur_char + 1]));
            emit_jieba_deglued(&w[first_len..], cur_char + 1, orig_bytes, base, out);
        } else {
            out.push((
                base + orig_bytes[cur_char],
                base + orig_bytes[cur_char + w_chars],
            ));
        }
        cur_char += w_chars;
    }
}

fn should_split_coverb_or_dem(w: &str, w_chars: usize) -> bool {
    if w_chars < 2 {
        return false;
    }
    let mut chars = w.chars();
    let Some(c0) = chars.next() else {
        return false;
    };
    let is_prefix = matches!(
        c0,
        '将' | '把'
            | '被'
            | '对'
            | '当'
            | '从'
            | '向'
            | '朝'
            | '往'
            | '给'
            | '连'
            | '和'
            | '与'
            | '跟'
            | '这'
            | '那'
            | '此'
    );
    if !is_prefix {
        return false;
    }
    if lexicalized_prefix_set().contains(w) {
        return false;
    }
    if matches!(c0, '这' | '那')
        && w_chars == 2
        && let Some(c1) = chars.next()
        && DEMONSTRATIVE_CLASSIFIERS.contains(c1)
    {
        return false;
    }
    true
}

/// Non-Han text: UAX 29 words, then clitics, hyphens, abbreviation dots and repeated marks.
fn word_ranges(s: &str, a0: usize, b0: usize, out: &mut Vec<(usize, usize)>) {
    let mut raw = Vec::new();
    for (i, w) in s[a0..b0].split_word_bound_indices() {
        let a = a0 + i;
        if !w.chars().all(char::is_whitespace) {
            split_clitic(s, a, a + w.len(), &mut raw);
        }
    }

    let mut merged: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let (a, b) = raw[i];
        let text = &s[a..b];
        let prev = merged.last().copied().filter(|&(_, pb)| pb == a);
        i += 1;
        let Some((pa, _)) = prev else {
            merged.push((a, b));
            continue;
        };
        let prev_text = &s[pa..a];
        if HYPHENS.contains(&text)
            && has_letter(prev_text)
            && let Some(&(na, nb)) = raw.get(i).filter(|&&(na, _)| na == b)
            && has_letter(&s[na..nb])
        {
            if let Some(last) = merged.last_mut() {
                last.1 = nb;
            }
            i += 1;
            continue;
        }
        // A dot after a listed abbreviation, a single capital, `etc`, or a word with an inner dot.
        if text == "."
            && (segment::is_abbreviation(prev_text)
                || prev_text.eq_ignore_ascii_case("etc")
                || (has_letter(prev_text) && prev_text.contains('.') && !prev_text.ends_with('.')))
        {
            if let Some(last) = merged.last_mut() {
                last.1 = b;
            }
            continue;
        }
        // A run of one repeatable char: the previous token is made of that same char only.
        let mut chars = text.chars();
        if let (Some(ch), None) = (chars.next(), chars.next())
            && REPEATABLE.contains(&ch)
            && prev_text.chars().all(|p| p == ch)
        {
            if let Some(last) = merged.last_mut() {
                last.1 = b;
            }
            continue;
        }
        merged.push((a, b));
    }
    out.extend(merged);
}

fn has_letter(text: &str) -> bool {
    text.chars().any(char::is_alphabetic)
}

/// Split English clitics off a word: `didn't` gives `did` and `n't`, `They'll` gives `They` and
/// `'ll`. Other words are kept whole.
fn split_clitic(s: &str, a: usize, b: usize, out: &mut Vec<(usize, usize)>) {
    let word = &s[a..b];
    let apostrophe = word
        .char_indices()
        .rev()
        .find(|&(_, ch)| ch == '\'' || ch == '’')
        .map(|(i, _)| i);
    // The stem must be non-empty, so the apostrophe is never the first char.
    if let Some(p) = apostrophe.filter(|&p| p > 0) {
        let suffix = word[p..].to_lowercase().replace('’', "'");
        let before = &word[..p];
        // `n't` alone has no stem, so the `n` must be preceded by something.
        if suffix == "'t" && before.ends_with(['n', 'N']) && before.len() > 1 {
            let at = p - 1;
            out.push((a, a + at));
            out.push((a + at, b));
            return;
        }
        if CLITICS.contains(&suffix.as_str()) {
            out.push((a, a + p));
            out.push((a + p, b));
            return;
        }
    }
    out.push((a, b));
}
