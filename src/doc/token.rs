//! Layer 2 tokenization of `nlp.doc.v1`. Tokens stay inside one sentence, and an opaque span
//! inside a sentence is always exactly one token. See `docs/doc-v1.md`.

use std::borrow::Cow;
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
    pub ja_lex: Option<crate::lexicon::ja::JaSegment>,
}

const CLITICS: &[&str] = &["'s", "'ll", "'re", "'ve", "'m", "'d"];
const REPEATABLE: &[char] = &[
    '.', '…', '‥', '⋯', '-', '—', '―', '─', '~', '〜', '～', '!', '?', '！', '？',
];
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
    "狼藉",
    "狼籍",
    "狼借",
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
    // 其 (2-char lexicalized words preserved from `其 + verb` de-gluing)
    "其实",
    "其中",
    "其他",
    "其它",
    "其次",
    "其余",
    "期间",
    "期待",
    "其间",
    "其后",
    "其前",
    "其内",
    "其外",
    "其上",
    "其下",
    "其旁",
    "其人",
    "其一",
    "其二",
    "其三",
    "其四",
    "其五",
    "其时",
    "其事",
];

/// Classifiers/measure/time suffixes allowed in 2-char `这<X>` / `那<X>` determiner tokens.
const DEMONSTRATIVE_CLASSIFIERS: &str = "个些种样么里儿边处时刻天日年夜晩晚次回趟阵番下点段截节集章卷篇部本册页封张幅件套双只条根枝枚颗粒块片滴团股道缕丝声句口盏壶杯碗盘锅瓶盒箱袋匹头峰座栋间家户门派宗路队群批帮伙类等项笔份场出幕台辆艘架尊具把柄支管面方位名般一";

/// Second characters in false 2-char `中 + <verb/prep/noun-frag>` merges under `normalize_traditional`.
const ZHONG_SPLIT_SECONDS: &str =
    "探方专專狂激射闪閃现現出飞飛落起看望凝露显顯带帶夹夾含透传傳响響变动變動转轉朝向往对對为為";

/// Rare radical/stroke infixes inserted between two Han chars by pirate anti-censorship scrapers.
pub(crate) fn is_scraper_infix(ch: char) -> bool {
    matches!(ch, '丅' | '丵' | '灬' | '堊' | '冇')
}

/// Strip single scraper anti-censorship infixes (`丅`, `丵`, `灬`, `堊`, `冇`) strictly sandwiched
/// between two non-infix Han characters, returning a zero-allocation `Cow::Borrowed` fast path
/// whenever no sandwiched infix is present.
pub(crate) fn strip_scraper_infix(s: &str) -> Cow<'_, str> {
    if s.len() < 9 || !s.chars().any(is_scraper_infix) {
        return Cow::Borrowed(s);
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut changed = false;
    for i in 0..chars.len() {
        let ch = chars[i];
        let sandwiched = i > 0
            && i + 1 < chars.len()
            && is_scraper_infix(ch)
            && is_han(chars[i - 1])
            && !is_scraper_infix(chars[i - 1])
            && is_han(chars[i + 1])
            && !is_scraper_infix(chars[i + 1]);
        if sandwiched {
            changed = true;
        } else {
            out.push(ch);
        }
    }
    if changed {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(s)
    }
}

/// The Han set of `nlp.doc.v1` tokens: CJK ideographs, compatibility ideographs, 々, 〆 and 〇.
/// Layer 1 keeps its own set (`segment::is_han`).
pub(super) fn is_han(ch: char) -> bool {
    matches!(
        ch,
        '\u{3400}'..='\u{4dbf}'
            | '\u{4e00}'..='\u{9fff}'
            | '\u{20000}'..='\u{2ceaf}'
            | '\u{f900}'..='\u{faff}'
            | '\u{3005}'
            | '\u{3006}'
            | '\u{3007}'
    )
}

/// Language of one token: `ja` with Kana or Japanese-context/exclusive Han, `ko` with Hangul,
/// `zh` with Han, `en` with another alphabetic letter, otherwise `und`.
pub(super) fn token_lang(text: &str, ja_context: bool) -> Lang {
    if text.chars().any(segment::is_kana)
        || (text.chars().any(is_han)
            && (ja_context || text.chars().any(segment::is_ja_exclusive_kanji)))
    {
        Lang::Ja
    } else if text.chars().any(segment::is_hangul) {
        Lang::Ko
    } else if text.chars().any(is_han) {
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

struct CachedAutomata {
    extra: Vec<String>,
    ja_ko_terms: Vec<String>,
    normalize_traditional: bool,
    surf_ac: Option<Arc<AhoCorasick>>,
    simp_ac: Option<Arc<AhoCorasick>>,
    ja_ko_ac: Option<Arc<AhoCorasick>>,
}

pub(super) struct TokenMatcher {
    surface: SurfaceTerms,
    simp_extra: Option<Arc<AhoCorasick>>,
    ja_ko_extra: Option<Arc<AhoCorasick>>,
    normalize_traditional: bool,
    use_doc_jieba: bool,
    first_party_zh_segmenter: bool,
}

impl TokenMatcher {
    /// Baseline matcher for `doc::line_tokens` (preserves exact parity with `crate::text::default_jieba`).
    pub(super) fn default_matcher() -> Self {
        Self {
            surface: SurfaceTerms::Default(terms()),
            simp_extra: None,
            ja_ko_extra: None,
            normalize_traditional: false,
            use_doc_jieba: false,
            first_party_zh_segmenter: false,
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

        let mut ja_ko_terms: Vec<String> = options
            .lexicon
            .iter()
            .chain(options.names.iter())
            .filter(|w| {
                !w.is_empty()
                    && !w.chars().any(char::is_whitespace)
                    && w.chars()
                        .any(|ch| is_han(ch) || segment::is_kana(ch) || segment::is_hangul(ch))
            })
            .cloned()
            .collect();
        ja_ko_terms.sort();
        ja_ko_terms.dedup();

        if extra.is_empty() && ja_ko_terms.is_empty() {
            return Self {
                surface: SurfaceTerms::Default(terms()),
                simp_extra: None,
                ja_ko_extra: None,
                normalize_traditional: options.normalize_traditional,
                use_doc_jieba: true,
                first_party_zh_segmenter: options.first_party_zh_segmenter,
            };
        }

        static CACHE: OnceLock<Mutex<Option<CachedAutomata>>> = OnceLock::new();
        let mutex = CACHE.get_or_init(|| Mutex::new(None));
        if let Ok(guard) = mutex.lock()
            && let Some(cached) = guard.as_ref()
            && cached.extra == extra
            && cached.ja_ko_terms == ja_ko_terms
            && cached.normalize_traditional == options.normalize_traditional
        {
            return Self {
                surface: match &cached.surf_ac {
                    Some(ac) => SurfaceTerms::Custom(Arc::clone(ac)),
                    None => SurfaceTerms::Default(terms()),
                },
                simp_extra: cached.simp_ac.as_ref().map(Arc::clone),
                ja_ko_extra: cached.ja_ko_ac.as_ref().map(Arc::clone),
                normalize_traditional: options.normalize_traditional,
                use_doc_jieba: true,
                first_party_zh_segmenter: options.first_party_zh_segmenter,
            };
        }

        let surf_ac = if extra.is_empty() {
            None
        } else {
            let mut patterns: Vec<String> = TAIWAN_TERMS.iter().map(|&s| s.to_owned()).collect();
            for w in &extra {
                if !patterns.contains(w) {
                    patterns.push(w.clone());
                }
            }
            Some(Arc::new(
                AhoCorasick::builder()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(&patterns)
                    .expect("surface term automaton builds"),
            ))
        };
        let simp_ac = if options.normalize_traditional && !extra.is_empty() {
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
        let ja_ko_ac = if ja_ko_terms.is_empty() {
            None
        } else {
            Some(Arc::new(
                AhoCorasick::builder()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(&ja_ko_terms)
                    .expect("ja/ko term automaton builds"),
            ))
        };

        if let Ok(mut guard) = mutex.lock() {
            *guard = Some(CachedAutomata {
                extra,
                ja_ko_terms,
                normalize_traditional: options.normalize_traditional,
                surf_ac: surf_ac.as_ref().map(Arc::clone),
                simp_ac: simp_ac.as_ref().map(Arc::clone),
                ja_ko_ac: ja_ko_ac.as_ref().map(Arc::clone),
            });
        }

        Self {
            surface: match surf_ac {
                Some(ac) => SurfaceTerms::Custom(ac),
                None => SurfaceTerms::Default(terms()),
            },
            simp_extra: simp_ac,
            ja_ko_extra: ja_ko_ac,
            normalize_traditional: options.normalize_traditional,
            use_doc_jieba: true,
            first_party_zh_segmenter: options.first_party_zh_segmenter,
        }
    }
}

/// Split one sentence into tokens with default options.
pub(super) fn tokenize(c: &[char], sentence: Span, inline: &[(Span, OpaqueKind)]) -> Vec<Piece> {
    tokenize_with_matcher(
        c,
        sentence,
        false,
        inline,
        &[],
        &TokenMatcher::default_matcher(),
    )
}

/// Split one sentence into tokens using caller `TokenMatcher`. `inline` holds the visible inline
/// opaque spans of the text (each one inside the sentence becomes a single piece), `silent` holds
/// suppressed inline noise spans that must be skipped without emitting a piece, and the text
/// between them is split by rule.
pub(super) fn tokenize_with_matcher(
    c: &[char],
    sentence: Span,
    ja_context: bool,
    inline: &[(Span, OpaqueKind)],
    silent: &[Span],
    matcher: &TokenMatcher,
) -> Vec<Piece> {
    let mut inside: Vec<(Span, Option<OpaqueKind>)> = inline
        .iter()
        .copied()
        .filter(|(s, _)| s.start_char >= sentence.start_char && s.end_char <= sentence.end_char)
        .map(|(s, k)| (s, Some(k)))
        .chain(
            silent
                .iter()
                .copied()
                .filter(|s| s.start_char < sentence.end_char && sentence.start_char < s.end_char)
                .map(|s| {
                    (
                        Span {
                            start_char: s.start_char.max(sentence.start_char),
                            end_char: s.end_char.min(sentence.end_char),
                        },
                        None,
                    )
                }),
        )
        .collect();
    inside.sort_by_key(|(s, _)| s.start_char);
    let mut out = Vec::new();
    let mut cursor = sentence.start_char;
    for (span, kind) in inside {
        if span.start_char < cursor {
            continue;
        }
        text_pieces(c, cursor, span.start_char, ja_context, matcher, &mut out);
        if let Some(kind) = kind {
            out.push(Piece {
                span,
                opaque: Some(kind),
                ja_lex: None,
            });
        }
        cursor = span.end_char;
    }
    text_pieces(c, cursor, sentence.end_char, ja_context, matcher, &mut out);
    out
}

fn text_pieces(
    c: &[char],
    start: usize,
    end: usize,
    ja_context: bool,
    matcher: &TokenMatcher,
    out: &mut Vec<Piece>,
) {
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

    let mut ja_ranges: Vec<(usize, usize, Option<crate::lexicon::ja::JaSegment>)> = Vec::new();
    let mut simple_ranges: Vec<(usize, usize)> = Vec::new();
    for (a, b, kind) in runs(&s, ja_context) {
        match kind {
            RunKind::Ja => ja_tokens(&s, a, b, matcher, &mut ja_ranges),
            RunKind::Zh => {
                simple_ranges.clear();
                han_tokens(&s, a, b, matcher, &mut simple_ranges);
                ja_ranges.extend(simple_ranges.drain(..).map(|(x, y)| (x, y, None)));
            }
            RunKind::Other => {
                simple_ranges.clear();
                other_tokens(&s, a, b, matcher, &mut simple_ranges);
                ja_ranges.extend(simple_ranges.drain(..).map(|(x, y)| (x, y, None)));
            }
        }
    }
    out.extend(ja_ranges.into_iter().map(|(a, b, ja_lex)| Piece {
        span: Span {
            start_char: start + index[a],
            end_char: start + index[b],
        },
        opaque: None,
        ja_lex,
    }));
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RunKind {
    Ja,
    Zh,
    Other,
}

/// Partition `s` into Japanese CJK+Kana runs, Chinese Han runs, and everything else.
fn runs(s: &str, ja_context: bool) -> Vec<(usize, usize, RunKind)> {
    let mut raw: Vec<(usize, usize, bool)> = Vec::new();
    for (b, ch) in s.char_indices() {
        let cjk_kana = is_han(ch) || segment::is_kana(ch);
        let end = b + ch.len_utf8();
        match raw.last_mut() {
            Some(last) if last.2 == cjk_kana => last.1 = end,
            _ => raw.push((b, end, cjk_kana)),
        }
    }
    raw.into_iter()
        .map(|(a, b, cjk_kana)| {
            let kind = if !cjk_kana {
                RunKind::Other
            } else if ja_context
                || s[a..b]
                    .chars()
                    .any(|ch| segment::is_kana(ch) || segment::is_ja_exclusive_kanji(ch))
            {
                RunKind::Ja
            } else {
                RunKind::Zh
            };
            (a, b, kind)
        })
        .collect()
}

/// Japanese CJK+Kana text: caller lexicon/names terms first, then first-party Japanese Viterbi segmentation.
fn ja_tokens(
    s: &str,
    a: usize,
    b: usize,
    matcher: &TokenMatcher,
    out: &mut Vec<(usize, usize, Option<crate::lexicon::ja::JaSegment>)>,
) {
    let run = &s[a..b];
    if let Some(ac) = &matcher.ja_ko_extra {
        let mut at = 0;
        for m in ac.find_iter(run) {
            ja_words(&run[at..m.start()], a + at, out);
            out.push((a + m.start(), a + m.end(), None));
            at = m.end();
        }
        ja_words(&run[at..], a + at, out);
    } else {
        ja_words(run, a, out);
    }
}

fn ja_words(
    stretch: &str,
    base: usize,
    out: &mut Vec<(usize, usize, Option<crate::lexicon::ja::JaSegment>)>,
) {
    if stretch.is_empty() {
        return;
    }
    let chars: Vec<char> = stretch.chars().collect();
    let mut byte_offs = Vec::with_capacity(chars.len() + 1);
    for (b, _) in stretch.char_indices() {
        byte_offs.push(b);
    }
    byte_offs.push(stretch.len());

    for seg in crate::lexicon::ja::segment_ja(&chars, 0) {
        let start_b = base + byte_offs[seg.span.start_char];
        let end_b = base + byte_offs[seg.span.end_char];
        out.push((start_b, end_b, Some(seg)));
    }
}

/// Non-CJK/Kana text: protect Hangul caller terms first when Hangul is present, then `word_ranges`.
fn other_tokens(
    s: &str,
    a: usize,
    b: usize,
    matcher: &TokenMatcher,
    out: &mut Vec<(usize, usize)>,
) {
    let run = &s[a..b];
    if let Some(ac) = &matcher.ja_ko_extra
        && run.chars().any(segment::is_hangul)
    {
        let mut at = 0;
        for m in ac.find_iter(run) {
            if run[m.start()..m.end()].chars().any(segment::is_hangul) {
                word_ranges(s, a + at, a + m.start(), out);
                out.push((a + m.start(), a + m.end()));
                at = m.end();
            }
        }
        word_ranges(s, a + at, b, out);
    } else {
        word_ranges(s, a, b, out);
    }
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
    if matcher.first_party_zh_segmenter {
        let chars: Vec<char> = text.chars().collect();
        let mut byte_offs = Vec::with_capacity(chars.len() + 1);
        for (b, _) in text.char_indices() {
            byte_offs.push(b);
        }
        byte_offs.push(text.len());
        for sp in crate::lexicon::zh::segment_zh(&chars, 0, &[]) {
            out.push((
                base + byte_offs[sp.start_char],
                base + byte_offs[sp.end_char],
            ));
        }
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

    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let m_len = chars.len();
    let mut orig_bytes = Vec::with_capacity(m_len + 1);
    for &(b, _) in &chars {
        orig_bytes.push(b);
    }
    orig_bytes.push(text.len());

    let mut proj_orig = Vec::with_capacity(m_len);
    let mut simp = String::with_capacity(text.len());
    for i in 0..m_len {
        let ch = chars[i].1;
        let skip = i > 0
            && i + 1 < m_len
            && is_scraper_infix(ch)
            && !is_scraper_infix(chars[i - 1].1)
            && !is_scraper_infix(chars[i + 1].1);
        if !skip {
            proj_orig.push(i);
            simp.push(t2s::to_simplified_char(ch));
        }
    }

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
                &proj_orig,
                &orig_bytes,
                base,
                out,
            );
            let cs = simp_byte_to_char[m.start()];
            let ce = simp_byte_to_char[m.end()];
            emit_projected_span(cs, ce, &proj_orig, &orig_bytes, base, out);
            at = m.end();
        }
        emit_jieba_deglued(
            &simp[at..],
            simp_byte_to_char[at],
            &proj_orig,
            &orig_bytes,
            base,
            out,
        );
    } else {
        emit_jieba_deglued(&simp, 0, &proj_orig, &orig_bytes, base, out);
    }
}

/// Emit the original byte range corresponding to projected char interval `[ps..pe)`, plus any
/// orphaned infix char that sat directly between `pe - 1` and `pe`, guaranteeing a zero-drift
/// contiguous byte partition of `text`.
fn emit_projected_span(
    ps: usize,
    pe: usize,
    proj_orig: &[usize],
    orig_bytes: &[usize],
    base: usize,
    out: &mut Vec<(usize, usize)>,
) {
    if ps >= pe {
        return;
    }
    let orig_start = proj_orig[ps];
    let orig_end = proj_orig[pe - 1] + 1;
    out.push((base + orig_bytes[orig_start], base + orig_bytes[orig_end]));
    if pe < proj_orig.len() && orig_end < proj_orig[pe] {
        out.push((
            base + orig_bytes[orig_end],
            base + orig_bytes[proj_orig[pe]],
        ));
    }
}

fn emit_jieba_deglued(
    simp_slice: &str,
    char_offset: usize,
    proj_orig: &[usize],
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
            emit_projected_span(cur_char, cur_char + 1, proj_orig, orig_bytes, base, out);
            emit_jieba_deglued(
                &w[first_len..],
                cur_char + 1,
                proj_orig,
                orig_bytes,
                base,
                out,
            );
        } else {
            emit_projected_span(
                cur_char,
                cur_char + w_chars,
                proj_orig,
                orig_bytes,
                base,
                out,
            );
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
    if c0 == '其' {
        return w_chars == 2 && !lexicalized_prefix_set().contains(w);
    }
    if c0 == '中' {
        return w_chars == 2
            && chars
                .next()
                .is_some_and(|c1| ZHONG_SPLIT_SECONDS.contains(c1));
    }
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

fn is_abbreviated_month(w: &str) -> bool {
    const MONTHS: &[&str] = &[
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    ];
    w.as_bytes().first().is_some_and(|b| b.is_ascii_uppercase())
        && MONTHS.iter().any(|m| w.eq_ignore_ascii_case(m))
}

/// Non-Han text: UAX 29 words, then clitics, hyphens, abbreviation dots and repeated marks.
fn word_ranges(s: &str, a0: usize, b0: usize, out: &mut Vec<(usize, usize)>) {
    if a0 >= b0 {
        return;
    }
    let mut raw = Vec::new();
    for (i, w) in s[a0..b0].split_word_bound_indices() {
        let a = a0 + i;
        if !w.chars().all(char::is_whitespace) {
            push_uax_segment(s, a, a + w.len(), &mut raw);
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
            && prev_text.chars().any(segment::is_hangul)
                == s[na..nb].chars().any(segment::is_hangul)
        {
            if let Some(last) = merged.last_mut() {
                last.1 = nb;
            }
            i += 1;
            continue;
        }
        // A dot after a listed abbreviation, a single capital, `etc`, a word with an inner dot,
        // or an abbreviated month followed by a token starting with an ASCII digit (`Nov. 13`).
        if text == "."
            && (segment::is_abbreviation(prev_text)
                || prev_text.eq_ignore_ascii_case("etc")
                || (has_letter(prev_text) && prev_text.contains('.') && !prev_text.ends_with('.'))
                || (is_abbreviated_month(prev_text)
                    && raw.get(i).is_some_and(|&(na, nb)| {
                        s[na..nb].starts_with(|c: char| c.is_ascii_digit())
                    })))
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

fn push_uax_segment(s: &str, a: usize, b: usize, raw: &mut Vec<(usize, usize)>) {
    let seg = &s[a..b];
    if !seg.chars().any(segment::is_hangul) {
        split_clitic(s, a, b, raw);
        return;
    }
    // UAX #29 treats Hangul as `ALetter` (WB9/WB10), gluing digits/Latin to Hangul (`3권의`).
    // Split at Hangul script transitions so `3` tags as `NUM` and `권의` as `Ko`.
    let mut sub_start = 0;
    let mut sub_hangul: Option<bool> = None;
    for (idx, ch) in seg.char_indices() {
        let h = segment::is_hangul(ch);
        if let Some(prev_h) = sub_hangul
            && prev_h != h
        {
            let slice_a = a + sub_start;
            let slice_b = a + idx;
            if prev_h {
                raw.push((slice_a, slice_b));
            } else {
                split_clitic(s, slice_a, slice_b, raw);
            }
            sub_start = idx;
        }
        sub_hangul = Some(h);
    }
    if sub_start < seg.len() {
        let slice_a = a + sub_start;
        if sub_hangul == Some(true) {
            raw.push((slice_a, b));
        } else {
            split_clitic(s, slice_a, b, raw);
        }
    }
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
