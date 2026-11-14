//! Track A: Deterministic, zero-model orthographic and lexical variant conversion
//! (`zh-Hant` <-> `zh-Hans` and `en-US` <-> `en-GB`) over `nlp.doc.v1` tokens, UPOS tags,
//! and the embedded `faberline_nlp::lexicon` (`FNLPLX01`).

use faberline_nlp::Span;
use faberline_nlp::doc::{Token, Upos};
use faberline_nlp::lexicon::{Variant, lookup_en, lookup_zh, zh};

use crate::schema::TargetVariant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZhScriptOrientation {
    Traditional,
    Simplified,
    Neutral,
}

/// Shared single characters that exist as distinct valid lexemes in Traditional Chinese (`后`
/// empress, `里` mile/neighborhood, `干` shield/stem/interfere, `面` face/surface, `系` department,
/// `台` platform/Taiwan, `余` surname I, `表` table/surface, `借` borrow, `斗` dipper, `谷` valley).
fn is_shared_polysemous_hant_char(c: char) -> bool {
    matches!(
        c,
        '后' | '里'
            | '干'
            | '面'
            | '系'
            | '台'
            | '余'
            | '表'
            | '借'
            | '斗'
            | '谷'
            | '云'
            | '叶'
            | '征'
            | '钟'
            | '卷'
            | '合'
            | '同'
            | '向'
            | '回'
    )
}

/// Detect whether a Chinese slice is predominantly Traditional, Simplified, or script-neutral.
pub fn detect_zh_orientation(text: &str) -> ZhScriptOrientation {
    let mut hant_votes = 0usize;
    let mut hans_votes = 0usize;
    let mut buf = [0u8; 4];

    for ch in text.chars() {
        if !is_han(ch) || is_shared_polysemous_hant_char(ch) {
            continue;
        }
        let s = ch.encode_utf8(&mut buf);
        if let Some(hit) = lookup_zh(s)
            && let Some(cp) = hit.counterpart.as_deref()
            && cp != s
        {
            match hit.variant {
                Variant::ZhHant => hant_votes += 1,
                Variant::ZhHans => hans_votes += 1,
                _ => {}
            }
        }
    }

    if hant_votes > hans_votes {
        ZhScriptOrientation::Traditional
    } else if hans_votes > hant_votes {
        ZhScriptOrientation::Simplified
    } else {
        ZhScriptOrientation::Neutral
    }
}

/// Convert a Chinese text string to `target` (`zh-Hant` or `zh-Hans`) while preserving `[[PHk]]`
/// span-lock placeholders verbatim.
pub(crate) fn convert_zh_text_to_target(text: &str, target: TargetVariant) -> String {
    if text.is_empty() || !target.is_zh() {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find("[[PH") {
        out.push_str(&convert_fragment_variant(&rest[..open], target));
        let after = &rest[open + 4..];
        if let Some(close) = after.find("]]") {
            out.push_str(&rest[open..open + 4 + close + 2]);
            rest = &after[close + 2..];
        } else {
            out.push_str("[[PH");
            rest = after;
        }
    }
    out.push_str(&convert_fragment_variant(rest, target));
    out
}

/// Convert an arbitrary standalone fragment (e.g. a heading subtitle or glossary surface) to
/// `target` when `target` is a Chinese or English variant.
pub fn convert_fragment_variant(fragment: &str, target: TargetVariant) -> String {
    if fragment.is_empty() {
        return String::new();
    }
    if target.is_zh() {
        let chars: Vec<char> = fragment.chars().collect();
        let orientation = detect_zh_orientation(fragment);
        let spans = zh::segment_zh(&chars, 0, &[]);
        let mut out = String::with_capacity(fragment.len());
        let mut cursor = 0usize;
        for sp in spans {
            if sp.start_char > cursor {
                for &c in &chars[cursor..sp.start_char] {
                    out.push(c);
                }
            }
            let surf: String = chars[sp.start_char..sp.end_char].iter().collect();
            let prev_char = sp
                .start_char
                .checked_sub(1)
                .and_then(|i| chars.get(i).copied());
            let next_char = chars.get(sp.end_char).copied();
            out.push_str(&convert_zh_word(
                &surf,
                None,
                prev_char,
                next_char,
                orientation,
                target,
            ));
            cursor = sp.end_char;
        }
        if cursor < chars.len() {
            for &c in &chars[cursor..] {
                out.push(c);
            }
        }
        out
    } else if target.is_en() {
        convert_en_fragment(fragment, target)
    } else {
        fragment.to_owned()
    }
}

/// Convert a sentence span `[sent_span.start_char, sent_span.end_char)` using `nlp.doc.v1` tokens
/// and UPOS tags while preserving exact non-locked spans (`locked_spans` are copied from
/// `locked_replacements` or handled by `SpanLockTable`).
pub fn convert_sentence_variant(
    chars: &[char],
    sent_span: Span,
    sent_tokens: &[&Token],
    target: TargetVariant,
    source_hint: Option<TargetVariant>,
) -> String {
    if sent_span.start_char >= sent_span.end_char {
        return String::new();
    }
    let raw: String = chars[sent_span.start_char..sent_span.end_char]
        .iter()
        .collect();

    if target.is_zh() {
        let orientation = match source_hint {
            Some(TargetVariant::ZhHant) => ZhScriptOrientation::Traditional,
            Some(TargetVariant::ZhHans) => ZhScriptOrientation::Simplified,
            _ => detect_zh_orientation(&raw),
        };
        // Re-segment contiguous Han runs with the first-party bilingual Viterbi segmenter so
        // Simplified compounds (`软件`, `头发`, `干净`, `王后`, `三天后`) never split into ambiguous
        // single characters even if the caller ran `analyze_document` with default `jieba` options,
        // while looking up any matching `nlp.doc.v1` token UPOS by span!
        let local_chars = &chars[sent_span.start_char..sent_span.end_char];
        let seg_spans = zh::segment_zh(local_chars, sent_span.start_char, &[]);
        let mut out = String::with_capacity(raw.len());
        let mut cursor = sent_span.start_char;

        for sp in seg_spans {
            if sp.start_char > cursor {
                for &c in &chars[cursor..sp.start_char] {
                    out.push(c);
                }
            }
            let surf: String = chars[sp.start_char..sp.end_char].iter().collect();
            let tok_upos = sent_tokens
                .iter()
                .find(|t| t.span == sp)
                .and_then(|t| t.upos);
            let prev_char = if sp.start_char > sent_span.start_char {
                chars.get(sp.start_char - 1).copied()
            } else {
                None
            };
            let next_char = if sp.end_char < sent_span.end_char {
                chars.get(sp.end_char).copied()
            } else {
                None
            };
            if surf.chars().any(is_han) {
                out.push_str(&convert_zh_word(
                    &surf,
                    tok_upos,
                    prev_char,
                    next_char,
                    orientation,
                    target,
                ));
            } else if target.is_en() {
                out.push_str(&convert_en_word(&surf, tok_upos, None, target));
            } else {
                out.push_str(&surf);
            }
            cursor = sp.end_char;
        }
        if cursor < sent_span.end_char {
            for &c in &chars[cursor..sent_span.end_char] {
                out.push(c);
            }
        }
        out
    } else if target.is_en() {
        let mut out = String::with_capacity(raw.len());
        let mut cursor = sent_span.start_char;
        for (idx, tok) in sent_tokens.iter().enumerate() {
            if tok.span.end_char <= sent_span.start_char
                || tok.span.start_char >= sent_span.end_char
            {
                continue;
            }
            let s = tok.span.start_char.max(cursor);
            let e = tok.span.end_char.min(sent_span.end_char);
            if s > cursor {
                for &c in &chars[cursor..s] {
                    out.push(c);
                }
            }
            let surf: String = chars[s..e].iter().collect();
            let prev_tok_info = idx.checked_sub(1).map(|pi| {
                let pt = sent_tokens[pi];
                let ps: String = chars[pt.span.start_char..pt.span.end_char].iter().collect();
                (ps, pt.upos)
            });
            let prev_ref = prev_tok_info.as_ref().map(|(ps, pu)| (ps.as_str(), *pu));
            out.push_str(&convert_en_word(&surf, tok.upos, prev_ref, target));
            cursor = e;
        }
        if cursor < sent_span.end_char {
            for &c in &chars[cursor..sent_span.end_char] {
                out.push(c);
            }
        }
        out
    } else {
        raw
    }
}

/// Convert a single segmented Chinese word between `zh-Hant` and `zh-Hans`.
pub fn convert_zh_word(
    surface: &str,
    upos: Option<Upos>,
    prev_char: Option<char>,
    next_char: Option<char>,
    orientation: ZhScriptOrientation,
    target: TargetVariant,
) -> String {
    let char_count = surface.chars().count();

    // 1. High-precision curated cross-strait and 1-to-many disambiguation tables.
    if target == TargetVariant::ZhHant {
        if let Some(exact) = curated_to_zh_hant(surface) {
            return exact.to_owned();
        }
    } else if target == TargetVariant::ZhHans
        && let Some(exact) = curated_to_zh_hans(surface)
    {
        return exact.to_owned();
    }

    // 2. Multi-character lexicon lookup (`FNLPLX01`).
    if char_count >= 2
        && let Some(hit) = lookup_zh(surface)
    {
        match (target, hit.variant, hit.counterpart.as_deref()) {
            (TargetVariant::ZhHans, Variant::ZhHant, Some(cp)) => return cp.to_owned(),
            (TargetVariant::ZhHant, Variant::ZhHans, Some(cp)) => return cp.to_owned(),
            (TargetVariant::ZhHant, Variant::ZhHant, _)
                if orientation == ZhScriptOrientation::Traditional =>
            {
                return surface.to_owned();
            }
            (TargetVariant::ZhHans, Variant::ZhHans, _)
                if orientation == ZhScriptOrientation::Simplified =>
            {
                return surface.to_owned();
            }
            _ => {}
        }
    }

    // 3. Single-character contextual & UPOS disambiguation + per-character fallback.
    if char_count == 1 {
        let ch = surface.chars().next().unwrap();
        return convert_single_han_char(ch, upos, prev_char, next_char, orientation, target)
            .to_string();
    }

    // Multi-char OOV compound: convert character-by-character with internal neighbor context.
    if target == TargetVariant::ZhHans
        && let Some(num_prefix) = surface.strip_suffix("公尺")
        && !num_prefix.is_empty()
        && num_prefix.chars().all(|c| c.is_ascii_digit() || c == '.')
    {
        return format!("{num_prefix}米");
    }
    let wchars: Vec<char> = surface.chars().collect();
    let mut out = String::with_capacity(surface.len());
    for (i, &ch) in wchars.iter().enumerate() {
        let p = if i > 0 {
            Some(wchars[i - 1])
        } else {
            prev_char
        };
        let n = if i + 1 < wchars.len() {
            Some(wchars[i + 1])
        } else {
            next_char
        };
        out.push(convert_single_han_char(ch, upos, p, n, orientation, target));
    }
    out
}

fn curated_to_zh_hant(surface: &str) -> Option<&'static str> {
    Some(match surface {
        // Cross-strait computing & engineering terms
        "软件" => "軟體",
        "硬件" => "硬體",
        "内存" => "記憶體",
        "网络" => "網路",
        "服务器" => "伺服器",
        "数据库" => "資料庫",
        "算法" => "演算法",
        "鼠标" => "滑鼠",
        "默认" => "預設",
        "调用" => "呼叫",
        "线程" => "執行緒",
        "数组" => "陣列",
        "指针" => "指標",
        "出租车" => "計程車",
        "自行车" => "腳踏車",
        "优盘" => "隨身碟",
        "硬盘" => "硬碟",
        "操作系统" => "作業系統",
        "人工智能" => "人工智慧",
        // 1-to-many: 发 -> 髮 vs 發
        "头发" | "頭髮" => "頭髮",
        "秀发" | "秀髮" => "秀髮",
        "白发" => "白髮",
        "黑发" | "黑髮" => "黑髮",
        "长发" | "長髮" => "長髮",
        "短发" | "短髮" => "短髮",
        "假发" => "假髮",
        "金发" | "金髮" => "金髮",
        "银发" | "銀髮" => "銀髮",
        "散发" | "散髮" => "散髮",
        "理发" => "理髮",
        "美发" => "美髮",
        "洗发" => "洗髮",
        "洗发水" => "洗髮精",
        "脱发" => "脫髮",
        "发型" => "髮型",
        "发丝" | "髮絲" => "髮絲",
        "千钧一发" => "千鈞一髮",
        "怒发冲冠" => "怒髮衝冠",
        "令人发指" => "令人髮指",
        "出发" => "出發",
        "发现" => "發現",
        "发生" => "發生",
        "发展" => "發展",
        "发情" | "發情" => "發情",
        "发抖" | "發抖" => "發抖",
        "发热" | "發熱" => "發熱",
        "发烫" | "發燙" => "發燙",
        "发软" | "發軟" => "發軟",
        "发泄" | "發洩" => "發洩",
        // 1-to-many: 后 -> 后 vs 後
        "皇后" => "皇后",
        "王后" => "王后",
        "太后" => "太后",
        "皇太后" => "皇太后",
        "母后" => "母后",
        "天后" => "天后",
        "影后" => "影后",
        "歌后" => "歌后",
        "以后" => "以後",
        "之后" => "之後",
        "最后" | "最後" => "最後",
        "后来" => "後來",
        "背后" | "背後" => "背後",
        "身后" | "身後" => "身後",
        "前后" => "前後",
        "然后" => "然後",
        "后背位" | "後背位" => "後背位",
        "后庭" | "後庭" => "後庭",
        "后穴" | "後穴" => "後穴",
        // 1-to-many: 干 -> 乾 vs 幹 vs 干
        "干净" | "乾淨" => "乾淨",
        "干燥" | "乾燥" => "乾燥",
        "干旱" => "乾旱",
        "干杯" => "乾杯",
        "干脆" => "乾脆",
        "干粮" => "乾糧",
        "饼干" | "餅乾" => "餅乾",
        "风干" => "風乾",
        "晾干" => "晾乾",
        "擦干" => "擦乾",
        "干涸" => "乾涸",
        "干枯" => "乾枯",
        "干坤" => "乾坤",
        "一干二净" => "一乾二淨",
        "干活" | "幹活" => "幹活",
        "干部" => "幹部",
        "骨干" => "骨幹",
        "能干" => "能幹",
        "主干" => "主幹",
        "树干" => "樹幹",
        "干什么" | "幹什麼" => "幹什麼",
        "干嘛" | "幹嘛" => "幹嘛",
        "干死" | "幹死" => "幹死",
        "操干" | "操幹" => "操幹",
        "狠干" | "狠幹" => "狠幹",
        "抽干" | "抽幹" => "抽幹",
        "干涉" => "干涉",
        "干预" => "干預",
        "干戈" => "干戈",
        "若干" => "若干",
        "相干" => "相干",
        "天干" => "天干",
        // 1-to-many: 里 -> 裡 vs 里
        "屋里" => "屋裡",
        "心里" | "心裡" => "心裡",
        "这里" => "這裡",
        "那里" => "那裡",
        "哪里" => "哪裡",
        "里面" | "裡面" => "裡面",
        "穴里" | "穴裡" => "穴裡",
        "身体里" | "身體裡" => "身體裡",
        "嘴里" | "嘴裡" => "嘴裡",
        "夜里" => "夜裡",
        "手里" => "手裡",
        "眼里" => "眼裡",
        "梦里" => "夢裡",
        "城里" => "城裡",
        "山里" => "山裡",
        "院里" => "院裡",
        "家里" => "家裡",
        "公里" => "公里",
        "英里" => "英里",
        "海里" => "海里",
        "万里" => "萬里",
        "千里" => "千里",
        "百里" => "百里",
        "十里" => "十里",
        "邻里" => "鄰里",
        "乡里" => "鄉里",
        "故里" => "故里",
        // 1-to-many: 欲 -> 慾 vs 欲
        "情欲" | "情慾" => "情慾",
        "性欲" | "性慾" => "性慾",
        "肉欲" | "肉慾" => "肉慾",
        "欲望" | "慾望" => "慾望",
        "淫欲" | "淫慾" => "淫慾",
        "色欲" | "色慾" => "色慾",
        "爱欲" | "愛慾" => "愛慾",
        "兽欲" | "獸慾" => "獸慾",
        "食欲" | "食慾" => "食慾",
        "贪欲" | "貪慾" => "貪慾",
        // 1-to-many: 湿 -> 濕, 荡 -> 蕩, 余 -> 餘, and literary/anatomical/adult terms
        "濡湿" | "濡濕" => "濡濕",
        "湿透" | "濕透" => "濕透",
        "湿润" | "濕潤" => "濕潤",
        "潮湿" | "潮濕" => "潮濕",
        "湿滑" | "濕滑" => "濕滑",
        "湿热" | "濕熱" => "濕熱",
        "湿濡" | "濕濡" => "濕濡",
        "淫荡" | "淫蕩" => "淫蕩",
        "放荡" | "放蕩" => "放蕩",
        "浪荡" | "浪蕩" => "浪蕩",
        "动荡" | "動盪" => "動盪",
        "震荡" | "震盪" => "震盪",
        "扫荡" | "掃蕩" => "掃蕩",
        "余韵" | "餘韻" => "餘韻",
        "娇喘" | "嬌喘" => "嬌喘",
        "痉挛" | "痙攣" => "痙攣",
        "爱抚" | "愛撫" => "愛撫",
        "绝顶" | "絕頂" => "絕頂",
        "阴茎" | "陰莖" => "陰莖",
        "阴道" | "陰道" => "陰道",
        "阴蒂" | "陰蒂" => "陰蒂",
        "阴唇" | "陰唇" => "陰唇",
        "乳头" | "乳頭" => "乳頭",
        "乳晕" | "乳暈" => "乳暈",
        "龟头" | "龜頭" => "龜頭",
        "紧缚" | "緊縛" => "緊縛",
        "调教" | "調教" => "調教",
        "内射" | "內射" => "內射",
        "颜射" | "顏射" => "顏射",
        "双乳" | "雙乳" => "雙乳",
        "酥胸" => "酥胸",
        "翘臀" | "翹臀" => "翹臀",
        "大腿内侧" | "大腿內側" => "大腿內側",
        "内侧" | "內側" => "內側",
        "颤抖" | "顫抖" => "顫抖",
        "抚摸" | "撫摸" => "撫摸",
        "缠绵" | "纏綿" => "纏綿",
        "痴汉" | "癡漢" => "癡漢",
        "触手" | "觸手" => "觸手",
        "媚药" | "媚藥" => "媚藥",
        "春药" | "春藥" => "春藥",
        // 1-to-many: 面 -> 麵 vs 面
        "面条" => "麵條",
        "拉面" => "拉麵",
        "牛肉面" => "牛肉麵",
        "泡面" => "泡麵",
        "方便面" => "泡麵",
        "煮面" => "煮麵",
        "吃面" => "吃麵",
        "炒面" => "炒麵",
        "拌面" => "拌麵",
        "汤面" => "湯麵",
        "面粉" => "麵粉",
        "面包" => "麵包",
        "面团" => "麵團",
        // 1-to-many: 系 -> 係 vs 繫 vs 系
        "关系" => "關係",
        "没关系" => "沒關係",
        "干系" => "干係",
        "联系" => "聯繫",
        "维系" => "維繫",
        "系统" => "系統",
        "系列" => "系列",
        "体系" => "體系",
        // Tech / media cross-strait terms
        "数字技术" => "數位技術",
        "数字化" => "數位化",
        "智能体育场" => "智慧體育場",
        "智能手机" => "智慧型手機",
        "传感器" => "感測器",
        _ => return None,
    })
}

fn curated_to_zh_hans(surface: &str) -> Option<&'static str> {
    Some(match surface {
        "公尺" => "米",
        "數位" => "数字",
        "數位技術" => "数字技术",
        "數位化" => "数字化",
        "智慧體育場" => "智能体育场",
        "智慧型手機" => "智能手机",
        "感測器" => "传感器",
        "姊姊" => "姐姐",
        "軟體" => "软件",
        "硬體" => "硬件",
        "記憶體" => "内存",
        "網路" => "网络",
        "伺服器" => "服务器",
        "資料庫" => "数据库",
        "演算法" => "算法",
        "滑鼠" => "鼠标",
        "預設" => "默认",
        "呼叫" => "调用",
        "執行緒" => "线程",
        "陣列" => "数组",
        "指標" => "指针",
        "計程車" => "出租车",
        "腳踏車" => "自行车",
        "隨身碟" => "优盘",
        "硬碟" => "硬盘",
        "作業系統" => "操作系统",
        "人工智慧" => "人工智能",
        "洗髮精" => "洗发水",
        "泡麵" => "方便面",
        "情慾" => "情欲",
        "性慾" => "性欲",
        "肉慾" => "肉欲",
        "慾望" => "欲望",
        "淫慾" => "淫欲",
        "色慾" => "色欲",
        "愛慾" => "爱欲",
        "獸慾" => "兽欲",
        "食慾" => "食欲",
        "貪慾" => "贪欲",
        "濡濕" => "濡湿",
        "濕透" => "湿透",
        "濕潤" => "湿润",
        "淫蕩" => "淫荡",
        "動盪" => "动荡",
        "震盪" => "震荡",
        "餘韻" => "余韵",
        "秀髮" => "秀发",
        "黑髮" => "黑发",
        "長髮" => "长发",
        "金髮" => "金发",
        "發情" => "发情",
        "發抖" => "发抖",
        "後背位" => "后背位",
        "後庭" => "后庭",
        "後穴" => "后穴",
        "穴裡" => "穴里",
        "身體裡" => "身体里",
        "嬌喘" => "娇喘",
        "痙攣" => "痉挛",
        "愛撫" => "爱抚",
        "絕頂" => "绝顶",
        "陰莖" => "阴茎",
        "陰道" => "阴道",
        "陰蒂" => "阴蒂",
        "陰唇" => "阴唇",
        "乳頭" => "乳头",
        "龜頭" => "龟头",
        "緊縛" => "紧缚",
        "調教" => "调教",
        "內射" => "内射",
        "顏射" => "颜射",
        "雙乳" => "双乳",
        "翹臀" => "翘臀",
        "大腿內側" => "大腿内侧",
        "癡漢" => "痴汉",
        "觸手" => "触手",
        "媚藥" => "媚药",
        "春藥" => "春药",
        _ => return None,
    })
}

fn convert_single_han_char(
    ch: char,
    upos: Option<Upos>,
    prev_char: Option<char>,
    next_char: Option<char>,
    orientation: ZhScriptOrientation,
    target: TargetVariant,
) -> char {
    if !is_han(ch) {
        return ch;
    }

    if target == TargetVariant::ZhHant {
        // Preserve already-Traditional shared polysemous characters (`立為后`, `三里路`, `天干`).
        if orientation == ZhScriptOrientation::Traditional && is_shared_polysemous_hant_char(ch) {
            return ch;
        }
        match ch {
            '发' => {
                if prev_char.is_some_and(|p| {
                    matches!(
                        p,
                        '頭' | '头'
                            | '秀'
                            | '白'
                            | '黑'
                            | '長'
                            | '长'
                            | '短'
                            | '假'
                            | '金'
                            | '銀'
                            | '银'
                            | '散'
                            | '理'
                            | '美'
                            | '洗'
                            | '脫'
                            | '脱'
                    )
                }) || next_char.is_some_and(|n| matches!(n, '絲' | '丝' | '型' | '簪' | '髻'))
                {
                    return '髮';
                }
                return '發';
            }
            '欲' => {
                if prev_char.is_some_and(|p| {
                    matches!(
                        p,
                        '情' | '性'
                            | '肉'
                            | '淫'
                            | '色'
                            | '愛'
                            | '爱'
                            | '獸'
                            | '兽'
                            | '貪'
                            | '贪'
                            | '物'
                            | '私'
                            | '食'
                            | '求'
                    )
                }) || next_char.is_some_and(|n| matches!(n, '望' | '火' | '念'))
                {
                    return '慾';
                }
                return '欲';
            }
            '荡' => {
                if prev_char.is_some_and(|p| matches!(p, '動' | '动' | '震' | '激' | '飄' | '飘'))
                {
                    return '盪';
                }
                return '蕩';
            }
            '余' => {
                if next_char.is_some_and(|n| {
                    matches!(
                        n,
                        '韻' | '韵' | '波' | '生' | '熱' | '热' | '溫' | '温' | '地' | '光'
                    )
                }) {
                    return '餘';
                }
                return '余';
            }
            '后' => {
                if prev_char
                    .is_some_and(|p| matches!(p, '皇' | '王' | '太' | '母' | '帝' | '影' | '歌'))
                {
                    return '后';
                }
                return '後';
            }
            '干' => {
                if upos == Some(Upos::Adj)
                    || next_char.is_some_and(|n| {
                        matches!(
                            n,
                            '衣' | '布'
                                | '柴'
                                | '草'
                                | '果'
                                | '貨'
                                | '货'
                                | '肉'
                                | '涸'
                                | '枯'
                                | '裂'
                                | '淨'
                                | '净'
                                | '燥'
                                | '旱'
                                | '杯'
                                | '脆'
                        )
                    })
                {
                    return '乾';
                }
                if next_char.is_some_and(|n| matches!(n, '涉' | '預' | '预' | '戈' | '支'))
                    || prev_char.is_some_and(|p| matches!(p, '若' | '相' | '天'))
                {
                    return '干';
                }
                return '幹';
            }
            '里' => {
                if prev_char.is_some_and(is_zh_measure_numeral_prefix) {
                    return '里';
                }
                return '裡';
            }
            '面' => {
                if prev_char.is_some_and(|p| {
                    matches!(p, '吃' | '煮' | '拉' | '炒' | '拌' | '泡' | '湯' | '汤')
                }) && !next_char.is_some_and(|n| {
                    matches!(
                        n,
                        '對' | '对'
                            | '前'
                            | '後'
                            | '后'
                            | '上'
                            | '下'
                            | '積'
                            | '积'
                            | '貌'
                            | '臨'
                            | '临'
                    )
                }) {
                    return '麵';
                }
                return '面';
            }
            _ => {}
        }

        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        if let Some(hit) = lookup_zh(s)
            && hit.variant == Variant::ZhHans
            && let Some(cp) = hit.counterpart.as_deref()
            && cp.chars().count() == 1
        {
            return cp.chars().next().unwrap();
        }
        ch
    } else {
        // `zh-Hans`
        match ch {
            '髮' | '發' => return '发',
            '後' => return '后',
            '乾' | '幹' => return '干',
            '裡' | '裏' => return '里',
            '麵' => return '面',
            '係' | '繫' => return '系',
            '臺' | '檯' | '颱' => return '台',
            '餘' => return '余',
            '錶' => return '表',
            '藉' => return '借',
            '鬥' => return '斗',
            '穀' => return '谷',
            '雲' => return '云',
            '葉' => return '叶',
            '徵' => return '征',
            '鐘' | '鍾' => return '钟',
            '捲' => return '卷',
            '濕' => return '湿',
            '燙' => return '烫',
            '著' => {} // keep 著/着 handled by lexicon compounds (`看著` -> `看着`)
            _ => {}
        }
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        if let Some(hit) = lookup_zh(s)
            && hit.variant == Variant::ZhHant
            && let Some(cp) = hit.counterpart.as_deref()
            && cp.chars().count() == 1
        {
            return cp.chars().next().unwrap();
        }
        ch
    }
}

fn is_zh_measure_numeral_prefix(c: char) -> bool {
    c.is_ascii_digit()
        || matches!(
            c,
            '一' | '二'
                | '兩'
                | '两'
                | '三'
                | '四'
                | '五'
                | '六'
                | '七'
                | '八'
                | '九'
                | '十'
                | '百'
                | '千'
                | '萬'
                | '万'
                | '零'
                | '〇'
                | '幾'
                | '几'
                | '數'
                | '数'
                | '公'
                | '英'
                | '海'
                | '華'
                | '华'
                | '鄰'
                | '邻'
                | '鄉'
                | '乡'
                | '故'
        )
}

fn is_han(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2CEAF | 0xF900..=0xFAFF
    )
}

// ============================================================================
// English (`en-US` <-> `en-GB`) orthographic & POS-aware variant engine
// ============================================================================

fn convert_en_fragment(fragment: &str, target: TargetVariant) -> String {
    let mut out = String::with_capacity(fragment.len());
    let mut prev_word: Option<String> = None;
    let mut i = 0usize;
    let bytes = fragment.as_bytes();
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() {
            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
            let word = &fragment[start..i];
            let prev_ref = prev_word.as_deref().map(|w| (w, None));
            let converted = convert_en_word(word, None, prev_ref, target);
            prev_word = Some(word.to_owned());
            out.push_str(&converted);
        } else {
            let ch = fragment[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Convert a single English token between `en-US` and `en-GB`, respecting `upos` (e.g.
/// `practice` NOUN vs `practise` VERB, `licence` NOUN vs `license` VERB) and preserving casing.
pub fn convert_en_word(
    surface: &str,
    upos: Option<Upos>,
    prev_tok: Option<(&str, Option<Upos>)>,
    target: TargetVariant,
) -> String {
    if surface.len() < 3 || !surface.bytes().all(|b| b.is_ascii_alphabetic()) {
        return surface.to_owned();
    }
    let lower = surface.to_ascii_lowercase();
    // Syntactic context (`to practice`, `doctors practice`, `daily practice`, `driving licence`)
    // overrides context-free lexicon default morphology on Noun/Verb homographs.
    let effective_pos = infer_en_pos_from_context(&lower, prev_tok).or(upos);

    // 1. POS-sensitive British/American pairs (`practice/practise`, `license/licence`).
    if let Some(mapped) = convert_pos_sensitive_en(&lower, effective_pos, target) {
        return preserve_case(surface, mapped);
    }

    // 2. Curated regular & irregular US <-> GB family table (covers base + `-s`, `-ed`, `-ing`,
    //    `-ation` derivations consistently).
    if let Some(mapped) = convert_curated_en_family(&lower, target) {
        return preserve_case(surface, &mapped);
    }

    // 3. Embedded lexicon (`FNLPLX01`) direct lookup (`Variant::EnUs` <-> `Variant::EnGb`).
    if let Some(hit) = lookup_en(&lower) {
        match (target, hit.variant, hit.counterpart.as_deref()) {
            (TargetVariant::EnGb, Variant::EnUs, Some(cp)) => return preserve_case(surface, cp),
            (TargetVariant::EnUs, Variant::EnGb, Some(cp)) => return preserve_case(surface, cp),
            _ => {}
        }
    }

    // 4. Regular plural `-s` over lexicon US/GB nouns (`colours` <-> `colors`, `centres` <-> `centers`).
    if let Some(stem) = lower.strip_suffix('s')
        && !stem.ends_with('s')
        && stem.len() >= 3
        && let Some(hit) = lookup_en(stem)
    {
        match (target, hit.variant, hit.counterpart.as_deref()) {
            (TargetVariant::EnGb, Variant::EnUs, Some(cp)) => {
                return preserve_case(surface, &format!("{cp}s"));
            }
            (TargetVariant::EnUs, Variant::EnGb, Some(cp)) => {
                return preserve_case(surface, &format!("{cp}s"));
            }
            _ => {}
        }
    }

    surface.to_owned()
}

fn infer_en_pos_from_context(lower: &str, prev_tok: Option<(&str, Option<Upos>)>) -> Option<Upos> {
    if lower.ends_with("ing") || lower.ends_with("ed") {
        return Some(Upos::Verb);
    }
    let (prev_surf, prev_upos) = prev_tok?;
    if matches!(
        prev_upos,
        Some(Upos::Det | Upos::Adp | Upos::Adj | Upos::Num)
    ) && !prev_surf.eq_ignore_ascii_case("to")
    {
        return Some(Upos::Noun);
    }
    if matches!(prev_upos, Some(Upos::Aux | Upos::Pron)) {
        return Some(Upos::Verb);
    }
    let pl = prev_surf.to_ascii_lowercase();
    // Plural noun subject (`doctors practice`, `councils license`) before bare stem.
    if matches!(prev_upos, None | Some(Upos::Noun | Upos::Propn))
        && pl.len() >= 4
        && pl.bytes().all(|b| b.is_ascii_alphabetic())
        && pl.ends_with('s')
        && !pl.ends_with("ss")
        && !pl.ends_with("us")
        && !pl.ends_with("is")
        && !pl.ends_with("ous")
        && !matches!(
            pl.as_str(),
            "this" | "thus" | "plus" | "does" | "goes" | "always" | "sometimes" | "perhaps"
        )
    {
        return Some(Upos::Verb);
    }
    if matches!(
        pl.as_str(),
        "to" | "will"
            | "would"
            | "can"
            | "could"
            | "shall"
            | "should"
            | "may"
            | "might"
            | "must"
            | "do"
            | "does"
            | "did"
            | "i"
            | "you"
            | "we"
            | "they"
            | "he"
            | "she"
    ) {
        return Some(Upos::Verb);
    }
    if matches!(
        pl.as_str(),
        "the"
            | "a"
            | "an"
            | "this"
            | "that"
            | "these"
            | "those"
            | "my"
            | "your"
            | "his"
            | "her"
            | "its"
            | "our"
            | "their"
            | "in"
            | "on"
            | "of"
            | "for"
            | "with"
            | "by"
            | "into"
            | "from"
            | "good"
            | "common"
            | "best"
            | "daily"
            | "medical"
            | "legal"
            | "driving"
    ) {
        return Some(Upos::Noun);
    }
    None
}

fn convert_pos_sensitive_en(
    lower: &str,
    pos: Option<Upos>,
    target: TargetVariant,
) -> Option<&'static str> {
    match target {
        TargetVariant::EnUs => match lower {
            "practise" => Some("practice"),
            "practises" => Some("practices"),
            "practised" => Some("practiced"),
            "practising" => Some("practicing"),
            "licence" => Some("license"),
            "licences" => Some("licenses"),
            _ => None,
        },
        TargetVariant::EnGb => match lower {
            "practice" if pos == Some(Upos::Verb) => Some("practise"),
            "practices" if pos == Some(Upos::Verb) => Some("practises"),
            "practiced" => Some("practised"),
            "practicing" => Some("practising"),
            "practice" if pos == Some(Upos::Noun) => Some("practice"),
            "practices" if pos == Some(Upos::Noun) => Some("practices"),
            "license" if pos == Some(Upos::Noun) => Some("licence"),
            "licenses" if pos == Some(Upos::Noun) => Some("licences"),
            "license" if pos == Some(Upos::Verb) => Some("license"),
            "licensed" => Some("licensed"),
            "licensing" => Some("licensing"),
            _ => None,
        },
        _ => None,
    }
}

/// Curated bilateral `(en-US, en-GB)` table covering inflections and `-ize/-ise` / `-ization/-isation`
/// / `-or/-our` / `-er/-re` / `-ense/-ence` / doubled-`l` families.
fn convert_curated_en_family(lower: &str, target: TargetVariant) -> Option<String> {
    const PAIRS: &[(&str, &str)] = &[
        ("color", "colour"),
        ("colors", "colours"),
        ("colored", "coloured"),
        ("coloring", "colouring"),
        ("colorful", "colourful"),
        ("colorless", "colourless"),
        ("behavior", "behaviour"),
        ("behaviors", "behaviours"),
        ("behavioral", "behavioural"),
        ("honor", "honour"),
        ("honors", "honours"),
        ("honored", "honoured"),
        ("honoring", "honouring"),
        ("honorable", "honourable"),
        ("favor", "favour"),
        ("favors", "favours"),
        ("favored", "favoured"),
        ("favoring", "favouring"),
        ("favorite", "favourite"),
        ("favorites", "favourites"),
        ("labor", "labour"),
        ("labors", "labours"),
        ("labored", "laboured"),
        ("laboring", "labouring"),
        ("neighbor", "neighbour"),
        ("neighbors", "neighbours"),
        ("neighborhood", "neighbourhood"),
        ("flavor", "flavour"),
        ("flavors", "flavours"),
        ("flavored", "flavoured"),
        ("humor", "humour"),
        ("rumor", "rumour"),
        ("rumors", "rumours"),
        ("harbor", "harbour"),
        ("harbors", "harbours"),
        ("center", "centre"),
        ("centers", "centres"),
        ("centered", "centred"),
        ("centering", "centring"),
        ("meter", "metre"),
        ("meters", "metres"),
        ("kilometer", "kilometre"),
        ("kilometers", "kilometres"),
        ("liter", "litre"),
        ("liters", "litres"),
        ("theater", "theatre"),
        ("theaters", "theatres"),
        ("fiber", "fibre"),
        ("fibers", "fibres"),
        ("defense", "defence"),
        ("defenses", "defences"),
        ("offense", "offence"),
        ("offenses", "offences"),
        ("pretense", "pretence"),
        ("analyze", "analyse"),
        ("analyzes", "analyses"),
        ("analyzed", "analysed"),
        ("analyzing", "analysing"),
        ("paralyze", "paralyse"),
        ("paralyzed", "paralysed"),
        ("paralyzing", "paralysing"),
        ("organize", "organise"),
        ("organizes", "organises"),
        ("organized", "organised"),
        ("organizing", "organising"),
        ("organization", "organisation"),
        ("organizations", "organisations"),
        ("realize", "realise"),
        ("realizes", "realises"),
        ("realized", "realised"),
        ("realizing", "realising"),
        ("realization", "realisation"),
        ("recognize", "recognise"),
        ("recognizes", "recognises"),
        ("recognized", "recognised"),
        ("recognizing", "recognising"),
        ("optimize", "optimise"),
        ("optimizes", "optimises"),
        ("optimized", "optimised"),
        ("optimizing", "optimising"),
        ("optimization", "optimisation"),
        ("normalize", "normalise"),
        ("normalizes", "normalises"),
        ("normalized", "normalised"),
        ("normalizing", "normalising"),
        ("normalization", "normalisation"),
        ("serialize", "serialise"),
        ("serializes", "serialises"),
        ("serialized", "serialised"),
        ("serializing", "serialising"),
        ("serialization", "serialisation"),
        ("tokenize", "tokenise"),
        ("tokenizes", "tokenises"),
        ("tokenized", "tokenised"),
        ("tokenizing", "tokenising"),
        ("tokenization", "tokenisation"),
        ("initialize", "initialise"),
        ("initializes", "initialises"),
        ("initialized", "initialised"),
        ("initializing", "initialising"),
        ("initialization", "initialisation"),
        ("visualize", "visualise"),
        ("visualized", "visualised"),
        ("visualization", "visualisation"),
        ("summarize", "summarise"),
        ("summarizes", "summarises"),
        ("summarized", "summarised"),
        ("summarizing", "summarising"),
        ("apologize", "apologise"),
        ("apologized", "apologised"),
        ("apologizing", "apologising"),
        ("traveled", "travelled"),
        ("traveling", "travelling"),
        ("traveler", "traveller"),
        ("travelers", "travellers"),
        ("canceled", "cancelled"),
        ("canceling", "cancelling"),
        ("modeled", "modelled"),
        ("modeling", "modelling"),
        ("labeled", "labelled"),
        ("labeling", "labelling"),
        ("signaled", "signalled"),
        ("signaling", "signalling"),
        ("fueled", "fuelled"),
        ("fueling", "fuelling"),
        ("catalog", "catalogue"),
        ("catalogs", "catalogues"),
        ("cataloged", "catalogued"),
        ("cataloging", "cataloguing"),
        ("analog", "analogue"),
        ("gray", "grey"),
        ("aluminum", "aluminium"),
        ("jewelry", "jewellery"),
        ("artifact", "artefact"),
        ("artifacts", "artefacts"),
        ("skeptical", "sceptical"),
        ("maneuver", "manoeuvre"),
        ("maneuvers", "manoeuvres"),
        ("fulfill", "fulfil"),
        ("fulfills", "fulfils"),
        ("fulfilled", "fulfilled"),
        ("enrollment", "enrolment"),
    ];

    for &(us, gb) in PAIRS {
        if target == TargetVariant::EnGb && lower == us {
            return Some(gb.to_owned());
        }
        if target == TargetVariant::EnUs && lower == gb {
            return Some(us.to_owned());
        }
    }
    None
}

fn preserve_case(original: &str, converted_lower: &str) -> String {
    if original.bytes().all(|b| b.is_ascii_uppercase()) {
        return converted_lower.to_ascii_uppercase();
    }
    let mut orig_chars = original.chars();
    if let Some(first) = orig_chars.next()
        && first.is_ascii_uppercase()
        && orig_chars.all(|c| c.is_ascii_lowercase())
    {
        let mut out = String::with_capacity(converted_lower.len());
        let mut conv = converted_lower.chars();
        if let Some(cf) = conv.next() {
            out.push(cf.to_ascii_uppercase());
            out.push_str(conv.as_str());
        }
        return out;
    }
    converted_lower.to_owned()
}
