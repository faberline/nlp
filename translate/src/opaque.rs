//! Stage 2: Structural & `Opaque` span handling (`VerbatimPassthrough`, `StructuralUnwrap`,
//! and `OmitEditorial`).

use faberline_nlp::Span;
use faberline_nlp::doc::{Opaque, OpaqueKind};

use crate::schema::{OpaquePolicy, TargetVariant};

/// Determine whether a top-level `Opaque` span is a standalone block (no sentence covers it)
/// vs an inline span inside a sentence.
pub fn is_block_opaque(opaque: &Opaque, sentence_spans: &[Span]) -> bool {
    match opaque.kind {
        OpaqueKind::Heading | OpaqueKind::ListItem => true,
        OpaqueKind::Code | OpaqueKind::Formula => !sentence_spans
            .iter()
            .any(|s| opaque.span.start_char >= s.start_char && opaque.span.end_char <= s.end_char),
        OpaqueKind::Citation | OpaqueKind::Url | OpaqueKind::Emoji => false,
    }
}

/// Detect whether a `Code` block span was produced by `mask_editorial` / HTML / banner noise
/// rather than a fenced Markdown code block (```` ``` ```` or `~~~`).
pub fn is_editorial_or_noise_code_block(raw_slice: &str) -> bool {
    let trimmed = raw_slice.trim_start();
    !(trimmed.starts_with("```") || trimmed.starts_with("~~~"))
}

/// Unwrapped heading components so the body title can be translated independently while keeping
/// Markdown `#` markers or localizing chapter numbers cleanly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnwrappedHeading<'a> {
    pub hashes: Option<&'a str>,
    pub kind: UnwrappedHeadingKind<'a>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnwrappedHeadingKind<'a> {
    Chapter {
        number: Option<u32>,
        is_episode: bool,
        title: &'a str,
    },
    Keyword {
        keyword_kind: SectionKeyword,
        subtitle: &'a str,
    },
    Plain(&'a str),
}

impl<'a> UnwrappedHeading<'a> {
    pub fn body(&self) -> &'a str {
        match &self.kind {
            UnwrappedHeadingKind::Chapter { title, .. } => title,
            UnwrappedHeadingKind::Keyword { subtitle, .. } => subtitle,
            UnwrappedHeadingKind::Plain(s) => s,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionKeyword {
    Prologue,
    Epilogue,
    Interlude,
    SideStory,
    FinalChapter,
}

impl SectionKeyword {
    pub fn render(self, target: TargetVariant) -> &'static str {
        match (self, target) {
            (Self::Prologue, TargetVariant::ZhHant | TargetVariant::ZhHans) => "序章",
            (Self::Prologue, TargetVariant::EnUs | TargetVariant::EnGb) => "Prologue",
            (Self::Prologue, TargetVariant::Ja) => "プロローグ",
            (Self::Prologue, TargetVariant::Ko) => "프롤로그",

            (Self::Epilogue, TargetVariant::ZhHant) => "終章",
            (Self::Epilogue, TargetVariant::ZhHans) => "终章",
            (Self::Epilogue, TargetVariant::EnUs | TargetVariant::EnGb) => "Epilogue",
            (Self::Epilogue, TargetVariant::Ja) => "エピローグ",
            (Self::Epilogue, TargetVariant::Ko) => "에필로그",

            (
                Self::Interlude,
                TargetVariant::ZhHant | TargetVariant::ZhHans | TargetVariant::Ja,
            ) => "幕間",
            (Self::Interlude, TargetVariant::EnUs | TargetVariant::EnGb) => "Interlude",
            (Self::Interlude, TargetVariant::Ko) => "막간",

            (Self::SideStory, TargetVariant::ZhHant) => "番外篇",
            (Self::SideStory, TargetVariant::ZhHans) => "番外篇",
            (Self::SideStory, TargetVariant::EnUs | TargetVariant::EnGb) => "Side Story",
            (Self::SideStory, TargetVariant::Ja) => "番外編",
            (Self::SideStory, TargetVariant::Ko) => "외전",

            (Self::FinalChapter, TargetVariant::ZhHant) => "最終章",
            (Self::FinalChapter, TargetVariant::ZhHans) => "最终章",
            (Self::FinalChapter, TargetVariant::EnUs | TargetVariant::EnGb) => "Final Chapter",
            (Self::FinalChapter, TargetVariant::Ja) => "最終章",
            (Self::FinalChapter, TargetVariant::Ko) => "종장",
        }
    }
}

pub fn unwrap_heading(raw: &str) -> UnwrappedHeading<'_> {
    let outer = raw.trim();

    // 1. Optional leading Markdown `#`..`######`
    let hash_len = outer.bytes().take_while(|&b| b == b'#').count();
    let (hashes, trimmed) =
        if (1..=6).contains(&hash_len) && outer[hash_len..].starts_with([' ', '\t']) {
            (Some(&outer[..hash_len]), outer[hash_len..].trim_start())
        } else {
            (None, outer)
        };

    // 2. Standalone section keywords (ZH / JA / KO / EN)
    let kw_table = [
        ("プロローグ", SectionKeyword::Prologue),
        ("프롤로그", SectionKeyword::Prologue),
        ("序章", SectionKeyword::Prologue),
        ("서장", SectionKeyword::Prologue),
        ("エピローグ", SectionKeyword::Epilogue),
        ("에필로그", SectionKeyword::Epilogue),
        ("終章", SectionKeyword::Epilogue),
        ("终章", SectionKeyword::Epilogue),
        ("종장", SectionKeyword::Epilogue),
        ("幕間", SectionKeyword::Interlude),
        ("番外編", SectionKeyword::SideStory),
        ("番外篇", SectionKeyword::SideStory),
        ("외전", SectionKeyword::SideStory),
        ("最終話", SectionKeyword::FinalChapter),
        ("最終章", SectionKeyword::FinalChapter),
        ("最终章", SectionKeyword::FinalChapter),
    ];
    for (kw, kind) in kw_table {
        if let Some(rest) = trimmed.strip_prefix(kw) {
            if rest.is_empty() {
                return UnwrappedHeading {
                    hashes,
                    kind: UnwrappedHeadingKind::Keyword {
                        keyword_kind: kind,
                        subtitle: "",
                    },
                };
            }
            let sub = strip_heading_sep(rest);
            if sub.len() < rest.len() {
                return UnwrappedHeading {
                    hashes,
                    kind: UnwrappedHeadingKind::Keyword {
                        keyword_kind: kind,
                        subtitle: sub,
                    },
                };
            }
        }
    }

    // 3. English `Chapter N` / `Episode N` / `Section N`
    let lower = trimmed.to_ascii_lowercase();
    for prefix in ["chapter ", "episode ", "section "] {
        if lower.starts_with(prefix) {
            let is_episode = prefix.starts_with("episode");
            let rest = trimmed[prefix.len()..].trim_start();
            let num_end = rest
                .find(|c: char| !c.is_ascii_alphanumeric())
                .unwrap_or(rest.len());
            let num_tok = &rest[..num_end];
            let number = num_tok.parse::<u32>().ok().or_else(|| parse_roman(num_tok));
            if number.is_some() {
                let tail = strip_heading_sep(&rest[num_end..]);
                return UnwrappedHeading {
                    hashes,
                    kind: UnwrappedHeadingKind::Chapter {
                        number,
                        is_episode,
                        title: tail,
                    },
                };
            }
        }
    }

    // 4. CJK `第<num><unit>` or Korean `제<num><unit>` or bare `<digits>話/화/章/장`
    if let Some((num, is_ep, tail)) = parse_cjk_chapter_header(trimmed) {
        return UnwrappedHeading {
            hashes,
            kind: UnwrappedHeadingKind::Chapter {
                number: Some(num),
                is_episode: is_ep,
                title: tail,
            },
        };
    }

    UnwrappedHeading {
        hashes,
        kind: UnwrappedHeadingKind::Plain(trimmed),
    }
}

pub fn render_heading(
    unwrapped: &UnwrappedHeading<'_>,
    translated_title: &str,
    target: TargetVariant,
) -> String {
    let clean_title = translated_title.trim();
    let core = match &unwrapped.kind {
        UnwrappedHeadingKind::Keyword { keyword_kind, .. } => {
            let kw = keyword_kind.render(target);
            if clean_title.is_empty() {
                kw.to_owned()
            } else if target.is_en() {
                format!("{kw}: {clean_title}")
            } else {
                format!("{kw} {clean_title}")
            }
        }
        UnwrappedHeadingKind::Chapter {
            number, is_episode, ..
        } => {
            let n = number.unwrap_or(1);
            let header = match (target, *is_episode) {
                (TargetVariant::ZhHant | TargetVariant::ZhHans, false) => {
                    format!("第{}章", to_zh_numeral(n))
                }
                (TargetVariant::ZhHant, true) => format!("第{n}話"),
                (TargetVariant::ZhHans, true) => format!("第{n}话"),
                (TargetVariant::EnUs | TargetVariant::EnGb, false) => format!("Chapter {n}"),
                (TargetVariant::EnUs | TargetVariant::EnGb, true) => format!("Episode {n}"),
                (TargetVariant::Ja, true) => format!("第{n}話"),
                (TargetVariant::Ja, false) => format!("第{n}章"),
                (TargetVariant::Ko, true) => format!("제{n}화"),
                (TargetVariant::Ko, false) => format!("제{n}장"),
            };
            if clean_title.is_empty() {
                header
            } else if target.is_en() {
                format!("{header}: {clean_title}")
            } else {
                format!("{header} {clean_title}")
            }
        }
        UnwrappedHeadingKind::Plain(_) => clean_title.to_owned(),
    };

    match unwrapped.hashes {
        Some(h) if core.is_empty() => h.to_owned(),
        Some(h) => format!("{h} {core}"),
        None => core,
    }
}

/// Unwrap a `ListItem` line into `(marker_prefix, item_body)` so `- `, `1. `, `※ ` stay intact.
pub fn unwrap_list_item(raw: &str) -> (&str, &str) {
    let trimmed = raw.trim_start();
    for marker in ["- ", "* ", "+ ", "• ", "・ ", "※ ", "-\t", "*\t", "※\t"] {
        if let Some(body) = trimmed.strip_prefix(marker) {
            let prefix_len = raw.len() - body.len();
            return (&raw[..prefix_len], body);
        }
    }
    // Fullwidth space after marker
    for ch in ['-', '*', '+', '•', '・', '※'] {
        if let Some(after) = trimmed.strip_prefix(ch)
            && let Some(body) = after.strip_prefix('\u{3000}')
        {
            let prefix_len = raw.len() - body.len();
            return (&raw[..prefix_len], body);
        }
    }
    // Numbered `1. ` or `12) `
    let digits = trimmed.bytes().take_while(|b| b.is_ascii_digit()).count();
    if (1..=3).contains(&digits) {
        let after = &trimmed[digits..];
        if let Some(rest) = after
            .strip_prefix(". ")
            .or_else(|| after.strip_prefix(") "))
        {
            let prefix_len = raw.len() - rest.len();
            return (&raw[..prefix_len], rest);
        }
    }
    ("", raw)
}

/// Choose the `OpaquePolicy` for a given `Opaque` span.
pub fn classify_opaque_policy(
    opaque: &Opaque,
    raw_slice: &str,
    omit_masked_editorial: bool,
) -> OpaquePolicy {
    match opaque.kind {
        OpaqueKind::Heading | OpaqueKind::ListItem => OpaquePolicy::StructuralUnwrap,
        OpaqueKind::Code => {
            if omit_masked_editorial && is_editorial_or_noise_code_block(raw_slice) {
                OpaquePolicy::OmitEditorial
            } else {
                OpaquePolicy::VerbatimPassthrough
            }
        }
        OpaqueKind::Formula | OpaqueKind::Url | OpaqueKind::Emoji | OpaqueKind::Citation => {
            OpaquePolicy::VerbatimPassthrough
        }
    }
}

fn strip_heading_sep(s: &str) -> &str {
    s.trim_start_matches([
        ' ', '\t', '\u{3000}', '\u{00a0}', '：', ':', '、', '・', '·', '-', '—', '―', '〜', '～',
        '.',
    ])
    .trim_start()
}

fn parse_cjk_chapter_header(s: &str) -> Option<(u32, bool, &str)> {
    let rest = s
        .strip_prefix('第')
        .or_else(|| s.strip_prefix('제'))
        .map(str::trim_start);

    if let Some(after_prefix) = rest {
        let (num_str, after_num) = split_cjk_numeral_prefix(after_prefix)?;
        let n = parse_cjk_or_ascii_num(num_str)?;
        let after_num = after_num.trim_start();
        let mut chars = after_num.chars();
        let unit = chars.next()?;
        let is_ep = matches!(unit, '話' | '话' | '回' | '화' | '회');
        let is_chap = matches!(
            unit,
            '章' | '節'
                | '节'
                | '卷'
                | '巻'
                | '篇'
                | '部'
                | '編'
                | '幕'
                | '장'
                | '절'
                | '권'
                | '편'
        );
        if is_ep || is_chap {
            let tail = strip_heading_sep(chars.as_str());
            return Some((n, is_ep, tail));
        }
    }

    // Bare `<digits><unit>` such as `12話 雨の駅` or `12화 제목`
    let (num_str, after_num) = split_cjk_numeral_prefix(s)?;
    let n = parse_cjk_or_ascii_num(num_str)?;
    let mut chars = after_num.chars();
    let unit = chars.next()?;
    let is_ep = matches!(unit, '話' | '话' | '화');
    let is_chap = matches!(unit, '章' | '장');
    if is_ep || is_chap {
        let tail = strip_heading_sep(chars.as_str());
        return Some((n, is_ep, tail));
    }
    None
}

fn split_cjk_numeral_prefix(s: &str) -> Option<(&str, &str)> {
    let mut end = 0usize;
    for (bi, ch) in s.char_indices() {
        let ok = ch.is_ascii_digit()
            || ('０'..='９').contains(&ch)
            || matches!(
                ch,
                '零' | '〇'
                    | '一'
                    | '二'
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
                    | '壱'
                    | '弐'
                    | '参'
                    | '일'
                    | '이'
                    | '삼'
                    | '사'
                    | '오'
                    | '육'
                    | '륙'
                    | '칠'
                    | '팔'
                    | '구'
                    | '십'
                    | '백'
                    | '천'
            );
        if ok {
            end = bi + ch.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        None
    } else {
        Some((&s[..end], &s[end..]))
    }
}

fn parse_cjk_or_ascii_num(s: &str) -> Option<u32> {
    if s.bytes().all(|b| b.is_ascii_digit()) {
        return s.parse::<u32>().ok();
    }
    // Fullwidth digits
    if s.chars().all(|c| ('０'..='９').contains(&c)) {
        let ascii: String = s
            .chars()
            .map(|c| char::from_u32((c as u32) - ('０' as u32) + ('0' as u32)).unwrap_or('0'))
            .collect();
        return ascii.parse::<u32>().ok();
    }
    // Han / Hangul Sino-Korean numerals up to 99999
    let mut total = 0u32;
    let mut current = 0u32;
    let mut all_digits = s.chars().count() >= 2;
    let mut digit_acc = 0u32;

    for ch in s.chars() {
        let d = match ch {
            '零' | '〇' => Some(0),
            '一' | '壱' | '일' => Some(1),
            '二' | '兩' | '两' | '弐' | '이' => Some(2),
            '三' | '参' | '삼' => Some(3),
            '四' | '사' => Some(4),
            '五' | '오' => Some(5),
            '六' | '육' | '륙' => Some(6),
            '七' | '칠' => Some(7),
            '八' | '팔' => Some(8),
            '九' | '구' => Some(9),
            _ => None,
        };
        if let Some(v) = d {
            current = v;
            digit_acc = digit_acc.saturating_mul(10).saturating_add(v);
        } else {
            all_digits = false;
            let mult = match ch {
                '十' | '십' => 10,
                '百' | '백' => 100,
                '千' | '천' => 1000,
                '萬' | '万' => 10000,
                _ => return None,
            };
            if mult == 10000 {
                total = (total + if current == 0 { 1 } else { current }) * 10000;
                current = 0;
            } else {
                total += (if current == 0 { 1 } else { current }) * mult;
                current = 0;
            }
        }
    }
    if all_digits {
        Some(digit_acc)
    } else {
        Some(total + current)
    }
}

fn to_zh_numeral(n: u32) -> String {
    const DIGITS: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    if n == 0 {
        return "零".to_owned();
    }
    if n > 9999 {
        return n.to_string();
    }
    let mut out = String::new();
    let q = n / 1000;
    let b = (n / 100) % 10;
    let s = (n / 10) % 10;
    let g = n % 10;

    if q > 0 {
        out.push(DIGITS[q as usize]);
        out.push('千');
    }
    if b > 0 {
        if q > 0 && b == 0 {
            out.push('零');
        }
        out.push(DIGITS[b as usize]);
        out.push('百');
    } else if q > 0 && (s > 0 || g > 0) {
        out.push('零');
    }
    if s > 0 {
        if !(n < 20 && s == 1 && q == 0 && b == 0) {
            out.push(DIGITS[s as usize]);
        }
        out.push('十');
    } else if b > 0 && g > 0 {
        out.push('零');
    }
    if g > 0 {
        out.push(DIGITS[g as usize]);
    }
    out
}

fn parse_roman(s: &str) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    let mut vals = Vec::with_capacity(s.len());
    for b in s.bytes() {
        let v = match b.to_ascii_uppercase() {
            b'I' => 1u32,
            b'V' => 5,
            b'X' => 10,
            b'L' => 50,
            b'C' => 100,
            _ => return None,
        };
        vals.push(v);
    }
    let mut sum = 0u32;
    for i in 0..vals.len() {
        if i + 1 < vals.len() && vals[i] < vals[i + 1] {
            sum = sum.checked_sub(vals[i])?;
        } else {
            sum += vals[i];
        }
    }
    if sum > 0 { Some(sum) } else { None }
}
