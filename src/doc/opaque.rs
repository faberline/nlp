//! Layer 1b of `nlp.doc.v1`: opaque spans. See `docs/doc-v1.md`.
//!
//! Opaque spans are found by rules and then masked before sentence and clause splitting, so no
//! boundary can fall inside one. All spans are offsets into the original text.

use super::{Opaque, OpaqueKind};
use crate::Span;

pub(super) struct Found {
    pub item: Opaque,
    /// Block spans cover whole lines. Inline spans stay inside one line.
    pub block: bool,
    /// Inline scraper/editorial/HTML/PUA noise suppressed when `mask_editorial` is enabled.
    pub silent_noise: bool,
}

pub(super) fn find(c: &[char]) -> Vec<Found> {
    find_with_options(c, false)
}

pub(super) fn find_with_options(c: &[char], mask_editorial: bool) -> Vec<Found> {
    let lines = lines(c);
    let mut out = Vec::new();
    let mut li = 0;
    while li < lines.len() {
        let (ls, le) = lines[li];
        let Some(s) = first_non_ws(c, ls, le) else {
            li += 1;
            continue;
        };
        let t_end = trimmed_end(c, s, le);
        let block = code_block(c, &lines, li, s)
            .or_else(|| formula_block(c, &lines, li, s))
            .or_else(|| box_banner_block(c, &lines, li, s))
            .or_else(|| {
                line_kind(c, ls, s, t_end, le, mask_editorial).map(|kind| (kind, t_end, li + 1))
            });
        match block {
            Some((kind, end, next)) => {
                out.push(Found {
                    item: Opaque {
                        kind,
                        span: Span {
                            start_char: s,
                            end_char: end,
                        },
                    },
                    block: true,
                    silent_noise: false,
                });
                li = next;
            }
            None => {
                inline(c, ls, le, mask_editorial, &mut out);
                li += 1;
            }
        }
    }
    out
}

/// Blank the block spans, mask `silent_noise` spans with non-breaking whitespace so noise-only
/// stretches are trimmed without triggering intra-line double-space sentence splits, and replace
/// every other inline opaque char with `x`. An emoji run that ends a sentence keeps a `!` as its
/// last char, so the sentence still ends there.
pub(super) fn mask(c: &[char], found: &[Found]) -> Vec<char> {
    let mut m = c.to_vec();
    for f in found {
        let Span {
            start_char: s,
            end_char: e,
        } = f.item.span;
        if f.block {
            for ch in &mut m[s..e] {
                if *ch != '\n' {
                    *ch = ' ';
                }
            }
        } else if f.silent_noise {
            for ch in &mut m[s..e] {
                if *ch != '\n' {
                    *ch = '\u{00a0}';
                }
            }
        } else {
            m[s..e].fill('x');
            if f.item.kind == OpaqueKind::Emoji && ends_sentence(c, s, e) {
                m[e - 1] = '!';
            }
        }
    }
    m
}

/// Attached to the text before it, and followed by whitespace or the end of the text.
fn ends_sentence(c: &[char], s: usize, e: usize) -> bool {
    let attached = s > 0 && !c[s - 1].is_whitespace();
    attached && (e == c.len() || c[e].is_whitespace())
}

/// Returns (kind, end, index of the next line) for a fenced code block or a display formula.
fn code_block(
    c: &[char],
    lines: &[(usize, usize)],
    li: usize,
    s: usize,
) -> Option<(OpaqueKind, usize, usize)> {
    let (ls, le) = lines[li];
    let fence = c[s];
    if s - ls > 3 || !matches!(fence, '`' | '~') {
        return None;
    }
    let n = run_len(c, s, le, fence);
    if n < 3 || c[s + n..le].contains(&fence) {
        return None;
    }
    for (k, &(cs, ce)) in lines.iter().enumerate().skip(li + 1) {
        let Some(f) = first_non_ws(c, cs, ce) else {
            continue;
        };
        let m = run_len(c, f, ce, fence);
        if c[f] == fence && m >= n && c[f + m..ce].iter().all(|ch| ch.is_whitespace()) {
            return Some((OpaqueKind::Code, f + m, k + 1));
        }
    }
    Some((OpaqueKind::Code, trimmed_end(c, s, c.len()), lines.len()))
}

/// A display formula opens with `$$` and closes at a line whose text ends with `$$`. The
/// opening line may close itself.
fn formula_block(
    c: &[char],
    lines: &[(usize, usize)],
    li: usize,
    s: usize,
) -> Option<(OpaqueKind, usize, usize)> {
    let (_, le) = lines[li];
    if s + 1 >= le || c[s] != '$' || c[s + 1] != '$' {
        return None;
    }
    for (k, &(cs, ce)) in lines.iter().enumerate().skip(li) {
        if k > li && first_non_ws(c, cs, ce).is_none() {
            return None;
        }
        let end = trimmed_end(c, cs, ce);
        let min = if k == li { s + 2 } else { cs };
        if end >= min + 2 && c[end - 2] == '$' && c[end - 1] == '$' {
            return Some((OpaqueKind::Formula, end, k + 1));
        }
    }
    None
}

/// Box-drawing scraper banner block opening with `┏━`/`┌─` and closing at `┗━`/`└─` (or single-line border).
fn box_banner_block(
    c: &[char],
    lines: &[(usize, usize)],
    li: usize,
    s: usize,
) -> Option<(OpaqueKind, usize, usize)> {
    let (_, le) = lines[li];
    if s + 1 >= le || !matches!((c[s], c[s + 1]), ('┏', '━' | '─') | ('┌', '─' | '━')) {
        return None;
    }
    for (k, &(cs, ce)) in lines
        .iter()
        .enumerate()
        .take(lines.len().min(li + 12))
        .skip(li + 1)
    {
        let Some(f) = first_non_ws(c, cs, ce) else {
            break;
        };
        let end = trimmed_end(c, cs, ce);
        let bottom_start =
            f + 1 < end && matches!((c[f], c[f + 1]), ('┗', '━' | '─') | ('└', '─' | '━'));
        let bottom_end = end > f + 1
            && matches!(c[end - 1], '┛' | '┘')
            && c[f..end].iter().any(|&ch| matches!(ch, '━' | '─'));
        if bottom_start || bottom_end {
            return Some((OpaqueKind::Code, end, k + 1));
        }
    }
    let t_end = trimmed_end(c, s, le);
    let single_line = t_end > s + 2
        && (matches!(c[t_end - 1], '┓' | '┐' | '┛' | '┘' | '━' | '─')
            || c[s..t_end]
                .windows(2)
                .any(|w| matches!((w[0], w[1]), ('┗', '━' | '─') | ('└', '─' | '━'))));
    single_line.then_some((OpaqueKind::Code, t_end, li + 1))
}

/// The kind of a line that is a heading, list item, whole-line raw markup/PUA noise, or opt-in
/// editorial block line.
fn line_kind(
    c: &[char],
    ls: usize,
    s: usize,
    t_end: usize,
    le: usize,
    mask_editorial: bool,
) -> Option<OpaqueKind> {
    let t = &c[s..t_end];
    if is_heading(t) {
        Some(OpaqueKind::Heading)
    } else if is_list_item(t) {
        Some(OpaqueKind::ListItem)
    } else if is_raw_noise_only_line(c, ls, s, t_end, le, mask_editorial)
        || (mask_editorial && is_editorial_block_line(t))
    {
        Some(OpaqueKind::Code)
    } else {
        None
    }
}

fn is_heading(t: &[char]) -> bool {
    markdown_heading(t)
        || chinese_heading(t)
        || japanese_heading(t)
        || korean_heading(t)
        || english_heading(t)
}

fn markdown_heading(t: &[char]) -> bool {
    let hashes = t.iter().take_while(|&&ch| ch == '#').count();
    (1..=6).contains(&hashes)
        && matches!(t.get(hashes).copied(), Some(' ' | '\t'))
        && t[hashes..].iter().any(|ch| !ch.is_whitespace())
}

const ZH_HEADING_NUMERALS: &str = "一二三四五六七八九十百千萬万零〇兩两";
const ZH_HEADING_UNITS: &str = "章節回卷篇部";

fn zh_numeral_len(s: &[char]) -> usize {
    s.iter()
        .take_while(|ch| ch.is_ascii_digit() || ZH_HEADING_NUMERALS.contains(**ch))
        .count()
}

fn heading_ws_len(s: &[char]) -> usize {
    s.iter()
        .take_while(|ch| matches!(**ch, ' ' | '\t' | '\u{3000}' | '\u{00a0}'))
        .count()
}

fn is_heading_sep(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '：' | ':' | '、' | '·' | '‧' | '»')
}

fn chinese_di_heading(s: &[char], allow_tight_title: bool) -> bool {
    if s.first() != Some(&'第') {
        return false;
    }
    let numerals = zh_numeral_len(&s[1..]);
    if numerals == 0 {
        return false;
    }
    let ws = heading_ws_len(&s[1 + numerals..]);
    let [unit, after @ ..] = &s[1 + numerals + ws..] else {
        return false;
    };
    if !ZH_HEADING_UNITS.contains(*unit) {
        return false;
    }
    match after {
        [] => ws == 0,
        [sep, title @ ..] if is_heading_sep(*sep) && has_non_decor_text(title) => true,
        _ => allow_tight_title && has_text(after),
    }
}

/// Standalone `第<zh_or_ascii_num>章<2..14 Han/punct title>` with zero whitespace after `章`,
/// total length `<= 30`, and a prose-continuation guard so phrases like `第三章的內容` stay prose.
fn tight_standalone_zhang_heading(s: &[char]) -> bool {
    if s.len() > 30 || s.first() != Some(&'第') {
        return false;
    }
    let numerals = zh_numeral_len(&s[1..]);
    if numerals == 0 || s.get(1 + numerals) != Some(&'章') {
        return false;
    }
    let title = &s[2 + numerals..];
    if !(2..=14).contains(&title.len()) || title[0].is_whitespace() {
        return false;
    }
    if matches!(
        title[0],
        '的' | '之'
            | '了'
            | '是'
            | '在'
            | '有'
            | '中'
            | '裡'
            | '里'
            | '內'
            | '内'
            | '講'
            | '讲'
            | '說'
            | '说'
            | '討'
            | '讨'
            | '介'
            | '論'
            | '论'
    ) {
        return false;
    }
    title.iter().any(|&ch| is_han(ch))
        && title.iter().all(|&ch| {
            is_han(ch)
                || ch.is_ascii_alphanumeric()
                || matches!(
                    ch,
                    '（' | '）' | '(' | ')' | '、' | '·' | '‧' | '—' | '-' | '《' | '》'
                )
        })
}

fn has_non_decor_text(chars: &[char]) -> bool {
    chars
        .iter()
        .any(|ch| !ch.is_whitespace() && !matches!(*ch, '·' | '‧' | '»'))
}

fn trim_heading_decor(chars: &[char]) -> &[char] {
    let start = chars
        .iter()
        .position(|ch| !ch.is_whitespace() && !matches!(*ch, '»' | '·' | '‧'))
        .unwrap_or(chars.len());
    trim(&chars[start..])
}

fn bare_numeral_zhang_heading(s: &[char]) -> bool {
    let numerals = zh_numeral_len(s);
    numerals >= 2
        && s.get(numerals) == Some(&'章')
        && s.get(numerals + 1).is_some_and(|sep| is_heading_sep(*sep))
        && has_non_decor_text(&s[numerals + 2..])
}

/// Standalone `第<zh_numerals><ws><short_title>` where `numerals >= 2` and `short_title` is
/// `1..=12` chars with no clause/sentence punctuation (handles web-novel titles missing `章`).
fn missing_unit_di_heading(s: &[char]) -> bool {
    if s.first() != Some(&'第') {
        return false;
    }
    let numerals = zh_numeral_len(&s[1..]);
    if numerals < 2 {
        return false;
    }
    let ws = heading_ws_len(&s[1 + numerals..]);
    if ws == 0 {
        return false;
    }
    let title = trim(&s[1 + numerals + ws..]);
    (1..=12).contains(&title.len())
        && !matches!(title.first(), Some('的' | '了' | '是'))
        && !title
            .iter()
            .any(|ch| "，。！？；：、,.;:!?'\"「」『』“”‘’".contains(*ch))
}

fn volume_then_chapter(s: &[char]) -> bool {
    if s.first() != Some(&'第') {
        return false;
    }
    let n = zh_numeral_len(&s[1..]);
    if n == 0 {
        return false;
    }
    let ws = heading_ws_len(&s[1 + n..]);
    if s.get(1 + n + ws) != Some(&'卷') {
        return false;
    }
    for k in (2 + n + ws)..s.len() {
        if (s[k] == '第' && (chinese_di_heading(&s[k..], true) || missing_unit_di_heading(&s[k..])))
            || bare_numeral_zhang_heading(&s[k..])
        {
            return true;
        }
    }
    false
}

fn prefixed_chinese_heading(rest: &[char]) -> bool {
    let rest = trim_heading_decor(rest);
    chinese_di_heading(rest, true)
        || tight_standalone_zhang_heading(rest)
        || volume_then_chapter(rest)
        || bare_numeral_zhang_heading(rest)
        || missing_unit_di_heading(rest)
}

/// `第` then a numeral, then one of 章節回卷篇部 (plus web-novel `VIP第...卷...`,
/// `正文第...卷...第...章...`, `<Prefix>\u{00a0}<Title>`, `<Prefix> 第...章`, `第...卷...第...章`,
/// `<numerals>章 <Title>`, `第<numerals>=2> <short_title>`, tight `第<num>章<title>`, and `»»»`/`·`
/// heading separators). At most 60 chars with no sentence end.
fn chinese_heading(t: &[char]) -> bool {
    if t.len() > 60 || t.iter().any(|ch| "。！？".contains(*ch)) {
        return false;
    }
    if chinese_di_heading(t, false) {
        return true;
    }
    if t.iter().any(|ch| "，；,;!?'\"「」『』“”‘’".contains(*ch)) {
        return false;
    }
    if tight_standalone_zhang_heading(t)
        || bare_numeral_zhang_heading(t)
        || missing_unit_di_heading(t)
        || volume_then_chapter(t)
    {
        return true;
    }
    if matches!(t.first(), Some('»' | '·' | '‧')) && prefixed_chinese_heading(t) {
        return true;
    }
    if starts_with_ci(t, 0, t.len(), "vip") && prefixed_chinese_heading(trim(&t[3..])) {
        return true;
    }
    if t.starts_with(&['正', '文']) && prefixed_chinese_heading(trim(&t[2..])) {
        return true;
    }
    if let Some(nbsp_idx) = t.iter().position(|&ch| ch == '\u{00a0}')
        && (1..=24).contains(&nbsp_idx)
        && let Some(rs) = (nbsp_idx + 1..t.len()).find(|&i| !t[i].is_whitespace())
        && prefixed_chinese_heading(&t[rs..])
    {
        return true;
    }
    for idx in 1..=t.len().saturating_sub(1).min(30) {
        if t[idx] == '第' && (t[idx - 1].is_whitespace() || matches!(t[idx - 1], '»' | '·' | '‧'))
        {
            let has_vol_or_decor = t[..idx]
                .iter()
                .any(|&ch| matches!(ch, '卷' | '»' | '·' | '‧'));
            if chinese_di_heading(&t[idx..], has_vol_or_decor)
                || (has_vol_or_decor && tight_standalone_zhang_heading(&t[idx..]))
                || volume_then_chapter(&t[idx..])
                || missing_unit_di_heading(&t[idx..])
            {
                return true;
            }
        }
    }
    false
}

/// Japanese headings: `第<num><unit>[sep title | tight non-particle title]`, bare `<num>話 <title>`,
/// and standalone structural headings (`プロローグ`, `エピローグ`, `幕間`, `序章`, `終章`, `最終話`, `最終章`, `番外編`).
const JA_HEADING_NUMERALS: &str = "一二三四五六七八九十百千万零〇壱弐参";
const JA_HEADING_UNITS: &str = "話巻章節回部編幕条項";
const JA_STRUCTURAL_HEADINGS: &[&str] = &[
    "プロローグ",
    "エピローグ",
    "幕間",
    "序章",
    "終章",
    "最終話",
    "最終章",
    "番外編",
];

fn is_ja_numeral(ch: char) -> bool {
    ch.is_ascii_digit() || ('０'..='９').contains(&ch) || JA_HEADING_NUMERALS.contains(ch)
}

fn ja_numeral_len(s: &[char]) -> usize {
    s.iter().take_while(|&&ch| is_ja_numeral(ch)).count()
}

fn is_ja_heading_sep(ch: char) -> bool {
    is_heading_sep(ch) || matches!(ch, '・' | '―' | '—' | '-' | '〜' | '～' | '\u{3000}')
}

fn is_ja_subtitle_sep(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '：' | ':' | '・' | '―' | '—' | '-' | '〜' | '～')
}

fn has_ja_heading_title(chars: &[char]) -> bool {
    chars.iter().any(|ch| {
        !ch.is_whitespace()
            && !matches!(
                *ch,
                '：' | ':' | '・' | '―' | '—' | '-' | '〜' | '～' | '·' | '‧' | '»'
            )
    })
}

fn starts_with_ja_particle(s: &[char]) -> bool {
    let Some(&first) = s.first() else {
        return false;
    };
    if matches!(
        first,
        'は' | 'が' | 'を' | 'に' | 'で' | 'の' | 'と' | 'も' | 'へ' | 'や'
    ) {
        return true;
    }
    if s == ['だ'] {
        return true;
    }
    [
        "から",
        "まで",
        "より",
        "だけ",
        "ほど",
        "など",
        "こそ",
        "さえ",
        "しか",
        "すら",
        "だった",
        "である",
        "です",
    ]
    .iter()
    .any(|p| matches_chars_at(s, 0, s.len(), p))
}

fn starts_with_zh_prose_continuation(s: &[char]) -> bool {
    s.first().is_some_and(|&ch| {
        matches!(
            ch,
            '的' | '之'
                | '了'
                | '是'
                | '在'
                | '有'
                | '中'
                | '裡'
                | '里'
                | '內'
                | '内'
                | '講'
                | '讲'
                | '說'
                | '说'
                | '討'
                | '讨'
                | '介'
                | '論'
                | '论'
        )
    })
}

fn japanese_dai_heading(t: &[char]) -> bool {
    if t.first() != Some(&'第') {
        return false;
    }
    let numerals = ja_numeral_len(&t[1..]);
    if numerals == 0 {
        return false;
    }
    let ws = heading_ws_len(&t[1 + numerals..]);
    let [unit, after @ ..] = &t[1 + numerals + ws..] else {
        return false;
    };
    if !JA_HEADING_UNITS.contains(*unit) {
        return false;
    }
    match after {
        [] => ws == 0,
        [sep, title @ ..] if is_ja_heading_sep(*sep) => has_ja_heading_title(title),
        _ => {
            ws == 0
                && (1..=20).contains(&after.len())
                && (matches!(*unit, '話' | '巻')
                    || after.len() >= 2
                    || after.iter().any(|&ch| is_kana(ch)))
                && !starts_with_ja_particle(after)
                && !starts_with_zh_prose_continuation(after)
                && !after
                    .iter()
                    .any(|ch| "，。！？；：、,.;:!?'\"「」『』“”‘’".contains(*ch))
        }
    }
}

fn japanese_bare_wa_heading(t: &[char]) -> bool {
    if t.len() > 40 || t.iter().any(|ch| matches!(*ch, '，' | '、' | ',')) {
        return false;
    }
    let numerals = ja_numeral_len(t);
    if !(1..=4).contains(&numerals) || t.get(numerals) != Some(&'話') {
        return false;
    }
    matches!(
        &t[numerals + 1..],
        [sep, title @ ..] if is_ja_subtitle_sep(*sep) && has_ja_heading_title(title)
    )
}

fn japanese_structural_heading(t: &[char]) -> bool {
    if t.len() > 40
        || t.iter()
            .any(|ch| matches!(*ch, '，' | '、' | ',' | '；' | ';'))
    {
        return false;
    }
    for kw in JA_STRUCTURAL_HEADINGS {
        if matches_chars_at(t, 0, t.len(), kw) {
            let rest = &t[kw.chars().count()..];
            return match rest {
                [] => true,
                [sep, subtitle @ ..] => is_ja_subtitle_sep(*sep) && has_ja_heading_title(subtitle),
            };
        }
    }
    false
}

fn japanese_heading(t: &[char]) -> bool {
    if t.is_empty() || t.len() > 60 || t.iter().any(|ch| matches!(*ch, '。' | '！' | '？' | '｡'))
    {
        return false;
    }
    japanese_dai_heading(t) || japanese_bare_wa_heading(t) || japanese_structural_heading(t)
}

/// Korean headings: `제<num><unit>[sep title]`, bare `<1..=4 digits>화` / `<1..=3 digits>장`,
/// and standalone structural headings (`프롤로그`, `에필로그`, `서장`, `종장`, `외전`).
const KO_HEADING_NUMERALS: &str = "일이삼사오육륙칠팔구십백천만영공";
const KO_HEADING_UNITS: &str = "장절화회권부편막";
const KO_STRUCTURAL_HEADINGS: &[&str] = &["프롤로그", "에필로그", "서장", "종장", "외전"];

fn is_ko_numeral(ch: char) -> bool {
    ch.is_ascii_digit() || ('０'..='９').contains(&ch) || KO_HEADING_NUMERALS.contains(ch)
}

fn ko_numeral_len(s: &[char]) -> usize {
    s.iter().take_while(|&&ch| is_ko_numeral(ch)).count()
}

fn is_ko_heading_sep(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, ':' | '：' | '-' | '—' | '·' | '‧' | '.' | '―')
}

fn has_ko_heading_title(chars: &[char]) -> bool {
    chars.iter().any(|ch| {
        !ch.is_whitespace() && !matches!(*ch, ':' | '：' | '-' | '—' | '·' | '‧' | '.' | '―')
    })
}

fn starts_with_ko_particle(s: &[char]) -> bool {
    ["에서", "으로"]
        .iter()
        .any(|p| matches_chars_at(s, 0, s.len(), p))
        || s.first().is_some_and(|&ch| {
            matches!(
                ch,
                '은' | '는'
                    | '이'
                    | '가'
                    | '을'
                    | '를'
                    | '의'
                    | '에'
                    | '도'
                    | '만'
                    | '과'
                    | '와'
                    | '로'
            )
        })
}

fn ends_with_ko_prose_verb(title: &[char]) -> bool {
    let title = trim(title);
    let Some(&last) = title.last() else {
        return false;
    };
    if matches!(last, '요' | '죠' | '까') {
        return true;
    }
    if last != '다' {
        return false;
    }
    let has_verb_suffix = [
        "이다",
        "하다",
        "한다",
        "했다",
        "된다",
        "됐다",
        "있다",
        "없다",
        "같다",
        "않다",
        "는다",
        "습니다",
        "입니다",
    ]
    .iter()
    .any(|suf| {
        let n = suf.chars().count();
        title.len() >= n && matches_chars_at(title, title.len() - n, title.len(), suf)
    });
    let has_past_ssangsiot = title.len() >= 2
        && is_hangul_syllable(title[title.len() - 2])
        && (title[title.len() - 2] as u32 - 0xAC00) % 28 == 20;
    has_verb_suffix || has_past_ssangsiot
}

fn korean_je_heading(t: &[char]) -> bool {
    if t.first() != Some(&'제') {
        return false;
    }
    let ws1 = heading_ws_len(&t[1..]);
    let numerals = ko_numeral_len(&t[1 + ws1..]);
    if numerals == 0 {
        return false;
    }
    let ws2 = heading_ws_len(&t[1 + ws1 + numerals..]);
    let [unit, after @ ..] = &t[1 + ws1 + numerals + ws2..] else {
        return false;
    };
    if !KO_HEADING_UNITS.contains(*unit) || starts_with_ko_particle(after) {
        return false;
    }
    match after {
        [] => true,
        [sep, title @ ..] => {
            is_ko_heading_sep(*sep)
                && has_ko_heading_title(title)
                && !ends_with_ko_prose_verb(title)
        }
    }
}

fn korean_bare_episode_heading(t: &[char]) -> bool {
    let digits = t.iter().take_while(|ch| ch.is_ascii_digit()).count();
    if digits == 0 {
        return false;
    }
    let Some(&unit) = t.get(digits) else {
        return false;
    };
    let valid_unit =
        (unit == '화' && (1..=4).contains(&digits)) || (unit == '장' && (1..=3).contains(&digits));
    if !valid_unit {
        return false;
    }
    let after = &t[digits + 1..];
    if starts_with_ko_particle(after) {
        return false;
    }
    match after {
        [] => true,
        [sep, raw_title @ ..]
            if matches!(*sep, ' ' | '\t' | '\u{3000}' | ':' | '：' | '-' | '—' | '.') =>
        {
            let title = trim(raw_title);
            (1..=30).contains(&title.len())
                && has_ko_heading_title(title)
                && !title.iter().any(|ch| matches!(*ch, ',' | '，'))
                && !matches!(title.last(), Some('다' | '요' | '죠' | '까' | '네'))
        }
        _ => false,
    }
}

fn korean_structural_heading(t: &[char]) -> bool {
    if t.len() > 40 || t.iter().any(|ch| matches!(*ch, ',' | '，' | ';' | '；')) {
        return false;
    }
    for kw in KO_STRUCTURAL_HEADINGS {
        if matches_chars_at(t, 0, t.len(), kw) {
            let rest = &t[kw.chars().count()..];
            if starts_with_ko_particle(rest) {
                return false;
            }
            return match rest {
                [] => true,
                [sep, subtitle @ ..] => {
                    (sep.is_whitespace()
                        || matches!(*sep, ':' | '：' | '-' | '—' | '·' | '‧' | '―'))
                        && has_ko_heading_title(subtitle)
                        && !ends_with_ko_prose_verb(subtitle)
                }
            };
        }
    }
    false
}

fn korean_heading(t: &[char]) -> bool {
    if t.is_empty()
        || t.len() > 60
        || t.last() == Some(&'.')
        || t.iter()
            .any(|ch| matches!(*ch, '!' | '?' | '。' | '！' | '？'))
    {
        return false;
    }
    korean_je_heading(t) || korean_bare_episode_heading(t) || korean_structural_heading(t)
}

/// `chapter`, `part` or `section`, one space, then digits or Roman numerals, then the end of the
/// line or a separator and a title. The line is at most 60 chars.
fn english_heading(t: &[char]) -> bool {
    if t.len() > 60 {
        return false;
    }
    ["chapter", "part", "section"].iter().any(|word| {
        let n = word.len();
        let starts = t.len() > n
            && t[..n]
                .iter()
                .zip(word.chars())
                .all(|(a, b)| a.eq_ignore_ascii_case(&b));
        starts && t[n] == ' ' && numeral_then_end_or_title(&t[n + 1..])
    })
}

fn numeral_then_end_or_title(rest: &[char]) -> bool {
    let numerals = rest
        .iter()
        .take_while(|ch| ch.is_ascii_digit() || "IVXLC".contains(**ch))
        .count();
    numerals > 0
        && match &rest[numerals..] {
            [] => true,
            [sep, title @ ..] => matches!(*sep, ':' | '.' | '-' | '—') && has_text(title),
        }
}

/// A bullet or a number of 1 to 3 digits, then `.` or `)`. Each marker needs the space after it.
fn is_list_item(t: &[char]) -> bool {
    const BULLETS: &[char] = &['-', '*', '+', '•', '・', '※'];
    match t {
        [marker, sep, ..] if BULLETS.contains(marker) => {
            matches!(*sep, ' ' | '\t' | '\u{3000}')
        }
        _ => {
            let digits = t.iter().take_while(|ch| ch.is_ascii_digit()).count();
            (1..=3).contains(&digits)
                && matches!(t.get(digits).copied(), Some('.' | ')'))
                && t.get(digits + 1).copied() == Some(' ')
        }
    }
}

/// True when every non-whitespace character on `c[s..t_end]` is raw HTML markup, HTML entity,
/// Unicode PUA noise, or (when `mask_editorial`) an inline scraper/editorial tag.
fn is_raw_noise_only_line(
    c: &[char],
    ls: usize,
    s: usize,
    t_end: usize,
    le: usize,
    mask_editorial: bool,
) -> bool {
    let mut i = s;
    let mut any = false;
    while i < t_end {
        if c[i].is_whitespace() {
            i += 1;
            continue;
        }
        let next = raw_html_tag(c, i, ls, le)
            .or_else(|| html_entity(c, i, le))
            .or_else(|| pua_run(c, i, le))
            .or_else(|| {
                if mask_editorial {
                    editorial_inline_at(c, i, ls, le)
                } else {
                    None
                }
            });
        match next {
            Some(end) if end <= t_end => {
                any = true;
                i = end;
            }
            _ => return false,
        }
    }
    any
}

const EDITORIAL_CUES: &[&str] = &[
    "月票",
    "雙倍月票",
    "双倍月票",
    "推薦票",
    "推荐票",
    "求月票",
    "求票",
    "求下票",
    "第一更",
    "第二更",
    "第三更",
    "第四更",
    "第五更",
    "加更",
    "碼字",
    "码字",
    "今天只能一更",
    "今日只能一更",
    "不得不向大家請假",
    "不得不向大家请假",
    "如果您喜歡這部作品",
    "如果您喜欢这部作品",
    "請為作者",
    "请为作者",
    "待續",
    "待续",
    "加入書架",
    "加入书架",
    "熱門書籍",
    "热门书籍",
    "熱門標簽",
    "熱門標籤",
    "热门标签",
    "書友正在讀",
    "书友正在读",
    "收集于網絡",
    "收集于网络",
    "收集於網路",
    "收集於網絡",
    "轉載作品",
    "转载作品",
    "請別忘本書",
    "请别忘本书",
    "鼎立支持",
    "票票",
    "投一票",
];

const SCRAPER_BOILERPLATE_CUES: &[&str] = &[
    "一一小說",
    "一一小说",
    "全本小說吧",
    "全本小说吧",
    "在線書吧",
    "在线书吧",
    "看書窩",
    "看书窝",
    "56小說",
    "56小说",
    "書家文學",
    "书家文学",
    "回味書庫",
    "回味书库",
    "理想文學",
    "理想文学",
    "云軒閣",
    "云轩阁",
    "雲軒閣",
    "七星閣",
    "七星阁",
    "第一中文",
    "搜/搜/9",
    "最新全本：",
    "最新全本:",
    "閱讀提示",
    "阅读提示",
    "瀏覽貼子",
    "浏览贴子",
    "轉貼次數",
    "转贴次数",
    "按左右鍵快速翻到上下章節",
    "按左右键快速翻到上下章节",
    "724小時不間斷更新",
    "724小时不间断更新",
    "手機,電腦.還可以下載電子書",
    "手机,电脑.还可以下载电子书",
    "還可以下載電子書",
    "还可以下载电子书",
    "热门小说标签",
    "熱門小說標簽",
    "熱門小說標籤",
    "熱門作品推薦",
    "热门作品推荐",
    "其他書友吐血推薦",
    "其他书友吐血推荐",
    "章節更多",
    "章节更多",
    "爬書",
    "爬书",
    "本周:",
    "本周：",
    "本週:",
    "本週：",
];

const SCRAPER_ASCII_CUES: &[&str] = &[
    "ddyueshu", "soso999", "3qzw", "qixingge", "qixinge", "qbxs8", "56xs", "wap77", "0_3_wx",
    "03wx",
];

const INLINE_PIRATE_SITES: &[&str] = &[
    "就到一一小說",
    "就到一一小说",
    "手機站直接訪問云軒閣",
    "手机站直接访问云轩阁",
    "手機站直接訪問雲軒閣",
    "書家文學",
    "书家文学",
    "56小說",
    "56小说",
    "一一小說",
    "一一小说",
    "云軒閣",
    "雲軒閣",
    "云轩阁",
    "七星閣",
    "七星阁",
    "回味書庫",
    "回味书库",
    "理想文學",
    "理想文学",
    "全本小說吧",
    "全本小说吧",
    "在線書吧",
    "在线书吧",
    "看書窩",
    "看书窝",
];

const WEB_FOOTER_META_CUES: &[&str] = &[
    "小說",
    "小说",
    "網址",
    "网址",
    "域名",
    "連載",
    "连载",
    "章節",
    "章节",
    "手機站",
    "手机站",
    "電腦站",
    "电脑站",
    "整理提供",
];

const UNBRACKETED_POSTSCRIPT_STARTERS: &[&str] = &[
    "如果您喜歡這部作品",
    "如果您喜欢这部作品",
    "如欲知后事如何",
    "如欲知後事如何",
    "另外雙倍月票",
    "另外双倍月票",
    "另外，雙倍月票",
    "另外，双倍月票",
    "今天只能一更",
    "今日只能一更",
    "不得不向大家請假",
    "不得不向大家请假",
];

fn contains_substr(chars: &[char], pat: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    !p.is_empty() && chars.windows(p.len()).any(|w| w == p.as_slice())
}

fn contains_substr_ci(chars: &[char], pat: &str) -> bool {
    let n = pat.len();
    n > 0
        && chars.windows(n).any(|w| {
            w.iter()
                .zip(pat.chars())
                .all(|(ch, p)| ch.to_ascii_lowercase() == p)
        })
}

fn contains_scraper_boilerplate(chars: &[char]) -> bool {
    SCRAPER_BOILERPLATE_CUES
        .iter()
        .any(|cue| contains_substr(chars, cue))
        || SCRAPER_ASCII_CUES
            .iter()
            .any(|cue| contains_substr_ci(chars, cue))
}

fn contains_bracketed_update_cue(chars: &[char]) -> bool {
    for cue in [
        "請假",
        "请假",
        "兩更",
        "两更",
        "書友",
        "书友",
        "更新",
        "碼出",
        "码出",
        "碼完",
        "码完",
        "章節數",
        "章节数",
        "章節名",
        "章节名",
        "大家見諒",
        "大家见谅",
        "，支持正",
        ",支持正",
        "支持正版",
    ] {
        if contains_substr(chars, cue) {
            return true;
        }
    }
    let has_author_time_or_audience = [
        "大家",
        "今天",
        "今日",
        "昨天",
        "昨日",
        "明天",
        "明日",
        "這兩日",
        "这两日",
        "晚上",
        "少發",
        "少发",
    ]
    .iter()
    .any(|w| contains_substr(chars, w));
    if has_author_time_or_audience
        && ["補上", "补上", "一章", "兩章", "两章", "三章", "碼", "码"]
            .iter()
            .any(|w| contains_substr(chars, w))
    {
        return true;
    }
    for (idx, w) in chars.windows(2).enumerate() {
        if w == ['一', '更'] {
            let next = chars.get(idx + 2).copied();
            if !matches!(next, Some('天' | '時' | '时')) {
                return true;
            }
        }
        if w == ['三', '更'] {
            let next = chars.get(idx + 2).copied();
            if !matches!(next, Some('半' | '天' | '時' | '时' | '夜')) {
                return true;
            }
        }
    }
    false
}

fn contains_any_editorial_cue(chars: &[char], include_baodi: bool) -> bool {
    EDITORIAL_CUES.iter().any(|cue| contains_substr(chars, cue))
        || contains_scraper_boilerplate(chars)
        || (include_baodi && contains_substr(chars, "保底"))
}

fn contains_inline_editorial_cue(chars: &[char]) -> bool {
    contains_any_editorial_cue(chars, true) || contains_bracketed_update_cue(chars)
}

fn multi_paren_tag_list_line(t: &[char]) -> bool {
    let mut i = 0;
    let mut groups = 0usize;
    while i < t.len() {
        while i < t.len() && (t[i].is_whitespace() || matches!(t[i], ',' | '，' | '、' | '/')) {
            i += 1;
        }
        if i >= t.len() {
            break;
        }
        if !matches!(t[i], '(' | '（') {
            return false;
        }
        let start = i + 1;
        let mut j = start;
        while j < t.len() && !matches!(t[j], '(' | '（' | ')' | '）') {
            j += 1;
        }
        if j >= t.len() || !matches!(t[j], ')' | '）') {
            return false;
        }
        let inner = trim(&t[start..j]);
        if inner.is_empty() || inner.len() > 24 {
            return false;
        }
        groups += 1;
        i = j + 1;
    }
    groups >= 4
}

/// Find the end offset (`k + 1`) of a leading editorial sentence inside an unclosed `(` line when
/// the remainder of the line is multi-sentence story prose with zero editorial/author-note cues.
fn unclosed_editorial_prefix_before_story(c: &[char], open_idx: usize, le: usize) -> Option<usize> {
    let end = trimmed_end(c, open_idx, le);
    if end <= open_idx + 2 {
        return None;
    }
    for k in (open_idx + 2)..end {
        if matches!(c[k], '。' | '！' | '？' | '!' | '?')
            && contains_inline_editorial_cue(&c[open_idx + 1..=k])
        {
            let tail = &c[k + 1..end];
            let tail_han = tail.iter().filter(|&&ch| is_han(ch)).count();
            let tail_sents = tail
                .iter()
                .filter(|&&ch| matches!(ch, '。' | '！' | '？' | '!' | '?'))
                .count();
            let tail_has_author_talk = contains_inline_editorial_cue(tail)
                || ["大家", "書友", "书友", "咱", "抱歉", "見諒", "见谅"]
                    .iter()
                    .any(|w| contains_substr(tail, w));
            if tail_han >= 20 && tail_sents >= 2 && !tail_has_author_talk {
                return Some(k + 1);
            }
        }
    }
    None
}

fn strip_inline_parens_and_postscripts(t: &[char], strip_watermarks: bool) -> Vec<char> {
    let mut out = Vec::with_capacity(t.len());
    let mut i = 0;
    while i < t.len() {
        if strip_watermarks
            && let Some(end) = editorial_inline_at(t, i, 0, t.len())
            && end > i
        {
            i = end;
            continue;
        }
        if matches!(t[i], '(' | '（' | '[' | '【' | '［') {
            let mut j = i + 1;
            let mut depth = 1usize;
            while j < t.len() && depth > 0 {
                if matches!(t[j], '(' | '（' | '[' | '【' | '［') {
                    depth += 1;
                } else if matches!(t[j], ')' | '）' | ']' | '】' | '］') {
                    depth -= 1;
                }
                j += 1;
            }
            if depth == 0 {
                i = j;
                continue;
            }
        }
        // If an unbracketed postscript starts after an earlier sentence on the same line, strip it
        // here so `is_editorial_block_line` does not blank the preceding story sentence; `inline`
        // will mask only the trailing postscript span.
        if i > 0 && matches!(t[i - 1], '。' | '！' | '？' | '!' | '?') && t.len() - i <= 240 {
            let start = if matches!(t[i], '，' | ',') {
                i + 1
            } else {
                i
            };
            if UNBRACKETED_POSTSCRIPT_STARTERS
                .iter()
                .any(|pat| matches_chars_at(t, start, t.len(), pat))
            {
                break;
            }
        }
        out.push(t[i]);
        i += 1;
    }
    out
}

fn count_prose_sentence_ends(chars: &[char]) -> usize {
    chars
        .iter()
        .filter(|&&ch| matches!(ch, '。' | '！' | '？' | '!' | '?'))
        .count()
}

fn is_editorial_block_line(t: &[char]) -> bool {
    if multi_paren_tag_list_line(t) {
        return true;
    }
    if matches!(t.first(), Some('(' | '（' | '【' | '[' | '［')) && contains_inline_editorial_cue(t)
    {
        let stripped_end = t
            .iter()
            .rposition(|ch| !matches!(*ch, '！' | '!' | '。' | '.' | '~' | '～'))
            .map_or(0, |idx| idx + 1);
        let closed =
            stripped_end > 1 && matches!(t[stripped_end - 1], ')' | '）' | '】' | ']' | '］');
        let mut depth = 1usize;
        let mut closed_early = false;
        for &ch in &t[1..stripped_end.max(1)] {
            if matches!(ch, '(' | '（' | '【' | '[' | '［') {
                depth += 1;
            } else if matches!(ch, ')' | '）' | '】' | ']' | '］') {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    closed_early = true;
                    break;
                }
            }
        }
        if closed
            || (!closed_early && unclosed_editorial_prefix_before_story(t, 0, t.len()).is_none())
        {
            return true;
        }
    }
    if t.len() > 400 || t.iter().any(|ch| "「」『』“”\"'".contains(*ch)) {
        return false;
    }
    let paren_stripped = strip_inline_parens_and_postscripts(t, false);
    let clean_outside = strip_inline_parens_and_postscripts(t, true);
    if clean_outside.iter().any(|ch| "「」『』“”\"'".contains(*ch)) {
        return false;
    }
    let clean_han = clean_outside.iter().filter(|&&ch| is_han(ch)).count();
    let periods = clean_outside.iter().filter(|&&ch| ch == '。').count();
    let sent_ends = count_prose_sentence_ends(&clean_outside);
    let commas = clean_outside
        .iter()
        .filter(|&&ch| matches!(ch, '，' | ','))
        .count();

    if EDITORIAL_CUES
        .iter()
        .any(|cue| contains_substr(&clean_outside, cue))
    {
        return t.len() <= 300 && periods <= 2 && sent_ends <= 4;
    }
    if contains_scraper_boilerplate(&clean_outside) {
        return clean_han <= 35 && periods <= 1 && sent_ends <= 2;
    }
    if contains_scraper_boilerplate(&paren_stripped) {
        let has_web_meta = WEB_FOOTER_META_CUES
            .iter()
            .any(|cue| contains_substr(&clean_outside, cue));
        return (clean_han <= 12 && periods == 0 && sent_ends == 0 && commas == 0)
            || (has_web_meta && clean_han <= 35 && periods <= 1 && sent_ends <= 2);
    }
    contains_substr(&clean_outside, "保底") && t.len() <= 80 && periods == 0 && clean_han <= 35
}

/// Scan one line for inline spans. At each position the kinds are tried in order.
fn inline(c: &[char], ls: usize, le: usize, mask_editorial: bool, out: &mut Vec<Found>) {
    let mut i = ls;
    while i < le {
        match inline_at(c, i, ls, le, mask_editorial) {
            Some((kind, end, is_noise_source)) => {
                out.push(Found {
                    item: Opaque {
                        kind,
                        span: Span {
                            start_char: i,
                            end_char: end,
                        },
                    },
                    block: false,
                    silent_noise: mask_editorial && is_noise_source,
                });
                i = end;
            }
            // A backtick run without a closing run is plain text, so skip the whole run.
            None if c[i] == '`' => i += run_len(c, i, le, '`'),
            None => i += 1,
        }
    }
}

fn inline_at(
    c: &[char],
    i: usize,
    ls: usize,
    le: usize,
    mask_editorial: bool,
) -> Option<(OpaqueKind, usize, bool)> {
    if c[i] == '`' {
        return code_span(c, i, le).map(|end| (OpaqueKind::Code, end, false));
    }
    raw_html_tag(c, i, ls, le)
        .map(|end| (OpaqueKind::Code, end, true))
        .or_else(|| html_entity(c, i, le).map(|end| (OpaqueKind::Code, end, true)))
        .or_else(|| pua_run(c, i, le).map(|end| (OpaqueKind::Code, end, true)))
        .or_else(|| {
            if mask_editorial {
                editorial_inline_at(c, i, ls, le).map(|end| (OpaqueKind::Code, end, true))
            } else {
                None
            }
        })
        .or_else(|| url(c, i, le).map(|end| (OpaqueKind::Url, end, false)))
        .or_else(|| citation(c, i, le).map(|end| (OpaqueKind::Citation, end, false)))
        .or_else(|| inline_formula(c, i, le).map(|end| (OpaqueKind::Formula, end, false)))
        .or_else(|| emoji(c, i, le).map(|end| (OpaqueKind::Emoji, end, false)))
}

/// Raw/unclosed HTML tags (`<P`, `</P`, `<p>`, `</p>`, `<br>`, `<br/>`, `<!go`, `<!over`,
/// `<quot;0" cellpadding=...>`, etc., case-insensitive).
fn raw_html_tag(c: &[char], i: usize, ls: usize, le: usize) -> Option<usize> {
    if c[i] != '<' {
        return None;
    }
    for closed in [
        "</p>", "<p>", "<br/>", "<br />", "<br>", "</div>", "<div>", "</span>", "<span>",
        "</table>", "<table>", "</tr>", "<tr>", "</td>", "<td>", "</font>", "<font>", "<hr/>",
        "<hr />", "<hr>",
    ] {
        if starts_with_ci(c, i, le, closed) {
            return Some(i + closed.len());
        }
    }
    // Single-line HTML comment `<!-- ... -->`.
    if starts_with_ci(c, i, le, "<!--")
        && let Some(rel) = c[i + 4..le].windows(3).position(|w| w == ['-', '-', '>'])
    {
        return Some(i + 4 + rel + 3);
    }
    // Unclosed or closed SGML scraper markers (`<!go`, `<!go>`, `<!over`, `<!over>`).
    for sgml in ["<!over", "<!go"] {
        if starts_with_ci(c, i, le, sgml) {
            let end = i + sgml.len();
            if end < le && c[end] == '>' {
                return Some(end + 1);
            }
            if end == le || !c[end].is_ascii_alphabetic() {
                return Some(end);
            }
        }
    }
    // Closing HTML tag `</` + 1..=12 ASCII letters, with optional closing `>` when bounded by
    // whitespace, line end, CJK, or non-alphanumeric punctuation (`</P`, `</p`, `</div`, etc.).
    if i + 2 < le && c[i + 1] == '/' && c[i + 2].is_ascii_alphabetic() {
        let letters = (i + 2..le.min(i + 15))
            .take_while(|&k| c[k].is_ascii_alphabetic())
            .count();
        if (1..=12).contains(&letters) {
            let end = i + 2 + letters;
            if end < le && c[end] == '>' {
                return Some(end + 1);
            }
            if end == le
                || c[end].is_whitespace()
                || is_han(c[end])
                || !c[end].is_ascii_alphanumeric()
            {
                return Some(end);
            }
        }
    }
    // Truncated HTML attribute fragments (`<quot;0" cellpadding=...>`) or closed HTML tags with
    // `attr=...` attributes (`<table border="0" ...>`, `<font color="...">`, `<a href="...">`).
    let scan_limit = le.min(i + 160);
    if let Some(rel_gt) = (i + 2..scan_limit)
        .take_while(|&k| c[k] != '<')
        .find(|&k| c[k] == '>')
    {
        if starts_with_ci(c, i, le, "<quot;") {
            return Some(rel_gt + 1);
        }
        let tag_letters = (i + 1..rel_gt)
            .take_while(|&k| c[k].is_ascii_alphabetic())
            .count();
        if (1..=10).contains(&tag_letters)
            && i + 1 + tag_letters < rel_gt
            && c[i + 1 + tag_letters].is_ascii_whitespace()
            && c[i + 1 + tag_letters..rel_gt].contains(&'=')
        {
            return Some(rel_gt + 1);
        }
    }
    // Standalone or boundary unclosed `<P` / `<p` followed by whitespace, line end, or Han.
    if starts_with_ci(c, i, le, "<p") {
        let at_boundary = i == ls || !c[i - 1].is_ascii_alphanumeric();
        let end = i + 2;
        let valid_after = end == le || c[end].is_whitespace() || is_han(c[end]);
        if at_boundary && valid_after {
            return Some(end);
        }
    }
    None
}

/// Corrupted or raw HTML entities (`&qut;`, `&quot;`, `&nbsp;`, `&amp;`, `&lt;`, `&gt;`, etc.).
fn html_entity(c: &[char], i: usize, le: usize) -> Option<usize> {
    if c[i] != '&' {
        return None;
    }
    for ent in [
        "&qut;", "&qu0t;", "&qout;", "&quot;", "&nbsp;", "&nbps;", "&amp;", "&lt;", "&gt;",
        "&apos;", "&ldquo;", "&rdquo;", "&lsquo;", "&rsquo;", "&hellip;", "&mdash;", "&ndash;",
    ] {
        if starts_with_ci(c, i, le, ent) {
            return Some(i + ent.len());
        }
    }
    // Numeric character entity `&#1234;` or `&#x1A2F;`.
    if i + 3 < le && c[i + 1] == '#' {
        let (hex, d_start) = if matches!(c[i + 2], 'x' | 'X') {
            (true, i + 3)
        } else {
            (false, i + 2)
        };
        let digits = (d_start..le.min(d_start + 7))
            .take_while(|&k| {
                if hex {
                    c[k].is_ascii_hexdigit()
                } else {
                    c[k].is_ascii_digit()
                }
            })
            .count();
        if (1..=6).contains(&digits) && d_start + digits < le && c[d_start + digits] == ';' {
            return Some(d_start + digits + 1);
        }
    }
    None
}

/// Run of Unicode Private Use Area chars (`\u{E000}..=\u{F8FF}`), plus an optional attached 1-2
/// char ASCII alphanumeric scraper tail when the PUA run has `>= 10` chars.
fn pua_run(c: &[char], i: usize, le: usize) -> Option<usize> {
    if !is_pua(c[i]) {
        return None;
    }
    let mut end = i + 1;
    while end < le && is_pua(c[end]) {
        end += 1;
    }
    if end - i >= 10 {
        let tail = (end..le.min(end + 3))
            .take_while(|&k| c[k].is_ascii_alphanumeric())
            .count();
        if (1..=2).contains(&tail) {
            end += tail;
        }
    }
    Some(end)
}

fn is_pua(ch: char) -> bool {
    matches!(ch, '\u{E000}'..='\u{F8FF}')
}

fn matches_chars_at(c: &[char], i: usize, le: usize, pat: &str) -> bool {
    let n = pat.chars().count();
    i + n <= le && c[i..i + n].iter().copied().eq(pat.chars())
}

fn is_obfuscated_domain_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '.' | '。' | '-' | '_' | 'ω' | '０'..='９')
}

fn consume_attached_asterisk_tag(c: &[char], pos: usize, le: usize) -> usize {
    if pos < le && c[pos] == '＊' {
        let limit = le.min(pos + 16);
        if let Some(close) = ((pos + 1)..limit).find(|&k| c[k] == '＊') {
            return close + 1;
        }
    }
    pos
}

/// Unbracketed inline pirate site watermarks (`☆⌒_⌒☆...`, `書家文學`, `就到一一小說`, `請：m2.ddyueshu`,
/// domain-prefixed site banners, etc.) matched strictly on the watermark substring.
fn inline_pirate_watermark_at(c: &[char], i: usize, le: usize) -> Option<usize> {
    let t_end = trimmed_end(c, i, le);
    if i >= t_end {
        return None;
    }

    // 1. Slash-obfuscated `<1..=2 Han>//<1..=3 Han>/首/發[/]` (`愛//書者/首/發`).
    if is_han(c[i]) {
        let h1 = (i..t_end.min(i + 3)).take_while(|&k| is_han(c[k])).count();
        if (1..=2).contains(&h1) && matches_chars_at(c, i + h1, t_end, "//") {
            let s2 = i + h1 + 2;
            let h2 = (s2..t_end.min(s2 + 4))
                .take_while(|&k| is_han(c[k]))
                .count();
            let sf = s2 + h2;
            if (1..=3).contains(&h2)
                && sf + 4 <= t_end
                && c[sf] == '/'
                && c[sf + 1] == '首'
                && c[sf + 2] == '/'
                && matches!(c[sf + 3], '發' | '发')
            {
                let end = if sf + 5 <= t_end && c[sf + 4] == '/' {
                    sf + 5
                } else {
                    sf + 4
                };
                return Some(end);
            }
        }
    }

    // 2. Kaomoji star banners (`☆⌒_⌒☆...☆⌒_⌒☆[ωωω。qiXingge | ＊...＊]`).
    for kao in ["☆⌒_⌒☆", "★⌒_⌒★"] {
        if matches_chars_at(c, i, t_end, kao) {
            let mut end = i + 5;
            let scan_lim = t_end.min(end + 20);
            for k in end..scan_lim {
                if matches_chars_at(c, k, t_end, "☆⌒_⌒☆") || matches_chars_at(c, k, t_end, "★⌒_⌒★")
                {
                    end = k + 5;
                    break;
                }
            }
            let mut dom_end = end;
            while dom_end < t_end && is_obfuscated_domain_char(c[dom_end]) {
                dom_end += 1;
            }
            if dom_end > end
                && c[end..dom_end]
                    .iter()
                    .any(|&ch| ch.is_ascii_alphanumeric() || ch == 'ω')
            {
                end = dom_end;
            }
            end = consume_attached_asterisk_tag(c, end, t_end);
            return Some(end);
        }
    }

    // 3. Triple-star tieba/pirate banners (`☆☆☆...☆☆☆`, `☆★☆...☆★☆`, `★☆★...★☆★`).
    if i + 6 <= t_end && c[i..i + 3].iter().all(|ch| matches!(*ch, '☆' | '★')) {
        let scan_lim = t_end.min(i + 80);
        if scan_lim >= i + 6 {
            for k in (i + 3)..=(scan_lim - 3) {
                if c[k..k + 3].iter().all(|ch| matches!(*ch, '☆' | '★')) {
                    return Some(k + 3);
                }
            }
        }
    }

    // 4. Trailing star-search notices (`最新最快的更新盡在☆...☆`, `緊急通知:...` with scraper cue).
    if matches_chars_at(c, i, t_end, "最新最快的更新") && c[i..t_end].contains(&'☆') {
        return Some(t_end);
    }
    if (matches_chars_at(c, i, t_end, "緊急通知:")
        || matches_chars_at(c, i, t_end, "緊急通知：")
        || matches_chars_at(c, i, t_end, "紧急通知:")
        || matches_chars_at(c, i, t_end, "紧急通知："))
        && contains_scraper_boilerplate(&c[i..t_end])
    {
        return Some(t_end);
    }

    // 5. `請：m2.ddyueshu` / `请：m2.ddyueshu` domain watermarks.
    if matches!(c[i], '請' | '请') && i + 2 < t_end && matches!(c[i + 1], '：' | ':') {
        let mut k = i + 2;
        while k < t_end && c[k].is_whitespace() {
            k += 1;
        }
        let dom_start = k;
        while k < t_end
            && (c[k].is_ascii_alphanumeric() || matches!(c[k], '.' | '-' | '_' | '/' | ':'))
        {
            k += 1;
        }
        if k > dom_start + 3 && contains_scraper_boilerplate(&c[dom_start..k]) {
            return Some(k);
        }
    }

    // 6. Domain-prefixed inline pirate banners (`www.0_3_wx.理想文學書友整理提供`, `w.０3_x.om理想文學...`).
    if starts_with_ci(c, i, t_end, "www.")
        || starts_with_ci(c, i, t_end, "wap.")
        || starts_with_ci(c, i, t_end, "w.")
    {
        let mut k = i;
        while k < t_end
            && (c[k].is_ascii_alphanumeric() || matches!(c[k], '.' | '_' | '-' | '０'..='９'))
        {
            k += 1;
        }
        if k > i + 3 {
            for site in INLINE_PIRATE_SITES {
                if matches_chars_at(c, k, t_end, site) {
                    let mut end = k + site.chars().count();
                    for suffix in ["書友整理提供", "书友整理提供"] {
                        if matches_chars_at(c, end, t_end, suffix) {
                            end += suffix.chars().count();
                            break;
                        }
                    }
                    return Some(end);
                }
            }
            if contains_scraper_boilerplate(&c[i..k]) {
                return Some(k);
            }
        }
    }

    // 7. Explicit inline pirate site names (`書家文學[.neΤ | www.]`, `就到一一小說`, `56小說[看(...)]`, etc.).
    for site in INLINE_PIRATE_SITES {
        if matches_chars_at(c, i, t_end, site) {
            let mut end = i + site.chars().count();
            if matches!(*site, "書家文學" | "书家文学") {
                let mut d = end;
                while d < t_end
                    && (c[d].is_ascii_alphanumeric()
                        || matches!(c[d], '.' | '-' | '_' | 'ω' | '\u{03A4}' | '\u{03C4}'))
                {
                    d += 1;
                }
                if d > end {
                    end = d;
                }
            } else if matches!(
                *site,
                "云軒閣"
                    | "雲軒閣"
                    | "云轩阁"
                    | "手機站直接訪問云軒閣"
                    | "手机站直接访问云轩阁"
                    | "手機站直接訪問雲軒閣"
            ) && starts_with_ci(c, end, t_end, "xs")
            {
                end += 2;
            } else if matches!(*site, "56小說" | "56小说")
                && end + 3 < t_end
                && c[end] == '看'
                && matches!(c[end + 1], '(' | '（')
            {
                let lim = t_end.min(end + 42);
                if let Some(close) = ((end + 2)..lim).find(|&k| matches!(c[k], ')' | '）'))
                    && contains_scraper_boilerplate(&c[end + 2..close])
                {
                    end = close + 1;
                }
            } else if matches!(*site, "理想文學" | "理想文学")
                && end + 1 == t_end
                && matches!(c[end], '：' | ':')
            {
                end += 1;
            }
            for suffix in [
                "書友整理提供",
                "书友整理提供",
                "書友整理上傳",
                "书友整理上传",
            ] {
                if matches_chars_at(c, end, t_end, suffix) {
                    end += suffix.chars().count();
                    break;
                }
            }
            return Some(end);
        }
    }

    None
}

/// Opt-in inline scraper/editorial tags (`/首/發`, `(..#..#..)`, `()()(...)`, mixed-bracket
/// `(…）` / `（…)` notes, unbracketed pirate watermarks, and trailing unbracketed postscripts).
fn editorial_inline_at(c: &[char], i: usize, ls: usize, le: usize) -> Option<usize> {
    let t_end = trimmed_end(c, i, le);
    if i >= t_end {
        return None;
    }

    if i + 4 <= t_end
        && c[i] == '/'
        && c[i + 1] == '首'
        && c[i + 2] == '/'
        && matches!(c[i + 3], '發' | '发')
    {
        let end = if i + 5 <= t_end && c[i + 4] == '/' {
            i + 5
        } else {
            i + 4
        };
        return Some(end);
    }

    // Orphan line-start closing editorial parenthesis (`...票票哦！）<story>`).
    if i == ls && !matches!(c[i], '(' | '（' | '[' | '【' | '［') {
        let scan_lim = t_end.min(ls + 72);
        let mut close_pos = None;
        for (idx, &ch) in c.iter().enumerate().take(scan_lim).skip(ls) {
            if matches!(ch, '(' | '（' | '[' | '【' | '［') {
                break;
            }
            if matches!(ch, ')' | '）') {
                close_pos = Some(idx);
                break;
            }
        }
        if let Some(cp) = close_pos
            && cp > ls
            && !c[ls..cp].iter().any(|ch| "「」『』“”\"'".contains(*ch))
            && !c[ls..cp].contains(&'。')
            && contains_inline_editorial_cue(&c[ls..cp.min(ls + 45)])
        {
            return Some(cp + 1);
        }
    }

    if matches!(c[i], '(' | '（' | '[' | '【' | '［') {
        // Advance past up to 4 consecutive leading empty paren pairs `()()(...)` when followed by `(`.
        let mut open_pos = i;
        let mut empty_skipped = 0usize;
        while empty_skipped < 4
            && open_pos + 2 < t_end
            && matches!(c[open_pos], '(' | '（')
            && matches!(c[open_pos + 1], ')' | '）')
            && matches!(c[open_pos + 2], '(' | '（')
        {
            open_pos += 2;
            empty_skipped += 1;
        }

        let limit = t_end.min(open_pos + 242);
        let mut j = open_pos + 1;
        let mut depth = 1usize;
        let mut saw_nested_open = false;
        while j < limit {
            if matches!(c[j], '(' | '（' | '[' | '【' | '［') {
                depth += 1;
                saw_nested_open = true;
            } else if matches!(c[j], ')' | '）' | ']' | '】' | '］') {
                depth -= 1;
                if depth == 0 {
                    let inner = &c[open_pos + 1..j];
                    let hashes = inner.iter().filter(|&&ch| ch == '#').count();
                    let stars = inner.iter().filter(|&&ch| matches!(ch, '☆' | '★')).count();
                    if !inner.is_empty()
                        && inner.len() <= 240
                        && (hashes >= 2 || stars >= 2 || contains_inline_editorial_cue(inner))
                    {
                        let mut end = j + 1;
                        while end + 1 < t_end
                            && matches!(c[end], '(' | '（')
                            && matches!(c[end + 1], ')' | '）')
                        {
                            end += 2;
                        }
                        end = consume_attached_asterisk_tag(c, end, t_end);
                        return Some(end);
                    }
                    break;
                }
            }
            j += 1;
        }
        // Unclosed editorial parenthesis: either split cleanly before multi-sentence story prose,
        // or run to `t_end` when trailing editorial noise reaches the end of the line.
        if depth > 0 && open_pos == i {
            if !saw_nested_open
                && let Some(split_end) = unclosed_editorial_prefix_before_story(c, i, t_end)
            {
                return Some(split_end);
            }
            if depth == 1
                && j == t_end
                && t_end > i + 1
                && t_end - (i + 1) <= 200
                && !c[i + 1..t_end]
                    .iter()
                    .any(|ch| "「」『』“”\"'".contains(*ch))
                && contains_inline_editorial_cue(&c[i + 1..t_end])
            {
                return Some(t_end);
            }
        }
    }

    if let Some(end) = inline_pirate_watermark_at(c, i, t_end) {
        return Some(end);
    }

    // Trailing unbracketed sentence-end postscripts (including an optional orphaned comma right
    // after sentence-final punctuation, e.g. `。，如果您喜歡這部作品...`).
    let at_postscript_boundary = i == ls
        || matches!(
            c[i - 1],
            '。' | '！'
                | '？'
                | '!'
                | '?'
                | '”'
                | '」'
                | '』'
                | '"'
                | ' '
                | '\t'
                | '\u{3000}'
                | '，'
                | ','
        );
    if at_postscript_boundary && t_end - i <= 240 {
        let start = if matches!(c[i], '，' | ',')
            && i > ls
            && matches!(c[i - 1], '。' | '！' | '？' | '!' | '?')
        {
            i + 1
        } else {
            i
        };
        if UNBRACKETED_POSTSCRIPT_STARTERS
            .iter()
            .any(|pat| matches_chars_at(c, start, t_end, pat))
        {
            return Some(t_end);
        }
    }
    None
}

/// A run of N backticks, then the next run of exactly N on the same line.
fn code_span(c: &[char], i: usize, le: usize) -> Option<usize> {
    let n = run_len(c, i, le, '`');
    let mut j = i + n;
    while j < le {
        if c[j] == '`' {
            let m = run_len(c, j, le, '`');
            if m == n {
                return Some(j + m);
            }
            j += m;
        } else {
            j += 1;
        }
    }
    None
}

fn url(c: &[char], i: usize, le: usize) -> Option<usize> {
    const SCHEMES: [&str; 3] = ["https://", "http://", "www."];
    const TRAILING: &[char] = &['.', ',', ';', ':', '!', '?', ')', ']', '}', '\''];
    let scheme = SCHEMES
        .iter()
        .find(|scheme| starts_with_ci(c, i, le, scheme))?;
    let body = i + scheme.len();
    let mut end = body;
    while end < le && c[end].is_ascii_graphic() && !matches!(c[end], '<' | '>' | '"') {
        end += 1;
    }
    while end > body && TRAILING.contains(&c[end - 1]) {
        end -= 1;
    }
    (end > body).then_some(end)
}

/// `prefix` is lowercase ASCII.
fn starts_with_ci(c: &[char], i: usize, le: usize, prefix: &str) -> bool {
    let n = prefix.len();
    i + n <= le
        && c[i..i + n]
            .iter()
            .zip(prefix.chars())
            .all(|(ch, p)| ch.to_ascii_lowercase() == p)
}

fn citation(c: &[char], i: usize, le: usize) -> Option<usize> {
    match c[i] {
        '[' => numeric_citation(c, i, le),
        '(' | '（' => author_year_citation(c, i, le),
        _ => None,
    }
}

/// `[` then digits, `,`, `-`, `–` and spaces with at least one digit, then `]`.
fn numeric_citation(c: &[char], i: usize, le: usize) -> Option<usize> {
    let mut j = i + 1;
    let mut digits = 0;
    while j < le && (c[j].is_ascii_digit() || matches!(c[j], ',' | '-' | '–' | ' ')) {
        digits += usize::from(c[j].is_ascii_digit());
        j += 1;
    }
    (j < le && c[j] == ']' && digits > 0).then_some(j + 1)
}

fn author_year_citation(c: &[char], i: usize, le: usize) -> Option<usize> {
    let mut j = i + 1;
    while j < le && !matches!(c[j], '(' | '（' | ')' | '）') {
        j += 1;
    }
    if j >= le || !matches!(c[j], ')' | '）') {
        return None;
    }
    let content = &c[i + 1..j];
    let valid = content.len() <= 80
        && content
            .split(|ch| matches!(*ch, ';' | '；'))
            .all(is_author_year_part);
    valid.then_some(j + 1)
}

/// Uppercase ASCII, Han, Kana, or Hangul syllable first, at most 60 chars, and a year at the end.
fn is_author_year_part(part: &[char]) -> bool {
    let part = trim(part);
    part.len() <= 60
        && part.first().is_some_and(|&ch| {
            ch.is_ascii_uppercase() || is_han(ch) || is_kana(ch) || is_hangul_syllable(ch)
        })
        && ends_with_year(part)
}

/// A year from 1500 to 2099, optionally followed by one lowercase letter, and preceded by a
/// space, fullwidth space, `,`, `，` or `、`.
fn ends_with_year(part: &[char]) -> bool {
    let mut end = part.len();
    if part.last().is_some_and(|ch| ch.is_ascii_lowercase()) {
        end -= 1;
    }
    if end < 5 {
        return false;
    }
    let year = &part[end - 4..end];
    let sep = part[end - 5];
    let digits: String = year.iter().collect();
    year.iter().all(|ch| ch.is_ascii_digit())
        && (1500..=2099).contains(&digits.parse::<u32>().unwrap_or(0))
        && matches!(sep, ' ' | '\u{3000}' | ',' | '，' | '、')
}

/// `$` then a char that is not whitespace, a digit or `$`. The next `$` closes it, and must be
/// preceded by a non-space and not followed by an ASCII digit.
fn inline_formula(c: &[char], i: usize, le: usize) -> Option<usize> {
    if i + 2 >= le || c[i] != '$' {
        return None;
    }
    let first = c[i + 1];
    if first.is_whitespace() || first.is_ascii_digit() || first == '$' {
        return None;
    }
    let j = (i + 2..le).find(|&j| c[j] == '$')?;
    let closes = !(c[j - 1].is_whitespace() || (j + 1 < le && c[j + 1].is_ascii_digit()));
    closes.then_some(j + 1)
}

/// A run of emoji chars that has at least one char other than a joiner.
fn emoji(c: &[char], i: usize, le: usize) -> Option<usize> {
    if !is_emoji_char(c[i]) {
        return None;
    }
    let end = (i..le).find(|&j| !is_emoji_char(c[j])).unwrap_or(le);
    c[i..end].iter().any(|&ch| !is_joiner(ch)).then_some(end)
}

fn is_emoji_char(ch: char) -> bool {
    matches!(ch as u32, 0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0x2B50 | 0x2B55) || is_joiner(ch)
}

fn is_joiner(ch: char) -> bool {
    matches!(ch, '\u{200D}' | '\u{FE0F}' | '\u{20E3}')
}

fn is_han(ch: char) -> bool {
    matches!(
        ch,
        '\u{4e00}'..='\u{9fff}'
            | '\u{3400}'..='\u{4dbf}'
            | '\u{20000}'..='\u{2ceaf}'
            | '\u{f900}'..='\u{faff}'
            | '\u{3005}'
            | '\u{3007}'
    )
}

fn is_kana(ch: char) -> bool {
    matches!(
        ch,
        '\u{3041}'..='\u{3096}'
            | '\u{309d}'..='\u{309f}'
            | '\u{30a1}'..='\u{30fa}'
            | '\u{30fc}'..='\u{30ff}'
            | '\u{31f0}'..='\u{31ff}'
            | '\u{ff66}'..='\u{ff9f}'
    )
}

fn is_hangul_syllable(ch: char) -> bool {
    ('\u{ac00}'..='\u{d7a3}').contains(&ch)
}

/// Start and end index of each line; the newline itself is excluded.
fn lines(c: &[char]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, &ch) in c.iter().enumerate() {
        if ch == '\n' {
            out.push((start, i));
            start = i + 1;
        }
    }
    out.push((start, c.len()));
    out
}

fn first_non_ws(c: &[char], from: usize, to: usize) -> Option<usize> {
    (from..to).find(|&j| !c[j].is_whitespace())
}

/// Index after the last non-whitespace char in `c[from..to]`, or `from` if there is none.
fn trimmed_end(c: &[char], from: usize, to: usize) -> usize {
    (from..to)
        .rev()
        .find(|&j| !c[j].is_whitespace())
        .map_or(from, |j| j + 1)
}

fn run_len(c: &[char], from: usize, to: usize, ch: char) -> usize {
    (from..to).take_while(|&j| c[j] == ch).count()
}

fn has_text(chars: &[char]) -> bool {
    chars.iter().any(|ch| !ch.is_whitespace())
}

fn trim(chars: &[char]) -> &[char] {
    let start = chars
        .iter()
        .position(|ch| !ch.is_whitespace())
        .unwrap_or(chars.len());
    let end = chars
        .iter()
        .rposition(|ch| !ch.is_whitespace())
        .map_or(start, |j| j + 1);
    &chars[start..end]
}
