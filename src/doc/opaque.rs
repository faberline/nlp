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

/// Blank the block spans, and replace every other opaque char with `x`. An emoji run that ends a
/// sentence keeps a `!` as its last char, so the sentence still ends there.
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
    markdown_heading(t) || chinese_heading(t) || english_heading(t)
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
        [sep, title @ ..]
            if (sep.is_whitespace() || matches!(*sep, '：' | ':' | '、')) && has_text(title) =>
        {
            true
        }
        _ => allow_tight_title && has_text(after),
    }
}

fn bare_numeral_zhang_heading(s: &[char]) -> bool {
    let numerals = zh_numeral_len(s);
    numerals >= 2
        && s.get(numerals) == Some(&'章')
        && s.get(numerals + 1).is_some_and(|sep| sep.is_whitespace())
        && has_text(&s[numerals + 2..])
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
    chinese_di_heading(rest, true)
        || volume_then_chapter(rest)
        || bare_numeral_zhang_heading(rest)
        || missing_unit_di_heading(rest)
}

/// `第` then a numeral, then one of 章節回卷篇部 (plus web-novel `VIP第...卷...`,
/// `正文第...卷...第...章...`, `<Prefix>\u{00a0}<Title>`, `<Prefix> 第...章`, `第...卷...第...章`,
/// `<numerals>章 <Title>`, and `第<numerals>=2> <short_title>`). At most 60 chars with no
/// sentence end.
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
    if bare_numeral_zhang_heading(t) || missing_unit_di_heading(t) || volume_then_chapter(t) {
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
    for idx in 1..=t.len().saturating_sub(1).min(24) {
        if t[idx] == '第' && t[idx - 1].is_whitespace() {
            let allow_tight = t[..idx].contains(&'卷');
            if chinese_di_heading(&t[idx..], allow_tight)
                || volume_then_chapter(&t[idx..])
                || missing_unit_di_heading(&t[idx..])
            {
                return true;
            }
        }
    }
    false
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
    const BULLETS: &[char] = &['-', '*', '+', '•', '・'];
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
                    editorial_inline_at(c, i, le)
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
    "推薦票",
    "推荐票",
    "第一更",
    "第二更",
    "第三更",
    "第四更",
    "第五更",
    "加更",
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
    "求下票",
];

fn contains_substr(chars: &[char], pat: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    !p.is_empty() && chars.windows(p.len()).any(|w| w == p.as_slice())
}

fn contains_any_editorial_cue(chars: &[char], include_baodi: bool) -> bool {
    EDITORIAL_CUES.iter().any(|cue| contains_substr(chars, cue))
        || (include_baodi && contains_substr(chars, "保底"))
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

fn strip_inline_parens(t: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(t.len());
    let mut i = 0;
    while i < t.len() {
        if matches!(t[i], '(' | '（') {
            let mut j = i + 1;
            while j < t.len() && !matches!(t[j], '(' | '（' | ')' | '）') {
                j += 1;
            }
            if j < t.len() && matches!(t[j], ')' | '）') {
                i = j + 1;
                continue;
            }
        }
        out.push(t[i]);
        i += 1;
    }
    out
}

fn is_editorial_block_line(t: &[char]) -> bool {
    if multi_paren_tag_list_line(t) {
        return true;
    }
    if matches!(t.first(), Some('(' | '（' | '【' | '[')) && contains_any_editorial_cue(t, true) {
        let stripped_end = t
            .iter()
            .rposition(|ch| !matches!(*ch, '！' | '!' | '。' | '.' | '~' | '～'))
            .map_or(0, |idx| idx + 1);
        let closed = stripped_end > 1 && matches!(t[stripped_end - 1], ')' | '）' | '】' | ']');
        let unclosed = !t[1..]
            .iter()
            .any(|ch| matches!(*ch, ')' | '）' | '】' | ']'));
        if closed || unclosed {
            return true;
        }
    }
    if t.len() > 300 || t.iter().any(|ch| "「」『』“”\"'".contains(*ch)) {
        return false;
    }
    let outside = strip_inline_parens(t);
    let periods = outside.iter().filter(|&&ch| ch == '。').count();
    if contains_any_editorial_cue(&outside, false) {
        return periods <= 2;
    }
    contains_substr(&outside, "保底") && t.len() <= 80 && periods == 0
}

/// Scan one line for inline spans. At each position the kinds are tried in order.
fn inline(c: &[char], ls: usize, le: usize, mask_editorial: bool, out: &mut Vec<Found>) {
    let mut i = ls;
    while i < le {
        match inline_at(c, i, ls, le, mask_editorial) {
            Some((kind, end)) => {
                out.push(Found {
                    item: Opaque {
                        kind,
                        span: Span {
                            start_char: i,
                            end_char: end,
                        },
                    },
                    block: false,
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
) -> Option<(OpaqueKind, usize)> {
    if c[i] == '`' {
        return code_span(c, i, le).map(|end| (OpaqueKind::Code, end));
    }
    url(c, i, le)
        .map(|end| (OpaqueKind::Url, end))
        .or_else(|| raw_html_tag(c, i, ls, le).map(|end| (OpaqueKind::Code, end)))
        .or_else(|| html_entity(c, i, le).map(|end| (OpaqueKind::Code, end)))
        .or_else(|| pua_run(c, i, le).map(|end| (OpaqueKind::Code, end)))
        .or_else(|| {
            if mask_editorial {
                editorial_inline_at(c, i, le).map(|end| (OpaqueKind::Code, end))
            } else {
                None
            }
        })
        .or_else(|| citation(c, i, le).map(|end| (OpaqueKind::Citation, end)))
        .or_else(|| inline_formula(c, i, le).map(|end| (OpaqueKind::Formula, end)))
        .or_else(|| emoji(c, i, le).map(|end| (OpaqueKind::Emoji, end)))
}

/// Raw/unclosed HTML tags (`<P`, `<p>`, `</p>`, `<br>`, `<br/>`, `<br />` case-insensitive).
fn raw_html_tag(c: &[char], i: usize, ls: usize, le: usize) -> Option<usize> {
    if c[i] != '<' {
        return None;
    }
    for closed in ["</p>", "<p>", "<br/>", "<br />", "<br>"] {
        if starts_with_ci(c, i, le, closed) {
            return Some(i + closed.len());
        }
    }
    // Standalone or line-start unclosed `<P` / `<p` followed by whitespace, line end, or Han.
    if starts_with_ci(c, i, le, "<p") {
        let at_start_or_ws = i == ls || c[i - 1].is_whitespace();
        let end = i + 2;
        let valid_after = end == le || c[end].is_whitespace() || is_han(c[end]);
        if at_start_or_ws && valid_after {
            return Some(end);
        }
    }
    None
}

/// Corrupted or raw HTML entities (`&qut;`, `&quot;`, `&nbsp;`, `&amp;`, `&lt;`, `&gt;`).
fn html_entity(c: &[char], i: usize, le: usize) -> Option<usize> {
    if c[i] != '&' {
        return None;
    }
    for ent in ["&qut;", "&quot;", "&nbsp;", "&amp;", "&lt;", "&gt;"] {
        if starts_with_ci(c, i, le, ent) {
            return Some(i + ent.len());
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

/// Opt-in inline scraper/editorial tags (`/首/發`, `(..#..#..)`, and short inline `(待續)` notes).
fn editorial_inline_at(c: &[char], i: usize, le: usize) -> Option<usize> {
    if i + 4 <= le
        && c[i] == '/'
        && c[i + 1] == '首'
        && c[i + 2] == '/'
        && matches!(c[i + 3], '發' | '发')
    {
        let end = if i + 5 <= le && c[i + 4] == '/' {
            i + 5
        } else {
            i + 4
        };
        return Some(end);
    }
    if matches!(c[i], '(' | '（') {
        let mut j = i + 1;
        while j < le.min(i + 62) && !matches!(c[j], '(' | '（' | ')' | '）') {
            j += 1;
        }
        if j < le && matches!(c[j], ')' | '）') {
            let inner = &c[i + 1..j];
            let hashes = inner.iter().filter(|&&ch| ch == '#').count();
            if !inner.is_empty()
                && inner.len() <= 60
                && (hashes >= 2 || contains_any_editorial_cue(inner, true))
            {
                return Some(j + 1);
            }
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

/// Uppercase ASCII or Han first, at most 60 chars, and a year at the end.
fn is_author_year_part(part: &[char]) -> bool {
    let part = trim(part);
    part.len() <= 60
        && part
            .first()
            .is_some_and(|&ch| ch.is_ascii_uppercase() || is_han(ch))
        && ends_with_year(part)
}

/// A year from 1500 to 2099, optionally followed by one lowercase letter, and preceded by a
/// space, `,` or `，`.
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
        && matches!(sep, ' ' | ',' | '，')
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
            | '\u{20000}'..='\u{2a6df}'
            | '\u{f900}'..='\u{faff}'
    )
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
