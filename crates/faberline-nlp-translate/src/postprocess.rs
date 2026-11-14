//! Stage 5: Target-language morphological and orthographic post-processing:
//! - Korean Hangul Batchim (종성) Josa postposition harmony (`은/는`, `이/가`, `을/를`, `과/와`,
//!   `으로/로`, `이랑/랑`, `이나/나`)
//! - English indefinite article (`a` / `an`) and sentence-initial capitalization repair
//! - Target-variant quotation mark localization & optional CJK–Latin spacing
//!   (strictly skipping locked `VerbatimPassthrough` spans).

use faberline_nlp::Span;

use crate::schema::TargetVariant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchimState {
    None,
    Rieul,
    OtherConsonant,
}

fn char_batchim(ch: char) -> Option<BatchimState> {
    let code = ch as u32;
    if (0xAC00..=0xD7A3).contains(&code) {
        let jong = (code - 0xAC00) % 28;
        return Some(match jong {
            0 => BatchimState::None,
            8 => BatchimState::Rieul,
            _ => BatchimState::OtherConsonant,
        });
    }
    // Sino-Korean digit pronunciation: 0(영), 1(일-ㄹ), 2(이), 3(삼), 4(사), 5(오), 6(육), 7(칠-ㄹ),
    // 8(팔-ㄹ), 9(구).
    match ch {
        '2' | '4' | '5' | '9' | '２' | '４' | '５' | '９' => Some(BatchimState::None),
        '1' | '7' | '8' | '１' | '７' | '８' => Some(BatchimState::Rieul),
        '0' | '3' | '6' | '０' | '３' | '６' => Some(BatchimState::OtherConsonant),
        _ => None,
    }
}

fn is_ko_boundary_after_particle(next_char: Option<char>) -> bool {
    match next_char {
        None => true,
        Some(c) => {
            c.is_whitespace()
                || matches!(
                    c,
                    '.' | ','
                        | '!'
                        | '?'
                        | ':'
                        | ';'
                        | '。'
                        | '，'
                        | '！'
                        | '？'
                        | ')'
                        | '）'
                        | ']'
                        | '”'
                        | '"'
                        | '\''
                        | '’'
                        | '」'
                        | '』'
                )
        }
    }
}

/// Repair Korean case/topic postpositions (`은/는`, `이/가`, `을/를`, `과/와`, `으로/로`, `이랑/랑`,
/// `이나/나`) immediately following any Hangul syllable or numeral (especially after glossary/entity
/// placeholder restoration), while preserving `protected_spans` verbatim and updating `tracked_spans`.
pub fn fix_korean_particles(
    text: &str,
    protected_spans: &[Span],
    tracked_spans: &mut [Span],
) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 2 {
        return text.to_owned();
    }

    let is_protected = |idx: usize| -> bool {
        protected_spans
            .iter()
            .any(|s| idx >= s.start_char && idx < s.end_char)
    };

    let mut out = String::with_capacity(text.len() + 8);
    let mut shifts: Vec<(usize, isize)> = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        if i > 0 && !is_protected(i) {
            let prev = chars[i - 1];
            if let Some(b_state) = char_batchim(prev) {
                let has_b = b_state != BatchimState::None;
                // 2-char particles (`으로`, `이랑`, `이나`)
                if i + 1 < chars.len()
                    && !is_protected(i + 1)
                    && is_ko_boundary_after_particle(chars.get(i + 2).copied())
                {
                    let pair = (chars[i], chars[i + 1]);
                    match pair {
                        ('으', '로') => {
                            if b_state == BatchimState::None || b_state == BatchimState::Rieul {
                                out.push('로');
                                shifts.push((i + 2, -1));
                                i += 2;
                                continue;
                            }
                        }
                        ('이', '랑') if !has_b => {
                            out.push('랑');
                            shifts.push((i + 2, -1));
                            i += 2;
                            continue;
                        }
                        ('이', '나') if !has_b => {
                            out.push('나');
                            shifts.push((i + 2, -1));
                            i += 2;
                            continue;
                        }
                        _ => {}
                    }
                }

                // 1-char particles (`은/는`, `이/가`, `을/를`, `과/와`, `로->으로`, `랑->이랑`, `나->이나`)
                // Only rewrite when `i` is NOT inside a protected glossary/opaque span and
                // stands at the end of the eojeol!
                if is_ko_boundary_after_particle(chars.get(i + 1).copied()) {
                    match chars[i] {
                        '은' | '는' => {
                            out.push(if has_b { '은' } else { '는' });
                            i += 1;
                            continue;
                        }
                        '이' | '가' => {
                            out.push(if has_b { '이' } else { '가' });
                            i += 1;
                            continue;
                        }
                        '을' | '를' => {
                            out.push(if has_b { '을' } else { '를' });
                            i += 1;
                            continue;
                        }
                        '과' | '와' => {
                            out.push(if has_b { '과' } else { '와' });
                            i += 1;
                            continue;
                        }
                        '로' if b_state == BatchimState::OtherConsonant => {
                            out.push_str("으로");
                            shifts.push((i + 1, 1));
                            i += 1;
                            continue;
                        }
                        '랑' if has_b => {
                            out.push_str("이랑");
                            shifts.push((i + 1, 1));
                            i += 1;
                            continue;
                        }
                        _ => {}
                    }
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }

    if !shifts.is_empty() {
        apply_span_shifts(tracked_spans, &shifts);
    }
    out
}

/// Repair English indefinite articles (`a` <-> `an`) before restored entities/nouns and ensure
/// sentence-initial capitalization when not starting with a protected inline code/URL span.
pub fn fix_english_articles_and_case(
    text: &str,
    protected_spans: &[Span],
    tracked_spans: &mut [Span],
    capitalize_initial: bool,
) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return String::new();
    }

    let is_protected = |idx: usize| -> bool {
        protected_spans
            .iter()
            .any(|s| idx >= s.start_char && idx < s.end_char)
    };

    let mut out = String::with_capacity(text.len() + 4);
    let mut shifts: Vec<(usize, isize)> = Vec::new();
    let mut capitalized = !capitalize_initial;
    let mut i = 0usize;

    while i < chars.len() {
        // Check standalone word `a` / `A` / `an` / `An` followed by whitespace + word/number.
        let at_word_start = i == 0 || !chars[i - 1].is_ascii_alphanumeric();
        if at_word_start && !is_protected(i) {
            let (art_len, is_upper) =
                if matches!(chars[i], 'a' | 'A') && chars.get(i + 1).is_some_and(|c| *c == ' ') {
                    (1usize, chars[i] == 'A')
                } else if matches!(chars[i], 'a' | 'A')
                    && chars.get(i + 1) == Some(&'n')
                    && chars.get(i + 2) == Some(&' ')
                {
                    (2usize, chars[i] == 'A')
                } else {
                    (0, false)
                };

            if art_len > 0 {
                let mut w_start = i + art_len + 1;
                while w_start < chars.len() && chars[w_start] == ' ' {
                    w_start += 1;
                }
                let mut w_end = w_start;
                while w_end < chars.len()
                    && (chars[w_end].is_ascii_alphanumeric() || chars[w_end] == '-')
                {
                    w_end += 1;
                }
                if w_end > w_start {
                    let next_word: String = chars[w_start..w_end].iter().collect();
                    let want_an = english_takes_an(&next_word);
                    let make_upper = is_upper || (!capitalized && i == 0);
                    if !capitalized && i == 0 {
                        capitalized = true;
                    }
                    let rendered = match (want_an, make_upper) {
                        (true, true) => "An",
                        (true, false) => "an",
                        (false, true) => "A",
                        (false, false) => "a",
                    };
                    out.push_str(rendered);
                    let delta = rendered.len() as isize - art_len as isize;
                    if delta != 0 {
                        shifts.push((i + art_len, delta));
                    }
                    i += art_len;
                    continue;
                }
            }
        }

        let ch = chars[i];
        if !capitalized {
            if is_protected(i) {
                capitalized = true;
                out.push(ch);
            } else if ch.is_ascii_alphabetic() {
                capitalized = true;
                out.push(ch.to_ascii_uppercase());
            } else {
                out.push(ch);
            }
        } else {
            out.push(ch);
        }
        i += 1;
    }

    if !shifts.is_empty() {
        apply_span_shifts(tracked_spans, &shifts);
    }
    out
}

fn english_takes_an(word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    // Numeric pronunciation (`8`, `8-core`, `8th`, `11`, `18`, `80`, `800`)
    if word.as_bytes()[0].is_ascii_digit() {
        let digits: String = word.chars().take_while(|c| c.is_ascii_digit()).collect();
        return digits.starts_with('8') || digits == "11" || digits == "18";
    }
    // All-caps initialism pronounced with vowel-initial letter name (`F`, `H`, `L`, `M`, `N`, `R`, `S`, `X`)
    if word.len() >= 2
        && word.bytes().all(|b| b.is_ascii_uppercase())
        && !matches!(word, "US" | "UK" | "UN" | "EU")
    {
        return matches!(
            word.as_bytes()[0],
            b'A' | b'E' | b'F' | b'H' | b'I' | b'L' | b'M' | b'N' | b'O' | b'R' | b'S' | b'X'
        );
    }

    let lower = word.to_ascii_lowercase();
    // Silent-h words
    for sh in ["hour", "honest", "honor", "honour", "heir"] {
        if lower.starts_with(sh) {
            return true;
        }
    }
    // Consonant-glide `/juː/` or `/w/` words (`user`, `university`, `unit`, `uniform`, `unique`,
    // `utility`, `ubiquitous` stays vowel! `european`, `one-way`, `once`)
    for cons_glide in [
        "uni", "use", "usu", "util", "urin", "uro", "eup", "eur", "one-",
    ] {
        if lower.starts_with(cons_glide)
            && !lower.starts_with("unid")
            && !lower.starts_with("unim")
            && !lower.starts_with("unin")
            && !lower.starts_with("unir")
        {
            return false;
        }
    }
    if lower == "one" || lower == "once" {
        return false;
    }
    matches!(lower.as_bytes()[0], b'a' | b'e' | b'i' | b'o' | b'u')
}

/// Localize paired quotation marks to `target`'s typographic convention while leaving
/// `protected_spans` (`code`, `formula`, `url`) untouched (1-to-1 scalar replacement, so spans
/// never shift!).
pub fn localize_quotation_marks(
    text: &str,
    target: TargetVariant,
    protected_spans: &[Span],
) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return String::new();
    }
    let is_protected = |idx: usize| -> bool {
        protected_spans
            .iter()
            .any(|s| idx >= s.start_char && idx < s.end_char)
    };

    let (outer_open, outer_close, inner_open, inner_close) = match target {
        TargetVariant::ZhHant | TargetVariant::Ja => ('「', '」', '『', '』'),
        TargetVariant::ZhHans | TargetVariant::EnUs | TargetVariant::Ko => ('“', '”', '‘', '’'),
        TargetVariant::EnGb => ('‘', '’', '“', '”'),
    };

    let mut out = String::with_capacity(text.len() + 8);
    let mut depth = 0usize;
    let mut single_open = 0usize;

    for (i, &ch) in chars.iter().enumerate() {
        if is_protected(i) {
            out.push(ch);
            continue;
        }
        match ch {
            '「' | '“' => {
                out.push(if depth == 0 { outer_open } else { inner_open });
                depth += 1;
            }
            '」' | '”' => {
                depth = depth.saturating_sub(1);
                out.push(if depth == 0 { outer_close } else { inner_close });
            }
            '『' => {
                out.push(inner_open);
                depth += 1;
            }
            '』' => {
                depth = depth.saturating_sub(1);
                out.push(inner_close);
            }
            '‘' => {
                out.push(if depth == 0 { outer_open } else { inner_open });
                depth += 1;
                single_open += 1;
            }
            '’' => {
                let is_contraction = i > 0
                    && i + 1 < chars.len()
                    && chars[i - 1].is_ascii_alphabetic()
                    && chars[i + 1].is_ascii_alphabetic();
                if single_open > 0 && !is_contraction {
                    single_open -= 1;
                    depth = depth.saturating_sub(1);
                    out.push(if depth == 0 { outer_close } else { inner_close });
                } else {
                    out.push(ch);
                }
            }
            _ => out.push(ch),
        }
    }
    out
}

/// Insert single ASCII spaces at CJK–ASCII alphanumeric boundaries outside `protected_spans`,
/// shifting `tracked_spans` accordingly.
pub fn apply_cjk_latin_spacing(
    text: &str,
    protected_spans: &[Span],
    tracked_spans: &mut [Span],
) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 2 {
        return text.to_owned();
    }
    let is_protected = |idx: usize| -> bool {
        protected_spans
            .iter()
            .any(|s| idx >= s.start_char && idx < s.end_char)
    };

    let mut out = String::with_capacity(text.len() + 8);
    let mut shifts: Vec<(usize, isize)> = Vec::new();

    for i in 0..chars.len() {
        out.push(chars[i]);
        if i + 1 < chars.len() && !(is_protected(i) && is_protected(i + 1)) {
            let a = chars[i];
            let b = chars[i + 1];
            if (is_cjk_letter(a) && b.is_ascii_alphanumeric())
                || (a.is_ascii_alphanumeric() && is_cjk_letter(b))
            {
                out.push(' ');
                shifts.push((i + 1, 1));
            }
        }
    }

    if !shifts.is_empty() {
        apply_span_shifts(tracked_spans, &shifts);
    }
    out
}

/// Clean up stray spaces between CJK characters and CJK punctuation outside `protected_spans`,
/// collapsing redundant ASCII spaces while preserving spaces around Latin/opaque spans and
/// updating `tracked_spans`.
pub fn cleanup_cjk_translation_spacing(
    text: &str,
    protected_spans: &[Span],
    tracked_spans: &mut [Span],
) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 2 {
        return text.to_owned();
    }
    let is_protected = |idx: usize| -> bool {
        protected_spans
            .iter()
            .any(|s| idx >= s.start_char && idx < s.end_char)
    };
    let is_cjk_or_punct = |c: char| -> bool {
        is_cjk_letter(c)
            || matches!(
                c,
                '，' | '。'
                    | '！'
                    | '？'
                    | '、'
                    | '「'
                    | '」'
                    | '『'
                    | '』'
                    | '“'
                    | '”'
                    | '‘'
                    | '’'
                    | '（'
                    | '）'
                    | '：'
                    | '；'
                    | '…'
                    | '—'
            )
    };

    let mut out = String::with_capacity(text.len());
    let mut shifts: Vec<(usize, isize)> = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == ' ' && !is_protected(i) {
            let prev = out.chars().next_back();
            let next = chars.get(i + 1).copied();
            let drop_space = match (prev, next) {
                (None, _) | (_, None) => true,
                (Some(' '), _) => true,
                (Some(p), Some(n)) if is_cjk_or_punct(p) && is_cjk_or_punct(n) => true,
                _ => false,
            };
            if drop_space {
                shifts.push((i + 1, -1));
                i += 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }

    if !shifts.is_empty() {
        apply_span_shifts(tracked_spans, &shifts);
    }
    out
}

fn is_cjk_letter(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x3040..=0x30FF
            | 0xAC00..=0xD7A3
            | 0x20000..=0x2CEAF
    )
}

fn apply_span_shifts(spans: &mut [Span], shifts: &[(usize, isize)]) {
    let map_pos = |pos: usize| -> usize {
        let mut delta = 0isize;
        for &(at, d) in shifts {
            if at <= pos {
                delta += d;
            }
        }
        (pos as isize + delta).max(0) as usize
    };
    for sp in spans.iter_mut() {
        sp.start_char = map_pos(sp.start_char);
        sp.end_char = map_pos(sp.end_char).max(sp.start_char);
    }
}
