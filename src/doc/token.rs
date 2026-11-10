//! Layer 2 tokenization of `nlp.doc.v1`. Tokens stay inside one sentence, and an opaque span
//! inside a sentence is always exactly one token. See `docs/doc-v1.md`.

use std::sync::OnceLock;

use aho_corasick::{AhoCorasick, MatchKind};
use unicode_segmentation::UnicodeSegmentation;

use super::{OpaqueKind, segment};
use crate::Span;
use crate::text::{TAIWAN_TERMS, default_jieba};

/// One token before tagging. `opaque` is set when the token is an opaque span.
pub(super) struct Piece {
    pub span: Span,
    pub opaque: Option<OpaqueKind>,
}

const CLITICS: &[&str] = &["'s", "'ll", "'re", "'ve", "'m", "'d"];
const REPEATABLE: &[char] = &['.', '…', '-', '—', '~', '!', '?', '！', '？'];

fn terms() -> &'static AhoCorasick {
    static TERMS: OnceLock<AhoCorasick> = OnceLock::new();
    TERMS.get_or_init(|| {
        AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(TAIWAN_TERMS)
            .expect("the Taiwan term list builds")
    })
}

/// Split one sentence into tokens. `inline` holds the opaque spans of the text; each one
/// inside the sentence becomes a single piece, and the text between them is split by rule.
pub(super) fn tokenize(c: &[char], sentence: Span, inline: &[(Span, OpaqueKind)]) -> Vec<Piece> {
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
        text_pieces(c, cursor, span.start_char, &mut out);
        out.push(Piece {
            span,
            opaque: Some(kind),
        });
        cursor = span.end_char;
    }
    text_pieces(c, cursor, sentence.end_char, &mut out);
    out
}

fn text_pieces(c: &[char], start: usize, end: usize, out: &mut Vec<Piece>) {
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
            han_tokens(&s, a, b, &mut ranges);
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
        let han = segment::is_han(ch);
        let end = b + ch.len_utf8();
        match out.last_mut() {
            Some(last) if last.2 == han => last.1 = end,
            _ => out.push((b, end, han)),
        }
    }
    out
}

/// Han text: Taiwan terms first, then jieba on the stretches between them.
fn han_tokens(s: &str, a: usize, b: usize, out: &mut Vec<(usize, usize)>) {
    let run = &s[a..b];
    let mut at = 0;
    for m in terms().find_iter(run) {
        han_words(&run[at..m.start()], a + at, out);
        out.push((a + m.start(), a + m.end()));
        at = m.end();
    }
    han_words(&run[at..], a + at, out);
}

fn han_words(text: &str, base: usize, out: &mut Vec<(usize, usize)>) {
    if text.is_empty() {
        return;
    }
    for word in default_jieba().cut(text, true) {
        let start = base + word.byte_start;
        out.push((start, start + word.word.len()));
    }
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
        if text == "-"
            && is_word(prev_text)
            && let Some(&(na, nb)) = raw.get(i).filter(|&&(na, _)| na == b)
            && is_word(&s[na..nb])
        {
            if let Some(last) = merged.last_mut() {
                last.1 = nb;
            }
            i += 1;
            continue;
        }
        if text == "."
            && (segment::is_abbreviation(prev_text) || prev_text.eq_ignore_ascii_case("etc"))
        {
            if let Some(last) = merged.last_mut() {
                last.1 = b;
            }
            continue;
        }
        let mut chars = text.chars();
        if let (Some(ch), None) = (chars.next(), chars.next())
            && REPEATABLE.contains(&ch)
            && prev_text == text
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

fn is_word(text: &str) -> bool {
    text.chars().any(char::is_alphanumeric)
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
    if let Some(p) = apostrophe.filter(|&p| p > 1) {
        let suffix = word[p..].to_lowercase().replace('’', "'");
        let before = &word[..p];
        if suffix == "'t" && before.ends_with(['n', 'N']) {
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
