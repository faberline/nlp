//! Chinese (`zh-Hant` + `zh-Hans`) first-party Viterbi segmentation and lexical analysis.

use std::collections::HashSet;

use super::format::LangGroup;
use super::{LexTokenAnalysis, db, lookup_zh};
use crate::Span;

const EXTRA_LEXICON_COST: u32 = 100;
const SINGLE_HAN_OOV_COST: u32 = 1600;
const MAX_WORD_CHARS: usize = 16;

#[inline]
fn is_ascii_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Segment a Chinese character slice using minimum-cost Viterbi DP over `zh_hant` + `zh_hans`
/// plus caller-supplied `extra_lexicon` words (matched at low cost `100`).
///
/// Returned `Span`s are in document Unicode scalar coordinates (`base_char + local_idx`).
pub fn segment_zh(chars: &[char], base_char: usize, extra_lexicon: &[String]) -> Vec<Span> {
    let n = chars.len();
    if n == 0 {
        return Vec::new();
    }

    // Build UTF-8 string and char-to-byte offset table once for zero-allocation prefix slicing.
    let mut text = String::with_capacity(n * 3);
    let mut byte_offs = Vec::with_capacity(n + 1);
    for &c in chars {
        byte_offs.push(text.len());
        text.push(c);
    }
    byte_offs.push(text.len());

    let extra_set: HashSet<&str> = extra_lexicon
        .iter()
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .collect();
    let max_extra_chars = extra_set
        .iter()
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(0)
        .min(32);

    let db = db();
    let mut dp_cost = vec![u32::MAX; n + 1];
    let mut back_len = vec![1usize; n + 1];
    dp_cost[0] = 0;

    let mut i = 0usize;
    while i < n {
        if dp_cost[i] == u32::MAX {
            i += 1;
            continue;
        }
        let base_cost = dp_cost[i];
        let ch = chars[i];

        // Skip whitespace cleanly without emitting a token.
        if ch.is_whitespace() {
            if base_cost < dp_cost[i + 1] {
                dp_cost[i + 1] = base_cost;
                back_len[i + 1] = 1;
            }
            i += 1;
            continue;
        }

        // Group contiguous ASCII alphanumeric runs cleanly.
        if is_ascii_word_char(ch) {
            let mut j = i + 1;
            while j < n && is_ascii_word_char(chars[j]) {
                j += 1;
            }
            let cand_cost = base_cost + 200;
            if cand_cost < dp_cost[j] {
                dp_cost[j] = cand_cost;
                back_len[j] = j - i;
            }
            i = j;
            continue;
        }

        // Default single-character fallback.
        let oov_cost = base_cost + SINGLE_HAN_OOV_COST;
        if oov_cost < dp_cost[i + 1] {
            dp_cost[i + 1] = oov_cost;
            back_len[i + 1] = 1;
        }

        // Check caller extra_lexicon matches starting at i.
        if max_extra_chars > 0 {
            let limit = (i + max_extra_chars).min(n);
            for end in (i + 1)..=limit {
                let slice = &text[byte_offs[i]..byte_offs[end]];
                if extra_set.contains(slice) {
                    let cand_cost = base_cost + EXTRA_LEXICON_COST;
                    if cand_cost < dp_cost[end] {
                        dp_cost[end] = cand_cost;
                        back_len[end] = end - i;
                    }
                }
            }
        }

        // Common-prefix search in the compiled Chinese lexicon.
        let rest = &text[byte_offs[i]..];
        let max_c = MAX_WORD_CHARS.min(n - i);
        db.common_prefix_search(
            LangGroup::Zh,
            rest,
            max_c,
            |_byte_len, char_len, entries| {
                let min_cost = entries.iter().map(|e| e.cost as u32).min().unwrap_or(1500);
                let end = i + char_len;
                let cand_cost = base_cost + min_cost;
                if cand_cost < dp_cost[end] {
                    dp_cost[end] = cand_cost;
                    back_len[end] = char_len;
                }
            },
        );

        i += 1;
    }

    // Backtrack from n to 0.
    let mut spans = Vec::new();
    let mut pos = n;
    while pos > 0 {
        let len = back_len[pos].max(1);
        let start = pos - len;
        if !chars[start..pos].iter().all(|c| c.is_whitespace()) {
            spans.push(Span {
                start_char: base_char + start,
                end_char: base_char + pos,
            });
        }
        pos = start;
    }
    spans.reverse();
    spans
}

/// Analyze a single Chinese token for UPOS, cross-strait variant counterpart, and semantic class.
pub fn analyze_zh_token(surface: &str) -> LexTokenAnalysis {
    let Some(hit) = lookup_zh(surface) else {
        return LexTokenAnalysis::default();
    };
    LexTokenAnalysis {
        decided_upos: hit.decided_upos,
        upos_mask: hit.upos_mask,
        upos_candidates: hit.upos_candidates,
        lemma: hit.lemma,
        canonical_variant: hit.counterpart,
        sem_class: hit.sem_class,
        morphemes: Vec::new(),
    }
}
