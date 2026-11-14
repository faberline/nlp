//! Japanese (`ja`) Viterbi minimum-cost segmentation, inflectional stem+tail matching
//! via `ja_inflect.tsv`, and UD-aligned morpheme splitting.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::format::{LangGroup, NONE_U32};
use super::{
    JaInflectRule, SemanticClass, code_to_upos, db, decided_from_mask, lookup_ja,
    mask_to_candidates,
};
use crate::Span;
use crate::doc::Upos;

const MAX_WORD_CHARS: usize = 16;
const MAX_STEM_CHARS: usize = 8;
const SINGLE_OOV_COST: u32 = 1800;
const KATAKANA_OOV_BASE_COST: u32 = 900;
const KATAKANA_OOV_PER_CHAR_COST: u32 = 80;

/// One segmented Japanese token (with conjugated verbs/adjectives already split into
/// UD-aligned morpheme tokens such as `買っ` + `た`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JaSegment {
    pub span: Span,
    pub decided_upos: Option<Upos>,
    pub upos_mask: u32,
    pub upos_candidates: Vec<Upos>,
    pub lemma: Option<String>,
    pub sem_class: Option<SemanticClass>,
}

#[derive(Clone)]
struct InflectEntry {
    rule: JaInflectRule,
    paradigm: &'static str,
    dict_ending: &'static str,
    tail_chars: usize,
}

struct JaInflectIndex {
    by_first_char: HashMap<char, Vec<InflectEntry>>,
}

fn paradigm_dict_ending(paradigm: &str) -> Option<&'static str> {
    match paradigm {
        "godan_ka" | "godan_iku" => Some("く"),
        "godan_ga" => Some("ぐ"),
        "godan_sa" => Some("す"),
        "godan_ta" => Some("つ"),
        "godan_na" => Some("ぬ"),
        "godan_ba" => Some("ぶ"),
        "godan_ma" => Some("む"),
        "godan_ra" => Some("る"),
        "godan_wa" => Some("う"),
        "ichidan" => Some("る"),
        "adj_i" => Some("い"),
        "sahen" => Some("する"),
        "kahen" => Some("来る"),
        _ => None,
    }
}

fn inflect_index() -> &'static JaInflectIndex {
    static IDX: OnceLock<JaInflectIndex> = OnceLock::new();
    IDX.get_or_init(|| {
        let db = db();
        let mut by_first_char: HashMap<char, Vec<InflectEntry>> = HashMap::new();
        for rule in db.ja_inflect_rules() {
            let Some(paradigm) = db.infl_str(rule.infl_id) else {
                continue;
            };
            let Some(dict_ending) = paradigm_dict_ending(paradigm) else {
                continue;
            };
            let Some(first_ch) = rule.conjugated_tail.chars().next() else {
                continue;
            };
            let tail_chars = rule.conjugated_tail.chars().count();
            by_first_char
                .entry(first_ch)
                .or_default()
                .push(InflectEntry {
                    rule,
                    paradigm,
                    dict_ending,
                    tail_chars,
                });
        }
        // Sort longest tail first for clean matching.
        for list in by_first_char.values_mut() {
            list.sort_by_key(|b| std::cmp::Reverse(b.tail_chars));
        }
        JaInflectIndex { by_first_char }
    })
}

#[inline]
fn is_katakana_run_char(c: char) -> bool {
    matches!(c, '\u{30A1}'..='\u{30FA}' | '\u{30FC}' | '\u{30FD}'..='\u{30FE}')
}

#[inline]
fn is_ascii_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

#[inline]
fn is_ja_stem_char(c: char) -> bool {
    matches!(
        c,
        '\u{4E00}'..='\u{9FFF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{3041}'..='\u{309F}'
            | '\u{30A1}'..='\u{30FF}'
            | '々'
            | '〆'
    )
}

#[derive(Clone)]
enum EdgeKind {
    Lexicon,
    Oov,
    Inflected {
        dict_lemma: String,
        sem_class: Option<SemanticClass>,
        split_morphemes: &'static str,
        stem_chars: usize,
    },
}

#[derive(Clone)]
struct BackEdge {
    len_chars: usize,
    kind: EdgeKind,
}

/// Look up the best-cost `zh-Hant` translation gloss (`counterpart`) for a Japanese surface form,
/// preferring entries matching `upos` when provided.
pub fn lookup_ja_gloss(surface: &str, upos: Option<Upos>) -> Option<&'static str> {
    let db = db();
    let entries = db.lookup_raw(LangGroup::Ja, surface);
    if entries.is_empty() {
        return None;
    }
    if let Some(u) = upos {
        let bit = 1u32 << (u as u8);
        if let Some(best) = entries
            .iter()
            .filter(|e| {
                e.extra_off != NONE_U32
                    && ((e.upos_mask & bit) != 0 || code_to_upos(e.decided_upos) == Some(u))
            })
            .min_by_key(|e| e.cost)
            && let Some(gloss) = db.str_at(best.extra_off)
        {
            return Some(gloss);
        }
    }
    entries
        .iter()
        .filter(|e| e.extra_off != NONE_U32)
        .min_by_key(|e| e.cost)
        .and_then(|e| db.str_at(e.extra_off))
}

/// Segment a Japanese character slice using Viterbi minimum-cost DP over `ja.tsv`
/// and `ja_inflect.tsv`, emitting UD-aligned tokens (splitting conjugated verbs/adjectives
/// into stem + auxiliary morphemes).
pub fn segment_ja(chars: &[char], base_char: usize) -> Vec<JaSegment> {
    segment_ja_with_lexicon(chars, base_char, &[])
}

/// Segment a Japanese character slice with caller-supplied custom lexicon terms honored at low cost.
pub fn segment_ja_with_lexicon(
    chars: &[char],
    base_char: usize,
    extra_lexicon: &[String],
) -> Vec<JaSegment> {
    let n = chars.len();
    if n == 0 {
        return Vec::new();
    }

    let custom_terms: Vec<(&str, usize)> = extra_lexicon
        .iter()
        .filter(|t| !t.is_empty() && !t.chars().any(char::is_whitespace))
        .map(|t| (t.as_str(), t.chars().count()))
        .collect();

    let mut text = String::with_capacity(n * 3);
    let mut byte_offs = Vec::with_capacity(n + 1);
    for &c in chars {
        byte_offs.push(text.len());
        text.push(c);
    }
    byte_offs.push(text.len());

    let db = db();
    let infl_idx = inflect_index();

    let mut dp_cost = vec![u32::MAX; n + 1];
    let mut back: Vec<Option<BackEdge>> = vec![None; n + 1];
    dp_cost[0] = 0;

    let mut i = 0usize;
    while i < n {
        if dp_cost[i] == u32::MAX {
            i += 1;
            continue;
        }
        let base_cost = dp_cost[i];
        let ch = chars[i];

        if ch.is_whitespace() {
            if base_cost < dp_cost[i + 1] {
                dp_cost[i + 1] = base_cost;
                back[i + 1] = Some(BackEdge {
                    len_chars: 1,
                    kind: EdgeKind::Oov,
                });
            }
            i += 1;
            continue;
        }

        // Contiguous ASCII alphanumeric run.
        if is_ascii_word_char(ch) {
            let mut j = i + 1;
            while j < n && is_ascii_word_char(chars[j]) {
                j += 1;
            }
            let cand_cost = base_cost + 200;
            if cand_cost < dp_cost[j] {
                dp_cost[j] = cand_cost;
                back[j] = Some(BackEdge {
                    len_chars: j - i,
                    kind: EdgeKind::Oov,
                });
            }
            i = j;
            continue;
        }

        // Single-char OOV fallback.
        let oov_cost = base_cost + SINGLE_OOV_COST;
        if oov_cost < dp_cost[i + 1] {
            dp_cost[i + 1] = oov_cost;
            back[i + 1] = Some(BackEdge {
                len_chars: 1,
                kind: EdgeKind::Oov,
            });
        }

        // Contiguous Katakana OOV run so loanwords don't shatter into single chars.
        if is_katakana_run_char(ch) {
            let mut j = i + 1;
            while j < n && is_katakana_run_char(chars[j]) {
                j += 1;
            }
            if j - i >= 2 {
                let kat_cost = base_cost
                    + KATAKANA_OOV_BASE_COST
                    + ((j - i) as u32) * KATAKANA_OOV_PER_CHAR_COST;
                if kat_cost < dp_cost[j] {
                    dp_cost[j] = kat_cost;
                    back[j] = Some(BackEdge {
                        len_chars: j - i,
                        kind: EdgeKind::Oov,
                    });
                }
            }
        }

        // Exact lexicon common-prefix matches.
        let rest = &text[byte_offs[i]..];
        for &(term, term_chars) in &custom_terms {
            if i + term_chars <= n && rest.starts_with(term) {
                let end = i + term_chars;
                let cand_cost = base_cost + 250;
                if cand_cost < dp_cost[end] {
                    dp_cost[end] = cand_cost;
                    back[end] = Some(BackEdge {
                        len_chars: term_chars,
                        kind: EdgeKind::Lexicon,
                    });
                }
            }
        }
        let max_c = MAX_WORD_CHARS.min(n - i);
        db.common_prefix_search(
            LangGroup::Ja,
            rest,
            max_c,
            |_byte_len, char_len, entries| {
                let min_cost = entries.iter().map(|e| e.cost as u32).min().unwrap_or(1500);
                let end = i + char_len;
                let cand_cost = base_cost + min_cost;
                if cand_cost < dp_cost[end] {
                    dp_cost[end] = cand_cost;
                    back[end] = Some(BackEdge {
                        len_chars: char_len,
                        kind: EdgeKind::Lexicon,
                    });
                }
            },
        );

        // Inflected verb/adjective lattice edges via ja_inflect.tsv.
        if is_ja_stem_char(ch) {
            let max_stem = MAX_STEM_CHARS.min(n - i);
            for stem_len in 0..=max_stem {
                let stem_end = i + stem_len;
                if stem_end >= n {
                    break;
                }
                if stem_len > 0 && !is_ja_stem_char(chars[stem_end - 1]) {
                    break;
                }
                let tail_ch = chars[stem_end];
                let Some(candidates) = infl_idx.by_first_char.get(&tail_ch) else {
                    continue;
                };
                let tail_rest = &text[byte_offs[stem_end]..];
                let stem_str = &text[byte_offs[i]..byte_offs[stem_end]];

                for inf in candidates {
                    if !tail_rest.starts_with(inf.rule.conjugated_tail) {
                        continue;
                    }
                    // Check stem_drop_chars compatibility: 0-char stem only for 2-char drop (sahen/kahen).
                    if stem_len == 0 && inf.rule.stem_drop_chars < 2 {
                        continue;
                    }
                    if stem_len > 0 && inf.rule.stem_drop_chars >= 2 {
                        continue;
                    }

                    let dict_form = if stem_len == 0 {
                        inf.dict_ending.to_owned()
                    } else {
                        format!("{stem_str}{}", inf.dict_ending)
                    };

                    // Also handle くる for kahen when hiragana.
                    let hit = lookup_ja(&dict_form).or_else(|| {
                        if inf.paradigm == "kahen" && stem_len == 0 {
                            lookup_ja("くる")
                        } else {
                            None
                        }
                    });
                    let Some(entry) = hit else {
                        continue;
                    };
                    if entry.infl != Some(inf.paradigm) {
                        continue;
                    }

                    let end = stem_end + inf.tail_chars;
                    let cand_cost = base_cost + (entry.cost as u32) + (inf.rule.cost_delta as u32);
                    if cand_cost < dp_cost[end] {
                        dp_cost[end] = cand_cost;
                        back[end] = Some(BackEdge {
                            len_chars: end - i,
                            kind: EdgeKind::Inflected {
                                dict_lemma: entry.lemma.unwrap_or(dict_form),
                                sem_class: entry.sem_class,
                                split_morphemes: inf.rule.split_morphemes,
                                stem_chars: stem_len,
                            },
                        });
                    }
                }
            }
        }

        i += 1;
    }

    // Reconstruct path in forward order.
    let mut path: Vec<(usize, usize, BackEdge)> = Vec::new();
    let mut pos = n;
    while pos > 0 {
        let edge = back[pos].clone().unwrap_or(BackEdge {
            len_chars: 1,
            kind: EdgeKind::Oov,
        });
        let start = pos - edge.len_chars;
        path.push((start, pos, edge));
        pos = start;
    }
    path.reverse();

    let mut out = Vec::with_capacity(path.len());
    for (start, end, edge) in path {
        if chars[start..end].iter().all(|c| c.is_whitespace()) {
            continue;
        }
        match edge.kind {
            EdgeKind::Lexicon => {
                let surf = &text[byte_offs[start]..byte_offs[end]];
                if let Some(hit) = lookup_ja(surf) {
                    out.push(JaSegment {
                        span: Span {
                            start_char: base_char + start,
                            end_char: base_char + end,
                        },
                        decided_upos: hit.decided_upos,
                        upos_mask: hit.upos_mask,
                        upos_candidates: hit.upos_candidates,
                        lemma: hit.lemma,
                        sem_class: hit.sem_class,
                    });
                } else {
                    out.push(JaSegment {
                        span: Span {
                            start_char: base_char + start,
                            end_char: base_char + end,
                        },
                        decided_upos: None,
                        upos_mask: 0,
                        upos_candidates: Vec::new(),
                        lemma: None,
                        sem_class: None,
                    });
                }
            }
            EdgeKind::Oov => {
                out.push(JaSegment {
                    span: Span {
                        start_char: base_char + start,
                        end_char: base_char + end,
                    },
                    decided_upos: None,
                    upos_mask: 0,
                    upos_candidates: Vec::new(),
                    lemma: None,
                    sem_class: None,
                });
            }
            EdgeKind::Inflected {
                dict_lemma,
                sem_class,
                split_morphemes,
                stem_chars,
            } => {
                let mut cur = start;
                let mut pieces_ok = true;
                let mut split_out = Vec::new();
                for (p_idx, piece) in split_morphemes.split('+').enumerate() {
                    let parts: Vec<&str> = piece.split('/').collect();
                    if parts.len() != 3 {
                        pieces_ok = false;
                        break;
                    }
                    let surf_pat = parts[0];
                    let lemma_pat = parts[1];
                    let upos_str = parts[2];
                    let piece_chars = if let Some(rest) = surf_pat.strip_prefix('*') {
                        stem_chars + rest.chars().count()
                    } else {
                        surf_pat.chars().count()
                    };
                    if piece_chars == 0 || cur + piece_chars > end {
                        pieces_ok = false;
                        break;
                    }
                    let piece_lemma = if lemma_pat == "<LEMMA>" {
                        dict_lemma.clone()
                    } else {
                        lemma_pat.to_owned()
                    };
                    let piece_upos =
                        super::format::upos_str_to_code(upos_str).and_then(code_to_upos);
                    let mask = piece_upos.map(|u| 1u32 << (u as u8)).unwrap_or(0);
                    let piece_surf: String = chars[cur..cur + piece_chars].iter().collect();
                    let emit_lemma = if piece_lemma == piece_surf {
                        None
                    } else {
                        Some(piece_lemma)
                    };
                    split_out.push(JaSegment {
                        span: Span {
                            start_char: base_char + cur,
                            end_char: base_char + cur + piece_chars,
                        },
                        decided_upos: piece_upos,
                        upos_mask: mask,
                        upos_candidates: mask_to_candidates(mask),
                        lemma: emit_lemma,
                        sem_class: if p_idx == 0 { sem_class } else { None },
                    });
                    cur += piece_chars;
                }
                if pieces_ok && cur == end && !split_out.is_empty() {
                    out.extend(split_out);
                } else {
                    // Safe fallback: emit single span with head lemma.
                    let mask = 1u32 << (Upos::Verb as u8);
                    out.push(JaSegment {
                        span: Span {
                            start_char: base_char + start,
                            end_char: base_char + end,
                        },
                        decided_upos: decided_from_mask(mask),
                        upos_mask: mask,
                        upos_candidates: mask_to_candidates(mask),
                        lemma: Some(dict_lemma),
                        sem_class,
                    });
                }
            }
        }
    }

    out
}
