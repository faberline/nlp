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
    HagakiTopic,
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

fn overlaps_custom_term(
    start: usize,
    end: usize,
    text: &str,
    byte_offs: &[usize],
    custom_terms: &[(&str, usize)],
) -> bool {
    custom_terms.iter().any(|(term, len)| {
        (start.saturating_sub(len.saturating_sub(1))..end)
            .any(|at| at + len < byte_offs.len() && text[byte_offs[at]..].starts_with(term))
    })
}

fn has_human_topic(surface: &str) -> bool {
    const PERSONAL_PRONOUNS: &[&str] = &[
        "私",
        "わたし",
        "あたし",
        "僕",
        "ぼく",
        "俺",
        "おれ",
        "彼",
        "彼女",
        "君",
        "きみ",
        "あなた",
        "お前",
        "私たち",
        "僕たち",
        "彼ら",
        "彼女たち",
    ];
    // These nouns denote people even when the dictionary only supplies POS.
    const HUMAN_NOUNS: &[&str] = &[
        "父", "母", "兄", "姉", "弟", "妹", "息子", "娘", "祖父", "祖母", "夫", "妻", "友達",
        "学生",
    ];
    lookup_ja(surface).is_some_and(|hit| {
        hit.sem_class == Some(SemanticClass::Person)
            || (hit.upos_candidates.contains(&Upos::Pron) && PERSONAL_PRONOUNS.contains(&surface))
            || (hit.decided_upos == Some(Upos::Noun) && HUMAN_NOUNS.contains(&surface))
    })
}

fn follows_with_age_or_copula(suffix: &str) -> bool {
    let phrase = suffix
        .split(['。', '、', '！', '？', '!', '?', ',', '」', '』'])
        .next()
        .unwrap_or(suffix);
    if let Some(tail) = phrase.strip_prefix('の') {
        return ["頃", "ころ", "時", "とき"]
            .into_iter()
            .filter_map(|time| tail.strip_prefix(time))
            .any(|rest| {
                rest.is_empty()
                    || rest.starts_with(['に', 'は'])
                    || ["から", "まで", "より"]
                        .into_iter()
                        .any(|case| rest.starts_with(case))
            });
    }
    let copula = ["よね", "かな", "か", "よ", "ね", "な"]
        .into_iter()
        .find_map(|particle| phrase.strip_suffix(particle))
        .unwrap_or(phrase);
    matches!(
        copula,
        "だ" | "だった"
            | "です"
            | "でした"
            | "じゃない"
            | "ではない"
            | "じゃなかった"
            | "ではなかった"
    )
}

fn resolve_hagaki_topic(
    path: &mut [(usize, usize, BackEdge)],
    text: &str,
    byte_offs: &[usize],
    custom_terms: &[(&str, usize)],
) {
    for index in 1..path.len() {
        let (subject_start, subject_end, subject) = &path[index - 1];
        let (start, end, edge) = &path[index];
        let (start, end) = (*start, *end);
        if end - start == 3
            && matches!(edge.kind, EdgeKind::Lexicon)
            && &text[byte_offs[start]..byte_offs[end]] == "はがき"
            && *subject_end == start
            && matches!(subject.kind, EdgeKind::Lexicon)
            && follows_with_age_or_copula(&text[byte_offs[end]..])
            && has_human_topic(&text[byte_offs[*subject_start]..byte_offs[start]])
            && !overlaps_custom_term(start, end, text, byte_offs, custom_terms)
        {
            // The age phrase or copula supports がき as a noun after topic は.
            // The alias supplies the noun's lemma while retaining the source.
            path[index].2.kind = EdgeKind::HagakiTopic;
        }
    }
}

fn resolve_noon_endpoint(
    path: &mut [(usize, usize, BackEdge)],
    chars: &[char],
    text: &str,
    byte_offs: &[usize],
    custom_terms: &[(&str, usize)],
    verified_sentence_tail: bool,
) {
    // These forms can continue 昼ま (daytime) with a copula or では even
    // when the lattice selected a separate で. Keep ambiguous readings.
    const DAYTIME_CONTINUATIONS: &[&str] = &[
        "して",
        "しょう",
        "ある",
        "あり",
        "あっ",
        "あろ",
        "ござ",
        "ない",
        "なく",
        "なかっ",
        "なけれ",
    ];
    for index in 1..path.len() {
        let (start, middle, first) = &path[index - 1];
        let (next_start, end, second) = &path[index];
        let (start, middle, next_start, end) = (*start, *middle, *next_start, *end);
        if first.len_chars != 2
            || second.len_chars != 1
            || middle != next_start
            || !matches!(first.kind, EdgeKind::Lexicon)
            || !matches!(second.kind, EdgeKind::Lexicon)
            || chars[start..end] != ['昼', 'ま', 'で']
        {
            continue;
        }
        let in_context = match chars.get(end) {
            Some('。' | '！' | '？' | '!' | '?') => true,
            Some(&ch) => is_ja_stem_char(ch),
            None => verified_sentence_tail,
        };
        let suffix = &text[byte_offs[end]..];
        let followed_by_topic_particle = suffix.starts_with('は')
            && path
                .get(index + 1)
                .is_some_and(|(_, _, edge)| edge.len_chars == 1);
        // Plain した may be the remainder of the daytime copula でした.
        // The selected する inflection with たい is a distinct action.
        let ambiguous_past = suffix.starts_with("した")
            && !path.get(index + 1).is_some_and(|(next, _, edge)| {
                *next == end
                    && matches!(
                        &edge.kind,
                        EdgeKind::Inflected {
                            dict_lemma,
                            split_morphemes,
                            stem_chars: 0,
                            ..
                        } if dict_lemma == "する"
                            && split_morphemes.split('+').any(|piece| piece.ends_with("/たい/AUX"))
                    )
            });
        if !in_context
            || followed_by_topic_particle
            || ambiguous_past
            || DAYTIME_CONTINUATIONS
                .iter()
                .any(|part| suffix.starts_with(part))
            || overlaps_custom_term(start, end, text, byte_offs, custom_terms)
        {
            continue;
        }

        // The dictionary favors 昼ま + で by nine cost points. Move the boundary
        // to emit the existing 昼 and まで entries without changing other costs.
        let boundary = start + 1;
        path[index - 1].1 = boundary;
        path[index - 1].2.len_chars = 1;
        path[index].0 = boundary;
        path[index].2.len_chars = 2;
    }
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
    segment_ja_with_context(chars, base_char, extra_lexicon, false)
}

fn lexicon_segment(span: Span, surface: &str) -> JaSegment {
    if let Some(hit) = lookup_ja(surface) {
        JaSegment {
            span,
            decided_upos: hit.decided_upos,
            upos_mask: hit.upos_mask,
            upos_candidates: hit.upos_candidates,
            lemma: hit.lemma,
            sem_class: hit.sem_class,
        }
    } else {
        JaSegment {
            span,
            decided_upos: None,
            upos_mask: 0,
            upos_candidates: Vec::new(),
            lemma: None,
            sem_class: None,
        }
    }
}

fn hagaki_topic_segments(start: usize, end: usize) -> [JaSegment; 2] {
    let boundary = start + 1;
    let mut noun = lexicon_segment(
        Span {
            start_char: boundary,
            end_char: end,
        },
        "ガキ",
    );
    noun.lemma.get_or_insert_with(|| "ガキ".to_owned());
    [
        lexicon_segment(
            Span {
                start_char: start,
                end_char: boundary,
            },
            "は",
        ),
        noun,
    ]
}

pub(crate) fn segment_ja_with_context(
    chars: &[char],
    base_char: usize,
    extra_lexicon: &[String],
    verified_sentence_tail: bool,
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

                    let dict_form = if stem_len == 0 {
                        inf.dict_ending.to_owned()
                    } else {
                        format!("{stem_str}{}", inf.dict_ending)
                    };

                    // Also handle くる for kahen when hiragana (standalone or compound).
                    let hit = lookup_ja(&dict_form).or_else(|| {
                        if inf.paradigm == "kahen" {
                            if stem_len == 0 {
                                lookup_ja("くる")
                            } else {
                                lookup_ja(&format!("{stem_str}くる"))
                            }
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
    if text.contains("はがき") {
        resolve_hagaki_topic(&mut path, &text, &byte_offs, &custom_terms);
    }
    resolve_noon_endpoint(
        &mut path,
        chars,
        &text,
        &byte_offs,
        &custom_terms,
        verified_sentence_tail,
    );

    // Only resolve the selected ambiguous lexical-stem + polite auxiliary
    // path when an existing dictionary-backed finite inflection describes
    // the same two source spans. Do not alter lattice costs.
    let mut index = 0;
    while index + 1 < path.len() {
        let (start, middle, first) = &path[index];
        let (aux_start, end, second) = &path[index + 1];
        let (start, middle, aux_start, end) = (*start, *middle, *aux_start, *end);
        let terminal = match chars.get(end) {
            None => verified_sentence_tail,
            Some('。' | '？' | '?' | '！' | '!') => true,
            Some('か') => match chars.get(end + 1) {
                None => verified_sentence_tail,
                Some('。' | '？' | '?') => true,
                _ => false,
            },
            _ => false,
        };
        if middle != aux_start
            || !terminal
            || !matches!(first.kind, EdgeKind::Lexicon)
            || !matches!(second.kind, EdgeKind::Lexicon)
            || &text[byte_offs[middle]..byte_offs[end]] != "ます"
        {
            index += 1;
            continue;
        }
        if overlaps_custom_term(start, end, &text, &byte_offs, &custom_terms) {
            index += 1;
            continue;
        }
        let stem = &text[byte_offs[start]..byte_offs[middle]];
        let Some(stem_hit) = lookup_ja(stem) else {
            index += 1;
            continue;
        };
        if !stem_hit.upos_candidates.contains(&Upos::Noun)
            || !stem_hit.upos_candidates.contains(&Upos::Adv)
            || stem_hit.upos_candidates.contains(&Upos::Verb)
            || !lookup_ja("ます").is_some_and(|hit| hit.decided_upos == Some(Upos::Aux))
        {
            index += 1;
            continue;
        }
        let replacement = infl_idx.by_first_char.get(&'ま').and_then(|rules| {
            let mut found = None;
            for inf in rules {
                if inf.rule.conjugated_tail != "ます"
                    || inf.rule.split_morphemes != "*/<LEMMA>/VERB+ます/ます/AUX"
                    || inf.tail_chars != end - middle
                    || inf.paradigm != "ichidan"
                    || inf.rule.stem_drop_chars != 1
                    || code_to_upos(inf.rule.head_upos) != Some(Upos::Verb)
                {
                    continue;
                }
                let dict_form = format!("{stem}{}", inf.dict_ending);
                let Some(entry) = lookup_ja(&dict_form) else {
                    continue;
                };
                if entry.infl != Some(inf.paradigm) || !entry.upos_candidates.contains(&Upos::Verb)
                {
                    continue;
                }
                if found.is_some() {
                    return None;
                }
                found = Some(BackEdge {
                    len_chars: end - start,
                    kind: EdgeKind::Inflected {
                        dict_lemma: entry.lemma.unwrap_or(dict_form),
                        sem_class: entry.sem_class,
                        split_morphemes: inf.rule.split_morphemes,
                        stem_chars: middle - start,
                    },
                });
            }
            found
        });
        if let Some(edge) = replacement {
            path.splice(index..index + 2, [(start, end, edge)]);
        }
        index += 1;
    }

    let mut out = Vec::with_capacity(path.len());
    for (start, end, edge) in path {
        if chars[start..end].iter().all(|c| c.is_whitespace()) {
            continue;
        }
        match edge.kind {
            EdgeKind::Lexicon => {
                out.push(lexicon_segment(
                    Span {
                        start_char: base_char + start,
                        end_char: base_char + end,
                    },
                    &text[byte_offs[start]..byte_offs[end]],
                ));
            }
            EdgeKind::HagakiTopic => {
                out.extend(hagaki_topic_segments(base_char + start, base_char + end));
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

/// Map a single Japanese Joyo/Jinmeiyo shinjitai Han character to its Traditional Han glyph
/// using Unicode-3.0 Unihan `kTraditionalVariant` / JIS X 0208 pure glyph pairs.
///
/// Ambiguous characters where Taiwan `zh-Hant` differs by word (such as `台`, `着`, `弁`, `欠`)
/// and lexical substitutions (such as `駅` -> `站`) are intentionally left unmapped.
pub fn map_ja_shinjitai_char(c: char) -> char {
    match c {
        '亜' => '亞',
        '悪' => '惡',
        '圧' => '壓',
        '囲' => '圍',
        '医' => '醫',
        '壱' => '壹',
        '営' => '營',
        '栄' => '榮',
        '駅' => '驛',
        '閲' => '閱',
        '円' => '圓',
        '塩' => '鹽',
        '縁' => '緣',
        '艶' => '豔',
        '応' => '應',
        '殴' => '毆',
        '桜' => '櫻',
        '奥' => '奧',
        '横' => '橫',
        '温' => '溫',
        '穏' => '穩',
        '仮' => '假',
        '価' => '價',
        '画' => '畫',
        '会' => '會',
        '壊' => '壞',
        '懐' => '懷',
        '絵' => '繪',
        '拡' => '擴',
        '殻' => '殼',
        '覚' => '覺',
        '学' => '學',
        '楽' => '樂',
        '渇' => '渴',
        '巻' => '卷',
        '勧' => '勸',
        '歓' => '歡',
        '観' => '觀',
        '関' => '關',
        '顔' => '顏',
        '気' => '氣',
        '帰' => '歸',
        '亀' => '龜',
        '戯' => '戲',
        '犠' => '犧',
        '旧' => '舊',
        '拠' => '據',
        '挙' => '舉',
        '虚' => '虛',
        '峡' => '峽',
        '狭' => '狹',
        '挟' => '挾',
        '郷' => '鄉',
        '暁' => '曉',
        '区' => '區',
        '駆' => '驅',
        '勲' => '勳',
        '薫' => '薰',
        '径' => '徑',
        '茎' => '莖',
        '恵' => '惠',
        '掲' => '揭',
        '渓' => '溪',
        '経' => '經',
        '継' => '繼',
        '蛍' => '螢',
        '軽' => '輕',
        '鶏' => '雞',
        '芸' => '藝',
        '撃' => '擊',
        '県' => '縣',
        '倹' => '儉',
        '剣' => '劍',
        '圏' => '圈',
        '険' => '險',
        '検' => '檢',
        '権' => '權',
        '献' => '獻',
        '顕' => '顯',
        '験' => '驗',
        '厳' => '嚴',
        '効' => '效',
        '広' => '廣',
        '恒' => '恆',
        '黄' => '黃',
        '鉱' => '礦',
        '号' => '號',
        '国' => '國',
        '黒' => '黑',
        '砕' => '碎',
        '済' => '濟',
        '斉' => '齊',
        '斎' => '齋',
        '剤' => '劑',
        '残' => '殘',
        '参' => '參',
        '惨' => '慘',
        '賛' => '贊',
        '蚕' => '蠶',
        '糸' => '絲',
        '歯' => '齒',
        '児' => '兒',
        '辞' => '辭',
        '湿' => '濕',
        '実' => '實',
        '写' => '寫',
        '舎' => '舍',
        '釈' => '釋',
        '寿' => '壽',
        '収' => '收',
        '渋' => '澀',
        '獣' => '獸',
        '従' => '從',
        '縦' => '縱',
        '粛' => '肅',
        '処' => '處',
        '叙' => '敘',
        '将' => '將',
        '奨' => '獎',
        '焼' => '燒',
        '称' => '稱',
        '証' => '證',
        '乗' => '乘',
        '剰' => '剩',
        '浄' => '淨',
        '状' => '狀',
        '条' => '條',
        '畳' => '疊',
        '嬢' => '孃',
        '譲' => '讓',
        '醸' => '釀',
        '嘱' => '囑',
        '触' => '觸',
        '寝' => '寢',
        '尽' => '盡',
        '図' => '圖',
        '粋' => '粹',
        '酔' => '醉',
        '随' => '隨',
        '髄' => '髓',
        '枢' => '樞',
        '数' => '數',
        '瀬' => '瀨',
        '声' => '聲',
        '静' => '靜',
        '窃' => '竊',
        '摂' => '攝',
        '説' => '說',
        '絶' => '絕',
        '専' => '專',
        '戦' => '戰',
        '浅' => '淺',
        '銭' => '錢',
        '潜' => '潛',
        '繊' => '纖',
        '禅' => '禪',
        '双' => '雙',
        '壮' => '壯',
        '捜' => '搜',
        '挿' => '插',
        '巣' => '巢',
        '争' => '爭',
        '痩' => '瘦',
        '装' => '裝',
        '総' => '總',
        '騒' => '騷',
        '増' => '增',
        '臓' => '臟',
        '蔵' => '藏',
        '属' => '屬',
        '続' => '續',
        '堕' => '墮',
        '体' => '體',
        '対' => '對',
        '帯' => '帶',
        '滞' => '滯',
        '滝' => '瀧',
        '択' => '擇',
        '沢' => '澤',
        '単' => '單',
        '担' => '擔',
        '胆' => '膽',
        '団' => '團',
        '弾' => '彈',
        '断' => '斷',
        '痴' => '癡',
        '遅' => '遲',
        '虫' => '蟲',
        '昼' => '晝',
        '鋳' => '鑄',
        '庁' => '廳',
        '徴' => '徵',
        '聴' => '聽',
        '転' => '轉',
        '鉄' => '鐵',
        '点' => '點',
        '伝' => '傳',
        '党' => '黨',
        '盗' => '盜',
        '灯' => '燈',
        '当' => '當',
        '闘' => '鬥',
        '徳' => '德',
        '独' => '獨',
        '読' => '讀',
        '届' => '屆',
        '縄' => '繩',
        '弐' => '貳',
        '悩' => '惱',
        '脳' => '腦',
        '覇' => '霸',
        '廃' => '廢',
        '拝' => '拜',
        '売' => '賣',
        '麦' => '麥',
        '発' => '發',
        '髪' => '髮',
        '抜' => '拔',
        '晩' => '晚',
        '蛮' => '蠻',
        '浜' => '濱',
        '払' => '拂',
        '仏' => '佛',
        '変' => '變',
        '辺' => '邊',
        '歩' => '步',
        '穂' => '穗',
        '宝' => '寶',
        '豊' => '豐',
        '没' => '沒',
        '毎' => '每',
        '万' => '萬',
        '満' => '滿',
        '黙' => '默',
        '薬' => '藥',
        '訳' => '譯',
        '与' => '與',
        '余' => '餘',
        '誉' => '譽',
        '揺' => '搖',
        '様' => '樣',
        '謡' => '謠',
        '来' => '來',
        '頼' => '賴',
        '乱' => '亂',
        '覧' => '覽',
        '竜' => '龍',
        '両' => '兩',
        '猟' => '獵',
        '緑' => '綠',
        '隣' => '鄰',
        '涙' => '淚',
        '塁' => '壘',
        '励' => '勵',
        '礼' => '禮',
        '霊' => '靈',
        '齢' => '齡',
        '歴' => '歷',
        '暦' => '曆',
        '恋' => '戀',
        '錬' => '鍊',
        '炉' => '爐',
        '労' => '勞',
        '楼' => '樓',
        '禄' => '祿',
        '録' => '錄',
        '湾' => '灣',
        '戸' => '戶',
        '氷' => '冰',
        '啓' => '啟',
        '呉' => '吳',
        '壌' => '壤',
        '姫' => '姬',
        '寛' => '寬',
        '巌' => '巖',
        '弥' => '彌',
        '祷' => '禱',
        '芦' => '蘆',
        '遥' => '遙',
        '鴎' => '鷗',
        '麺' => '麵',
        '麹' => '麴',
        '尭' => '堯',
        '槙' => '槇',
        '曽' => '曾',
        '彦' => '彥',
        '渉' => '涉',
        '聡' => '聰',
        '厘' => '釐',
        '舗' => '鋪',
        '雑' => '雜',
        '産' => '產',
        '値' => '值',
        '隠' => '隱',
        '陥' => '陷',
        '践' => '踐',
        '逓' => '遞',
        '鋭' => '銳',
        '隷' => '隸',
        '頬' => '頰',
        '剥' => '剝',
        '篭' => '籠',
        '歳' => '歲',
        '悦' => '悅',
        '荘' => '莊',
        '諌' => '諫',
        '遡' => '溯',
        '脱' => '脫',
        '査' => '查',
        '税' => '稅',
        '廻' => '迴',
        _ => c,
    }
}

/// Convert Japanese shinjitai characters in `s` to Traditional Han glyphs via
/// [`map_ja_shinjitai_char`].
pub fn map_ja_shinjitai(s: &str) -> String {
    s.chars().map(map_ja_shinjitai_char).collect()
}

#[cfg(test)]
mod noon_endpoint_segmentation_tests {
    use super::*;

    #[test]
    fn noon_endpoint_and_deadline_keep_source_spans() {
        for (source, particles) in [
            ("昼までに提出します。", &["昼", "まで", "に"][..]),
            ("昼まで店を開けます。", &["昼", "まで"]),
            ("昼までです。", &["昼", "まで", "です"]),
        ] {
            let chars: Vec<_> = source.chars().collect();
            let base = 11;
            let segments = segment_ja(&chars, base);
            let mut offset = 0;
            for (segment, surface) in segments.iter().zip(particles) {
                let end = offset + surface.chars().count();
                assert_eq!(
                    segment.span,
                    Span {
                        start_char: base + offset,
                        end_char: base + end
                    }
                );
                assert_eq!(chars[offset..end].iter().collect::<String>(), *surface);
                offset = end;
            }
            assert!(segments.len() >= particles.len());
            assert_eq!(segments[1].decided_upos, Some(Upos::Adp));
        }
    }

    #[test]
    fn noon_endpoint_before_a_desired_action_preserves_the_inflection() {
        for (source, continuation) in [
            ("仕事を昼までしたいです。", "たい"),
            ("仕事を昼までしたかったです。", "たかっ"),
            ("勉強を昼までしたくないです。", "たく"),
        ] {
            let chars: Vec<_> = source.chars().collect();
            let segments = segment_ja(&chars, 0);
            let time = source.chars().position(|ch| ch == '昼').unwrap();
            let part: Vec<_> = segments
                .iter()
                .filter(|segment| segment.span.start_char >= time)
                .take(4)
                .collect();
            let surfaces: Vec<String> = part
                .iter()
                .map(|segment| {
                    chars[segment.span.start_char..segment.span.end_char]
                        .iter()
                        .collect()
                })
                .collect();
            assert_eq!(surfaces, ["昼", "まで", "し", continuation], "{source}");
            assert_eq!(part[1].decided_upos, Some(Upos::Adp));
            assert_eq!(part[2].lemma.as_deref(), Some("する"));
            assert_eq!(part[3].decided_upos, Some(Upos::Aux));
        }
    }

    #[test]
    fn daytime_and_explicit_boundaries_keep_their_lexical_reading() {
        for (source, base) in [
            ("昼まです。", 0),
            ("昼までも暗いです。", 0),
            ("昼までした。", 0),
            ("昼までは明るいです。", 0),
            ("昼でなく、昼まではない。", 5),
            ("昼までございます。", 0),
            ("昼まに会います。", 0),
            ("昼ま で働きます。", 0),
            ("「昼ま」で働きます。", 1),
            ("昼まで、もう一方は夜でした。", 0),
        ] {
            let chars: Vec<_> = source.chars().collect();
            let segments = segment_ja(&chars, 0);
            assert!(
                segments.iter().any(|segment| {
                    segment.span
                        == Span {
                            start_char: base,
                            end_char: base + 2,
                        }
                }),
                "{source}"
            );
        }
        let chars: Vec<_> = "昼までに出します。".chars().collect();
        let custom = ["昼ま".to_owned()];
        let segments = segment_ja_with_lexicon(&chars, 0, &custom);
        assert_eq!(
            segments[0].span,
            Span {
                start_char: 0,
                end_char: 2
            }
        );
    }
}

#[cfg(test)]
mod finite_polite_segmentation_tests {
    use super::*;

    #[test]
    fn existing_finite_inflection_wins_only_at_clause_end() {
        for source in ["先生は授業を始めます。", "先生は授業を始めますか。"]
        {
            let chars: Vec<_> = source.chars().collect();
            let segments = segment_ja(&chars, 0);
            let stem = segments.iter().find(|s| s.span.start_char == 6).unwrap();
            assert_eq!(stem.span.end_char, 8);
            assert_eq!(stem.decided_upos, Some(Upos::Verb));
            assert_eq!(stem.lemma.as_deref(), Some("始める"));
            let aux = segments.iter().find(|s| s.span.start_char == 8).unwrap();
            assert_eq!(aux.span.end_char, 10);
            assert_eq!(aux.decided_upos, Some(Upos::Aux));
        }
    }

    #[test]
    fn nominal_and_custom_boundaries_are_not_reclassified() {
        for source in ["本の始めは大切です。", "私は初めに名前を書きます。"]
        {
            let chars: Vec<_> = source.chars().collect();
            let segments = segment_ja(&chars, 0);
            let nominal = segments.iter().find(|s| s.span.start_char == 2).unwrap();
            assert_ne!(nominal.decided_upos, Some(Upos::Verb));
        }
        let chars: Vec<_> = "始めます。".chars().collect();
        let segments = segment_ja_with_lexicon(&chars, 0, &["始めます".to_owned()]);
        assert!(segments.iter().any(|s| s.span
            == Span {
                start_char: 0,
                end_char: 4
            }));
        let segments = segment_ja_with_lexicon(&chars, 0, &["始め".to_owned()]);
        let stem = segments.iter().find(|s| s.span.start_char == 0).unwrap();
        assert_ne!(stem.decided_upos, Some(Upos::Verb));
        let connective: Vec<_> = "始めますから".chars().collect();
        let segments = segment_ja(&connective, 0);
        assert_ne!(segments[0].decided_upos, Some(Upos::Verb));
        let two_sentences: Vec<_> = "始めます。続けます。".chars().collect();
        let segments = segment_ja(&two_sentences, 0);
        assert_eq!(segments[0].decided_upos, Some(Upos::Verb));
    }
}

#[cfg(test)]
mod hagaki_topic_segmentation_tests {
    use super::*;

    fn recovers_topic(previous: &str, suffix: &str, custom_terms: &[(&str, usize)]) -> bool {
        let text = format!("{previous}はがき{suffix}");
        let byte_offs: Vec<_> = text
            .char_indices()
            .map(|(byte, _)| byte)
            .chain(std::iter::once(text.len()))
            .collect();
        let start = previous.chars().count();
        let end = start + 3;
        let edge = |start, end| {
            (
                start,
                end,
                BackEdge {
                    len_chars: end - start,
                    kind: EdgeKind::Lexicon,
                },
            )
        };
        let mut path = [edge(0, start), edge(start, end)];
        resolve_hagaki_topic(&mut path, &text, &byte_offs, custom_terms);
        matches!(path[1].2.kind, EdgeKind::HagakiTopic)
    }

    #[test]
    fn human_topic_keeps_its_particle_before_childhood_and_copulas() {
        for (person, suffix) in [
            ("彼", "の頃から走っていた。"),
            ("私", "のころからここに住んでいます。"),
            ("僕", "の時に泳ぎを覚えた。"),
            ("中村", "のときに絵を習った。"),
            ("彼女", "だった。"),
            ("彼", "じゃないよ。"),
            ("彼女", "ではないよね。"),
            ("僕", "じゃなかったか。"),
            ("兄", "のころ、雨でも外で遊んだ。"),
            ("祖母", "の頃に川で泳いだ。"),
            ("娘", "ではないよ。"),
            ("友達", "じゃないよ。"),
            ("学生", "のときにサッカーを始めた。"),
        ] {
            assert!(
                recovers_topic(person, suffix, &[]),
                "{person}はがき{suffix}"
            );
        }

        let [topic, noun] = hagaki_topic_segments(12, 15);
        assert_eq!((topic.span.start_char, topic.span.end_char), (12, 13));
        assert_eq!(topic.decided_upos, Some(Upos::Adp));
        assert_eq!((noun.span.start_char, noun.span.end_char), (13, 15));
        assert_eq!(noun.decided_upos, Some(Upos::Noun));
        assert_eq!(noun.lemma.as_deref(), Some("ガキ"));
    }

    #[test]
    fn postcard_and_ambiguous_readings_keep_the_selected_word() {
        for (previous, suffix) in [
            ("と", "を送りました。"),
            ("、", "を送った。"),
            ("は", "の表に住所を書いた。"),
            ("新しい", "の束を買いました。"),
            ("これ", "ですか。"),
            ("これ", "ではないよ。"),
            ("昔", "だった。"),
            ("彼", "の表を見た。"),
            ("兄", "の表に住所を書いた。"),
        ] {
            assert!(
                !recovers_topic(previous, suffix, &[]),
                "{previous}はがき{suffix}"
            );
        }
        for term in [("はがき", 3), ("彼はがき", 4), ("がきの", 3)] {
            assert!(
                !recovers_topic("彼", "の頃に絵を習った。", &[term]),
                "{term:?}"
            );
        }
    }
}
