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
