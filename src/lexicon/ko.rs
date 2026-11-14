//! Korean (`ko`) eojeol (어절) morphological decomposition, batchim & irregular stem recovery,
//! and exact character-span morpheme partitioning.

use std::sync::OnceLock;

use super::format::{KoBatchimCond, KoMorphKind, NONE_U8};
use super::{
    KoMorphRule, LexLookup, LexTokenAnalysis, Morpheme, code_to_upos, db, decided_from_mask,
    lookup_ko, mask_to_candidates,
};
use crate::Span;

#[derive(Clone)]
struct KoRuleEntry {
    rule: KoMorphRule,
    suffix_chars: usize,
}

fn ko_rules() -> &'static [KoRuleEntry] {
    static RULES: OnceLock<Vec<KoRuleEntry>> = OnceLock::new();
    RULES.get_or_init(|| {
        let mut list: Vec<KoRuleEntry> = db()
            .ko_morph_rules()
            .filter(|r| !r.surface_suffix.is_empty())
            .map(|rule| KoRuleEntry {
                suffix_chars: rule.surface_suffix.chars().count(),
                rule,
            })
            .collect();
        // Longest suffix first, then lowest cost.
        list.sort_by(|a, b| {
            b.suffix_chars
                .cmp(&a.suffix_chars)
                .then_with(|| a.rule.cost.cmp(&b.rule.cost))
        });
        list
    })
}

#[inline]
fn hangul_jongseong(c: char) -> Option<u32> {
    let code = c as u32;
    if (0xAC00..=0xD7A3).contains(&code) {
        Some((code - 0xAC00) % 28)
    } else {
        None
    }
}

#[inline]
fn has_batchim(c: char) -> bool {
    hangul_jongseong(c).is_some_and(|j| j != 0)
}

#[inline]
fn is_rieul_batchim(c: char) -> bool {
    hangul_jongseong(c) == Some(8)
}

#[inline]
fn replace_jongseong(c: char, new_jong: u32) -> Option<char> {
    let code = c as u32;
    if !(0xAC00..=0xD7A3).contains(&code) {
        return None;
    }
    let base = code - ((code - 0xAC00) % 28);
    char::from_u32(base + new_jong)
}

fn check_batchim_cond(stem_last: char, cond: KoBatchimCond) -> bool {
    match cond {
        KoBatchimCond::Any => true,
        KoBatchimCond::Batchim => has_batchim(stem_last),
        KoBatchimCond::NoBatchim => !has_batchim(stem_last),
        KoBatchimCond::RieulOrNoBatchim => !has_batchim(stem_last) || is_rieul_batchim(stem_last),
    }
}

fn parse_decomp_morphemes(
    decomp: &str,
    base_char: usize,
    expected_chars: usize,
) -> Option<Vec<Morpheme>> {
    let mut out = Vec::new();
    let mut cur = 0usize;
    for part in decomp.split('+') {
        let cols: Vec<&str> = part.split('/').collect();
        if cols.len() != 3 {
            return None;
        }
        let surf = cols[0];
        let lemma = cols[1];
        let upos = super::format::upos_str_to_code(cols[2]).and_then(code_to_upos);
        let len = surf.chars().count();
        if len == 0 {
            return None;
        }
        out.push(Morpheme {
            span: Span {
                start_char: base_char + cur,
                end_char: base_char + cur + len,
            },
            surface: surf.to_owned(),
            lemma: lemma.to_owned(),
            upos,
        });
        cur += len;
    }
    if cur == expected_chars {
        Some(out)
    } else {
        None
    }
}

fn lookup_ko_stem_with_irregulars(stem: &str, kind: KoMorphKind) -> Option<LexLookup> {
    if let Some(hit) = lookup_ko(stem) {
        return Some(hit);
    }
    if kind != KoMorphKind::Eomi {
        return None;
    }
    let mut chars: Vec<char> = stem.chars().collect();
    let last = *chars.last()?;

    // 1. ㄷ-irregular: ㄹ(8) -> ㄷ(7) (e.g., 걸 -> 걷다)
    if hangul_jongseong(last) == Some(8)
        && let Some(c2) = replace_jongseong(last, 7)
    {
        *chars.last_mut().unwrap() = c2;
        let cand: String = chars.iter().collect();
        if let Some(hit) = lookup_ko(&cand)
            && hit.infl == Some("irr_d")
        {
            return Some(hit);
        }
        *chars.last_mut().unwrap() = last;
    }

    // 2. ㅅ-irregular: no batchim(0) -> ㅅ(19) (e.g., 지 -> 짓다)
    if hangul_jongseong(last) == Some(0)
        && let Some(c2) = replace_jongseong(last, 19)
    {
        *chars.last_mut().unwrap() = c2;
        let cand: String = chars.iter().collect();
        if let Some(hit) = lookup_ko(&cand)
            && hit.infl == Some("irr_s")
        {
            return Some(hit);
        }
        *chars.last_mut().unwrap() = last;
    }

    // 3. ㅎ-irregular: no batchim(0) -> ㅎ(27) (e.g., 하야 -> 하얗다)
    if hangul_jongseong(last) == Some(0)
        && let Some(c2) = replace_jongseong(last, 27)
    {
        *chars.last_mut().unwrap() = c2;
        let cand: String = chars.iter().collect();
        if let Some(hit) = lookup_ko(&cand)
            && hit.infl == Some("irr_h")
        {
            return Some(hit);
        }
        *chars.last_mut().unwrap() = last;
    }

    // 4. ㅂ-irregular: trailing 우(0xC6B0) dropped and previous syllable gets ㅂ(17) (e.g., 아름다우 -> 아름답다)
    if last == '우' && chars.len() >= 2 {
        let prev = chars[chars.len() - 2];
        if hangul_jongseong(prev) == Some(0)
            && let Some(p2) = replace_jongseong(prev, 17)
        {
            let mut cand_chars = chars[..chars.len() - 1].to_vec();
            *cand_chars.last_mut().unwrap() = p2;
            let cand: String = cand_chars.into_iter().collect();
            if let Some(hit) = lookup_ko(&cand)
                && hit.infl == Some("irr_b")
            {
                return Some(hit);
            }
        }
    }

    None
}

const NOMINAL_MASK: u32 = (1u32 << 7) | (1u32 << 11) | (1u32 << 10) | (1u32 << 8); // NOUN | PROPN | PRON | NUM
const PREDICATE_MASK: u32 = (1u32 << 15) | (1u32 << 0) | (1u32 << 3); // VERB | ADJ | AUX

/// Analyze one Korean eojeol (`chars`) starting at document Unicode scalar offset `base_char`.
pub fn analyze_ko_eojeol(chars: &[char], base_char: usize) -> LexTokenAnalysis {
    let n = chars.len();
    if n == 0 {
        return LexTokenAnalysis::default();
    }
    let surface: String = chars.iter().collect();

    // 1. Exact lookup in ko.tsv (handles standalone words & Inflect.csv pre-analyzed forms).
    if let Some(hit) = lookup_ko(&surface) {
        let morphemes = hit
            .decomp
            .and_then(|d| parse_decomp_morphemes(d, base_char, n))
            .unwrap_or_default();
        return LexTokenAnalysis {
            decided_upos: hit.decided_upos,
            upos_mask: hit.upos_mask,
            upos_candidates: hit.upos_candidates,
            lemma: hit.lemma,
            canonical_variant: None,
            sem_class: hit.sem_class,
            morphemes,
        };
    }

    // 2. Strip longest/lowest-cost compatible suffix from ko_morph.tsv.
    let mut best_score = u32::MAX;
    let mut best_analysis: Option<LexTokenAnalysis> = None;

    for rule_entry in ko_rules() {
        let s_len = rule_entry.suffix_chars;
        if s_len >= n {
            continue;
        }
        let stem_chars = n - s_len;
        let Some(suffix_slice) = surface.strip_suffix(rule_entry.rule.surface_suffix) else {
            continue;
        };
        let _ = suffix_slice;
        let stem_last = chars[stem_chars - 1];
        if !check_batchim_cond(stem_last, rule_entry.rule.cond) {
            continue;
        }
        let stem: String = chars[..stem_chars].iter().collect();
        let Some(stem_hit) = lookup_ko_stem_with_irregulars(&stem, rule_entry.rule.kind) else {
            continue;
        };

        // Validate POS compatibility between stem and suffix kind.
        let compatible = match rule_entry.rule.kind {
            KoMorphKind::Josa | KoMorphKind::Cop | KoMorphKind::Xsv | KoMorphKind::Xsa => {
                (stem_hit.upos_mask & NOMINAL_MASK) != 0
            }
            KoMorphKind::Eomi => (stem_hit.upos_mask & PREDICATE_MASK) != 0,
        };
        if !compatible {
            continue;
        }

        let score =
            (stem_hit.cost as u32) + (rule_entry.rule.cost as u32) - ((s_len as u32) * 30).min(150);
        if score >= best_score {
            continue;
        }

        let Some(suffix_morphemes) =
            parse_decomp_morphemes(rule_entry.rule.decomp, base_char + stem_chars, s_len)
        else {
            continue;
        };

        let (eojeol_mask, eojeol_upos) = if rule_entry.rule.eojeol_upos_effect != NONE_U8 {
            let u = code_to_upos(rule_entry.rule.eojeol_upos_effect);
            let m = u.map(|x| 1u32 << (x as u8)).unwrap_or(0);
            (m, u)
        } else {
            let filtered_mask = match rule_entry.rule.kind {
                KoMorphKind::Josa | KoMorphKind::Cop => stem_hit.upos_mask & NOMINAL_MASK,
                KoMorphKind::Eomi => stem_hit.upos_mask & PREDICATE_MASK,
                _ => stem_hit.upos_mask,
            };
            (filtered_mask, decided_from_mask(filtered_mask))
        };

        let stem_lemma = stem_hit.lemma.clone().unwrap_or_else(|| stem.clone());
        let mut morphemes = Vec::with_capacity(1 + suffix_morphemes.len());
        morphemes.push(Morpheme {
            span: Span {
                start_char: base_char,
                end_char: base_char + stem_chars,
            },
            surface: stem,
            lemma: stem_lemma.clone(),
            upos: stem_hit.decided_upos,
        });
        morphemes.extend(suffix_morphemes);

        best_score = score;
        best_analysis = Some(LexTokenAnalysis {
            decided_upos: eojeol_upos,
            upos_mask: eojeol_mask,
            upos_candidates: mask_to_candidates(eojeol_mask),
            lemma: Some(stem_lemma),
            canonical_variant: None,
            sem_class: stem_hit.sem_class,
            morphemes,
        });
    }

    best_analysis.unwrap_or_default()
}
