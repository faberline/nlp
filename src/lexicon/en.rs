//! English (`en-US` & `en-GB`) lexical lookup, regular morphological lemmatization,
//! US/GB spelling counterpart resolution, and semantic class tagging.

use super::{LexTokenAnalysis, lookup_en, mask_to_candidates};
use crate::doc::Upos;

const VERB_MASK: u32 = 1u32 << 15;
const NOUN_MASK: u32 = 1u32 << 7;
const ADJ_MASK: u32 = 1u32 << 0;

#[inline]
fn has_any_mask(mask: u32, target: u32) -> bool {
    (mask & target) != 0
}

fn try_stem_with_mask(stem: &str, required_mask: u32) -> Option<super::LexLookup> {
    if stem.len() < 2 {
        return None;
    }
    let hit = lookup_en(stem)?;
    if has_any_mask(hit.upos_mask, required_mask) {
        Some(hit)
    } else {
        None
    }
}

/// Analyze an English token:
/// 1. Exact / ASCII-lowercased lookup in `en.tsv` (covers base words, closed classes, irregulars, US/GB pairs).
/// 2. Regular English inflection fallback (`-ies`, `-es`, `-s`, `-ied`, `-ed` with doubled-consonant
///    & silent-e recovery, `-ing` verified against dictionary verbs) when not already in `en.tsv`
///    or when `en.tsv` has an inflected form (like `stopped` as `ADJ|VERB`) without an explicit lemma.
pub fn analyze_en_token(surface: &str) -> LexTokenAnalysis {
    if surface.is_empty() {
        return LexTokenAnalysis::default();
    }

    let lower = surface.to_ascii_lowercase();
    if let Some(hit) = lookup_en(surface) {
        let lemma = if hit.lemma.is_some() {
            hit.lemma
        } else if lower.is_ascii()
            && lower.len() >= 4
            && (lower.ends_with("ed") || lower.ends_with("ing") || lower.ends_with("ies"))
        {
            regular_fallback(&lower).and_then(|f| f.lemma)
        } else {
            None
        };
        return LexTokenAnalysis {
            decided_upos: hit.decided_upos,
            upos_mask: hit.upos_mask,
            upos_candidates: hit.upos_candidates,
            lemma,
            canonical_variant: hit.counterpart,
            sem_class: hit.sem_class,
            morphemes: Vec::new(),
        };
    }

    if !lower.is_ascii() || lower.len() < 3 {
        return LexTokenAnalysis::default();
    }
    regular_fallback(&lower).unwrap_or_default()
}

fn make_fallback_analysis(hit: super::LexLookup, lemma: String) -> LexTokenAnalysis {
    LexTokenAnalysis {
        decided_upos: hit.decided_upos,
        upos_mask: hit.upos_mask,
        upos_candidates: hit.upos_candidates,
        lemma: Some(lemma),
        canonical_variant: hit.counterpart,
        sem_class: hit.sem_class,
        morphemes: Vec::new(),
    }
}

fn regular_fallback(lower: &str) -> Option<LexTokenAnalysis> {
    // 1. Plural / 3sg -ies -> -y (e.g. studies -> study, cities -> city)
    if let Some(stem) = lower.strip_suffix("ies") {
        let cand = format!("{stem}y");
        if let Some(hit) = try_stem_with_mask(&cand, NOUN_MASK | VERB_MASK) {
            return Some(make_fallback_analysis(hit, cand));
        }
    }

    // 2. -es after sibilants or -o (boxes -> box, watches -> watch, goes -> go)
    if let Some(stem) = lower.strip_suffix("es")
        && (stem.ends_with('s')
            || stem.ends_with('x')
            || stem.ends_with('z')
            || stem.ends_with("ch")
            || stem.ends_with("sh")
            || stem.ends_with('o'))
        && let Some(hit) = try_stem_with_mask(stem, NOUN_MASK | VERB_MASK)
    {
        return Some(make_fallback_analysis(hit, stem.to_owned()));
    }

    // 3. Regular -s (books -> book, runs -> run, colours -> colour)
    if let Some(stem) = lower.strip_suffix('s')
        && !stem.ends_with('s')
        && !stem.ends_with('\'')
        && let Some(hit) = try_stem_with_mask(stem, NOUN_MASK | VERB_MASK)
    {
        return Some(make_fallback_analysis(hit, stem.to_owned()));
    }

    // 4. Past / participle -ied -> -y (studied -> study)
    if let Some(stem) = lower.strip_suffix("ied") {
        let cand = format!("{stem}y");
        if let Some(hit) = try_stem_with_mask(&cand, VERB_MASK | ADJ_MASK) {
            return Some(LexTokenAnalysis {
                decided_upos: Some(Upos::Verb),
                upos_mask: VERB_MASK,
                upos_candidates: mask_to_candidates(VERB_MASK),
                lemma: Some(cand),
                canonical_variant: hit.counterpart,
                sem_class: hit.sem_class,
                morphemes: Vec::new(),
            });
        }
    }

    // 5. Past / participle -ed (walked -> walk, baked -> bake, stopped -> stop)
    if let Some(stem) = lower.strip_suffix("ed") {
        let bytes = stem.as_bytes();
        let doubled = bytes.len() >= 3
            && bytes[bytes.len() - 1] == bytes[bytes.len() - 2]
            && bytes[bytes.len() - 1].is_ascii_alphabetic();

        let candidates: [Option<String>; 3] = [
            if doubled {
                Some(stem[..stem.len() - 1].to_owned())
            } else {
                None
            },
            Some(stem.to_owned()),
            Some(format!("{stem}e")),
        ];
        for cand in candidates.into_iter().flatten() {
            if let Some(hit) = try_stem_with_mask(&cand, VERB_MASK) {
                return Some(make_fallback_analysis(hit, cand));
            }
        }
    }

    // 6. Progressive / gerund -ing (walking -> walk, making -> make, running -> run)
    if let Some(stem) = lower.strip_suffix("ing") {
        let bytes = stem.as_bytes();
        let doubled = bytes.len() >= 3
            && bytes[bytes.len() - 1] == bytes[bytes.len() - 2]
            && bytes[bytes.len() - 1].is_ascii_alphabetic();

        let candidates: [Option<String>; 3] = [
            if doubled {
                Some(stem[..stem.len() - 1].to_owned())
            } else {
                None
            },
            Some(stem.to_owned()),
            Some(format!("{stem}e")),
        ];
        for cand in candidates.into_iter().flatten() {
            if let Some(hit) = try_stem_with_mask(&cand, VERB_MASK) {
                return Some(make_fallback_analysis(hit, cand));
            }
        }
    }

    None
}
