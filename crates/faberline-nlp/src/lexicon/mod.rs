//! First-party multilingual morphological and semantic lexicon (`faberline_nlp::lexicon`).
//!
//! All lexical tables are compiled at build time by `build.rs` from `lexicon/data/*.tsv`
//! into a single deterministic, little-endian binary blob embedded via `include_bytes!`.

pub mod en;
pub(crate) mod format;
pub mod ja;
pub mod ko;
pub mod zh;

use serde::{Deserialize, Serialize};

use crate::Span;
use crate::doc::Upos;
use format::{
    BUCKET_COUNT, HEADER_SIZE, KoBatchimCond, KoMorphKind, LangGroup, MAGIC, NONE_U8, NONE_U32,
    SemCode, VERSION, VariantCode, bucket_index,
};

static LEXICON_BIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/lexicon_v1.bin"));

/// Language/script variant of a lexicon entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Variant {
    #[serde(rename = "zh-Hant")]
    ZhHant,
    #[serde(rename = "zh-Hans")]
    ZhHans,
    #[serde(rename = "zh")]
    Zh,
    #[serde(rename = "en-US")]
    EnUs,
    #[serde(rename = "en-GB")]
    EnGb,
    #[serde(rename = "en")]
    En,
    #[serde(rename = "ja")]
    Ja,
    #[serde(rename = "ko")]
    Ko,
}

impl Variant {
    fn from_code(code: VariantCode) -> Self {
        match code {
            VariantCode::ZhHant => Self::ZhHant,
            VariantCode::ZhHans => Self::ZhHans,
            VariantCode::Zh => Self::Zh,
            VariantCode::EnUs => Self::EnUs,
            VariantCode::EnGb => Self::EnGb,
            VariantCode::En => Self::En,
            VariantCode::Ja => Self::Ja,
            VariantCode::Ko => Self::Ko,
        }
    }
}

/// Coarse semantic / entity-hint class from the first-party lexicon.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticClass {
    Person,
    Location,
    Organization,
    Time,
    Quantity,
    Artifact,
}

impl SemanticClass {
    fn from_code(code: SemCode) -> Option<Self> {
        match code {
            SemCode::None => None,
            SemCode::Person => Some(Self::Person),
            SemCode::Location => Some(Self::Location),
            SemCode::Organization => Some(Self::Organization),
            SemCode::Time => Some(Self::Time),
            SemCode::Quantity => Some(Self::Quantity),
            SemCode::Artifact => Some(Self::Artifact),
        }
    }
}

/// A sub-token morpheme (e.g., Korean stem + josa/eomi, or Japanese verb stem + auxiliary).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Morpheme {
    pub span: Span,
    pub surface: String,
    pub lemma: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upos: Option<Upos>,
}

/// Morphological and semantic analysis of a single token or eojeol.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexTokenAnalysis {
    pub decided_upos: Option<Upos>,
    pub upos_mask: u32,
    pub upos_candidates: Vec<Upos>,
    pub lemma: Option<String>,
    pub canonical_variant: Option<String>,
    pub sem_class: Option<SemanticClass>,
    pub morphemes: Vec<Morpheme>,
}

/// Merged surface-level lexicon lookup result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LexLookup {
    pub decided_upos: Option<Upos>,
    pub upos_mask: u32,
    pub upos_candidates: Vec<Upos>,
    pub cost: u16,
    pub variant: Variant,
    pub lemma: Option<String>,
    pub counterpart: Option<String>,
    pub sem_class: Option<SemanticClass>,
    pub infl: Option<&'static str>,
    pub decomp: Option<&'static str>,
}

/// Summary statistics for the embedded binary lexicon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LexiconStats {
    pub total_bytes: usize,
    pub zh_hant: usize,
    pub zh_hans: usize,
    pub zh: usize,
    pub en_us: usize,
    pub en_gb: usize,
    pub en: usize,
    pub ja: usize,
    pub ko: usize,
    pub ja_inflect_rules: usize,
    pub ko_morph_rules: usize,
}

impl LexiconStats {
    pub fn total_entries(&self) -> usize {
        self.zh_hant
            + self.zh_hans
            + self.zh
            + self.en_us
            + self.en_gb
            + self.en
            + self.ja
            + self.ko
    }
}

#[inline]
pub(crate) fn code_to_upos(code: u8) -> Option<Upos> {
    match code {
        0 => Some(Upos::Adj),
        1 => Some(Upos::Adp),
        2 => Some(Upos::Adv),
        3 => Some(Upos::Aux),
        4 => Some(Upos::Cconj),
        5 => Some(Upos::Det),
        6 => Some(Upos::Intj),
        7 => Some(Upos::Noun),
        8 => Some(Upos::Num),
        9 => Some(Upos::Part),
        10 => Some(Upos::Pron),
        11 => Some(Upos::Propn),
        12 => Some(Upos::Punct),
        13 => Some(Upos::Sconj),
        14 => Some(Upos::Sym),
        15 => Some(Upos::Verb),
        16 => Some(Upos::X),
        _ => None,
    }
}

pub(crate) fn mask_to_candidates(mask: u32) -> Vec<Upos> {
    let mut out = Vec::with_capacity(mask.count_ones() as usize);
    for bit in 0..17u8 {
        if (mask & (1u32 << bit)) != 0
            && let Some(u) = code_to_upos(bit)
        {
            out.push(u);
        }
    }
    out
}

pub(crate) fn decided_from_mask(mask: u32) -> Option<Upos> {
    if mask.count_ones() == 1 {
        code_to_upos(mask.trailing_zeros() as u8)
    } else {
        None
    }
}

#[inline]
fn read_u16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

#[inline]
fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawEntryRef {
    pub variant: VariantCode,
    pub decided_upos: u8,
    pub sem_class: SemCode,
    pub infl_id: u8,
    pub cost: u16,
    pub upos_mask: u32,
    pub lemma_off: u32,
    pub extra_off: u32,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct JaInflectRule {
    pub infl_id: u8,
    pub stem_drop_chars: u8,
    pub head_upos: u8,
    pub cost_delta: u16,
    pub conjugated_tail: &'static str,
    pub split_morphemes: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct KoMorphRule {
    pub kind: KoMorphKind,
    pub cond: KoBatchimCond,
    pub eojeol_upos_effect: u8,
    pub cost: u16,
    pub surface_suffix: &'static str,
    pub decomp: &'static str,
}

pub(crate) struct LexiconDb {
    bin: &'static [u8],
    variant_counts: [u32; 8],
    group_offsets: [usize; 4],
    ja_inflect_offset: usize,
    ko_morph_offset: usize,
    infl_table_offset: usize,
    string_pool_offset: usize,
}

impl LexiconDb {
    fn from_bytes(bin: &'static [u8]) -> Self {
        assert!(
            bin.len() >= HEADER_SIZE,
            "corrupt lexicon_v1.bin: too short"
        );
        assert_eq!(&bin[0..8], MAGIC, "corrupt lexicon_v1.bin: bad magic");
        assert_eq!(
            read_u32(bin, 8),
            VERSION,
            "unsupported lexicon_v1.bin version"
        );

        let mut variant_counts = [0u32; 8];
        for (i, c) in variant_counts.iter_mut().enumerate() {
            *c = read_u32(bin, 16 + i * 4);
        }
        let mut group_offsets = [0usize; 4];
        for (i, g) in group_offsets.iter_mut().enumerate() {
            *g = read_u32(bin, 48 + i * 4) as usize;
        }
        let ja_inflect_offset = read_u32(bin, 64) as usize;
        let ko_morph_offset = read_u32(bin, 68) as usize;
        let infl_table_offset = read_u32(bin, 72) as usize;
        let string_pool_offset = read_u32(bin, 76) as usize;

        Self {
            bin,
            variant_counts,
            group_offsets,
            ja_inflect_offset,
            ko_morph_offset,
            infl_table_offset,
            string_pool_offset,
        }
    }

    #[inline]
    pub(crate) fn str_at(&self, off: u32) -> Option<&'static str> {
        if off == NONE_U32 {
            return None;
        }
        let pos = self.string_pool_offset + off as usize;
        let len = read_u16(self.bin, pos) as usize;
        let bytes = &self.bin[pos + 2..pos + 2 + len];
        std::str::from_utf8(bytes).ok()
    }

    #[inline]
    pub(crate) fn infl_str(&self, infl_id: u8) -> Option<&'static str> {
        if infl_id == 0 {
            return None;
        }
        let count = read_u32(self.bin, self.infl_table_offset) as usize;
        let idx = (infl_id as usize) - 1;
        if idx >= count {
            return None;
        }
        let off = read_u32(self.bin, self.infl_table_offset + 4 + idx * 4);
        self.str_at(off)
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn infl_id_by_name(&self, name: &str) -> Option<u8> {
        let count = read_u32(self.bin, self.infl_table_offset) as usize;
        for idx in 0..count {
            let off = read_u32(self.bin, self.infl_table_offset + 4 + idx * 4);
            if self.str_at(off) == Some(name) {
                return Some((idx + 1) as u8);
            }
        }
        None
    }

    fn group_view(&self, group: LangGroup) -> GroupView<'_> {
        let base = self.group_offsets[group as usize];
        let num_keys = read_u32(self.bin, base) as usize;
        let num_entries = read_u32(self.bin, base + 4) as usize;
        let buckets_off = base + 8;
        let keys_off = buckets_off + (BUCKET_COUNT + 1) * 4;
        let entries_off = keys_off + num_keys * 12;
        GroupView {
            db: self,
            num_keys,
            _num_entries: num_entries,
            buckets_off,
            keys_off,
            entries_off,
        }
    }

    pub(crate) fn lookup_raw(&self, group: LangGroup, surface: &str) -> Vec<RawEntryRef> {
        let gv = self.group_view(group);
        gv.lookup(surface)
    }

    pub(crate) fn common_prefix_search<F>(
        &self,
        group: LangGroup,
        text: &str,
        max_chars: usize,
        mut f: F,
    ) where
        F: FnMut(usize, usize, &[RawEntryRef]),
    {
        let gv = self.group_view(group);
        if gv.num_keys == 0 || text.is_empty() {
            return;
        }
        let mut prev_lo = 0usize;
        let mut scratch = Vec::with_capacity(4);

        for (char_idx, (byte_idx, ch)) in text.char_indices().enumerate() {
            let char_len = char_idx + 1;
            if char_len > max_chars {
                break;
            }
            let byte_end = byte_idx + ch.len_utf8();
            let prefix = &text[..byte_end];
            let bucket = bucket_index(prefix.as_bytes());
            let b_lo = gv.bucket(bucket);
            let b_hi = gv.bucket(bucket + 1);
            let lo = prev_lo.max(b_lo);
            let hi = b_hi;

            if lo >= hi {
                if lo >= gv.num_keys || !gv.key_str(lo).starts_with(prefix) {
                    break;
                }
                prev_lo = lo;
                continue;
            }

            match gv.binary_search_range(lo, hi, prefix.as_bytes()) {
                Ok(idx) => {
                    gv.fill_entries(idx, &mut scratch);
                    f(byte_end, char_len, &scratch);
                    prev_lo = idx + 1;
                    if prev_lo >= gv.num_keys || !gv.key_str(prev_lo).starts_with(prefix) {
                        break;
                    }
                }
                Err(idx) => {
                    prev_lo = idx;
                    if idx >= gv.num_keys || !gv.key_str(idx).starts_with(prefix) {
                        break;
                    }
                }
            }
        }
    }

    pub(crate) fn ja_inflect_rules(&self) -> impl Iterator<Item = JaInflectRule> + '_ {
        let count = read_u32(self.bin, self.ja_inflect_offset) as usize;
        let base = self.ja_inflect_offset + 4;
        (0..count).map(move |i| {
            let off = base + i * 16;
            JaInflectRule {
                infl_id: self.bin[off],
                stem_drop_chars: self.bin[off + 1],
                head_upos: self.bin[off + 2],
                cost_delta: read_u16(self.bin, off + 4),
                conjugated_tail: self.str_at(read_u32(self.bin, off + 8)).unwrap_or(""),
                split_morphemes: self.str_at(read_u32(self.bin, off + 12)).unwrap_or(""),
            }
        })
    }

    pub(crate) fn ko_morph_rules(&self) -> impl Iterator<Item = KoMorphRule> + '_ {
        let count = read_u32(self.bin, self.ko_morph_offset) as usize;
        let base = self.ko_morph_offset + 4;
        (0..count).filter_map(move |i| {
            let off = base + i * 16;
            let kind = KoMorphKind::from_u8(self.bin[off])?;
            let cond = KoBatchimCond::from_u8(self.bin[off + 1]);
            let eojeol_upos_effect = self.bin[off + 2];
            let cost = read_u16(self.bin, off + 4);
            let surface_suffix = self.str_at(read_u32(self.bin, off + 8)).unwrap_or("");
            let decomp = self.str_at(read_u32(self.bin, off + 12)).unwrap_or("");
            Some(KoMorphRule {
                kind,
                cond,
                eojeol_upos_effect,
                cost,
                surface_suffix,
                decomp,
            })
        })
    }
}

struct GroupView<'a> {
    db: &'a LexiconDb,
    num_keys: usize,
    _num_entries: usize,
    buckets_off: usize,
    keys_off: usize,
    entries_off: usize,
}

impl<'a> GroupView<'a> {
    #[inline]
    fn bucket(&self, idx: usize) -> usize {
        read_u32(self.db.bin, self.buckets_off + idx * 4) as usize
    }

    #[inline]
    fn key_str(&self, key_idx: usize) -> &'static str {
        let str_off = read_u32(self.db.bin, self.keys_off + key_idx * 12);
        self.db.str_at(str_off).unwrap_or("")
    }

    fn binary_search_range(
        &self,
        mut lo: usize,
        mut hi: usize,
        target: &[u8],
    ) -> Result<usize, usize> {
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let s = self.key_str(mid).as_bytes();
            match s.cmp(target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Err(lo)
    }

    fn fill_entries(&self, key_idx: usize, out: &mut Vec<RawEntryRef>) {
        out.clear();
        let key_off = self.keys_off + key_idx * 12;
        let entry_start = read_u32(self.db.bin, key_off + 4) as usize;
        let entry_count = read_u16(self.db.bin, key_off + 8) as usize;
        for i in 0..entry_count {
            let e_off = self.entries_off + (entry_start + i) * 20;
            let bin = self.db.bin;
            let Some(variant) = VariantCode::from_u8(bin[e_off]) else {
                continue;
            };
            out.push(RawEntryRef {
                variant,
                decided_upos: bin[e_off + 1],
                sem_class: SemCode::from_u8(bin[e_off + 2]),
                infl_id: bin[e_off + 3],
                cost: read_u16(bin, e_off + 4),
                upos_mask: read_u32(bin, e_off + 8),
                lemma_off: read_u32(bin, e_off + 12),
                extra_off: read_u32(bin, e_off + 16),
            });
        }
    }

    fn lookup(&self, surface: &str) -> Vec<RawEntryRef> {
        if self.num_keys == 0 || surface.is_empty() {
            return Vec::new();
        }
        let bucket = bucket_index(surface.as_bytes());
        let lo = self.bucket(bucket);
        let hi = self.bucket(bucket + 1);
        let mut out = Vec::new();
        if let Ok(idx) = self.binary_search_range(lo, hi, surface.as_bytes()) {
            self.fill_entries(idx, &mut out);
        }
        out
    }
}

pub(crate) fn db() -> &'static LexiconDb {
    static DB: std::sync::OnceLock<LexiconDb> = std::sync::OnceLock::new();
    DB.get_or_init(|| LexiconDb::from_bytes(LEXICON_BIN))
}

pub(crate) fn merge_entries(group: LangGroup, entries: &[RawEntryRef]) -> Option<LexLookup> {
    if entries.is_empty() {
        return None;
    }
    let db = db();
    let mut combined_mask = 0u32;
    let mut best_idx = 0usize;
    let mut best_cost = u16::MAX;
    let mut lemma_off = NONE_U32;
    let mut extra_off = NONE_U32;
    let mut sem_class: Option<SemanticClass> = None;
    let mut sem_conflict = false;
    let mut infl_id = 0u8;

    for (i, e) in entries.iter().enumerate() {
        combined_mask |= e.upos_mask;
        if e.cost < best_cost {
            best_cost = e.cost;
            best_idx = i;
        }
        if lemma_off == NONE_U32 && e.lemma_off != NONE_U32 {
            lemma_off = e.lemma_off;
        }
        if extra_off == NONE_U32 && e.extra_off != NONE_U32 {
            extra_off = e.extra_off;
        }
        if infl_id == 0 && e.infl_id != 0 {
            infl_id = e.infl_id;
        }
        if let Some(sc) = SemanticClass::from_code(e.sem_class) {
            match sem_class {
                None => sem_class = Some(sc),
                Some(prev) if prev != sc => sem_conflict = true,
                _ => {}
            }
        }
    }

    // Never guess: decided_upos is Some only if the combined upos_mask has exactly 1 bit
    // AND every entry that specifies a decided_upos agrees on it.
    let decided_upos = if combined_mask.count_ones() == 1 {
        let candidate = decided_from_mask(combined_mask);
        let all_agree = entries
            .iter()
            .all(|e| e.decided_upos == NONE_U8 || code_to_upos(e.decided_upos) == candidate);
        if all_agree { candidate } else { None }
    } else {
        None
    };

    let best = &entries[best_idx];
    let extra = db.str_at(extra_off);
    let (counterpart, decomp) = match group {
        LangGroup::Ko => (None, extra),
        _ => (extra.map(str::to_owned), None),
    };

    Some(LexLookup {
        decided_upos,
        upos_mask: combined_mask,
        upos_candidates: mask_to_candidates(combined_mask),
        cost: best.cost,
        variant: Variant::from_code(best.variant),
        lemma: db.str_at(lemma_off).map(str::to_owned),
        counterpart,
        sem_class: if sem_conflict { None } else { sem_class },
        infl: db.infl_str(infl_id),
        decomp,
    })
}

/// Return entry counts and embedded binary size for the compiled lexicon.
pub fn stats() -> LexiconStats {
    let db = db();
    let ja_inflect_rules = read_u32(db.bin, db.ja_inflect_offset) as usize;
    let ko_morph_rules = read_u32(db.bin, db.ko_morph_offset) as usize;
    LexiconStats {
        total_bytes: db.bin.len(),
        zh_hant: db.variant_counts[VariantCode::ZhHant as usize] as usize,
        zh_hans: db.variant_counts[VariantCode::ZhHans as usize] as usize,
        zh: db.variant_counts[VariantCode::Zh as usize] as usize,
        en_us: db.variant_counts[VariantCode::EnUs as usize] as usize,
        en_gb: db.variant_counts[VariantCode::EnGb as usize] as usize,
        en: db.variant_counts[VariantCode::En as usize] as usize,
        ja: db.variant_counts[VariantCode::Ja as usize] as usize,
        ko: db.variant_counts[VariantCode::Ko as usize] as usize,
        ja_inflect_rules,
        ko_morph_rules,
    }
}

/// Look up a Chinese word across `zh-Hant`, `zh-Hans`, and `zh`.
pub fn lookup_zh(surface: &str) -> Option<LexLookup> {
    let entries = db().lookup_raw(LangGroup::Zh, surface);
    merge_entries(LangGroup::Zh, &entries)
}

/// Look up an English word across `en-US`, `en-GB`, and `en` (exact or ASCII-lowercased).
pub fn lookup_en(surface: &str) -> Option<LexLookup> {
    let db = db();
    let entries = db.lookup_raw(LangGroup::En, surface);
    if !entries.is_empty() {
        return merge_entries(LangGroup::En, &entries);
    }
    let lower = surface.to_ascii_lowercase();
    if lower != surface {
        let lower_entries = db.lookup_raw(LangGroup::En, &lower);
        return merge_entries(LangGroup::En, &lower_entries);
    }
    None
}

/// Look up a Japanese word in `ja.tsv`.
pub fn lookup_ja(surface: &str) -> Option<LexLookup> {
    let entries = db().lookup_raw(LangGroup::Ja, surface);
    merge_entries(LangGroup::Ja, &entries)
}

/// Look up a Korean word or pre-analyzed form in `ko.tsv`.
pub fn lookup_ko(surface: &str) -> Option<LexLookup> {
    let entries = db().lookup_raw(LangGroup::Ko, surface);
    merge_entries(LangGroup::Ko, &entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexicon_stats_and_all_variants() {
        let s = stats();
        assert!(s.total_bytes > 1_000_000, "bin bytes={}", s.total_bytes);
        assert!(s.zh_hant > 50_000, "zh_hant={}", s.zh_hant);
        assert!(s.zh_hans > 50_000, "zh_hans={}", s.zh_hans);
        assert!(s.en_us > 5_000, "en_us={}", s.en_us);
        assert!(s.en_gb > 5_000, "en_gb={}", s.en_gb);
        assert!(s.en > 30_000, "en={}", s.en);
        assert!(s.ja > 50_000, "ja={}", s.ja);
        assert!(s.ko > 50_000, "ko={}", s.ko);
        assert!(
            s.ja_inflect_rules > 400,
            "ja_inflect={}",
            s.ja_inflect_rules
        );
        assert!(s.ko_morph_rules > 100, "ko_morph={}", s.ko_morph_rules);

        // zh-Hant & zh-Hans
        let hit = lookup_zh("軟體").expect("軟體 should be in zh_hant");
        assert_eq!(hit.counterpart.as_deref(), Some("软件"));
        let res = zh::analyze_zh_token("研究");
        assert_eq!(res.decided_upos, None, "研究 is NOUN|VERB");
        assert!(res.upos_candidates.contains(&Upos::Noun));
        assert!(res.upos_candidates.contains(&Upos::Verb));

        let zh_text: Vec<char> = "台灣的軟體產業發展".chars().collect();
        let spans = zh::segment_zh(&zh_text, 0, &[]);
        let segs: Vec<String> = spans
            .iter()
            .map(|sp| zh_text[sp.start_char..sp.end_char].iter().collect())
            .collect();
        assert!(segs.contains(&"軟體".to_owned()), "got {segs:?}");

        let extra = vec!["產業發展".to_owned()];
        let spans_extra = zh::segment_zh(&zh_text, 5, &extra);
        let segs_extra: Vec<String> = spans_extra
            .iter()
            .map(|sp| {
                zh_text[(sp.start_char - 5)..(sp.end_char - 5)]
                    .iter()
                    .collect()
            })
            .collect();
        assert!(
            segs_extra.contains(&"產業發展".to_owned()),
            "got {segs_extra:?}"
        );

        // en-US & en-GB
        let went = en::analyze_en_token("Went");
        assert_eq!(went.lemma.as_deref(), Some("go"));
        assert_eq!(went.decided_upos, Some(Upos::Verb));

        let colour = en::analyze_en_token("colour");
        assert_eq!(colour.canonical_variant.as_deref(), Some("color"));

        let analyze = en::analyze_en_token("analyze");
        assert_eq!(analyze.canonical_variant.as_deref(), Some("analyse"));

        let studies = en::analyze_en_token("studies");
        assert_eq!(studies.lemma.as_deref(), Some("study"));

        let stopped = en::analyze_en_token("stopped");
        assert_eq!(stopped.lemma.as_deref(), Some("stop"));

        let walking = en::analyze_en_token("walking");
        assert_eq!(walking.lemma.as_deref(), Some("walk"));

        let sing = en::analyze_en_token("sing");
        assert_ne!(sing.lemma.as_deref(), Some("s"));

        let open = en::analyze_en_token("open");
        assert_eq!(open.decided_upos, None);
        assert!(open.upos_candidates.len() > 1);

        // ja
        type JaWordRow = (String, Option<Upos>, Option<String>, Option<SemanticClass>);
        let ja_chars: Vec<char> = "東京駅で3冊の本を買った。".chars().collect();
        let ja_segs = ja::segment_ja(&ja_chars, 0);
        let ja_words: Vec<JaWordRow> = ja_segs
            .iter()
            .map(|seg| {
                (
                    ja_chars[seg.span.start_char..seg.span.end_char]
                        .iter()
                        .collect(),
                    seg.decided_upos,
                    seg.lemma.clone(),
                    seg.sem_class,
                )
            })
            .collect();
        assert!(
            ja_words.iter().any(|(w, u, _, sc)| w == "東京駅"
                && *u == Some(Upos::Propn)
                && *sc == Some(SemanticClass::Location)),
            "ja_words={ja_words:?}"
        );
        assert!(
            ja_words.iter().any(|(w, u, l, _)| w == "買っ"
                && *u == Some(Upos::Verb)
                && l.as_deref() == Some("買う")),
            "ja_words={ja_words:?}"
        );
        assert!(
            ja_words
                .iter()
                .any(|(w, u, _, _)| w == "た" && *u == Some(Upos::Aux)),
            "ja_words={ja_words:?}"
        );

        let ja_kat: Vec<char> = "スマートフォンを使う".chars().collect();
        let ja_kat_segs = ja::segment_ja(&ja_kat, 0);
        let ja_kat_words: Vec<String> = ja_kat_segs
            .iter()
            .map(|seg| {
                ja_kat[seg.span.start_char..seg.span.end_char]
                    .iter()
                    .collect()
            })
            .collect();
        assert!(
            ja_kat_words.contains(&"スマートフォン".to_owned()),
            "ja_kat_words={ja_kat_words:?}"
        );

        // ko
        for (word, expected_upos) in [
            ("서울역에서", Some(Upos::Propn)),
            ("책을", Some(Upos::Noun)),
            ("샀다", Some(Upos::Verb)),
            ("회의가", Some(Upos::Noun)),
            ("열렸다", Some(Upos::Verb)),
        ] {
            let chars: Vec<char> = word.chars().collect();
            let a = ko::analyze_ko_eojeol(&chars, 10);
            assert_eq!(a.decided_upos, expected_upos, "word={word}, analysis={a:?}");
            assert!(!a.morphemes.is_empty(), "expected morphemes for {word}");
            assert_eq!(a.morphemes.first().unwrap().span.start_char, 10);
            assert_eq!(a.morphemes.last().unwrap().span.end_char, 10 + chars.len());
            for w in a.morphemes.windows(2) {
                assert_eq!(w[0].span.end_char, w[1].span.start_char);
            }
        }
    }
}
