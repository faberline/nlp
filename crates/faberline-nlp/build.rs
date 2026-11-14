//! Pure-Rust compile-time lexicon compiler (`lexicon/data/*.tsv` -> `$OUT_DIR/lexicon_v1.bin`).

#[path = "src/lexicon/format.rs"]
mod format;

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use format::{
    BUCKET_COUNT, HEADER_SIZE, KoBatchimCond, KoMorphKind, MAGIC, NONE_U8, NONE_U32, SemCode,
    VERSION, VariantCode, bucket_index, upos_str_to_code,
};

const MAIN_TSVS: &[&str] = &[
    "lexicon/data/zh_hant.tsv",
    "lexicon/data/zh_hans.tsv",
    "lexicon/data/en.tsv",
    "lexicon/data/ja.tsv",
    "lexicon/data/ko.tsv",
];
const JA_INFLECT_TSV: &str = "lexicon/data/ja_inflect.tsv";
const KO_MORPH_TSV: &str = "lexicon/data/ko_morph.tsv";

#[derive(Clone)]
struct RawEntry {
    surface: String,
    variant: VariantCode,
    cost: u16,
    decided_upos: u8,
    upos_mask: u32,
    sem_class: SemCode,
    infl_id: u8,
    lemma: Option<String>,
    extra: Option<String>, // counterpart (zh/en) or decomp (ko)
}

struct JaInflectRow {
    infl_id: u8,
    stem_drop_chars: u8,
    conjugated_tail: String,
    cost_delta: u16,
    head_upos: u8,
    split_morphemes: String,
}

struct KoMorphRow {
    surface_suffix: String,
    kind: KoMorphKind,
    cond: KoBatchimCond,
    eojeol_upos_effect: u8,
    cost: u16,
    decomp: String,
}

struct StringPool {
    buf: Vec<u8>,
    map: HashMap<String, u32>,
}

impl StringPool {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(1024 * 1024),
            map: HashMap::with_capacity(128 * 1024),
        }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&off) = self.map.get(s) {
            return off;
        }
        let off = u32::try_from(self.buf.len()).expect("string pool exceeded 4 GiB");
        let len = u16::try_from(s.len()).expect("lexicon string exceeded 65535 bytes");
        self.buf.extend_from_slice(&len.to_le_bytes());
        self.buf.extend_from_slice(s.as_bytes());
        self.map.insert(s.to_owned(), off);
        off
    }
}

struct InflInterner {
    list: Vec<String>,
    map: HashMap<String, u8>,
}

impl InflInterner {
    fn new() -> Self {
        Self {
            list: Vec::new(),
            map: HashMap::new(),
        }
    }

    fn intern(&mut self, s: &str) -> u8 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        assert!(
            self.list.len() < 254,
            "too many distinct inflection paradigms (>254)"
        );
        self.list.push(s.to_owned());
        let id = self.list.len() as u8; // 1-based; 0 = none
        self.map.insert(s.to_owned(), id);
        id
    }
}

fn parse_upos_mask(mask_str: &str) -> u32 {
    if mask_str == "-" || mask_str.is_empty() {
        return 0;
    }
    let mut mask = 0u32;
    for part in mask_str.split('|') {
        if let Some(code) = upos_str_to_code(part.trim()) {
            mask |= 1u32 << code;
        } else {
            panic!("unknown UPOS tag in upos_mask: {part:?}");
        }
    }
    mask
}

fn load_main_tsv(
    path: &Path,
    allow_missing: bool,
    groups: &mut [Vec<RawEntry>; 4],
    variant_counts: &mut [u32; 8],
    infl_interner: &mut InflInterner,
) {
    if !path.exists() {
        if allow_missing {
            return;
        }
        panic!("missing required lexicon file: {}", path.display());
    }

    let file =
        File::open(path).unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    let reader = BufReader::new(file);

    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res.unwrap_or_else(|e| panic!("{}:{}: {e}", path.display(), line_no + 1));
        let line = line.trim_end_matches(&['\r', '\n'][..]);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 8 {
            panic!(
                "{}:{}: expected 8 tab-separated columns, got {}",
                path.display(),
                line_no + 1,
                cols.len()
            );
        }

        let surface = cols[0];
        let variant = VariantCode::from_str(cols[1]).unwrap_or_else(|| {
            panic!(
                "{}:{}: invalid variant {:?}",
                path.display(),
                line_no + 1,
                cols[1]
            )
        });
        let cost: u16 = cols[2].parse().unwrap_or_else(|e| {
            panic!(
                "{}:{}: invalid cost {:?}: {e}",
                path.display(),
                line_no + 1,
                cols[2]
            )
        });
        let upos_mask = parse_upos_mask(cols[4]);
        let mut decided_upos = if cols[3] == "-" || cols[3].is_empty() {
            NONE_U8
        } else {
            upos_str_to_code(cols[3]).unwrap_or_else(|| {
                panic!(
                    "{}:{}: invalid decided_upos {:?}",
                    path.display(),
                    line_no + 1,
                    cols[3]
                )
            })
        };
        if upos_mask.count_ones() > 1 {
            decided_upos = NONE_U8;
        }

        let lemma = if cols[5] == "-" || cols[5].is_empty() || cols[5] == surface {
            None
        } else {
            Some(cols[5].to_owned())
        };

        let counterpart = if cols[6] == "-" || cols[6].is_empty() {
            None
        } else {
            Some(cols[6].to_owned())
        };

        let mut sem_class = SemCode::None;
        let mut infl_id = 0u8;
        let mut decomp: Option<String> = None;

        if cols[7] != "-" && !cols[7].is_empty() {
            for kv in cols[7].split([',', ';']) {
                let kv = kv.trim();
                if let Some(val) = kv.strip_prefix("sem=") {
                    sem_class = SemCode::from_str(val).unwrap_or_else(|| {
                        panic!(
                            "{}:{}: invalid sem= value {:?}",
                            path.display(),
                            line_no + 1,
                            val
                        )
                    });
                } else if let Some(val) = kv.strip_prefix("infl=") {
                    infl_id = infl_interner.intern(val);
                } else if let Some(val) = kv.strip_prefix("decomp=") {
                    decomp = Some(val.to_owned());
                }
            }
        }

        let extra = counterpart.or(decomp);
        let group = variant.lang_group() as usize;
        variant_counts[variant as usize] += 1;
        groups[group].push(RawEntry {
            surface: surface.to_owned(),
            variant,
            cost,
            decided_upos,
            upos_mask,
            sem_class,
            infl_id,
            lemma,
            extra,
        });
    }
}

fn load_ja_inflect(
    path: &Path,
    allow_missing: bool,
    infl_interner: &mut InflInterner,
) -> Vec<JaInflectRow> {
    if !path.exists() {
        if allow_missing {
            return Vec::new();
        }
        panic!("missing required lexicon file: {}", path.display());
    }

    let file =
        File::open(path).unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    let reader = BufReader::new(file);
    let mut rows = Vec::new();

    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res.unwrap_or_else(|e| panic!("{}:{}: {e}", path.display(), line_no + 1));
        let line = line.trim_end_matches(&['\r', '\n'][..]);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 6 {
            panic!(
                "{}:{}: expected 6 columns, got {}",
                path.display(),
                line_no + 1,
                cols.len()
            );
        }
        let infl_id = infl_interner.intern(cols[0]);
        let stem_drop_chars: u8 = cols[1].parse().unwrap_or_else(|e| {
            panic!(
                "{}:{}: invalid stem_drop_chars: {e}",
                path.display(),
                line_no + 1
            )
        });
        let conjugated_tail = cols[2].to_owned();
        let cost_delta: u16 = cols[3].parse().unwrap_or_else(|e| {
            panic!(
                "{}:{}: invalid cost_delta: {e}",
                path.display(),
                line_no + 1
            )
        });
        let head_upos = upos_str_to_code(cols[4]).unwrap_or_else(|| {
            panic!(
                "{}:{}: invalid head_upos {:?}",
                path.display(),
                line_no + 1,
                cols[4]
            )
        });
        let split_morphemes = cols[5].to_owned();
        rows.push(JaInflectRow {
            infl_id,
            stem_drop_chars,
            conjugated_tail,
            cost_delta,
            head_upos,
            split_morphemes,
        });
    }

    rows
}

fn load_ko_morph(path: &Path, allow_missing: bool) -> Vec<KoMorphRow> {
    if !path.exists() {
        if allow_missing {
            return Vec::new();
        }
        panic!("missing required lexicon file: {}", path.display());
    }

    let file =
        File::open(path).unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    let reader = BufReader::new(file);
    let mut rows = Vec::new();

    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res.unwrap_or_else(|e| panic!("{}:{}: {e}", path.display(), line_no + 1));
        let line = line.trim_end_matches(&['\r', '\n'][..]);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 6 {
            panic!(
                "{}:{}: expected 6 columns, got {}",
                path.display(),
                line_no + 1,
                cols.len()
            );
        }
        let surface_suffix = cols[0].to_owned();
        let kind = KoMorphKind::from_str(cols[1]).unwrap_or_else(|| {
            panic!(
                "{}:{}: invalid kind {:?}",
                path.display(),
                line_no + 1,
                cols[1]
            )
        });
        let cond = KoBatchimCond::from_str(cols[2]).unwrap_or_else(|| {
            panic!(
                "{}:{}: invalid cond {:?}",
                path.display(),
                line_no + 1,
                cols[2]
            )
        });
        let cost: u16 = cols[3]
            .parse()
            .unwrap_or_else(|e| panic!("{}:{}: invalid cost: {e}", path.display(), line_no + 1));
        let eojeol_upos_effect = if cols[4] == "KEEP_STEM" || cols[4] == "-" {
            NONE_U8
        } else {
            upos_str_to_code(cols[4]).unwrap_or_else(|| {
                panic!(
                    "{}:{}: invalid eojeol_upos_effect {:?}",
                    path.display(),
                    line_no + 1,
                    cols[4]
                )
            })
        };
        let decomp = cols[5].to_owned();
        rows.push(KoMorphRow {
            surface_suffix,
            kind,
            cond,
            eojeol_upos_effect,
            cost,
            decomp,
        });
    }

    rows
}

fn write_group(out: &mut Vec<u8>, entries: &mut [RawEntry], pool: &mut StringPool) {
    // Deterministic sort by UTF-8 bytes of surface, then variant, then cost.
    entries.sort_by(|a, b| {
        a.surface
            .as_bytes()
            .cmp(b.surface.as_bytes())
            .then_with(|| (a.variant as u8).cmp(&(b.variant as u8)))
            .then_with(|| a.cost.cmp(&b.cost))
    });

    // Group consecutive entries with identical surface.
    struct KeyRec {
        str_off: u32,
        entry_start: u32,
        entry_count: u16,
        bucket: usize,
    }

    let mut keys: Vec<KeyRec> = Vec::new();
    let mut i = 0usize;
    while i < entries.len() {
        let start = i;
        let surf = &entries[i].surface;
        while i < entries.len() && entries[i].surface == *surf {
            i += 1;
        }
        let count = u16::try_from(i - start).expect("too many homographs for one surface");
        let str_off = pool.intern(surf);
        let bucket = bucket_index(surf.as_bytes());
        keys.push(KeyRec {
            str_off,
            entry_start: start as u32,
            entry_count: count,
            bucket,
        });
    }

    let num_keys = keys.len() as u32;
    let num_entries = entries.len() as u32;

    // Build monotonic 4097-entry bucket directory.
    let mut buckets = vec![num_keys; BUCKET_COUNT + 1];
    for (k_idx, k) in keys.iter().enumerate().rev() {
        buckets[k.bucket] = k_idx as u32;
    }
    for b in (0..BUCKET_COUNT).rev() {
        if buckets[b] > buckets[b + 1] {
            buckets[b] = buckets[b + 1];
        }
    }

    // Group header: num_keys (u32), num_entries (u32)
    out.extend_from_slice(&num_keys.to_le_bytes());
    out.extend_from_slice(&num_entries.to_le_bytes());

    // Bucket table: (BUCKET_COUNT + 1) * u32
    for b in buckets {
        out.extend_from_slice(&b.to_le_bytes());
    }

    // Key table: num_keys * 12 bytes (str_off: u32, entry_start: u32, entry_count: u16, pad: u16)
    for k in &keys {
        out.extend_from_slice(&k.str_off.to_le_bytes());
        out.extend_from_slice(&k.entry_start.to_le_bytes());
        out.extend_from_slice(&k.entry_count.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
    }

    // PackedEntry table: num_entries * 20 bytes
    for e in entries.iter() {
        let lemma_off = e
            .lemma
            .as_deref()
            .map(|s| pool.intern(s))
            .unwrap_or(NONE_U32);
        let extra_off = e
            .extra
            .as_deref()
            .map(|s| pool.intern(s))
            .unwrap_or(NONE_U32);
        out.push(e.variant as u8);
        out.push(e.decided_upos);
        out.push(e.sem_class as u8);
        out.push(e.infl_id);
        out.extend_from_slice(&e.cost.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&e.upos_mask.to_le_bytes());
        out.extend_from_slice(&lemma_off.to_le_bytes());
        out.extend_from_slice(&extra_off.to_le_bytes());
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/lexicon/format.rs");
    println!("cargo:rerun-if-env-changed=LEXICON_ALLOW_MISSING");
    for p in MAIN_TSVS {
        println!("cargo:rerun-if-changed={p}");
    }
    println!("cargo:rerun-if-changed={JA_INFLECT_TSV}");
    println!("cargo:rerun-if-changed={KO_MORPH_TSV}");

    let allow_missing = env::var("LEXICON_ALLOW_MISSING").as_deref() == Ok("1");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let mut groups: [Vec<RawEntry>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut variant_counts = [0u32; 8];
    let mut infl_interner = InflInterner::new();

    for rel in MAIN_TSVS {
        load_main_tsv(
            &manifest_dir.join(rel),
            allow_missing,
            &mut groups,
            &mut variant_counts,
            &mut infl_interner,
        );
    }
    let ja_inflect = load_ja_inflect(
        &manifest_dir.join(JA_INFLECT_TSV),
        allow_missing,
        &mut infl_interner,
    );
    let ko_morph = load_ko_morph(&manifest_dir.join(KO_MORPH_TSV), allow_missing);

    let mut pool = StringPool::new();
    let mut body = Vec::with_capacity(8 * 1024 * 1024);

    // Placeholder for 80-byte header
    body.resize(HEADER_SIZE, 0u8);

    let mut group_offsets = [0u32; 4];
    for (g_idx, entries) in groups.iter_mut().enumerate() {
        group_offsets[g_idx] = body.len() as u32;
        write_group(&mut body, entries, &mut pool);
    }

    // Write ja_inflect table
    let ja_inflect_offset = body.len() as u32;
    body.extend_from_slice(&(ja_inflect.len() as u32).to_le_bytes());
    for r in &ja_inflect {
        let tail_off = pool.intern(&r.conjugated_tail);
        let split_off = pool.intern(&r.split_morphemes);
        body.push(r.infl_id);
        body.push(r.stem_drop_chars);
        body.push(r.head_upos);
        body.push(0u8);
        body.extend_from_slice(&r.cost_delta.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes());
        body.extend_from_slice(&tail_off.to_le_bytes());
        body.extend_from_slice(&split_off.to_le_bytes());
    }

    // Write ko_morph table
    let ko_morph_offset = body.len() as u32;
    body.extend_from_slice(&(ko_morph.len() as u32).to_le_bytes());
    for r in &ko_morph {
        let suffix_off = pool.intern(&r.surface_suffix);
        let decomp_off = pool.intern(&r.decomp);
        body.push(r.kind as u8);
        body.push(r.cond as u8);
        body.push(r.eojeol_upos_effect);
        body.push(0u8);
        body.extend_from_slice(&r.cost.to_le_bytes());
        body.extend_from_slice(&0u16.to_le_bytes());
        body.extend_from_slice(&suffix_off.to_le_bytes());
        body.extend_from_slice(&decomp_off.to_le_bytes());
    }

    // Write infl_table (1-based IDs)
    let infl_table_offset = body.len() as u32;
    body.extend_from_slice(&(infl_interner.list.len() as u32).to_le_bytes());
    for name in &infl_interner.list {
        let off = pool.intern(name);
        body.extend_from_slice(&off.to_le_bytes());
    }

    // Write string pool
    let string_pool_offset = body.len() as u32;
    body.extend_from_slice(&pool.buf);

    // Fill 80-byte header
    body[0..8].copy_from_slice(MAGIC);
    body[8..12].copy_from_slice(&VERSION.to_le_bytes());
    body[12..16].copy_from_slice(&0u32.to_le_bytes());
    for (i, c) in variant_counts.iter().enumerate() {
        let off = 16 + i * 4;
        body[off..off + 4].copy_from_slice(&c.to_le_bytes());
    }
    for (i, g_off) in group_offsets.iter().enumerate() {
        let off = 48 + i * 4;
        body[off..off + 4].copy_from_slice(&g_off.to_le_bytes());
    }
    body[64..68].copy_from_slice(&ja_inflect_offset.to_le_bytes());
    body[68..72].copy_from_slice(&ko_morph_offset.to_le_bytes());
    body[72..76].copy_from_slice(&infl_table_offset.to_le_bytes());
    body[76..80].copy_from_slice(&string_pool_offset.to_le_bytes());

    let out_path = out_dir.join("lexicon_v1.bin");
    let file = File::create(&out_path)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", out_path.display()));
    let mut writer = BufWriter::new(file);
    writer.write_all(&body).unwrap();
    writer.flush().unwrap();
    let _ = fs::metadata(&out_path);
}
