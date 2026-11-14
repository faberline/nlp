//! Binary format definitions and UPOS/variant encodings shared between `build.rs` and `crate::lexicon`.
//!
//! This module uses `std` only so `build.rs` can include it directly via `#[path]`.

pub const MAGIC: &[u8; 8] = b"FNLPLX01";
pub const VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 80;
pub const BUCKET_COUNT: usize = 4096;
pub const NONE_U32: u32 = u32::MAX;
pub const NONE_U8: u8 = u8::MAX;

/// Language/script variant tag stored in each packed entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VariantCode {
    ZhHant = 0,
    ZhHans = 1,
    Zh = 2,
    EnUs = 3,
    EnGb = 4,
    En = 5,
    Ja = 6,
    Ko = 7,
}

impl VariantCode {
    #[allow(dead_code)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "zh-Hant" => Some(Self::ZhHant),
            "zh-Hans" => Some(Self::ZhHans),
            "zh" => Some(Self::Zh),
            "en-US" => Some(Self::EnUs),
            "en-GB" => Some(Self::EnGb),
            "en" => Some(Self::En),
            "ja" => Some(Self::Ja),
            "ko" => Some(Self::Ko),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::ZhHant),
            1 => Some(Self::ZhHans),
            2 => Some(Self::Zh),
            3 => Some(Self::EnUs),
            4 => Some(Self::EnGb),
            5 => Some(Self::En),
            6 => Some(Self::Ja),
            7 => Some(Self::Ko),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn lang_group(self) -> LangGroup {
        match self {
            Self::ZhHant | Self::ZhHans | Self::Zh => LangGroup::Zh,
            Self::EnUs | Self::EnGb | Self::En => LangGroup::En,
            Self::Ja => LangGroup::Ja,
            Self::Ko => LangGroup::Ko,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LangGroup {
    Zh = 0,
    En = 1,
    Ja = 2,
    Ko = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SemCode {
    None = 0,
    Person = 1,
    Location = 2,
    Organization = 3,
    Time = 4,
    Quantity = 5,
    Artifact = 6,
}

impl SemCode {
    #[allow(dead_code)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "person" => Some(Self::Person),
            "location" => Some(Self::Location),
            "organization" => Some(Self::Organization),
            "time" => Some(Self::Time),
            "quantity" => Some(Self::Quantity),
            "artifact" => Some(Self::Artifact),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Person,
            2 => Self::Location,
            3 => Self::Organization,
            4 => Self::Time,
            5 => Self::Quantity,
            6 => Self::Artifact,
            _ => Self::None,
        }
    }
}

/// UD v2 UPOS tag index (0..17).
pub fn upos_str_to_code(s: &str) -> Option<u8> {
    match s {
        "ADJ" => Some(0),
        "ADP" => Some(1),
        "ADV" => Some(2),
        "AUX" => Some(3),
        "CCONJ" => Some(4),
        "DET" => Some(5),
        "INTJ" => Some(6),
        "NOUN" => Some(7),
        "NUM" => Some(8),
        "PART" => Some(9),
        "PRON" => Some(10),
        "PROPN" => Some(11),
        "PUNCT" => Some(12),
        "SCONJ" => Some(13),
        "SYM" => Some(14),
        "VERB" => Some(15),
        "X" => Some(16),
        _ => None,
    }
}

/// Compute the monotonic 12-bit bucket index `0..4096` for UTF-8 bytes.
#[inline]
pub fn bucket_index(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        return 0;
    }
    let b0 = bytes[0] as usize;
    let b1 = if bytes.len() > 1 {
        (bytes[1] as usize) >> 4
    } else {
        0
    };
    (b0 << 4) | b1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum KoMorphKind {
    Josa = 0,
    Cop = 1,
    Xsv = 2,
    Xsa = 3,
    Eomi = 4,
}

impl KoMorphKind {
    #[allow(dead_code)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "josa" => Some(Self::Josa),
            "cop" => Some(Self::Cop),
            "xsv" => Some(Self::Xsv),
            "xsa" => Some(Self::Xsa),
            "eomi" => Some(Self::Eomi),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Josa),
            1 => Some(Self::Cop),
            2 => Some(Self::Xsv),
            3 => Some(Self::Xsa),
            4 => Some(Self::Eomi),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum KoBatchimCond {
    Any = 0,
    Batchim = 1,
    NoBatchim = 2,
    RieulOrNoBatchim = 3,
}

impl KoBatchimCond {
    #[allow(dead_code)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "any" => Some(Self::Any),
            "batchim" => Some(Self::Batchim),
            "no_batchim" => Some(Self::NoBatchim),
            "rieul_or_no_batchim" => Some(Self::RieulOrNoBatchim),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Batchim,
            2 => Self::NoBatchim,
            3 => Self::RieulOrNoBatchim,
            _ => Self::Any,
        }
    }
}
