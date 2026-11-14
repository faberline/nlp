//! Core data structures and JSON contracts for `nlp.translate.v1`.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use faberline_nlp::Span;
use faberline_nlp::doc::{Lang, OpaqueKind};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

pub const TRANSLATE_SCHEMA: &str = "nlp.translate.v1";
pub const TRANSLATE_GLOSSARY_SCHEMA: &str = "nlp.translate.glossary.v1";
pub const TRANSLATE_BATCH_RECEIPT_SCHEMA: &str = "nlp.translate.receipt.v1";
pub const SEQ2SEQ_MANIFEST_SCHEMA: &str = "nlp.models.seq2seq.v1";
pub const DEFAULT_TRANSLATE_MODEL_ID: &str = "doc-translate-v1";

fn is_false(b: &bool) -> bool {
    !*b
}

/// Supported target language and orthographic variants (aligned with `faberline_nlp::lexicon::Variant`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TargetVariant {
    #[serde(rename = "zh-Hant")]
    ZhHant,
    #[serde(rename = "zh-Hans")]
    ZhHans,
    #[serde(rename = "en-US")]
    EnUs,
    #[serde(rename = "en-GB")]
    EnGb,
    #[serde(rename = "ja")]
    Ja,
    #[serde(rename = "ko")]
    Ko,
}

impl TargetVariant {
    pub const ALL: [Self; 6] = [
        Self::ZhHant,
        Self::ZhHans,
        Self::EnUs,
        Self::EnGb,
        Self::Ja,
        Self::Ko,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ZhHant => "zh-Hant",
            Self::ZhHans => "zh-Hans",
            Self::EnUs => "en-US",
            Self::EnGb => "en-GB",
            Self::Ja => "ja",
            Self::Ko => "ko",
        }
    }

    pub fn lang_family(self) -> Lang {
        match self {
            Self::ZhHant | Self::ZhHans => Lang::Zh,
            Self::EnUs | Self::EnGb => Lang::En,
            Self::Ja => Lang::Ja,
            Self::Ko => Lang::Ko,
        }
    }

    pub fn is_zh(self) -> bool {
        matches!(self, Self::ZhHant | Self::ZhHans)
    }

    pub fn is_en(self) -> bool {
        matches!(self, Self::EnUs | Self::EnGb)
    }

    /// Control prefix token prepended to seq2seq encoder inputs.
    pub fn control_tag(self) -> &'static str {
        match self {
            Self::ZhHant => "<2zh_Hant>",
            Self::ZhHans => "<2zh_Hans>",
            Self::EnUs => "<2en_US>",
            Self::EnGb => "<2en_GB>",
            Self::Ja => "<2ja>",
            Self::Ko => "<2ko>",
        }
    }
}

impl fmt::Display for TargetVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for TargetVariant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "zh-Hant" | "zh-TW" | "zh-HK" | "zh_Hant" | "zh_tw" => Ok(Self::ZhHant),
            "zh-Hans" | "zh-CN" | "zh-SG" | "zh_Hans" | "zh_cn" => Ok(Self::ZhHans),
            "en-US" | "en_US" | "en-us" | "en" => Ok(Self::EnUs),
            "en-GB" | "en_GB" | "en-gb" | "en-UK" => Ok(Self::EnGb),
            "ja" | "ja-JP" | "jp" => Ok(Self::Ja),
            "ko" | "ko-KR" | "kr" => Ok(Self::Ko),
            other => Err(format!(
                "unsupported target variant '{other}' (expected zh-Hant, zh-Hans, en-US, en-GB, ja, or ko)"
            )),
        }
    }
}

/// Execution track chosen for a sentence or document.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranslationTrack {
    VariantRule,
    NeuralSeq2Seq,
    Passthrough,
    Hybrid,
}

/// Handling strategy applied to a `doc.segmentation.opaque` span.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpaquePolicy {
    VerbatimPassthrough,
    StructuralUnwrap,
    OmitEditorial,
}

/// Grammatical gender and number hint for pronoun injection in target languages with overt subjects.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PronounClass {
    MasculineSingular,
    FeminineSingular,
    NeuterSingular,
    AnimatePlural,
    FirstSingular,
    SecondSingular,
    #[default]
    Unspecified,
}

impl PronounClass {
    pub fn en_subject_pronoun(self) -> Option<&'static str> {
        match self {
            Self::MasculineSingular => Some("he"),
            Self::FeminineSingular => Some("she"),
            Self::NeuterSingular => Some("it"),
            Self::AnimatePlural => Some("they"),
            Self::FirstSingular => Some("I"),
            Self::SecondSingular => Some("you"),
            Self::Unspecified => None,
        }
    }

    pub fn control_code(self) -> Option<&'static str> {
        match self {
            Self::MasculineSingular => Some("3sg.m"),
            Self::FeminineSingular => Some("3sg.f"),
            Self::NeuterSingular => Some("3sg.n"),
            Self::AnimatePlural => Some("3pl"),
            Self::FirstSingular => Some("1sg"),
            Self::SecondSingular => Some("2sg"),
            Self::Unspecified => None,
        }
    }
}

/// Speech register / honorific level for dialogue attribution (`ja` keigo, `ko` speech levels, etc.).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechRegister {
    Plain,
    Polite,
    Formal,
    Archaic,
    #[default]
    Unspecified,
}

impl SpeechRegister {
    pub fn control_code(self) -> Option<&'static str> {
        match self {
            Self::Plain => Some("plain"),
            Self::Polite => Some("polite"),
            Self::Formal => Some("formal"),
            Self::Archaic => Some("archaic"),
            Self::Unspecified => None,
        }
    }
}

/// Quantized `[0.000, 1.000]` milli-score (`0..=1000`) for byte-identical cross-backend JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Score(pub u16);

impl Score {
    pub fn from_f32(v: f32) -> Self {
        let clamped = v.clamp(0.0, 1.0);
        Self((clamped * 1000.0).round() as u16)
    }

    pub fn milli(self) -> u16 {
        self.0
    }
}

impl Serialize for Score {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(f64::from(self.0) / 1000.0)
    }
}

impl<'de> Deserialize<'de> for Score {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v = f64::deserialize(deserializer)?;
        let clamped = v.clamp(0.0, 1.0);
        Ok(Self((clamped * 1000.0).round() as u16))
    }
}

/// Options controlling post-NLP translation and target rendering.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslateOptions {
    pub target: TargetVariant,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_hint: Option<TargetVariant>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub mask_editorial: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub omit_masked_editorial: bool,
    #[serde(default = "default_true")]
    pub localize_quotes: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub localize_entities: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub cjk_latin_spacing: bool,
    #[serde(default = "default_beam")]
    pub beam_size: usize,
    #[serde(default = "default_model_id")]
    pub model_id: String,
}

fn default_true() -> bool {
    true
}

fn default_beam() -> usize {
    1
}

fn default_model_id() -> String {
    DEFAULT_TRANSLATE_MODEL_ID.to_owned()
}

impl TranslateOptions {
    pub fn new(target: TargetVariant) -> Self {
        Self {
            target,
            source_hint: None,
            mask_editorial: false,
            omit_masked_editorial: false,
            localize_quotes: true,
            localize_entities: false,
            cjk_latin_spacing: false,
            beam_size: 1,
            model_id: DEFAULT_TRANSLATE_MODEL_ID.to_owned(),
        }
    }
}

/// Category of a locked glossary entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlossaryKind {
    Actor,
    Place,
    Object,
    DomainTerm,
}

/// One work- or domain-specific vocabulary entry with per-variant target renderings.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlossaryEntry {
    pub canonical: String,
    pub surfaces: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub targets: BTreeMap<TargetVariant, String>,
    #[serde(default, skip_serializing_if = "is_unspecified_pronoun")]
    pub pronoun_class: PronounClass,
    #[serde(default, skip_serializing_if = "is_unspecified_register")]
    pub speech_register: SpeechRegister,
}

fn is_unspecified_pronoun(p: &PronounClass) -> bool {
    *p == PronounClass::Unspecified
}

fn is_unspecified_register(r: &SpeechRegister) -> bool {
    *r == SpeechRegister::Unspecified
}

/// Multilingual glossary extending `narrative::WorkVocabulary` with target-variant renderings.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslationGlossary {
    #[serde(default = "default_glossary_schema")]
    pub schema: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actors: Vec<GlossaryEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub places: Vec<GlossaryEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<GlossaryEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub domain_terms: Vec<GlossaryEntry>,
}

fn default_glossary_schema() -> String {
    TRANSLATE_GLOSSARY_SCHEMA.to_owned()
}

impl TranslationGlossary {
    pub fn new() -> Self {
        Self {
            schema: TRANSLATE_GLOSSARY_SCHEMA.to_owned(),
            ..Self::default()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.actors.is_empty()
            && self.places.is_empty()
            && self.objects.is_empty()
            && self.domain_terms.is_empty()
    }

    /// Collect all surface forms to pass into `faberline_nlp::doc::Options` (`names` and `lexicon`).
    pub fn surface_terms(&self) -> (Vec<String>, Vec<String>) {
        let mut names = Vec::new();
        let mut lex = Vec::new();
        for e in self.actors.iter().chain(&self.places) {
            for s in &e.surfaces {
                if !s.is_empty() {
                    names.push(s.clone());
                    lex.push(s.clone());
                }
            }
        }
        for e in self.objects.iter().chain(&self.domain_terms) {
            for s in &e.surfaces {
                if !s.is_empty() {
                    lex.push(s.clone());
                }
            }
        }
        names.sort();
        names.dedup();
        lex.sort();
        lex.dedup();
        (names, lex)
    }

    pub fn find_actor(&self, canonical: &str) -> Option<&GlossaryEntry> {
        self.actors.iter().find(|a| a.canonical == canonical)
    }

    #[cfg(feature = "narrative")]
    pub fn to_work_vocabulary(&self) -> faberline_nlp_narrative::WorkVocabulary {
        use faberline_nlp_narrative::{AliasEntry, WorkVocabulary};
        let to_aliases = |entries: &[GlossaryEntry]| -> Vec<AliasEntry> {
            entries
                .iter()
                .flat_map(|e| {
                    e.surfaces.iter().map(move |s| AliasEntry {
                        surface: s.clone(),
                        canonical: e.canonical.clone(),
                    })
                })
                .collect()
        };
        let domain_terms = self
            .domain_terms
            .iter()
            .flat_map(|e| e.surfaces.iter().cloned())
            .collect();
        WorkVocabulary {
            actors: to_aliases(&self.actors),
            places: to_aliases(&self.places),
            objects: to_aliases(&self.objects),
            domain_terms,
        }
    }
}

/// Structured NLP/narrative conditioning evidence attached to a translated sentence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SentenceConditioning {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_pro_drop: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pronoun_hint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_register: Option<SpeechRegister>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factuality: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discourse_relation: Option<String>,
    #[serde(default = "default_one_usize", skip_serializing_if = "is_one_or_zero")]
    pub window_count: usize,
}

fn default_one_usize() -> usize {
    1
}

impl Default for SentenceConditioning {
    fn default() -> Self {
        Self {
            resolved_pro_drop: None,
            pronoun_hint: None,
            speaker: None,
            speech_register: None,
            factuality: None,
            discourse_relation: None,
            window_count: 1,
        }
    }
}

fn is_one_or_zero(n: &usize) -> bool {
    *n <= 1
}

impl SentenceConditioning {
    pub fn is_empty(&self) -> bool {
        self.resolved_pro_drop.is_none()
            && self.pronoun_hint.is_none()
            && self.speaker.is_none()
            && self.speech_register.is_none()
            && self.factuality.is_none()
            && self.discourse_relation.is_none()
            && self.window_count <= 1
    }
}

/// Paragraph-level bidirectional alignment between source and rendered target text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslatedParagraph {
    pub paragraph: usize,
    pub source_span: Span,
    pub target_span: Span,
}

/// Clause-level alignment record inside a sentence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClauseAlignment {
    pub clause: usize,
    pub sentence: usize,
    pub source_span: Span,
    pub target_span: Span,
}

/// Sentence-level bidirectional alignment and translation metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslatedSentence {
    pub sentence: usize,
    pub paragraph: usize,
    pub source_lang: Lang,
    pub track: TranslationTrack,
    pub source_span: Span,
    pub target_span: Span,
    pub target_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<Score>,
    #[serde(default, skip_serializing_if = "SentenceConditioning::is_empty")]
    pub conditioning: SentenceConditioning,
}

/// Provenance record for each glossary term locked and rendered in the output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlossaryHit {
    pub canonical: String,
    pub kind: GlossaryKind,
    pub sentence: Option<usize>,
    pub source_span: Span,
    pub target_span: Span,
}

/// Provenance record for each `opaque` span preserved or structurally unwrapped in the output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpaqueSpanRecord {
    pub kind: OpaqueKind,
    pub policy: OpaquePolicy,
    pub source_span: Span,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_span: Option<Span>,
}

/// Reasons a sentence or span requires review or could not be translated by a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranslateUnknownKind {
    NoModel,
    UnsupportedLanguage,
    LowConfidence,
    PlaceholderRepaired,
    PlaceholderDropped,
    AmbiguousVariant,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslateUnknown {
    pub kind: TranslateUnknownKind,
    pub source_span: Span,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sentence: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Complete `nlp.translate.v1` output document.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslatedDocument {
    pub schema: String,
    pub source_chars: usize,
    pub source_sha256: String,
    pub target_chars: usize,
    pub target_sha256: String,
    pub source_lang: Lang,
    pub target_variant: TargetVariant,
    pub track: TranslationTrack,
    pub target_text: String,
    pub paragraphs: Vec<TranslatedParagraph>,
    pub sentences: Vec<TranslatedSentence>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clause_alignments: Vec<ClauseAlignment>,
    pub glossary_hits: Vec<GlossaryHit>,
    pub opaque_spans: Vec<OpaqueSpanRecord>,
    pub unknown: Vec<TranslateUnknown>,
}

/// Corpus-level structural receipt written by `nlp_translate_batch`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslateManifestReceipt {
    pub schema: String,
    pub target_variant: TargetVariant,
    pub chapters_total: usize,
    pub chapters_translated: usize,
    pub chapters_resumed_cached: usize,
    pub total_source_chars: usize,
    pub total_target_chars: usize,
    pub total_sentences: usize,
    pub variant_rule_sentences: usize,
    pub neural_seq2seq_sentences: usize,
    pub passthrough_sentences: usize,
    pub glossary_hits: usize,
    pub verbatim_opaque_spans: usize,
    pub verbatim_opaque_byte_identical: usize,
    pub placeholder_repairs: usize,
    pub unknown_count: usize,
    pub chapters: Vec<ChapterTranslateReceipt>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChapterTranslateReceipt {
    pub chapter: u32,
    pub file: String,
    pub source_sha256: String,
    pub target_sha256: String,
    pub sentences: usize,
    pub glossary_hits: usize,
    pub unknown_count: usize,
}

pub fn sha256_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
