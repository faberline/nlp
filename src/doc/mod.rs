//! Layer 1 of `nlp.doc.v1`. See `docs/doc-v1.md`.

mod entity;
pub mod model;
mod opaque;
mod pos;
mod relation;
mod segment;
pub mod syntax;
mod t2s;
mod token;

use crate::Span;
pub(crate) use entity::NameRule;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(any(feature = "onnx", feature = "coreml"))]
pub use syntax::analyze_document_with_loaded_model;
pub use syntax::{
    DEPRELS, SYNTAX_MODEL_ID, SYNTAX_THRESHOLD, Score, SyntaxToken, analyze_document_with_model,
};

pub const SCHEMA: &str = "nlp.doc.v1";

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Options {
    /// Names for the name rule. Matched as exact text, leftmost-longest.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub names: Vec<String>,
    /// Mask generic web-novel editorial/scraper boilerplate lines and inline scraper tags.
    #[serde(default, skip_serializing_if = "is_false")]
    pub mask_editorial: bool,
    /// Project Traditional Han characters onto Simplified equivalents during Chinese word
    /// segmentation and split false coverb/demonstrative HMM merges while preserving exact
    /// original character offsets.
    #[serde(default, skip_serializing_if = "is_false")]
    pub normalize_traditional: bool,
    /// Extra domain vocabulary matched leftmost-longest before Chinese word segmentation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lexicon: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    Zh,
    En,
    Ja,
    Ko,
    Mixed,
    Und,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    NoModel,
    Ambiguous,
    LowConfidence,
    UnsupportedLanguage,
    NotApplicable,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unknown {
    pub layer: String,
    pub span: Span,
    pub reason: Reason,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Paragraph {
    pub span: Span,
    pub lang: Lang,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sentence {
    pub span: Span,
    pub lang: Lang,
    pub paragraph: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clause {
    pub span: Span,
    pub lang: Lang,
    pub sentence: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpaqueKind {
    Code,
    Formula,
    Heading,
    ListItem,
    Citation,
    Url,
    Emoji,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Opaque {
    pub kind: OpaqueKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segmentation {
    pub paragraphs: Vec<Paragraph>,
    pub sentences: Vec<Sentence>,
    pub clauses: Vec<Clause>,
    pub opaque: Vec<Opaque>,
}

/// Universal POS tags (UD v2 names).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Upos {
    Adj,
    Adp,
    Adv,
    Aux,
    Cconj,
    Det,
    Intj,
    Noun,
    Num,
    Part,
    Pron,
    Propn,
    Punct,
    Sconj,
    Sym,
    Verb,
    X,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
    pub span: Span,
    pub lang: Lang,
    pub sentence: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upos: Option<Upos>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    Sequence,
    Cause,
    Contrast,
    Parallel,
    Condition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationLevel {
    Clause,
    Sentence,
}

/// A discourse relation between two layer-1 units. See `docs/doc-v1.md`, section Relations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relation {
    pub kind: RelationKind,
    pub level: RelationLevel,
    /// The first connector in the text.
    pub connector: Span,
    /// The second connector of a pair, such as `所以` after `因為`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlate: Option<Span>,
    pub arg1: Span,
    pub arg2: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityKind {
    Number,
    Date,
    RelativeDate,
    Time,
    Quantity,
    Url,
    Email,
    Name,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub kind: EntityKind,
    pub span: Span,
    pub sentence: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    pub schema: String,
    pub text_chars: usize,
    pub text_sha256: String,
    pub options: Options,
    pub segmentation: Segmentation,
    pub tokens: Vec<Token>,
    pub entities: Vec<Entity>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub syntax: Vec<SyntaxToken>,
    pub relations: Vec<Relation>,
    pub unknown: Vec<Unknown>,
}

pub(crate) fn sort_unknown(unknown: &mut [Unknown]) {
    unknown.sort_by(|a, b| {
        (a.span.start_char, a.span.end_char, a.layer.as_str()).cmp(&(
            b.span.start_char,
            b.span.end_char,
            b.layer.as_str(),
        ))
    });
}

pub fn analyze_document(text: &str, options: &Options) -> Document {
    let c: Vec<char> = text.chars().collect();
    let found = opaque::find_with_options(&c, options.mask_editorial);
    // Splitting runs on the masked text, so no boundary falls inside an opaque span.
    let m = opaque::mask(&c, &found);
    let token_matcher = token::TokenMatcher::from_options(options);
    let mut seg = Segmentation {
        paragraphs: Vec::new(),
        sentences: Vec::new(),
        clauses: Vec::new(),
        opaque: found.iter().map(|f| f.item.clone()).collect(),
    };
    let mut unknown = Vec::new();
    let mut tokens = Vec::new();
    let inline: Vec<(Span, OpaqueKind)> = found
        .iter()
        .filter(|f| {
            !f.block
                && !f.silent_noise
                && !matches!(f.item.kind, OpaqueKind::Heading | OpaqueKind::ListItem)
        })
        .map(|f| (f.item.span, f.item.kind))
        .collect();
    let silent: Vec<Span> = found
        .iter()
        .filter(|f| !f.block && f.silent_noise)
        .map(|f| f.item.span)
        .collect();
    // Supported sentences only, as (sentence index, span). Unsupported text gets no entities.
    let mut scanned = Vec::new();

    for paragraph in segment::paragraphs(&m) {
        let paragraph_index = seg.paragraphs.len();
        let (para_lang, _) = segment::lang_in_context(&c, paragraph, false);
        let para_ja = para_lang == Lang::Ja || segment::has_ja_cue(&c, paragraph);
        seg.paragraphs.push(Paragraph {
            span: paragraph,
            lang: para_lang,
        });

        for span in segment::sentences(&m, paragraph) {
            let (lang, unsupported) = segment::lang_in_context(&c, span, para_ja);
            let sent_ja = lang == Lang::Ja
                || (para_ja && !segment::has_ja_or_ko(&c, span))
                || segment::has_ja_cue(&c, span);
            let sentence_index = seg.sentences.len();
            if unsupported {
                unknown.push(Unknown {
                    layer: "segmentation".to_owned(),
                    span,
                    reason: Reason::UnsupportedLanguage,
                });
            } else if !matches!(lang, Lang::Ja | Lang::Ko) && !segment::has_ja_or_ko(&c, span) {
                scanned.push((sentence_index, span));
            }
            seg.sentences.push(Sentence {
                span,
                lang,
                paragraph: paragraph_index,
            });

            for clause in segment::clauses_with_lang(&m, span, sent_ja) {
                let lang = segment::lang_in_context(&c, clause, sent_ja).0;
                seg.clauses.push(Clause {
                    span: clause,
                    lang,
                    sentence: sentence_index,
                });
            }

            if unsupported {
                continue;
            }
            for piece in
                token::tokenize_with_matcher(&c, span, sent_ja, &inline, &silent, &token_matcher)
            {
                let text: String = c[piece.span.start_char..piece.span.end_char]
                    .iter()
                    .collect();
                let lang = token::token_lang(&text, sent_ja);
                let undecided = |reason| Unknown {
                    layer: "pos".to_owned(),
                    span: piece.span,
                    reason,
                };
                let upos = match pos::tag(&text, lang, piece.opaque) {
                    pos::Tag::Decided(upos) => Some(upos),
                    pos::Tag::Ambiguous => {
                        unknown.push(undecided(Reason::Ambiguous));
                        None
                    }
                    pos::Tag::Missing => {
                        unknown.push(undecided(Reason::NoModel));
                        None
                    }
                    pos::Tag::UnsupportedLang => {
                        unknown.push(undecided(Reason::UnsupportedLanguage));
                        None
                    }
                };
                tokens.push(Token {
                    span: piece.span,
                    lang,
                    sentence: sentence_index,
                    upos,
                });
            }
        }
    }

    let (entities, entity_unknown) = entity::extract(
        &c,
        &found,
        &scanned,
        &tokens,
        &options.names,
        options.normalize_traditional,
    );
    unknown.extend(entity_unknown);
    let (relations, relation_unknown) = relation::relations(&c, &seg);
    unknown.extend(relation_unknown);
    sort_unknown(&mut unknown);

    Document {
        schema: SCHEMA.to_owned(),
        text_chars: c.len(),
        text_sha256: Sha256::digest(text.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
        options: options.clone(),
        segmentation: seg,
        tokens,
        entities,
        syntax: Vec::new(),
        relations,
        unknown,
    }
}

/// Layers 1 and 3 over one line of text, in byte ranges of that line. The narrative code reads
/// this instead of its own segmentation. `terms` are exact caller surfaces for the name rule.
pub(crate) struct LineView {
    /// Sentences, in order. Block opaque spans (heading, list item, fenced code, display
    /// formula) belong to no sentence.
    pub sentences: Vec<std::ops::Range<usize>>,
    /// Name hits, in order of position.
    pub names: Vec<std::ops::Range<usize>>,
    /// True when the line holds an opaque span of any kind.
    pub has_opaque: bool,
}

pub(crate) fn line_view(text: &str, terms: &entity::NameRule) -> LineView {
    let c: Vec<char> = text.chars().collect();
    let found = opaque::find(&c);
    let m = opaque::mask(&c, &found);
    let mut bytes = Vec::with_capacity(c.len() + 1);
    let mut at = 0;
    for ch in &c {
        bytes.push(at);
        at += ch.len_utf8();
    }
    bytes.push(at);
    let range = |s: Span| bytes[s.start_char]..bytes[s.end_char];
    let mut spans = Vec::new();
    for paragraph in segment::paragraphs(&m) {
        spans.extend(segment::sentences(&m, paragraph));
    }
    let names = entity::name_spans(&c, &found, &spans, terms)
        .into_iter()
        .map(range)
        .collect();
    LineView {
        sentences: spans.into_iter().map(range).collect(),
        names,
        has_opaque: !found.is_empty(),
    }
}

/// Layer 2 token ranges of one line, in bytes of that line. `None` when a sentence of the line
/// has an unsupported language (layer 2 leaves it untokenized) or when the line holds any opaque
/// span (heading, list item, code, formula, citation, URL, emoji): the caller keeps its own
/// tokenizer.
pub(crate) fn line_tokens(text: &str) -> Option<Vec<std::ops::Range<usize>>> {
    let c: Vec<char> = text.chars().collect();
    let found = opaque::find(&c);
    if !found.is_empty() {
        return None;
    }
    let m = opaque::mask(&c, &found);
    let inline: Vec<(Span, OpaqueKind)> = Vec::new();
    let mut bytes = Vec::with_capacity(c.len() + 1);
    let mut at = 0;
    for ch in &c {
        bytes.push(at);
        at += ch.len_utf8();
    }
    bytes.push(at);
    let mut out = Vec::new();
    for paragraph in segment::paragraphs(&m) {
        for sentence in segment::sentences(&m, paragraph) {
            if segment::lang(&c, sentence).1 || segment::has_ja_or_ko(&c, sentence) {
                return None;
            }
            out.extend(
                token::tokenize(&c, sentence, &inline)
                    .into_iter()
                    .map(|p| bytes[p.span.start_char]..bytes[p.span.end_char]),
            );
        }
    }
    Some(out)
}
