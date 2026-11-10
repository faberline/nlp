//! Layer 1 of `nlp.doc.v1`. See `docs/doc-v1.md`.

mod entity;
mod opaque;
mod pos;
mod relation;
mod segment;
mod token;

use crate::Span;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA: &str = "nlp.doc.v1";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Options {
    /// Names for the name rule. Matched as exact text, leftmost-longest.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub names: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    Zh,
    En,
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
    pub relations: Vec<Relation>,
    pub unknown: Vec<Unknown>,
}

pub fn analyze_document(text: &str, options: &Options) -> Document {
    let c: Vec<char> = text.chars().collect();
    let found = opaque::find(&c);
    // Splitting runs on the masked text, so no boundary falls inside an opaque span.
    let m = opaque::mask(&c, &found);
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
        .filter(|f| !f.block && !matches!(f.item.kind, OpaqueKind::Heading | OpaqueKind::ListItem))
        .map(|f| (f.item.span, f.item.kind))
        .collect();
    // Supported sentences only, as (sentence index, span). Unsupported text gets no entities.
    let mut scanned = Vec::new();

    for paragraph in segment::paragraphs(&m) {
        let paragraph_index = seg.paragraphs.len();
        seg.paragraphs.push(Paragraph {
            span: paragraph,
            lang: segment::lang(&c, paragraph).0,
        });

        for span in segment::sentences(&m, paragraph) {
            let (lang, unsupported) = segment::lang(&c, span);
            let sentence_index = seg.sentences.len();
            if unsupported {
                unknown.push(Unknown {
                    layer: "segmentation".to_owned(),
                    span,
                    reason: Reason::UnsupportedLanguage,
                });
            } else {
                scanned.push((sentence_index, span));
            }
            seg.sentences.push(Sentence {
                span,
                lang,
                paragraph: paragraph_index,
            });

            for clause in segment::clauses(&m, span) {
                let lang = segment::lang(&c, clause).0;
                seg.clauses.push(Clause {
                    span: clause,
                    lang,
                    sentence: sentence_index,
                });
            }

            if unsupported {
                continue;
            }
            for piece in token::tokenize(&c, span, &inline) {
                let text: String = c[piece.span.start_char..piece.span.end_char]
                    .iter()
                    .collect();
                let lang = token::token_lang(&text);
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

    let (entities, entity_unknown) = entity::extract(&c, &found, &scanned, &options.names);
    unknown.extend(entity_unknown);
    let (relations, relation_unknown) = relation::relations(&c, &seg);
    unknown.extend(relation_unknown);
    unknown.sort_by(|a, b| {
        (a.span.start_char, a.span.end_char, a.layer.as_str()).cmp(&(
            b.span.start_char,
            b.span.end_char,
            b.layer.as_str(),
        ))
    });

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
        relations,
        unknown,
    }
}
