//! Layer 1 of `nlp.doc.v1`. See `docs/doc-v1.md`.

mod opaque;
mod segment;

use crate::Span;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCHEMA: &str = "nlp.doc.v1";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Options {}

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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    pub schema: String,
    pub text_chars: usize,
    pub text_sha256: String,
    pub options: Options,
    pub segmentation: Segmentation,
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
        }
    }

    Document {
        schema: SCHEMA.to_owned(),
        text_chars: c.len(),
        text_sha256: Sha256::digest(text.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
        options: options.clone(),
        segmentation: seg,
        unknown,
    }
}
