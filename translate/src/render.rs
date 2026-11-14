//! Output formatters (`json`, `text`, `bilingual-md`, `bilingual-jsonl`) over `TranslatedDocument`.

use serde::Serialize;

use crate::schema::TranslatedDocument;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    JsonPretty,
    Text,
    BilingualMarkdown,
    BilingualJsonl,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "json" => Ok(Self::Json),
            "json-pretty" | "pretty" => Ok(Self::JsonPretty),
            "text" | "plain" | "md" => Ok(Self::Text),
            "bilingual-md" | "bilingual" => Ok(Self::BilingualMarkdown),
            "bilingual-jsonl" | "jsonl" => Ok(Self::BilingualJsonl),
            other => Err(format!(
                "unsupported format '{other}' (expected json, json-pretty, text, bilingual-md, or bilingual-jsonl)"
            )),
        }
    }
}

/// Render `doc` in the requested `OutputFormat`.
pub fn render_output(
    doc: &TranslatedDocument,
    source_text: &str,
    format: OutputFormat,
) -> anyhow::Result<String> {
    match format {
        OutputFormat::Json => Ok(serde_json::to_string(doc)?),
        OutputFormat::JsonPretty => Ok(serde_json::to_string_pretty(doc)?),
        OutputFormat::Text => Ok(doc.target_text.clone()),
        OutputFormat::BilingualMarkdown => Ok(render_bilingual_markdown(doc, source_text)),
        OutputFormat::BilingualJsonl => render_bilingual_jsonl(doc, source_text),
    }
}

/// Render paragraph-by-paragraph interleaved source (`> ...`) and target text in Markdown.
pub fn render_bilingual_markdown(doc: &TranslatedDocument, source_text: &str) -> String {
    let src_chars: Vec<char> = source_text.chars().collect();
    let tgt_chars: Vec<char> = doc.target_text.chars().collect();

    if doc.paragraphs.is_empty() {
        if source_text.trim().is_empty() {
            return doc.target_text.clone();
        }
        return format!("> {}\n\n{}", source_text.trim(), doc.target_text.trim());
    }

    let mut blocks = Vec::with_capacity(doc.paragraphs.len());
    for p in &doc.paragraphs {
        let src_slice: String = src_chars[p.source_span.start_char.min(src_chars.len())
            ..p.source_span.end_char.min(src_chars.len())]
            .iter()
            .collect();
        let tgt_slice: String = tgt_chars[p.target_span.start_char.min(tgt_chars.len())
            ..p.target_span.end_char.min(tgt_chars.len())]
            .iter()
            .collect();

        let quoted_src = src_slice
            .lines()
            .map(|l| format!("> {l}"))
            .collect::<Vec<_>>()
            .join("\n");
        blocks.push(format!("{quoted_src}\n\n{tgt_slice}"));
    }
    blocks.join("\n\n")
}

#[derive(Serialize)]
struct BilingualSentenceRow<'a> {
    sentence: usize,
    paragraph: usize,
    source_span: faberline_nlp::Span,
    target_span: faberline_nlp::Span,
    source_text: String,
    target_text: &'a str,
    track: crate::schema::TranslationTrack,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<crate::schema::Score>,
}

/// Render one JSON object per sentence with paired `source_text` and `target_text`.
pub fn render_bilingual_jsonl(
    doc: &TranslatedDocument,
    source_text: &str,
) -> anyhow::Result<String> {
    let src_chars: Vec<char> = source_text.chars().collect();
    let mut lines = Vec::with_capacity(doc.sentences.len());
    for s in &doc.sentences {
        let src_slice: String = src_chars[s.source_span.start_char.min(src_chars.len())
            ..s.source_span.end_char.min(src_chars.len())]
            .iter()
            .collect();
        let row = BilingualSentenceRow {
            sentence: s.sentence,
            paragraph: s.paragraph,
            source_span: s.source_span,
            target_span: s.target_span,
            source_text: src_slice,
            target_text: &s.target_text,
            track: s.track,
            confidence: s.confidence,
        };
        lines.push(serde_json::to_string(&row)?);
    }
    Ok(lines.join("\n"))
}
