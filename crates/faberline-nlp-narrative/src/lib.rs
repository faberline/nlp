//! Source-linked narrative event, dialogue, and actor-chain extraction over `nlp.doc.v1`.

pub mod chain;
pub mod dialogue;
pub mod event;
pub mod graph;
pub mod schema;

pub use graph::{
    BookEventRef, BookTimelineGraph, ChapterGraphBuild, ChapterTimelineDag,
    build_chapter_dag_from_narrative, build_chapter_graph,
};
pub use schema::{
    ActorMention, AliasEntry, ArgumentBasis, BOOK_RECEIPT_SCHEMA, CHAPTER_NARRATIVE_SCHEMA,
    CandidateEvent, ChapterNarrative, ChronologyOrder, ContextZone, DialogueSpan, EdgeEvidenceKind,
    EdgeKind, EdgeStatus, EventArgument, EventRole, Factuality, LocatedSpan, MentionKind,
    NARRATIVE_SCHEMA, NarrativeManifestReceipt, NarrativeUnknown, NarrativeUnknownKind,
    SpeakerBasis, TemporalAnchor, TimelineEdge, WorkVocabulary, text_sha256_hex,
};

use crate::chain::resolve_chains;
use crate::dialogue::extract_dialogues;
use crate::event::extract_events;
use faberline_nlp::Span;
use faberline_nlp::doc::{Document, EntityKind, Upos};

/// Combined Wave-1 core extraction output prior to graph/timeline construction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoreExtraction {
    pub dialogues: Vec<DialogueSpan>,
    pub events: Vec<CandidateEvent>,
    pub mentions: Vec<ActorMention>,
    pub unknown: Vec<NarrativeUnknown>,
}

/// Runs quote pairing, guarded candidate event extraction, and intra-sentence topic-chain
/// / mention resolution over `text` and its `nlp.doc.v1` [`Document`].
pub fn extract_core(
    chapter: u32,
    text: &str,
    doc: &Document,
    vocab: &WorkVocabulary,
) -> CoreExtraction {
    let dialogue_res = extract_dialogues(chapter, text, doc, vocab);
    let mut event_res = extract_events(chapter, text, doc, vocab, &dialogue_res.dialogues);
    let chain_res = resolve_chains(
        chapter,
        text,
        doc,
        vocab,
        &dialogue_res.dialogues,
        &mut event_res.events,
    );

    let mut unknown = dialogue_res.unknown;
    unknown.extend(event_res.unknown);
    unknown.extend(chain_res.unknown);
    unknown.sort_by_key(|u| (u.span.start_char, u.span.end_char));

    CoreExtraction {
        dialogues: dialogue_res.dialogues,
        events: event_res.events,
        mentions: chain_res.mentions,
        unknown,
    }
}

/// Validates `doc` against `text`, extracts Wave-1 core narrative structures, constructs the
/// cycle-free timeline graph, and returns a self-contained [`ChapterNarrative`].
pub fn analyze_chapter_narrative(
    chapter: u32,
    title: Option<String>,
    text: &str,
    doc: &Document,
    vocab: &WorkVocabulary,
) -> anyhow::Result<ChapterNarrative> {
    let text_chars = text.chars().count();
    if doc.text_chars != text_chars {
        anyhow::bail!(
            "chapter {chapter}: document text_chars ({}) does not match text scalar count ({text_chars})",
            doc.text_chars
        );
    }
    let text_sha256 = text_sha256_hex(text);
    if doc.text_sha256 != text_sha256 {
        anyhow::bail!(
            "chapter {chapter}: document text_sha256 ({}) does not match text SHA-256 ({text_sha256})",
            doc.text_sha256
        );
    }

    let core = extract_core(chapter, text, doc, vocab);
    let graph_build = build_chapter_graph(chapter, text, doc, &core.events);

    if graph_build.event_dag_layers.len() != core.events.len() {
        anyhow::bail!(
            "chapter {chapter}: event_dag_layers length ({}) != events length ({})",
            graph_build.event_dag_layers.len(),
            core.events.len()
        );
    }
    if graph_build.dag.has_cycle() {
        anyhow::bail!("chapter {chapter}: condensed timeline DAG contains a cycle");
    }

    let mut unknown = core.unknown;
    unknown.extend(graph_build.cycle_unknowns);
    unknown.sort_by_key(|u| (u.span.start_char, u.span.end_char));

    Ok(ChapterNarrative {
        schema: CHAPTER_NARRATIVE_SCHEMA.to_string(),
        chapter,
        title,
        text_chars,
        text_sha256,
        dialogues: core.dialogues,
        events: core.events,
        mentions: core.mentions,
        edges: graph_build.edges,
        event_dag_layers: graph_build.event_dag_layers,
        conflict_cycles_broken: graph_build.conflict_cycles_broken,
        unknown,
    })
}

#[derive(Clone, Debug)]
pub(crate) struct ActorCandidate {
    pub span: Span,
    pub canonical: Option<String>,
    #[allow(dead_code)]
    pub is_explicit_name: bool,
}

pub(crate) fn span_text(chars: &[char], span: Span) -> String {
    let start = span.start_char.min(chars.len());
    let end = span.end_char.min(chars.len());
    if start >= end {
        String::new()
    } else {
        chars[start..end].iter().collect()
    }
}

pub(crate) fn spans_overlap(a: Span, b: Span) -> bool {
    a.start_char < b.end_char && b.start_char < a.end_char
}

pub(crate) fn span_contains(outer: Span, inner: Span) -> bool {
    inner.start_char >= outer.start_char && inner.end_char <= outer.end_char
}

pub(crate) fn is_inside_opaque(opaque_spans: &[Span], char_idx: usize) -> bool {
    opaque_spans
        .iter()
        .any(|o| char_idx >= o.start_char && char_idx < o.end_char)
}

const GLUED_PRONOUN_NON_SPLITS: &[&str] = &[
    "他們", "她們", "它們", "他们", "她们", "它们", "他人", "他日", "他鄉", "他乡", "他年", "他山",
    "他殺", "他杀", "她人", "其它", "其他", "其實", "其实", "其中", "其餘", "其余",
];

/// When Chinese word segmentation glues a 3rd-person singular pronoun (`他`/`她`/`它`/`牠`) onto
/// the following verb (e.g. `他端`), returns the 1-char pronoun span if `surf` is not a lexical word.
pub(crate) fn glued_pronoun_prefix(surf: &str, tok_span: Span) -> Option<Span> {
    let mut it = surf.chars();
    let first = it.next()?;
    let second = it.next()?;
    if !matches!(first, '他' | '她' | '它' | '牠') {
        return None;
    }
    if GLUED_PRONOUN_NON_SPLITS.contains(&surf) {
        return None;
    }
    // Only split if the remaining part starts with a Han character.
    if ('\u{4e00}'..='\u{9fff}').contains(&second) {
        Some(Span {
            start_char: tok_span.start_char,
            end_char: tok_span.start_char + 1,
        })
    } else {
        None
    }
}

pub(crate) fn is_personal_pronoun(surf: &str) -> bool {
    let lower = surf.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "他" | "她"
            | "它"
            | "牠"
            | "其"
            | "他們"
            | "她們"
            | "它們"
            | "他们"
            | "她们"
            | "它们"
            | "眾人"
            | "众人"
            | "大夥"
            | "大家"
            | "我們"
            | "我们"
            | "你們"
            | "你们"
            | "咱們"
            | "咱们"
            | "我"
            | "你"
            | "妳"
            | "老夫"
            | "老身"
            | "在下"
            | "妾身"
            | "貧道"
            | "贫道"
            | "本座"
            | "本人"
            | "小生"
            | "晚輩"
            | "晚辈"
            | "吾"
            | "余"
            | "予"
            | "私"
            | "わたし"
            | "あたし"
            | "わたくし"
            | "僕"
            | "ぼく"
            | "俺"
            | "おれ"
            | "あなた"
            | "あんた"
            | "君"
            | "きみ"
            | "お前"
            | "おまえ"
            | "彼"
            | "彼女"
            | "彼ら"
            | "彼女ら"
            | "あいつ"
            | "こいつ"
            | "自分"
            | "he"
            | "him"
            | "his"
            | "she"
            | "her"
            | "hers"
            | "it"
            | "its"
            | "they"
            | "them"
            | "their"
            | "theirs"
            | "i"
            | "me"
            | "my"
            | "we"
            | "us"
            | "our"
            | "you"
            | "your"
    )
}

const JA_ACTOR_HONORIFIC_SUFFIXES: &[&str] = &[
    "お嬢様",
    "先輩",
    "先生",
    "ちゃん",
    "さん",
    "くん",
    "さま",
    "様",
    "殿",
    "氏",
    "嬢",
];

pub(crate) fn strip_ja_honorific(s: &str) -> &str {
    for &suffix in JA_ACTOR_HONORIFIC_SUFFIXES {
        if let Some(stem) = s.strip_suffix(suffix)
            && !stem.is_empty()
        {
            return stem;
        }
    }
    s
}

pub(crate) fn normalize_ja_actor_key(vocab: &WorkVocabulary, surf: &str) -> Option<String> {
    if let Some(canon) = vocab.lookup_actor(surf) {
        return Some(canon.to_owned());
    }
    let stripped = strip_ja_honorific(surf);
    if stripped != surf {
        return Some(
            vocab
                .lookup_actor(stripped)
                .map_or_else(|| stripped.to_owned(), str::to_owned),
        );
    }
    None
}

pub(crate) fn collect_actor_candidates(
    _chapter: u32,
    text: &str,
    chars: &[char],
    doc: &Document,
    vocab: &WorkVocabulary,
    opaque_spans: &[Span],
) -> Vec<ActorCandidate> {
    let mut out: Vec<ActorCandidate> = Vec::new();
    let places = vocab.find_place_spans(text);
    let objects = vocab.find_object_spans(text);

    for ent in &doc.entities {
        if ent.kind != EntityKind::Name || ent.span.end_char > chars.len() {
            continue;
        }
        if opaque_spans.iter().any(|o| spans_overlap(*o, ent.span)) {
            continue;
        }
        let surf = span_text(chars, ent.span);
        out.push(ActorCandidate {
            span: ent.span,
            canonical: normalize_ja_actor_key(vocab, &surf),
            is_explicit_name: true,
        });
    }

    for (span, canonical) in vocab.find_actor_spans(text) {
        if span.end_char > chars.len() || opaque_spans.iter().any(|o| spans_overlap(*o, span)) {
            continue;
        }
        if out.iter().any(|c| spans_overlap(c.span, span)) {
            continue;
        }
        out.push(ActorCandidate {
            span,
            canonical: Some(canonical),
            is_explicit_name: true,
        });
    }

    for tok in &doc.tokens {
        if tok.upos != Some(Upos::Propn) || tok.span.end_char > chars.len() {
            continue;
        }
        if opaque_spans.iter().any(|o| spans_overlap(*o, tok.span)) {
            continue;
        }
        if out.iter().any(|c| spans_overlap(c.span, tok.span)) {
            continue;
        }
        if places.iter().any(|(s, _)| spans_overlap(*s, tok.span))
            || objects.iter().any(|(s, _)| spans_overlap(*s, tok.span))
        {
            continue;
        }
        if doc
            .entities
            .iter()
            .any(|e| e.kind != EntityKind::Name && spans_overlap(e.span, tok.span))
        {
            continue;
        }
        let surf = span_text(chars, tok.span);
        out.push(ActorCandidate {
            span: tok.span,
            canonical: normalize_ja_actor_key(vocab, &surf),
            is_explicit_name: true,
        });
    }

    out.sort_by_key(|c| (c.span.start_char, c.span.end_char));
    out
}
