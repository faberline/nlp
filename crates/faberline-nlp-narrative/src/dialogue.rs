//! Quotation pairing, context-zone classification, and conservative speaker attribution.

use crate::schema::{
    ContextZone, DialogueSpan, NarrativeUnknown, NarrativeUnknownKind, SpeakerBasis, WorkVocabulary,
};
use crate::{
    ActorCandidate, collect_actor_candidates, is_inside_opaque, span_contains, span_text,
    spans_overlap,
};
use faberline_nlp::Span;
use faberline_nlp::doc::{Document, SyntaxToken, Upos};
use std::cmp::Reverse;
use std::collections::HashSet;

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuoteKind {
    Corner,
    DoubleCorner,
    CurlyDouble,
    CurlySingle,
    AsciiDouble,
}

/// Output of dialogue pairing and speaker attribution for a chapter.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DialogueExtraction {
    pub dialogues: Vec<DialogueSpan>,
    pub unknown: Vec<NarrativeUnknown>,
}

const CJK_SPEECH_VERBS: &[&str] = &[
    "說道",
    "说道",
    "問道",
    "问道",
    "答道",
    "笑道",
    "冷笑道",
    "喝道",
    "怒道",
    "嘆道",
    "叹道",
    "歎道",
    "喊道",
    "叫道",
    "嚷道",
    "唸道",
    "念道",
    "喃喃道",
    "低聲道",
    "低声道",
    "沉聲道",
    "沉声道",
    "朗聲道",
    "朗声道",
    "緩緩道",
    "缓缓道",
    "淡淡道",
    "輕聲道",
    "轻声道",
    "開口道",
    "开口道",
    "點頭道",
    "点头道",
    "搖頭道",
    "摇头道",
    "傳音道",
    "传音道",
    "拱手道",
    "抱拳道",
    "言道",
    "講道",
    "讲道",
    "勸道",
    "劝道",
    "回答",
    "解釋",
    "解释",
    "補充",
    "补充",
    "追問",
    "追问",
    "反問",
    "反问",
    "回應",
    "回应",
    "詢問",
    "询问",
    "告訴",
    "告诉",
    "叮囑",
    "叮嘱",
    "吩咐",
    "交代",
    "插嘴",
    "開口",
    "开口",
    "說",
    "说",
    "問",
    "问",
    "答",
    "喊",
    "叫",
    "講",
    "讲",
];

const DAO_NON_SPEECH_PREV: &[char] = &[
    '知', '難', '难', '通', '大', '小', '山', '石', '官', '河', '軌', '轨', '味', '公', '天', '人',
    '正', '邪', '門', '门', '同', '順', '顺', '一', '霸', '王', '武', '醫', '医', '茶', '花', '書',
    '书', '寫', '写', '報', '报', '世', '古', '車', '车', '街', '巷', '樓', '楼', '力', '氣', '气',
    '修', '得', '悟', '證', '证', '成', '求', '尋', '寻', '步', '棧', '栈', '林', '小', '窄', '寬',
    '宽', '彎', '弯', '要', '應', '应',
];

const EN_SPEECH_VERBS: &[&str] = &[
    "said",
    "say",
    "says",
    "asked",
    "ask",
    "asks",
    "replied",
    "reply",
    "replies",
    "whispered",
    "whisper",
    "whispers",
    "shouted",
    "shout",
    "shouts",
    "murmured",
    "murmur",
    "murmurs",
    "muttered",
    "mutter",
    "exclaimed",
    "exclaim",
    "answered",
    "answer",
    "added",
    "told",
    "cried",
    "called",
];

/// Classifies whether `span` lies inside a direct quotation or in narrator prose.
pub fn zone_for_span(dialogues: &[DialogueSpan], span: Span) -> ContextZone {
    for d in dialogues {
        if span.start_char == span.end_char {
            if span.start_char >= d.span.start_char && span.start_char < d.span.end_char {
                return ContextZone::DirectQuote {
                    quote_index: d.quote_index,
                };
            }
        } else if span_contains(d.span, span) {
            return ContextZone::DirectQuote {
                quote_index: d.quote_index,
            };
        }
    }
    ContextZone::Narrative
}

/// Pairs balanced/nested direct quotes per paragraph and attributes speakers conservatively.
pub fn extract_dialogues(
    chapter: u32,
    text: &str,
    doc: &Document,
    vocab: &WorkVocabulary,
) -> DialogueExtraction {
    let chars: Vec<char> = text.chars().collect();
    let opaque_spans: Vec<Span> = doc.segmentation.opaque.iter().map(|o| o.span).collect();
    let raw_quotes = pair_outer_quotes(&chars, doc, &opaque_spans);
    let all_actors = collect_actor_candidates(chapter, text, &chars, doc, vocab, &opaque_spans);

    let mut syntax_by_tok: Vec<Option<&SyntaxToken>> = vec![None; doc.tokens.len()];
    for st in &doc.syntax {
        if st.token < doc.tokens.len() {
            syntax_by_tok[st.token] = Some(st);
        }
    }

    let mut dialogues = Vec::with_capacity(raw_quotes.len());
    let mut unknown = Vec::new();

    for (quote_index, (para_idx, outer_span)) in raw_quotes.iter().copied().enumerate() {
        let content_span = Span {
            start_char: (outer_span.start_char + 1).min(outer_span.end_char),
            end_char: outer_span
                .end_char
                .saturating_sub(1)
                .max((outer_span.start_char + 1).min(outer_span.end_char)),
        };
        dialogues.push(DialogueSpan {
            quote_index,
            span: outer_span,
            content_span,
            paragraph: para_idx,
            speech_verb: None,
            speaker_span: None,
            speaker_canonical: None,
            basis: SpeakerBasis::Unknown,
        });
    }

    let para_quotes = group_quotes_by_paragraph(&dialogues, doc.segmentation.paragraphs.len());

    for idx in 0..dialogues.len() {
        let para_idx = dialogues[idx].paragraph;
        let para_span = doc.segmentation.paragraphs[para_idx].span;
        let q_span = dialogues[idx].span;

        let prev_q_end = prev_quote_end_in_para(&dialogues, &para_quotes[para_idx], idx)
            .unwrap_or(para_span.start_char);
        let next_q_start = next_quote_start_in_para(&dialogues, &para_quotes[para_idx], idx)
            .unwrap_or(para_span.end_char);

        // 1. Pre-quote framing window (in the same sentence, after any earlier quote).
        let sent_start = sentence_start_at(&chars, doc, q_span.start_char, para_span.start_char);
        let pre_start = prev_q_end.max(sent_start);
        let pre_win = trim_window(&chars, pre_start, q_span.start_char);

        let mut attributed = false;
        if let Some(pre) = pre_win
            && let Some(sv_span) = find_pre_quote_speech_verb(&chars, doc, pre)
        {
            attributed = true;
            resolve_framing_speaker(
                chapter,
                &chars,
                doc,
                &syntax_by_tok,
                &all_actors,
                pre,
                sv_span,
                SpeakerBasis::PreQuoteSpeechVerb,
                &mut dialogues[idx],
                &mut unknown,
            );
        }

        // 2. Post-quote framing window (up to next sentence end or next quote in the paragraph).
        if !attributed {
            let post_limit = post_quote_window_end(&chars, doc, q_span.end_char, next_q_start);
            if let Some(post) = trim_window(&chars, q_span.end_char, post_limit)
                && !post_introduces_next_quote(&chars, post, next_q_start, q_span)
                && let Some(sv_span) = find_post_quote_speech_verb(&chars, doc, post)
            {
                attributed = true;
                resolve_framing_speaker(
                    chapter,
                    &chars,
                    doc,
                    &syntax_by_tok,
                    &all_actors,
                    post,
                    sv_span,
                    SpeakerBasis::PostQuoteSpeechVerb,
                    &mut dialogues[idx],
                    &mut unknown,
                );
            }
        }

        // 3. Fallback: single unambiguous narrative actor in the same paragraph.
        if !attributed {
            let para_actors = narrative_actors_in_span(&chars, &all_actors, &dialogues, para_span);
            if para_actors.len() == 1 {
                dialogues[idx].basis = SpeakerBasis::IntraParagraphSingleActor;
                dialogues[idx].speaker_span = Some(para_actors[0].span);
                dialogues[idx].speaker_canonical = para_actors[0].canonical.clone();
            } else if para_actors.len() >= 2 {
                unknown.push(NarrativeUnknown {
                    kind: NarrativeUnknownKind::SpeakerAmbiguous,
                    chapter,
                    span: q_span,
                    detail: format!(
                        "{} narrative actors in paragraph with no framing speech verb",
                        para_actors.len()
                    ),
                });
            }
        }
    }

    DialogueExtraction { dialogues, unknown }
}

fn pair_outer_quotes(chars: &[char], doc: &Document, opaque_spans: &[Span]) -> Vec<(usize, Span)> {
    let mut out = Vec::new();
    for (para_idx, para) in doc.segmentation.paragraphs.iter().enumerate() {
        let p_start = para.span.start_char.min(chars.len());
        let p_end = para.span.end_char.min(chars.len());
        let mut stack: Vec<(QuoteKind, usize)> = Vec::new();
        let mut matched: Vec<(usize, usize)> = Vec::new();

        for i in p_start..p_end {
            if is_inside_opaque(opaque_spans, i) {
                continue;
            }
            match chars[i] {
                '「' => stack.push((QuoteKind::Corner, i)),
                '『' => stack.push((QuoteKind::DoubleCorner, i)),
                '\u{201c}' => stack.push((QuoteKind::CurlyDouble, i)),
                '\u{2018}' => {
                    let prev_alnum = i > p_start && chars[i - 1].is_ascii_alphanumeric();
                    if !prev_alnum {
                        stack.push((QuoteKind::CurlySingle, i));
                    }
                }
                '」' => close_kind(&mut stack, QuoteKind::Corner, i, &mut matched),
                '』' => close_kind(&mut stack, QuoteKind::DoubleCorner, i, &mut matched),
                '\u{201d}' => close_kind(&mut stack, QuoteKind::CurlyDouble, i, &mut matched),
                '\u{2019}' => {
                    let prev_alnum = i > p_start && chars[i - 1].is_ascii_alphanumeric();
                    let next_alnum = i + 1 < p_end && chars[i + 1].is_ascii_alphanumeric();
                    if !(prev_alnum && next_alnum) {
                        close_kind(&mut stack, QuoteKind::CurlySingle, i, &mut matched);
                    }
                }
                '"' => {
                    if let Some(pos) = stack
                        .iter()
                        .rposition(|(k, _)| *k == QuoteKind::AsciiDouble)
                    {
                        let (_, open) = stack[pos];
                        stack.truncate(pos);
                        matched.push((open, i));
                    } else {
                        stack.push((QuoteKind::AsciiDouble, i));
                    }
                }
                _ => {}
            }
        }

        matched.sort_by_key(|&(open, close)| (open, Reverse(close)));
        let mut last_end = p_start;
        for (open, close) in matched {
            if open >= last_end {
                out.push((
                    para_idx,
                    Span {
                        start_char: open,
                        end_char: close + 1,
                    },
                ));
                last_end = close + 1;
            }
        }
    }
    out
}

fn close_kind(
    stack: &mut Vec<(QuoteKind, usize)>,
    kind: QuoteKind,
    close_idx: usize,
    matched: &mut Vec<(usize, usize)>,
) {
    if let Some(pos) = stack.iter().rposition(|(k, _)| *k == kind) {
        let (_, open) = stack[pos];
        stack.truncate(pos);
        matched.push((open, close_idx));
    }
}

fn group_quotes_by_paragraph(dialogues: &[DialogueSpan], num_paras: usize) -> Vec<Vec<usize>> {
    let mut groups = vec![Vec::new(); num_paras];
    for (idx, d) in dialogues.iter().enumerate() {
        if d.paragraph < num_paras {
            groups[d.paragraph].push(idx);
        }
    }
    groups
}

fn prev_quote_end_in_para(
    dialogues: &[DialogueSpan],
    para_indices: &[usize],
    current: usize,
) -> Option<usize> {
    let pos = para_indices.iter().position(|&i| i == current)?;
    if pos == 0 {
        None
    } else {
        Some(dialogues[para_indices[pos - 1]].span.end_char)
    }
}

fn next_quote_start_in_para(
    dialogues: &[DialogueSpan],
    para_indices: &[usize],
    current: usize,
) -> Option<usize> {
    let pos = para_indices.iter().position(|&i| i == current)?;
    para_indices
        .get(pos + 1)
        .map(|&next| dialogues[next].span.start_char)
}

fn sentence_start_at(chars: &[char], doc: &Document, pos: usize, fallback: usize) -> usize {
    for s in &doc.segmentation.sentences {
        if pos >= s.span.start_char && pos <= s.span.end_char {
            return s.span.start_char.min(chars.len());
        }
    }
    fallback.min(chars.len())
}

fn post_quote_window_end(
    chars: &[char],
    doc: &Document,
    after_quote: usize,
    next_quote_start: usize,
) -> usize {
    let limit = next_quote_start.min(chars.len());
    let mut i = after_quote.min(limit);
    while i < limit && (chars[i].is_whitespace() || matches!(chars[i], '，' | ',')) {
        i += 1;
    }
    for s in &doc.segmentation.sentences {
        if i >= s.span.start_char && i < s.span.end_char {
            return s.span.end_char.min(limit);
        }
    }
    limit
}

fn post_introduces_next_quote(
    chars: &[char],
    post: Span,
    next_q_start: usize,
    current_q: Span,
) -> bool {
    if next_q_start >= chars.len() {
        return false;
    }
    let mut end = post.end_char;
    while end > post.start_char && chars[end - 1].is_whitespace() {
        end -= 1;
    }
    let ends_with_colon = end > post.start_char && matches!(chars[end - 1], '：' | ':');
    if !ends_with_colon {
        return false;
    }
    // Check whether the current quote itself ended with a continuation comma (e.g. `"Hi," he said, "bye."`).
    let q_inner_end = current_q.end_char.saturating_sub(1);
    let mut prev = q_inner_end;
    while prev > current_q.start_char && chars[prev - 1].is_whitespace() {
        prev -= 1;
    }
    let quote_continued = prev > current_q.start_char && matches!(chars[prev - 1], '，' | ',');
    !quote_continued
}

fn trim_window(chars: &[char], start: usize, end: usize) -> Option<Span> {
    let mut s = start.min(chars.len());
    let mut e = end.min(chars.len());
    while s < e && chars[s].is_whitespace() {
        s += 1;
    }
    while e > s && chars[e - 1].is_whitespace() {
        e -= 1;
    }
    if s < e {
        Some(Span {
            start_char: s,
            end_char: e,
        })
    } else {
        None
    }
}

fn find_pre_quote_speech_verb(chars: &[char], doc: &Document, window: Span) -> Option<Span> {
    let mut end = window.end_char;
    while end > window.start_char
        && (chars[end - 1].is_whitespace() || matches!(chars[end - 1], '：' | ':' | '，' | ','))
    {
        end -= 1;
    }
    if end <= window.start_char {
        return None;
    }
    let trimmed = Span {
        start_char: window.start_char,
        end_char: end,
    };
    // Look in the trailing clause of the pre-quote window.
    let mut clause_start = trimmed.start_char;
    for i in (trimmed.start_char..trimmed.end_char).rev() {
        if matches!(chars[i], '，' | '；' | ',' | ';') {
            clause_start = i + 1;
            break;
        }
    }
    let clause_span = trim_window(chars, clause_start, trimmed.end_char)?;
    find_speech_verb_in_span(chars, doc, clause_span, true)
}

fn find_post_quote_speech_verb(chars: &[char], doc: &Document, window: Span) -> Option<Span> {
    // Only inspect the first framing clause immediately following the closing quote.
    let mut start = window.start_char;
    while start < window.end_char
        && (chars[start].is_whitespace() || matches!(chars[start], '，' | ','))
    {
        start += 1;
    }
    let mut clause_end = window.end_char;
    for (i, &ch) in chars.iter().enumerate().take(window.end_char).skip(start) {
        if matches!(ch, '，' | '；' | '：' | ',' | ';' | ':') {
            clause_end = i;
            break;
        }
    }
    let clause_span = trim_window(chars, start, clause_end)?;
    find_speech_verb_in_span(chars, doc, clause_span, false)
}

fn find_speech_verb_in_span(
    chars: &[char],
    doc: &Document,
    span: Span,
    at_end_only_for_dao: bool,
) -> Option<Span> {
    let mut tokens_in_span: Vec<(usize, &faberline_nlp::doc::Token)> = doc
        .tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| span_contains(span, t.span))
        .collect();
    tokens_in_span.sort_by_key(|(_, t)| t.span.start_char);

    // Search from right to left in pre-quote (closest to quote), or left to right in post-quote.
    let ordered: Vec<(usize, &faberline_nlp::doc::Token)> = if at_end_only_for_dao {
        tokens_in_span.into_iter().rev().collect()
    } else {
        tokens_in_span
    };

    for (_, tok) in &ordered {
        let surf = span_text(chars, tok.span);
        let lower = surf.to_ascii_lowercase();
        if EN_SPEECH_VERBS.contains(&lower.as_str()) {
            return Some(tok.span);
        }
        if surf == "道"
            && (!at_end_only_for_dao || is_trailing_token(chars, tok.span, span.end_char))
        {
            // Merge `說`+`道`, `問`+`道`, `答`+`道`, `笑`+`道`, etc. if segmented as two tokens.
            if tok.span.start_char > span.start_char
                && matches!(
                    chars[tok.span.start_char - 1],
                    '說' | '说'
                        | '問'
                        | '问'
                        | '答'
                        | '笑'
                        | '喝'
                        | '怒'
                        | '嘆'
                        | '叹'
                        | '歎'
                        | '喊'
                        | '叫'
                        | '言'
                        | '講'
                        | '讲'
                        | '勸'
                        | '劝'
                )
            {
                return Some(Span {
                    start_char: tok.span.start_char - 1,
                    end_char: tok.span.end_char,
                });
            }
            let prev_char = tok.span.start_char.checked_sub(1).map(|p| chars[p]);
            if !prev_char.is_some_and(|ch| DAO_NON_SPEECH_PREV.contains(&ch)) {
                return Some(tok.span);
            }
        }
        for &verb in CJK_SPEECH_VERBS {
            if surf == verb {
                // If `說` is immediately followed by `道` inside `span`, extend to `說道`.
                if matches!(
                    surf.as_str(),
                    "說" | "说" | "問" | "问" | "答" | "講" | "讲"
                ) && tok.span.end_char < span.end_char
                    && chars[tok.span.end_char] == '道'
                {
                    return Some(Span {
                        start_char: tok.span.start_char,
                        end_char: tok.span.end_char + 1,
                    });
                }
                return Some(tok.span);
            }
            if surf.ends_with(verb) && verb.chars().count() >= 2 {
                let verb_chars = verb.chars().count();
                return Some(Span {
                    start_char: tok.span.end_char - verb_chars,
                    end_char: tok.span.end_char,
                });
            }
        }
    }

    // Fallback substring scan if tokenizer glued adverb+speech_verb (e.g. `低聲說道` or `道`).
    let slice_text = span_text(chars, span);
    for &verb in CJK_SPEECH_VERBS {
        if verb.chars().count() >= 2
            && let Some(byte_pos) = slice_text.rfind(verb)
        {
            let char_off = slice_text[..byte_pos].chars().count();
            let char_len = verb.chars().count();
            return Some(Span {
                start_char: span.start_char + char_off,
                end_char: span.start_char + char_off + char_len,
            });
        }
    }
    if span.end_char > span.start_char {
        let mut last = span.end_char;
        while last > span.start_char
            && matches!(chars[last - 1], '。' | '！' | '？' | '.' | '!' | '?')
        {
            last -= 1;
        }
        if last > span.start_char && chars[last - 1] == '道' {
            let prev_char = (last - 1).checked_sub(1).map(|p| chars[p]);
            if !prev_char.is_some_and(|ch| DAO_NON_SPEECH_PREV.contains(&ch)) {
                return Some(Span {
                    start_char: last - 1,
                    end_char: last,
                });
            }
        }
    }
    None
}

fn is_trailing_token(chars: &[char], tok_span: Span, span_end: usize) -> bool {
    (tok_span.end_char..span_end).all(|i| {
        chars[i].is_whitespace() || matches!(chars[i], '了' | '著' | '着' | '。' | '！' | '？')
    })
}

#[allow(clippy::too_many_arguments)]
fn resolve_framing_speaker(
    chapter: u32,
    chars: &[char],
    doc: &Document,
    syntax_by_tok: &[Option<&SyntaxToken>],
    all_actors: &[ActorCandidate],
    frame_win: Span,
    sv_span: Span,
    basis: SpeakerBasis,
    out_dialogue: &mut DialogueSpan,
    unknown: &mut Vec<NarrativeUnknown>,
) {
    out_dialogue.speech_verb = Some(sv_span);

    // Narrow to the framing clause containing `sv_span`.
    let frame_clause = doc
        .segmentation
        .clauses
        .iter()
        .find(|c| spans_overlap(c.span, sv_span) && spans_overlap(c.span, frame_win))
        .map(|c| Span {
            start_char: c.span.start_char.max(frame_win.start_char),
            end_char: c.span.end_char.min(frame_win.end_char),
        })
        .unwrap_or(frame_win);

    // Collect candidate actors inside `frame_clause`.
    let mut candidates: Vec<ActorCandidate> = all_actors
        .iter()
        .filter(|a| span_contains(frame_clause, a.span))
        .cloned()
        .collect();

    // Also include personal pronouns or syntax `nsubj` inside `frame_clause` if not already covered.
    let sv_tok_idx = doc
        .tokens
        .iter()
        .position(|t| spans_overlap(t.span, sv_span));

    for (t_idx, tok) in doc.tokens.iter().enumerate() {
        if !span_contains(frame_clause, tok.span) || spans_overlap(tok.span, sv_span) {
            continue;
        }
        if candidates.iter().any(|c| spans_overlap(c.span, tok.span)) {
            continue;
        }
        let surf = span_text(chars, tok.span);
        let is_pron = tok.upos == Some(Upos::Pron) || crate::is_personal_pronoun(&surf);
        let is_syntax_nsubj = syntax_by_tok.get(t_idx).and_then(|s| *s).is_some_and(|st| {
            st.deprel == "nsubj"
                && st.score.milli() >= 350
                && sv_tok_idx.is_some_and(|sv_i| st.head == Some(sv_i))
                && matches!(tok.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
        });
        if is_pron || is_syntax_nsubj {
            candidates.push(ActorCandidate {
                span: tok.span,
                canonical: None,
                is_explicit_name: false,
            });
        }
    }

    candidates.sort_by_key(|c| (c.span.start_char, c.span.end_char));
    let distinct = dedup_distinct_actors(&candidates, chars);

    if distinct.len() == 1 {
        out_dialogue.basis = basis;
        out_dialogue.speaker_span = Some(distinct[0].span);
        out_dialogue.speaker_canonical = distinct[0].canonical.clone();
        return;
    }

    if distinct.len() >= 2 {
        // Check if Layer-4 syntax cleanly disambiguates a single `nsubj` attached to `sv_tok_idx`.
        if let Some(sv_i) = sv_tok_idx {
            let nsubj_hits: Vec<usize> = doc
                .tokens
                .iter()
                .enumerate()
                .filter_map(|(t_idx, tok)| {
                    let st = syntax_by_tok.get(t_idx).and_then(|s| *s)?;
                    if st.head == Some(sv_i)
                        && st.deprel == "nsubj"
                        && st.score.milli() >= 350
                        && span_contains(frame_clause, tok.span)
                        && matches!(tok.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
                    {
                        Some(t_idx)
                    } else {
                        None
                    }
                })
                .collect();

            if nsubj_hits.len() == 1 {
                let subj_span = doc.tokens[nsubj_hits[0]].span;
                if let Some(matched) = distinct.iter().find(|c| spans_overlap(c.span, subj_span)) {
                    out_dialogue.basis = basis;
                    out_dialogue.speaker_span = Some(matched.span);
                    out_dialogue.speaker_canonical = matched.canonical.clone();
                    return;
                }
            }
        }

        out_dialogue.basis = SpeakerBasis::Unknown;
        unknown.push(NarrativeUnknown {
            kind: NarrativeUnknownKind::SpeakerAmbiguous,
            chapter,
            span: frame_clause,
            detail: format!(
                "{} candidate actors in speech-verb framing clause without syntax disambiguation",
                distinct.len()
            ),
        });
    }
}

fn narrative_actors_in_span(
    chars: &[char],
    all_actors: &[ActorCandidate],
    dialogues: &[DialogueSpan],
    para_span: Span,
) -> Vec<ActorCandidate> {
    let mut filtered = Vec::new();
    let mut seen_keys: HashSet<String> = HashSet::new();
    for actor in all_actors {
        if !span_contains(para_span, actor.span) {
            continue;
        }
        if !matches!(zone_for_span(dialogues, actor.span), ContextZone::Narrative) {
            continue;
        }
        let key = actor
            .canonical
            .clone()
            .unwrap_or_else(|| span_text(chars, actor.span));
        if seen_keys.insert(key) {
            filtered.push(actor.clone());
        }
    }
    filtered
}

fn dedup_distinct_actors(candidates: &[ActorCandidate], chars: &[char]) -> Vec<ActorCandidate> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for c in candidates {
        let key = c
            .canonical
            .clone()
            .unwrap_or_else(|| span_text(chars, c.span));
        if seen.insert(key) {
            out.push(c.clone());
        }
    }
    out
}
