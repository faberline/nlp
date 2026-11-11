use std::collections::{HashMap, HashSet};

use crate::language::*;
use crate::recap_time;
use crate::text::{Analyzer, Locale, TokenKind};
use crate::{Analysis, Event, Lexicon, Review, Span, TextSection, TextUnit};

#[cfg(test)]
mod doc_line_tests;
mod linking;
#[cfg(test)]
mod object_coverage_tests;
mod speech;
mod transfer_roles;

pub use linking::{
    ClauseCandidate, ClauseOrderAdvisory, ClauseOrderBinding, ClauseOrderCue, ClauseOrderCueKind,
    ClauseOrderDiagnostic, ClauseOrderEndpoint, ClauseOrderProbe, ClauseOrderSide,
    ClauseOrderWindow, ClausePrecedence, ClauseResultLink, ClauseScope, DurationObservation,
    LinkedClauses, RankedClauseOrderCue, ReadingAction, ScopedClause, VerifiedClauseEvidence,
    VerifiedClauseHead,
};
pub use speech::{SpeechAct, SpeechActKind};
pub use transfer_roles::{
    WrittenTransferRoleAlternatives, WrittenTransferRoleCandidate, WrittenTransferUnknown,
};

struct Source<'a> {
    text: &'a str,
    bytes: Vec<usize>,
    actor_tokens: Option<Vec<(usize, usize)>>,
}

impl<'a> Source<'a> {
    fn new(text: &'a str) -> Self {
        let mut bytes: Vec<_> = text.char_indices().map(|(offset, _)| offset).collect();
        bytes.push(text.len());
        Self {
            text,
            bytes,
            actor_tokens: None,
        }
    }

    fn slice(&self, span: &Span) -> &'a str {
        &self.text[self.bytes[span.start_char]..self.bytes[span.end_char]]
    }

    fn span(&self, start: usize, end: usize) -> Span {
        Span {
            start_char: self
                .bytes
                .binary_search(&start)
                .expect("character boundary"),
            end_char: self.bytes.binary_search(&end).expect("character boundary"),
        }
    }

    fn trim(&self, start: usize, end: usize) -> Option<Span> {
        let text = &self.text[start..end];
        let is_margin = |ch: char| ch.is_whitespace() || ch == '\u{feff}';
        let leading = text.trim_start_matches(is_margin);
        let body = leading.trim_end_matches(is_margin);
        (!body.is_empty()).then(|| {
            let start = start + text.len() - leading.len();
            self.span(start, start + body.len())
        })
    }

    fn first_name<'lexicon>(
        &self,
        start: usize,
        end: usize,
        lexicon: &Lexicon<'lexicon>,
    ) -> Option<(usize, &'lexicon str, &'lexicon str)> {
        let Some(tokens) = &self.actor_tokens else {
            return first_name(&self.text[start..end], lexicon);
        };
        let &(offset, actor) = tokens.get(tokens.partition_point(|&(offset, _)| offset < start))?;
        let (word, name) = lexicon.actors[actor];
        (offset + word.len() <= end).then_some((offset - start, word, name))
    }

    fn has_speech(&self, text: &str) -> bool {
        contains_any(text, SPEECH)
            || (self.actor_tokens.is_some() && contains_any(text, SPEECH_CONTINUATION))
    }
}

const SPEECH_CONTINUATION: &[&str] = &["說下去", "说下去"];

fn contains_any(text: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| text.contains(term))
}

fn starts_any(text: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| text.starts_with(term))
}

fn ends_any(text: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| text.ends_with(term))
}

fn quote_end(ch: char) -> Option<char> {
    match ch {
        '“' => Some('”'),
        '‘' => Some('’'),
        '「' => Some('」'),
        '『' => Some('』'),
        '"' => Some('"'),
        _ => None,
    }
}

// Malformed, crossed, or cross-line quote spans are not safe to treat as
// ordinary narration. Preserve text and ranges but stop asserting occurrence.
fn source_quote_damage_start(text: &str) -> Option<usize> {
    let mut stack: Vec<(char, usize)> = Vec::new();
    for (position, ch) in text.chars().enumerate() {
        if matches!(ch, '\n' | '\r') && !stack.is_empty() {
            return stack.first().map(|entry| entry.1);
        }
        if stack.last().is_some_and(|entry| entry.0 == ch) {
            stack.pop();
        } else if let Some(end) = quote_end(ch) {
            stack.push((end, position));
        } else if matches!(ch, '」' | '』' | '”' | '’') {
            // A stray closer might close a missing opener anywhere earlier.
            return Some(stack.first().map_or(0, |entry| entry.1));
        }
    }
    stack.first().map(|entry| entry.1)
}

// The chapter analyzer has a title and caller footers, so it scans only the
// text admitted by the same filtering used for chapter units. Standalone
// speech/transfer APIs intentionally retain the raw-text behavior above.
fn admitted_quote_damage_start(
    source: &Source<'_>,
    title: &str,
    lexicon: &Lexicon<'_>,
) -> Option<usize> {
    let mut stack: Vec<(char, usize)> = Vec::new();
    for line in admitted_lines(source.text, title, lexicon) {
        for (relative, ch) in line.text.char_indices() {
            if matches!(ch, '\n' | '\r') && !stack.is_empty() {
                return stack.first().map(|entry| entry.1);
            }
            if stack.last().is_some_and(|entry| entry.0 == ch) {
                stack.pop();
            } else if let Some(end) = quote_end(ch) {
                let byte = line.start + relative;
                let position = source
                    .bytes
                    .binary_search(&byte)
                    .expect("character boundary");
                stack.push((end, position));
            } else if matches!(ch, '」' | '』' | '”' | '’') {
                // A stray closer might close a missing opener anywhere earlier.
                return Some(stack.first().map_or(0, |entry| entry.1));
            }
        }
        // A line footer can remove the physical line ending from the admitted
        // prefix. It must still prevent a quote from crossing into a later line.
        if line.removed_newline && !stack.is_empty() {
            return stack.first().map(|entry| entry.1);
        }
    }
    stack.first().map(|entry| entry.1)
}

fn track_quote(text: &str, offset: usize, ch: char, stack: &mut Vec<char>) -> bool {
    if stack.last() == Some(&ch) {
        stack.pop();
    } else if let Some(end) = quote_end(ch)
        && text[offset + ch.len_utf8()..].contains(end)
    {
        stack.push(end);
    } else {
        return false;
    }
    true
}

fn split_outside_quotes(text: &str, boundary: impl Fn(char) -> bool) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut stack = Vec::new();
    let mut start = 0;
    for (offset, ch) in text.char_indices() {
        if !track_quote(text, offset, ch, &mut stack) && stack.is_empty() && boundary(ch) {
            let end = offset + ch.len_utf8();
            ranges.push((start, end));
            start = end;
        }
    }
    if start < text.len() {
        ranges.push((start, text.len()));
    }
    ranges
}

struct Unit {
    evidence: Span,
    sentence: Span,
    speech_context: Span,
    quote: bool,
    first_in_line: bool,
}

fn append_narration(
    source: &Source<'_>,
    units: &mut Vec<Unit>,
    sentence: &Span,
    start: usize,
    end: usize,
    first_in_line: &mut bool,
) {
    for (left, right) in split_outside_quotes(&source.text[start..end], |ch| {
        matches!(ch, '，' | ',' | '；' | ';')
    }) {
        if let Some(evidence) = source.trim(start + left, start + right) {
            units.push(Unit {
                evidence,
                sentence: *sentence,
                speech_context: *sentence,
                quote: false,
                first_in_line: std::mem::take(first_in_line),
            });
        }
    }
}

fn is_dialogue(source: &Source<'_>, body: &str, before: &str, after: &str) -> bool {
    // An adjacent hearing cue makes the quotation part of the surrounding
    // written hearing construction, even if its content ends in a question.
    if ends_any(
        before.trim_end(),
        &["聽到了", "听到了", "聽見", "听见", "聽到", "听到"],
    ) {
        return false;
    }
    contains_any(body, &["。", "！", "？", "!", "?"])
        || (body.chars().count() >= 5
            && (source.has_speech(before)
                || source.has_speech(after)
                || (before.trim().is_empty() && after.trim().is_empty())))
}

fn append_sentence(
    source: &Source<'_>,
    units: &mut Vec<Unit>,
    start: usize,
    end: usize,
    first_in_line: &mut bool,
) {
    let Some(sentence) = source.trim(start, end) else {
        return;
    };
    let text = &source.text[start..end];
    let mut stack = Vec::new();
    let mut open_at = 0;
    let mut next = 0;
    let mut previous_quote: Option<usize> = None;
    for (offset, ch) in text.char_indices() {
        if stack.last() == Some(&ch) {
            stack.pop();
            if stack.is_empty() {
                let open = open_at + text[open_at..].chars().next().unwrap().len_utf8();
                let end_quote = offset + ch.len_utf8();
                if is_dialogue(
                    source,
                    &text[open..offset],
                    &text[next..open_at],
                    &text[end_quote..],
                ) {
                    let evidence = source.span(start + open_at, start + end_quote);
                    let mut speech_context = sentence;
                    if source.actor_tokens.is_some() {
                        speech_context.start_char = source
                            .span(start + next, start + next)
                            .start_char
                            .max(sentence.start_char);
                        if let Some(index) = previous_quote {
                            units[index].speech_context.end_char = evidence.start_char;
                        }
                    }
                    append_narration(
                        source,
                        units,
                        &sentence,
                        start + next,
                        start + open_at,
                        first_in_line,
                    );
                    previous_quote = Some(units.len());
                    units.push(Unit {
                        evidence,
                        sentence,
                        speech_context,
                        quote: true,
                        first_in_line: std::mem::take(first_in_line),
                    });
                    next = end_quote;
                }
            }
        } else if let Some(close) = quote_end(ch)
            && text[offset + ch.len_utf8()..].contains(close)
        {
            if stack.is_empty() {
                open_at = offset;
            }
            stack.push(close);
        }
    }
    append_narration(source, units, &sentence, start + next, end, first_in_line);
}

fn split_units(
    source: &mut Source<'_>,
    title: &str,
    lexicon: &Lexicon<'_>,
    narrative: Option<&NarrativeAnalyzer<'_>>,
) -> Vec<Unit> {
    split_units_impl(source, title, lexicon, narrative, None)
}

pub(crate) struct AdmittedLine<'a> {
    pub(crate) start: usize,
    pub(crate) text: &'a str,
    removed_newline: bool,
}

// Share filtering between unit splitting and quote damage detection. `start`
// stays in the original input, including bytes skipped by earlier footers.
pub(crate) fn admitted_lines<'a>(
    text: &'a str,
    title: &'a str,
    lexicon: &'a Lexicon<'_>,
) -> impl Iterator<Item = AdmittedLine<'a>> + 'a {
    admitted_lines_with_footer_mode(text, title, lexicon, false)
}

// The scoped API folds both sides of chapter-footer comparisons. The legacy
// analyzer keeps its old behavior for compatibility with existing streams.
pub(crate) fn admitted_lines_scoped<'a>(
    text: &'a str,
    title: &'a str,
    lexicon: &'a Lexicon<'_>,
) -> impl Iterator<Item = AdmittedLine<'a>> + 'a {
    admitted_lines_with_footer_mode(text, title, lexicon, true)
}

fn admitted_lines_with_footer_mode<'a>(
    text: &'a str,
    title: &'a str,
    lexicon: &'a Lexicon<'_>,
    fold_marker: bool,
) -> impl Iterator<Item = AdmittedLine<'a>> + 'a {
    // ASCII folding preserves UTF-8 byte lengths, so matches index `text`.
    let folded_text = text.to_ascii_lowercase();
    let end = (if fold_marker {
        lexicon
            .footers
            .iter()
            .filter_map(|marker| {
                let folded_marker = marker.to_ascii_lowercase();
                folded_text.find(folded_marker.as_str())
            })
            .min()
    } else {
        earliest(&folded_text, lexicon.footers).map(|(offset, _)| offset)
    })
    .unwrap_or(text.len());
    let mut start = 0;
    text[..end]
        .split_inclusive('\n')
        .enumerate()
        .filter_map(move |(line_number, line)| {
            let line_start = start;
            start += line.len();
            let trimmed = line.trim_matches(|ch: char| ch.is_whitespace() || ch == '\u{feff}');
            if !title.is_empty()
                && line_number < 4
                && (trimmed == title
                    || trimmed
                        .strip_suffix(title)
                        .is_some_and(|prefix| (1..=24).contains(&prefix.trim().chars().count())))
            {
                return None;
            }
            let end = ["<script", "<div", "<a href", "<iframe", "<!--"]
                .iter()
                .chain(lexicon.line_footers.iter())
                .filter_map(|marker| line.find(marker))
                .min()
                .unwrap_or(line.len());
            Some(AdmittedLine {
                start: line_start,
                text: &line[..end],
                removed_newline: end < line.len() && line.ends_with('\n'),
            })
        })
}

fn split_units_impl(
    source: &mut Source<'_>,
    title: &str,
    lexicon: &Lexicon<'_>,
    narrative: Option<&NarrativeAnalyzer<'_>>,
    quote_damage: Option<usize>,
) -> Vec<Unit> {
    let text = source.text;
    let mut units = Vec::new();
    for admitted in admitted_lines(text, title, lexicon) {
        let line_start = admitted.start;
        let line = admitted.text;
        let end = line.len();
        if let Some(narrative) = narrative {
            let mut first_in_line = true;
            // Preserve a completed sentence even when the segmenter joins it
            // to a later damaged quote. Do not split an unfinished sentence.
            let safe_cut = quote_damage
                .and_then(|pos| source.bytes.get(pos).copied())
                .and_then(|byte| byte.checked_sub(line_start))
                .filter(|&cut| cut > 0 && cut < end)
                .filter(|&cut| {
                    line[..cut]
                        .trim_end()
                        .chars()
                        .last()
                        .is_some_and(|ch| matches!(ch, '。' | '！' | '？' | '!' | '?'))
                });
            if let Some(cut) = safe_cut {
                narrative.append_line(
                    source,
                    &mut units,
                    line_start,
                    &line[..cut],
                    &mut first_in_line,
                );
                narrative.append_line(
                    source,
                    &mut units,
                    line_start + cut,
                    &line[cut..end],
                    &mut first_in_line,
                );
            } else {
                narrative.append_line(
                    source,
                    &mut units,
                    line_start,
                    &line[..end],
                    &mut first_in_line,
                );
            }
            continue;
        }
        let mut first_in_line = true;
        for (left, right) in split_outside_quotes(&line[..end], |ch| {
            matches!(ch, '。' | '！' | '？' | '!' | '?')
        }) {
            append_sentence(
                source,
                &mut units,
                line_start + left,
                line_start + right,
                &mut first_in_line,
            );
        }
    }
    units
}

#[derive(Clone)]
struct Actor {
    name: String,
    evidence: Span,
}

fn first_name<'a>(text: &str, lexicon: &Lexicon<'a>) -> Option<(usize, &'a str, &'a str)> {
    lexicon
        .actors
        .iter()
        .filter_map(|&(word, name)| text.find(word).map(|offset| (offset, word, name)))
        .min_by_key(|(offset, word, _)| (*offset, std::cmp::Reverse(word.len())))
}

fn actor_at(source: &Source<'_>, base: usize, hit: (usize, &str, &str)) -> Actor {
    let (offset, word, name) = hit;
    Actor {
        name: name.to_owned(),
        evidence: source.span(base + offset, base + offset + word.len()),
    }
}

fn named_subject(source: &Source<'_>, span: &Span, lexicon: &Lexicon<'_>) -> Option<Actor> {
    let text = source.slice(span);
    let start = source.bytes[span.start_char];
    let Some(hit) = source.first_name(start, start + text.len(), lexicon) else {
        return title_subject(source, start, text, lexicon);
    };
    let prefix = &text[..hit.0];
    let suffix = &text[hit.0 + hit.1.len()..];
    if prefix.chars().count() <= 16
        && predicate(prefix, lexicon).is_none()
        && !contains_any(
            prefix,
            &[
                "把", "將", "讓", "望著", "看著", "陪著", "不負", "直奔", "幫助", "輔助", "原諒",
                "按照",
            ],
        )
        && !starts_any(
            suffix,
            &[
                "的功法",
                "的身影",
                "的事情",
                "的身體",
                "身側",
                "身后",
                "身後",
                "身邊",
                "所在",
                "所說",
            ],
        )
    {
        Some(actor_at(source, start, hit))
    } else {
        title_subject(source, start, text, lexicon)
    }
}

const SUBJECT_TITLES: &[&str] = &[
    "護法", "长老", "長老", "仙子", "真人", "老祖", "掌門", "宗主", "堂主", "道友", "居士", "尊者",
    "大師", "大夫", "老者", "道士", "童子", "魔君", "上人",
];

fn title_subject(
    source: &Source<'_>,
    base: usize,
    text: &str,
    lexicon: &Lexicon<'_>,
) -> Option<Actor> {
    let lead = text.trim_start_matches(|ch: char| {
        ch.is_whitespace() || matches!(ch, '，' | ',' | '。' | '：' | ':')
    });
    let lead = [
        "那位",
        "這位",
        "一位",
        "那名",
        "這名",
        "一名",
        "這時",
        "這時候",
    ]
    .iter()
    .find_map(|prefix| lead.strip_prefix(prefix))
    .unwrap_or(lead);
    let (offset, title) = earliest(lead, SUBJECT_TITLES)?;
    let qualifier = &lead[..offset];
    if !(1..=4).contains(&qualifier.chars().count())
        || !qualifier.chars().all(|ch| ('一'..='龯').contains(&ch))
        || qualifier
            .chars()
            .any(|ch| "的了著在有把與將還向那這是從問道告讓找由聽見".contains(ch))
    {
        return None;
    }
    let end = offset + title.len();
    let suffix = &lead[end..];
    let following = suffix
        .strip_prefix("突然")
        .or_else(|| suffix.strip_prefix("當即"))
        .or_else(|| suffix.strip_prefix("微微"))
        .or_else(|| suffix.strip_prefix("也"))
        .unwrap_or(suffix);
    let direct_predicate = predicate(following, lexicon).is_some_and(|(_, at, _)| at == 0)
        || (following.starts_with(['把', '將', '将']) && predicate(following, lexicon).is_some())
        || starts_any(following, SPEECH)
        || starts_any(following, &["聞言", "闻言", "笑了", "冷哼"]);
    if !direct_predicate
        && !suffix
            .trim_matches([',', '，', '。', ' ', '：', ':'])
            .is_empty()
    {
        return None;
    }
    let start = base + text.len() - lead.len();
    Some(Actor {
        name: lead[..end].to_owned(),
        evidence: source.span(start, start + end),
    })
}

fn named_speaker_after(
    source: &Source<'_>,
    start: usize,
    end: usize,
    lexicon: &Lexicon<'_>,
) -> Option<Actor> {
    let after = &source.text[start..end];
    if let Some(hit) = source.first_name(start, end, lexicon) {
        let phrase = &after[hit.0 + hit.1.len()..];
        if after[..hit.0].chars().count() < 6 && source.has_speech(phrase) {
            return Some(actor_at(source, start, hit));
        }
    }
    None
}

fn speech_actor(source: &Source<'_>, unit: &Unit, lexicon: &Lexicon<'_>) -> Option<Actor> {
    let left = source.bytes[unit.speech_context.start_char];
    let right = source.bytes[unit.speech_context.end_char];
    let start = source.bytes[unit.evidence.start_char];
    let end = source.bytes[unit.evidence.end_char];
    let before = &source.text[left..start];
    let after = &source.text[end..right];
    let named_before = || {
        let hit = source.first_name(left, start, lexicon)?;
        source
            .has_speech(&before[hit.0 + hit.1.len()..])
            .then(|| actor_at(source, left, hit))
    };
    let title_before = || {
        source
            .has_speech(before)
            .then(|| title_subject(source, left, before, lexicon))
            .flatten()
    };
    // A tag that introduces the next quote must not replace this quote's explicit speaker.
    if unit.speech_context.end_char < unit.sentence.end_char
        && let Some(actor) = named_before().or_else(title_before)
    {
        return Some(actor);
    }
    if let Some(actor) = named_speaker_after(source, end, right, lexicon) {
        return Some(actor);
    }
    if let Some(actor) = named_before() {
        return Some(actor);
    }
    if source.has_speech(after)
        && let Some(actor) = title_subject(source, end, after, lexicon)
    {
        return Some(actor);
    }
    if let Some(actor) = title_before() {
        return Some(actor);
    }
    let right = if unit.speech_context.end_char < unit.sentence.end_char {
        right
    } else {
        source.bytes[(unit.evidence.end_char + 165).min(source.bytes.len() - 1)]
    };
    let remainder = &source.text[end..right];
    if let Some((offset, _)) = remainder.match_indices("正是").next() {
        let lead = &remainder[..offset];
        let identity = remainder[offset..]
            .split(['。', '\n'])
            .next()
            .unwrap_or_default();
        if contains_any(lead, &["面容", "面孔", "容貌"])
            && let Some(hit) =
                source.first_name(end + offset, end + offset + identity.len(), lexicon)
        {
            return Some(actor_at(source, end + offset, hit));
        }
    }
    None
}

fn earliest<'a>(text: &str, words: &'a [&'a str]) -> Option<(usize, &'a str)> {
    words
        .iter()
        .filter_map(|word| text.find(word).map(|offset| (offset, *word)))
        .min_by_key(|(offset, word)| (*offset, std::cmp::Reverse(word.len())))
}

fn directly_attributed(
    source: &Source<'_>,
    unit: &Unit,
    actor: &Actor,
    lexicon: &Lexicon<'_>,
) -> bool {
    let tag_end = if actor.evidence.start_char >= unit.evidence.end_char {
        unit.speech_context.end_char
    } else {
        unit.evidence.start_char
    };
    if actor.evidence.start_char < unit.speech_context.start_char
        || actor.evidence.end_char > tag_end
    {
        return false;
    }
    let start = source.bytes[actor.evidence.end_char];
    let tag = &source.text[start..source.bytes[tag_end]];
    let cue = [earliest(tag, SPEECH), earliest(tag, SPEECH_CONTINUATION)]
        .into_iter()
        .flatten()
        .map(|(offset, _)| offset)
        .min();
    cue.is_some_and(|offset| {
        let lead = &tag[..offset];
        source.first_name(start, start + offset, lexicon).is_none()
            && !actual_dream(lead)
            && !contains_any(lead, PLANNED)
            && !contains_any(
                lead,
                &[
                    "聽", "听", "看著", "看着", "本想", "原想", "正想", "差點", "差点", "險些",
                    "险些",
                ],
            )
            && !ends_any(lead.trim_end(), &["想", "欲", "要"])
    })
}

fn trim_clause_end(text: &str) -> &str {
    text.trim_end_matches(|ch: char| ch.is_whitespace() || "。.!！,，;；…".contains(ch))
}

fn ends_with_question_particle(text: &str) -> bool {
    trim_clause_end(text).ends_with(['嗎', '吗'])
}

// Limit the entire suffix scan, including any leading spaces, not merely the
// eventual matched tag. The bound is in Unicode scalars, not UTF-8 bytes.
const MAX_CESSATION_SUFFIX_SCALARS: usize = 256;
const MAX_CESSATION_PREFIX_BYTES: usize = MAX_CESSATION_SUFFIX_SCALARS * 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CessationScan {
    Matched(Span),
    NoMatch,
    Unexamined(&'static str),
}

// This is only a local written boundary. It does not determine whether the
// quoted passage is narrated, imagined, or true in the wider story.
fn completed_prefix_allows_cessation(prefix: &str) -> bool {
    let prefix = prefix.trim();
    if prefix.is_empty() {
        return true;
    }
    if !prefix.ends_with(['。', '！', '？'])
        || prefix
            .chars()
            .any(|ch| quote_end(ch).is_some() || matches!(ch, '」' | '』' | '”' | '’'))
    {
        return false;
    }
    let before_terminator = &prefix[..prefix.len() - prefix.chars().last().unwrap().len_utf8()];
    let prior_sentence = before_terminator
        .rsplit(['。', '！', '？', '\n', '\r'])
        .next()
        .unwrap_or("");
    prior_sentence.chars().any(char::is_alphanumeric)
        && !actual_dream(prior_sentence)
        && !contains_any(prior_sentence, HYPOTHETICAL)
        && !starts_any(prior_sentence.trim_start(), &["若", "假使"])
        && !contains_any(prior_sentence, PLANNED)
}

#[cfg(test)]
mod cessation_completed_prefix_tests {
    use super::completed_prefix_allows_cessation;

    #[test]
    fn requires_a_real_completed_local_sentence_without_uncertain_scope() {
        for prefix in ["", " \t", "𠮷。", "旁白。𠮷。", "前文！", "前文？"] {
            assert!(completed_prefix_allows_cessation(prefix), "{prefix}");
        }
        for prefix in [
            "𠮷，",
            "𠮷：",
            "𠮷.",
            "。",
            "如果下雨。",
            "若下雨。",
            "他夢見山谷。",
            "「前文。」",
        ] {
            assert!(!completed_prefix_allows_cessation(prefix), "{prefix}");
        }
    }
}

// Match a bounded written continuation-stop construction after a quotation.
// A surface name is vocabulary supplied by the caller, not identity evidence.
// This recognizes the source form only: it does not establish that the written
// subject spoke the quotation, nor that the quotation's contents happened.
fn written_cessation_tag(
    source: &Source<'_>,
    unit: &Unit,
    context: &str,
    lexicon: &Lexicon<'_>,
) -> CessationScan {
    if !unit.quote || context != "main" {
        return CessationScan::NoMatch;
    }
    let sentence_start = source.bytes[unit.sentence.start_char];
    let quote_start = source.bytes[unit.evidence.start_char];
    let prefix = &source.text[sentence_start..quote_start];
    if prefix.len() > MAX_CESSATION_PREFIX_BYTES {
        return CessationScan::Unexamined("prefix_limit");
    }
    // Some segmenters retain a completed prior sentence in the same unit.
    if !completed_prefix_allows_cessation(prefix) {
        return CessationScan::NoMatch;
    }
    let quote_end_byte = source.bytes[unit.evidence.end_char];
    let after_quote = &source.text[quote_end_byte..];
    let mut terminal = None;
    for (count, (byte, ch)) in after_quote.char_indices().enumerate() {
        if count == MAX_CESSATION_SUFFIX_SCALARS {
            return CessationScan::Unexamined("suffix_limit");
        }
        if quote_end(ch).is_some() || matches!(ch, '」' | '』' | '”' | '’') {
            return CessationScan::NoMatch;
        }
        if matches!(ch, '。' | '.' | '！' | '!' | '？' | '?' | '\r' | '\n') {
            terminal = Some((byte, ch));
            break;
        }
    }
    let Some((terminal_byte, terminal_char)) = terminal else {
        // A completed quote followed only by EOF/spacing has no partially
        // written postposed tag to review. Its speaker remains unknown.
        if after_quote.trim_matches([' ', '\t']).is_empty() {
            return CessationScan::NoMatch;
        }
        return CessationScan::Unexamined("end_of_input");
    };
    if !matches!(terminal_char, '。' | '.') {
        return CessationScan::NoMatch;
    }
    let prefix = &after_quote[..terminal_byte];
    let clause = prefix.trim_start_matches([' ', '\t']);
    let tag_start = quote_end_byte + prefix.len() - clause.len();
    let mut matched: Option<&str> = None;
    for &(surface, _) in lexicon.actors {
        if surface.is_empty() || matched == Some(surface) {
            continue;
        }
        let Some(after_subject) = clause.strip_prefix(surface) else {
            continue;
        };
        let Some(after_stop) = after_subject.strip_prefix("停了") else {
            continue;
        };
        let Some(after_duration) = ["一會", "一会", "一下", "片刻"]
            .iter()
            .find_map(|duration| after_stop.strip_prefix(*duration))
        else {
            continue;
        };
        let Some(after_comma) = after_duration
            .strip_prefix('，')
            .or_else(|| after_duration.strip_prefix(','))
        else {
            continue;
        };
        if !matches!(
            after_comma.trim_start(),
            "沒再說下去" | "没再说下去" | "沒有再說下去" | "没有再说下去"
        ) {
            continue;
        }
        if matched.is_some() {
            return CessationScan::NoMatch;
        }
        matched = Some(surface);
    }
    if matched.is_none() {
        return CessationScan::NoMatch;
    }
    CessationScan::Matched(source.span(
        tag_start,
        quote_end_byte + terminal_byte + terminal_char.len_utf8(),
    ))
}

fn append_reported_actions(
    source: &Source<'_>,
    unit: &Unit,
    following: Option<&Unit>,
    actor: Option<&Actor>,
    lexicon: &Lexicon<'_>,
    events: &mut Vec<Event>,
    mut predicates: Option<&mut Vec<MentionPredicateObservation>>,
) {
    if actor.is_some_and(|actor| !directly_attributed(source, unit, actor, lexicon)) {
        return;
    }
    let start = source.bytes[unit.evidence.start_char + 1];
    let end = source.bytes[unit.evidence.end_char - 1];
    let body = &source.text[start..end];
    if body.contains(['？', '?'])
        || ends_with_question_particle(body)
        || contains_any(body, HYPOTHETICAL)
        || actual_dream(body)
    {
        return;
    }
    for (left, right) in split_outside_quotes(body, |ch| {
        matches!(ch, '，' | ',' | '；' | ';' | '。' | '！' | '？' | '!' | '?')
    }) {
        let Some(span) = source.trim(start + left, start + right) else {
            continue;
        };
        let clause = source.slice(&span);
        let Some((kind, offset, word)) = predicate(clause, lexicon) else {
            continue;
        };
        let lead = clause[..offset].trim();
        let Some(modifier) = ["我", "讓我", "让我"]
            .into_iter()
            .find_map(|subject| lead.strip_prefix(subject))
        else {
            continue;
        };
        let modifier = modifier.trim();
        let tail = &clause[offset + word.len()..];
        let experienced =
            matches!(modifier, "曾" | "曾經" | "曾经") && tail.starts_with(['過', '过']);
        if !matches!(
            modifier,
            "" | "曾"
                | "曾經"
                | "曾经"
                | "已經"
                | "已经"
                | "無意"
                | "无意"
                | "碰巧"
                | "剛才"
                | "刚才"
        ) || !(word.ends_with('了') || tail.starts_with('了') || experienced)
            || ends_with_question_particle(clause)
            || contains_any(clause, PLANNED)
            || realm(clause, body, lexicon).is_some()
        {
            continue;
        }
        let event_index = events.len();
        let predicate_span = predicates.as_ref().map(|_| {
            source.span(
                source.bytes[span.start_char] + offset,
                source.bytes[span.start_char] + offset + word.len(),
            )
        });
        events.push(Event {
            kind: kind.to_owned(),
            status: "reported".into(),
            actor: actor.map(|actor| actor.name.clone()),
            object: object_after(source, span, offset, word, kind, lexicon),
            action: excerpt(clause),
            evidence: actor.map_or_else(
                || {
                    let mut evidence = vec![span, unit.speech_context];
                    if let Some(next) = following
                        && !evidence.contains(&next.evidence)
                    {
                        evidence.push(next.evidence);
                    }
                    evidence
                },
                |actor| vec![span, actor.evidence],
            ),
            recap: true,
        });
        if let Some(predicate_span) = predicate_span {
            record_mention_predicate(
                predicates.as_deref_mut(),
                event_index,
                predicate_span,
                MentionPredicateBasis::ReportedClausePredicate,
            );
        }
    }
}

#[derive(Clone, Copy)]
struct QuotePredicate {
    start: usize,
    end: usize,
    kind: &'static str,
    rank_byte: Option<usize>,
    rank_end: Option<usize>,
}

// Extra written-form candidates belong to the source-mention path only.
// They do not establish an object, participant, or completed action.
const SOURCE_ACTION_CUES: &[&str] = &["拿起", "拾起", "舉起", "举起"];
const WRITTEN_GIVE_CUES: &[&str] = &[
    "交給", "交给", "送給", "送给", "贈給", "赠给", "贈予", "赠予", "遞給", "递给", "交付", "交予",
    "給予", "给予",
];
const WRITTEN_RECEIVE_CUES: &[&str] = &["收到", "收下", "接過", "接过", "獲得", "获得", "得到"];

fn source_kind(kind: &'static str, word: &str) -> &'static str {
    if kind == "acquisition" && WRITTEN_GIVE_CUES.contains(&word) {
        "transfer"
    } else {
        kind
    }
}

/// The written form of a transfer cue, not a verified transfer direction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WrittenTransferDirection {
    Give,
    Receive,
}

/// An exact written cue without any established event or participant binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WrittenTransferCue {
    pub cue: Span,
    pub direction: WrittenTransferDirection,
}

/// Find written predicate ranges once per clause. Overlapping vocabulary
/// matches prefer the longest form and a directly attached rank expression.
fn quote_predicates(text: &str, lexicon: &Lexicon<'_>) -> Vec<QuotePredicate> {
    let mut found = Vec::new();
    for (kind, words) in [
        ("movement", MOVEMENT),
        ("encounter", ENCOUNTER),
        ("decision", DECISION),
        ("combat", COMBAT),
        ("acquisition", ACQUISITION),
        ("action", ACTION),
        ("action", SOURCE_ACTION_CUES),
        ("transfer", WRITTEN_GIVE_CUES),
        ("acquisition", WRITTEN_RECEIVE_CUES),
    ] {
        for &word in words {
            for (start, _) in text.match_indices(word) {
                if kind == "movement"
                    && !movement_candidate_allowed(text, start, word, lexicon)
                    && !unverified_destination_before_abstract(text, start, word)
                {
                    continue;
                }
                found.push(QuotePredicate {
                    start,
                    end: start + word.len(),
                    kind: source_kind(kind, word),
                    rank_byte: None,
                    rank_end: None,
                });
            }
        }
    }
    for &(surface, _) in lexicon.ranks {
        if surface.is_empty() {
            continue;
        }
        for (rank_byte, _) in text.match_indices(surface) {
            if let Some(range) = rank_predicate(text, rank_byte) {
                found.push(QuotePredicate {
                    start: range.start,
                    end: range.end,
                    kind: "realm_change",
                    rank_byte: Some(rank_byte),
                    rank_end: Some(rank_byte + surface.len()),
                });
            }
        }
    }
    found.sort_unstable_by_key(|item| {
        (
            item.start,
            item.rank_byte.is_none(),
            std::cmp::Reverse(item.end - item.start),
            std::cmp::Reverse(item.rank_end.unwrap_or(item.end)),
        )
    });
    let mut result: Vec<QuotePredicate> = Vec::new();
    for item in found {
        if result.last().is_some_and(|prior| item.start < prior.end) {
            continue;
        }
        result.push(item);
    }
    result
}

/// Preserve a written destination-shaped phrase even when it is absent from
/// caller vocabulary. This is only an unverified source candidate.
fn unverified_destination_before_abstract(text: &str, offset: usize, word: &str) -> bool {
    let Some(target) = text.get(offset + word.len()..) else {
        return false;
    };
    target
        .char_indices()
        .take_while(|(byte, _)| *byte <= 256)
        .filter(|(_, ch)| matches!(ch, '後' | '后'))
        .any(|(boundary, ch)| {
            let written_target = target[..boundary].trim();
            let following: String = target[boundary + ch.len_utf8()..]
                .chars()
                .take(24)
                .collect();
            !written_target.is_empty()
                && written_target.chars().take(17).count() <= 16
                && !contains_any(
                    written_target,
                    &[
                        "沉思",
                        "夢境",
                        "梦境",
                        "睡夢",
                        "睡梦",
                        "修仙之路",
                        "話題",
                        "话题",
                        "幻覺",
                        "幻觉",
                    ],
                )
                && contains_any(
                    &following,
                    &[
                        "沉思",
                        "睡夢中",
                        "睡梦中",
                        "夢境中",
                        "梦境中",
                        "修仙之路",
                        "修煉狀態",
                        "修炼状态",
                        "話題",
                        "话题",
                    ],
                )
        })
}

/// Preserve written action candidates in a quotation without assigning a
/// participant, asserting occurrence, or merging them with other events.
fn append_quote_source_candidates(
    source: &Source<'_>,
    unit: &Unit,
    lexicon: &Lexicon<'_>,
    events: &mut Vec<Event>,
    first_event: usize,
    primary_realm_byte: Option<usize>,
    mut predicates: Option<&mut Vec<MentionPredicateObservation>>,
) {
    let start = source.bytes[unit.evidence.start_char + 1];
    let end = source.bytes[unit.evidence.end_char - 1];
    let body = &source.text[start..end];
    let seen: HashSet<_> = events[first_event..]
        .iter()
        .filter_map(|event| event.evidence.first())
        .map(|span| (span.start_char, span.end_char))
        .collect();
    for (left, right) in split_outside_quotes(body, |ch| {
        matches!(ch, '，' | ',' | '；' | ';' | '。' | '！' | '？' | '!' | '?')
    }) {
        let Some(span) = source.trim(start + left, start + right) else {
            continue;
        };
        let clause = source.slice(&span);
        let candidates = quote_predicates(clause, lexicon);
        let existing_clause = seen.contains(&(span.start_char, span.end_char));
        for (index, candidate) in candidates.iter().enumerate() {
            let rank_byte = candidate
                .rank_byte
                .map(|byte| source.bytes[span.start_char] + byte);
            if rank_byte.is_some() && rank_byte == primary_realm_byte {
                continue;
            }
            if existing_clause && index == 0 {
                continue;
            }
            let start = source.bytes[span.start_char] + candidate.start;
            let predicate = source.span(start, source.bytes[span.start_char] + candidate.end);
            let end = candidates
                .get(index + 1)
                .map_or(clause.len(), |next| next.start);
            let specific = candidate
                .rank_end
                .filter(|&rank_end| {
                    candidate.end <= rank_end
                        && rank_end <= end
                        && clause.get(candidate.start..rank_end).is_some()
                })
                .map(|rank_end| source.span(start, source.bytes[span.start_char] + rank_end))
                .unwrap_or(predicate);
            let evidence = if specific == predicate {
                vec![predicate, span]
            } else {
                vec![specific, predicate, span]
            };
            let event_index = events.len();
            events.push(Event {
                kind: candidate.kind.to_owned(),
                status: "uncertain".into(),
                actor: None,
                object: None,
                action: excerpt(&clause[candidate.start..end]),
                evidence,
                recap: false,
            });
            record_mention_predicate(
                predicates.as_deref_mut(),
                event_index,
                predicate,
                MentionPredicateBasis::AdditionalWrittenCue,
            );
        }
    }
}

/// Retain additional written predicates in narration without binding roles or
/// asserting events. Complete quoted ranges are handled separately.
// Keep the source, parsing inputs, and output sidecar explicit at this boundary.
#[allow(clippy::too_many_arguments)]
fn append_narration_source_candidates(
    source: &Source<'_>,
    unit: &Unit,
    text: &str,
    quotes: &[std::ops::Range<usize>],
    lexicon: &Lexicon<'_>,
    events: &mut Vec<Event>,
    first_event: usize,
    primary_predicate: Option<(usize, &'static str)>,
    primary_realm_byte: Option<usize>,
    mut predicates: Option<&mut Vec<MentionPredicateObservation>>,
) {
    let unit_start = source.bytes[unit.evidence.start_char];
    let seen: HashSet<_> = events[first_event..]
        .iter()
        .filter_map(|event| event.evidence.first())
        .map(|span| (span.start_char, span.end_char))
        .collect();
    let mut cursor = 0;
    let mut segments = Vec::new();
    for quote in quotes {
        if cursor < quote.start {
            segments.push(cursor..quote.start);
        }
        cursor = quote.end;
    }
    if cursor < text.len() {
        segments.push(cursor..text.len());
    }
    for segment in segments {
        let body = &text[segment.clone()];
        let candidates = quote_predicates(body, lexicon);
        for (index, candidate) in candidates.iter().enumerate() {
            let relative = segment.start + candidate.start;
            if primary_predicate
                .is_some_and(|(offset, kind)| offset == relative && kind == candidate.kind)
            {
                continue;
            }
            let rank_byte = candidate
                .rank_byte
                .map(|byte| unit_start + segment.start + byte);
            if rank_byte.is_some() && rank_byte == primary_realm_byte {
                continue;
            }
            let begin = unit_start + relative;
            let end = unit_start + segment.start + candidate.end;
            let predicate = source.span(begin, end);
            if seen.contains(&(predicate.start_char, predicate.end_char)) {
                continue;
            }
            let next = candidates
                .get(index + 1)
                .map_or(body.len(), |next| next.start);
            let context = source.span(unit_start + segment.start, unit_start + segment.end);
            let evidence = if context == predicate {
                vec![predicate]
            } else {
                vec![predicate, context]
            };
            let event_index = events.len();
            events.push(Event {
                kind: candidate.kind.to_owned(),
                status: "uncertain".into(),
                actor: None,
                object: None,
                action: excerpt(&body[candidate.start..next]),
                evidence,
                recap: false,
            });
            record_mention_predicate(
                predicates.as_deref_mut(),
                event_index,
                predicate,
                MentionPredicateBasis::AdditionalWrittenCue,
            );
        }
    }
}

fn movement_candidate_allowed(
    text: &str,
    offset: usize,
    word: &str,
    lexicon: &Lexicon<'_>,
) -> bool {
    let target = &text[offset + word.len()..];
    if word == "出了" && place_name(target, lexicon).is_none() {
        return false;
    }
    let prefix: String = target.chars().take(14).collect();
    let abstract_targets = [
        "睡夢中",
        "睡梦中",
        "夢境中",
        "梦境中",
        "沉思",
        "修煉狀態",
        "修炼状态",
        "修仙之路",
        "歧途",
        "話題",
        "话题",
        "軌道",
        "轨道",
    ];
    let Some((abstract_at, _)) = earliest(&prefix, &abstract_targets) else {
        return true;
    };
    // A separate later clause must not erase an earlier written place.
    // A place inside a dream or topic is not by itself a destination.
    earliest(&prefix, lexicon.places).is_some_and(|(place_at, place)| {
        let place_end = place_at + place.len();
        if place_end > abstract_at {
            return false;
        }
        let between = prefix[place_end..abstract_at].trim_start();
        starts_any(between, &["後", "后", "之後", "之后", "以後", "以后"])
            && !starts_any(between, &["後的", "后的", "後面", "后面"])
    })
}

fn predicate(text: &str, lexicon: &Lexicon<'_>) -> Option<(&'static str, usize, &'static str)> {
    [
        ("movement", MOVEMENT),
        ("encounter", ENCOUNTER),
        ("decision", DECISION),
        ("combat", COMBAT),
        ("acquisition", ACQUISITION),
        ("action", ACTION),
    ]
    .into_iter()
    .filter_map(|(kind, terms)| {
        earliest(text, terms)
            .filter(|(offset, word)| {
                if kind != "movement" {
                    return true;
                }
                movement_candidate_allowed(text, *offset, word, lexicon)
            })
            .map(|(offset, word)| (kind, offset, word))
    })
    .min_by_key(|(_, offset, word)| (*offset, std::cmp::Reverse(word.len())))
}

/// Detect an explicit negation or near miss in the same written clause as a
/// predicate. An uncertain classification is deliberate: double negation and
/// narrow or ambiguous scope cannot establish a positive or failed event.
/// Adversative clause boundaries stop an earlier unrelated negation from
/// changing a subsequent affirmative predicate.
fn predicate_has_unasserted_scope(text: &str, predicate_byte: usize) -> bool {
    let Some(prefix) = text.get(..predicate_byte) else {
        return true;
    };
    let mut local = prefix;
    for boundary in [
        "但是", "然而", "不過", "不过", "於是", "于是", "卻", "却", "但", "，", ",", "；", ";",
        "。", "！", "？", "\n", "\r",
    ] {
        if let Some((_, after)) = local.rsplit_once(boundary) {
            local = after;
        }
    }
    if contains_any(
        local,
        &[
            "沒有",
            "没有",
            "沒",
            "没",
            "未曾",
            "不曾",
            "尚未",
            "並未",
            "并未",
            "從未",
            "从未",
            "未能",
            "不能",
            "無法",
            "无法",
            "不會",
            "不会",
            "未",
            "差點",
            "差点",
            "差一點",
            "差一点",
            "險些",
            "险些",
            "試圖",
            "试图",
            "嘗試",
            "尝试",
            "以為",
            "以为",
            "原本要",
            "本來要",
            "本来要",
            "原先要",
        ],
    ) {
        return true;
    }
    let nonnegative = [
        "不但",
        "不僅",
        "不仅",
        "不只",
        "不光",
        "不時",
        "不时",
        "不由",
        "不免",
        "不久",
        "不慎",
        "不小心",
        "不經意",
        "不经意",
        "不知不覺",
        "不知不觉",
        "不約而同",
        "不约而同",
    ];
    local.match_indices('不').any(|(at, _)| {
        !nonnegative.iter().any(|word| {
            local
                .match_indices(word)
                .any(|(start, found)| start <= at && at < start + found.len())
        })
    })
}

/// Exact byte ranges of complete quoted passages in a single unit. A damaged
/// delimiter or ambiguous ASCII escape has no safe segmentation.
fn written_quote_ranges(text: &str) -> Option<Vec<std::ops::Range<usize>>> {
    let mut stack = Vec::<char>::new();
    let mut start = 0;
    let mut ranges = Vec::new();
    for (offset, ch) in text.char_indices() {
        if stack.last() == Some(&ch) {
            stack.pop();
            if stack.is_empty() {
                ranges.push(start..offset + ch.len_utf8());
            }
        } else if ch == '\\' && stack.last() == Some(&'"') {
            return None;
        } else if let Some(end) = quote_end(ch) {
            if stack.is_empty() {
                start = offset;
            }
            stack.push(end);
        } else if matches!(ch, '」' | '』' | '”' | '’') {
            return None;
        }
    }
    stack.is_empty().then_some(ranges)
}

/// Remove complete quoted passages from scope checks without joining words
/// across the removed text. An invalid boundary remains unknown.
fn unquoted_scope_text(text: &str) -> Option<std::borrow::Cow<'_, str>> {
    let quotes = written_quote_ranges(text)?;
    if quotes.is_empty() {
        return Some(std::borrow::Cow::Borrowed(text));
    }
    let mut output = String::with_capacity(text.len());
    let mut start = 0;
    for quote in quotes {
        output.push_str(&text[start..quote.start]);
        output.push('，');
        start = quote.end;
    }
    output.push_str(&text[start..]);
    Some(std::borrow::Cow::Owned(output))
}

/// A trailing written question marker outside every complete quotation. Its
/// exact source span is retained; this does not resolve the marker's scope.
fn outside_question_cue(
    text: &str,
    quotes: &[std::ops::Range<usize>],
) -> Option<std::ops::Range<usize>> {
    let trimmed = trim_clause_end(text);
    let (last_start, last) = trimmed.char_indices().next_back()?;
    let start = if matches!(last, '？' | '?') {
        trimmed[..last_start]
            .char_indices()
            .next_back()
            .filter(|(_, ch)| matches!(ch, '嗎' | '吗'))
            .map_or(last_start, |(start, _)| start)
    } else if matches!(last, '嗎' | '吗') {
        last_start
    } else {
        return None;
    };
    let cue = start..trimmed.len();
    (!quotes
        .iter()
        .any(|quote| cue.start < quote.end && quote.start < cue.end))
    .then_some(cue)
}

const WRITTEN_MODALITY_CUES: &[&str] = &[
    "可能",
    "聲稱",
    "声称",
    "自稱",
    "自称",
    "假稱",
    "假称",
    "否認",
    "否认",
    "猜測",
    "猜测",
    "想像",
    "想象",
    "懷疑",
    "怀疑",
    "說他",
    "说他",
    "說她",
    "说她",
    "說自己",
    "说自己",
];

// Additional source-mode observations are unverified wording only. Keep the
// legacy analyze entry unchanged until its separate compatibility review.
const SOURCE_ONLY_MODALITY_CUES: &[&str] = &[
    "假裝", "假装", "佯裝", "佯装", "謊稱", "谎称", "佯稱", "佯称",
];

fn extent_possible(text: &str, start: usize) -> bool {
    ["盡", "尽", "儘"]
        .iter()
        .any(|prefix| text[..start].ends_with(prefix))
        && text.get(start + "可能".len()..).is_some_and(|tail| {
            ["快地", "快速地", "迅速地", "早地", "慢地"]
                .iter()
                .any(|suffix| tail.starts_with(suffix))
        })
}

/// Find a written attempt cue in the predicate's punctuation-delimited clause,
/// excluding quoted text. The association is conservative and unverified.
fn outside_attempt_cue(
    text: &str,
    quotes: &[std::ops::Range<usize>],
    predicate: usize,
    prefix_words: &[&str],
    suffix_words: &[&str],
) -> Option<std::ops::Range<usize>> {
    if !text.is_char_boundary(predicate) || predicate >= text.len() {
        return None;
    }
    let mut clause_start = 0;
    let mut clause_end = text.len();
    let outside = |range: &std::ops::Range<usize>| {
        let index = quotes.partition_point(|quote| quote.end <= range.start);
        quotes
            .get(index)
            .is_none_or(|quote| range.end <= quote.start)
    };
    for (at, ch) in text.char_indices() {
        if !outside(&(at..at + ch.len_utf8())) {
            continue;
        }
        if matches!(
            ch,
            '，' | ',' | '；' | ';' | '。' | '！' | '？' | '!' | '?' | '\r' | '\n'
        ) {
            if at < predicate {
                clause_start = at + ch.len_utf8();
            } else {
                clause_end = at;
                break;
            }
        }
    }
    for marker in ["但是", "然而", "不過", "不过", "卻", "却", "但"] {
        for (at, _) in text.match_indices(marker) {
            let end = at + marker.len();
            if !outside(&(at..end)) {
                continue;
            }
            if marker == "但" && (text[..at].ends_with('不') || text[at..].starts_with("但凡"))
            {
                continue;
            }
            if end <= predicate && end > clause_start {
                clause_start = end;
            } else if at >= predicate && at < clause_end {
                clause_end = at;
            }
        }
    }
    for &word in prefix_words {
        for (relative, _) in text[clause_start..predicate].match_indices(word) {
            let start = clause_start + relative;
            let range = start..start + word.len();
            // In "盡可能快地", possible describes extent, not event modality.
            if word == "可能" && extent_possible(text, start) {
                continue;
            }
            if outside(&range) {
                return Some(range);
            }
        }
    }
    for &word in suffix_words {
        for (relative, _) in text[predicate..clause_end].match_indices(word) {
            let start = predicate + relative;
            let range = start..start + word.len();
            if outside(&range) {
                return Some(range);
            }
        }
    }
    None
}

/// A belief cue followed by a clause separator may cover later clauses in the
/// same sentence. Adversatives and hard boundaries stop this conservative hint.
fn earlier_written_scope_cue(text: &str, words: &[&str]) -> Option<std::ops::Range<usize>> {
    let quotes = written_quote_ranges(text)?;
    let outside = |range: &std::ops::Range<usize>| {
        !quotes
            .iter()
            .any(|quote| range.start < quote.end && quote.start < range.end)
    };
    let mut selected: Option<std::ops::Range<usize>> = None;
    for &word in words {
        for (start, _) in text.match_indices(word) {
            let end = start + word.len();
            if !outside(&(start..end)) {
                continue;
            }
            if word == "可能" && extent_possible(text, start) {
                continue;
            }
            // The believed clause can contain words before the first comma.
            // Keep the later action uncertain when that scope is unresolved.
            if !text[end..].char_indices().any(|(relative, ch)| {
                matches!(ch, '，' | ',' | '：' | ':')
                    && outside(&(end + relative..end + relative + ch.len_utf8()))
            }) {
                continue;
            }
            let blocked = text[end..].char_indices().any(|(relative, ch)| {
                matches!(
                    ch,
                    '。' | '！' | '？' | '!' | '?' | '；' | ';' | '\r' | '\n'
                ) && outside(&(end + relative..end + relative + ch.len_utf8()))
            }) || ["但是", "然而", "不過", "不过", "卻", "却", "但", "可是"]
                .iter()
                .any(|marker| {
                    text[end..].match_indices(marker).any(|(relative, _)| {
                        let at = end + relative;
                        outside(&(at..at + marker.len()))
                            && !(*marker == "但"
                                && (text[..at].ends_with('不') || text[at..].starts_with("但凡")))
                    })
                });
            if !blocked && selected.as_ref().is_none_or(|cue| start > cue.start) {
                selected = Some(start..end);
            }
        }
    }
    selected
}

fn unquoted_conditional_cue(text: &str) -> Option<(usize, &'static str)> {
    let quotes = written_quote_ranges(text)?;
    let mut start = 0;
    for quote in &quotes {
        let gap = &text[start..quote.start];
        if let Some((offset, word)) = earliest(gap, HYPOTHETICAL).or_else(|| {
            ["若", "假使"]
                .into_iter()
                .find(|cue| gap.starts_with(cue))
                .map(|cue| (0, cue))
        }) {
            return Some((start + offset, word));
        }
        start = quote.end;
    }
    let gap = &text[start..];
    earliest(gap, HYPOTHETICAL)
        .or_else(|| {
            ["若", "假使"]
                .into_iter()
                .find(|cue| gap.starts_with(cue))
                .map(|cue| (0, cue))
        })
        .map(|(offset, word)| (start + offset, word))
}

/// Predicate and source extent within narration only. The quoted passage is
/// never used as the predicate or object of the enclosing speaker's action.
fn predicate_outside_quotes(
    text: &str,
    quotes: &[std::ops::Range<usize>],
    lexicon: &Lexicon<'_>,
) -> Option<(&'static str, usize, &'static str, std::ops::Range<usize>)> {
    let mut start = 0;
    for quote in quotes {
        let end = quote.start;
        let segment = &text[start..end];
        if let Some((kind, offset, word)) = predicate(segment, lexicon) {
            return Some((kind, start + offset, word, start..end));
        }
        start = quote.end;
    }
    predicate(&text[start..], lexicon)
        .map(|(kind, offset, word)| (kind, start + offset, word, start..text.len()))
}

fn outside_quote_segment(
    text: &str,
    quotes: &[std::ops::Range<usize>],
    at: usize,
) -> Option<std::ops::Range<usize>> {
    let mut start = 0;
    for quote in quotes {
        if start <= at && at < quote.start {
            return Some(start..quote.start);
        }
        if quote.start <= at && at < quote.end {
            return None;
        }
        start = quote.end;
    }
    (start <= at && at < text.len()).then_some(start..text.len())
}

/// An explicit hearing cue adjacent to a complete quotation. The cue and
/// written quote remain separate evidence. This does not assert the quote's
/// contents or identify its speaker.
fn hearing_quote(
    text: &str,
    quotes: &[std::ops::Range<usize>],
) -> Option<(std::ops::Range<usize>, std::ops::Range<usize>)> {
    const HEARING: &[&str] = &["聽到了", "听到了", "聽見", "听见", "聽到", "听到"];
    let first = quotes.first()?;
    let before = &text[..first.start];
    let (at, cue) = earliest(before, HEARING)?;
    if !before[at + cue.len()..].trim().is_empty() {
        return None;
    }
    Some((at..at + cue.len(), first.clone()))
}

fn realm_outside_quotes<'a>(
    text: &str,
    quotes: &[std::ops::Range<usize>],
    lexicon: &Lexicon<'a>,
) -> Option<Realm<'a>> {
    let mut start = 0;
    for quote in quotes {
        let segment = &text[start..quote.start];
        if let Some(mut hit) = realm(segment, segment, lexicon) {
            hit.offset += start;
            hit.end_offset += start;
            return Some(hit);
        }
        start = quote.end;
    }
    realm(&text[start..], &text[start..], lexicon).map(|mut hit| {
        hit.offset += start;
        hit.end_offset += start;
        hit
    })
}

fn place_name<'a>(text: &str, lexicon: &Lexicon<'a>) -> Option<&'a str> {
    earliest(text, lexicon.places).map(|(_, place)| place)
}

const RANK_ADVANCEMENT: &[&str] = &[
    "進入", "进入", "進階", "进阶", "晉階", "晋阶", "突破", "跨入", "踏入", "踏進", "邁入", "達",
    "躋身",
];

fn rank_predicate(text: &str, rank_offset: usize) -> Option<std::ops::Range<usize>> {
    let prefix = text
        .get(..rank_offset)?
        .trim_end_matches(['了', '的', '到', '至']);
    for word in ["到達", "抵達", "到达", "抵达"] {
        if prefix.ends_with(word) {
            return Some(prefix.len() - word.len()..prefix.len());
        }
    }
    RANK_ADVANCEMENT.iter().find_map(|word| {
        prefix
            .ends_with(word)
            .then(|| prefix.len() - word.len()..prefix.len())
    })
}

struct Realm<'a> {
    offset: usize,
    end_offset: usize,
    target: &'a str,
    outcome: &'static str,
}

fn realm<'a>(text: &str, local_sentence: &str, lexicon: &Lexicon<'a>) -> Option<Realm<'a>> {
    let candidates = lexicon.ranks.iter().flat_map(|&(word, target)| {
        text.match_indices(word).filter_map(move |(offset, _)| {
            let before = &text[..offset];
            let after = &text[offset + word.len()..];
            let tail = after.strip_prefix('期').unwrap_or(after);
            if after.starts_with('丹')
                || ends_any(before, &["第二", "主"])
                || ends_any(before, lexicon.rank_blocked_prefixes)
                || contains_any(before, &["侵占", "奪舍", "夺舍", "同化", "反被", "抹去"])
                || starts_any(tail, &["成功率", "幾率", "率", "的希望", "的關鍵", "經驗"])
                || contains_any(
                    tail,
                    &["成功的關鍵", "成功的条件", "成功的條件", "成功的幾率"],
                )
            {
                return None;
            }
            let advancement = RANK_ADVANCEMENT.iter().copied().find(|verb| {
                before
                    .trim_end_matches(['了', '的', '到', '至'])
                    .ends_with(verb)
            });
            let assertion_scope = before
                .rsplit(['。', '！', '？', '!', '?', '，', ','])
                .next()
                .unwrap_or_default();
            let ability = contains_any(
                assertion_scope,
                &[
                    "如何", "就可", "便可", "方可", "方能", "就能", "才可", "才能", "可以", "能夠",
                    "能够", "有望", "需要", "足以", "想要", "辦法", "方法", "秘訣", "秘方", "以期",
                    "倘若", "假如",
                ],
            ) || starts_any(tail, &["的方法", "的辦法", "的途徑", "的秘訣", "之前"]);
            let realized = advancement.is_some_and(|verb| {
                let without_end = before.trim_end_matches(['了', '的', '到', '至']);
                let prefix = without_end.strip_suffix(verb).unwrap_or_default();
                before[without_end.len()..].contains('了')
                    || ends_any(
                        prefix,
                        &[
                            "終于", "終於", "终于", "總算", "已", "已經", "已然", "真的", "當真",
                            "果然", "竟然", "這才", "才", "遂",
                        ],
                    )
                    || starts_any(tail, &["了", "成功"])
            });
            let outcome = if starts_any(
                tail,
                &["未成", "未成功", "沒成", "失敗", "無望", "根本無望"],
            ) || ends_any(before, &["未能", "沒能", "無法", "未曾"])
            {
                "failed"
            } else if ability && (advancement.is_some() || starts_any(tail, &["成功", "大成"]))
            {
                "hypothetical"
            } else if (lexicon.formed_ranks.contains(&word)
                && ends_any(before, &["來結成", "来结成"]))
                || starts_any(tail, &["沖關", "衝關"])
                || contains_any(before, PLANNED)
                || ends_any(before, &["要", "想", "開始", "試著", "嘗試", "趕緊"])
            {
                "planned"
            } else if starts_any(tail, &["成功", "大成"])
                || realized
                || (lexicon.formed_ranks.contains(&word)
                    && ends_any(before, &["結成了", "結成", "结成了", "结成"]))
                || (starts_any(after, &["期", "境界"])
                    && contains_any(local_sentence, &["沖關成功", "衝關成功", "突破成功"])
                    && ends_any(before, &["是", "成為"]))
            {
                "occurred"
            } else if advancement.is_some() {
                "uncertain"
            } else {
                return None;
            };
            Some(Realm {
                offset,
                end_offset: offset + word.len(),
                target,
                outcome,
            })
        })
    });
    candidates.min_by_key(|hit| (!matches!(hit.outcome, "occurred" | "failed"), hit.offset))
}

fn temporal_shift(text: &str) -> Option<&'static str> {
    let text = text
        .trim_start_matches(|ch: char| matches!(ch, '…' | '。' | '“' | '「') || ch.is_whitespace());
    if starts_any(
        text,
        &[
            "次日",
            "翌日",
            "第二天",
            "第三天",
            "隔天",
            "轉眼",
            "沒過多久",
            "不久后",
            "不久後",
            "片刻后",
            "片刻後",
            "這一日",
            "到了晚上",
            "夜深",
            "到了傍晚",
        ],
    ) {
        return Some("main");
    }
    let mut chars = text.chars().peekable();
    let mut counted = false;
    while chars.peek().is_some_and(|ch| {
        "一二三四五六七八九十百千萬兩幾數半多個个余餘零〇第".contains(*ch) || ch.is_ascii_digit()
    }) {
        counted = true;
        chars.next();
    }
    let unit = chars.next()?;
    if !counted || !"年月日時天刻晚".contains(unit) {
        return None;
    }
    match chars.next() {
        Some('前') => Some("flashback"),
        Some('後' | '后') => Some("main"),
        _ if text.starts_with('第') && "日天晚".contains(unit) => Some("main"),
        _ => None,
    }
}

fn actual_dream(text: &str) -> bool {
    contains_any(text, DREAM) || (text.starts_with('在') && text.contains("睡夢中"))
}

fn context_for(text: &str, previous: &'static str) -> &'static str {
    if actual_dream(text) {
        "dream"
    } else if (previous == "dream"
        && (contains_any(text, &["醒了", "醒來", "夢醒", "梦醒", "早上起來"])
            || temporal_shift(text) == Some("main")))
        || starts_any(
            text,
            &["回過神", "回过神", "回到眼前", "眼下", "如今", "現在"],
        )
    {
        "main"
    } else if contains_any(
        text,
        &["回想著", "回想起", "回憶起", "回忆起", "想起了", "想起當年"],
    ) || starts_any(text, &["當年", "早在", "多年前"])
        || temporal_shift(text) == Some("flashback")
    {
        "flashback"
    } else if starts_any(text, REPORTED) {
        "reported"
    } else if matches!(previous, "flashback" | "dream") {
        previous
    } else {
        "main"
    }
}

fn context_for_unit(unit: &Unit, text: &str, previous: &'static str) -> &'static str {
    if !unit.quote
        && (unit.first_in_line
            || actual_dream(text)
            || (previous == "dream" && contains_any(text, &["醒來", "醒了", "早上起來"]))
            || contains_any(text, &["回想著", "回想起", "想起了", "回憶起"]))
    {
        context_for(text, previous)
    } else {
        previous
    }
}

fn event_status(
    text: &str,
    earlier: &str,
    context: &str,
    quote: bool,
    known_actor: bool,
    realm: Option<&Realm<'_>>,
    kind: &str,
) -> &'static str {
    if context == "dream" {
        "dream"
    } else if realm.is_some_and(|hit| hit.outcome == "hypothetical")
        || contains_any(text, HYPOTHETICAL)
        || starts_any(text, &["若", "假使"])
        || ((contains_any(earlier, HYPOTHETICAL) || starts_any(earlier, &["若", "假使"]))
            && !starts_any(
                text,
                &["但", "然而", "不過", "結果", "最終", "終于", "終於"],
            ))
    {
        "hypothetical"
    } else if realm.is_some_and(|hit| hit.outcome == "failed")
        || (realm.is_none() && starts_any(text, FAILURE))
    {
        "failed"
    } else if (contains_any(text, PLANNED) && kind != "decision")
        || realm.is_some_and(|hit| hit.outcome == "planned")
    {
        "planned"
    } else if context == "reported"
        || contains_any(text, REPORTED)
        || (quote && (!known_actor || realm.is_none()))
    {
        "reported"
    } else if !known_actor || realm.is_some_and(|hit| hit.outcome == "uncertain") {
        "uncertain"
    } else {
        "occurred"
    }
}

fn object_after(
    source: &Source<'_>,
    evidence: Span,
    offset: usize,
    word: &str,
    kind: &str,
    lexicon: &Lexicon<'_>,
) -> Option<String> {
    let start = source.bytes[evidence.start_char] + offset + word.len();
    let end = source.bytes[evidence.end_char];
    let tail = &source.text[start..end];
    if kind == "movement" {
        return place_name(tail, lexicon).map(str::to_owned);
    }
    if matches!(kind, "encounter" | "combat")
        && let Some((_, _, name)) = source.first_name(start, end, lexicon)
    {
        return Some(name.into());
    }
    lexicon
        .objects
        .iter()
        .filter_map(|&(surface, canonical)| {
            tail.find(surface).map(|index| (index, surface, canonical))
        })
        .min_by_key(|(index, surface, _)| (*index, std::cmp::Reverse(surface.len())))
        .map(|(_, _, canonical)| canonical.to_owned())
}

/// Source ranges for a structured event in an analysis.
///
/// The linked event contains parser hints, not verified occurrence or identity.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ClauseFrame {
    /// Index into the analysis's event list.
    pub event_index: usize,
    /// The original verb or verb phrase.
    pub predicate: Span,
    /// The source range of a single written subject, even if its identity is unknown.
    /// Compound or unclear subjects have no range.
    pub actor_surface: Option<Span>,
    /// The person or object that the action affects.
    pub patient: Span,
    /// The stated result, when the text links it to the action.
    pub result: Option<Span>,
}

struct DisposalFrame<'lexicon> {
    kind: &'static str,
    verb: &'static str,
    predicate: Span,
    actor_surface: Option<Span>,
    actor: Option<Actor>,
    inherit_actor: bool,
    patient: Span,
    patient_name: Option<&'lexicon str>,
    result: Option<Span>,
    completed: bool,
    blocked: bool,
}

fn nominal_measure(text: &str) -> Option<&str> {
    let rest =
        text.trim_start_matches(|ch: char| "一二三四五六七八九十兩两幾几0123456789".contains(ch));
    (rest.len() < text.len()).then_some(())?;
    [
        '扇', '本', '卷', '條', '条', '張', '张', '塊', '块', '片', '枚', '個', '个', '只', '件',
        '根', '盞', '盏', '把', '面', '台', '臺',
    ]
    .into_iter()
    .find_map(|classifier| rest.strip_prefix(classifier))
    .map(str::trim_start)
}

fn result_value(source: &Source<'_>, start: usize, end: usize, markers: &[&str]) -> Option<Span> {
    let text = source.text[start..end].trim_start();
    if text.contains(['？', '?'])
        || ends_with_question_particle(text)
        || trim_clause_end(text).ends_with('吧')
    {
        return None;
    }
    let text = text.strip_prefix('了').unwrap_or(text).trim_start();
    let tail = markers
        .iter()
        .find_map(|marker| text.strip_prefix(marker))?;
    let value = trim_clause_end(tail.trim_start());
    if !value.chars().any(char::is_alphanumeric)
        || value.chars().count() > 24
        || ends_any(value, &["次", "遍", "趟", "回"])
        || value.chars().any(|ch| quote_end(ch).is_some())
        || contains_any(value, HYPOTHETICAL)
    {
        return None;
    }
    let start = end - tail.trim_start().len();
    Some(source.span(start, start + value.len()))
}

const SUBJECT_CLASSIFIERS: &[&str] = &[
    "一名", "一位", "這名", "这名", "這位", "这位", "那名", "那位", "這個", "这个", "那個", "那个",
];

fn participant_description(text: &str) -> bool {
    let role = SUBJECT_CLASSIFIERS
        .iter()
        .find_map(|prefix| text.strip_prefix(prefix))
        .unwrap_or(text);
    matches!(
        role,
        "人" | "陌生人"
            | "匠人"
            | "守衛"
            | "守卫"
            | "侍者"
            | "工人"
            | "士兵"
            | "修士"
            | "學生"
            | "学生"
            | "老師"
            | "老师"
            | "旅人"
            | "行人"
            | "孩子"
    )
}

fn coordinated_actors(text: &str, lexicon: &Lexicon<'_>) -> bool {
    text.split_once(['和', '與', '与', '跟', '及'])
        .is_some_and(|(left, right)| {
            [left.trim(), right.trim()]
                .into_iter()
                .all(|part| lexicon.actors.iter().any(|&(word, _)| word == part))
        })
}

fn disposal_clause_frame<'lexicon>(
    source: &Source<'_>,
    unit: &Unit,
    next: Option<&Unit>,
    lexicon: &Lexicon<'lexicon>,
) -> Option<DisposalFrame<'lexicon>> {
    let text = source.slice(&unit.evidence);
    if unit.quote
        || !text.contains(['把', '將', '将'])
        || text.chars().any(|ch| quote_end(ch).is_some())
    {
        return None;
    }
    let base = source.bytes[unit.evidence.start_char];
    let end = source.bytes[unit.evidence.end_char];
    let negations = &[
        "沒有", "没有", "沒能", "没能", "未能", "不能", "不要", "別", "别", "沒", "没", "未", "不",
    ];
    let modifiers = &[
        "當即", "当即", "隨即", "随即", "立即", "急忙", "連忙", "连忙", "趕忙", "赶忙", "並也",
        "并也", "同樣", "同样", "便", "就", "先", "又", "也", "亦",
    ];
    text.match_indices(['把', '將', '将'])
        .find_map(|(marker, word)| {
            let lead = text[..marker].trim_end();
            if word == "把" && lead.ends_with('一') {
                return None;
            }
            let exact_actor = source
                .first_name(base, base + lead.len(), lexicon)
                .filter(|&(offset, surface, _)| offset == 0 && surface.len() == lead.len())
                .map(|hit| actor_at(source, base, hit));
            let subject = negations
                .iter()
                .chain(modifiers)
                .find_map(|modifier| lead.strip_suffix(modifier))
                .unwrap_or(lead)
                .trim_end();
            let actor_surface = source.trim(base, base + subject.len()).filter(|span| {
                let text = source.slice(span);
                !CONTINUATIONS.contains(&text)
                    && !modifiers.contains(&text)
                    && !matches!(text, "但是" | "然而" | "可是" | "請" | "请")
            });
            let actor = exact_actor.or_else(|| {
                actor_surface.and_then(|span| {
                    let start = source.bytes[span.start_char];
                    let end = source.bytes[span.end_char];
                    let hit = source.first_name(start, end, lexicon)?;
                    let prefix = source.text[start..start + hit.0].trim();
                    let suffix = source.text[start + hit.0 + hit.1.len()..end].trim();
                    ((matches!(
                        prefix,
                        "" | "如果"
                            | "假如"
                            | "若是"
                            | "若"
                            | "請"
                            | "请"
                            | "忽然"
                            | "突然"
                            | "剛才"
                            | "刚才"
                            | "但是"
                            | "然而"
                            | "可是"
                    ) || CONTINUATIONS.contains(&prefix)
                        || SUBJECT_CLASSIFIERS.contains(&prefix))
                        && (matches!(
                            suffix,
                            "" | "已" | "已經" | "已经" | "早已" | "剛剛" | "刚刚"
                        ) || PLANNED.contains(&suffix)))
                    .then(|| actor_at(source, start, hit))
                })
            });
            if actor.is_none() && (predicate(lead, lexicon).is_some() || subject.contains('的')) {
                return None;
            }
            let mut inherit_actor = actor_surface.is_none();
            let mut unclear_subject = false;
            let actor_surface = if let Some(actor) = &actor {
                Some(actor.evidence)
            } else if let Some(span) = actor_surface {
                let text = source.slice(&span);
                let surface = continuation(text).unwrap_or(text).trim_start();
                let pronoun = matches!(surface, "我" | "你" | "他" | "她" | "它");
                inherit_actor = matches!(surface, "他" | "她");
                let end = source.bytes[span.end_char];
                let start = end - surface.len();
                if pronoun
                    || (participant_description(surface)
                        && source.first_name(start, end, lexicon).is_none())
                {
                    Some(source.span(start, end))
                } else {
                    unclear_subject = !coordinated_actors(surface, lexicon);
                    None
                }
            } else {
                None
            };
            let body = text[marker + word.len()..].trim_start();
            let (kind, verb, predicate, patient, patient_name) = std::iter::once(body)
                .chain(nominal_measure(body))
                .flat_map(|phrase| {
                    lexicon.objects.iter().chain(lexicon.actors).filter_map(
                        move |&(surface, canonical)| {
                            let tail = phrase.strip_prefix(surface)?.trim_start();
                            if surface.is_empty() {
                                return None;
                            }
                            let (kind, offset, verb) = predicate(tail, lexicon)?;
                            if !matches!(
                                tail[..offset].trim_end(),
                                "" | "輕輕" | "轻轻" | "緩緩" | "缓缓" | "慢慢" | "小心地" | "用力"
                            ) {
                                return None;
                            }
                            let patient_start = end - phrase.len();
                            let verb_start = end - tail.len() + offset;
                            Some((
                                kind,
                                verb,
                                source.span(verb_start, verb_start + verb.len()),
                                source.span(patient_start, patient_start + surface.len()),
                                canonical,
                            ))
                        },
                    )
                })
                .max_by_key(|(_, _, _, patient, _)| patient.end_char - patient.start_char)?;
            let verb_end = source.bytes[predicate.end_char];
            let result = result_value(source, verb_end, end, &["成", "為", "为"]).or_else(|| {
                let next = next.filter(|next| {
                    !next.quote && next.sentence == unit.sentence && text.ends_with(['，', ','])
                })?;
                let patient = source.slice(&patient);
                let result_clause = source.slice(&next.evidence);
                result_clause.strip_prefix(patient)?;
                result_value(
                    source,
                    source.bytes[next.evidence.start_char] + patient.len(),
                    source.bytes[next.evidence.end_char],
                    &["裂成", "斷成", "断成", "變成", "变成", "化為", "化为"],
                )
            });
            let tail = source.text[verb_end..end].trim_start();
            let completed = tail
                .strip_prefix('了')
                .is_some_and(|tail| trim_clause_end(tail).is_empty());
            let surface = source.slice(&patient);
            let conflicting_patient = lexicon
                .objects
                .iter()
                .chain(lexicon.actors)
                .any(|&(form, canonical)| form == surface && canonical != patient_name);
            Some(DisposalFrame {
                kind,
                verb,
                predicate,
                actor_surface,
                actor,
                inherit_actor,
                patient,
                patient_name: (!conflicting_patient).then_some(patient_name),
                result,
                completed: completed || result.is_some(),
                blocked: unclear_subject
                    || ends_any(lead, negations)
                    || lead.trim_start().starts_with(['請', '请'])
                    || tail.strip_prefix('了').unwrap_or(tail).starts_with('的')
                    || text.contains(['？', '?'])
                    || ends_with_question_particle(text)
                    || trim_clause_end(text).ends_with('吧'),
            })
        })
}

fn transformed_patient<'source>(
    source: &Source<'source>,
    previous: &Unit,
    unit: &Unit,
    predicate_start: usize,
) -> Option<&'source str> {
    if previous.quote || previous.sentence != unit.sentence {
        return None;
    }
    let prefix = source.slice(&unit.evidence)[..predicate_start]
        .trim_end()
        .strip_suffix('被')?
        .trim();
    if !matches!(
        prefix,
        "" | "頓時" | "顿时" | "隨即" | "随即" | "旋即" | "立即" | "立刻"
    ) {
        return None;
    }
    let clause = source.slice(&previous.evidence).strip_suffix(['，', ','])?;
    let (offset, word) = earliest(clause, &["化為", "化为", "變成", "变成", "凝成", "融成"])?;
    let lead = clause[..offset].trim();
    let patient = clause[offset + word.len()..].trim();
    if lead.is_empty()
        || ends_any(
            lead,
            &[
                "不",
                "未",
                "沒",
                "没",
                "無法",
                "无法",
                "不能",
                "不會",
                "不会",
                "不可能",
                "幾乎",
                "几乎",
                "差點",
                "差点",
            ],
        )
        || contains_any(lead, PLANNED)
        || patient.chars().any(|ch| quote_end(ch).is_some())
    {
        return None;
    }
    patient
        .chars()
        .any(char::is_alphanumeric)
        .then_some(patient)
}

fn excerpt(text: &str) -> String {
    let mut chars = text.chars();
    let mut result: String = chars.by_ref().take(88).collect();
    if chars.next().is_some() {
        result.push('…');
    }
    result
}

fn possible_aliases(
    source: &Source<'_>,
    span: &Span,
    output: &mut Vec<Review>,
    lexicon: &Lexicon<'_>,
) {
    let text = source.slice(span);
    let nearby = source.slice(&Span {
        start_char: span.start_char.saturating_sub(150),
        end_char: (span.end_char + 150).min(source.bytes.len() - 1),
    });
    let cues = &[
        "所在", "說道", "问道", "問道", "聞言", "身邊", "身後", "身后", "低聲",
    ];
    for (offset, first) in text.char_indices() {
        let rest = &text[offset + first.len_utf8()..];
        let Some(last) = rest.chars().next() else {
            continue;
        };
        if !('一'..='龯').contains(&last) || !starts_any(&rest[last.len_utf8()..], cues) {
            continue;
        }
        let end = offset + first.len_utf8() + last.len_utf8();
        let name = &text[offset..end];
        if lexicon.actors.iter().any(|(surface, _)| *surface == name) {
            continue;
        }
        let candidates: std::collections::BTreeSet<_> = lexicon
            .actors
            .iter()
            .filter(|(surface, _)| {
                surface.starts_with(first)
                    && surface.chars().count() == 2
                    && nearby.contains(surface)
            })
            .map(|(_, canonical)| *canonical)
            .collect();
        if candidates.len() == 1 {
            let start = source.bytes[span.start_char];
            output.push(Review {
                kind: "possible_alias".into(),
                detail: format!("{name} 可能指 {}", candidates.first().unwrap()),
                evidence: source.span(start + offset, start + end),
            });
        }
    }
}

/// Whether a source-derived observation is verified. This API never verifies it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MentionActorUnknown {
    Unknown,
}

/// How the parser selected an unverified written predicate candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MentionPredicateBasis {
    SelectedNarrationPredicate,
    AdditionalWrittenCue,
    ReportedClausePredicate,
    HearingCue,
    RecognizedQuotedRankPredicate,
    Unavailable,
}

/// Source-only sidecar. A range never confirms an event, person or relation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct MentionPredicateObservation {
    pub event_index: usize,
    pub written_predicate: Option<Span>,
    pub basis: MentionPredicateBasis,
    pub identity_status: MentionActorUnknown,
    pub occurrence_status: MentionActorUnknown,
    pub association_status: MentionActorUnknown,
    pub narrative_scope_status: MentionActorUnknown,
}

fn record_mention_predicate(
    observations: Option<&mut Vec<MentionPredicateObservation>>,
    event_index: usize,
    written_predicate: Span,
    basis: MentionPredicateBasis,
) {
    if let Some(observations) = observations {
        observations.push(MentionPredicateObservation {
            event_index,
            written_predicate: Some(written_predicate),
            basis,
            identity_status: MentionActorUnknown::Unknown,
            occurrence_status: MentionActorUnknown::Unknown,
            association_status: MentionActorUnknown::Unknown,
            narrative_scope_status: MentionActorUnknown::Unknown,
        });
    }
}

/// What can be established about the origin of a legacy label.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MentionLabelBasis {
    CallerVocabularyExactSurfaceAndLabel,
    LegacyParserLabelUnverified,
    Unavailable,
}

/// Number of distinct caller-supplied labels for the exact written surface.
/// It does not establish the identity of a person.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MentionLabelMappingStatus {
    NoWrittenSurface,
    Unmapped,
    Unique,
    Ambiguous,
}

/// The parser's source association is a candidate, not a verified role.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MentionAssociationBasis {
    ExplicitWrittenSubjectCandidate,
    PriorWrittenReferenceCandidate,
    Unavailable,
}

/// Unverified source evidence associated with one extracted event.
/// A caller label is not a verified identity and different mentions are not
/// proven to be the same or different person. Ranges index the original input.
#[derive(Clone, Debug, serde::Serialize)]
pub struct MentionActorObservation {
    pub event_index: usize,
    pub written_surface: Option<Span>,
    pub legacy_label_hint: Option<String>,
    pub label_basis: MentionLabelBasis,
    pub caller_labels: Vec<String>,
    pub label_mapping_status: MentionLabelMappingStatus,
    pub identity_status: MentionActorUnknown,
    pub role_status: MentionActorUnknown,
    pub occurrence_status: MentionActorUnknown,
    pub narrative_scope_status: MentionActorUnknown,
    pub association_basis: MentionAssociationBasis,
}

// Both the individual and combined APIs use these normalizers. Missing
// observations must retain the original API's unknown fallbacks.
fn finalize_mention_actors(
    analysis: &Analysis,
    recorded: Vec<MentionActorObservation>,
) -> Vec<MentionActorObservation> {
    let mut by_index: std::collections::BTreeMap<usize, MentionActorObservation> = recorded
        .into_iter()
        .map(|item| (item.event_index, item))
        .collect();
    analysis
        .events
        .iter()
        .enumerate()
        .map(|(event_index, event)| {
            by_index
                .remove(&event_index)
                .unwrap_or_else(|| MentionActorObservation {
                    event_index,
                    written_surface: None,
                    legacy_label_hint: event.actor.clone(),
                    label_basis: if event.actor.is_some() {
                        MentionLabelBasis::LegacyParserLabelUnverified
                    } else {
                        MentionLabelBasis::Unavailable
                    },
                    caller_labels: Vec::new(),
                    label_mapping_status: MentionLabelMappingStatus::NoWrittenSurface,
                    identity_status: MentionActorUnknown::Unknown,
                    role_status: MentionActorUnknown::Unknown,
                    occurrence_status: MentionActorUnknown::Unknown,
                    narrative_scope_status: MentionActorUnknown::Unknown,
                    association_basis: MentionAssociationBasis::Unavailable,
                })
        })
        .collect()
}

fn finalize_mention_predicates(
    analysis: &Analysis,
    recorded: Vec<MentionPredicateObservation>,
    text_len: usize,
) -> Vec<MentionPredicateObservation> {
    let mut by_index =
        std::collections::BTreeMap::<usize, Option<MentionPredicateObservation>>::new();
    for observation in recorded {
        by_index
            .entry(observation.event_index)
            .and_modify(|slot| *slot = None)
            .or_insert(Some(observation));
    }
    analysis
        .events
        .iter()
        .enumerate()
        .map(|(event_index, _)| {
            by_index
                .remove(&event_index)
                .flatten()
                .filter(|observation| {
                    observation.written_predicate.is_some_and(|span| {
                        span.start_char < span.end_char && span.end_char <= text_len
                    })
                })
                .unwrap_or(MentionPredicateObservation {
                    event_index,
                    written_predicate: None,
                    basis: MentionPredicateBasis::Unavailable,
                    identity_status: MentionActorUnknown::Unknown,
                    occurrence_status: MentionActorUnknown::Unknown,
                    association_status: MentionActorUnknown::Unknown,
                    narrative_scope_status: MentionActorUnknown::Unknown,
                })
        })
        .collect()
}

fn capture_mention_actor(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    event_index: usize,
    actor: Option<&Actor>,
    explicit_subject: bool,
) -> MentionActorObservation {
    let written_surface = actor.map(|a| a.evidence);
    let legacy_label_hint = actor.map(|a| a.name.clone());
    let caller_labels: Vec<String> = actor
        .map(|a| {
            let written = source.slice(&a.evidence);
            lexicon
                .actors
                .iter()
                .filter_map(|(surface, label)| (*surface == written).then_some((*label).to_owned()))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect()
        })
        .unwrap_or_default();
    let label_mapping_status = match (written_surface, caller_labels.len()) {
        (None, _) => MentionLabelMappingStatus::NoWrittenSurface,
        (Some(_), 0) => MentionLabelMappingStatus::Unmapped,
        (Some(_), 1) => MentionLabelMappingStatus::Unique,
        (Some(_), _) => MentionLabelMappingStatus::Ambiguous,
    };
    let label_basis = match actor {
        Some(actor)
            if lexicon.actors.iter().any(|(surface, label)| {
                *surface == source.slice(&actor.evidence) && *label == actor.name.as_str()
            }) =>
        {
            MentionLabelBasis::CallerVocabularyExactSurfaceAndLabel
        }
        Some(_) => MentionLabelBasis::LegacyParserLabelUnverified,
        None => MentionLabelBasis::Unavailable,
    };
    MentionActorObservation {
        event_index,
        written_surface,
        legacy_label_hint,
        label_basis,
        caller_labels,
        label_mapping_status,
        identity_status: MentionActorUnknown::Unknown,
        role_status: MentionActorUnknown::Unknown,
        occurrence_status: MentionActorUnknown::Unknown,
        narrative_scope_status: MentionActorUnknown::Unknown,
        association_basis: if actor.is_none() {
            MentionAssociationBasis::Unavailable
        } else if explicit_subject {
            MentionAssociationBasis::ExplicitWrittenSubjectCandidate
        } else {
            MentionAssociationBasis::PriorWrittenReferenceCandidate
        },
    }
}

fn push_event(
    source: &Source<'_>,
    events: &mut Vec<Event>,
    mut event: Event,
    source_mentions: bool,
) {
    if !source_mentions
        && event.kind == "realm_change"
        && event.status == "occurred"
        && event.actor.is_some()
        && let Some(previous) = events.iter_mut().rev().find(|previous| {
            previous.kind == event.kind
                && previous.actor == event.actor
                && previous.object == event.object
                && previous.status == event.status
                && (previous.recap || event.recap)
        })
    {
        let unresolved = previous
            .evidence
            .first()
            .zip(event.evidence.first())
            .is_some_and(|(old, new)| {
                recap_time::preserve_unresolved(
                    recap_time::local_context(source.text, &source.bytes, *old),
                    recap_time::local_context(source.text, &source.bytes, *new),
                    event.actor.as_deref().unwrap_or(""),
                    false,
                )
            });
        let distinct = previous
            .evidence
            .first()
            .zip(event.evidence.first())
            .and_then(|(old, new)| {
                let (prior_cue, next_cue) = recap_time::distinct_cues(
                    recap_time::local_context(source.text, &source.bytes, *old),
                    recap_time::local_context(source.text, &source.bytes, *new),
                    event.actor.as_deref().unwrap_or(""),
                    false,
                )?;
                Some((
                    recap_time::cue_span(source.text, &source.bytes, *old, prior_cue)?,
                    recap_time::cue_span(source.text, &source.bytes, *new, next_cue)?,
                ))
            });
        if let Some((prior_cue, next_cue)) = distinct {
            if !previous.evidence.contains(&prior_cue) {
                previous.evidence.push(prior_cue);
            }
            if !event.evidence.contains(&next_cue) {
                event.evidence.push(next_cue);
            }
        } else if unresolved {
            if let Some(span) = previous
                .evidence
                .first()
                .and_then(|old| recap_time::primary_cue_span(source.text, &source.bytes, *old))
                && !previous.evidence.contains(&span)
            {
                previous.evidence.push(span);
            }
            if let Some(span) = event
                .evidence
                .first()
                .and_then(|new| recap_time::primary_cue_span(source.text, &source.bytes, *new))
                && !event.evidence.contains(&span)
            {
                event.evidence.push(span);
            }
        } else {
            previous.evidence.extend(event.evidence);
            previous.recap &= event.recap;
            return;
        }
    }
    events.push(event);
}

const CONTINUATIONS: &[&str] = &[
    "同時", "只見", "頓時", "此刻", "此時", "接著", "隨後", "然后", "然後", "旋即", "於是", "于是",
    "原來", "卻", "但", "而",
];

fn continuation(text: &str) -> Option<&str> {
    CONTINUATIONS
        .iter()
        .find_map(|marker| text.strip_prefix(marker))
}

fn introduces_subject(text: &str) -> bool {
    starts_any(
        continuation(text).unwrap_or(text),
        &[
            "有人",
            "一人",
            "此人",
            "一個人",
            "一個",
            "這人",
            "那人",
            "一名",
            "這名",
            "那名",
            "一位",
            "這位",
            "那位",
            "兩名",
            "另一",
            "眾人",
            "他們",
            "她們",
            "他人",
            "他们",
            "她们",
        ],
    )
}

fn carries_subject(text: &str) -> bool {
    let body = continuation(text).unwrap_or(text);
    matches!(body.chars().next(), Some('他' | '她'))
        && !introduces_subject(body)
        && !starts_any(body, &["他的", "她的"])
}

fn detached_predecessor<'units>(
    source: &Source<'_>,
    preceding: &'units [Unit],
    unit: &Unit,
) -> Option<&'units Unit> {
    let [.., named, cue] = preceding else {
        return None;
    };
    if named.quote
        || cue.quote
        || cue.first_in_line
        || cue.sentence != unit.sentence
        || cue.evidence.start_char != unit.sentence.start_char
        || named.sentence == unit.sentence
    {
        return None;
    }
    let marker = source
        .slice(&cue.evidence)
        .strip_suffix(['，', ','])?
        .trim_end();
    matches!(
        marker,
        "隨後" | "随后" | "接著" | "接着" | "然後" | "然后" | "旋即"
    )
    .then_some(named)
}

fn omitted_disposal_subject(text: &str, predicate_start: usize) -> bool {
    text[..predicate_start]
        .strip_prefix('把')
        .is_some_and(|patient| patient.chars().any(char::is_alphanumeric))
        && !contains_any(text, &["的是", "才可以", "就可以"])
        && !text.contains(['？', '?'])
        && !ends_with_question_particle(text)
}

struct Focus {
    actor: Actor,
    last_reference: Span,
    sentence: usize,
}

impl Focus {
    fn carry(&self, unit: &Unit, text: &str, lexicon: &Lexicon<'_>) -> Option<Actor> {
        if introduces_subject(text)
            || unit
                .evidence
                .start_char
                .saturating_sub(self.actor.evidence.end_char)
                > 350
        {
            return None;
        }
        let nearby = unit
            .evidence
            .start_char
            .saturating_sub(self.last_reference.end_char)
            <= 125;
        let explicit_continuation = continuation(text)
            .is_some_and(|body| predicate(body, lexicon).is_some_and(|(_, offset, _)| offset == 0));
        (unit.sentence.start_char == self.sentence
            || (nearby && (carries_subject(text) || explicit_continuation)))
            .then(|| self.actor.clone())
    }

    fn carry_from(&self, named: &Unit, last_event: &Event) -> Option<Actor> {
        (self.last_reference == named.evidence
            && self.actor.evidence.start_char >= named.evidence.start_char
            && self.actor.evidence.end_char <= named.evidence.end_char
            && last_event.evidence.first() == Some(&named.evidence)
            && last_event.status == "occurred")
            .then(|| self.actor.clone())
    }
}

/// True when the `nlp.doc.v1` readers may serve the line. This is an allow list of chars on
/// which layer 1, layer 2 and the old segmenter agree. Every other line (an ASCII letter, digit
/// or punctuation, `…`, other brackets and closers, emoji, a tab, `\r`, another script, a
/// doubled or edge space) keeps the old ICU and tokenizer path. A char leaves the set when a
/// differential test finds a line it breaks; a layer or an expected value never changes.
///
/// The set: Han ideographs; `。！？，、；：「」『』`; one ASCII space or U+3000 that is not next
/// to another space and not first or last in the line; a final `\n`.
pub(crate) fn doc_applies(line: &str) -> bool {
    let body = line.strip_suffix('\n').unwrap_or(line);
    let mut previous_space = true; // a leading space is refused like a doubled one
    for ch in body.chars() {
        let is_space = matches!(ch, ' ' | '\u{3000}');
        let ok = match ch {
            '\u{4E00}'..='\u{9FFF}'
            | '\u{3400}'..='\u{4DBF}'
            | '。'
            | '！'
            | '？'
            | '，'
            | '、'
            | '；'
            | '：'
            | '「'
            | '」'
            | '『'
            | '』' => true,
            _ => is_space && !previous_space,
        };
        if !ok {
            return false;
        }
        previous_space = is_space;
    }
    body.is_empty() || !previous_space
}

/// The `nlp.doc.v1` path: layer 1 sentences and layer 3 names of one line. It returns `None`
/// for a line where the old path must stay (see `doc_applies`).
fn doc_line<'a>(line: &'a str, terms: &crate::doc::NameRule) -> Option<LineRead<'a>> {
    if !doc_applies(line) {
        return None;
    }
    let view = crate::doc::line_view(line, terms);
    if view.has_opaque {
        return None;
    }
    let sentences = view
        .sentences
        .into_iter()
        .map(|r| LineSentence {
            text: &line[r.clone()],
            start: r.start,
            end: r.end,
        })
        .collect();
    let hits = view.names.into_iter().map(|r| (r.start, r.end)).collect();
    Some((sentences, hits))
}

/// Sentences of one line and the byte ranges of its term hits.
type LineRead<'a> = (Vec<LineSentence<'a>>, Vec<(usize, usize)>);

/// One sentence of a line: trimmed text and its byte range in the line.
struct LineSentence<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

/// Reuse the same locale and work vocabulary across chapters.
pub struct NarrativeAnalyzer<'lexicon> {
    /// Terms for the `nlp.doc.v1` name rule. `None` keeps the old ICU path (see `docs/doc-v1.md`).
    doc_terms: Option<crate::doc::NameRule>,
    segmenter: Analyzer,
    lexicon: Lexicon<'lexicon>,
    actors: HashMap<&'lexicon str, usize>,
}

impl<'lexicon> NarrativeAnalyzer<'lexicon> {
    pub(crate) fn matches_lexicon(&self, other: &Lexicon<'_>) -> bool {
        let own = &self.lexicon;
        own.actors == other.actors
            && own.places == other.places
            && own.objects == other.objects
            && own.ranks == other.ranks
            && own.formed_ranks == other.formed_ranks
            && own.rank_blocked_prefixes == other.rank_blocked_prefixes
            && own.footers == other.footers
            && own.line_footers == other.line_footers
    }

    /// Build a sentence and word analyzer with the work's surface spellings.
    ///
    /// Spellings must be nonempty and have no surrounding whitespace or line breaks.
    pub fn new(locale: Locale, lexicon: &Lexicon<'lexicon>) -> anyhow::Result<Self> {
        let mut actors = HashMap::new();
        for (index, &(word, _)) in lexicon.actors.iter().enumerate() {
            actors.entry(word).or_insert(index);
        }
        let terms: Vec<_> = lexicon
            .actors
            .iter()
            .chain(lexicon.objects)
            .chain(lexicon.ranks)
            .map(|&(word, _)| word)
            .chain(lexicon.places.iter().copied())
            .collect();
        // Only the zh-TW locale has a layer in `nlp.doc.v1`; other locales stay on the ICU path.
        let doc_terms = (locale == Locale::ChineseTaiwan).then(|| {
            let names: Vec<String> = crate::text::TAIWAN_TERMS
                .iter()
                .chain(&terms)
                .map(|term| (*term).to_owned())
                .collect();
            crate::doc::NameRule::new(&names)
        });
        Ok(Self {
            doc_terms,
            segmenter: Analyzer::with_terms(locale, &terms)?,
            lexicon: *lexicon,
            actors,
        })
    }

    /// Analyze one chapter. Evidence counts Unicode scalar values in the original text.
    pub fn analyze(&self, chapter_text: &str, title: &str) -> Analysis {
        analyze_chapter(
            chapter_text,
            title,
            &self.lexicon,
            Some(self),
            None,
            false,
            None,
            None,
        )
    }

    /// Keep each extracted source mention. This does not establish whether two
    /// mentions refer to the same actual event, or claim complete coverage.
    pub fn analyze_source_mentions(&self, chapter_text: &str, title: &str) -> Analysis {
        analyze_chapter(
            chapter_text,
            title,
            &self.lexicon,
            Some(self),
            None,
            true,
            None,
            None,
        )
    }

    /// Find written give/receive forms, independent of events and identities.
    /// Quotation, negation, intent, and role association remain unresolved.
    /// The returned ranges count Unicode scalars in the unmodified text.
    pub fn discover_written_transfer_cues(&self, chapter_text: &str) -> Vec<WrittenTransferCue> {
        let mut found = Vec::new();
        for (direction, words) in [
            (WrittenTransferDirection::Give, WRITTEN_GIVE_CUES),
            (WrittenTransferDirection::Receive, WRITTEN_RECEIVE_CUES),
        ] {
            for &word in words {
                for (start, _) in chapter_text.match_indices(word) {
                    found.push((start, start + word.len(), direction));
                }
            }
        }
        found.sort_unstable_by_key(|&(start, end, _)| (start, std::cmp::Reverse(end - start)));
        let mut result = Vec::new();
        let mut byte_cursor = 0;
        let mut scalar_cursor = 0;
        for (start, next, direction) in found {
            if start < byte_cursor {
                continue;
            }
            let start_char = scalar_cursor + chapter_text[byte_cursor..start].chars().count();
            let end_char = start_char + chapter_text[start..next].chars().count();
            result.push(WrittenTransferCue {
                cue: Span {
                    start_char,
                    end_char,
                },
                direction,
            });
            byte_cursor = next;
            scalar_cursor = end_char;
        }
        result
    }

    /// Propose local written fragments. These do not establish roles or events.
    pub fn discover_written_transfer_role_candidates(
        &self,
        chapter_text: &str,
    ) -> Vec<WrittenTransferRoleCandidate> {
        let cues = self.discover_written_transfer_cues(chapter_text);
        transfer_roles::discover(chapter_text, &cues)
    }

    /// Keep possible source boundaries without establishing identities or roles.
    pub fn discover_written_transfer_role_alternatives(
        &self,
        chapter_text: &str,
    ) -> Vec<WrittenTransferRoleAlternatives> {
        let cues = self.discover_written_transfer_cues(chapter_text);
        transfer_roles::discover_alternatives(chapter_text, &cues)
    }

    /// Keep extracted source mentions and clause ranges indexed into them.
    pub fn analyze_source_mentions_with_clause_frames(
        &self,
        chapter_text: &str,
        title: &str,
    ) -> (Analysis, Vec<ClauseFrame>) {
        let mut frames = Vec::new();
        let analysis = analyze_chapter(
            chapter_text,
            title,
            &self.lexicon,
            Some(self),
            Some(&mut frames),
            true,
            None,
            None,
        );
        (analysis, frames)
    }

    /// Analyze extracted mentions with clause frames and unverified source roles.
    /// Every returned observation has the corresponding event index. A missing
    /// written surface stays absent; labels are hints, not resolved identities.
    pub fn analyze_source_mentions_with_actor_observations(
        &self,
        chapter_text: &str,
        title: &str,
    ) -> (Analysis, Vec<ClauseFrame>, Vec<MentionActorObservation>) {
        let mut frames = Vec::new();
        let mut recorded = Vec::new();
        let analysis = analyze_chapter(
            chapter_text,
            title,
            &self.lexicon,
            Some(self),
            Some(&mut frames),
            true,
            Some(&mut recorded),
            None,
        );
        let observations = finalize_mention_actors(&analysis, recorded);
        (analysis, frames, observations)
    }

    /// Analyze source mentions and retain parser-selected written predicate
    /// ranges. These candidates do not establish identity, occurrence,
    /// narrative scope, or the relation between written mentions.
    /// Every output entry has the index of the corresponding source mention.
    pub fn analyze_source_mentions_with_predicate_observations(
        &self,
        chapter_text: &str,
        title: &str,
    ) -> (Analysis, Vec<MentionPredicateObservation>) {
        let mut recorded = Vec::new();
        let analysis = analyze_chapter(
            chapter_text,
            title,
            &self.lexicon,
            Some(self),
            None,
            true,
            None,
            Some(&mut recorded),
        );
        let observations =
            finalize_mention_predicates(&analysis, recorded, chapter_text.chars().count());
        (analysis, observations)
    }

    /// Analyze once and retain clause, written-actor and written-predicate
    /// candidates indexed into the same source-mention analysis. These
    /// observations do not establish identities, roles, events or relations.
    pub fn analyze_source_mentions_with_observations(
        &self,
        chapter_text: &str,
        title: &str,
    ) -> (
        Analysis,
        Vec<ClauseFrame>,
        Vec<MentionActorObservation>,
        Vec<MentionPredicateObservation>,
    ) {
        let mut frames = Vec::new();
        let mut actors = Vec::new();
        let mut predicates = Vec::new();
        let analysis = analyze_chapter(
            chapter_text,
            title,
            &self.lexicon,
            Some(self),
            Some(&mut frames),
            true,
            Some(&mut actors),
            Some(&mut predicates),
        );
        let actors = finalize_mention_actors(&analysis, actors);
        let predicates =
            finalize_mention_predicates(&analysis, predicates, chapter_text.chars().count());
        (analysis, frames, actors, predicates)
    }

    /// Analyze a chapter and return source ranges for recognized disposal clauses.
    ///
    /// The ranges count Unicode scalar values in the original chapter text.
    pub fn analyze_with_clause_frames(
        &self,
        chapter_text: &str,
        title: &str,
    ) -> (Analysis, Vec<ClauseFrame>) {
        let mut frames = Vec::new();
        let analysis = analyze_chapter(
            chapter_text,
            title,
            &self.lexicon,
            Some(self),
            Some(&mut frames),
            false,
            None,
            None,
        );
        (analysis, frames)
    }

    /// Find parser-suggested speech and decision acts with source ranges.
    ///
    /// A payload records what the act concerns. It does not assert that the
    /// actions in that text occurred.
    pub fn discover_speech_acts(&self, chapter_text: &str, title: &str) -> Vec<SpeechAct> {
        speech::discover(self, chapter_text, title)
    }

    /// Find completed reading and checking actions with an explicit known subject and direct object.
    /// The actor and the whole object must be grounded in the supplied vocabulary.
    /// These source ranges are returned separately from chapter events and temporal links.
    pub fn discover_reading_actions(&self, chapter_text: &str, title: &str) -> Vec<ReadingAction> {
        linking::discover_reading_actions(self, chapter_text, title)
    }

    /// Find observations with a written completion marker and an explicit short duration.
    /// The evidence is separate from chapter events and temporal links.
    pub fn discover_duration_observations(
        &self,
        chapter_text: &str,
        title: &str,
    ) -> Vec<DurationObservation> {
        linking::discover_duration_observations(self, chapter_text, title)
    }

    /// Anchor a batch of candidate ranges to clauses in the original chapter.
    ///
    /// The result preserves candidate scope and explicit local result links.
    /// It does not add events to the narrative or timeline.
    pub fn link_clause_candidates(
        &self,
        chapter_text: &str,
        title: &str,
        candidates: &[ClauseCandidate],
    ) -> LinkedClauses {
        linking::link(self, chapter_text, title, candidates, None)
    }

    /// Link a batch of candidates and return the local order written in the source.
    ///
    /// The precedence links refer to the accepted clauses. This does not build an event timeline.
    pub fn link_clause_candidates_with_precedence(
        &self,
        chapter_text: &str,
        title: &str,
        candidates: &[ClauseCandidate],
    ) -> (LinkedClauses, Vec<ClausePrecedence>) {
        let mut precedence = Vec::new();
        let clauses = linking::link(
            self,
            chapter_text,
            title,
            candidates,
            Some(linking::TemporalOutput::Precedence(&mut precedence)),
        );
        (clauses, precedence)
    }

    /// Discover written order cues and bind each side when the source supports it.
    ///
    /// An empty candidate slice still returns cues with their original ranges.
    /// Endpoint indices refer to the accepted clauses in the first return value.
    pub fn link_clause_candidates_with_cues(
        &self,
        chapter_text: &str,
        title: &str,
        candidates: &[ClauseCandidate],
    ) -> (LinkedClauses, Vec<ClauseOrderCue>) {
        let mut cues = Vec::new();
        let clauses = linking::link(
            self,
            chapter_text,
            title,
            candidates,
            Some(linking::TemporalOutput::Cues(&mut cues)),
        );
        (clauses, cues)
    }

    /// Discover written cues and return up to five evidenced alternatives for each side.
    ///
    /// The ranking is advisory. A side binds only when one candidate meets its binding rule.
    /// A sequence cue can rank an earlier action across one static description on the same line.
    /// That earlier alternative is ineligible for binding.
    pub fn link_clause_candidates_with_ranked_cues(
        &self,
        chapter_text: &str,
        title: &str,
        candidates: &[ClauseCandidate],
    ) -> (LinkedClauses, Vec<RankedClauseOrderCue>) {
        self.link_clause_candidates_with_ranked_cues_and_evidence(
            chapter_text,
            title,
            candidates,
            &[],
        )
    }

    /// Rank endpoints with optional action heads independently verified by the caller.
    /// These heads can add alternatives for cues the analyzer has found. The cue inventory,
    /// cue kinds, strict endpoint decisions, and linked clauses are independent of the hints.
    /// Invalid or conflicting ranges for a candidate are ignored.
    pub fn link_clause_candidates_with_ranked_cues_and_heads(
        &self,
        chapter_text: &str,
        title: &str,
        candidates: &[ClauseCandidate],
        heads: &[VerifiedClauseHead],
    ) -> (LinkedClauses, Vec<RankedClauseOrderCue>) {
        let evidence: Vec<_> = heads
            .iter()
            .copied()
            .map(|head| VerifiedClauseEvidence {
                head,
                subject: None,
            })
            .collect();
        self.link_clause_candidates_with_ranked_cues_and_evidence(
            chapter_text,
            title,
            candidates,
            &evidence,
        )
    }

    /// Rank endpoints with action heads and optional subjects verified by the caller.
    ///
    /// Verify each role in the exact, unmodified chapter text. Ranges count Unicode
    /// scalar values from the start of that text. The evidence can add alternatives
    /// without changing the discovered cues, strict endpoints, or linked clauses.
    pub fn link_clause_candidates_with_ranked_cues_and_evidence(
        &self,
        chapter_text: &str,
        title: &str,
        candidates: &[ClauseCandidate],
        evidence: &[VerifiedClauseEvidence],
    ) -> (LinkedClauses, Vec<RankedClauseOrderCue>) {
        let mut cues = Vec::new();
        let clauses = linking::link(
            self,
            chapter_text,
            title,
            candidates,
            Some(linking::TemporalOutput::RankedCues(&mut cues, evidence)),
        );
        (clauses, cues)
    }

    /// Return ranked cues and explain the requested endpoint decisions.
    ///
    /// Probes use cue ranges in the original chapter and indices in `candidates`.
    /// Diagnostics follow the order of `probes`. Probes do not affect cue discovery,
    /// ranking, binding, or the linked clauses.
    pub fn link_clause_candidates_with_cue_diagnostics(
        &self,
        chapter_text: &str,
        title: &str,
        candidates: &[ClauseCandidate],
        evidence: &[VerifiedClauseEvidence],
        probes: &[ClauseOrderProbe],
    ) -> (
        LinkedClauses,
        Vec<RankedClauseOrderCue>,
        Vec<ClauseOrderDiagnostic>,
    ) {
        let mut cues = Vec::new();
        let mut diagnostics = Vec::new();
        let clauses = linking::link(
            self,
            chapter_text,
            title,
            candidates,
            Some(linking::TemporalOutput::DiagnosedCues {
                cues: &mut cues,
                diagnostics: &mut diagnostics,
                evidence,
                probes,
            }),
        );
        (clauses, cues, diagnostics)
    }

    /// Sentences of one line and the byte ranges of the vocabulary terms in them.
    ///
    /// The zh-TW locale reads `nlp.doc.v1` layer 1 (sentences) and layer 3 (names). Other
    /// locales keep the ICU segmenter, because the document has no layer for them.
    fn line_sentences<'a>(&self, line: &'a str) -> LineRead<'a> {
        if let Some(terms) = &self.doc_terms
            && let Some(found) = doc_line(line, terms)
        {
            return found;
        }
        self.icu_line(line)
    }

    /// The old path: ICU sentences and `Term` tokens from the text segmenter.
    fn icu_line<'a>(&self, line: &'a str) -> LineRead<'a> {
        let sentences = self.segmenter.analyze(line);
        let hits = sentences
            .iter()
            .flat_map(|sentence| sentence.tokens.iter())
            .filter(|token| token.kind == TokenKind::Term)
            .map(|token| (token.bytes.start, token.bytes.end))
            .collect();
        let sentences = sentences
            .iter()
            .map(|s| LineSentence {
                text: s.text,
                start: s.bytes.start,
                end: s.bytes.end,
            })
            .collect();
        (sentences, hits)
    }

    fn append_line(
        &self,
        source: &mut Source<'_>,
        units: &mut Vec<Unit>,
        base: usize,
        line: &str,
        first_in_line: &mut bool,
    ) {
        let (sentences, hits) = self.line_sentences(line);
        let actor_tokens = source.actor_tokens.get_or_insert_with(Vec::new);
        actor_tokens.extend(hits.into_iter().filter_map(|(start, end)| {
            let &actor = self.actors.get(&line[start..end])?;
            Some((base + start, actor))
        }));

        let mut stack = Vec::new();
        let mut start = 0;
        let mut scanned = 0;
        for (index, sentence) in sentences.iter().enumerate() {
            let end = sentence.end;
            let mut closed_quote = false;
            for (offset, ch) in line[scanned..end].char_indices() {
                if stack.len() == 1 && stack.last() == Some(&ch) {
                    closed_quote = true;
                } else if !ch.is_whitespace() && !matches!(ch, ')' | '）' | ']' | '】') {
                    closed_quote = false;
                }
                track_quote(line, scanned + offset, ch, &mut stack);
            }
            scanned = end;
            if !stack.is_empty() {
                continue;
            }
            // ICU can split immediately after a question inside a heard
            // quote. Keep an explicit adjacent 後/后 + predicate with its
            // hearing construction so its subject and source remain local.
            if closed_quote && let Some(next) = sentences.get(index + 1) {
                let next_text = next.text.trim_start();
                let continuation = next_text
                    .strip_prefix('後')
                    .or_else(|| next_text.strip_prefix('后'));
                let explicit_predicate = continuation.is_some_and(|tail| {
                    predicate(tail.trim_start(), &self.lexicon)
                        .is_some_and(|(_, offset, _)| offset == 0)
                });
                let explicit_question = starts_any(next_text, &["嗎", "吗", "了嗎", "了吗"])
                    && (next_text.trim_end().ends_with(['？', '?'])
                        || ends_with_question_particle(next_text));
                let current = &line[start..end];
                let has_hearing = written_quote_ranges(current)
                    .and_then(|quotes| hearing_quote(current, &quotes))
                    .is_some();
                if (explicit_predicate || explicit_question) && has_hearing {
                    continue;
                }
            }
            if closed_quote
                && let Some(next) = sentences.get(index + 1)
                && self.has_speaker_tag(source, base + next.start, next.text)
            {
                continue;
            }
            append_sentence(source, units, base + start, base + end, first_in_line);
            start = end;
        }
        if start < scanned {
            append_sentence(source, units, base + start, base + scanned, first_in_line);
        }
    }

    fn has_speaker_tag(&self, source: &Source<'_>, start: usize, text: &str) -> bool {
        let clause = text
            .split(|ch| matches!(ch, '；' | ';') || quote_end(ch).is_some())
            .next()
            .unwrap_or(text);
        source.has_speech(clause)
            && (named_speaker_after(source, start, start + clause.len(), &self.lexicon).is_some()
                || title_subject(source, start, clause, &self.lexicon).is_some())
    }
}

/// Analyze one chapter while preserving offsets into its original text.
pub fn analyze(chapter_text: &str, title: &str, lexicon: &Lexicon<'_>) -> Analysis {
    analyze_chapter(chapter_text, title, lexicon, None, None, false, None, None)
}

// The stored intervals are disjoint. A later, overlapping vocabulary spelling
// is treated as an alternative surface, not a second written occurrence.
fn remember_written_rank(spans: &mut std::collections::BTreeMap<usize, usize>, span: Span) -> bool {
    if spans
        .range(..span.end_char)
        .next_back()
        .is_some_and(|(_, end)| *end > span.start_char)
    {
        return false;
    }
    spans.insert(span.start_char, span.end_char);
    true
}

// Build the same non-overlapping object matches as `str::match_indices` once
// per damaged unit. Only the furthest end at each start can cover a rank.
// The frontier retains only strictly increasing ends, so a binary search gives
// the earliest start associated with the furthest end in the queried prefix.
// Cap unique starts; if exceeded, use the original exact scan for each query.
const MAX_OBJECT_COVERAGE_STARTS: usize = 65_536;

enum ObjectCoverage {
    Indexed(Vec<(usize, usize)>),
    Scan,
}

impl ObjectCoverage {
    fn new(text: &str, lexicon: &Lexicon<'_>) -> Self {
        let mut starts = std::collections::BTreeMap::<usize, usize>::new();
        for &(object, _) in lexicon.objects {
            if object.is_empty() {
                continue;
            }
            for (start, _) in text.match_indices(object) {
                let end = start + object.len();
                starts
                    .entry(start)
                    .and_modify(|old| *old = (*old).max(end))
                    .or_insert(end);
                if starts.len() > MAX_OBJECT_COVERAGE_STARTS {
                    return Self::Scan;
                }
            }
        }
        let mut frontier = Vec::new();
        let mut furthest = 0;
        for (start, end) in starts {
            if end > furthest {
                frontier.push((start, end));
                furthest = end;
            }
        }
        Self::Indexed(frontier)
    }

    fn covers(&self, text: &str, lexicon: &Lexicon<'_>, start: usize, end: usize) -> bool {
        match self {
            Self::Indexed(frontier) => {
                let count = frontier.partition_point(|(left, _)| *left <= start);
                if count == 0 {
                    return false;
                }
                let (object_start, object_end) = frontier[count - 1];
                object_end > end || (object_end == end && object_start < start)
            }
            Self::Scan => lexicon.objects.iter().any(|&(object, _)| {
                !object.is_empty()
                    && object.len() > end - start
                    && text.match_indices(object).any(|(object_start, _)| {
                        object_start <= start && end <= object_start + object.len()
                    })
            }),
        }
    }
}

// A caller's rank spelling can also occur inside a longer written object.
// Check one occurrence at a time so a later rank in the same unit does not
// decide whether this one is retained. This is only a written-form check.
fn recovery_rank_allowed(
    text: &str,
    lexicon: &Lexicon<'_>,
    start: usize,
    end: usize,
    object_coverage: &mut Option<ObjectCoverage>,
) -> bool {
    let Some(surface) = text.get(start..end) else {
        return false;
    };
    let Some(&(_, target)) = lexicon.ranks.iter().find(|(rank, _)| *rank == surface) else {
        return false;
    };
    let tail = text[end..].strip_prefix('期').unwrap_or(&text[end..]);
    if starts_any(tail, &["之前", "以前"]) {
        return false;
    }
    if object_coverage
        .get_or_insert_with(|| ObjectCoverage::new(text, lexicon))
        .covers(text, lexicon, start, end)
    {
        return false;
    }
    let left = text[..start]
        .rfind(surface)
        .map_or(0, |offset| offset + surface.len());
    let right = text[end..]
        .find(surface)
        .map_or(text.len(), |offset| end + offset);
    let Some(window) = text.get(left..right) else {
        return false;
    };
    let only_rank = [(surface, target)];
    let vocabulary = Lexicon {
        ranks: &only_rank,
        ..*lexicon
    };
    realm(window, window, &vocabulary)
        .is_some_and(|hit| hit.offset + left == start && hit.end_offset + left == end)
}

// The optional output sidecars preserve the existing analyzer entry points.
#[allow(clippy::too_many_arguments)]
fn analyze_chapter(
    chapter_text: &str,
    title: &str,
    lexicon: &Lexicon<'_>,
    narrative: Option<&NarrativeAnalyzer<'_>>,
    mut clause_frames: Option<&mut Vec<ClauseFrame>>,
    source_mentions: bool,
    mut mention_actors: Option<&mut Vec<MentionActorObservation>>,
    output_predicates: Option<&mut Vec<MentionPredicateObservation>>,
) -> Analysis {
    // All source-mention entry points must use the same predicate bookkeeping,
    // even when their caller does not request the observation sidecar. This
    // also gives the damaged-text fallback one consistent set for deduplication.
    let mut recorded_predicates = Vec::new();
    let mut mention_predicates = source_mentions.then_some(&mut recorded_predicates);
    // These disjoint intervals track written rank surfaces, not identities or
    // outcomes. Keeping source positions avoids conflating repeated spellings.
    let mut represented_ranks = std::collections::BTreeMap::<usize, usize>::new();
    let mut source = Source::new(chapter_text);
    let quote_damage = admitted_quote_damage_start(&source, title, lexicon);
    let units = split_units_impl(&mut source, title, lexicon, narrative, quote_damage);
    let mut result = Analysis::default();
    let mut context = "main";
    let mut focus: Option<Focus> = None;
    for (index, unit) in units.iter().enumerate() {
        let first_event = result.events.len();
        let text = source.slice(&unit.evidence);
        let sentence = source.slice(&unit.sentence);
        if !text.chars().any(|ch| ch.is_alphanumeric()) {
            continue;
        }
        let damaged_quote = quote_damage.is_some_and(|start| unit.evidence.end_char > start);
        let prior_context = context;
        let quote_ranges = (!unit.quote).then(|| written_quote_ranges(text)).flatten();
        let context_text = if unit.quote {
            None
        } else {
            unquoted_scope_text(text)
        };
        context = context_for_unit(unit, context_text.as_deref().unwrap_or(text), context);
        let realm = if unit.quote || quote_ranges.as_ref().is_some_and(Vec::is_empty) {
            realm(text, sentence, lexicon)
        } else {
            quote_ranges
                .as_ref()
                .and_then(|quotes| realm_outside_quotes(text, quotes, lexicon))
        };
        let start = source.bytes[unit.evidence.start_char];
        // Disposal analysis may construct several verb/patient spans. Do not
        // permit any such frame if the unit contains a quoted passage or its
        // delimiter state is damaged; those spans must come from narration.
        let frame = if narrative.is_some()
            && realm.is_none()
            && !unit.quote
            && quote_ranges.as_ref().is_some_and(Vec::is_empty)
        {
            disposal_clause_frame(&source, unit, units.get(index + 1), lexicon)
        } else {
            None
        };
        let outside_pred = quote_ranges
            .as_ref()
            .and_then(|quotes| predicate_outside_quotes(text, quotes, lexicon));
        let pred = frame
            .as_ref()
            .map(|frame| {
                (
                    frame.kind,
                    source.bytes[frame.predicate.start_char] - start,
                    frame.verb,
                )
            })
            .or_else(|| {
                if unit.quote {
                    predicate(text, lexicon)
                } else {
                    outside_pred
                        .as_ref()
                        .map(|(kind, offset, word, _)| (*kind, *offset, *word))
                }
            });
        let earlier = &source.text
            [source.bytes[unit.sentence.start_char]..source.bytes[unit.evidence.start_char]];
        let mut quote_conflicting = false;
        let written_speech = if unit.quote {
            speech::quoted_act(
                &source,
                unit,
                units.get(index + 1),
                context,
                lexicon,
                &mut quote_conflicting,
            )
            .filter(|act| {
                quote_damage.is_none_or(|start| {
                    act.cue.end_char <= start
                        && act.payload.end_char <= start
                        && act
                            .speaker_surface
                            .is_none_or(|span| span.end_char <= start)
                })
            })
        } else {
            None
        };
        if quote_damage.is_some() {
            // A malformed following context cannot establish two competing tags.
            quote_conflicting = false;
        }
        let quote_identity_unknown = written_speech
            .as_ref()
            .is_some_and(|act| act.speaker_surface.is_some() && act.speaker.is_none());
        let subject = if let Some(frame) = &frame {
            frame.actor.clone()
        } else if unit.quote {
            written_speech.and_then(|act| {
                let surface = act.speaker_surface?;
                let name = act.speaker?;
                // This is the caller's vocabulary label, not a verified
                // identity. Its written surface remains in the evidence.
                Some(Actor {
                    name,
                    evidence: surface,
                })
            })
        } else {
            named_subject(&source, &unit.evidence, lexicon).filter(|actor| {
                quote_ranges.as_ref().is_some_and(|quotes| {
                    let actor_start = source.bytes[actor.evidence.start_char] - start;
                    let actor_end = source.bytes[actor.evidence.end_char] - start;
                    !quotes
                        .iter()
                        .any(|quote| actor_start < quote.end && quote.start < actor_end)
                })
            })
        };
        let actor_from_subject = subject.is_some();
        let explicit_subject = frame.as_ref().is_some_and(|frame| !frame.inherit_actor);
        let mut available_actor = subject.or_else(|| {
            if unit.quote || explicit_subject {
                None
            } else {
                focus.as_ref()?.carry(unit, text, lexicon)
            }
        });
        // A nearby speech tag does not establish who the quoted rank phrase
        // refers to. The separate written-speech output retains that tag.
        if source_mentions && unit.quote && realm.is_some() {
            available_actor = None;
        }
        // Scope checks must inspect the same unquoted segment as the selected
        // narrative predicate. A negation or condition inside a written quote
        // is not a qualifier on the outside action.
        let selected_offset = realm
            .as_ref()
            .map(|hit| hit.offset)
            .or_else(|| pred.map(|(_, offset, _)| offset));
        let selected_segment = if !unit.quote {
            quote_ranges
                .as_ref()
                .and_then(|quotes| outside_quote_segment(text, quotes, selected_offset?))
        } else {
            None
        };
        let status_text = selected_segment
            .as_ref()
            .map_or(text, |range| &text[range.clone()]);
        let attempt_cue = if unit.quote {
            None
        } else {
            quote_ranges
                .as_ref()
                .and_then(|quotes| {
                    outside_attempt_cue(
                        text,
                        quotes,
                        selected_offset?,
                        &[
                            "試圖",
                            "试图",
                            "嘗試",
                            "尝试",
                            "以為",
                            "以为",
                            "原本要",
                            "本來要",
                            "本来要",
                            "原先要",
                        ],
                        &["未遂", "失敗", "失败"],
                    )
                })
                .map(|cue| source.span(start + cue.start, start + cue.end))
        };
        // Written possibility and attributed claims do not establish that a
        // neighboring action occurred. Keep the cue independent of the event.
        let modality_cue = if unit.quote {
            None
        } else {
            quote_ranges
                .as_ref()
                .and_then(|quotes| {
                    let offset = selected_offset?;
                    outside_attempt_cue(text, quotes, offset, WRITTEN_MODALITY_CUES, &[]).or_else(
                        || {
                            if source_mentions {
                                outside_attempt_cue(
                                    text,
                                    quotes,
                                    offset,
                                    SOURCE_ONLY_MODALITY_CUES,
                                    &[],
                                )
                            } else {
                                None
                            }
                        },
                    )
                })
                .map(|cue| source.span(start + cue.start, start + cue.end))
        };
        let plan_cue = if unit.quote {
            None
        } else {
            quote_ranges
                .as_ref()
                .and_then(|quotes| {
                    outside_attempt_cue(
                        text,
                        quotes,
                        selected_offset?,
                        &["原本要", "本來要", "本来要", "原先要"],
                        &[],
                    )
                })
                .map(|cue| source.span(start + cue.start, start + cue.end))
        };
        // An adjacent written cancellation is a source cue only. Its relation
        // to this plan remains unverified and must not authenticate an event.
        let cancellation_cue = plan_cue.and_then(|cue| {
            if !matches!(
                source.slice(&cue),
                "原本要" | "本來要" | "本来要" | "原先要"
            ) {
                return None;
            }
            let after_start = source.bytes[unit.evidence.end_char];
            let after_end = source.bytes[unit.sentence.end_char];
            let after = source.text.get(after_start..after_end)?;
            let trimmed =
                after.trim_start_matches(|ch: char| ch.is_whitespace() || matches!(ch, '，' | ','));
            let skipped = after.len() - trimmed.len();
            for phrase in ["後來作罷", "后来作罢"] {
                if trimmed.starts_with(phrase) {
                    let word = if phrase.ends_with("作罷") {
                        "作罷"
                    } else {
                        "作罢"
                    };
                    let at = after_start + skipped + phrase.len() - word.len();
                    return Some(source.span(at, at + word.len()));
                }
            }
            None
        });
        let question_cue = if unit.quote {
            None
        } else {
            quote_ranges
                .as_ref()
                .and_then(|quotes| outside_question_cue(text, quotes))
                .map(|cue| source.span(start + cue.start, start + cue.end))
                .or_else(|| {
                    let quotes = written_quote_ranges(sentence)?;
                    if quotes.is_empty() {
                        return None;
                    }
                    let cue = outside_question_cue(sentence, &quotes)?;
                    let sentence_start = source.bytes[unit.sentence.start_char];
                    Some(source.span(sentence_start + cue.start, sentence_start + cue.end))
                })
        };
        let earlier_scope = if unit.quote {
            Some(std::borrow::Cow::Borrowed(earlier))
        } else {
            unquoted_scope_text(earlier)
        };
        let earlier_belief = if unit.quote
            || starts_any(
                status_text.trim_start(),
                &["但", "然而", "不過", "不过", "卻", "却"],
            ) {
            None
        } else {
            earlier_written_scope_cue(earlier, &["以為", "以为"]).map(|cue| {
                let base = source.bytes[unit.sentence.start_char];
                source.span(base + cue.start, base + cue.end)
            })
        };
        let earlier_modality = if unit.quote
            || starts_any(
                status_text.trim_start(),
                &[
                    "但", "然而", "不過", "不过", "卻", "却", "隨後", "随后", "接著", "接着",
                    "然後", "然后", "於是", "于是",
                ],
            ) {
            None
        } else {
            earlier_written_scope_cue(earlier, WRITTEN_MODALITY_CUES)
                .or_else(|| {
                    if source_mentions {
                        earlier_written_scope_cue(earlier, SOURCE_ONLY_MODALITY_CUES)
                    } else {
                        None
                    }
                })
                .map(|cue| {
                    let base = source.bytes[unit.sentence.start_char];
                    source.span(base + cue.start, base + cue.end)
                })
        };
        let mut status = event_status(
            status_text,
            earlier_scope.as_deref().unwrap_or(earlier),
            if !unit.quote
                && realm.is_some()
                && unquoted_scope_text(sentence).is_some_and(|scope| contains_any(&scope, REPORTED))
            {
                "reported"
            } else {
                context
            },
            unit.quote,
            available_actor.is_some(),
            realm.as_ref(),
            if realm.is_some() {
                "realm_change"
            } else {
                pred.map_or("action", |value| value.0)
            },
        );
        let unasserted_predicate = !unit.quote
            && (question_cue.is_some() || attempt_cue.is_some() || modality_cue.is_some() || earlier_belief.is_some() || earlier_modality.is_some()
                || status_text.trim_end().ends_with(['？', '?'])
                || ends_with_question_particle(status_text)
                || selected_offset.is_some_and(|offset| {
                    let segment_start = selected_segment.as_ref().map_or(0, |range| range.start);
                    offset.checked_sub(segment_start)
                        .and_then(|local| status_text.get(..local))
                        .is_none_or(|prefix| contains_any(prefix, &["似乎", "彷彿", "仿佛"]))
                })
                // A directly postposed 不了 may express inability or may
                // belong to a following word such as 不了解. Either way the
                // written scope alone does not establish an occurrence.
                || pred.is_some_and(|(_, offset, word)| {
                    let segment_start = selected_segment.as_ref().map_or(0, |range| range.start);
                    offset.checked_sub(segment_start)
                        .and_then(|local| local.checked_add(word.len()))
                        .and_then(|end| status_text.get(end..))
                        .is_some_and(|tail| tail.starts_with("不了"))
                })
                || selected_offset.is_some_and(|offset| {
                    let segment_start = selected_segment.as_ref().map_or(0, |range| range.start);
                    offset.checked_sub(segment_start)
                        .filter(|&local| status_text.is_char_boundary(local))
                        .is_none_or(|local| predicate_has_unasserted_scope(status_text, local))
                }));
        if unasserted_predicate && matches!(status, "occurred" | "failed") {
            status = "uncertain";
        }
        if source_mentions && unit.quote && matches!(status, "occurred" | "failed") {
            status = "uncertain";
        }
        if !unit.quote && earlier_scope.is_none() && matches!(status, "occurred" | "failed") {
            status = "uncertain";
        }
        if damaged_quote {
            available_actor = None;
            if matches!(status, "occurred" | "failed" | "reported") {
                status = "uncertain";
            }
        }
        if let Some(frame) = &frame {
            status = match status {
                "occurred" if frame.blocked => "uncertain",
                "uncertain"
                    if !damaged_quote
                        && !unasserted_predicate
                        && frame.completed
                        && !frame.blocked =>
                {
                    "occurred"
                }
                _ => status,
            };
        }
        if !damaged_quote
            && !unasserted_predicate
            && narrative.is_some()
            && !unit.quote
            && frame.as_ref().is_none_or(|frame| !frame.blocked)
            && realm.is_none()
            && status == "uncertain"
            && let Some((_, offset, _)) = pred
            && omitted_disposal_subject(text, offset)
            && !contains_any(text, THOUGHT)
            && let Some(named) = detached_predecessor(&source, &units[..index], unit)
            && source
                .first_name(start, start + text.len(), lexicon)
                .is_none()
            && let (Some(focus), Some(last_event)) = (&focus, result.events.last())
            && let Some(actor) = focus.carry_from(named, last_event)
        {
            available_actor = Some(actor);
            // An uncertain action outside a realm change lacks only its actor.
            status = "occurred";
        }
        if !unit.quote {
            if let Some(actor) = &available_actor {
                focus = Some(Focus {
                    actor: actor.clone(),
                    last_reference: unit.evidence,
                    sentence: unit.sentence.start_char,
                });
            } else if explicit_subject || introduces_subject(text) {
                focus = None;
            }
        }
        let destination = pred.and_then(|(kind, offset, word)| {
            let limit = outside_pred
                .as_ref()
                .filter(|(_, candidate, _, _)| *candidate == offset)
                .map_or(text.len(), |(_, _, _, segment)| segment.end);
            (kind == "movement"
                && status == "occurred"
                && !starts_any(word, &["離", "离", "出", "逃", "遁離"]))
            .then(|| place_name(&text[offset + word.len()..limit], lexicon))
            .flatten()
        });
        let establishes_place = (!damaged_quote
            && starts_any(text, &["在", "到了", "進了", "洞府內", "密室內"]))
        .then(|| place_name(text, lexicon))
        .flatten();
        let changed_place = destination.is_some_and(|place| {
            result
                .sections
                .last()
                .and_then(|scene| scene.place.as_deref())
                != Some(place)
        });
        let new_scene = result.sections.is_empty()
            || context != prior_context
            || changed_place
            || (!unit.quote && unit.first_in_line && temporal_shift(text).is_some());
        if new_scene {
            result.sections.push(TextSection {
                evidence: unit.evidence,
                place: destination.or(establishes_place).map(str::to_owned),
                context: context.into(),
                units: Vec::new(),
            });
        }
        let scene = result.sections.last_mut().unwrap();
        scene.evidence.end_char = unit.evidence.end_char;
        let thought = if unit.quote {
            contains_any(
                earlier,
                &["暗想", "暗道", "心想", "心中想", "心里想", "心裡想"],
            )
        } else if source_mentions {
            // A later cue or a cue inside a quote must not erase an outside action.
            pred.map_or_else(
                || {
                    context_text
                        .as_deref()
                        .is_none_or(|scope| contains_any(scope, THOUGHT))
                },
                |(_, offset, _)| {
                    unquoted_scope_text(&text[..offset])
                        .as_deref()
                        .is_none_or(|scope| contains_any(scope, THOUGHT))
                },
            )
        } else {
            contains_any(text, THOUGHT)
        };
        let frame_kind = if damaged_quote {
            "unresolved_quote"
        } else if thought {
            "thought"
        } else if unit.quote {
            "dialogue"
        } else if (pred.is_some()
            || quote_ranges
                .as_ref()
                .and_then(|ranges| hearing_quote(text, ranges))
                .is_some())
            && (matches!(status, "occurred" | "failed" | "dream")
                || (status == "uncertain" && realm.is_none()))
        {
            "action"
        } else {
            "narration"
        };
        scene.units.push(TextUnit {
            kind: frame_kind.into(),
            speaker: available_actor
                .as_ref()
                .filter(|_| unit.quote)
                .map(|actor| actor.name.to_owned()),
            evidence: unit.evidence,
        });
        if !unit.quote {
            possible_aliases(&source, &unit.evidence, &mut result.reviews, lexicon);
        }
        if !unit.quote
            && !damaged_quote
            && let Some((cue, quote)) = quote_ranges
                .as_ref()
                .and_then(|ranges| hearing_quote(text, ranges))
        {
            let outside_before = &text[..cue.end];
            let outside_after = &text[quote.end..];
            let uncertain = contains_any(
                outside_before,
                &[
                    "沒有", "没有", "沒", "没", "未", "不", "無", "无", "差點", "差点", "幾乎",
                    "几乎",
                ],
            ) || contains_any(outside_before, HYPOTHETICAL)
                || contains_any(outside_before, PLANNED)
                || outside_after.contains(['？', '?'])
                || ends_with_question_particle(outside_after)
                || question_cue.is_some()
                || available_actor.is_none();
            let hearing_status = match context {
                "dream" => "dream",
                "reported" => "reported",
                _ if uncertain => "uncertain",
                _ => "occurred",
            };
            let cue_span = source.span(start + cue.start, start + cue.end);
            let quote_span = source.span(start + quote.start, start + quote.end);
            let mut evidence = vec![cue_span, quote_span];
            if let Some(question) = question_cue
                && !evidence.contains(&question)
            {
                evidence.push(question);
            }
            if let Some(actor) = &available_actor
                && !evidence.contains(&actor.evidence)
            {
                evidence.push(actor.evidence);
            }
            if let Some(observations) = mention_actors.as_deref_mut() {
                observations.push(capture_mention_actor(
                    &source,
                    lexicon,
                    result.events.len(),
                    available_actor.as_ref(),
                    actor_from_subject,
                ));
            }
            let event_index = result.events.len();
            result.events.push(Event {
                kind: "action".to_owned(),
                status: hearing_status.to_owned(),
                actor: available_actor.as_ref().map(|actor| actor.name.clone()),
                object: None,
                action: source.slice(&cue_span).to_owned(),
                evidence,
                recap: false,
            });
            record_mention_predicate(
                mention_predicates.as_deref_mut(),
                event_index,
                cue_span,
                MentionPredicateBasis::HearingCue,
            );
        }
        // Complete inline quotes, including content heard by a listener, are
        // independent written candidates. They do not identify the listener
        // as the actor and do not establish whether the content happened.
        if source_mentions
            && !unit.quote
            && !damaged_quote
            && narrative.is_some()
            && let Some(quotes) = &quote_ranges
        {
            for quote in quotes {
                let first_quote_event = result.events.len();
                let quoted_unit = Unit {
                    evidence: source.span(start + quote.start, start + quote.end),
                    sentence: unit.sentence,
                    speech_context: unit.speech_context,
                    quote: true,
                    first_in_line: false,
                };
                append_quote_source_candidates(
                    &source,
                    &quoted_unit,
                    lexicon,
                    &mut result.events,
                    first_quote_event,
                    None,
                    mention_predicates.as_deref_mut(),
                );
            }
        }
        if source_mentions
            && !unit.quote
            && !damaged_quote
            && narrative.is_some()
            && let Some(quotes) = &quote_ranges
        {
            let primary_predicate = if !thought && realm.is_none() {
                pred.map(|(kind, offset, word)| (offset, source_kind(kind, word)))
            } else {
                None
            };
            let primary_realm_byte = realm
                .as_ref()
                .filter(|_| !thought)
                .map(|hit| start + hit.offset);
            append_narration_source_candidates(
                &source,
                unit,
                text,
                quotes,
                lexicon,
                &mut result.events,
                first_event,
                primary_predicate,
                primary_realm_byte,
                mention_predicates.as_deref_mut(),
            );
        }
        let cessation = if !damaged_quote
            && quote_damage.is_none()
            && narrative.is_some()
            && unit.quote
            && !thought
        {
            written_cessation_tag(&source, unit, context, lexicon)
        } else {
            CessationScan::NoMatch
        };
        if let CessationScan::Unexamined(reason) = cessation {
            result.reviews.push(Review {
                kind: "unexamined_cessation_tag".into(),
                detail: reason.into(),
                evidence: unit.evidence,
            });
        }
        if let CessationScan::Matched(tag) = cessation {
            let first_new_event = result.events.len();
            append_reported_actions(
                &source,
                unit,
                units.get(index + 1),
                None,
                lexicon,
                &mut result.events,
                mention_predicates.as_deref_mut(),
            );
            for event in &mut result.events[first_new_event..] {
                // Legacy output has no role/identity certainty fields. Preserve
                // the quote and continuation-stop source, not inferred roles.
                event.actor = None;
                event.object = None;
                event.evidence.truncate(1);
                event.evidence.push(tag);
            }
        } else if !damaged_quote
            && narrative.is_some()
            && unit.quote
            && !thought
            && context != "dream"
            && (available_actor.is_some() || quote_conflicting || quote_identity_unknown)
        {
            append_reported_actions(
                &source,
                unit,
                units.get(index + 1),
                available_actor.as_ref(),
                lexicon,
                &mut result.events,
                mention_predicates.as_deref_mut(),
            );
        }
        if source_mentions && unit.quote && !damaged_quote && narrative.is_some() {
            append_quote_source_candidates(
                &source,
                unit,
                lexicon,
                &mut result.events,
                first_event,
                realm
                    .as_ref()
                    .filter(|_| !thought)
                    .map(|hit| start + hit.offset),
                mention_predicates.as_deref_mut(),
            );
        }
        let narrowed_evidence = if !unit.quote {
            let event_offset = realm
                .as_ref()
                .map(|hit| hit.offset)
                .or_else(|| pred.map(|(_, offset, _)| offset));
            quote_ranges
                .as_ref()
                .filter(|quotes| !quotes.is_empty())
                .and_then(|quotes| {
                    let range = outside_quote_segment(text, quotes, event_offset?)?;
                    source.trim(start + range.start, start + range.end)
                })
        } else {
            None
        };
        let primary_written_predicate =
            if source_mentions && !unit.quote && !thought && realm.is_none() {
                pred.map(|(_, offset, word)| {
                    source.span(start + offset, start + offset + word.len())
                })
            } else {
                None
            };
        let quoted_rank_ranges = if source_mentions && unit.quote {
            realm.as_ref().and_then(|hit| {
                let range = rank_predicate(text, hit.offset)?;
                text.get(range.start..hit.end_offset)?;
                Some((
                    source.span(start + range.start, start + hit.end_offset),
                    source.span(start + range.start, start + range.end),
                ))
            })
        } else {
            None
        };
        // A damaged non-quote unit can still contain an explicitly written
        // rank predicate. Keep only its source location; it does not establish
        // a participant, occurrence, or quoted/narrated interpretation.
        let damaged_rank_predicate = if source_mentions && damaged_quote && !unit.quote && !thought
        {
            realm
                .as_ref()
                .and_then(|hit| rank_predicate(text, hit.offset))
                .map(|range| source.span(start + range.start, start + range.end))
        } else {
            None
        };
        let primary_rank_span = if source_mentions {
            realm
                .as_ref()
                .map(|hit| source.span(start + hit.offset, start + hit.end_offset))
        } else {
            None
        };
        let (kind, mut object) = if thought {
            continue;
        } else if let Some(realm) = realm {
            ("realm_change", Some(realm.target.to_owned()))
        } else if let Some((kind, offset, word)) = pred.filter(|_| !unit.quote) {
            let object = if kind == "movement" || frame.is_none() {
                if let Some((_, _, _, segment)) = outside_pred
                    .as_ref()
                    .filter(|(_, candidate, _, _)| *candidate == offset)
                {
                    object_after(
                        &source,
                        source.span(start + segment.start, start + segment.end),
                        offset - segment.start,
                        word,
                        kind,
                        lexicon,
                    )
                } else {
                    object_after(&source, unit.evidence, offset, word, kind, lexicon)
                }
            } else {
                None
            };
            (
                kind,
                object.or_else(|| frame.as_ref()?.patient_name.map(str::to_owned)),
            )
        } else {
            continue;
        };
        let kind = if source_mentions {
            pred.map_or(kind, |(_, _, word)| source_kind(kind, word))
        } else {
            kind
        };
        if source_mentions && kind == "transfer" {
            object = None;
        }
        if damaged_quote {
            object = None;
        }
        let context_evidence = narrowed_evidence.unwrap_or(unit.evidence);
        let mut evidence = if let Some((phrase, predicate)) = quoted_rank_ranges {
            let mut spans = vec![phrase];
            if !spans.contains(&predicate) {
                spans.push(predicate);
            }
            if !spans.contains(&context_evidence) {
                spans.push(context_evidence);
            }
            spans
        } else {
            vec![context_evidence]
        };
        if let Some(span) = primary_written_predicate
            && !evidence.contains(&span)
        {
            evidence.push(span);
        }
        for span in [modality_cue, earlier_modality].into_iter().flatten() {
            result.reviews.push(Review {
                kind: "written_modality_cue".into(),
                detail: "unverified_association".into(),
                evidence: span,
            });
        }
        if let Some(span) = earlier_belief
            && !evidence.contains(&span)
        {
            evidence.push(span);
        }
        if let Some(span) = cancellation_cue {
            // The written cue is retained separately; association with this
            // or any other action is not established by adjacency alone.
            result.reviews.push(Review {
                kind: "written_cancellation_cue".into(),
                detail: "unverified_association".into(),
                evidence: span,
            });
        }
        if let Some(span) = attempt_cue
            && !evidence.iter().any(|existing| {
                existing.start_char <= span.start_char && span.end_char <= existing.end_char
            })
        {
            evidence.push(span);
        }
        if let Some(span) = question_cue
            && !evidence.iter().any(|existing| {
                existing.start_char <= span.start_char && span.end_char <= existing.end_char
            })
        {
            evidence.push(span);
        }
        // The cue may come from an earlier source clause, but quoted words do
        // not qualify an outside action. Preserve only an outside literal cue.
        if status == "hypothetical"
            && !earlier.is_empty()
            && let Some((offset, word)) = unquoted_conditional_cue(earlier)
        {
            let sentence_start = source.bytes[unit.sentence.start_char];
            let span = source.span(
                sentence_start + offset,
                sentence_start + offset + word.len(),
            );
            if !evidence.contains(&span) {
                evidence.push(span);
            }
        }
        if let Some(actor) = available_actor.as_ref()
            && (actor.evidence.start_char < evidence[0].start_char
                || actor.evidence.end_char > evidence[0].end_char)
        {
            evidence.push(actor.evidence);
        }
        let mut action_span = narrowed_evidence.unwrap_or(unit.evidence);
        if let Some(frame) = &frame {
            let result_span = frame
                .result
                .filter(|span| status == "occurred" || span.end_char <= unit.evidence.end_char);
            if let Some(span) = result_span {
                action_span.end_char = action_span.end_char.max(span.end_char);
            }
            for span in [
                frame.actor_surface,
                Some(frame.predicate),
                Some(frame.patient),
                result_span,
            ]
            .into_iter()
            .flatten()
            {
                if !evidence.contains(&span) {
                    evidence.push(span);
                }
            }
            if let Some(frames) = &mut clause_frames {
                frames.push(ClauseFrame {
                    event_index: result.events.len(),
                    predicate: frame.predicate,
                    actor_surface: frame.actor_surface,
                    patient: frame.patient,
                    result: result_span,
                });
            }
        }
        if narrative.is_some()
            && kind == "action"
            && status == "occurred"
            && object.is_none()
            && let Some((_, offset, _)) = pred
            && let Some(previous) = index.checked_sub(1).map(|index| &units[index])
            && let Some(patient) = transformed_patient(&source, previous, unit, offset)
        {
            object = Some(patient.to_owned());
            // Keep the action as the first span for timeline ordering.
            evidence.push(previous.evidence);
        }
        if kind == "realm_change" && status == "uncertain" {
            let (kind, detail) = if available_actor.is_none() {
                ("unknown_actor", "境界變化的當事人待確認")
            } else {
                ("uncertain_outcome", "境界變化是否已發生待確認")
            };
            result.reviews.push(Review {
                kind: kind.into(),
                detail: detail.into(),
                evidence: evidence[0],
            });
        }
        if let Some(observations) = mention_actors.as_deref_mut() {
            observations.push(capture_mention_actor(
                &source,
                lexicon,
                result.events.len(),
                available_actor.as_ref(),
                actor_from_subject,
            ));
        }
        let predicate_observation = primary_written_predicate
            .map(|span| (span, MentionPredicateBasis::SelectedNarrationPredicate))
            .or_else(|| {
                quoted_rank_ranges
                    .map(|(_, span)| (span, MentionPredicateBasis::RecognizedQuotedRankPredicate))
            })
            .or_else(|| {
                damaged_rank_predicate
                    .filter(|span| {
                        evidence.iter().any(|proof| {
                            proof.start_char <= span.start_char && span.end_char <= proof.end_char
                        })
                    })
                    .map(|span| (span, MentionPredicateBasis::SelectedNarrationPredicate))
            });
        let event_index = result.events.len();
        push_event(
            &source,
            &mut result.events,
            Event {
                kind: kind.into(),
                status: status.into(),
                actor: available_actor.map(|actor| actor.name.to_owned()),
                object,
                action: excerpt(source.slice(&action_span)),
                evidence,
                recap: context == "flashback" || contains_any(text, RECALL),
            },
            source_mentions,
        );
        if source_mentions
            && result.events.len() == event_index + 1
            && kind == "realm_change"
            && let Some(span) = primary_rank_span
        {
            remember_written_rank(&mut represented_ranks, span);
        }
        if source_mentions
            && result.events.len() == event_index + 1
            && let Some((span, basis)) = predicate_observation
        {
            record_mention_predicate(mention_predicates.as_deref_mut(), event_index, span, basis);
        }
    }
    // In damaged material the ordinary parser avoids assigning participants
    // or asserting events, but that must not erase an independently visible
    // written predicate. Scan only admitted units: split_units_impl has already
    // excluded titles and configured document/line footers. A cue is not proof
    // of a quotation, occurrence, actor, or any semantic relationship.
    if source_mentions
        && narrative.is_some()
        && let Some(damage_start) = quote_damage
    {
        let mut seen: HashSet<(usize, usize)> = recorded_predicates
            .iter()
            .filter_map(|item| item.written_predicate)
            .map(|span| (span.start_char, span.end_char))
            .collect();
        let realm_predicates: HashSet<(usize, usize)> = recorded_predicates
            .iter()
            .filter(|item| {
                result
                    .events
                    .get(item.event_index)
                    .is_some_and(|event| event.kind == "realm_change")
            })
            .filter_map(|item| item.written_predicate)
            .map(|span| (span.start_char, span.end_char))
            .collect();
        for unit in units
            .iter()
            .filter(|unit| unit.evidence.end_char > damage_start)
        {
            let body = source.slice(&unit.evidence);
            let start = source.bytes[unit.evidence.start_char];
            let mut object_coverage = None;
            for mut candidate in quote_predicates(body, lexicon) {
                if let Some((left, right)) = candidate.rank_byte.zip(candidate.rank_end)
                    && !recovery_rank_allowed(body, lexicon, left, right, &mut object_coverage)
                {
                    // Preserve the exact written verb without asserting a rank
                    // classification from an object or modifier substring.
                    candidate.kind = "action";
                    candidate.rank_byte = None;
                    candidate.rank_end = None;
                }
                let predicate = source.span(start + candidate.start, start + candidate.end);
                let rank_span = candidate
                    .rank_byte
                    .zip(candidate.rank_end)
                    .map(|(left, right)| source.span(start + left, start + right));
                if let Some(rank) = rank_span
                    && realm_predicates.contains(&(predicate.start_char, predicate.end_char))
                {
                    remember_written_rank(&mut represented_ranks, rank);
                }
                if !seen.insert((predicate.start_char, predicate.end_char)) {
                    continue;
                }
                if let Some(rank) = rank_span
                    && !remember_written_rank(&mut represented_ranks, rank)
                {
                    continue;
                }
                let event_index = result.events.len();
                let mut evidence = vec![predicate];
                if let Some(rank) = rank_span
                    && rank != predicate
                {
                    evidence.push(rank);
                }
                result.events.push(Event {
                    kind: candidate.kind.to_owned(),
                    status: "uncertain".into(),
                    actor: None,
                    object: None,
                    action: source.slice(&predicate).to_owned(),
                    evidence,
                    recap: false,
                });
                record_mention_predicate(
                    Some(&mut recorded_predicates),
                    event_index,
                    predicate,
                    MentionPredicateBasis::AdditionalWrittenCue,
                );
            }
            // Some recognized rank phrases have no safely locatable verb.
            // Inspect each caller-provided surface independently; the source
            // sidecar remains unavailable and the classification unverified.
            let mut ranks = Vec::new();
            for &(surface, _) in lexicon.ranks {
                if surface.is_empty() {
                    continue;
                }
                for (left, _) in body.match_indices(surface) {
                    let right = left + surface.len();
                    if recovery_rank_allowed(body, lexicon, left, right, &mut object_coverage) {
                        ranks.push((left, right));
                    }
                }
            }
            ranks.sort_unstable_by_key(|&(left, right)| (left, std::cmp::Reverse(right - left)));
            for (left, right) in ranks {
                let rank = source.span(start + left, start + right);
                if !remember_written_rank(&mut represented_ranks, rank) {
                    continue;
                }
                result.events.push(Event {
                    kind: "realm_change".into(),
                    status: "uncertain".into(),
                    actor: None,
                    object: None,
                    action: source.slice(&rank).to_owned(),
                    evidence: vec![rank],
                    recap: false,
                });
            }
        }
    }
    if let Some(output) = output_predicates {
        output.extend(recorded_predicates);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: Lexicon<'static> = Lexicon {
        actors: &[("林嶼", "林嶼"), ("沈牧", "沈牧"), ("阿嶼", "林嶼")],
        places: &["屋子", "石室", "山谷", "院子"],
        objects: &[("令牌", "令牌"), ("藥瓶", "藥瓶")],
        ranks: &[
            ("築基", "築基"),
            ("筑基", "築基"),
            ("結丹", "結丹"),
            ("金丹", "結丹"),
            ("煞丹", "結丹"),
            ("元嬰", "元嬰"),
            ("化神", "化神"),
        ],
        formed_ranks: &["煞丹"],
        rank_blocked_prefixes: &[],
        footers: &["<!fiction-footer"],
        line_footers: &[],
    };

    fn analyze_fixture(text: &str, title: &str) -> Analysis {
        analyze(text, title, &FIXTURE)
    }

    #[test]
    fn offsets_cover_quotes_and_supplementary_characters() {
        let text =
            "\u{feff}第一章 試章\n\n某小說 第一章 試章\n𠮷。林嶼抬手：“現在走吧！”  他回到院子。";
        let result = analyze_fixture(text, "第一章 試章");
        let frames: Vec<_> = result
            .sections
            .iter()
            .flat_map(|scene| &scene.units)
            .collect();
        let source = Source::new(text);
        assert_eq!(source.slice(&frames[0].evidence), "𠮷。");
        assert_eq!(
            source.slice(
                &frames
                    .iter()
                    .find(|frame| frame.kind == "dialogue")
                    .unwrap()
                    .evidence
            ),
            "“現在走吧！”"
        );
    }

    #[test]
    fn cultivation_outcomes_require_a_rank_and_context() {
        let text = "林嶼打算吞服筑基丹，準備築基沖關。林嶼凝丹后取出了藥丸。\n林嶼結丹未成。\n沈牧終究結成了煞丹。\n林嶼終于金丹大成了！\n若林嶼侵占別人元嬰成功，就可使用秘術。";
        let result = analyze_fixture(text, "");
        let outcomes: Vec<_> = result
            .events
            .iter()
            .filter(|event| event.kind == "realm_change")
            .map(|event| {
                (
                    event.actor.as_deref(),
                    event.object.as_deref(),
                    event.status.as_str(),
                )
            })
            .collect();
        assert_eq!(
            outcomes,
            [
                (Some("林嶼"), Some("築基"), "planned"),
                (Some("林嶼"), Some("結丹"), "failed"),
                (Some("沈牧"), Some("結丹"), "occurred"),
                (Some("林嶼"), Some("結丹"), "occurred"),
            ]
        );
    }

    #[test]
    fn aliases_require_a_nearby_name_and_a_naming_cue() {
        let text = "林嶼來到屋內。客人直奔林羽所在的屋子。阿嶼姓林名嶼。";
        let result = analyze_fixture(text, "");
        assert_eq!(result.reviews.len(), 1);
        assert_eq!(Source::new(text).slice(&result.reviews[0].evidence), "林羽");
        assert!(
            result
                .events
                .iter()
                .all(|event| event.actor.as_deref() != Some("林羽"))
        );
    }

    #[test]
    fn dream_and_report_do_not_become_asserted_story_events() {
        let result = analyze_fixture(
            "林嶼夢見自己擊敗了少年。\n第二天他醒來。\n據說林嶼已返回山谷。",
            "",
        );
        assert!(
            result
                .events
                .iter()
                .any(|event| event.kind == "combat" && event.status == "dream")
        );
        assert!(
            result
                .events
                .iter()
                .any(|event| event.kind == "movement" && event.status == "reported")
        );
    }

    #[test]
    fn attributes_an_unlisted_title_only_when_it_acts() {
        let result = analyze_fixture(
            "白袍長老取出令牌。林嶼望著白袍長老，抬手接過。\n“收好！”青衡真人說道。",
            "",
        );
        assert!(
            result
                .events
                .iter()
                .any(|event| event.actor.as_deref() == Some("白袍長老"))
        );
        assert!(
            result.events.iter().any(
                |event| event.actor.as_deref() == Some("林嶼") && event.action.contains("抬手")
            )
        );
        assert!(
            result
                .sections
                .iter()
                .flat_map(|scene| &scene.units)
                .any(|frame| frame.kind == "dialogue"
                    && frame.speaker.as_deref() == Some("青衡真人"))
        );
    }

    #[test]
    fn strips_inline_publisher_footer_and_following_lines_without_shifting_offsets() {
        let body = "𠮷。林嶼推開石室。".repeat(320);
        let text = format!("{body}<!FICTION-FOOTER(假想網頁提示)\n快打開網站。\n");
        let source = Source::new(&text);
        let result = analyze_fixture(&text, "");
        let frame = result.sections.last().unwrap().units.last().unwrap();
        assert_eq!(frame.evidence.end_char, body.chars().count());
        assert_eq!(source.slice(&frame.evidence), "林嶼推開石室。");
        assert!(
            result
                .events
                .iter()
                .all(|event| !event.action.contains("網站"))
        );
    }

    #[test]
    fn keeps_advancement_methods_distinct_from_realized_outcomes() {
        let result = analyze_fixture(
            "林嶼研讀如何進入化神期。林嶼習得此法，便可踏入元嬰期。林嶼已進入了結丹期。",
            "",
        );
        let outcomes: Vec<_> = result
            .events
            .iter()
            .filter(|event| event.kind == "realm_change")
            .map(|event| (event.object.as_deref(), event.status.as_str()))
            .collect();
        assert_eq!(
            outcomes,
            [
                (Some("化神"), "hypothetical"),
                (Some("元嬰"), "hypothetical"),
                (Some("結丹"), "occurred"),
            ]
        );
    }

    #[test]
    fn carries_explicit_pronouns_until_another_subject_appears() {
        let result = analyze_fixture(
            "林嶼揮手。只見他抬手。此刻他取出令牌。一名修士取出藥瓶。此時他收起藥瓶。",
            "",
        );
        let actor_for = |clause: &str| {
            result
                .events
                .iter()
                .find(|event| event.action.contains(clause))
                .unwrap()
        };
        assert_eq!(actor_for("只見他抬手").actor.as_deref(), Some("林嶼"));
        assert_eq!(actor_for("此刻他取出").actor.as_deref(), Some("林嶼"));
        let other = actor_for("此時他收起");
        assert_eq!(other.actor, None);
        assert_eq!(other.status, "uncertain");
        assert!(
            result
                .sections
                .iter()
                .flat_map(|scene| &scene.units)
                .any(|frame| frame.kind == "action")
        );
    }
}
