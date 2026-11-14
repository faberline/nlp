//! Reconciles event evidence and derives a timeline from temporal cues in the source.

use crate::language::HYPOTHETICAL;
use crate::recap_time;
use crate::{Event, Span};
use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, BTreeSet};

/// Source text and its extracted events. Order is the chapter's source ordinal.
#[derive(Clone, Copy, Debug)]
pub struct TimelineDocument<'a> {
    pub order: u32,
    pub text: &'a str,
    pub events: &'a [Event],
}

/// A chapter-local range in Unicode scalar offsets, with an exclusive end.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocatedSpan {
    pub chapter: u32,
    pub span: Span,
}

/// A source event and the evidence it retains after reconciliation.
#[derive(Debug)]
pub struct TimelineEvent<'a> {
    pub chapter: u32,
    pub index: usize,
    pub event: &'a Event,
    pub evidence: Vec<LocatedSpan>,
}

/// A temporal relation with references to the before event, cue, and after event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Relation {
    pub before: usize,
    pub after: usize,
    pub evidence: [LocatedSpan; 3],
    pub supported: bool,
}

/// Events in source order, relations indexed into those events, and the story view.
#[derive(Debug)]
pub struct Timeline<'a> {
    pub events: Vec<TimelineEvent<'a>>,
    pub relations: Vec<Relation>,
    pub story_sequence: Vec<usize>,
    pub story_positions: Vec<Option<usize>>,
}

struct Text<'a> {
    document: TimelineDocument<'a>,
    bytes: Vec<usize>,
}

impl<'a> Text<'a> {
    fn new(document: TimelineDocument<'a>) -> Self {
        let bytes = document
            .text
            .char_indices()
            .map(|(byte, _)| byte)
            .chain([document.text.len()])
            .collect();
        Self { document, bytes }
    }

    fn slice(&self, start: usize, end: usize) -> &'a str {
        &self.document.text[self.bytes[start]..self.bytes[end]]
    }

    fn evidence(&self, span: Span) -> Result<LocatedSpan> {
        ensure!(
            span.start_char < span.end_char && span.end_char < self.bytes.len(),
            "章節 {} 的原文範圍無效：{}..{}",
            self.document.order,
            span.start_char,
            span.end_char
        );
        Ok(LocatedSpan {
            chapter: self.document.order,
            span,
        })
    }
}

type Precedence = BTreeMap<(usize, usize), LocatedSpan>;

const QUOTE_OPENERS: &[char] = &['“', '‘', '「', '『', '"', '《', '〈'];
const QUOTE_CLOSERS: &[char] = &['”', '’', '」', '』', '"', '》', '〉'];
const SENTENCE_ENDS: &[char] = &['。', '！', '？', '.', '!', '?'];

fn occurred(event: &Event) -> bool {
    matches!(event.status.as_str(), "occurred" | "failed")
}

// A day expression qualified by a bound such as "before the next day" is
// not the same cue as the unqualified day. Its bound alone does not establish
// order against the preceding event. This check is deliberately conservative;
// an unrecognized qualifier is not proof of a supported relationship.
fn has_qualified_calendar_cue(prefix: &str) -> bool {
    const CALENDAR: &[&str] = &["次日", "翌日", "隔天", "第二天", "昨天"];
    const BOUNDS: &[&str] = &[
        "之前", "以前", "之後", "之后", "以後", "以后", "過後", "过后",
    ];
    CALENDAR.iter().any(|cue| {
        prefix.match_indices(*cue).any(|(at, _)| {
            let tail = prefix[at + cue.len()..].trim_start();
            let clause = tail
                .split(['，', ',', '。', '！', '？', '；', ';', '\n', '\r'])
                .next()
                .unwrap_or("");
            let directional = clause.starts_with(['後', '后'])
                || (clause.starts_with('前')
                    && !["前往", "前行", "前進", "前进", "前去"]
                        .iter()
                        .any(|word| clause.starts_with(*word)));
            directional || BOUNDS.iter().any(|bound| clause.contains(*bound))
        })
    })
}

#[cfg(test)]
mod qualified_calendar_cue_tests {
    use super::{Event, Span, TimelineDocument, build, build_with_indefinite_delays};

    fn make_event(text: &str, phrase: &str) -> Event {
        let start = text.find(phrase).expect("test phrase");
        let start_char = text[..start].chars().count();
        Event {
            kind: "movement".into(),
            status: "occurred".into(),
            actor: Some("甲".into()),
            object: None,
            action: phrase.into(),
            evidence: vec![Span {
                start_char,
                end_char: start_char + phrase.chars().count(),
            }],
            recap: false,
        }
    }

    fn text_at(text: &str, span: Span) -> String {
        text.chars()
            .skip(span.start_char)
            .take(span.end_char - span.start_char)
            .collect()
    }

    #[test]
    fn qualified_calendar_bounds_do_not_establish_event_order() {
        for cue in [
            "次日之前",
            "翌日以前",
            "隔天之後",
            "第二天之前",
            "第二天前",
            "次日傍晚之前",
            "昨天之前",
        ] {
            let source = format!("𠮷。甲進入密室。\r\n{cue}，甲才離開洞府。");
            let events = [
                make_event(&source, "甲進入密室"),
                make_event(&source, "甲才離開洞府"),
            ];
            let docs = [TimelineDocument {
                order: 1,
                text: &source,
                events: &events,
            }];
            for timeline in [
                build(&docs).unwrap(),
                build_with_indefinite_delays(&docs).unwrap(),
            ] {
                assert_eq!(timeline.events.len(), 2, "{source}");
                assert_eq!(
                    text_at(&source, timeline.events[0].evidence[0].span),
                    "甲進入密室"
                );
                assert_eq!(
                    text_at(&source, timeline.events[1].evidence[0].span),
                    "甲才離開洞府"
                );
                assert!(
                    timeline.relations.iter().all(|r| !r.supported),
                    "{source}: {timeline:?}"
                );
            }
        }
    }

    #[test]
    fn unqualified_calendar_cues_remain_available_with_source_ranges() {
        for cue in ["次日", "翌日", "隔天", "第二天"] {
            let source = format!("𠮷。甲進入密室。\r\n{cue}，甲離開洞府。");
            let events = [
                make_event(&source, "甲進入密室"),
                make_event(&source, "甲離開洞府"),
            ];
            let timeline = build(&[TimelineDocument {
                order: 1,
                text: &source,
                events: &events,
            }])
            .unwrap();
            let [relation] = timeline.relations.as_slice() else {
                panic!("{source}: {timeline:?}")
            };
            assert!(relation.supported, "{source}");
            assert_eq!(text_at(&source, relation.evidence[1].span), cue);
        }
    }

    #[test]
    fn verb_after_next_day_is_not_misread_as_a_before_bound() {
        let source = "甲進入密室。次日前往山谷。";
        let events = [
            make_event(source, "甲進入密室"),
            make_event(source, "前往山谷"),
        ];
        let timeline = build(&[TimelineDocument {
            order: 1,
            text: source,
            events: &events,
        }])
        .unwrap();
        assert!(
            timeline
                .relations
                .iter()
                .any(|r| r.supported && text_at(source, r.evidence[1].span) == "次日")
        );
    }
}

fn signal(source: &Text<'_>, span: LocatedSpan, floor: usize) -> Option<(bool, Span)> {
    const CUES: &[(&str, bool)] = &[
        ("緊接著", false),
        ("紧接着", false),
        ("接下來", false),
        ("隨後", false),
        ("随后", false),
        ("隨后", false),
        ("緊接着", false),
        ("接著", false),
        ("接着", false),
        ("此後", false),
        ("此后", false),
        ("事後", false),
        ("事后", false),
        ("於是", false),
        ("于是", false),
        ("然後", false),
        ("然后", false),
        ("之後", false),
        ("之后", false),
        ("轉眼間", false),
        ("转眼间", false),
        ("次日", false),
        ("翌日", false),
        ("隔天", false),
        ("第二天", false),
        ("先前", true),
        ("早先", true),
        ("此前", true),
        ("昨天", true),
    ];
    let mut start = span.span.start_char.saturating_sub(30).max(floor);
    let end = span
        .span
        .end_char
        .min(span.span.start_char.saturating_add(36));
    if start >= end {
        return None;
    }
    // A cue in a preceding sentence, paragraph, or quotation belongs to that earlier unit.
    if let Some(boundary) = source
        .slice(start, span.span.start_char)
        .rfind(['。', '！', '？', '\n', '”', '」', '』'])
    {
        let prefix = source.slice(start, span.span.start_char);
        start += prefix[..boundary].chars().count() + 1;
    }
    let governing = source.slice(span.span.start_char, end);
    let end = governing
        .find(['，', ',', '。', '！', '？', '；', ';'])
        .map_or(end, |at| {
            span.span.start_char + governing[..at].chars().count()
        });
    let window = source.slice(start, end);
    let prefix = window
        .find(QUOTE_OPENERS)
        .map_or(window, |at| &window[..at]);
    // Do this before both the explicit-cue scan and fallback inference. In
    // particular, do not rediscover the inner "之前" or a later "才".
    if has_qualified_calendar_cue(prefix) {
        return None;
    }
    let mut found = None;
    for (cue, backward) in CUES {
        if let Some(at) = prefix.find(cue) {
            let position = start + prefix[..at].chars().count();
            if found
                .as_ref()
                .is_none_or(|(_, old): &(bool, Span)| position < old.start_char)
            {
                found = Some((
                    *backward,
                    Span {
                        start_char: position,
                        end_char: position + cue.chars().count(),
                    },
                ));
            }
        }
    }
    let explicit = found.or_else(|| relative_cue(prefix, start));
    if explicit.is_some() || floor == 0 || span.span.start_char.saturating_sub(floor) > 16 {
        return explicit;
    }
    let through_verb = source.slice(floor, end.min(span.span.start_char + 9));
    if through_verb.contains(['。', '！', '？', '\n', '”', '」', '』'])
        || through_verb.contains(QUOTE_OPENERS)
    {
        return None;
    }
    through_verb
        .chars()
        .enumerate()
        .find(|(_, ch)| matches!(ch, '便' | '遂' | '才'))
        .map(|(at, _)| {
            (
                false,
                Span {
                    start_char: floor + at,
                    end_char: floor + at + 1,
                },
            )
        })
}

fn relative_cue(prefix: &str, offset: usize) -> Option<(bool, Span)> {
    const UNITS: &[&str] = &["時辰", "时辰", "刻鐘", "刻钟", "年", "月", "日", "天"];
    measured_cue(prefix, offset, UNITS, false)
}

pub(crate) fn written_duration(prefix: &str, offset: usize) -> Option<(bool, Span)> {
    const UNITS: &[&str] = &[
        "盞茶的工夫",
        "盞茶工夫",
        "盞茶功夫",
        "柱香的工夫",
        "柱香工夫",
        "時辰",
        "时辰",
        "刻鐘",
        "刻钟",
        "分鐘",
        "分钟",
        "小時",
        "小时",
        "星期",
        "盞茶",
        "柱香",
        "年",
        "月",
        "日",
        "天",
        "刻",
        "秒",
        "周",
        "週",
    ];
    measured_cue(prefix, offset, UNITS, true)
}

fn measured_cue(
    prefix: &str,
    offset: usize,
    units: &[&str],
    compound: bool,
) -> Option<(bool, Span)> {
    const NUMBER: &str = "零〇一二兩两三四五六七八九十百千數数幾几半0123456789";
    let chars: Vec<_> = prefix.chars().collect();
    let mut cursor = 0;
    while cursor < chars.len() {
        let start = cursor;
        while cursor < chars.len() && NUMBER.contains(chars[cursor]) {
            cursor += 1;
        }
        if cursor == start {
            cursor += 1;
            continue;
        }
        if compound
            && matches!(chars.get(cursor), Some('.' | '點' | '点'))
            && chars.get(cursor + 1).is_some_and(|ch| NUMBER.contains(*ch))
        {
            cursor += 1;
            while cursor < chars.len() && NUMBER.contains(chars[cursor]) {
                cursor += 1;
            }
        }
        if matches!(chars.get(cursor), Some('多' | '餘' | '余')) {
            cursor += 1;
        }
        if matches!(chars.get(cursor), Some('個' | '个')) {
            cursor += 1;
            if matches!(chars.get(cursor), Some('多' | '餘' | '余')) {
                cursor += 1;
            }
        }
        for unit in units {
            let mut suffix = chars[cursor..].iter().copied();
            if !unit.chars().all(|ch| suffix.next() == Some(ch)) {
                continue;
            }
            let mut marker = suffix.next();
            let lead = usize::from(compound && matches!(marker, Some('之' | '以')));
            if lead == 1 {
                marker = suffix.next();
            }
            if let Some(marker @ ('前' | '後' | '后')) = marker {
                if compound && matches!(suffix.next(), Some('前' | '後' | '后')) {
                    continue;
                }
                return Some((
                    marker == '前',
                    Span {
                        start_char: offset + start,
                        end_char: offset + cursor + unit.chars().count() + lead + 1,
                    },
                ));
            }
        }
    }
    None
}

pub(crate) fn indefinite_duration(text: &str) -> bool {
    let elapsed = text.strip_suffix(['後', '后']).unwrap_or(text);
    let lead = elapsed.strip_prefix("不知").unwrap_or(elapsed);
    let Some(duration) = lead
        .strip_prefix("過了")
        .or_else(|| lead.strip_prefix("过了"))
    else {
        return false;
    };
    if lead.len() < elapsed.len() {
        duration == "多久"
    } else {
        matches!(
            duration,
            "片刻" | "半晌" | "許久" | "许久" | "一會" | "一会" | "一會兒" | "一会儿"
        )
    }
}

fn has_open_quote(text: &str) -> bool {
    let mut closers = Vec::new();
    for ch in text.chars() {
        if closers.last() == Some(&ch) {
            closers.pop();
        } else if let Some(index) = QUOTE_OPENERS.iter().position(|&open| open == ch) {
            closers.push(QUOTE_CLOSERS[index]);
        }
    }
    !closers.is_empty()
}

fn indefinite_delay(
    source: &Text<'_>,
    previous: Span,
    current: Span,
    actor: &str,
) -> Option<(bool, Span)> {
    if actor.is_empty() {
        return None;
    }
    let end = current.end_char.min(current.start_char.saturating_add(36));
    let window = source.slice(previous.end_char, end);
    let prior = source.slice(previous.start_char, previous.end_char);
    let next = if prior.trim_end().ends_with(SENTENCE_ENDS) {
        window
    } else {
        let (tail, next) = window.split_once(SENTENCE_ENDS)?;
        let tail = tail.trim();
        let stem = tail.strip_suffix('了').unwrap_or(tail);
        if !matches!(
            stem,
            "" | "過去" | "过去" | "過來" | "过来" | "下去" | "下來" | "下来" | "起來" | "起来"
        ) {
            return None;
        }
        next
    };
    let lead = next.trim_start();
    let (marker, governed) = lead.split_once(['，', ','])?;
    let cue = marker.trim_end();
    if !indefinite_duration(cue) {
        return None;
    }
    let start = end - lead.chars().count();
    let clause = governed
        .split(|ch| {
            SENTENCE_ENDS.contains(&ch) || matches!(ch, '，' | ',' | '；' | ';' | '\n' | '\r')
        })
        .next()?;
    let action = clause.trim_start().strip_prefix(actor)?.trim_start();
    let clause_end = end - governed.chars().count() + clause.chars().count();
    if start - previous.end_char > 24
        || current.start_char >= clause_end
        || action.is_empty()
        || action.starts_with(['若', '的'])
    {
        return None;
    }
    let before = &source.document.text[..source.bytes[previous.start_char]];
    let prefix = before
        .rsplit(|ch| SENTENCE_ENDS.contains(&ch) || matches!(ch, '\n' | '\r'))
        .next()
        .unwrap_or_default();
    let scope = &source.document.text[before.len() - prefix.len()..source.bytes[clause_end]];
    if scope
        .chars()
        .any(|ch| QUOTE_OPENERS.contains(&ch) || QUOTE_CLOSERS.contains(&ch))
        || scope.contains(['？', '?'])
        || scope.trim_start().starts_with('若')
        || scope.contains("假使")
        || HYPOTHETICAL.iter().any(|cue| scope.contains(cue))
        || has_open_quote(before)
    {
        return None;
    }
    Some((
        false,
        Span {
            start_char: start,
            end_char: start + cue.chars().count(),
        },
    ))
}

fn clause_transition(
    source: &Text<'_>,
    previous: LocatedSpan,
    current: LocatedSpan,
) -> Option<(bool, Span)> {
    if current
        .span
        .start_char
        .saturating_sub(previous.span.end_char)
        > 12
    {
        return None;
    }
    let previous_text = source.slice(previous.span.start_char, previous.span.end_char);
    let connector = previous_text
        .trim_end_matches(|ch: char| matches!(ch, '，' | ',' | '、') || ch.is_whitespace());
    if connector.is_empty()
        || previous_text.ends_with(['。', '！', '？', '”', '」', '』'])
        || previous_text.contains(QUOTE_OPENERS)
    {
        return None;
    }
    let gap = source.slice(previous.span.end_char, current.span.start_char);
    if gap.contains(['。', '！', '？', '”', '」', '』']) {
        return None;
    }
    for (words, backward) in [
        (&["之後", "之后", "過後", "过后"][..], false),
        (&["之前", "以前"][..], true),
    ] {
        for word in words {
            if connector.ends_with(*word) {
                let end_char = previous.span.start_char + connector.chars().count();
                return Some((
                    backward,
                    Span {
                        start_char: end_char - word.chars().count(),
                        end_char,
                    },
                ));
            }
        }
    }
    None
}

fn collect<'a>(
    documents: &[TimelineDocument<'a>],
    include_indefinite_delays: bool,
    source_mentions: bool,
) -> Result<(Vec<TimelineEvent<'a>>, Precedence)> {
    let mut entries: Vec<TimelineEvent<'a>> = Vec::new();
    let mut primary_contexts: Vec<&'a str> = Vec::new();
    let mut primary_cue_spans: Vec<Option<Span>> = Vec::new();
    let mut edges = Precedence::new();
    let mut last: Option<usize> = None;
    let mut recent_outcomes = BTreeMap::new();
    for &document in documents {
        let source = Text::new(document);
        let chapter = document.order;
        let mut events: Vec<_> = document.events.iter().enumerate().collect();
        events.sort_by_key(|(index, event)| {
            (event.evidence.first().map(|span| span.start_char), *index)
        });
        for (index, event) in events {
            let mut evidence = event
                .evidence
                .iter()
                .map(|span| source.evidence(*span))
                .collect::<Result<Vec<_>>>()?;
            let anchor = *evidence.first().context("事件缺少原文範圍")?;
            // A recap may refer to the same nearby realm transition. Keep its own original span.
            let outcome = match (&event.actor, &event.object) {
                (Some(actor), Some(object))
                    if event.kind == "realm_change" && event.status == "occurred" =>
                {
                    Some((actor.as_str(), object.as_str()))
                }
                _ => None,
            };
            if !source_mentions && let Some(key) = outcome {
                if let Some(&prior) = recent_outcomes.get(&key) {
                    let prior_entry: &mut TimelineEvent<'_> = &mut entries[prior];
                    let distinct = recap_time::distinct_cues(
                        primary_contexts[prior],
                        recap_time::local_context(document.text, &source.bytes, anchor.span),
                        key.0,
                        chapter != prior_entry.chapter,
                    );
                    let unresolved = recap_time::preserve_unresolved(
                        primary_contexts[prior],
                        recap_time::local_context(document.text, &source.bytes, anchor.span),
                        key.0,
                        chapter != prior_entry.chapter,
                    );
                    let same_transition = event.recap
                        && chapter.saturating_sub(prior_entry.chapter) <= 1
                        && distinct.is_none()
                        && !unresolved
                        && ![
                            "另一次",
                            "上一次",
                            "第一次",
                            "前一次",
                            "亦曾",
                            "也曾",
                            "又一次",
                            "再一次",
                        ]
                        .iter()
                        .any(|phrase| {
                            source
                                .slice(
                                    anchor.span.start_char.saturating_sub(12),
                                    anchor.span.end_char,
                                )
                                .contains(phrase)
                        });
                    if same_transition {
                        for span in evidence {
                            if !prior_entry.evidence.contains(&span) {
                                prior_entry.evidence.push(span);
                            }
                        }
                        continue;
                    }
                    if let Some((_, next_cue)) = distinct {
                        if let Some(span) = primary_cue_spans[prior] {
                            let cue = LocatedSpan {
                                chapter: prior_entry.chapter,
                                span,
                            };
                            if !prior_entry.evidence.contains(&cue) {
                                prior_entry.evidence.push(cue);
                            }
                        }
                        if let Some(span) = recap_time::cue_span(
                            document.text,
                            &source.bytes,
                            anchor.span,
                            next_cue,
                        ) {
                            let cue = source.evidence(span)?;
                            if !evidence.contains(&cue) {
                                evidence.push(cue);
                            }
                        }
                        if chapter != prior_entry.chapter
                            && let Some(span) = recap_time::next_day_cue_span(
                                document.text,
                                &source.bytes,
                                anchor.span,
                            )
                        {
                            let cue = source.evidence(span)?;
                            if !evidence.contains(&cue) {
                                evidence.push(cue);
                            }
                        }
                    }
                    if unresolved {
                        if let Some(span) = primary_cue_spans[prior] {
                            let cue = LocatedSpan {
                                chapter: prior_entry.chapter,
                                span,
                            };
                            if !prior_entry.evidence.contains(&cue) {
                                prior_entry.evidence.push(cue);
                            }
                        }
                        if let Some(span) =
                            recap_time::primary_cue_span(document.text, &source.bytes, anchor.span)
                        {
                            let cue = source.evidence(span)?;
                            if !evidence.contains(&cue) {
                                evidence.push(cue);
                            }
                        }
                        if chapter != prior_entry.chapter
                            && let Some(span) = recap_time::next_day_cue_span(
                                document.text,
                                &source.bytes,
                                anchor.span,
                            )
                        {
                            let cue = source.evidence(span)?;
                            if !evidence.contains(&cue) {
                                evidence.push(cue);
                            }
                        }
                    }
                }
                recent_outcomes.insert(key, entries.len());
            }
            let current = entries.len();
            primary_contexts.push(recap_time::local_context(
                document.text,
                &source.bytes,
                anchor.span,
            ));
            primary_cue_spans.push(recap_time::primary_cue_span(
                document.text,
                &source.bytes,
                anchor.span,
            ));
            if occurred(event) && !event.recap {
                if let Some(previous) = last {
                    let other = &entries[previous];
                    let old = other.evidence[0];
                    let same_chapter = old.chapter == chapter;
                    let near = if same_chapter {
                        anchor.span.start_char >= old.span.end_char
                            && anchor.span.start_char - old.span.end_char <= 320
                    } else {
                        old.chapter.checked_add(1) == Some(chapter) && anchor.span.start_char <= 160
                    };
                    let same_actor = event.actor.is_some() && event.actor == other.event.actor;
                    let floor = if same_chapter { old.span.end_char } else { 0 };
                    if near && (same_chapter || same_actor) {
                        let transition = same_chapter
                            .then(|| clause_transition(&source, old, anchor))
                            .flatten();
                        let cue = transition
                            .or_else(|| {
                                if !include_indefinite_delays
                                    || !same_chapter
                                    || !same_actor
                                    || previous + 1 != current
                                {
                                    return None;
                                }
                                indefinite_delay(
                                    &source,
                                    old.span,
                                    anchor.span,
                                    event.actor.as_deref()?,
                                )
                            })
                            .or_else(|| {
                                signal(&source, anchor, floor)
                                    .filter(|(backward, _)| !backward || same_actor)
                            });
                        if let Some((backward, cue)) = cue {
                            let pair = if backward {
                                (current, previous)
                            } else {
                                (previous, current)
                            };
                            edges.insert(pair, source.evidence(cue)?);
                        }
                    }
                }
                last = Some(current);
            } else if !occurred(event) {
                // A non-asserted intervening event may be the written cue's
                // intended anchor. Do not silently skip it and attach a later
                // cue to an earlier asserted event. Keep the entry and its
                // source evidence; the next asserted event starts a new chain.
                last = None;
            }
            entries.push(TimelineEvent {
                chapter,
                index,
                event,
                evidence,
            });
        }
    }
    Ok((entries, edges))
}

fn topological(size: usize, edges: impl Iterator<Item = (usize, usize)>) -> Vec<usize> {
    let mut outgoing = vec![Vec::new(); size];
    let mut incoming = vec![0; size];
    for (before, after) in edges {
        outgoing[before].push(after);
        incoming[after] += 1;
    }
    let mut ready: BTreeSet<_> = (0..size).filter(|&index| incoming[index] == 0).collect();
    let mut sorted = Vec::with_capacity(size);
    while let Some(before) = ready.pop_first() {
        sorted.push(before);
        for &after in &outgoing[before] {
            incoming[after] -= 1;
            if incoming[after] == 0 {
                ready.insert(after);
            }
        }
    }
    sorted
}

struct Story {
    sequence: Vec<usize>,
    positions: Vec<Option<usize>>,
    conflicts: BTreeSet<(usize, usize)>,
}

fn story_order(size: usize, edges: &Precedence) -> Story {
    let mut reached = vec![false; size];
    for index in topological(size, edges.keys().copied()) {
        reached[index] = true;
    }
    let conflicts: BTreeSet<_> = edges
        .keys()
        .copied()
        .filter(|(a, b)| !reached[*a] && !reached[*b])
        .collect();
    let supported = || {
        edges
            .keys()
            .copied()
            .filter(|edge| !conflicts.contains(edge))
    };
    let mut linked = vec![false; size];
    for (before, after) in supported() {
        linked[before] = true;
        linked[after] = true;
    }
    let mut sorted = topological(size, supported());
    sorted.sort_by_key(|&index| !linked[index]);
    let mut positions = vec![None; size];
    for (position, &index) in sorted.iter().enumerate() {
        if linked[index] {
            positions[index] = Some(position + 1);
        }
    }
    Story {
        sequence: sorted,
        positions,
        conflicts,
    }
}

/// Build the source and story views. Chapter ordinals must be unique.
pub fn build<'a>(documents: &[TimelineDocument<'a>]) -> Result<Timeline<'a>> {
    build_timeline(documents, false, false)
}

/// Build the source and story views with support for indefinite delay cues.
pub fn build_with_indefinite_delays<'a>(
    documents: &[TimelineDocument<'a>],
) -> Result<Timeline<'a>> {
    build_timeline(documents, true, false)
}

/// Keep every extracted source mention as a separate entry. Distinct entries
/// do not assert that their underlying real or fictional events are different.
/// This only preserves mentions present in the provided analyses.
pub fn build_source_mentions<'a>(documents: &[TimelineDocument<'a>]) -> Result<Timeline<'a>> {
    build_timeline(documents, false, true)
}

/// Keep every extracted source mention, and retain indefinite-delay hints.
pub fn build_source_mentions_with_indefinite_delays<'a>(
    documents: &[TimelineDocument<'a>],
) -> Result<Timeline<'a>> {
    build_timeline(documents, true, true)
}

fn build_timeline<'a>(
    documents: &[TimelineDocument<'a>],
    include_indefinite_delays: bool,
    source_mentions: bool,
) -> Result<Timeline<'a>> {
    let mut documents = documents.to_vec();
    documents.sort_by_key(|document| document.order);
    ensure!(
        documents
            .windows(2)
            .all(|pair| pair[0].order != pair[1].order),
        "章節順序不可重複"
    );
    let (events, edges) = collect(&documents, include_indefinite_delays, source_mentions)?;
    let Story {
        sequence,
        positions,
        conflicts,
    } = story_order(events.len(), &edges);
    let relations = edges
        .into_iter()
        .map(|((before, after), cue)| Relation {
            before,
            after,
            evidence: [events[before].evidence[0], cue, events[after].evidence[0]],
            supported: !conflicts.contains(&(before, after)),
        })
        .collect();

    Ok(Timeline {
        events,
        relations,
        story_sequence: sequence,
        story_positions: positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(source: &str, pieces: &[(&str, bool)]) -> Vec<Event> {
        pieces
            .iter()
            .map(|(phrase, recap)| {
                let byte = source.find(phrase).unwrap();
                let start_char = source[..byte].chars().count();
                Event {
                    kind: "realm_change".into(),
                    status: "occurred".into(),
                    actor: Some("甲".into()),
                    object: Some(
                        if phrase.contains("築基") {
                            "築基"
                        } else {
                            "結丹"
                        }
                        .into(),
                    ),
                    action: (*phrase).into(),
                    evidence: vec![Span {
                        start_char,
                        end_char: start_char + phrase.chars().count(),
                    }],
                    recap: *recap,
                }
            })
            .collect()
    }

    fn slice(text: &str, span: Span) -> String {
        text.chars()
            .skip(span.start_char)
            .take(span.end_char - span.start_char)
            .collect()
    }

    #[test]
    fn recall_can_sort_before_its_source_neighbor() {
        let text = "𠀀甲結丹成功。三年前，甲築基成功。";
        let events = sample(text, &[("甲結丹成功", false), ("甲築基成功", false)]);
        let timeline = build(&[TimelineDocument {
            order: 1,
            text,
            events: &events,
        }])
        .unwrap();

        assert_eq!(timeline.events.len(), 2);
        assert_eq!(timeline.relations.len(), 1);
        let edge = timeline.relations[0];
        assert_eq!((edge.before, edge.after, edge.supported), (1, 0, true));
        assert_eq!(slice(text, edge.evidence[1].span), "三年前");
        assert_eq!(timeline.story_sequence, [1, 0]);
        assert_eq!(timeline.story_positions, [Some(2), Some(1)]);
    }

    #[test]
    fn a_temporal_clause_can_bind_distinct_actors() {
        let text = "甲結丹之後，乙築基成功。";
        let mut events = sample(text, &[("甲結丹之後", false), ("乙築基成功", false)]);
        events[1].actor = Some("乙".into());
        let timeline = build(&[TimelineDocument {
            order: 1,
            text,
            events: &events,
        }])
        .unwrap();
        let edge = timeline.relations[0];
        assert_eq!((edge.before, edge.after), (0, 1));
        assert_eq!(slice(text, edge.evidence[1].span), "之後");
    }

    #[test]
    fn a_duration_with_classifier_preserves_the_original_marker() {
        let text = "一個多月后，甲出門";
        let (backward, cue) = relative_cue(text, 0).unwrap();
        assert!(!backward);
        assert_eq!(slice(text, cue), "一個多月后");
    }

    #[test]
    fn an_indefinite_delay_binds_the_action_after_the_cue() {
        for (text, cue, previous, current) in [
            (
                "𠮷。向蔓昏迷過去了。\r\n不知過了多久後，向蔓緩緩坐起。",
                "不知過了多久後",
                "昏迷",
                "坐起",
            ),
            (
                "向蔓收起罗盘。过了片刻，向蔓推开木窗。",
                "过了片刻",
                "向蔓收起罗盘。",
                "过了片刻，向蔓推开木窗。",
            ),
        ] {
            let mut events = sample(text, &[(previous, false), (current, false)]);
            for event in &mut events {
                event.actor = Some("向蔓".into());
            }
            let documents = [TimelineDocument {
                order: 1,
                text,
                events: &events,
            }];
            assert!(build(&documents).unwrap().relations.is_empty(), "{text}");
            let timeline = build_with_indefinite_delays(&documents).unwrap();
            let [edge] = timeline.relations.as_slice() else {
                panic!("expected one explicit relation for {text}");
            };
            assert_eq!((edge.before, edge.after, edge.supported), (0, 1, true));
            assert_eq!(slice(text, edge.evidence[0].span), previous);
            assert_eq!(slice(text, edge.evidence[1].span), cue);
            assert_eq!(slice(text, edge.evidence[2].span), current);
        }
    }

    #[test]
    fn an_indefinite_delay_does_not_cross_other_actions_or_assertion_boundaries() {
        for (text, intervening) in [
            ("甲昏迷。門外敲鐘。過了片刻，甲坐起。", None),
            ("甲昏迷。過了片刻，乙敲鐘，甲坐起。", None),
            ("甲昏迷。過了片刻，乙敲鐘，甲坐起。", Some("敲鐘")),
            ("如果甲昏迷。過了片刻，甲坐起。", None),
            ("甲昏迷？過了片刻，甲坐起。", None),
            ("甲昏迷。過了片刻，甲若是坐起。", None),
            ("「甲昏迷。過了片刻，甲坐起。」", None),
            ("「外面有人。\n甲昏迷。過了片刻，甲坐起。」", None),
            ("甲昏迷。有人喊：「過了片刻，甲坐起。」", None),
            ("甲昏迷。甲坐起。", None),
        ] {
            let mut pieces = vec![("昏迷", false)];
            pieces.extend(intervening.map(|action| (action, false)));
            pieces.push(("坐起", false));
            let mut events = sample(text, &pieces);
            if intervening.is_some() {
                events[1].actor = Some("乙".into());
            }
            let timeline = build_with_indefinite_delays(&[TimelineDocument {
                order: 1,
                text,
                events: &events,
            }])
            .unwrap();
            assert!(timeline.relations.is_empty(), "{text}");
        }
    }

    #[test]
    fn text_on_a_quoted_object_does_not_set_story_time() {
        let text = "甲結丹成功。甲拿出一封寫著「三年前」的信。";
        let events = sample(
            text,
            &[
                ("甲結丹成功", false),
                ("甲拿出一封寫著「三年前」的信", false),
            ],
        );
        let timeline = build(&[TimelineDocument {
            order: 1,
            text,
            events: &events,
        }])
        .unwrap();
        assert!(timeline.relations.is_empty());
    }

    #[test]
    fn an_explicit_separate_attempt_keeps_its_own_event() {
        let text = "甲築基成功。先前甲亦曾築基成功，後來修為全失。";
        let events = sample(text, &[("甲築基成功", false), ("先前甲亦曾築基成功", true)]);
        let timeline = build(&[TimelineDocument {
            order: 1,
            text,
            events: &events,
        }])
        .unwrap();
        assert_eq!(timeline.events.len(), 2);
    }

    #[test]
    fn a_nearby_recap_keeps_both_evidence_spans_with_unordered_input() {
        let first = "甲築基成功。";
        let first_events = sample(first, &[("甲築基成功", false)]);
        let second = "甲回想起當日。甲築基成功。";
        let second_events = sample(second, &[("甲築基成功", true)]);
        let timeline = build(&[
            TimelineDocument {
                order: 216,
                text: second,
                events: &second_events,
            },
            TimelineDocument {
                order: 215,
                text: first,
                events: &first_events,
            },
        ])
        .unwrap();

        assert_eq!(timeline.events.len(), 1);
        let event = &timeline.events[0];
        assert_eq!((event.chapter, event.index), (215, 0));
        assert_eq!(
            event
                .evidence
                .iter()
                .map(|span| span.chapter)
                .collect::<Vec<_>>(),
            [215, 216]
        );
        assert_eq!(timeline.story_positions, [None]);
    }

    #[test]
    fn rejects_duplicate_chapters_and_out_of_bounds_evidence() {
        let text = "甲築基成功。";
        let mut events = sample(text, &[("甲築基成功", false)]);
        let document = TimelineDocument {
            order: 1,
            text,
            events: &events,
        };
        assert!(build(&[document, document]).is_err());
        events[0].evidence[0].end_char = 100;
        assert!(
            build(&[TimelineDocument {
                order: 1,
                text,
                events: &events
            }])
            .is_err()
        );
    }
}
