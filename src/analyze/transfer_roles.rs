//! Conservative written-role candidates for two local transfer constructions.

use super::{Span, WrittenTransferCue, WrittenTransferDirection, source_quote_damage_start};

/// An explicitly unknown source, role, event, or association status.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WrittenTransferUnknown {
    Unknown,
}

/// Written fragments are proposals only; none establishes a person or event.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WrittenTransferRoleCandidate {
    pub cue: Span,
    pub construction: Span,
    pub direction: WrittenTransferDirection,
    pub giver: Option<Span>,
    pub item: Option<Span>,
    pub recipient: Option<Span>,
    pub identity_status: WrittenTransferUnknown,
    pub role_status: WrittenTransferUnknown,
    pub occurrence_status: WrittenTransferUnknown,
    pub association_status: WrittenTransferUnknown,
}

/// Alternative written boundaries only. The spans do not establish a person,
/// role, event, relation, or even that the suffix is an independent word.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WrittenTransferRoleAlternatives {
    pub cue: Span,
    pub construction: Span,
    pub giver_alternatives: Vec<Span>,
    pub recipient_alternatives: Vec<Span>,
    pub boundary_status: WrittenTransferUnknown,
    pub interpretation_status: WrittenTransferUnknown,
    pub identity_status: WrittenTransferUnknown,
    pub role_status: WrittenTransferUnknown,
    pub occurrence_status: WrittenTransferUnknown,
    pub association_status: WrittenTransferUnknown,
}

const AMBIGUOUS_SUBJECT_SUFFIXES: &[&str] = &[
    "沒有", "没有", "未曾", "不曾", "打算", "準備", "准备", "可能", "沒", "没", "未", "不", "想",
];

fn boundary(ch: char) -> bool {
    matches!(
        ch,
        '。' | '！'
            | '？'
            | '!'
            | '?'
            | '，'
            | ','
            | '；'
            | ';'
            | '\n'
            | '\r'
            | '「'
            | '」'
            | '『'
            | '』'
            | '“'
            | '”'
            | '‘'
            | '’'
            | '"'
            | '：'
            | ':'
    )
}

// Stop at a nearby written boundary. A long unbounded context stays unknown.
fn clause_bounds(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let mut left = 0;
    for (seen, (at, ch)) in text[..start].char_indices().rev().enumerate() {
        if boundary(ch) {
            left = at + ch.len_utf8();
            break;
        }
        if seen >= 255 {
            return None;
        }
    }
    let mut right = text.len();
    for (seen, (at, ch)) in text[end..].char_indices().enumerate() {
        if boundary(ch) {
            right = end + at;
            break;
        }
        if seen >= 255 {
            return None;
        }
    }
    Some((left, right))
}

fn trimmed(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let raw = text.get(start..end)?;
    let left = raw.len() - raw.trim_start().len();
    let right = raw.trim_end().len();
    (left < right).then_some((start + left, start + right))
}

fn subject(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let (start, end) = trimmed(text, start, end)?;
    // A suffix such as 想 or 未 may also be part of a name. Without an
    // independent boundary, neither the shorter nor the longer role is safe.
    if AMBIGUOUS_SUBJECT_SUFFIXES
        .iter()
        .any(|suffix| text[start..end].ends_with(*suffix))
    {
        return None;
    }
    phrase(text, start, end, 8, true)
}

// Keep both possible written boundaries. The longer range can include a
// modifier; neither range is asserted to be a person or the correct role.
fn ambiguous_subject_ranges(text: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
    let Some((start, end)) = trimmed(text, start, end) else {
        return Vec::new();
    };
    let word = &text[start..end];
    if word.chars().count() > 8
        || !word.chars().all(|ch| ch.is_alphanumeric() || ch == '·')
        || ["與", "与", "和", "及", "跟", "被"]
            .iter()
            .any(|term| word.contains(term))
    {
        return Vec::new();
    }
    let mut result = vec![(start, end)];
    for suffix in AMBIGUOUS_SUBJECT_SUFFIXES {
        let Some(lead) = word.strip_suffix(*suffix) else {
            continue;
        };
        let cut = start + lead.len();
        // Only a plausible short fragment is proposed; this filter is a
        // heuristic, not proof of its interpretation or a complete list.
        if phrase(text, start, cut, 8, true).is_some() && cut > start {
            result.push((start, cut));
        }
    }
    result.sort_unstable();
    result.dedup();
    if result.len() < 2 { Vec::new() } else { result }
}

fn phrase(
    text: &str,
    start: usize,
    end: usize,
    limit: usize,
    person: bool,
) -> Option<(usize, usize)> {
    let (start, end) = trimmed(text, start, end)?;
    let word = &text[start..end];
    if word.chars().count() > limit || !word.chars().all(|ch| ch.is_alphanumeric() || ch == '·') {
        return None;
    }
    if person
        && ["與", "与", "和", "及", "跟", "被"]
            .iter()
            .any(|term| word.contains(term))
    {
        return None;
    }
    // Segmentation is only a warning signal. It cannot prove a role or a
    // boundary, and unfamiliar names may still be omitted or mis-segmented.
    if word.chars().any(|ch| matches!(ch, '後' | '后' | '又')) || word.ends_with('了') {
        return None;
    }
    if crate::text::default_jieba()
        .tag(word, true)
        .iter()
        .any(|tag| tag.tag.starts_with('v') || matches!(tag.tag, "d" | "c" | "p" | "u" | "i"))
    {
        return None;
    }
    Some((start, end))
}

fn scalar_span(
    text: &str,
    cue_byte: usize,
    cue_scalar: usize,
    range: (usize, usize),
) -> Option<Span> {
    fn scalar(text: &str, cue_byte: usize, cue_scalar: usize, byte: usize) -> Option<usize> {
        if byte <= cue_byte {
            cue_scalar.checked_sub(text.get(byte..cue_byte)?.chars().count())
        } else {
            Some(cue_scalar + text.get(cue_byte..byte)?.chars().count())
        }
    }
    let start_char = scalar(text, cue_byte, cue_scalar, range.0)?;
    let end_char = scalar(text, cue_byte, cue_scalar, range.1)?;
    (start_char < end_char).then_some(Span {
        start_char,
        end_char,
    })
}

pub(super) fn discover(
    text: &str,
    cues: &[WrittenTransferCue],
) -> Vec<WrittenTransferRoleCandidate> {
    let damaged = source_quote_damage_start(text);
    let mut byte_cursor = 0;
    let mut scalar_cursor = 0;
    let mut result = Vec::with_capacity(cues.len());
    for cue in cues {
        let Some(relative) = cue.cue.start_char.checked_sub(scalar_cursor) else {
            continue;
        };
        let Some((offset, _)) = text[byte_cursor..].char_indices().nth(relative) else {
            continue;
        };
        let cue_start = byte_cursor + offset;
        let width = cue.cue.end_char - cue.cue.start_char;
        let Some((end_offset, _)) = text[cue_start..].char_indices().nth(width).or_else(|| {
            (text[cue_start..].chars().count() == width).then_some((text.len() - cue_start, '\0'))
        }) else {
            continue;
        };
        let cue_end = cue_start + end_offset;
        byte_cursor = cue_end;
        scalar_cursor = cue.cue.end_char;
        let mut candidate = WrittenTransferRoleCandidate {
            cue: cue.cue,
            construction: cue.cue,
            direction: cue.direction,
            giver: None,
            item: None,
            recipient: None,
            identity_status: WrittenTransferUnknown::Unknown,
            role_status: WrittenTransferUnknown::Unknown,
            occurrence_status: WrittenTransferUnknown::Unknown,
            association_status: WrittenTransferUnknown::Unknown,
        };
        if damaged.is_some_and(|at| cue.cue.start_char >= at) {
            result.push(candidate);
            continue;
        }
        let Some((left, right)) = clause_bounds(text, cue_start, cue_end) else {
            result.push(candidate);
            continue;
        };
        let Some(construction) = trimmed(text, left, right)
            .and_then(|range| scalar_span(text, cue_start, cue.cue.start_char, range))
        else {
            result.push(candidate);
            continue;
        };
        candidate.construction = construction;
        let prefix = &text[left..cue_start];
        let written_cue = &text[cue_start..cue_end];
        let ranges = if cue.direction == WrittenTransferDirection::Give
            && matches!(written_cue, "交給" | "交给")
        {
            let marker = prefix
                .char_indices()
                .filter(|(_, ch)| matches!(ch, '把' | '將' | '将'))
                .collect::<Vec<_>>();
            if marker.len() != 1 {
                None
            } else {
                let (at, ch) = marker[0];
                let lead = &prefix[..at];
                let token_marker = crate::text::default_jieba()
                    .tag(&text[left..right], true)
                    .iter()
                    .any(|tag| tag.byte_start == at && tag.byte_end == at + ch.len_utf8());
                if !token_marker || lead.ends_with(['一', '二', '三', '四', '五', '那', '這', '这'])
                {
                    None
                } else {
                    Some((
                        subject(text, left, left + at),
                        phrase(text, left + at + ch.len_utf8(), cue_start, 12, false),
                        phrase(text, cue_end, right, 8, true),
                    ))
                }
            }
        } else if cue.direction == WrittenTransferDirection::Receive
            && matches!(written_cue, "接過" | "接过")
            && prefix.ends_with("手中")
        {
            prefix
                .char_indices()
                .rev()
                .find(|(_, ch)| matches!(ch, '從' | '从'))
                .map(|(at, ch)| {
                    (
                        phrase(
                            text,
                            left + at + ch.len_utf8(),
                            cue_start - "手中".len(),
                            8,
                            true,
                        ),
                        phrase(text, cue_end, right, 12, false),
                        subject(text, left, left + at),
                    )
                })
        } else {
            None
        };
        if let Some((giver, item, recipient)) = ranges {
            candidate.giver =
                giver.and_then(|range| scalar_span(text, cue_start, cue.cue.start_char, range));
            candidate.item =
                item.and_then(|range| scalar_span(text, cue_start, cue.cue.start_char, range));
            candidate.recipient =
                recipient.and_then(|range| scalar_span(text, cue_start, cue.cue.start_char, range));
        }
        result.push(candidate);
    }
    result
}

/// Preserve ambiguous local role boundaries without changing the legacy
/// single-role proposal. Every input cue has an output, even if unexamined.
pub(super) fn discover_alternatives(
    text: &str,
    cues: &[WrittenTransferCue],
) -> Vec<WrittenTransferRoleAlternatives> {
    let roles = discover(text, cues);
    let damaged = source_quote_damage_start(text);
    let mut byte_cursor = 0;
    let mut scalar_cursor = 0;
    let mut result = Vec::with_capacity(roles.len());
    for role in roles {
        let mut observation = WrittenTransferRoleAlternatives {
            cue: role.cue,
            construction: role.construction,
            giver_alternatives: Vec::new(),
            recipient_alternatives: Vec::new(),
            boundary_status: WrittenTransferUnknown::Unknown,
            interpretation_status: WrittenTransferUnknown::Unknown,
            identity_status: WrittenTransferUnknown::Unknown,
            role_status: WrittenTransferUnknown::Unknown,
            occurrence_status: WrittenTransferUnknown::Unknown,
            association_status: WrittenTransferUnknown::Unknown,
        };
        let positions = role
            .cue
            .start_char
            .checked_sub(scalar_cursor)
            .and_then(|relative| {
                let (offset, _) = text.get(byte_cursor..)?.char_indices().nth(relative)?;
                let start = byte_cursor + offset;
                let width = role.cue.end_char.checked_sub(role.cue.start_char)?;
                let tail = text.get(start..)?;
                let end = match tail.char_indices().nth(width) {
                    Some((offset, _)) => start + offset,
                    None if tail.chars().count() == width => text.len(),
                    None => return None,
                };
                Some((start, end))
            });
        let Some((cue_start, cue_end)) = positions else {
            result.push(observation);
            continue;
        };
        byte_cursor = cue_end;
        scalar_cursor = role.cue.end_char;
        if damaged.is_some_and(|at| role.cue.start_char >= at) {
            result.push(observation);
            continue;
        }
        let Some((left, right)) = clause_bounds(text, cue_start, cue_end) else {
            result.push(observation);
            continue;
        };
        let Some(construction) = trimmed(text, left, right)
            .and_then(|range| scalar_span(text, cue_start, role.cue.start_char, range))
        else {
            result.push(observation);
            continue;
        };
        if construction != role.construction {
            result.push(observation);
            continue;
        }
        let prefix = &text[left..cue_start];
        let cue = &text[cue_start..cue_end];
        let mut ranges = Vec::new();
        let mut giving = false;
        if role.direction == WrittenTransferDirection::Give
            && role.giver.is_none()
            && matches!(cue, "交給" | "交给")
        {
            let markers: Vec<_> = prefix
                .char_indices()
                .filter(|(_, ch)| matches!(ch, '把' | '將' | '将'))
                .collect();
            if let [(at, ch)] = markers.as_slice() {
                let lead = &prefix[..*at];
                let marker_is_token = crate::text::default_jieba()
                    .tag(&text[left..right], true)
                    .iter()
                    .any(|tag| tag.byte_start == *at && tag.byte_end == *at + ch.len_utf8());
                let item = phrase(text, left + *at + ch.len_utf8(), cue_start, 12, false);
                if marker_is_token
                    && item.is_some()
                    && !lead.ends_with(['一', '二', '三', '四', '五', '那', '這', '这'])
                {
                    ranges = ambiguous_subject_ranges(text, left, left + *at);
                    giving = true;
                }
            }
        } else if role.direction == WrittenTransferDirection::Receive
            && role.recipient.is_none()
            && matches!(cue, "接過" | "接过")
            && prefix.ends_with("手中")
            && let Some((at, ch)) = prefix
                .char_indices()
                .rev()
                .find(|(_, ch)| matches!(ch, '從' | '从'))
        {
            let giver = phrase(
                text,
                left + at + ch.len_utf8(),
                cue_start - "手中".len(),
                8,
                true,
            );
            let item = phrase(text, cue_end, right, 12, false);
            if giver.is_some() && item.is_some() {
                ranges = ambiguous_subject_ranges(text, left, left + at);
            }
        }
        let spans: Vec<_> = ranges
            .into_iter()
            .filter_map(|range| scalar_span(text, cue_start, role.cue.start_char, range))
            .filter(|span| {
                construction.start_char <= span.start_char && span.end_char <= construction.end_char
            })
            .collect();
        if spans.len() >= 2 {
            if giving {
                observation.giver_alternatives = spans;
            } else {
                observation.recipient_alternatives = spans;
            }
        }
        result.push(observation);
    }
    result
}
