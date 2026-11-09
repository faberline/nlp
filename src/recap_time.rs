//! Conservative, source-backed comparison of explicit relative dates.
//! No match means unknown, not that two mentions describe the same event.
use crate::Span;

pub(super) fn local_context<'a>(text: &'a str, bytes: &[usize], span: Span) -> &'a str {
    let start = bytes[span.start_char];
    let end = bytes[span.end_char];
    let beginning = text[..start]
        .char_indices()
        .rev()
        .find_map(|(offset, ch)| {
            matches!(ch, '。' | '！' | '？' | '.' | '!' | '?' | '\n' | '\r')
                .then_some(offset + ch.len_utf8())
        })
        .unwrap_or(0);
    &text[beginning..end]
}

fn chinese_digit(ch: char) -> Option<u32> {
    match ch {
        '一' => Some(1),
        '二' | '兩' | '两' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    }
}

fn positive_number(number: &str) -> Option<u32> {
    if number.chars().all(|ch| ch.is_ascii_digit()) {
        return number.parse::<u32>().ok().filter(|n| *n > 0);
    }
    let mut total = 0u32;
    let mut pending = None;
    let mut last_unit = 10_000u32;
    let mut seen_zero = false;
    let mut pending_after_zero = false;
    for ch in number.chars() {
        if let Some(digit) = chinese_digit(ch) {
            if pending.replace(digit).is_some() {
                return None;
            }
            pending_after_zero = seen_zero;
            seen_zero = false;
        } else if matches!(ch, '零' | '〇') {
            if pending.is_some() || seen_zero || total == 0 {
                return None;
            }
            seen_zero = true;
        } else {
            let unit = match ch {
                '十' => 10,
                '百' => 100,
                '千' => 1000,
                _ => return None,
            };
            if unit >= last_unit || seen_zero {
                return None;
            }
            let digit = match pending.take() {
                Some(n) => n,
                None if unit == 10 && total == 0 => 1,
                _ => return None,
            };
            total = total.checked_add(digit * unit)?;
            last_unit = unit;
        }
    }
    if seen_zero {
        return None;
    }
    // A trailing digit after hundreds or thousands without 零 can be a
    // colloquial abbreviation: 一百二 may mean 102 or 120. Do not guess.
    if pending.is_some() && last_unit > 10 && last_unit < 10_000 && !pending_after_zero {
        return None;
    }
    total.checked_add(pending.unwrap_or(0)).filter(|n| *n > 0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelativeTime {
    Day(u32),
    Years(u32),
}

// Return the relative time and exact written cue, not a normalized date.
fn beginning_cue(context: &str) -> Option<(RelativeTime, &str)> {
    for (cue, day) in [("今日", 0), ("今天", 0), ("昨日", 1), ("昨天", 1)] {
        if let Some(rest) = context.strip_prefix(cue)
            && rest.starts_with(['，', ','])
        {
            return Some((RelativeTime::Day(day), &context[..cue.len()]));
        }
    }
    let end = context
        .char_indices()
        .take_while(|(_, ch)| {
            ch.is_ascii_digit() || "零〇一二三四五六七八九十百千兩两".contains(*ch)
        })
        .map(|(offset, ch)| offset + ch.len_utf8())
        .last()?;
    let years = positive_number(&context[..end])?;
    let rest = context[end..].strip_prefix("年前")?;
    if !rest.starts_with(['，', ','])
        && !["曾經", "曾经", "曾"]
            .iter()
            .any(|word| rest.starts_with(*word))
    {
        return None;
    }
    Some((RelativeTime::Years(years), &context[..end + "年前".len()]))
}

fn marker<'a>(
    context: &'a str,
    actor: &str,
    next_chapter: bool,
) -> Option<(RelativeTime, &'a str)> {
    let context = context.trim_start();
    if context
        .chars()
        .any(|ch| "「」『』“”‘’\"《》〈〉".contains(ch))
    {
        return None;
    }
    if !next_chapter {
        // A leading date before 回想 times the recollection, not necessarily
        // the event described inside it. Do not compare it as the event date.
        if has_timed_recall(context) {
            return None;
        }
        return beginning_cue(context);
    }
    let tail = ["次日，", "次日,", "翌日，", "翌日,", "隔天，", "隔天,"]
        .iter()
        .find_map(|cue| context.strip_prefix(*cue))?;
    let tail = tail.strip_prefix(actor)?;
    let tail = ["回想", "回憶", "回忆", "想起"]
        .iter()
        .find_map(|cue| tail.strip_prefix(*cue))?;
    for (cue, day) in [("今日", 0), ("今天", 0), ("昨日", 1), ("昨天", 1)] {
        if let Some(rest) = tail.strip_prefix(cue)
            && ![
                "之前", "以前", "之後", "之后", "以後", "以后", "前後", "前后",
            ]
            .iter()
            .any(|qualifier| rest.starts_with(*qualifier))
        {
            return Some((RelativeTime::Day(day), &tail[..cue.len()]));
        }
    }
    beginning_cue(tail)
}

fn leading_time(context: &str) -> Option<(&str, &str)> {
    let context = context.trim_start();
    for cue in ["今日", "今天", "昨日", "昨天"] {
        if let Some(rest) = context.strip_prefix(cue)
            && let Some(rest) = rest.strip_prefix(['，', ','])
        {
            return Some((&context[..cue.len()], rest));
        }
    }
    let end = context
        .char_indices()
        .take_while(|(_, ch)| {
            ch.is_ascii_digit() || "零〇一二三四五六七八九十百千兩两".contains(*ch)
        })
        .map(|(offset, ch)| offset + ch.len_utf8())
        .last()?;
    let rest = context[end..].strip_prefix("年前")?;
    let rest = rest.strip_prefix(['，', ','])?;
    Some((&context[..end + "年前".len()], rest))
}

fn has_timed_recall(context: &str) -> bool {
    let Some((_, body)) = leading_time(context) else {
        return false;
    };
    ["回想", "回憶", "回忆", "想起", "追憶", "追忆"]
        .iter()
        .any(|word| body.contains(*word))
}

/// Keep separate source mentions if a written time is ambiguous or timestamps
/// a recollection. This does not establish whether the occurrences are equal.
pub(super) fn preserve_unresolved(
    prior: &str,
    current: &str,
    actor: &str,
    next_chapter: bool,
) -> bool {
    // Day-relative words in the next chapter have a different reference day.
    // Keep both mentions without assuming whether they describe one occurrence.
    if next_chapter
        && matches!(
            marker(current, actor, true),
            Some((RelativeTime::Day(_), _))
        )
    {
        return true;
    }
    if !next_chapter && (has_timed_recall(prior) || has_timed_recall(current)) {
        return true;
    }
    let Some((old, _)) = leading_time(prior) else {
        return false;
    };
    let Some((new, _)) = leading_time(current) else {
        return false;
    };
    let ambiguous = |cue: &str| {
        cue.strip_suffix("年前")
            .is_some_and(|number| positive_number(number).is_none())
    };
    ambiguous(old) || ambiguous(new)
}

pub(super) fn distinct_cues<'a, 'b>(
    prior: &'a str,
    current: &'b str,
    actor: &str,
    next_chapter: bool,
) -> Option<(&'a str, &'b str)> {
    let (old, old_cue) = marker(prior, actor, false)?;
    let (new, new_cue) = marker(current, actor, next_chapter)?;
    if next_chapter && matches!(new, RelativeTime::Day(_)) {
        return None;
    }
    (old != new).then_some((old_cue, new_cue))
}

pub(super) fn cue_span(text: &str, bytes: &[usize], primary: Span, cue: &str) -> Option<Span> {
    let context = local_context(text, bytes, primary);
    let context_byte = bytes[primary.end_char].checked_sub(context.len())?;
    let cue_start = context_byte + context.find(cue)?;
    Some(Span {
        start_char: bytes.binary_search(&cue_start).ok()?,
        end_char: bytes.binary_search(&(cue_start + cue.len())).ok()?,
    })
}

pub(super) fn primary_cue_span(text: &str, bytes: &[usize], primary: Span) -> Option<Span> {
    let context = local_context(text, bytes, primary);
    if has_timed_recall(context) {
        return None;
    }
    let (cue, _) = leading_time(context)?;
    cue_span(text, bytes, primary, cue)
}

pub(super) fn next_day_cue_span(text: &str, bytes: &[usize], primary: Span) -> Option<Span> {
    let context = local_context(text, bytes, primary).trim_start();
    let cue = ["次日", "翌日", "隔天"].iter().find(|cue| {
        context
            .strip_prefix(**cue)
            .is_some_and(|rest| rest.starts_with(['，', ',']))
    })?;
    cue_span(text, bytes, primary, cue)
}

#[cfg(test)]
mod tests {
    use super::{distinct_cues, preserve_unresolved};
    #[test]
    fn compare_numbers_and_remain_conservative() {
        assert!(
            distinct_cues(
                "三年前，甲築基成功。",
                "十年前，甲曾經築基成功。",
                "甲",
                false
            )
            .is_some()
        );
        assert!(
            distinct_cues(
                "今日，甲築基成功。",
                "次日，甲回想十年前曾經築基成功。",
                "甲",
                true
            )
            .is_some()
        );
        assert!(
            distinct_cues(
                "昨日，甲築基成功。",
                "十年前，甲曾經築基成功。",
                "甲",
                false
            )
            .is_some()
        );
        assert!(
            distinct_cues(
                "一百零二年前，甲築基成功。",
                "一百二十年前，甲曾經築基成功。",
                "甲",
                false
            )
            .is_some()
        );
        let ambiguous = ("一百二年前，甲築基成功。", "一百二十年前，甲曾經築基成功。");
        assert!(distinct_cues(ambiguous.0, ambiguous.1, "甲", false).is_none());
        assert!(preserve_unresolved(ambiguous.0, ambiguous.1, "甲", false));
        let recall = ("十年前，甲築基成功。", "三年前，甲回想七年前曾經築基成功。");
        assert!(distinct_cues(recall.0, recall.1, "甲", false).is_none());
        assert!(preserve_unresolved(recall.0, recall.1, "甲", false));
        assert!(
            distinct_cues(
                "今日，甲築基成功。",
                "次日，甲回想昨日築基成功。",
                "甲",
                true
            )
            .is_none()
        );
        assert!(preserve_unresolved(
            "今日，甲築基成功。",
            "次日，甲回想昨日築基成功。",
            "甲",
            true
        ));
        for (a, b) in [
            ("三年前，甲築基成功。", "三年前，甲曾經築基成功。"),
            ("甲築基成功。", "十年前，甲曾經築基成功。"),
            ("今日，甲築基成功。", "十年前後，甲曾經築基成功。"),
            ("今日，甲築基成功。", "零年前，甲曾經築基成功。"),
            ("今日，甲築基成功。", "不是十年前，甲曾經築基成功。"),
        ] {
            assert!(distinct_cues(a, b, "甲", false).is_none());
        }
    }
}
