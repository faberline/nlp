use super::{
    Actor, HYPOTHETICAL, Lexicon, NarrativeAnalyzer, PLANNED, REPORTED, SUBJECT_TITLES, Source,
    Span, Unit, actual_dream, contains_any, context_for_unit, continuation, earliest, ends_any,
    ends_with_question_particle, participant_description, predicate, quote_end, speech_actor,
    split_units, starts_any, trim_clause_end,
};

/// The kind of act performed in the narration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechActKind {
    Statement,
    Question,
    Request,
    Refusal,
    Warning,
    Decision,
}

/// A performed act and its source evidence.
///
/// Ranges count Unicode scalar values. The payload does not assert that any
/// action discussed in it occurred.
#[derive(Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpeechAct {
    pub kind: SpeechActKind,
    /// The explicit act cue in the narration.
    pub cue: Span,
    /// Written surface selected as a speaker candidate; the role is unverified.
    pub speaker_surface: Option<Span>,
    /// Legacy parser-selected label, often from the caller vocabulary; not a verified identity.
    pub speaker: Option<String>,
    /// The content of the act, without the outer quotation marks.
    pub payload: Span,
}

const SPECIALIZED_CUES: &[(SpeechActKind, &[&str])] = &[
    (
        SpeechActKind::Request,
        &["請求", "请求", "要求", "懇求", "恳求", "吩咐"],
    ),
    (SpeechActKind::Refusal, &["拒絕", "拒绝", "回絕", "回绝"]),
    (SpeechActKind::Warning, &["警告", "告誡", "告诫", "提醒"]),
    (SpeechActKind::Decision, &["決定", "决定"]),
];

fn specialized_cue(text: &str) -> Option<(usize, &'static str, SpeechActKind)> {
    SPECIALIZED_CUES
        .iter()
        .filter_map(|(kind, words)| earliest(text, words).map(|(at, word)| (at, word, *kind)))
        .min_by_key(|(at, word, _)| (*at, std::cmp::Reverse(word.len())))
}

fn assertion_scope(lead: &str, context: &str) -> bool {
    !matches!(context, "dream" | "reported")
        && !actual_dream(lead)
        && !contains_any(lead, HYPOTHETICAL)
        && !contains_any(lead, PLANNED)
        && !contains_any(lead, REPORTED)
        && !contains_any(
            lead,
            &[
                "本想", "原想", "正想", "差點", "差点", "險些", "险些", "沒有", "没有", "未曾",
                "尚未", "不曾", "從未", "从未", "並未", "并未", "聽見", "听见",
            ],
        )
        && !lead.trim_start().starts_with('若')
        && !ends_any(
            lead.trim_end(),
            &["不", "沒", "没", "未", "想", "欲", "要", "將", "将"],
        )
}

fn speaker_from_lead(
    source: &Source<'_>,
    start: usize,
    end: usize,
    lexicon: &Lexicon<'_>,
) -> Option<(Span, Option<String>)> {
    let lead = source.text[start..end]
        .trim_start_matches(|ch: char| ch.is_whitespace() || matches!(ch, ',' | '，'));
    let lead = continuation(lead).unwrap_or(lead).trim_start();
    let base = end - lead.len();
    if let Some((0, word, identity)) = source.first_name(base, end, lexicon)
        && matches!(
            lead[word.len()..].trim(),
            "" | "低聲"
                | "低声"
                | "大聲"
                | "大声"
                | "輕聲"
                | "轻声"
                | "立刻"
                | "已"
                | "已經"
                | "已经"
                | "當即"
                | "当即"
                | "終於"
                | "终于"
        )
    {
        return Some((
            source.span(base, base + word.len()),
            Some(identity.to_owned()),
        ));
    }
    let subject = lead.trim_end();
    (matches!(subject, "我" | "他" | "她") || participant_description(subject))
        .then(|| (source.span(base, base + subject.len()), None))
}

fn narrated_act(
    source: &Source<'_>,
    unit: &Unit,
    context: &str,
    lexicon: &Lexicon<'_>,
) -> Option<SpeechAct> {
    let text = source.slice(&unit.evidence);
    if text.chars().any(|ch| quote_end(ch).is_some())
        || text.contains(['？', '?'])
        || ends_with_question_particle(text)
        || trim_clause_end(text).ends_with('吧')
    {
        return None;
    }
    let (offset, cue, kind) = specialized_cue(text)?;
    let base = source.bytes[unit.evidence.start_char];
    let cue_start = base + offset;
    let cue_end = cue_start + cue.len();
    let earlier = &source.text[source.bytes[unit.sentence.start_char]..cue_start];
    if !assertion_scope(earlier, context) {
        return None;
    }
    let (speaker_surface, speaker) = speaker_from_lead(source, base, cue_start, lexicon)?;
    let tail = text[offset + cue.len()..].trim_start();
    let content = tail.strip_prefix('了').unwrap_or(tail).trim_start();
    if content.starts_with(['的', '是', ':', '：'])
        || !trim_clause_end(content).chars().any(char::is_alphanumeric)
    {
        return None;
    }
    let end = source.bytes[unit.evidence.end_char];
    Some(SpeechAct {
        kind,
        cue: source.span(cue_start, cue_end),
        speaker_surface: Some(speaker_surface),
        speaker,
        payload: source.span(end - content.len(), end),
    })
}

pub(super) fn discover(
    analyzer: &NarrativeAnalyzer<'_>,
    chapter_text: &str,
    title: &str,
) -> Vec<SpeechAct> {
    let mut source = Source::new(chapter_text);
    let units = split_units(&mut source, title, &analyzer.lexicon, Some(analyzer));
    let mut acts = Vec::new();
    let mut context = "main";
    for (index, unit) in units.iter().enumerate() {
        let text = source.slice(&unit.evidence);
        if !text.chars().any(char::is_alphanumeric) {
            continue;
        }
        let lead = specialized_cue(text).map_or(text, |(offset, _, _)| &text[..offset]);
        context = context_for_unit(unit, lead, context);
        let act = if unit.quote {
            let mut conflicting = false;
            quoted_act(
                &source,
                unit,
                units.get(index + 1),
                context,
                &analyzer.lexicon,
                &mut conflicting,
            )
        } else if [
            index.checked_sub(1).and_then(|index| units.get(index)),
            units.get(index + 1),
        ]
        .into_iter()
        .flatten()
        .any(|neighbor| neighbor.quote && neighbor.sentence == unit.sentence)
        {
            None
        } else {
            narrated_act(&source, unit, context, &analyzer.lexicon)
        };
        // Legacy API: preserve the caller-supplied alias label. This is a
        // vocabulary mapping, not independent identity verification. The
        // source-evidence API carries both the spelling and unknown identity.
        acts.extend(act);
    }
    if let Some(start) = super::source_quote_damage_start(chapter_text) {
        acts.retain(|act| {
            act.cue.end_char <= start
                && act.payload.end_char <= start
                && act
                    .speaker_surface
                    .is_none_or(|surface| surface.end_char <= start)
        });
    }
    acts
}

const QUOTED_CUES: &[(&str, SpeechActKind)] = &[
    ("追問", SpeechActKind::Question),
    ("追问", SpeechActKind::Question),
    ("問道", SpeechActKind::Question),
    ("问道", SpeechActKind::Question),
    ("問", SpeechActKind::Question),
    ("问", SpeechActKind::Question),
    ("宣佈", SpeechActKind::Statement),
    ("宣布", SpeechActKind::Statement),
    ("開口道", SpeechActKind::Statement),
    ("开口道", SpeechActKind::Statement),
    ("說道", SpeechActKind::Statement),
    ("说道", SpeechActKind::Statement),
    ("答道", SpeechActKind::Statement),
    ("回道", SpeechActKind::Statement),
    ("笑道", SpeechActKind::Statement),
    ("喊道", SpeechActKind::Statement),
    ("叫道", SpeechActKind::Statement),
    ("喝道", SpeechActKind::Statement),
    ("說", SpeechActKind::Statement),
    ("说", SpeechActKind::Statement),
];

fn quoted_payload_kind(payload: &str) -> SpeechActKind {
    if payload.ends_with(['？', '?'])
        || ends_with_question_particle(payload)
        || starts_any(payload, &["請問", "请问"])
    {
        SpeechActKind::Question
    } else if direct_performative(payload, &["我拒絕", "我拒绝"]) {
        SpeechActKind::Refusal
    } else if direct_performative(payload, &["我決定", "我决定"]) {
        SpeechActKind::Decision
    } else if (payload.starts_with("小心") && !starts_any(payload, &["小心的", "小心地"]))
        || direct_performative(payload, &["我警告", "我提醒"])
    {
        SpeechActKind::Warning
    } else if starts_any(payload, &["拜託", "拜托"])
        || direct_performative(payload, &["我請求", "我请求"])
        || (payload.starts_with(['請', '请'])
            && !starts_any(
                payload,
                &[
                    "請柬", "请柬", "請帖", "请帖", "請求", "请求", "請願", "请愿",
                ],
            ))
    {
        SpeechActKind::Request
    } else {
        SpeechActKind::Statement
    }
}

fn direct_performative(payload: &str, prefixes: &[&str]) -> bool {
    prefixes
        .iter()
        .find_map(|prefix| payload.strip_prefix(prefix))
        .is_some_and(|tail| {
            !tail.starts_with(['了', '過', '过', '的'])
                && trim_clause_end(tail).chars().any(char::is_alphanumeric)
        })
}

fn strip_quoted_manner(text: &str, post_quote: bool) -> &str {
    if let Some(subject) = [
        "輕聲", "轻声", "低聲", "低声", "沉聲", "沉声", "大聲", "大声", "緩緩", "缓缓",
    ]
    .iter()
    .find_map(|modifier| text.strip_suffix(modifier))
    {
        return subject.trim_end();
    }
    let stem = text.strip_suffix(['的', '地']).unwrap_or(text);
    if post_quote
        && let Some(subject) = [
            "眼珠一轉",
            "眼珠一转",
            "神色凝重",
            "凝重",
            "不慌不忙",
            "觀察後",
            "觀察后",
            "观察后",
            "打量後",
            "打量后",
        ]
        .iter()
        .find_map(|modifier| stem.strip_suffix(modifier))
    {
        return subject
            .strip_suffix(['也', '則', '则'])
            .unwrap_or(subject)
            .trim_end();
    }
    let mut ending = stem.char_indices().rev();
    if let (Some((_, last)), Some((start, first))) = (ending.next(), ending.next())
        && start > 0
        && (last == first || last == '然')
    {
        return text[..start].trim_end();
    }
    text
}

fn numbered_utterance<'a>(lead: &'a str, payload: &str, lexicon: &Lexicon<'_>) -> Option<&'a str> {
    let phrase = ["一個", "一个", "這個", "这个"]
        .iter()
        .find_map(|prefix| lead.strip_prefix(prefix))?;
    let (description, tail) = ["數字", "数字"]
        .iter()
        .find_map(|noun| phrase.split_once(noun))?;
    if !description.is_empty() {
        let reaction = description
            .strip_suffix('的')?
            .strip_prefix(['令', '讓', '让', '使'])?;
        let participant = [
            "無法置信",
            "无法置信",
            "難以置信",
            "难以置信",
            "不敢相信",
            "吃驚",
            "吃惊",
            "驚訝",
            "惊讶",
            "震驚",
            "震惊",
        ]
        .iter()
        .find_map(|ending| reaction.strip_suffix(ending))?
        .trim();
        if participant.is_empty()
            || !(matches!(participant, "我" | "你" | "他" | "她")
                || lexicon
                    .actors
                    .iter()
                    .any(|&(alias, _)| alias == participant)
                || participant_description(participant)
                || described_speaker(participant))
        {
            return None;
        }
    }
    // The label must refer to a number written in this quote.
    const NUMERALS: &str = "〇零一二兩两三四五六七八九十百千萬万億亿";
    let payload = trim_clause_end(payload).trim();
    (payload.chars().any(|ch| ch.is_ascii_digit())
        || payload.split(|ch| !NUMERALS.contains(ch)).any(|part| {
            part.starts_with(|ch| "〇零一二兩两三四五六七八九十".contains(ch))
                && (part == payload || part.chars().count() > 1)
        }))
    .then_some(tail)
}

fn voiced_by(
    source: &Source<'_>,
    lead: &str,
    cue_start: usize,
    payload: &str,
    lexicon: &Lexicon<'_>,
) -> Option<(Span, String)> {
    let base = cue_start - lead.len();
    let lead = lead.trim_end();
    let role = ["口中", "口裡", "口里"]
        .iter()
        .find_map(|suffix| lead.strip_suffix(suffix))?;
    let utterance = [
        "這句話",
        "这句话",
        "那句話",
        "那句话",
        "這幾句話",
        "这几句话",
        "這番話",
        "这番话",
        "這些話",
        "这些话",
        "這話",
        "这话",
        "此話",
        "此话",
        "這一聲",
        "这一声",
        "這聲音",
        "这声音",
        "聲音",
        "声音",
        "話語",
        "话语",
        "此言",
        "此語",
        "此语",
        "此聲",
        "此声",
    ]
    .iter()
    .find_map(|noun| role.strip_prefix(noun))
    .or_else(|| numbered_utterance(role, payload, lexicon))?
    .trim_start();
    let utterance = ["卻", "却", "竟然", "竟", "緩緩", "缓缓", "忽然"]
        .iter()
        .find_map(|modifier| utterance.strip_prefix(modifier))
        .unwrap_or(utterance)
        .trim_start();
    let argument = utterance.strip_prefix(['從', '从', '由', '自'])?;
    let actor = argument.trim();
    let identity = lexicon
        .actors
        .iter()
        .find(|&&(alias, _)| !alias.is_empty() && alias == actor)?
        .1;
    let actor_start = base + role.len() - argument.trim_start().len();
    Some((
        source.span(actor_start, actor_start + actor.len()),
        identity.to_owned(),
    ))
}

fn described_speaker(subject: &str) -> bool {
    const ROLES: &[&str] = &["老者", "青年", "少女", "少婦", "童子", "少年"];
    let subject = [
        "一名", "一位", "這名", "这名", "那名", "這位", "这位", "那位",
    ]
    .iter()
    .find_map(|prefix| subject.strip_prefix(prefix))
    .unwrap_or(subject);
    ROLES.contains(&subject)
        || ROLES
            .iter()
            .chain(SUBJECT_TITLES)
            .filter_map(|role| subject.strip_suffix(role))
            .any(|qualifier| {
                if matches!(
                    qualifier,
                    "高瘦" | "矮胖" | "瘦小" | "瘦削" | "年輕" | "年轻" | "年邁" | "年迈" | "中年"
                ) {
                    return true;
                }
                let mut chars = qualifier.chars();
                let (Some(first), Some(last), None) = (chars.next(), chars.next(), chars.next())
                else {
                    return false;
                };
                ("白黑灰青紅红黃黄藍蓝紫綠绿銀银金素錦锦布儒".contains(first)
                    && "衣袍衫髮发眉鬚须面臉脸".contains(last))
                    || (('一'..='龯').contains(&first) && last == '姓')
            })
}

fn quoted_tag(
    source: &Source<'_>,
    (base, tag): (usize, &str),
    payload: Span,
    context: &str,
    attributed: Option<&Actor>,
    lexicon: &Lexicon<'_>,
    post_quote: bool,
) -> Option<SpeechAct> {
    let trimmed = tag.trim();
    let tag_start = base + tag.len() - tag.trim_start().len();
    if post_quote
        && let Some((lead, cue)) = ["說了出來", "说了出来", "說出口", "说出口", "說出", "说出"]
            .iter()
            .find_map(|&cue| Some((trimmed.strip_suffix(cue)?, cue)))
    {
        let cue_start = tag_start + lead.len();
        if assertion_scope(lead, context)
            && let Some((surface, speaker)) =
                voiced_by(source, lead, cue_start, source.slice(&payload), lexicon)
        {
            return Some(SpeechAct {
                kind: quoted_payload_kind(source.slice(&payload)),
                cue: source.span(cue_start, cue_start + cue.len()),
                speaker_surface: Some(surface),
                speaker: Some(speaker),
                payload,
            });
        }
    }
    let (lead, cue, kind) = SPECIALIZED_CUES
        .iter()
        .flat_map(|(kind, words)| words.iter().map(move |&word| (word, *kind)))
        .chain(QUOTED_CUES.iter().copied())
        .find_map(|(cue, kind)| Some((trimmed.strip_suffix(cue)?, cue, kind)))?;
    if !assertion_scope(lead, context) {
        return None;
    }
    let cue_start = tag_start + lead.len();
    let lead_start = tag_start + lead.len() - lead.trim_start().len();
    let lead = lead.trim();
    let subject = ["忽然", "突然", "當即", "当即", "隨即", "随即"]
        .iter()
        .find_map(|modifier| lead.strip_prefix(modifier))
        .unwrap_or(lead)
        .trim_start();
    let subject_start = lead_start + lead.len() - subject.len();
    let subject = subject.trim_end();
    let exact_name = source
        .first_name(subject_start, subject_start + subject.len(), lexicon)
        .is_some_and(|(offset, word, _)| offset == 0 && word.len() == subject.len());
    let subject = if exact_name {
        subject
    } else {
        strip_quoted_manner(subject, post_quote)
    };
    let speaker_surface =
        (!subject.is_empty()).then(|| source.span(subject_start, subject_start + subject.len()));
    let mut speaker = None;
    if let Some(surface) = speaker_surface {
        let subject_end = subject_start + subject.len();
        if let Some((0, word, canonical)) = source.first_name(subject_start, subject_end, lexicon)
            && word.len() == subject.len()
        {
            speaker = Some(canonical.to_owned());
        } else if let Some(actor) = attributed.filter(|actor| actor.evidence == surface) {
            speaker = Some(actor.name.clone());
        } else if !(matches!(subject, "他" | "她" | "有人")
            || participant_description(subject)
            || post_quote && described_speaker(subject))
        {
            return None;
        }
    }
    Some(SpeechAct {
        kind: if matches!(kind, SpeechActKind::Statement) {
            quoted_payload_kind(source.slice(&payload))
        } else {
            kind
        },
        cue: source.span(cue_start, cue_start + cue.len()),
        speaker_surface,
        speaker,
        payload,
    })
}

// Bounded local form: a written actor performs one action, then a subjectless
// speech cue follows. A written surface does not authenticate a real identity.
fn action_then_speech(
    source: &Source<'_>,
    unit: &Unit,
    payload: Span,
    context: &str,
    attributed: Option<&Actor>,
    lexicon: &Lexicon<'_>,
) -> Option<SpeechAct> {
    let actor = attributed?;
    let close = source.bytes[unit.evidence.end_char];
    let limit = source.bytes[(unit.evidence.end_char + 80).min(source.bytes.len() - 1)];
    let after = &source.text[close..limit];
    let first = after.trim_start_matches([' ', '\t']);
    let first_start = close + after.len() - first.len();
    if source.bytes[actor.evidence.start_char] != first_start {
        return None;
    }
    let comma = first.find([',', '，'])?;
    let clause = &first[..comma];
    let surface = source.slice(&actor.evidence);
    let action = clause.strip_prefix(surface)?;
    let (kind, offset, word) = predicate(action, lexicon)?;
    if kind != "action" || offset != 0 || word.len() != action.len() {
        return None;
    }
    let after_comma = &first[comma..];
    let marker = after_comma.chars().next()?;
    let rest = &after_comma[marker.len_utf8()..];
    let rest = rest.trim_start_matches([' ', '\t']);
    let rest_start = first_start + first.len() - rest.len();
    let end = rest.find(|ch: char| {
        matches!(
            ch,
            '。' | '.'
                | '！'
                | '!'
                | '？'
                | '?'
                | '，'
                | ','
                | '；'
                | ';'
                | '：'
                | ':'
                | '\n'
                | '\r'
        ) || quote_end(ch).is_some()
    })?;
    if !rest[end..].starts_with(['。', '.', '！', '!']) {
        return None;
    }
    let mut act = quoted_tag(
        source,
        (rest_start, &rest[..end]),
        payload,
        context,
        None,
        lexicon,
        false,
    )?;
    if act.speaker_surface.is_some() || act.speaker.is_some() {
        return None;
    }
    act.speaker_surface = Some(actor.evidence);
    act.speaker = Some(actor.name.clone());
    Some(act)
}

pub(super) fn quoted_act(
    source: &Source<'_>,
    unit: &Unit,
    following: Option<&Unit>,
    context: &str,
    lexicon: &Lexicon<'_>,
    conflicting: &mut bool,
) -> Option<SpeechAct> {
    if !unit.quote || unit.evidence.end_char - unit.evidence.start_char < 3 {
        return None;
    }
    let left = source.bytes[unit.speech_context.start_char];
    let right = source.bytes[unit.speech_context.end_char];
    let open = source.bytes[unit.evidence.start_char];
    let close = source.bytes[unit.evidence.end_char];
    let before = &source.text[left..open];
    if !assertion_scope(before, context) {
        return None;
    }
    // Only consume an immediately matched pair around this exact quote.
    // The parentheses themselves do not prove identity or occurrence.
    let wrapper = match (before.chars().last(), source.text[close..].chars().next()) {
        (Some('（'), Some('）')) => Some('）'),
        (Some('('), Some(')')) => Some(')'),
        _ => None,
    };
    let payload = source.trim(
        source.bytes[unit.evidence.start_char + 1],
        source.bytes[unit.evidence.end_char - 1],
    )?;
    let attributed = speech_actor(source, unit, lexicon);
    let pre_scope = before
        .trim_end_matches(|ch: char| ch.is_whitespace() || matches!(ch, ':' | '：' | ',' | '，'));
    let pre = pre_scope
        .rsplit(['，', ',', '；', ';'])
        .next()
        .unwrap_or(pre_scope);
    let pre_start = left + pre_scope.len() - pre.len();
    let pre = quoted_tag(
        source,
        (pre_start, pre),
        payload,
        context,
        attributed.as_ref(),
        lexicon,
        false,
    );
    // An anonymous speaker can follow a sentence break at the closing quote.
    // The following unit must already have passed chapter and line cleanup.
    let right = if right == close {
        following
            .filter(|next| !next.quote && !next.first_in_line)
            .filter(|next| {
                let next_start = source.bytes[next.evidence.start_char];
                let gap = &source.text[close..next_start];
                let gap = wrapper
                    .and_then(|closing| gap.strip_prefix(closing))
                    .unwrap_or(gap);
                gap.chars().all(|ch| {
                    (ch.is_whitespace() && !matches!(ch, '\n' | '\r')) || matches!(ch, '，' | ',')
                })
            })
            .map_or(close, |next| {
                source.bytes[(unit.evidence.end_char + 32).min(next.evidence.end_char)]
            })
    } else {
        right
    };
    let after = &source.text[close..right];
    let after = wrapper
        .and_then(|closing| after.strip_prefix(closing))
        .unwrap_or(after);
    let after_start = close + wrapper.map_or(0, char::len_utf8);
    let post = after.trim_start_matches(|ch: char| {
        (ch.is_whitespace() && !matches!(ch, '\n' | '\r')) || matches!(ch, '，' | ',')
    });
    let post_start = after_start + after.len() - post.len();
    let post_end = post
        .find(|ch| {
            matches!(
                ch,
                '，' | ','
                    | '。'
                    | '.'
                    | '！'
                    | '!'
                    | '？'
                    | '?'
                    | '；'
                    | ';'
                    | '：'
                    | ':'
                    | '\n'
                    | '\r'
            ) || quote_end(ch).is_some()
        })
        .unwrap_or(post.len());
    let next_quote = post[post_end..]
        .chars()
        .next()
        .is_some_and(|ch| matches!(ch, '：' | ':') || quote_end(ch).is_some());
    let post = if next_quote {
        None
    } else {
        let tag = (post_start, &post[..post_end]);
        let same_line = !source.text[close..post_start + post_end]
            .contains(|ch| matches!(ch, '\n'..='\r' | '\u{85}' | '\u{2028}' | '\u{2029}'));
        let existing = quoted_tag(
            source,
            tag,
            payload,
            context,
            attributed.as_ref(),
            lexicon,
            false,
        );
        if existing.is_none() && pre.is_none() && same_line && !tag.1.trim().is_empty() {
            quoted_tag(
                source,
                tag,
                payload,
                context,
                attributed.as_ref(),
                lexicon,
                true,
            )
        } else {
            existing
        }
    };
    let post = if post.is_none() && pre.is_none() {
        action_then_speech(source, unit, payload, context, attributed.as_ref(), lexicon)
    } else {
        post
    };
    match (pre, post) {
        (Some(before), Some(after)) => {
            let before_surface = before.speaker_surface.map(|span| source.slice(&span));
            let after_surface = after.speaker_surface.map(|span| source.slice(&span));
            if before_surface.is_none() {
                Some(after)
            } else if after_surface.is_none() || before_surface == after_surface {
                Some(before)
            } else {
                // Different written surfaces are competing clues even if a
                // caller maps them to the same canonical name.
                *conflicting = true;
                None
            }
        }
        (before, after) => before.or(after),
    }
}
