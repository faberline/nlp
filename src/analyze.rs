use crate::language::*;
use crate::{Analysis, Event, Lexicon, Review, Span, TextSection, TextUnit};

struct Source<'a> {
    text: &'a str,
    bytes: Vec<usize>,
}

impl<'a> Source<'a> {
    fn new(text: &'a str) -> Self {
        let mut bytes: Vec<_> = text.char_indices().map(|(offset, _)| offset).collect();
        bytes.push(text.len());
        Self { text, bytes }
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
}

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

fn split_outside_quotes(text: &str, boundary: impl Fn(char) -> bool) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut stack = Vec::new();
    let mut start = 0;
    for (offset, ch) in text.char_indices() {
        if stack.last() == Some(&ch) {
            stack.pop();
        } else if let Some(end) = quote_end(ch)
            && text[offset + ch.len_utf8()..].contains(end)
        {
            stack.push(end);
        } else if stack.is_empty() && boundary(ch) {
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
                quote: false,
                first_in_line: std::mem::take(first_in_line),
            });
        }
    }
}

fn is_dialogue(body: &str, before: &str, after: &str) -> bool {
    contains_any(body, &["。", "！", "？", "!", "?"])
        || (body.chars().count() >= 5
            && (contains_any(before, SPEECH)
                || contains_any(after, SPEECH)
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
    for (offset, ch) in text.char_indices() {
        if stack.last() == Some(&ch) {
            stack.pop();
            if stack.is_empty() {
                let open = open_at + text[open_at..].chars().next().unwrap().len_utf8();
                let end_quote = offset + ch.len_utf8();
                if is_dialogue(
                    &text[open..offset],
                    &text[next..open_at],
                    &text[end_quote..],
                ) {
                    append_narration(
                        source,
                        units,
                        &sentence,
                        start + next,
                        start + open_at,
                        first_in_line,
                    );
                    units.push(Unit {
                        evidence: source.span(start + open_at, start + end_quote),
                        sentence,
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

fn split_units(source: &Source<'_>, title: &str, lexicon: &Lexicon<'_>) -> Vec<Unit> {
    let end = earliest(&source.text.to_ascii_lowercase(), lexicon.footers)
        .map_or(source.text.len(), |(offset, _)| offset);
    let mut units = Vec::new();
    let mut start = 0;
    for (line_number, line) in source.text[..end].split_inclusive('\n').enumerate() {
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
            continue;
        }
        let end = ["<script", "<div", "<a href", "<iframe", "<!--"]
            .iter()
            .chain(lexicon.line_footers.iter())
            .filter_map(|marker| line.find(marker))
            .min()
            .unwrap_or(line.len());
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
    let Some(hit) = first_name(text, lexicon) else {
        return title_subject(source, source.bytes[span.start_char], text, lexicon);
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
        Some(actor_at(source, source.bytes[span.start_char], hit))
    } else {
        title_subject(source, source.bytes[span.start_char], text, lexicon)
    }
}

fn title_subject(
    source: &Source<'_>,
    base: usize,
    text: &str,
    lexicon: &Lexicon<'_>,
) -> Option<Actor> {
    const TITLES: &[&str] = &[
        "護法", "长老", "長老", "仙子", "真人", "老祖", "掌門", "宗主", "堂主", "道友", "居士",
        "尊者", "大師", "大夫", "老者", "道士", "童子", "魔君", "上人",
    ];
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
    let (offset, title) = earliest(lead, TITLES)?;
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

fn speech_actor(source: &Source<'_>, unit: &Unit, lexicon: &Lexicon<'_>) -> Option<Actor> {
    let left = source.bytes[unit.sentence.start_char];
    let right = source.bytes[unit.sentence.end_char];
    let start = source.bytes[unit.evidence.start_char];
    let end = source.bytes[unit.evidence.end_char];
    let after = &source.text[end..right];
    if let Some(hit) = first_name(after, lexicon) {
        let phrase = &after[hit.0 + hit.1.len()..];
        if after[..hit.0].chars().count() < 6 && contains_any(phrase, SPEECH) {
            return Some(actor_at(source, end, hit));
        }
    }
    let before = &source.text[left..start];
    if let Some(hit) = first_name(before, lexicon)
        && contains_any(&before[hit.0 + hit.1.len()..], SPEECH)
    {
        return Some(actor_at(source, left, hit));
    }
    if contains_any(after, SPEECH)
        && let Some(actor) = title_subject(source, end, after, lexicon)
    {
        return Some(actor);
    }
    if contains_any(before, SPEECH)
        && let Some(actor) = title_subject(source, left, before, lexicon)
    {
        return Some(actor);
    }
    let right = source.bytes[(unit.evidence.end_char + 165).min(source.bytes.len() - 1)];
    let remainder = &source.text[end..right];
    if let Some((offset, _)) = remainder.match_indices("正是").next() {
        let lead = &remainder[..offset];
        let identity = remainder[offset..]
            .split(['。', '\n'])
            .next()
            .unwrap_or_default();
        if contains_any(lead, &["面容", "面孔", "容貌"])
            && let Some(hit) = first_name(identity, lexicon)
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
                let target = &text[offset + word.len()..];
                if *word == "出了" && place_name(target, lexicon).is_none() {
                    return false;
                }
                let prefix: String = target.chars().take(14).collect();
                !contains_any(
                    &prefix,
                    &[
                        "睡夢中",
                        "夢境中",
                        "沉思",
                        "修煉狀態",
                        "修仙之路",
                        "歧途",
                        "話題",
                        "軌道",
                    ],
                )
            })
            .map(|(offset, word)| (kind, offset, word))
    })
    .min_by_key(|(_, offset, word)| (*offset, std::cmp::Reverse(word.len())))
}

fn place_name<'a>(text: &str, lexicon: &Lexicon<'a>) -> Option<&'a str> {
    earliest(text, lexicon.places).map(|(_, place)| place)
}

struct Realm<'a> {
    offset: usize,
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
            let advancement = [
                "進入", "进入", "進階", "进阶", "晉階", "晋阶", "突破", "跨入", "踏入", "踏進",
                "邁入", "達", "躋身",
            ]
            .into_iter()
            .find(|verb| {
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
    text: &str,
    offset: usize,
    word: &str,
    kind: &str,
    lexicon: &Lexicon<'_>,
) -> Option<String> {
    let tail = &text[offset + word.len()..];
    if kind == "movement" {
        return place_name(tail, lexicon).map(str::to_owned);
    }
    if matches!(kind, "encounter" | "combat")
        && let Some((_, _, name)) = first_name(tail, lexicon)
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

fn push_event(events: &mut Vec<Event>, event: Event) {
    if event.kind == "realm_change"
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
        previous.evidence.extend(event.evidence);
        previous.recap &= event.recap;
        return;
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
}

/// Analyze one chapter while preserving offsets into its original text.
pub fn analyze(chapter_text: &str, title: &str, lexicon: &Lexicon<'_>) -> Analysis {
    let source = Source::new(chapter_text);
    let units = split_units(&source, title, lexicon);
    let mut result = Analysis::default();
    let mut context = "main";
    let mut focus: Option<Focus> = None;
    for unit in units {
        let text = source.slice(&unit.evidence);
        let sentence = source.slice(&unit.sentence);
        if !text.chars().any(|ch| ch.is_alphanumeric()) {
            continue;
        }
        let prior_context = context;
        if !unit.quote
            && (unit.first_in_line
                || actual_dream(text)
                || (context == "dream" && contains_any(text, &["醒來", "醒了", "早上起來"]))
                || contains_any(text, &["回想著", "回想起", "想起了", "回憶起"]))
        {
            context = context_for(text, context);
        }
        let realm = realm(text, sentence, lexicon);
        let pred = predicate(text, lexicon);
        let subject = if unit.quote {
            speech_actor(&source, &unit, lexicon)
        } else {
            named_subject(&source, &unit.evidence, lexicon)
        };
        let available_actor = subject.or_else(|| {
            if unit.quote {
                None
            } else {
                focus.as_ref()?.carry(&unit, text, lexicon)
            }
        });
        if !unit.quote {
            if let Some(actor) = &available_actor {
                focus = Some(Focus {
                    actor: actor.clone(),
                    last_reference: unit.evidence,
                    sentence: unit.sentence.start_char,
                });
            } else if introduces_subject(text) {
                focus = None;
            }
        }
        let earlier = &source.text
            [source.bytes[unit.sentence.start_char]..source.bytes[unit.evidence.start_char]];
        let status = event_status(
            text,
            earlier,
            if !unit.quote && realm.is_some() && contains_any(sentence, REPORTED) {
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
        let destination = pred.and_then(|(kind, offset, word)| {
            (kind == "movement"
                && status == "occurred"
                && !starts_any(word, &["離", "出", "逃", "遁離"]))
            .then(|| place_name(&text[offset + word.len()..], lexicon))
            .flatten()
        });
        let establishes_place = starts_any(text, &["在", "到了", "進了", "洞府內", "密室內"])
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
        } else {
            contains_any(text, THOUGHT)
        };
        let frame_kind = if thought {
            "thought"
        } else if unit.quote {
            "dialogue"
        } else if pred.is_some()
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
        let (kind, object) = if thought {
            continue;
        } else if let Some(realm) = realm {
            ("realm_change", Some(realm.target.to_owned()))
        } else if let Some((kind, offset, word)) = pred.filter(|_| !unit.quote) {
            (kind, object_after(text, offset, word, kind, lexicon))
        } else {
            continue;
        };
        let mut evidence = vec![unit.evidence];
        if let Some(actor) = available_actor.as_ref()
            && (actor.evidence.start_char < evidence[0].start_char
                || actor.evidence.end_char > evidence[0].end_char)
        {
            evidence.push(actor.evidence);
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
        push_event(
            &mut result.events,
            Event {
                kind: kind.into(),
                status: status.into(),
                actor: available_actor.map(|actor| actor.name.to_owned()),
                object,
                action: excerpt(text),
                evidence,
                recap: context == "flashback" || contains_any(text, RECALL),
            },
        );
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
