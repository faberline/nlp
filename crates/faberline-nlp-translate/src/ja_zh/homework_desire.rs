//! Recognize the desire to submit or assign homework before assigning roles.
use super::{
    Atom, Document, TargetVariant, Upos, compose_relative_daypart, group_into_bunsetsus,
    merge_adnominal_bunsetsus, try_translate_homework_transfer,
};

fn clock_qualifier<'a>(
    prefix: &'a [Atom],
    time: &Atom,
    interrogative: bool,
    doc: &Document,
) -> Option<(usize, &'a Atom)> {
    match prefix {
        [.., day, no]
            if !day.is_locked
                && !no.is_locked
                && day.deprel == "nmod"
                && no.surface == "の"
                && no.deprel == "case"
                && no.head == day.doc_tok_idx
                && day.span.end_char == no.span.start_char
                && no.span.end_char == time.span.start_char
                && day
                    .head
                    .and_then(|id| doc.tokens.get(id))
                    .is_some_and(|head| {
                        (time.span.start_char..time.span.end_char).contains(&head.span.start_char)
                    }) =>
        {
            Some((prefix.len() - 2, day))
        }
        // In 午前何時 the analyzer can attach both 午前 and 何時 to 出す.
        [.., period]
            if interrogative
                && !period.is_locked
                && super::is_daypart_surface(&period.surface)
                && period.deprel == "advmod"
                && (period.head == time.head || period.head == time.doc_tok_idx)
                && period.span.end_char == time.span.start_char =>
        {
            Some((prefix.len() - 1, period))
        }
        _ => None,
    }
}

pub(super) fn modal(tail: &[Atom]) -> Option<&'static str> {
    let suffix: String = tail.iter().map(|a| a.surface.as_str()).collect();
    let plain = suffix
        .strip_suffix("のです")
        .or_else(|| suffix.strip_suffix("んです"))
        .or_else(|| suffix.strip_suffix("です"))
        .unwrap_or(&suffix);
    Some(match plain {
        "たい" => "想",
        "たかった" => "當時想",
        "たくない" | "たくはない" | "たくありません" | "たくはありません" => {
            "不想"
        }
        "たくなかった" | "たくはなかった" | "たくありませんでした" | "たくはありませんでした" => {
            "當時不想"
        }
        _ => return None,
    })
}

pub(super) fn intention_adverb(time: &Atom) -> Option<String> {
    if time.is_locked || time.deprel != "advmod" {
        return None;
    }
    Some(match time.surface.as_str() {
        "今朝" => "今天早上".to_owned(),
        "昨夜" | "昨晩" => "昨晚".to_owned(),
        _ => super::render_temporal_made_atom(time)?,
    })
}

pub(super) fn report_time(args: &[Atom]) -> Option<String> {
    let args = match args {
        [prefix @ .., comma] if super::homework_relative::is_argument_comma(comma) => prefix,
        _ => args,
    };
    let mut time = super::normalize_ja_atoms(args);
    super::calendar::normalize(&mut time, false);
    match time.as_slice() {
        [] => Some(String::new()),
        [time] if !time.is_locked && time.deprel == "compound" => match time.surface.as_str() {
            "今朝" => Some("今天早上".to_owned()),
            "昨夜" | "昨晩" => Some("昨晚".to_owned()),
            _ => None,
        },
        [time] => intention_adverb(time),
        [time, ni]
            if ni.surface == "に"
                && (super::is_clock_atom(time) || super::calendar::is_date(time)) =>
        {
            Some(format!("在{}", time.surface))
        }
        _ => None,
    }
}

/// A time between a classroom topic and its recipient can date the intention.
fn intention_frame(args: &[Atom], transfer: &Atom) -> Option<(usize, String)> {
    let pos = args.windows(2).position(|w| {
        matches!(w[0].deprel.as_str(), "obl" | "iobj")
            && w[0].head == transfer.doc_tok_idx
            && w[1].surface == "に"
            && super::dasu_case_link(&w[0], &w[1])
            && super::classroom_role(&[], &w[0]).is_some()
    })?;
    let frame = &args[..pos];
    let framed = frame
        .last()
        .is_some_and(super::homework_relative::is_argument_comma);
    let relative_past = frame.first().is_some_and(|a| {
        matches!(
            a.surface.as_str(),
            "今" | "昨日" | "今朝" | "昨夜" | "昨晩" | "先週" | "先月" | "去年"
        )
    });
    let dated = frame.iter().any(|a| a.surface == "に");
    if !(relative_past || dated && (framed || has_event_deadline(&args[pos + 2..], transfer))) {
        return None;
    }
    let time = report_time(frame)?;
    (!time.is_empty()).then_some((pos, time))
}

#[derive(Clone, Copy)]
pub(super) struct Desire<'a> {
    pub(super) modal: &'a str,
    pub(super) question: bool,
    pub(super) within: Option<&'a str>,
}

/// Keep a bounded calendar period attached to 出す. 中に following a place
/// or a class activity has a different meaning.
fn within_period(period: &Atom, marker: &Atom, verb: &Atom) -> Option<String> {
    if period.is_locked
        || marker.is_locked
        || period.span.end_char != marker.span.start_char
        || marker.surface != "中に"
        || marker.deprel != "advmod"
        || marker.head != verb.doc_tok_idx
        || period.head != marker.doc_tok_idx
        || period.deprel != "advmod"
        || !matches!(
            period.surface.as_str(),
            "今日"
                | "明日"
                | "昨日"
                | "今週"
                | "来週"
                | "先週"
                | "今月"
                | "来月"
                | "先月"
                | "今年"
                | "来年"
                | "去年"
        )
    {
        return None;
    }
    Some(format!(
        "在{}之內",
        super::render_temporal_made_atom(period)?
    ))
}

pub(super) fn has_event_deadline(args: &[Atom], verb: &Atom) -> bool {
    args.windows(2).any(|pair| {
        let [start, end] = pair else { return false };
        within_period(start, end, verb).is_some()
            || (!start.is_locked
                && !end.is_locked
                && start.surface == "まで"
                && end.surface == "に"
                && start.deprel == "case"
                && start.head.is_some()
                && start.head == end.head)
    })
}

/// Keep the topic of 思っている and its time outside a separately marked
/// が-clause. With just one classroom topic, the thought is its own intention.
pub(super) fn reported(
    source: &[Atom],
    doc: &Document,
    sentence: usize,
    target: TargetVariant,
) -> Option<String> {
    let thought_idx = source
        .iter()
        .rposition(|a| a.lemma.as_deref() == Some("思う"))?;
    let [te, ongoing, ending @ ..] = &source[thought_idx + 1..] else {
        return None;
    };
    let thought = &source[thought_idx];
    if te.surface != "て"
        || te.head != thought.doc_tok_idx
        || ongoing.lemma.as_deref() != Some("いる")
        || thought.head != ongoing.doc_tok_idx
        || source.iter().any(|a| {
            a.surface.contains("[[PH")
                || (a.is_locked && !super::is_clock_atom(a) && !super::calendar::is_date(a))
        })
    {
        return None;
    }
    let (ending, question) = match ending {
        [affixes @ .., ka]
            if !ka.is_locked
                && ka.surface == "か"
                && ka.deprel == "discourse"
                && ka.head == ongoing.doc_tok_idx =>
        {
            (affixes, true)
        }
        _ => (ending, false),
    };
    let suffix: String = ending.iter().map(|a| a.surface.as_str()).collect();
    let (past, negative) = match suffix.as_str() {
        "" | "ます" => (false, false),
        "た" | "ました" => (true, false),
        "ない" | "ません" => (false, true),
        "なかった" | "ませんでした" => (true, true),
        _ => return None,
    };
    let reported = match &source[..thought_idx] {
        [phrase @ .., focus]
            if negative
                && focus.surface == "は"
                && focus.deprel == "case"
                && phrase
                    .last()
                    .is_some_and(|q| q.surface == "と" && q.head == focus.head) =>
        {
            phrase
        }
        phrase => phrase,
    };
    let (quote, content) = reported.split_last()?;
    if quote.surface != "と" || quote.deprel != "case" {
        return None;
    }
    let pred = content
        .iter()
        .rposition(|a| a.lemma.as_deref() == Some("出す"))?;
    let transfer = &content[pred];
    let modal = modal(&content[pred + 1..])?;
    if transfer.head != thought.doc_tok_idx || quote.head != transfer.doc_tok_idx {
        return None;
    }
    let idx = content.windows(2).position(|w| {
        w[0].deprel == "nsubj" && w[1].surface == "は" && super::dasu_case_link(&w[0], &w[1])
    })?;
    let topic = &content[idx];
    let before_topic = &content[..idx];
    let after_topic = &content[idx + 2..pred];
    let second = after_topic.windows(2).position(|w| {
        w[0].head == transfer.doc_tok_idx
            && w[0].deprel == "nsubj"
            && w[1].surface == "が"
            && super::dasu_case_link(&w[0], &w[1])
            && super::classroom_role(&[], &w[0]).is_some()
    });
    let before = report_time(before_topic)?;
    let mut matrix = String::new();
    let mut desire = modal;
    let inner;
    let mut scoped_content;
    let dated_desire;
    if let Some(second) = second {
        if !super::is_companion_person(topic) {
            return None;
        }
        let after = report_time(&after_topic[..second])?;
        if !before.is_empty() && !after.is_empty() {
            return None;
        }
        let person =
            super::translate_single_ja_term(&topic.surface, topic.lemma.as_deref(), topic.upos);
        matrix.push_str(&person);
        matrix.push_str(&before);
        matrix.push_str(&after);
        if past && before.is_empty() && after.is_empty() {
            matrix.push_str("當時");
        }
        matrix.push_str(if negative { "不覺得" } else { "覺得" });
        inner = &content[idx + 2 + second..];
    } else {
        if !before.is_empty() || super::classroom_role(&[], topic).is_none() || modal != "想" {
            return None;
        }
        desire = match (past, negative) {
            (false, false) => modal,
            (true, false) => "當時想",
            (false, true) => "沒有打算",
            (true, true) => "當時沒有打算",
        };
        if let Some((length, time)) = intention_frame(after_topic, transfer) {
            scoped_content = content[..idx + 2].to_vec();
            scoped_content.extend_from_slice(&content[idx + 2 + length..]);
            inner = &scoped_content;
            dated_desire = format!("{time}{}", desire.strip_prefix("當時").unwrap_or(desire));
            desire = &dated_desire;
        } else {
            inner = content;
        }
    }
    let mut normalized = super::normalize_building_floors(
        super::normalize_ja_atoms(&super::apply_phrase_matches(inner)),
        doc,
    );
    super::calendar::normalize(&mut normalized, false);
    let embedded = render(
        inner,
        &normalized,
        doc,
        sentence,
        target,
        Some((desire, question)),
    )?;
    matrix.push_str(&embedded);
    Some(matrix)
}

pub(super) fn render(
    source: &[Atom],
    normalized: &[Atom],
    doc: &Document,
    sentence: usize,
    target: TargetVariant,
    report: Option<(&str, bool)>,
) -> Option<String> {
    let (clause, question) = match source {
        [clause @ .., ka] if !ka.is_locked && ka.surface == "か" && ka.deprel == "discourse" => {
            (clause, true)
        }
        _ => (source, false),
    };
    let i = clause.iter().rposition(|a| {
        !a.is_locked && a.upos == Some(Upos::Verb) && a.lemma.as_deref() == Some("出す")
    })?;
    let (verb, tail) = clause[i..].split_first()?;
    let first = tail.first()?;
    if !matches!(
        first.surface.as_str(),
        "たい" | "たく" | "たかっ" | "たかった"
    ) || tail.iter().any(|a| a.is_locked)
        || !clause[i..]
            .windows(2)
            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
    {
        return None;
    }
    // Polite negative forms can parse as 出し → たく → あり. In either parse
    // the arguments must belong to 出す and the suffix must finish this predicate.
    if !(first.head == verb.doc_tok_idx && first.deprel == "aux"
        || verb.head == first.doc_tok_idx && verb.deprel == "acl" && first.surface == "たく")
    {
        return None;
    }
    let (modal, question) = report.unwrap_or((modal(tail)?, question));
    let idx = normalized
        .iter()
        .position(|a| a.doc_tok_idx == verb.doc_tok_idx && a.span == verb.span)?;
    let mut transfer = normalized[..=idx].to_vec();
    // Group the arguments of 出す without turning the past tense of wanting
    // into a completed transfer. The public source analysis is kept intact.
    let action = transfer.last_mut()?;
    action.deprel = "root".to_owned();
    action.head = None;
    // Merge a date with its own clock before の renders the date as an
    // unattached Chinese modifier. Preserve the source clock and case links.
    let mut end = transfer.len();
    let mut asks_qualified_time = false;
    while end >= 2 {
        let time = &transfer[end - 1];
        let interrogative =
            super::deadline_time_question(time).filter(|_| time.head == verb.doc_tok_idx);
        if (time.lemma.as_deref() == Some(super::CLOCK_ATOM)
            || super::is_daypart_surface(&time.surface)
            || interrogative.is_some())
            && let Some((start, day)) =
                clock_qualifier(&transfer[..end - 1], time, interrogative.is_some(), doc)
            && let Some(day_label) = super::render_temporal_made_atom(day)
            && let Some(clock_label) =
                super::render_temporal_made_atom(time).or_else(|| interrogative.map(str::to_owned))
        {
            asks_qualified_time |= interrogative.is_some();
            let mut qualified = time.clone();
            qualified.span.start_char = day.span.start_char;
            qualified.surface = format!("{day_label}{clock_label}");
            qualified.lemma = Some(super::CLOCK_ATOM.to_owned());
            qualified.is_locked = true;
            transfer.splice(start..end, [qualified]);
            end = start + 1;
        } else {
            end -= 1;
        }
    }
    let within = transfer
        .windows(2)
        .enumerate()
        .find_map(|(i, pair)| within_period(&pair[0], &pair[1], verb).map(|label| (i, label)))
        .map(|(i, label)| {
            transfer.drain(i..i + 2);
            label
        });
    let groups = compose_relative_daypart(group_into_bunsetsus(&transfer, doc, sentence));
    let groups = merge_adnominal_bunsetsus(groups, &clause[..=i], doc, target);
    try_translate_homework_transfer(
        &groups,
        &clause[..=i],
        doc,
        target,
        Some(Desire {
            modal,
            question: question && !asks_qualified_time,
            within: within.as_deref(),
        }),
        None,
    )
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
    };

    #[test]
    fn desire_scopes_over_the_classroom_transfer() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        for (source, expected) in [
            (
                "生徒は先生に六時に宿題を出したいです。",
                "學生想在六點把作業交給老師。",
            ),
            ("先生は学生に宿題を出したいです。", "老師想出作業給學生。"),
            (
                "生徒は昨日先生に宿題を出したいと思っていました。",
                "學生昨天想把作業交給老師。",
            ),
            (
                "生徒は昨日、先生に今日中に宿題を出したいと思っていました。",
                "學生昨天想在今天之內把作業交給老師。",
            ),
            (
                "生徒は六月七日に、先生に六月八日までに宿題を出したいと思っていました。",
                "學生在六月七日想最晚在六月八日把作業交給老師。",
            ),
            (
                "生徒は昨日先生に今日中に宿題を出したいと思っていませんでしたか。",
                "學生昨天沒有打算在今天之內把作業交給老師嗎？",
            ),
            (
                "私は昨日先生が生徒に今日中に宿題を出したいと思っていました。",
                "我昨天覺得老師想在今天之內出作業給學生。",
            ),
            (
                "先生は昨日来た生徒に宿題を出したいと思っていました。",
                "老師當時想出作業給昨天來的學生。",
            ),
            (
                "先生は昨日、学生に今週中に宿題を出したいと思っていました。",
                "老師昨天想在這週之內出作業給學生。",
            ),
            (
                "私は六時に、先生が生徒に七時までに宿題を出したいと思っていました。",
                "我在六點覺得老師想最晚在七點出作業給學生。",
            ),
            (
                "生徒は昨日、先生に宿題を出したいと思っていましたか。",
                "學生昨天想把作業交給老師嗎？",
            ),
            (
                "私は昨日、先生が生徒に宿題を出したいと思っていました。",
                "我昨天覺得老師想出作業給學生。",
            ),
            (
                "母は先週、生徒が先生に明日の朝に宿題を出したいと思っていました。",
                "母親上週覺得學生想在明天早上把作業交給老師。",
            ),
            (
                "私は今朝、先生が生徒に明日の朝に宿題を出したいと思っていました。",
                "我今天早上覺得老師想在明天早上出作業給學生。",
            ),
            (
                "生徒は昨日、先生に宿題を出したいと思っていました。今日、生徒は先生に宿題を出しました。",
                "學生昨天想把作業交給老師。學生今天把作業交給老師了。",
            ),
            (
                "私は先生が明日、生徒に宿題を出したいと思っています。",
                "我覺得老師想明天出作業給學生。",
            ),
            (
                "生徒は先生に明日宿題を出したいと思っていました。",
                "學生當時想明天把作業交給老師。",
            ),
            (
                "私は昨日、先生が生徒に宿題を出したくないと思っていました。",
                "我昨天覺得老師不想出作業給學生。",
            ),
            (
                "生徒は昨日、先生に明日の朝に宿題を出したいと思っていませんでした。",
                "學生昨天沒有打算在明天早上把作業交給老師。",
            ),
            (
                "学生は先生に宿題を出したくないですか。",
                "學生不想把作業交給老師嗎？",
            ),
            (
                "生徒は先生に宿題を出したかったです。",
                "學生當時想把作業交給老師。",
            ),
            (
                "先生は学生に宿題を出したくありませんでした。",
                "老師當時不想出作業給學生。",
            ),
            (
                "先生は学生に宿題を出したくはありません。",
                "老師不想出作業給學生。",
            ),
            (
                "学生は先生に五時までに宿題を出したいです。",
                "學生想最晚在五點把作業交給老師。",
            ),
            (
                "生徒は先生に何時に宿題を出したいですか。",
                "學生想在幾點把作業交給老師？",
            ),
            (
                "生徒は先生に明日の何時に宿題を出したいですか。",
                "學生想在明天幾點把作業交給老師？",
            ),
            (
                "先生は生徒に日曜日の何時に宿題を出したくなかったですか。",
                "老師當時不想在星期日幾點出作業給學生？",
            ),
            (
                "生徒は先生に明日の午前何時に宿題を出したいですか。",
                "學生想在明天上午幾點把作業交給老師？",
            ),
            (
                "先生は明日の宿題を今日の何時に生徒に出したいですか。",
                "老師想在今天幾點出明天的作業給學生？",
            ),
            (
                "生徒は先生に明日の九時に宿題を出したいですか。",
                "學生想在明天九點把作業交給老師嗎？",
            ),
            (
                "生徒は先生にいつ宿題を出したいですか。",
                "學生想什麼時候把作業交給老師？",
            ),
            (
                "生徒は先生に五時まで宿題を出したくありません。",
                "學生直到五點都不想把作業交給老師。",
            ),
            (
                "先生は学生に正午から宿題を出したいです。",
                "老師想從正午起出作業給學生。",
            ),
            (
                "先生は学生に正午まで宿題を出したいです。",
                "老師想出作業給學生，一直到正午。",
            ),
            (
                "生徒は先生に明日の六時に宿題を出したいです。",
                "學生想在明天六點把作業交給老師。",
            ),
            (
                "先生は生徒に月曜日の三時から宿題を出したいです。",
                "老師想從星期一三點起出作業給學生。",
            ),
            (
                "先生は生徒に今日の正午から宿題を出したいです。",
                "老師想從今天正午起出作業給學生。",
            ),
        ] {
            let actual = analyze_and_translate(
                source,
                &options,
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(actual.target_text, expected, "{source}");
        }
    }
}
