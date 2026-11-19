//! Recognize the desire to submit or assign homework before assigning roles.
use super::{
    Atom, Document, TargetVariant, Upos, compose_relative_daypart, group_into_bunsetsus,
    merge_adnominal_bunsetsus, try_translate_homework_transfer,
};

pub(super) fn render(
    source: &[Atom],
    normalized: &[Atom],
    doc: &Document,
    sentence: usize,
    target: TargetVariant,
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
    let suffix: String = tail.iter().map(|a| a.surface.as_str()).collect();
    let plain = suffix
        .strip_suffix("のです")
        .or_else(|| suffix.strip_suffix("んです"))
        .or_else(|| suffix.strip_suffix("です"))
        .unwrap_or(&suffix);
    let modal = match plain {
        "たい" => "想",
        "たかった" => "當時想",
        "たくない" | "たくはない" | "たくありません" | "たくはありません" => {
            "不想"
        }
        "たくなかった" | "たくはなかった" | "たくありませんでした" | "たくはありませんでした" => {
            "當時不想"
        }
        _ => return None,
    };
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
    while end >= 3 {
        let start = end - 3;
        let (day, no, time) = (&transfer[start], &transfer[start + 1], &transfer[start + 2]);
        if !day.is_locked
            && !no.is_locked
            && day.deprel == "nmod"
            && no.surface == "の"
            && no.deprel == "case"
            && no.head == day.doc_tok_idx
            && day.span.end_char == no.span.start_char
            && no.span.end_char == time.span.start_char
            && (time.lemma.as_deref() == Some(super::CLOCK_ATOM)
                || super::is_daypart_surface(&time.surface))
            && let Some(raw_head) = day.head.and_then(|id| doc.tokens.get(id))
            && (time.span.start_char..time.span.end_char).contains(&raw_head.span.start_char)
            && let Some(day_label) = super::render_temporal_made_atom(day)
            && let Some(clock_label) = super::render_temporal_made_atom(time)
        {
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
    let groups = compose_relative_daypart(group_into_bunsetsus(&transfer, doc, sentence));
    let groups = merge_adnominal_bunsetsus(groups, &clause[..=i], doc, target);
    try_translate_homework_transfer(&groups, &clause[..=i], doc, target, Some((modal, question)))
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
