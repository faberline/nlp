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
