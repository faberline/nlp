//! Render a schedule ending at a time: 会議は五時までです.
use super::{
    Atom, CLOCK_ATOM, Document, TargetVariant, Upos, deadline_weekday, group_into_bunsetsus,
    is_daypart_surface, merge_adnominal_bunsetsus, render_bunsetsu_np, render_clock_surface,
    render_temporal_made_atom,
};

const LOCKED_QUESTION: &str = "__ja_copular_time_question";

fn time_question(source: &str) -> Option<&'static str> {
    match source {
        "何時" => Some("幾點"),
        "いつ" => Some("什麼時候"),
        "何曜日" => Some("星期幾"),
        "何日" => Some("幾號"),
        _ => None,
    }
}

// The caller supplies the display text. Preserve the question only when the
// complete source question word was protected by a glossary entry.
pub(super) fn mark_questions(atoms: &mut [Atom], chars: &[char], locks: &super::SpanLockTable) {
    if locks.locks.is_empty() || !atoms.iter().any(|a| a.surface == "まで") {
        return;
    }
    for i in 1..atoms.len() {
        if !atoms[i].is_locked
            && atoms[i].deprel == "case"
            && matches!(atoms[i].surface.as_str(), "まで" | "から")
            && atoms[i - 1].is_locked
            && atoms[i - 1].doc_tok_idx == atoms[i].head
            && let Some(lock) = locks
                .locks
                .iter()
                .find(|l| l.placeholder == atoms[i - 1].surface)
            && matches!(lock.origin, super::LockOrigin::Glossary { .. })
            && chars
                .get(lock.source_span.start_char..lock.source_span.end_char)
                .is_some_and(|raw| time_question(&raw.iter().collect::<String>()).is_some())
        {
            atoms[i - 1].lemma = Some(LOCKED_QUESTION.to_owned());
        }
    }
}

fn clock_label(label: String) -> String {
    // A late PM hour in a schedule is in the evening in Taiwanese Mandarin.
    if let Some(rest) = label.strip_prefix("下午")
        && let Some((hour, _)) = rest.split_once('點')
        && super::clock_number(hour).is_some_and(|hour| (7..=11).contains(&hour))
    {
        return format!("晚上{rest}");
    }
    label
}

fn clock(words: &[Atom], doc: &Document) -> Option<(String, bool)> {
    let first = words.first()?;
    let last = words.last()?;
    if words.len() == 1 && first.lemma.as_deref() == Some(LOCKED_QUESTION) {
        return Some((first.surface.clone(), true));
    }
    if words.len() == 1 && first.lemma.as_deref() == Some(CLOCK_ATOM) {
        return Some((clock_label(first.surface.clone()), false));
    }
    if words
        .iter()
        .any(|a| a.is_locked && !matches!(a.lemma.as_deref(), Some(CLOCK_ATOM | LOCKED_QUESTION)))
        || !words
            .windows(2)
            .all(|p| p[0].span.end_char == p[1].span.start_char)
    {
        return None;
    }
    if let [day, no, rest @ ..] = words
        && no.surface == "の"
        && no.deprel == "case"
        && no.head == day.doc_tok_idx
        && day.deprel == "nmod"
        && let Some(head) = day.head.and_then(|id| doc.tokens.get(id))
        && rest.iter().any(|a| {
            a.span.start_char <= head.span.start_char && a.span.end_char >= head.span.end_char
        })
        && let Some(label) = render_temporal_made_atom(day)
    {
        let (time, wh) = clock(rest, doc)?;
        return Some((format!("{label}{time}"), wh));
    }
    let spelling: String = words.iter().map(|a| a.surface.as_str()).collect();
    if let Some(time) = time_question(&spelling) {
        return Some((time.to_owned(), true));
    }
    if let Some((time, false)) = render_clock_surface(&spelling) {
        return Some((clock_label(time), false));
    }
    if spelling == "一時" {
        return Some(("一點".to_owned(), false));
    }
    if words.len() > 1 && is_daypart_surface(&first.surface) && first.deprel == "advmod" {
        let (time, wh) = clock(&words[1..], doc)?;
        return Some((format!("{}{time}", render_temporal_made_atom(first)?), wh));
    }
    if words.len() == 1
        && (is_daypart_surface(&last.surface)
            || deadline_weekday(&last.surface).is_some()
            || matches!(
                last.surface.as_str(),
                "今" | "今日"
                    | "昨日"
                    | "明日"
                    | "明後日"
                    | "翌日"
                    | "今週"
                    | "先週"
                    | "来週"
                    | "週末"
                    | "今月"
                    | "先月"
                    | "来月"
                    | "月末"
                    | "今年"
                    | "去年"
                    | "来年"
                    | "年末"
            ))
    {
        return Some((render_temporal_made_atom(last)?, false));
    }
    None
}

fn linked_time(words: &[Atom], case: &Atom, doc: &Document) -> Option<(String, bool)> {
    let last = words.last()?;
    let governor = &doc.tokens.get(case.head?)?.span;
    if case.is_locked
        || case.deprel != "case"
        || case.upos != Some(Upos::Adp)
        || last.span.end_char != case.span.start_char
        || governor.start_char < last.span.start_char
        || governor.end_char > last.span.end_char
    {
        return None;
    }
    clock(words, doc)
}

fn subject(
    words: &[Atom],
    doc: &Document,
    sentence: usize,
    target: TargetVariant,
) -> Option<String> {
    let mut words = words.to_vec();
    if let Some(i) = words.windows(3).position(|p| {
        !p.iter().any(|a| a.is_locked)
            && matches!(p[0].surface.as_str(), "誰" | "だれ")
            && p[1].surface == "か"
            && p[1].deprel == "discourse"
            && p[2].surface == "の"
            && p[2].deprel == "case"
            && p[0].doc_tok_idx == p[1].head
            && p[0].doc_tok_idx == p[2].head
    }) {
        words[i].surface = "某人".to_owned();
        words[i].is_locked = true;
        words[i].span.end_char = words[i + 1].span.end_char;
        words.remove(i + 1);
    }
    let noun = words.last_mut()?;
    // These nouns name an activity or service when its hours are the predicate.
    if !noun.is_locked
        && let Some(label) = match noun.surface.as_str() {
            "授業" => Some("課"),
            "受付" => Some("受理"),
            "試験" => Some("考試"),
            _ => None,
        }
    {
        noun.surface = label.to_owned();
        noun.is_locked = true;
    }
    if let [which, head] = words.as_slice()
        && !which.is_locked
        && which.surface == "どの"
        && which.deprel == "det"
        && which.head == head.doc_tok_idx
    {
        let group = group_into_bunsetsus(std::slice::from_ref(head), doc, sentence);
        let [group] = group.as_slice() else {
            return None;
        };
        return Some(format!("哪個{}", render_bunsetsu_np(group, target)));
    }
    let groups = group_into_bunsetsus(&words, doc, sentence);
    let merged = merge_adnominal_bunsetsus(groups, &words, doc, target);
    let [phrase] = merged.as_slice() else {
        return None;
    };
    Some(render_bunsetsu_np(phrase, target))
}

pub(super) fn render(
    atoms: &[Atom],
    doc: &Document,
    sentence: usize,
    target: TargetVariant,
) -> Option<String> {
    let at = atoms
        .iter()
        .rposition(|a| !a.is_locked && a.surface == "まで")?;
    let (prefix, suffix) = atoms.split_at(at);
    let (made, suffix) = suffix.split_first()?;
    let (suffix, ending) = match suffix {
        [words @ .., yo, ne]
            if !yo.is_locked
                && !ne.is_locked
                && yo.surface == "よ"
                && ne.surface == "ね"
                && yo.deprel == "discourse"
                && ne.deprel == "discourse" =>
        {
            (words, "呢")
        }
        [words @ .., part] if !part.is_locked && part.deprel == "discourse" => (
            words,
            match part.surface.as_str() {
                "か" => "嗎",
                "ね" => "呢",
                "よ" => "喔",
                _ => return None,
            },
        ),
        _ => (suffix, ""),
    };
    if suffix.is_empty()
        || suffix.iter().any(|a| a.is_locked)
        || !atoms[at..]
            .windows(2)
            .all(|p| p[0].span.end_char == p[1].span.start_char)
    {
        return None;
    }
    let copula: String = suffix.iter().map(|a| a.surface.as_str()).collect();
    let tense = match copula.as_str() {
        "です" | "だ" => "",
        "でした" | "だった" => "當時",
        "ではない"
        | "ではありません"
        | "じゃない"
        | "じゃありません"
        | "ではないです"
        | "じゃないです" => "不是",
        "ではなかった"
        | "ではなかったです"
        | "ではありませんでした"
        | "じゃなかった"
        | "じゃなかったです"
        | "じゃありませんでした" => "當時不是",
        _ => return None,
    };
    let mut text = String::new();
    let mut wh = false;
    let time_words = if let Some(i) = prefix.iter().position(|a| {
        !a.is_locked && a.deprel == "case" && matches!(a.surface.as_str(), "は" | "も" | "が")
    }) {
        let head = prefix.get(i.checked_sub(1)?)?;
        if head.deprel != "nsubj"
            || prefix[i].head != head.doc_tok_idx
            || !(head.head == made.head || suffix.iter().any(|a| a.doc_tok_idx == head.head))
        {
            return None;
        }
        text = subject(&prefix[..i], doc, sentence, target)?;
        wh = prefix[..i].iter().enumerate().any(|(index, a)| {
            !a.is_locked
                && matches!(
                    a.surface.as_str(),
                    "誰" | "だれ" | "どなた" | "どの" | "どちら" | "どこ" | "どれ" | "何"
                )
                && !prefix
                    .get(index + 1)
                    .is_some_and(|next| matches!(next.surface.as_str(), "か" | "も" | "でも"))
        });
        if prefix[i].surface == "も" {
            text.push('也');
        }
        &prefix[i + 1..]
    } else {
        prefix
    };
    text.push_str(tense);
    let end_words = if let Some(i) = time_words
        .iter()
        .position(|a| !a.is_locked && a.surface == "から")
    {
        let (start, asks) = linked_time(&time_words[..i], &time_words[i], doc)?;
        text.push_str(&format!("從{start}"));
        wh |= asks;
        &time_words[i + 1..]
    } else {
        time_words
    };
    let (time, asks) = linked_time(end_words, made, doc)?;
    text.push_str(&format!("到{time}"));
    if ending != "嗎" || !(wh || asks) {
        text.push_str(ending);
    }
    Some(text)
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary,
        analyze_and_translate,
    };

    fn translate(jp: &str, glossary: &TranslationGlossary) -> String {
        analyze_and_translate(
            jp,
            &TranslateOptions::new(TargetVariant::ZhHant),
            glossary,
            None,
            Backend::Onnx,
        )
        .unwrap()
        .target_text
    }

    #[test]
    fn temporal_copula_keeps_the_subject_and_the_limits() {
        let glossary = TranslationGlossary::new();
        for (jp, zh) in [
            ("午前の作業は十二時までです。", "上午的作業到十二點。"),
            ("今日の会議は五時まででした。", "今天的會議當時到五點。"),
            ("会議は三時までじゃなかったです。", "會議當時不是到三點。"),
            ("先生の授業は十時まででしたか。", "老師的課當時到十點嗎？"),
            ("会議も三時までです。", "會議也到三點。"),
            ("授業は何時までですか。", "課到幾點？"),
            ("誰の授業が三時までですか。", "誰的課到三點？"),
            ("授業は何時から五時までですか。", "課從幾點到五點？"),
            (
                "営業は朝九時から午後五時までです。",
                "營業從早上九點到下午五點。",
            ),
            (
                "停電は木曜日の午後十一時から金曜日の午前二時までです。",
                "停電從星期四晚上十一點到星期五凌晨兩點。",
            ),
            ("会議は五時までですよね。", "會議到五點呢。"),
            ("会議は三時です。", "會議是三點。"),
            ("駅まで歩きます。", "走到車站。"),
            ("三時までに書類を出します。", "最晚在三點拿出書類。"),
        ] {
            assert_eq!(translate(jp, &glossary), zh, "{jp}");
        }
        let spatial = translate("この電車は東京までです。", &glossary);
        assert!(
            spatial.contains("東京") && !spatial.contains("點"),
            "{spatial}"
        );
    }

    #[test]
    fn opaque_subject_times_and_questions_preserve_the_user_spelling() {
        let mut glossary = TranslationGlossary::new();
        for (source, text) in [
            ("作業", "試作"),
            ("十二時", "約定鐘點"),
            ("何時", "指定疑時"),
        ] {
            glossary.domain_terms.push(GlossaryEntry {
                canonical: source.to_owned(),
                surfaces: vec![source.to_owned()],
                targets: [(TargetVariant::ZhHant, text.to_owned())]
                    .into_iter()
                    .collect(),
                ..Default::default()
            });
        }
        assert_eq!(
            translate("午前の作業は十二時までです。", &glossary),
            "上午的試作到約定鐘點。"
        );
        assert_eq!(
            translate("会議は何時までですか。", &glossary),
            "會議到指定疑時？"
        );
    }
}
