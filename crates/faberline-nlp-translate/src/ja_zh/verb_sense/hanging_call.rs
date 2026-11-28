//! Resolve four finite uses of 掛ける by their arguments.

use super::{
    Aspect, Atom, Nominal, RenderedFrame, Upos, adjacent, dasu_case_link, linked, read_nominal,
};
use crate::{ja_zh::elapsed_number, opaque::to_zh_numeral};

pub(super) fn is_predicate(atom: &Atom) -> bool {
    atom.deprel == "root" && matches!(atom.lemma.as_deref(), Some("掛ける" | "かける"))
}

fn split_case<'a>(
    words: &'a [Atom],
    spellings: &[&str],
) -> Option<(&'a [Atom], &'a Atom, &'a [Atom])> {
    let index = words
        .iter()
        .position(|word| spellings.contains(&word.surface.as_str()) && word.deprel == "case")?;
    Some((&words[..index], &words[index], &words[index + 1..]))
}

fn argument<'a>(
    words: &'a [Atom],
    case: &Atom,
    head: &Atom,
    relation: &str,
) -> Option<Nominal<'a>> {
    let noun = read_nominal(words)?;
    (dasu_case_link(noun.head(), case) && linked(noun.head(), head, relation)).then_some(noun)
}

fn actor_text(actor: Nominal<'_>) -> String {
    match spelling(actor).as_str() {
        "作業員" => "工作人員".to_owned(),
        "担当者" => "負責人".to_owned(),
        _ => actor.text(),
    }
}

fn spelling(noun: Nominal<'_>) -> String {
    noun.0.iter().map(|word| word.surface.as_str()).collect()
}

fn clothes_line<'a>(words: &'a [Atom], ni: &Atom, verb: &Atom) -> Option<Option<&'a Atom>> {
    match words {
        [thing, dry, pole]
            if thing.surface == "物"
                && thing.upos == Some(Upos::Noun)
                && dry.lemma.as_deref() == Some("干す")
                && dry.surface == "干し"
                && dry.upos == Some(Upos::Verb)
                && pole.surface == "竿"
                && pole.upos == Some(Upos::Noun)
                && adjacent(thing, dry)
                && adjacent(dry, pole)
                && linked(thing, dry, "obl")
                && linked(dry, pole, "acl")
                && dasu_case_link(pole, ni)
                && linked(pole, verb, "obl") =>
        {
            // The compound is split into a noun, a verb and a noun. Only this
            // connected, adjacent compound can repair the misplaced object arc.
            Some(Some(dry))
        }
        _ => {
            let support = argument(words, ni, verb, "obl")?;
            (spelling(support) == "物干し竿").then_some(None)
        }
    }
}

fn hanging(subject: &[Atom], subject_case: &Atom, words: &[Atom], verb: &Atom) -> Option<String> {
    let (garment, wo, rest) = split_case(words, &["を"])?;
    let (ni, support) = rest.split_last()?;
    if ni.surface != "に" {
        return None;
    }
    let split_support = clothes_line(support, ni, verb)?;
    let [wet, past, coat] = garment else {
        return None;
    };
    if wet.lemma.as_deref() != Some("濡れる")
        || wet.upos != Some(Upos::Verb)
        || past.surface != "た"
        || past.upos != Some(Upos::Aux)
        || !linked(past, wet, "aux")
        || !adjacent(wet, past)
        || !adjacent(past, coat)
        || !linked(wet, coat, "acl")
        || coat.upos != Some(Upos::Noun)
        || !dasu_case_link(coat, wo)
        || !(linked(coat, verb, "obj") || split_support.is_some_and(|dry| linked(coat, dry, "obj")))
    {
        return None;
    }
    let coat = match coat.surface.as_str() {
        "上着" | "コート" => "外套",
        "シャツ" => "襯衫",
        _ => return None,
    };
    let actor = read_nominal(subject)?;
    if !dasu_case_link(actor.head(), subject_case)
        || !(linked(actor.head(), verb, "nsubj")
            || split_support.is_some() && linked(actor.head(), wet, "nsubj"))
    {
        return None;
    }
    Some(format!("{}把濕{coat}掛在曬衣桿上", actor_text(actor)))
}

fn reading_before(words: &[Atom], verb: &Atom) -> Option<&'static str> {
    let [reading, wo, begins, before, ni] = words else {
        return None;
    };
    (reading.surface == "読書"
        && reading.upos == Some(Upos::Noun)
        && wo.surface == "を"
        && begins.surface == "始める"
        && begins.upos == Some(Upos::Verb)
        && before.surface == "前"
        && before.upos == Some(Upos::Noun)
        && ni.surface == "に"
        && dasu_case_link(reading, wo)
        && linked(reading, begins, "obj")
        && linked(begins, before, "acl")
        && dasu_case_link(before, ni)
        && linked(before, verb, "obl"))
    .then_some("開始閱讀前，")
}

fn glasses(prefix: &[Atom], words: &[Atom], actor: &str, verb: &Atom) -> Option<RenderedFrame> {
    let (pause, timing) = prefix.split_last()?;
    let starts = timing.get(2)?;
    if !matches!(pause.surface.as_str(), "、" | "，") || !linked(pause, starts, "punct") {
        return None;
    }
    let before = reading_before(timing, verb)?;
    let (wo, glasses) = words.split_last()?;
    let glasses = argument(glasses, wo, verb, "obj")?;
    if wo.surface != "を" || !matches!(spelling(glasses).as_str(), "眼鏡" | "メガネ" | "めがね")
    {
        return None;
    }
    Some(RenderedFrame {
        text: format!("{before}{actor}戴上眼鏡"),
        cut: Some((pause.span, before.chars().count())),
    })
}

fn call(words: &[Atom], verb: &Atom, actor: &str) -> Option<String> {
    let (recipient, ni, rest) = split_case(words, &["に"])?;
    let (wo, phone) = rest.split_last()?;
    let recipient = argument(recipient, ni, verb, "obl")?;
    let phone = argument(phone, wo, verb, "obj")?;
    if wo.surface != "を" || spelling(phone) != "電話" {
        return None;
    }
    let recipient = match spelling(recipient).as_str() {
        "受付" => "櫃台".to_owned(),
        _ if recipient.person() => actor_text(recipient),
        _ => return None,
    };
    Some(format!("{actor}打電話給{recipient}"))
}

fn spent_time(words: &[Atom], verb: &Atom, actor: &str) -> Option<String> {
    let (task, ni, duration) = split_case(words, &["に"])?;
    let [count, hours] = duration else {
        return None;
    };
    if count.upos != Some(Upos::Num)
        || hours.surface != "時間"
        || !adjacent(count, hours)
        || !(linked(count, verb, "obl") && linked(hours, verb, "advmod")
            || linked(hours, verb, "obl")
                && (linked(count, hours, "nummod") || linked(count, hours, "compound")))
    {
        return None;
    }
    let count = match elapsed_number(&count.surface)? {
        2 => "兩".to_owned(),
        value => to_zh_numeral(value),
    };
    let inspection = argument(task, ni, verb, "obl")?;
    if inspection.head().surface != "点検" {
        return None;
    }
    let action = match task {
        [_] => "做檢查".to_owned(),
        [object @ .., no, _] if no.surface == "の" => {
            let object = read_nominal(object)?;
            let target = match spelling(object).as_str() {
                "機械" => "機器".to_owned(),
                _ => object.text(),
            };
            format!("檢查{target}")
        }
        _ => return None,
    };
    Some(format!("{actor}花了{count}小時{action}"))
}

pub(super) fn render(prefix: &[Atom], verb: &Atom, aspect: Aspect) -> Option<RenderedFrame> {
    if aspect != Aspect::Past {
        return None;
    }
    if let Some(index) = prefix
        .iter()
        .position(|word| matches!(word.surface.as_str(), "、" | "，"))
    {
        let (timing, main) = prefix.split_at(index + 1);
        let (subject, case, words) = split_case(main, &["は", "が"])?;
        let actor = actor_text(argument(subject, case, verb, "nsubj")?);
        return glasses(timing, words, &actor, verb);
    }
    let (subject, case, words) = split_case(prefix, &["は", "が"])?;
    let text = hanging(subject, case, words, verb).or_else(|| {
        let actor = actor_text(argument(subject, case, verb, "nsubj")?);
        call(words, verb, &actor).or_else(|| spent_time(words, verb, &actor))
    })?;
    Some(RenderedFrame { text, cut: None })
}

#[cfg(test)]
mod tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;
    use serde_json::Value;

    #[test]
    fn hanging_call_uses_proven_roles_and_excludes_homophones() {
        let bank: Value = serde_json::from_str(include_str!(
            "../../../tests/data/ja_zh/verb_hanging_call_dev_v1.json"
        ))
        .unwrap();
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let rows = bank["items"].as_array().unwrap();
        assert_eq!(rows.len(), 6);
        for row in rows {
            assert_eq!(row["heldout"], false);
            assert_eq!(row["visibility"], "development-visible");
            let id = row["id"].as_str().unwrap();
            let source = row["jp"].as_str().unwrap();
            let result =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            let expected = match id {
                "KAKERU-DEV-02" => "開始閱讀前，她戴上眼鏡。",
                "KAKERU-DEV-05" => "團隊給決勝戰的勝利賭了全て。",
                "KAKERU-DEV-06" => "選手連ゴール都奔馳了。",
                _ => row["zh_hant"].as_str().unwrap(),
            };
            assert_eq!(result.target_text, expected, "{id}");
            if id == "KAKERU-DEV-02" {
                let [before, after] = &result.clause_alignments[..] else {
                    panic!("expected two aligned clauses for {id}");
                };
                assert_eq!(
                    (before.source_span.start_char, before.source_span.end_char),
                    (0, 8)
                );
                assert_eq!(
                    (after.source_span.start_char, after.source_span.end_char),
                    (9, 19)
                );
                assert_eq!(
                    (before.target_span.start_char, before.target_span.end_char),
                    (0, 6)
                );
                assert_eq!(
                    (after.target_span.start_char, after.target_span.end_char),
                    (6, expected.chars().count())
                );
            }
        }
    }
}
