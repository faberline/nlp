//! Preserve explicit destinations and separate nonliteral adversity from motion.

use super::{
    Aspect, Atom, Nominal, RenderedFrame, Upos, adjacent, dasu_case_link, linked, read_nominal,
};

pub(super) fn is_predicate(atom: &Atom) -> bool {
    atom.deprel == "root" && matches!(atom.lemma.as_deref(), Some("参る" | "まいる"))
}

fn day(atom: &Atom) -> Option<(&'static str, bool)> {
    match atom.surface.as_str() {
        "明日" => Some(("明天", true)),
        "明後日" => Some(("後天", true)),
        "今日" => Some(("今天", false)),
        _ => None,
    }
}

fn scheduled_time<'a>(atoms: &'a [Atom], verb: &Atom) -> Option<(&'a [Atom], String, bool)> {
    if let [subject @ .., date, no, period] = atoms
        && no.surface == "の"
        && dasu_case_link(date, no)
        && adjacent(no, period)
        && linked(period, verb, "advmod")
    {
        let time = match period.surface.as_str() {
            "午前" => "上午",
            "午後" => "下午",
            _ => return None,
        };
        let (date, future) = day(date)?;
        return Some((subject, format!("{date}{time}"), future));
    }
    let (date, subject) = atoms.split_last()?;
    if !linked(date, verb, "advmod") {
        return None;
    }
    let (date, future) = day(date)?;
    Some((subject, date.to_owned(), future))
}

fn subject(words: &[Atom], verb: &Atom) -> Option<String> {
    if words.is_empty() {
        return Some(String::new());
    }
    let (case, phrase) = words.split_last()?;
    let noun = read_nominal(phrase)?;
    if !matches!(case.surface.as_str(), "が" | "は")
        || !dasu_case_link(noun.head(), case)
        || !linked(noun.head(), verb, "nsubj")
    {
        return None;
    }
    if noun.key() == "担当者" && noun.0.iter().all(|word| word.upos == Some(Upos::Noun)) {
        Some("負責人".to_owned())
    } else {
        Some(noun.text())
    }
}

fn place_name(noun: Nominal<'_>) -> Option<String> {
    if noun.0.iter().all(|word| word.upos == Some(Upos::Noun)) {
        match noun.key().as_str() {
            "別館" => return Some("另一棟".to_owned()),
            "会議室" => return Some("會議室".to_owned()),
            _ => {}
        }
    }
    noun.place().then(|| noun.text())
}

fn location(words: &[Atom]) -> Option<(Nominal<'_>, String)> {
    if let Some(index) = words.iter().rposition(|atom| atom.surface == "の") {
        let (owner, rest) = words.split_at(index);
        let owner = read_nominal(owner)?;
        let (no, room) = rest.split_first()?;
        let place = read_nominal(room)?;
        if !dasu_case_link(owner.head(), no)
            || !adjacent(no, room.first()?)
            || !linked(owner.head(), place.head(), "nmod")
        {
            return None;
        }
        return Some((
            place,
            format!("{}的{}", place_name(owner)?, place_name(place)?),
        ));
    }
    let place = read_nominal(words)?;
    Some((place, place_name(place)?))
}

fn movement(words: &[Atom], verb: &Atom) -> Option<String> {
    let (case, goal) = words.split_last()?;
    if !matches!(case.surface.as_str(), "に" | "へ") || !adjacent(case, verb) {
        return None;
    }
    let (place, text) = match goal {
        [side] if side.upos == Some(Upos::Pron) => {
            let heading = match side.surface.as_str() {
                "そちら" => "到你那邊",
                "こちら" => "到這邊",
                "あちら" => "到那邊",
                _ => return None,
            };
            (side, heading.to_owned())
        }
        _ => {
            let (place, text) = location(goal)?;
            (place.head(), format!("前往{text}"))
        }
    };
    (dasu_case_link(place, case) && linked(place, verb, "obl")).then_some(text)
}

fn motion(prefix: &[Atom], verb: &Atom) -> Option<RenderedFrame> {
    if let Some(index) = prefix.iter().position(|atom| atom.surface == "、") {
        let (before, rest) = prefix.split_at(index);
        let (pause, goal) = rest.split_first()?;
        let (words, time, future) = scheduled_time(before, verb)?;
        let mut text = format!("{}{time}", subject(words, verb)?);
        let cut = Some((pause.span, text.chars().count()));
        if future {
            text.push('會');
        }
        text.push_str(&movement(goal, verb)?);
        return Some(RenderedFrame { text, cut });
    }
    // A complete motion phrase can also omit a scheduled time.
    let (actor, goal) = prefix
        .iter()
        .position(|atom| matches!(atom.surface.as_str(), "は" | "が") && atom.deprel == "case")
        .map_or((&[][..], prefix), |index| prefix.split_at(index + 1));
    Some(RenderedFrame {
        text: format!("{}{}", subject(actor, verb)?, movement(goal, verb)?),
        cut: None,
    })
}

fn repeated(words: &[Atom], action: &Atom) -> bool {
    let quantity = match words {
        [quantity] if matches!(quantity.surface.as_str(), "何度" | "何回") => quantity,
        [how, count]
            if how.surface == "何"
                && matches!(count.surface.as_str(), "度" | "回")
                && linked(how, count, "compound") =>
        {
            count
        }
        _ => return false,
    };
    linked(quantity, action, "obl") || linked(quantity, action, "advmod")
}

fn adversity(prefix: &[Atom], verb: &Atom) -> Option<RenderedFrame> {
    let [
        frequency @ ..,
        action,
        te,
        even,
        same,
        question,
        ga,
        continues,
        pause,
    ] = prefix
    else {
        return None;
    };
    // A repeated attempt that failed to stop the same questions supplies
    // positive evidence of being at a loss. A missing destination alone does not.
    if action.lemma.as_deref() != Some("説明する")
        || action.upos != Some(Upos::Verb)
        || !repeated(frequency, action)
        || te.surface != "て"
        || te.upos != Some(Upos::Sconj)
        || !linked(te, action, "mark")
        || even.surface != "も"
        || !linked(even, action, "case")
        || !(linked(action, continues, "nsubj") || linked(action, continues, "advcl"))
        || same.surface != "同じ"
        || !linked(same, question, "det")
        || question.surface != "質問"
        || ga.surface != "が"
        || !dasu_case_link(question, ga)
        || !linked(question, continues, "nsubj")
        || continues.lemma.as_deref() != Some("続く")
        || !linked(continues, verb, "advcl")
        || pause.surface != "、"
        || !prefix.windows(2).all(|pair| adjacent(&pair[0], &pair[1]))
        || !adjacent(pause, verb)
    {
        return None;
    }
    let before = "解釋好幾次後還是一直被問同樣的問題，";
    Some(RenderedFrame {
        text: format!("{before}真讓人沒轍"),
        cut: Some((pause.span, before.chars().count())),
    })
}

pub(super) fn render(prefix: &[Atom], verb: &Atom, aspect: Aspect) -> Option<RenderedFrame> {
    match aspect {
        Aspect::Plain => motion(prefix, verb),
        Aspect::Past => adversity(prefix, verb),
        Aspect::Continuing => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;
    use serde_json::Value;

    #[test]
    fn humble_motion_keeps_direction_and_adversity_without_changing_ordinary_motion() {
        let bank: Value = serde_json::from_str(include_str!(
            "../../../tests/data/ja_zh/verb_humble_motion_dev_v1.json"
        ))
        .unwrap();
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let rows = bank["items"].as_array().unwrap();
        assert_eq!(rows.len(), 5);
        for row in rows {
            assert_eq!(row["heldout"], false);
            assert_eq!(row["visibility"], "development-visible");
            let id = row["id"].as_str().unwrap();
            let source = row["jp"].as_str().unwrap();
            let result =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            let aligned_prefix = match id {
                "MAIRU-DEV-01" => "負責人明天",
                "MAIRU-DEV-02" => "明天下午",
                "MAIRU-DEV-05" => "解釋好幾次後還是一直被問同樣的問題，",
                "MAIRU-DEV-03" => {
                    assert_eq!(result.target_text, "擔當者明天，往別館的會議室去。");
                    continue;
                }
                "MAIRU-DEV-04" => {
                    assert_eq!(result.target_text, "明天下午，往這邊來。");
                    continue;
                }
                _ => panic!("unexpected adopted case: {id}"),
            };
            let expected = row["zh_hant"].as_str().unwrap();
            assert_eq!(result.target_text, expected, "{id}");
            let [before, after] = &result.clause_alignments[..] else {
                panic!("expected two aligned clauses for {id}");
            };
            assert_eq!(before.source_span.start_char, 0);
            assert_eq!(
                before.source_span.end_char + 1,
                after.source_span.start_char
            );
            assert_eq!(after.source_span.end_char, source.chars().count());
            assert_eq!(before.target_span.start_char, 0);
            assert_eq!(before.target_span.end_char, aligned_prefix.chars().count());
            assert_eq!(after.target_span.start_char, before.target_span.end_char);
            assert_eq!(after.target_span.end_char, expected.chars().count());
            assert!(expected.starts_with(aligned_prefix), "{id}");
        }
    }
}
