//! Distinguish parts coming together from a matter settled through discussion.

use super::{Aspect, Atom, RenderedFrame, Upos, adjacent, dasu_case_link, linked, read_nominal};

pub(super) fn is_predicate(atom: &Atom) -> bool {
    matches!(atom.lemma.as_deref(), Some("まとまる" | "纏まる"))
}

fn assembly(atoms: &[Atom], verb: &Atom, past: &str) -> Option<String> {
    let [subject @ .., part, ga, one, no, form, ni] = atoms else {
        return None;
    };
    let part_name = match part.surface.as_str() {
        "部品" | "パーツ" => "零件",
        "破片" => "碎片",
        "断片" => "片段",
        _ => return None,
    };
    let result = match form.surface.as_str() {
        "形" => "一個整體",
        "形状" => "一種形狀",
        "全体" => "一個整體",
        _ => return None,
    };
    if part.upos != Some(Upos::Noun)
        || ga.surface != "が"
        || !dasu_case_link(part, ga)
        || !linked(part, verb, "nsubj")
        || !matches!(one.surface.as_str(), "一つ" | "ひとつ")
        || no.surface != "の"
        || !dasu_case_link(one, no)
        || !adjacent(no, form)
        || !linked(one, form, "nmod")
        || form.upos != Some(Upos::Noun)
        || ni.surface != "に"
        || !dasu_case_link(form, ni)
        || !linked(form, verb, "obl")
        || !adjacent(ni, verb)
    {
        return None;
    }
    let prior_state = match subject {
        [] => "",
        [scattered, was]
            if scattered.surface == "ばらばら"
                && matches!(scattered.upos, Some(Upos::Adv | Upos::Adj))
                && was.surface == "だった"
                && was.upos == Some(Upos::Aux)
                && linked(was, scattered, "aux")
                && (linked(scattered, verb, "advmod") || linked(scattered, part, "acl"))
                && adjacent(scattered, was)
                && adjacent(was, part) =>
        {
            "原本零散的"
        }
        _ => return None,
    };
    Some(format!("{prior_state}{part_name}組合成{result}{past}"))
}

fn discussion(event: &Atom, case: &Atom, verb: &Atom) -> Option<&'static str> {
    if case.surface != "で" || !adjacent(case, verb) {
        return None;
    }
    let name = match event.surface.as_str() {
        "話し合い" | "議論" | "協議" => "討論",
        "相談" => "商量",
        "交渉" => "協商",
        _ => return None,
    };
    let nominal =
        event.upos == Some(Upos::Noun) && dasu_case_link(event, case) && linked(event, verb, "obl");
    // The analyzer can read the noun 話し合い as the stem of 話し合う.
    let verbal_stem = event.lemma.as_deref() == Some("話し合う")
        && event.upos == Some(Upos::Verb)
        && case.upos == Some(Upos::Sconj)
        && linked(case, event, "mark")
        && linked(event, verb, "advcl")
        && adjacent(event, case);
    (nominal || verbal_stem).then_some(name)
}

fn matter(words: &[Atom], topic: &Atom, event: &Atom, verb: &Atom) -> Option<String> {
    let (name, head) = match words {
        [owner @ .., no, use_stem, way]
            if use_stem.surface == "使い"
                && way.surface == "方"
                && linked(use_stem, way, "compound")
                && adjacent(use_stem, way) =>
        {
            let owner = read_nominal(owner)?;
            if no.surface != "の"
                || !dasu_case_link(owner.head(), no)
                || !linked(owner.head(), way, "nmod")
                || !adjacent(no, use_stem)
            {
                return None;
            }
            let owner_name = match owner.0 {
                [place] if place.surface == "会場" => "場地".to_owned(),
                _ => owner.text(),
            };
            (format!("{owner_name}的使用方式"), way)
        }
        _ => {
            let noun = read_nominal(words)?;
            if !matches!(
                noun.key().as_str(),
                "使い方" | "方法" | "日程" | "条件" | "案" | "計画" | "方針" | "手順"
            ) {
                return None;
            }
            (noun.text(), noun.head())
        }
    };
    (dasu_case_link(head, topic) && (linked(head, verb, "nsubj") || linked(head, event, "nsubj")))
        .then_some(name)
}

fn agreement(atoms: &[Atom], verb: &Atom, past: &str) -> Option<RenderedFrame> {
    let topic_end = atoms
        .iter()
        .position(|atom| matches!(atom.surface.as_str(), "は" | "が") && atom.deprel == "case")?;
    let (words, rest) = atoms.split_at(topic_end);
    let (topic, after_topic) = rest.split_first()?;
    let (pause, after_topic) = match after_topic {
        [pause, rest @ ..] if matches!(pause.surface.as_str(), "、" | "，") => {
            (Some(pause), rest)
        }
        _ => (None, after_topic),
    };
    let [timing @ .., event, case] = after_topic else {
        return None;
    };
    let setting = discussion(event, case, verb)?;
    let mut text = matter(words, topic, event, verb)?;
    let time = match timing {
        [] => "",
        [day, no]
            if no.surface == "の"
                && dasu_case_link(day, no)
                && adjacent(no, event)
                && (linked(day, event, "nmod") || linked(day, verb, "nmod")) =>
        {
            match day.surface.as_str() {
                "今日" => "今天的",
                "昨日" => "昨天的",
                "先日" => "前幾天的",
                _ => return None,
            }
        }
        _ => return None,
    };
    let cut = pause.map(|pause| {
        text.push('，');
        (pause.span, text.chars().count())
    });
    text.push_str(&format!("在{time}{setting}中定案{past}"));
    Some(RenderedFrame { text, cut })
}

pub(super) fn render(atoms: &[Atom], verb: &Atom, aspect: Aspect) -> Option<RenderedFrame> {
    let past = match aspect {
        Aspect::Plain => "",
        Aspect::Past => "了",
        Aspect::Continuing => return None,
    };
    if let Some(text) = assembly(atoms, verb, past) {
        return Some(RenderedFrame { text, cut: None });
    }
    agreement(atoms, verb, past)
}

#[cfg(test)]
mod tests {
    use super::super::super::build_sentence_atoms;
    use super::super::{Document, render};
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::{analyze_document, model::Backend};
    use serde_json::Value;

    fn adopted_contrast() -> [String; 3] {
        let bank: Value = serde_json::from_str(include_str!(
            "../../../tests/data/ja_zh/verb_peer_followup_dev_v1.json"
        ))
        .unwrap();
        assert_eq!(bank["status"], "development-visible");
        ["VP-N5-13", "VP-N5-14", "VP-N5-15"].map(|id| {
            let row = bank["items"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["id"] == id)
                .unwrap();
            assert_eq!(row["heldout"], false);
            assert_eq!(row["visibility"], "development-visible");
            row["jp"].as_str().unwrap().to_owned()
        })
    }

    #[test]
    fn intransitive_cohesion_preserves_the_result_and_both_clauses() {
        let [assembly, agreement, _] = adopted_contrast();
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        for (source, phrases) in [
            (assembly, &["原本零散的零件組合成一個整體了。"][..]),
            (
                agreement,
                &["場地的使用方式，", "在今天的討論中定案了。"][..],
            ),
        ] {
            let result =
                analyze_and_translate(&source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(result.target_text, phrases.concat());
            assert_eq!(result.clause_alignments.len(), phrases.len());
            for (link, phrase) in result.clause_alignments.iter().zip(phrases) {
                let target: String = result
                    .target_text
                    .chars()
                    .skip(link.target_span.start_char)
                    .take(link.target_span.end_char - link.target_span.start_char)
                    .collect();
                assert_eq!(target, *phrase);
            }
        }
    }

    #[test]
    fn transitive_sorting_does_not_select_an_intransitive_sense() {
        let [_, _, source] = adopted_contrast();
        let chars: Vec<_> = source.chars().collect();
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let doc: Document = analyze_document(
            &source,
            &crate::doc_options_for_translate(&options, &glossary),
        );
        let atoms = build_sentence_atoms(
            0,
            &chars,
            &doc,
            &doc.tokens.iter().collect::<Vec<_>>(),
            &crate::lock::SpanLockTable::new(),
        );
        let clauses: Vec<_> = doc
            .segmentation
            .clauses
            .iter()
            .enumerate()
            .map(|(index, clause)| (index, clause.span))
            .collect();
        assert!(
            atoms
                .iter()
                .any(|atom| atom.lemma.as_deref() == Some("まとめる"))
        );
        assert!(render(&atoms, &doc, &chars, 0, &clauses).is_none());
    }
}
