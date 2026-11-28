//! Preserve the two physical participants in an intransitive contact clause.

use super::{
    Aspect, Atom, Nominal, RenderedFrame, Upos, adjacent, dasu_case_link, linked, read_nominal,
};

pub(super) fn is_predicate(atom: &Atom) -> bool {
    atom.deprel == "root"
        && (atom.lemma.as_deref() == Some("擦れる")
            || (atom.surface == "擦れ" && atom.lemma.as_deref() == Some("擦る")))
}

pub(super) fn is_split_plain(verb: &Atom, ending: &Atom) -> bool {
    // The analyzer can split 擦れる as the homographic stem and an auxiliary.
    verb.surface == "擦れ"
        && is_predicate(verb)
        && ending.surface == "る"
        && ending.upos == Some(Upos::Aux)
        && adjacent(verb, ending)
        && linked(ending, verb, "aux")
}

fn participant_text(noun: Nominal<'_>) -> String {
    match noun.0 {
        [head] if head.surface == "袖" => "袖子".into(),
        [head] if head.surface == "紐" => "繩子".into(),
        [head] if head.surface == "箱" => "箱子".into(),
        [det, head] if head.surface == "紐" => match det.surface.as_str() {
            "この" => "這條繩子".into(),
            "その" | "あの" => "那條繩子".into(),
            _ => noun.text(),
        },
        [owner @ .., no, head] if no.surface == "の" => {
            let owner = participant_text(Nominal(owner));
            let part = participant_text(Nominal(std::slice::from_ref(head)));
            format!("{owner}的{part}")
        }
        _ => noun.text(),
    }
}

pub(super) fn render(atoms: &[Atom], verb: &Atom, aspect: Aspect) -> Option<RenderedFrame> {
    if aspect != Aspect::Plain {
        return None;
    }
    let (mut text, cut, body) = match atoms {
        [action, condition, pause, body @ ..] if matches!(pause.surface.as_str(), "、" | "，") => {
            // The case-like と belongs to the activity, not to a participant.
            if condition.surface != "と"
                || !dasu_case_link(action, condition)
                || action.upos != Some(Upos::Verb)
                || !linked(action, verb, "obl")
                || !adjacent(condition, pause)
            {
                return None;
            }
            let activity = match super::super::travel_manner(action)? {
                "走" => "走路",
                "跑" => "跑步",
                "游" => "游泳",
                manner => manner,
            };
            let leading = format!("{activity}時，");
            let cut = Some((pause.span, leading.chars().count()));
            (leading, cut, body)
        }
        _ => (String::new(), None, atoms),
    };

    let subject_end = body
        .iter()
        .position(|word| word.deprel == "case" && matches!(word.surface.as_str(), "は" | "が"))?;
    let subject = read_nominal(&body[..subject_end])?;
    let [subject_case, rest @ .., contact_case] = &body[subject_end..] else {
        return None;
    };
    let contact = read_nominal(rest)?;
    if subject.head().upos != Some(Upos::Noun)
        || subject.person()
        || contact.head().upos != Some(Upos::Noun)
        || !dasu_case_link(subject.head(), subject_case)
        || !linked(subject.head(), verb, "nsubj")
        || contact_case.surface != "に"
        || !dasu_case_link(contact.head(), contact_case)
        || !linked(contact.head(), verb, "obl")
        || !adjacent(contact_case, verb)
    {
        return None;
    }
    text.push_str(&participant_text(subject));
    text.push_str("會擦到");
    text.push_str(&participant_text(contact));
    Some(RenderedFrame { text, cut })
}

#[cfg(test)]
mod tests {
    use super::super::super::build_sentence_atoms;
    use super::super::render;
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::{analyze_document, model::Backend};
    use serde_json::Value;

    fn cases_from_bank() -> Vec<String> {
        let bank: Value = serde_json::from_str(include_str!(
            "../../../tests/data/ja_zh/verb_friction_dev_v1.json"
        ))
        .unwrap();
        assert_eq!(bank["status"], "development-visible");
        bank["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                assert_eq!(row["heldout"], false);
                assert_eq!(row["visibility"], "development-visible");
                row["jp"].as_str().unwrap().to_owned()
            })
            .collect()
    }

    fn assert_segments(cases: &[(usize, &[&str])]) {
        let source_cases = cases_from_bank();
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        for &(index, segments) in cases {
            let source = &source_cases[index];
            let result =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(result.target_text, segments.concat());
            assert_eq!(result.clause_alignments.len(), segments.len());
            for (link, segment) in result.clause_alignments.iter().zip(segments) {
                let target: String = result
                    .target_text
                    .chars()
                    .skip(link.target_span.start_char)
                    .take(link.target_span.end_char - link.target_span.start_char)
                    .collect();
                assert_eq!(target, *segment);
            }
        }
    }

    #[test]
    fn contact_keeps_both_participants_and_the_leading_condition() {
        assert_segments(&[
            (0, &["走路時，", "袖子會擦到書桌。"]),
            (1, &["這條繩子會擦到箱子的角。"]),
        ]);
    }

    #[test]
    fn wear_keeps_the_result_subject_and_optional_rubbing_event() {
        assert_segments(&[
            (2, &["反覆摩擦後，", "鞋跟磨破了。"]),
            (3, &["繩子摩擦後，", "表層纖維鬆脫了。"]),
            (5, &["舊袋子的底部磨破了。"]),
        ]);
    }

    #[test]
    fn contact_does_not_replace_wear_or_transitive_rubbing() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let case_bank = cases_from_bank();
        for source in &case_bank[2..] {
            let chars: Vec<_> = source.chars().collect();
            let doc = analyze_document(
                source,
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
                .map(|(i, clause)| (i, clause.span))
                .collect();
            assert!(!atoms.iter().any(super::is_predicate), "{source}");
            if source == &case_bank[4] {
                assert!(
                    render(&atoms, &doc, &chars, 0, &clauses).is_none(),
                    "{source}"
                );
            }
        }
    }
}
