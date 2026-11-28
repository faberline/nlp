//! Keep the affected participant and the cause of an intransitive fall.

use super::{
    Aspect, Atom, Nominal, RenderedFrame, Upos, adjacent, dasu_case_link, linked, read_nominal,
};

pub(super) fn is_predicate(atom: &Atom) -> bool {
    matches!(atom.lemma.as_deref(), Some("倒れる" | "たおれる"))
        && matches!(
            atom.surface.as_str(),
            "倒れ" | "倒れる" | "たおれ" | "たおれる"
        )
}

struct Argument<'a> {
    owner: Option<Nominal<'a>>,
    noun: Nominal<'a>,
    case: &'a Atom,
}

impl Argument<'_> {
    fn text(&self) -> String {
        let noun = match self.noun.key().as_str() {
            "看板" => "招牌".to_owned(),
            _ => self.noun.text(),
        };
        match self.owner {
            Some(owner) => format!("{}的{noun}", owner.text()),
            None => noun,
        }
    }
}

// Accept a complete nominal argument, including a possessive with a
// multi-token head. A case marker inside the possessive does not end it.
fn take_argument(atoms: &[Atom]) -> Option<(Argument<'_>, &[Atom])> {
    let index = atoms.iter().position(|atom| {
        atom.upos == Some(Upos::Adp) && atom.deprel == "case" && atom.surface != "の"
    })?;
    let case = &atoms[index];
    let words = &atoms[..index];
    let (owner, noun) = match words.iter().rposition(|atom| atom.surface == "の") {
        Some(index) => {
            let owner = read_nominal(&words[..index])?;
            let noun = read_nominal(&words[index + 1..])?;
            let no = &words[index];
            if !dasu_case_link(owner.head(), no)
                || !linked(owner.head(), noun.head(), "nmod")
                || !adjacent(no, &words[index + 1])
            {
                return None;
            }
            (Some(owner), noun)
        }
        None => (None, read_nominal(words)?),
    };
    dasu_case_link(noun.head(), case)
        .then_some((Argument { owner, noun, case }, &atoms[index + 1..]))
}

fn wind_topples(atoms: &[Atom], fall: &Atom, past: &str) -> Option<String> {
    let (cause, rest) = take_argument(atoms)?;
    let (affected, tail) = take_argument(rest)?;
    if !tail.is_empty()
        || cause.case.surface != "で"
        || !linked(cause.noun.head(), fall, "obl")
        || !matches!(cause.noun.key().as_str(), "風" | "強風" | "突風")
        || !matches!(affected.case.surface.as_str(), "が" | "は")
        || !linked(affected.noun.head(), fall, "nsubj")
        || !matches!(
            affected.noun.key().as_str(),
            "看板" | "木" | "棚" | "椅子" | "自転車" | "旗" | "箱"
        )
    {
        return None;
    }
    Some(format!("{}被{}吹倒{past}", affected.text(), cause.text()))
}

fn slipping(atoms: &[Atom], fall: &Atom, past: &str) -> Option<String> {
    let [arguments @ .., slip, suffix, te] = atoms else {
        return None;
    };
    if slip.lemma.as_deref() != Some("滑る")
        || slip.upos != Some(Upos::Verb)
        || !linked(slip, fall, "advcl")
        || suffix.surface != "せ"
        || suffix.lemma.as_deref() != Some("せる")
        || suffix.upos != Some(Upos::Aux)
        || !linked(suffix, slip, "aux")
        || te.surface != "て"
        || te.upos != Some(Upos::Sconj)
        || !linked(te, slip, "mark")
        || !adjacent(slip, suffix)
        || !adjacent(suffix, te)
        || !adjacent(te, fall)
    {
        return None;
    }

    let (person, rest) = take_argument(arguments)?;
    let (place, rest) = take_argument(rest)?;
    let (foot, tail) = take_argument(rest)?;
    if !tail.is_empty()
        || !person.noun.person()
        || !matches!(person.case.surface.as_str(), "は" | "が")
        || !linked(person.noun.head(), slip, "nsubj")
        || place.case.surface != "で"
        || !linked(place.noun.head(), slip, "obl")
        || !(place.noun.place()
            || matches!(
                place.noun.key().as_str(),
                "階段" | "床" | "歩道" | "道路" | "路面" | "廊下" | "通路"
            ))
        || foot.owner.is_some()
        || foot.noun.0.len() != 1
        || !matches!(foot.noun.key().as_str(), "足" | "脚")
        || foot.case.surface != "を"
        || !linked(foot.noun.head(), slip, "obj")
    {
        return None;
    }
    Some(format!("{}在{}滑倒{past}", person.text(), place.text()))
}

fn financial_failure(atoms: &[Atom], fall: &Atom, past: &str) -> Option<String> {
    let [arguments @ .., continue_, without] = atoms else {
        return None;
    };
    if continue_.lemma.as_deref() != Some("続く")
        || continue_.upos != Some(Upos::Verb)
        || !linked(continue_, fall, "advcl")
        || without.surface != "ず"
        || without.upos != Some(Upos::Aux)
        || !linked(without, continue_, "aux")
        || !adjacent(continue_, without)
        || !adjacent(without, fall)
    {
        return None;
    }

    let (company, rest) = take_argument(arguments)?;
    let (funding, tail) = take_argument(rest)?;
    if !tail.is_empty()
        || !matches!(
            company.noun.key().as_str(),
            "会社" | "企業" | "商店" | "団体"
        )
        || company.case.surface != "は"
        || !(linked(company.noun.head(), fall, "nsubj")
            || linked(company.noun.head(), continue_, "nsubj"))
        || !matches!(funding.noun.key().as_str(), "資金" | "運転資金")
        || funding.case.surface != "が"
        || !linked(funding.noun.head(), continue_, "nsubj")
    {
        return None;
    }
    let company_text = match company.noun.0 {
        [det, _] if matches!(det.surface.as_str(), "この" | "その" | "あの") => {
            let deictic = if det.surface == "この" {
                "這"
            } else {
                "那"
            };
            format!("{deictic}家{}", Nominal(&company.noun.0[1..]).text())
        }
        _ => company.text(),
    };
    Some(format!(
        "{company_text}因{}難以為繼而倒閉{past}",
        funding.text()
    ))
}

// A reading pause can separate a time-setting clause after an economic
// downturn from the business event. Preserve each clause's target range.
fn after_downturn(atoms: &[Atom]) -> Option<(String, &Atom, &[Atom])> {
    let i = atoms
        .iter()
        .position(|atom| matches!(atom.surface.as_str(), "、" | "，"))?;
    let [period @ .., no, after] = &atoms[..i] else {
        return None;
    };
    let pause = &atoms[i];
    let rest = &atoms[i + 1..];
    let downturn = read_nominal(period)?;
    let condition = match downturn.key().as_str() {
        "不況" | "不景気" => "不景氣",
        "景気後退" => "景氣衰退",
        _ => return None,
    };
    let duration = match period {
        [adj, _] if adj.surface == "長い" && linked(adj, downturn.head(), "amod") => "長期",
        [adj, _] if adj.surface == "短い" && linked(adj, downturn.head(), "amod") => "短期",
        _ if period[0].upos == Some(Upos::Noun) => "",
        _ => return None,
    };
    // The source analyzer can attach の to the duration adjective in a
    // contiguous time phrase. Keep both words only when it owns the noun.
    if no.surface != "の"
        || no.upos != Some(Upos::Adp)
        || no.deprel != "case"
        || !(no.head == downturn.head().doc_tok_idx
            || !duration.is_empty() && no.head == period[0].doc_tok_idx)
        || after.surface != "あと"
        || after.upos != Some(Upos::Adv)
        || ![downturn.head(), no, after, pause]
            .windows(2)
            .all(|p| adjacent(p[0], p[1]))
    {
        return None;
    }
    let [.., continue_, _] = rest else {
        return None;
    };
    if !linked(after, continue_, "advmod") {
        return None;
    }
    Some((format!("{duration}{condition}後，"), pause, rest))
}

pub(super) fn render(atoms: &[Atom], fall: &Atom, aspect: Aspect) -> Option<RenderedFrame> {
    let past = match aspect {
        Aspect::Past => "了",
        Aspect::Plain => "",
        Aspect::Continuing => return None,
    };

    if let Some((time, pause, rest)) = after_downturn(atoms) {
        let main = financial_failure(rest, fall, past)?;
        let cut = Some((pause.span, time.chars().count()));
        return Some(RenderedFrame {
            text: time + &main,
            cut,
        });
    }
    let text = wind_topples(atoms, fall, past)
        .or_else(|| slipping(atoms, fall, past))
        .or_else(|| financial_failure(atoms, fall, past))?;
    Some(RenderedFrame { text, cut: None })
}

#[cfg(test)]
mod tests {
    use super::super::super::build_sentence_atoms;
    use super::super::{Document, render};
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::{analyze_document, model::Backend};
    use serde_json::Value;

    fn public_cases() -> [String; 3] {
        let bank: Value = serde_json::from_str(include_str!(
            "../../../tests/data/ja_zh/verb_peer_followup_dev_v1.json"
        ))
        .unwrap();
        assert_eq!(bank["status"], "development-visible");
        ["VP-N5-19", "VP-N5-20", "VP-N5-21"].map(|id| {
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
    fn source_roles_choose_object_human_and_business_readings() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let expected = [
            ("被強風吹倒", 1),
            ("樓梯滑倒", 1),
            ("因資金難以為繼而倒閉", 2),
        ];
        for (source, (reading, clauses)) in public_cases().into_iter().zip(expected) {
            let result =
                analyze_and_translate(&source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert!(
                result.target_text.contains(reading),
                "{}",
                result.target_text
            );
            assert_eq!(result.clause_alignments.len(), clauses);
            if let [left, right] = result.clause_alignments.as_slice() {
                assert_eq!(left.target_span.end_char, right.target_span.start_char);
                assert!(left.target_span.end_char > left.target_span.start_char);
                assert!(right.target_span.end_char > right.target_span.start_char);
            }
        }
    }

    #[test]
    fn unrelated_causes_and_opaque_arguments_fall_back() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let doc_options = crate::doc_options_for_translate(&options, &glossary);
        for (source, boundary) in public_cases().into_iter().zip(["で", "せ", "ず"]) {
            let chars = source.chars().collect::<Vec<_>>();
            let mut doc = analyze_document(&source, &doc_options);
            let atom_index = doc
                .tokens
                .iter()
                .position(|token| {
                    chars[token.span.start_char..token.span.end_char]
                        .iter()
                        .copied()
                        .eq(boundary.chars())
                })
                .unwrap();
            let clauses = doc
                .segmentation
                .clauses
                .iter()
                .enumerate()
                .map(|(index, clause)| (index, clause.span))
                .collect::<Vec<_>>();
            let atoms = |doc: &Document| {
                build_sentence_atoms(
                    0,
                    &chars,
                    doc,
                    &doc.tokens.iter().collect::<Vec<_>>(),
                    &crate::lock::SpanLockTable::new(),
                )
            };
            let mut locked = atoms(&doc);
            let (original, _) = render(&locked, &doc, &chars, 0, &clauses).unwrap();
            locked[atom_index].is_locked = true;
            assert!(render(&locked, &doc, &chars, 0, &clauses).is_none());

            let verb = doc
                .tokens
                .iter_mut()
                .find(|token| token.lemma.as_deref() == Some("倒れる"))
                .unwrap();
            verb.lemma = Some("たおれる".into());
            let (kana_lemma, _) = render(&atoms(&doc), &doc, &chars, 0, &clauses).unwrap();
            assert_eq!(kana_lemma, original);

            doc.syntax[atom_index].head = None;
            assert!(render(&atoms(&doc), &doc, &chars, 0, &clauses).is_none());
        }
    }
}
