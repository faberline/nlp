//! Keep the experiencer and the proposition that they lament.

use super::{
    Aspect, Atom, Nominal, RenderedFrame, Upos, adjacent, dasu_case_link, linked, read_nominal,
};

pub(super) fn is_predicate(atom: &Atom) -> bool {
    matches!(atom.lemma.as_deref(), Some("嘆く" | "なげく"))
        && matches!(atom.surface.as_str(), "嘆い" | "嘆く" | "なげい" | "なげく")
}

struct Event {
    text: String,
    cause_marker: &'static str,
    feeling: &'static str,
}

fn noun_case<'a>(atoms: &'a [Atom], marker: &str) -> Option<Nominal<'a>> {
    let (case, nominal) = atoms.split_last()?;
    let noun = read_nominal(nominal)?;
    (case.surface == marker && dasu_case_link(noun.head(), case)).then_some(noun)
}

fn past_event(action: &Atom, past: &Atom, clause: &Atom) -> bool {
    action.upos == Some(Upos::Verb)
        && past.upos == Some(Upos::Aux)
        && matches!(past.surface.as_str(), "た" | "だ")
        && linked(past, action, "aux")
        && linked(action, clause, "acl")
        && adjacent(action, past)
        && adjacent(past, clause)
}

fn portable_item(noun: Nominal<'_>) -> bool {
    matches!(
        noun.key().as_str(),
        "写真"
            | "財布"
            | "鍵"
            | "書類"
            | "手紙"
            | "切符"
            | "パスポート"
            | "かばん"
            | "ノート"
            | "本"
            | "傘"
            | "携帯電話"
            | "スマホ"
    )
}

// 大切な can be attached through another adjective. Its noun and all the
// adjectives still belong to the object of the one loss event.
fn lost_item(atoms: &[Atom], action: &Atom, requires_portable_item: bool) -> Option<String> {
    if let Some(noun) = noun_case(atoms, "を")
        && linked(noun.head(), action, "obj")
        && (!requires_portable_item || portable_item(noun))
    {
        return Some(noun.text());
    }
    let [important, na, nominal @ ..] = atoms else {
        return None;
    };
    let noun = noun_case(nominal, "を")?;
    let first = nominal.first()?;
    if important.surface != "大切"
        || important.upos != Some(Upos::Adj)
        || na.surface != "な"
        || na.upos != Some(Upos::Part)
        || !linked(na, important, "discourse")
        || !(linked(important, first, "advcl") || linked(important, noun.head(), "amod"))
        || !adjacent(important, na)
        || !adjacent(na, first)
        || !linked(noun.head(), action, "obj")
        || requires_portable_item && !portable_item(noun)
    {
        return None;
    }
    let item = match noun.0 {
        [adj, word] if adj.surface == "古い" && linked(adj, word, "amod") => {
            format!("舊{}", Nominal(std::slice::from_ref(word)).text())
        }
        _ => noun.text(),
    };
    Some(format!("珍貴的{item}"))
}

fn loss(atoms: &[Atom], clause: &Atom) -> Option<Event> {
    let [arguments @ .., action, past] = atoms else {
        return None;
    };
    if !past_event(action, past, clause) {
        return None;
    }
    // The analyzer can split なくした as なく + し + た. Only the exact
    // contiguous stem, its own past suffix and an explicit object support
    // the loss reading here.
    let item = if action
        .lemma
        .as_deref()
        .is_some_and(|lemma| matches!(lemma, "無くす" | "なくす" | "失くす"))
    {
        lost_item(arguments, action, action.lemma.as_deref() != Some("失くす"))?
    } else {
        let (stem, item) = arguments.split_last()?;
        if action.lemma.as_deref() != Some("する")
            || action.surface != "し"
            || !matches!(stem.surface.as_str(), "なく" | "無く" | "失く")
            || !linked(stem, action, "advcl")
            || !adjacent(stem, action)
        {
            return None;
        }
        lost_item(item, action, stem.surface != "失く")?
    };
    Some(Event {
        text: format!("遺失{item}"),
        cause_marker: "為",
        feeling: "難過",
    })
}

fn changed_state(atoms: &[Atom], clause: &Atom) -> Option<Event> {
    let [arguments @ .., action, past] = atoms else {
        return None;
    };
    let subject = noun_case(arguments, "が")?;
    if action.lemma.as_deref() != Some("変わる")
        || !past_event(action, past, clause)
        || !linked(subject.head(), action, "nsubj")
    {
        return None;
    }
    let name = if subject.0.len() == 1 && subject.key() == "予定" {
        "行程".to_owned()
    } else {
        subject.text()
    };
    Some(Event {
        text: format!("{name}變動"),
        cause_marker: "對",
        feeling: "遺憾",
    })
}

fn continuing_tree_removal(atoms: &[Atom], clause: &Atom) -> Option<Event> {
    let [prefix @ .., cut, passive, te, continuation] = atoms else {
        return None;
    };
    if cut.upos != Some(Upos::Verb)
        || !matches!(cut.lemma.as_deref(), Some("切る" | "伐る" | "切り倒す"))
        || !linked(cut, continuation, "advcl")
        || passive.upos != Some(Upos::Aux)
        || !matches!(passive.lemma.as_deref(), Some("れる" | "られる"))
        || !linked(passive, cut, "aux")
        || te.surface != "て"
        || te.upos != Some(Upos::Sconj)
        || !linked(te, cut, "mark")
        || continuation.upos != Some(Upos::Verb)
        || !matches!(continuation.surface.as_str(), "いく" | "行く")
        || !linked(continuation, clause, "acl")
        || ![cut, passive, te, continuation, clause]
            .windows(2)
            .all(|pair| adjacent(pair[0], pair[1]))
    {
        return None;
    }

    let (nominal, timing) = match prefix {
        [nominal @ .., repetition] if repetition.surface == "次々" => {
            if repetition.upos != Some(Upos::Adv) || !linked(repetition, cut, "advmod") {
                return None;
            }
            (nominal, "接連")
        }
        _ => (prefix, "正"),
    };
    let affected = noun_case(nominal, "が")?;
    let is_tree = match affected.key().as_str() {
        "樹木" | "木々" | "街路樹" => true,
        "木" => matches!(affected.0, [owner @ .., no, _] if no.surface == "の"
        && read_nominal(owner).is_some_and(|owner| {
            owner.place() || matches!(owner.key().as_str(), "森" | "林" | "庭")
        })),
        _ => false,
    };
    if !is_tree || !linked(affected.head(), cut, "nsubj") {
        return None;
    }
    Some(Event {
        text: format!("{}{timing}被砍掉", affected.text()),
        cause_marker: "對",
        feeling: "痛心",
    })
}

pub(super) fn render(atoms: &[Atom], lament: &Atom, aspect: Aspect) -> Option<RenderedFrame> {
    let time = match aspect {
        Aspect::Past => "曾",
        Aspect::Plain => "",
        Aspect::Continuing => return None,
    };
    let [prefix @ .., clause, wo] = atoms else {
        return None;
    };
    if clause.surface != "こと"
        || clause.upos != Some(Upos::Noun)
        || wo.surface != "を"
        || !dasu_case_link(clause, wo)
        || !linked(clause, lament, "obj")
    {
        return None;
    }

    let index = prefix
        .iter()
        .position(|atom| atom.surface == "は" && atom.deprel == "case")?;
    let experiencer = noun_case(&prefix[..=index], "は")?;
    let event_atoms = &prefix[index + 1..];
    if !experiencer.person()
        || !(linked(experiencer.head(), lament, "nsubj")
            || event_atoms.iter().any(|event_head| {
                matches!(event_head.upos, Some(Upos::Verb | Upos::Adj))
                    && linked(experiencer.head(), event_head, "nsubj")
            }))
    {
        return None;
    }
    let event = loss(event_atoms, clause)
        .or_else(|| continuing_tree_removal(event_atoms, clause))
        .or_else(|| changed_state(event_atoms, clause))?;
    Some(RenderedFrame {
        text: format!(
            "{}{time}{}{}感到{}",
            experiencer.text(),
            event.cause_marker,
            event.text,
            event.feeling
        ),
        cut: None,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::build_sentence_atoms;
    use super::super::{Document, render};
    use super::{Aspect, is_predicate, render as render_lament};
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::{analyze_document, model::Backend};
    use serde_json::Value;

    fn public_cases() -> [String; 3] {
        let bank: Value = serde_json::from_str(include_str!(
            "../../../tests/data/ja_zh/verb_peer_followup_dev_v1.json"
        ))
        .unwrap();
        assert_eq!(bank["status"], "development-visible");
        ["VP-N5-07", "VP-N5-08", "VP-N5-09"].map(|id| {
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
    fn lament_keeps_its_experiencer_and_the_event_they_react_to() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let expected = [
            "她曾為遺失珍貴的舊照片感到難過。",
            "他曾對公園的樹接連被砍掉感到痛心。",
            "她曾對行程變動感到遺憾。",
        ];
        for (source, target) in public_cases().into_iter().zip(expected) {
            let output =
                analyze_and_translate(&source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(output.target_text, target);
            assert_eq!(output.clause_alignments.len(), 1);
            assert_eq!(
                output.clause_alignments[0].target_span.end_char,
                target.chars().count()
            );
        }
    }

    #[test]
    fn an_unproven_embedded_event_or_caller_lock_falls_back() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let opts = crate::doc_options_for_translate(&options, &glossary);
        for (source, connected_term) in public_cases().into_iter().zip(["なく", "れ", "が"]) {
            let chars = source.chars().collect::<Vec<_>>();
            let mut doc = analyze_document(&source, &opts);
            let index = doc
                .tokens
                .iter()
                .position(|token| {
                    chars[token.span.start_char..token.span.end_char]
                        .iter()
                        .copied()
                        .eq(connected_term.chars())
                })
                .unwrap();
            let clauses = doc
                .segmentation
                .clauses
                .iter()
                .enumerate()
                .map(|(id, clause)| (id, clause.span))
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
            let (target, _) = render(&locked, &doc, &chars, 0, &clauses).unwrap();
            let verb_index = locked.iter().position(is_predicate).unwrap();
            let prefix = &locked[..verb_index];
            let predicate = &locked[verb_index];
            let plain = render_lament(prefix, predicate, Aspect::Plain).unwrap();
            let past = render_lament(prefix, predicate, Aspect::Past).unwrap();
            assert_eq!(past.text.replacen('曾', "", 1), plain.text);
            assert_ne!(past.text, plain.text);
            assert!(render_lament(prefix, predicate, Aspect::Continuing).is_none());

            locked[index].is_locked = true;
            assert!(render(&locked, &doc, &chars, 0, &clauses).is_none());

            let verb = doc
                .tokens
                .iter_mut()
                .find(|token| token.lemma.as_deref() == Some("嘆く"))
                .unwrap();
            verb.lemma = Some("なげく".into());
            assert_eq!(
                render(&atoms(&doc), &doc, &chars, 0, &clauses).unwrap().0,
                target
            );

            doc.syntax[index].head = None;
            assert!(render(&atoms(&doc), &doc, &chars, 0, &clauses).is_none());
        }
    }
}
