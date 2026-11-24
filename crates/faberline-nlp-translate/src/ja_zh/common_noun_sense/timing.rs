//! Small source-linked renderers for temporal arguments and temporal copulas.

use super::super::{Atom, Document, Upos, dasu_case_link, render_temporal_made_atom};
use super::{attached_to, boundary, finite_affirmative, like, noun_phrase, render as noun_reading};

/// Render the return-trip buying/counting frames and a noun phrase that starts
/// from a stated day.
pub(super) fn render(atoms: &[Atom], doc: &Document, sentence: usize) -> Option<String> {
    if atoms.iter().any(boundary) {
        return None;
    }
    return_trip_action(atoms).or_else(|| starting_day(atoms, doc, sentence))
}

fn return_trip_action(atoms: &[Atom]) -> Option<String> {
    let (root_index, verb) = atoms
        .iter()
        .enumerate()
        .find(|(_, atom)| atom.deprel == "root")?;
    let aspect = finite_affirmative(verb, &atoms[root_index + 1..])?;
    let args = &atoms[..root_index];
    let [return_trip, time_case, object, object_case] = args else {
        return None;
    };
    if !like(return_trip, &["帰り"])
        || !attached_to(return_trip, time_case, verb, "obl", "に")
        || !attached_to(object, object_case, verb, "obj", "を")
    {
        return None;
    }
    if return_trip.upos != Some(Upos::Noun) || return_trip.is_locked {
        return None;
    }
    let time = "回程";
    if like(verb, &["買う"]) {
        let item = match object.surface.as_str() {
            "アイス" => "冰",
            "アイスクリーム" => "冰淇淋",
            "ソフト"
                if noun_reading(&object.surface, object.lemma.as_deref()) == Some("霜淇淋") =>
            {
                "霜淇淋"
            }
            _ => return None,
        };
        return Some(format!("{time}路上買{aspect}{item}"));
    }
    if like(verb, &["数える"]) && like(object, &["数"]) {
        return Some(format!("{time}時數{aspect}數量"));
    }
    None
}

fn starting_day(atoms: &[Atom], doc: &Document, sentence: usize) -> Option<String> {
    let [subject, topic, day, from, copula] = atoms else {
        return None;
    };
    if subject.upos != Some(Upos::Noun)
        || subject.deprel != "nsubj"
        || subject.head != day.doc_tok_idx
        || subject.is_locked
        || topic.surface != "は"
        || topic.is_locked
        || !dasu_case_link(subject, topic)
        || day.deprel != "root"
        || day.is_locked
        || from.surface != "から"
        || from.is_locked
        || !dasu_case_link(day, from)
        || copula.surface != "です"
        || copula.upos != Some(Upos::Aux)
        || copula.deprel != "cop"
        || copula.is_locked
        || copula.head != day.doc_tok_idx
    {
        return None;
    }
    let subject = noun_phrase(std::slice::from_ref(subject), doc, sentence)?;
    let day = render_temporal_made_atom(day)?;
    Some(format!("{subject}從{day}開始"))
}
