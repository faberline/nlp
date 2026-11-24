//! Render sports activities and talk about a sport from their local syntax.

use super::{
    Atom, Document, Upos, attached_to, dasu_case_link, finite_affirmative, like, noun_phrase,
    render as noun_reading,
};

/// Render a source-linked sports activity or a statement about baseball.
///
/// The caller may pass the sentence-final punctuation. Every other atom must
/// belong to this sentence and map to its original document token.
pub(super) fn render(atoms: &[Atom], doc: &Document, sentence: usize) -> Option<String> {
    let body = sentence_body(atoms, doc, sentence)?;
    timed_softball_game(body)
        .or_else(|| place_softball_play(body, doc, sentence))
        .or_else(|| baseball_talk(body))
}

fn sentence_body<'a>(atoms: &'a [Atom], doc: &Document, sentence: usize) -> Option<&'a [Atom]> {
    let span = doc.segmentation.sentences.get(sentence)?.span;
    if atoms.iter().any(|atom| {
        atom.is_locked
            || atom.doc_tok_idx.is_none()
            || atom.span.start_char < span.start_char
            || atom.span.end_char > span.end_char
            || atom.doc_tok_idx.is_some_and(|index| {
                doc.tokens
                    .get(index)
                    .is_none_or(|token| token.span != atom.span || token.upos != atom.upos)
            })
    }) {
        return None;
    }

    match atoms.last() {
        Some(last) if matches!(last.surface.as_str(), "。" | "！" | "？") => {
            Some(&atoms[..atoms.len() - 1])
        }
        _ => Some(atoms),
    }
}

fn timed_softball_game(atoms: &[Atom]) -> Option<String> {
    let (time, time_case, comma, rest) = match atoms {
        [time, time_case, comma, rest @ ..] if comma.surface == "、" => {
            (time, time_case, Some(comma), rest)
        }
        [time, time_case, rest @ ..] => (time, time_case, None, rest),
        _ => return None,
    };

    if !like(time, &["昼休み"])
        || time.upos != Some(Upos::Noun)
        || !matches!(time.deprel.as_str(), "obl" | "advmod")
        || time_case.surface != "に"
        || time_case.deprel != "case"
        || !matches!(time_case.upos, Some(Upos::Adp))
        || time_case.head != time.doc_tok_idx
    {
        return None;
    }

    let [
        everyone,
        everyone_case,
        sport,
        genitive,
        match_word,
        object_case,
        verb,
        tail @ ..,
    ] = rest
    else {
        return None;
    };

    if !like(verb, &["する"])
        || verb.deprel != "root"
        || !like(everyone, &["みんな"])
        || !matches!(everyone.upos, Some(Upos::Pron | Upos::Noun))
        || !attached_to(everyone, everyone_case, verb, "obl", "で")
        || sport.head != match_word.doc_tok_idx
        || sport.deprel != "nmod"
        || genitive.surface != "の"
        || !dasu_case_link(sport, genitive)
        || !like(match_word, &["試合"])
        || match_word.deprel != "obj"
        || !matches!(
            noun_reading(&sport.surface, sport.lemma.as_deref()),
            Some("壘球")
        )
        || !attached_to(match_word, object_case, verb, "obj", "を")
        || !attached_to(time, time_case, verb, "obl", "に")
    {
        return None;
    }
    let aspect = finite_affirmative(verb, tail)?;
    let comma = comma.map_or("", |_| "，");
    Some(format!("午休時{comma}大家一起打{aspect}壘球比賽"))
}

fn place_softball_play(atoms: &[Atom], doc: &Document, sentence: usize) -> Option<String> {
    let [place, place_case, sport, sport_case, verb, tail @ ..] = atoms else {
        return None;
    };
    if !like(place, &["公園", "校庭", "球場", "グラウンド", "運動場"])
        || !matches!(place.upos, Some(Upos::Noun | Upos::Propn))
        || !attached_to(place, place_case, verb, "obl", "で")
        || !like(sport, &["ソフト"])
        || !matches!(
            noun_reading(&sport.surface, sport.lemma.as_deref()),
            Some("壘球")
        )
        || !attached_to(sport, sport_case, verb, "obj", "を")
        || !like(verb, &["する"])
        || verb.deprel != "root"
    {
        return None;
    }
    let aspect = finite_affirmative(verb, tail)?;
    let place = noun_phrase(std::slice::from_ref(place), doc, sentence)?;
    Some(format!("在{place}打{aspect}壘球"))
}

fn baseball_talk(atoms: &[Atom]) -> Option<String> {
    let [baseball, genitive, topic, object_case, verb, tail @ ..] = atoms else {
        return None;
    };
    if !like(baseball, &["野球"])
        || baseball.deprel != "nmod"
        || baseball.head != topic.doc_tok_idx
        || genitive.surface != "の"
        || !dasu_case_link(baseball, genitive)
        || !like(topic, &["話"])
        || !attached_to(topic, object_case, verb, "obj", "を")
        || !like(verb, &["する"])
    {
        return None;
    }
    let aspect = finite_affirmative(verb, tail)?;
    let baseball = match baseball.surface.as_str() {
        "野球" => "棒球",
        _ => return None,
    };
    Some(format!("聊{aspect}{baseball}"))
}
