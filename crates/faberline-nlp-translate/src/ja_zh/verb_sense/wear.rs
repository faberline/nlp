//! Keep a physical wear result distinct from an earlier rubbing event.

use super::{
    Aspect, Atom, Nominal, RenderedFrame, SemanticClass, Upos, adjacent, dasu_case_link, linked,
    lookup_ja, read_nominal, translate_single_ja_term,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    WornThrough,
    Frayed,
}

fn outcome<'a>(prefix: &'a [Atom], verb: &Atom) -> Option<(Outcome, &'a [Atom])> {
    if verb.deprel != "root" || verb.upos != Some(Upos::Verb) {
        return None;
    }
    match verb.lemma.as_deref()? {
        "擦り切れる" | "すり切れる" | "すりきれる" => {
            Some((Outcome::WornThrough, prefix))
        }
        "ほつれる" => Some((Outcome::Frayed, prefix)),
        "切れる" => {
            // Mixed kana/kanji spelling can be split into す + り + 切れ.
            // Check the whole written compound and the observed links.
            let [before @ .., su, ri] = prefix else {
                return None;
            };
            (su.surface == "す"
                && su.upos == Some(Upos::Noun)
                && ri.surface == "り"
                && ri.upos == Some(Upos::Aux)
                && matches!(verb.surface.as_str(), "切れ" | "切れる")
                && adjacent(su, ri)
                && adjacent(ri, verb)
                && linked(su, verb, "obl")
                && linked(ri, su, "aux"))
            .then_some((Outcome::WornThrough, before))
        }
        _ => None,
    }
}

pub(super) fn is_predicate(prefix: &[Atom], verb: &Atom) -> bool {
    outcome(prefix, verb).is_some()
}

fn affected<'a>(words: &'a [Atom], verb: &Atom) -> Option<Nominal<'a>> {
    let (case, phrase) = words.split_last()?;
    let noun = read_nominal(phrase)?;
    (matches!(case.surface.as_str(), "が" | "は")
        && noun.head().upos == Some(Upos::Noun)
        && !noun.person()
        && dasu_case_link(noun.head(), case)
        && linked(noun.head(), verb, "nsubj"))
    .then_some(noun)
}

fn physical_text(noun: Nominal<'_>) -> String {
    match noun.0 {
        [head] => match head.surface.as_str() {
            "ロープ" => "繩子".into(),
            "袋" => "袋子".into(),
            "底" => "底部".into(),
            _ => noun.text(),
        },
        [owner @ .., no, head] if no.surface == "の" => match (owner, head.surface.as_str()) {
            ([shoe], "かかと") if shoe.surface == "靴" => "鞋跟".into(),
            ([surface], "糸") if surface.surface == "表面" => "表層纖維".into(),
            _ => format!(
                "{}的{}",
                physical_text(Nominal(owner)),
                physical_text(Nominal(std::slice::from_ref(head)))
            ),
        },
        [description, bag] if description.upos == Some(Upos::Adj) && bag.surface == "袋" => {
            let adjective = translate_single_ja_term(
                &description.surface,
                description.lemma.as_deref(),
                description.upos,
            );
            let adjective = adjective.trim_end_matches('的');
            let link = if adjective.chars().count() == 1 {
                ""
            } else {
                "的"
            };
            format!("{adjective}{link}袋子")
        }
        _ => noun.text(),
    }
}

fn earlier_rubbing(atoms: &[Atom], result: &Atom, pause: &Atom) -> Option<String> {
    let [arguments @ .., rub, te] = atoms else {
        return None;
    };
    // 擦れて is the intransitive te form even when the stem's lemma is
    // tagged 擦る. The transitive te form would be 擦って.
    if rub.surface != "擦れ"
        || !matches!(rub.lemma.as_deref(), Some("擦る" | "擦れる"))
        || rub.upos != Some(Upos::Verb)
        || !linked(rub, result, "advcl")
        || te.surface != "て"
        || te.upos != Some(Upos::Sconj)
        || !linked(te, rub, "mark")
        || !adjacent(rub, te)
        || !adjacent(te, pause)
    {
        return None;
    }
    match arguments {
        [] => Some("摩擦後，".into()),
        [how, times, mo]
            if how.surface == "何"
                && times.surface == "度"
                && mo.surface == "も"
                && adjacent(how, times)
                && linked(how, times, "compound")
                && dasu_case_link(times, mo)
                && times.head == rub.doc_tok_idx =>
        {
            Some("反覆摩擦後，".into())
        }
        _ => Some(format!(
            "{}摩擦後，",
            physical_text(affected(arguments, rub)?)
        )),
    }
}

pub(super) fn render(atoms: &[Atom], verb: &Atom, aspect: Aspect) -> Option<RenderedFrame> {
    let (outcome, prefix) = outcome(atoms, verb)?;
    let aspect_suffix = match aspect {
        Aspect::Plain => "",
        Aspect::Past => "了",
        Aspect::Continuing => return None,
    };
    let (mut text, cut, subject) = match prefix
        .iter()
        .position(|atom| matches!(atom.surface.as_str(), "、" | "，"))
    {
        Some(index) => {
            let pause = &prefix[index];
            let text = earlier_rubbing(&prefix[..index], verb, pause)?;
            let cut = Some((pause.span, text.chars().count()));
            (text, cut, &prefix[index + 1..])
        }
        None if outcome == Outcome::WornThrough => (String::new(), None, prefix),
        None => return None,
    };
    let subject = affected(subject, verb)?;
    let key = subject.key();
    // Keep figurative wear out of the material-damage route.
    if !matches!(
        key.as_str(),
        "底" | "かかと" | "踵" | "糸" | "布" | "布地" | "生地" | "紐" | "ロープ"
    ) && !lookup_ja(&key).is_some_and(|entry| entry.sem_class == Some(SemanticClass::Artifact))
    {
        return None;
    }
    text.push_str(&physical_text(subject));
    text.push_str(match outcome {
        Outcome::WornThrough => "磨破",
        Outcome::Frayed => "鬆脫",
    });
    text.push_str(aspect_suffix);
    Some(RenderedFrame { text, cut })
}
