//! Separate physical routes, telephone connections, and human comprehension.

use super::{
    Aspect, Atom, Nominal, RenderedFrame, Upos, adjacent, aspect, dasu_case_link, linked,
    read_frame, read_nominal, translate_single_ja_term,
};

#[derive(Clone, Copy)]
enum Form {
    Plain,
    Past,
    State,
    Negative,
    NegativePast,
}

pub(super) fn is_predicate(verb: &Atom, tail: &[Atom]) -> bool {
    matches!(verb.lemma.as_deref(), Some("通じる" | "つうじる"))
        && (verb.deprel == "root" || matches!(aspect(verb, tail), Some(Aspect::Continuing)))
}

fn form(verb: &Atom, tail: &[Atom]) -> Option<Form> {
    if let Some(affirmative) = aspect(verb, tail) {
        return Some(match affirmative {
            Aspect::Plain => Form::Plain,
            Aspect::Past => Form::Past,
            Aspect::Continuing => Form::State,
        });
    }
    let auxiliary = |atom: &Atom| atom.upos == Some(Upos::Aux) && linked(atom, verb, "aux");
    match tail {
        [negative]
            if negative.surface == "ない" && adjacent(verb, negative) && auxiliary(negative) =>
        {
            Some(Form::Negative)
        }
        [negative, past]
            if negative.surface == "なかっ"
                && negative.lemma.as_deref() == Some("ない")
                && past.surface == "た"
                && adjacent(verb, negative)
                && adjacent(negative, past)
                && auxiliary(negative)
                && auxiliary(past) =>
        {
            Some(Form::NegativePast)
        }
        _ => None,
    }
}

fn demonstrative(atom: &Atom) -> Option<&'static str> {
    match (atom.upos, atom.surface.as_str()) {
        (Some(Upos::Det), "この") => Some("這"),
        (Some(Upos::Det), "その" | "あの") => Some("那"),
        _ => None,
    }
}

// A named compound must be one proven noun. Its determiner is retained
// separately so the caller can choose a Taiwan Chinese classifier.
fn named_noun(atoms: &[Atom]) -> Option<(Nominal<'_>, Option<&'static str>)> {
    let (first, rest) = atoms.split_first()?;
    let (meaning, words) = if let Some(meaning) = demonstrative(first) {
        (Some(meaning), rest)
    } else {
        (None, atoms)
    };
    let noun = read_nominal(words)?;
    if words.iter().any(|atom| atom.upos != Some(Upos::Noun))
        || meaning.is_some()
            && (!adjacent(first, words.first()?) || !linked(first, noun.head(), "det"))
    {
        return None;
    }
    Some((noun, meaning))
}

fn split_subject(prefix: &[Atom]) -> Option<(&[Atom], &Atom, &[Atom])> {
    let index = prefix.iter().position(|atom| {
        matches!(atom.surface.as_str(), "は" | "が")
            && atom.upos == Some(Upos::Adp)
            && atom.deprel == "case"
    })?;
    let (words, tail) = prefix.split_at(index);
    let (case, rest) = tail.split_first()?;
    Some((words, case, rest))
}

fn route(prefix: &[Atom], verb: &Atom, form: Form) -> Option<String> {
    if !matches!(form, Form::Plain | Form::State) {
        return None;
    }
    let (subject, topic, rest) = split_subject(prefix)?;
    let (road, demonstrative) = named_noun(subject)?;
    let (until, destination) = rest.split_last()?;
    let destination = read_nominal(destination)?;
    if !dasu_case_link(road.head(), topic)
        || !linked(road.head(), verb, "nsubj")
        || until.surface != "まで"
        || !dasu_case_link(destination.head(), until)
        || !linked(destination.head(), verb, "obl")
        || !destination.place()
    {
        return None;
    }
    let road = match road.key().as_str() {
        "小道" => "小路",
        "道" | "道路" => "路",
        "通路" => "通道",
        _ => return None,
    };
    let prefix = demonstrative.map_or_else(String::new, |word| format!("{word}條"));
    Some(format!("{prefix}{road}通到{}", destination.text()))
}

fn area<'a>(prefix: &'a [Atom], verb: &Atom) -> Option<(&'static str, &'a [Atom])> {
    let index = prefix.iter().position(|atom| atom.surface == "で")?;
    let [det, words @ ..] = &prefix[..index] else {
        return None;
    };
    let label = match demonstrative(det)? {
        "這" => "這一帶",
        _ => "那一帶",
    };
    let head = match words {
        [head] if matches!(head.surface.as_str(), "辺り" | "付近" | "近辺") => head,
        [head, suffix]
            if head.surface == "辺"
                && suffix.surface == "り"
                && suffix.upos == Some(Upos::Aux)
                && linked(suffix, head, "aux")
                && adjacent(head, suffix) =>
        {
            head
        }
        _ => return None,
    };
    let [case, topic, tail @ ..] = &prefix[index..] else {
        return None;
    };
    if head.upos != Some(Upos::Noun)
        || !linked(det, head, "det")
        || !adjacent(det, head)
        || !(linked(head, verb, "obl") || linked(head, verb, "nsubj"))
        || !(linked(case, head, "case") || linked(case, head, "mark"))
        || !adjacent(words.last()?, case)
        || topic.surface != "は"
        || topic.upos != Some(Upos::Adp)
        || !linked(topic, head, "case")
        || !adjacent(case, topic)
    {
        return None;
    }
    Some((label, tail))
}

fn telephone(prefix: &[Atom], verb: &Atom, form: Form) -> Option<String> {
    let (area, args) = area(prefix, verb).unwrap_or(("", prefix));
    let frame = read_frame(args, verb)?;
    if frame.object.is_some()
        || frame.direction.is_some()
        || frame.source.is_some()
        || frame.setting.is_some()
        || frame.topic.is_some()
        || frame.time.is_some()
        || frame.adverb.is_some()
        || frame.pause.is_some()
    {
        return None;
    }
    let (telephone, det) = named_noun(frame.subject?.0)?;
    let (label, classifier) = match telephone.key().as_str() {
        "電話" => ("電話", "通"),
        "携帯電話" => ("手機", "支"),
        _ => return None,
    };
    let status = match form {
        Form::Plain => "接通",
        Form::Past => "接通了",
        Form::Negative => "打不通",
        Form::NegativePast => "沒接通",
        Form::State => return None,
    };
    let receiver = match frame.goal {
        Some((receiver, "に"))
            if label == "電話"
                && (receiver.place()
                    || receiver.key() == "店"
                    || experiencing_person(receiver.0).is_some()) =>
        {
            let name = match receiver.0 {
                [shop] if shop.surface == "店" => "店裡".to_owned(),
                _ => receiver.text(),
            };
            format!("打給{name}的")
        }
        None => String::new(),
        _ => return None,
    };
    let det = det.map_or_else(String::new, |word| format!("{word}{classifier}"));
    Some(format!("{area}{det}{receiver}{label}{status}"))
}

fn content_text(content: Nominal<'_>) -> Option<String> {
    let word = match content.head().surface.as_str() {
        "説明" => "說明",
        "冗談" => "笑話",
        "話" => "話",
        "意味" => "意思",
        _ => return None,
    };
    match content.0 {
        [_] => Some(word.to_owned()),
        [det, _] if demonstrative(det).is_some() => {
            Some(format!("{}個{word}", demonstrative(det)?))
        }
        [description, _] if description.upos == Some(Upos::Adj) => {
            let text = translate_single_ja_term(
                &description.surface,
                description.lemma.as_deref(),
                description.upos,
            );
            let adjective = text.trim_end_matches('的');
            let adjective = if adjective == "短" {
                "簡短"
            } else {
                adjective
            };
            Some(format!("{adjective}的{word}"))
        }
        _ => None,
    }
}

fn experiencing_person(words: &[Atom]) -> Option<(&Atom, String)> {
    if let [stem, suffix] = words
        && stem.surface == "子"
        && stem.upos == Some(Upos::Noun)
        && suffix.surface == "ども"
        && suffix.upos == Some(Upos::Sconj)
        && linked(suffix, stem, "mark")
        && adjacent(stem, suffix)
    {
        return Some((stem, "小孩".to_owned()));
    }
    let person = read_nominal(words)?;
    match person.0 {
        [child] if matches!(child.surface.as_str(), "子ども" | "子供") => {
            Some((child, "小孩".to_owned()))
        }
        [pronoun] if pronoun.upos == Some(Upos::Pron) => matches!(
            pronoun.surface.as_str(),
            "私" | "わたし" | "僕" | "俺" | "あなた" | "君" | "彼" | "彼女" | "誰"
        )
        .then(|| (pronoun, person.text())),
        _ if person.head().upos != Some(Upos::Pron) && person.person() => {
            Some((person.head(), person.text()))
        }
        _ => None,
    }
}

fn comprehension(prefix: &[Atom], verb: &Atom, form: Form) -> Option<String> {
    let (words, topic, rest) = split_subject(prefix)?;
    let content = read_nominal(words)?;
    if !dasu_case_link(content.head(), topic) || !linked(content.head(), verb, "nsubj") {
        return None;
    }
    let (subject, extra) = match rest {
        [words @ .., focus] if matches!(focus.surface.as_str(), "も" | "は") => {
            (words, Some(focus))
        }
        _ => (rest, None),
    };
    let (case, person_words) = subject.split_last()?;
    let (person, name) = experiencing_person(person_words)?;
    if case.surface != "に"
        || case.upos != Some(Upos::Adp)
        || !linked(case, person, "case")
        || !adjacent(person_words.last()?, case)
        || !(linked(person, verb, "nsubj")
            || linked(person, verb, "obl")
            || linked(person, verb, "iobj"))
        || extra.is_some_and(|extra| {
            extra.upos != Some(Upos::Adp)
                || !linked(extra, person, "case")
                || !adjacent(case, extra)
        })
    {
        return None;
    }
    let reader = if extra.is_some_and(|extra| extra.surface == "も") {
        format!("連{name}也")
    } else {
        name
    };
    let meaning = match form {
        Form::Plain | Form::State => "理解",
        Form::Past => "理解了",
        Form::Negative => "不理解",
        Form::NegativePast => "沒有理解",
    };
    Some(format!("{reader}{meaning}{}", content_text(content)?))
}

pub(super) fn render(prefix: &[Atom], verb: &Atom, tail: &[Atom]) -> Option<RenderedFrame> {
    let form = form(verb, tail)?;
    let text = route(prefix, verb, form)
        .or_else(|| telephone(prefix, verb, form))
        .or_else(|| comprehension(prefix, verb, form))?;
    Some(RenderedFrame { text, cut: None })
}

#[cfg(test)]
mod tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;
    use serde_json::Value;

    #[test]
    fn connection_keeps_roles_tense_and_negation_without_capturing_transitive_passage() {
        let bank: Value = serde_json::from_str(include_str!(
            "../../../tests/data/ja_zh/verb_connection_dev_v1.json"
        ))
        .unwrap();
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let rows = bank["items"].as_array().unwrap();
        assert_eq!(rows.len(), 6);
        for row in rows {
            assert_eq!(row["heldout"], false);
            assert_eq!(row["visibility"], "development-visible");
            let source = row["jp"].as_str().unwrap();
            let result =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            let expected = match row["id"].as_str().unwrap() {
                "TSUJI-DEV-05" => "他沒有理解簡短的笑話。",
                // This is the pre-change output of the distinct 通す control.
                "TSUJI-DEV-06" => "驛員從改札嘴讓過了乘客。",
                _ => row["zh_hant"].as_str().unwrap(),
            };
            assert_eq!(result.target_text, expected, "{}", row["id"]);
            let [alignment] = &result.clause_alignments[..] else {
                panic!("one clause expected for {}", row["id"]);
            };
            assert_eq!(alignment.source_span.start_char, 0);
            assert_eq!(alignment.source_span.end_char, source.chars().count());
            assert_eq!(alignment.target_span.start_char, 0);
            assert_eq!(alignment.target_span.end_char, expected.chars().count());
        }
    }
}
