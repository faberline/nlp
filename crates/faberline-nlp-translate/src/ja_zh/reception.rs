//! The service counter and the act of accepting applications share a Japanese noun.
use super::{Atom, Upos, is_building_floor};

pub(super) const COUNTER: &str = "櫃檯";
pub(super) const SERVICE: &str = "受理";

pub(super) fn location_relation(inflection: &str) -> Option<&'static str> {
    match inflection {
        "" => Some("在"),
        "た" => Some("當時在"),
        "ない" => Some("不在"),
        "なかった" => Some("當時不在"),
        _ => None,
    }
}

fn reading(atoms: &[Atom], index: usize) -> Option<&'static str> {
    let noun = atoms.get(index)?;
    if noun.is_locked || noun.surface != "受付" || noun.upos != Some(Upos::Noun) {
        return None;
    }
    let head = noun.head?;
    let governor = atoms.iter().find(|a| a.doc_tok_idx == Some(head))?;
    let particle = atoms
        .get(index + 1)
        .filter(|a| super::dasu_case_link(noun, a))?;
    let case = particle.surface.as_str();
    let lemma = governor.lemma.as_deref().unwrap_or(&governor.surface);
    if matches!(noun.deprel.as_str(), "obj" | "nsubj" | "obl")
        && matches!(case, "を" | "は" | "が")
        && !governor.is_locked
        && governor.upos == Some(Upos::Verb)
        && matches!(lemma, "始まる" | "始める" | "終わる" | "終える")
    {
        return Some(SERVICE);
    }
    if noun.deprel == "nsubj"
        && matches!(case, "は" | "が")
        && (is_building_floor(governor) || governor.surface == "階")
    {
        return Some(COUNTER);
    }
    if noun.deprel == "nmod"
        && case == "の"
        && !governor.is_locked
        && governor.upos == Some(Upos::Noun)
    {
        if matches!(
            governor.surface.as_str(),
            "人" | "職員" | "係員" | "店員" | "先生" | "担当者"
        ) || (governor.surface == "者"
            && atoms.iter().any(|a| {
                a.surface == "担当"
                    && a.head == Some(head)
                    && a.deprel == "compound"
                    && a.span.end_char == governor.span.start_char
            }))
        {
            return Some(COUNTER);
        }
        // 前 can also be a time. Require a location linked to an existential verb.
        if matches!(governor.surface.as_str(), "前" | "横" | "隣" | "中" | "奥")
            && governor.deprel == "obl"
            && atoms
                .iter()
                .any(|a| a.head == Some(head) && a.surface == "に" && a.deprel == "case")
            && atoms.iter().any(|a| {
                a.doc_tok_idx == governor.head
                    && !a.is_locked
                    && a.upos == Some(Upos::Verb)
                    && matches!(a.lemma.as_deref(), Some("いる" | "ある"))
            })
        {
            return Some(COUNTER);
        }
    }
    if governor.is_locked || governor.upos != Some(Upos::Verb) {
        return None;
    }
    match (noun.deprel.as_str(), case, lemma) {
        ("obl" | "nsubj", "に", "いる" | "ある") => Some(COUNTER),
        (
            "obl",
            "に" | "へ" | "から" | "まで",
            "行く" | "来る" | "戻る" | "向かう" | "寄る" | "着く",
        ) => Some(COUNTER),
        (
            "obl" | "advmod",
            "で",
            "待つ"
            | "会う"
            | "支払う"
            | "払う"
            | "働く"
            | "受け取る"
            | "聞く"
            | "渡す"
            | "出す"
            | super::application_form::SUBMIT
            | super::application_form::CAN_SUBMIT,
        ) => Some(COUNTER),
        ("nsubj", "は" | "が", "受け付ける") => Some(COUNTER),
        _ => None,
    }
}

pub(super) fn normalize(atoms: &mut Vec<Atom>) {
    for i in (0..atoms.len()).rev() {
        if let Some((end, phrase)) = reception_compound(atoms, i) {
            atoms.splice(i..=end, [phrase]);
            continue;
        }
        if let Some(label) = reading(atoms, i) {
            // Keep the source surface for the case and floor rules.
            atoms[i].lemma = Some(label.to_owned());
            if label == SERVICE && i >= 2 {
                let modifier = &atoms[i - 2];
                let case = &atoms[i - 1];
                if !modifier.is_locked
                    && modifier.deprel == "nmod"
                    && modifier.head == atoms[i].doc_tok_idx
                    && case.surface == "の"
                    && super::dasu_case_link(modifier, case)
                    && case.span.end_char == atoms[i].span.start_char
                    && matches!(
                        modifier.surface.as_str(),
                        "申請" | "申し込み" | "申込み" | "申込"
                    )
                {
                    let term = super::translate_single_ja_term(
                        &modifier.surface,
                        modifier.lemma.as_deref(),
                        modifier.upos,
                    );
                    let start = modifier.span.start_char;
                    atoms[i].surface = format!("{term}{SERVICE}");
                    atoms[i].span.start_char = start;
                    atoms[i].is_locked = true;
                    atoms.drain(i - 2..i);
                    continue;
                }
            }
            // In 受付には先生がいる the analyzer can mark both nouns as subjects.
            // に locates the counter. が identifies the person at that place.
            if label == COUNTER
                && let Some([ni, wa]) = atoms.get(i + 1..i + 3)
                && ni.surface == "に"
                && wa.surface == "は"
                && !wa.is_locked
                && wa.deprel == "case"
                && wa.head == atoms[i].doc_tok_idx
                && ni.span.end_char == wa.span.start_char
                && atoms.iter().any(|verb| {
                    verb.doc_tok_idx == atoms[i].head
                        && !verb.is_locked
                        && verb.lemma.as_deref() == Some("いる")
                })
                && atoms[i + 3..].windows(2).any(|pair| {
                    let [whom, ga] = pair else { return false };
                    whom.deprel == "nsubj"
                        && whom.head == atoms[i].head
                        && ga.surface == "が"
                        && super::dasu_case_link(whom, ga)
                })
            {
                atoms[i + 1].span.end_char = wa.span.end_char;
                atoms[i].deprel = "obl".to_owned();
                atoms.remove(i + 2);
            }
        }
    }
    for i in (0..atoms.len()).rev() {
        if let Some((end, person)) = person_at_counter(atoms, i) {
            atoms.splice(i..=end, [person]);
        }
    }
}

fn reception_compound(atoms: &[Atom], i: usize) -> Option<(usize, Atom)> {
    let reception = atoms.get(i)?;
    if reception.is_locked || reception.surface != "受付" || reception.deprel != "compound" {
        return None;
    }
    let next = atoms.get(i + 1)?;
    let (end, label) = match next.surface.as_str() {
        "時間" => (i + 1, "受理時間"),
        "係員" => (i + 1, "櫃檯人員"),
        "係" if reception.head == next.doc_tok_idx => (i + 1, "櫃檯人員"),
        "係" if next.deprel == "compound"
            && next.head == reception.head
            && atoms.get(i + 2).is_some_and(|a| a.surface == "員") =>
        {
            (i + 2, "櫃檯人員")
        }
        _ => return None,
    };
    let head = &atoms[end];
    if reception.head.is_none()
        || reception.head != head.doc_tok_idx
        || atoms[i..=end]
            .iter()
            .any(|a| a.is_locked || a.upos != Some(Upos::Noun))
        || atoms[i..=end]
            .windows(2)
            .any(|w| w[0].span.end_char != w[1].span.start_char)
    {
        return None;
    }
    let mut phrase = head.clone();
    phrase.surface = label.to_owned();
    phrase.is_locked = true;
    phrase.span.start_char = reception.span.start_char;
    Some((end, phrase))
}

fn person_at_counter(atoms: &[Atom], i: usize) -> Option<(usize, Atom)> {
    let [counter, ni, be] = atoms.get(i..i + 3)? else {
        return None;
    };
    if counter.lemma.as_deref() != Some(COUNTER)
        || ni.surface != "に"
        || !super::dasu_case_link(counter, ni)
        || be.deprel != "acl"
        || be.upos != Some(Upos::Verb)
        || be.is_locked
        || be.lemma.as_deref().unwrap_or(&be.surface) != "いる"
        || counter.head != be.doc_tok_idx
    {
        return None;
    }
    let end = i
        + 3
        + atoms[i + 3..]
            .iter()
            .position(|a| a.doc_tok_idx == be.head)?;
    let noun = &atoms[end];
    if !super::is_person_like_atom(noun) && !matches!(noun.surface.as_str(), "職員" | "係員") {
        return None;
    }
    let phrase = &atoms[i..=end];
    if phrase
        .windows(2)
        .any(|w| w[0].span.end_char != w[1].span.start_char)
        || atoms[..i]
            .iter()
            .chain(&atoms[end + 1..])
            .any(|a| a.head == counter.doc_tok_idx || a.head == be.doc_tok_idx)
    {
        return None;
    }
    let inflection = &atoms[i + 3..end];
    if !inflection
        .iter()
        .all(|a| !a.is_locked && a.deprel == "aux" && a.head == be.doc_tok_idx)
    {
        return None;
    }
    let relation = location_relation(
        &inflection
            .iter()
            .map(|a| a.surface.as_str())
            .collect::<String>(),
    )?;
    let role = if noun.is_locked {
        noun.surface.clone()
    } else {
        super::translate_single_ja_term(&noun.surface, noun.lemma.as_deref(), noun.upos)
    };
    let mut person = noun.clone();
    person.surface = format!("{relation}{COUNTER}的{role}");
    person.is_locked = true;
    person.span.start_char = counter.span.start_char;
    Some((end, person))
}

// If the participant comes first (X が受付にいる), Chinese can keep X as the
// subject of 在. For 受付に X がいる the existing 櫃檯有 X order is natural.
pub(super) fn subject_before_counter(atoms: &[Atom]) -> bool {
    atoms.iter().enumerate().any(|(i, counter)| {
        counter.surface == "受付"
            && counter.lemma.as_deref() == Some(COUNTER)
            && atoms[..i]
                .iter()
                .any(|subject| subject.deprel == "nsubj" && subject.head == counter.head)
    })
}

#[cfg(test)]
mod tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

    #[test]
    fn counter_location_preserves_presence_and_acceptance_preserves_the_activity() {
        let translate = |source| {
            analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap()
            .target_text
        };
        for (source, expected) in [
            ("父が受付にいます。", "父親在櫃檯。"),
            ("父は受付にいません。", "父親不在櫃檯。"),
            ("受付には先生がいます。", "櫃檯有老師。"),
            ("申請の受付は終わりました。", "申請受理結束了。"),
            ("受付にいない人が来ました。", "不在櫃檯的人來了。"),
            ("受付は五時までです。", "受理到五點。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }
}
