//! The service counter and the act of accepting applications share a Japanese noun.
use super::{Atom, Upos, dasu_case_link, is_building_floor, return_endpoint};

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

fn is_direct_genitive(modifier: &Atom, particle: &Atom, noun: &Atom) -> bool {
    modifier.deprel == "nmod"
        && modifier.head == noun.doc_tok_idx
        && particle.surface == "の"
        && particle.span.end_char == noun.span.start_char
        && dasu_case_link(modifier, particle)
}

fn is_unanchored_nearby(atoms: &[Atom], modifier: &Atom) -> bool {
    !modifier.is_locked
        && modifier.upos == Some(Upos::Adj)
        && modifier.surface == "近く"
        && modifier.lemma.as_deref() == Some("近い")
        && modifier.doc_tok_idx.is_some()
        // A dependent can anchor the proximity in time, as in 締切近く.
        && !atoms.iter().any(|atom| {
            atom.head == modifier.doc_tok_idx
                && !matches!(atom.deprel.as_str(), "case" | "punct")
        })
}

// A nominal qualifier can mark an intake method or activity. An attached
// place genitive or unanchored 近くの can qualify a physical counter.
fn has_only_place_qualifiers(atoms: &[Atom], counter: &Atom) -> bool {
    atoms
        .iter()
        .enumerate()
        .filter(|(_, modifier)| {
            modifier.head == counter.doc_tok_idx
                && matches!(
                    modifier.deprel.as_str(),
                    "compound" | "nmod" | "amod" | "nummod"
                )
        })
        .all(|(index, modifier)| {
            atoms
                .get(index + 1)
                .is_some_and(|particle| is_direct_genitive(modifier, particle, counter))
                && (return_endpoint::is_place(modifier) || is_unanchored_nearby(atoms, modifier))
        })
}

fn is_relative_counter_location(atoms: &[Atom], counter: &Atom, no: &Atom, place: &Atom) -> bool {
    if !matches!(place.surface.as_str(), "前" | "横" | "隣" | "中" | "奥") || place.deprel != "obl"
    {
        return false;
    }
    let Some(predicate) = atoms
        .iter()
        .find(|a| a.doc_tok_idx == place.head && !a.is_locked && a.upos == Some(Upos::Verb))
    else {
        return false;
    };
    let lemma = predicate.lemma.as_deref().unwrap_or(&predicate.surface);
    atoms.iter().any(|case| {
        if case.head != place.doc_tok_idx || case.deprel != "case" {
            return false;
        }
        match (case.surface.as_str(), lemma) {
            ("に", "いる" | "ある") => {
                matches!(predicate.lemma.as_deref(), Some("いる" | "ある"))
            }
            // An attached position of waiting or standing is spatial. 前に
            // with a separate activity can still give a temporal relation.
            ("で", "待つ") | ("に", "立つ") => {
                no.span.end_char == place.span.start_char
                    && dasu_case_link(place, case)
                    && has_only_place_qualifiers(atoms, counter)
            }
            _ => false,
        }
    })
}

fn reading(atoms: &[Atom], index: usize) -> Option<&'static str> {
    let noun = atoms.get(index)?;
    if noun.is_locked || noun.surface != "受付" || noun.upos != Some(Upos::Noun) {
        return None;
    }
    let head = noun.head?;
    let governor = atoms.iter().find(|a| a.doc_tok_idx == Some(head))?;
    let particle = atoms.get(index + 1).filter(|a| dasu_case_link(noun, a))?;
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
        if is_relative_counter_location(atoms, noun, particle, governor) {
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
        // In 受付で申請を受け付ける, the place and the object belong to
        // the same acceptance predicate. Keep method-qualified 受付 separate.
        ("obl" | "nsubj", "で", "受け付ける")
            if has_only_place_qualifiers(atoms, noun)
                && atoms.windows(2).any(|pair| {
                    let [object, wo] = pair else { return false };
                    object.head == Some(head)
                        && object.deprel == "obj"
                        && object.upos == Some(Upos::Noun)
                        && wo.surface == "を"
                        && dasu_case_link(object, wo)
                }) =>
        {
            Some(COUNTER)
        }
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
            if label == COUNTER
                && i >= 2
                && is_direct_genitive(&atoms[i - 2], &atoms[i - 1], &atoms[i])
                && is_unanchored_nearby(atoms, &atoms[i - 2])
                && has_only_place_qualifiers(atoms, &atoms[i])
            {
                let modifier = &mut atoms[i - 2];
                modifier.surface = "附近".to_owned();
                modifier.lemma = None;
                modifier.upos = Some(Upos::Noun);
            }
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
    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
    };
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

    #[test]
    fn physical_counter_belongs_to_the_action_that_proves_its_location() {
        let translate = |source, glossary: &TranslationGlossary| {
            analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                glossary,
                None,
                Backend::Onnx,
            )
            .unwrap()
        };
        let empty = TranslationGlossary::new();
        for (source, expected) in [
            ("受付の前で待ちました。", "在櫃檯前面等待了。"),
            ("受付の前に立ちました。", "在櫃檯前面站了。"),
            ("受付で申請を受け付けます。", "在櫃檯受理申請。"),
            ("受付で申請を受け付けない。", "在櫃檯不受理申請。"),
            ("近くの受付の前で待ちました。", "在附近的櫃檯前面等待了。"),
            ("近くの受付で申請を受け付けます。", "在附近的櫃檯受理申請。"),
            (
                "近くの電話受付で申請を受け付けます。",
                "用近近地的電話受付受理申請。",
            ),
            (
                "近くのオンライン受付で申請を受け付けます。",
                "在近近地的オンライン受付受理申請。",
            ),
            (
                "締切近くの受付で申請を受け付けます。",
                "在近近地的受付締切受理申請。",
            ),
            (
                "近くの受付の前に申請しました。",
                "在近近地的受付前面申請了。",
            ),
            (
                "図書館の受付で申請を受け付けます。",
                "在圖書館的櫃檯受理申請。",
            ),
            ("受付の前に申請しました。", "在受付前面申請了。"),
            ("受付が始まる前に待ちました。", "在受理開始的前面等待了。"),
            ("申請の受付の前に立ちました。", "在申請的受付前面站了。"),
            (
                "オンライン受付で申請を受け付けます。",
                "在オンライン受付受理申請。",
            ),
            (
                "受付で案内して、別室では申請を受け付けます。",
                "在受付說明，在另外的室受理申請。",
            ),
        ] {
            assert_eq!(translate(source, &empty).target_text, expected, "{source}");
        }

        let mut glossary = TranslationGlossary::new();
        glossary.places.push(GlossaryEntry {
            canonical: "受付".to_owned(),
            surfaces: vec!["受付".to_owned()],
            targets: [(TargetVariant::ZhHant, "服務台".to_owned())]
                .into_iter()
                .collect(),
            ..Default::default()
        });
        let quoted = "「受付の前で待ちました」と一字ずつ紙に書きました。受付で申請を受け付けます。";
        let result = translate(quoted, &glossary);
        let text = &result.target_text;
        assert!(text.contains("「受付の前で待ちました」"), "{text}");
        assert!(text.ends_with("在服務台受理申請。"), "{text}");
        assert_eq!(result.glossary_hits.len(), 1);
        assert_eq!(result.glossary_hits[0].canonical, "受付");
        assert_eq!(result.glossary_hits[0].sentence, Some(1));
    }
}
