//! Keep restrictions on the group that receives an item.

use super::{Atom, Upos};

pub(super) const EXCLUSION_OUTSIDE: &str = "__ja_group_exclusion_outside";

fn adjacent(atoms: &[&Atom]) -> bool {
    atoms
        .windows(2)
        .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
}

fn excludes_people(atoms: &[Atom], index: usize) -> bool {
    let Some(start) = index.checked_sub(1) else {
        return false;
    };
    let [group, from, outside, no, person, ..] = &atoms[start..] else {
        return false;
    };
    let phrase = [group, from, outside, no, person];
    phrase
        .iter()
        .all(|atom| !atom.is_locked && atom.doc_tok_idx.is_some())
        && adjacent(&phrase)
        && matches!(group.upos, Some(Upos::Noun | Upos::Propn))
        && group.deprel == "compound"
        && group.head == outside.doc_tok_idx
        && from.surface == "以"
        && from.deprel == "compound"
        && from.head == outside.doc_tok_idx
        && outside.surface == "外"
        && outside.deprel == "nmod"
        && outside.head == person.doc_tok_idx
        && no.surface == "の"
        && no.upos == Some(Upos::Adp)
        && no.deprel == "case"
        && no.head == outside.doc_tok_idx
        && person.surface == "人"
        && person.upos == Some(Upos::Noun)
}

pub(super) fn mark_exclusions(atoms: &mut [Atom]) {
    for index in 1..atoms.len().saturating_sub(3) {
        if excludes_people(atoms, index) {
            atoms[index + 1].lemma = Some(EXCLUSION_OUTSIDE.to_owned());
        }
    }
}

pub(super) fn distributes_only_to(atoms: &[Atom]) -> bool {
    let [.., recipient, only, ni, verb, polite] = atoms else {
        return false;
    };
    let phrase = [recipient, only, ni, verb, polite];
    phrase
        .iter()
        .all(|atom| !atom.is_locked && atom.doc_tok_idx.is_some())
        && adjacent(&phrase)
        && matches!(recipient.upos, Some(Upos::Noun | Upos::Propn))
        && recipient.deprel == "obl"
        && recipient.head == verb.doc_tok_idx
        && only.surface == "だけ"
        && only.upos == Some(Upos::Part)
        && only.deprel == "discourse"
        && only.head == recipient.doc_tok_idx
        && ni.surface == "に"
        && ni.upos == Some(Upos::Adp)
        && ni.deprel == "case"
        && ni.head == recipient.doc_tok_idx
        && verb.lemma.as_deref() == Some("配る")
        && verb.upos == Some(Upos::Verb)
        && verb.deprel == "root"
        && polite.surface == "ます"
        && polite.upos == Some(Upos::Aux)
        && polite.deprel == "aux"
        && polite.head == verb.doc_tok_idx
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::model::Backend;

    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};

    #[test]
    fn keeps_the_recipient_restriction_and_the_eligibility_question() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let translate = |source| {
            analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                .unwrap()
                .target_text
        };
        assert_eq!(
            translate("入場券は学生だけに配ります。学生以外の人は入場券をいただけませんか。"),
            "入場券只發給學生。學生以外的人不能得到入場券嗎？"
        );
        assert!(translate("建物の外の人は待ってください。").contains("外面"));
    }
}
