//! Recognize a meal followed by a completed trip.

use faberline_nlp::doc::SyntaxToken;

use super::{Atom, Document, LOCKED_OBJECT_ATOM, LOCKED_PLACE_ATOM, Token, Upos, dasu_case_link};

fn source_identity<'a>(atom: &Atom, doc: &'a Document) -> Option<(&'a SyntaxToken, &'a Token)> {
    let id = atom.doc_tok_idx?;
    let token = doc.tokens.get(id)?;
    let link = doc.syntax.get(id)?;
    (token.span == atom.span
        && link.token == id
        && link.span == token.span
        && link.sentence == token.sentence
        && link.head == atom.head
        && link.deprel == atom.deprel)
        .then_some((link, token))
}

fn source_link<'a>(atom: &Atom, doc: &'a Document) -> Option<&'a SyntaxToken> {
    let (link, token) = source_identity(atom, doc)?;
    (!atom.is_locked && token.upos == atom.upos).then_some(link)
}

fn linked_noun(noun: &Atom, predicate: &SyntaxToken, allowed_lock: &str, doc: &Document) -> bool {
    source_identity(noun, doc).is_some_and(|(link, token)| {
        let trusted_kind = if noun.is_locked {
            noun.lemma.as_deref() == Some(allowed_lock)
        } else {
            noun.upos == token.upos
        };
        trusted_kind
            && matches!(token.upos, Some(Upos::Noun | Upos::Propn))
            && link.head == Some(predicate.token)
            && link.sentence == predicate.sentence
    })
}

fn only_adverbs(atoms: &[Atom], predicate: &SyntaxToken, doc: &Document) -> bool {
    atoms.iter().all(|atom| {
        atom.upos == Some(Upos::Adv)
            && atom.deprel == "advmod"
            && atom.head == Some(predicate.token)
            && source_link(atom, doc).is_some_and(|link| link.sentence == predicate.sentence)
    })
}

fn is_meal_noun(source: &[char]) -> bool {
    matches!(
        source,
        ['ご', '飯']
            | ['御', '飯']
            | ['朝' | '昼' | '夕', 'ご', '飯']
            | ['朝' | '昼' | '夕', '食']
            | ['食', '事']
    )
}

pub(super) fn prior_meal(
    before: &[Atom],
    trip: &[Atom],
    doc: &Document,
    source: &[char],
) -> Option<usize> {
    let [leading @ .., meal, object_case, eat, te] = before else {
        return None;
    };
    let [place, direction, go, ending @ ..] = trip else {
        return None;
    };
    let action = source_link(eat, doc)?;
    let motion = source_link(go, doc)?;
    let meal_text = source.get(meal.span.start_char..meal.span.end_char)?;
    let past = match ending {
        [plain] if plain.surface == "た" => plain,
        [polite, past] if polite.surface == "まし" && past.surface == "た" => past,
        _ => return None,
    };
    if eat.upos != Some(Upos::Verb)
        || eat.lemma.as_deref() != Some("食べる")
        || go.upos != Some(Upos::Verb)
        || go.lemma.as_deref() != Some("行く")
        || action.deprel != "advcl"
        || action.head != Some(motion.token)
        || action.sentence != motion.sentence
        || motion.deprel != "root"
        || meal.deprel != "obj"
        || !linked_noun(meal, action, LOCKED_OBJECT_ATOM, doc)
        || !is_meal_noun(meal_text)
        || object_case.surface != "を"
        || !dasu_case_link(meal, object_case)
        || source_link(object_case, doc).is_none()
        || !only_adverbs(leading, action, doc)
        || te.surface != "て"
        || te.upos != Some(Upos::Sconj)
        || te.deprel != "mark"
        || te.head != Some(action.token)
        || eat.span.end_char != te.span.start_char
        || source_link(te, doc).is_none()
        || place.deprel != "obl"
        || !linked_noun(place, motion, LOCKED_PLACE_ATOM, doc)
        || !matches!(direction.surface.as_str(), "に" | "へ")
        || !dasu_case_link(place, direction)
        || source_link(direction, doc).is_none()
        || go.span.end_char != ending[0].span.start_char
        || ending
            .windows(2)
            .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
        || ending.iter().any(|atom| {
            atom.upos != Some(Upos::Aux)
                || atom.deprel != "aux"
                || atom.head != Some(motion.token)
                || source_link(atom, doc).is_none()
        })
    {
        return None;
    }
    // A question or quotation can survive token cleanup at the clause boundary.
    let sentence = doc.segmentation.sentences.get(motion.sentence)?.span;
    let tail = source.get(past.span.end_char..sentence.end_char)?;
    tail.iter()
        .all(|ch| matches!(ch, '。' | '.') || ch.is_whitespace())
        .then_some(action.token)
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::model::Backend;

    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslatedDocument, TranslationGlossary,
        analyze_and_translate,
    };

    fn translate(source: &str, glossary: &TranslationGlossary) -> TranslatedDocument {
        analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            glossary,
            None,
            Backend::Onnx,
        )
        .unwrap()
    }

    #[test]
    fn marks_a_meal_as_complete_before_a_past_trip() {
        let mut glossary = TranslationGlossary::new();
        let meal = translate("ご飯を食べて、駅に行った。", &glossary);
        assert!(meal.target_text.starts_with("吃了飯後，"));

        glossary.objects.push(GlossaryEntry {
            canonical: "朝ご飯".into(),
            surfaces: vec!["朝ご飯".into()],
            targets: [(TargetVariant::ZhHant, "早餐".into())].into(),
            ..GlossaryEntry::default()
        });
        let locked = translate("朝ご飯を食べて、駅へ行きました。", &glossary);
        assert!(locked.target_text.starts_with("吃了早餐後，"));
        let [hit] = locked.glossary_hits.as_slice() else {
            panic!("expected one meal glossary hit");
        };
        assert_eq!(hit.canonical, "朝ご飯");
        let span = hit.target_span;
        let alias: String = locked
            .target_text
            .chars()
            .skip(span.start_char)
            .take(span.end_char - span.start_char)
            .collect();
        assert_eq!(alias, "早餐");
    }

    #[test]
    fn leaves_other_relations_and_non_assertions_alone() {
        let glossary = TranslationGlossary::new();
        for near in [
            "ご飯を食べながら、駅へ行きました。",
            "ご飯を食べて、駅へ行きます。",
            "ご飯を食べて、駅へ行きましたか。",
            "ご飯を食べて、駅には行きませんでした。",
            "兄はご飯を食べて、私は駅へ行きました。",
        ] {
            let actual = translate(near, &glossary);
            assert!(!actual.target_text.contains("吃了飯後"), "{near}");
        }
    }
}
