//! Express the contrast between an acted-on topic and a negated trip.

use faberline_nlp::doc::SyntaxToken;

use super::{Atom, Document, Upos, dasu_case_link, is_companion_person};

fn source_link<'a>(atom: &Atom, doc: &'a Document) -> Option<&'a SyntaxToken> {
    let id = atom.doc_tok_idx?;
    let token = doc.tokens.get(id)?;
    let link = doc.syntax.get(id)?;
    (!atom.is_locked
        && token.span == atom.span
        && token.upos == atom.upos
        && link.token == id
        && link.span == token.span
        && link.sentence == token.sentence
        && link.head == atom.head
        && link.deprel == atom.deprel)
        .then_some(link)
}

fn qualifies_topic(noun: &Atom, predicate: &SyntaxToken, doc: &Document) -> bool {
    matches!(noun.upos, Some(Upos::Noun | Upos::Propn))
        && matches!(noun.deprel.as_str(), "nsubj" | "obj" | "obl")
        && noun.head == Some(predicate.token)
        && source_link(noun, doc).is_some_and(|link| link.sentence == predicate.sentence)
}

fn only_adverbs(atoms: &[Atom], predicate: &SyntaxToken, doc: &Document) -> bool {
    atoms.iter().all(|atom| {
        atom.upos == Some(Upos::Adv)
            && atom.deprel == "advmod"
            && atom.head == Some(predicate.token)
            && source_link(atom, doc).is_some_and(|link| link.sentence == predicate.sentence)
    })
}

fn past_negative(verb: &Atom, tail: &[Atom], doc: &Document) -> bool {
    let correct_ending = match tail {
        [negative, past] => negative.surface == "なかっ" && past.surface == "た",
        [polite, negative, copula, past] => {
            polite.surface == "ませ"
                && negative.surface == "ん"
                && copula.surface == "でし"
                && past.surface == "た"
        }
        _ => false,
    };
    correct_ending
        && tail[0].span.start_char == verb.span.end_char
        && tail
            .windows(2)
            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
        && tail.iter().all(|atom| {
            atom.upos == Some(Upos::Aux)
                && atom.deprel == "aux"
                && atom.head == verb.doc_tok_idx
                && source_link(atom, doc).is_some()
        })
}

pub(super) fn matches(previous: &[Atom], current: &[Atom], doc: &Document) -> bool {
    let [first_leading @ .., object, first_topic, action, connector] = previous else {
        return false;
    };
    let Some(go_index) = current
        .iter()
        .position(|atom| atom.upos == Some(Upos::Verb))
    else {
        return false;
    };
    let [
        second_leading @ ..,
        destination,
        direction,
        second_topic,
        go,
    ] = &current[..=go_index]
    else {
        return false;
    };
    let Some(first) = source_link(action, doc) else {
        return false;
    };
    let Some(second) = source_link(go, doc) else {
        return false;
    };
    // An object topic of a deliberate action supports a contrast. A weather
    // event or change of state can instead give a reason for the negated trip.
    first.sentence == second.sentence
        && first.deprel == "advcl"
        && first.head == Some(second.token)
        && action.upos == Some(Upos::Verb)
        && matches!(
            doc.tokens[first.token].lemma.as_deref(),
            Some("食べる" | "飲む" | "読む" | "見る" | "する")
        )
        && second.deprel == "root"
        && doc.tokens[second.token].lemma.as_deref() == Some("行く")
        && connector.upos == Some(Upos::Sconj)
        && matches!(connector.surface.as_str(), "て" | "で")
        && connector.deprel == "mark"
        && connector.head == Some(first.token)
        && action.span.end_char == connector.span.start_char
        && source_link(connector, doc).is_some()
        && first_topic.surface == "は"
        && dasu_case_link(object, first_topic)
        && source_link(first_topic, doc).is_some()
        && qualifies_topic(object, first, doc)
        && !is_companion_person(object)
        && matches!(direction.surface.as_str(), "に" | "へ")
        && dasu_case_link(destination, direction)
        && source_link(direction, doc).is_some()
        && second_topic.surface == "は"
        && second_topic.upos == Some(Upos::Adp)
        && second_topic.deprel == "case"
        && second_topic.head == destination.doc_tok_idx
        && direction.span.end_char == second_topic.span.start_char
        && source_link(second_topic, doc).is_some()
        && qualifies_topic(destination, second, doc)
        && object.surface != destination.surface
        && only_adverbs(first_leading, first, doc)
        && only_adverbs(second_leading, second, doc)
        && past_negative(go, &current[go_index + 1..], doc)
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
    fn paired_topics_put_the_contrast_in_the_negative_clause() {
        let glossary = TranslationGlossary::new();
        for (source, suffix) in [
            (
                "本は読んで、図書館へは行かなかった。",
                "，但往圖書館沒有去。",
            ),
            (
                "コーヒーは飲んで、会社には行きませんでした。",
                "，但沒有去到公司。",
            ),
        ] {
            let translated = translate(source, &glossary);
            assert!(translated.target_text.ends_with(suffix), "{source}");
            let [first, second] = translated.clause_alignments.as_slice() else {
                panic!("expected the two clauses to have separate alignments");
            };
            assert_eq!(first.target_span.end_char, second.target_span.start_char);
            assert_eq!(
                translated
                    .target_text
                    .chars()
                    .nth(second.target_span.start_char),
                Some('但')
            );
        }
    }

    #[test]
    fn other_te_connections_keep_their_existing_relation() {
        let glossary = TranslationGlossary::new();
        for source in [
            "雨は降って、学校には行きませんでした。",
            "新聞を読んで、駅には行きませんでした。",
            "新聞は読んで、駅に行きませんでした。",
            "新聞は読んで、駅には行きました。",
            "新聞は読んで、駅には行きませんでしたか。",
            "「新聞は読んで、駅へは行きませんでした」と言いました。",
        ] {
            assert!(
                !translate(source, &glossary).target_text.contains('但'),
                "{source}"
            );
        }
    }

    #[test]
    fn a_locked_topic_keeps_its_original_rendering() {
        let mut glossary = TranslationGlossary::new();
        glossary.objects.push(GlossaryEntry {
            canonical: "新聞".into(),
            surfaces: vec!["新聞".into()],
            targets: [(TargetVariant::ZhHant, "晨報".into())].into(),
            ..GlossaryEntry::default()
        });
        let translated = translate("新聞は読んで、駅へは行きませんでした。", &glossary);
        assert!(translated.target_text.contains("晨報"));
        assert!(!translated.target_text.contains('但'));
        assert_eq!(translated.glossary_hits[0].canonical, "新聞");
    }
}
