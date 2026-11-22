//! Directional reading of bare 外を as the object of 見る.

use faberline_nlp::Span;
use faberline_nlp::doc::{Options, SyntaxToken, analyze_document};

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, LOCKED_OBJECT_ATOM, LOCKED_PLACE_ATOM, Upos,
    dasu_case_link,
};

fn original_link<'a>(atom: &Atom, doc: &'a Document) -> Option<&'a SyntaxToken> {
    let id = atom.doc_tok_idx?;
    let token = doc.tokens.get(id)?;
    let link = doc.syntax.get(id)?;
    (!atom.is_locked
        && token.span == atom.span
        && token.upos == atom.upos
        && link.span == atom.span
        && link.sentence == token.sentence
        && link.token == id
        && link.head == atom.head
        && link.deprel == atom.deprel)
        .then_some(link)
}

pub(super) fn restore_locked_object(
    groups: &mut [Bunsetsu],
    source: &[Atom],
    doc: &Document,
    chars: &[char],
) {
    let mut candidates = groups.iter().enumerate().filter_map(|(index, group)| {
        let [atom] = group.atoms.as_slice() else {
            return None;
        };
        (atom.is_locked
            && matches!(
                atom.lemma.as_deref(),
                Some(LOCKED_OBJECT_ATOM | LOCKED_PLACE_ATOM)
            )
            && chars.get(atom.span.start_char..atom.span.end_char) == Some(&['外', 'を']))
        .then_some((index, atom.span))
    });
    let Some((index, lock)) = candidates.next() else {
        return;
    };
    if candidates.next().is_some() {
        return;
    }
    let mut verbs = source.iter().filter(|atom| atom.upos == Some(Upos::Verb));
    let Some(look) = verbs.next() else {
        return;
    };
    if verbs.next().is_some()
        || look.lemma.as_deref() != Some("見る")
        || !original_link(look, doc)
            .is_some_and(|link| matches!(link.deprel.as_str(), "root" | "advcl"))
    {
        return;
    }
    let Some((first, last)) = source.first().zip(source.last()) else {
        return;
    };
    let start = first.span.start_char;
    let Some(text) = chars.get(start..last.span.end_char) else {
        return;
    };
    let unlocked = analyze_document(
        &text.iter().collect::<String>(),
        &Options {
            normalize_traditional: true,
            enrich_morphology: true,
            cpu_syntax: true,
            ..Default::default()
        },
    );
    if unlocked.segmentation.sentences.len() != 1 {
        return;
    }
    let absolute = |span: Span| Span {
        start_char: start + span.start_char,
        end_char: start + span.end_char,
    };
    let Some(marker) = unlocked.syntax.iter().find(|link| {
        let span = absolute(link.span);
        span.start_char == lock.start_char + 1
            && span.end_char == lock.end_char
            && link.deprel == "case"
            && unlocked.tokens[link.token].upos == Some(Upos::Adp)
    }) else {
        return;
    };
    let Some(noun) = marker.head.and_then(|index| unlocked.syntax.get(index)) else {
        return;
    };
    let Some(verb) = noun.head.and_then(|index| unlocked.syntax.get(index)) else {
        return;
    };
    if noun.deprel == "obj"
        && unlocked.tokens[noun.token].upos == Some(Upos::Noun)
        && absolute(noun.span).start_char == lock.start_char
        && absolute(noun.span).end_char == lock.start_char + 1
        && absolute(verb.span) == look.span
        && unlocked.tokens[verb.token].lemma.as_deref() == Some("見る")
        && !unlocked.syntax.iter().any(|link| {
            (link.head == Some(noun.token) && link.token != marker.token)
                || (link.head == Some(verb.token)
                    && link.deprel == "obj"
                    && link.token != noun.token)
        })
    {
        // Keep the placeholder and its full glossary span. Only recover the
        // object position that was hidden by the locked case marker.
        groups[index].role = BunsetsuRole::Obj;
    }
}

pub(super) fn is_direction(source: &[Atom], predicates: &[Bunsetsu], doc: &Document) -> bool {
    let [predicate] = predicates else {
        return false;
    };
    let mut verbs = predicate
        .atoms
        .iter()
        .filter(|atom| atom.upos == Some(Upos::Verb));
    let Some(look_link) = verbs.next().and_then(|look| original_link(look, doc)) else {
        return false;
    };
    if verbs.next().is_some()
        || doc.tokens[look_link.token].lemma.as_deref() != Some("見る")
        || !matches!(look_link.deprel.as_str(), "root" | "advcl")
    {
        return false;
    }

    let mut objects = doc
        .syntax
        .iter()
        .filter(|link| link.head == Some(look_link.token) && link.deprel == "obj");
    let Some(object) = objects.next() else {
        return false;
    };
    if objects.next().is_some() {
        return false;
    }

    source.windows(2).any(|pair| {
        let [outside, marker] = pair else {
            return false;
        };
        if outside.surface != "外"
            || outside.upos != Some(Upos::Noun)
            || outside.doc_tok_idx != Some(object.token)
            || marker.surface != "を"
            || !dasu_case_link(outside, marker)
            || !original_link(outside, doc).is_some_and(|link| link.sentence == look_link.sentence)
        {
            return false;
        }
        let Some(case_link) = original_link(marker, doc) else {
            return false;
        };
        // A modifier or coordinated noun can make 外 refer to a specific exterior.
        case_link.sentence == look_link.sentence
            && !doc
                .syntax
                .iter()
                .any(|link| link.head == Some(object.token) && link.token != case_link.token)
    })
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
    fn an_outside_lock_with_its_case_marker_keeps_the_object_position() {
        let source = "窓を開けて、外を見ました。";
        for surface in ["外", "外を"] {
            let entry = GlossaryEntry {
                canonical: surface.to_owned(),
                surfaces: vec![surface.to_owned()],
                targets: [(TargetVariant::ZhHant, "戶外".to_owned())].into(),
                ..GlossaryEntry::default()
            };
            let mut object = TranslationGlossary::new();
            object.objects.push(entry.clone());
            let mut place = TranslationGlossary::new();
            place.places.push(entry);

            for glossary in [object, place] {
                let result = translate(source, &glossary);
                assert_eq!(result.target_text, "打開窗戶，看了戶外。");
                assert_eq!(result.glossary_hits.len(), 1);
                let hit = &result.glossary_hits[0];
                assert_eq!(hit.source_span.start_char, 6);
                assert_eq!(hit.source_span.end_char, 6 + surface.chars().count());
                assert_eq!(hit.target_span.start_char, 7);
                assert_eq!(hit.target_span.end_char, 9);
            }
        }
    }

    #[test]
    fn bare_outside_is_a_direction_with_look() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("ドアを開けて、外を見ました。", "打開門，往外看了。"),
            ("窓から外を見ました。", "從窗戶往外看了。"),
            ("外を見ませんでした。", "沒有往外看。"),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn other_uses_of_outside_keep_their_original_role() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("子どもは外の雲を見ました。", "孩子看了外面的雲。"),
            ("建物の外を見ました。", "看了建物外面。"),
            ("外で本を見ました。", "在外面看了書。"),
            ("外から犬が見えました。", "狗從外面看得見了。"),
            ("外と中を見ました。", "看了外面和裡面。"),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn glossary_lock_wins_even_when_its_rendering_matches_the_default() {
        let mut glossary = TranslationGlossary::new();
        glossary.objects.push(GlossaryEntry {
            canonical: "外".into(),
            surfaces: vec!["外".into()],
            targets: [(TargetVariant::ZhHant, "外面".into())].into(),
            ..GlossaryEntry::default()
        });
        let result = translate("外を見ました。", &glossary);
        assert_eq!(result.target_text, "看了外面。");
        assert_eq!(result.glossary_hits[0].canonical, "外");
    }
}
