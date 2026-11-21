//! Select the smoking sense of 吸う when its original object is tobacco.

use faberline_nlp::doc::{SyntaxToken, Token};

use super::{Atom, Bunsetsu, Document, LOCKED_OBJECT_ATOM, Upos, dasu_case_link};

fn source_identity<'a>(atom: &Atom, doc: &'a Document) -> Option<(&'a SyntaxToken, &'a Token)> {
    let id = atom.doc_tok_idx?;
    let token = doc.tokens.get(id)?;
    let link = doc.syntax.get(id)?;
    (token.span == atom.span
        && link.span == token.span
        && link.sentence == token.sentence
        && link.token == id
        && link.head == atom.head
        && link.deprel == atom.deprel)
        .then_some((link, token))
}

pub(super) fn tobacco_object<'a>(
    source: &'a [Atom],
    predicates: &[Bunsetsu],
    doc: &Document,
    chars: &[char],
) -> Option<&'a Atom> {
    let mut verbs = predicates
        .iter()
        .flat_map(|group| &group.atoms)
        .filter(|atom| !atom.is_locked && atom.upos == Some(Upos::Verb))
        .filter_map(|atom| {
            let (link, token) = source_identity(atom, doc)?;
            (token.upos == Some(Upos::Verb)
                && token.lemma.as_deref() == Some("吸う")
                && matches!(link.deprel.as_str(), "root" | "advcl" | "conj"))
            .then_some(link)
        });
    let verb = verbs.next()?;
    if verbs.next().is_some() {
        return None;
    }
    let mut objects = doc
        .syntax
        .iter()
        .filter(|link| link.head == Some(verb.token) && link.deprel == "obj");
    let object = objects.next()?;
    if objects.next().is_some() {
        return None;
    }

    source.windows(2).find_map(|pair| {
        let [noun, marker] = pair else {
            return None;
        };
        if noun.doc_tok_idx != Some(object.token)
            || marker.surface != "を"
            || !dasu_case_link(noun, marker)
        {
            return None;
        }
        let (noun_link, token) = source_identity(noun, doc)?;
        let (case_link, case_token) = source_identity(marker, doc)?;
        let trusted_noun = if noun.is_locked {
            noun.lemma.as_deref() == Some(LOCKED_OBJECT_ATOM)
        } else {
            noun.upos == token.upos
        };
        (trusted_noun
            && token.upos == Some(Upos::Noun)
            && noun_link.sentence == verb.sentence
            && case_link.sentence == verb.sentence
            && case_token.upos == Some(Upos::Adp)
            && matches!(
                chars.get(noun.span.start_char..noun.span.end_char),
                Some(['た', 'ば', 'こ'] | ['タ', 'バ', 'コ'] | ['煙', '草'])
            )
            && chars.get(marker.span.start_char..marker.span.end_char) == Some(&['を']))
        .then_some(noun)
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
    fn tobacco_object_selects_smoking_and_keeps_the_original_mood() {
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("姉はタバコを吸いました。", "姊姊抽了菸。"),
            ("店の前で煙草を吸いません。", "在商店前面不抽菸。"),
            ("公園でタバコを吸わないでください。", "請不要在公園抽菸。"),
            ("彼は安い煙草を吸いました。", "他抽了便宜的菸。"),
        ] {
            assert_eq!(
                translate(source, &glossary).target_text,
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn smoke_air_and_mentions_in_other_roles_do_not_select_smoking() {
        let glossary = TranslationGlossary::new();
        for source in [
            "少年は煙を吸いました。",
            "一緒に息を吸いました。",
            "弟はタバコを買いました。",
            "タバコの箱を吸いました。",
            "煙草を買って、息を吸いました。",
        ] {
            assert!(
                !translate(source, &glossary).target_text.contains('抽'),
                "{source}"
            );
        }
    }

    #[test]
    fn a_tobacco_glossary_alias_keeps_its_spelling_and_span() {
        let mut glossary = TranslationGlossary::new();
        glossary.objects.push(GlossaryEntry {
            canonical: "たばこ".into(),
            surfaces: vec!["たばこ".into()],
            targets: [(TargetVariant::ZhHant, "紙菸".into())].into(),
            ..GlossaryEntry::default()
        });
        let translated = translate("公園でたばこを吸わないでください。", &glossary);
        assert_eq!(translated.target_text, "請不要在公園抽紙菸。");
        let [hit] = translated.glossary_hits.as_slice() else {
            panic!("expected one tobacco glossary hit");
        };
        assert_eq!(hit.canonical, "たばこ");
        let span = hit.target_span;
        let alias: String = translated
            .target_text
            .chars()
            .skip(span.start_char)
            .take(span.end_char - span.start_char)
            .collect();
        assert_eq!(alias, "紙菸");
    }
}
