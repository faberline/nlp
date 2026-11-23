//! Directional reading of bare 外を as the object of 見る.

use faberline_nlp::doc::{SyntaxToken, analyze_document};

use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Bunsetsu, Document, LOCKED_OBJECT_ATOM, Span, Upos, build_sentence_atoms, dasu_case_link,
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

pub(super) fn is_direction(source: &[Atom], predicates: &[Bunsetsu], doc: &Document) -> bool {
    let [predicate] = predicates else {
        return false;
    };
    let mut verbs = predicate
        .atoms
        .iter()
        .filter(|atom| atom.upos == Some(Upos::Verb));
    let Some(look) = verbs.next() else {
        return false;
    };
    verbs.next().is_none()
        && outside_case(source, look, doc).is_some_and(|(noun, marker)| {
            // A modifier or coordination can make 外 a specific exterior.
            !doc.syntax
                .iter()
                .any(|link| link.head == noun.doc_tok_idx && Some(link.token) != marker.doc_tok_idx)
        })
}

fn outside_case<'a>(
    source: &'a [Atom],
    look: &Atom,
    doc: &Document,
) -> Option<(&'a Atom, &'a Atom)> {
    let look_link = original_link(look, doc)?;
    if doc.tokens[look_link.token].lemma.as_deref() != Some("見る")
        || !matches!(look_link.deprel.as_str(), "root" | "advcl")
    {
        return None;
    }

    let mut objects = doc
        .syntax
        .iter()
        .filter(|link| link.head == Some(look_link.token) && link.deprel == "obj");
    let object = objects.next()?;
    if objects.next().is_some() {
        return None;
    }

    source.windows(2).find_map(|pair| {
        let [outside, marker] = pair else {
            return None;
        };
        (outside.surface == "外"
            && outside.upos == Some(Upos::Noun)
            && outside.doc_tok_idx == Some(object.token)
            && original_link(outside, doc).is_some_and(|link| link.sentence == look_link.sentence)
            && marker.surface == "を"
            && dasu_case_link(outside, marker)
            && original_link(marker, doc).is_some_and(|link| link.sentence == look_link.sentence))
        .then_some((outside, marker))
    })
}

pub(super) fn restore_case(
    atoms: &mut Vec<Atom>,
    chars: &[char],
    sentence: Span,
    doc: &Document,
    locks: &SpanLockTable,
) {
    if !doc.options.enrich_morphology || !doc.options.cpu_syntax {
        return;
    }
    let matches: Vec<_> = locks
        .locks
        .iter()
        .filter(|lock| {
            lock.mandatory
                && matches!(
                    lock.origin,
                    LockOrigin::Glossary {
                        kind: GlossaryKind::Object | GlossaryKind::Place | GlossaryKind::DomainTerm,
                        ..
                    }
                )
                && lock.source_span.start_char >= sentence.start_char
                && lock.source_span.end_char <= sentence.end_char
                && chars.get(lock.source_span.start_char..lock.source_span.end_char)
                    == Some(&['外', 'を'])
        })
        .collect();
    if matches.is_empty() {
        return;
    }
    let Some(text) = chars.get(sentence.start_char..sentence.end_char) else {
        return;
    };
    let mut options = doc.options.clone();
    options.names.retain(|hint| hint != "外を");
    options.lexicon.retain(|hint| hint != "外を");
    let parsed = analyze_document(&text.iter().collect::<String>(), &options);
    if parsed.segmentation.sentences.len() != 1 {
        return;
    }
    let tokens: Vec<_> = parsed.tokens.iter().collect();
    let source = build_sentence_atoms(0, text, &parsed, &tokens, &SpanLockTable::new());
    let absolute = |span: Span| Span {
        start_char: sentence.start_char + span.start_char,
        end_char: sentence.start_char + span.end_char,
    };
    for look in source.iter().filter(|atom| atom.upos == Some(Upos::Verb)) {
        let Some((noun, marker)) = outside_case(&source, look, &parsed) else {
            continue;
        };
        let case = absolute(marker.span);
        let span = Span {
            start_char: absolute(noun.span).start_char,
            end_char: case.end_char,
        };
        let Some(lock) = matches.iter().find(|lock| lock.source_span == span) else {
            continue;
        };
        let Some(head) = atoms
            .iter()
            .find(|atom| {
                !atom.is_locked
                    && atom.span == absolute(look.span)
                    && atom.upos == Some(Upos::Verb)
                    && atom.lemma.as_deref() == Some("見る")
            })
            .and_then(|verb| verb.doc_tok_idx)
        else {
            continue;
        };
        let Some(index) = atoms.iter().position(|atom| {
            atom.is_locked && atom.span == span && atom.surface == lock.placeholder
        }) else {
            continue;
        };
        // The original lock and its complete source match still own the hit.
        let mut object = atoms[index].clone();
        object.span.end_char = case.start_char;
        object.lemma = Some(LOCKED_OBJECT_ATOM.to_owned());
        object.doc_tok_idx = None;
        object.head = Some(head);
        object.deprel = "obj".to_owned();
        let marker = Atom {
            span: case,
            surface: "を".to_owned(),
            lemma: None,
            upos: Some(Upos::Adp),
            doc_tok_idx: None,
            head: None,
            deprel: "case".to_owned(),
            is_locked: false,
        };
        atoms.splice(index..=index, [object, marker]);
    }
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::model::Backend;

    use crate::{
        GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslatedDocument,
        TranslationGlossary, analyze_and_translate,
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

    #[test]
    fn whole_object_lock_keeps_the_glossary_literal_after_look() {
        for kind in [
            GlossaryKind::Object,
            GlossaryKind::Place,
            GlossaryKind::DomainTerm,
        ] {
            let entry = GlossaryEntry {
                canonical: "外を".into(),
                surfaces: vec!["外を".into()],
                targets: [(TargetVariant::ZhHant, "戶外".into())].into(),
                ..GlossaryEntry::default()
            };
            let mut glossary = TranslationGlossary::new();
            match kind {
                GlossaryKind::Object => glossary.objects.push(entry),
                GlossaryKind::Place => glossary.places.push(entry),
                GlossaryKind::DomainTerm => glossary.domain_terms.push(entry),
                _ => unreachable!(),
            }
            let translated = translate("窓を開けて、外を見ました。", &glossary);
            assert_eq!(translated.target_text, "打開窗戶，看了戶外。");
            let [hit] = translated.glossary_hits.as_slice() else {
                panic!("expected one full glossary hit");
            };
            assert_eq!(hit.kind, kind);
            assert_eq!(hit.source_span.start_char, 6);
            assert_eq!(hit.source_span.end_char, 8);
            assert_eq!(hit.target_span.start_char, 7);
            assert_eq!(hit.target_span.end_char, 9);
            assert_eq!(
                translate("外を見ませんでした。", &glossary).target_text,
                "沒有看戶外。"
            );
            assert_eq!(
                translate("外を走りました。", &glossary).target_text,
                "戶外跑了。"
            );
        }
    }
}
