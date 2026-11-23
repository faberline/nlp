//! Select the smoking sense of 吸う when its original object is tobacco.

use faberline_nlp::Span;
use faberline_nlp::doc::{Options, SyntaxToken, Token, analyze_document};

use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{Atom, Bunsetsu, BunsetsuRole, Document, LOCKED_OBJECT_ATOM, Upos, dasu_case_link};

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

fn smoking_verb<'a>(
    atoms: impl Iterator<Item = &'a Atom>,
    doc: &'a Document,
) -> Option<&'a SyntaxToken> {
    let mut verbs = atoms
        .filter(|atom| !atom.is_locked && atom.upos == Some(Upos::Verb))
        .filter_map(|atom| {
            let (link, token) = source_identity(atom, doc)?;
            (token.upos == Some(Upos::Verb)
                && token.lemma.as_deref() == Some("吸う")
                && matches!(link.deprel.as_str(), "root" | "advcl" | "conj"))
            .then_some(link)
        });
    let verb = verbs.next()?;
    verbs.next().is_none().then_some(verb)
}

fn is_tobacco(word: &[char]) -> bool {
    matches!(word, ['た', 'ば', 'こ'] | ['タ', 'バ', 'コ'] | ['煙', '草'])
}

pub(super) fn restore_locked_object<'a>(
    groups: &mut [Bunsetsu],
    source: &'a [Atom],
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<&'a Atom> {
    let mut objects = source.iter().enumerate().filter_map(|(idx, atom)| {
        if !atom.is_locked {
            return None;
        }
        let word = chars.get(atom.span.start_char..atom.span.end_char)?;
        let (noun, noun_end, case_end) = if let [noun @ .., 'を'] = word {
            (noun, atom.span.end_char - 1, atom.span.end_char)
        } else {
            let case = source.get(idx + 1)?;
            if case.is_locked || case.surface != "を" || case.span.start_char != atom.span.end_char
            {
                return None;
            }
            (word, atom.span.end_char, case.span.end_char)
        };
        (is_tobacco(noun)
            && locks.locks.iter().any(|entry| {
                entry.mandatory
                    && entry.source_span == atom.span
                    && entry.placeholder == atom.surface
                    && matches!(
                        entry.origin,
                        LockOrigin::Glossary {
                            kind: GlossaryKind::Object | GlossaryKind::DomainTerm,
                            ..
                        }
                    )
            }))
        .then_some((atom, noun, noun_end, case_end))
    });
    let (lock, noun, noun_end, case_end) = objects.next()?;
    if objects.next().is_some() {
        return None;
    }
    let verb = smoking_verb(source.iter(), doc)?;
    let start = source.first()?.span.start_char;
    let original = chars.get(start..source.last()?.span.end_char)?;
    let analysis = analyze_document(
        &original.iter().collect::<String>(),
        &Options {
            normalize_traditional: true,
            enrich_morphology: true,
            cpu_syntax: true,
            lexicon: vec![noun.iter().collect::<String>()],
            ..Default::default()
        },
    );
    if analysis.segmentation.sentences.len() != 1 {
        return None;
    }
    let absolute = |span: Span| Span {
        start_char: span.start_char + start,
        end_char: span.end_char + start,
    };
    let case = analysis.syntax.iter().find(|link| {
        let span = absolute(link.span);
        span.start_char == noun_end
            && span.end_char == case_end
            && link.deprel == "case"
            && analysis.tokens[link.token].upos == Some(Upos::Adp)
    })?;
    let object = analysis.syntax.get(case.head?)?;
    let governor = analysis.syntax.get(object.head?)?;
    let noun_span = absolute(object.span);
    if noun_span.start_char != lock.span.start_char
        || noun_span.end_char != absolute(case.span).start_char
        || object.deprel != "obj"
        || analysis.tokens[object.token].upos != Some(Upos::Noun)
        || absolute(governor.span) != verb.span
        || analysis.tokens[governor.token].lemma.as_deref() != Some("吸う")
        || analysis.syntax.iter().any(|link| {
            link.head == Some(governor.token) && link.deprel == "obj" && link.token != object.token
        })
    {
        return None;
    }
    let group = groups.iter_mut().find(|group| {
        let Some((atom, rest)) = group.atoms.split_first() else {
            return false;
        };
        atom.is_locked
            && atom.span == lock.span
            && match rest {
                [] => true,
                [case] => {
                    !case.is_locked
                        && case.surface == "を"
                        && case.span.start_char == noun_end
                        && case.span.end_char == case_end
                }
                _ => false,
            }
    })?;
    // The exact glossary placeholder is opaque to rendering; only the
    // attested object role is recovered here.
    group.role = BunsetsuRole::Obj;
    Some(lock)
}

pub(super) fn tobacco_object<'a>(
    source: &'a [Atom],
    predicates: &[Bunsetsu],
    doc: &Document,
    chars: &[char],
) -> Option<&'a Atom> {
    let verb = smoking_verb(predicates.iter().flat_map(|group| &group.atoms), doc)?;
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
            && chars
                .get(noun.span.start_char..noun.span.end_char)
                .is_some_and(is_tobacco)
            && chars.get(marker.span.start_char..marker.span.end_char) == Some(&['を']))
        .then_some(noun)
    })
}

#[cfg(test)]
mod term_glossary_tests {
    use faberline_nlp::Span;
    use faberline_nlp::doc::{analyze_document, model::Backend};

    use crate::{
        GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
        analyze_and_translate, doc_options_for_translate, translate_document,
    };

    #[test]
    fn tobacco_domain_term_keeps_its_declared_category_and_object_role() {
        let source = "ここでたばこを吸わないでください。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let doc = analyze_document(
            source,
            &doc_options_for_translate(&options, &TranslationGlossary::new()),
        );
        for (word, end) in [("たばこ", 6), ("たばこを", 7)] {
            let mut glossary = TranslationGlossary::new();
            glossary.domain_terms.push(GlossaryEntry {
                canonical: word.into(),
                surfaces: vec![word.into()],
                targets: [(TargetVariant::ZhHant, "雪茄".into())].into(),
                ..GlossaryEntry::default()
            });
            let hinted =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            let unhinted = translate_document(
                source,
                &doc,
                #[cfg(feature = "narrative")]
                None,
                &options,
                &glossary,
                None,
            )
            .unwrap();
            assert_eq!(hinted.target_text, "請不要在這裡抽雪茄。");
            assert_eq!(hinted.target_text, unhinted.target_text);
            assert_eq!(hinted.glossary_hits, unhinted.glossary_hits);
            let [hit] = hinted.glossary_hits.as_slice() else {
                panic!("expected one complete glossary hit");
            };
            assert_eq!(hit.kind, GlossaryKind::DomainTerm);
            assert_eq!(
                hit.source_span,
                Span {
                    start_char: 3,
                    end_char: end
                }
            );
            let target: String = hinted
                .target_text
                .chars()
                .skip(hit.target_span.start_char)
                .take(hit.target_span.end_char - hit.target_span.start_char)
                .collect();
            assert_eq!(target, "雪茄");
        }
    }
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
    fn a_tobacco_object_lock_keeps_its_position_when_it_includes_the_case_marker() {
        let source = "ここでたばこを吸わないでください。";
        for surface in ["たばこ", "たばこを"] {
            let mut glossary = TranslationGlossary::new();
            glossary.objects.push(GlossaryEntry {
                canonical: surface.into(),
                surfaces: vec![surface.into()],
                targets: [(TargetVariant::ZhHant, "菸".into())].into(),
                ..GlossaryEntry::default()
            });
            let result = translate(source, &glossary);
            assert_eq!(result.target_text, "請不要在這裡抽菸。");
            assert_eq!(result.glossary_hits.len(), 1);
            let hit = &result.glossary_hits[0];
            assert_eq!(hit.source_span.start_char, 3);
            assert_eq!(hit.source_span.end_char, 3 + surface.chars().count());
            assert_eq!(hit.target_span.start_char, 7);
            assert_eq!(hit.target_span.end_char, 8);
        }
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
