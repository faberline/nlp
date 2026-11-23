//! Directional reading of bare 外を as the object of 見る.

use faberline_nlp::doc::SyntaxToken;

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{Atom, Bunsetsu, Document, Span, Upos, dasu_case_link, is_punct_atom};

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
    verbs.next().is_none() && outside_object(source, look, doc).is_some()
}

fn outside_object<'a>(
    source: &'a [Atom],
    verb: &Atom,
    doc: &Document,
) -> Option<(&'a Atom, &'a Atom)> {
    let look_link = original_link(verb, doc)?;
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
        if outside.surface != "外"
            || outside.upos != Some(Upos::Noun)
            || outside.doc_tok_idx != Some(object.token)
            || marker.surface != "を"
            || !dasu_case_link(outside, marker)
            || !original_link(outside, doc).is_some_and(|link| link.sentence == look_link.sentence)
        {
            return None;
        }
        let case_link = original_link(marker, doc)?;
        // A modifier or coordinated noun can make 外 refer to a specific exterior.
        (case_link.sentence == look_link.sentence
            && !doc
                .syntax
                .iter()
                .any(|link| link.head == Some(object.token) && link.token != case_link.token))
        .then_some((outside, marker))
    })
}

pub(super) fn covers_outside_case(chars: &[char], lock: &LockedSpan) -> bool {
    lock.mandatory
        && matches!(
            lock.origin,
            LockOrigin::Glossary {
                kind: GlossaryKind::Object | GlossaryKind::Place,
                ..
            }
        )
        && chars.get(lock.source_span.start_char..lock.source_span.end_char) == Some(&['外', 'を'])
}

/// Return the final predicate of a direct gaze at bare 外を.
/// The caller checks the attached tense ending and the exact glossary span.
pub(super) fn gaze_predicate<'a>(doc: &Document, atoms: &'a [Atom]) -> Option<&'a [Atom]> {
    let mut roots = atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| atom.deprel == "root");
    let (index, head) = roots.next()?;
    if roots.next().is_some()
        || head.upos != Some(Upos::Verb)
        || head.head.is_some()
        || atoms
            .iter()
            .any(|atom| matches!(atom.surface.as_str(), "?" | "？"))
    {
        return None;
    }
    outside_object(atoms, head, doc)?;
    let predicate = &atoms[index..];
    let last = predicate.iter().rposition(|atom| !is_punct_atom(atom))?;
    Some(&predicate[..=last])
}

fn linked_verb_stem(
    doc: &Document,
    atoms: &[Atom],
    lock: &LockedSpan,
    gaze_head: usize,
) -> Option<Span> {
    if !lock.mandatory
        || !matches!(
            lock.origin,
            LockOrigin::Glossary {
                kind: GlossaryKind::DomainTerm,
                ..
            }
        )
    {
        return None;
    }
    let gaze = doc.syntax.get(gaze_head)?;
    atoms.windows(2).find_map(|pair| {
        let [verb, ending] = pair else {
            return None;
        };
        let first = original_link(verb, doc)?;
        let connector = original_link(ending, doc)?;
        (verb.span.start_char == lock.source_span.start_char
            && ending.span.end_char == lock.source_span.end_char
            && verb.span.end_char == ending.span.start_char
            && verb.upos == Some(Upos::Verb)
            && matches!(first.deprel.as_str(), "advcl" | "conj")
            && first.head == Some(gaze.token)
            && first.sentence == gaze.sentence
            && connector.sentence == gaze.sentence
            && connector.head == Some(first.token)
            && connector.deprel == "mark"
            && ending.upos == Some(Upos::Sconj)
            && matches!(ending.surface.as_str(), "て" | "で"))
        .then_some(verb.span)
    })
}

pub(super) fn restore_case(
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
    atoms: &[Atom],
    gaze_head: Option<usize>,
    whole: Option<usize>,
) -> Option<SpanLockTable> {
    if whole.is_none()
        && !locks
            .locks
            .iter()
            .any(|lock| covers_outside_case(chars, lock))
    {
        return None;
    }
    let mut render = locks.clone();
    for lock in &mut render.locks {
        if covers_outside_case(chars, lock) {
            let span = lock.source_span;
            let (_, marker) = atoms
                .iter()
                .filter(|atom| atom.upos == Some(Upos::Verb))
                .filter_map(|verb| outside_object(atoms, verb, doc))
                .find(|(noun, marker)| {
                    noun.span.start_char == span.start_char && marker.span.end_char == span.end_char
                })?;
            // Expose the native particle to rendering while the original
            // lock retains the full caller match.
            lock.source_span.end_char = marker.span.start_char;
        } else if whole != Some(lock.index)
            && !atoms.iter().any(|atom| atom.span == lock.source_span)
        {
            // Render the caller's linked action as a verb while the native
            // connector still joins the two clauses. The original match is kept.
            lock.source_span = linked_verb_stem(doc, atoms, lock, gaze_head?)?;
        }
    }
    Some(render)
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
    fn keeps_the_gaze_after_a_glossary_lock_on_the_linked_verb() {
        let source = "窓を開けて、外を見ました。";
        let entry = |term: &str, rendering: &str| GlossaryEntry {
            canonical: term.into(),
            surfaces: vec![term.into()],
            targets: [(TargetVariant::ZhHant, rendering.into())].into(),
            ..GlossaryEntry::default()
        };
        for (look, place, expected) in [
            (None, true, "推開窗戶，看了門外。"),
            (Some(("見", "察看")), true, "推開窗戶，察看了門外。"),
            (Some(("見ました", "察看了")), true, "推開窗戶，察看了門外。"),
            (
                Some(("見ました", "察看了")),
                false,
                "推開窗戶，往外察看了。",
            ),
        ] {
            let mut glossary = TranslationGlossary::new();
            glossary.domain_terms.push(entry("開けて", "推開"));
            if let Some((term, rendering)) = look {
                glossary.domain_terms.push(entry(term, rendering));
            }
            if place {
                glossary.places.push(entry("外を", "門外"));
            }
            let output = translate(source, &glossary);
            assert_eq!(output.target_text, expected);
            assert_eq!(
                output.glossary_hits.len(),
                1 + usize::from(look.is_some()) + usize::from(place)
            );
            let hit = output
                .glossary_hits
                .iter()
                .find(|hit| hit.canonical == "開けて")
                .unwrap();
            assert_eq!(hit.source_span.start_char..hit.source_span.end_char, 2..5);
            assert_eq!(hit.target_span.start_char..hit.target_span.end_char, 0..2);
        }

        let glossary = TranslationGlossary {
            domain_terms: vec![entry("窓を開けて", "開窗"), entry("見ました", "察看了")],
            ..TranslationGlossary::new()
        };
        let unchanged = translate(source, &glossary);
        assert_eq!(unchanged.target_text, "開窗，察看了外面。");
        assert_eq!(unchanged.glossary_hits.len(), 2);
    }
}
