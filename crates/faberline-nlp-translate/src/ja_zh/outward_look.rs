//! Directional reading of bare 外を as the object of 見る.

use faberline_nlp::doc::SyntaxToken;

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{Atom, Bunsetsu, Document, Upos, dasu_case_link, is_punct_atom};

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

pub(super) fn restore_case(
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
    atoms: &[Atom],
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
            return None;
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
}
