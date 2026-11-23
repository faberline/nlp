//! Keep the source-linked group in affirmative polite statements and questions.

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, JointManner, LOCKED_RESIDENT_ATOM, Segment, Span, Upos,
    dasu_case_link, group_into_bunsetsus, joint_action, split_ja_atoms,
};

pub(super) fn is_polite_ending(verb: &Atom, tail: &[Atom]) -> bool {
    let (polite, question) = match tail {
        [polite] => (polite, None),
        [polite, question] => (polite, Some(question)),
        _ => return false,
    };
    verb.doc_tok_idx.is_some()
        && !verb.is_locked
        && verb.upos == Some(Upos::Verb)
        && verb.deprel == "root"
        && !polite.is_locked
        && polite.surface == "ます"
        && polite.upos == Some(Upos::Aux)
        && polite.deprel == "aux"
        && polite.head == verb.doc_tok_idx
        && verb.span.end_char == polite.span.start_char
        && question.is_none_or(|ka| {
            !ka.is_locked
                && ka.surface == "か"
                && ka.deprel == "discourse"
                && ka.head == verb.doc_tok_idx
                && polite.span.end_char == ka.span.start_char
        })
}

fn predicate(source: &[Atom]) -> Option<(&Atom, Option<usize>)> {
    let (body, verb) = match source {
        [body @ .., verb, _, ka] if ka.surface == "か" => (body, verb),
        [body @ .., verb, _] => (body, verb),
        _ => return None,
    };
    let head = verb.doc_tok_idx?;
    is_polite_ending(verb, &source[body.len() + 1..])
        .then(|| (verb, joint_action(body, verb, head).doc_tok_idx))
}

fn group_case<'a>(
    group: &'a Bunsetsu,
    verb: &Atom,
    action: Option<usize>,
    chars: &[char],
) -> Option<(&'a Atom, &'a Atom)> {
    let [everyone, de] = group.atoms.as_slice() else {
        return None;
    };
    let collective = if everyone.is_locked {
        everyone.lemma.as_deref() == Some(LOCKED_RESIDENT_ATOM)
            && matches!(
                chars.get(everyone.span.start_char..everyone.span.end_char),
                Some(['み', 'ん', 'な'] | ['皆'])
            )
    } else {
        matches!(everyone.surface.as_str(), "みんな" | "皆")
    };
    (group.role == BunsetsuRole::Obl
        && collective
        && de.surface == "で"
        && matches!(everyone.deprel.as_str(), "obl" | "advmod")
        && (everyone.head == verb.doc_tok_idx || action.is_some() && everyone.head == action)
        && dasu_case_link(everyone, de))
    .then_some((everyone, de))
}

pub(super) fn mark(groups: &mut [Bunsetsu], source: &[Atom], chars: &[char]) {
    let Some((verb, action)) = predicate(source) else {
        return;
    };
    for group in groups {
        if group_case(group, verb, action, chars).is_some() {
            group.role = BunsetsuRole::JointManner {
                manner: JointManner::Everyone,
                invitation: false,
                wh_question: false,
            };
        }
    }
}

fn actor_lock(lock: &LockedSpan, chars: &[char]) -> bool {
    lock.mandatory
        && matches!(
            lock.origin,
            LockOrigin::Glossary {
                kind: GlossaryKind::Actor,
                ..
            }
        )
        && matches!(
            chars.get(lock.source_span.start_char..lock.source_span.end_char),
            Some(['み', 'ん', 'な'] | ['み', 'ん', 'な', 'で'] | ['皆'] | ['皆', 'で'])
        )
}

pub(super) fn may_restore_actor(chars: &[char], locks: &SpanLockTable) -> bool {
    locks.locks.iter().any(|lock| actor_lock(lock, chars))
}

pub(super) fn restore_actor_case(
    sentence: usize,
    atoms: &[Atom],
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<SpanLockTable> {
    if !may_restore_actor(chars, locks) {
        return None;
    }
    let sentence_span = doc.segmentation.sentences.get(sentence)?.span;
    let clauses = split_ja_atoms(atoms, doc, sentence_span, Some(chars));
    let [Segment::Clause { atoms, .. }] = clauses.as_slice() else {
        return None;
    };
    let (verb, action) = predicate(atoms)?;
    let groups = group_into_bunsetsus(atoms, doc, sentence);
    let mut joint = groups
        .iter()
        .filter_map(|group| group_case(group, verb, action, chars));
    let (everyone, de) = joint.next()?;
    if joint.next().is_some() {
        return None;
    }
    let phrase = Span {
        start_char: everyone.span.start_char,
        end_char: de.span.end_char,
    };
    let mut render = locks.clone();
    let lock = render.locks.iter_mut().find(|lock| {
        actor_lock(lock, chars) && (lock.source_span == everyone.span || lock.source_span == phrase)
    })?;
    // Only the render table exposes で. The match keeps its original span.
    lock.source_span = everyone.span;
    Some(render)
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::{analyze_document, model::Backend};

    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
        doc_options_for_translate, translate_document,
    };

    use super::Span;

    fn entry(source: &str, target: &str) -> GlossaryEntry {
        GlossaryEntry {
            canonical: source.into(),
            surfaces: vec![source.into()],
            targets: [(TargetVariant::ZhHant, target.into())].into(),
            ..GlossaryEntry::default()
        }
    }

    fn slice(text: &str, span: Span) -> String {
        text.chars()
            .skip(span.start_char)
            .take(span.end_char - span.start_char)
            .collect()
    }

    #[test]
    fn polite_group_composes_with_an_actor_lock_and_a_verb_stem_lock() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        for (source, punctuation) in [
            ("みんなで歌います。", "。"),
            ("みんなで歌いますか。", "嗎？"),
        ] {
            for (actor, label) in [
                (None, "大家"),
                (Some("みんな"), "全組同學"),
                (Some("みんなで"), "全組同學"),
            ] {
                for (stem, action) in [(None, "哼唱"), (Some("歌い"), "合唱歌曲")] {
                    let mut glossary = TranslationGlossary::new();
                    if let Some(actor) = actor {
                        glossary.actors.push(entry(actor, label));
                    }
                    if let Some(stem) = stem {
                        glossary.domain_terms.push(entry(stem, action));
                    }
                    let expected = format!("{label}一起{action}{punctuation}");
                    let analyzed =
                        analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                            .unwrap();
                    let doc =
                        analyze_document(source, &doc_options_for_translate(&options, &glossary));
                    let direct = translate_document(
                        source,
                        &doc,
                        #[cfg(feature = "narrative")]
                        None,
                        &options,
                        &glossary,
                        None,
                    )
                    .unwrap();
                    assert_eq!(
                        analyzed.target_text, expected,
                        "{source} {actor:?} {stem:?}"
                    );
                    assert_eq!(direct.target_text, expected);
                    assert_eq!(direct.glossary_hits, analyzed.glossary_hits);
                    assert_eq!(direct.clause_alignments, analyzed.clause_alignments);
                    let expected_hits = actor
                        .into_iter()
                        .zip([label])
                        .chain(stem.into_iter().zip([action]));
                    assert_eq!(analyzed.glossary_hits.len(), expected_hits.clone().count());
                    for (canonical, target) in expected_hits {
                        let hit = analyzed
                            .glossary_hits
                            .iter()
                            .find(|hit| hit.canonical == canonical)
                            .unwrap();
                        assert_eq!(slice(source, hit.source_span), canonical);
                        assert_eq!(slice(&analyzed.target_text, hit.target_span), target);
                    }
                }
            }
        }
    }

    #[test]
    fn a_group_lookalike_locked_as_an_object_keeps_its_original_behavior() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let mut glossary = TranslationGlossary::new();
        glossary.objects.push(entry("みんなで", "現場集合"));
        let result = analyze_and_translate(
            "みんなで歌いますか。",
            &options,
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_eq!(result.target_text, "現場集合哼唱嗎？");
        assert_eq!(result.glossary_hits.len(), 1);
    }
}
