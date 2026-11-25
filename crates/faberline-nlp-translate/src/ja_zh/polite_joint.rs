//! Keep the source-linked group in affirmative polite statements and questions.

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, JointManner, LOCKED_RESIDENT_ATOM, Segment, Span,
    Token, Upos, build_sentence_atoms, dasu_case_link, group_into_bunsetsus, joint_action,
    predicate_glossary, split_ja_atoms,
};

fn is_polite_ending(verb: &Atom, tail: &[Atom]) -> bool {
    let (polite, question) = match tail {
        [polite] => (polite, None),
        [polite, question] => (polite, Some(question)),
        _ => return false,
    };
    verb.doc_tok_idx.is_some()
        && !verb.is_locked
        && verb.upos == Some(Upos::Verb)
        && verb.deprel == "root"
        && verb.head.is_none()
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

fn predicate_words(source: &[Atom]) -> Option<(&[Atom], &Atom, &[Atom])> {
    let (body, verb) = match source {
        [body @ .., verb, _, ka] if ka.surface == "か" => (body, verb),
        [body @ .., verb, _] => (body, verb),
        _ => return None,
    };
    Some((body, verb, &source[body.len() + 1..]))
}

fn predicate(source: &[Atom]) -> Option<(&Atom, Option<usize>)> {
    let (body, verb, tail) = predicate_words(source)?;
    let head = verb.doc_tok_idx?;
    is_polite_ending(verb, tail).then(|| (verb, joint_action(body, verb, head).doc_tok_idx))
}

fn group_case(
    group: &Bunsetsu,
    verb: &Atom,
    action: Option<usize>,
    doc: &Document,
    chars: &[char],
    source_locks: Option<&SpanLockTable>,
) -> Option<(Span, Span)> {
    let everyone = group.atoms.first()?;
    if !matches!(everyone.deprel.as_str(), "obl" | "advmod")
        || !(everyone.head == verb.doc_tok_idx || action.is_some() && everyone.head == action)
    {
        return None;
    }
    let actor = everyone.is_locked && everyone.lemma.as_deref() == Some(LOCKED_RESIDENT_ATOM);
    let text = chars.get(everyone.span.start_char..everyone.span.end_char);
    let [_, de] = group.atoms.as_slice() else {
        // The caller may have analyzed a glossary surface as one token.
        // Its exact source spelling and retained attachment supply the role.
        let covered_actor = actor
            || source_locks.is_some_and(|locks| {
                locks
                    .locks
                    .iter()
                    .any(|lock| lock.source_span == everyone.span && actor_lock(lock, chars))
            });
        return (group.atoms.len() == 1
            && covered_actor
            && matches!(text, Some(['み', 'ん', 'な', 'で'] | ['皆', 'で']))
            && collapsed_actor_has_source_link(everyone, doc, text?))
        .then_some((everyone.span, everyone.span));
    };
    let collective = if everyone.is_locked {
        actor && matches!(text, Some(['み', 'ん', 'な'] | ['皆']))
    } else {
        matches!(everyone.surface.as_str(), "みんな" | "皆")
    };
    (group.role == BunsetsuRole::Obl
        && collective
        && de.surface == "で"
        && dasu_case_link(everyone, de))
    .then_some((
        everyone.span,
        Span {
            start_char: everyone.span.start_char,
            end_char: de.span.end_char,
        },
    ))
}

fn collapsed_actor_has_source_link(actor: &Atom, doc: &Document, text: &[char]) -> bool {
    actor.doc_tok_idx.is_some_and(|index| {
        doc.tokens.get(index).is_some_and(|token| {
            token.span == actor.span
                && token.upos == Some(Upos::Propn)
                && doc.syntax.get(index).is_some_and(|arc| {
                    arc.token == index
                        && arc.sentence == token.sentence
                        && arc.span == actor.span
                        && arc.head == actor.head
                        && arc.deprel == actor.deprel
                })
        })
    }) && doc
        .options
        .names
        .iter()
        .any(|name| name.chars().eq(text.iter().copied()))
}

pub(super) fn mark(groups: &mut [Bunsetsu], source: &[Atom], doc: &Document, chars: &[char]) {
    let Some((verb, action)) = predicate(source) else {
        return;
    };
    for group in groups {
        if group_case(group, verb, action, doc, chars, None).is_some() {
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

pub(crate) fn may_restore(text: &str, chars: &[char], locks: &SpanLockTable) -> bool {
    text.contains("ます")
        && (text.contains("みんな") || text.contains('皆'))
        && locks.locks.iter().any(|lock| {
            actor_lock(lock, chars)
                || predicate_glossary::kind(lock) == Some(GlossaryKind::DomainTerm)
        })
}

fn overlaps(a: Span, b: Span) -> bool {
    a.start_char < b.end_char && b.start_char < a.end_char
}

fn recover_glossary_stem(
    stem: &Atom,
    ending: &[Atom],
    doc: &Document,
    chars: &[char],
) -> Option<Atom> {
    // A caller's lexical hint can affect the displayed part of speech.
    // Check the original inflection and keep the supplied attachments.
    let polite = ending.first()?;
    let surface = chars.get(stem.span.start_char..stem.span.end_char)?;
    if !doc
        .options
        .lexicon
        .iter()
        .any(|word| word.chars().eq(surface.iter().copied()))
    {
        return None;
    }
    let span = stem.span.start_char..polite.span.end_char;
    let segmented = faberline_nlp::lexicon::ja::segment_ja(chars.get(span)?, stem.span.start_char);
    let [original, suffix] = segmented.as_slice() else {
        return None;
    };
    if original.span != stem.span
        || original.decided_upos != Some(Upos::Verb)
        || original.lemma.is_none()
        || suffix.span != polite.span
        || suffix.decided_upos != Some(Upos::Aux)
    {
        return None;
    }
    let mut recovered = stem.clone();
    recovered.upos = Some(Upos::Verb);
    is_polite_ending(&recovered, ending).then_some(recovered)
}

pub(crate) fn restore(
    sentence: usize,
    doc: &Document,
    tokens: &[&Token],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<(SpanLockTable, Option<predicate_glossary::PredicateTerms>)> {
    let sentence_span = doc.segmentation.sentences.get(sentence)?.span;
    let original = build_sentence_atoms(sentence, chars, doc, tokens, &SpanLockTable::new());
    let mut clauses = split_ja_atoms(&original, doc, sentence_span, Some(chars));
    let [Segment::Clause { atoms, .. }] = clauses.as_mut_slice() else {
        return None;
    };
    let (prefix, stem, ending) = predicate_words(atoms)?;
    if stem.upos == Some(Upos::Adj)
        && locks.locks.iter().any(|lock| {
            lock.source_span == stem.span
                && predicate_glossary::kind(lock) == Some(GlossaryKind::DomainTerm)
        })
    {
        let candidate = recover_glossary_stem(stem, ending, doc, chars)?;
        let index = prefix.len();
        atoms[index] = candidate;
    }
    let (verb, action) = predicate(atoms)?;
    let groups = group_into_bunsetsus(atoms, doc, sentence);
    let mut joint = groups
        .iter()
        .filter_map(|group| group_case(group, verb, action, doc, chars, Some(locks)));
    let (everyone, phrase) = joint.next()?;
    if joint.next().is_some() {
        return None;
    }
    let predicate_span = Span {
        start_char: verb.span.start_char,
        end_char: atoms.last()?.span.end_char,
    };
    let mut render = SpanLockTable::new();
    let mut terms = predicate_glossary::PredicateTerms::default();
    let mut changed = false;
    for lock in &locks.locks {
        let mut active = lock.clone();
        if overlaps(lock.source_span, phrase) {
            if !actor_lock(lock, chars)
                || (lock.source_span != everyone && lock.source_span != phrase)
            {
                return None;
            }
            // The original lock table still owns the complete glossary hit.
            active.source_span = everyone;
            changed = true;
        } else if overlaps(lock.source_span, predicate_span) {
            if predicate_glossary::kind(lock) != Some(GlossaryKind::DomainTerm)
                || lock.source_span != verb.span
                || action != verb.doc_tok_idx
            {
                return None;
            }
            terms.add_polite_joint_stem(verb.span, lock);
            changed = true;
            continue;
        }
        render.locks.push(active);
    }
    changed.then(|| (render, (!terms.is_empty()).then_some(terms)))
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
    fn joint_light_verbs_keep_the_group_or_location_and_question() {
        let bank: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/data/ja_zh/polite_joint_light_verb_dev_v1.json"
        ))
        .unwrap();
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let rows = bank["items"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        for row in rows {
            assert_eq!(row["visibility"], "development-visible");
            assert_eq!(row["heldout"], false);
            let output = analyze_and_translate(
                row["jp"].as_str().unwrap(),
                &options,
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            let references =
                std::iter::once(&row["zh_hant"]).chain(row["allowed_variants"].as_array().unwrap());
            assert!(
                references
                    .filter_map(|text| text.as_str())
                    .any(|text| text == output.target_text),
                "{}: {}",
                row["id"],
                output.target_text
            );
        }
    }

    #[test]
    fn supplied_conflicting_analysis_keeps_glossary_hits_without_a_joint_role() {
        let source = "みんなで歌いますか。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let translate = |doc: &faberline_nlp::doc::Document, glossary: &TranslationGlossary| {
            translate_document(
                source,
                doc,
                #[cfg(feature = "narrative")]
                None,
                &options,
                glossary,
                None,
            )
            .unwrap()
        };

        let mut stem_glossary = TranslationGlossary::new();
        stem_glossary.domain_terms.push(entry("歌い", "合唱歌曲"));
        let doc_options = doc_options_for_translate(&options, &TranslationGlossary::new());
        let mut supplied = analyze_document(source, &doc_options);
        let stem = supplied
            .tokens
            .iter_mut()
            .find(|token| slice(source, token.span) == "歌い")
            .unwrap();
        stem.upos = Some(faberline_nlp::doc::Upos::Adj);
        let output = translate(&supplied, &stem_glossary);
        assert!(!output.target_text.contains("一起"));
        assert_eq!(output.glossary_hits.len(), 1);

        let mut actor_glossary = TranslationGlossary::new();
        actor_glossary.actors.push(entry("みんなで", "全組同學"));
        let doc_options = doc_options_for_translate(&options, &actor_glossary);
        let mut supplied = analyze_document(source, &doc_options);
        let group = supplied
            .syntax
            .iter_mut()
            .find(|arc| slice(source, arc.span) == "みんなで")
            .unwrap();
        group.deprel = "nsubj".to_owned();
        let output = translate(&supplied, &actor_glossary);
        assert!(!output.target_text.contains("一起"));
        assert_eq!(output.glossary_hits.len(), 1);
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
