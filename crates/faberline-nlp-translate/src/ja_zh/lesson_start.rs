//! Preserve the lesson object and polite question of a classroom start.

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::{GlossaryKind, TargetVariant};
use faberline_nlp::doc::{Document, Token, analyze_document};

use super::{
    Atom, Span, Upos, build_sentence_atoms, dasu_case_link, group_into_bunsetsus, localize_punct,
    merge_adnominal_bunsetsus, render_bunsetsu_np,
};

const LESSON_TERMS: [&str; 8] = [
    "授業",
    "授業を",
    "始め",
    "はじめ",
    "始めます",
    "はじめます",
    "始めますか",
    "はじめますか",
];

#[derive(Clone, Copy)]
pub(crate) enum SourceAnalysis {
    Provided,
    OwnedCpu,
}

fn may_render(ch: &[char], sentence: Span, locks: &SpanLockTable) -> bool {
    ch[sentence.start_char..sentence.end_char]
        .windows(2)
        .any(|pair| pair == ['授', '業'])
        && locks.locks.iter().any(|lock| {
            lock.mandatory
                && matches!(
                    lock.origin,
                    LockOrigin::Glossary {
                        kind: GlossaryKind::Object | GlossaryKind::DomainTerm,
                        ..
                    }
                )
                && lock.source_span.start_char >= sentence.start_char
                && lock.source_span.end_char <= sentence.end_char
                && ch
                    .get(lock.source_span.start_char..lock.source_span.end_char)
                    .is_some_and(|span| {
                        LESSON_TERMS
                            .iter()
                            .any(|term| span.iter().copied().eq(term.chars()))
                    })
        })
}

fn root_starts_lesson(atom: &Atom) -> bool {
    !atom.is_locked
        && atom.upos == Some(Upos::Verb)
        && atom.deprel == "root"
        && atom.lemma.as_deref() == Some("始める")
}

fn lesson_head(noun: &Atom, case: &Atom) -> Option<usize> {
    if !noun.is_locked
        && noun.surface == "授業"
        && noun.upos == Some(Upos::Noun)
        && noun.deprel == "obj"
        && case.surface == "を"
        && dasu_case_link(noun, case)
    {
        noun.head
    } else {
        None
    }
}

pub(super) fn uses_native_object(group: &[Atom], source: &[Atom]) -> bool {
    let [noun, case] = group else { return false };
    lesson_head(noun, case).is_some_and(|head| {
        source
            .iter()
            .any(|atom| atom.doc_tok_idx == Some(head) && root_starts_lesson(atom))
    })
}

fn is_term(lock: &LockedSpan, kind: GlossaryKind) -> bool {
    lock.mandatory && matches!(lock.origin, LockOrigin::Glossary { kind: k, .. } if k == kind)
}

fn subject(
    source: &[Atom],
    masked: &[Atom],
    verb: &Atom,
    doc: &Document,
    sentence: usize,
) -> Option<String> {
    let Some((topic, words)) = source.split_last() else {
        return Some(String::new());
    };
    let noun = words.last()?;
    if !matches!(topic.surface.as_str(), "は" | "が" | "も")
        || noun.deprel != "nsubj"
        || noun.head != verb.doc_tok_idx
        || !dasu_case_link(noun, topic)
    {
        return None;
    }
    let masked_end = masked.partition_point(|atom| atom.span.end_char <= noun.span.end_char);
    let role_atoms = &masked[..masked_end];
    let groups = group_into_bunsetsus(role_atoms, doc, sentence);
    let merged = merge_adnominal_bunsetsus(groups, role_atoms, doc, TargetVariant::ZhHant);
    let [phrase] = merged.as_slice() else {
        return None;
    };
    let mut text = render_bunsetsu_np(phrase, TargetVariant::ZhHant);
    if text.is_empty() {
        return None;
    }
    if topic.surface == "も" {
        text.push('也');
    }
    Some(text)
}

pub(super) fn render_locked(
    chars: &[char],
    sentence: usize,
    doc: &Document,
    tokens: &[&Token],
    masked: &[Atom],
    locks: &SpanLockTable,
    source_analysis: SourceAnalysis,
) -> Option<String> {
    let span = doc.segmentation.sentences[sentence].span;
    if !may_render(chars, span, locks) {
        return None;
    }
    let unlocked = SpanLockTable::new();
    let original = build_sentence_atoms(sentence, chars, doc, tokens, &unlocked);
    if let Some(text) = render_lesson(&original, sentence, doc, masked, locks) {
        return Some(text);
    }
    if !matches!(source_analysis, SourceAnalysis::OwnedCpu)
        || doc.segmentation.opaque.iter().any(|opaque| {
            opaque.span.start_char < span.end_char && span.start_char < opaque.span.end_char
        })
    {
        return None;
    }

    // Caller terms can fuse the object with its case marker or the verb with the
    // question particle. Recover those links only in a sentence analyzed here.
    let mut options = doc.options.clone();
    options
        .lexicon
        .retain(|term| !LESSON_TERMS.contains(&term.as_str()));
    if options.lexicon == doc.options.lexicon {
        return None;
    }
    let local_chars = &chars[span.start_char..span.end_char];
    let text: String = local_chars.iter().collect();
    let local = analyze_document(&text, &options);
    let local_span = Span {
        start_char: 0,
        end_char: local_chars.len(),
    };
    let ([local_sentence], [local_clause]) = (
        local.segmentation.sentences.as_slice(),
        local.segmentation.clauses.as_slice(),
    ) else {
        return None;
    };
    if local_sentence.span != local_span
        || local_clause.span != local_span
        || !local.segmentation.opaque.is_empty()
    {
        return None;
    }
    let local_tokens: Vec<_> = local.tokens.iter().collect();
    let mut original = build_sentence_atoms(0, local_chars, &local, &local_tokens, &unlocked);
    for atom in &mut original {
        atom.span.start_char += span.start_char;
        atom.span.end_char += span.start_char;
    }
    render_lesson(&original, sentence, doc, masked, locks)
}

fn render_lesson(
    original: &[Atom],
    sentence: usize,
    doc: &Document,
    masked: &[Atom],
    locks: &SpanLockTable,
) -> Option<String> {
    let object = original
        .windows(2)
        .position(|pair| lesson_head(&pair[0], &pair[1]).is_some())?;
    let [noun, case, verb, polite, tail @ ..] = &original[object..] else {
        return None;
    };
    if verb.doc_tok_idx != lesson_head(noun, case)
        || !root_starts_lesson(verb)
        || polite.is_locked
        || polite.surface != "ます"
        || polite.upos != Some(Upos::Aux)
        || polite.deprel != "aux"
        || polite.head != verb.doc_tok_idx
        || verb.span.end_char != polite.span.start_char
    {
        return None;
    }
    let (question, terminal) = match tail {
        [marker, rest @ ..]
            if !marker.is_locked
                && marker.surface == "か"
                && marker.upos == Some(Upos::Part)
                && marker.deprel == "discourse"
                && marker.head == verb.doc_tok_idx
                && polite.span.end_char == marker.span.start_char =>
        {
            (Some(marker), rest)
        }
        rest => (None, rest),
    };
    let punctuation = match terminal {
        [] => "",
        [end] if !end.is_locked && matches!(end.surface.as_str(), "。" | "？" | "?") => {
            end.surface.as_str()
        }
        _ => return None,
    };
    let object_lock = locks.locks.iter().find(|lock| {
        is_term(lock, GlossaryKind::Object)
            && lock.source_span.start_char == noun.span.start_char
            && [noun.span.end_char, case.span.end_char].contains(&lock.source_span.end_char)
    });
    let verb_lock = locks.locks.iter().find(|lock| {
        is_term(lock, GlossaryKind::DomainTerm)
            && lock.source_span.start_char == verb.span.start_char
            && ([verb.span.end_char, polite.span.end_char].contains(&lock.source_span.end_char)
                || question.is_some_and(|q| q.span.end_char == lock.source_span.end_char))
    });
    if object_lock.is_none() && verb_lock.is_none() {
        return None;
    }
    let subject = subject(&original[..object], masked, verb, doc, sentence)?;
    let whole_question = verb_lock.filter(|lock| {
        question.is_some_and(|marker| lock.source_span.end_char == marker.span.end_char)
    });
    let object = object_lock.map_or_else(
        || {
            if whole_question.is_some() {
                "課程"
            } else if verb_lock.is_some() {
                "授業"
            } else {
                "上課"
            }
        },
        |lock| lock.placeholder.as_str(),
    );
    let predicate = verb_lock.map_or("開始", |lock| lock.placeholder.as_str());
    let rendered = if let Some(lock) = whole_question
        && lock
            .replacement
            .trim_end()
            .ends_with(['嗎', '吗', '呢', '？', '?'])
    {
        let ending = if lock.replacement.trim_end().ends_with(['？', '?']) {
            ""
        } else {
            "？"
        };
        format!("{object}，{subject}{predicate}{ending}")
    } else if question.is_some() {
        format!("{subject}{predicate}{object}嗎？")
    } else {
        format!(
            "{subject}{predicate}{object}{}",
            localize_punct(punctuation)
        )
    };
    let span = doc.segmentation.sentences[sentence].span;
    locks
        .locks
        .iter()
        .filter(|lock| {
            lock.source_span.start_char < span.end_char
                && span.start_char < lock.source_span.end_char
        })
        .all(|lock| rendered.matches(lock.placeholder.as_str()).count() == 1)
        .then_some(rendered)
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, GlossaryEntry, Span, TargetVariant, TranslateOptions, TranslationGlossary,
        analyze_and_translate, doc_options_for_translate, translate_document,
    };
    use faberline_nlp::doc::analyze_document;

    fn term(source: &str, replacement: &str) -> GlossaryEntry {
        GlossaryEntry {
            canonical: source.to_owned(),
            surfaces: vec![source.to_owned()],
            targets: [(TargetVariant::ZhHant, replacement.to_owned())].into(),
            ..GlossaryEntry::default()
        }
    }

    #[test]
    fn starts_a_lesson_with_ordinary_wording_and_question() {
        for (source, expected) in [
            ("先生は授業を始めます。", "老師開始上課。"),
            ("先生は授業を始めますか。", "老師開始上課嗎？"),
        ] {
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(doc.target_text, expected);
            assert!(doc.glossary_hits.is_empty());
        }
    }

    #[test]
    fn retains_full_glossary_spans_and_uses_one_question_mark() {
        let source = "先生は授業を始めますか。";
        for (predicate, replacement, expected) in [
            ("始め", "展開", "老師展開專題課嗎？"),
            ("始めます", "開展", "老師開展專題課嗎？"),
            ("始めますか", "已開展嗎？", "專題課，老師已開展嗎？"),
            ("始めますか", "開展", "老師開展專題課嗎？"),
        ] {
            let glossary = TranslationGlossary {
                objects: vec![term("授業を", "專題課")],
                domain_terms: vec![term(predicate, replacement)],
                ..TranslationGlossary::new()
            };
            let doc = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(doc.target_text, expected);
            let [object, action] = doc.glossary_hits.as_slice() else {
                panic!("expected object and predicate hits");
            };
            let at = |text: &str, span: Span| {
                text.chars()
                    .skip(span.start_char)
                    .take(span.end_char - span.start_char)
                    .collect::<String>()
            };
            assert_eq!(at(source, object.source_span), "授業を");
            assert_eq!(at(expected, object.target_span), "專題課");
            assert_eq!(at(source, action.source_span), predicate);
            assert_eq!(at(expected, action.target_span), replacement);
        }
    }

    #[test]
    fn uses_only_the_grammar_provided_with_a_document() {
        let source = "先生は授業を始めますか。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary {
            objects: vec![term("授業を", "專題課")],
            domain_terms: vec![term("始めますか", "已開展嗎？")],
            ..TranslationGlossary::new()
        };
        let mut input = analyze_document(
            source,
            &doc_options_for_translate(&options, &TranslationGlossary::new()),
        );
        let translate = |input: &faberline_nlp::doc::Document| {
            translate_document(
                source,
                input,
                #[cfg(feature = "narrative")]
                None,
                &options,
                &glossary,
                None,
            )
            .unwrap()
        };
        let expected = "專題課，老師已開展嗎？";
        assert_eq!(translate(&input).target_text, expected);

        let object = input
            .syntax
            .iter_mut()
            .find(|arc| arc.deprel == "obj")
            .unwrap();
        object.deprel = "obl".to_owned();
        let actual = translate(&input);
        assert_ne!(actual.target_text, expected);
        assert_eq!(actual.glossary_hits.len(), 2);
    }
}
