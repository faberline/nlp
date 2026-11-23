//! Express the contrast between an acted-on topic and a negated trip.

use faberline_nlp::{Span, doc::SyntaxToken};

use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Document, LOCKED_OBJECT_ATOM, LOCKED_PLACE_ATOM, Segment, Token, Upos, classify_bunsetsu,
    dasu_case_link, focused_physical_goal_head, is_companion_person, source_time_head,
    split_ja_atoms,
};

#[derive(Clone, Copy)]
pub(super) struct AffirmativeMeal {
    pub(super) predicate: usize,
    pub(super) topic: usize,
    pub(super) time_first: bool,
}

fn source_identity<'a>(atom: &Atom, doc: &'a Document) -> Option<(&'a SyntaxToken, &'a Token)> {
    let id = atom.doc_tok_idx?;
    let token = doc.tokens.get(id)?;
    let link = doc.syntax.get(id)?;
    (token.span == atom.span
        && link.token == id
        && link.span == token.span
        && link.sentence == token.sentence
        && link.head == atom.head
        && link.deprel == atom.deprel)
        .then_some((link, token))
}

fn source_link<'a>(atom: &Atom, doc: &'a Document) -> Option<&'a SyntaxToken> {
    let (link, token) = source_identity(atom, doc)?;
    (!atom.is_locked && token.upos == atom.upos).then_some(link)
}

fn qualifies_topic(
    noun: &Atom,
    predicate: &SyntaxToken,
    allowed_lock: Option<&str>,
    doc: &Document,
) -> bool {
    let identity = source_identity(noun, doc).or_else(|| {
        if !noun.is_locked || allowed_lock != Some(LOCKED_OBJECT_ATOM) {
            return None;
        }
        let id = noun.doc_tok_idx?;
        let token = doc.tokens.get(id)?;
        let before = id.checked_sub(1)?;
        let prefix = doc.tokens.get(before)?;
        let relation = doc.syntax.get(before)?;
        if prefix.span.start_char != noun.span.start_char
            || prefix.span.end_char != token.span.start_char
            || prefix.sentence != token.sentence
            || prefix.upos != Some(Upos::Adv)
            || relation.token != before
            || relation.span != prefix.span
            || relation.sentence != prefix.sentence
            || relation.head != Some(predicate.token)
            || relation.deprel != "advmod"
            || token.span.end_char != noun.span.end_char
        {
            return None;
        }
        // The locked meal can also include its native time prefix.
        let head = Atom {
            span: token.span,
            ..noun.clone()
        };
        source_identity(&head, doc)
    });
    identity.is_some_and(|(link, token)| {
        let trusted = if noun.is_locked {
            allowed_lock.is_some_and(|kind| noun.lemma.as_deref() == Some(kind))
        } else {
            token.upos == noun.upos
        };
        trusted
            && matches!(token.upos, Some(Upos::Noun | Upos::Propn))
            && matches!(link.deprel.as_str(), "nsubj" | "obj" | "obl")
            && link.head == Some(predicate.token)
            && link.sentence == predicate.sentence
    })
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

pub(super) fn negative_question_body<'a>(atoms: &'a [Atom], doc: &Document) -> Option<&'a [Atom]> {
    let (particle, body) = atoms.split_last()?;
    let go_index = body.iter().position(|atom| atom.upos == Some(Upos::Verb))?;
    let go = &body[go_index];
    (go.lemma.as_deref() == Some("行く")
        && go.deprel == "root"
        && particle.surface == "か"
        && particle.upos == Some(Upos::Part)
        && source_link(particle, doc)
            .is_some_and(|link| link.deprel == "discourse" && link.head == go.doc_tok_idx)
        && body.last()?.span.end_char == particle.span.start_char
        && past_negative(go, &body[go_index + 1..], doc))
    .then_some(body)
}

fn meal_topic(noun: &Atom, verb: &Atom, doc: &Document, chars: &[char]) -> bool {
    verb.doc_tok_idx
        .and_then(|id| doc.tokens.get(id))
        .is_some_and(|token| token.lemma.as_deref() == Some("食べる"))
        && chars
            .get(noun.span.start_char..noun.span.end_char)
            .is_some_and(is_meal)
}

fn is_meal(word: &[char]) -> bool {
    matches!(
        word,
        ['ご', '飯']
            | ['御', '飯']
            | ['朝' | '昼' | '夕', 'ご', '飯']
            | ['朝' | '昼' | '夕', '食']
            | ['食', '事']
    )
}

pub(super) fn matches(previous: &[Atom], current: &[Atom], doc: &Document, chars: &[char]) -> bool {
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
    let meal = meal_topic(object, action, doc, chars);
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
        && qualifies_topic(object, first, meal.then_some(LOCKED_OBJECT_ATOM), doc)
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
        && qualifies_topic(destination, second, meal.then_some(LOCKED_PLACE_ATOM), doc)
        && object.surface != destination.surface
        && only_adverbs(first_leading, first, doc)
        && only_adverbs(second_leading, second, doc)
        && past_negative(go, &current[go_index + 1..], doc)
}

pub(super) fn affirmative_meal(
    previous: &[Atom],
    current: &[Atom],
    doc: &Document,
    chars: &[char],
) -> Option<AffirmativeMeal> {
    let [leading @ .., topic, _, verb, _] = previous else {
        return None;
    };
    if !meal_topic(topic, verb, doc, chars) || !matches(previous, current, doc, chars) {
        return None;
    }
    Some(AffirmativeMeal {
        predicate: verb.doc_tok_idx?,
        topic: topic.doc_tok_idx?,
        time_first: leading.iter().any(source_time_head),
    })
}

pub(super) fn ascii_meal_comma(
    meal: &[Atom],
    following: &[Atom],
    doc: &Document,
    chars: &[char],
) -> bool {
    let trip = match following.split_last() {
        Some((end, trip))
            if !end.is_locked && matches!(end.surface.as_str(), "。" | "？" | "?" | "！" | "!") =>
        {
            trip
        }
        _ => following,
    };
    let trip = negative_question_body(trip, doc).unwrap_or(trip);
    affirmative_meal(meal, trip, doc, chars).is_some()
}

pub(super) fn ends_meal_question(
    prefix: &[Segment],
    current: &[Atom],
    doc: &Document,
    chars: &[char],
) -> bool {
    let [
        Segment::Clause {
            atoms: meal,
            trailing_punct: Some(separator),
            ..
        },
    ] = prefix
    else {
        return false;
    };
    matches!(separator.as_str(), "、" | "，")
        && negative_question_body(current, doc)
            .is_some_and(|trip| affirmative_meal(meal, trip, doc, chars).is_some())
}

pub(super) fn may_restore_topics(chars: &[char], locks: &SpanLockTable) -> bool {
    locks.locks.iter().any(|lock| {
        let last = chars.get(lock.source_span.end_char.saturating_sub(1));
        lock.mandatory
            && matches!(
                (&lock.origin, last),
                (
                    LockOrigin::Glossary {
                        kind: GlossaryKind::Object,
                        ..
                    },
                    Some('は')
                ) | (
                    LockOrigin::Glossary {
                        kind: GlossaryKind::Place,
                        ..
                    },
                    Some('に' | 'へ' | 'は')
                ) | (
                    LockOrigin::Glossary {
                        kind: GlossaryKind::DomainTerm,
                        ..
                    },
                    Some('た' | 'て')
                )
            )
    })
}

pub(super) fn restore_topics(
    sentence: usize,
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
    atoms: &[Atom],
) -> Option<(SpanLockTable, Option<usize>)> {
    if !may_restore_topics(chars, locks) {
        return None;
    }
    let sentence_span = doc.segmentation.sentences.get(sentence)?.span;
    let segments = split_ja_atoms(atoms, doc, sentence_span, Some(chars));
    let [
        Segment::Clause {
            atoms: previous,
            trailing_punct: Some(separator),
            ..
        },
        Segment::Clause {
            atoms: current,
            trailing_punct,
            ..
        },
    ] = segments.as_slice()
    else {
        return None;
    };
    if !matches!(separator.as_str(), "、" | "，") {
        return None;
    }
    // The question particle belongs to the clause and stays outside the glossary match.
    let trip = if matches!(trailing_punct.as_deref(), Some("?" | "？")) {
        negative_question_body(current, doc).unwrap_or(current)
    } else {
        current
    };
    affirmative_meal(previous, trip, doc, chars)?;
    let [leading @ .., meal, topic, action, connector] = previous.as_slice() else {
        return None;
    };
    let go_index = trip.iter().position(|atom| atom.upos == Some(Upos::Verb))?;
    let [.., place, direction, focus, go] = &trip[..=go_index] else {
        return None;
    };
    let goal = classify_bunsetsu(
        vec![place.clone(), direction.clone(), focus.clone()],
        doc,
        sentence,
    );
    if focused_physical_goal_head(&goal, doc, chars) != go.doc_tok_idx {
        return None;
    }
    let finite_trip = Span {
        start_char: go.span.start_char,
        end_char: trip.last()?.span.end_char,
    };
    let linked_meal = Span {
        start_char: action.span.start_char,
        end_char: connector.span.end_char,
    };
    let meal_prefix = leading.last().filter(|prefix| {
        prefix.span.end_char == meal.span.start_char
            && prefix.head == action.doc_tok_idx
            && is_meal(&chars[prefix.span.start_char..meal.span.end_char])
    });
    let mut render = locks.clone();
    let mut whole_trip = None;
    for lock in &mut render.locks {
        let LockOrigin::Glossary { kind, .. } = lock.origin else {
            return None;
        };
        if !lock.mandatory {
            return None;
        }
        let span = lock.source_span;
        let end_char = match kind {
            GlossaryKind::Object
                if (span.start_char == meal.span.start_char
                    || meal_prefix
                        .is_some_and(|prefix| span.start_char == prefix.span.start_char))
                    && [meal.span.end_char, topic.span.end_char].contains(&span.end_char) =>
            {
                meal.span.end_char
            }
            GlossaryKind::Place
                if span.start_char == place.span.start_char
                    && [
                        place.span.end_char,
                        direction.span.end_char,
                        focus.span.end_char,
                    ]
                    .contains(&span.end_char) =>
            {
                place.span.end_char
            }
            GlossaryKind::DomainTerm
                if (span == action.span || span == go.span)
                    && crate::verb_glossary::is_stem_lock(lock, doc, sentence) =>
            {
                span.end_char
            }
            GlossaryKind::DomainTerm if span == linked_meal => {
                // Use the native connector for rendering. The original lock
                // retains the full caller match and replacement.
                action.span.end_char
            }
            GlossaryKind::DomainTerm if span == finite_trip => {
                whole_trip = Some(lock.index);
                span.end_char
            }
            _ => return None,
        };
        lock.source_span = Span { end_char, ..span };
    }
    Some((render, whole_trip))
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
        for (source, destination) in [
            ("本は読んで、図書館へは行かなかった。", "圖書館"),
            ("コーヒーは飲んで、会社には行きませんでした。", "公司"),
        ] {
            let translated = translate(source, &glossary);
            let [first, second] = translated.clause_alignments.as_slice() else {
                panic!("expected the two clauses to have separate alignments");
            };
            assert_eq!(first.target_span.end_char, second.target_span.start_char);
            let negative_clause: String = translated
                .target_text
                .chars()
                .skip(second.target_span.start_char)
                .collect();
            assert!(
                negative_clause.starts_with('但')
                    && negative_clause.contains("沒有去")
                    && negative_clause.contains(destination),
                "{source}"
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
    fn affirmative_meal_preserves_the_time_and_glossary_terms() {
        let source = "朝ご飯は食べて、学校には行きませんでした。";
        let assert_contrast = |text: &str, meal: &str, place: &str| {
            let (first, second) = text.split_once('，').unwrap();
            assert_eq!(first, meal);
            let destination = second.strip_prefix("但沒有去").unwrap();
            let destination = destination.strip_prefix('到').unwrap_or(destination);
            assert_eq!(destination.strip_suffix('。'), Some(place));
        };
        let mut glossary = TranslationGlossary::new();
        let natural = translate(source, &glossary);
        let native_meal = match faberline_nlp::lexicon::ja::lookup_ja_gloss(
            "朝ご飯",
            Some(faberline_nlp::doc::Upos::Noun),
        ) {
            None => "早上有吃飯",
            Some("早餐") => "早餐有吃",
            Some(gloss) => panic!("unexpected breakfast gloss: {gloss}"),
        };
        assert_contrast(&natural.target_text, native_meal, "學校");

        let term = |canonical: &str, alias: &str| GlossaryEntry {
            canonical: canonical.to_owned(),
            surfaces: vec![canonical.to_owned()],
            targets: [(TargetVariant::ZhHant, alias.to_owned())].into(),
            ..GlossaryEntry::default()
        };
        glossary.objects.push(term("朝ご飯", "早餐"));
        glossary.places.push(term("学校", "新校舍"));
        let locked = translate(source, &glossary);
        assert_contrast(&locked.target_text, "早餐有吃", "新校舍");
        assert_eq!(locked.glossary_hits.len(), 2);
        for (hit, alias) in locked.glossary_hits.iter().zip(["早餐", "新校舍"]) {
            let span = hit.target_span;
            let text: String = locked
                .target_text
                .chars()
                .skip(span.start_char)
                .take(span.end_char - span.start_char)
                .collect();
            assert_eq!(text, alias);
        }
        assert!(
            !translate("ご飯は食べて、学校に行きませんでした。", &glossary)
                .target_text
                .contains("有吃")
        );
    }

    #[test]
    fn paired_glossary_topics_preserve_both_verified_verb_stems() {
        let source = "朝ご飯は食べて、学校には行きませんでした。";
        let entry = |canonical: &str, alias: &str| GlossaryEntry {
            canonical: canonical.into(),
            surfaces: vec![canonical.into()],
            targets: [(TargetVariant::ZhHant, alias.into())].into(),
            ..GlossaryEntry::default()
        };
        let mut glossary = TranslationGlossary::new();
        glossary.objects.push(entry("朝ご飯は", "早餐"));
        glossary.places.push(entry("学校には", "新校舍"));
        glossary.domain_terms.push(entry("食べ", "享用"));
        glossary.domain_terms.push(entry("行き", "前往"));
        let result = translate(source, &glossary);
        assert_eq!(result.target_text, "早餐有享用，但沒有前往新校舍。");
        let expected = [
            ("朝ご飯は", "早餐"),
            ("食べ", "享用"),
            ("学校には", "新校舍"),
            ("行き", "前往"),
        ];
        assert_eq!(result.glossary_hits.len(), expected.len());
        let text_at = |text: &str, span: faberline_nlp::Span| -> String {
            text.chars()
                .skip(span.start_char)
                .take(span.end_char - span.start_char)
                .collect()
        };
        for (hit, (term, alias)) in result.glossary_hits.iter().zip(expected) {
            assert_eq!(hit.canonical, term);
            assert_eq!(text_at(source, hit.source_span), term);
            assert_eq!(text_at(&result.target_text, hit.target_span), alias);
        }

        glossary.domain_terms[0] = entry("食べて", "享用過");
        let complete = translate(source, &glossary);
        assert_eq!(complete.target_text.matches("享用過").count(), 1);
        assert!(!complete.target_text.contains("有享用過"));
    }

    #[test]
    fn the_whole_negative_trip_keeps_its_contrast_and_place() {
        let source = "朝ご飯は食べて、学校には行きませんでした。";
        let entry = |canonical: &str, alias: &str| GlossaryEntry {
            canonical: canonical.into(),
            surfaces: vec![canonical.into()],
            targets: [(TargetVariant::ZhHant, alias.into())].into(),
            ..GlossaryEntry::default()
        };
        let mut glossary = TranslationGlossary::new();
        glossary.objects.push(entry("朝ご飯は", "早餐"));
        glossary.places.push(entry("学校には", "新校舍"));
        glossary.domain_terms.push(entry("食べ", "享用"));
        glossary
            .domain_terms
            .push(entry("行きませんでした", "沒有前往"));
        let translated = translate(source, &glossary);
        assert_eq!(translated.target_text, "早餐有享用，但沒有前往新校舍。");
        assert_eq!(translated.target_text.matches("沒有").count(), 1);
        let [first, second] = translated.clause_alignments.as_slice() else {
            panic!("expected alignments for both topics");
        };
        assert_eq!(first.target_span.end_char, second.target_span.start_char);
        assert_eq!(translated.glossary_hits.len(), 4);
        let trip = translated.glossary_hits.last().unwrap();
        assert_eq!(trip.canonical, "行きませんでした");
        assert_eq!(trip.source_span.end_char + 1, source.chars().count());
        assert_eq!(
            trip.source_span.end_char - trip.source_span.start_char,
            "行きませんでした".chars().count()
        );
        assert_eq!(
            trip.target_span.start_char,
            second.target_span.start_char + 1
        );
        assert_eq!(trip.target_span.end_char - trip.target_span.start_char, 4);

        let question = translate("朝ご飯は食べて、学校には行きませんでした？", &glossary);
        assert!(question.target_text.ends_with('？'));
        assert_eq!(question.target_text.matches("沒有前往").count(), 1);
        assert!(!question.target_text.contains('但'));
    }

    #[test]
    fn the_whole_meal_term_keeps_both_topics_and_its_own_affirmation() {
        let source = "朝ご飯は食べて、学校には行きませんでした。";
        let entry = |canonical: &str, alias: &str| GlossaryEntry {
            canonical: canonical.into(),
            surfaces: vec![canonical.into()],
            targets: [(TargetVariant::ZhHant, alias.into())].into(),
            ..GlossaryEntry::default()
        };
        for (replacement, first, target_start) in [
            ("享用", "早餐有享用", 3),
            ("有吃", "早餐有吃", 2),
            ("吃過", "早餐吃過", 2),
        ] {
            for (go, second) in [
                (None, "但沒有去新校舍。"),
                (Some(("行き", "前往")), "但沒有前往新校舍。"),
                (Some(("行きませんでした", "沒有前往")), "但沒有前往新校舍。"),
            ] {
                let mut glossary = TranslationGlossary::new();
                glossary.objects.push(entry("朝ご飯は", "早餐"));
                glossary.places.push(entry("学校には", "新校舍"));
                glossary.domain_terms.push(entry("食べて", replacement));
                if let Some((term, alias)) = go {
                    glossary.domain_terms.push(entry(term, alias));
                }
                let output = translate(source, &glossary);
                assert_eq!(output.target_text, format!("{first}，{second}"));
                assert_eq!(output.glossary_hits.len(), 3 + usize::from(go.is_some()));
                let meal = output
                    .glossary_hits
                    .iter()
                    .find(|hit| hit.canonical == "食べて")
                    .unwrap();
                assert_eq!(meal.source_span.start_char..meal.source_span.end_char, 4..7);
                assert_eq!(
                    meal.target_span.start_char..meal.target_span.end_char,
                    target_start..target_start + replacement.chars().count()
                );
            }
        }

        let glossary = TranslationGlossary {
            domain_terms: vec![
                entry("食べて", "享用"),
                entry("行きませんでした", "沒有前往"),
            ],
            ..TranslationGlossary::new()
        };
        let question = translate("朝ご飯は食べて、学校には行きませんでした？", &glossary);
        assert!(question.target_text.ends_with('？'));
        assert!(!question.target_text.contains('但'));
        assert_eq!(question.glossary_hits.len(), 2);
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
