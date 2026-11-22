//! Express the contrast between an acted-on topic and a negated trip.

use faberline_nlp::Span;
use faberline_nlp::doc::{Options, SyntaxToken, analyze_document};

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, LOCKED_OBJECT_ATOM, LOCKED_PLACE_ATOM, SpanLockTable,
    Token, Upos, build_sentence_atoms, dasu_case_link, focused_goal, is_companion_person,
    source_time_head,
};

#[derive(Clone, Copy)]
enum MealTopic {
    Token(usize),
    LockedSpan(Span),
}

#[derive(Clone, Copy)]
pub(super) struct AffirmativeMeal {
    pub(super) predicate: usize,
    topic: MealTopic,
    time_first: bool,
}

impl AffirmativeMeal {
    pub(super) fn matches_topic(self, group: &Bunsetsu) -> bool {
        match self.topic {
            MealTopic::Token(topic) => {
                group.role == BunsetsuRole::Subj
                    && group.case_particle.as_deref() == Some("は")
                    && group
                        .atoms
                        .iter()
                        .any(|atom| atom.doc_tok_idx == Some(topic))
            }
            MealTopic::LockedSpan(span) => matches!(
                group.atoms.as_slice(),
                [atom] if atom.is_locked
                    && atom.span == span
                    && atom.lemma.as_deref() == Some(LOCKED_OBJECT_ATOM)
            ),
        }
    }

    pub(super) fn prepare_topic(self, group: &mut Bunsetsu) {
        if self.time_first {
            group.role = BunsetsuRole::Obj;
        } else if matches!(self.topic, MealTopic::LockedSpan(_)) {
            group.role = BunsetsuRole::Subj;
            group.case_particle = Some("は".to_owned());
        }
    }
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
    source_identity(noun, doc).is_some_and(|(link, token)| {
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

fn meal_topic(noun: &Atom, verb: &Atom, doc: &Document, chars: &[char]) -> bool {
    verb.doc_tok_idx
        .and_then(|id| doc.tokens.get(id))
        .is_some_and(|token| token.lemma.as_deref() == Some("食べる"))
        && matches!(
            chars.get(noun.span.start_char..noun.span.end_char),
            Some(
                ['ご', '飯']
                    | ['御', '飯']
                    | ['朝' | '昼' | '夕', 'ご', '飯']
                    | ['朝' | '昼' | '夕', '食']
                    | ['食', '事']
            )
        )
}

pub(super) fn matches(previous: &[Atom], current: &[Atom], doc: &Document, chars: &[char]) -> bool {
    matches_separate_topic(previous, current, doc, chars)
        || recover_locked_meal(previous, current, doc, chars).is_some()
}

fn matches_separate_topic(
    previous: &[Atom],
    current: &[Atom],
    doc: &Document,
    chars: &[char],
) -> bool {
    let [first_leading @ .., object, first_topic, action, connector] = previous else {
        return false;
    };
    let Some(go_index) = current
        .iter()
        .position(|atom| atom.upos == Some(Upos::Verb))
    else {
        return false;
    };
    let go = &current[go_index];
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
        && only_adverbs(first_leading, first, doc)
        && past_negative(go, &current[go_index + 1..], doc)
        && focused_destination(current, go_index, second, object, meal, doc, chars)
}

fn focused_destination(
    current: &[Atom],
    go_index: usize,
    predicate: &SyntaxToken,
    first_topic: &Atom,
    meal: bool,
    doc: &Document,
    chars: &[char],
) -> bool {
    let before_go = &current[..go_index];
    if let [leading @ .., destination, direction, topic] = before_go
        && matches!(direction.surface.as_str(), "に" | "へ")
        && dasu_case_link(destination, direction)
        && source_link(direction, doc).is_some()
        && topic.surface == "は"
        && topic.upos == Some(Upos::Adp)
        && topic.deprel == "case"
        && topic.head == destination.doc_tok_idx
        && direction.span.end_char == topic.span.start_char
        && source_link(topic, doc).is_some()
        && qualifies_topic(
            destination,
            predicate,
            meal.then_some(LOCKED_PLACE_ATOM),
            doc,
        )
    {
        return first_topic.surface != destination.surface && only_adverbs(leading, predicate, doc);
    }

    // A whole Place lock can contain the directional case and contrast
    // particle. The original clause must still prove the same negated trip.
    if !meal {
        return false;
    }
    let Some(goal) = focused_goal::recover(current, doc, chars) else {
        return false;
    };
    let (leading, destination) = match before_go {
        [leading @ .., lock] if goal.includes_focus => (leading, lock),
        [leading @ .., lock, focus] if !goal.includes_focus && focus.surface == "は" => {
            (leading, lock)
        }
        _ => return false,
    };
    destination.span == goal.span
        && goal.predicate == predicate.token
        && first_topic.surface != destination.surface
        && only_adverbs(leading, predicate, doc)
}

pub(super) fn affirmative_meal(
    previous: &[Atom],
    current: &[Atom],
    doc: &Document,
    chars: &[char],
) -> Option<AffirmativeMeal> {
    if let [leading @ .., topic, _, verb, _] = previous
        && meal_topic(topic, verb, doc, chars)
        && matches_separate_topic(previous, current, doc, chars)
    {
        return Some(AffirmativeMeal {
            predicate: verb.doc_tok_idx?,
            topic: MealTopic::Token(topic.doc_tok_idx?),
            time_first: leading.iter().any(source_time_head),
        });
    }
    recover_locked_meal(previous, current, doc, chars)
}

fn recover_locked_meal(
    previous: &[Atom],
    current: &[Atom],
    doc: &Document,
    chars: &[char],
) -> Option<AffirmativeMeal> {
    let [leading @ .., lock, action, connector] = previous else {
        return None;
    };
    if !lock.is_locked || lock.lemma.as_deref() != Some(LOCKED_OBJECT_ATOM) {
        return None;
    }
    let [noun @ .., 'は'] = chars.get(lock.span.start_char..lock.span.end_char)? else {
        return None;
    };
    if noun.is_empty() {
        return None;
    }
    let first = source_link(action, doc)?;
    let go_index = current
        .iter()
        .position(|atom| atom.upos == Some(Upos::Verb))?;
    let go = &current[go_index];
    let second = source_link(go, doc)?;
    if first.head != Some(second.token)
        || first.deprel != "advcl"
        || first.sentence != second.sentence
        || second.deprel != "root"
        || !only_adverbs(leading, first, doc)
        || !past_negative(go, &current[go_index + 1..], doc)
        || !focused_destination(current, go_index, second, lock, true, doc, chars)
    {
        return None;
    }

    // Recover the two source roles from the same sentence. The renderer
    // still consumes one locked meal replacement over its full range.
    let start = previous.first()?.span.start_char;
    let text = chars.get(start..current.last()?.span.end_char)?;
    let original = analyze_document(
        &text.iter().collect::<String>(),
        &Options {
            lexicon: vec![noun.iter().collect()],
            normalize_traditional: true,
            enrich_morphology: true,
            cpu_syntax: true,
            ..Default::default()
        },
    );
    if original.segmentation.sentences.len() != 1 {
        return None;
    }
    let source_tokens: Vec<_> = original.tokens.iter().collect();
    let atoms = build_sentence_atoms(
        0,
        text,
        &original,
        &source_tokens,
        &SpanLockTable::default(),
    );
    let absolute = |span: Span| Span {
        start_char: start + span.start_char,
        end_char: start + span.end_char,
    };
    let meal_end =
        atoms.partition_point(|atom| absolute(atom.span).end_char <= connector.span.end_char);
    let trip_start =
        atoms.partition_point(|atom| absolute(atom.span).start_char < current[0].span.start_char);
    let [
        ..,
        original_noun,
        topic,
        original_action,
        original_connector,
    ] = &atoms[..meal_end]
    else {
        return None;
    };
    let trip = atoms.get(trip_start..)?;
    let original_go = trip.iter().find(|atom| atom.upos == Some(Upos::Verb))?;
    if meal_end > trip_start
        || absolute(original_noun.span).start_char != lock.span.start_char
        || absolute(topic.span).end_char != lock.span.end_char
        || absolute(original_action.span) != action.span
        || absolute(original_connector.span) != connector.span
        || absolute(original_go.span) != go.span
        || !meal_topic(original_noun, original_action, &original, text)
        || !matches_separate_topic(&atoms[..meal_end], trip, &original, text)
    {
        return None;
    }
    Some(AffirmativeMeal {
        predicate: first.token,
        topic: MealTopic::LockedSpan(lock.span),
        time_first: leading.iter().any(source_time_head),
    })
}

#[cfg(test)]
mod wide_meal_topic_tests {
    use faberline_nlp::doc::{analyze_document, model::Backend};

    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
        doc_options_for_translate, translate_document,
    };

    fn entry(surface: &str, replacement: &str) -> GlossaryEntry {
        GlossaryEntry {
            canonical: surface.into(),
            surfaces: vec![surface.into()],
            targets: [(TargetVariant::ZhHant, replacement.into())].into(),
            ..GlossaryEntry::default()
        }
    }

    #[test]
    fn locked_meal_topic_keeps_the_contrast_and_full_glossary_hits() {
        let source = "朝ご飯は食べて、学校には行きませんでした。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let mut glossary = TranslationGlossary::new();
        let doc = analyze_document(source, &doc_options_for_translate(&options, &glossary));
        glossary.objects.push(entry("朝ご飯は", "早餐"));
        for (places, expected) in [
            (vec![], "早餐有吃，但沒有去學校。"),
            (
                vec![entry("学校には", "新校舍")],
                "早餐有吃，但沒有去新校舍。",
            ),
        ] {
            glossary.places = places;
            let analyzed =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
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
            assert_eq!(analyzed.target_text, expected);
            assert_eq!(direct.target_text, expected);
            assert_eq!(direct.glossary_hits, analyzed.glossary_hits);
            assert_eq!(direct.glossary_hits.len(), 1 + glossary.places.len());
            for hit in direct.glossary_hits {
                let at = |text: &str, span: faberline_nlp::Span| -> String {
                    text.chars()
                        .skip(span.start_char)
                        .take(span.end_char - span.start_char)
                        .collect()
                };
                let replacement = if hit.canonical == "朝ご飯は" {
                    "早餐"
                } else {
                    "新校舍"
                };
                assert_eq!(at(source, hit.source_span), hit.canonical);
                assert_eq!(at(expected, hit.target_span), replacement);
            }
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
        assert_contrast(&natural.target_text, "早上有吃飯", "學校");

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
