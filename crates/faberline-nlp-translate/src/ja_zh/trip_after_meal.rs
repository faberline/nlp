//! Recognize a meal followed by a completed trip.

use faberline_nlp::Span;
use faberline_nlp::doc::{Options, SyntaxToken, analyze_document};

use crate::lock::LockOrigin;
use crate::schema::GlossaryKind;

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, LOCKED_OBJECT_ATOM, LOCKED_PLACE_ATOM, Segment,
    SpanLockTable, Token, Upos, build_sentence_atoms, dasu_case_link, finished_go,
    group_into_bunsetsus, split_ja_atoms,
};

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

fn linked_noun(noun: &Atom, predicate: &SyntaxToken, allowed_lock: &str, doc: &Document) -> bool {
    source_identity(noun, doc).is_some_and(|(link, token)| {
        let trusted_kind = if noun.is_locked {
            noun.lemma.as_deref() == Some(allowed_lock)
        } else {
            noun.upos == token.upos
        };
        trusted_kind
            && matches!(token.upos, Some(Upos::Noun | Upos::Propn))
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

fn is_meal_noun(source: &[char]) -> bool {
    matches!(
        source,
        ['ご', '飯']
            | ['御', '飯']
            | ['朝' | '昼' | '夕', 'ご', '飯']
            | ['朝' | '昼' | '夕', '食']
            | ['食', '事']
    )
}

#[derive(Clone, Copy)]
pub(super) struct RecoveredTrip {
    pub(super) meal_predicate: usize,
    motion_predicate: usize,
    wide_meal: Option<Span>,
    wide_place: Option<Span>,
    place: Span,
    marker: Span,
    direction: &'static str,
}

impl RecoveredTrip {
    pub(super) fn restore_arguments(
        self,
        groups: &mut [Bunsetsu],
    ) -> Option<finished_go::FinishedGo> {
        let wide_lock = |group: &Bunsetsu, span: Option<Span>, kind: &str| {
            matches!(group.atoms.as_slice(), [atom] if atom.is_locked
                && Some(atom.span) == span && atom.lemma.as_deref() == Some(kind))
        };
        if let Some(group) = groups
            .iter_mut()
            .find(|group| wide_lock(group, self.wide_meal, LOCKED_OBJECT_ATOM))
        {
            group.role = BunsetsuRole::Obj;
        }
        let goal_group = groups.iter().position(|group| {
            wide_lock(group, self.wide_place, LOCKED_PLACE_ATOM)
                || matches!(group.atoms.as_slice(), [place, marker]
                    if place.span == self.place && marker.span == self.marker
                        && marker.surface == self.direction && !marker.is_locked)
        })?;
        // Keep the one placeholder and its original lock span. The case
        // marker is needed by the renderer only for the role of this group.
        groups[goal_group].role = BunsetsuRole::Obl;
        groups[goal_group].case_particle = Some(self.direction.to_owned());
        Some(finished_go::FinishedGo {
            predicate: self.motion_predicate,
            goal_group,
        })
    }
}

/// Use source evidence for the meal-to-trip link before removing a whole verb lock from the render parse.
pub(super) fn whole_trip_lock(
    sentence: usize,
    atoms: &[Atom],
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<usize> {
    let span = doc.segmentation.sentences.get(sentence)?.span;
    let segments = split_ja_atoms(atoms, doc, span, Some(chars));
    let [
        Segment::Clause {
            atoms: meal,
            trailing_punct: Some(comma),
            ..
        },
        Segment::Clause { atoms: trip, .. },
    ] = segments.as_slice()
    else {
        return None;
    };
    if !matches!(comma.as_str(), "、" | "，") {
        return None;
    }
    let before = prior_meal(meal, trip, doc, chars)?;
    let [.., meal_noun, object_case, eat, _] = meal.as_slice() else {
        return None;
    };
    let [place, direction, go, ..] = trip.as_slice() else {
        return None;
    };
    let verb_span = Span {
        start_char: go.span.start_char,
        end_char: trip.last()?.span.end_char,
    };
    let verb = locks.locks.iter().find(|lock| {
        lock.mandatory
            && lock.source_span == verb_span
            && matches!(
                lock.origin,
                LockOrigin::Glossary {
                    kind: GlossaryKind::DomainTerm,
                    ..
                }
            )
    })?;
    if !locks.locks.iter().all(|lock| {
        if !lock.mandatory {
            return false;
        }
        let span = lock.source_span;
        match lock.origin {
            LockOrigin::Glossary {
                kind: GlossaryKind::Object,
                ..
            } => {
                (span.end_char == meal_noun.span.end_char
                    || span.end_char == object_case.span.end_char)
                    && meal
                        .iter()
                        .take_while(|atom| atom.span.start_char <= meal_noun.span.start_char)
                        .any(|atom| atom.span.start_char == span.start_char)
                    && chars
                        .get(span.start_char..meal_noun.span.end_char)
                        .is_some_and(is_meal_noun)
            }
            LockOrigin::Glossary {
                kind: GlossaryKind::Place,
                ..
            } => {
                span.start_char == place.span.start_char
                    && (span.end_char == place.span.end_char
                        || span.end_char == direction.span.end_char)
            }
            LockOrigin::Glossary {
                kind: GlossaryKind::DomainTerm,
                ..
            } => {
                lock.index == verb.index
                    || (span == eat.span && crate::verb_glossary::is_stem_lock(lock, doc, sentence))
            }
            _ => false,
        }
    }) {
        return None;
    }
    let groups = group_into_bunsetsus(trip, doc, sentence);
    finished_go::resolve(&groups, trip, doc, chars, Some(before))?;
    Some(verb.index)
}

pub(super) fn recover(
    before: &[Atom],
    trip: &[Atom],
    doc: &Document,
    chars: &[char],
) -> Option<RecoveredTrip> {
    let mut lexical_words = Vec::new();
    let mut wide = [None, None];
    for ((part, kind, markers), covered) in [
        (before, LOCKED_OBJECT_ATOM, &['を'][..]),
        (trip, LOCKED_PLACE_ATOM, &['に', 'へ'][..]),
    ]
    .into_iter()
    .zip(&mut wide)
    {
        let mut locks = part.iter().filter(|atom| atom.is_locked);
        if let Some(lock) = locks.next() {
            if locks.next().is_some() || lock.lemma.as_deref() != Some(kind) {
                return None;
            }
            let word = chars.get(lock.span.start_char..lock.span.end_char)?;
            let (last, prefix) = word.split_last()?;
            let noun = if markers.contains(last) && !prefix.is_empty() {
                *covered = Some(lock.span);
                prefix
            } else {
                word
            };
            lexical_words.push(noun.iter().collect::<String>());
        }
    }
    let [wide_meal, wide_place] = wide;
    // An analysis supplied by the caller can lack the glossary's noun hints.
    if wide_meal.is_none()
        && wide_place.is_none()
        && lexical_words
            .iter()
            .all(|word| doc.options.lexicon.contains(word))
    {
        return None;
    }
    let [.., eat, _] = before else { return None };
    let go = trip.iter().find(|atom| atom.upos == Some(Upos::Verb))?;
    let meal = source_link(eat, doc)?;
    let motion = source_link(go, doc)?;
    if eat.lemma.as_deref() != Some("食べる")
        || go.lemma.as_deref() != Some("行く")
        || meal.sentence != motion.sentence
        || meal.deprel != "advcl"
        || meal.head != Some(motion.token)
        || motion.deprel != "root"
    {
        return None;
    }

    let span = doc.segmentation.sentences.get(motion.sentence)?.span;
    let local = chars.get(span.start_char..span.end_char)?;
    let analysis = analyze_document(
        &local.iter().collect::<String>(),
        &Options {
            normalize_traditional: true,
            enrich_morphology: true,
            cpu_syntax: true,
            lexicon: lexical_words,
            ..Default::default()
        },
    );
    if analysis.segmentation.sentences.len() != 1 {
        return None;
    }
    let tokens: Vec<_> = analysis.tokens.iter().collect();
    let atoms = build_sentence_atoms(0, local, &analysis, &tokens, &SpanLockTable::default());
    let absolute = |range: Span| Span {
        start_char: span.start_char + range.start_char,
        end_char: span.start_char + range.end_char,
    };
    let part = |source: &[Atom]| -> Option<Vec<Atom>> {
        let start = source.first()?.span.start_char;
        let end = source.last()?.span.end_char;
        Some(
            atoms
                .iter()
                .filter(|atom| {
                    let range = absolute(atom.span);
                    start <= range.start_char && range.end_char <= end
                })
                .cloned()
                .collect(),
        )
    };
    let original_meal = part(before)?;
    let original_trip = part(trip)?;
    let original_meal_predicate = prior_meal(&original_meal, &original_trip, &analysis, local)?;
    let [.., meal_noun, meal_case, original_eat, _] = original_meal.as_slice() else {
        return None;
    };
    let [place, marker, original_go, ..] = original_trip.as_slice() else {
        return None;
    };
    let matches_lock = |lock: Option<Span>, noun: &Atom, case: &Atom| {
        lock.is_none_or(|lock| {
            lock.start_char == absolute(noun.span).start_char
                && lock.end_char == absolute(case.span).end_char
        })
    };
    if absolute(original_eat.span) != eat.span
        || absolute(original_go.span) != go.span
        || !matches_lock(wide_meal, meal_noun, meal_case)
        || !matches_lock(wide_place, place, marker)
    {
        return None;
    }
    let groups = group_into_bunsetsus(&original_trip, &analysis, 0);
    finished_go::resolve(
        &groups,
        &original_trip,
        &analysis,
        local,
        Some(original_meal_predicate),
    )?;
    let direction = if marker.surface == "へ" {
        "へ"
    } else {
        "に"
    };
    Some(RecoveredTrip {
        meal_predicate: meal.token,
        motion_predicate: motion.token,
        wide_meal,
        wide_place,
        place: absolute(place.span),
        marker: absolute(marker.span),
        direction,
    })
}

pub(super) fn prior_meal(
    before: &[Atom],
    trip: &[Atom],
    doc: &Document,
    source: &[char],
) -> Option<usize> {
    let [leading @ .., meal, object_case, eat, te] = before else {
        return None;
    };
    let [place, direction, go, ending @ ..] = trip else {
        return None;
    };
    let action = source_link(eat, doc)?;
    let motion = source_link(go, doc)?;
    let meal_text = source.get(meal.span.start_char..meal.span.end_char)?;
    let past = match ending {
        [plain] if plain.surface == "た" => plain,
        [polite, past] if polite.surface == "まし" && past.surface == "た" => past,
        _ => return None,
    };
    if eat.upos != Some(Upos::Verb)
        || eat.lemma.as_deref() != Some("食べる")
        || go.upos != Some(Upos::Verb)
        || go.lemma.as_deref() != Some("行く")
        || action.deprel != "advcl"
        || action.head != Some(motion.token)
        || action.sentence != motion.sentence
        || motion.deprel != "root"
        || meal.deprel != "obj"
        || !linked_noun(meal, action, LOCKED_OBJECT_ATOM, doc)
        || !is_meal_noun(meal_text)
        || object_case.surface != "を"
        || !dasu_case_link(meal, object_case)
        || source_link(object_case, doc).is_none()
        || !only_adverbs(leading, action, doc)
        || te.surface != "て"
        || te.upos != Some(Upos::Sconj)
        || te.deprel != "mark"
        || te.head != Some(action.token)
        || eat.span.end_char != te.span.start_char
        || source_link(te, doc).is_none()
        || place.deprel != "obl"
        || !linked_noun(place, motion, LOCKED_PLACE_ATOM, doc)
        || !matches!(direction.surface.as_str(), "に" | "へ")
        || !dasu_case_link(place, direction)
        || source_link(direction, doc).is_none()
        || go.span.end_char != ending[0].span.start_char
        || ending
            .windows(2)
            .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
        || ending.iter().any(|atom| {
            atom.upos != Some(Upos::Aux)
                || atom.deprel != "aux"
                || atom.head != Some(motion.token)
                || source_link(atom, doc).is_none()
        })
    {
        return None;
    }
    // A question or quotation can survive token cleanup at the clause boundary.
    let sentence = doc.segmentation.sentences.get(motion.sentence)?.span;
    let tail = source.get(past.span.end_char..sentence.end_char)?;
    tail.iter()
        .all(|ch| matches!(ch, '。' | '.') || ch.is_whitespace())
        .then_some(action.token)
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

    fn entry(surface: &str, target: &str) -> GlossaryEntry {
        GlossaryEntry {
            canonical: surface.into(),
            surfaces: vec![surface.into()],
            targets: [(TargetVariant::ZhHant, target.into())].into(),
            ..GlossaryEntry::default()
        }
    }

    #[test]
    fn locks_covering_the_meal_and_destination_particles_keep_the_sequence() {
        let source = "朝ご飯を食べて、学校へ行きました。";
        let mut options = crate::TranslateOptions::new(TargetVariant::ZhHant);
        options.source_hint = Some(TargetVariant::Ja);
        let doc = faberline_nlp::doc::analyze_document(
            source,
            &crate::doc_options_for_translate(&options, &TranslationGlossary::new()),
        );
        for (meal, place) in [
            ("朝ご飯", "学校"),
            ("朝ご飯を", "学校"),
            ("朝ご飯", "学校へ"),
            ("朝ご飯を", "学校へ"),
        ] {
            let mut glossary = TranslationGlossary::new();
            glossary.objects.push(entry(meal, "早餐"));
            glossary.places.push(entry(place, "新校舍"));
            let result = translate(source, &glossary);
            assert_eq!(result.target_text, "吃了早餐後，去了新校舍。");
            let direct = crate::translate_document(
                source,
                &doc,
                #[cfg(feature = "narrative")]
                None,
                &options,
                &glossary,
                None,
            )
            .unwrap();
            assert_eq!(direct.target_text, result.target_text, "{meal} / {place}");
            assert_eq!(direct.glossary_hits, result.glossary_hits);
            let [meal_hit, place_hit] = result.glossary_hits.as_slice() else {
                panic!("expected two glossary matches");
            };
            assert_eq!(meal_hit.source_span.start_char, 0);
            assert_eq!(meal_hit.source_span.end_char, meal.chars().count());
            assert_eq!(meal_hit.target_span.start_char, 2);
            assert_eq!(meal_hit.target_span.end_char, 4);
            assert_eq!(place_hit.source_span.start_char, 8);
            assert_eq!(place_hit.source_span.end_char, 8 + place.chars().count());
            assert_eq!(place_hit.target_span.start_char, 8);
            assert_eq!(place_hit.target_span.end_char, 11);

            let omitted = translate("朝ご飯を食べないで、学校へ行きました。", &glossary);
            assert!(!omitted.target_text.contains("吃了早餐後"));
        }
    }

    #[test]
    fn a_whole_trip_lock_keeps_the_completed_meal_and_destination() {
        let source = "朝ご飯を食べて、学校へ行きました。";
        let mut glossary = TranslationGlossary::new();
        glossary.objects.push(entry("朝ご飯を", "早餐"));
        glossary.places.push(entry("学校へ", "新校舍"));
        glossary.domain_terms.push(entry("食べ", "享用"));
        glossary.domain_terms.push(entry("行きました", "已前往"));
        let result = translate(source, &glossary);
        assert_eq!(result.target_text, "享用了早餐後，已前往新校舍。");
        assert_eq!(result.glossary_hits.len(), 4);
        let trip = result
            .glossary_hits
            .iter()
            .find(|hit| hit.canonical == "行きました")
            .unwrap();
        assert_eq!(
            trip.source_span,
            faberline_nlp::Span {
                start_char: 11,
                end_char: 16
            }
        );
        let [meal_clause, trip_clause] = result.clause_alignments.as_slice() else {
            panic!("expected separate meal and trip clauses");
        };
        assert_eq!(
            meal_clause.target_span.end_char,
            trip_clause.target_span.start_char
        );
        assert_eq!(
            trip.target_span.start_char,
            trip_clause.target_span.start_char
        );
        assert_eq!(trip.target_span.end_char - trip.target_span.start_char, 3);
    }

    #[test]
    fn a_whole_trip_lock_in_a_question_does_not_assert_a_completed_meal() {
        let source = "朝ご飯を食べて、学校へ行きましたか。";
        let mut glossary = TranslationGlossary::new();
        glossary.domain_terms.push(entry("行きました", "已前往"));
        let result = translate(source, &glossary);
        assert!(!result.target_text.contains('後'));
        assert_eq!(result.target_text.matches("已前往").count(), 1);
        assert_eq!(result.glossary_hits.len(), 1);
    }

    #[test]
    fn marks_a_meal_as_complete_before_a_past_trip() {
        let mut glossary = TranslationGlossary::new();
        let source = "朝ご飯を食べて、学校へ行きました。";
        let meal = translate(source, &glossary);
        let native = match faberline_nlp::lexicon::ja::lookup_ja_gloss(
            "朝ご飯",
            Some(faberline_nlp::doc::Upos::Noun),
        ) {
            None => "早上吃了飯後，去了學校。",
            Some("早餐") => "吃了早餐後，去了學校。",
            Some(gloss) => panic!("unexpected breakfast gloss: {gloss}"),
        };
        assert_eq!(meal.target_text, native);
        let plain = translate("ご飯を食べて、駅に行った。", &glossary);
        assert_eq!(plain.target_text, "吃了飯後，去了車站。");

        glossary.objects.push(GlossaryEntry {
            canonical: "朝ご飯".into(),
            surfaces: vec!["朝ご飯".into()],
            targets: [(TargetVariant::ZhHant, "早餐".into())].into(),
            ..GlossaryEntry::default()
        });
        let locked = translate(source, &glossary);
        assert_eq!(locked.target_text, "吃了早餐後，去了學校。");
        let [hit] = locked.glossary_hits.as_slice() else {
            panic!("expected one meal glossary hit");
        };
        assert_eq!(hit.canonical, "朝ご飯");
        let span = hit.target_span;
        let alias: String = locked
            .target_text
            .chars()
            .skip(span.start_char)
            .take(span.end_char - span.start_char)
            .collect();
        assert_eq!(alias, "早餐");
    }

    #[test]
    fn leaves_other_relations_and_non_assertions_alone() {
        let glossary = TranslationGlossary::new();
        for near in [
            "ご飯を食べながら、駅へ行きました。",
            "ご飯を食べて、駅へ行きます。",
            "ご飯を食べて、駅へ行きましたか。",
            "ご飯を食べて、駅には行きませんでした。",
            "兄はご飯を食べて、私は駅へ行きました。",
        ] {
            let actual = translate(near, &glossary);
            assert!(!actual.target_text.contains("吃了飯後"), "{near}");
        }
    }
}
