//! Source evidence for one completed trip to a physical destination.

use faberline_nlp::doc::{EntityKind, Token};

use super::{
    Atom, Bunsetsu, BunsetsuRole, CLOCK_ATOM, Document, LOCKED_PLACE_ATOM, SemanticClass, Span,
    Upos, clock_case_governor, dasu_case_link, is_clock_atom, physical_goal_head,
};

pub(super) struct FinishedGo {
    pub(super) predicate: usize,
    pub(super) goal_group: usize,
}

struct DirectGo {
    predicate: usize,
    place: usize,
    marker: usize,
    clock_marker: Option<usize>,
}

fn raw(source: &[char], span: Span) -> Option<&[char]> {
    source.get(span.start_char..span.end_char)
}

fn source_head<'a>(atom: &Atom, doc: &'a Document) -> Option<&'a Token> {
    let id = atom.doc_tok_idx?;
    let token = doc.tokens.get(id)?;
    let link = doc.syntax.get(id)?;
    (link.token == id
        && link.span == token.span
        && link.sentence == token.sentence
        && link.head == atom.head
        && link.deprel == atom.deprel
        && atom.span.start_char <= token.span.start_char
        && atom.span.end_char == token.span.end_char)
        .then_some(token)
}

fn completed_go(verb: &Atom, tail: &[Atom], doc: &Document, source: &[char]) -> Option<usize> {
    let predicate = verb.doc_tok_idx?;
    let original = source_head(verb, doc)?;
    let sentence = doc.segmentation.sentences.get(original.sentence)?.span;
    let first_token = doc
        .tokens
        .partition_point(|token| token.span.start_char < sentence.start_char);
    let end_token = doc
        .tokens
        .partition_point(|token| token.span.start_char < sentence.end_char);
    if verb.is_locked
        || original.span != verb.span
        || original.upos != Some(Upos::Verb)
        || original.lemma.as_deref() != Some("行く")
        || verb.deprel != "root"
        || verb.head.is_some()
        || raw(source, sentence)?
            .iter()
            .any(|ch| matches!(ch, '？' | '?' | '「' | '」' | '『' | '』' | '"' | '“' | '”'))
    {
        return None;
    }
    let affirmative_past = match tail {
        [past] => past.surface == "た",
        [polite, past] => polite.surface == "まし" && past.surface == "た",
        _ => false,
    };
    if !affirmative_past
        || tail[0].span.start_char != verb.span.end_char
        || tail
            .windows(2)
            .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
        || tail.iter().any(|ending| {
            ending.is_locked
                || ending.head != Some(predicate)
                || ending.deprel != "aux"
                || source_head(ending, doc)
                    .is_none_or(|token| token.span != ending.span || token.upos != Some(Upos::Aux))
        })
        || doc.tokens[first_token..end_token]
            .iter()
            .enumerate()
            .any(|(offset, item)| {
                first_token + offset != predicate
                    && (item.upos == Some(Upos::Verb)
                        || matches!(
                            raw(source, item.span),
                            Some(
                                ['何']
                                    | ['何', '時']
                                    | ['誰']
                                    | ['ど', 'こ']
                                    | ['い', 'つ']
                                    | ['ど', 'れ']
                                    | ['ど', 'う']
                                    | ['ど', 'ち', 'ら']
                                    | ['な', 'ぜ']
                                    | ['か']
                                    | ['時', '間']
                                    | ['分', '間']
                                    | ['日', '間']
                                    | ['週', '間']
                                    | ['年', '間']
                            )
                        ))
            })
    {
        return None;
    }
    Some(predicate)
}

// A clock can be normalized from split tokens, including a fused 時に.
// The existing source helper distinguishes both forms from a place or recipient に.
fn attached_clock(clock: &Atom, marker: &Atom, predicate: usize, doc: &Document) -> Option<usize> {
    if clock.lemma.as_deref() != Some(CLOCK_ATOM)
        || !clock.is_locked
        || marker.is_locked
        || marker.surface != "に"
        || marker.upos != Some(Upos::Adp)
        || clock.span.end_char != marker.span.start_char
        || clock_case_governor(clock, marker, doc) != Some(predicate)
    {
        return None;
    }
    marker.doc_tok_idx
}

fn direct_destination(
    atoms: &[Atom],
    predicate: usize,
    doc: &Document,
    source: &[char],
) -> Option<DirectGo> {
    let mut clock_marker = None;
    for (i, atom) in atoms.iter().enumerate() {
        if is_clock_atom(atom) {
            let marker = attached_clock(atom, atoms.get(i + 1)?, predicate, doc)?;
            if clock_marker.replace(marker).is_some() {
                return None;
            }
        }
    }
    let mut destination = None;
    for link in doc.syntax.iter().filter(|link| link.deprel == "case") {
        let Some(argument) = link.head.and_then(|head| doc.syntax.get(head)) else {
            continue;
        };
        if argument.head != Some(predicate) || Some(link.token) == clock_marker {
            continue;
        }
        match (argument.deprel.as_str(), raw(source, link.span)?) {
            ("nsubj", ['は' | 'が']) => {}
            ("obl", ['に' | 'へ']) if destination.is_none() => {
                destination = Some((argument.token, link.token));
            }
            _ => return None,
        }
    }
    let (place, marker) = destination?;
    Some(DirectGo {
        predicate,
        place,
        marker,
        clock_marker,
    })
}

fn physical_destination(
    group: &Bunsetsu,
    going: &DirectGo,
    doc: &Document,
    source: &[char],
) -> bool {
    let [place, marker] = group.atoms.as_slice() else {
        return false;
    };
    let Some(noun) = source_head(place, doc) else {
        return false;
    };
    if place.doc_tok_idx != Some(going.place)
        || marker.doc_tok_idx != Some(going.marker)
        || place.head != Some(going.predicate)
        || noun.span != place.span
        || !matches!(noun.upos, Some(Upos::Noun | Upos::Propn))
        || source_head(marker, doc).is_none_or(|token| token.span != marker.span)
        || !dasu_case_link(place, marker)
        || !matches!(
            (raw(source, marker.span), group.case_particle.as_deref()),
            (Some(['に']), Some("に")) | (Some(['へ']), Some("へ"))
        )
        || doc.syntax.iter().any(|node| {
            node.head == Some(going.place)
                && matches!(
                    node.deprel.as_str(),
                    "acl" | "amod" | "compound" | "nmod" | "nummod"
                )
        })
    {
        return false;
    }
    if !place.is_locked {
        return physical_goal_head(group) == Some(going.predicate);
    }
    // Accept only a full Place lock on the original noun. A direction or a
    // person in an incorrectly typed glossary does not establish a destination.
    if !(place.surface.starts_with("[[PH")
        && place.lemma.as_deref() == Some(LOCKED_PLACE_ATOM)
        && matches!(noun.sem_class, None | Some(SemanticClass::Location))
        && !matches!(
            raw(source, noun.span),
            Some(['方'] | ['方', '向'] | ['方', '角'] | ['方', '面'] | ['向', 'こ', 'う'])
        ))
    {
        return false;
    }
    if noun.sem_class == Some(SemanticClass::Location) {
        return true;
    }
    let Some(surface) = raw(source, noun.span) else {
        return false;
    };
    // The glossary feeds both Actor and Place surfaces into the analyzer's
    // name list. A Name observation alone cannot decide which kind this is.
    let mut source_place = place.clone();
    source_place.surface = surface.iter().collect();
    source_place.upos = noun.upos;
    source_place.lemma.clone_from(&noun.lemma);
    source_place.is_locked = false;
    let source_group = Bunsetsu {
        atoms: vec![source_place, marker.clone()],
        role: group.role,
        case_particle: group.case_particle.clone(),
        sconj_tail: None,
        sfp_tail: None,
    };
    physical_goal_head(&source_group) == Some(going.predicate)
}

pub(super) fn resolve(
    groups: &[Bunsetsu],
    atoms: &[Atom],
    doc: &Document,
    source: &[char],
) -> Option<FinishedGo> {
    if doc.text_chars != source.len() {
        return None;
    }
    let mut predicates = groups.iter().filter(|b| b.role == BunsetsuRole::Predicate);
    let group = predicates.next()?;
    let [verb, tail @ ..] = group.atoms.as_slice() else {
        return None;
    };
    if predicates.next().is_some() || group.sconj_tail.is_some() || group.sfp_tail.is_some() {
        return None;
    }
    let predicate = completed_go(verb, tail, doc, source)?;
    let going = direct_destination(atoms, predicate, doc, source)?;
    // The renderer must have the original subject group in this same clause.
    let mut subjects = doc
        .syntax
        .iter()
        .filter(|node| node.head == Some(predicate) && node.deprel == "nsubj");
    if let Some(subject) = subjects.next()
        && (subjects.next().is_some()
            || !groups.iter().flat_map(|group| &group.atoms).any(|atom| {
                atom.doc_tok_idx == Some(subject.token) && source_head(atom, doc).is_some()
            }))
    {
        return None;
    }
    let goal_group = groups
        .iter()
        .position(|group| physical_destination(group, &going, doc, source))?;
    Some(FinishedGo {
        predicate,
        goal_group,
    })
}

/// A reading comma after a source-linked time can keep all arguments of the
/// same trip in one segment. Keep ambiguous or event-separating commas intact.
pub(super) fn reading_pause(atoms: &[Atom], doc: &Document, source: &[char]) -> Option<usize> {
    if doc.text_chars != source.len() {
        return None;
    }
    let comma = atoms
        .iter()
        .position(|atom| matches!(atom.surface.as_str(), "、" | "，"))?;
    let pause = &atoms[comma];
    let [place, marker, verb, ending @ ..] = &atoms[comma + 1..] else {
        return None;
    };
    let tail = match ending {
        [tail @ .., period] if !period.is_locked && period.surface == "。" => tail,
        _ => ending,
    };
    let predicate = completed_go(verb, tail, doc, source)?;
    let going = direct_destination(atoms, predicate, doc, source)?;
    if pause.is_locked
        || pause.deprel != "punct"
        || pause.head != Some(predicate)
        || source_head(pause, doc).is_none()
    {
        return None;
    }
    let prefix = match &atoms[..comma] {
        [prefix @ .., clock, ni]
            if attached_clock(clock, ni, predicate, doc) == going.clock_marker
                && going.clock_marker.is_some() =>
        {
            prefix
        }
        [prefix @ .., day] if relative_day(day, predicate, doc) => prefix,
        _ => return None,
    };
    if !topic_has_same_governor(prefix, predicate, doc) {
        return None;
    }
    let goal = Bunsetsu {
        atoms: vec![place.clone(), marker.clone()],
        role: BunsetsuRole::Obl,
        case_particle: Some(marker.surface.clone()),
        sconj_tail: None,
        sfp_tail: None,
    };
    physical_destination(&goal, &going, doc, source).then_some(comma)
}

fn relative_day(day: &Atom, predicate: usize, doc: &Document) -> bool {
    !day.is_locked
        && day.deprel == "advmod"
        && day.head == Some(predicate)
        && source_head(day, doc).is_some_and(|token| {
            token.span == day.span
                && (matches!(day.surface.as_str(), "今朝" | "昨夜" | "昨晩")
                    || doc.entities.iter().any(|entity| {
                        entity.kind == EntityKind::RelativeDate && entity.span == token.span
                    }))
        })
}

fn topic_has_same_governor(atoms: &[Atom], predicate: usize, doc: &Document) -> bool {
    let [prefix @ .., subject, case] = atoms else {
        return atoms.is_empty();
    };
    subject.deprel == "nsubj"
        && subject.head == Some(predicate)
        && source_head(subject, doc).is_some()
        && matches!(case.surface.as_str(), "は" | "が")
        && dasu_case_link(subject, case)
        && source_head(case, doc).is_some()
        && prefix.iter().all(|atom| {
            !atom.is_locked
                && atom.deprel == "compound"
                && atom.head == subject.doc_tok_idx
                && source_head(atom, doc).is_some()
        })
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::analyze_document;

    use super::super::{
        GlossaryKind, LockOrigin, SpanLockTable, core_non_punct, group_into_bunsetsus,
        merge_adnominal_bunsetsus, normalize_ja_atoms, normalized_sentence_atoms,
    };
    use super::*;
    use crate::{GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary};

    fn source_result(text: &str, place: Option<&str>) -> (Option<usize>, Option<String>) {
        let mut glossary = TranslationGlossary::new();
        let mut candidates = Vec::new();
        let mut lock = |canonical: &str, target: &str, kind| {
            let byte = text.find(canonical).unwrap();
            let start_char = text[..byte].chars().count();
            candidates.push((
                1,
                Span {
                    start_char,
                    end_char: start_char + canonical.chars().count(),
                },
                target.to_owned(),
                LockOrigin::Glossary {
                    canonical: canonical.to_owned(),
                    kind,
                },
                true,
            ));
            GlossaryEntry {
                canonical: canonical.to_owned(),
                surfaces: vec![canonical.to_owned()],
                targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
                ..GlossaryEntry::default()
            }
        };
        if let Some(place) = place {
            glossary
                .places
                .push(lock(place, "指定地點", GlossaryKind::Place));
            if text.contains("田中さん") {
                glossary
                    .actors
                    .push(lock("田中さん", "田中代表", GlossaryKind::Actor));
            }
        }
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let doc = analyze_document(text, &crate::doc_options_for_translate(&options, &glossary));
        let source: Vec<_> = text.chars().collect();
        let tokens: Vec<_> = doc
            .tokens
            .iter()
            .filter(|token| token.sentence == 0)
            .collect();
        let atoms =
            normalized_sentence_atoms(0, &source, &doc, &tokens, &SpanLockTable::build(candidates));
        let pause = reading_pause(&atoms, &doc, &source);
        let body: Vec<_> = core_non_punct(&atoms)
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != pause)
            .map(|(_, atom)| atom.clone())
            .collect();
        let normalized = normalize_ja_atoms(&body);
        let groups = merge_adnominal_bunsetsus(
            group_into_bunsetsus(&normalized, &doc, 0),
            &body,
            &doc,
            TargetVariant::ZhHant,
        );
        let goal = resolve(&groups, &body, &doc, &source).map(|result| {
            assert_eq!(doc.tokens[result.predicate].lemma.as_deref(), Some("行く"));
            let head = &groups[result.goal_group].atoms[0];
            raw(&source, head.span).unwrap().iter().collect()
        });
        (pause, goal)
    }

    #[test]
    fn attached_times_do_not_compete_with_a_completed_destination() {
        for (source, expected_pause, place) in [
            ("田中さんは昨日、学校へ行きました。", true, "学校"),
            ("佐藤さんは昨日図書館へ行きました。", false, "図書館"),
            ("彼女は今朝、公園へ行った。", true, "公園"),
            ("午前九時に学校へ行きました。", false, "学校"),
            ("午前九時に学校に行きました。", false, "学校"),
            ("午後二時に学校へ行きました。", false, "学校"),
            ("田中さんは午前九時に、学校へ行きました。", true, "学校"),
        ] {
            let (pause, goal) = source_result(source, None);
            assert_eq!(pause.is_some(), expected_pause, "{source}");
            assert_eq!(goal.as_deref(), Some(place), "{source}");
        }
        let (pause, goal) = source_result("田中さんは午前九時に、学校へ行きました。", Some("学校"));
        assert!(pause.is_some());
        assert_eq!(goal.as_deref(), Some("学校"));
    }

    #[test]
    fn ambiguous_times_routes_roles_and_quotes_keep_the_existing_path() {
        for (source, glossary) in [
            ("田中さんは午前九時、学校へ行きました。", None),
            ("午前九時に駅で待ち、学校へ行きました。", None),
            ("午前九時までに学校へ行きました。", None),
            ("午前九時に学校へ行きましたか。", None),
            ("午前九時に学校へ行きませんでした。", None),
            ("午前九時に学校の方へ行きました。", Some("方")),
            ("田中さんは午前九時に伊藤に行きました。", Some("伊藤")),
            ("午前九時に会議へ行きました。", Some("会議")),
            ("午前九時に学校まで行きました。", None),
            ("午前九時に学校まで歩いて行きました。", None),
            ("午前九時に学校へ本を返しに行きました。", None),
            (
                "紙に「午前九時に学校へ行きました」と一字ずつ書きました。",
                None,
            ),
        ] {
            assert_eq!(source_result(source, glossary), (None, None), "{source}");
        }
    }
}
