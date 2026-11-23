//! Source evidence for one completed trip to a physical destination.

use faberline_nlp::doc::{EntityKind, Token};

use crate::{
    GlossaryKind,
    lock::{LockOrigin, SpanLockTable},
};

use super::{
    Atom, Bunsetsu, BunsetsuRole, CLOCK_ATOM, Document, LOCKED_PLACE_ATOM, LOCKED_RESIDENT_ATOM,
    Segment, SemanticClass, Span, Upos, clock_case_governor, dasu_case_link, group_into_bunsetsus,
    is_clock_atom, is_companion_person, physical_goal_head, return_companion_owner,
    source_time_head, split_atoms_into_segments,
};

#[cfg(test)]
mod companion_tests;

pub(super) struct FinishedGo {
    pub(super) predicate: usize,
    pub(super) goal_group: usize,
}

const READING_PAUSE: &str = "__ja_finished_go_reading_pause";

pub(super) fn mark_reading_pause(atoms: &mut [Atom], doc: &Document, source: &[char]) {
    if let Some(index) = reading_pause(atoms, doc, source) {
        atoms[index].lemma = Some(READING_PAUSE.to_owned());
    }
}

pub(super) fn is_reading_pause(atom: &Atom) -> bool {
    atom.lemma.as_deref() == Some(READING_PAUSE)
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
    is_companion_case: impl Fn(usize) -> bool,
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
        if argument.head != Some(predicate)
            || Some(link.token) == clock_marker
            || is_companion_case(link.token)
        {
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

fn source_destination<'a>(
    group: &'a Bunsetsu,
    going: &DirectGo,
    doc: &'a Document,
    source: &[char],
) -> Option<(&'a Atom, &'a Atom, &'a Token)> {
    let [place, marker] = group.atoms.as_slice() else {
        return None;
    };
    let noun = source_head(place, doc)?;
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
        return None;
    }
    Some((place, marker, noun))
}

fn physical_destination(
    group: &Bunsetsu,
    going: &DirectGo,
    doc: &Document,
    source: &[char],
) -> bool {
    let Some((place, marker, noun)) = source_destination(group, going, doc, source) else {
        return false;
    };
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

fn companion_market_destination(
    group: &Bunsetsu,
    going: &DirectGo,
    doc: &Document,
    source: &[char],
) -> bool {
    // 市場 has no Location tag. With a proved companion, the direct に/へ
    // argument of completed 行く still identifies the market as a place.
    source_destination(group, going, doc, source).is_some_and(|(place, _, _)| {
        raw(source, place.span) == Some(&['市', '場'])
            && (!place.is_locked || place.lemma.as_deref() == Some(LOCKED_PLACE_ATOM))
    })
}

fn completed_predicate(groups: &[Bunsetsu], doc: &Document, source: &[char]) -> Option<usize> {
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
    completed_go(verb, tail, doc, source)
}

fn has_local_subject(groups: &[Bunsetsu], predicate: usize, doc: &Document) -> bool {
    // The renderer must have the original subject group in this same clause.
    let mut subjects = doc
        .syntax
        .iter()
        .filter(|node| node.head == Some(predicate) && node.deprel == "nsubj");
    subjects.next().is_none_or(|subject| {
        subjects.next().is_none()
            && groups.iter().flat_map(|group| &group.atoms).any(|atom| {
                atom.doc_tok_idx == Some(subject.token) && source_head(atom, doc).is_some()
            })
    })
}

fn original_actor(atom: &Atom, doc: &Document, source: &[char]) -> Option<Atom> {
    if !atom.is_locked || atom.lemma.as_deref() != Some(LOCKED_RESIDENT_ATOM) {
        return None;
    }
    let token = source_head(atom, doc)?;
    if token.span != atom.span {
        return None;
    }
    let mut original = atom.clone();
    original.surface = raw(source, atom.span)?.iter().collect();
    original.upos = token.upos;
    original.lemma.clone_from(&token.lemma);
    original.is_locked = false;
    Some(original)
}

fn companion_owner(
    noun: &Atom,
    marker: &Atom,
    atoms: &[Atom],
    doc: &Document,
    source: &[char],
) -> Option<usize> {
    let original = original_actor(noun, doc, source);
    if !is_companion_person(original.as_ref().unwrap_or(noun)) {
        return None;
    }
    let position = atoms
        .iter()
        .position(|atom| atom.doc_tok_idx == marker.doc_tok_idx)?;
    if original.is_none() {
        return return_companion_owner(atoms, position);
    }
    // Check the original actor roles while keeping placeholders in the render atoms.
    let evidence: Vec<_> = atoms
        .iter()
        .map(|atom| original_actor(atom, doc, source).unwrap_or_else(|| atom.clone()))
        .collect();
    return_companion_owner(&evidence, position)
}

pub(super) fn mark_companions(
    groups: &mut [Bunsetsu],
    atoms: &[Atom],
    doc: &Document,
    source: &[char],
) {
    let Some(predicate) = completed_predicate(groups, doc, source) else {
        return;
    };
    let mut person = None;
    let mut together = None;
    for (index, group) in groups.iter().enumerate() {
        let [noun, marker] = group.atoms.as_slice() else {
            continue;
        };
        if noun.head != Some(predicate)
            || noun.deprel != "obl"
            || source_head(noun, doc).is_none_or(|original| original.span != noun.span)
            || source_head(marker, doc).is_none_or(|original| original.span != marker.span)
            || !dasu_case_link(noun, marker)
        {
            continue;
        }
        let candidate = (index, marker.doc_tok_idx);
        let previous = match (raw(source, noun.span), raw(source, marker.span)) {
            (_, Some(['と']))
                if companion_owner(noun, marker, atoms, doc, source) == Some(predicate) =>
            {
                person.replace(candidate)
            }
            (Some(['一', '緒']), Some(['に'])) if !noun.is_locked => together.replace(candidate),
            _ => continue,
        };
        if previous.is_some() {
            return;
        }
    }
    if person.is_none() && together.is_none() {
        return;
    }
    let allows = |id| {
        person.is_some_and(|(_, marker)| marker == Some(id))
            || together.is_some_and(|(_, marker)| marker == Some(id))
    };
    let Some(going) = direct_destination(atoms, predicate, doc, source, allows) else {
        return;
    };
    if !has_local_subject(groups, predicate, doc)
        || !groups.iter().any(|group| {
            physical_destination(group, &going, doc, source)
                || companion_market_destination(group, &going, doc, source)
                // へ proves a direction even for a noun without a location
                // tag. Keep its existing directional order in that case.
                || group.case_particle.as_deref() == Some("へ")
                    && source_destination(group, &going, doc, source)
                        .is_some_and(|(place, _, _)| !place.is_locked && !source_time_head(place))
        })
    {
        return;
    }
    if let Some((index, _)) = person {
        groups[index].role = BunsetsuRole::TravelCompanion;
    }
    if let Some((index, _)) = together {
        groups[index].role = BunsetsuRole::TravelTogether;
    }
}

pub(super) fn resolve(
    groups: &[Bunsetsu],
    atoms: &[Atom],
    doc: &Document,
    source: &[char],
) -> Option<FinishedGo> {
    let predicate = completed_predicate(groups, doc, source)?;
    let going = direct_destination(atoms, predicate, doc, source, |id| {
        groups.iter().any(|group| {
            is_companion_group(group)
                && group.atoms.last().and_then(|atom| atom.doc_tok_idx) == Some(id)
        })
    })?;
    if !has_local_subject(groups, predicate, doc) {
        return None;
    }
    let has_companions = groups.iter().any(is_companion_group);
    let goal_group = groups.iter().position(|group| {
        physical_destination(group, &going, doc, source)
            || has_companions && companion_market_destination(group, &going, doc, source)
    })?;
    Some(FinishedGo {
        predicate,
        goal_group,
    })
}

fn is_companion_group(group: &Bunsetsu) -> bool {
    matches!(
        group.role,
        BunsetsuRole::TravelCompanion | BunsetsuRole::TravelTogether
    )
}

/// Expose the original role markers while keeping the caller's replacements.
pub(super) fn restore_companion_locks(
    sentence: usize,
    atoms: &[Atom],
    doc: &Document,
    source: &[char],
    locks: &SpanLockTable,
) -> Option<SpanLockTable> {
    let span = doc.segmentation.sentences.get(sentence)?.span;
    let clauses = split_atoms_into_segments(atoms, doc, span);
    let [Segment::Clause { atoms, .. }] = clauses.as_slice() else {
        return None;
    };
    let mut groups = group_into_bunsetsus(atoms, doc, sentence);
    mark_companions(&mut groups, atoms, doc, source);
    if !groups.iter().any(is_companion_group) {
        return None;
    }
    let going = resolve(&groups, atoms, doc, source)?;
    let mut render = locks.clone();
    let mut actor = false;
    let mut place = false;
    let mut exposed_marker = false;
    for lock in render.locks.iter_mut().filter(|lock| {
        lock.source_span.start_char < span.end_char && span.start_char < lock.source_span.end_char
    }) {
        if !lock.mandatory {
            return None;
        }
        let group = match lock.origin {
            LockOrigin::Glossary {
                kind: GlossaryKind::Actor,
                ..
            } if !actor => {
                actor = true;
                groups
                    .iter()
                    .find(|group| group.role == BunsetsuRole::TravelCompanion)?
            }
            LockOrigin::Glossary {
                kind: GlossaryKind::Place,
                ..
            } if !place => {
                place = true;
                &groups[going.goal_group]
            }
            _ => return None,
        };
        let [noun, marker] = group.atoms.as_slice() else {
            return None;
        };
        let end = lock.source_span.end_char;
        if lock.source_span.start_char != noun.span.start_char
            || (end != noun.span.end_char && end != marker.span.end_char)
        {
            return None;
        }
        exposed_marker |= lock.source_span != noun.span;
        lock.source_span = noun.span;
    }
    (actor || exposed_marker).then_some(render)
}

/// A reading comma after a source-linked time can keep all arguments of the
/// same trip in one segment. Keep ambiguous or event-separating commas intact.
fn reading_pause(atoms: &[Atom], doc: &Document, source: &[char]) -> Option<usize> {
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
    let going = direct_destination(atoms, predicate, doc, source, |_| false)?;
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
    use faberline_nlp::doc::model::Backend;

    use super::super::{
        GlossaryKind, LockOrigin, SpanLockTable, core_non_punct, group_into_bunsetsus,
        merge_adnominal_bunsetsus, normalize_ja_atoms, normalized_sentence_atoms,
    };
    use super::*;
    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
    };

    fn glossary_entry(canonical: &str, target: &str) -> GlossaryEntry {
        GlossaryEntry {
            canonical: canonical.to_owned(),
            surfaces: vec![canonical.to_owned()],
            targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
            ..GlossaryEntry::default()
        }
    }

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
            glossary_entry(canonical, target)
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

    #[test]
    fn completed_trips_use_their_own_subject_clock_and_goal() {
        let glossary = TranslationGlossary::new();
        for hint in [None, Some(TargetVariant::Ja)] {
            let mut options = TranslateOptions::new(TargetVariant::ZhHant);
            options.source_hint = hint;
            for (source, expected) in [
                ("田中さんは昨日、学校へ行きました。", "田中昨天去了學校。"),
                ("午前九時に学校へ行った。", "在上午九點去了學校。"),
                ("午前九時に学校に行きました。", "在上午九點去了學校。"),
                ("午後二時に学校に行きました。", "在下午兩點去了學校。"),
                (
                    "田中さんは午前九時に、学校へ行きました。",
                    "田中在上午九點去了學校。",
                ),
                (
                    "田中さんは午前九時、学校へ行きました。",
                    "田中上午九點，往學校去了。",
                ),
                (
                    "午前九時に駅で待ち、学校へ行きました。",
                    "在上午九點在車站等待，往學校去了。",
                ),
                ("私は学校の方へ行きました。", "我往學校的位去了。"),
                ("昨日学校へ本を返しに行きました。", "昨天往學校去還書了。"),
                (
                    "午前九時までに学校へ行きました。",
                    "最晚在上午九點往學校去了。",
                ),
                (
                    "先生は生徒に資料を渡し、教室へ行きました。",
                    "老師把資料交給學生，往教室去了。",
                ),
                (
                    "紙に「図書館へ行きました」と一字ずつ書きました。",
                    "給紙，「図書館へ行きました」一字ずつ寫了。",
                ),
            ] {
                let result =
                    analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                        .unwrap();
                assert_eq!(result.target_text, expected, "{source}");
            }
        }
    }

    #[test]
    fn completed_trips_keep_original_actor_and_place_glossary_spans() {
        let mut glossary = TranslationGlossary::new();
        glossary.actors.push(glossary_entry("田中さん", "田中代表"));
        glossary.places.push(glossary_entry("学校", "甲學校"));
        let source = "田中さんは午前九時に、学校へ行きました。";
        for hint in [None, Some(TargetVariant::Ja)] {
            let mut options = TranslateOptions::new(TargetVariant::ZhHant);
            options.source_hint = hint;
            let result =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(result.target_text, "田中代表在上午九點去了甲學校。");
            let slice = |text: &str, span: Span| {
                text.chars()
                    .skip(span.start_char)
                    .take(span.end_char - span.start_char)
                    .collect::<String>()
            };
            let hits: Vec<_> = result
                .glossary_hits
                .iter()
                .map(|hit| {
                    (
                        slice(source, hit.source_span),
                        slice(&result.target_text, hit.target_span),
                    )
                })
                .collect();
            assert_eq!(
                hits,
                [
                    ("田中さん".to_owned(), "田中代表".to_owned()),
                    ("学校".to_owned(), "甲學校".to_owned()),
                ]
            );
        }
    }
}
