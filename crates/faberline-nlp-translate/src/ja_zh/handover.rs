//! Source-bound arguments of a finite handover.

use faberline_nlp::doc::{EntityKind, Token};

use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, LOCKED_OBJECT_ATOM, LOCKED_PLACE_ATOM,
    LOCKED_RESIDENT_ATOM, Segment, SemanticClass, Span, TargetVariant, Upos, classroom_role,
    dasu_case_link, daytime, honorific, is_demonstrative_person_noun, is_time_or_place_bunsetsu,
    purpose_stem_lemma, render_bunsetsu_np, return_endpoint, spatial_route_marker,
};

pub(super) struct Handover {
    pub(super) predicate: usize,
    pub(super) object_group: usize,
    pub(super) recipient_group: usize,
    pub(super) object: String,
    pub(super) recipient: String,
    pub(super) action: &'static str,
}

pub(crate) struct HandoverRenderLocks {
    pub(crate) table: SpanLockTable,
    course_time: Option<(Span, usize)>,
    subject_pause: Option<Span>,
    teacher: Option<Span>,
}

impl HandoverRenderLocks {
    pub(super) fn apply(&self, atoms: &mut [Atom]) {
        for atom in atoms {
            if self.subject_pause == Some(atom.span) {
                super::homework_relative::keep_argument_pause(atom);
            } else if self.teacher == Some(atom.span) && atom.is_locked {
                super::homework_relative::mark_locked_teacher(atom);
            } else if let Some((span, governor)) = self.course_time
                && atom.span == span
                && atom.is_locked
            {
                daytime::mark_locked_course_time(atom, governor);
            }
        }
    }
}

pub(crate) fn has_handover_glossary(text: &str, locks: &SpanLockTable) -> bool {
    text.contains('渡')
        && !text.contains(['「', '」', '『', '』'])
        && locks.locks.iter().any(|lock| {
            matches!(
                lock.origin,
                LockOrigin::Glossary {
                    kind: GlossaryKind::Actor | GlossaryKind::Object | GlossaryKind::DomainTerm,
                    ..
                }
            )
        })
}

/// Use the ordinary source parse when a caller's term hides handover syntax.
/// Only the render copy of a lock gives a grammatical particle back to
/// the parser. The original table still records the caller's full match.
pub(crate) fn handover_render_locks(
    sentence: usize,
    source: &Document,
    tokens: &[&Token],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<HandoverRenderLocks> {
    let mut roots = source
        .syntax
        .iter()
        .filter(|node| node.sentence == sentence && node.deprel == "root");
    let verb = roots.next()?.token;
    if roots.next().is_some()
        || !matches!(
            source.tokens.get(verb)?.lemma.as_deref(),
            Some("渡す" | "手渡す")
        )
    {
        return None;
    }

    let original =
        super::build_sentence_atoms(sentence, chars, source, tokens, &SpanLockTable::new());
    if original.iter().any(|atom| {
        atom.upos == Some(Upos::Verb) && atom.doc_tok_idx != Some(verb) && atom.deprel != "acl"
    }) {
        return None;
    }
    let predicate_start = original.iter().position(|a| a.doc_tok_idx == Some(verb))?;
    let predicate = super::classify_bunsetsu(
        super::core_non_punct(&original[predicate_start..]).to_vec(),
        source,
        sentence,
    );
    finite_handover(&[predicate], source)?;
    let teacher_pause = super::homework_relative::teaching_argument_pause(&original, source);
    let mut render = HandoverRenderLocks {
        table: locks.clone(),
        course_time: None,
        subject_pause: None,
        teacher: None,
    };
    for lock in &mut render.table.locks {
        let LockOrigin::Glossary { kind, .. } = lock.origin else {
            return None;
        };
        let span = lock.source_span;
        if kind == GlossaryKind::DomainTerm {
            let full = original.iter().enumerate().find_map(|(i, noun)| {
                if noun.span.start_char != span.start_char {
                    return None;
                }
                let (end, owner, _) = daytime::course_time(noun, &original[i + 1..], &original)?;
                let short = span.end_char + 1 == end.span.end_char
                    && chars.get(span.end_char) == Some(&'に');
                (owner == verb && (span.end_char == end.span.end_char || short)).then_some(Span {
                    start_char: span.start_char,
                    end_char: end.span.end_char,
                })
            })?;
            if render.course_time.replace((full, verb)).is_some() {
                return None;
            }
            lock.source_span = full;
            continue;
        }
        let first = source
            .tokens
            .partition_point(|token| token.span.start_char < span.start_char);
        let last = source
            .tokens
            .partition_point(|token| token.span.end_char <= span.end_char)
            .checked_sub(1)?;
        let covered = source.tokens.get(first..=last)?;
        if covered.first()?.span.start_char != span.start_char
            || covered.last()?.span.end_char != span.end_char
            || !covered
                .windows(2)
                .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
        {
            return None;
        }
        let tail = source.syntax.get(last)?;
        let (head, absorbed) = if tail.deprel == "case" {
            (source.syntax.get(tail.head?)?, Some(tail))
        } else {
            (tail, None)
        };
        let token = source.tokens.get(head.token)?;
        if head.sentence != sentence
            || head.head != Some(verb)
            || head.token < first
            || !matches!(token.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
        {
            return None;
        }
        let case = absorbed.or_else(|| {
            source.syntax.get(last + 1).filter(|node| {
                node.deprel == "case"
                    && node.head == Some(head.token)
                    && node.span.start_char == head.span.end_char
            })
        })?;
        if head.span.end_char != case.span.start_char
            || !matches!(
                (kind, head.deprel.as_str(), written_case(chars, case.span)),
                (GlossaryKind::Object, "obj", Some('を'))
                    | (GlossaryKind::Actor, "nsubj", Some('は' | 'が'))
                    | (GlossaryKind::Actor, "obl" | "iobj", Some('に' | 'へ'))
            )
        {
            return None;
        }
        // The caller may wrap a whole relative clause, name, or compound.
        // Each covered token must still belong to this one argument.
        for index in first..=last {
            let mut node = index;
            for _ in first..=last {
                if node == head.token {
                    break;
                }
                node = source
                    .syntax
                    .get(node)?
                    .head
                    .filter(|next| (first..=last).contains(next))?;
            }
            if node != head.token {
                return None;
            }
        }
        lock.source_span.end_char = head.span.end_char;
        if head.deprel == "nsubj"
            && let Some(pause) = teacher_pause
            && original[pause].span.start_char == case.span.end_char
        {
            render.subject_pause = Some(original[pause].span);
            if lock.source_span == head.span {
                render.teacher = Some(head.span);
            }
        }
    }
    if render
        .table
        .locks
        .windows(2)
        .any(|pair| pair[0].source_span.end_char > pair[1].source_span.start_char)
    {
        return None;
    }
    Some(render)
}

// The rendered atom can cover an entire name or glossary phrase. Its final
// original token must still prove the attachment to the case and governor.
fn written_head<'a>(atom: &Atom, doc: &'a Document) -> Option<&'a Token> {
    let index = atom.doc_tok_idx?;
    let token = doc.tokens.get(index)?;
    let syntax = doc.syntax.get(index)?;
    (syntax.token == index
        && syntax.span == token.span
        && syntax.sentence == token.sentence
        && syntax.head == atom.head
        && syntax.deprel == atom.deprel
        && atom.span.start_char <= token.span.start_char
        && atom.span.end_char == token.span.end_char)
        .then_some(token)
}

fn whole_glossary_atom(atom: &Atom, kind: &str, doc: &Document) -> bool {
    if !atom.is_locked || !atom.surface.starts_with("[[PH") || atom.lemma.as_deref() != Some(kind) {
        return false;
    }
    // The lock builder assigns these labels only to mandatory, token-aligned
    // glossaries. Check that the surviving atom still ends at its source head
    // and covers complete, adjacent tokens of the same sentence.
    let (Some(head), Some(end)) = (written_head(atom, doc), atom.doc_tok_idx) else {
        return false;
    };
    let start = doc
        .tokens
        .partition_point(|token| token.span.start_char < atom.span.start_char);
    doc.tokens.get(start..=end).is_some_and(|tokens| {
        tokens
            .first()
            .is_some_and(|token| token.span.start_char == atom.span.start_char)
            && tokens.iter().all(|token| token.sentence == head.sentence)
            && tokens
                .windows(2)
                .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
    })
}

fn finite_handover(groups: &[Bunsetsu], doc: &Document) -> Option<(usize, &'static str)> {
    let mut predicates = groups
        .iter()
        .filter(|group| group.role == BunsetsuRole::Predicate);
    let group = predicates.next()?;
    let [verb, tail @ ..] = group.atoms.as_slice() else {
        return None;
    };
    let id = verb.doc_tok_idx?;
    let original = written_head(verb, doc)?;
    if predicates.next().is_some()
        || group.sconj_tail.is_some()
        || verb.is_locked
        || verb.span != original.span
        || original.upos != Some(Upos::Verb)
        || verb.deprel != "root"
        || verb.head.is_some()
    {
        return None;
    }
    let action = match original.lemma.as_deref().unwrap_or(&verb.surface) {
        "渡す" => "交給",
        "手渡す" => "親手交給",
        _ => return None,
    };
    let tail = match tail {
        [rest @ .., question]
            if !question.is_locked
                && question.surface == "か"
                && question.deprel == "discourse"
                && question.head == Some(id)
                && written_head(question, doc).is_some() =>
        {
            rest
        }
        _ => tail,
    };
    if group
        .atoms
        .windows(2)
        .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
        || tail.iter().any(|atom| {
            atom.is_locked
                || atom.upos != Some(Upos::Aux)
                || atom.head != Some(id)
                || atom.deprel != "aux"
                || written_head(atom, doc).is_none_or(|token| token.span != atom.span)
        })
    {
        return None;
    }
    let ending: String = tail.iter().map(|atom| atom.surface.as_str()).collect();
    match ending.as_str() {
        "" if matches!(verb.surface.as_str(), "渡す" | "手渡す") => Some((id, action)),
        "た" | "ない" | "なかった" | "ます" | "ました" | "ません" | "ませんでした" => {
            Some((id, action))
        }
        _ => None,
    }
}

fn human_recipient(group: &Bunsetsu, head: &Atom, token: &Token, doc: &Document) -> bool {
    if head.is_locked && head.surface.starts_with("[[PH") {
        return whole_glossary_atom(head, LOCKED_RESIDENT_ATOM, doc);
    }
    // A person mentioned inside a place name is not its grammatical head.
    // The name must cover the source head, including a linked さん suffix.
    let has_name = doc.entities.iter().any(|entity| {
        entity.kind == EntityKind::Name
            && entity.sentence == token.sentence
            && entity.span.end_char == head.span.end_char
            && entity.span.start_char >= group.atoms[0].span.start_char
            && entity.span.start_char <= token.span.start_char
    });
    if head.is_locked {
        return honorific::is_bound_person(head) && has_name;
    }
    if is_place_or_time(head, token, doc)
        || matches!(
            token.sem_class,
            Some(SemanticClass::Organization | SemanticClass::Artifact)
        )
    {
        return false;
    }
    has_name
        || token.sem_class == Some(SemanticClass::Person)
        || token.upos == Some(Upos::Noun)
            && head.upos == Some(Upos::Noun)
            && is_demonstrative_person_noun(head)
        || token.upos == Some(Upos::Pron)
            && matches!(
                head.surface.as_str(),
                "私" | "僕" | "俺" | "あなた" | "君" | "彼" | "彼女"
            )
}

fn is_place_or_time(head: &Atom, token: &Token, doc: &Document) -> bool {
    if head.is_locked {
        return whole_glossary_atom(head, LOCKED_PLACE_ATOM, doc);
    }
    matches!(
        token.sem_class,
        Some(SemanticClass::Location | SemanticClass::Time)
    ) || is_time_or_place_bunsetsu(std::slice::from_ref(head))
}

/// Check the source object on a particular predicate, including a complete
/// Object glossary. The same check can guard a reading pause before assembly.
pub(super) fn is_explicit_object(
    head: &Atom,
    case: &Atom,
    predicate: usize,
    doc: &Document,
) -> bool {
    head.head == Some(predicate)
        && head.deprel == "obj"
        && case.surface == "を"
        && dasu_case_link(head, case)
        && written_head(case, doc).is_some_and(|token| token.span == case.span)
        && if head.is_locked {
            whole_glossary_atom(head, LOCKED_OBJECT_ATOM, doc)
        } else {
            written_head(head, doc).is_some_and(|token| {
                matches!(token.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
            })
        }
}

fn written_case(source: &[char], span: Span) -> Option<char> {
    let [marker] = source.get(span.start_char..span.end_char)? else {
        return None;
    };
    Some(*marker)
}

fn travel_place(place: &Atom, doc: &Document, source: &[char]) -> bool {
    let Some(token) = written_head(place, doc) else {
        return false;
    };
    if !matches!(token.upos, Some(Upos::Noun | Upos::Propn))
        || matches!(
            token.sem_class,
            Some(SemanticClass::Person | SemanticClass::Time)
        )
        || matches!(
            source.get(token.span.start_char..token.span.end_char),
            Some(['方'] | ['方', '向'])
        )
    {
        return false;
    }
    if place.is_locked {
        return whole_glossary_atom(place, LOCKED_PLACE_ATOM, doc);
    }
    token.span == place.span
        && (token.sem_class == Some(SemanticClass::Location)
            || return_endpoint::is_place(place)
            || place.surface == "教室")
}

fn traversed_space(place: &Atom, doc: &Document, source: &[char]) -> bool {
    let Some(token) = written_head(place, doc) else {
        return false;
    };
    if token.upos != Some(Upos::Noun)
        || token.span != place.span
        || place.is_locked && !whole_glossary_atom(place, LOCKED_PLACE_ATOM, doc)
    {
        return false;
    }
    // A locked route hides the noun. Use its original surface so the same
    // spatial vocabulary applies to locked and unlocked routes.
    source
        .get(token.span.start_char..token.span.end_char)
        .is_some_and(|chars| spatial_route_marker(&chars.iter().collect::<String>()).is_some())
}

fn travel_past(past: &[Atom], motion_id: usize, doc: &Document) -> bool {
    let affirmative = match past {
        [ta] => ta.surface == "た",
        [mas, ta] => mas.surface == "まし" && ta.surface == "た",
        _ => false,
    };
    affirmative
        && past.iter().all(|atom| {
            !atom.is_locked
                && atom.head == Some(motion_id)
                && atom.deprel == "aux"
                && written_head(atom, doc)
                    .is_some_and(|token| token.span == atom.span && token.upos == Some(Upos::Aux))
        })
}

// A bare 渡し before a comma can be tagged as a noun modifying the next
// place. Recover the continuative only from a complete left argument frame
// and an independently anchored finite movement on the right.
pub(super) fn before_travel(
    atoms: &[Atom],
    following: &[Segment],
    doc: &Document,
    source: &[char],
) -> Option<String> {
    let [
        speaker,
        topic,
        first,
        first_case,
        second,
        second_case,
        hand,
        continuation @ ..,
    ] = atoms
    else {
        return None;
    };
    let [
        Segment::Clause {
            atoms: travel,
            trailing_punct,
            ..
        },
    ] = following
    else {
        return None;
    };
    let [place, case, motion, past @ ..] = travel.as_slice() else {
        return None;
    };
    let motion_id = motion.doc_tok_idx?;
    let motion_source = written_head(motion, doc)?;
    if motion.is_locked
        || motion.span != motion_source.span
        || motion_source.upos != Some(Upos::Verb)
        || motion.deprel != "root"
        || motion.head.is_some()
        || place.head != Some(motion_id)
        || !dasu_case_link(place, case)
        || written_head(case, doc).is_none()
        || !matches!(trailing_punct.as_deref(), None | Some("。"))
        || !travel_past(past, motion_id, doc)
        || travel
            .windows(2)
            .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
    {
        return None;
    }
    let path_is_proven = match (
        motion_source.lemma.as_deref(),
        place.deprel.as_str(),
        case.surface.as_str(),
    ) {
        (Some("行く"), "obl", "へ" | "に") => travel_place(place, doc, source),
        (Some("渡る"), "obj", "を") => traversed_space(place, doc, source),
        _ => false,
    };
    if !path_is_proven {
        return None;
    }
    let hand_id = hand.doc_tok_idx?;
    let hand_source = written_head(hand, doc)?;
    if hand.is_locked || hand.surface != "渡し" || hand.span != hand_source.span {
        return None;
    }
    let (owner, end) = match continuation {
        [] if hand_source.upos == Some(Upos::Noun)
            && hand.deprel == "nmod"
            && hand.head == place.doc_tok_idx
            && purpose_stem_lemma(hand).as_deref() == Some("渡す") =>
        {
            (motion_id, hand.span.end_char)
        }
        [te] if hand_source.upos == Some(Upos::Verb)
            && hand_source.lemma.as_deref() == Some("渡す")
            && hand.deprel == "advcl"
            && hand.head == Some(motion_id)
            && !te.is_locked
            && te.surface == "て"
            && te.deprel == "mark"
            && te.head == Some(hand_id)
            && written_head(te, doc).is_some() =>
        {
            (hand_id, te.span.end_char)
        }
        _ => return None,
    };
    let sentence = doc.segmentation.sentences.get(motion_source.sentence)?.span;
    if atoms[0].span.start_char != sentence.start_char
        || atoms
            .windows(2)
            .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
        || !matches!(source.get(end..place.span.start_char), Some(['、' | '，']))
        || !doc.syntax.iter().any(|node| {
            node.span.start_char == end
                && node.span.end_char == place.span.start_char
                && node.head == Some(owner)
                && node.deprel == "punct"
        })
    {
        return None;
    }
    let (object, wo, receiver, dative) = if first_case.surface == "を" {
        (first, first_case, second, second_case)
    } else {
        (second, second_case, first, first_case)
    };
    if speaker.head != Some(owner)
        || speaker.deprel != "nsubj"
        || !matches!(topic.surface.as_str(), "は" | "が")
        || !dasu_case_link(speaker, topic)
        || written_head(topic, doc).is_none()
        || !is_explicit_object(object, wo, owner, doc)
        || receiver.head != Some(owner)
        || !matches!(receiver.deprel.as_str(), "obl" | "iobj")
        || !matches!(dative.surface.as_str(), "に" | "へ")
        || !dasu_case_link(receiver, dative)
        || written_head(dative, doc).is_none()
    {
        return None;
    }
    let nominal = |head: &Atom, case: &Atom, role| Bunsetsu {
        atoms: vec![head.clone(), case.clone()],
        role,
        case_particle: Some(case.surface.clone()),
        sconj_tail: None,
        sfp_tail: None,
    };
    let mover = nominal(speaker, topic, BunsetsuRole::Subj);
    let recipient = nominal(receiver, dative, BunsetsuRole::IObj);
    let item = nominal(object, wo, BunsetsuRole::Obj);
    if !human_recipient(&mover, speaker, written_head(speaker, doc)?, doc)
        || !human_recipient(&recipient, receiver, written_head(receiver, doc)?, doc)
    {
        return None;
    }
    let render_person = |group, atom: &Atom| {
        if !atom.is_locked
            && let Some((_, name)) = classroom_role(&[], atom)
        {
            return name.to_owned();
        }
        render_bunsetsu_np(group, TargetVariant::ZhHant)
    };
    let mover = render_person(&mover, speaker);
    let item = render_bunsetsu_np(&item, TargetVariant::ZhHant);
    let recipient = render_person(&recipient, receiver);
    if mover.is_empty() || item.is_empty() || recipient.is_empty() {
        return None;
    }
    Some(format!("{mover}把{item}交給{recipient}"))
}

/// Resolve both arguments before the caller consumes either group. This
/// leaves time, voice, modality and quotation rendering with the assembler.
pub(super) fn resolve(
    groups: &[Bunsetsu],
    source: &[char],
    doc: &Document,
    target: TargetVariant,
) -> Option<Handover> {
    if target != TargetVariant::ZhHant || source.len() != doc.text_chars {
        return None;
    }
    let (predicate, action) = finite_handover(groups, doc)?;
    let mut object_nodes = doc
        .syntax
        .iter()
        .filter(|node| node.head == Some(predicate) && node.deprel == "obj");
    let only_object = object_nodes.next()?.token;
    if object_nodes.next().is_some() {
        return None;
    }
    // A separate counter needs its own object/count assembly. Keep it on
    // the current path rather than moving only the uncounted noun.
    if doc.syntax.iter().any(|node| {
        node.deprel == "nummod"
            && node
                .head
                .and_then(|index| doc.syntax.get(index))
                .is_some_and(|counter| counter.deprel == "obl" && counter.head == Some(predicate))
    }) {
        return None;
    }
    let mut subjects = doc
        .syntax
        .iter()
        .filter(|node| node.head == Some(predicate) && node.deprel == "nsubj");
    if let Some(subject) = subjects.next()
        && (subjects.next().is_some()
            || !groups.iter().flat_map(|group| &group.atoms).any(|atom| {
                atom.doc_tok_idx == Some(subject.token) && written_head(atom, doc).is_some()
            }))
    {
        return None;
    }
    let mut object_group = None;
    let mut recipient_group = None;
    for source_case in doc.syntax.iter().filter(|node| node.deprel == "case") {
        let Some(argument) = source_case.head.and_then(|index| doc.syntax.get(index)) else {
            continue;
        };
        let Some(marker) = written_case(source, source_case.span) else {
            continue;
        };
        if matches!(marker, 'と' | 'や')
            && (argument.head == Some(predicate)
                || argument.deprel == "conj"
                    && argument
                        .head
                        .and_then(|index| doc.syntax.get(index))
                        .is_some_and(|head| head.head == Some(predicate)))
        {
            return None;
        }
        if argument.head != Some(predicate) || !matches!(marker, 'を' | 'に' | 'へ') {
            continue;
        }
        // Search by original case id. An argument hidden by a comma cannot
        // be inferred from a rendered name in another clause.
        let mut rendered = groups.iter().enumerate().filter_map(|(i, group)| {
            let [prefix @ .., head, case] = group.atoms.as_slice() else {
                return None;
            };
            (case.doc_tok_idx == Some(source_case.token)).then_some((i, group, prefix, head, case))
        });
        let Some((index, group, prefix, head, case)) = rendered.next() else {
            if marker == 'に'
                && groups.iter().flat_map(|group| &group.atoms).any(|atom| {
                    daytime::is_modifier(atom)
                        && atom.head == Some(predicate)
                        && atom.span.start_char <= argument.span.start_char
                        && atom.span.end_char == source_case.span.end_char
                })
            {
                continue;
            }
            return None;
        };
        let token = written_head(head, doc)?;
        let case_token = written_head(case, doc)?;
        if rendered.next().is_some()
            || !dasu_case_link(head, case)
            || case_token.sentence != token.sentence
            || case_token.span != case.span
            || !matches!(
                (marker, case.surface.as_str()),
                ('を', "を") | ('に', "に") | ('へ', "へ" | "に")
            )
            || prefix.iter().any(|atom| {
                atom.is_locked
                    && !(atom.doc_tok_idx.is_none()
                        && atom.upos.is_none()
                        && !atom.surface.contains("[[PH"))
            })
        {
            return None;
        }
        if marker == 'を' {
            if head.doc_tok_idx != Some(only_object)
                || !is_explicit_object(head, case, predicate, doc)
                || object_group.replace(index).is_some()
            {
                return None;
            }
        } else if !matches!(head.deprel.as_str(), "obl" | "iobj") {
            return None;
        } else if human_recipient(group, head, token, doc) {
            if recipient_group.replace(index).is_some() {
                return None;
            }
        } else if !is_place_or_time(head, token, doc) {
            // An unresolved に／へ on this same action can compete for the
            // recipient slot. Its text is not evidence for a person.
            return None;
        }
    }
    let object_group = object_group?;
    let recipient_group = recipient_group?;
    let object = render_bunsetsu_np(&groups[object_group], target);
    let recipient = render_bunsetsu_np(&groups[recipient_group], target);
    if object.is_empty() || recipient.is_empty() {
        return None;
    }
    Some(Handover {
        predicate,
        object_group,
        recipient_group,
        object,
        recipient,
        action,
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
    };
    use faberline_nlp::doc::model::Backend;

    const HS_12: &str = "授業を担当する先生は、授業後に佐藤さんへ資料を渡しました。";

    #[test]
    fn a_handover_keeps_its_owner_recipient_and_polarity() {
        let glossary = TranslationGlossary::new();
        for hint in [None, Some(TargetVariant::Ja)] {
            let mut options = TranslateOptions::new(TargetVariant::ZhHant);
            options.source_hint = hint;
            for (source, expected) in [
                (HS_12, "負責授課的老師下課後把資料交給佐藤。"),
                ("先生は佐藤さんへ資料を渡しました。", "老師把資料交給佐藤。"),
                ("先生は佐藤さんに資料を渡しました。", "老師把資料交給佐藤。"),
                ("佐藤さんは先生へ資料を渡しました。", "佐藤把資料交給老師。"),
                (
                    "先生は佐藤さんに資料を渡しませんでした。",
                    "老師沒有把資料交給佐藤。",
                ),
                ("先生は川を渡りました。", "老師渡過了川。"),
                ("先生は佐藤さんの家へ行きました。", "老師往佐藤的家去了。"),
                (
                    "先生は佐藤さんに資料を渡してもらいました。",
                    "佐藤幫老師遞給了資料。",
                ),
            ] {
                let result =
                    analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                        .unwrap();
                assert_eq!(result.target_text, expected, "{source}");
            }
            let literal = "「佐藤さんへ資料を渡す」";
            let source = format!("紙に{literal}と一字ずつ写しました。");
            let result =
                analyze_and_translate(&source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert!(result.target_text.contains(literal));
        }
    }

    #[test]
    fn a_whole_object_and_actor_glossary_preserve_the_handover() {
        let mut glossary = TranslationGlossary::new();
        let entry = |source: &str, target: &str| GlossaryEntry {
            canonical: source.to_owned(),
            surfaces: vec![source.to_owned()],
            targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
            ..GlossaryEntry::default()
        };
        glossary.actors.push(entry("佐藤さん", "佐藤代表"));
        glossary.objects.push(entry("資料", "講義材料"));
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let result =
            analyze_and_translate(HS_12, &options, &glossary, None, Backend::Onnx).unwrap();
        assert_eq!(
            result.target_text,
            "負責授課的老師下課後把講義材料交給佐藤代表。"
        );
        let slice = |text: &str, start, end| {
            text.chars()
                .skip(start)
                .take(end - start)
                .collect::<String>()
        };
        let hits: Vec<_> = result
            .glossary_hits
            .iter()
            .map(|hit| {
                (
                    slice(HS_12, hit.source_span.start_char, hit.source_span.end_char),
                    slice(
                        &result.target_text,
                        hit.target_span.start_char,
                        hit.target_span.end_char,
                    ),
                )
            })
            .collect();
        assert_eq!(hits.len(), 2);
        assert!(
            hits.iter()
                .any(|(src, dst)| src == "佐藤さん" && dst == "佐藤代表")
        );
        assert!(
            hits.iter()
                .any(|(src, dst)| src == "資料" && dst == "講義材料")
        );
    }

    #[test]
    fn coordinated_actions_keep_the_handover_and_travel_roles_separate() {
        let glossary = TranslationGlossary::new();
        for hint in [None, Some(TargetVariant::Ja)] {
            let mut options = TranslateOptions::new(TargetVariant::ZhHant);
            options.source_hint = hint;
            for (source, expected) in [
                (
                    "先生は生徒に資料を渡し、教室へ行きました。",
                    "老師把資料交給學生，往教室去了。",
                ),
                (
                    "先生は資料を生徒に渡し、図書館へ行きました。",
                    "老師把資料交給學生，往圖書館去了。",
                ),
                (
                    "先生は生徒に資料を渡して、教室へ行きました。",
                    "老師把資料交給學生，往教室去了。",
                ),
                (
                    "先生は生徒に資料を渡し、橋を渡りました。",
                    "老師把資料交給學生，渡過了橋。",
                ),
                (
                    "先生は資料を生徒に渡し、川を渡りました。",
                    "老師把資料交給學生，渡過了川。",
                ),
                (
                    "先生は生徒に資料を渡して、橋を渡りました。",
                    "老師把資料交給學生，渡過了橋。",
                ),
                (
                    "先生は資料を渡し、教室へ行きました。",
                    "老師渡し資料，往教室去了。",
                ),
                (
                    "先生は生徒に資料を渡さず、教室へ行きました。",
                    "老師給生徒不遞給資料，往教室去了。",
                ),
                (
                    "先生は生徒に資料を渡さず、橋を渡りました。",
                    "老師給生徒不遞給資料，渡過了橋。",
                ),
                (
                    "先生は生徒に資料を渡し、資料を渡りました。",
                    "老師給生徒渡し資料，渡過了資料。",
                ),
                (
                    "先生は橋を渡って図書館へ行きました。",
                    "老師渡過橋，往圖書館去了。",
                ),
                (
                    "先生は生徒に資料を渡し、教室の方へ行きました。",
                    "老師給生徒渡し資料，往教室的位去了。",
                ),
                ("私は渡しの近くへ行きました。", "我往渡し的近近地去了。"),
            ] {
                let result =
                    analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                        .unwrap();
                assert_eq!(result.target_text, expected, "{source}");
            }
            let literal = "「先生は生徒に資料を渡し、教室へ行きました」";
            let source = format!("紙に{literal}と一字ずつ写しました。");
            let result =
                analyze_and_translate(&source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert!(result.target_text.contains(literal));
        }
    }

    #[test]
    fn coordinated_actions_keep_all_glossary_spans() {
        let mut glossary = TranslationGlossary::new();
        let entry = |source: &str, target: &str| GlossaryEntry {
            canonical: source.to_owned(),
            surfaces: vec![source.to_owned()],
            targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
            ..GlossaryEntry::default()
        };
        glossary.actors.push(entry("先生", "教師甲"));
        glossary.actors.push(entry("生徒", "學生乙"));
        glossary.objects.push(entry("資料", "指定資料"));
        glossary.places.push(entry("教室", "指定教室"));
        glossary.places.push(entry("橋", "指定橋"));
        let slice = |text: &str, start, end| {
            text.chars()
                .skip(start)
                .take(end - start)
                .collect::<String>()
        };
        for hint in [None, Some(TargetVariant::Ja)] {
            let mut options = TranslateOptions::new(TargetVariant::ZhHant);
            options.source_hint = hint;
            for (source, movement, route, glossed_route) in [
                (
                    "先生は生徒に資料を渡し、教室へ行きました。",
                    "往指定教室去了",
                    "教室",
                    "指定教室",
                ),
                (
                    "先生は資料を生徒に渡し、教室へ行きました。",
                    "往指定教室去了",
                    "教室",
                    "指定教室",
                ),
                (
                    "先生は生徒に資料を渡して、教室へ行きました。",
                    "往指定教室去了",
                    "教室",
                    "指定教室",
                ),
                (
                    "先生は生徒に資料を渡し、橋を渡りました。",
                    "渡過了指定橋",
                    "橋",
                    "指定橋",
                ),
            ] {
                let output =
                    analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                        .unwrap();
                assert_eq!(
                    output.target_text,
                    format!("教師甲把指定資料交給學生乙，{movement}。"),
                    "{source}"
                );
                let mut hits: Vec<_> = output
                    .glossary_hits
                    .iter()
                    .map(|hit| {
                        (
                            slice(source, hit.source_span.start_char, hit.source_span.end_char),
                            slice(
                                &output.target_text,
                                hit.target_span.start_char,
                                hit.target_span.end_char,
                            ),
                        )
                    })
                    .collect();
                hits.sort();
                let mut expected = [
                    ("先生".to_owned(), "教師甲".to_owned()),
                    (route.to_owned(), glossed_route.to_owned()),
                    ("生徒".to_owned(), "學生乙".to_owned()),
                    ("資料".to_owned(), "指定資料".to_owned()),
                ];
                expected.sort();
                assert_eq!(hits, expected, "{source}");
            }
        }
    }
}
