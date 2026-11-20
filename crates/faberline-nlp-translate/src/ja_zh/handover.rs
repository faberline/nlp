//! Source-bound arguments of a finite handover.

use faberline_nlp::doc::{EntityKind, Token};

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, LOCKED_OBJECT_ATOM, LOCKED_PLACE_ATOM,
    LOCKED_RESIDENT_ATOM, SemanticClass, Span, TargetVariant, Upos, dasu_case_link, daytime,
    honorific, is_demonstrative_person_noun, is_time_or_place_bunsetsu, render_bunsetsu_np,
};

pub(super) struct Handover {
    pub(super) predicate: usize,
    pub(super) object_group: usize,
    pub(super) recipient_group: usize,
    pub(super) object: String,
    pub(super) recipient: String,
    pub(super) action: &'static str,
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
