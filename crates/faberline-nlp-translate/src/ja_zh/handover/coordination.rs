//! Recover separate handover roles within their written clause boundaries.

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, GlossaryKind, LockOrigin, Span, SpanLockTable,
    TargetVariant, Token, Upos, classroom_role, dasu_case_link, human_recipient,
    purpose_stem_lemma, render_bunsetsu_np, travel_past, written_head,
};
use crate::ja_zh::{apply_phrase_matches, build_sentence_atoms, normalize_ja_atoms};
use crate::schema::ClauseAlignment;

#[derive(Clone, Copy)]
struct Argument<'a> {
    words: &'a [Atom],
    head: &'a Atom,
    case: &'a Atom,
    kind: GlossaryKind,
}

impl Argument<'_> {
    fn span(self) -> Span {
        Span {
            start_char: self.words[0].span.start_char,
            end_char: self.head.span.end_char,
        }
    }
}

#[derive(Default)]
struct Arguments<'a> {
    actor: Option<Argument<'a>>,
    object: Option<Argument<'a>>,
    recipient: Option<Argument<'a>>,
}

fn arguments(atoms: &[Atom], owner: usize) -> Option<Arguments<'_>> {
    let mut result = Arguments::default();
    for phrase in atoms.split_inclusive(|atom| {
        atom.deprel == "case" && matches!(atom.surface.as_str(), "が" | "は" | "を" | "に" | "へ")
    }) {
        let [words @ .., case] = phrase else {
            return None;
        };
        let head = words.last()?;
        if !dasu_case_link(head, case) || head.head != Some(owner) {
            return None;
        }
        let (slot, kind) = match (head.deprel.as_str(), case.surface.as_str()) {
            ("nsubj", "は" | "が") => (&mut result.actor, GlossaryKind::Actor),
            ("obj", "を") => (&mut result.object, GlossaryKind::Object),
            ("obl" | "iobj", "に" | "へ") => (&mut result.recipient, GlossaryKind::Actor),
            _ => return None,
        };
        let head_id = head.doc_tok_idx?;
        for atom in words {
            let nominal = matches!(
                atom.upos,
                Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::Det | Upos::Num)
            );
            if !(nominal || atom.deprel == "case" && atom.surface == "の") {
                return None;
            }
            let mut id = atom.doc_tok_idx?;
            for _ in 0..words.len() {
                if id == head_id {
                    break;
                }
                id = words
                    .iter()
                    .find(|word| word.doc_tok_idx == Some(id))?
                    .head?;
            }
            if id != head_id {
                return None;
            }
        }
        if slot
            .replace(Argument {
                words,
                head,
                case,
                kind,
            })
            .is_some()
        {
            return None;
        }
    }
    Some(result)
}

fn nominal(atoms: Vec<Atom>) -> Bunsetsu {
    Bunsetsu {
        atoms,
        role: BunsetsuRole::Other,
        case_particle: None,
        sconj_tail: None,
        sfp_tail: None,
    }
}

fn words_in(atoms: &[Atom], span: Span) -> Option<&[Atom]> {
    let first = atoms.partition_point(|atom| atom.span.start_char < span.start_char);
    let end = atoms.partition_point(|atom| atom.span.end_char <= span.end_char);
    let words = atoms.get(first..end)?;
    (words.first()?.span.start_char == span.start_char
        && words.last()?.span.end_char == span.end_char)
        .then_some(words)
}

fn role_name(role: &Atom) -> Option<&str> {
    classroom_role(&[], role)
        .map(|(_, name)| name)
        .or(match role.surface.as_str() {
            "客" | "お客" => Some("客人"),
            "店員" => Some("店員"),
            _ => None,
        })
}

// A role followed by の can identify a named person. Its dependent must
// belong to that name, not to a place or object outside this argument.
fn apposition(words: &[Atom]) -> Option<(&Atom, &[Atom], bool)> {
    let (role, no, name, other) = match words {
        [other, genitive, role, no, name @ ..]
            if other.surface == "別"
                && dasu_case_link(other, genitive)
                && genitive.surface == "の"
                && other.head == role.doc_tok_idx =>
        {
            (role, no, name, true)
        }
        [role, no, name @ ..] => (role, no, name, false),
        _ => return None,
    };
    (dasu_case_link(role, no)
        && no.surface == "の"
        && role.deprel == "nmod"
        && role.head == name.last()?.doc_tok_idx
        && role_name(role).is_some())
    .then_some((role, name, other))
}

fn person(argument: Argument<'_>, doc: &Document) -> bool {
    written_head(argument.head, doc).is_some_and(|token| {
        human_recipient(&nominal(argument.words.to_vec()), argument.head, token, doc)
            || role_name(argument.head).is_some()
    })
}

fn nominal_text(words: &[Atom]) -> String {
    render_bunsetsu_np(
        &nominal(normalize_ja_atoms(&apply_phrase_matches(words))),
        TargetVariant::ZhHant,
    )
}

fn person_text(words: &[Atom]) -> String {
    // The title さん is neutral. The name or stated role already supplies
    // the address without adding a gender absent from the source.
    let words = match words {
        [name @ .., suffix]
            if !name.is_empty() && !suffix.is_locked && suffix.surface == "さん" =>
        {
            name
        }
        _ => words,
    };
    nominal_text(words)
}

fn render_argument(argument: Argument<'_>, rendered: &[Atom]) -> Option<String> {
    let words = words_in(rendered, argument.span())?;
    if let [atom] = words {
        if atom.is_locked {
            return Some(atom.surface.clone());
        }
        if argument.kind == GlossaryKind::Actor
            && let Some(role) = role_name(atom)
        {
            return Some(role.to_owned());
        }
    }
    if argument.kind == GlossaryKind::Actor
        && let Some((role, name, other)) = apposition(argument.words)
        && let [rendered_role] = words_in(rendered, role.span)?
    {
        let name_span = Span {
            start_char: name.first()?.span.start_char,
            end_char: name.last()?.span.end_char,
        };
        let role = if rendered_role.is_locked {
            &rendered_role.surface
        } else {
            role_name(role)?
        };
        let determiner = if other { "另一位" } else { "" };
        return Some(format!(
            "{determiner}{role}{}",
            person_text(words_in(rendered, name_span)?)
        ));
    }
    let text = if argument.kind == GlossaryKind::Actor {
        person_text(words)
    } else {
        nominal_text(words)
    };
    (!text.is_empty()).then_some(text)
}

fn render_locks(locks: &SpanLockTable, arguments: &[Argument<'_>]) -> Option<SpanLockTable> {
    let mut render = locks.clone();
    for lock in &mut render.locks {
        let LockOrigin::Glossary { kind, .. } = lock.origin else {
            return None;
        };
        let span = lock.source_span;
        let argument = arguments.iter().find(|argument| {
            kind == argument.kind
                && span.start_char >= argument.span().start_char
                && span.start_char < argument.span().end_char
                && span.end_char <= argument.case.span.end_char
        })?;
        let end = span.end_char.min(argument.head.span.end_char);
        let covered = Span {
            start_char: span.start_char,
            end_char: end,
        };
        if !lock.mandatory || (span.end_char > end && span.end_char != argument.case.span.end_char)
        {
            return None;
        }
        let inside = words_in(argument.words, covered)?;
        if argument.words.iter().any(|word| {
            (word.span.start_char < covered.start_char || word.span.end_char > covered.end_char)
                && word.head != argument.head.doc_tok_idx
                && inside.iter().any(|parent| parent.doc_tok_idx == word.head)
        }) {
            return None;
        }
        // The caller still reports the full match, including any case marker.
        lock.source_span = covered;
    }
    Some(render)
}

fn handover_word(lemma: &str) -> Option<&'static str> {
    match lemma {
        "渡す" => Some("交給"),
        "手渡す" => Some("親手交給"),
        _ => None,
    }
}

fn preceding<'a>(
    atoms: &'a [Atom],
    right_words: &[Atom],
    comma: &Atom,
    root: usize,
    doc: &Document,
) -> Option<(&'a [Atom], &'static str, usize, bool)> {
    if let [words @ .., verb, link] = atoms
        && link.surface == "て"
        && link.deprel == "mark"
        && link.head == verb.doc_tok_idx
        && verb.deprel == "advcl"
        && verb.head == Some(root)
        && comma.head == verb.doc_tok_idx
    {
        return Some((
            words,
            handover_word(verb.lemma.as_deref()?)?,
            verb.doc_tok_idx?,
            false,
        ));
    }
    if let [words @ .., stem] = atoms
        && stem.upos == Some(Upos::Noun)
        && stem.deprel == "nmod"
        && right_words.iter().any(|word| word.doc_tok_idx == stem.head)
        && comma.head == Some(root)
    {
        // The parser can attach a bare continuative as a nominal modifier of
        // the right clause. Both local argument frames still need to pass.
        return Some((
            words,
            handover_word(&purpose_stem_lemma(stem)?)?,
            root,
            false,
        ));
    }
    if let [words @ .., stem] = atoms
        && stem.upos == Some(Upos::Verb)
        && matches!(stem.deprel.as_str(), "advcl" | "conj")
        && stem.head == Some(root)
        && stem
            .doc_tok_idx
            .is_some_and(|id| matches!(comma.head, Some(parent) if parent == id || parent == root))
        && let Some(lemma) = purpose_stem_lemma(stem)
        && stem.lemma.as_deref() == Some(lemma.as_str())
    {
        return Some((words, handover_word(&lemma)?, stem.doc_tok_idx?, false));
    }
    let [clause @ .., link] = atoms else {
        return None;
    };
    let index = clause
        .iter()
        .position(|atom| atom.upos == Some(Upos::Verb))?;
    let (words, predicate) = clause.split_at(index);
    let [verb, past @ ..] = predicate else {
        return None;
    };
    let id = verb.doc_tok_idx?;
    if link.surface != "が"
        || !matches!(link.deprel.as_str(), "case" | "mark" | "cc")
        || link.head != Some(id)
        || !matches!(verb.deprel.as_str(), "nsubj" | "advcl" | "conj")
        || verb.head != Some(root)
        || comma.head != Some(id)
        || !travel_past(past, id, doc)
    {
        return None;
    }
    Some((words, handover_word(verb.lemma.as_deref()?)?, id, true))
}

pub(in crate::ja_zh) fn render(
    source: &[char],
    doc: &Document,
    sentence: usize,
    tokens: &[&Token],
    locks: &SpanLockTable,
    clauses: &[(usize, Span)],
) -> Option<(String, Vec<ClauseAlignment>)> {
    let [(left_clause, left_span), (right_clause, right_span)] = clauses else {
        return None;
    };
    let sentence_span = doc.segmentation.sentences.get(sentence)?.span;
    if source.len() != doc.text_chars
        || left_span.start_char != sentence_span.start_char
        || right_span.end_char != sentence_span.end_char
        || !matches!(
            source.get(left_span.end_char..right_span.start_char),
            Some(['、' | '，'])
        )
        || source.get(sentence_span.end_char.checked_sub(1)?) != Some(&'。')
        || doc.segmentation.opaque.iter().any(|opaque| {
            opaque.span.start_char < sentence_span.end_char
                && sentence_span.start_char < opaque.span.end_char
        })
    {
        return None;
    }
    let original = build_sentence_atoms(sentence, source, doc, tokens, &SpanLockTable::new());
    if original.first()?.span.start_char != sentence_span.start_char
        || original.last()?.span.end_char != sentence_span.end_char
        || original
            .iter()
            .any(|atom| written_head(atom, doc).is_none_or(|token| token.sentence != sentence))
        || original
            .windows(2)
            .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
    {
        return None;
    }
    let boundary = original
        .iter()
        .position(|atom| atom.span.start_char == left_span.end_char)?;
    let (left, rest) = original.split_at(boundary);
    let [comma, right @ .., period] = rest else {
        return None;
    };
    if comma.deprel != "punct"
        || period.deprel != "punct"
        || comma.span.end_char != right_span.start_char
    {
        return None;
    }
    let right_verb_index = right
        .iter()
        .position(|atom| atom.upos == Some(Upos::Verb))?;
    let (right_words, predicate) = right.split_at(right_verb_index);
    let [right_verb, right_past @ ..] = predicate else {
        return None;
    };
    let root = right_verb.doc_tok_idx?;
    if right_verb.deprel != "root"
        || right_verb.head.is_some()
        || period.head != Some(root)
        || !travel_past(right_past, root, doc)
    {
        return None;
    }
    let (left_words, action, owner, contrast) = preceding(left, right_words, comma, root, doc)?;
    let left = arguments(left_words, owner)?;
    let right = arguments(right_words, root)?;
    let (left_actor, left_object, left_recipient, right_object) =
        (left.actor?, left.object?, left.recipient?, right.object?);

    // A role-labelled recipient can be repeated as an overt subject. The
    // parser can leave such a personal name untagged in both clauses.
    let repeated_name = right.actor.is_some_and(|actor| {
        let same_name = |name: &Atom| name.surface == actor.head.surface;
        contrast
            && actor.words.len() == 1
            && apposition(left_recipient.words)
                .is_some_and(|(_, name, _)| matches!(name, [word] if same_name(word)))
            && written_head(left_recipient.head, doc).is_some_and(|token| token.sem_class.is_none())
    });
    if !person(left_actor, doc)
        || !(person(left_recipient, doc) || repeated_name)
        || right
            .actor
            .is_some_and(|actor| !(person(actor, doc) || repeated_name))
    {
        return None;
    }
    let arguments = [
        left.actor,
        left.object,
        left.recipient,
        right.actor,
        right.object,
        right.recipient,
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    let render = render_locks(locks, &arguments)?;
    let rendered = build_sentence_atoms(sentence, source, doc, tokens, &render);
    let noun = |argument| render_argument(argument, &rendered);
    let left_out = format!(
        "{}把{}{action}{}，",
        noun(left_actor)?,
        noun(left_object)?,
        noun(left_recipient)?
    );
    let mut right_out = if contrast {
        "但".to_owned()
    } else {
        String::new()
    };
    if let Some(actor) = right.actor {
        right_out.push_str(&noun(actor)?);
    } else if !contrast {
        right_out.push('也');
    }
    let right_lemma = right_verb.lemma.as_deref()?;
    if let Some(action) = handover_word(right_lemma) {
        let recipient = right.recipient.filter(|argument| person(*argument, doc))?;
        right_out.push_str(&format!(
            "把{}{action}{}",
            noun(right_object)?,
            noun(recipient)?
        ));
    } else if contrast
        && right.actor.is_some()
        && right.recipient.is_none()
        && right_lemma == "尋ねる"
    {
        let object = words_in(&rendered, right_object.span())?;
        right_out.push_str("問了");
        if matches!(object, [road] if !road.is_locked && road.surface == "道") {
            // In an inquiry, 道を尋ねる is asking for directions.
            right_out.push('路');
        } else {
            right_out.push_str(&noun(right_object)?);
        }
    } else {
        return None;
    }
    right_out.push('。');
    let left_end = left_out.chars().count();
    let right_end = left_end + right_out.chars().count();
    let output = left_out + &right_out;
    if locks
        .locks
        .iter()
        .any(|lock| output.matches(&lock.placeholder).count() != 1)
    {
        return None;
    }
    let alignment = |clause, source_span, start_char, end_char| ClauseAlignment {
        clause,
        sentence,
        source_span,
        target_span: Span {
            start_char,
            end_char,
        },
    };
    Some((
        output,
        vec![
            alignment(*left_clause, *left_span, 0, left_end),
            alignment(*right_clause, *right_span, left_end, right_end),
        ],
    ))
}
