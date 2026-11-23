//! Retain focus and direction when a glossary match covers a negated destination.

use faberline_nlp::doc::analyze_document;

use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Bunsetsu, Document, LOCKED_PLACE_ATOM, Span, Upos, build_sentence_atoms,
    classify_bunsetsu, focused_physical_goal_head,
};

pub(super) struct FocusedGoal {
    noun: Span,
    case: Span,
    focus: Span,
    head: usize,
}

fn has_direct_negation(go: &Atom, tail: &[Atom]) -> bool {
    let Some(first) = tail.first() else {
        return false;
    };
    let attached = |atom: &Atom, start| {
        atom.span.start_char == start
            && atom.upos == Some(Upos::Aux)
            && atom.deprel == "aux"
            && atom.head == go.doc_tok_idx
    };
    attached(first, go.span.end_char)
        && (matches!(first.surface.as_str(), "ない" | "なかっ" | "ぬ")
            || first.surface == "ませ"
                && tail.get(1).is_some_and(|next| {
                    attached(next, first.span.end_char) && next.surface == "ん"
                }))
}

pub(super) fn restore(
    atoms: &mut Vec<Atom>,
    chars: &[char],
    sentence: Span,
    doc: &Document,
    locks: &SpanLockTable,
) -> Vec<FocusedGoal> {
    if !doc.options.enrich_morphology || !doc.options.cpu_syntax {
        return Vec::new();
    }
    let candidates: Vec<_> = locks
        .locks
        .iter()
        .filter_map(|lock| {
            let span = lock.source_span;
            if !lock.mandatory
                || !matches!(
                    lock.origin,
                    LockOrigin::Glossary {
                        kind: GlossaryKind::Place | GlossaryKind::DomainTerm,
                        ..
                    }
                )
                || span.start_char < sentence.start_char
                || span.end_char > sentence.end_char
            {
                return None;
            }
            let text = chars.get(span.start_char..span.end_char)?;
            let covers_focus = matches!(text, [_, .., 'に' | 'へ', 'は']);
            let covers_case = matches!(text, [_, .., 'に' | 'へ'])
                && span.end_char < sentence.end_char
                && chars.get(span.end_char) == Some(&'は');
            (covers_focus || covers_case).then(|| (lock, text.iter().collect::<String>()))
        })
        .collect();
    if candidates.is_empty() {
        return Vec::new();
    }
    let mut options = doc.options.clone();
    let supplied = |word: &String| candidates.iter().any(|(_, surface)| surface == word);
    options.lexicon.retain(|word| !supplied(word));
    options.names.retain(|word| !supplied(word));
    let text = &chars[sentence.start_char..sentence.end_char];
    let parsed = analyze_document(&text.iter().collect::<String>(), &options);
    if parsed.segmentation.sentences.len() != 1 {
        return Vec::new();
    }
    let tokens: Vec<_> = parsed.tokens.iter().collect();
    let source = build_sentence_atoms(0, text, &parsed, &tokens, &SpanLockTable::new());
    let absolute = |span: Span| Span {
        start_char: sentence.start_char + span.start_char,
        end_char: sentence.start_char + span.end_char,
    };
    let mut restored = Vec::new();
    for (lock, _) in candidates {
        let span = lock.source_span;
        let Some(start) = source
            .iter()
            .position(|atom| absolute(atom.span).start_char == span.start_char)
        else {
            continue;
        };
        let Some((offset, [case, focus])) =
            source[start + 1..]
                .windows(2)
                .enumerate()
                .find_map(|(offset, pair)| {
                    let [case, focus] = pair else { return None };
                    (matches!(case.surface.as_str(), "に" | "へ")
                        && focus.surface == "は"
                        && [absolute(case.span).end_char, absolute(focus.span).end_char]
                            .contains(&span.end_char))
                    .then_some((offset, [case, focus]))
                })
        else {
            continue;
        };
        let through_focus = start + offset + 3;
        let group = classify_bunsetsu(source[start..through_focus].to_vec(), &parsed, 0);
        let Some(go_id) = focused_physical_goal_head(&group, &parsed, text) else {
            continue;
        };
        let Some(go_index) = source
            .iter()
            .position(|atom| atom.doc_tok_idx == Some(go_id))
        else {
            continue;
        };
        let go = &source[go_index];
        if go_index < through_focus
            || go.lemma.as_deref() != Some("行く")
            || go.upos != Some(Upos::Verb)
            || !matches!(go.deprel.as_str(), "root" | "conj")
            || !has_direct_negation(go, &source[go_index + 1..])
        {
            continue;
        }
        let Some(head) = atoms
            .iter()
            .find(|atom| {
                !atom.is_locked
                    && atom.span == absolute(go.span)
                    && atom.lemma.as_deref() == Some("行く")
                    && atom.upos == Some(Upos::Verb)
            })
            .and_then(|atom| atom.doc_tok_idx)
        else {
            continue;
        };
        let Some(index) = atoms.iter().position(|atom| {
            atom.is_locked && atom.span == span && atom.surface == lock.placeholder
        }) else {
            continue;
        };
        let mut case_atom = case.clone();
        let mut focus_atom = focus.clone();
        for atom in [&mut case_atom, &mut focus_atom] {
            atom.span = absolute(atom.span);
            atom.doc_tok_idx = None;
            atom.head = None;
        }
        let last = if span.end_char == focus_atom.span.end_char {
            index
        } else {
            let Some(visible) = atoms.get(index + 1) else {
                continue;
            };
            if visible.span != focus_atom.span || visible.surface != "は" || visible.is_locked {
                continue;
            }
            index + 1
        };
        let mut noun = atoms[index].clone();
        noun.span.end_char = case_atom.span.start_char;
        noun.lemma = Some(LOCKED_PLACE_ATOM.to_owned());
        noun.doc_tok_idx = None;
        noun.head = Some(head);
        noun.deprel = "obl".to_owned();
        restored.push(FocusedGoal {
            noun: noun.span,
            case: case_atom.span,
            focus: focus_atom.span,
            head,
        });
        // Keep the original lock table so the caller's hit still includes its particles.
        atoms.splice(index..=last, [noun, case_atom, focus_atom]);
    }
    restored
}

pub(super) fn restored_head(group: &Bunsetsu, restored: &[FocusedGoal]) -> Option<usize> {
    let [noun, case, focus] = group.atoms.as_slice() else {
        return None;
    };
    if !noun.is_locked
        || case.is_locked
        || focus.is_locked
        || !matches!(case.surface.as_str(), "に" | "へ")
        || focus.surface != "は"
    {
        return None;
    }
    restored
        .iter()
        .find(|goal| (noun.span, case.span, focus.span) == (goal.noun, goal.case, goal.focus))
        .map(|goal| goal.head)
}
