//! Recover the role of a place glossary entry that includes its case marker.

use faberline_nlp::Span;
use faberline_nlp::doc::{Options, analyze_document};

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, LOCKED_PLACE_ATOM, SpanLockTable, Upos,
    build_sentence_atoms, classify_bunsetsu, focused_physical_goal_head,
};

#[derive(Clone, Copy)]
pub(super) struct LockedGoal {
    pub(super) span: Span,
    pub(super) predicate: usize,
    pub(super) includes_focus: bool,
    direction: &'static str,
}

impl LockedGoal {
    pub(super) fn matches_group(self, group: &Bunsetsu) -> bool {
        let same_lock = |atom: &Atom| {
            atom.is_locked
                && atom.lemma.as_deref() == Some(LOCKED_PLACE_ATOM)
                && atom.span == self.span
        };
        match group.atoms.as_slice() {
            [lock] => self.includes_focus && same_lock(lock),
            [lock, focus] => {
                !self.includes_focus
                    && same_lock(lock)
                    && !focus.is_locked
                    && focus.surface == "は"
                    && focus.span.start_char == self.span.end_char
            }
            _ => false,
        }
    }

    pub(super) fn reclassify(self, groups: &mut [Bunsetsu]) -> bool {
        let Some(group) = groups.iter_mut().find(|group| self.matches_group(group)) else {
            return false;
        };
        // The placeholder still covers the original glossary range.
        group.role = BunsetsuRole::Obl;
        group.case_particle = Some(self.direction.to_owned());
        true
    }
}

pub(super) fn recover(source: &[Atom], doc: &Document, chars: &[char]) -> Option<LockedGoal> {
    let mut places = source
        .iter()
        .enumerate()
        .filter(|(_, atom)| atom.is_locked && atom.lemma.as_deref() == Some(LOCKED_PLACE_ATOM));
    let (lock_index, lock) = places.next()?;
    if places.next().is_some() {
        return None;
    }
    let word = chars.get(lock.span.start_char..lock.span.end_char)?;
    let (direction, includes_focus) = match word {
        [_, .., 'に', 'は'] => ("に", true),
        [_, .., 'へ', 'は'] => ("へ", true),
        [_, .., 'に'] => ("に", false),
        [_, .., 'へ'] => ("へ", false),
        _ => return None,
    };
    if !includes_focus
        && !source.get(lock_index + 1).is_some_and(|atom| {
            !atom.is_locked && atom.surface == "は" && atom.span.start_char == lock.span.end_char
        })
    {
        return None;
    }

    let mut verbs = source.iter().filter(|atom| atom.upos == Some(Upos::Verb));
    let go = verbs.next()?;
    let predicate = go.doc_tok_idx?;
    let token = doc.tokens.get(predicate)?;
    let link = doc.syntax.get(predicate)?;
    if verbs.next().is_some()
        || go.is_locked
        || link.token != predicate
        || link.span != go.span
        || token.span != go.span
        || token.lemma.as_deref() != Some("行く")
        || link.deprel != "root"
    {
        return None;
    }
    let negated = source.iter().any(|atom| {
        !atom.is_locked
            && atom.upos == Some(Upos::Aux)
            && atom.head == Some(predicate)
            && atom.deprel == "aux"
            && matches!(
                atom.lemma.as_deref().unwrap_or(&atom.surface),
                "ない" | "ん" | "なかっ"
            )
    });
    if !negated {
        return None;
    }

    let start = source.first()?.span.start_char;
    let original_chars = chars.get(start..source.last()?.span.end_char)?;
    let original = analyze_document(
        &original_chars.iter().collect::<String>(),
        &Options {
            normalize_traditional: true,
            enrich_morphology: true,
            cpu_syntax: true,
            ..Default::default()
        },
    );
    if original.segmentation.sentences.len() != 1 {
        return None;
    }
    let tokens: Vec<_> = original.tokens.iter().collect();
    let original_atoms = build_sentence_atoms(
        0,
        original_chars,
        &original,
        &tokens,
        &SpanLockTable::default(),
    );
    let absolute = |span: Span| Span {
        start_char: start + span.start_char,
        end_char: start + span.end_char,
    };
    let original_go = original_atoms.iter().find(|atom| {
        absolute(atom.span) == go.span
            && atom.lemma.as_deref() == Some("行く")
            && atom.deprel == "root"
    })?;
    let original_head = original_go.doc_tok_idx?;
    let group = original_atoms.windows(3).find(|atoms| {
        let [noun, case, focus] = atoms else {
            return false;
        };
        absolute(noun.span).start_char == lock.span.start_char
            && case.surface == direction
            && focus.surface == "は"
            && absolute(if includes_focus {
                focus.span
            } else {
                case.span
            })
            .end_char
                == lock.span.end_char
    })?;
    let probe = classify_bunsetsu(group.to_vec(), &original, 0);
    if focused_physical_goal_head(&probe, &original, original_chars) != Some(original_head)
        || original_atoms.iter().any(|atom| {
            atom.doc_tok_idx != group[1].doc_tok_idx
                && matches!(atom.surface.as_str(), "に" | "へ" | "まで")
                && atom.deprel == "case"
                && atom
                    .head
                    .and_then(|head| original.syntax.get(head))
                    .is_some_and(|arg| arg.head == Some(original_head))
        })
    {
        return None;
    }

    Some(LockedGoal {
        span: lock.span,
        predicate,
        includes_focus,
        direction,
    })
}
