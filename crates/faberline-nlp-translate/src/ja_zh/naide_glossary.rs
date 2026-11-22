//! Recover argument case covered by a caller glossary in an omitted action.

use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Document, Span, Token, Upos, build_sentence_atoms, classify_bunsetsu, dasu_case_link,
    naide_adverbial, physical_goal_head,
};

pub(crate) fn may_restore_glossary_syntax(
    text: &str,
    chars: &[char],
    locks: &SpanLockTable,
) -> bool {
    let naide = text.contains("ないで");
    let zuni = text.contains("ずに");
    (naide || zuni)
        && !text.contains(['「', '」', '『', '』', '"', '“', '”'])
        && locks.locks.iter().any(|lock| {
            let last = chars.get(lock.source_span.end_char.saturating_sub(1));
            lock.mandatory
                && match lock.origin {
                    LockOrigin::Glossary {
                        kind: GlossaryKind::Place,
                        ..
                    } => zuni || (naide && matches!(last, Some('に' | 'へ'))),
                    LockOrigin::Glossary {
                        kind: GlossaryKind::Object,
                        ..
                    } => naide && last == Some(&'を'),
                    _ => false,
                }
        })
}

/// The render table exposes the case token. The original glossary table
/// continues to own the full source match and replacement.
pub(crate) fn glossary_render_locks(
    sentence: usize,
    doc: &Document,
    tokens: &[&Token],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<SpanLockTable> {
    let atoms = build_sentence_atoms(sentence, chars, doc, tokens, &SpanLockTable::new());
    let mut omitted = (3..atoms.len()).filter_map(|end| {
        let remaining = &atoms[end..];
        let following = if matches!(remaining[0].surface.as_str(), "、" | "，" | ",") {
            &remaining[1..]
        } else {
            remaining
        };
        naide_adverbial(&atoms[..end], following, None, chars)
    });
    let omission = omitted.next()?;
    let omitted_action = omission.action;
    if omitted.next().is_some() {
        return None;
    }
    let action = atoms
        .iter()
        .find(|a| a.doc_tok_idx == Some(omitted_action))?;
    let later = action.head?;
    let movement = atoms.iter().find(|a| {
        a.doc_tok_idx == Some(later) && matches!(a.lemma.as_deref(), Some("行く" | "入る"))
    });

    let mut render = locks.clone();
    for lock in &mut render.locks {
        let LockOrigin::Glossary { kind, .. } = lock.origin else {
            return None;
        };
        if !lock.mandatory {
            return None;
        }
        let span = lock.source_span;
        let first = atoms
            .iter()
            .position(|a| a.span.start_char == span.start_char)?;
        let last = atoms
            .iter()
            .position(|a| a.span.end_char == span.end_char)?;
        if first > last
            || atoms[first..=last]
                .windows(2)
                .any(|p| p[0].span.end_char != p[1].span.start_char)
        {
            return None;
        }
        let (head_index, marker) = if atoms[last].deprel == "case" {
            (last.checked_sub(1)?, &atoms[last])
        } else {
            (last, atoms.get(last + 1)?)
        };
        let noun = &atoms[head_index];
        if head_index < first
            || !dasu_case_link(noun, marker)
            || !matches!(noun.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
        {
            return None;
        }
        match (kind, marker.surface.as_str(), noun.deprel.as_str()) {
            (GlossaryKind::Object, "を", "obj") if noun.head == Some(omitted_action) => {
                let prefix = &atoms[first..head_index];
                // The unhinted tokenizer may give 朝 its own adverb role.
                // A full breakfast lock must keep that source character too.
                let breakfast = matches!(prefix, [morning]
                    if morning.surface == "朝"
                        && noun.surface == "ご飯"
                        && morning.head == Some(omitted_action)
                        && morning.deprel == "advmod");
                if !breakfast
                    && prefix.iter().any(|a| {
                        a.head != noun.doc_tok_idx
                            || !matches!(a.deprel.as_str(), "compound" | "amod" | "nummod")
                    })
                {
                    return None;
                }
            }
            (GlossaryKind::Place, "に" | "へ", "obl")
                if noun.head == Some(later) && first == head_index && movement.is_some() =>
            {
                let goal = classify_bunsetsu(vec![noun.clone(), marker.clone()], doc, sentence);
                if physical_goal_head(&goal) != Some(later)
                    || atoms.windows(2).any(|pair| {
                        let [other, case] = pair else { return false };
                        other.span != noun.span
                            && other.head == Some(later)
                            && other.deprel == "obl"
                            && matches!(case.surface.as_str(), "に" | "へ" | "まで")
                            && dasu_case_link(other, case)
                    })
                {
                    return None;
                }
            }
            (GlossaryKind::Place, "を", "obj")
                if omission.zuni && noun.head == Some(later) && first == head_index => {}
            _ => return None,
        }
        lock.source_span = Span {
            start_char: span.start_char,
            end_char: noun.span.end_char,
        };
    }
    Some(render)
}
