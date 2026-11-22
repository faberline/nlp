//! Recover argument case covered by a caller glossary in linked clauses.

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Document, Span, Token, Upos, build_sentence_atoms, classify_bunsetsu, dasu_case_link,
    finished_go, group_into_bunsetsus, household_entry, is_punct_atom, naide_adverbial,
    outward_look, physical_goal_head, polarity_contrast, trip_after_meal,
};

fn may_cover_past_predicate(lock: &LockedSpan, chars: &[char]) -> bool {
    lock.mandatory
        && matches!(
            lock.origin,
            LockOrigin::Glossary {
                kind: GlossaryKind::DomainTerm,
                ..
            }
        )
        && chars.get(lock.source_span.end_char.saturating_sub(1)) == Some(&'た')
}

fn covers_case_marker(lock: &LockedSpan, chars: &[char]) -> bool {
    matches!(
        (
            &lock.origin,
            chars.get(lock.source_span.end_char.saturating_sub(1))
        ),
        (
            LockOrigin::Glossary {
                kind: GlossaryKind::Object,
                ..
            },
            Some('を')
        ) | (
            LockOrigin::Glossary {
                kind: GlossaryKind::Place,
                ..
            },
            Some('に' | 'へ')
        )
    )
}

pub(crate) fn may_restore_glossary_syntax(
    text: &str,
    chars: &[char],
    locks: &SpanLockTable,
) -> bool {
    if text.contains(['「', '」', '『', '』', '"', '“', '”']) {
        return false;
    }
    if locks
        .locks
        .iter()
        .any(|lock| outward_look::covers_outside_case(chars, lock))
    {
        return true;
    }
    if text.matches('は').nth(1).is_some() && polarity_contrast::may_restore_topics(chars, locks) {
        return true;
    }
    if text.contains('て')
        && locks
            .locks
            .iter()
            .any(|lock| may_cover_past_predicate(lock, chars))
    {
        return true;
    }
    let naide = text.contains("ないで");
    let zuni = text.contains("ずに");
    (naide || zuni)
        && locks.locks.iter().any(|lock| {
            lock.mandatory
                && (if zuni {
                    matches!(
                        lock.origin,
                        LockOrigin::Glossary {
                            kind: GlossaryKind::Place,
                            ..
                        }
                    )
                } else {
                    false
                } || naide
                    && (covers_case_marker(lock, chars) || may_cover_past_predicate(lock, chars)))
        })
}

/// Expose verified case tokens and identify a full predicate kept verbatim.
/// The original glossary table owns the source matches and replacements.
pub(crate) fn glossary_render_locks(
    sentence: usize,
    doc: &Document,
    tokens: &[&Token],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<(SpanLockTable, Option<usize>)> {
    let atoms = build_sentence_atoms(sentence, chars, doc, tokens, &SpanLockTable::new());
    omitted_render_locks(sentence, doc, chars, locks, &atoms)
        .or_else(|| {
            if !locks
                .locks
                .iter()
                .any(|lock| may_cover_past_predicate(lock, chars))
            {
                return None;
            }
            trip_after_meal::whole_trip_lock(sentence, &atoms, doc, chars, locks)
                .map(|index| (locks.clone(), Some(index)))
        })
        .or_else(|| polarity_contrast::restore_topics(sentence, doc, chars, locks, &atoms))
        .or_else(|| {
            outward_look::restore_case(doc, chars, locks, &atoms).map(|locks| (locks, None))
        })
}

fn omitted_render_locks(
    sentence: usize,
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
    atoms: &[Atom],
) -> Option<(SpanLockTable, Option<usize>)> {
    let mut omitted = (3..atoms.len()).filter_map(|end| {
        let remaining = &atoms[end..];
        let following = if matches!(remaining[0].surface.as_str(), "、" | "，" | ",") {
            &remaining[1..]
        } else {
            remaining
        };
        naide_adverbial(&atoms[..end], following, None, chars).map(|omitted| (omitted, following))
    });
    let (omission, following) = omitted.next()?;
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
    let trip_predicate = match movement {
        Some(motion)
            if !omission.zuni
                && omission.join_unstated_subject
                && motion.lemma.as_deref() == Some("行く") =>
        {
            completed_trip_span(sentence, doc, chars, following, motion, omitted_action)
        }
        _ => None,
    };
    let whole_predicate = if omission.zuni {
        None
    } else if let Some(span) = trip_predicate {
        whole_domain_lock(locks, span)
    } else {
        whole_entry_lock(sentence, doc, chars, following, later, locks)
    };
    if whole_predicate.is_none()
        && !omission.zuni
        && !locks
            .locks
            .iter()
            .any(|lock| covers_case_marker(lock, chars))
    {
        return None;
    }

    let mut render = locks.clone();
    for lock in &mut render.locks {
        let LockOrigin::Glossary { kind, .. } = lock.origin else {
            return None;
        };
        if !lock.mandatory {
            return None;
        }
        let span = lock.source_span;
        if whole_predicate == Some(lock.index) {
            continue;
        }
        // Preserve proved verb stems while exposing argument case markers.
        let carrying_stem = span == action.span && action.lemma.as_deref() == Some("持つ");
        let entry_stem = movement.is_some_and(|entry| {
            entry.lemma.as_deref() == Some("入る") && (span == action.span || span == entry.span)
        });
        let trip_stem = trip_predicate.is_some()
            && movement.is_some_and(|trip| span == action.span || span == trip.span);
        if (carrying_stem || entry_stem || trip_stem)
            && crate::verb_glossary::is_stem_lock(lock, doc, sentence)
        {
            continue;
        }
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
    Some((render, whole_predicate))
}

fn completed_trip_span(
    sentence: usize,
    doc: &Document,
    chars: &[char],
    following: &[Atom],
    motion: &Atom,
    omitted_action: usize,
) -> Option<Span> {
    let clause = following.split(is_punct_atom).next()?;
    let groups = group_into_bunsetsus(clause, doc, sentence);
    let trip = finished_go::resolve(&groups, clause, doc, chars, Some(omitted_action))?;
    (motion.doc_tok_idx == Some(trip.predicate)).then_some(Span {
        start_char: motion.span.start_char,
        end_char: clause.last()?.span.end_char,
    })
}

fn whole_domain_lock(locks: &SpanLockTable, span: Span) -> Option<usize> {
    locks.locks.iter().find_map(|lock| {
        (lock.mandatory
            && lock.source_span == span
            && matches!(
                lock.origin,
                LockOrigin::Glossary {
                    kind: GlossaryKind::DomainTerm,
                    ..
                }
            ))
        .then_some(lock.index)
    })
}

fn whole_entry_lock(
    sentence: usize,
    doc: &Document,
    chars: &[char],
    following: &[Atom],
    governor: usize,
    locks: &SpanLockTable,
) -> Option<usize> {
    let clause = following.split(is_punct_atom).next()?;
    let head = clause
        .iter()
        .position(|atom| atom.doc_tok_idx == Some(governor))?;
    let (arguments, predicate) = clause.split_at(head);
    let [entry, ending @ ..] = predicate else {
        return None;
    };
    let simple_past = match ending {
        [past] => matches!(past.surface.as_str(), "た" | "ました"),
        [polite, past] => polite.surface == "まし" && past.surface == "た",
        _ => false,
    };
    if entry.lemma.as_deref() != Some("入る")
        || !simple_past
        || !ending.iter().all(|atom| {
            atom.upos == Some(Upos::Aux) && atom.deprel == "aux" && atom.head == Some(governor)
        })
        || !predicate
            .windows(2)
            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
    {
        return None;
    }
    let span = Span {
        start_char: entry.span.start_char,
        end_char: predicate.last()?.span.end_char,
    };
    let lock = whole_domain_lock(locks, span)?;
    let mut goals = arguments.windows(2).filter(|pair| {
        let [noun, marker] = pair else { return false };
        noun.head == Some(governor)
            && noun.deprel == "obl"
            && matches!(marker.surface.as_str(), "に" | "へ" | "まで")
            && dasu_case_link(noun, marker)
    });
    let [noun, marker] = goals.next()? else {
        return None;
    };
    let goal = classify_bunsetsu(vec![noun.clone(), marker.clone()], doc, sentence);
    (marker.surface != "まで"
        && goals.next().is_none()
        && physical_goal_head(&goal) == Some(governor)
        && household_entry::allows_physical_entry(doc, chars, governor))
    .then_some(lock)
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
    fn preserves_omission_and_destination_with_completed_trip_locks() {
        let source = "朝ご飯を食べないで、学校へ行きました。";
        for (term, replacement, expected) in [
            ("行き", "前往", "沒有享用早餐就前往了新校舍。"),
            ("行きました", "已前往", "沒有享用早餐就已前往新校舍。"),
        ] {
            let glossary = TranslationGlossary {
                objects: vec![entry("朝ご飯を", "早餐")],
                places: vec![entry("学校へ", "新校舍")],
                domain_terms: vec![entry("食べ", "享用"), entry(term, replacement)],
                ..TranslationGlossary::new()
            };
            let output = translate(source, &glossary);
            assert_eq!(output.target_text, expected);
            assert_eq!(output.glossary_hits.len(), 4);
            let hit = output
                .glossary_hits
                .iter()
                .find(|hit| hit.canonical == term)
                .unwrap();
            let source_slice: String = source
                .chars()
                .skip(hit.source_span.start_char)
                .take(hit.source_span.end_char - hit.source_span.start_char)
                .collect();
            let target_slice: String = output
                .target_text
                .chars()
                .skip(hit.target_span.start_char)
                .take(hit.target_span.end_char - hit.target_span.start_char)
                .collect();
            assert_eq!(source_slice, term);
            assert_eq!(target_slice, replacement);
        }
    }

    #[test]
    fn a_completed_trip_lock_does_not_assert_a_trip_in_a_question() {
        let glossary = TranslationGlossary {
            domain_terms: vec![entry("行きました", "已前往")],
            ..TranslationGlossary::new()
        };
        let output = translate("朝ご飯を食べないで、学校へ行きましたか。", &glossary);
        assert!(!output.target_text.contains('就'));
        assert_eq!(output.target_text.matches("已前往").count(), 1);
        assert_eq!(output.glossary_hits.len(), 1);
    }
}
