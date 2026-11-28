//! Narrow source-linked handling for ordinary reported speech.

use super::*;

/// Move a simple 「...」と言う report ahead of its quotation.
///
/// The quotation itself remains with the caller, which renders it with the
/// existing locks. Reject shapes with extra arguments or nested quotations.
pub(super) fn reported_speech(
    segments: &[Segment],
    doc: &Document,
    locks: &SpanLockTable,
) -> Option<(String, Span, usize)> {
    let quote_index = segments
        .iter()
        .take(3)
        .position(|segment| matches!(segment, Segment::Quote { .. }))?;
    let [
        Segment::Quote { inner_atoms, .. },
        Segment::Clause {
            atoms: report,
            span,
            ..
        },
        ..,
    ] = &segments[quote_index..]
    else {
        return None;
    };
    if inner_atoms.iter().any(is_quote_mark) {
        return None;
    }
    let (marker, rest) = report.split_first()?;
    if marker.is_locked || marker.surface != "と" {
        return None;
    }
    let verb_index = rest
        .iter()
        .position(|atom| !atom.is_locked && atom.lemma.as_deref() == Some("言う"))?;
    let (after, verb_and_tail) = rest.split_at(verb_index);
    let (verb, tail) = verb_and_tail.split_first()?;
    if verb.doc_tok_idx.is_none()
        || (!verb.surface.ends_with('た') && tail.last().is_none_or(|atom| atom.surface != "た"))
        || marker.head != verb.doc_tok_idx
        || tail.iter().any(|atom| {
            atom.is_locked || !matches!(atom.surface.as_str(), "ます" | "まし" | "た" | "だ")
        })
    {
        return None;
    }
    let mut roles = DirectQuoteRoles::default();
    for before in &segments[..quote_index] {
        let Segment::Clause {
            atoms,
            trailing_punct,
            ..
        } = before
        else {
            return None;
        };
        if !is_direct_quote_prefix_separator(trailing_punct.as_deref()) {
            return None;
        }
        read_direct_quote_roles(atoms, doc, locks, None, &mut roles)?;
    }
    read_direct_quote_roles(after, doc, locks, Some(verb), &mut roles)?;
    let speaker = roles.speaker?;
    let listener = roles.listener?;
    let time = roles.time.unwrap_or_default();
    let listener = if listener == "客" {
        "客人"
    } else {
        &listener
    };
    Some((
        format!("{speaker}{time}對{listener}說："),
        *span,
        quote_index,
    ))
}

/// Reading a quoted sentence to understand its meaning uses the normal quoted
/// clause translation. Move the reading verb ahead of its quoted object.
pub(super) fn reading_for_meaning(
    segments: &[Segment],
    doc: &Document,
    sentence: usize,
    chars: &[char],
    locks: &SpanLockTable,
    clauses: &[(usize, Span)],
) -> Option<(String, Vec<ClauseAlignment>)> {
    let [
        Segment::Clause { atoms: topic, .. },
        Segment::Quote {
            open_q,
            inner_atoms,
            close_q,
            span: quote,
        },
        Segment::Clause {
            atoms: reading,
            trailing_punct: Some(comma),
            ..
        },
        Segment::Clause {
            atoms: result,
            trailing_punct,
            ..
        },
    ] = segments
    else {
        return None;
    };
    let [first, second] = clauses else {
        return None;
    };
    if !matches!(comma.as_str(), "、" | "，")
        || inner_atoms.iter().any(is_quote_mark)
        || locks.locks.iter().any(|lock| {
            lock.origin == LockOrigin::QuotedSource
                || (lock.source_span.start_char >= quote.end_char)
        })
    {
        return None;
    }

    let (topic_particle, actor) = topic.split_last()?;
    if topic_particle.is_locked || topic_particle.surface != "は" {
        return None;
    }
    let person = render_direct_quote_person(actor, doc, locks)?;
    let [label, text, object, verb, conjunctive] = reading.as_slice() else {
        return None;
    };
    if reading.iter().any(|atom| atom.is_locked)
        || label.surface != "という"
        || !matches!(text.surface.as_str(), "文" | "文章" | "一文")
        || object.surface != "を"
        || object.head != text.doc_tok_idx
        || text.head != verb.doc_tok_idx
        || verb.lemma.as_deref() != Some("読む")
        || conjunctive.head != verb.doc_tok_idx
        || !matches!(conjunctive.surface.as_str(), "て" | "で")
    {
        return None;
    }
    let [meaning, subject, understood, polite, past] = result.as_slice() else {
        return None;
    };
    if result.iter().any(|atom| atom.is_locked)
        || meaning.surface != "意味"
        || subject.surface != "が"
        || understood.lemma.as_deref() != Some("分かる")
        || meaning.head != understood.doc_tok_idx
        || verb.head != understood.doc_tok_idx
        || polite.surface != "まし"
        || past.surface != "た"
    {
        return None;
    }

    let quoted = match split_atoms_into_segments(inner_atoms, doc, *quote).as_slice() {
        [
            Segment::Clause {
                atoms,
                trailing_punct,
                ..
            },
        ] => {
            let (mut text, _) = translate_clause_atoms(
                atoms,
                doc,
                sentence,
                None,
                false,
                &ClauseDiscourseCtx {
                    source_chars: chars,
                    source_locks: Some(locks),
                    inside_quote: true,
                    ..Default::default()
                },
                TargetVariant::ZhHant,
            );
            if future_confirmation(atoms) {
                text.insert(0, '會');
            }
            if let Some(punct) = trailing_punct {
                text.push_str(&localize_punct(punct));
            }
            text
        }
        _ => return None,
    };
    let mut translated = format!("{person}讀了{open_q}{quoted}{close_q}這句話，");
    let result_start = translated.chars().count();
    translated.push_str("明白了它的意思");
    if let Some(punct) = trailing_punct {
        translated.push_str(&localize_punct(punct));
    }
    let result_end = translated.chars().count();
    let alignments = [
        (
            first,
            Span {
                start_char: 0,
                end_char: result_start,
            },
        ),
        (
            second,
            Span {
                start_char: result_start,
                end_char: result_end,
            },
        ),
    ]
    .iter()
    .map(|((clause, source_span), target_span)| ClauseAlignment {
        clause: *clause,
        sentence,
        source_span: *source_span,
        target_span: *target_span,
    })
    .collect();
    Some((translated, alignments))
}

/// A complete short confirmation announcement in polite nonpast expresses
/// a commitment to check. Leave longer or modified quoted clauses alone.
pub(super) fn future_confirmation(atoms: &[Atom]) -> bool {
    let [object, case, verb, polite] = core_non_punct(atoms) else {
        return false;
    };
    (matches!(object.upos, Some(Upos::Noun | Upos::Propn)) || object.is_locked)
        && !case.is_locked
        && case.surface == "を"
        && !verb.is_locked
        && verb.lemma.as_deref() == Some("確認する")
        && !polite.is_locked
        && polite.surface == "ます"
}
