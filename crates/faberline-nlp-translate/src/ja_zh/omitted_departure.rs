//! Retain an omitted action directly linked to a completed departure.

use faberline_nlp::doc::analyze_document;

use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, LOCKED_PLACE_ATOM, Segment, Span, Upos,
    build_sentence_atoms, is_existing_spatial_head, is_punct_atom, spatial_route_marker,
};

#[derive(Clone, Copy)]
pub(super) struct OmittedDeparture {
    action: usize,
    negation: usize,
    pub(super) connective: usize,
    pub(super) governor: usize,
    locked_source: Option<ProtectedOrigin>,
}

#[derive(Clone, Copy)]
struct ProtectedOrigin {
    lock: Span,
    noun_end: usize,
}

impl OmittedDeparture {
    pub(super) fn matches_action(self, predicates: &[Bunsetsu]) -> bool {
        matches!(predicates, [predicate] if [self.action, self.negation].into_iter().all(|index|
            predicate.atoms.iter().any(|atom| !atom.is_locked && atom.doc_tok_idx == Some(index))))
    }

    pub(super) fn expose_source_case(self, atoms: &mut Vec<Atom>) {
        let Some(origin) = self.locked_source else {
            return;
        };
        let Some(index) = atoms
            .iter()
            .position(|atom| atom.is_locked && atom.span == origin.lock)
        else {
            return;
        };
        let atom = &mut atoms[index];
        let head = atom.doc_tok_idx;
        atom.lemma = Some(LOCKED_PLACE_ATOM.to_owned());
        atom.head = Some(self.governor);
        atom.deprel = "obj".to_owned();
        if origin.noun_end == origin.lock.end_char {
            return;
        }
        atom.span.end_char = origin.noun_end;
        atoms.insert(
            index + 1,
            Atom {
                span: Span {
                    start_char: origin.noun_end,
                    end_char: origin.lock.end_char,
                },
                surface: "を".to_owned(),
                lemma: None,
                upos: Some(Upos::Adp),
                doc_tok_idx: None,
                head,
                deprel: "case".to_owned(),
                is_locked: false,
            },
        );
    }

    pub(super) fn restore_source(self, groups: &mut [Bunsetsu], source: &[Atom]) -> bool {
        let Some(origin) = self.locked_source else {
            return true;
        };
        let Some(placeholder) = source
            .iter()
            .find(|atom| atom.is_locked && atom.span == origin.lock)
        else {
            return false;
        };
        let Some(group) = groups.iter_mut().find(|group| {
            group.atoms.iter().any(|atom| {
                atom.span.start_char <= origin.lock.start_char
                    && atom.span.end_char >= origin.noun_end
                    && atom.surface.contains(&placeholder.surface)
            })
        }) else {
            return false;
        };
        group.role = BunsetsuRole::Obj;
        true
    }
}

fn glossary_argument(atom: &Atom, expected: GlossaryKind, locks: &SpanLockTable) -> bool {
    locks.locks.iter().any(|entry| {
        entry.mandatory
            && entry.source_span == atom.span
            && entry.placeholder == atom.surface
            && matches!(entry.origin, LockOrigin::Glossary { kind, .. }
                if kind == expected || kind == GlossaryKind::DomainTerm)
    })
}

pub(super) fn segments(
    atoms: &[Atom],
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<(Vec<Segment>, OmittedDeparture)> {
    let (last, rest) = atoms.split_last()?;
    let (core, punctuation) = if !last.is_locked && matches!(last.surface.as_str(), "。" | ".") {
        (rest, Some(last.surface.clone()))
    } else {
        (atoms, None)
    };
    for (index, triple) in core.windows(3).enumerate() {
        let [action, negative, connective] = triple else {
            continue;
        };
        let (Some(action_id), Some(negation), Some(marker)) = (
            action.doc_tok_idx,
            negative.doc_tok_idx,
            connective.doc_tok_idx,
        ) else {
            continue;
        };
        if action.is_locked
            || negative.is_locked
            || connective.is_locked
            || action.upos != Some(Upos::Verb)
            || action.deprel != "obl"
            || negative.surface != "ず"
            || negative.upos != Some(Upos::Aux)
            || negative.deprel != "aux"
            || negative.head != Some(action_id)
            || connective.surface != "に"
            || connective.upos != Some(Upos::Adp)
            || connective.deprel != "case"
            || connective.head != Some(action_id)
            || action.span.end_char != negative.span.start_char
            || negative.span.end_char != connective.span.start_char
        {
            continue;
        }
        let (before, after) = core.split_at(index + 3);
        let (after, comma) = match after {
            [separator, rest @ ..]
                if !separator.is_locked
                    && matches!(separator.surface.as_str(), "、" | "，" | ",") =>
            {
                (rest, separator.surface.as_str())
            }
            _ => (after, "，"),
        };
        if before.iter().any(is_punct_atom)
            || after.iter().any(is_punct_atom)
            || before[..index].iter().any(|atom| {
                if atom.is_locked {
                    !glossary_argument(atom, GlossaryKind::Object, locks)
                } else {
                    atom.upos == Some(Upos::Verb)
                }
            })
        {
            continue;
        }
        let Some(governor) = after
            .iter()
            .find(|atom| !atom.is_locked && atom.deprel == "root" && atom.upos == Some(Upos::Verb))
            .and_then(|atom| atom.doc_tok_idx)
        else {
            continue;
        };
        let Some(locked_source) = departure_source(after, action, governor, doc, chars, locks)
        else {
            continue;
        };
        let clause = |part: &[Atom], trailing_punct: Option<String>| -> Option<Segment> {
            Some(Segment::Clause {
                atoms: part.to_vec(),
                trailing_punct,
                span: Span {
                    start_char: part.first()?.span.start_char,
                    end_char: part.last()?.span.end_char,
                },
            })
        };
        let frame = OmittedDeparture {
            action: action_id,
            negation,
            connective: marker,
            governor,
            locked_source,
        };
        return Some((
            vec![
                clause(before, Some(comma.to_owned()))?,
                clause(after, punctuation)?,
            ],
            frame,
        ));
    }
    None
}

fn completed_verb(after: &[Atom], governor: usize) -> Option<(&[Atom], &Atom)> {
    let index = after
        .iter()
        .position(|atom| atom.doc_tok_idx == Some(governor))?;
    let (arguments, predicate) = after.split_at(index);
    let [verb, ending @ ..] = predicate else {
        return None;
    };
    if verb.is_locked
        || verb.upos != Some(Upos::Verb)
        || verb.deprel != "root"
        || verb.lemma.as_deref() != Some("出る")
        || !match ending {
            [past] => past.surface == "た",
            [polite, past] => polite.surface == "まし" && past.surface == "た",
            _ => false,
        }
        || verb.span.end_char != ending[0].span.start_char
        || ending
            .windows(2)
            .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
        || ending.iter().any(|atom| {
            atom.is_locked
                || atom.upos != Some(Upos::Aux)
                || atom.deprel != "aux"
                || atom.head != Some(governor)
        })
        || arguments.iter().any(|atom| {
            !atom.is_locked && atom.upos == Some(Upos::Verb)
                || atom.deprel == "nsubj" && atom.head == Some(governor)
        })
    {
        return None;
    }
    Some((arguments, verb))
}

fn physical_origin(arguments: &[Atom], governor: usize) -> Option<Span> {
    let mut sources = arguments
        .iter()
        .filter(|atom| atom.head == Some(governor) && atom.deprel == "obj");
    let place = sources.next()?;
    if sources.next().is_some()
        || place.is_locked
        || place.upos != Some(Upos::Noun)
        || !(is_existing_spatial_head(place) || spatial_route_marker(&place.surface).is_some())
    {
        return None;
    }
    let place_id = place.doc_tok_idx?;
    let marker = arguments.iter().find(|marker| {
        !marker.is_locked
            && marker.surface == "を"
            && marker.upos == Some(Upos::Adp)
            && marker.deprel == "case"
            && marker.head == Some(place_id)
            && marker.span.start_char == place.span.end_char
    })?;
    Some(Span {
        start_char: place.span.start_char,
        end_char: marker.span.end_char,
    })
}

fn departure_source(
    after: &[Atom],
    omission: &Atom,
    governor: usize,
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<Option<ProtectedOrigin>> {
    let (arguments, verb) = completed_verb(after, governor)?;
    let mut protected = arguments.iter().filter(|atom| atom.is_locked);
    let Some(lock) = protected.next() else {
        if omission.head != Some(governor) {
            return None;
        }
        return physical_origin(arguments, governor).map(|_| None);
    };
    if protected.next().is_some()
        || !glossary_argument(lock, GlossaryKind::Place, locks)
        || !doc.options.enrich_morphology
        || !doc.options.cpu_syntax
    {
        return None;
    }
    let linked = omission.head?;
    if linked != governor {
        let span = doc.tokens.get(linked)?.span;
        if span.start_char < lock.span.start_char || span.end_char > lock.span.end_char {
            return None;
        }
    }

    // A caller's lock can cover を. Recover the case from the unhinted source
    // while retaining the original lock table for the replacement and its hit.
    let sentence = doc.tokens.get(omission.doc_tok_idx?)?.sentence;
    let start = doc.segmentation.sentences.get(sentence)?.span.start_char;
    let source = chars.get(start..after.last()?.span.end_char)?;
    let surface: String = chars
        .get(lock.span.start_char..lock.span.end_char)?
        .iter()
        .collect();
    let mut options = doc.options.clone();
    options.lexicon.retain(|hint| hint != &surface);
    options.names.retain(|hint| hint != &surface);
    let parsed = analyze_document(&source.iter().collect::<String>(), &options);
    if parsed.segmentation.sentences.len() != 1 {
        return None;
    }
    let tokens: Vec<_> = parsed.tokens.iter().collect();
    let original = build_sentence_atoms(0, source, &parsed, &tokens, &SpanLockTable::new());
    let original_governor = original
        .iter()
        .find(|atom| {
            atom.span.start_char + start == verb.span.start_char
                && atom.span.end_char + start == verb.span.end_char
        })?
        .doc_tok_idx?;
    let original_omission = original.iter().find(|atom| {
        atom.span.start_char + start == omission.span.start_char
            && atom.span.end_char + start == omission.span.end_char
    })?;
    if original_omission.head != Some(original_governor)
        || original_omission.deprel != "obl"
        || original_omission.upos != Some(Upos::Verb)
    {
        return None;
    }
    let tail = after.first()?.span.start_char.checked_sub(start)?;
    let tail_start = original.partition_point(|atom| atom.span.start_char < tail);
    let (arguments, _) = completed_verb(&original[tail_start..], original_governor)?;
    let source = physical_origin(arguments, original_governor)?;
    let end = start + source.end_char;
    (lock.span.start_char == start + source.start_char
        && [end, end - 1].contains(&lock.span.end_char))
    .then_some(Some(ProtectedOrigin {
        lock: lock.span,
        noun_end: end - 1,
    }))
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::{analyze_document, model::Backend};

    use crate::{
        GlossaryEntry, GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary,
        analyze_and_translate, doc_options_for_translate, translate_document,
    };

    #[test]
    fn completed_departure_keeps_the_omission_and_protected_origin() {
        let source = "朝ご飯を食べずに家を出た。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let entry = |surface: &str, target: &str| GlossaryEntry {
            canonical: surface.into(),
            surfaces: vec![surface.into()],
            targets: [(TargetVariant::ZhHant, target.into())].into(),
            ..GlossaryEntry::default()
        };
        let empty = TranslationGlossary::new();
        let doc = analyze_document(source, &doc_options_for_translate(&options, &empty));
        let native = analyze_and_translate(source, &options, &empty, None, Backend::Onnx).unwrap();
        let expected = match faberline_nlp::lexicon::ja::lookup_ja_gloss(
            "朝ご飯",
            Some(faberline_nlp::doc::Upos::Noun),
        ) {
            None => "早上沒有吃飯就離開家了。",
            Some("早餐") => "沒有吃早餐就離開家了。",
            Some(gloss) => panic!("unexpected breakfast gloss: {gloss}"),
        };
        assert_eq!(native.target_text, expected);
        for origin in ["家", "家を"] {
            for kind in [GlossaryKind::Place, GlossaryKind::DomainTerm] {
                let mut glossary = TranslationGlossary::new();
                glossary.objects.push(entry("朝ご飯", "早餐"));
                if kind == GlossaryKind::Place {
                    glossary.places.push(entry(origin, "舊屋"));
                } else {
                    glossary.domain_terms.push(entry(origin, "舊屋"));
                }
                let translated =
                    analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                        .unwrap();
                let direct = translate_document(
                    source,
                    &doc,
                    #[cfg(feature = "narrative")]
                    None,
                    &options,
                    &glossary,
                    None,
                )
                .unwrap();
                assert_eq!(
                    translated.target_text, "沒有吃早餐就離開舊屋了。",
                    "{kind:?} {origin}"
                );
                assert_eq!(translated.target_text, direct.target_text);
                assert_eq!(translated.glossary_hits, direct.glossary_hits);
                let [_, place] = translated.glossary_hits.as_slice() else {
                    panic!("expected a meal and a place hit");
                };
                let extract = |text: &str, span: faberline_nlp::Span| {
                    text.chars()
                        .skip(span.start_char)
                        .take(span.end_char - span.start_char)
                        .collect::<String>()
                };
                assert_eq!(place.kind, kind);
                assert_eq!(extract(source, place.source_span), origin);
                assert_eq!(extract(&translated.target_text, place.target_span), "舊屋");
            }
        }
    }
}
