//! Keep the caller's object in a request proven by the original Japanese syntax.

use super::{
    Atom, Document, LOCKED_OBJECT_ATOM, LockedSpan, SpanLockTable, TargetVariant, Token, Upos,
    attached_demonstrative, core_non_punct, dasu_case_link, demonstrative_classifier,
    is_object_or_domain_lock, nominal_receiving_glossary, normalized_sentence_atoms,
    te_help_request, translate_single_ja_term,
};

pub(super) fn restore_object<'source>(
    atoms: &[Atom],
    chars: &[char],
    doc: &Document,
    tokens: &[&Token],
    sentence: usize,
    locks: &SpanLockTable,
    unhinted: &dyn Fn() -> Option<&'source Document>,
) -> Option<Vec<Atom>> {
    let body = core_non_punct(atoms);
    let end = body.last()?;
    if end.is_locked
        || end.surface != "か"
        || !body
            .iter()
            .rev()
            .take(4)
            .any(|atom| !atom.is_locked && matches!(atom.surface.as_str(), "いただけ" | "もらえ"))
    {
        return None;
    }
    let mut protected = body.iter().enumerate().filter(|(_, atom)| atom.is_locked);
    let (index, atom) = protected.next()?;
    if protected.next().is_some() {
        return None;
    }
    let lock = locks.locks.iter().find(|lock| {
        lock.mandatory
            && lock.source_span == atom.span
            && lock.placeholder == atom.surface
            && is_object_or_domain_lock(lock)
    })?;
    let start = body.first()?.span.start_char;
    let end = end.span.end_char;
    if nominal_receiving_glossary::has_quote_marks(chars.get(start..end)?) {
        return None;
    }

    let restore = |source: &Document, source_tokens: &[&Token]| {
        if source
            .segmentation
            .opaque
            .iter()
            .any(|part| part.span.start_char < end && start < part.span.end_char)
        {
            return None;
        }
        let plain = normalized_sentence_atoms(
            sentence,
            chars,
            source,
            source_tokens,
            &SpanLockTable::default(),
        );
        let plain = core_non_punct(&plain);
        if plain.first()?.span.start_char != start || plain.last()?.span.end_char != end {
            return None;
        }
        let request = te_help_request(plain, source, None)?;
        let action = request.action.last()?;
        let SourceObject {
            noun,
            case_is_locked,
            determiner,
        } = direct_object(request.action, action, lock)?;
        // Glossary hints can change token indices. Use the verified source
        // span to find the action in the document that will be rendered.
        let action_index = body
            .iter()
            .find(|other| {
                !other.is_locked
                    && other.span == action.span
                    && other.surface == action.surface
                    && other.lemma == action.lemma
            })?
            .doc_tok_idx?;
        let mut restored = atom.clone();
        restored.doc_tok_idx = None;
        restored.upos = Some(Upos::Noun);
        restored.lemma = Some(LOCKED_OBJECT_ATOM.to_owned());
        restored.deprel = "obj".to_owned();
        restored.head = Some(action_index);
        let left = if let Some((det, demonstrative)) = determiner
            && lock.source_span.start_char == noun.span.start_char
        {
            let left = index.checked_sub(1)?;
            let actual = &body[left];
            if actual.is_locked || actual.span != det.span || actual.surface != det.surface {
                return None;
            }
            let unit = match noun.surface.as_str() {
                "荷物" => "件",
                "書類" => "份",
                _ => {
                    let word =
                        translate_single_ja_term(&noun.surface, noun.lemma.as_deref(), noun.upos);
                    demonstrative_classifier(&word, noun, Some(TargetVariant::ZhHant))
                }
            };
            if unit.is_empty() && !case_is_locked {
                return None;
            }
            restored.span.start_char = det.span.start_char;
            restored
                .surface
                .insert_str(0, &format!("{demonstrative}{unit}"));
            left
        } else if case_is_locked {
            index
        } else {
            return None;
        };
        let mut result = atoms.to_vec();
        result.splice(left..=index, [restored]);
        te_help_request(core_non_punct(&result), doc, None)?;
        Some(result)
    };

    restore(doc, tokens).or_else(|| {
        let source = unhinted()?;
        let first = source
            .tokens
            .partition_point(|token| token.sentence < sentence);
        let last = source
            .tokens
            .partition_point(|token| token.sentence <= sentence);
        let tokens: Vec<_> = source.tokens[first..last].iter().collect();
        restore(source, &tokens)
    })
}

struct SourceObject<'a> {
    noun: &'a Atom,
    case_is_locked: bool,
    determiner: Option<(&'a Atom, char)>,
}

fn direct_object<'a>(
    action: &'a [Atom],
    verb: &Atom,
    lock: &LockedSpan,
) -> Option<SourceObject<'a>> {
    let verb_index = verb.doc_tok_idx?;
    action.windows(2).enumerate().find_map(|(index, pair)| {
        let [noun, particle] = pair else { return None };
        let case_is_locked = lock.source_span.end_char == particle.span.end_char;
        if !matches!(noun.upos, Some(Upos::Noun | Upos::Propn))
            || noun.head != Some(verb_index)
            || noun.deprel != "obj"
            || particle.surface != "を"
            || !dasu_case_link(noun, particle)
            || !(case_is_locked || lock.source_span.end_char == noun.span.end_char)
        {
            return None;
        }
        let determiner = index
            .checked_sub(1)
            .and_then(|previous| action.get(previous))
            .and_then(|det| attached_demonstrative(det, noun).map(|word| (det, word)));
        (lock.source_span.start_char == noun.span.start_char
            || case_is_locked
                && determiner
                    .is_some_and(|(det, _)| det.span.start_char == lock.source_span.start_char))
        .then_some(SourceObject {
            noun,
            case_is_locked,
            determiner,
        })
    })
}
