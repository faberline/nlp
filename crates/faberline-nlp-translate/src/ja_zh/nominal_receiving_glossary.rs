//! Keep a caller's protected noun in a request proven by the source syntax.
use super::{
    Atom, Document, GlossaryKind, LockedSpan, NominalRequestContext, SpanLockTable, TargetVariant,
    Token, build_sentence_atoms, nominal_possession, predicate_glossary,
    render_nominal_receiving_request,
};

const QUOTE_MARKS: [char; 7] = ['「', '」', '『', '』', '"', '“', '”'];

fn is_object_term(lock: &LockedSpan) -> bool {
    matches!(
        predicate_glossary::kind(lock),
        Some(GlossaryKind::Object | GlossaryKind::DomainTerm)
    )
}

pub(super) fn has_quote_marks(chars: &[char]) -> bool {
    chars.iter().any(|ch| QUOTE_MARKS.contains(ch))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render<'source>(
    atoms: &[Atom],
    chars: &[char],
    doc: &Document,
    sent_tokens: &[&Token],
    sentence: usize,
    locks: &SpanLockTable,
    unhinted: &dyn Fn() -> Option<&'source Document>,
    context: NominalRequestContext,
) -> Option<String> {
    let mut locked = atoms.iter().filter(|atom| atom.is_locked);
    let atom = locked.next()?;
    if locked.next().is_some()
        || !matches!(atoms, [.., verb, masu, neg, question]
            if verb.surface == "いただけ" && masu.surface == "ませ"
                && neg.surface == "ん" && question.surface == "か")
    {
        return None;
    }
    let lock = locks.locks.iter().find(|lock| {
        lock.source_span == atom.span && lock.placeholder == atom.surface && is_object_term(lock)
    })?;
    let start = atoms.first()?.span.start_char;
    let end = atoms.last()?.span.end_char;
    let prove = |source: &Document, tokens: &[&Token]| {
        if source
            .segmentation
            .opaque
            .iter()
            .any(|op| op.span.start_char < end && start < op.span.end_char)
        {
            return None;
        }
        let plain =
            build_sentence_atoms(sentence, chars, source, tokens, &SpanLockTable::default());
        let left = plain.partition_point(|a| a.span.end_char <= start);
        let right = plain.partition_point(|a| a.span.start_char < end);
        let clause = plain.get(left..right)?;
        if clause.first()?.span.start_char != start || clause.last()?.span.end_char != end {
            return None;
        }
        render_nominal_receiving_request(clause, source, sentence, Some(lock), chars, context)
    };
    prove(doc, sent_tokens).or_else(|| {
        // Only the built-in entry points may recover source analysis before
        // their glossary hints combined the object and its case particle.
        let source = unhinted()?;
        let first = source
            .tokens
            .partition_point(|token| token.sentence < sentence);
        let last = source
            .tokens
            .partition_point(|token| token.sentence <= sentence);
        let tokens: Vec<_> = source.tokens[first..last].iter().collect();
        prove(source, &tokens)
    })
}

pub(super) fn render_object(
    phrase: &[Atom],
    lock: &LockedSpan,
    doc: &Document,
    sentence: usize,
    source: &[char],
    time_request: bool,
) -> Option<String> {
    let [modifiers @ .., noun, particle] = phrase else {
        return None;
    };
    if lock.source_span.end_char != noun.span.end_char
        && lock.source_span.end_char != particle.span.end_char
    {
        return None;
    }
    if lock.source_span.start_char == phrase.first()?.span.start_char {
        return Some(lock.placeholder.clone());
    }
    if lock.source_span.start_char != noun.span.start_char {
        return None;
    }
    // The whole request supplies the courtesy conveyed by お.
    if time_request
        && matches!(modifiers, [prefix] if prefix.surface == "お" && prefix.span.end_char == noun.span.start_char)
    {
        return Some(lock.placeholder.clone());
    }
    let mut object = phrase[..phrase.len() - 1].to_vec();
    let masked_noun = object.last_mut()?;
    masked_noun.surface.clone_from(&lock.placeholder);
    masked_noun.is_locked = true;
    nominal_possession::choice_phrase(&object, doc, sentence, TargetVariant::ZhHant, source)
}
