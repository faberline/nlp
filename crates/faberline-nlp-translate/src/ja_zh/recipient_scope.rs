//! Preserve the recipient restriction and the matching eligibility question.

use faberline_nlp::Span;
use faberline_nlp::doc::{Document, Token, Upos};

use crate::lock::{LockedSpan, SpanLockTable};
use crate::schema::TargetVariant;

use super::{
    Atom, apply_phrase_matches, build_sentence_atoms, core_non_punct, is_object_or_domain_lock,
    nominal_possession, normalize_ja_atoms,
};

pub(super) const EXCLUSION_OUTSIDE: &str = "__ja_group_exclusion_outside";

fn adjacent(atoms: &[&Atom]) -> bool {
    atoms
        .windows(2)
        .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
}

fn excludes_people(atoms: &[Atom], from: usize) -> bool {
    let Some(start) = from.checked_sub(1) else {
        return false;
    };
    let [group, from, outside, no, person, ..] = &atoms[start..] else {
        return false;
    };
    let phrase = [group, from, outside, no, person];
    phrase
        .iter()
        .all(|atom| !atom.is_locked && atom.doc_tok_idx.is_some())
        && adjacent(&phrase)
        && matches!(group.upos, Some(Upos::Noun | Upos::Propn))
        && group.deprel == "compound"
        && group.head == outside.doc_tok_idx
        && from.surface == "以"
        && from.deprel == "compound"
        && from.head == outside.doc_tok_idx
        && outside.surface == "外"
        && outside.deprel == "nmod"
        && outside.head == person.doc_tok_idx
        && no.surface == "の"
        && no.upos == Some(Upos::Adp)
        && no.deprel == "case"
        && no.head == outside.doc_tok_idx
        && person.surface == "人"
        && person.upos == Some(Upos::Noun)
}

pub(super) fn mark_exclusions(atoms: &mut [Atom]) {
    for index in 1..atoms.len().saturating_sub(3) {
        if excludes_people(atoms, index) {
            atoms[index + 1].lemma = Some(EXCLUSION_OUTSIDE.to_owned());
        }
    }
}

pub(super) fn distributes_only_to(atoms: &[Atom]) -> bool {
    let [.., recipient, only, ni, verb, polite] = atoms else {
        return false;
    };
    let phrase = [recipient, only, ni, verb, polite];
    phrase
        .iter()
        .all(|atom| !atom.is_locked && atom.doc_tok_idx.is_some())
        && adjacent(&phrase)
        && matches!(recipient.upos, Some(Upos::Noun | Upos::Propn))
        && recipient.deprel == "obl"
        && recipient.head == verb.doc_tok_idx
        && only.surface == "だけ"
        && only.upos == Some(Upos::Part)
        && only.deprel == "discourse"
        && only.head == recipient.doc_tok_idx
        && ni.surface == "に"
        && ni.upos == Some(Upos::Adp)
        && ni.deprel == "case"
        && ni.head == recipient.doc_tok_idx
        && verb.lemma.as_deref() == Some("配る")
        && verb.upos == Some(Upos::Verb)
        && verb.deprel == "root"
        && polite.surface == "ます"
        && polite.upos == Some(Upos::Aux)
        && polite.deprel == "aux"
        && polite.head == verb.doc_tok_idx
}

struct Question<'a> {
    group: &'a [Atom],
    item: &'a [Atom],
    case: &'a Atom,
}

fn question(atoms: &[Atom]) -> Option<Question<'_>> {
    let [prefix @ .., noun, case, root, masu, negative, ka] = atoms else {
        return None;
    };
    let item_start = compound_start(prefix, noun);
    let item = &atoms[item_start..=prefix.len()];
    let [group @ .., _, outside, _, receiver, topic] = &prefix[..item_start] else {
        return None;
    };
    let [group_modifiers @ .., group_head] = group else {
        return None;
    };
    let predicate = Some(root.doc_tok_idx?);
    if !excludes_people(atoms, group.len())
        || !group_modifiers
            .iter()
            .all(|member| compound_of(member, group_head) || compound_of(member, outside))
        || receiver.deprel != "nsubj"
        || receiver.head != predicate
        || topic.surface != "は"
        || !super::dasu_case_link(receiver, topic)
        || !matches!(noun.upos, Some(Upos::Noun | Upos::Propn))
        || noun.deprel != "obj"
        || noun.head != predicate
        || case.surface != "を"
        || !super::dasu_case_link(noun, case)
        || root.lemma.as_deref() != Some("いただける")
        || root.upos != Some(Upos::Verb)
        || root.deprel != "root"
        || masu.surface != "ませ"
        || negative.surface != "ん"
        || ka.surface != "か"
        || masu.upos != Some(Upos::Aux)
        || negative.upos != Some(Upos::Aux)
        || ka.upos != Some(Upos::Part)
        || [masu, negative, ka]
            .iter()
            .any(|atom| atom.head != predicate)
        || !contiguous(atoms)
    {
        return None;
    }
    Some(Question { group, item, case })
}

fn compound_start(prefix: &[Atom], head: &Atom) -> usize {
    prefix
        .iter()
        .rposition(|atom| !compound_of(atom, head))
        .map_or(0, |index| index + 1)
}

fn compound_of(atom: &Atom, head: &Atom) -> bool {
    !atom.is_locked
        && matches!(atom.upos, Some(Upos::Noun | Upos::Propn))
        && atom.deprel == "compound"
        && head.doc_tok_idx.is_some()
        && atom.head == head.doc_tok_idx
}

fn contiguous(atoms: &[Atom]) -> bool {
    atoms
        .windows(2)
        .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
}

fn distribution_item(atoms: &[Atom]) -> Option<(&[Atom], &[Atom])> {
    if !distributes_only_to(atoms) {
        return None;
    }
    let [.., recipient, _, _, verb, _] = atoms else {
        return None;
    };
    let topic_index = atoms.iter().position(|atom| atom.surface == "は")?;
    let [modifiers @ .., noun, topic] = &atoms[..=topic_index] else {
        return None;
    };
    let item_start = compound_start(modifiers, noun);
    let item = &atoms[item_start..topic_index];
    let group = atoms.get(topic_index + 1..atoms.len().checked_sub(4)?)?;
    let [compounds @ .., _] = group else {
        return None;
    };
    let item_start_char = item.first()?.span.start_char;
    if noun.upos != Some(Upos::Noun)
        || noun.deprel != "nsubj"
        || noun.head != verb.doc_tok_idx
        || !super::dasu_case_link(noun, topic)
        || !contiguous(item)
        || !contiguous(group)
        || modifiers[..item_start]
            .iter()
            .any(|atom| super::attached_demonstrative_at(atom, noun, item_start_char).is_none())
        || !compounds.iter().all(|atom| compound_of(atom, recipient))
    {
        return None;
    }
    Some((group, item))
}

fn tokens_in(doc: &Document, sentence: usize) -> Vec<&Token> {
    let start = doc
        .tokens
        .partition_point(|token| token.sentence < sentence);
    let end = doc
        .tokens
        .partition_point(|token| token.sentence <= sentence);
    doc.tokens[start..end].iter().collect()
}

fn matches_distribution(
    question: &Question<'_>,
    sentence: usize,
    doc: &Document,
    chars: &[char],
    unlocked: &SpanLockTable,
) -> Option<()> {
    let previous_idx = sentence.checked_sub(1)?;
    let previous = &doc.segmentation.sentences[previous_idx];
    let current = &doc.segmentation.sentences[sentence];
    if previous.paragraph != current.paragraph
        || doc.segmentation.opaque.iter().any(|part| {
            part.span.start_char < previous.span.end_char
                && previous.span.start_char < part.span.end_char
        })
        || super::nominal_receiving_glossary::has_quote_marks(
            chars.get(previous.span.start_char..previous.span.end_char)?,
        )
        || !chars
            .get(previous.span.end_char..current.span.start_char)?
            .iter()
            .all(|ch| ch.is_whitespace())
    {
        return None;
    }
    let tokens = tokens_in(doc, previous_idx);
    let statement = build_sentence_atoms(previous_idx, chars, doc, &tokens, unlocked);
    if statement.last()?.surface != "。" {
        return None;
    }
    let (group, item) = distribution_item(core_non_punct(&statement))?;
    let text =
        |atoms: &[Atom]| chars.get(atoms.first()?.span.start_char..atoms.last()?.span.end_char);
    (text(group)? == text(question.group)? && text(item)? == text(question.item)?).then_some(())
}

fn object_lock<'a>(atoms: &[Atom], locks: &'a SpanLockTable) -> Option<Option<&'a LockedSpan>> {
    let mut protected = atoms.iter().filter(|atom| atom.is_locked);
    let Some(atom) = protected.next() else {
        return Some(None);
    };
    if protected.next().is_some() {
        return None;
    }
    locks
        .locks
        .iter()
        .find(|lock| {
            lock.source_span == atom.span
                && lock.placeholder == atom.surface
                && is_object_or_domain_lock(lock)
        })
        .map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render_question<'source>(
    masked: &[Atom],
    chars: &[char],
    doc: &Document,
    sent_tokens: &[&Token],
    sentence: usize,
    locks: &SpanLockTable,
    unhinted: &dyn Fn() -> Option<&'source Document>,
) -> Option<String> {
    let [.., verb, masu, neg, ka] = masked else {
        return None;
    };
    if verb.surface != "いただけ"
        || masu.surface != "ませ"
        || neg.surface != "ん"
        || ka.surface != "か"
        || !masked
            .windows(2)
            .any(|pair| pair[0].surface == "以" && pair[1].surface == "外")
    {
        return None;
    }
    let lock = object_lock(masked, locks)?;
    let span = Span {
        start_char: masked.first()?.span.start_char,
        end_char: ka.span.end_char,
    };
    let unlocked = SpanLockTable::default();
    let prove = |analysis: &Document, tokens: &[&Token]| {
        if analysis.segmentation.opaque.iter().any(|part| {
            part.span.start_char < span.end_char && span.start_char < part.span.end_char
        }) {
            return None;
        }
        let plain = build_sentence_atoms(sentence, chars, analysis, tokens, &unlocked);
        let body = core_non_punct(&plain);
        if body.first()?.span.start_char != span.start_char
            || body.last()?.span.end_char != span.end_char
        {
            return None;
        }
        let question = question(body)?;
        let item_end = question.item.last()?.span.end_char;
        if let Some(term) = lock
            && (term.source_span.start_char != question.item.first()?.span.start_char
                || (term.source_span.end_char != item_end
                    && term.source_span.end_char != question.case.span.end_char))
        {
            return None;
        }
        let eligible =
            matches_distribution(&question, sentence, analysis, chars, &unlocked).is_some();
        if !eligible
            && !lock.is_some_and(|term| term.source_span.end_char == question.case.span.end_char)
        {
            return None;
        }
        let render_nominal = |phrase| {
            let atoms = normalize_ja_atoms(&apply_phrase_matches(phrase));
            nominal_possession::noun_phrase(&atoms, analysis, sentence, TargetVariant::ZhHant)
                .filter(|text| !text.is_empty())
        };
        let group = render_nominal(question.group)?;
        let item = match lock {
            Some(term) => term.placeholder.clone(),
            None => render_nominal(question.item)?,
        };
        let action = if eligible { "領取" } else { "得到" };
        Some((eligible, format!("{group}以外的人不能{action}{item}嗎")))
    };
    let existing = prove(doc, sent_tokens);
    if existing.as_ref().is_some_and(|(eligible, _)| *eligible) || doc.options.lexicon.is_empty() {
        return existing.map(|(_, text)| text);
    }
    unhinted()
        .and_then(|source| prove(source, &tokens_in(source, sentence)))
        .or(existing)
        .map(|(_, text)| text)
}
