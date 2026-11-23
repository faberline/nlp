//! Keep restrictions on the group that receives an item.

use faberline_nlp::Span;
use faberline_nlp::doc::Document;

use super::naide_glossary::sole_object_lock;
use super::{Atom, LOCKED_OBJECT_ATOM, Upos, build_sentence_atoms, core_non_punct};
use crate::lock::SpanLockTable;

pub(super) const EXCLUSION_OUTSIDE: &str = "__ja_group_exclusion_outside";

fn adjacent(atoms: &[&Atom]) -> bool {
    atoms
        .windows(2)
        .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
}

fn excludes_people(atoms: &[Atom], index: usize) -> bool {
    let Some(start) = index.checked_sub(1) else {
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

pub(super) fn may_restore(text: &str, locks: &SpanLockTable) -> bool {
    text.contains("以外")
        && text.contains("いただけませんか")
        && (locks.is_empty() || sole_object_lock(locks).is_some())
}

struct CoveredObject {
    lock: Span,
    receiver: Span,
    verb: Span,
    particle: Atom,
}

#[derive(Default)]
pub(super) struct EligibilitySyntax {
    object: Option<CoveredObject>,
    pub(super) pickup: Option<Span>,
}

impl EligibilitySyntax {
    pub(super) fn restore_object(&self, atoms: &mut Vec<Atom>) {
        let Some(covered) = &self.object else { return };
        let body = core_non_punct(atoms);
        let [prefix @ .., lock, verb, _, _, _] = body else {
            return;
        };
        if !lock.is_locked
            || lock.lemma.as_deref() != Some(LOCKED_OBJECT_ATOM)
            || lock.span != covered.lock
            || verb.is_locked
            || verb.span != covered.verb
        {
            return;
        }
        let Some(predicate) = verb.doc_tok_idx else {
            return;
        };
        let Some(receiver) = prefix
            .iter()
            .position(|atom| !atom.is_locked && atom.span == covered.receiver)
        else {
            return;
        };
        let index = prefix.len();
        let mut object = lock.clone();
        let mut particle = covered.particle.clone();
        object.span.end_char = particle.span.start_char;
        object.upos = Some(Upos::Noun);
        object.doc_tok_idx = None;
        object.head = Some(predicate);
        object.deprel = "obj".to_owned();
        particle.doc_tok_idx = None;
        particle.head = None;
        atoms[receiver].head = Some(predicate);
        // The original table still restores the full caller match.
        atoms.splice(index..=index, [object, particle]);
    }
}

pub(super) fn prove(
    atoms: &[Atom],
    sentence: usize,
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<EligibilitySyntax> {
    let request = eligibility_question(core_non_punct(atoms))?;
    let lock = if locks.is_empty() {
        None
    } else {
        Some(sole_object_lock(locks)?)
    };
    let start = request.item.first()?.span.start_char;
    let case_end = request.case.span.end_char;
    if lock.is_some_and(|entry| {
        entry.source_span.start_char < start || entry.source_span.end_char > case_end
    }) {
        return None;
    }
    let object = lock
        .filter(|entry| {
            entry.source_span.start_char == start && entry.source_span.end_char == case_end
        })
        .map(|entry| CoveredObject {
            lock: entry.source_span,
            receiver: request.receiver.span,
            verb: request.root.span,
            particle: request.case.clone(),
        });
    let pickup = matches_distribution(&request, sentence, doc, chars).then_some(request.root.span);
    (object.is_some() || pickup.is_some()).then_some(EligibilitySyntax { object, pickup })
}

pub(super) fn matches_pickup_predicate(atoms: &[Atom], proof: Option<Span>) -> bool {
    matches!(core_non_punct(atoms), [.., verb, polite, negative, question]
        if !verb.is_locked && Some(verb.span) == proof
            && verb.lemma.as_deref() == Some("いただける")
            && polite.surface == "ませ" && negative.surface == "ん" && question.surface == "か")
}

struct EligibilityQuestion<'a> {
    group: &'a [Atom],
    receiver: &'a Atom,
    item: &'a [Atom],
    case: &'a Atom,
    root: &'a Atom,
}

fn eligibility_question(atoms: &[Atom]) -> Option<EligibilityQuestion<'_>> {
    let [prefix @ .., noun, case, root, masu, negative, ka] = atoms else {
        return None;
    };
    let item_start = compound_start(prefix, noun);
    let item = &atoms[item_start..=prefix.len()];
    let [group @ .., _, _, _, receiver, topic] = &prefix[..item_start] else {
        return None;
    };
    let predicate = Some(root.doc_tok_idx?);
    if !excludes_people(atoms, group.len())
        || receiver.deprel != "nsubj"
        || receiver.head != predicate
        || topic.surface != "は"
        || topic.upos != Some(Upos::Adp)
        || topic.deprel != "case"
        || topic.head != receiver.doc_tok_idx
        || !matches!(noun.upos, Some(Upos::Noun | Upos::Propn))
        || noun.deprel != "obj"
        || noun.head != predicate
        || case.surface != "を"
        || case.upos != Some(Upos::Adp)
        || case.deprel != "case"
        || case.head != noun.doc_tok_idx
        || root.lemma.as_deref() != Some("いただける")
        || root.upos != Some(Upos::Verb)
        || root.deprel != "root"
        || masu.surface != "ませ"
        || negative.surface != "ん"
        || ka.surface != "か"
        || [masu, negative, ka]
            .iter()
            .any(|atom| atom.head != predicate)
        || !contiguous(item)
        || !adjacent(&[noun, case, root, masu, negative, ka])
    {
        return None;
    }
    Some(EligibilityQuestion {
        group,
        receiver,
        item,
        case,
        root,
    })
}

fn matches_distribution(
    request: &EligibilityQuestion<'_>,
    sentence: usize,
    doc: &Document,
    chars: &[char],
) -> bool {
    let Some(previous_idx) = sentence.checked_sub(1) else {
        return false;
    };
    let previous = &doc.segmentation.sentences[previous_idx];
    let current = &doc.segmentation.sentences[sentence];
    if previous.paragraph != current.paragraph
        || !chars
            .get(previous.span.end_char..current.span.start_char)
            .is_some_and(|gap| gap.iter().all(|ch| ch.is_whitespace()))
    {
        return false;
    }
    let first = doc
        .tokens
        .partition_point(|token| token.sentence < previous_idx);
    let last = doc
        .tokens
        .partition_point(|token| token.sentence <= previous_idx);
    let tokens = doc.tokens[first..last].iter().collect::<Vec<_>>();
    let statement = build_sentence_atoms(previous_idx, chars, doc, &tokens, &SpanLockTable::new());
    if !statement.last().is_some_and(|atom| atom.surface == "。") {
        return false;
    }
    let Some((group, item)) = distribution_item(core_non_punct(&statement)) else {
        return false;
    };
    let source =
        |atoms: &[Atom]| chars.get(atoms.first()?.span.start_char..atoms.last()?.span.end_char);
    source(item).is_some_and(|text| Some(text) == source(request.item))
        && source(group).is_some_and(|text| Some(text) == source(request.group))
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
    if noun.upos != Some(Upos::Noun)
        || noun.deprel != "nsubj"
        || noun.head != verb.doc_tok_idx
        || topic.deprel != "case"
        || topic.head != noun.doc_tok_idx
        || !adjacent(&[noun, topic])
        || !contiguous(item)
        || !contiguous(group)
        || modifiers[..item_start].iter().any(|atom| {
            atom.upos != Some(Upos::Det) || atom.deprel != "det" || atom.head != noun.doc_tok_idx
        })
        || !compounds.iter().all(|atom| compound_of(atom, recipient))
    {
        return None;
    }
    Some((group, item))
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

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::model::Backend;

    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
    };

    #[test]
    fn keeps_the_recipient_restriction_and_the_eligibility_question() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let translate = |source| {
            analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                .unwrap()
                .target_text
        };
        assert_eq!(
            translate("入場券は学生だけに配ります。学生以外の人は入場券をいただけませんか。"),
            "入場券只發給學生。學生以外的人不能領取入場券嗎？"
        );
        assert!(translate("建物の外の人は待ってください。").contains("外面"));
    }

    #[test]
    fn pickup_requires_the_same_group_and_item_in_the_previous_statement() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let translate = |source: &str| {
            analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                .unwrap()
                .target_text
        };
        let question = "参加者以外の人は資料をいただけませんか。";
        for preceding in [
            "",
            "この資料は学生だけに配ります。",
            "入場券は参加者だけに配ります。",
            "この資料は参加者だけに配ります。\n\n",
        ] {
            let text = format!("{preceding}{question}");
            assert!(translate(&text).ends_with("不能得到資料嗎？"), "{text}");
        }
        assert_eq!(
            translate("この資料をいただけませんか。"),
            "可以給我這份資料嗎？"
        );
    }

    #[test]
    fn eligibility_preserves_glossary_spans_in_either_sentence() {
        let source = "この資料は参加者だけに配ります。参加者以外の人は資料をいただけませんか。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let slice = |text: &str, span: faberline_nlp::Span| -> String {
            text.chars()
                .skip(span.start_char)
                .take(span.end_char - span.start_char)
                .collect()
        };
        for (term, gloss, expected, count) in [
            (
                "資料を",
                "講義",
                "這份資料只發給參加者。參加者以外的人不能領取講義嗎？",
                1,
            ),
            (
                "この資料",
                "本資料",
                "本資料只發給參加者。參加者以外的人不能領取資料嗎？",
                1,
            ),
            (
                "資料",
                "講義",
                "這份講義只發給參加者。參加者以外的人不能領取講義嗎？",
                2,
            ),
        ] {
            let mut glossary = TranslationGlossary::new();
            glossary.objects.push(GlossaryEntry {
                canonical: term.to_owned(),
                surfaces: vec![term.to_owned()],
                targets: [(TargetVariant::ZhHant, gloss.to_owned())].into(),
                ..GlossaryEntry::default()
            });
            let translated =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(translated.target_text, expected);
            assert_eq!(translated.glossary_hits.len(), count);
            for hit in &translated.glossary_hits {
                assert_eq!(hit.kind, crate::GlossaryKind::Object);
                assert_eq!(slice(source, hit.source_span), term);
                assert_eq!(slice(expected, hit.target_span), gloss);
            }
        }
    }
}
