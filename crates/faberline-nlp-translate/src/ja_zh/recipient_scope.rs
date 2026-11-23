//! Keep restrictions on the group that receives an item.

use faberline_nlp::doc::{Document, analyze_document};

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

pub(super) fn restore_eligibility_object(atoms: &mut Vec<Atom>, doc: &Document, chars: &[char]) {
    let Some((object_index, receiver_index, mut particle)) =
        eligibility_object(core_non_punct(atoms), doc, chars)
    else {
        return;
    };
    let predicate = atoms[object_index + 1].doc_tok_idx;
    let mut object = atoms[object_index].clone();
    object.span.end_char = particle.span.start_char;
    object.upos = Some(Upos::Noun);
    object.doc_tok_idx = None;
    object.head = predicate;
    object.deprel = "obj".to_owned();
    particle.doc_tok_idx = None;
    particle.head = None;
    // The compound hint can make the receiver depend on the masked object
    // as a verb. Restore its attachment to the finite receiving verb.
    atoms[receiver_index].head = predicate;
    // The lock table still owns the placeholder and the full source span.
    atoms.splice(object_index..=object_index, [object, particle]);
}

fn eligibility_object(
    atoms: &[Atom],
    doc: &Document,
    chars: &[char],
) -> Option<(usize, usize, Atom)> {
    let [subject @ .., lock, verb, polite, neg, question] = atoms else {
        return None;
    };
    if lock.lemma.as_deref() != Some(LOCKED_OBJECT_ATOM)
        || !lock.is_locked
        || atoms.iter().filter(|atom| atom.is_locked).count() != 1
        || !doc.options.cpu_syntax
        || !doc.options.enrich_morphology
        || verb.lemma.as_deref() != Some("いただける")
        || polite.surface != "ませ"
        || neg.surface != "ん"
        || question.surface != "か"
    {
        return None;
    }
    verb.doc_tok_idx?;
    let term = chars.get(lock.span.start_char..lock.span.end_char)?;
    if term.last() != Some(&'を') {
        return None;
    }

    let start = subject.first()?.span.start_char;
    let end = question.span.end_char;
    let source = chars.get(start..end)?;
    let term: String = term.iter().collect();
    let mut options = doc.options.clone();
    options.lexicon.retain(|hint| hint != &term);
    options.names.retain(|hint| hint != &term);
    let parsed = analyze_document(&source.iter().collect::<String>(), &options);
    if parsed.segmentation.sentences.len() != 1 {
        return None;
    }
    let tokens: Vec<_> = parsed.tokens.iter().collect();
    let raw = build_sentence_atoms(0, source, &parsed, &tokens, &SpanLockTable::default());
    let EligibilityQuestion {
        receiver,
        item,
        case,
        root,
        ..
    } = eligibility_question(core_non_punct(&raw))?;
    if item.first()?.span.start_char + start != lock.span.start_char
        || case.span.end_char + start != lock.span.end_char
        || root.span.start_char + start != verb.span.start_char
        || root.span.end_char + start != verb.span.end_char
    {
        return None;
    }
    let receiver_index = subject.iter().position(|atom| {
        atom.span.start_char == receiver.span.start_char + start
            && atom.span.end_char == receiver.span.end_char + start
    })?;
    let mut particle = case.clone();
    particle.span.start_char += start;
    particle.span.end_char += start;
    Some((subject.len(), receiver_index, particle))
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

pub(super) fn receives_distributed_item(atoms: &[Atom], doc: &Document, chars: &[char]) -> bool {
    receiving_distribution(atoms, doc, chars).is_some()
}

fn receiving_distribution(atoms: &[Atom], doc: &Document, chars: &[char]) -> Option<()> {
    let atoms = core_non_punct(atoms);
    let [.., verb, polite, negative, question] = atoms else {
        return None;
    };
    if verb.is_locked
        || verb.lemma.as_deref() != Some("いただける")
        || polite.surface != "ませ"
        || negative.surface != "ん"
        || question.surface != "か"
        || !doc.options.cpu_syntax
        || !doc.options.enrich_morphology
    {
        return None;
    }
    let sentence = doc.tokens.get(verb.doc_tok_idx?)?.sentence;
    let previous = doc.segmentation.sentences.get(sentence.checked_sub(1)?)?;
    let current = doc.segmentation.sentences.get(sentence)?;
    if previous.paragraph != current.paragraph {
        return None;
    }
    let passage = chars.get(previous.span.start_char..current.span.end_char)?;
    let mut options = doc.options.clone();
    let mut locks = atoms.iter().enumerate().filter(|(_, atom)| atom.is_locked);
    if let Some((index, lock)) = locks.next() {
        if lock.lemma.as_deref() != Some(LOCKED_OBJECT_ATOM) || locks.next().is_some() {
            return None;
        }
        // Restoration can split the original object lock from its case marker.
        // Omit either spelling as a tokenization hint for this syntax check.
        let bare: String = chars
            .get(lock.span.start_char..lock.span.end_char)?
            .iter()
            .collect();
        let case_end = atoms
            .get(index + 1)
            .filter(|next| next.surface == "を" && next.span.start_char == lock.span.end_char)
            .map_or(lock.span.end_char, |next| next.span.end_char);
        let with_case: String = chars.get(lock.span.start_char..case_end)?.iter().collect();
        options
            .lexicon
            .retain(|hint| hint != &bare && hint != &with_case);
        options
            .names
            .retain(|hint| hint != &bare && hint != &with_case);
    }
    let parsed = analyze_document(&passage.iter().collect::<String>(), &options);
    if parsed.segmentation.sentences.len() != 2 {
        return None;
    }
    let raw_atoms = |sentence| {
        let tokens: Vec<_> = parsed
            .tokens
            .iter()
            .filter(|token| token.sentence == sentence)
            .collect();
        build_sentence_atoms(
            sentence,
            passage,
            &parsed,
            &tokens,
            &SpanLockTable::default(),
        )
    };
    let statement = raw_atoms(0);
    let question = raw_atoms(1);
    if statement.last()?.surface != "。" {
        return None;
    }
    let (group, item) = distribution_item(core_non_punct(&statement))?;
    let request = eligibility_question(core_non_punct(&question))?;
    let same_predicate = request.root.span.start_char + previous.span.start_char
        == verb.span.start_char
        && request.root.span.end_char + previous.span.start_char == verb.span.end_char;
    let span_text =
        |group: &[Atom]| passage.get(group.first()?.span.start_char..group.last()?.span.end_char);
    (same_predicate
        && span_text(item)? == span_text(request.item)?
        && span_text(group)? == span_text(request.group)?)
    .then_some(())
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
    fn eligibility_keeps_the_glossary_object_after_the_negated_verb() {
        let source = "この資料は参加者だけに配ります。参加者以外の人は資料をいただけませんか。";
        let expected = "這份資料只發給參加者。參加者以外的人不能領取講義嗎？";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let mut glossary = TranslationGlossary::new();
        glossary.objects.push(GlossaryEntry {
            canonical: "資料を".to_owned(),
            surfaces: vec!["資料を".to_owned()],
            targets: [(TargetVariant::ZhHant, "講義".to_owned())].into(),
            ..GlossaryEntry::default()
        });
        let translated =
            analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
        assert_eq!(translated.target_text, expected);
        let [hit] = translated.glossary_hits.as_slice() else {
            panic!("expected one object glossary hit");
        };
        let slice = |text: &str, span: faberline_nlp::Span| -> String {
            text.chars()
                .skip(span.start_char)
                .take(span.end_char - span.start_char)
                .collect()
        };
        assert_eq!(slice(source, hit.source_span), "資料を");
        assert_eq!(slice(expected, hit.target_span), "講義");
    }
}
