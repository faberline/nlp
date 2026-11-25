//! Render a payment-method noun phrase and a displayed code example.

use super::{Atom, ClauseAlignment, Document, Span, Upos, dasu_case_link, localize_punct};
use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::GlossaryKind;
use faberline_nlp::doc::OpaqueKind;

fn raw<'a>(atom: &Atom, chars: &'a [char]) -> Option<&'a [char]> {
    chars.get(atom.span.start_char..atom.span.end_char)
}

fn term(
    atom: &Atom,
    chars: &[char],
    locks: &SpanLockTable,
    expected: &str,
    fallback: &str,
) -> Option<String> {
    if raw(atom, chars)?.iter().collect::<String>() != expected {
        return None;
    }
    if !atom.is_locked {
        return Some(fallback.to_owned());
    }
    locks.locks.iter().find(|lock| {
        lock.mandatory
            && lock.source_span == atom.span
            && lock.placeholder == atom.surface
            && matches!(
                lock.origin,
                LockOrigin::Glossary {
                    kind: GlossaryKind::Object | GlossaryKind::DomainTerm,
                    ..
                }
            )
    })?;
    Some(atom.surface.clone())
}

fn output(doc: &Document, sentence: usize, text: String) -> Option<(String, Vec<ClauseAlignment>)> {
    let mut clauses = doc
        .segmentation
        .clauses
        .iter()
        .enumerate()
        .filter(|(_, c)| c.sentence == sentence);
    let (index, clause) = clauses.next()?;
    if clauses.next().is_some() {
        return None;
    }
    let length = text.chars().count();
    Some((
        text,
        vec![ClauseAlignment {
            clause: index,
            sentence,
            source_span: clause.span,
            target_span: Span {
                start_char: 0,
                end_char: length,
            },
        }],
    ))
}

pub(super) fn render(
    atoms: &[Atom],
    doc: &Document,
    sentence: usize,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<(String, Vec<ClauseAlignment>)> {
    let [body @ .., punctuation] = atoms else {
        return None;
    };
    if punctuation.is_locked
        || !matches!(punctuation.surface.as_str(), "。" | "！" | "？" | "!" | "?")
    {
        return None;
    }
    let suffix = localize_punct(&punctuation.surface);
    if let Some(text) = payment_method(body, chars, locks) {
        return output(doc, sentence, format!("{text}{suffix}"));
    }
    let text = example(body, chars, locks)?;
    output(doc, sentence, format!("{text}{suffix}"))
}

fn payment_method(atoms: &[Atom], chars: &[char], locks: &SpanLockTable) -> Option<String> {
    let [amount @ .., genitive, payment, method, topic, cash, copula] = atoms else {
        return None;
    };
    let (head, amount_text) = match amount {
        [whole] => (whole, term(whole, chars, locks, "切符代", "票款")?),
        [ticket, fee]
            if !ticket.is_locked
                && !fee.is_locked
                && ticket.surface == "切符"
                && fee.surface == "代"
                && ticket.deprel == "compound"
                && ticket.head == fee.doc_tok_idx
                && ticket.span.end_char == fee.span.start_char =>
        {
            (fee, "票款".to_owned())
        }
        _ => return None,
    };
    let payment_text = term(payment, chars, locks, "支払い", "付款")?;
    let cash_text = term(cash, chars, locks, "現金", "現金")?;
    if genitive.surface != "の"
        || !dasu_case_link(head, genitive)
        || head.deprel != "nmod"
        || head.head != method.doc_tok_idx
        || !matches!(
            payment.deprel.as_str(),
            "acl" | "amod" | "compound" | "nmod"
        )
        || payment.head != method.doc_tok_idx
        || method.is_locked
        || method.surface != "方法"
        || method.upos != Some(Upos::Noun)
        || method.deprel != "nsubj"
        || method.head != cash.doc_tok_idx
        || topic.surface != "は"
        || !dasu_case_link(method, topic)
        || cash.deprel != "root"
        || cash.doc_tok_idx.is_none()
        || copula.is_locked
        || copula.surface != "です"
        || copula.upos != Some(Upos::Aux)
        || copula.deprel != "cop"
        || copula.head != cash.doc_tok_idx
    {
        return None;
    }
    Some(format!("{amount_text}的{payment_text}方式是{cash_text}"))
}

fn example(atoms: &[Atom], chars: &[char], locks: &SpanLockTable) -> Option<String> {
    let [example, as_case, quoted, object_case, show, polite] = atoms else {
        return None;
    };
    if example.is_locked
        || example.surface != "例"
        || example.upos != Some(Upos::Noun)
        || example.deprel != "obl"
        || example.head != show.doc_tok_idx
        || as_case.surface != "として"
        || !dasu_case_link(example, as_case)
        || !quoted.is_locked
        || !matches!(raw(quoted, chars)?, ['`', _, .., '`'])
        || quoted.deprel != "obj"
        || quoted.head != show.doc_tok_idx
        || quoted.doc_tok_idx.is_none()
        || object_case.is_locked
        || object_case.surface != "を"
        || object_case.upos != Some(Upos::Adp)
        || object_case.deprel != "case"
        || object_case.head != quoted.doc_tok_idx
        || !chars
            .get(quoted.span.end_char..object_case.span.start_char)?
            .iter()
            .all(|ch| ch.is_whitespace())
        || show.is_locked
        || show.upos != Some(Upos::Verb)
        || show.lemma.as_deref() != Some("示す")
        || show.deprel != "root"
        || show.doc_tok_idx.is_none()
        || polite.is_locked
        || polite.surface != "ます"
        || polite.upos != Some(Upos::Aux)
        || polite.deprel != "aux"
        || polite.head != show.doc_tok_idx
        || atoms.iter().filter(|atom| atom.is_locked).count() != 1
        || !locks.locks.iter().any(|lock| {
            lock.mandatory
                && lock.source_span == quoted.span
                && lock.placeholder == quoted.surface
                && matches!(lock.origin, LockOrigin::InlineOpaque(OpaqueKind::Code))
        })
    {
        return None;
    }
    Some(format!("作為範例，列出 {}", quoted.surface))
}
