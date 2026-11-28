//! Keep a school teaching relative on the person seen by the matrix subject.

use super::{
    Atom, Document, Span, Token, Upos, build_sentence_atoms, core_non_punct, dasu_case_link,
    translate_single_ja_term,
};
use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::{ClauseAlignment, GlossaryKind};

struct TeachingObject<'a> {
    observer: &'a Atom,
    school: &'a Atom,
    teacher: &'a Atom,
    name: Span,
    address: Span,
    mark: Option<Span>,
}

fn school_name(atom: &Atom) -> Option<&'static str> {
    match atom.surface.as_str() {
        "小学校" => Some("小學"),
        "中学校" => Some("國中"),
        "高校" | "高等学校" => Some("高中"),
        "大学" => Some("大學"),
        _ => None,
    }
}

fn prove(atoms: &[Atom]) -> Option<TeachingObject<'_>> {
    let [
        observer,
        topic,
        school,
        genitive,
        teacher,
        teacher_case,
        work,
        link,
        state,
        rest @ ..,
    ] = atoms
    else {
        return None;
    };
    let verb_pos = rest
        .iter()
        .position(|atom| atom.deprel == "root" && atom.lemma.as_deref() == Some("見る"))?;
    let (matrix_case, identity) = rest[..verb_pos].split_last()?;
    let (see, tense) = rest[verb_pos..].split_first()?;
    let past = match tense {
        [past] => past,
        [polite, past] if polite.surface == "まし" && polite.head == see.doc_tok_idx => past,
        _ => return None,
    };

    let person = identity.last()?;
    let first = identity.first()?;
    let suffix = (person.surface == "さん").then_some(person);
    let names = if suffix.is_some() {
        &identity[..identity.len() - 1]
    } else {
        identity
    };
    let last_name = names.last()?;
    if school.upos != Some(Upos::Noun)
        || school_name(school).is_none()
        || genitive.surface != "の"
        || school.deprel != "nmod"
        || school.head != teacher.doc_tok_idx
        || !dasu_case_link(school, genitive)
        || !matches!(teacher.surface.as_str(), "先生" | "教師" | "教員")
        || teacher.upos != Some(Upos::Noun)
        || teacher.deprel != "obj"
        || teacher.head != work.doc_tok_idx
        || teacher_case.surface != "を"
        || !dasu_case_link(teacher, teacher_case)
        || work.lemma.as_deref() != Some("する")
        || work.deprel != "advcl"
        || work.head != state.doc_tok_idx
        || link.surface != "て"
        || link.head != work.doc_tok_idx
        || state.surface != "いる"
        || state.deprel != "acl"
        || state.head != person.doc_tok_idx
        || names.iter().any(|atom| {
            atom.upos != Some(Upos::Propn)
                || (atom.doc_tok_idx != person.doc_tok_idx
                    && (atom.deprel != "compound" || atom.head != person.doc_tok_idx))
        })
        || person.deprel != "obj"
        || person.head != see.doc_tok_idx
        || matrix_case.surface != "を"
        || !dasu_case_link(person, matrix_case)
        || observer.upos != Some(Upos::Pron)
        || !dasu_case_link(observer, topic)
        || observer.deprel != "nsubj"
        || !matches!(topic.surface.as_str(), "は" | "が")
        // The parser can attach a matrix topic to the earlier light verb.
        // A が-subject headed by that verb still belongs to the relative.
        || (observer.head != see.doc_tok_idx
            && !(topic.surface == "は" && observer.head == work.doc_tok_idx))
        || past.surface != "た"
        || past.head != see.doc_tok_idx
        || [observer, school, teacher, work, state, person, see]
            .iter().any(|atom| atom.doc_tok_idx.is_none())
    {
        return None;
    }
    Some(TeachingObject {
        observer,
        school,
        teacher,
        name: Span {
            start_char: first.span.start_char,
            end_char: last_name.span.end_char,
        },
        address: Span {
            start_char: first.span.start_char,
            end_char: person.span.end_char,
        },
        mark: suffix.map(|atom| atom.span),
    })
}

fn allowed_lock(lock: &LockedSpan, span: Span, kinds: &[GlossaryKind]) -> bool {
    lock.source_span == span
        && matches!(&lock.origin, LockOrigin::Glossary { kind, .. } if kinds.contains(kind))
}

fn render_object(
    object: TeachingObject<'_>,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<String> {
    let mut observer = None;
    let mut school = None;
    let mut teacher = None;
    let mut name = None;
    let mut address = None;
    let mut mark = None;
    let terms = &[GlossaryKind::Object, GlossaryKind::DomainTerm];
    let people = &[
        GlossaryKind::Actor,
        GlossaryKind::Object,
        GlossaryKind::DomainTerm,
    ];
    for lock in &locks.locks {
        if allowed_lock(lock, object.observer.span, people) {
            observer = Some(lock.placeholder.as_str());
        } else if allowed_lock(lock, object.school.span, terms) {
            school = Some(lock.placeholder.as_str());
        } else if allowed_lock(lock, object.teacher.span, terms) {
            teacher = Some(lock.placeholder.as_str());
        } else if object.address != object.name && allowed_lock(lock, object.address, people) {
            address = Some(lock);
        } else if allowed_lock(lock, object.name, people) {
            name = Some(lock);
        } else if object
            .mark
            .is_some_and(|span| allowed_lock(lock, span, people))
        {
            mark = Some(lock);
        } else {
            return None;
        }
    }

    let observed_name: String = chars[object.name.start_char..object.name.end_char]
        .iter()
        .collect();
    let mut person = match address.or(name) {
        Some(lock) => lock.placeholder.clone(),
        None => observed_name,
    };
    if let Some(lock) = mark {
        person.push_str(&lock.placeholder);
    } else if address.is_none() && name.is_none() && teacher.is_none() {
        // A caller's chosen name may already include its own form of address.
        person.push_str("老師");
    }
    let observer = observer.map(str::to_owned).unwrap_or_else(|| {
        translate_single_ja_term(
            &object.observer.surface,
            object.observer.lemma.as_deref(),
            object.observer.upos,
        )
    });
    let school = school.or_else(|| school_name(object.school))?;
    let relative = if let Some(role) = teacher {
        format!("在{school}擔任{role}")
    } else {
        format!("在{school}教書")
    };
    Some(format!("{observer}看見{relative}的{person}了"))
}

fn render_document(
    chars: &[char],
    sentence: usize,
    doc: &Document,
    tokens: &[&Token],
    locks: &SpanLockTable,
) -> Option<String> {
    let source = build_sentence_atoms(sentence, chars, doc, tokens, &SpanLockTable::default());
    let core = core_non_punct(&source);
    let [terminal] = &source[core.len()..] else {
        return None;
    };
    let punctuation = match terminal.surface.as_str() {
        "。" => "。",
        "!" | "！" => "！",
        // Questions and other unfinished sentence forms keep the general render path.
        _ => return None,
    };
    let object = prove(core)?;
    render_object(object, chars, locks).map(|text| text + punctuation)
}

pub(super) fn render<'source>(
    chars: &[char],
    sentence: usize,
    doc: &Document,
    tokens: &[&Token],
    locks: &SpanLockTable,
    without_glossary: &dyn Fn() -> Option<&'source Document>,
) -> Option<(String, Vec<ClauseAlignment>)> {
    let span = doc.segmentation.sentences.get(sentence)?.span;
    if !chars
        .get(span.start_char..span.end_char)?
        .windows(5)
        .any(|part| part == ['を', 'し', 'て', 'い', 'る'])
    {
        return None;
    }
    let mut clauses = doc
        .segmentation
        .clauses
        .iter()
        .enumerate()
        .filter(|(_, clause)| clause.sentence == sentence);
    let (clause, source) = clauses.next()?;
    if source.span != span || clauses.next().is_some() {
        return None;
    }
    let text = render_document(chars, sentence, doc, tokens, locks).or_else(|| {
        // A caller term can absorb part of the person's name. For the built-in
        // parser, recover the grammatical evidence without glossary hints while
        // still emitting the original lock and its exact offsets.
        if locks.locks.is_empty() {
            return None;
        }
        let source = without_glossary()?;
        let first = source
            .tokens
            .partition_point(|token| token.sentence < sentence);
        let end = source
            .tokens
            .partition_point(|token| token.sentence <= sentence);
        let tokens = source.tokens[first..end].iter().collect::<Vec<_>>();
        render_document(chars, sentence, source, &tokens, locks)
    })?;
    let target_span = Span {
        start_char: 0,
        end_char: text.chars().count(),
    };
    Some((
        text,
        vec![ClauseAlignment {
            clause,
            sentence,
            source_span: span,
            target_span,
        }],
    ))
}
