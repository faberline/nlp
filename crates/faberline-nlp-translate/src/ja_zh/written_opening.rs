//! Keep the opening of a text attached to its written source.

use crate::lock::{LockedSpan, SpanLockTable};

use super::{
    Atom, ClauseDiscourseCtx, Document, Span, Token, Upos, build_sentence_atoms, core_non_punct,
    dasu_case_link, is_object_or_domain_lock,
};

fn is_text(word: &str) -> bool {
    matches!(
        word,
        "本" | "書籍" | "小説" | "物語" | "文章" | "段落" | "章" | "手紙"
    )
}

fn is_opening(word: &str) -> bool {
    matches!(word, "始め" | "初め")
}

fn source_word(chars: &[char], span: Span) -> Option<String> {
    Some(chars.get(span.start_char..span.end_char)?.iter().collect())
}

fn term_at<'a>(atom: &Atom, locks: &'a [LockedSpan]) -> Option<&'a LockedSpan> {
    locks.iter().find(|lock| {
        is_object_or_domain_lock(lock)
            && lock.source_span == atom.span
            && lock.placeholder == atom.surface
    })
}

fn nominal(atom: &Atom, spelling: fn(&str) -> bool, chars: &[char], locks: &[LockedSpan]) -> bool {
    if atom.is_locked {
        term_at(atom, locks).is_some()
            && source_word(chars, atom.span).is_some_and(|word| spelling(&word))
    } else {
        atom.upos == Some(Upos::Noun) && spelling(&atom.surface)
    }
}

fn linked_opening(
    text: &Atom,
    genitive: &Atom,
    opening: &Atom,
    chars: &[char],
    locks: &[LockedSpan],
) -> bool {
    opening.doc_tok_idx.is_some()
        && text.deprel == "nmod"
        && text.head == opening.doc_tok_idx
        && genitive.surface == "の"
        && dasu_case_link(text, genitive)
        && genitive.span.end_char == opening.span.start_char
        && nominal(text, is_text, chars, locks)
        && nominal(opening, is_opening, chars, locks)
}

fn resources<'a>(ctx: Option<&'a ClauseDiscourseCtx<'_>>) -> (&'a [char], &'a [LockedSpan]) {
    ctx.map(|ctx| {
        (
            ctx.source_chars,
            ctx.source_locks
                .map_or(&[][..], |table| table.locks.as_slice()),
        )
    })
    .unwrap_or_default()
}

/// Render the nominal sense and indicate whether the modifier already includes 的.
pub(super) fn localize(
    opening: &mut Atom,
    modifier: &[Atom],
    ctx: Option<&ClauseDiscourseCtx<'_>>,
) -> bool {
    if ctx.is_some_and(|context| context.inside_quote) {
        return false;
    }
    let [.., text, genitive] = modifier else {
        return false;
    };
    let (chars, locks) = resources(ctx);
    if !linked_opening(text, genitive, opening, chars, locks) {
        return false;
    }
    if !opening.is_locked {
        opening.surface = "開頭".to_owned();
        opening.lemma = None;
    }
    text.is_locked
        && term_at(text, locks).is_some_and(|lock| lock.replacement.trim_end().ends_with('的'))
}

pub(super) fn localize_importance(atoms: &mut [Atom], ctx: &ClauseDiscourseCtx<'_>) {
    let end = core_non_punct(atoms).len();
    let [text, genitive, opening, topic, adjective, polite] = &mut atoms[..end] else {
        return;
    };
    let (chars, locks) = resources(Some(ctx));
    if linked_opening(text, genitive, opening, chars, locks)
        && opening.deprel == "nsubj"
        && opening.head == adjective.doc_tok_idx
        && topic.surface == "は"
        && dasu_case_link(opening, topic)
        && !adjective.is_locked
        && adjective.surface == "大切"
        && adjective.upos == Some(Upos::Adj)
        && adjective.doc_tok_idx.is_some()
        && adjective.deprel == "root"
        && !polite.is_locked
        && polite.surface == "です"
        && polite.deprel == "cop"
        && polite.head == adjective.doc_tok_idx
    {
        adjective.surface = "很重要".to_owned();
        adjective.lemma = None;
    }
}

pub(crate) fn may_restore(text: &str, chars: &[char], locks: &SpanLockTable) -> bool {
    (text.contains("の始め") || text.contains("の初め"))
        && locks.locks.iter().any(|lock| {
            is_object_or_domain_lock(lock)
                && source_word(chars, lock.source_span).is_some_and(|word| {
                    is_opening(&word) || is_text(word.strip_suffix('の').unwrap_or(&word))
                })
        })
}

pub(crate) fn restore(
    sentence: usize,
    doc: &Document,
    tokens: &[&Token],
    chars: &[char],
    render_locks: &mut SpanLockTable,
) -> bool {
    let span = doc.segmentation.sentences[sentence].span;
    if doc
        .segmentation
        .opaque
        .iter()
        .any(|part| part.span.start_char < span.end_char && span.start_char < part.span.end_char)
    {
        return false;
    }
    let atoms = build_sentence_atoms(sentence, chars, doc, tokens, &SpanLockTable::default());
    let mut restored = false;
    for window in atoms.windows(3) {
        let [text, genitive, opening] = window else {
            continue;
        };
        if !linked_opening(text, genitive, opening, &[], &[]) {
            continue;
        }
        for lock in render_locks
            .locks
            .iter_mut()
            .filter(|lock| is_object_or_domain_lock(lock))
        {
            if lock.source_span.start_char == text.span.start_char
                && [text.span.end_char, genitive.span.end_char].contains(&lock.source_span.end_char)
            {
                // Expose の for rendering. Restoration still uses the complete caller hit.
                lock.source_span = text.span;
                restored = true;
            } else if lock.source_span == opening.span {
                restored = true;
            }
        }
    }
    restored
}
