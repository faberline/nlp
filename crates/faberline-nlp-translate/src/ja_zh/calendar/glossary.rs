//! Restore a typed time label and its native case markers before rendering.

use super::{Atom, Upos, expression};
use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::GlossaryKind;
use faberline_nlp::Span;
use faberline_nlp::doc::{Document, Token};

use super::super::{
    build_sentence_atoms, dasu_case_link, daytime, deadline_weekday, is_clock_atom,
    is_daypart_surface, is_named_event_time, normalize_clock_atoms, render_clock_surface,
    render_temporal_made_atom, selects_time_value,
};

const CLOCK: &str = "__ja_protected_event_clock";
const DAYPART: &str = "__ja_protected_event_daypart";
const DATE: &str = "__ja_protected_event_date";
const QUESTION: &str = "__ja_protected_event_time_question";

pub(in crate::ja_zh) fn is_date(atom: &Atom) -> bool {
    matches!(atom.lemma.as_deref(), Some(DATE | QUESTION))
}

pub(in crate::ja_zh) fn is_question(atom: &Atom) -> bool {
    atom.lemma.as_deref() == Some(QUESTION)
}

pub(in crate::ja_zh) fn is_clock_or_daypart(atom: &Atom) -> bool {
    matches!(atom.lemma.as_deref(), Some(CLOCK | DAYPART))
}

pub(in crate::ja_zh) fn is_time(atom: &Atom) -> bool {
    is_date(atom) || is_clock_or_daypart(atom)
}

struct TimeLabel {
    span: Span,
    token: usize,
    governor: usize,
    lemma: &'static str,
    fused_point: Option<Span>,
}

pub(crate) struct TimeRenderLocks {
    pub(crate) table: SpanLockTable,
    labels: Vec<TimeLabel>,
}

impl TimeRenderLocks {
    pub(in crate::ja_zh) fn apply(&self, atoms: &mut Vec<Atom>) {
        for label in &self.labels {
            let Some(index) = atoms
                .iter()
                .position(|a| a.is_locked && a.span == label.span)
            else {
                continue;
            };
            let atom = &mut atoms[index];
            atom.lemma = Some(label.lemma.to_owned());
            atom.upos = Some(Upos::Noun);
            atom.doc_tok_idx = Some(label.token);
            atom.head = Some(label.governor);
            atom.deprel = "obl".to_owned();
            if let Some(span) = label.fused_point {
                atoms.insert(
                    index + 1,
                    Atom {
                        span,
                        surface: "に".to_owned(),
                        lemma: None,
                        upos: Some(Upos::Adp),
                        doc_tok_idx: None,
                        head: Some(label.token),
                        deprel: "case".to_owned(),
                        is_locked: false,
                    },
                );
            }
        }
    }
}

fn spelling(chars: &[char], span: Span) -> String {
    chars[span.start_char..span.end_char].iter().collect()
}

pub(in crate::ja_zh) fn may_cover_time(chars: &[char], locks: &SpanLockTable) -> bool {
    locks.locks.iter().any(|lock| {
        if !matches!(
            lock.origin,
            LockOrigin::Glossary {
                kind: GlossaryKind::DomainTerm,
                ..
            }
        ) {
            return false;
        }
        let raw = spelling(chars, lock.source_span);
        let label = raw
            .strip_suffix("までに")
            .or_else(|| raw.strip_suffix("まで"))
            .or_else(|| raw.strip_suffix('に'))
            .unwrap_or(&raw);
        render_clock_surface(label).is_some()
            || deadline_weekday(label).is_some()
            || is_named_event_time(label)
            || label.ends_with('日')
            || label == "日中"
    })
}

fn label_for_time(
    original: &[Atom],
    last: usize,
    span: Span,
    chars: &[char],
) -> Option<&'static str> {
    let raw = spelling(chars, span);
    if render_clock_surface(&raw).is_some_and(|(_, particle)| !particle) {
        Some(CLOCK)
    } else if raw == "日中" && original[last + 1].surface == "に" {
        daytime::glossary_role(&['日', '中'])
    } else if is_daypart_surface(&raw) || raw == "日中" {
        Some(DAYPART)
    } else if deadline_weekday(&raw).is_some() {
        Some(DATE)
    } else if let Some((start, rendered)) = expression(original, last, false)
        && original[start].span.start_char == span.start_char
    {
        if rendered == "幾號" || rendered == "星期幾" {
            Some(QUESTION)
        } else {
            Some(DATE)
        }
    } else {
        let head = &original[last];
        (span == head.span && render_temporal_made_atom(head).is_some()).then_some(DATE)
    }
}

fn time_label(original: &[Atom], chars: &[char], lock: Span) -> Option<TimeLabel> {
    let first = original
        .iter()
        .position(|a| a.span.start_char == lock.start_char)?;
    for index in first..original.len() {
        let head = &original[index];
        if head.span.end_char > lock.end_char {
            break;
        }
        let Some(particle) = original.get(index + 1) else {
            break;
        };
        if !matches!(head.deprel.as_str(), "obl" | "iobj")
            || !matches!(particle.surface.as_str(), "まで" | "に")
            || !dasu_case_link(head, particle)
            || original[first..index].iter().any(|part| {
                part.head != head.doc_tok_idx
                    || !matches!(part.deprel.as_str(), "nummod" | "compound")
            })
            || original[first..=index]
                .windows(2)
                .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
        {
            continue;
        }
        let until = particle.surface == "まで";
        let deadline = until
            .then(|| original.get(index + 2))
            .flatten()
            .filter(|next| {
                next.surface == "に"
                    && next.upos == Some(Upos::Adp)
                    && next.deprel == "case"
                    && next.head == head.doc_tok_idx
                    && next.span.start_char == particle.span.end_char
            });
        if lock.end_char != head.span.end_char
            && lock.end_char != particle.span.end_char
            && deadline.is_none_or(|a| lock.end_char != a.span.end_char)
        {
            continue;
        }
        let governor = head.head?;
        let action = original
            .iter()
            .find(|a| a.doc_tok_idx == Some(governor) && a.upos == Some(Upos::Verb))?;
        if !until && selects_time_value(action.lemma.as_deref().unwrap_or(&action.surface)) {
            continue;
        }
        let span = Span {
            start_char: lock.start_char,
            end_char: head.span.end_char,
        };
        let lemma = label_for_time(original, index, span, chars)?;
        return Some(TimeLabel {
            span,
            token: head.doc_tok_idx?,
            governor,
            lemma,
            fused_point: None,
        });
    }
    None
}

// The ordinary tokenizer can emit 時に as a single adverb. Confirm the
// clock with the existing normalizer and keep the local point marker.
fn fused_clock_point(original: &[Atom], clocks: &[Atom], lock: Span) -> Option<TimeLabel> {
    clocks.windows(2).find_map(|pair| {
        let [clock, marker] = pair else { return None };
        if !is_clock_atom(clock)
            || clock.span.start_char != lock.start_char
            || marker.surface != "に"
            || clock.span.end_char != marker.span.start_char
            || lock.end_char != marker.span.end_char
        {
            return None;
        }
        let token = marker.doc_tok_idx?;
        let tail = original.iter().find(|a| a.doc_tok_idx == Some(token))?;
        let governor = tail.head?;
        if tail.span.start_char >= marker.span.start_char
            || tail.span.end_char != marker.span.end_char
            || tail.deprel != "advmod"
        {
            return None;
        }
        let action = original
            .iter()
            .find(|a| a.doc_tok_idx == Some(governor) && a.upos == Some(Upos::Verb))?;
        if selects_time_value(action.lemma.as_deref().unwrap_or(&action.surface)) {
            return None;
        }
        Some(TimeLabel {
            span: clock.span,
            token,
            governor,
            lemma: CLOCK,
            fused_point: Some(marker.span),
        })
    })
}

pub(in crate::ja_zh) fn time_render_locks(
    sentence: usize,
    source: &Document,
    tokens: &[&Token],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<TimeRenderLocks> {
    if !may_cover_time(chars, locks) {
        return None;
    }
    let original = build_sentence_atoms(sentence, chars, source, tokens, &SpanLockTable::new());
    let clocks = normalize_clock_atoms(&original);
    let mut table = locks.clone();
    let mut labels = Vec::new();
    for lock in &mut table.locks {
        let LockOrigin::Glossary { kind, .. } = lock.origin else {
            return None;
        };
        if kind == GlossaryKind::DomainTerm
            && let Some(label) = time_label(&original, chars, lock.source_span)
                .or_else(|| fused_clock_point(&original, &clocks, lock.source_span))
        {
            lock.source_span = label.span;
            labels.push(label);
        } else if !tokens.iter().any(|t| t.span == lock.source_span) {
            // A second, unrelated composite lock may hide its own syntax.
            return None;
        }
    }
    (!labels.is_empty()).then_some(TimeRenderLocks { table, labels })
}
