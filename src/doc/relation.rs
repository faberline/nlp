//! Layer 7 of `nlp.doc.v1`: discourse relations from connectors at the start of a clause. Only
//! layer 1 is read. See `docs/doc-v1.md`.

use super::segment;
use super::{Reason, Relation, RelationKind, RelationLevel, Segmentation, Unknown};
use crate::Span;
use RelationKind as K;
use RelationLevel::{Clause as Cl, Sentence as Se};
use Role::{Ambiguous, Back, Open};

#[derive(Clone, Copy)]
enum Role {
    /// The connector sits in arg2; arg1 is the unit before it.
    Back(K),
    /// The connector sits in arg1; the next clause may start with one of the results.
    Open(K, &'static [&'static str]),
    /// Several readings: an unknown, no relation.
    Ambiguous,
}

const CAUSE_RESULT: &[&str] = &["所以", "因此", "so", "therefore", "thus"];
const COND_RESULT: &[&str] = &["就", "那麼", "便", "then"];
const SEQ_RESULT: &[&str] = &["然後", "接著", "再", "隨後"];
const PAR_RESULT: &[&str] = &["另一方面"];
const SO_NOT: &[&str] = &[
    "far", "much", "many", "long", "as", "that", "good", "bad", "little", "few", "called", "on",
];
const OPENING: &[char] = &['「', '『', '“', '‘', '（', '(', '[', '"', '\''];
const CLOSING: &[char] = &[
    '。', '！', '？', '.', '!', '?', '」', '』', '”', '’', '）', ')', '"',
];

const ENTRIES: &[(&str, Role)] = &[
    ("然後", Back(K::Sequence)),
    ("接著", Back(K::Sequence)),
    ("隨後", Back(K::Sequence)),
    ("then", Back(K::Sequence)),
    ("afterwards", Back(K::Sequence)),
    ("after that", Back(K::Sequence)),
    ("所以", Back(K::Cause)),
    ("因此", Back(K::Cause)),
    ("so", Back(K::Cause)),
    ("therefore", Back(K::Cause)),
    ("thus", Back(K::Cause)),
    ("但是", Back(K::Contrast)),
    ("然而", Back(K::Contrast)),
    ("可是", Back(K::Contrast)),
    ("但", Back(K::Contrast)),
    ("however", Back(K::Contrast)),
    ("but", Back(K::Contrast)),
    ("另一方面", Back(K::Parallel)),
    ("另外", Back(K::Parallel)),
    ("此外", Back(K::Parallel)),
    ("而且", Back(K::Parallel)),
    ("並且", Back(K::Parallel)),
    ("同樣", Back(K::Parallel)),
    ("also", Back(K::Parallel)),
    ("moreover", Back(K::Parallel)),
    ("furthermore", Back(K::Parallel)),
    ("likewise", Back(K::Parallel)),
    ("similarly", Back(K::Parallel)),
    ("in addition", Back(K::Parallel)),
    ("因為", Open(K::Cause, CAUSE_RESULT)),
    ("because", Open(K::Cause, CAUSE_RESULT)),
    ("如果", Open(K::Condition, COND_RESULT)),
    ("假如", Open(K::Condition, COND_RESULT)),
    ("要是", Open(K::Condition, COND_RESULT)),
    ("若", Open(K::Condition, COND_RESULT)),
    ("if", Open(K::Condition, COND_RESULT)),
    ("先", Open(K::Sequence, SEQ_RESULT)),
    ("一方面", Open(K::Parallel, PAR_RESULT)),
    ("同時", Ambiguous),
    ("while", Ambiguous),
    ("since", Ambiguous),
];

/// Relations and the `relations` unknowns (connector spans), before the caller sorts them.
pub(super) fn relations(c: &[char], seg: &Segmentation) -> (Vec<Relation>, Vec<Unknown>) {
    let unsupported: Vec<bool> = seg
        .sentences
        .iter()
        .map(|s| segment::lang(c, s.span).1)
        .collect();
    let mut units = vec![Vec::new(); seg.sentences.len()];
    for (i, clause) in seg.clauses.iter().enumerate() {
        units[clause.sentence].push(i);
    }
    let mut pass = Pass {
        c,
        seg,
        rels: Vec::new(),
        unknown: Vec::new(),
        used: vec![false; seg.clauses.len()],
    };
    for s in 0..seg.sentences.len() {
        if unsupported[s] {
            continue;
        }
        let same_paragraph = s > 0
            && !unsupported[s - 1]
            && seg.sentences[s - 1].paragraph == seg.sentences[s].paragraph;
        pass.sentence(s, same_paragraph.then(|| s - 1), &units[s]);
    }
    pass.rels.sort_by_key(|r| r.connector.start_char);
    (pass.rels, pass.unknown)
}

struct Pass<'a> {
    c: &'a [char],
    seg: &'a Segmentation,
    rels: Vec<Relation>,
    unknown: Vec<Unknown>,
    /// Clauses consumed as the correlate of an opener.
    used: Vec<bool>,
}

impl Pass<'_> {
    fn clause(&self, i: usize) -> Span {
        self.seg.clauses[i].span
    }

    fn unknown(&mut self, span: Span) {
        self.unknown.push(Unknown {
            layer: "relations".to_owned(),
            span,
            reason: Reason::Ambiguous,
        });
    }

    fn emit(
        &mut self,
        kind: K,
        level: RelationLevel,
        connector: Span,
        correlate: Option<Span>,
        arg1: Span,
        arg2: Span,
    ) {
        self.rels.push(Relation {
            kind,
            level,
            connector,
            correlate,
            arg1,
            arg2,
        });
    }

    /// Sentence `s` with clauses `ids`; `prev` is the previous sentence in the same paragraph.
    fn sentence(&mut self, s: usize, prev: Option<usize>, ids: &[usize]) {
        let sentence = self.seg.sentences[s].span;
        for k in 0..ids.len() {
            let ci = ids[k];
            if self.used[ci] {
                continue;
            }
            let clause = self.clause(ci);
            let Some((role, conn)) = connector(self.c, clause, sentence.end_char) else {
                continue;
            };
            let next = ids.get(k + 1).copied();
            let bare = is_bare(self.c, conn.end_char, clause.end_char);
            match role {
                Ambiguous => self.unknown(conn),
                // `then` after an `if` clause is the correlate of that `if`, never a sequence.
                Back(_)
                    if k > 0
                        && is_then(self.c, conn)
                        && opens_if(self.c, self.clause(ids[k - 1]), sentence.end_char) => {}
                Back(kind) if k == 0 => match prev {
                    Some(p) => {
                        let a = self.seg.sentences[p].span;
                        self.emit(kind, Se, conn, None, a, sentence);
                    }
                    None => self.unknown(conn),
                },
                Back(kind) if !bare => {
                    let a = self.clause(ids[k - 1]);
                    self.emit(kind, Cl, conn, None, a, clause);
                }
                Back(kind) => match next {
                    Some(n) => {
                        let (a, b) = (self.clause(ids[k - 1]), self.clause(n));
                        self.emit(kind, Cl, conn, None, a, b);
                    }
                    None => self.unknown(conn),
                },
                Open(kind, results) => {
                    if kind == K::Cause && (k > 0 || next.is_none()) {
                        if k > 0 {
                            let b = self.clause(ids[k - 1]);
                            self.emit(kind, Cl, conn, None, clause, b);
                        } else {
                            match prev {
                                Some(p) => {
                                    let b = self.seg.sentences[p].span;
                                    self.emit(kind, Se, conn, None, sentence, b);
                                }
                                None => self.unknown(conn),
                            }
                        }
                        continue;
                    }
                    // Sequence and parallel openers stay silent unless their correlate follows.
                    let silent = matches!(kind, K::Sequence | K::Parallel);
                    let Some(n) = next.filter(|_| !bare) else {
                        if !silent {
                            self.unknown(conn);
                        }
                        continue;
                    };
                    let n_span = self.clause(n);
                    match correlate(self.c, n_span, sentence.end_char, results) {
                        Some(cor) => {
                            self.emit(kind, Cl, conn, Some(cor), clause, n_span);
                            self.used[n] = true;
                        }
                        None if silent => {}
                        None if connector(self.c, n_span, sentence.end_char).is_some() => {
                            self.unknown(conn);
                        }
                        None => self.emit(kind, Cl, conn, None, clause, n_span),
                    }
                }
            }
        }
    }
}

/// The first connector of a clause, with its role. Special cases are applied here.
fn connector(c: &[char], clause: Span, sentence_end: usize) -> Option<(Role, Span)> {
    let from = skip_opening(c, clause);
    let (text, span) = longest(c, from, clause.end_char, ENTRIES.iter().map(|e| e.0))?;
    let role = ENTRIES.iter().find(|e| e.0 == text)?.1;
    let end = span.end_char;
    let next = next_nonspace(c, end, sentence_end);
    let word = next_word(c, end, sentence_end);
    let at = |i: usize, set: &str| c.get(i).is_some_and(|ch| set.contains(*ch));
    let role = match text {
        "但" if at(end, "願凡求見") => return None,
        "若" if at(end, "干無非是即") => return None,
        "先" if at(end, "生前後") => return None,
        "if" if word == "only" => return None,
        "however" if next != Some(',') => return None,
        "so" if SO_NOT.contains(&word.as_str()) => return None,
        "so" if next == Some(',') => Ambiguous,
        _ => role,
    };
    Some((role, span))
}

/// A correlate that starts the clause `clause`, if any. `so` counts only where it is a cause-result
/// connector (not `so far`, not `So,`).
fn correlate(
    c: &[char],
    clause: Span,
    sentence_end: usize,
    texts: &'static [&'static str],
) -> Option<Span> {
    let from = skip_opening(c, clause);
    let (text, span) = longest(c, from, clause.end_char, texts.iter().copied())?;
    let ok = text != "so"
        || matches!(
            connector(c, clause, sentence_end),
            Some((Back(K::Cause), _))
        );
    ok.then_some(span)
}

/// Whether the connector span reads `then` (ASCII, case-insensitive).
fn is_then(c: &[char], conn: Span) -> bool {
    c[conn.start_char..conn.end_char]
        .iter()
        .collect::<String>()
        .eq_ignore_ascii_case("then")
}

/// Whether the clause starts with an `if` opener.
fn opens_if(c: &[char], clause: Span, sentence_end: usize) -> bool {
    matches!(
        connector(c, clause, sentence_end),
        Some((Open(K::Condition, _), _))
    )
}

/// The longest text that matches at `from` ending by `limit`. ASCII letters match case-insensitively,
/// and an English text needs a word boundary after it.
fn longest(
    c: &[char],
    from: usize,
    limit: usize,
    texts: impl Iterator<Item = &'static str>,
) -> Option<(&'static str, Span)> {
    let mut best: Option<(&'static str, usize)> = None;
    for text in texts {
        let end = from + text.chars().count();
        if end > limit
            || !text
                .chars()
                .zip(&c[from..end])
                .all(|(a, b)| a.eq_ignore_ascii_case(b))
        {
            continue;
        }
        let boundary = text.is_ascii()
            && c.get(end)
                .is_some_and(|&ch| ch.is_ascii_alphanumeric() || matches!(ch, '\'' | '’' | '-'));
        if !boundary && best.is_none_or(|(_, e)| end > e) {
            best = Some((text, end));
        }
    }
    best.map(|(text, end)| {
        (
            text,
            Span {
                start_char: from,
                end_char: end,
            },
        )
    })
}

fn skip_opening(c: &[char], clause: Span) -> usize {
    (clause.start_char..clause.end_char)
        .find(|&i| !(c[i].is_whitespace() || OPENING.contains(&c[i])))
        .unwrap_or(clause.end_char)
}

/// The connector ends the clause, or only closers and terminators follow it inside the clause.
fn is_bare(c: &[char], from: usize, end: usize) -> bool {
    c[from..end]
        .iter()
        .all(|ch| ch.is_whitespace() || CLOSING.contains(ch))
}

fn next_nonspace(c: &[char], from: usize, end: usize) -> Option<char> {
    c[from..end].iter().copied().find(|ch| !ch.is_whitespace())
}

/// The ASCII word after the next non-space char, lowercased.
fn next_word(c: &[char], from: usize, end: usize) -> String {
    let start = (from..end).find(|&i| !c[i].is_whitespace()).unwrap_or(end);
    c[start..end]
        .iter()
        .take_while(|ch| ch.is_ascii_alphabetic())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}
