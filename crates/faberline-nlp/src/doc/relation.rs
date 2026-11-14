//! Layer 7 of `nlp.doc.v1`: discourse relations from connectors at the start of a clause. Only
//! layer 1 is read. See `docs/doc-v1.md`.

use super::segment;
use super::{
    Lang, Opaque, OpaqueKind, Reason, Relation, RelationKind, RelationLevel, Segmentation, Token,
    Unknown,
};
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
const COND_RESULT: &[&str] = &["就", "那麼", "那么", "便", "then"];
const SEQ_RESULT: &[&str] = &[
    "然後",
    "然后",
    "接著",
    "接着",
    "再",
    "隨後",
    "随后",
    "隨即",
    "随即",
    "緊接著",
    "紧接着",
    "緊接着",
    "紧接著",
    "旋即",
];
const PAR_RESULT: &[&str] = &["另一方面"];
const POST_SEQ_FRAMES: &[&str] = &["之後", "之后", "之際", "之际", "之時", "之时"];
const KINSHIP_AFTER_STEMS: &[&str] = &["故人", "高人", "仙人", "名門", "名门", "將門", "将门"];
const SPATIAL_AFTER_VERBS: &[&str] = &[
    "出現在",
    "出现在",
    "跟在",
    "落在",
    "站在",
    "排在",
    "躲在",
    "藏在",
    "懸在",
    "悬在",
    "立在",
    "立於",
    "立于",
    "位於",
    "位于",
];
const TEMPORAL_TAIL_WORDS: &[&str] = &[
    "片刻",
    "半晌",
    "良久",
    "許久",
    "许久",
    "頃刻",
    "顷刻",
    "一會兒",
    "一会儿",
    "時辰",
    "时辰",
    "工夫",
    "功夫",
    "分鐘",
    "分钟",
    "小時",
    "小时",
    "數息",
    "数息",
    "幾息",
    "几息",
];
const SO_NOT: &[&str] = &[
    "far", "much", "many", "long", "as", "that", "good", "bad", "little", "few", "called", "on",
];
/// Chars that, right after a correlate, show it is part of a longer word and not a correlate:
/// `便宜`, `再見`, `就近`. A stop list of following chars, not a longest-word match.
const CORRELATE_STOP: &[(&str, &str)] = &[
    ("便", "宜利當條於民秘"),
    ("再", "見次度三也說加"),
    ("就", "近業讀地"),
];
const OPENING: &[char] = &['「', '『', '“', '‘', '（', '(', '[', '"', '\''];
const CLOSING: &[char] = &[
    '。', '！', '？', '.', '!', '?', '」', '』', '”', '’', '）', ')', '"',
];

const ENTRIES: &[(&str, Role)] = &[
    ("然後", Back(K::Sequence)),
    ("然后", Back(K::Sequence)),
    ("接著", Back(K::Sequence)),
    ("接着", Back(K::Sequence)),
    ("隨後", Back(K::Sequence)),
    ("随后", Back(K::Sequence)),
    ("隨即", Back(K::Sequence)),
    ("随即", Back(K::Sequence)),
    ("緊接著", Back(K::Sequence)),
    ("紧接着", Back(K::Sequence)),
    ("緊接着", Back(K::Sequence)),
    ("紧接著", Back(K::Sequence)),
    ("旋即", Back(K::Sequence)),
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
    ("不過", Back(K::Contrast)),
    ("不过", Back(K::Contrast)),
    ("但", Back(K::Contrast)),
    ("however", Back(K::Contrast)),
    ("but", Back(K::Contrast)),
    ("否則", Back(K::Condition)),
    ("否则", Back(K::Condition)),
    ("另一方面", Back(K::Parallel)),
    ("與此同時", Back(K::Parallel)),
    ("与此同时", Back(K::Parallel)),
    ("与此同時", Back(K::Parallel)),
    ("與此同时", Back(K::Parallel)),
    ("而與此同時", Back(K::Parallel)),
    ("而与此同时", Back(K::Parallel)),
    ("而与此同時", Back(K::Parallel)),
    ("而與此同时", Back(K::Parallel)),
    ("幾乎與此同時", Back(K::Parallel)),
    ("几乎与此同时", Back(K::Parallel)),
    ("幾乎与此同時", Back(K::Parallel)),
    ("几乎與此同時", Back(K::Parallel)),
    ("就在與此同時", Back(K::Parallel)),
    ("就在与此同时", Back(K::Parallel)),
    ("就在与此同時", Back(K::Parallel)),
    ("就在與此同时", Back(K::Parallel)),
    ("另外", Back(K::Parallel)),
    ("此外", Back(K::Parallel)),
    ("而且", Back(K::Parallel)),
    ("並且", Back(K::Parallel)),
    ("并且", Back(K::Parallel)),
    ("同樣", Back(K::Parallel)),
    ("also", Back(K::Parallel)),
    ("moreover", Back(K::Parallel)),
    ("furthermore", Back(K::Parallel)),
    ("likewise", Back(K::Parallel)),
    ("similarly", Back(K::Parallel)),
    ("in addition", Back(K::Parallel)),
    ("因為", Open(K::Cause, CAUSE_RESULT)),
    ("因为", Open(K::Cause, CAUSE_RESULT)),
    ("because", Open(K::Cause, CAUSE_RESULT)),
    ("如果", Open(K::Condition, COND_RESULT)),
    ("假如", Open(K::Condition, COND_RESULT)),
    ("要是", Open(K::Condition, COND_RESULT)),
    ("若是", Open(K::Condition, COND_RESULT)),
    ("若", Open(K::Condition, COND_RESULT)),
    ("if", Open(K::Condition, COND_RESULT)),
    ("先", Open(K::Sequence, SEQ_RESULT)),
    ("一方面", Open(K::Parallel, PAR_RESULT)),
    ("同時", Ambiguous),
    ("同时", Ambiguous),
    ("while", Ambiguous),
    ("since", Ambiguous),
    // Japanese
    ("それにもかかわらず", Back(K::Contrast)),
    ("にもかかわらず", Back(K::Contrast)),
    ("しかしながら", Back(K::Contrast)),
    ("けれども", Back(K::Contrast)),
    ("ところが", Back(K::Contrast)),
    ("しかし", Back(K::Contrast)),
    ("けれど", Back(K::Contrast)),
    ("一方で", Back(K::Contrast)),
    ("だが", Back(K::Contrast)),
    ("とはいえ", Back(K::Contrast)),
    ("したがって", Back(K::Cause)),
    ("そのため", Back(K::Cause)),
    ("ゆえに", Back(K::Cause)),
    ("だから", Back(K::Cause)),
    ("よって", Back(K::Cause)),
    ("なぜなら", Open(K::Cause, CAUSE_RESULT)),
    ("というのも", Open(K::Cause, CAUSE_RESULT)),
    ("万が一", Open(K::Condition, COND_RESULT)),
    ("もしも", Open(K::Condition, COND_RESULT)),
    ("もし", Open(K::Condition, COND_RESULT)),
    ("仮に", Open(K::Condition, COND_RESULT)),
    ("それから", Back(K::Sequence)),
    ("その後", Back(K::Sequence)),
    ("次いで", Back(K::Sequence)),
    ("続いて", Back(K::Sequence)),
    ("やがて", Back(K::Sequence)),
    ("次に", Back(K::Sequence)),
    ("ならびに", Back(K::Parallel)),
    ("さらに", Back(K::Parallel)),
    ("しかも", Back(K::Parallel)),
    ("加えて", Back(K::Parallel)),
    ("その上", Back(K::Parallel)),
    ("また", Back(K::Parallel)),
    // Korean
    ("그럼에도 불구하고", Back(K::Contrast)),
    ("그렇지만", Back(K::Contrast)),
    ("그러나", Back(K::Contrast)),
    ("하지만", Back(K::Contrast)),
    ("반면에", Back(K::Contrast)),
    ("그럼에도", Back(K::Contrast)),
    ("그러므로", Back(K::Cause)),
    ("따라서", Back(K::Cause)),
    ("그래서", Back(K::Cause)),
    ("그 결과", Back(K::Cause)),
    ("이로 인해", Back(K::Cause)),
    ("왜냐하면", Open(K::Cause, CAUSE_RESULT)),
    ("만약에", Open(K::Condition, COND_RESULT)),
    ("만일에", Open(K::Condition, COND_RESULT)),
    ("만약", Open(K::Condition, COND_RESULT)),
    ("만일", Open(K::Condition, COND_RESULT)),
    ("가령", Open(K::Condition, COND_RESULT)),
    ("그러고 나서", Back(K::Sequence)),
    ("다음으로", Back(K::Sequence)),
    ("이어서", Back(K::Sequence)),
    ("그 후", Back(K::Sequence)),
    ("그 뒤", Back(K::Sequence)),
    ("뿐만 아니라", Back(K::Parallel)),
    ("게다가", Back(K::Parallel)),
    ("더구나", Back(K::Parallel)),
    ("아울러", Back(K::Parallel)),
    ("또한", Back(K::Parallel)),
];

/// Relations and the `relations` unknowns (connector spans), before the caller sorts them.
pub(super) fn relations(
    c: &[char],
    seg: &Segmentation,
    tokens: &[Token],
) -> (Vec<Relation>, Vec<Unknown>) {
    let unsupported: Vec<bool> = seg
        .sentences
        .iter()
        .map(|s| segment::lang(c, s.span).1)
        .collect();
    let mut units = vec![Vec::new(); seg.sentences.len()];
    for (i, clause) in seg.clauses.iter().enumerate() {
        units[clause.sentence].push(i);
    }
    let mut sent_tok_ranges = vec![0..0; seg.sentences.len()];
    let mut tok_cursor = 0usize;
    for (s, range) in sent_tok_ranges.iter_mut().enumerate() {
        while tok_cursor < tokens.len() && tokens[tok_cursor].sentence < s {
            tok_cursor += 1;
        }
        let start = tok_cursor;
        while tok_cursor < tokens.len() && tokens[tok_cursor].sentence == s {
            tok_cursor += 1;
        }
        *range = start..tok_cursor;
    }
    let mut pass = Pass {
        c,
        seg,
        tokens,
        sent_tok_ranges,
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
    tokens: &'a [Token],
    sent_tok_ranges: Vec<std::ops::Range<usize>>,
    rels: Vec<Relation>,
    unknown: Vec<Unknown>,
    /// Clauses consumed as the correlate of an opener.
    used: Vec<bool>,
}

impl Pass<'_> {
    fn clause(&self, i: usize) -> Span {
        self.seg.clauses[i].span
    }

    fn sent_tokens(&self, s: usize) -> &[Token] {
        &self.tokens[self.sent_tok_ranges[s].clone()]
    }

    /// Whether the last clause of sentence `s` opens with an `if` opener.
    fn last_opens_if(&self, s: usize) -> bool {
        let toks = self.sent_tokens(s);
        let sent_lang = self.seg.sentences[s].lang;
        self.seg
            .clauses
            .iter()
            .rposition(|c| c.sentence == s)
            .is_some_and(|i| {
                let clause = self.clause(i);
                let lang =
                    effective_clause_lang(self.c, sent_lang, clause, self.seg.clauses[i].lang);
                opens_if(
                    self.c,
                    &self.seg.opaque,
                    clause,
                    self.seg.sentences[s].span.end_char,
                    toks,
                    lang,
                )
            })
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
        let sent_lang = self.seg.sentences[s].lang;
        let toks = &self.tokens[self.sent_tok_ranges[s].clone()];
        for k in 0..ids.len() {
            let ci = ids[k];
            if self.used[ci] {
                continue;
            }
            let clause = self.clause(ci);
            let clause_lang =
                effective_clause_lang(self.c, sent_lang, clause, self.seg.clauses[ci].lang);
            let next = ids.get(k + 1).copied();
            let initial = connector(
                self.c,
                &self.seg.opaque,
                clause,
                sentence.end_char,
                toks,
                clause_lang,
            );
            if let Some((role, conn)) = initial {
                let bare = is_bare(self.c, conn.end_char, clause.end_char);
                match role {
                    Ambiguous => self.unknown(conn),
                    // `then` after an `if` clause is the correlate of that `if`, never a sequence.
                    Back(_)
                        if k > 0
                            && is_then(self.c, conn)
                            && opens_if(
                                self.c,
                                &self.seg.opaque,
                                self.clause(ids[k - 1]),
                                sentence.end_char,
                                toks,
                                effective_clause_lang(
                                    self.c,
                                    sent_lang,
                                    self.clause(ids[k - 1]),
                                    self.seg.clauses[ids[k - 1]].lang,
                                ),
                            ) => {}
                    Back(kind) if k == 0 => match prev {
                        // `then` that starts a sentence after an `if` clause is that `if`'s correlate.
                        Some(p) if is_then(self.c, conn) && self.last_opens_if(p) => {}
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
                        let n_lang = effective_clause_lang(
                            self.c,
                            sent_lang,
                            n_span,
                            self.seg.clauses[n].lang,
                        );
                        let opaque = &self.seg.opaque;
                        match correlate(
                            self.c,
                            opaque,
                            n_span,
                            sentence.end_char,
                            results,
                            toks,
                            n_lang,
                        ) {
                            Some(cor) => {
                                self.used[n] = true;
                                // A bare correlate (`then,`) takes the clause after it as arg2.
                                let arg2 = if is_bare(self.c, cor.end_char, n_span.end_char) {
                                    let after = ids.get(k + 2).copied().filter(|&n2| {
                                        let n2_span = self.clause(n2);
                                        let n2_lang = effective_clause_lang(
                                            self.c,
                                            sent_lang,
                                            n2_span,
                                            self.seg.clauses[n2].lang,
                                        );
                                        connector(
                                            self.c,
                                            opaque,
                                            n2_span,
                                            sentence.end_char,
                                            toks,
                                            n2_lang,
                                        )
                                        .is_none()
                                    });
                                    let Some(n2) = after else {
                                        if !silent {
                                            self.unknown(conn);
                                        }
                                        continue;
                                    };
                                    self.used[n2] = true;
                                    self.clause(n2)
                                } else {
                                    n_span
                                };
                                self.emit(kind, Cl, conn, Some(cor), clause, arg2);
                            }
                            None if silent => {}
                            None if connector(
                                self.c,
                                opaque,
                                n_span,
                                sentence.end_char,
                                toks,
                                n_lang,
                            )
                            .is_some() =>
                            {
                                self.unknown(conn);
                            }
                            None => self.emit(kind, Cl, conn, None, clause, n_span),
                        }
                        continue;
                    }
                }
            }

            // Clause-end post-positional temporal frames (`…之後`, `…之際`, `…之時`).
            if !matches!(initial, Some((Open(..) | Ambiguous, _)))
                && let Some(n) = next
                && !self.used[n]
                && let Some(frame_conn) = post_temporal_frame(self.c, &self.seg.opaque, clause)
            {
                let n_span = self.clause(n);
                let n_lang =
                    effective_clause_lang(self.c, sent_lang, n_span, self.seg.clauses[n].lang);
                let n_start = skip_opening(self.c, &self.seg.opaque, n_span);
                if !is_bare(self.c, n_start, n_span.end_char)
                    && connector(
                        self.c,
                        &self.seg.opaque,
                        n_span,
                        sentence.end_char,
                        toks,
                        n_lang,
                    )
                    .is_none()
                {
                    self.emit(K::Sequence, Cl, frame_conn, None, clause, n_span);
                }
            }
        }
    }
}

fn effective_clause_lang(c: &[char], sent_lang: Lang, clause: Span, clause_lang: Lang) -> Lang {
    if clause_lang == Lang::Ja || (sent_lang == Lang::Ja && clause_lang != Lang::Zh) {
        return Lang::Ja;
    }
    if clause_lang == Lang::Ko || (sent_lang == Lang::Ko && clause_lang != Lang::Zh) {
        return Lang::Ko;
    }
    if sent_lang == Lang::Mixed || clause_lang == Lang::Mixed {
        let slice = &c[clause.start_char..clause.end_char];
        if slice
            .iter()
            .any(|&ch| matches!(ch, '\u{3040}'..='\u{30FF}'))
        {
            return Lang::Ja;
        }
        if slice
            .iter()
            .any(|&ch| matches!(ch, '\u{AC00}'..='\u{D7A3}'))
        {
            return Lang::Ko;
        }
    }
    clause_lang
}

/// The first connector of a clause, with its role. Special cases are applied here.
fn connector(
    c: &[char],
    opaque: &[Opaque],
    clause: Span,
    sentence_end: usize,
    sent_tokens: &[Token],
    lang: Lang,
) -> Option<(Role, Span)> {
    let from = skip_opening(c, opaque, clause);
    // A connector never starts inside code, a citation or another opaque span.
    if hides(opaque, from) {
        return None;
    }
    let (text, span) = longest(
        c,
        from,
        clause.end_char,
        ENTRIES
            .iter()
            .filter(|&&(entry, _)| match lang {
                Lang::Ja => is_ja_entry(entry) || entry.is_ascii(),
                Lang::Ko => is_ko_entry(entry) || entry.is_ascii(),
                _ => true,
            })
            .map(|e| e.0),
    )?;
    let role = ENTRIES.iter().find(|e| e.0 == text)?.1;
    let end = span.end_char;
    let next = next_nonspace(c, end, sentence_end);
    let word = next_word(c, end, sentence_end);
    let at = |i: usize, set: &str| c.get(i).is_some_and(|ch| set.contains(*ch));
    let starts_after = |from: usize, pat: &str| {
        let n = pat.chars().count();
        from + n <= sentence_end && pat.chars().eq(c[from..from + n].iter().copied())
    };
    let role = match text {
        "但" if at(end, "願凡求見") => return None,
        "若" if at(end, "干無非是即") => return None,
        "先" if at(end, "生前後") => return None,
        "不過" | "不过"
            if ["如此", "爾爾", "尔尔"]
                .iter()
                .any(|p| starts_after(end, p)) =>
        {
            return None;
        }
        "if" if word == "only" => return None,
        "however" if next != Some(',') => return None,
        "so" if SO_NOT.contains(&word.as_str()) => return None,
        "so" if next == Some(',') => Ambiguous,
        _ => role,
    };
    if is_ko_entry(text) && !ko_boundary_ok(c, end, clause.end_char) {
        return None;
    }
    if is_ja_entry(text) {
        if text == "また"
            && c.get(end)
                .is_some_and(|&ch| matches!(ch, 'は' | 'が' | 'だ'))
        {
            return None;
        }
        if text == "もし" && starts_after(end, "もし") {
            return None;
        }
        if !ja_boundary_ok(c, end, clause.end_char, sent_tokens) {
            return None;
        }
    }
    Some((role, span))
}

fn is_ko_entry(s: &str) -> bool {
    s.chars().any(|ch| matches!(ch, '\u{AC00}'..='\u{D7A3}'))
}

fn ko_boundary_ok(c: &[char], end: usize, clause_end: usize) -> bool {
    end == clause_end
        || c.get(end).is_some_and(|&ch| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    ',' | '，'
                        | '、'
                        | '.'
                        | '。'
                        | '!'
                        | '！'
                        | '?'
                        | '？'
                        | ';'
                        | '；'
                        | ':'
                        | '：'
                )
                || CLOSING.contains(&ch)
        })
}

fn is_ja_entry(s: &str) -> bool {
    s.chars().any(|ch| matches!(ch, '\u{3040}'..='\u{30FF}')) || s == "その後"
}

fn ja_boundary_ok(c: &[char], end: usize, clause_end: usize, sent_tokens: &[Token]) -> bool {
    end == clause_end
        || c.get(end).is_some_and(|&ch| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    '、' | ','
                        | '，'
                        | '.'
                        | '。'
                        | '!'
                        | '！'
                        | '?'
                        | '？'
                        | ';'
                        | '；'
                        | ':'
                        | '：'
                )
                || CLOSING.contains(&ch)
        })
        || sent_tokens.iter().any(|tok| tok.span.end_char == end)
}

/// A post-positional temporal frame (`之後`, `之后`, `之際`, `之际`, `之時`, `之时`) ending a clause
/// whose trimmed content is at least 3 chars long (ignoring trailing quotes and closers).
fn post_temporal_frame(c: &[char], opaque: &[Opaque], clause: Span) -> Option<Span> {
    let start = skip_opening(c, opaque, clause);
    let mut end = clause.end_char;
    while end > start
        && !hides(opaque, end - 1)
        && (c[end - 1].is_whitespace() || CLOSING.contains(&c[end - 1]) || c[end - 1] == '\'')
    {
        end -= 1;
    }
    if end.saturating_sub(start) < 3 {
        return None;
    }
    for &suffix in POST_SEQ_FRAMES {
        let n = suffix.chars().count();
        if end >= start + n
            && suffix.chars().eq(c[end - n..end].iter().copied())
            && (end - n..end).all(|idx| !hides(opaque, idx))
        {
            if matches!(suffix, "之後" | "之后") && is_non_temporal_after(&c[start..end - n]) {
                return None;
            }
            return Some(Span {
                start_char: end - n,
                end_char: end,
            });
        }
    }
    None
}

fn chars_ends_with(slice: &[char], pat: &str) -> bool {
    let n = pat.chars().count();
    slice.len() >= n && pat.chars().eq(slice[slice.len() - n..].iter().copied())
}

fn rfind_chars(slice: &[char], pat: &str) -> Option<usize> {
    let pat_len = pat.chars().count();
    if slice.len() < pat_len {
        return None;
    }
    (0..=slice.len() - pat_len)
        .rev()
        .find(|&i| pat.chars().eq(slice[i..i + pat_len].iter().copied()))
        .map(|i| i + pat_len)
}

/// Suppresses kinship (`故人之後` = descendant) and spatial (`站在石門之後` = behind) senses of
/// clause-final `之後` / `之后`.
fn is_non_temporal_after(stem: &[char]) -> bool {
    for &kin in KINSHIP_AFTER_STEMS {
        if chars_ends_with(stem, kin) {
            let kin_len = kin.chars().count();
            let prev = stem.len().checked_sub(kin_len + 1).map(|i| stem[i]);
            if !matches!(prev, Some('到' | '過' | '过' | '了' | '完')) {
                return true;
            }
        }
    }
    let mut latest_verb_end: Option<usize> = None;
    for &verb in SPATIAL_AFTER_VERBS {
        if let Some(v_end) = rfind_chars(stem, verb) {
            latest_verb_end = Some(latest_verb_end.map_or(v_end, |cur| cur.max(v_end)));
        }
    }
    if let Some(v_end) = latest_verb_end {
        let tail = &stem[v_end..];
        if !tail.is_empty() && !TEMPORAL_TAIL_WORDS.iter().any(|w| chars_ends_with(tail, w)) {
            return true;
        }
    }
    false
}

/// A correlate that starts the clause `clause`, if any. `so` counts only where it is a cause-result
/// connector (not `so far`, not `So,`).
fn correlate(
    c: &[char],
    opaque: &[Opaque],
    clause: Span,
    sentence_end: usize,
    texts: &'static [&'static str],
    sent_tokens: &[Token],
    lang: Lang,
) -> Option<Span> {
    let from = skip_opening(c, opaque, clause);
    if hides(opaque, from) {
        return None;
    }
    let (text, span) = longest(c, from, clause.end_char, texts.iter().copied())?;
    let ok = text != "so"
        || matches!(
            connector(c, opaque, clause, sentence_end, sent_tokens, lang),
            Some((Back(K::Cause), _))
        );
    let part_of_word = CORRELATE_STOP.iter().any(|&(word, stop)| {
        word == text && c.get(span.end_char).is_some_and(|ch| stop.contains(*ch))
    });
    (ok && !part_of_word).then_some(span)
}

/// Whether the connector span reads `then` (ASCII, case-insensitive).
fn is_then(c: &[char], conn: Span) -> bool {
    c[conn.start_char..conn.end_char]
        .iter()
        .collect::<String>()
        .eq_ignore_ascii_case("then")
}

/// Whether the clause starts with an `if` opener.
fn opens_if(
    c: &[char],
    opaque: &[Opaque],
    clause: Span,
    sentence_end: usize,
    sent_tokens: &[Token],
    lang: Lang,
) -> bool {
    matches!(
        connector(c, opaque, clause, sentence_end, sent_tokens, lang),
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

/// The start of the clause text after spaces, opening marks, emoji and symbols. It stops at an
/// opaque span other than an emoji.
fn skip_opening(c: &[char], opaque: &[Opaque], clause: Span) -> usize {
    let mut i = clause.start_char;
    while i < clause.end_char {
        if let Some(o) = opaque
            .iter()
            .find(|o| o.span.start_char <= i && i < o.span.end_char)
        {
            if o.kind != OpaqueKind::Emoji {
                break;
            }
            i = o.span.end_char;
        } else if c[i].is_whitespace() || OPENING.contains(&c[i]) || is_symbol(c[i]) {
            i += 1;
        } else {
            break;
        }
    }
    i.min(clause.end_char)
}

/// True when `i` is inside an opaque span that is not an emoji.
fn hides(opaque: &[Opaque], i: usize) -> bool {
    opaque
        .iter()
        .any(|o| o.kind != OpaqueKind::Emoji && o.span.start_char <= i && i < o.span.end_char)
}

/// Symbol and pictograph chars that can start a clause before a connector: arrows, dingbats,
/// emoji, and the marks `© ® ™`.
fn is_symbol(ch: char) -> bool {
    matches!(
        ch,
        '\u{A9}' | '\u{AE}' | '\u{2122}' | '\u{2190}'..='\u{2BFF}' | '\u{1F000}'..='\u{1FAFF}'
            | '\u{200D}' | '\u{FE0F}'
    )
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
