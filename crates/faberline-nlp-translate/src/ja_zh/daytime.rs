use super::{Atom, Upos, is_daypart_surface, selects_time_value, source_time_head};

const GLOSSARY_DAYTIME: &str = "__ja_glossary_daytime";
const GLOSSARY_TIME_PHRASE: &str = "__ja_glossary_daytime_phrase";
const DAYTIME: &str = "__ja_activity_daytime";

pub(super) fn glossary_role(chars: &[char]) -> Option<&'static str> {
    match chars {
        ['日', '中'] => Some(GLOSSARY_DAYTIME),
        ['日', '中', 'に'] | ['日', '中', 'に', 'は' | 'も'] => Some(GLOSSARY_TIME_PHRASE),
        _ => None,
    }
}

pub(super) fn is_modifier(atom: &Atom) -> bool {
    atom.is_locked && atom.lemma.as_deref() == Some(DAYTIME)
}

// A user term covering 日中に can tokenize as a single acl. Repair this
// before relative clauses are split, using the predicate of the noun which
// the term was attached to. A subject of that spurious acl follows it.
pub(super) fn restore_glossary_attachment(atoms: &mut [Atom]) {
    for i in 0..atoms.len() {
        let day = &atoms[i];
        if day.lemma.as_deref() != Some(GLOSSARY_TIME_PHRASE) || day.deprel != "acl" {
            continue;
        }
        let Some(argument) = atoms.iter().find(|a| {
            a.doc_tok_idx.is_some()
                && a.doc_tok_idx == day.head
                && a.upos == Some(Upos::Noun)
                && matches!(a.deprel.as_str(), "obj" | "obl" | "nsubj")
        }) else {
            continue;
        };
        let Some(verb) = atoms.iter().find(|a| {
            a.doc_tok_idx.is_some() && a.doc_tok_idx == argument.head && a.upos == Some(Upos::Verb)
        }) else {
            continue;
        };
        let (day_id, owner) = (day.doc_tok_idx, verb.doc_tok_idx);
        atoms[i].head = owner;
        atoms[i].deprel = "obl".to_owned();
        for a in &mut atoms[..i] {
            if a.head == day_id && a.deprel == "nsubj" {
                a.head = owner;
            }
        }
    }
}

// A 会 meeting may be tokenized as 説明 + 会. Only the object of this
// particular verb establishes its event reading of 開く.
fn meeting_object<'a>(source: &'a [Atom], verb: &Atom) -> Option<&'a str> {
    let noun = source.iter().find(|n| {
        !n.is_locked
            && n.head == verb.doc_tok_idx
            && n.deprel == "obj"
            && matches!(n.surface.as_str(), "会議" | "会合" | "会" | "大会")
    })?;
    Some(&noun.surface)
}

pub(super) fn meeting_verb(source: &[Atom], day: &Atom) -> Option<(super::Span, &'static str)> {
    let verb = source
        .iter()
        .find(|a| !a.is_locked && a.doc_tok_idx == day.head)?;
    if !matches!(
        verb.lemma.as_deref().unwrap_or(&verb.surface),
        "開く" | "開催" | "開催する"
    ) {
        return None;
    }
    let label = match meeting_object(source, verb)? {
        "会議" | "会合" => "召開",
        _ => "舉辦",
    };
    Some((verb.span, label))
}

// A separate destination or addressee for this verb leaves 日中 free to
// modify the event. A participant of another predicate cannot do so.
fn other_goal<'a>(source: &'a [Atom], day: &Atom, verb: &Atom) -> Option<&'a Atom> {
    source.windows(2).find_map(|pair| {
        let noun = &pair[0];
        let case = &pair[1];
        (noun.doc_tok_idx.is_some()
            && noun.doc_tok_idx != day.doc_tok_idx
            && noun.head == verb.doc_tok_idx
            && matches!(noun.deprel.as_str(), "iobj" | "obl")
            && !source_time_head(noun)
            && !is_daypart_surface(&noun.surface)
            && noun.surface != "日中"
            && !case.is_locked
            && case.deprel == "case"
            && case.head == noun.doc_tok_idx
            && matches!(case.surface.as_str(), "に" | "へ"))
        .then_some(case)
    })
}

// An unfilled participant or destination of these predicates can denote
// Japan and China. The contrast is lexical even when both parses use obl.
fn selects_participant(stem: &str) -> bool {
    matches!(
        stem,
        "送る"
            | "送付"
            | "郵送"
            | "発送"
            | "届ける"
            | "渡す"
            | "配る"
            | "配布"
            | "与える"
            | "贈る"
            | "伝える"
            | "知らせる"
            | "頼む"
            | "求める"
            | "要請"
            | "提案"
            | "打診"
            | "勧告"
            | "要求"
            | "依頼"
            | "期待"
            | "呼びかける"
            | "働きかける"
            | "輸出"
            | "輸入"
            | "投資"
            | "侵攻"
            | "抗議"
            | "制裁"
            | "譲渡"
    )
}

// The analyzer can split 日中にも as 日 + 中にも. Keep both source tokens
// in the marker's span without manufacturing syntax tokens for に and も.
fn form(a: &Atom, next: &Atom, following: Option<&Atom>) -> Option<(usize, usize, bool)> {
    if a.lemma.as_deref() == Some(GLOSSARY_TIME_PHRASE) {
        return Some((a.span.end_char, a.head?, false));
    }
    let glossary_day = a.lemma.as_deref() == Some(GLOSSARY_DAYTIME);
    if a.is_locked && !glossary_day
        || next.is_locked
        || a.span.end_char != next.span.start_char
        || a.doc_tok_idx.is_none()
        || next.doc_tok_idx.is_none()
    {
        return None;
    }
    if a.surface == "日"
        && next.surface == "中にも"
        && a.head == next.doc_tok_idx
        && a.deprel == "advmod"
        && next.deprel == "advmod"
    {
        return Some((next.span.end_char, next.head?, true));
    }
    if !(glossary_day || a.surface == "日中" && matches!(a.upos, Some(Upos::Noun | Upos::Adv)))
        || next.surface != "に"
        || next.upos != Some(Upos::Adp)
        || next.deprel != "case"
        || next.head != a.doc_tok_idx
    {
        return None;
    }
    let focus = following.filter(|f| {
        !f.is_locked
            && matches!(f.surface.as_str(), "は" | "も")
            && f.deprel == "case"
            && f.head == a.doc_tok_idx
            && f.span.start_char == next.span.end_char
    });
    if !(matches!(a.deprel.as_str(), "obl" | "iobj" | "advmod")
        || focus.is_some() && a.deprel == "nsubj")
    {
        return None;
    }
    Some((
        focus.unwrap_or(next).span.end_char,
        a.head?,
        focus.is_some_and(|f| f.surface == "も"),
    ))
}

pub(super) fn bind(atoms: &mut Vec<Atom>, source: &[Atom]) {
    for (idx, word) in source.iter().enumerate().rev() {
        let Some((end, owner, also)) = source
            .get(idx + 1)
            .and_then(|case| form(word, case, source.get(idx + 2)))
        else {
            continue;
        };
        if source
            .get(..idx)
            .and_then(|prev| prev.last())
            .is_some_and(|prev| {
                prev.span.end_char == word.span.start_char
                    && (prev.upos == Some(Upos::Num) || prev.is_locked)
            })
        {
            continue;
        }
        let Some(verb) = source
            .iter()
            .find(|a| !a.is_locked && a.doc_tok_idx == Some(owner) && a.upos == Some(Upos::Verb))
        else {
            continue;
        };
        // Passive agents and causees can occupy に even with a reading verb.
        if source.iter().any(|a| {
            a.head == Some(owner)
                && !a.is_locked
                && matches!(
                    a.lemma.as_deref().unwrap_or(&a.surface),
                    "れる" | "られる" | "せる" | "させる"
                )
        }) {
            continue;
        }
        // The parser sometimes assigns the matrix topic to a nested relative
        // verb. Leave the two predicates to the relative-clause renderer.
        if verb.deprel == "acl"
            && source.iter().any(|a| {
                a.head == Some(owner)
                    && a.deprel == "nsubj"
                    && source
                        .iter()
                        .any(|p| p.head == a.doc_tok_idx && p.deprel == "case" && p.surface == "は")
            })
        {
            continue;
        }
        let lemma = verb.lemma.as_deref().unwrap_or(&verb.surface);
        let stem = lemma.strip_suffix("する").unwrap_or(lemma);
        let recipient = other_goal(source, word, verb);
        // A market can be opened to two countries. A shop opening for the
        // day and a meeting being held instead have an event-time modifier.
        let takes_participant = selects_participant(stem)
            || matches!(stem, "開く" | "開放")
                && source.iter().any(|n| {
                    !n.is_locked
                        && n.head == verb.doc_tok_idx
                        && n.deprel == "obj"
                        && n.surface == "市場"
                })
            || stem.is_empty()
                && source.iter().any(|n| {
                    !n.is_locked && n.head == Some(owner) && selects_participant(&n.surface)
                });
        if selects_time_value(lemma) || takes_participant && recipient.is_none() {
            continue;
        }
        let Some(start) = atoms
            .iter()
            .position(|a| a.span.start_char == word.span.start_char)
        else {
            continue;
        };
        let Some(last) = atoms[start..].iter().position(|a| a.span.end_char == end) else {
            continue;
        };
        let range = start..=start + last;
        if atoms[range.clone()].iter().any(|a| {
            a.is_locked
                && !matches!(
                    a.lemma.as_deref(),
                    Some(GLOSSARY_DAYTIME | GLOSSARY_TIME_PHRASE)
                )
        }) {
            continue;
        }
        if takes_participant
            && let Some(case) = recipient.filter(|a| a.surface == "へ")
            && let Some(direction) = atoms
                .iter_mut()
                .find(|a| !a.is_locked && a.span == case.span)
        {
            direction.surface = "に".to_owned();
        }
        let mut day = word.clone();
        if !day.is_locked {
            day.surface = "白天".to_owned();
        }
        if also {
            day.surface.push('也');
        }
        day.lemma = Some(DAYTIME.to_owned());
        day.span.end_char = end;
        day.head = Some(owner);
        day.deprel = "advmod".to_owned();
        day.upos = Some(Upos::Adv);
        day.is_locked = true;
        atoms.splice(range, [day]);
    }
}
