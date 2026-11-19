use super::{Atom, LOCKED_RESIDENT_ATOM, Upos, is_companion_person, is_demonstrative_person_noun};

const DAYTIME: &str = "__ja_activity_daytime";

pub(super) fn is_modifier(atom: &Atom) -> bool {
    atom.is_locked && atom.lemma.as_deref() == Some(DAYTIME)
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

// Another person's に or へ identifies the recipient only for the very
// same act of sending. A letter by itself cannot distinguish the readings.
fn recipient_case<'a>(source: &'a [Atom], verb: &Atom) -> Option<&'a Atom> {
    source.windows(2).find_map(|pair| {
        let noun = &pair[0];
        let case = &pair[1];
        (noun.doc_tok_idx.is_some()
            && noun.head == verb.doc_tok_idx
            && matches!(noun.deprel.as_str(), "iobj" | "obl")
            && (noun.is_locked && noun.lemma.as_deref() == Some(LOCKED_RESIDENT_ATOM)
                || !noun.is_locked
                    && (is_companion_person(noun)
                        || noun.upos == Some(Upos::Noun)
                            && (is_demonstrative_person_noun(noun) || noun.surface == "客")))
            && !case.is_locked
            && case.deprel == "case"
            && case.head == noun.doc_tok_idx
            && matches!(case.surface.as_str(), "に" | "へ"))
        .then_some(case)
    })
}

fn ordinary_activity(stem: &str) -> bool {
    matches!(
        stem,
        // Work, rest, weather and a bounded set of ordinary activities do
        // not require a second participant. Requests and negotiations do.
        "働く"
            | "休む"
            | "寝る"
            | "眠る"
            | "起きる"
            | "暮らす"
            | "過ごす"
            | "読む"
            | "食べる"
            | "飲む"
            | "洗う"
            | "片付ける"
            | "掃除"
            | "料理"
            | "勉強"
            | "練習"
            | "泳ぐ"
            | "走る"
            | "歩く"
            | "散歩"
            | "遊ぶ"
            | "降る"
            | "晴れる"
            | "咲く"
            | "始まる"
            | "終わる"
            | "開催"
    )
}

// The analyzer can split 日中にも as 日 + 中にも. Keep both source tokens
// in the marker's span without manufacturing syntax tokens for に and も.
fn form(a: &Atom, next: &Atom, following: Option<&Atom>) -> Option<(usize, usize, bool)> {
    if a.is_locked
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
    if a.surface != "日中"
        || !matches!(a.upos, Some(Upos::Noun | Upos::Adv))
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
        let recipient = match stem {
            "送る" | "送付" | "郵送" | "発送" | "届ける" | "渡す" => {
                let Some(case) = recipient_case(source, verb) else {
                    continue;
                };
                Some(case)
            }
            "開く" if meeting_object(source, verb).is_some() => None,
            "" if source.iter().any(|n| {
                !n.is_locked
                    && n.deprel == "obj"
                    && n.head == Some(owner)
                    && n.span.start_char >= end
                    && ordinary_activity(&n.surface)
            }) =>
            {
                None
            }
            activity if ordinary_activity(activity) => None,
            _ => continue,
        };
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
        if atoms[range.clone()].iter().any(|a| a.is_locked) {
            continue;
        }
        if let Some(case) = recipient.filter(|a| a.surface == "へ")
            && let Some(direction) = atoms
                .iter_mut()
                .find(|a| !a.is_locked && a.span == case.span)
        {
            direction.surface = "に".to_owned();
        }
        let mut day = word.clone();
        day.surface = if also { "白天也" } else { "白天" }.to_owned();
        day.lemma = Some(DAYTIME.to_owned());
        day.span.end_char = end;
        day.head = Some(owner);
        day.deprel = "advmod".to_owned();
        day.upos = Some(Upos::Adv);
        day.is_locked = true;
        atoms.splice(range, [day]);
    }
}
