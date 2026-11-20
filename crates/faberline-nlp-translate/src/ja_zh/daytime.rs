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

// A glossary entry covering 日中に can be analyzed as a verb. Recover
// the real head directly for advcl, or through the modified noun for acl.
pub(super) fn restore_glossary_attachment(atoms: &mut [Atom]) {
    for i in 0..atoms.len() {
        let day = &atoms[i];
        if day.lemma.as_deref() != Some(GLOSSARY_TIME_PHRASE) || day.doc_tok_idx.is_none() {
            continue;
        }
        let owner = match day.deprel.as_str() {
            "advcl" => day.head,
            "acl" => atoms
                .iter()
                .find(|a| {
                    a.doc_tok_idx.is_some()
                        && a.doc_tok_idx == day.head
                        && a.upos == Some(Upos::Noun)
                        && matches!(a.deprel.as_str(), "obj" | "obl" | "nsubj")
                })
                .and_then(|noun| noun.head),
            _ => continue,
        };
        let Some(verb) = atoms.iter().find(|a| {
            a.doc_tok_idx.is_some() && a.doc_tok_idx == owner && a.upos == Some(Upos::Verb)
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
        if let Some(comma) = atoms.get_mut(i + 1)
            && comma.head == day_id
            && comma.deprel == "punct"
            && !comma.is_locked
            && matches!(comma.surface.as_str(), "、" | "，" | ",")
        {
            comma.head = owner;
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
fn other_goal<'a>(
    source: &'a [Atom],
    day: &Atom,
    verb: &Atom,
    day_also: bool,
) -> Option<(&'a Atom, &'a Atom)> {
    source.windows(2).find_map(|pair| {
        let noun = &pair[0];
        let case = &pair[1];
        let focus = source.iter().find(|a| {
            !a.is_locked
                && a.head == noun.doc_tok_idx
                && a.deprel == "case"
                && matches!(a.surface.as_str(), "は" | "も")
                && a.span.start_char == case.span.end_char
        });
        (noun.doc_tok_idx.is_some()
            && noun.doc_tok_idx != day.doc_tok_idx
            && noun.head == verb.doc_tok_idx
            && (matches!(noun.deprel.as_str(), "iobj" | "obl")
                || noun.deprel == "nsubj" && focus.is_some())
            // 日中にも…米国にも can coordinate two political recipients.
            && !(day_also && focus.is_some_and(|p| p.surface == "も"))
            && !source_time_head(noun)
            && !is_daypart_surface(&noun.surface)
            && noun.surface != "日中"
            && !case.is_locked
            && case.deprel == "case"
            && case.head == noun.doc_tok_idx
            && matches!(case.surface.as_str(), "に" | "へ"))
        .then_some((noun, case))
    })
}

// A second へ can name a shipping destination. Only prefer 給 when
// its own head is a person, including a caller-designated actor.
fn personal_addressee(noun: &Atom, source: &[Atom], doc: &super::Document) -> bool {
    if noun.is_locked {
        return super::honorific::is_bound_person(noun);
    }
    let is_person = |a: &Atom| {
        a.doc_tok_idx
            .and_then(|id| doc.tokens.get(id))
            .is_some_and(|t| t.span == a.span && t.sem_class == Some(super::SemanticClass::Person))
    };
    is_person(noun)
        || noun.upos != Some(Upos::Propn) && super::is_demonstrative_person_noun(noun)
        || (super::is_honorific_suffix(&noun.surface) || noun.surface == "者")
            && source.iter().any(|a| {
                a.head == noun.doc_tok_idx
                    && a.deprel == "compound"
                    && (noun.surface == "者" || is_person(a))
            })
}

// A date used as the topic supplies the day whose daytime is meant.
// Distribution can omit the recipients, as long as the same verb has
// an overt distributed object. A date alone cannot settle a request.
fn dated_distribution(source: &[Atom], day_index: usize, verb: &Atom) -> bool {
    let lemma = verb.lemma.as_deref().unwrap_or(&verb.surface);
    if !matches!(lemma.strip_suffix("する").unwrap_or(lemma), "配る" | "配布") {
        return false;
    }
    let [.., date, topic] = &source[..day_index] else {
        return false;
    };
    !date.is_locked
        && date.doc_tok_idx.is_some()
        && date.head == verb.doc_tok_idx
        && (matches!(
            date.surface.as_str(),
            "今日" | "本日" | "昨日" | "一昨日" | "明日" | "明後日" | "当日"
        ) || super::deadline_weekday(&date.surface).is_some())
        && !topic.is_locked
        && matches!(topic.surface.as_str(), "は" | "も")
        && topic.deprel == "case"
        && topic.head == date.doc_tok_idx
        && topic.span.end_char == source[day_index].span.start_char
        && source.iter().any(|a| {
            a.head == verb.doc_tok_idx
                && a.deprel == "obj"
                && a.doc_tok_idx.is_some()
                && source.iter().any(|p| {
                    !p.is_locked
                        && p.surface == "を"
                        && p.deprel == "case"
                        && p.head == a.doc_tok_idx
                })
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

fn parallel_goals(source: &[Atom], day: &Atom, owner: usize) -> bool {
    source.windows(3).any(|p| {
        let (other, case, also) = (&p[0], &p[1], &p[2]);
        other.head == Some(owner)
            && other.doc_tok_idx != day.doc_tok_idx
            && !source_time_head(other)
            && !is_daypart_surface(&other.surface)
            && !case.is_locked
            && case.surface == "に"
            && case.deprel == "case"
            && case.head == other.doc_tok_idx
            && !also.is_locked
            && also.surface == "も"
            && also.deprel == "case"
            && also.head == other.doc_tok_idx
    })
}

fn passive_or_causative(source: &[Atom], owner: usize) -> bool {
    source.iter().any(|a| {
        a.head == Some(owner)
            && !a.is_locked
            && matches!(
                a.lemma.as_deref().unwrap_or(&a.surface),
                "れる" | "られる" | "せる" | "させる"
            )
    })
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

pub(super) fn bind(atoms: &mut Vec<Atom>, source: &[Atom], doc: &super::Document) {
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
        if passive_or_causative(source, owner) {
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
        let recipient = other_goal(source, word, verb, also);
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
        if selects_time_value(lemma)
            || also
                && matches!(stem, "出す" | "提出" | super::application_form::SUBMIT)
                && parallel_goals(source, word, owner)
            || takes_participant && recipient.is_none() && !dated_distribution(source, idx, verb)
        {
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
            && let Some((_, case)) = recipient.filter(|(noun, case)| {
                case.surface == "へ" && personal_addressee(noun, source, doc)
            })
            && let Some(direction) = atoms
                .iter_mut()
                .find(|a| !a.is_locked && a.span == case.span)
        {
            direction.surface = "に".to_owned();
        }
        atoms.splice(range, [time_modifier(word, end, owner, also)]);
    }
    bind_saury_meal(atoms, source);
}

fn case_for(noun: &Atom, case: &Atom, particle: &str) -> bool {
    !noun.is_locked
        && !case.is_locked
        && noun.doc_tok_idx.is_some()
        && case.head == noun.doc_tok_idx
        && case.deprel == "case"
        && case.surface == particle
        && noun.span.end_char == case.span.start_char
}

fn meal_name<'a>(previous: Option<&'a Atom>, meal: &'a Atom) -> Option<(&'a Atom, &'static str)> {
    match meal.surface.as_str() {
        "朝食" | "朝ご飯" => Some((meal, "早餐")),
        "昼食" | "昼ご飯" => Some((meal, "午餐")),
        "夕食" | "晩ご飯" => Some((meal, "晚餐")),
        "ご飯" => {
            let part = previous.filter(|part| {
                !part.is_locked
                    && (part.deprel == "compound" && part.head == meal.doc_tok_idx
                        || part.deprel == "advmod" && part.head == meal.head)
                    && part.span.end_char == meal.span.start_char
            })?;
            match part.surface.as_str() {
                "朝" => Some((part, "早餐")),
                "昼" => Some((part, "午餐")),
                "晩" => Some((part, "晚餐")),
                _ => None,
            }
        }
        _ => None,
    }
}

// The meal and the fish must both belong to this one cooking action. An
// object belonging to a nearby verb cannot disambiguate the meal's に.
fn bind_saury_meal(atoms: &mut Vec<Atom>, source: &[Atom]) {
    for (idx, pair) in source.windows(2).enumerate().rev() {
        let [meal, case] = pair else { continue };
        if meal.upos != Some(Upos::Noun) || meal.deprel != "obl" || !case_for(meal, case, "に") {
            continue;
        }
        let Some((first, label)) = meal_name(idx.checked_sub(1).map(|i| &source[i]), meal) else {
            continue;
        };
        let Some(owner) = meal.head else { continue };
        let end = case.span.end_char;
        if source.iter().any(|a| {
            a.head == meal.doc_tok_idx
                && (matches!(a.deprel.as_str(), "nmod" | "amod")
                    || a.deprel == "case" && a.span.start_char == end)
        }) || passive_or_causative(source, owner)
        {
            continue;
        }
        let Some(verb) = source
            .iter()
            .find(|a| !a.is_locked && a.doc_tok_idx == Some(owner) && a.upos == Some(Upos::Verb))
        else {
            continue;
        };
        let verb_label = match verb.lemma.as_deref().unwrap_or(&verb.surface) {
            "焼く" => Some("烤"),
            "煮る" => Some("煮"),
            "食べる" => None,
            _ => continue,
        };
        let Some(fish) = source.windows(2).find_map(|pair| {
            let [fish, case] = pair else { return None };
            (fish.upos == Some(Upos::Noun)
                && fish.deprel == "obj"
                && fish.head == Some(owner)
                && fish.lemma.as_deref().unwrap_or(&fish.surface) == "サンマ"
                && fish.span.start_char >= end
                && case.span.end_char <= verb.span.start_char
                && case_for(fish, case, "を"))
            .then_some(fish)
        }) else {
            continue;
        };
        let Some(start) = atoms
            .iter()
            .position(|a| a.span.start_char == first.span.start_char)
        else {
            continue;
        };
        let Some(last) = atoms[start..].iter().position(|a| a.span.end_char == end) else {
            continue;
        };
        let range = start..=start + last;
        let Some(fish) = atoms
            .iter()
            .position(|a| !a.is_locked && a.span == fish.span)
        else {
            continue;
        };
        let Some(verb) = atoms
            .iter()
            .position(|a| !a.is_locked && a.span == verb.span)
        else {
            continue;
        };
        if atoms[range.clone()].iter().any(|a| a.is_locked) {
            continue;
        }
        atoms[fish].surface = "秋刀魚".to_owned();
        atoms[fish].is_locked = true;
        if let Some(label) = verb_label {
            atoms[verb].surface = label.to_owned();
            atoms[verb].is_locked = true;
        }
        let mut modifier = time_modifier(meal, end, owner, false);
        modifier.span.start_char = first.span.start_char;
        modifier.surface = label.to_owned();
        atoms.splice(range, [modifier]);
    }
}

fn time_modifier(word: &Atom, end: usize, owner: usize, also: bool) -> Atom {
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
    day
}

// Identify the time before selecting the filing or extraction sense of 出す.
// A comma in a series of actions can depend on the final verb. The time and
// the form must instead share the actual action's own dependency node.
pub(super) fn bind_form_time(atoms: &mut Vec<Atom>, chars: &[char], doc: &super::Document) {
    for i in (0..atoms.len().saturating_sub(1)).rev() {
        let word = &atoms[i];
        let Some((end, owner, also)) = atoms
            .get(i + 1)
            .and_then(|case| form(word, case, atoms.get(i + 2)))
        else {
            continue;
        };
        if i.checked_sub(1).is_some_and(|idx| {
            let a = &atoms[idx];
            a.span.end_char == word.span.start_char && (a.upos == Some(Upos::Num) || a.is_locked)
        }) {
            continue;
        }
        let Some(action) = atoms.iter().position(|a| a.doc_tok_idx == Some(owner)) else {
            continue;
        };
        if action <= i
            || passive_or_causative(atoms, owner)
            || !super::application_form::has_form_object(atoms, &atoms[action], chars)
        {
            continue;
        }
        let Some(last) = (i..action).find(|&j| atoms[j].span.end_char == end) else {
            continue;
        };
        if also && parallel_goals(atoms, word, owner) {
            super::application_form::paired_recipients(atoms, i, last, owner, doc);
            continue;
        }
        if !super::application_form::has_form_role(atoms, owner, chars) {
            continue;
        }
        let pause = atoms.get(last + 1).is_some_and(|a| {
            !a.is_locked
                && a.deprel == "punct"
                && a.span.start_char == end
                && matches!(a.surface.as_str(), "、" | "，" | ",")
        });
        let tail = last + usize::from(pause);
        if atoms[tail + 1..action]
            .iter()
            .any(|a| super::is_punct_atom(a) || a.upos == Some(Upos::Verb) || a.deprel == "nsubj")
        {
            continue;
        }
        let day = time_modifier(word, end, owner, also);
        atoms.splice(i..=tail, [day]);
    }
}

// A pause between two modifiers need not end the clause: in 日中に、患者に
// 送った the time, recipient and pause all depend on the same sending verb.
pub(super) fn join_recipient_pauses(atoms: &mut Vec<Atom>) {
    for split in (1..atoms.len()).rev() {
        let comma = &atoms[split];
        if comma.is_locked
            || comma.deprel != "punct"
            || !matches!(comma.surface.as_str(), "、" | "，" | ",")
        {
            continue;
        }
        let Some(owner) = comma.head else { continue };
        let Some(offset) = atoms[split + 1..].iter().position(|a| {
            a.doc_tok_idx == Some(owner) && !a.is_locked && a.upos == Some(Upos::Verb)
        }) else {
            continue;
        };
        let verb_index = split + 1 + offset;
        let verb = &atoms[verb_index];
        let lemma = verb.lemma.as_deref().unwrap_or(&verb.surface);
        if !selects_participant(lemma.strip_suffix("する").unwrap_or(lemma))
            || atoms[split + 1..verb_index].iter().any(|a| {
                super::is_punct_atom(a)
                    || a.upos == Some(Upos::Verb)
                    || a.deprel == "nsubj"
                        && !atoms[split + 1..verb_index].iter().any(|p| {
                            !p.is_locked
                                && p.deprel == "case"
                                && p.head == a.doc_tok_idx
                                && matches!(p.surface.as_str(), "を" | "に" | "へ")
                        })
            })
        {
            continue;
        }
        let joins = (split.saturating_sub(3)..split).any(|idx| {
            let day = &atoms[idx];
            let Some((end, head, also)) = form(day, &atoms[idx + 1], atoms.get(idx + 2)) else {
                return false;
            };
            end == comma.span.start_char
                && head == owner
                && !atoms[..idx].last().is_some_and(|a| {
                    a.span.end_char == day.span.start_char
                        && (a.upos == Some(Upos::Num) || a.is_locked)
                })
                && (dated_distribution(atoms, idx, verb)
                    || other_goal(atoms, day, verb, also).is_some_and(|(_, case)| {
                        comma.span.end_char <= case.span.start_char
                            && case.span.end_char <= verb.span.start_char
                    }))
        });
        if joins {
            atoms.remove(split);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

    #[test]
    fn a_meal_and_its_fish_object_follow_the_same_cooking_action() {
        let mut options = TranslateOptions::new(TargetVariant::ZhHant);
        options.source_hint = Some(TargetVariant::Ja);
        let translate = |source| {
            analyze_and_translate(
                source,
                &options,
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap()
            .target_text
        };
        for (source, expected) in [
            ("夕食にさんまを焼きました。", "晚餐烤了秋刀魚。"),
            ("昼ご飯にさんまを食べました。", "午餐吃了秋刀魚。"),
            ("朝食にさんまを煮ました。", "早餐煮了秋刀魚。"),
            ("夕食にさんまを焼きませんでした。", "晚餐沒有烤秋刀魚。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        assert!(
            translate("紙に「夕食にさんまを焼きました」と一字ずつそのまま写しました。")
                .contains("「夕食にさんまを焼きました」")
        );
        for source in [
            "明石家さんまがテレビに出ました。",
            "夕食に魚を買い、台所でさんまを焼きました。",
        ] {
            assert!(!translate(source).contains("秋刀魚"), "{source}");
        }
    }
}
