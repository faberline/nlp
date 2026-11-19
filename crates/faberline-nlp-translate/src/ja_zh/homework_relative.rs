//! Locate classroom participants when a verb modifies a person.
use super::{
    Atom, Bunsetsu, BunsetsuRole, ClassroomRole, Document, TargetVariant, Upos, classroom_role,
    dasu_case_link, deadline_governor, prepend_locked_atom, try_translate_homework_transfer,
};

fn modifier_of<'a>(person: &Atom, source: &'a [Atom]) -> Option<&'a Atom> {
    source.iter().find(|a| {
        !a.is_locked
            && a.upos == Some(Upos::Verb)
            && a.deprel == "acl"
            && person.doc_tok_idx.is_some()
            && a.head == person.doc_tok_idx
    })
}

pub(super) fn person(group: &Bunsetsu, source: &[Atom]) -> Option<(ClassroomRole, String)> {
    let [modifiers @ .., noun, marker] = group.atoms.as_slice() else {
        return None;
    };
    if !dasu_case_link(noun, marker) {
        return None;
    }
    let (prefix, modifiers) = match modifiers {
        [folded, rest @ ..]
            if folded.is_locked
                && folded.doc_tok_idx.is_none()
                && folded.surface.ends_with('的')
                && modifier_of(noun, source).is_some() =>
        {
            (folded.surface.as_str(), rest)
        }
        _ => ("", modifiers),
    };
    let (role, name) = classroom_role(modifiers, noun)?;
    Some((role, format!("{prefix}{name}")))
}

pub(super) fn has_modified_person(groups: &[Bunsetsu], source: &[Atom]) -> bool {
    groups.iter().any(|g| {
        g.atoms
            .first()
            .is_some_and(|a| a.is_locked && a.doc_tok_idx.is_none())
            && person(g, source).is_some()
    })
}

pub(super) fn asks_modifier_time(groups: &[Bunsetsu], source: &[Atom]) -> bool {
    groups.iter().any(|g| {
        let [.., noun, _] = g.atoms.as_slice() else {
            return false;
        };
        person(g, source).is_some()
            && modifier_of(noun, source).is_some_and(|verb| {
                source.iter().any(|time| {
                    time.head == verb.doc_tok_idx && super::deadline_time_question(time).is_some()
                })
            })
    })
}

pub(super) fn modifies_classroom_argument(verb: &Atom, noun: &Atom, source: &[Atom]) -> bool {
    !verb.is_locked
        && verb.upos == Some(Upos::Verb)
        && verb.deprel == "acl"
        && noun.doc_tok_idx.is_some()
        && verb.head == noun.doc_tok_idx
        && source.iter().any(|obj| {
            !obj.is_locked
                && obj.surface == "宿題"
                && obj.deprel == "obj"
                && source.iter().any(|action| {
                    !action.is_locked
                        && action.lemma.as_deref().unwrap_or(&action.surface) == "出す"
                        && action.doc_tok_idx == obj.head
                        && (action.doc_tok_idx == verb.doc_tok_idx
                            || action.doc_tok_idx == noun.head)
                })
        })
}

pub(super) fn attached_predicate(predicate: &Bunsetsu, group: &Bunsetsu, source: &[Atom]) -> bool {
    let Some(verb) = predicate
        .atoms
        .iter()
        .find(|a| !a.is_locked && a.deprel == "acl" && a.upos == Some(Upos::Verb))
    else {
        return false;
    };
    group.atoms.iter().enumerate().any(|(i, noun)| {
        classroom_role(&group.atoms[..i], noun).is_some()
            && modifies_classroom_argument(verb, noun, source)
    })
}

pub(super) fn matrix_topic(
    topic: &Atom,
    marker: &Atom,
    verb: Option<usize>,
    source: &[Atom],
) -> bool {
    // In Aは [六時に来た B]に 宿題を出す the parser may attach the topic to 来た.
    // Recover は only when that relative's head is an explicit recipient of 出す.
    marker.surface == "は"
        && topic.deprel == "nsubj"
        && source.windows(2).any(|pair| {
            let [receiver, ni] = pair else { return false };
            receiver.head == verb
                && matches!(receiver.deprel.as_str(), "obl" | "iobj")
                && ni.surface == "に"
                && dasu_case_link(receiver, ni)
                && modifier_of(receiver, source).is_some_and(|acl| {
                    acl.doc_tok_idx == topic.head && acl.span.start_char >= marker.span.end_char
                })
        })
}

/// A teacher or student can arrive at noon and at school in the same clause.
/// Both arguments carry に. Only a time head can supply the arrival time.
pub(super) fn arrival_times(groups: &mut [Bunsetsu], source: &[Atom]) {
    let Some(verb) = groups.last().and_then(|g| g.atoms.first()) else {
        return;
    };
    if !matches!(
        verb.lemma.as_deref().unwrap_or(&verb.surface),
        "来る" | "行く" | "帰る" | "戻る" | "着く"
    ) {
        return;
    }
    let id = verb.doc_tok_idx;
    for group in groups {
        let [mods @ .., time, ni] = group.atoms.as_slice() else {
            continue;
        };
        if ni.surface != "に" || !dasu_case_link(time, ni) || time.head != id || time.is_locked {
            continue;
        }
        let label = super::deadline_time_question(time)
            .map(str::to_owned)
            .or_else(|| super::deadline_weekday(&time.surface).map(str::to_owned))
            .or_else(|| {
                super::is_daypart_surface(&time.surface)
                    .then(|| super::render_temporal_made_atom(time))
                    .flatten()
            });
        let Some(mut label) = label else { continue };
        if !mods.is_empty() {
            let [folded] = mods else { continue };
            if !folded.is_locked || folded.doc_tok_idx.is_some() {
                continue;
            }
            let Some(prefix) = source.windows(3).find_map(|w| {
                let [day, no, head] = w else { return None };
                (head.doc_tok_idx == time.doc_tok_idx
                    && no.surface == "の"
                    && dasu_case_link(day, no)
                    && day.deprel == "nmod"
                    && day.head == time.doc_tok_idx
                    && !day.is_locked)
                    .then(|| super::render_temporal_made_atom(day))
                    .flatten()
            }) else {
                continue;
            };
            label.insert_str(0, &prefix);
        }
        group.atoms = vec![Atom {
            surface: label,
            is_locked: true,
            ..time.clone()
        }];
        group.case_particle = None;
        group.role = BunsetsuRole::Adv;
    }
}

/// Keep the outer topic and the outer action's time for their own predicate.
pub(super) fn fold(
    groups: &mut Vec<Bunsetsu>,
    predicate: &Bunsetsu,
    noun_group: &mut Bunsetsu,
    source: &[Atom],
    doc: &Document,
    target: TargetVariant,
) -> bool {
    let Some(verb) = predicate.atoms.first() else {
        return false;
    };
    if verb.is_locked
        || verb.deprel != "acl"
        || verb.lemma.as_deref().unwrap_or(&verb.surface) != "出す"
    {
        return false;
    }
    let Some((role, name, noun)) = noun_group.atoms.iter().enumerate().find_map(|(i, n)| {
        let (role, name) = classroom_role(&noun_group.atoms[..i], n)?;
        (n.doc_tok_idx.is_some() && n.doc_tok_idx == verb.head).then_some((role, name, i))
    }) else {
        return false;
    };
    let start = groups
        .iter()
        .rposition(|g| {
            g.case_particle.as_deref() == Some("は")
                || (!g.atoms.iter().any(|a| a.head == verb.doc_tok_idx)
                    && deadline_governor(g, doc) != verb.doc_tok_idx)
        })
        .map_or(0, |i| i + 1);
    let mut args = Vec::new();
    for g in &groups[start..] {
        // 昨日先生に can be grouped together although 昨日 modifies 出す.
        let mut rest = g.clone();
        while rest.atoms.len() > 1
            && rest.atoms[0].deprel == "advmod"
            && rest.atoms[0].head == verb.doc_tok_idx
            && rest.atoms[0].upos == Some(Upos::Adv)
        {
            args.push(Bunsetsu {
                atoms: vec![rest.atoms.remove(0)],
                role: BunsetsuRole::Adv,
                case_particle: None,
                sconj_tail: None,
                sfp_tail: None,
            });
        }
        args.push(rest);
    }
    let mut pred = predicate.clone();
    pred.role = BunsetsuRole::Predicate;
    let modal = super::homework_desire::modal(&pred.atoms[1..]).filter(|_| {
        pred.atoms[1].head == verb.doc_tok_idx
            && pred.atoms[1].deprel == "aux"
            && pred.atoms.iter().all(|a| !a.is_locked)
            && pred
                .atoms
                .windows(2)
                .all(|w| w[0].span.end_char == w[1].span.start_char)
    });
    if modal.is_some() {
        pred.atoms.truncate(1);
    }
    args.push(pred);
    let Some(first) = args.first().and_then(|g| g.atoms.first()) else {
        return false;
    };
    // A physical source cannot disappear because of an intervening outer group.
    if source.iter().any(|a| {
        a.head == verb.doc_tok_idx
            && a.span.end_char <= first.span.start_char
            && matches!(a.deprel.as_str(), "obj" | "obl" | "iobj" | "advmod")
    }) {
        return false;
    }
    let Some(inner) = try_translate_homework_transfer(
        &args,
        source,
        doc,
        target,
        modal.map(|m| (m, false)),
        Some(role),
    ) else {
        return false;
    };
    groups.truncate(start);
    // Keep the original noun identity for an outer transfer. 大学 + 生
    // translates as one participant, with the case attached to the last token.
    let mut head = noun_group.atoms[noun].clone();
    head.lemma = Some(
        noun_group.atoms[..=noun]
            .iter()
            .map(|a| a.surface.as_str())
            .collect(),
    );
    head.span.start_char = noun_group.atoms[0].span.start_char;
    head.surface = name.to_owned();
    noun_group.atoms.splice(..=noun, [head]);
    prepend_locked_atom(noun_group, format!("{}的", inner.trim_end_matches('了')));
    true
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
    };

    #[test]
    fn transfers_follow_the_participant_modified_by_the_relative_clause() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        for (source, expected) in [
            (
                "六時に来た先生は生徒に宿題を出しました。",
                "在六點來的老師出作業給學生了。",
            ),
            (
                "六月七日に来た先生は生徒に宿題を出しました。",
                "在六月七日來的老師出作業給學生了。",
            ),
            (
                "六月七日に来た生徒は先生に宿題を出しました。",
                "在六月七日來的學生把作業交給老師了。",
            ),
            (
                "先生は六月七日に学校に来た生徒に六月八日までに宿題を出しませんでした。",
                "老師沒有在六月八日前出作業給在六月七日來到學校的學生。",
            ),
            (
                "生徒は六月七日に学校に来た先生に七時に宿題を出しました。",
                "學生在七點把作業交給在六月七日來到學校的老師了。",
            ),
            (
                "六月七日に先生に宿題を出した生徒は六月十日に学校に来ました。",
                "在六月七日把作業交給老師的學生在六月十日來到學校了。",
            ),
            (
                "六月七日に来た生徒は先生に何日に宿題を出しますか。",
                "在六月七日來的學生幾號把作業交給老師？",
            ),
            (
                "六月七日に来た生徒は机の引き出しから宿題を出しました。",
                "在六月七日來的生徒從書桌的抽屜拿出了作業。",
            ),
            (
                "生徒は六時に来た先生に七時に宿題を出しました。",
                "學生在七點把作業交給在六點來的老師了。",
            ),
            (
                "明日来る先生は生徒に宿題を出します。",
                "明天來的老師出作業給學生。",
            ),
            (
                "本を読んだ先生は生徒に宿題を出しました。",
                "讀了書的老師出作業給學生了。",
            ),
            (
                "正午に学校に来た先生は生徒に宿題を出しました。",
                "正午來到學校的老師出作業給學生了。",
            ),
            (
                "先生は昨日宿題を出した生徒に宿題を出しました。",
                "老師出作業給昨天交作業的學生了。",
            ),
            (
                "宿題を出したい生徒は今日学校に来ました。",
                "想交作業的學生今天來到學校了。",
            ),
            (
                "六時に来た先生は生徒に宿題を出したいです。",
                "在六點來的老師想出作業給學生。",
            ),
            (
                "先生は明日の何時に帰る生徒に宿題を出したいですか。",
                "老師想出作業給明天幾點回去的學生？",
            ),
            (
                "先生は六時に来た生徒に宿題を出しました。",
                "老師出作業給在六點來的學生了。",
            ),
            (
                "昨日先生に宿題を出した生徒は今日学校に来ました。",
                "昨天把作業交給老師的學生今天來到學校了。",
            ),
            (
                "昨日宿題を出した学生は本を買いました。",
                "昨天交作業的學生買了書。",
            ),
            (
                "生徒に宿題を出した先生は本を読みました。",
                "出作業給學生的老師讀了書。",
            ),
            (
                "先生に宿題を出さなかった生徒は六時に来ました。",
                "沒有把作業交給老師的學生在六點來了。",
            ),
            (
                "六時に来た先生は机の引き出しから宿題を出しました。",
                "在六點來的老師從書桌的抽屜拿出了作業。",
            ),
            (
                "昨日宿題を出した人は本を買いました。",
                "昨天拿出了作業的人買了書。",
            ),
            (
                "六時に来た先生の友達は生徒に宿題を出しました。",
                "在六點來的老師的朋友給生徒拿出了作業。",
            ),
        ] {
            let actual = analyze_and_translate(
                source,
                &options,
                &TranslationGlossary::new(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(actual.target_text, expected, "{source}");
        }
    }
}
