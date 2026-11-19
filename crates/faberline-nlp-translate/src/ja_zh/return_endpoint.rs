//! Distinguish a restored time setting from a physical destination of 戻る.
use super::{
    Atom, Bunsetsu, Document, TargetVariant, Upos, clock_case_governor, explicit_clock_case,
    is_building_floor, is_daypart_surface, is_existing_spatial_head, is_plain_clock_ni,
    is_time_setting, linked_joint_case, predicates_has_passive, render_bunsetsu_np,
    setting_time_label,
};

pub(super) fn is_place(noun: &Atom) -> bool {
    use faberline_nlp::lexicon::{SemanticClass, lookup_ja};
    !noun.is_locked
        && (is_existing_spatial_head(noun)
            || is_building_floor(noun)
            || matches!(noun.surface.as_str(), "店" | "岸" | "島" | "庭")
            || lookup_ja(&noun.surface).is_some_and(|entry| {
                entry.decided_upos == noun.upos && entry.sem_class == Some(SemanticClass::Location)
            }))
}

// In 夕方に古い店に戻る the analyzer can attach the time to 古い. Age
// identifies the place here. A temporary property such as 暗い may itself
// hold only at the named time.
pub(super) fn time_before_place_modifier(time: &Atom, source: &[Atom]) -> bool {
    let Some(adjective) = source.iter().find(|a| {
        a.doc_tok_idx.is_some()
            && a.doc_tok_idx == time.head
            && !a.is_locked
            && a.deprel == "amod"
            && a.upos == Some(Upos::Adj)
            && matches!(a.surface.as_str(), "古い" | "新しい")
    }) else {
        return false;
    };
    let Some(place) = source.iter().find(|a| {
        a.doc_tok_idx.is_some()
            && a.doc_tok_idx == adjective.head
            && a.deprel == "obl"
            && is_place(a)
    }) else {
        return false;
    };
    source.iter().any(|a| {
        a.doc_tok_idx.is_some()
            && a.doc_tok_idx == place.head
            && !a.is_locked
            && a.lemma.as_deref() == Some("戻る")
    }) && source
        .iter()
        .any(|case| linked_joint_case(place, case) && matches!(case.surface.as_str(), "に" | "へ"))
}

fn schedule_noun(noun: &Atom, source: &[Atom]) -> bool {
    !noun.is_locked
        && (matches!(
            noun.surface.as_str(),
            "時間" | "時刻" | "日時" | "日程" | "日付" | "予定"
        ) || matches!(noun.surface.as_str(), "枠" | "帯") && is_time_setting(noun, source))
}

// In 元に戻る the former state is explicit. Another time on the same
// verb describes when it was restored. A possessor's 元 can mean their side.
fn original_schedule_value(noun: &Atom, source: &[Atom]) -> bool {
    let Some(id) = noun.doc_tok_idx else {
        return false;
    };
    let has_genitive = |n: &Atom| {
        source
            .iter()
            .any(|a| a.deprel == "nmod" && a.head == n.doc_tok_idx)
    };
    if noun.surface == "元" {
        return !noun.is_locked && !has_genitive(noun);
    }
    schedule_noun(noun, source)
        && source.windows(2).any(|w| {
            !w[0].is_locked
                && w[0].surface == "元"
                && w[0].deprel == "nmod"
                && w[0].head == Some(id)
                && !has_genitive(&w[0])
                && !w[1].is_locked
                && w[1].surface == "の"
                && w[1].deprel == "case"
                && w[0].doc_tok_idx.is_some()
                && w[1].head == w[0].doc_tok_idx
        })
}

pub(super) fn has_original_value(source: &[Atom], head: usize) -> bool {
    source.windows(2).any(|w| {
        w[0].head == Some(head)
            && original_schedule_value(&w[0], source)
            && w[1].surface == "に"
            && linked_joint_case(&w[0], &w[1])
    })
}

// A time or a place can be the result of 戻る. Match its own governor and
// subject before removing the case marker; an event time keeps its usual role.
pub(super) fn resolve(
    groups: &[Bunsetsu],
    predicates: &[Bunsetsu],
    source: &[Atom],
    doc: &Document,
    target: TargetVariant,
) -> Option<(usize, String)> {
    let [predicate] = predicates else { return None };
    let mut verbs = predicate
        .atoms
        .iter()
        .filter(|a| a.upos == Some(Upos::Verb));
    let verb = verbs.next()?;
    if verb.is_locked
        || verb.lemma.as_deref() != Some("戻る")
        || verbs.next().is_some()
        || predicates_has_passive(predicates)
        || predicate
            .atoms
            .iter()
            .any(|a| matches!(a.lemma.as_deref(), Some("せる" | "させる")))
    {
        return None;
    }
    let head = verb.doc_tok_idx?;
    let schedule = source
        .iter()
        .any(|a| a.head == Some(head) && a.deprel == "nsubj" && schedule_noun(a, source));
    let explicit_original = schedule && has_original_value(source, head);
    let mut endpoints = groups.iter().enumerate().filter_map(|(index, group)| {
        if schedule && !explicit_original && is_plain_clock_ni(group) {
            let (clock, case) = explicit_clock_case(group)?;
            return (clock_case_governor(clock, case, doc) == Some(head)).then(|| {
                (
                    index,
                    format!("恢復到{}", setting_time_label(group, target)),
                )
            });
        }
        let (atoms, focused) = match group.atoms.as_slice() {
            [prefix @ .., case, topic]
                if matches!(topic.surface.as_str(), "は" | "も")
                    && !topic.is_locked
                    && topic.head == case.head =>
            {
                (&group.atoms[..prefix.len() + 1], true)
            }
            atoms => (atoms, false),
        };
        let [.., noun, case] = atoms else { return None };
        if noun.is_locked
            || case.is_locked
            || noun.head != Some(head)
            || noun.doc_tok_idx.is_none()
            || case.head != noun.doc_tok_idx
            || case.deprel != "case"
            || case.upos != Some(Upos::Adp)
            || noun.span.end_char != case.span.start_char
            || !(matches!(noun.deprel.as_str(), "obl" | "advmod")
                || focused && noun.deprel == "nsubj")
            || !matches!(case.surface.as_str(), "に" | "へ")
        {
            return None;
        }
        if schedule && case.surface == "に" {
            if explicit_original && original_schedule_value(noun, source) {
                if noun.surface == "元" && atoms.len() == 2 {
                    return Some((index, "恢復原狀".to_owned()));
                }
                let label = render_bunsetsu_np(group, target);
                return label
                    .strip_prefix("元的")
                    .map(|time| (index, format!("恢復到原本的{time}")));
            }
            if !explicit_original && is_daypart_surface(&noun.surface) {
                return Some((
                    index,
                    format!("恢復到{}", setting_time_label(group, target)),
                ));
            }
        }
        is_place(noun).then(|| (index, format!("回到{}", render_bunsetsu_np(group, target))))
    });
    let endpoint = endpoints.next()?;
    endpoints.next().is_none().then_some(endpoint)
}
