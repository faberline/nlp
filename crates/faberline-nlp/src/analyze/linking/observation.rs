use super::{
    ClauseCandidate, ClauseScope, NarrativeAnalyzer, Source, Span, Unit, UnitScope,
    candidate_scope, continuation, inside_lexical_term, participant_description,
    reading::asserted_unit, scene_transition, scopes_for_units, split_units, starts_any,
    trim_clause_end,
};
use crate::Lexicon;

/// A written observation with its explicit completion marker and duration.
/// Ranges count Unicode scalar values in the original chapter.
#[derive(Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DurationObservation {
    pub clause: Span,
    pub predicate: Span,
    pub completion: Span,
    pub duration: Span,
    /// Legacy parser-selected label, not a verified identity.
    pub actor: Option<String>,
    pub actor_surface: Option<Span>,
    /// The object stays open in the closed-duration construction.
    pub object: Option<Span>,
}

const DURATIONS: &[&str] = &[
    "好一會兒",
    "好一会儿",
    "一小會兒",
    "一小会儿",
    "一會兒",
    "一会儿",
    "好一會",
    "好一会",
    "一小會",
    "一小会",
    "一會",
    "一会",
    "片刻",
    "半晌",
    "半響",
    "半响",
];
const OBSERVATION_VERBS: &[&str] = &[
    "觀看", "观看", "察看", "查看", "翻看", "細看", "细看", "觀察", "观察", "閱讀", "阅读", "翻閱",
    "翻阅", "查閱", "查阅", "細讀", "细读", "看", "讀", "读", "閱", "阅",
];
const EARLIER_OBSERVATIONS: &[&str] = &[
    "看著", "看着", "看向", "盯著", "盯着", "望著", "望着", "注視", "注视", "看了", "讀了", "读了",
    "閱了", "阅了",
];

const PRONOUNS: &[&str] = &["我們", "我们", "他們", "他们", "我", "你", "他", "她", "它"];
const LOCATIVES: &[char] = &['內', '内', '中', '前', '旁', '邊', '边', '裡', '里'];
const NONACTUAL: &[&str] = &[
    "差一點",
    "差一点",
    "差點",
    "差点",
    "險些",
    "险些",
    "本可以",
    "本可",
    "原可以",
    "假裝",
    "假装",
    "佯裝",
    "佯装",
    "裝作",
    "装作",
];
const DEPENDENT_NEXT: &[&str] = &[
    "再",
    "務必",
    "务必",
    "記得",
    "记得",
    "就可以",
    "才可以",
    "便可",
    "就要",
    "將要",
    "将要",
];

#[derive(Default)]
struct Observer<'a> {
    actor: Option<&'a str>,
    surface: Option<Span>,
}

fn written_observer<'a>(
    source: &Source<'_>,
    lexicon: &Lexicon<'a>,
    start: usize,
    text: &str,
) -> Option<Observer<'a>> {
    let Some(surface) = source.trim(start, start + text.len()) else {
        return Some(Observer::default());
    };
    let name = source.slice(&surface);
    if let Some((0, word, actor)) = source.first_name(
        source.bytes[surface.start_char],
        source.bytes[surface.end_char],
        lexicon,
    ) && word == name
    {
        return Some(Observer {
            actor: Some(actor),
            surface: Some(surface),
        });
    }
    (PRONOUNS.contains(&name) || participant_description(name)).then_some(Observer {
        actor: None,
        surface: Some(surface),
    })
}

fn located_observer<'a>(
    source: &Source<'_>,
    lexicon: &Lexicon<'a>,
    start: usize,
    lead: &str,
) -> Option<Observer<'a>> {
    lead.match_indices(['在', '於', '于'])
        .find_map(|(offset, marker)| {
            let before = &lead[..offset];
            let after = lead[offset + marker.len()..].trim_start();
            let mut tails = lexicon
                .places
                .iter()
                .filter(|place| !place.is_empty())
                .filter_map(|place| after.strip_prefix(place))
                .flat_map(|tail| {
                    [tail.strip_prefix(LOCATIVES), Some(tail)]
                        .into_iter()
                        .flatten()
                });
            if before.trim().is_empty() {
                tails
                    .filter_map(|tail| {
                        let tail_start = start + lead.len() - tail.len();
                        written_observer(source, lexicon, tail_start, tail)
                            .map(|actor| (tail.len(), actor))
                    })
                    .min_by_key(|(remaining, _)| *remaining)
                    .map(|(_, actor)| actor)
            } else if tails.any(|tail| tail.trim().is_empty()) {
                written_observer(source, lexicon, start, before)
            } else {
                None
            }
        })
}

fn observer<'a>(
    source: &Source<'_>,
    lexicon: &Lexicon<'a>,
    unit: &Unit,
    head_start: usize,
    manner_suffixes: &[char],
) -> Option<Observer<'a>> {
    let base = source.bytes[unit.evidence.start_char];
    let prefix = source.text[base..head_start].trim_end();
    let lead = prefix.trim_start();
    let lead = continuation(lead).unwrap_or(lead).trim_start();
    let start = base + prefix.len() - lead.len();
    let parse = |lead| {
        written_observer(source, lexicon, start, lead)
            .or_else(|| located_observer(source, lexicon, start, lead))
    };
    parse(lead).or_else(|| {
        let manner = lead
            .strip_suffix(manner_suffixes)
            .unwrap_or(lead)
            .trim_end();
        let before = [
            "仔細", "仔细", "細細", "细细", "略微", "稍微", "足足", "靜靜", "静静", "默默", "認真",
            "认真", "專注", "专注", "緩緩", "缓缓", "略",
        ]
        .iter()
        .find_map(|modifier| manner.strip_suffix(modifier))?;
        parse(before)
    })
}

fn nonactual_modifier(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    unit: &Unit,
    end: usize,
    detached: bool,
) -> Option<Span> {
    let start = source.bytes[unit.evidence.start_char];
    let prefix = &source.text[start..end];
    NONACTUAL.iter().find_map(|word| {
        prefix.match_indices(word).find_map(|(offset, word)| {
            let marker_start = start + offset;
            let marker_end = marker_start + word.len();
            let marker = source.span(marker_start, marker_end);
            ((!detached || marker_end == end)
                && !inside_lexical_term(source, lexicon, unit.evidence, marker)
                && observer(source, lexicon, unit, marker_start, &['地', '的']).is_some())
            .then_some(marker)
        })
    })
}

fn continued_observation(
    source: &Source<'_>,
    lexicon: &Lexicon<'_>,
    unit: &Unit,
    current: &Observer<'_>,
    previous: &Unit,
    scope: &UnitScope,
) -> bool {
    let text = source.slice(&previous.evidence);
    if unit.first_in_line
        || previous.sentence != unit.sentence
        || !text.ends_with(['，', ','])
        || !asserted_unit(source, previous, scope)
    {
        return false;
    }
    let start = source.bytes[previous.evidence.start_char];
    EARLIER_OBSERVATIONS
        .iter()
        .chain(
            OBSERVATION_VERBS
                .iter()
                .filter(|verb| verb.chars().count() > 1),
        )
        .any(|verb| {
            text.match_indices(verb).any(|(offset, word)| {
                let head = source.span(start + offset, start + offset + word.len());
                // A bare observation word after 的 can be a noun, not an earlier action.
                let Some(prior) = observer(source, lexicon, previous, start + offset, &['地'])
                else {
                    return false;
                };
                let tail = trim_clause_end(&text[offset + word.len()..]).trim();
                if tail.is_empty()
                    || tail.starts_with(['的', '者'])
                    || candidate_scope(source, scope, &ClauseCandidate::new(head)).0
                        != ClauseScope::Narrated
                    || inside_lexical_term(source, lexicon, previous.evidence, head)
                {
                    return false;
                }
                current
                    .surface
                    .is_none_or(|current_surface| match (current.actor, prior.actor) {
                        (Some(current), Some(prior)) => current == prior,
                        _ => prior.surface.is_some_and(|prior_surface| {
                            source.slice(&current_surface) == source.slice(&prior_surface)
                        }),
                    })
            })
        })
}

fn dependent_sequel(source: &Source<'_>, lexicon: &Lexicon<'_>, unit: &Unit, next: &Unit) -> bool {
    if next.sentence != unit.sentence {
        return false;
    }
    let text = source.slice(&next.evidence);
    let lead = continuation(text).unwrap_or(text).trim_start();
    if scene_transition(lead) || starts_any(lead, DEPENDENT_NEXT) {
        return true;
    }
    let end = source.bytes[next.evidence.end_char];
    let start = end - lead.len();
    let subject = source
        .first_name(start, end, lexicon)
        .and_then(|(offset, name, _)| (offset == 0).then_some(name))
        .or_else(|| {
            PRONOUNS
                .iter()
                .copied()
                .find(|pronoun| lead.starts_with(pronoun))
        });
    subject.is_some_and(|name| starts_any(lead[name.len()..].trim_start(), DEPENDENT_NEXT))
}

pub(in crate::analyze) fn discover(
    analyzer: &NarrativeAnalyzer<'_>,
    chapter_text: &str,
    title: &str,
) -> Vec<DurationObservation> {
    if !chapter_text
        .match_indices('了')
        .any(|(offset, marker)| starts_any(&chapter_text[offset + marker.len()..], DURATIONS))
    {
        return Vec::new();
    }
    let mut source = Source::new(chapter_text);
    let lexicon = &analyzer.lexicon;
    let units = split_units(&mut source, title, lexicon, Some(analyzer));
    let scopes = scopes_for_units(&source, &units, true);
    let mut observations = Vec::new();
    for (index, (unit, scope)) in units.iter().zip(&scopes).enumerate() {
        let body = trim_clause_end(source.slice(&unit.evidence));
        let Some((lead, duration)) = DURATIONS.iter().find_map(|&duration| {
            Some((body.strip_suffix(duration)?.strip_suffix('了')?, duration))
        }) else {
            continue;
        };
        let start = source.bytes[unit.evidence.start_char];
        let marker = start + lead.len();
        let duration_start = marker + '了'.len_utf8();
        let previous = index
            .checked_sub(1)
            .map(|prior| (&units[prior], &scopes[prior]));
        if nonactual_modifier(&source, lexicon, unit, marker, false).is_some()
            || previous.is_some_and(|(earlier, _)| {
                let text = source.slice(&earlier.evidence);
                !unit.first_in_line
                    && earlier.sentence == unit.sentence
                    && text.ends_with(['，', ','])
                    && !source.text[source.bytes[earlier.evidence.start_char]..start].contains(
                        |ch| matches!(ch, '\n'..='\r' | '\u{85}' | '\u{2028}' | '\u{2029}'),
                    )
                    && nonactual_modifier(
                        &source,
                        lexicon,
                        earlier,
                        source.bytes[earlier.evidence.start_char] + trim_clause_end(text).len(),
                        true,
                    )
                    .is_some()
            })
        {
            continue;
        }
        // Try the whole verb first, but require it to fit the written observer lead.
        let Some((predicate, observer)) = OBSERVATION_VERBS.iter().find_map(|verb| {
            let head_start = start + lead.strip_suffix(verb)?.len();
            let observer = observer(&source, lexicon, unit, head_start, &['地', '的'])?;
            Some((source.span(head_start, marker), observer))
        }) else {
            continue;
        };
        if !asserted_unit(&source, unit, scope)
            || units
                .get(index + 1)
                .is_some_and(|next| dependent_sequel(&source, lexicon, unit, next))
        {
            continue;
        }
        let candidate = ClauseCandidate {
            actor: observer.surface,
            ..ClauseCandidate::new(predicate)
        };
        if candidate_scope(&source, scope, &candidate).0 != ClauseScope::Narrated
            || inside_lexical_term(&source, lexicon, unit.evidence, predicate)
            || previous.is_some_and(|(earlier, earlier_scope)| {
                let plan = earlier.sentence == unit.sentence
                    && ["打算", "計畫", "计划", "原想", "原定", "準備", "准备"]
                        .iter()
                        .any(|word| {
                            trim_clause_end(source.slice(&earlier.evidence)).ends_with(word)
                        });
                plan || continued_observation(
                    &source,
                    lexicon,
                    unit,
                    &observer,
                    earlier,
                    earlier_scope,
                )
            })
        {
            continue;
        }
        observations.push(DurationObservation {
            clause: unit.evidence,
            predicate,
            completion: source.span(marker, duration_start),
            duration: source.span(duration_start, duration_start + duration.len()),
            actor: observer.actor.map(str::to_owned),
            actor_surface: observer.surface,
            object: None,
        });
    }
    observations
}
