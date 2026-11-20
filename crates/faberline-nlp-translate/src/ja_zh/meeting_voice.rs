//! Source evidence for a meeting counterpart in an ambiguous polite negative.

use super::{
    Atom, Bunsetsu, BunsetsuRole, Document, LOCKED_OBJECT_ATOM, LOCKED_PLACE_ATOM, SemanticClass,
    honorific, is_companion_person, is_honorific_suffix, is_time_or_place_bunsetsu,
};
use faberline_nlp::doc::{EntityKind, Upos};

/// Return the one `に` counterpart group only when all links point to the same
/// unlocked `会う` predicate. The frame's inferred ARG0 is not evidence here.
pub(super) fn counterpart(
    groups: &[Bunsetsu],
    source: &[Atom],
    doc: &Document,
    chars: &[char],
) -> Option<usize> {
    if source
        .iter()
        .any(|a| matches!(a.surface.as_str(), "「" | "」" | "『" | "』"))
    {
        return None;
    }
    let mut chains = source.windows(4).filter(|tokens| {
        let [verb, voice, polite, negative] = *tokens else {
            return false;
        };
        !tokens.iter().any(|a| a.is_locked)
            && verb.upos == Some(Upos::Verb)
            && verb.lemma.as_deref() == Some("会う")
            && verb.deprel == "root"
            && verb.doc_tok_idx.is_some()
            && matches!(
                (voice.surface.as_str(), voice.lemma.as_deref()),
                ("れ", Some("れる")) | ("られ", Some("られる"))
            )
            && polite.surface == "ませ"
            && polite.lemma.as_deref() == Some("ます")
            && negative.surface == "ん"
            && [voice, polite, negative].iter().all(|a| {
                a.upos == Some(Upos::Aux) && a.deprel == "aux" && a.head == verb.doc_tok_idx
            })
            && tokens
                .windows(2)
                .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
    });
    let chain = chains.next()?;
    if chains.next().is_some() {
        return None;
    }
    let verb = &chain[0];
    let head = verb.doc_tok_idx?;
    // A second verb or subject can indicate a separate event or affected
    // participant. Leave such cases to the existing translation path.
    if source.iter().filter(|a| a.upos == Some(Upos::Verb)).count() != 1
        || doc
            .syntax
            .iter()
            .filter(|edge| edge.deprel == "nsubj" && edge.head == Some(head))
            .count()
            != 1
    {
        return None;
    }

    // Normalization can expose a source-linked marker not retained as one
    // original token. It must not introduce a second role or explicit agent.
    let visible_cases: Vec<_> = source
        .iter()
        .filter(|case| {
            !case.is_locked
                && case.deprel == "case"
                && matches!(
                    case.surface.as_str(),
                    "に" | "と" | "によって" | "により" | "から"
                )
                && case.head.is_some_and(|id| {
                    source
                        .iter()
                        .any(|noun| noun.doc_tok_idx == Some(id) && noun.head == Some(head))
                })
        })
        .collect();
    if visible_cases.len() != 1 || visible_cases[0].surface != "に" {
        return None;
    }

    // Inspect original Document edges, including cases hidden by a glossary
    // lock. A hidden or competing case is not evidence of a unique partner.
    let mut linked_cases = 0;
    for case in doc.syntax.iter().filter(|edge| edge.deprel == "case") {
        let Some(noun) = case.head else { continue };
        if !doc
            .syntax
            .iter()
            .any(|edge| edge.token == noun && edge.head == Some(head))
        {
            continue;
        }
        let token = doc.tokens.get(case.token)?;
        let marker: String = chars
            .get(token.span.start_char..token.span.end_char)?
            .iter()
            .collect();
        if source.iter().any(|atom| {
            atom.is_locked
                && atom.span.start_char < token.span.end_char
                && token.span.start_char < atom.span.end_char
        }) {
            return None;
        }
        // A compound case such as に由って is not a plain counterpart.
        if marker == "から"
            || ((marker.starts_with('に') || marker.starts_with('と'))
                && !matches!(marker.as_str(), "に" | "と"))
        {
            return None;
        }
        if matches!(marker.as_str(), "に" | "と") {
            linked_cases += 1;
        }
    }
    if linked_cases != 1 {
        return None;
    }

    let mut partners = source.iter().filter_map(|case| {
        if case.is_locked
            || case.upos != Some(Upos::Adp)
            || case.deprel != "case"
            || case.doc_tok_idx.is_none()
            || case.surface != "に"
        {
            return None;
        }
        let id = case.head?;
        let noun = source.iter().find(|a| a.doc_tok_idx == Some(id))?;
        let original = doc.tokens.get(id)?;
        let original_case = doc.tokens.get(case.doc_tok_idx?)?;
        if noun.head != Some(head)
            || !matches!(noun.deprel.as_str(), "obl" | "iobj")
            || noun.span.end_char != case.span.start_char
            // Intervening text may contain a locked agent marker. Until it
            // can be analyzed, do not consume the argument or alter voice.
            || case.span.end_char != verb.span.start_char
            || original.span.start_char < noun.span.start_char
            || original.span.end_char != noun.span.end_char
            || original_case.span != case.span
            || (!noun.is_locked
                && !matches!(noun.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron)))
        {
            return None;
        }
        let mut matching = groups.iter().enumerate().filter(|(_, group)| {
            !matches!(group.role, BunsetsuRole::Subj | BunsetsuRole::Predicate)
                && group.case_particle.as_deref() == Some(case.surface.as_str())
                && group.atoms.iter().any(|a| a.doc_tok_idx == Some(id))
                && group
                    .atoms
                    .iter()
                    .any(|a| a.doc_tok_idx == case.doc_tok_idx && a.span == case.span)
                && !group
                    .atoms
                    .iter()
                    .any(|a| matches!(a.deprel.as_str(), "nsubj" | "obj" | "root" | "advcl"))
                && !is_time_or_place_bunsetsu(&group.atoms)
                && person_evidence(noun, group, source, doc, chars)
        });
        let (index, _) = matching.next()?;
        matching.next().is_none().then_some(index)
    });
    let index = partners.next()?;
    partners.next().is_none().then_some(index)
}

/// Keep a recognized multi-token name together when grouping split off one of
/// its source-linked compound tokens. Never consume a neighboring phrase.
pub(super) fn expand_name(
    groups: &[Bunsetsu],
    index: usize,
    doc: &Document,
) -> Option<(std::ops::RangeInclusive<usize>, Bunsetsu)> {
    let group = groups.get(index)?;
    let case = group
        .atoms
        .iter()
        .find(|a| !a.is_locked && a.surface == "に" && a.deprel == "case" && a.head.is_some())?;
    let head = case.head?;
    let noun = group.atoms.iter().find(|a| a.doc_tok_idx == Some(head))?;
    let mut entities = doc.entities.iter().filter(|entity| {
        entity.kind == EntityKind::Name
            && entity.span.start_char <= noun.span.start_char
            && entity.span.end_char == noun.span.end_char
    });
    let Some(name) = entities.next() else {
        return Some((index..=index, group.clone()));
    };
    if entities.next().is_some() {
        return None;
    }
    let mut first = index;
    let mut start = group.atoms.first()?.span.start_char;
    while start > name.span.start_char {
        let previous = groups.get(first.checked_sub(1)?)?;
        if previous.role != BunsetsuRole::Other
            || previous.case_particle.is_some()
            || previous.atoms.is_empty()
            || previous.atoms.last()?.span.end_char != start
            || !previous.atoms.iter().all(|atom| {
                !atom.is_locked
                    && atom.doc_tok_idx.is_some()
                    && atom.deprel == "compound"
                    && atom.head == Some(head)
                    && atom.span.start_char >= name.span.start_char
                    && atom.span.end_char <= name.span.end_char
            })
        {
            return None;
        }
        first -= 1;
        start = previous.atoms.first()?.span.start_char;
    }
    if start != name.span.start_char {
        return None;
    }
    let mut expanded = group.clone();
    expanded.atoms = groups[first..=index]
        .iter()
        .flat_map(|part| part.atoms.iter().cloned())
        .collect();
    Some((first..=index, expanded))
}

/// Use context-specific wording only for an unlocked, proved counterpart.
pub(super) fn localized_counterpart(
    group: &Bunsetsu,
    source: &[Atom],
    doc: &Document,
    chars: &[char],
) -> Option<&'static str> {
    let case = group
        .atoms
        .iter()
        .find(|atom| !atom.is_locked && atom.surface == "に" && atom.deprel == "case")?;
    let mut start = group.atoms.first()?.span.start_char;
    let end = case.span.start_char;
    let noun = case.head?;
    let predicate = doc.syntax.iter().find(|edge| edge.token == noun)?.head?;
    let raw: String = chars.get(start..end)?.iter().collect();
    if raw == "客様"
        && start > 0
        && chars.get(start - 1) == Some(&'お')
        && source.iter().any(|atom| {
            !atom.is_locked
                && atom.surface == "お"
                && atom.span.start_char == start - 1
                && atom.span.end_char == start
                && atom.head == Some(predicate)
        })
    {
        start -= 1;
    }
    let raw: String = chars.get(start..end)?.iter().collect();
    if source
        .iter()
        .any(|atom| atom.is_locked && atom.span.start_char < end && start < atom.span.end_char)
    {
        return None;
    }
    if raw == "お客様" {
        return Some("客人");
    }
    if raw != "保護者" {
        return None;
    }
    let subject = doc
        .syntax
        .iter()
        .find(|edge| edge.deprel == "nsubj" && edge.head == Some(predicate))?;
    let span = doc.tokens.get(subject.token)?.span;
    let subject: String = chars.get(span.start_char..span.end_char)?.iter().collect();
    (subject == "校長").then_some("家長")
}

fn person_evidence(
    noun: &Atom,
    group: &Bunsetsu,
    source: &[Atom],
    doc: &Document,
    chars: &[char],
) -> bool {
    let named_span = doc
        .entities
        .iter()
        .any(|entity| entity.kind == EntityKind::Name && entity.span == noun.span);
    let person_token = doc.tokens.iter().any(|token| {
        token.span.start_char >= noun.span.start_char
            && token.span.end_char <= noun.span.end_char
            && token.sem_class == Some(SemanticClass::Person)
    });
    if noun.is_locked {
        if matches!(
            noun.lemma.as_deref(),
            Some(LOCKED_PLACE_ATOM | LOCKED_OBJECT_ATOM)
        ) {
            return false;
        }
        if person_token || honorific::is_bound_person(noun) {
            return true;
        }
        // A DomainTerm lock may merge a recognized name into one token and
        // lose its Person class. Require an exact, independently recognized
        // name plus a nonempty stem and a source honorific.
        let Some(name) = chars.get(noun.span.start_char..noun.span.end_char) else {
            return false;
        };
        let name: String = name.iter().collect();
        return named_span
            && !doc.options.names.iter().any(|provided| provided == &name)
            && ["さん", "様", "さま", "先生", "先輩", "ちゃん", "くん"]
                .iter()
                .any(|suffix| {
                    name.strip_suffix(suffix)
                        .is_some_and(|stem| !stem.is_empty())
                });
    }
    named_span && is_honorific_suffix(&noun.surface)
        || person_token
        || group
            .atoms
            .iter()
            .any(|atom| !atom.is_locked && is_companion_person(atom))
        || matches!(noun.surface.as_str(), "様" | "さま" | "者")
            && source.iter().any(|part| {
                !part.is_locked
                    && part.deprel == "compound"
                    && part.head == noun.doc_tok_idx
                    && part.span.end_char == noun.span.start_char
            })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faberline_nlp::Span;

    fn meeting() -> (Vec<Atom>, Vec<Bunsetsu>, Document) {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/meeting-voice-documents.json"
        ))
        .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["id"] == "P01")
            .unwrap();
        let source = case["source"].as_str().unwrap();
        let chars: Vec<_> = source.chars().collect();
        let doc: Document = serde_json::from_value(case["documents"][1]["doc"].clone()).unwrap();
        let atoms: Vec<_> = doc
            .tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let edge = doc.syntax.iter().find(|edge| edge.token == index).unwrap();
                Atom {
                    span: token.span,
                    surface: chars[token.span.start_char..token.span.end_char]
                        .iter()
                        .collect(),
                    lemma: token.lemma.clone(),
                    upos: token.upos,
                    doc_tok_idx: Some(index),
                    head: edge.head,
                    deprel: edge.deprel.clone(),
                    is_locked: false,
                }
            })
            .collect();
        let groups = vec![Bunsetsu {
            atoms: atoms[2..6].to_vec(),
            role: BunsetsuRole::IObj,
            case_particle: Some("に".to_owned()),
            sconj_tail: None,
            sfp_tail: None,
        }];
        (atoms, groups, doc)
    }

    #[test]
    fn localized_counterpart_respects_source_context_and_locks() {
        let (source, groups, doc) = meeting();
        let chars: Vec<_> = "先生はお客様に会われません。".chars().collect();
        assert_eq!(
            localized_counterpart(&groups[0], &source, &doc, &chars),
            Some("客人")
        );
        let mut split_prefix = groups[0].clone();
        split_prefix.atoms.remove(0);
        assert_eq!(
            localized_counterpart(&split_prefix, &source, &doc, &chars),
            Some("客人")
        );
        for index in 2..=4 {
            let mut locked = source.clone();
            locked[index].is_locked = true;
            assert_eq!(
                localized_counterpart(&groups[0], &locked, &doc, &chars),
                None,
                "locked source token {index}"
            );
            assert_eq!(
                localized_counterpart(&split_prefix, &locked, &doc, &chars),
                None,
                "locked source token {index} with split prefix"
            );
        }

        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/meeting-voice-documents.json"
        ))
        .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["id"] == "M04")
            .unwrap();
        let chars: Vec<_> = case["source"].as_str().unwrap().chars().collect();
        let doc: Document = serde_json::from_value(case["documents"][1]["doc"].clone()).unwrap();
        let source: Vec<_> = doc
            .tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let edge = doc.syntax.iter().find(|edge| edge.token == index).unwrap();
                Atom {
                    span: token.span,
                    surface: chars[token.span.start_char..token.span.end_char]
                        .iter()
                        .collect(),
                    lemma: token.lemma.clone(),
                    upos: token.upos,
                    doc_tok_idx: Some(index),
                    head: edge.head,
                    deprel: edge.deprel.clone(),
                    is_locked: false,
                }
            })
            .collect();
        let group = Bunsetsu {
            atoms: source[4..7].to_vec(),
            role: BunsetsuRole::IObj,
            case_particle: Some("に".to_owned()),
            sconj_tail: None,
            sfp_tail: None,
        };
        assert_eq!(
            localized_counterpart(&group, &source, &doc, &chars),
            Some("家長")
        );
        let mut non_school = chars.clone();
        non_school[0] = '部';
        assert_eq!(
            localized_counterpart(&group, &source, &doc, &non_school),
            None
        );
        for index in 4..=5 {
            let mut locked = source.clone();
            locked[index].is_locked = true;
            assert_eq!(
                localized_counterpart(&group, &locked, &doc, &chars),
                None,
                "locked source token {index}"
            );
        }
    }

    #[test]
    fn meeting_requires_unlocked_same_head_auxiliaries_and_subject() {
        let (source, groups, doc) = meeting();
        let chars: Vec<_> = "先生はお客様に会われません。".chars().collect();
        assert_eq!(counterpart(&groups, &source, &doc, &chars), Some(0));
        for index in 7..=9 {
            let mut locked = source.clone();
            locked[index].is_locked = true;
            assert_eq!(counterpart(&groups, &locked, &doc, &chars), None);
            let mut other_head = source.clone();
            other_head[index].head = None;
            assert_eq!(counterpart(&groups, &other_head, &doc, &chars), None);
        }
        let mut separated = source.clone();
        separated[8].span.start_char += 1;
        assert_eq!(counterpart(&groups, &separated, &doc, &chars), None);
        let mut no_subject = doc.clone();
        no_subject.syntax[0].head = None;
        assert_eq!(counterpart(&groups, &source, &no_subject, &chars), None);
    }

    #[test]
    fn meeting_rejects_an_unproven_person_agent_or_competing_case() {
        let (source, groups, doc) = meeting();
        let chars: Vec<_> = "先生はお客様に会われません。".chars().collect();
        let mut unknown_source = source.clone();
        let mut unknown_group = groups.clone();
        unknown_source[4].surface = "机".to_owned();
        unknown_group[0].atoms[2].surface = "机".to_owned();
        assert_eq!(
            counterpart(&unknown_group, &unknown_source, &doc, &chars),
            None
        );
        let mut locked_source = source.clone();
        let mut locked_group = groups.clone();
        locked_source[4].is_locked = true;
        locked_group[0].atoms[2].is_locked = true;
        assert_eq!(
            counterpart(&locked_group, &locked_source, &doc, &chars),
            None
        );

        let mut explicit_agent = source.clone();
        let mut marker = source[5].clone();
        marker.surface = "によって".to_owned();
        explicit_agent.push(marker);
        assert_eq!(counterpart(&groups, &explicit_agent, &doc, &chars), None);

        let mut competing = source.clone();
        let mut marker = source[1].clone();
        marker.surface = "と".to_owned();
        competing.push(marker);
        assert_eq!(counterpart(&groups, &competing, &doc, &chars), None);

        let mut quoted = source.clone();
        let mut marker = source[1].clone();
        marker.surface = "「".to_owned();
        quoted.push(marker);
        assert_eq!(counterpart(&groups, &quoted, &doc, &chars), None);
    }

    #[test]
    fn a_glossary_lock_cannot_hide_a_competing_source_case() {
        for (suffix, case_surface) in [
            ("と", "と"),
            ("によって", "によって"),
            ("によって", "に"),
            ("に依って", "に依って"),
            ("に依って", "に"),
            ("に因って", "に"),
            ("に依り", "に"),
            ("に由って", "に由って"),
            ("に由り", "に由り"),
            ("に拠って", "に拠って"),
        ] {
            let (source, groups, mut doc) = meeting();
            let mut chars: Vec<_> = "先生はお客様に会われません。".chars().collect();
            let start = chars.len();
            chars.extend(format!("同僚{suffix}").chars());
            let noun_id = doc.tokens.len();
            let case_id = noun_id + 1;
            let mut noun_token = doc.tokens[4].clone();
            noun_token.span = Span {
                start_char: start,
                end_char: start + 2,
            };
            let mut case_token = doc.tokens[5].clone();
            case_token.span = Span {
                start_char: start + 2,
                end_char: start + 2 + case_surface.chars().count(),
            };
            doc.tokens.extend([noun_token.clone(), case_token.clone()]);
            let mut noun_edge = doc.syntax[4].clone();
            noun_edge.token = noun_id;
            noun_edge.head = Some(6);
            noun_edge.span = noun_token.span;
            let mut case_edge = doc.syntax[5].clone();
            case_edge.token = case_id;
            case_edge.head = Some(noun_id);
            case_edge.span = case_token.span;
            doc.syntax.extend([noun_edge, case_edge]);

            let mut visible = source.clone();
            let mut noun = source[4].clone();
            noun.surface = "同僚".to_owned();
            noun.span = noun_token.span;
            noun.doc_tok_idx = Some(noun_id);
            noun.head = Some(6);
            let mut marker = source[5].clone();
            marker.surface = case_surface.to_owned();
            marker.span = case_token.span;
            marker.doc_tok_idx = Some(case_id);
            marker.head = Some(noun_id);
            visible.extend([noun, marker.clone()]);
            assert_eq!(
                counterpart(&groups, &visible, &doc, &chars),
                None,
                "{suffix}/{case_surface}"
            );

            let mut locked = source.clone();
            marker.surface = "[[PH999]]".to_owned();
            marker.span = Span {
                start_char: start,
                end_char: chars.len(),
            };
            marker.is_locked = true;
            locked.push(marker);
            assert_eq!(
                counterpart(&groups, &locked, &doc, &chars),
                None,
                "{suffix}/{case_surface}"
            );
        }
    }

    #[test]
    fn locked_agent_suffix_after_visible_ni_is_still_detected() {
        for suffix in ["依って", "由って", "拠って"] {
            let (mut source, groups, mut doc) = meeting();
            let mut chars: Vec<_> = "先生はお客様に会われません。".chars().collect();
            let start = 7;
            chars.splice(start..start, suffix.chars());
            for token in &mut doc.tokens {
                if token.span.start_char >= start {
                    token.span.start_char += 3;
                    token.span.end_char += 3;
                }
            }
            for edge in &mut doc.syntax {
                if edge.span.start_char >= start {
                    edge.span.start_char += 3;
                    edge.span.end_char += 3;
                }
            }
            for atom in &mut source {
                if atom.span.start_char >= start {
                    atom.span.start_char += 3;
                    atom.span.end_char += 3;
                }
            }
            let mut locked = source[2].clone();
            locked.span = Span {
                start_char: start,
                end_char: start + 3,
            };
            locked.surface = "[[PH998]]".to_owned();
            locked.upos = Some(Upos::Noun);
            locked.doc_tok_idx = None;
            locked.head = None;
            locked.deprel.clear();
            locked.is_locked = true;
            source.insert(6, locked);
            assert_eq!(
                counterpart(&groups, &source, &doc, &chars),
                None,
                "{suffix}"
            );
        }
    }

    #[test]
    fn a_locked_name_requires_original_name_and_honorific_evidence() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/meeting-voice-documents.json"
        ))
        .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["id"] == "M08")
            .unwrap();
        let chars: Vec<_> = case["source"].as_str().unwrap().chars().collect();
        let doc: Document = serde_json::from_value(case["documents"][2]["doc"].clone()).unwrap();
        let mut source: Vec<_> = doc
            .tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let edge = doc.syntax.iter().find(|edge| edge.token == index).unwrap();
                Atom {
                    span: token.span,
                    surface: chars[token.span.start_char..token.span.end_char]
                        .iter()
                        .collect(),
                    lemma: token.lemma.clone(),
                    upos: token.upos,
                    doc_tok_idx: Some(index),
                    head: edge.head,
                    deprel: edge.deprel.clone(),
                    is_locked: false,
                }
            })
            .collect();
        source[5].surface = "[[PH998]]".to_owned();
        source[5].is_locked = true;
        let groups = vec![Bunsetsu {
            atoms: source[5..7].to_vec(),
            role: BunsetsuRole::IObj,
            case_particle: Some("に".to_owned()),
            sconj_tail: None,
            sfp_tail: None,
        }];
        assert_eq!(counterpart(&groups, &source, &doc, &chars), Some(0));

        let mut without_name = doc.clone();
        without_name
            .entities
            .retain(|entity| entity.span != source[5].span);
        assert_eq!(counterpart(&groups, &source, &without_name, &chars), None);
        let mut caller_name = doc.clone();
        caller_name.options.names.push("森川さん".to_owned());
        assert_eq!(counterpart(&groups, &source, &caller_name, &chars), None);
        let mut without_suffix = chars.clone();
        without_suffix[11] = '記';
        without_suffix[12] = '号';
        assert_eq!(counterpart(&groups, &source, &doc, &without_suffix), None);
        for marker in [
            super::super::LOCKED_PLACE_ATOM,
            super::super::LOCKED_OBJECT_ATOM,
        ] {
            let mut other_lock = source.clone();
            other_lock[5].lemma = Some(marker.to_owned());
            assert_eq!(counterpart(&groups, &other_lock, &doc, &chars), None);
        }
    }

    #[test]
    fn a_locked_name_without_an_honorific_is_not_person_evidence() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/meeting-voice-documents.json"
        ))
        .unwrap();
        let case = &fixture["negative_controls"][0];
        assert_eq!(case["id"], "M08-no-honorific");
        assert_eq!(case["glossary"]["domain_terms"][0]["canonical"], "森川");
        let chars: Vec<_> = case["source"].as_str().unwrap().chars().collect();
        let doc: Document = serde_json::from_value(case["doc"].clone()).unwrap();
        let mut source: Vec<_> = doc
            .tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let edge = doc.syntax.iter().find(|edge| edge.token == index).unwrap();
                Atom {
                    span: token.span,
                    surface: chars[token.span.start_char..token.span.end_char]
                        .iter()
                        .collect(),
                    lemma: token.lemma.clone(),
                    upos: token.upos,
                    doc_tok_idx: Some(index),
                    head: edge.head,
                    deprel: edge.deprel.clone(),
                    is_locked: false,
                }
            })
            .collect();
        assert_eq!(source[5].surface, "森川");
        assert!(
            doc.entities
                .iter()
                .any(|entity| entity.kind == EntityKind::Name && entity.span == source[5].span)
        );
        assert_ne!(doc.tokens[5].sem_class, Some(SemanticClass::Person));
        assert!(!doc.options.names.iter().any(|name| name == "森川"));
        source[3].surface = "[[PH-place]]".to_owned();
        source[3].lemma = Some(LOCKED_PLACE_ATOM.to_owned());
        source[3].is_locked = true;
        source[5].surface = "[[PH-name]]".to_owned();
        source[5].lemma = None;
        source[5].is_locked = true;
        let groups = vec![Bunsetsu {
            atoms: source[5..7].to_vec(),
            role: BunsetsuRole::IObj,
            case_particle: Some("に".to_owned()),
            sconj_tail: None,
            sfp_tail: None,
        }];
        assert_eq!(counterpart(&groups, &source, &doc, &chars), None);
    }

    #[test]
    fn a_split_name_is_reassembled_only_from_its_compound_group() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/meeting-voice-documents.json"
        ))
        .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["id"] == "M08")
            .unwrap();
        let chars: Vec<_> = case["source"].as_str().unwrap().chars().collect();
        let doc: Document = serde_json::from_value(case["documents"][1]["doc"].clone()).unwrap();
        let atoms: Vec<_> = doc
            .tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                let edge = doc.syntax.iter().find(|edge| edge.token == index).unwrap();
                Atom {
                    span: token.span,
                    surface: chars[token.span.start_char..token.span.end_char]
                        .iter()
                        .collect(),
                    lemma: token.lemma.clone(),
                    upos: token.upos,
                    doc_tok_idx: Some(index),
                    head: edge.head,
                    deprel: edge.deprel.clone(),
                    is_locked: false,
                }
            })
            .collect();
        let make_group = |atoms: Vec<Atom>, role, case_particle| Bunsetsu {
            atoms,
            role,
            case_particle,
            sconj_tail: None,
            sfp_tail: None,
        };
        let groups = vec![
            make_group(vec![atoms[7].clone()], BunsetsuRole::Other, None),
            make_group(
                atoms[8..11].to_vec(),
                BunsetsuRole::IObj,
                Some("に".to_owned()),
            ),
        ];
        let (range, combined) = expand_name(&groups, 1, &doc).unwrap();
        assert_eq!(range, 0..=1);
        assert_eq!(
            combined
                .atoms
                .iter()
                .map(|a| a.surface.as_str())
                .collect::<String>(),
            "森川さんに"
        );

        let mut other_head = groups.clone();
        other_head[0].atoms[0].head = Some(5);
        assert!(expand_name(&other_head, 1, &doc).is_none());
        let mut locked = groups.clone();
        locked[0].atoms[0].is_locked = true;
        assert!(expand_name(&locked, 1, &doc).is_none());
        let mut subject = groups.clone();
        subject[0].role = BunsetsuRole::Subj;
        assert!(expand_name(&subject, 1, &doc).is_none());
    }
}
