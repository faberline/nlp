//! Distinguish a household destination from a linked family role.

use faberline_nlp::doc::{Document, SyntaxToken, Upos};

const FAMILY_ROLES: &[&str] = &["婿", "嫁", "養子", "養女", "婿養子", "家族"];
const HOME_ROLES: &[&str] = &["主婦", "主夫", "専業主婦", "専業主夫"];
const DOORS: &[&str] = &["戸", "扉", "ドア"];

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum EntryUse {
    NativeWording,
    CallerDestination,
}

pub(super) fn allows_physical_entry(
    doc: &Document,
    source: &[char],
    entry: usize,
    usage: EntryUse,
) -> bool {
    let Some(token) = doc.tokens.get(entry) else {
        return false;
    };
    let Some(sentence) = doc.segmentation.sentences.get(token.sentence) else {
        return false;
    };
    if source
        .get(sentence.span.start_char..sentence.span.end_char)
        .is_none()
    {
        return false;
    }
    let start = doc
        .tokens
        .partition_point(|token| token.span.start_char < sentence.span.start_char);
    let end = doc
        .tokens
        .partition_point(|token| token.span.start_char < sentence.span.end_char);
    let Some(links) = doc.syntax.get(start..end) else {
        return false;
    };
    let Some(entry_link) = doc.syntax.get(entry).filter(|link| {
        link.token == entry && link.span == token.span && link.sentence == token.sentence
    }) else {
        return false;
    };
    let children = |head, relation: &'static str| {
        links
            .iter()
            .filter(move |link| link.head == Some(head) && link.deprel == relation)
    };
    let surface_is = |link: &SyntaxToken, spelling: &str| {
        source
            .get(link.span.start_char..link.span.end_char)
            .is_some_and(|chars| chars.iter().copied().eq(spelling.chars()))
    };
    let form_is = |link: &SyntaxToken, lemma: &str| {
        doc.tokens[link.token].lemma.as_deref() == Some(lemma) || surface_is(link, lemma)
    };
    let has_case = |noun: &SyntaxToken, particle| {
        children(noun.token, "case")
            .any(|case| noun.span.end_char == case.span.start_char && surface_is(case, particle))
    };
    let has_argument = |verb, relation, spellings: &[&str], particle| {
        children(verb, relation).any(|noun| {
            doc.tokens[noun.token].upos == Some(Upos::Noun)
                && spellings.iter().any(|word| surface_is(noun, word))
                && has_case(noun, particle)
        })
    };
    if links.iter().enumerate().any(|(offset, link)| {
        link.token != start + offset
            || link.sentence != token.sentence
            || link.span != doc.tokens[link.token].span
    }) {
        return false;
    }

    if has_argument(entry, "obl", FAMILY_ROLES, "として") {
        return false;
    }
    // Adoption goes with the preceding reception predicate. An active
    // reception can describe a different actor's action before going inside.
    if children(entry, "advcl").any(|prior| {
        prior.token < entry
            && form_is(prior, "迎える")
            && has_argument(prior.token, "obl", FAMILY_ROLES, "に")
            && children(prior.token, "aux").any(|suffix| form_is(suffix, "られる"))
    }) {
        return false;
    }

    // Do not infer a shared actor when each action has an explicit subject.
    let can_share_subject = |left, right| {
        children(left, "nsubj").next().is_none() || children(right, "nsubj").next().is_none()
    };
    let can_share_actor = |other| can_share_subject(entry, other);
    let has_te_kara = |verb| {
        children(verb, "mark").any(|mark| surface_is(mark, "て"))
            && children(verb, "case").any(|case| surface_is(case, "から"))
    };
    let result = entry_link
        .head
        .filter(|head| {
            *head > entry
                && *head < end
                && (entry_link.deprel == "advcl"
                    || entry_link.deprel == "obl" && has_te_kara(entry))
        })
        .and_then(|head| doc.syntax.get(head));

    let family_succession = result.is_some_and(|next| {
        let inherits = form_is(next, "継ぐ")
            || form_is(next, "引き継ぐ")
            // The morphology can read the past form's だ as a copula.
            || (surface_is(next, "継い") || surface_is(next, "引き継い"))
                && ["aux", "cop"].into_iter().any(|relation| {
                    children(next.token, relation).any(|suffix| {
                        next.span.end_char == suffix.span.start_char && surface_is(suffix, "だ")
                    })
                });
        can_share_actor(next.token)
            && inherits
            && children(next.token, "obj").any(|noun| {
                has_case(noun, "を")
                    && (surface_is(noun, "家業")
                        || surface_is(noun, "業")
                            && children(noun.token, "compound").any(|prefix| {
                                prefix.span.end_char == noun.span.start_char
                                    && surface_is(prefix, "家")
                            }))
            })
    });
    // A caller can supply the complete destination. The uncertain reading
    // only limits the idiom chosen by the engine for an unlocked noun.
    let mut prior_heads = vec![entry];
    if usage == EntryUse::NativeWording {
        // The parser can attach an earlier marriage clause to an omitted
        // action instead of to 入る. Keep that context when rendering ないで
        // as a separate clause. Only follow original, attached omission links.
        let mut next = 0;
        while let Some(&head) = prior_heads.get(next) {
            prior_heads.extend(
                children(head, "advcl")
                    .filter(|prior| {
                        prior.token < head
                            && can_share_actor(prior.token)
                            && can_share_subject(head, prior.token)
                            && doc.tokens[prior.token].upos == Some(Upos::Verb)
                            && children(prior.token, "aux").any(|negative| {
                                prior.span.end_char == negative.span.start_char
                                    && surface_is(negative, "ない")
                                    && children(prior.token, "mark").any(|marker| {
                                        negative.span.end_char == marker.span.start_char
                                            && surface_is(marker, "で")
                                    })
                            })
                    })
                    .map(|prior| prior.token),
            );
            next += 1;
        }
    }
    let linked_marriages = || {
        prior_heads.iter().flat_map(|&head| {
            children(head, "advcl")
                .chain(children(head, "obl").filter(|prior| has_te_kara(prior.token)))
                .filter(move |prior| {
                    prior.token < head
                        && can_share_actor(prior.token)
                        && can_share_subject(head, prior.token)
                        && form_is(prior, "結婚する")
                })
        })
    };
    if usage == EntryUse::NativeWording
        && (family_succession || linked_marriages().next().is_some())
    {
        let actor_agrees = |action| {
            can_share_actor(action)
                && children(action, "nsubj").all(|actor| {
                    linked_marriages()
                        .flat_map(|prior| children(prior.token, "nsubj"))
                        .all(|prior_actor| {
                            // Same-sentence first person has a fixed referent.
                            ["私", "僕", "俺"].iter().any(|name| {
                                surface_is(actor, name) && surface_is(prior_actor, name)
                            })
                        })
                })
        };
        let action_is_asserted = |action| {
            children(action, "aux").all(|suffix| {
                ["た", "だ", "ます", "まし"]
                    .iter()
                    .any(|form| surface_is(suffix, form))
            }) && !children(action, "discourse").any(|particle| surface_is(particle, "か"))
                && !children(action, "mark").any(|marker| {
                    ["か", "なら", "ば", "たら", "と"]
                        .iter()
                        .any(|form| surface_is(marker, form))
                })
                && !children(action, "punct")
                    .any(|punct| surface_is(punct, "？") || surface_is(punct, "?"))
        };
        // A linked, asserted door or threshold action can establish actual entry.
        let before_door = children(entry, "advcl").any(|prior| {
            prior.token < entry
                && actor_agrees(prior.token)
                && action_is_asserted(prior.token)
                && (form_is(prior, "開ける") && has_argument(prior.token, "obj", DOORS, "を")
                    || form_is(prior, "脱ぐ")
                        && has_argument(prior.token, "obj", &["靴"], "を")
                        && has_argument(prior.token, "obl", &["玄関"], "で"))
        });
        let after_door = result.is_some_and(|next| {
            actor_agrees(next.token)
                && action_is_asserted(next.token)
                && form_is(next, "閉める")
                && has_argument(next.token, "obj", DOORS, "を")
        });
        if !before_door && !after_door {
            return false;
        }
    }

    let Some(result) = result.filter(|_| entry_link.deprel == "advcl") else {
        return true;
    };
    // A linked change in role or a move from employment to household
    // care leaves the physical interpretation unproven.
    let homemaking = form_is(result, "なる") && has_argument(result.token, "obl", HOME_ROLES, "に");
    let caregiving = form_is(result, "専念")
        && has_argument(result.token, "obl", &["子育て", "育児", "家事"], "に")
        && children(entry, "advcl").any(|prior| {
            prior.token < entry
                && (form_is(prior, "退職する")
                    || (form_is(prior, "辞める")
                        && has_argument(prior.token, "obj", &["仕事", "職"], "を")))
        });
    !homemaking && !caregiving
}
