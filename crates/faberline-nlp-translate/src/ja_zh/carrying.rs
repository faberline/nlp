//! Render a portable item that accompanies the same actor's movement.

use faberline_nlp::doc::{Document, SyntaxToken, Upos, analyze_document};

use crate::lock::{LockOrigin, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{Atom, Bunsetsu, LOCKED_OBJECT_ATOM, Segment, Span};

pub(super) struct GlossaryCarry {
    object: Span,
    hold: Span,
    motion: Span,
}

pub(super) fn with_motion(
    source: &[Atom],
    predicates: &[Bunsetsu],
    following: &[Segment],
    doc: &Document,
    chars: &[char],
    locks: Option<&SpanLockTable>,
) -> bool {
    let Some((hold, motion)) = linked_actions(predicates, following, doc) else {
        return false;
    };
    let Some((object, noun, _)) = portable_object(doc, chars, hold, motion) else {
        return false;
    };
    if let Some(atom) = source.iter().find(|atom| {
        atom.is_locked
            && atom.span.start_char < noun.end_char
            && noun.start_char < atom.span.end_char
    }) {
        // The native syntax already proves the object and its separate case marker.
        // An exact noun match keeps that role while changing the emitted noun.
        return atom.span == noun
            && atom.doc_tok_idx == Some(object)
            && locks.is_some_and(|table| {
                table.locks.iter().any(|lock| {
                    lock.source_span == noun
                        && lock.placeholder == atom.surface
                        && lock.mandatory
                        && matches!(
                            lock.origin,
                            LockOrigin::Glossary {
                                kind: GlossaryKind::Object | GlossaryKind::DomainTerm,
                                ..
                            }
                        )
                })
            });
    }
    source
        .iter()
        .any(|atom| !atom.is_locked && atom.doc_tok_idx == Some(object))
}

fn portable_object(
    doc: &Document,
    chars: &[char],
    hold: usize,
    motion: usize,
) -> Option<(usize, Span, Span)> {
    let hold_token = doc.tokens.get(hold)?;
    let sentence = doc.segmentation.sentences.get(hold_token.sentence)?;
    let start = doc
        .tokens
        .partition_point(|token| token.span.start_char < sentence.span.start_char);
    let end = doc
        .tokens
        .partition_point(|token| token.span.start_char < sentence.span.end_char);
    let links = doc.syntax.get(start..end)?;
    if !((start..end).contains(&motion) && hold < motion)
        || links.iter().enumerate().any(|(offset, link)| {
            link.token != start + offset
                || link.sentence != hold_token.sentence
                || link.span != doc.tokens[link.token].span
        })
    {
        return None;
    }
    let subject = |head| {
        links
            .iter()
            .find(|link| link.head == Some(head) && link.deprel == "nsubj")
            .map(|link| link.token)
    };
    if matches!((subject(hold), subject(motion)), (Some(a), Some(b)) if a != b) {
        return None;
    }
    let tail = doc.syntax.get(hold + 1)?;
    let mark = |link: &SyntaxToken, spelling| {
        link.head == Some(hold) && link.deprel == "mark" && is_spelling(link, chars, spelling)
    };
    let connected = hold_token.span.end_char == tail.span.start_char
        && (mark(tail, "て")
            || (tail.head == Some(hold)
                && tail.deprel == "aux"
                && is_spelling(tail, chars, "ない")
                && doc.syntax.get(hold + 2).is_some_and(|next| {
                    tail.span.end_char == next.span.start_char && mark(next, "で")
                })));
    if !connected {
        return None;
    }
    let direct_argument = |relation| {
        links
            .iter()
            .filter(move |link| link.head == Some(hold) && link.deprel == relation)
    };
    let argument_case = |arg: &SyntaxToken, spelling| {
        links.iter().find(|link| {
            link.head == Some(arg.token)
                && link.deprel == "case"
                && arg.span.end_char == link.span.start_char
                && is_spelling(link, chars, spelling)
        })
    };
    if direct_argument("obl").any(|arg| {
        ["手", "片手", "両手", "右手", "左手", "手のひら"]
            .iter()
            .any(|word| is_spelling(arg, chars, word))
            && (argument_case(arg, "で").is_some() || argument_case(arg, "に").is_some())
    }) {
        return None;
    }
    let mut objects = direct_argument("obj");
    let object = objects.next()?;
    if objects.next().is_some() || doc.tokens[object.token].upos != Some(Upos::Noun) {
        return None;
    }
    let particle = argument_case(object, "を")?.span;
    let noun = compound_span(object, links);
    portable_word(chars.get(noun.start_char..noun.end_char)?).then_some((
        object.token,
        noun,
        particle,
    ))
}

fn linked_actions(
    predicates: &[Bunsetsu],
    following: &[Segment],
    doc: &Document,
) -> Option<(usize, usize)> {
    let [predicate] = predicates else { return None };
    let mut verbs = predicate
        .atoms
        .iter()
        .filter(|atom| atom.upos == Some(Upos::Verb));
    let hold = verbs.next()?;
    if verbs.next().is_some()
        || predicate.atoms.iter().any(|atom| atom.is_locked)
        || hold.lemma.as_deref() != Some("持つ")
    {
        return None;
    }
    let hold_id = hold.doc_tok_idx?;
    let link = doc.syntax.get(hold_id)?;
    let motion_id = link.head?;
    if link.token != hold_id || link.span != hold.span || link.deprel != "advcl" {
        return None;
    }
    let Some(Segment::Clause { atoms, .. }) = following.first() else {
        return None;
    };
    let motion = atoms
        .iter()
        .find(|atom| atom.doc_tok_idx == Some(motion_id))?;
    let motion_link = doc.syntax.get(motion_id)?;
    (!motion.is_locked
        && motion.upos == Some(Upos::Verb)
        && motion_link.token == motion_id
        && motion_link.span == motion.span
        && motion_link.deprel == "root"
        && is_motion(motion.lemma.as_deref()))
    .then_some((hold_id, motion_id))
}

fn is_motion(lemma: Option<&str>) -> bool {
    matches!(
        lemma,
        Some(
            "行く"
                | "出かける"
                | "出掛ける"
                | "出る"
                | "帰る"
                | "戻る"
                | "向かう"
                | "移動する"
                | "出発する"
        )
    )
}

fn compound_span(object: &SyntaxToken, links: &[SyntaxToken]) -> Span {
    let mut start = object.span.start_char;
    for link in links.iter().rev().filter(|link| link.token < object.token) {
        if link.head != Some(object.token)
            || link.deprel != "compound"
            || link.span.end_char != start
        {
            break;
        }
        start = link.span.start_char;
    }
    Span {
        start_char: start,
        end_char: object.span.end_char,
    }
}

fn portable_word(word: &[char]) -> bool {
    // An artifact can also be a vehicle or a building. Keep this
    // vocabulary to named items that a person can take along.
    [
        "傘",
        "雨傘",
        "日傘",
        "財布",
        "水筒",
        "切符",
        "鍵",
        "かばん",
        "鞄",
        "バッグ",
        "リュック",
        "手荷物",
        "荷物",
        "書類",
        "袋",
        "箱",
        "弁当",
        "薬",
        "手帳",
        "カメラ",
        "スマホ",
        "携帯電話",
        "杖",
        "本",
    ]
    .iter()
    .any(|spelling| word.iter().copied().eq(spelling.chars()))
}

fn is_spelling(link: &SyntaxToken, chars: &[char], spelling: &str) -> bool {
    chars
        .get(link.span.start_char..link.span.end_char)
        .is_some_and(|word| word.iter().copied().eq(spelling.chars()))
}

pub(super) fn restore_case(
    atoms: &mut Vec<Atom>,
    chars: &[char],
    sentence: Span,
    doc: &Document,
    locks: &SpanLockTable,
) -> Vec<GlossaryCarry> {
    if !doc.options.enrich_morphology || !doc.options.cpu_syntax {
        return Vec::new();
    }
    let candidates: Vec<_> = locks
        .locks
        .iter()
        .filter_map(|lock| {
            let span = lock.source_span;
            if !lock.mandatory
                || !matches!(
                    lock.origin,
                    LockOrigin::Glossary {
                        kind: GlossaryKind::Object | GlossaryKind::DomainTerm,
                        ..
                    }
                )
                || span.start_char < sentence.start_char
                || span.end_char > sentence.end_char
            {
                return None;
            }
            let [word @ .., 'を'] = chars.get(span.start_char..span.end_char)? else {
                return None;
            };
            portable_word(word).then(|| {
                (
                    lock,
                    chars[span.start_char..span.end_char]
                        .iter()
                        .collect::<String>(),
                )
            })
        })
        .collect();
    if candidates.is_empty() {
        return Vec::new();
    }
    let mut options = doc.options.clone();
    let is_hint = |word: &String| candidates.iter().any(|(_, surface)| surface == word);
    options.names.retain(|word| !is_hint(word));
    options.lexicon.retain(|word| !is_hint(word));
    let text = &chars[sentence.start_char..sentence.end_char];
    let parsed = analyze_document(&text.iter().collect::<String>(), &options);
    if parsed.segmentation.sentences.len() != 1 {
        return Vec::new();
    }
    let absolute = |span: Span| Span {
        start_char: sentence.start_char + span.start_char,
        end_char: sentence.start_char + span.end_char,
    };
    let mut restored = Vec::new();
    for hold in &parsed.syntax {
        let Some(motion) = hold.head.and_then(|head| parsed.syntax.get(head)) else {
            continue;
        };
        let hold_token = &parsed.tokens[hold.token];
        let motion_token = &parsed.tokens[motion.token];
        if hold.deprel != "advcl"
            || motion.deprel != "root"
            || hold_token.upos != Some(Upos::Verb)
            || motion_token.upos != Some(Upos::Verb)
            || hold_token.lemma.as_deref() != Some("持つ")
            || !is_motion(motion_token.lemma.as_deref())
        {
            continue;
        }
        let Some((_, noun, particle)) = portable_object(&parsed, text, hold.token, motion.token)
        else {
            continue;
        };
        let noun = absolute(noun);
        let particle = absolute(particle);
        let hold = absolute(hold.span);
        let motion = absolute(motion.span);
        let span = Span {
            start_char: noun.start_char,
            end_char: particle.end_char,
        };
        let Some((lock, _)) = candidates.iter().find(|(lock, _)| lock.source_span == span) else {
            continue;
        };
        let Some(head) = atoms
            .iter()
            .find(|atom| {
                !atom.is_locked && atom.span == hold && atom.lemma.as_deref() == Some("持つ")
            })
            .and_then(|atom| atom.doc_tok_idx)
        else {
            continue;
        };
        if !atoms
            .iter()
            .any(|atom| !atom.is_locked && atom.span == motion && is_motion(atom.lemma.as_deref()))
        {
            continue;
        }
        let Some(index) = atoms.iter().position(|atom| {
            atom.is_locked && atom.span == span && atom.surface == lock.placeholder
        }) else {
            continue;
        };
        let mut object = atoms[index].clone();
        object.span = noun;
        object.lemma = Some(LOCKED_OBJECT_ATOM.to_owned());
        object.doc_tok_idx = None;
        object.head = Some(head);
        object.deprel = "obj".to_owned();
        let marker = Atom {
            span: particle,
            surface: "を".to_owned(),
            lemma: None,
            upos: Some(Upos::Adp),
            doc_tok_idx: None,
            head: None,
            deprel: "case".to_owned(),
            is_locked: false,
        };
        // Only the render atoms change; the original glossary hit still covers を.
        atoms.splice(index..=index, [object, marker]);
        restored.push(GlossaryCarry {
            object: noun,
            hold,
            motion,
        });
    }
    restored
}

pub(super) fn restored_carry(
    source: &[Atom],
    predicates: &[Bunsetsu],
    following: &[Segment],
    restored: &[GlossaryCarry],
) -> bool {
    if restored.is_empty() {
        return false;
    }
    let [predicate] = predicates else {
        return false;
    };
    let Some(Segment::Clause { atoms, .. }) = following.first() else {
        return false;
    };
    restored.iter().any(|link| {
        source
            .iter()
            .any(|atom| atom.is_locked && atom.span == link.object)
            && predicate
                .atoms
                .iter()
                .any(|atom| !atom.is_locked && atom.span == link.hold)
            && atoms
                .iter()
                .any(|atom| !atom.is_locked && atom.span == link.motion)
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
        doc_options_for_translate, translate_document,
    };
    use faberline_nlp::doc::{analyze_document, model::Backend};

    fn translate(text: &str, glossary: &TranslationGlossary) -> String {
        analyze_and_translate(
            text,
            &TranslateOptions::new(TargetVariant::ZhHant),
            glossary,
            None,
            Backend::Onnx,
        )
        .expect("translate on CPU")
        .target_text
    }

    #[test]
    fn a_portable_object_can_accompany_the_next_movement() {
        let glossary = TranslationGlossary::new();
        for text in [
            "水筒を持って、駅へ行きました。",
            "水筒を持って駅へ行きました。",
        ] {
            assert_eq!(translate(text, &glossary), "帶著水筒，往車站去了。");
        }
    }

    #[test]
    fn a_handhold_or_different_actor_keeps_the_existing_rendering() {
        let glossary = TranslationGlossary::new();
        assert_eq!(
            translate("片手で傘を持って、駅へ行きました。", &glossary),
            "用片手拿著傘，往車站去了。"
        );
        assert_eq!(
            translate("父が傘を持って、私は先に出かけました。", &glossary),
            "父親拿著傘，我先に出門了。"
        );
        assert_eq!(
            translate("箱を両手で持っています。", &glossary),
            "用兩手拿著箱。"
        );
    }

    #[test]
    fn an_opaque_item_keeps_its_glossary_rendering() {
        let mut glossary = TranslationGlossary::new();
        glossary.places.push(GlossaryEntry {
            canonical: "傘".to_owned(),
            surfaces: vec!["傘".to_owned()],
            targets: [(TargetVariant::ZhHant, "雨具".to_owned())].into(),
            ..GlossaryEntry::default()
        });
        assert_eq!(
            translate("傘を持って、駅へ行きました。", &glossary),
            "拿著雨具，往車站去了。"
        );
    }

    #[test]
    fn an_item_lock_keeps_the_object_and_the_carrying_action() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        for (source, expected, target_start) in [
            ("傘を持って、駅へ行きました。", "帶著雨具，往車站去了。", 2),
            ("傘を持たないで、出かけました。", "沒有帶雨具就出門了。", 3),
        ] {
            let unhinted = analyze_document(
                source,
                &doc_options_for_translate(&options, &TranslationGlossary::new()),
            );
            for term in ["傘", "傘を"] {
                let entry = GlossaryEntry {
                    canonical: term.to_owned(),
                    surfaces: vec![term.to_owned()],
                    targets: [(TargetVariant::ZhHant, "雨具".to_owned())].into(),
                    ..GlossaryEntry::default()
                };
                for glossary in [
                    TranslationGlossary {
                        objects: vec![entry.clone()],
                        ..TranslationGlossary::new()
                    },
                    TranslationGlossary {
                        domain_terms: vec![entry],
                        ..TranslationGlossary::new()
                    },
                ] {
                    let result =
                        analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                            .expect("translate on CPU");
                    let direct = translate_document(
                        source,
                        &unhinted,
                        #[cfg(feature = "narrative")]
                        None,
                        &options,
                        &glossary,
                        None,
                    )
                    .expect("translate pre-analyzed text");
                    assert_eq!(result.target_text, expected);
                    assert_eq!(direct.target_text, result.target_text);
                    assert_eq!(direct.glossary_hits, result.glossary_hits);
                    let [hit] = result.glossary_hits.as_slice() else {
                        panic!("expected one object match");
                    };
                    assert_eq!(hit.source_span.start_char, 0);
                    assert_eq!(hit.source_span.end_char, term.chars().count());
                    assert_eq!(hit.target_span.start_char, target_start);
                    assert_eq!(hit.target_span.end_char, target_start + 2);
                    for control in [
                        "片手で傘を持って、駅へ行きました。",
                        "父が傘を持って、私は先に出かけました。",
                    ] {
                        let output = translate(control, &glossary);
                        assert!(!output.contains("帶著"), "{output}");
                    }
                }
            }
        }
    }
}
