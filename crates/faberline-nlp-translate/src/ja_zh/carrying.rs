//! Render a portable item that accompanies the same actor's movement.

use faberline_nlp::doc::{Document, SyntaxToken, Upos};

use super::{Atom, Bunsetsu, LOCKED_OBJECT_ATOM, Segment};

pub(super) fn with_motion(
    source: &[Atom],
    predicates: &[Bunsetsu],
    following: &[Segment],
    doc: &Document,
    chars: &[char],
    omitted: bool,
) -> bool {
    let Some((hold, motion)) = linked_actions(predicates, following, doc) else {
        return false;
    };
    let Some(hold_token) = doc.tokens.get(hold) else {
        return false;
    };
    let Some(sentence) = doc.segmentation.sentences.get(hold_token.sentence) else {
        return false;
    };
    let start = doc
        .tokens
        .partition_point(|token| token.span.start_char < sentence.span.start_char);
    let end = doc
        .tokens
        .partition_point(|token| token.span.start_char < sentence.span.end_char);
    let Some(links) = doc.syntax.get(start..end) else {
        return false;
    };
    if !((start..end).contains(&motion) && hold < motion)
        || links.iter().enumerate().any(|(offset, link)| {
            link.token != start + offset
                || link.sentence != hold_token.sentence
                || link.span != doc.tokens[link.token].span
        })
    {
        return false;
    }
    let subject = |head| {
        links
            .iter()
            .find(|link| link.head == Some(head) && link.deprel == "nsubj")
            .map(|link| link.token)
    };
    if matches!((subject(hold), subject(motion)), (Some(a), Some(b)) if a != b) {
        return false;
    }
    let Some(tail) = doc.syntax.get(hold + 1) else {
        return false;
    };
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
        return false;
    }
    let direct_argument = |relation| {
        links
            .iter()
            .filter(move |link| link.head == Some(hold) && link.deprel == relation)
    };
    let has_case = |arg: &SyntaxToken, spelling| {
        links.iter().any(|link| {
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
            && (has_case(arg, "で") || has_case(arg, "に"))
    }) {
        return false;
    }
    let mut objects = direct_argument("obj");
    let Some(object) = objects.next() else {
        return false;
    };
    objects.next().is_none()
        && has_case(object, "を")
        && doc.tokens[object.token].upos == Some(Upos::Noun)
        && object_is_portable(object, links, source, chars, omitted)
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
        && matches!(
            motion.lemma.as_deref(),
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
        ))
    .then_some((hold_id, motion_id))
}

fn object_is_portable(
    object: &SyntaxToken,
    links: &[SyntaxToken],
    source: &[Atom],
    chars: &[char],
    omitted: bool,
) -> bool {
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
    let lock = source.iter().find(|atom| {
        atom.is_locked && atom.span.start_char < object.span.end_char && start < atom.span.end_char
    });
    let available = match lock {
        Some(atom) => {
            omitted
                && atom.doc_tok_idx == Some(object.token)
                && atom.lemma.as_deref() == Some(LOCKED_OBJECT_ATOM)
                && atom.span.start_char == start
                && atom.span.end_char == object.span.end_char
        }
        None => source
            .iter()
            .any(|atom| !atom.is_locked && atom.doc_tok_idx == Some(object.token)),
    };
    if !available {
        return false;
    }
    let Some(word) = chars.get(start..object.span.end_char) else {
        return false;
    };
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

#[cfg(test)]
mod tests {
    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
    };
    use faberline_nlp::doc::model::Backend;

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
        glossary.objects.push(GlossaryEntry {
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
}
