//! Render a source-linked genitive complement as a single noun phrase after 是.
use super::{
    Atom, BunsetsuRole, Document, TargetVariant, Upos, group_into_bunsetsus,
    is_demonstrative_person_noun, merge_adnominal_bunsetsus, render_bunsetsu_np,
};

fn noun_phrase(
    atoms: &[Atom],
    doc: &Document,
    sentence: usize,
    target: TargetVariant,
) -> Option<String> {
    let mut groups = group_into_bunsetsus(atoms, doc, sentence);
    // The original head may be the root precisely because the copula was attached
    // to it. Inside this complement it takes its ordinary nominal modifiers.
    groups.last_mut()?.role = BunsetsuRole::Other;
    let merged = merge_adnominal_bunsetsus(groups, atoms, doc, target);
    let [phrase] = merged.as_slice() else {
        return None;
    };
    Some(render_bunsetsu_np(phrase, target))
}

pub(super) fn render(
    atoms: &[Atom],
    doc: &Document,
    sentence: usize,
    target: TargetVariant,
) -> Option<String> {
    let (body, ending) = match atoms {
        [body @ .., yo, ne]
            if yo.surface == "よ"
                && ne.surface == "ね"
                && [yo, ne]
                    .iter()
                    .all(|a| !a.is_locked && a.deprel == "discourse") =>
        {
            (body, "呢")
        }
        [body @ .., part] if !part.is_locked && part.deprel == "discourse" => {
            match part.surface.as_str() {
                "か" => (body, "嗎"),
                "ね" | "よね" => (body, "呢"),
                "よ" => (body, "喔"),
                _ => return None,
            }
        }
        _ => (atoms, ""),
    };
    let head_index = body
        .iter()
        .rposition(|a| matches!(a.upos, Some(Upos::Noun | Upos::Propn)))?;
    let (nominals, tail) = body.split_at(head_index + 1);
    let head = nominals.last()?;
    // 「鳥のことです」 can define what a word refers to. こと is not a possessed item.
    if matches!(head.surface.as_str(), "こと" | "事") {
        return None;
    }
    if tail.is_empty()
        || tail.len() > 7
        || body.iter().any(|a| a.is_locked)
        || !body[head_index..]
            .windows(2)
            .all(|p| p[0].span.end_char == p[1].span.start_char)
    {
        return None;
    }
    let suffix: String = tail.iter().map(|a| a.surface.as_str()).collect();
    let copula = match suffix.as_str() {
        "です" | "だ" if tail[0].upos == Some(Upos::Aux) => "是",
        "だった" | "でした" => "當時是",
        "じゃない" | "ではない" | "じゃありません" | "ではありません" => {
            "不是"
        }
        "じゃなかった" | "ではなかった" | "じゃありませんでした" | "ではありませんでした" => {
            "當時不是"
        }
        _ => return None,
    };

    let split = nominals
        .iter()
        .position(|a| a.deprel == "case" && matches!(a.surface.as_str(), "は" | "が" | "も"));
    let (topic, phrase) = match split {
        Some(i) => (&nominals[..i], &nominals[i + 1..]),
        None => (&[][..], nominals),
    };
    // An object or adjunct before an instrumental で and する is a different
    // construction. Every atom here must belong to the topic or its nominal complement.
    if !topic.iter().chain(phrase).all(|a| {
        matches!(
            a.upos,
            Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::Num | Upos::Det)
        ) || (a.surface == "の" && a.deprel == "case")
    }) {
        return None;
    }
    let head_id = head.doc_tok_idx?;
    if !phrase.windows(2).any(|p| {
        p[0].deprel == "nmod"
            && p[0].head == Some(head_id)
            && p[1].surface == "の"
            && p[1].deprel == "case"
            && p[1].head == p[0].doc_tok_idx
    }) {
        return None;
    }
    let subject = match (topic, split) {
        ([], None) if !(suffix == "でした" && tail[0].upos == Some(Upos::Adp)) => String::new(),
        ([], _) => return None,
        (_, Some(i)) => {
            let noun = topic.last()?;
            if noun.deprel != "nsubj" || nominals[i].head != noun.doc_tok_idx {
                return None;
            }
            if suffix == "でした"
                && tail[0].upos == Some(Upos::Adp)
                && !(matches!(
                    noun.surface.as_str(),
                    "これ" | "それ" | "あれ" | "ここ" | "そこ" | "あそこ"
                ) || is_demonstrative_person_noun(head)
                    || noun.surface == head.surface
                    || topic.first().is_some_and(|a| a.upos == Some(Upos::Det)))
            {
                return None;
            }
            let mut subject = match topic {
                [a] if a.surface == "これ" => "這".to_owned(),
                [a] if matches!(a.surface.as_str(), "それ" | "あれ") => "那".to_owned(),
                _ => noun_phrase(topic, doc, sentence, target)?,
            };
            if nominals[i].surface == "も" {
                subject.push('也');
            }
            subject
        }
        _ => return None,
    };
    // In a nominal identification, the head もの can stand for an already
    // mentioned item. The explicit owner carries that reference in Chinese.
    let complement = match phrase {
        [owner @ .., no, thing] if thing.surface == "もの" && no.surface == "の" => {
            format!("{}的", noun_phrase(owner, doc, sentence, target)?)
        }
        _ => noun_phrase(phrase, doc, sentence, target)?,
    };
    let ending = if ending == "嗎"
        && nominals.iter().any(|a| {
            matches!(
                a.surface.as_str(),
                "誰" | "だれ" | "何" | "どれ" | "どの" | "どこ" | "どちら"
            )
        }) {
        ""
    } else {
        ending
    };
    Some(format!("{subject}{copula}{complement}{ending}"))
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
    };

    fn translate(source: &str) -> String {
        analyze_and_translate(
            source,
            &TranslateOptions::new(TargetVariant::ZhHant),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap()
        .target_text
    }

    #[test]
    fn possessive_copula_keeps_the_owner_inside_the_complement() {
        for (source, expected) in [
            ("それは先生の手紙です。", "那是老師的信。"),
            ("これは母の本ではありません。", "這不是母親的書。"),
            ("あれは友達の本ですか。", "那是朋友的書嗎？"),
            ("これは妹の手紙でした。", "這當時是妹妹的信。"),
            ("これは誰の本ですか。", "這是誰的書？"),
            ("この本は私のものです。", "這本書是我的。"),
            ("これは母の本ですね。", "這是母親的書呢。"),
            ("私の本を読みました。", "讀了我的書。"),
            ("私は学生です。", "我是學生。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
    }
}
