use super::{
    Atom, Upos, attached_demonstrative_at, core_non_punct, is_japanese_numeral,
    translate_single_ja_term,
};
use std::ops::Range;

/// The counter may float next to the verb or modify the object with の.
struct Measure {
    numeral: String,
    unit: &'static str,
    span: Range<usize>,
}

fn measure_at(atoms: &[Atom], i: usize, noun: &Atom, verb: usize) -> Option<Measure> {
    let first = &atoms[i];
    let native_digit = |n: &str| {
        matches!(
            n,
            "一" | "二" | "三" | "四" | "五" | "六" | "七" | "八" | "九"
        ) || (n.chars().count() == 1 && n.chars().all(|c| matches!(c, '1'..='9' | '１'..='９')))
    };
    let (numeral, unit, kind, end) = if first.upos == Some(Upos::Noun)
        && let Some(number) = first.surface.strip_suffix('つ')
        && native_digit(number)
    {
        (number, first, "つ", i + 1)
    } else {
        let suffix = atoms.get(i + 1)?;
        if first.span.end_char != suffix.span.start_char || !is_japanese_numeral(&first.surface) {
            return None;
        }
        match suffix.surface.as_str() {
            "つ" if native_digit(&first.surface)
                && matches!(first.upos, Some(Upos::Noun | Upos::Num))
                && suffix.upos == Some(Upos::Adp)
                && suffix.deprel == "case"
                && suffix.head == Some(first.doc_tok_idx?) =>
            {
                (first.surface.as_str(), first, "つ", i + 2)
            }
            "本" if suffix.upos == Some(Upos::Noun)
                && first.head == Some(suffix.doc_tok_idx?)
                && ((first.upos == Some(Upos::Num) && first.deprel == "nummod")
                    || (first.upos == Some(Upos::Noun)
                        && first.deprel == "compound"
                        && first.surface.chars().all(|c| ('０'..='９').contains(&c)))) =>
            {
                (first.surface.as_str(), suffix, "本", i + 2)
            }
            _ => return None,
        }
    };
    let end = match unit.deprel.as_str() {
        "obl" if unit.head == Some(verb) => end,
        "nmod" if unit.head == noun.doc_tok_idx => {
            let no = atoms.get(end)?;
            if no.surface != "の" || no.deprel != "case" || no.head != unit.doc_tok_idx {
                return None;
            }
            end + 1
        }
        _ => return None,
    };
    // A tokenizer can split 十二 and 二十一 across tokens, with each part
    // attached to the counter or to another part of the same numeral.
    let mut start = i;
    if kind == "本" {
        while start > 0 {
            let prev = &atoms[start - 1];
            if prev.span.end_char != atoms[start].span.start_char
                || !is_japanese_numeral(&prev.surface)
            {
                break;
            }
            if prev.head.is_none()
                || !matches!(prev.upos, Some(Upos::Num | Upos::Noun))
                || !atoms[start..=i + 1]
                    .iter()
                    .any(|part| part.doc_tok_idx == prev.head)
            {
                return None;
            }
            start -= 1;
        }
    }
    let span = start..end;
    // Other modifiers, e.g. ずつ or だけ, change what the quantity means.
    // Leave the entire construction together when their scope is unresolved.
    if atoms.iter().enumerate().any(|(idx, a)| {
        !span.contains(&idx)
            && a.head.is_some()
            && atoms[span.clone()]
                .iter()
                .any(|part| part.doc_tok_idx == a.head)
    }) {
        return None;
    }
    let mut number = String::new();
    for part in &atoms[start..i] {
        number.push_str(&part.surface);
    }
    number.push_str(numeral);
    Some(Measure {
        numeral: number,
        unit: kind,
        span,
    })
}

pub(super) fn classifier(unit: &str, noun: &str) -> Option<&'static str> {
    Some(match unit {
        "枚" => match noun {
            "紙" | "地圖" | "照片" | "票" | "海報" | "明信片" | "表格" => "張",
            _ => "枚",
        },
        "冊" => "本",
        "個" => "個",
        "台" => match noun {
            "車" | "公車" | "汽車" | "卡車" | "機車" | "自行車" | "腳踏車" => "輛",
            _ => "台",
        },
        "件" => "件",
        "匹" => "隻",
        "杯" => "杯",
        // 本 is a shape counter. The target depends on the object itself.
        "本" => match noun {
            "筆" | "鉛筆" | "鋼筆" | "原子筆" => "支",
            "水" | "酒" | "啤酒" | "果汁" => "瓶",
            _ => return None,
        },
        "つ" => match noun {
            "橘子" | "蘋果" | "梨子" | "桃子" => "顆",
            "箱" | "蛋糕" | "包包" | "便當" => "個",
            _ => return None,
        },
        _ => return None,
    })
}

pub(super) fn push_quantity(out: &mut String, numeral: &str, counter: &str) {
    if numeral == "二" {
        out.push('兩');
    } else {
        out.push_str(&translate_single_ja_term(numeral, None, Some(Upos::Num)));
    }
    out.push_str(counter);
}

/// Keep all other arguments and verb inflections on the normal clause path.
/// Only collapse a uniquely linked, visible object with an established count reading.
pub(super) fn bind(atoms: &[Atom]) -> Option<Vec<Atom>> {
    if atoms.iter().any(|a| a.is_locked) {
        return None;
    }
    let mut objects = atoms.windows(2).enumerate().filter(|(_, pair)| {
        let (noun, wo) = (&pair[0], &pair[1]);
        noun.upos == Some(Upos::Noun)
            && noun.deprel == "obj"
            && noun.doc_tok_idx.is_some()
            && wo.surface == "を"
            && wo.upos == Some(Upos::Adp)
            && wo.deprel == "case"
            && wo.head == noun.doc_tok_idx
    });
    let (object_index, pair) = objects.next()?;
    let noun = &pair[0];
    let governor = noun.head?;
    let verb_index = atoms
        .iter()
        .position(|a| a.doc_tok_idx == Some(governor) && a.upos == Some(Upos::Verb))?;
    let verb = &atoms[verb_index];
    if objects.next().is_some()
        || !(matches!(verb.lemma.as_deref(), Some("買う" | "貸す")) || verb.surface == "ください")
        || atoms
            .iter()
            .any(|a| a.upos == Some(Upos::Adp) && matches!(a.surface.as_str(), "と" | "や"))
    {
        return None;
    }
    let mut measures = (0..atoms.len()).filter_map(|i| measure_at(atoms, i, noun, governor));
    let amount = measures.next()?;
    if measures.next().is_some() {
        return None;
    }
    let mut start = object_index;
    while let Some(prev) = start.checked_sub(1).map(|i| &atoms[i]) {
        if prev.deprel != "compound"
            || prev.head != noun.doc_tok_idx
            || prev.upos != Some(Upos::Noun)
        {
            break;
        }
        start -= 1;
    }
    // An additional nominal modifier can change the scope or the type of the
    // counted object. Keep that construction on the ordinary renderer.
    if atoms.iter().enumerate().any(|(i, a)| {
        a.head == noun.doc_tok_idx
            && matches!(
                a.deprel.as_str(),
                "nmod" | "compound" | "acl" | "amod" | "nummod"
            )
            && !(start..object_index).contains(&i)
            && !amount.span.contains(&i)
    }) {
        return None;
    }
    let object: String = atoms[start..=object_index]
        .iter()
        .map(|a| translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos))
        .collect();
    let classifier = classifier(amount.unit, &object)?;
    let mut text = String::new();
    if start > 0
        && let Some(det) =
            attached_demonstrative_at(&atoms[start - 1], noun, atoms[start].span.start_char)
    {
        text.push(det);
        start -= 1;
    }
    push_quantity(&mut text, &amount.numeral, classifier);
    text.push_str(if object == "箱" { "箱子" } else { &object });

    let request = match core_non_punct(&atoms[verb_index..]) {
        [please] if please.surface == "ください" && please.deprel == "root" => Some("請給我"),
        [lend, te, please]
            if lend.lemma.as_deref() == Some("貸す")
                && lend.deprel == "advcl"
                && lend.head == please.doc_tok_idx
                && te.surface == "て"
                && te.head == lend.doc_tok_idx
                && please.surface == "ください"
                && please.deprel == "root" =>
        {
            Some("借我")
        }
        _ => None,
    };
    let implicit_recipient = !atoms
        .iter()
        .any(|a| a.deprel == "iobj" || a.surface == "に");
    let mut result = Vec::with_capacity(atoms.len());
    for (i, atom) in atoms.iter().enumerate() {
        if amount.span.contains(&i) || (start..object_index).contains(&i) {
            continue;
        }
        let mut atom = atom.clone();
        if i == object_index {
            atom.surface.clone_from(&text);
            atom.is_locked = true;
        } else if i == verb_index
            && implicit_recipient
            && let Some(zh) = request
        {
            atom.surface = zh.to_owned();
            atom.lemma = None;
            atom.is_locked = true;
        }
        result.push(atom);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

    fn translate(jp: &str, target: TargetVariant) -> String {
        analyze_and_translate(
            jp,
            &TranslateOptions::new(target),
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap()
        .target_text
    }

    #[test]
    fn counted_objects_keep_quantity_and_speech_act() {
        for (jp, zh) in [
            ("ジュースを二本買いました。", "買了兩瓶果汁。"),
            ("二本の水を買いました。", "買了兩瓶水。"),
            ("私は水を二本買いました。", "我買了兩瓶水。"),
            ("水を二本買いませんでした。", "沒有買兩瓶水。"),
            ("水を三本買いますか。", "買三瓶水嗎？"),
            ("水を十二本買いました。", "買了十二瓶水。"),
            ("水を二十一本買いました。", "買了二十一瓶水。"),
            ("ビールを三本ください。", "請給我三瓶啤酒。"),
            ("鉛筆を二本貸してください。", "請借我兩支鉛筆。"),
            ("鉛筆を一本貸しました。", "出借了一支鉛筆。"),
            ("二つのみかんを買いました。", "買了兩顆橘子。"),
            ("箱を二つ買いました。", "買了兩個箱子。"),
            ("その鉛筆を一本貸してください。", "請借我那一支鉛筆。"),
            ("みかんを2つ買いました。", "買了2顆橘子。"),
            ("３つのみかんを買いました。", "買了３顆橘子。"),
            ("ビールを３本ください。", "請給我３瓶啤酒。"),
        ] {
            assert_eq!(translate(jp, TargetVariant::ZhHant), zh, "{jp}");
        }
        assert_eq!(
            translate("みかんを二つ買いました。", TargetVariant::ZhHans),
            "买了两颗橘子。"
        );
    }

    #[test]
    fn standalone_book_and_ambiguous_counts_remain_distinct() {
        assert_eq!(
            translate("私はその本を読みます。", TargetVariant::ZhHant),
            "我讀那本書。"
        );
        for (jp, object) in [
            ("本を三冊買いました。", "書"),
            ("木を三本買いました。", "樹"),
        ] {
            let zh = translate(jp, TargetVariant::ZhHant);
            assert!(
                zh.contains(object) && !zh.contains(['瓶', '支']),
                "{jp} {zh}"
            );
        }
        let explicit_recipient = translate("妹に鉛筆を一本貸してください。", TargetVariant::ZhHant);
        assert!(explicit_recipient.contains("妹妹") && !explicit_recipient.contains('我'));
    }
}
