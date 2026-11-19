//! Render a source-linked genitive complement as a single noun phrase after 是.
use super::{
    Atom, BunsetsuRole, Document, TargetVariant, Upos, group_into_bunsetsus,
    is_demonstrative_person_noun, merge_adnominal_bunsetsus, render_bunsetsu_np,
};

pub(super) fn noun_phrase(
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

// Keep a protected noun as its original placeholder. A partial token or a
// protected clause cannot supply the syntactic head of an owned noun phrase.
pub(super) fn whole_nominal_lock(a: &Atom, doc: &Document) -> bool {
    if !(a.surface.starts_with("[[PH") && a.surface.ends_with("]]")) {
        return false;
    }
    let Some(last) = a.doc_tok_idx else {
        return false;
    };
    let first = doc
        .tokens
        .partition_point(|t| t.span.start_char < a.span.start_char);
    let Some(tokens) = doc.tokens.get(first..=last) else {
        return false;
    };
    tokens
        .first()
        .is_some_and(|t| t.span.start_char == a.span.start_char)
        && tokens
            .last()
            .is_some_and(|t| t.span.end_char == a.span.end_char)
        && tokens.iter().all(|t| {
            matches!(
                t.upos,
                Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::X)
            )
        })
        && tokens
            .windows(2)
            .all(|w| w[0].span.end_char == w[1].span.start_char)
}

// The analyzer can split ものでは into も + ので + は. In that position ので
// is part of the copula, while a later ので may still introduce a reason.
pub(super) fn contains_copular_node(atoms: &[Atom], at: usize, doc: &Document) -> bool {
    let node = &atoms[at];
    if node.surface != "ので" || node.is_locked || node.deprel != "mark" {
        return false;
    }
    let before = &atoms[..at];
    let owner = match before {
        [.., owner, no, mo]
            if !no.is_locked
                && no.surface == "の"
                && no.head == owner.doc_tok_idx
                && !mo.is_locked
                && mo.surface == "も"
                && mo.head == owner.doc_tok_idx =>
        {
            owner
        }
        [.., owner] => owner,
        _ => return false,
    };
    if !owner_atom(owner, doc) || owner.doc_tok_idx != node.head {
        return false;
    }
    match &atoms[at + 1..] {
        [wa, neg, ..] => {
            !wa.is_locked
                && !neg.is_locked
                && ((wa.surface == "は"
                    && matches!(
                        neg.surface.as_str(),
                        "ない" | "なかっ" | "あり" | "なく" | "ある" | "あっ"
                    ))
                    || (wa.surface == "し" && neg.surface == "た"))
        }
        _ => false,
    }
}

fn owner_atom(a: &Atom, doc: &Document) -> bool {
    matches!(a.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
        && (!a.is_locked || a.lemma.as_deref() == Some(MODIFIED_NOUN) || whole_nominal_lock(a, doc))
}

const MODIFIED_NOUN: &str = "__ja_possession_modified_noun";

fn original_pos(atom: &Atom, doc: &Document) -> Option<Upos> {
    if !atom.is_locked {
        return atom.upos;
    }
    let token = doc.tokens.get(atom.doc_tok_idx?)?;
    (atom.span == token.span).then_some(token.upos).flatten()
}

fn noun_classifier(word: &str, noun: &Atom) -> &'static str {
    match word {
        "鑰匙" | "傘" => "把",
        "房間" => "間",
        "包包" | "背包" => "個",
        "帽子" => "頂",
        "筆記本" | "課本" => "本",
        "鞋子" => "雙",
        _ => super::demonstrative_classifier(word, noun),
    }
}

// Fold adjective phrases only into the noun identified by their source arcs.
// The following の remains the possessor marker, outside this noun phrase.
fn modified_noun(words: &[Atom], doc: &Document, chars: &[char]) -> Option<Vec<Atom>> {
    let (head, prefix) = words.split_last()?;
    if !owner_atom(head, doc) || head.doc_tok_idx.is_none() {
        return None;
    }
    let start = prefix
        .iter()
        .rposition(|a| {
            !a.is_locked && a.deprel == "case" && matches!(a.surface.as_str(), "の" | "と")
        })
        .map_or(0, |i| i + 1);
    let mut modifiers = &prefix[start..];
    let mut demonstrative = "";
    if let [det, rest @ ..] = modifiers
        && !det.is_locked
        && det.deprel == "det"
        && det.head == head.doc_tok_idx
    {
        demonstrative = match det.surface.as_str() {
            "この" => "這",
            "その" | "あの" => "那",
            "どの" => "哪",
            _ => "",
        };
        if !demonstrative.is_empty() {
            modifiers = rest;
        }
    }
    let mut descriptions = Vec::new();
    while !modifiers.is_empty() {
        let (degree, content) = match modifiers {
            [adverb, adj, ..]
                if !adverb.is_locked
                    && adverb.deprel == "advmod"
                    && adverb.head == adj.doc_tok_idx
                    && original_pos(adj, doc) == Some(Upos::Adj) =>
            {
                (
                    super::translate_single_ja_term(
                        &adverb.surface,
                        adverb.lemma.as_deref(),
                        adverb.upos,
                    ),
                    &modifiers[1..],
                )
            }
            _ => (String::new(), modifiers),
        };
        let (adj, rest) = content.split_first()?;
        let adjective = original_pos(adj, doc) == Some(Upos::Adj)
            || !adj.is_locked
                && matches!(adj.surface.as_str(), "大きな" | "小さな")
                && adj.deprel == "det";
        if !adjective
            || !(adj.head == head.doc_tok_idx
                && matches!(adj.deprel.as_str(), "amod" | "acl" | "det")
                || adj.deprel == "advcl"
                    && rest.iter().any(|next| {
                        next.doc_tok_idx == adj.head
                            && original_pos(next, doc) == Some(Upos::Adj)
                            && next.head == head.doc_tok_idx
                    }))
        {
            return None;
        }
        let n = rest
            .iter()
            .take_while(|a| {
                !a.is_locked
                    && a.head == adj.doc_tok_idx
                    && (a.deprel == "aux"
                        || a.deprel == "discourse" && a.surface == "な"
                        || a.deprel == "mark" && a.surface == "て" && adj.deprel == "advcl")
            })
            .count();
        let morph: String = rest[..n].iter().map(|a| a.surface.as_str()).collect();
        let tense = match morph.as_str() {
            "" | "な" | "て" => "",
            "ない" => "不",
            "た" => "原本",
            "なかった" => "原本不",
            _ => return None,
        };
        let adjective = if adj.is_locked {
            adj.surface.clone()
        } else if adj.surface == "大きな" {
            "大".to_owned()
        } else if adj.surface == "小さな" {
            "小".to_owned()
        } else {
            let stem = adj.lemma.clone().or_else(|| {
                adj.surface
                    .strip_suffix("かっ")
                    .or_else(|| adj.surface.strip_suffix('く'))
                    .map(|x| format!("{x}い"))
            });
            super::translate_single_ja_term(
                stem.as_deref().unwrap_or(&adj.surface),
                stem.as_deref(),
                Some(Upos::Adj),
            )
        };
        let word = adjective.trim_end_matches('的');
        let hue = matches!(word, "紅" | "黑" | "白" | "藍" | "黃" | "綠" | "紫" | "灰");
        let description = if hue && degree.is_empty() && descriptions.is_empty() && n == rest.len()
        {
            let tense = match tense {
                "不" => "不是",
                "原本不" => "原本不是",
                "原本" => "原本是",
                _ => "",
            };
            format!("{tense}{word}色")
        } else {
            format!("{tense}{degree}{word}")
        };
        descriptions.push(description);
        modifiers = &rest[n..];
    }
    if descriptions.is_empty() {
        return None;
    }
    let raw: String = chars
        .get(head.span.start_char..head.span.end_char)?
        .iter()
        .collect();
    let word = super::translate_single_ja_term(&raw, None, original_pos(head, doc));
    let mut noun = head.clone();
    let classifier = if demonstrative.is_empty() {
        ""
    } else {
        noun_classifier(&word, head)
    };
    let text = if head.is_locked {
        head.surface.as_str()
    } else if word == "箱" {
        "箱子"
    } else {
        &word
    };
    noun.surface = format!(
        "{demonstrative}{classifier}{}{}的{text}",
        if descriptions.len() > 1 { "又" } else { "" },
        descriptions.join("又")
    );
    noun.span.start_char = words[start].span.start_char;
    noun.is_locked = true;
    noun.lemma = Some(MODIFIED_NOUN.to_owned());
    let mut result = words[..start].to_vec();
    result.push(noun);
    Some(result)
}

fn elliptical_owner(
    words: &[Atom],
    doc: &Document,
    sentence: usize,
    target: TargetVariant,
) -> Option<String> {
    if let Some(i) = words
        .iter()
        .position(|a| !a.is_locked && a.surface == "と" && a.deprel == "case")
    {
        let left = words.get(i.checked_sub(1)?)?;
        let right = words.last()?;
        if words[i].head != left.doc_tok_idx
            || !(left.head == right.doc_tok_idx || left.head == right.head)
        {
            return None;
        }
        return Some(format!(
            "{}和{}",
            elliptical_owner(&words[..i], doc, sentence, target)?,
            elliptical_owner(&words[i + 1..], doc, sentence, target)?
        ));
    }
    // The polite suffix さん does not supply the owner's gender.
    if let [.., name, honorific] = words
        && !honorific.is_locked
        && honorific.surface == "さん"
        && name.upos == Some(Upos::Propn)
        && name.head == honorific.doc_tok_idx
        && name.span.end_char == honorific.span.start_char
    {
        return noun_phrase(&words[..words.len() - 1], doc, sentence, target);
    }
    noun_phrase(words, doc, sentence, target)
}

fn render_ellipsis(
    body: &[Atom],
    ending: &str,
    doc: &Document,
    sentence: usize,
    target: TargetVariant,
    chars: &[char],
) -> Option<String> {
    let at = body.windows(2).rposition(|p| {
        owner_atom(&p[0], doc)
            && !p[1].is_locked
            && p[1].head == p[0].doc_tok_idx
            && matches!(
                (p[1].surface.as_str(), p[1].deprel.as_str()),
                ("の", "case") | ("ので", "mark")
            )
    })?;
    let suffix = &body[at + 1..];
    if suffix.iter().any(|a| a.is_locked)
        || !body[at..]
            .windows(2)
            .all(|p| p[0].span.end_char == p[1].span.start_char)
    {
        return None;
    }
    let text: String = suffix.iter().map(|a| a.surface.as_str()).collect();
    let tail = text
        .strip_prefix("のもの")
        .or_else(|| text.strip_prefix('の'))?;
    // In 母のか、妹のか分かりません each か belongs to an embedded choice.
    if tail.is_empty() && ending == "嗎" && body[at].deprel == "nmod" {
        return None;
    }
    let copula = match tail {
        "" | "だ" | "です" | "ではある" | "ではあります" => "是",
        "でした" | "だった" | "ではあった" | "ではありました" => "當時是",
        "ではなく"
        | "じゃなく"
        | "じゃない"
        | "ではない"
        | "じゃないです"
        | "ではないです"
        | "じゃありません"
        | "ではありません" => "不是",
        "じゃなかった"
        | "ではなかった"
        | "じゃなかったです"
        | "ではなかったです"
        | "じゃありませんでした"
        | "ではありませんでした" => "當時不是",
        _ => return None,
    };
    let prefix = &body[..=at];
    let topic_case = prefix.iter().position(|a| {
        !a.is_locked && a.deprel == "case" && matches!(a.surface.as_str(), "は" | "が" | "も")
    });
    let (topic, owner_phrase) = if let Some(i) = topic_case {
        let topic = &prefix[..i];
        let head = topic.last()?;
        if head.doc_tok_idx != prefix[i].head || head.deprel != "nsubj" {
            return None;
        }
        (topic, &prefix[i + 1..])
    } else {
        if tail.is_empty() {
            return None;
        }
        (&[][..], prefix)
    };
    let modified_topic = modified_noun(topic, doc, chars);
    let modified_owner = modified_noun(owner_phrase, doc, chars);
    let topic = modified_topic.as_deref().unwrap_or(topic);
    let owner_phrase = modified_owner.as_deref().unwrap_or(owner_phrase);
    if !topic.iter().chain(owner_phrase).all(|a| {
        owner_atom(a, doc)
            || !a.is_locked
                && (matches!(a.upos, Some(Upos::Det | Upos::Num))
                    || matches!(a.surface.as_str(), "の" | "と") && a.deprel == "case")
    }) {
        return None;
    }
    let mut subject = match topic {
        [] => String::new(),
        [a] if a.surface == "これ" => "這".to_owned(),
        [a] if matches!(a.surface.as_str(), "それ" | "あれ") => "那".to_owned(),
        [det, noun]
            if !det.is_locked
                && det.head == noun.doc_tok_idx
                && (matches!(noun.surface.as_str(), "鍵" | "傘")
                    || noun.is_locked && whole_nominal_lock(noun, doc))
                && matches!(det.surface.as_str(), "この" | "その" | "あの") =>
        {
            let mut source = noun.clone();
            source.surface = chars
                .get(noun.span.start_char..noun.span.end_char)?
                .iter()
                .collect();
            source.is_locked = false;
            source.upos = doc.tokens.get(noun.doc_tok_idx?)?.upos;
            let word = super::translate_single_ja_term(&source.surface, None, source.upos);
            let classifier = noun_classifier(&word, &source);
            format!(
                "{}{classifier}{}",
                if det.surface == "この" {
                    "這"
                } else {
                    "那"
                },
                if noun.is_locked { &noun.surface } else { &word }
            )
        }
        _ => noun_phrase(topic, doc, sentence, target)?,
    };
    if topic_case.is_some_and(|i| prefix[i].surface == "も") {
        subject.push('也');
    }
    let owner_text = elliptical_owner(owner_phrase, doc, sentence, target)?;
    let wh = prefix.iter().any(|a| {
        let raw = if a.is_locked {
            chars
                .get(a.span.start_char..a.span.end_char)
                .map(|s| s.iter().collect::<String>())
        } else {
            None
        };
        matches!(
            raw.as_deref().unwrap_or(&a.surface),
            "誰" | "だれ" | "どなた" | "何" | "どれ" | "どの" | "どこ" | "どちら"
        )
    });
    let ending = if wh && ending == "嗎" { "" } else { ending };
    Some(format!("{subject}{copula}{owner_text}的{ending}"))
}

pub(super) fn render(
    atoms: &[Atom],
    doc: &Document,
    sentence: usize,
    target: TargetVariant,
    chars: &[char],
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
    if let [predicate @ .., cause] = body
        && !cause.is_locked
        && matches!(cause.surface.as_str(), "ので" | "から")
        && cause.deprel == "mark"
        && predicate.iter().any(|a| {
            a.doc_tok_idx == cause.head
                && matches!(a.upos, Some(Upos::Verb | Upos::Adj | Upos::Aux))
        })
        && let Some(text) = render_ellipsis(predicate, "", doc, sentence, target, chars)
    {
        return Some(format!("因為{text}"));
    }
    if let [predicate @ .., conjunction] = body
        && !conjunction.is_locked
        && matches!(
            conjunction.surface.as_str(),
            "が" | "けど" | "けれど" | "けれども"
        )
        && matches!(conjunction.deprel.as_str(), "case" | "mark")
        && predicate.iter().any(|a| a.doc_tok_idx == conjunction.head)
        && let Some(text) = render_ellipsis(predicate, "", doc, sentence, target, chars)
    {
        return Some(format!("雖然{text}"));
    }
    if let Some(i) = body
        .iter()
        .position(|a| !a.is_locked && a.surface == "なく")
        && let Some(left) = render_ellipsis(&body[..=i], "", doc, sentence, target, chars)
        && let Some(right) = render_ellipsis(&body[i + 1..], ending, doc, sentence, target, chars)
    {
        return Some(format!("{left}，而{right}"));
    }
    if let Some(text) = render_ellipsis(body, ending, doc, sentence, target, chars) {
        return Some(text);
    }
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
        || tail.iter().any(|a| a.is_locked)
        || nominals
            .iter()
            .any(|a| a.is_locked && !whole_nominal_lock(a, doc))
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
    fn elliptical_possession_keeps_its_subject_owner_and_polarity() {
        for (source, expected) in [
            ("その黒い傘は店長のですか。", "那把黑色的傘是店長的嗎？"),
            ("この赤い本は姉のです。", "這本紅色的書是姊姊的。"),
            (
                "この赤い本は弟のではありません。",
                "這本紅色的書不是弟弟的。",
            ),
            ("古い本は誰のですか。", "舊的書是誰的？"),
            ("これは若い先生のです。", "這是年輕的老師的。"),
            ("この静かな部屋は友達のです。", "這間安靜的房間是朋友的。"),
            (
                "あの大きな古い鞄は先生のです。",
                "那個又大又舊的包包是老師的。",
            ),
            ("このとても古い本は誰のですか。", "這本非常舊的書是誰的？"),
            ("どの古い本が学校のですか。", "哪本舊的書是學校的？"),
            ("この本は私のです。", "這本書是我的。"),
            ("その本は誰のですか。", "那本書是誰的？"),
            ("この鍵は私のものではありません。", "這把鑰匙不是我的。"),
            ("この本はあなたのですか。", "這本書是你的嗎？"),
            ("その本は先生のでした。", "那本書當時是老師的。"),
            ("これは私のではなかったです。", "這當時不是我的。"),
            ("これは母と父のです。", "這是母親和父親的。"),
            ("そのカメラは田中さんのです。", "那相機是田中的。"),
            ("その本は先生の友達のです。", "那本書是老師的朋友的。"),
            ("それも母のです。", "那也是母親的。"),
            (
                "これは彼のものではあるが、私のものではない。",
                "雖然這是他的，不是我的。",
            ),
            (
                "本は母のものではなく、父のものです。",
                "書不是母親的，是父親的。",
            ),
            (
                "「この鍵は私のものではない」と先生が言いました。",
                "「這把鑰匙不是我的」老師說了。",
            ),
            (
                "この本は私のです。その本は妹のではありません。",
                "這本書是我的。那本書不是妹妹的。",
            ),
            ("父は今日帰るのです。", "父親今天回去。"),
            ("先生の本なので読みました。", "因為老師的書，讀了。"),
        ] {
            assert_eq!(translate(source), expected, "{source}");
        }
        for source in [
            "人の物を取るものではない。",
            "これは私が書いたものではありません。",
            "この赤い本は私が書いたのです。",
            "この古い本を妹が読みました。",
        ] {
            let result = translate(source);
            assert!(
                !result.contains("不是人的") && !result.contains("不是我的"),
                "{source}: {result}"
            );
        }
    }

    #[test]
    fn possessive_copula_preserves_a_caller_named_object_once() {
        let mut glossary = TranslationGlossary::new();
        for (source, translation) in [("鍵", "銅鑰匙"), ("月灯箱", "月光信匣")] {
            glossary.objects.push(crate::GlossaryEntry {
                canonical: source.to_owned(),
                surfaces: vec![source.to_owned()],
                targets: [(TargetVariant::ZhHant, translation.to_owned())]
                    .into_iter()
                    .collect(),
                ..Default::default()
            });
        }
        for (source, expected) in [
            ("これは会社の鍵です。", "這是公司的銅鑰匙。"),
            ("それは会社の鍵ではありません。", "那不是公司的銅鑰匙。"),
            ("それは会社の鍵ですか。", "那是公司的銅鑰匙嗎？"),
            ("それは会社の鍵ですか！", "那是公司的銅鑰匙嗎！"),
            ("この鍵は私のものではありません。", "這把銅鑰匙不是我的。"),
            ("この鍵は誰のですか。", "這把銅鑰匙是誰的？"),
            (
                "これは学校の月灯箱ではありません。",
                "這不是學校的月光信匣。",
            ),
        ] {
            let result = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert_eq!(result.target_text, expected, "{source}");
            assert_eq!(result.glossary_hits.len(), 1);
        }
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
