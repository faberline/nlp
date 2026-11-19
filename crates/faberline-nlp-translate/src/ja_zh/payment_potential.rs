//! Resolve the potential of 払う from its arguments within one clause.

use super::*;

fn potential(atom: &Atom) -> bool {
    !atom.is_locked
        && atom.upos == Some(Upos::Verb)
        && matches!(
            atom.lemma.as_deref().unwrap_or(&atom.surface),
            "支払える" | "払える"
        )
}

// A quoted conjugation can be either an utterance or the Japanese spelling.
// Resolve the role from the surrounding report. A bare form has no monetary
// assertion of its own when its use is unclear.
pub(super) fn quoted_clause(
    inner: &[Atom],
    clause: &[Atom],
    previous: Option<&Segment>,
    after: &[Segment],
    chars: &[char],
    target: TargetVariant,
) -> Option<String> {
    let word = clause.iter().find(|atom| potential(atom))?;
    if reported_content(word, previous, after) {
        let (body, suffix) = match clause.split_last() {
            Some((p, body)) if !p.is_locked && p.upos == Some(Upos::Part) => {
                match p.surface.as_str() {
                    "よ" => (body, "喔"),
                    "ね" | "よね" => (body, "呢"),
                    _ => (clause, ""),
                }
            }
            _ => (clause, ""),
        };
        return render(body, target, true).map(|mut text| {
            text.push_str(suffix);
            text
        });
    }
    let (first, tail) = core_non_punct(inner).split_first()?;
    if !potential(first)
        || !tail
            .iter()
            .all(|a| !a.is_locked && a.upos == Some(Upos::Aux) && a.head == first.doc_tok_idx)
    {
        return None;
    }
    let end = tail.last().unwrap_or(first).span.end_char;
    Some(chars[first.span.start_char..end].iter().collect())
}

fn answer_noun(atom: &Atom) -> bool {
    !atom.is_locked
        && matches!(
            atom.surface.as_str(),
            "返事" | "返答" | "回答" | "返信" | "応答" | "答え"
        )
}

// A stop inside the quotation can close a document sentence just before
// the reporting と. Read only that adjacent report, on the same source line.
pub(super) fn adjacent_report(
    inner: &[Atom],
    span: Span,
    doc: &Document,
    sentence_idx: usize,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<Vec<Segment>> {
    let next = doc.segmentation.sentences.get(sentence_idx + 1)?.span;
    if !inner.iter().any(potential)
        || doc.segmentation.sentences[sentence_idx].span.end_char != span.end_char
        || !chars[span.end_char..next.start_char]
            .iter()
            .all(|c| matches!(c, ' ' | '　'))
        || !matches!(chars.get(next.start_char), Some('と' | 'っ'))
    {
        return None;
    }
    let tokens = doc
        .tokens
        .iter()
        .filter(|t| t.sentence == sentence_idx + 1)
        .collect::<Vec<_>>();
    let atoms = normalized_sentence_atoms(sentence_idx + 1, chars, doc, &tokens, locks);
    Some(split_atoms_into_segments(&atoms, doc, next))
}

fn reported_content(word: &Atom, previous: Option<&Segment>, after: &[Segment]) -> bool {
    let before = match previous {
        Some(Segment::Clause {
            atoms,
            trailing_punct: None,
            ..
        }) => atoms.as_slice(),
        _ => &[],
    };
    let next = match after.first() {
        Some(Segment::Clause { atoms, .. }) => atoms.as_slice(),
        _ => &[],
    };
    if before.last().is_some_and(|n| {
        !n.is_locked
            && n.head == word.doc_tok_idx
            && n.span.end_char + 1 == word.span.start_char
            && matches!(
                n.surface.as_str(),
                "語" | "単語" | "言葉" | "用語" | "動詞" | "表現"
            )
    }) {
        return false;
    }
    // In 返信は「…」でした the quotation is the value of the answer topic.
    let reply_topic = matches!(before, [.., noun, topic] if answer_noun(noun)
        && !topic.is_locked && matches!(topic.surface.as_str(), "は" | "が")
        && topic.head == noun.doc_tok_idx);
    if reply_topic
        && (next.is_empty() || matches!(joined(next).as_str(), "だ" | "です" | "だった" | "でした"))
    {
        return true;
    }
    let Some((marker, rest)) = next.split_first() else {
        return false;
    };
    if marker.is_locked {
        return false;
    }
    // A continuative answer before a comma can be tagged as an answer noun
    // instead of the verb stem, as in 「…」と答え、次に… .
    if marker.surface == "と"
        && matches!(rest, [reply] if answer_noun(reply))
        && matches!(after.first(), Some(Segment::Clause { trailing_punct: Some(punct), .. })
            if matches!(punct.as_str(), "、" | "，"))
    {
        return true;
    }
    if marker.surface == "と"
        && matches!(rest, [no, reply, ..] if !no.is_locked
        && no.surface == "の" && no.head == marker.doc_tok_idx && answer_noun(reply)
        && marker.head == reply.doc_tok_idx)
    {
        return true;
    }
    match marker.surface.as_str() {
        "という" => {
            // A quotation gives the content of an answer or its asserted meaning.
            rest.first().is_some_and(answer_noun)
                || matches!(rest, [meaning, tail @ ..]
                if !meaning.is_locked && meaning.surface == "意味" && !tail.is_empty()
                    && tail.iter().all(|p| !p.is_locked && matches!(p.surface.as_str(), "だ" | "です" | "よ" | "ね")))
        }
        "と" | "って" | "を" => {
            let Some(report) = rest.iter().find(|a| {
                !a.is_locked
                    && a.upos == Some(Upos::Verb)
                    && a.doc_tok_idx.is_some()
                    && a.doc_tok_idx == word.head.or(marker.head)
            }) else {
                return false;
            };
            let lemma = report.lemma.as_deref().unwrap_or(&report.surface);
            if marker.surface == "を" {
                return lemma == "聞く";
            }
            let attached_answer = |noun: &Atom| {
                answer_noun(noun)
                    && (noun.head == report.doc_tok_idx || noun.head == word.doc_tok_idx)
            };
            if lemma == "する" && rest.iter().any(attached_answer)
                || (answer_noun(report)
                    && rest.iter().any(|v| {
                        v.doc_tok_idx == report.head
                            && !v.is_locked
                            && v.lemma.as_deref() == Some("する")
                    }))
            {
                return true;
            }
            match lemma {
                "言う" | "いう" | "答える" | "話す" | "伝える" | "告げる" | "聞く" | "返事する"
                | "返信する" | "返答する" | "回答する" | "応答する" => true,
                "書く" => {
                    // Writing can either copy characters or communicate a reply.
                    // Look for the answer attached to this writing event (or its
                    // next て-linked verb), not an unrelated later statement.
                    before.iter().chain(rest).any(attached_answer)
                        || (rest.last().is_some_and(|a| {
                            !a.is_locked && a.surface == "て" && a.head == report.doc_tok_idx
                        }) && matches!(after.get(1), Some(Segment::Clause { atoms, .. })
                                if atoms.iter().any(|a| answer_noun(a) && a.head == report.head)))
                }
                _ => false,
            }
        }
        _ => false,
    }
}

pub(super) fn ends_in_question(inner: &[Atom]) -> bool {
    inner.last().is_some_and(|q| {
        q.surface == "か"
            && q.head.is_some()
            && inner
                .iter()
                .any(|v| potential(v) && v.doc_tok_idx == q.head)
    })
}

pub(super) fn reported_question(
    before: &[Segment],
    inner: &[Atom],
    report: &[Atom],
    doc: &Document,
    locks: &SpanLockTable,
) -> Option<String> {
    let [
        Segment::Clause {
            atoms: who,
            trailing_punct: None,
            ..
        },
    ] = before
    else {
        return None;
    };
    let [marker, hear, tail @ ..] = report else {
        return None;
    };
    if report.iter().any(|a| a.is_locked)
        || marker.surface != "と"
        || marker.head.is_none()
        || marker.head != hear.doc_tok_idx
        || hear.lemma.as_deref().unwrap_or(&hear.surface) != "聞く"
        || !matches!(joined(tail).as_str(), "" | "ます" | "ました" | "た")
    {
        return None;
    }
    let question = inner.last()?;
    if question.surface != "か"
        || question.head.is_none()
        || !inner
            .iter()
            .any(|v| potential(v) && v.doc_tok_idx == question.head && v.head == hear.doc_tok_idx)
    {
        return None;
    }
    let (case, speaker) = who.split_last()?;
    let head = speaker.last()?;
    if !matches!(case.surface.as_str(), "は" | "が")
        || case.head.is_none()
        || case.head != head.doc_tok_idx
        || head.deprel != "nsubj"
    {
        return None;
    }
    let who = render_direct_quote_person(speaker, doc, locks).or_else(|| {
        // A topic outside the quote is the speaker even when the parser
        // attaches its subject edge to the inside of the quote.
        (speaker
            .iter()
            .all(|a| !a.is_locked && a.upos == Some(Upos::Noun))
            && (head.head == question.head || head.head == hear.doc_tok_idx))
            .then(|| nominal(speaker, TargetVariant::ZhHant))
    })?;
    Some(format!("{who}問："))
}

#[derive(Clone, Copy)]
enum Sense {
    Money,
    Brush,
    Respect,
    Attention,
}

fn joined(atoms: &[Atom]) -> String {
    atoms.iter().map(|a| a.surface.as_str()).collect()
}

fn nominal(words: &[Atom], target: TargetVariant) -> String {
    let atoms = normalize_ja_atoms(&apply_phrase_matches(words));
    let group = Bunsetsu {
        atoms,
        role: BunsetsuRole::Other,
        case_particle: None,
        sconj_tail: None,
        sfp_tail: None,
    };
    render_bunsetsu_np(&group, target)
}

fn dust(a: &Atom) -> bool {
    !a.is_locked && matches!(a.surface.as_str(), "ほこり" | "埃" | "ホコリ")
}

fn attention_target(words: &[Atom], target: TargetVariant) -> String {
    let mut localized = Vec::new();
    let mut iter = words.iter().peekable();
    while let Some(word) = iter.next() {
        let mut word = word.clone();
        if word.surface == "子"
            && iter.peek().is_some_and(|suffix| {
                suffix.surface == "ども"
                    && suffix.head == word.doc_tok_idx
                    && suffix.span.start_char == word.span.end_char
            })
        {
            word.surface = "子供".to_owned();
            word.span.end_char = iter.next().unwrap().span.end_char;
        } else if word.surface == "動き" && word.deprel == "obl" {
            word.surface = "動作".to_owned();
            word.is_locked = true;
        }
        localized.push(word);
    }
    nominal(&localized, target)
}

// Only the head and its contiguous compound select a sense. A noun in a
// modifier of the object cannot supply the reading of the verb.
fn object(
    words: &[Atom],
    target: TargetVariant,
    explicit_payment: bool,
) -> Option<(Sense, String)> {
    let head = words.last()?;
    let mut start = words.len() - 1;
    while start > 0
        && words[start - 1].deprel == "compound"
        && words[start - 1].head == head.doc_tok_idx
        && words[start - 1].span.end_char == words[start].span.start_char
    {
        start -= 1;
    }
    let name = joined(&words[start..]);
    let (sense, zh) = match name.as_str() {
        "家賃" => (Sense::Money, "房租"),
        "代金" => (Sense::Money, "貨款"),
        "電気代" => (Sense::Money, "電費"),
        "授業料" => (Sense::Money, "學費"),
        "入場料" => (Sense::Money, "入場費"),
        "料金" | "費用" => (Sense::Money, "費用"),
        "ほこり" | "埃" | "ホコリ" if !explicit_payment => (Sense::Brush, "灰塵"),
        "雪" if !explicit_payment => (Sense::Brush, "雪"),
        "敬意" if !explicit_payment && start == 0 => (Sense::Respect, "敬意"),
        "注意" if !explicit_payment && start == 0 => (Sense::Attention, ""),
        _ if explicit_payment || head.surface == "費" => {
            return Some((Sense::Money, nominal(words, target)));
        }
        _ => return None,
    };
    let modifiers = &words[..start];
    let mut text = match (sense, modifiers) {
        (Sense::Brush, [coat, genitive])
            if coat.surface == "上着"
                && genitive.surface == "の"
                && genitive.head == coat.doc_tok_idx =>
        {
            "外套上的".to_owned()
        }
        _ => nominal(modifiers, target),
    };
    text.push_str(zh);
    Some((sense, text))
}

// Keep the ordinary assembly path for clauses with a more complex adjunct.
// Only specialize its verb when an explicit monetary object selects the sense.
pub(super) fn localize_money_verb(atoms: &mut [Atom], source: &[Atom], target: TargetVariant) {
    for verb in atoms
        .iter_mut()
        .filter(|a| potential(a) && a.lemma.as_deref().unwrap_or(&a.surface) == "払える")
    {
        let Some(governor) = verb.doc_tok_idx else {
            continue;
        };
        let financial = source.windows(2).enumerate().any(|(i, pair)| {
            let noun = &pair[0];
            let case = &pair[1];
            if noun.is_locked
                || case.is_locked
                || noun.head != Some(governor)
                || noun.deprel != "obj"
                || case.surface != "を"
                || case.deprel != "case"
                || case.head.is_none()
                || case.head != noun.doc_tok_idx
            {
                return false;
            }
            let mut begin = i;
            while begin > 0
                && source[begin - 1].head == noun.doc_tok_idx
                && source[begin - 1].deprel == "compound"
                && !source[begin - 1].is_locked
            {
                begin -= 1;
            }
            matches!(
                object(&source[begin..=i], target, false),
                Some((Sense::Money, _))
            )
        });
        if financial {
            verb.surface = "能付".to_owned();
            verb.is_locked = true;
        }
    }
}

#[derive(Default)]
struct Arguments<'a> {
    subject: Option<String>,
    places: String,
    instrument: Option<String>,
    goal: Option<&'a [Atom]>,
    object: Option<(Sense, String)>,
}

impl<'a> Arguments<'a> {
    fn read(
        mut words: &'a [Atom],
        governor: usize,
        target: TargetVariant,
        explicit_payment: bool,
    ) -> Option<Self> {
        let mut args = Self::default();
        let dust_object = words
            .iter()
            .find(|a| dust(a) && a.deprel == "obj" && a.head == Some(governor))
            .and_then(|a| a.doc_tok_idx);
        while !words.is_empty() {
            let i = words.windows(2).position(|pair| {
                let [noun, case] = pair else { unreachable!() };
                case.upos == Some(Upos::Adp)
                    && case.deprel == "case"
                    && case.head.is_some()
                    && matches!(
                        case.surface.as_str(),
                        "は" | "が" | "を" | "で" | "に" | "から"
                    )
                    && noun.doc_tok_idx == case.head
                    && (noun.head == Some(governor)
                        || (case.surface == "で"
                            && noun.head.is_some()
                            && noun.head == dust_object
                            && noun.deprel == "obl"
                            && matches!(noun.surface.as_str(), "ブラシ" | "刷毛")))
            })? + 1;
            let phrase = &words[..i];
            let head = phrase.last()?;
            let case = &words[i];
            words = &words[i + 1..];
            // Keep only modifiers belonging to this argument. Do not absorb
            // another predicate, argument or clause boundary.
            if phrase[..phrase.len() - 1].iter().enumerate().any(|(i, a)| {
                a.upos == Some(Upos::Punct)
                    || a.head.is_none()
                    || !(phrase.iter().any(|parent| parent.doc_tok_idx == a.head)
                        || (dust(head)
                            && a.head == Some(governor)
                            && a.deprel == "nmod"
                            && phrase
                                .get(i + 1)
                                .is_some_and(|no| no.surface == "の" && no.head == a.doc_tok_idx))
                        || (i == 0
                            && matches!(a.surface.as_str(), "お" | "ご")
                            && a.deprel == "dep"
                            && a.head == Some(governor)
                            && a.span.end_char == phrase[i + 1].span.start_char))
            }) {
                return None;
            }
            let focused = words
                .first()
                .is_some_and(|a| a.surface == "は" && a.deprel == "case" && a.head == case.head);
            if focused {
                words = &words[1..];
            }
            let text = || nominal(phrase, target);
            match (case.surface.as_str(), head.deprel.as_str()) {
                ("を", "obj") if args.object.is_none() => {
                    args.object = Some(object(phrase, target, explicit_payment)?)
                }
                ("が", "nsubj")
                    if args.object.is_none() && object(phrase, target, false).is_some() =>
                {
                    args.object = object(phrase, target, false)
                }
                ("は" | "が", "nsubj") if args.subject.is_none() => {
                    args.subject = Some(if joined(phrase) == "お客さん" {
                        "客人".to_owned()
                    } else {
                        text()
                    });
                }
                ("で", "obl" | "nsubj") if head.deprel == "obl" || focused => {
                    let brush =
                        dust_object.is_some() && matches!(head.surface.as_str(), "ブラシ" | "刷毛");
                    let instrument = is_payment_medium(head, phrase.iter().rev().nth(1))
                        || matches!(head.surface.as_str(), "スマホ" | "携帯")
                        || brush;
                    if instrument {
                        if args.instrument.is_some() {
                            return None;
                        }
                        let label = match phrase {
                            [det, noun]
                                if let Some(demonstrative) = attached_demonstrative(det, noun)
                                    && (noun.surface == "カード" || brush) =>
                            {
                                format!("{demonstrative}{}", if brush { "支刷子" } else { "張卡" })
                            }
                            _ if brush => "刷子".to_owned(),
                            _ => text(),
                        };
                        args.instrument = Some(format!("用{label}"));
                    } else {
                        let venue = match joined(phrase).as_str() {
                            "窓口" => "窗口".to_owned(),
                            "売店" => "小賣店".to_owned(),
                            _ => text(),
                        };
                        args.places.push_str(&format!("在{venue}"));
                    }
                }
                ("に", "obl" | "iobj") if args.goal.is_none() => args.goal = Some(phrase),
                ("から", "obl") => args.places.push_str(&format!("從{}", text())),
                _ => return None,
            }
        }
        Some(args)
    }
}

pub(super) fn render(source: &[Atom], target: TargetVariant, reported: bool) -> Option<String> {
    let verb_index = source.iter().position(potential)?;
    let verb = &source[verb_index];
    let governor = verb.doc_tok_idx?;
    if source.iter().any(|a| a.is_locked) {
        return None;
    }
    let tail = &source[verb_index + 1..];
    let (tail, question) = match tail.split_last() {
        Some((q, t)) if q.surface == "か" && q.head == Some(governor) => (t, true),
        _ => (tail, false),
    };
    if tail
        .iter()
        .any(|a| a.head != Some(governor) || a.upos != Some(Upos::Aux))
    {
        return None;
    }
    let modal = match joined(tail).as_str() {
        "" | "ます" => "能",
        "ない" | "ません" => "不能",
        // The past inflection supplies no lexical time adverb. An explicit
        // date or time belongs to its own argument (possibly before a comma).
        "た" | "ました" => "有辦法",
        "なかった" | "ませんでした" => "沒能",
        _ => return None,
    };
    let explicit_payment = verb.lemma.as_deref().unwrap_or(&verb.surface) == "支払える";
    let args = Arguments::read(&source[..verb_index], governor, target, explicit_payment)?;
    let (sense, obj) = match args.object {
        Some(pair) => pair,
        None if explicit_payment
            && (reported
                || args.instrument.is_some()
                || lookup_ja_lexicon_gloss(&verb.surface, verb.lemma.as_deref(), verb.upos)
                    .is_some()) =>
        {
            (Sense::Money, String::new())
        }
        _ => return None,
    };
    let action = match (sense, args.goal) {
        (Sense::Money, None) if obj.is_empty() => "付款".to_owned(),
        (Sense::Money, None) => format!("付{obj}"),
        (Sense::Money, Some(who)) if !obj.is_empty() && who.iter().any(is_person_like_atom) => {
            format!("付{obj}給{}", nominal(who, target))
        }
        (Sense::Brush, None) => format!("掃掉{obj}"),
        (Sense::Respect, who) => format!(
            "{}表達{obj}",
            who.map(|words| format!("向{}", nominal(words, target)))
                .unwrap_or_default()
        ),
        (Sense::Attention, scope) => format!(
            "留意{}",
            scope
                .map(|words| attention_target(words, target))
                .unwrap_or_default()
        ),
        _ => return None,
    };
    Some(format!(
        "{}{}{modal}{}{action}{}",
        args.subject.unwrap_or_default(),
        args.places,
        args.instrument.unwrap_or_default(),
        if question { "嗎" } else { "" }
    ))
}

#[cfg(test)]
mod tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

    fn translate(jp: &str) -> String {
        let mut options = TranslateOptions::new(TargetVariant::ZhHant);
        options.source_hint = Some(TargetVariant::Ja);
        analyze_and_translate(
            jp,
            &options,
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap()
        .target_text
    }

    #[test]
    fn case_marked_object_selects_the_sense_and_keeps_polarity() {
        for (jp, expected) in [
            ("母は今月の家賃が払えません。", "母親不能付這個月的房租。"),
            ("入場料を払えましたか。", "有辦法付入場費嗎？"),
            ("当時、家賃を払えました。", "當時，有辦法付房租。"),
            ("昨日はカードで支払えませんでした。", "昨天沒能用卡片付款。"),
            ("弟は姉に家賃を払えます。", "弟弟能付房租給姊姊。"),
            ("帽子の埃を払えます。", "能掃掉帽子的灰塵。"),
            ("机のほこりを払えません。", "不能掃掉書桌的灰塵。"),
            ("相手に敬意を払えますか。", "能向對方表達敬意嗎？"),
            ("子どもの動きに注意を払えません。", "不能留意孩子的動作。"),
            ("花を払えますか。", "払え花嗎？"),
        ] {
            assert_eq!(translate(jp), expected, "{jp}");
        }
    }

    #[test]
    fn medium_and_venue_have_different_roles() {
        for (jp, expected) in [
            ("このカードでは支払えませんか。", "不能用這張卡付款嗎？"),
            ("窓口では現金で支払えますか。", "在窗口能用現金付款嗎？"),
            (
                "このブラシで上着のほこりを払えますか。",
                "能用這支刷子掃掉外套上的灰塵嗎？",
            ),
            ("このカードでは遊べますか。", "在這卡片能玩嗎？"),
        ] {
            assert_eq!(translate(jp), expected, "{jp}");
        }
    }

    #[test]
    fn terms_and_quoted_questions_keep_their_scope() {
        for (jp, expected) in [
            ("「支払える」は動詞です。", "「支払える」是動詞。"),
            (
                "店員は「このカードで支払えますか」と聞きました。",
                "店員問：「能用這張卡付款嗎？」",
            ),
            (
                "先生は「家賃を払えますか」と聞きました。",
                "老師問：「能付房租嗎？」",
            ),
            ("二人は家賃の話を聞きました。", "二人聽了家賃的話。"),
        ] {
            assert_eq!(translate(jp), expected, "{jp}");
        }
    }

    #[test]
    fn naming_and_dictionary_lookup_preserve_a_cited_word() {
        for jp in [
            "「支払える」という動詞を調べました。",
            "先生は「支払える」という言葉を書きました。",
            "「支払える」を辞書で調べます。",
            "辞書で「支払える」を調べました。",
            "「支払える」を辞典で引きました。",
            "「支払える」の用例を二つ探しました。",
            "「支払える」とはどういう意味ですか。",
        ] {
            assert!(translate(jp).contains("「支払える」"), "{jp}");
        }
        for jp in [
            "「支払える」という返事を聞きました。",
            "彼は「支払える」と言いました。",
        ] {
            assert!(translate(jp).contains("「能付款」"), "{jp}");
        }
    }

    #[test]
    fn written_forms_and_reported_answers_use_different_quotation_rules() {
        for (jp, quotes) in [
            (
                "日本語の単語「支払える」をノートに書きました。",
                "「支払える」",
            ),
            ("黒板の「支払える」に下線を引きました。", "「支払える」"),
            (
                "編集者は「支払える」の送り仮名を確認しました。",
                "「支払える」",
            ),
            (
                "「支払える」と「支払えない」は、意味が反対です。",
                "「支払える」「支払えない」",
            ),
            (
                "生徒は「支払える」と「払える」の表記を見比べました。",
                "「支払える」「払える」",
            ),
            ("先生は「支払える」と発音しました。", "「支払える」"),
            ("先生は単語「支払える」を聞いて覚えました。", "「支払える」"),
            ("ノートの「支払える」を赤で囲みました。", "「支払える」"),
            ("紙には「支払えない」。", "「支払えない」"),
            (
                "料金について、客は「支払えない」と答えました。",
                "「不能付款」",
            ),
            ("受付は「支払える」とメールで返信しました。", "「能付款」"),
            ("返事は「支払える」でした。", "「能付款」"),
            ("「支払える」との回答です。", "「能付款」"),
            ("彼は「支払える」と返事をしました。", "「能付款」"),
            ("電話では「支払える」と返答してくれました。", "「能付款」"),
            ("彼は「支払える。」と答えました。", "「能付款。」"),
            ("彼は紙に「支払える」と書いて返事をしました。", "「能付款」"),
            ("「支払える」を聞いて安心しました。", "「能付款」"),
        ] {
            let actual = translate(jp);
            assert!(actual.contains(quotes), "{jp} -> {actual}");
        }
        let mixed = translate("店長は「支払える」と答え、その単語「支払える」をノートに書いた。");
        let (_, following) = mixed.split_once("「能付款」").expect(&mixed);
        assert!(following.contains("「支払える」"), "{mixed}");
    }
}
