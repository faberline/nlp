//! Resolve payment and potential 払う from their source arguments.

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

fn answer_verb(atom: &Atom) -> bool {
    !atom.is_locked
        && atom.upos == Some(Upos::Verb)
        && matches!(
            atom.lemma.as_deref(),
            Some("答える" | "返事する" | "返信する" | "返答する" | "回答する" | "応答する")
        )
}

fn nominal_answer_report(rest: &[Atom], word: &Atom) -> bool {
    let rest = &rest[rest
        .iter()
        .take_while(|a| !a.is_locked && a.upos == Some(Upos::Adv))
        .count()..];
    match rest {
        // 答えた is a finite reply even when the parser tags the stem as a noun,
        // including relative clauses such as 答えた声.
        [stem, past, ..]
            if !stem.is_locked
                && stem.surface == "答え"
                && !past.is_locked
                && past.upos == Some(Upos::Aux)
                && past.surface == "た"
                && past.head == stem.doc_tok_idx =>
        {
            true
        }
        // The quotation supplies the content of the reply that came or existed.
        [reply, case, ..]
            if answer_noun(reply)
                && !case.is_locked
                && matches!(case.surface.as_str(), "が" | "は")
                && case.head == reply.doc_tok_idx =>
        {
            rest.iter().any(|v| {
                !v.is_locked
                    && v.upos == Some(Upos::Verb)
                    && v.doc_tok_idx == reply.head
                    && v.doc_tok_idx == word.head
                    && matches!(v.lemma.as_deref(), Some("ある" | "来る" | "届く"))
            })
        }
        _ => false,
    }
}

// A writing verb can copy characters. An answer field tied to the payment
// question supplies the missing evidence for the content reading.
fn payment_answer_field(before: &[Atom], word: &Atom, writing: &Atom) -> bool {
    let [field_atoms @ .., field, ni] = before else {
        return false;
    };
    if ni.is_locked
        || ni.surface != "に"
        || ni.head != field.doc_tok_idx
        || field.is_locked
        || field.surface != "欄"
        || (field.head != word.doc_tok_idx && field.head != writing.doc_tok_idx)
    {
        return false;
    }
    let Some(answer) = field_atoms
        .iter()
        .find(|a| answer_noun(a) && a.head == field.doc_tok_idx)
    else {
        return false;
    };
    field_atoms.array_windows().any(|[verb, ka, to, question]| {
        potential(verb)
            && !ka.is_locked
            && ka.surface == "か"
            && ka.head == verb.doc_tok_idx
            && !to.is_locked
            && to.surface == "という"
            && !question.is_locked
            && matches!(question.surface.as_str(), "質問" | "問い")
            && (question.head == field.doc_tok_idx || question.head == answer.doc_tok_idx)
    })
}

fn modifies_voice(atoms: &[Atom], modifier: &Atom) -> bool {
    let Some(end) = atoms
        .iter()
        .position(|a| !a.is_locked && a.deprel == "obj" && a.head == modifier.doc_tok_idx)
    else {
        return false;
    };
    let start = compound_start(&atoms[..=end]);
    matches!(
        joined(&atoms[start..=end]).as_str(),
        "声" | "声量" | "語尾" | "語気" | "口調" | "調子"
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
    if matches!(marker.surface.as_str(), "と" | "って") && nominal_answer_report(rest, word) {
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
                // Hearing a sound does not by itself identify quoted speech.
                // A linked emotional response supplies evidence that the
                // listener reacted to the content of the answer.
                return lemma == "聞く"
                    && matches!(after.get(1), Some(Segment::Clause { atoms, .. })
                    if atoms.iter().any(|v| !v.is_locked && v.doc_tok_idx == report.head
                        && matches!(v.lemma.as_deref().unwrap_or(&v.surface),
                            "安心する" | "安堵する" | "喜ぶ" | "驚く")));
            }
            let attached_answer = |noun: &Atom| {
                answer_noun(noun)
                    && (noun.head == report.doc_tok_idx || noun.head == word.doc_tok_idx)
            };
            // In 語尾を強めて答える the first verb takes its own object. The
            // quotative particle attaches to the reply verb governing this
            // て-clause. Do not cross another report or a free-standing clause.
            if report.deprel == "advcl"
                && report.head == marker.head
                && modifies_voice(rest, report)
                && rest.last().is_some_and(|a| {
                    !a.is_locked && a.surface == "て" && a.head == report.doc_tok_idx
                })
                && matches!(after.get(1), Some(Segment::Clause { atoms, .. })
                    if atoms.first().is_some_and(|a| a.doc_tok_idx == report.head && answer_verb(a)))
            {
                return true;
            }
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
                "書く" | "記入" | "記入する" => {
                    // Writing can either copy characters or communicate a reply.
                    // Look for the answer attached to this writing event (or its
                    // next て-linked verb), not an unrelated later statement.
                    before.iter().chain(rest).any(attached_answer)
                        || payment_answer_field(before, word, report)
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

fn compound_start(words: &[Atom]) -> usize {
    let mut start = words.len() - 1;
    let head = &words[start];
    while start > 0
        && words[start - 1].deprel == "compound"
        && words[start - 1].head == head.doc_tok_idx
        && words[start - 1].span.end_char == words[start].span.start_char
    {
        start -= 1;
    }
    start
}

// Only the head and its contiguous compound select a sense. A noun in a
// modifier of the object cannot supply the reading of the verb.
fn object(
    words: &[Atom],
    target: TargetVariant,
    explicit_payment: bool,
) -> Option<(Sense, String)> {
    let head = words.last()?;
    let start = compound_start(words);
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

pub(crate) struct FareRoles {
    amount: Span,
    medium: Span,
    medium_with_case: Span,
    stem: Span,
    native_medium: &'static str,
    ending: FareEnding,
    report: Option<FareReport>,
}

impl FareRoles {
    fn render(&self, overrides: &[&crate::lock::LockedSpan]) -> Option<String> {
        let mut amount = "票款";
        let mut method = self.native_medium.to_owned();
        let mut action = None;
        for lock in overrides {
            if lock.source_span == self.amount {
                amount = &lock.placeholder;
            } else if lock.source_span == self.medium {
                method = format!("用{}", lock.placeholder);
            } else if lock.source_span == self.medium_with_case {
                method.clone_from(&lock.placeholder);
            } else if lock.source_span == self.stem
                && matches!(
                    lock.origin,
                    LockOrigin::Glossary {
                        kind: GlossaryKind::DomainTerm,
                        ..
                    }
                )
            {
                action = Some(lock.placeholder.as_str());
            } else {
                return None;
            }
        }
        if self.report.is_none() && self.ending.is_affirmative_present() {
            return Some(format!("{method}{}{amount}", action.unwrap_or("付")));
        }
        // Extended forms recover grammar for a protected stem. Keep the
        // ordinary renderer for statements that do not carry such a term.
        let action = action?;
        let marker = match (self.ending.negative, self.ending.past) {
            (true, true) => "沒有",
            (true, false) => "不",
            (false, true) => "已",
            (false, false) => "會",
        };
        let closing = if self.ending.question { "嗎？" } else { "" };
        Some(format!("{amount}{marker}{method}{action}{closing}"))
    }
}

pub(super) fn render_fare_statement(
    source: &[Atom],
    doc: &Document,
    sentence: usize,
    chars: &[char],
    locks: &SpanLockTable,
    unhinted_roles: &dyn Fn(Span) -> Option<FareRoles>,
) -> Option<(String, Option<&'static str>)> {
    let sentence_span = doc.segmentation.sentences.get(sentence)?.span;
    let span = atom_span(source)?;
    if !chars
        .get(sentence_span.start_char..span.start_char)?
        .iter()
        .all(|c| c.is_whitespace())
        || !chars
            .get(span.end_char..sentence_span.end_char)?
            .iter()
            .all(|c| matches!(c, '。' | '.' | '?' | '？') || c.is_whitespace())
    {
        return None;
    }
    let (mut roles, overrides) = recover_fare(source, doc, sentence, chars, locks, unhinted_roles)?;
    if roles.report.is_some() {
        return None;
    }
    roles.ending.question |= chars[span.end_char..sentence_span.end_char]
        .iter()
        .any(|c| matches!(c, '?' | '？'));
    let factuality = if roles.ending.question {
        Some("hypothetical")
    } else if roles.ending.negative {
        Some("negated")
    } else {
        None
    };
    Some((roles.render(&overrides)?, factuality))
}

pub(super) fn render_reported_fare(
    segments: &[Segment],
    doc: &Document,
    sentence: usize,
    chars: &[char],
    locks: &SpanLockTable,
    unhinted_roles: &dyn Fn(Span) -> Option<FareRoles>,
) -> Option<String> {
    let [
        Segment::Clause {
            atoms: prefix,
            trailing_punct: None,
            ..
        },
        Segment::Quote {
            open_q,
            inner_atoms,
            close_q,
            ..
        },
        Segment::Clause {
            atoms: suffix,
            trailing_punct,
            ..
        },
    ] = segments
    else {
        return None;
    };
    let inner = core_non_punct(inner_atoms);
    if inner_atoms.iter().any(is_quote_mark)
        || suffix.iter().any(|atom| atom.is_locked)
        || !matches!(trailing_punct.as_deref(), None | Some("。" | "."))
    {
        return None;
    }
    let (mut roles, overrides) = recover_fare(inner, doc, sentence, chars, locks, unhinted_roles)?;
    let report = roles.report.as_ref()?;
    let (case, speaker) = prefix.split_last()?;
    if !matches!(case.surface.as_str(), "は" | "が")
        || case.is_locked
        || atom_span(speaker)? != report.speaker
        || atom_span(core_non_punct(suffix))? != report.clause
    {
        return None;
    }
    let who = render_direct_quote_person(speaker, doc, locks).or_else(|| {
        // The source proof already established a speech subject. Some role
        // nouns have no person class in the supplied token analysis.
        speaker
            .iter()
            .all(|atom| {
                !atom.is_locked && matches!(atom.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron))
            })
            .then(|| nominal(speaker, TargetVariant::ZhHant))
    })?;
    let speech = if report.past { "曾說" } else { "說" };
    let inner_span = atom_span(inner)?;
    let inner_end = inner_atoms.last()?.span.end_char;
    roles.ending.question |= chars
        .get(inner_span.end_char..inner_end)?
        .iter()
        .any(|c| matches!(c, '?' | '？'));
    let spoken = roles.render(&overrides)?;
    let terminal = if spoken.ends_with('？') { "" } else { "。" };
    let open = localize_open_quote(*open_q, TargetVariant::ZhHant);
    let close = localize_close_quote(*close_q, TargetVariant::ZhHant);
    Some(format!("{who}{speech}：{open}{spoken}{terminal}{close}"))
}

fn atom_span(atoms: &[Atom]) -> Option<Span> {
    Some(Span {
        start_char: atoms.first()?.span.start_char,
        end_char: atoms.last()?.span.end_char,
    })
}

fn recover_fare<'a>(
    source: &[Atom],
    doc: &Document,
    sentence: usize,
    chars: &[char],
    locks: &'a SpanLockTable,
    unhinted_roles: &dyn Fn(Span) -> Option<FareRoles>,
) -> Option<(FareRoles, Vec<&'a crate::lock::LockedSpan>)> {
    let content = atom_span(source)?;
    let Span {
        start_char: start,
        end_char: end,
    } = content;
    if !chars.get(start..end)?.starts_with(&['切', '符', '代'])
        || doc
            .segmentation
            .opaque
            .iter()
            .any(|opaque| opaque.span.start_char < end && start < opaque.span.end_char)
    {
        return None;
    }
    let overrides: Vec<_> = locks
        .locks
        .iter()
        .filter(|lock| lock.source_span.start_char < end && start < lock.source_span.end_char)
        .collect();
    if overrides.is_empty() {
        return Some((fare_roles(source, None)?, overrides));
    }
    if overrides.iter().any(|lock| {
        !lock.mandatory
            || !matches!(
                lock.origin,
                LockOrigin::Glossary {
                    kind: GlossaryKind::Object | GlossaryKind::DomainTerm,
                    ..
                }
            )
            || !source.iter().any(|atom| {
                atom.is_locked && atom.span == lock.source_span && atom.surface == lock.placeholder
            })
    }) || source.iter().filter(|atom| atom.is_locked).count() != overrides.len()
    {
        return None;
    }

    // The lock can hide で. Prove the roles using the source analysis while
    // keeping the active placeholders and their original ranges.
    let roles = fare_roles_for_sentence(doc, sentence, chars, content)
        .or_else(|| unhinted_roles(content))?;
    Some((roles, overrides))
}

pub(crate) fn fare_roles_for_sentence(
    doc: &Document,
    sentence: usize,
    chars: &[char],
    content: Span,
) -> Option<FareRoles> {
    let first = doc
        .tokens
        .partition_point(|token| token.sentence < sentence);
    let last = doc
        .tokens
        .partition_point(|token| token.sentence <= sentence);
    let tokens: Vec<_> = doc.tokens[first..last].iter().collect();
    let original = build_sentence_atoms(sentence, chars, doc, &tokens, &SpanLockTable::default());
    let original = core_non_punct(&original);
    let first = original.partition_point(|atom| atom.span.end_char <= content.start_char);
    let last = original.partition_point(|atom| atom.span.start_char < content.end_char);
    let body = original.get(first..last)?;
    if atom_span(body)? != content {
        return None;
    }
    if first == 0 && last == original.len() {
        return fare_roles(body, None);
    }
    let (report, governor) = fare_report(&original[..first], &original[last..], body)?;
    let mut roles = fare_roles(body, Some(governor))?;
    roles.report = Some(report);
    Some(roles)
}

#[derive(Clone, Copy)]
struct FareEnding {
    negative: bool,
    past: bool,
    question: bool,
}

impl FareEnding {
    fn is_affirmative_present(self) -> bool {
        !self.negative && !self.past && !self.question
    }

    fn parse(verb: &Atom, suffix: &[Atom]) -> Option<Self> {
        let (aux, question) = match suffix.split_last() {
            Some((last, aux))
                if last.surface == "か"
                    && last.upos == Some(Upos::Part)
                    && last.deprel == "discourse"
                    && last.head == verb.doc_tok_idx =>
            {
                (aux, true)
            }
            _ => (suffix, false),
        };
        if suffix.first()?.span.start_char != verb.span.end_char
            || suffix
                .windows(2)
                .any(|pair| pair[0].span.end_char != pair[1].span.start_char)
            || suffix.iter().any(|atom| atom.is_locked)
            || aux.iter().any(|atom| {
                atom.upos != Some(Upos::Aux)
                    || atom.deprel != "aux"
                    || atom.head != verb.doc_tok_idx
            })
        {
            return None;
        }
        let (negative, past) = match joined(aux).as_str() {
            "ます" => (false, false),
            "ました" => (false, true),
            "ません" => (true, false),
            "ませんでした" => (true, true),
            _ => return None,
        };
        Some(Self {
            negative,
            past,
            question,
        })
    }
}

struct FareReport {
    speaker: Span,
    clause: Span,
    past: bool,
}

fn fare_report(before: &[Atom], after: &[Atom], body: &[Atom]) -> Option<(FareReport, usize)> {
    let [speaker @ .., topic, open] = before else {
        return None;
    };
    let after = match after {
        [punct, rest @ ..]
            if matches!(punct.surface.as_str(), "。" | "." | "？" | "?") && !punct.is_locked =>
        {
            rest
        }
        _ => after,
    };
    let [close, marker, verb, ending @ ..] = after else {
        return None;
    };
    let report = verb.doc_tok_idx?;
    let speech = FareEnding::parse(verb, ending)?;
    let speaker_head = speaker.last()?;
    if !matches!(
        (open.surface.as_str(), close.surface.as_str()),
        ("「", "」") | ("『", "』")
    ) || !matches!(topic.surface.as_str(), "は" | "が")
        || !dasu_case_link(speaker_head, topic)
        || speaker_head.deprel != "nsubj"
        || !(speaker_head.head == Some(report)
            || body.iter().any(|a| {
                a.lemma.as_deref() == Some("支払う") && a.doc_tok_idx == speaker_head.head
            }))
        || !speaker
            .iter()
            .all(|a| matches!(a.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron)))
        || marker.surface != "と"
        || marker.head != Some(report)
        || verb.upos != Some(Upos::Verb)
        || verb.lemma.as_deref() != Some("言う")
        || verb.deprel != "root"
        || verb.head.is_some()
        || speech.negative
        || speech.question
    {
        return None;
    }
    let clause = atom_span(&after[1..])?;
    Some((
        FareReport {
            speaker: atom_span(speaker)?,
            clause,
            past: speech.past,
        },
        report,
    ))
}

fn fare_roles(source: &[Atom], parent: Option<usize>) -> Option<FareRoles> {
    let verb_idx = source.iter().position(|atom| {
        atom.upos == Some(Upos::Verb) && atom.lemma.as_deref() == Some("支払う")
    })?;
    let [charge @ .., topic, medium, by] = &source[..verb_idx] else {
        return None;
    };
    let verb = &source[verb_idx];
    let ending = FareEnding::parse(verb, &source[verb_idx + 1..])?;
    if source.iter().any(|atom| atom.is_locked) {
        return None;
    }
    let fare = match charge {
        [fare] if fare.surface == "切符代" => fare,
        [ticket, fare]
            if ticket.surface == "切符"
                && ticket.upos == Some(Upos::Noun)
                && ticket.deprel == "compound"
                && ticket.head == fare.doc_tok_idx
                && ticket.span.end_char == fare.span.start_char
                && fare.surface == "代" =>
        {
            fare
        }
        _ => return None,
    };
    let governor = verb.doc_tok_idx?;
    if fare.upos != Some(Upos::Noun)
        || fare.deprel != "nsubj"
        || fare.head != Some(governor)
        || topic.surface != "は"
        || !dasu_case_link(fare, topic)
        || medium.upos != Some(Upos::Noun)
        || medium.deprel != "obl"
        || medium.head != Some(governor)
        || by.surface != "で"
        || !dasu_case_link(medium, by)
        || verb.head != parent
        || match parent {
            Some(_) => !matches!(verb.deprel.as_str(), "advcl" | "ccomp"),
            None => verb.deprel != "root",
        }
    {
        return None;
    }
    let native_medium = match medium.surface.as_str() {
        "カード" => "刷卡",
        "現金" => "用現金",
        _ => return None,
    };
    Some(FareRoles {
        amount: Span {
            start_char: charge.first()?.span.start_char,
            end_char: fare.span.end_char,
        },
        medium: medium.span,
        medium_with_case: Span {
            start_char: medium.span.start_char,
            end_char: by.span.end_char,
        },
        stem: verb.span,
        native_medium,
        ending,
        report: None,
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        GlossaryEntry, Span, TargetVariant, TranslateOptions, TranslationGlossary,
        analyze_and_translate,
    };
    use faberline_nlp::doc::model::Backend;
    use faberline_nlp::doc::{Document, analyze_document};

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
    fn adopted_protected_fare_keeps_endings_and_boundaries() {
        let bank: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/data/ja_zh/payment_fare_stem_dev_v1.json"
        ))
        .unwrap();
        for row in bank["items"].as_array().unwrap() {
            assert_eq!(row["visibility"], "development-visible");
            assert_eq!(row["heldout"], false);
            let id = row["id"].as_str().unwrap();
            let source = row["jp"].as_str().unwrap();
            let fixture = &row["fixture"];
            let options = serde_json::from_value(fixture["translate_options"].clone()).unwrap();
            let glossary = serde_json::from_value(fixture["glossary"].clone()).unwrap();
            let translated =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(
                translated.target_text,
                row["zh_hant"].as_str().unwrap(),
                "{id}"
            );
            let factuality = match id {
                "PFST-N5-02" | "PFST-N5-03" => Some("negated"),
                "PFST-N5-04" | "PFST-N5-05" => Some("hypothetical"),
                _ => None,
            };
            if let Some(expected) = factuality {
                assert_eq!(
                    translated.sentences[0].conditioning.factuality.as_deref(),
                    Some(expected),
                    "{id}"
                );
            }
            let expected = row["lock_expectation"]["expected_hits"].as_array().unwrap();
            assert_eq!(translated.glossary_hits.len(), expected.len(), "{id}");
            for reference in expected {
                let scalar = |span: &serde_json::Value| Span {
                    start_char: span[0].as_u64().unwrap() as usize,
                    end_char: span[1].as_u64().unwrap() as usize,
                };
                let source_span = scalar(&reference["source_span"]["scalar"]);
                let canonical = reference["canonical"].as_str().unwrap();
                let hit = translated
                    .glossary_hits
                    .iter()
                    .find(|hit| hit.source_span == source_span && hit.canonical == canonical)
                    .unwrap_or_else(|| panic!("{id}: missing hit for {canonical}"));
                assert_eq!(
                    hit.target_span,
                    scalar(&reference["target_spans"][0]["scalar"]),
                    "{id}"
                );
            }
        }
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
    fn ticket_fare_is_the_amount_paid() {
        for (source, expected) in [
            (
                "窓口で切符を買います。切符代はカードで支払います。",
                "刷卡付票款。",
            ),
            (
                "窓口で切符を買います。切符代は現金で支払います。",
                "用現金付票款。",
            ),
        ] {
            let translated = translate(source);
            assert_eq!(
                translated.split_once('。').map(|(_, payment)| payment),
                Some(expected)
            );
        }
    }

    fn glossary_entry(source: &str, replacement: &str) -> GlossaryEntry {
        GlossaryEntry {
            canonical: source.into(),
            surfaces: vec![source.into()],
            targets: [(TargetVariant::ZhHant, replacement.into())].into(),
            ..GlossaryEntry::default()
        }
    }

    #[test]
    fn fare_glossary_keeps_the_amount_payment_method_and_stem() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let slice = |text: &str, span: Span| {
            text.chars()
                .skip(span.start_char)
                .take(span.end_char - span.start_char)
                .collect::<String>()
        };
        for (method, native, replacement, whole, whole_replacement) in [
            ("カード", "刷卡", "信用卡", "カードで", "以信用卡"),
            ("現金", "用現金", "現鈔", "現金で", "以現鈔"),
        ] {
            let source = format!("窓口で切符を買います。切符代は{method}で支払います。");
            for (amount, medium, stem, expected) in [
                (true, None, false, format!("{native}付車資。")),
                (false, None, true, format!("{native}結清票款。")),
                (
                    false,
                    Some((method, replacement)),
                    false,
                    format!("用{replacement}付票款。"),
                ),
                (
                    false,
                    Some((method, replacement)),
                    true,
                    format!("用{replacement}結清票款。"),
                ),
                (
                    false,
                    Some((whole, whole_replacement)),
                    false,
                    format!("{whole_replacement}付票款。"),
                ),
                (
                    true,
                    Some((whole, whole_replacement)),
                    false,
                    format!("{whole_replacement}付車資。"),
                ),
                (
                    true,
                    Some((whole, whole_replacement)),
                    true,
                    format!("{whole_replacement}結清車資。"),
                ),
            ] {
                let mut glossary = TranslationGlossary::new();
                let mut expected_hits = Vec::new();
                if amount {
                    glossary.objects.push(glossary_entry("切符代", "車資"));
                    expected_hits.push(("切符代", "車資"));
                }
                if let Some((word, gloss)) = medium {
                    glossary.domain_terms.push(glossary_entry(word, gloss));
                    expected_hits.push((word, gloss));
                }
                if stem {
                    glossary.domain_terms.push(glossary_entry("支払い", "結清"));
                    expected_hits.push(("支払い", "結清"));
                }
                let result =
                    analyze_and_translate(&source, &options, &glossary, None, Backend::Onnx)
                        .unwrap();
                assert_eq!(result.target_text.split_once('。').unwrap().1, expected);
                assert_eq!(result.glossary_hits.len(), expected_hits.len());
                for (hit, (word, gloss)) in result.glossary_hits.iter().zip(expected_hits) {
                    assert_eq!(slice(&source, hit.source_span), word);
                    assert_eq!(slice(&result.target_text, hit.target_span), gloss);
                }
            }
        }
    }

    #[test]
    fn fare_reordering_respects_the_callers_payment_dependency() {
        let source = "窓口で切符を買います。切符代はカードで支払います。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let mut glossary = TranslationGlossary::new();
        glossary
            .domain_terms
            .push(glossary_entry("カードで", "以信用卡"));
        glossary.domain_terms.push(glossary_entry("支払い", "結清"));
        let mut analysis = analyze_document(
            source,
            &crate::doc_options_for_translate(&options, &TranslationGlossary::new()),
        );
        let translate = |input: &Document| {
            crate::translate_document(
                source,
                input,
                #[cfg(feature = "narrative")]
                None,
                &options,
                &glossary,
                None,
            )
            .unwrap()
        };
        assert!(
            translate(&analysis)
                .target_text
                .ends_with("以信用卡結清票款。")
        );

        let medium = analysis
            .syntax
            .iter_mut()
            .find(|arc| {
                source
                    .chars()
                    .skip(arc.span.start_char)
                    .take(arc.span.end_char - arc.span.start_char)
                    .eq("カード".chars())
            })
            .unwrap();
        medium.deprel = "obj".to_owned();
        let output = translate(&analysis);
        assert!(!output.target_text.ends_with("以信用卡結清票款。"));
        assert_eq!(output.glossary_hits.len(), 2);
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
            (
                "録音の「支払える」を聞いてアクセントを確認しました。",
                "「支払える」",
            ),
            ("授業で「支払える」を初めて聞きました。", "「支払える」"),
        ] {
            let actual = translate(jp);
            assert!(actual.contains(quotes), "{jp} -> {actual}");
        }
        let mixed = translate("店長は「支払える」と答え、その単語「支払える」をノートに書いた。");
        let (_, following) = mixed.split_once("「能付款」").expect(&mixed);
        assert!(following.contains("「支払える」"), "{mixed}");
    }

    #[test]
    fn quoted_answers_keep_the_governing_predicate_with_modifiers() {
        for (jp, quote) in [
            ("受付から「支払える」と回答がありました。", "能付款"),
            (
                "録音には、支払いの質問に「支払える」と答えた声が残っていました。",
                "能付款",
            ),
            ("客は「支払える」と語尾を強めて答えました。", "能付款"),
            (
                "料金を払えるかという質問の回答欄に「支払える」と記入しました。",
                "能付款",
            ),
            (
                "送り仮名の質問の回答欄に「支払える」と記入しました。",
                "支払える",
            ),
            ("客は「支払える」と声を落として答えました。", "能付款"),
            ("生徒は「支払える」と声を落として発音しました。", "支払える"),
            ("店員は「支払える」とゆっくり答えた。", "能付款"),
            (
                "読み方を聞かれた生徒は「支払える」と文字を読んで答えました。",
                "支払える",
            ),
        ] {
            let actual = translate(jp);
            assert!(actual.contains(&format!("「{quote}」")), "{jp} -> {actual}");
        }
    }
}
