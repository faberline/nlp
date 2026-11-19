//! Select the use of a written application and the sense of 出す from its
//! own object, destination and source. Preserve the ordinary predicate path.
use super::{Atom, Document, Upos, dasu_case_link, direct_quote_time, is_clock_atom};

pub(super) const SUBMIT: &str = "__ja_submit_application";
pub(super) const CAN_SUBMIT: &str = "__ja_can_submit_application";
pub(super) const OFFICE: &str = "__ja_application_recipient";
pub(super) const SHARED_FORM_RECIPIENT: &str = "__ja_shared_form_recipient";
pub(super) const TIME_QUESTION: &str = "__ja_application_time_question";
pub(super) const CONTRAST: &str = "__ja_application_contrast";
const PRESENT_RECIPIENT: &str = "__ja_located_application_recipient";
pub(super) const PURPOSE: &str = "__ja_application_purpose";
pub(super) const RENT: &str = "__ja_rent_home";
pub(super) const ATTEND: &str = "__ja_attend_course";
pub(super) const PROTECTED_VERB: &str = "__ja_protected_purpose_verb";

fn spelling(a: &Atom, chars: &[char]) -> String {
    chars[a.span.start_char..a.span.end_char].iter().collect()
}

fn form_start(atoms: &[Atom], end: usize, chars: &[char]) -> Option<usize> {
    let form = &atoms[end];
    if !matches!(form.upos, Some(Upos::Noun | Upos::Propn)) {
        return None;
    }
    if !form.is_locked && !form.surface.ends_with('書') {
        return None;
    }
    let raw = spelling(form, chars);
    if matches!(raw.as_str(), "申込書" | "申込み書" | "申し込み書") {
        return Some(end);
    }
    if raw != "書" || end == 0 {
        return None;
    }
    let prefix = &atoms[end - 1];
    (prefix.head == form.doc_tok_idx
        && prefix.deprel == "compound"
        && prefix.span.end_char == form.span.start_char
        && matches!(
            spelling(prefix, chars).as_str(),
            "申込" | "申込み" | "申し込み"
        ))
    .then_some(end - 1)
}

fn purpose(
    atoms: &[Atom],
    start: usize,
    form: &Atom,
    chars: &[char],
) -> Option<(usize, &'static str)> {
    let i = start.checked_sub(1)?;
    let last = &atoms[i];
    let i = if last.surface == "の" {
        let index = i.checked_sub(1)?;
        let owner = &atoms[index];
        if !dasu_case_link(owner, last) || owner.head != form.doc_tok_idx || owner.deprel != "nmod"
        {
            return None;
        }
        index
    } else if last.deprel == "compound" && last.head == form.doc_tok_idx {
        i
    } else {
        return None;
    };
    let owner = &atoms[i];
    let label = spelling(owner, chars);
    let subject = atoms[..i].iter().any(|a| {
        a.head == owner.doc_tok_idx && a.deprel == "compound" && taught_subject(&spelling(a, chars))
    });
    Some((i, named_purpose(&label, subject)?))
}

fn taught_subject(name: &str) -> bool {
    matches!(
        name,
        "日本語" | "英語" | "会話" | "料理" | "音楽" | "水泳" | "書道" | "絵画" | "ダンス"
    )
}

fn named_purpose(label: &str, subject: bool) -> Option<&'static str> {
    if ["講座", "コース", "講習", "授業", "研修"]
        .iter()
        .any(|name| label.ends_with(name))
        || label
            .strip_suffix("教室")
            .is_some_and(|name| subject || taught_subject(name))
    {
        return Some("課程報名表");
    }
    [
        ("大会", "大會報名表"),
        ("イベント", "活動報名表"),
        ("セミナー", "研討會報名表"),
        ("入居", "入住申請表"),
        ("住宅", "住宅申請表"),
    ]
    .into_iter()
    .find_map(|(end, text)| label.ends_with(end).then_some(text))
}

// The glossary can make a protected qualified noun one source token.
// Its visible case and source spelling identify the filing argument.
fn covered_purpose(form: &Atom, chars: &[char]) -> bool {
    if !form.is_locked {
        return false;
    }
    let raw = spelling(form, chars);
    ["申込書", "申込み書", "申し込み書"]
        .iter()
        .find_map(|ending| raw.strip_suffix(ending))
        .and_then(|name| named_purpose(name.strip_suffix('の').unwrap_or(name), false))
        .is_some()
}

fn writable_surface(atoms: &[Atom], end: usize) -> bool {
    let form = &atoms[end];
    atoms
        .get(end + 1)
        .is_some_and(|c| c.surface == "に" && dasu_case_link(form, c))
        && atoms.iter().any(|a| {
            a.doc_tok_idx == form.head
                && !a.is_locked
                && matches!(
                    a.lemma.as_deref().unwrap_or(&a.surface),
                    "書く" | "記入" | "記載"
                )
        })
        && atoms
            .iter()
            .any(|a| a.head == form.head && a.deprel == "obj")
}

fn filing_reading(a: &Atom, atoms: &[Atom]) -> Option<&'static str> {
    if a.is_locked || a.upos != Some(Upos::Verb) {
        return None;
    }
    match a.lemma.as_deref().unwrap_or(&a.surface) {
        "出す" | "提出" => Some(SUBMIT),
        "出せる" => Some(CAN_SUBMIT),
        "出せ"
            if atoms.iter().any(|aux| {
                aux.head == a.doc_tok_idx
                    && aux.deprel == "aux"
                    && !aux.is_locked
                    && matches!(
                        aux.surface.as_str(),
                        "ます" | "まし" | "ませ" | "ました" | "ません"
                    )
            }) =>
        {
            Some(CAN_SUBMIT)
        }
        _ => None,
    }
}

// An office or person receives the document only when its case belongs to this action.
fn recipient(atoms: &[Atom], case: usize, verb: usize, chars: &[char]) -> bool {
    let Some(i) = case.checked_sub(1) else {
        return false;
    };
    let noun = &atoms[i];
    if noun.head != Some(verb) || !dasu_case_link(noun, &atoms[case]) {
        return false;
    }
    let mut first = i;
    while first > 0
        && atoms[first - 1].head == noun.doc_tok_idx
        && atoms[first - 1].deprel == "compound"
        && atoms[first - 1].span.end_char == atoms[first].span.start_char
    {
        first -= 1
    }
    let raw: String = chars[atoms[first].span.start_char..noun.span.end_char]
        .iter()
        .collect();
    is_person(noun, chars)
        || matches!(
            raw.as_str(),
            "受付" | "窓口" | "事務局" | "事務室" | "管理会社" | "担当者" | "係員"
        )
}

fn is_person(noun: &Atom, chars: &[char]) -> bool {
    if matches!(
        noun.lemma.as_deref(),
        Some(PRESENT_RECIPIENT | super::LOCKED_RESIDENT_ATOM)
    ) {
        return true;
    }
    if !noun.is_locked {
        return super::is_companion_person(noun);
    }
    let mut original = noun.clone();
    original.upos = Some(Upos::Noun);
    let raw = spelling(noun, chars);
    // A whole glossary entry is one noun in the source document. Its
    // final noun still distinguishes a recipient from that person's desk.
    let name = [
        "にいる",
        "にいた",
        "にいない",
        "にいなかった",
        "に来る",
        "に来た",
        "の",
    ]
    .iter()
    .filter_map(|suffix| raw.rsplit_once(suffix))
    .max_by_key(|(prefix, _)| prefix.len())
    .map_or(raw.as_str(), |(_, head)| head);
    original.surface = name.to_owned();
    matches!(name, "担当者" | "係員") || super::is_companion_person(&original)
}

fn physical_argument(
    atoms: &[Atom],
    verb: usize,
    chars: &[char],
    doc: &Document,
    has_recipient: bool,
) -> bool {
    // A glossary entry may include the container's source particle.
    if atoms.iter().any(|a| {
        a.is_locked
            && (a.head == Some(verb)
                || (a.deprel == "case"
                    && a.head
                        .and_then(|h| doc.syntax.iter().find(|s| s.token == h))
                        .is_some_and(|s| s.head == Some(verb))))
            && spelling(a, chars).strip_suffix("から").is_some_and(|name| {
                [
                    "かばん",
                    "鞄",
                    "バッグ",
                    "箱",
                    "封筒",
                    "引き出し",
                    "ポケット",
                ]
                .iter()
                .any(|end| name.ends_with(end))
            })
    }) {
        return true;
    }
    atoms.windows(2).enumerate().any(|(i, pair)| {
        let [noun, case] = pair else { return false };
        if noun.head != Some(verb) || !dasu_case_link(noun, case) {
            return false;
        }
        let raw = spelling(noun, chars);
        if has_recipient && case.surface == "に" && purpose_head(atoms, noun, doc).is_some() {
            return false;
        }
        let temporal = direct_quote_time(noun, doc).is_some()
            || is_clock_atom(noun)
            || noun.lemma.as_deref() == Some(TIME_QUESTION)
            || matches!(
                raw.as_str(),
                "日" | "月" | "年" | "時" | "分" | "翌日" | "平日"
            )
            || [
                "月曜日",
                "火曜日",
                "水曜日",
                "木曜日",
                "金曜日",
                "土曜日",
                "日曜日",
            ]
            .contains(&raw.as_str())
            || ["日", "時", "月", "年"].iter().any(|s| {
                raw.strip_suffix(s)
                    .and_then(crate::opaque::parse_cjk_or_ascii_num)
                    .is_some()
            });
        match case.surface.as_str() {
            "から" => !temporal && noun.upos != Some(Upos::Verb),
            "に" if has_recipient && raw == "一緒" => false,
            "に" | "へ" => !temporal && !recipient(atoms, i + 1, verb, chars),
            _ => false,
        }
    })
}

fn purpose_head<'a>(atoms: &'a [Atom], reason: &Atom, doc: &Document) -> Option<&'a Atom> {
    if reason.is_locked || reason.surface != "ため" {
        return None;
    }
    atoms.iter().find(|v| {
        v.deprel == "acl"
            && v.head == reason.doc_tok_idx
            && ((!v.is_locked && v.upos == Some(Upos::Verb))
                || (v.is_locked
                    && v.doc_tok_idx
                        .and_then(|i| doc.tokens.get(i))
                        .is_some_and(|t| t.upos == Some(Upos::Verb) && t.span == v.span)))
    })
}

fn bind_filing_purpose(atoms: &mut Vec<Atom>, doc: &Document, chars: &[char]) {
    for i in (1..atoms.len().saturating_sub(1)).rev() {
        let Some(reason) = atoms.get(i) else { continue };
        let Some(ni) = atoms
            .get(i + 1)
            .filter(|ni| ni.surface == "に" && dasu_case_link(reason, ni))
        else {
            continue;
        };
        let Some(verb) = purpose_head(atoms, reason, doc) else {
            continue;
        };
        let Some(outer) = atoms.iter().find(|a| a.doc_tok_idx == reason.head) else {
            continue;
        };
        let physical_form = filing_reading(outer, atoms).is_some()
            && atoms.iter().enumerate().any(|(k, a)| {
                a.deprel == "obj"
                    && a.head == outer.doc_tok_idx
                    && (covered_purpose(a, chars) || form_start(atoms, k, chars).is_some())
            })
            && atoms.windows(2).any(|p| {
                p[0].head == outer.doc_tok_idx
                    && p[1].surface == "から"
                    && dasu_case_link(&p[0], &p[1])
            });
        if !physical_form && !matches!(outer.lemma.as_deref(), Some(SUBMIT | CAN_SUBMIT)) {
            continue;
        }
        let inner = verb.doc_tok_idx;
        let outer = outer.doc_tok_idx;
        let Some(sentence) = verb
            .doc_tok_idx
            .and_then(|h| doc.syntax.iter().find(|n| n.token == h))
            .map(|n| n.sentence)
        else {
            continue;
        };
        // A preceding は is the matrix topic. Include the objects of the
        // purpose predicate in the embedded clause, leaving that topic outside.
        let start = atoms[..i]
            .iter()
            .rposition(|a| a.surface == "は" || matches!(a.upos, Some(Upos::Punct)))
            .map_or(0, |k| k + 1);
        if !atoms[start..i].iter().any(|a| a.doc_tok_idx == inner)
            || atoms[start..i]
                .iter()
                .any(|a| a.head == outer && a.deprel == "obj")
        {
            continue;
        }
        let ctx = super::ClauseDiscourseCtx {
            source_chars: chars,
            ..Default::default()
        };
        let mut clause = atoms[start..=i].to_vec();
        if verb.is_locked
            && let Some(v) = clause.iter_mut().find(|a| a.doc_tok_idx == inner)
        {
            v.upos = Some(Upos::Verb);
            v.is_locked = false;
            v.lemma = Some(PROTECTED_VERB.to_owned());
        }
        // In a housing application, 部屋を借りる is renting a room. An
        // unrelated loan or a glossary-protected verb keeps its own reading.
        if verb.lemma.as_deref().unwrap_or(&verb.surface) == "借りる"
            && clause.iter().any(|a| {
                a.deprel == "obj"
                    && a.head == inner
                    && matches!(
                        spelling(a, chars).as_str(),
                        "部屋" | "家" | "住宅" | "アパート"
                    )
            })
            && atoms.iter().any(|a| {
                a.deprel == "obj" && a.head == outer && {
                    let raw = spelling(a, chars);
                    ["申込書", "申込み書", "申し込み書"]
                        .iter()
                        .find_map(|suffix| raw.strip_suffix(suffix))
                        .is_some_and(|qualifier| {
                            matches!(qualifier.trim_end_matches('の'), "住宅" | "入居")
                        })
                }
            })
            && let Some(borrow) = clause.iter_mut().find(|a| a.doc_tok_idx == inner)
        {
            borrow.lemma = Some(RENT.to_owned());
        }
        if verb.lemma.as_deref().unwrap_or(&verb.surface) == "受ける"
            && clause.iter().any(|a| {
                a.deprel == "obj"
                    && a.head == inner
                    && matches!(
                        spelling(a, chars).as_str(),
                        "授業" | "講座" | "講習" | "研修"
                    )
            })
            && let Some(take) = clause.iter_mut().find(|a| a.doc_tok_idx == inner)
        {
            take.lemma = Some(ATTEND.to_owned());
        }
        let Some((text, _)) = super::try_translate_purposive_tame(
            &clause,
            doc,
            sentence,
            &ctx,
            super::TargetVariant::ZhHant,
        ) else {
            continue;
        };
        let mut phrase = reason.clone();
        phrase.surface = text;
        phrase.lemma = Some(PURPOSE.to_owned());
        phrase.upos = Some(Upos::Adv);
        phrase.deprel = "advmod".to_owned();
        phrase.is_locked = true;
        phrase.span.start_char = atoms[start].span.start_char;
        phrase.span.end_char = ni.span.end_char;
        for subject in &mut atoms[..start] {
            if subject.deprel == "nsubj" && subject.head == inner {
                subject.head = outer
            }
        }
        atoms.splice(start..=i + 1, [phrase]);
    }
}

fn mark_filing_time(atoms: &mut [Atom], verb: usize, chars: &[char]) {
    for i in 0..atoms.len().saturating_sub(1) {
        let (noun, case) = (&atoms[i], &atoms[i + 1]);
        if noun.head != Some(verb) || case.surface != "に" || !dasu_case_link(noun, case) {
            continue;
        }
        let raw = spelling(noun, chars);
        let time = if noun.is_locked {
            raw.rsplit_once('の')
                .and_then(|(qualifier, head)| {
                    let mut day = noun.clone();
                    day.is_locked = false;
                    day.upos = Some(Upos::Noun);
                    day.surface = qualifier.to_owned();
                    day.lemma = None;
                    super::render_temporal_made_atom(&day).map(|_| head)
                })
                .unwrap_or(raw.as_str())
        } else {
            raw.as_str()
        };
        let text = match time {
            "何時" => "幾點",
            "いつ" => "什麼時候",
            _ => continue,
        };
        let noun = &mut atoms[i];
        if !noun.is_locked {
            noun.surface = text.to_owned();
        }
        noun.is_locked = true;
        noun.lemma = Some(TIME_QUESTION.to_owned());
    }
}

fn office_name(a: &mut Atom) {
    if a.is_locked {
        return;
    }
    let name = match a.surface.as_str() {
        "窓口" => "窗口",
        "受付" => "櫃檯",
        _ => return,
    };
    a.surface = name.to_owned();
    a.is_locked = true;
}

// In 表を窓口にいる人に出す the parser may attach を to the intransitive
// いる. The form belongs to 出す; the place belongs to the person.
fn noun_start(atoms: &[Atom], end: usize) -> usize {
    let mut start = end;
    while start > 0
        && atoms[start - 1].deprel == "compound"
        && atoms[start - 1].head == atoms[end].doc_tok_idx
        && atoms[start - 1].span.end_char == atoms[start].span.start_char
    {
        start -= 1;
    }
    start
}

fn noun_text(words: &[Atom], chars: &[char]) -> String {
    let text = |a: &Atom| {
        if a.is_locked {
            a.surface.clone()
        } else {
            super::translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos)
        }
    };
    if words.iter().any(|a| a.is_locked) {
        return words.iter().map(text).collect();
    }
    let raw: String = chars[words[0].span.start_char..words.last().unwrap().span.end_char]
        .iter()
        .collect();
    match raw.as_str() {
        "受付" => "櫃檯".to_owned(),
        "窓口" => "窗口".to_owned(),
        "管理会社" => "管理公司".to_owned(),
        "担当者" => "負責人".to_owned(),
        "担当" => "負責".to_owned(),
        _ => super::translate_single_ja_term(&raw, None, Some(Upos::Noun)),
    }
}

fn relative_time(
    atoms: &[Atom],
    end: usize,
    predicate: &Atom,
    doc: &Document,
    chars: &[char],
) -> Option<(usize, String)> {
    let last = end.checked_sub(1)?;
    let marker = &atoms[last];
    let (i, case) = if matches!(marker.surface.as_str(), "に" | "から" | "まで" | "は") {
        let i = last.checked_sub(1)?;
        (i, marker.surface.as_str())
    } else {
        (last, "")
    };
    let time = &atoms[i];
    // Merging 午前・九時 can drop its token id. The case particle
    // still points to the original clock head in the source document.
    let merged_clock = is_clock_atom(time)
        && marker.head.is_some_and(|id| {
            doc.tokens.get(id).is_some_and(|t| {
                time.span.start_char <= t.span.start_char && t.span.end_char <= time.span.end_char
            }) && doc.syntax.iter().any(|n| {
                n.token == id
                    && n.head == predicate.doc_tok_idx
                    && matches!(n.deprel.as_str(), "advmod" | "obl")
            })
        });
    if !merged_clock
        && (time.head != predicate.doc_tok_idx
            || !(matches!(time.deprel.as_str(), "advmod" | "obl")
                || (case == "は" && time.deprel == "nsubj"))
            || (!case.is_empty() && !dasu_case_link(time, marker)))
    {
        return None;
    }
    let label = |a: &Atom| {
        if is_clock_atom(a) {
            return Some(a.surface.clone());
        }
        let source = spelling(a, chars);
        let known = match source.as_str() {
            "今朝" => Some("今天早上".to_owned()),
            "今夜" | "今晩" => Some("今晚".to_owned()),
            "昨夜" | "昨晩" => Some("昨晚".to_owned()),
            _ if case == "は"
                && source
                    .strip_suffix('日')
                    .and_then(crate::opaque::parse_cjk_or_ascii_num)
                    .is_some_and(|day| (1..=31).contains(&day)) =>
            {
                Some(source.clone())
            }
            _ => {
                let (prefix, clock) = source
                    .strip_prefix("午前")
                    .map(|s| ("上午", s))
                    .or_else(|| source.strip_prefix("午後").map(|s| ("下午", s)))
                    .unwrap_or(("", source.as_str()));
                clock
                    .strip_suffix('時')
                    .and_then(super::clock_number)
                    .map(|hour| format!("{prefix}{}點", super::clock_hour_text(hour)))
            }
        };
        if let Some(value) = known {
            return Some(if a.is_locked && !is_clock_atom(a) {
                a.surface.clone()
            } else {
                value
            });
        }
        if let Some(text) = super::render_temporal_made_atom(a) {
            return Some(text);
        }
        if !a.is_locked {
            return direct_quote_time(a, doc).or_else(|| {
                // An adverb attached to いる modifies the person's presence.
                (a.upos == Some(Upos::Adv) && a.deprel == "advmod").then(|| {
                    super::translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos)
                })
            });
        }
        let mut source = a.clone();
        source.is_locked = false;
        source.surface = spelling(a, chars);
        source.upos = Some(Upos::Noun);
        source.lemma = None;
        let adverb = a.deprel == "advmod"
            && faberline_nlp::lexicon::lookup_ja(&source.surface)
                .is_some_and(|h| h.upos_candidates.contains(&Upos::Adv));
        (adverb
            || direct_quote_time(&source, doc).is_some()
            || super::render_temporal_made_atom(&source).is_some())
        .then(|| a.surface.clone())
    };
    let mut text = label(time)?;
    let mut start = i;
    if i >= 2
        && atoms[i - 1].surface == "の"
        && dasu_case_link(&atoms[i - 2], &atoms[i - 1])
        && (atoms[i - 2].head == time.doc_tok_idx
            || (matches!(spelling(time, chars).as_str(), "朝" | "昼" | "夜")
                && atoms[i - 2].deprel == "nmod"
                && atoms
                    .get(end)
                    .is_some_and(|place| atoms[i - 2].head == place.doc_tok_idx)))
        && let Some(prefix) = label(&atoms[i - 2])
    {
        text = prefix + &text;
        start -= 2;
    } else if i > 0
        && matches!(spelling(&atoms[i - 1], chars).as_str(), "午前" | "午後")
        && (atoms[i - 1].head == time.doc_tok_idx || atoms[i - 1].head == predicate.doc_tok_idx)
        && let Some(prefix) = label(&atoms[i - 1])
    {
        text = prefix + &text;
        start -= 1;
    }
    match case {
        "から" => text = format!("自{text}起"),
        "まで" => text = format!("直到{text}"),
        _ => {}
    }
    Some((start, text))
}

fn recipient_description(
    atoms: &[Atom],
    modified: usize,
    chars: &[char],
) -> (usize, usize, String) {
    let mut i = modified;
    let mut start = noun_start(atoms, i);
    let counter = &atoms[i];
    let raw_counter = spelling(counter, chars);
    let number = i.checked_sub(1).filter(|&k| {
        let num = &atoms[k];
        matches!(raw_counter.as_str(), "人" | "名")
            && num.head == counter.doc_tok_idx
            && num.span.end_char == counter.span.start_char
            && matches!(num.deprel.as_str(), "nummod" | "compound")
            && crate::opaque::parse_cjk_or_ascii_num(&super::normalize_decimal_digits(&spelling(
                num, chars,
            )))
            .is_some()
    });
    let mut count = String::new();
    let mut unit = None;
    if let Some(k) = number {
        start = k;
        if atoms[k].is_locked {
            count = atoms[k].surface.clone();
        } else {
            count = super::render_joint_people(&atoms[k].surface)
                .trim_end_matches('人')
                .to_owned();
        }
        unit = Some(counter);
    } else if counter.is_locked
        && raw_counter.strip_suffix(['人', '名']).is_some_and(|n| {
            crate::opaque::parse_cjk_or_ascii_num(&super::normalize_decimal_digits(n)).is_some()
        })
    {
        count = counter.surface.clone();
    }
    let mut name = if count.is_empty() {
        noun_text(&atoms[start..=i], chars)
    } else {
        String::new()
    };
    while atoms[i].deprel == "nmod"
        && atoms
            .get(i + 1)
            .is_some_and(|no| no.surface == "の" && dasu_case_link(&atoms[i], no))
    {
        let Some(n) = atoms.iter().position(|a| a.doc_tok_idx == atoms[i].head) else {
            break;
        };
        if n <= i + 1 || noun_start(atoms, n) != i + 2 {
            break;
        }
        if !name.is_empty() {
            name.push('的');
        }
        name.push_str(&noun_text(&atoms[i + 2..=n], chars));
        i = n;
    }
    if let Some(unit) = unit {
        count.push_str(if unit.is_locked {
            &unit.surface
        } else if name.is_empty() {
            "人"
        } else {
            "位"
        });
    }
    (start, i, count + &name)
}

fn located_recipient(atoms: &mut Vec<Atom>, chars: &[char], doc: &Document) {
    for be_idx in (2..atoms.len()).rev() {
        let Some(be) = atoms.get(be_idx) else {
            continue;
        };
        let action = be.lemma.as_deref().unwrap_or(&be.surface);
        if be.is_locked || !matches!(action, "いる" | "居る" | "来る") || be.deprel != "acl" {
            continue;
        }
        // In 待っていた職員 the location is attached to 待つ. The
        // auxiliary いる still modifies the recipient.
        let activity = atoms
            .get(be_idx - 2)
            .filter(|v| {
                !v.is_locked
                    && v.head == be.doc_tok_idx
                    && v.deprel == "advcl"
                    && atoms[be_idx - 1].surface == "て"
                    && atoms[be_idx - 1].head == v.doc_tok_idx
            })
            .and_then(|v| match v.lemma.as_deref() {
                Some("待つ") => Some((v, "等候")),
                Some("勤める" | "働く") => Some((v, "工作")),
                _ => None,
            });
        let (event, suffix) = activity.unwrap_or((be, ""));
        let event_idx = if activity.is_some() {
            be_idx - 2
        } else {
            be_idx
        };
        let (end_place, after) = relative_time(atoms, event_idx, event, doc, chars)
            .unwrap_or((event_idx, String::new()));
        if end_place < 2 {
            continue;
        }
        let place = &atoms[end_place - 2];
        let ni = &atoms[end_place - 1];
        if !(ni.surface == "に" || (activity.is_some() && ni.surface == "で"))
            || !dasu_case_link(place, ni)
            || place.head != event.doc_tok_idx
        {
            continue;
        }
        let Some(modified) = atoms.iter().position(|a| a.doc_tok_idx == be.head) else {
            continue;
        };
        let (start_person, person_idx, role) = recipient_description(atoms, modified, chars);
        if start_person <= be_idx {
            continue;
        }
        let person = &atoms[person_idx];
        let Some(to) = atoms.get(person_idx + 1) else {
            continue;
        };
        if !matches!(to.surface.as_str(), "に" | "へ" | "と") || !dasu_case_link(person, to) {
            continue;
        }
        let Some(outer) = person.head.filter(|id| {
            atoms
                .iter()
                .any(|v| v.doc_tok_idx == Some(*id) && filing_reading(v, atoms).is_some())
        }) else {
            continue;
        };
        let accompanies = to.surface == "と"
            && together(atoms, outer)
                .and_then(|(_, noun)| noun)
                .is_some_and(|(id, _)| person.doc_tok_idx == Some(id));
        if (to.surface == "と" && !accompanies) || !recipient(atoms, person_idx + 1, outer, chars)
        {
            continue;
        }
        let inflection = &atoms[be_idx + 1..start_person];
        if !inflection
            .iter()
            .all(|a| !a.is_locked && a.head == be.doc_tok_idx && a.deprel == "aux")
        {
            continue;
        }
        let conjugation = inflection
            .iter()
            .map(|a| a.surface.as_str())
            .collect::<String>();
        let relation = match (action == "来る", conjugation.as_str()) {
            (false, "") => "在",
            (false, "た") => "當時在",
            (false, "ない") => "不在",
            (false, "なかった") => "當時不在",
            (true, "" | "た") => "來到",
            (true, "ない") => "不來",
            (true, "なかった") => "沒來",
            _ => continue,
        };
        let place_first = noun_start(atoms, end_place - 2);
        let (first, before) = relative_time(atoms, place_first, event, doc, chars)
            .unwrap_or((place_first, String::new()));
        let form = (0..atoms.len()).find(|&k| {
            atoms[k].deprel == "obj"
                && ((k < first && atoms[k].head == event.doc_tok_idx)
                    || ((k < first || k > person_idx) && atoms[k].head == Some(outer)))
                && atoms
                    .get(k + 1)
                    .is_some_and(|c| c.surface == "を" && dasu_case_link(&atoms[k], c))
                && (form_start(atoms, k, chars).is_some() || covered_purpose(&atoms[k], chars))
        });
        // The first predicate may link to the relative いる. Keep its
        // own form and connect the two actions through the recipient.
        let preceding = (0..first).rev().find(|&v| {
            let action = &atoms[v];
            action.deprel == "advcl"
                && (action.head == event.doc_tok_idx || action.head == Some(outer))
                && linked_clause(atoms, v)
                && (0..v).any(|k| {
                    atoms[k].deprel == "obj"
                        && atoms[k].head == action.doc_tok_idx
                        && atoms
                            .get(k + 1)
                            .is_some_and(|a| a.surface == "を" && dasu_case_link(&atoms[k], a))
                        && (form_start(atoms, k, chars).is_some()
                            || covered_purpose(&atoms[k], chars))
                })
        });
        if form.is_none() && preceding.is_none() {
            continue;
        }
        let mut receiver = person.clone();
        let location = noun_text(&atoms[place_first..end_place - 1], chars);
        let time = before + &after;
        let relation = if time.is_empty() {
            relation
        } else {
            relation.trim_start_matches("當時")
        };
        receiver.surface = format!("{time}{relation}{location}{suffix}的{role}");
        receiver.lemma = Some(PRESENT_RECIPIENT.to_owned());
        receiver.is_locked = true;
        receiver.span.start_char = atoms[first].span.start_char;
        let inner = event.doc_tok_idx;
        for i in 0..first {
            if atoms[i].deprel == "nsubj"
                && atoms[i].head == inner
                && atoms
                    .get(i + 1)
                    .is_some_and(|case| case.surface == "は" && dasu_case_link(&atoms[i], case))
            {
                atoms[i].head = Some(outer);
            }
        }
        if let Some(form) = form {
            atoms[form].head = Some(outer);
        }
        if let Some(v) = preceding {
            atoms[v].head = Some(outer);
        }
        atoms.splice(first..=person_idx, [receiver]);
    }
}

fn filing_recipient(atoms: &[Atom], verb: usize, chars: &[char]) -> Option<usize> {
    atoms.iter().enumerate().find_map(|(i, a)| {
        (matches!(a.surface.as_str(), "に" | "へ") && recipient(atoms, i, verb, chars))
            .then(|| i - 1)
    })
}

// 一緒に belongs to the filing action. An immediately preceding Nと
// can name the person or item that accompanies the application.
pub(super) fn together(atoms: &[Atom], verb: usize) -> Option<(usize, Option<(usize, bool)>)> {
    let i = atoms.windows(2).position(|pair| {
        let [word, case] = pair else { return false };
        !word.is_locked
            && word.surface == "一緒"
            && word.head == Some(verb)
            && case.surface == "に"
            && dasu_case_link(word, case)
    })?;
    let companion = i.checked_sub(2).and_then(|start| {
        let noun = &atoms[start];
        let to = &atoms[start + 1];
        (noun.head == Some(verb)
            && to.surface == "と"
            && dasu_case_link(noun, to)
            && to.span.end_char == atoms[i].span.start_char)
            .then_some((
                noun.doc_tok_idx?,
                super::is_companion_person(noun)
                    || matches!(
                        noun.lemma.as_deref(),
                        Some(super::LOCKED_RESIDENT_ATOM | PRESENT_RECIPIENT)
                    ),
            ))
    });
    Some((atoms[i].doc_tok_idx?, companion))
}

pub(super) fn companion_label(
    source: &[Atom],
    id: usize,
    person: bool,
    group: &super::Bunsetsu,
    target: super::TargetVariant,
    chars: &[char],
) -> Option<String> {
    let end = source.iter().position(|a| a.doc_tok_idx == Some(id))?;
    let person = person
        || (source[end].is_locked
            && source[end].span.end_char <= chars.len()
            && is_person(&source[end], chars));
    let mut start = end;
    while let Some(i) = start.checked_sub(1) {
        let before = &source[i];
        let attached = matches!(
            before.deprel.as_str(),
            "nmod" | "amod" | "compound" | "nummod" | "det"
        ) && source[start..=end]
            .iter()
            .any(|a| a.doc_tok_idx == before.head);
        let genitive = before.surface == "の"
            && i > 0
            && dasu_case_link(&source[i - 1], before)
            && source[start..=end]
                .iter()
                .any(|a| a.doc_tok_idx == source[i - 1].head);
        if !attached && !genitive {
            break;
        }
        start = i;
    }
    let counted = (start..=end).find_map(|i| {
        if !person
            || source[end].span.end_char > chars.len()
            || !matches!(spelling(&source[i], chars).as_str(), "人" | "名")
        {
            return None;
        }
        let (first, last, text) = recipient_description(source, i, chars);
        (first == start && last == end).then_some(text)
    });
    let noun = counted.unwrap_or_else(|| {
        let mut noun = group.clone();
        noun.atoms = source[start..=end].to_vec();
        super::render_bunsetsu_np(&noun, target)
    });
    Some(if person {
        format!("和{noun}一起")
    } else {
        format!("連同{noun}")
    })
}

fn mark_recipient(atoms: &mut [Atom], i: usize, role: &str) {
    let person = atoms[i].doc_tok_idx;
    for modifier in &mut atoms[..i] {
        if modifier.head == person && modifier.deprel == "nmod" {
            office_name(modifier);
        }
    }
    office_name(&mut atoms[i]);
    atoms[i].lemma = Some(role.to_owned());
}

// A coordinated action can keep the preceding action's object. Both the
// link between the verbs and the receiving argument belong to the source.
fn contrast_link(atoms: &[Atom], verb: usize) -> Option<usize> {
    let action = &atoms[verb];
    let id = action.doc_tok_idx?;
    if action.is_locked || action.upos != Some(Upos::Verb) {
        return None;
    }
    let link = (verb + 1..atoms.len()).find(|&i| {
        let a = &atoms[i];
        a.is_locked || a.head != Some(id) || a.deprel != "aux" || a.upos != Some(Upos::Aux)
    })?;
    let mark = &atoms[link];
    (!mark.is_locked
        && mark.head == Some(id)
        && ((mark.surface == "が" && matches!(mark.deprel.as_str(), "case" | "mark"))
            || (matches!(mark.surface.as_str(), "けれど" | "けれども" | "けど")
                && mark.deprel == "mark")))
        .then_some(link)
}

fn linked_clause(atoms: &[Atom], verb: usize) -> bool {
    atoms
        .get(verb + 1)
        .is_some_and(|a| !a.is_locked && matches!(a.surface.as_str(), "て" | "、" | ","))
        || contrast_link(atoms, verb).is_some()
}

fn mark_form_contrasts(atoms: &mut [Atom], chars: &[char]) {
    for i in 0..atoms.len() {
        let form = &atoms[i];
        if form.deprel != "obj"
            || !atoms
                .get(i + 1)
                .is_some_and(|a| a.surface == "を" && dasu_case_link(form, a))
            || (form_start(atoms, i, chars).is_none() && !covered_purpose(form, chars))
        {
            continue;
        }
        let Some(v) = (i + 1..atoms.len()).find(|&k| atoms[k].doc_tok_idx == form.head) else {
            continue;
        };
        let Some(link) = contrast_link(atoms, v) else {
            continue;
        };
        // が after this finite action attaches to the verb. が on a
        // noun is the subject marker and keeps its original analysis.
        if atoms[link].surface == "が"
            && matches!(atoms[v].deprel.as_str(), "nsubj" | "advcl")
            && atoms[link + 1..]
                .iter()
                .any(|a| a.doc_tok_idx.is_some() && a.doc_tok_idx == atoms[v].head)
        {
            atoms[v].deprel = "advcl".to_owned();
            atoms[link].upos = Some(Upos::Sconj);
            atoms[link].deprel = "mark".to_owned();
            atoms[link].lemma = Some(CONTRAST.to_owned());
        }
    }
}

fn mark_shared_form(
    atoms: &mut [Atom],
    form: usize,
    first: usize,
    chars: &[char],
    doc: &Document,
) -> Option<()> {
    // A verbal noun such as 印刷 can govern the form while its light
    // verb する carries the clause link.
    let first = match atoms.get(first + 1) {
        Some(light)
            if !light.is_locked
                && light.lemma.as_deref() == Some("する")
                && atoms[first].deprel == "advcl"
                && atoms[first].head == light.doc_tok_idx =>
        {
            first + 1
        }
        _ => first,
    };
    let previous = &atoms[first];
    let after = previous.deprel == "obl"
        && atoms.get(first + 1..first + 3).is_some_and(|p| {
            p[0].surface == "て"
                && p[1].surface == "から"
                && p.iter()
                    .all(|a| a.head == previous.doc_tok_idx && !a.is_locked)
        });
    if form >= first
        || (previous.deprel != "advcl" && !after)
        || previous.is_locked
        || !linked_clause(atoms, first)
        || atoms.windows(2).any(|w| {
            w[0].head == previous.doc_tok_idx
                && w[1].surface == "と"
                && dasu_case_link(&w[0], &w[1])
                && !previous
                    .doc_tok_idx
                    .and_then(|v| together(atoms, v))
                    .and_then(|(_, companion)| companion)
                    .is_some_and(|(id, _)| w[0].doc_tok_idx == Some(id))
        })
    {
        return None;
    }
    let mut next = (first + 1..atoms.len()).find(|&i| atoms[i].doc_tok_idx == previous.head)?;
    // An adjective on the accompanying item can become the parser's
    // next predicate. Its item must link to the same 一緒に filing.
    if atoms[next].deprel == "amod" {
        let item = atoms.iter().find(|a| a.doc_tok_idx == atoms[next].head)?;
        let owner = item.head?;
        let (_, companion) = together(atoms, owner)?;
        if companion.map(|(id, _)| id) != item.doc_tok_idx {
            return None;
        }
        next = (next + 1..atoms.len()).find(|&i| atoms[i].doc_tok_idx == Some(owner))?;
    }
    let reading = filing_reading(&atoms[next], atoms)?;
    let verb = atoms[next].doc_tok_idx?;
    let office = filing_recipient(atoms, verb, chars)?;
    if atoms[form + 1..next]
        .iter()
        .any(|a| super::is_quote_mark(a) || matches!(a.surface.as_str(), "“" | "”" | "\""))
        || atoms
            .iter()
            .any(|a| a.head == Some(verb) && a.deprel == "obj")
    {
        return None;
    }
    // A new subject cancels the shared object. A time topicalized with は
    // can receive the same subject tag without introducing another actor.
    let mut times = Vec::new();
    for (i, a) in atoms.iter().enumerate().filter(|(i, a)| {
        *i != office
            && a.head == Some(verb)
            && a.deprel == "nsubj"
            && a.span.start_char > previous.span.end_char
    }) {
        let wa = atoms.get(i + 1)?;
        if wa.surface != "は" || !dasu_case_link(a, wa) {
            return None;
        }
        let (start, time) = relative_time(atoms, i + 2, &atoms[next], doc, chars)?;
        if start == i && !a.is_locked {
            times.push((i, time));
        }
    }
    mark_filing_time(atoms, verb, chars);
    if physical_argument(atoms, verb, chars, doc, true) {
        return None;
    }
    for (i, time) in times {
        atoms[i].surface = time;
        atoms[i].is_locked = true;
    }
    atoms[next].lemma = Some(reading.to_owned());
    mark_recipient(atoms, office, SHARED_FORM_RECIPIENT);
    Some(())
}

pub(super) fn normalize(atoms: &mut Vec<Atom>, chars: &[char], doc: &Document) {
    mark_form_contrasts(atoms, chars);
    located_recipient(atoms, chars, doc);
    for end in (0..atoms.len()).rev() {
        let Some(form) = atoms.get(end) else { continue };
        let covered = covered_purpose(form, chars);
        let Some(start) = form_start(atoms, end, chars).or_else(|| covered.then_some(end)) else {
            continue;
        };
        let context = purpose(atoms, start, form, chars);
        let mut filing = false;
        if form.deprel == "obj"
            && atoms
                .get(end + 1)
                .is_some_and(|wo| wo.surface == "を" && dasu_case_link(form, wo))
            && let Some(v) = atoms.iter().position(|a| a.doc_tok_idx == form.head)
        {
            filing = mark_shared_form(atoms, end, v, chars, doc).is_some();
            if let Some(reading) = filing_reading(&atoms[v], atoms)
                && let Some(verb) = atoms[v].doc_tok_idx
            {
                mark_filing_time(atoms, verb, chars);
                let office = filing_recipient(atoms, verb, chars);
                if atoms[v].surface == "提出"
                    || ((context.is_some() || office.is_some() || covered)
                        && !physical_argument(atoms, verb, chars, doc, office.is_some()))
                {
                    atoms[v].lemma = Some(reading.to_owned());
                    filing = true;
                    if let Some(i) = office {
                        mark_recipient(atoms, i, OFFICE);
                    }
                }
            }
        }
        let writing = writable_surface(atoms, end);
        if writing {
            atoms[end + 1].surface = "で".to_owned();
        }
        let form = &atoms[end];
        if form.is_locked && start == end && context.is_none() {
            if writing {
                atoms[end].surface.push('上');
            }
            continue;
        }
        let (first, target) = match context {
            Some(pair) => pair,
            None if filing => (start, "申請表"),
            None => continue,
        };
        let mut noun = form.clone();
        let mut first = first;
        let mut prefix = String::new();
        if first < start {
            let owner = atoms[first].doc_tok_idx;
            let finish = first;
            while first > 0
                && atoms[first - 1].head == owner
                && atoms[first - 1].deprel == "compound"
                && atoms[first - 1].span.end_char == atoms[first].span.start_char
            {
                first -= 1
            }
            let words = &atoms[first..finish];
            let raw = words.iter().map(|a| a.surface.as_str()).collect::<String>();
            prefix = match raw.as_str() {
                "日本語" => "日語".to_owned(),
                "英会話" => "英語會話".to_owned(),
                "料理" if target == "課程報名表" => "烹飪".to_owned(),
                _ => words
                    .iter()
                    .map(|a| {
                        if a.is_locked {
                            a.surface.clone()
                        } else {
                            super::translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos)
                        }
                    })
                    .collect(),
            };
            for before in &mut atoms[..first] {
                if before.head == owner {
                    before.head = noun.doc_tok_idx;
                }
            }
        }
        let qualifier = target.trim_end_matches("報名表").trim_end_matches("申請表");
        let kind = target.strip_prefix(qualifier).unwrap_or(target);
        let domain = context
            .map(|(i, _)| &atoms[i])
            .filter(|a| a.is_locked)
            .map_or(qualifier, |a| a.surface.as_str());
        let form_name = if noun.is_locked && start == end {
            noun.surface.clone()
        } else if noun.is_locked {
            format!("{}{}", kind.trim_end_matches('表'), noun.surface)
        } else if start < end && atoms[start].is_locked {
            format!("{}表", atoms[start].surface)
        } else {
            kind.to_owned()
        };
        noun.surface = prefix + domain + &form_name;
        if writing {
            noun.surface.push('上');
        }
        noun.span.start_char = atoms[first].span.start_char;
        noun.is_locked = true;
        atoms.splice(first..=end, [noun]);
    }
    bind_filing_purpose(atoms, doc, chars);
}
