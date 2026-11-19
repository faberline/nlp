//! Select the use of a written application and the sense of 出す from its
//! own object, destination and source. Preserve the ordinary predicate path.
use super::{Atom, Document, Upos, dasu_case_link, direct_quote_time, is_clock_atom};

pub(super) const SUBMIT: &str = "__ja_submit_application";
pub(super) const CAN_SUBMIT: &str = "__ja_can_submit_application";
pub(super) const OFFICE: &str = "__ja_application_recipient";
pub(super) const TIME_QUESTION: &str = "__ja_application_time_question";
const PRESENT_RECIPIENT: &str = "__ja_located_application_recipient";

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
    let name = raw
        .rsplit_once("にいる")
        .or_else(|| raw.rsplit_once('の'))
        .map_or(raw.as_str(), |(_, head)| head);
    original.surface = name.to_owned();
    super::is_companion_person(&original)
}

fn physical_argument(atoms: &[Atom], verb: usize, chars: &[char], doc: &Document) -> bool {
    atoms.windows(2).enumerate().any(|(i, pair)| {
        let [noun, case] = pair else { return false };
        if noun.head != Some(verb) || !dasu_case_link(noun, case) {
            return false;
        }
        let raw = spelling(noun, chars);
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
            "から" => !temporal,
            "に" | "へ" => !temporal && !recipient(atoms, i + 1, verb, chars),
            _ => false,
        }
    })
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
fn located_recipient(atoms: &mut Vec<Atom>, chars: &[char]) {
    for i in (0..atoms.len().saturating_sub(4)).rev() {
        let Some([place, ni, be, person, to]) = atoms.get(i..i + 5) else {
            continue;
        };
        if ni.surface != "に"
            || !dasu_case_link(place, ni)
            || place.head != be.doc_tok_idx
            || be.is_locked
            || !matches!(be.lemma.as_deref().unwrap_or(&be.surface), "いる" | "居る")
            || be.deprel != "acl"
            || be.head != person.doc_tok_idx
            || !matches!(to.surface.as_str(), "に" | "へ")
            || !dasu_case_link(person, to)
            || !is_person(person, chars)
        {
            continue;
        }
        let Some(outer) = person.head.filter(|id| {
            atoms
                .iter()
                .any(|v| v.doc_tok_idx == Some(*id) && filing_reading(v, atoms).is_some())
        }) else {
            continue;
        };
        let Some(form) = (0..i).rev().find(|&k| {
            atoms[k].deprel == "obj"
                && atoms[k].head == be.doc_tok_idx
                && atoms
                    .get(k + 1)
                    .is_some_and(|c| c.surface == "を" && dasu_case_link(&atoms[k], c))
                && (form_start(atoms, k, chars).is_some() || covered_purpose(&atoms[k], chars))
        }) else {
            continue;
        };
        let mut receiver = person.clone();
        let mut location = place.clone();
        office_name(&mut location);
        let text = |a: &Atom| {
            if a.is_locked {
                a.surface.clone()
            } else {
                super::translate_single_ja_term(&a.surface, a.lemma.as_deref(), a.upos)
            }
        };
        receiver.surface = format!("在{}的{}", text(&location), text(person));
        receiver.lemma = Some(PRESENT_RECIPIENT.to_owned());
        receiver.is_locked = true;
        receiver.span.start_char = place.span.start_char;
        let inner = be.doc_tok_idx;
        for subject in &mut atoms[..form] {
            if subject.deprel == "nsubj" && subject.head == inner {
                subject.head = Some(outer);
            }
        }
        atoms[form].head = Some(outer);
        atoms.splice(i..i + 4, [receiver]);
    }
}

pub(super) fn normalize(atoms: &mut Vec<Atom>, chars: &[char], doc: &Document) {
    located_recipient(atoms, chars);
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
            && let Some(reading) = filing_reading(&atoms[v], atoms)
            && let Some(verb) = atoms[v].doc_tok_idx
        {
            mark_filing_time(atoms, verb, chars);
            let office = atoms.iter().enumerate().find_map(|(i, a)| {
                (matches!(a.surface.as_str(), "に" | "へ") && recipient(atoms, i, verb, chars))
                    .then(|| i - 1)
            });
            if atoms[v].surface == "提出"
                || ((context.is_some() || office.is_some() || covered)
                    && !physical_argument(atoms, verb, chars, doc))
            {
                atoms[v].lemma = Some(reading.to_owned());
                filing = true;
                if let Some(i) = office {
                    let person = atoms[i].doc_tok_idx;
                    for modifier in &mut atoms[..i] {
                        if modifier.head == person && modifier.deprel == "nmod" {
                            office_name(modifier);
                        }
                    }
                    office_name(&mut atoms[i]);
                    atoms[i].lemma = Some(OFFICE.to_owned());
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
}
