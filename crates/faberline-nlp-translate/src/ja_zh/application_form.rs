//! Select the use of a written application and the sense of 出す from its
//! own object, destination and source. Preserve the ordinary predicate path.
use super::{Atom, Document, Upos, dasu_case_link, direct_quote_time, is_clock_atom};

pub(super) const SUBMIT: &str = "__ja_submit_application";
pub(super) const OFFICE: &str = "__ja_application_recipient";

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
    (!form.is_locked
        && !prefix.is_locked
        && prefix.head == form.doc_tok_idx
        && prefix.deprel == "compound"
        && prefix.span.end_char == form.span.start_char
        && matches!(prefix.surface.as_str(), "申込" | "申込み" | "申し込み"))
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
    let text = match spelling(owner, chars).as_str() {
        "講座" | "コース" | "講習" => "課程報名表",
        "教室"
            if atoms[..i]
                .iter()
                .any(|a| a.head == owner.doc_tok_idx && a.deprel == "compound") =>
        {
            "課程報名表"
        }
        "大会" => "大會報名表",
        "イベント" => "活動報名表",
        "セミナー" => "研討會報名表",
        "入居" => "入住申請表",
        "住宅" => "住宅申請表",
        _ => return None,
    };
    Some((i, text))
}

// Return a receiving office only when its case belongs to this action.
fn receiving_office(atoms: &[Atom], case: usize, verb: usize, chars: &[char]) -> bool {
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
    matches!(
        raw.as_str(),
        "受付" | "窓口" | "事務局" | "事務室" | "管理会社" | "担当者" | "係員"
    )
}

fn physical_argument(atoms: &[Atom], verb: usize, chars: &[char], doc: &Document) -> bool {
    atoms.windows(2).any(|pair| {
        let [noun, case] = pair else { return false };
        if noun.head != Some(verb) || !dasu_case_link(noun, case) {
            return false;
        }
        let raw = spelling(noun, chars);
        match case.surface.as_str() {
            "から" => {
                direct_quote_time(noun, doc).is_none()
                    && !is_clock_atom(noun)
                    && !raw.ends_with(['時', '日', '月', '年'])
            }
            "に" | "へ" => matches!(
                raw.as_str(),
                "かばん" | "鞄" | "引き出し" | "封筒" | "袋" | "机" | "箱" | "上" | "中" | "外"
            ),
            _ => false,
        }
    })
}

pub(super) fn normalize(atoms: &mut Vec<Atom>, chars: &[char], doc: &Document) {
    for end in (0..atoms.len()).rev() {
        let Some(form) = atoms.get(end) else { continue };
        let Some(start) = form_start(atoms, end, chars) else {
            continue;
        };
        let context = purpose(atoms, start, form, chars);
        let mut filing = false;
        if form.deprel == "obj"
            && atoms
                .get(end + 1)
                .is_some_and(|wo| wo.surface == "を" && dasu_case_link(form, wo))
            && let Some(v) = atoms.iter().position(|a| {
                a.doc_tok_idx == form.head
                    && !a.is_locked
                    && a.upos == Some(Upos::Verb)
                    && a.lemma.as_deref() == Some("出す")
            })
            && let Some(verb) = atoms[v].doc_tok_idx
            && !physical_argument(atoms, verb, chars, doc)
        {
            let office = atoms.iter().enumerate().find_map(|(i, a)| {
                (matches!(a.surface.as_str(), "に" | "へ")
                    && receiving_office(atoms, i, verb, chars))
                .then(|| i - 1)
            });
            if context.is_some() || office.is_some() {
                atoms[v].lemma = Some(SUBMIT.to_owned());
                filing = true;
                if let Some(i) = office {
                    let a = &mut atoms[i];
                    if !a.is_locked
                        && let Some(text) = match a.surface.as_str() {
                            "窓口" => Some("窗口"),
                            "受付" => Some("櫃檯"),
                            _ => None,
                        }
                    {
                        a.surface = text.to_owned();
                        a.is_locked = true;
                    }
                    a.lemma = Some(OFFICE.to_owned());
                }
            }
        }
        let form = &atoms[end];
        if form.is_locked {
            continue;
        }
        let (first, target) = match context {
            Some((first, text)) if !atoms[first].is_locked => (first, text),
            Some((_, text)) => (
                start,
                if text.ends_with("報名表") {
                    "報名表"
                } else {
                    "申請表"
                },
            ),
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
        noun.surface = prefix + target;
        noun.span.start_char = atoms[first].span.start_char;
        noun.is_locked = true;
        atoms.splice(first..=end, [noun]);
    }
}
