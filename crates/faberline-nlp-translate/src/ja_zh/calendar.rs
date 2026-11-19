//! Bind a calendar day to its case particles before noun-phrase grouping.
mod dates;
mod quantities;
pub(super) mod report;
use super::{Atom, Upos, normalize_decimal_digits, selects_time_value};
use crate::opaque::parse_cjk_or_ascii_num;
pub(crate) use dates::DateContext;
pub(super) use quantities::bind_relative_contrast;

const DATE: &str = "__ja_calendar_day";
const DAY_DURATION: &str = "__ja_day_duration";
const DATE_LIST: &str = "__ja_calendar_dates";
const INDIVIDUALS: &str = "__ja_one_person_at_a_time";
pub(super) fn individual_arrivals(atom: &Atom) -> bool {
    atom.lemma.as_deref() == Some(INDIVIDUALS)
}
const EACH_COUNT: &str = "__ja_calendar_each_count";
const EACH_SUBJECT: &str = "__ja_calendar_each_subject";
const INTERVAL: &str = "__ja_day_interval";
const MEASURED_SUBJECT: &str = "__ja_interval_subject_count";
const DISTRIBUTED_SUBJECT: &str = "__ja_distributed_subject_count";
const ONLY_MEASURED_SUBJECT: &str = "__ja_only_interval_subject_count";
const PROTECTED_INTERVAL: &str = "__ja_protected_day_interval";

fn regular_rate(surface: &str) -> bool {
    matches!(
        surface,
        "毎日" | "毎週" | "毎月" | "毎年" | "毎回" | "毎朝" | "毎晩"
    )
}

// A Japanese pause after the frequency can fall inside a single predicate.
// Keep a fronted time or a new subject separate from the following clause.
pub(super) fn continues_habitual_action(
    prefix: &[Atom],
    following: &[Atom],
    chars: &[char],
) -> bool {
    let Some((rate, before)) = prefix.split_last() else {
        return false;
    };
    let Some(governor) = rate.head else {
        return false;
    };
    let subject_of_action = |atom: &Atom| atom.deprel == "nsubj" && atom.head == Some(governor);
    if rate.is_locked
        || rate.deprel != "advmod"
        || !regular_rate(&rate.surface)
        || !before.iter().any(subject_of_action)
        || following.iter().any(subject_of_action)
    {
        return false;
    }
    let Some(action) = following
        .iter()
        .position(|a| a.doc_tok_idx == Some(governor))
    else {
        return false;
    };
    let verb = &following[action];
    if verb.upos != Some(Upos::Verb) || !matches!(verb.deprel.as_str(), "root" | "advcl" | "conj") {
        return false;
    }
    // Inspect source characters even when a glossary covers the delimiters or
    // tokenization omits them. Skip only the original pause after the time.
    let Some(between) = chars.get(rate.span.end_char..verb.span.start_char) else {
        return false;
    };
    let Some(pause) = between.iter().position(|c| !c.is_whitespace()) else {
        return false;
    };
    if !matches!(between[pause], '、' | '，') {
        return false;
    }
    !between[pause + 1..].iter().enumerate().any(|(offset, c)| {
        let index = pause + 1 + offset;
        match c {
            '「' | '」' | '『' | '』' | '"' | '＂' | '“' | '”' | '‘' | '、' | '，' | '。'
            | '！' | '？' | '!' | '?' | '…' => true,
            '\'' | '’' => {
                !(between
                    .get(index - 1)
                    .is_some_and(char::is_ascii_alphanumeric)
                    && between
                        .get(index + 1)
                        .is_some_and(char::is_ascii_alphanumeric))
            }
            ',' => {
                !(between[index - 1].is_ascii_digit()
                    && between[index + 1..]
                        .iter()
                        .take_while(|c| c.is_ascii_digit())
                        .count()
                        == 3)
            }
            _ => false,
        }
    })
}

pub(super) fn contrasts_recurrence(atoms: &[Atom]) -> bool {
    let core = super::core_non_punct(atoms);
    matches!(core, [rate, ..] if matches!(core.len(),3|4) && !rate.is_locked &&
        regular_rate(&rate.surface) &&
        super::clause_contains_dewa_naku(core))
}

pub(super) fn is_counted_subject(atom: &Atom) -> bool {
    matches!(
        atom.lemma.as_deref(),
        Some(MEASURED_SUBJECT | ONLY_MEASURED_SUBJECT | DISTRIBUTED_SUBJECT | EACH_SUBJECT)
    )
}

pub(super) fn arrival_noun(b: &super::Bunsetsu, target: super::TargetVariant) -> String {
    // 位 supplies the polite register of さん on a counted common role.
    if let [noun, title, case] = b.atoms.as_slice()
        && !noun.is_locked
        && noun.upos == Some(Upos::Noun)
        && !title.is_locked
        && title.surface == "さん"
        && case.surface == "が"
    {
        let name = super::translate_single_ja_term(&noun.surface, noun.lemma.as_deref(), noun.upos);
        return if name == "客" {
            "客人".to_owned()
        } else {
            name
        };
    }
    super::render_bunsetsu_np(b, target)
}
const REPETITIONS: &str = "__ja_day_repetitions";
const ONLY_REPETITIONS: &str = "__ja_day_only_repetitions";

pub(super) fn is_habit(atom: &Atom) -> bool {
    atom.lemma.as_deref() == Some(quantities::HABIT)
}

// A protected count cannot supply a translated classifier. Its visible
// distributive ずつ still belongs to the verb of the counted noun phrase.
pub(super) fn share_governor(atoms: &[Atom], doc: &super::Document) -> Option<usize> {
    let marker = atoms
        .iter()
        .find(|a| !a.is_locked && a.surface == "ずつ" && a.deprel == "discourse")?;
    let unit = doc.syntax.iter().find(|n| Some(n.token) == marker.head)?;
    let head = unit.head?;
    if doc.tokens.get(head)?.upos == Some(Upos::Verb) {
        return Some(head);
    }
    if unit.deprel == "nmod" && matches!(doc.tokens.get(head)?.upos, Some(Upos::Noun | Upos::Propn))
    {
        return doc.syntax.iter().find(|n| n.token == head)?.head;
    }
    None
}

pub(super) fn is_interval(atom: &Atom) -> bool {
    atom.lemma.as_deref() == Some(INTERVAL)
}

pub(super) fn is_day_duration(atom: &Atom) -> bool {
    atom.lemma.as_deref() == Some(DAY_DURATION)
}

fn counted_days(source: &[Atom]) -> Option<String> {
    let spelling = source
        .iter()
        .map(|a| a.surface.as_str())
        .collect::<String>();
    super::elapsed_duration_surface(&spelling)
}

// A focus marker immediately after a duration limits that amount. しか
// contributes "only" together with the negative ending of its own verb.
pub(super) fn take_focused_days(
    groups: &mut Vec<super::Bunsetsu>,
    source: &[Atom],
) -> Option<(String, usize, bool)> {
    for args in source.windows(2) {
        let [unit, marker] = args else {
            continue;
        };
        if unit.is_locked
            || unit.surface != "日間"
            || unit.doc_tok_idx.is_none()
            || marker.is_locked
            || marker.head != unit.doc_tok_idx
            || unit.span.end_char != marker.span.start_char
        {
            continue;
        }
        let shika = match (marker.surface.as_str(), marker.deprel.as_str()) {
            ("だけ", "discourse") => false,
            ("しか", "case") => true,
            _ => continue,
        };
        let owner = unit.head?;
        let predicate = groups.iter().find(|g| {
            g.role == super::BunsetsuRole::Predicate
                && g.atoms.iter().any(|a| {
                    !a.is_locked
                        && (a.doc_tok_idx == Some(owner)
                            || source.iter().any(|word| {
                                word.doc_tok_idx == Some(owner)
                                    && word.head == a.doc_tok_idx
                                    && a.lemma.as_deref() == Some("する")
                            }))
                })
        });
        let Some(predicate) = predicate else { continue };
        let negative = predicate.atoms.iter().any(|a| {
            !a.is_locked
                && (matches!(a.lemma.as_deref(), Some("ない" | "ぬ")) || a.surface == "ませ")
        });
        if shika && !negative {
            continue;
        }
        let Some(day) = groups
            .iter()
            .flat_map(|g| &g.atoms)
            .find(|a| is_day_duration(a) && a.doc_tok_idx == unit.doc_tok_idx)
        else {
            continue;
        };
        let amount = day.surface.clone();
        let span = day.span;
        if !groups
            .iter()
            .flat_map(|g| &g.atoms)
            .any(|a| a.span == marker.span && !a.is_locked)
        {
            continue;
        }
        for g in groups.iter_mut() {
            g.atoms.retain(|a| a.span != span && a.span != marker.span);
        }
        groups.retain(|g| !g.atoms.is_empty());
        return Some((amount, owner, shika));
    }
    None
}

// A counted 日間 can attach to an intervening noun or share its governor
// with a verbal noun. Use the original dependencies even across glossary spans.
pub(super) fn bind_day_arguments(atoms: &mut Vec<Atom>, doc: &super::Document) {
    for idx in (1..atoms.len()).rev() {
        let day = &atoms[idx];
        if day.is_locked || day.surface != "日間" || day.doc_tok_idx.is_none() {
            continue;
        }
        // The number and 日間 may share the head of a following noun phrase.
        let shared = is_numeral(&atoms[idx - 1])
            && atoms[idx - 1].deprel == "nummod"
            && atoms[idx - 1].span.end_char == day.span.start_char
            && atoms[idx - 1].head == day.head;
        let count_start = if shared {
            idx - 1
        } else if let Some((start, _)) = number_before(atoms, idx) {
            start
        } else {
            continue;
        };
        if counted_days(&atoms[count_start..=idx]).is_none() {
            continue;
        }
        if let [word, verb, ..] = &atoms[idx + 1..]
            && !word.is_locked
            && !verb.is_locked
            && word.deprel == "compound"
            && verb.lemma.as_deref().unwrap_or(&verb.surface) == "する"
            && word.head == verb.doc_tok_idx
            && day.head == verb.doc_tok_idx
            && word.span.end_char == verb.span.start_char
        {
            if shared {
                atoms[idx - 1].head = atoms[idx].doc_tok_idx;
            }
            atoms[idx].deprel = "obl".to_owned();
            continue;
        }
        let is_comma = |a: &Atom| !a.is_locked && matches!(a.surface.as_str(), "、" | "，" | ",");
        let pause = atoms.get(idx + 1).is_some_and(is_comma);
        let start = idx + 1 + usize::from(pause);
        let Some(case_pos) = atoms[start..]
            .iter()
            .position(|a| {
                a.upos == Some(Upos::Verb)
                    || super::is_punct_atom(a)
                    || matches!(a.surface.as_str(), "を" | "で")
            })
            .map(|offset| start + offset)
        else {
            continue;
        };
        let case = &atoms[case_pos];
        let Some(action) = atoms.get(case_pos + 1) else {
            continue;
        };
        let (Some(owner), Some(object)) = (action.doc_tok_idx, case.head) else {
            continue;
        };
        // With 「父は、二日間、家で休んだ」 both commas enclose a modifier
        // of the subject's own verb. A plain fronted interval keeps its pause.
        let topic_pause = matches!(&atoms[..count_start], [.., topic, comma]
            if pause && is_comma(comma) && !topic.is_locked && topic.surface == "は"
                && topic.deprel == "case" && doc.syntax.iter().any(|s|
                    Some(s.token) == topic.head && s.deprel == "nsubj" && s.head == Some(owner)));
        let relation = match case.surface.as_str() {
            "を" => "obj",
            "で" if topic_pause => "obl",
            _ => continue,
        };
        if case.is_locked
            || case_pos == start
            || case.deprel != "case"
            || action.is_locked
            || action.upos != Some(Upos::Verb)
            || !doc
                .syntax
                .iter()
                .any(|s| s.token == object && s.deprel == relation && s.head == Some(owner))
        {
            continue;
        }
        let phrase = |s: &faberline_nlp::doc::SyntaxToken| {
            s.span.start_char >= atoms[start].span.start_char
                && s.span.end_char <= case.span.end_char
        };
        let owns = |mut index: usize| {
            for _ in 0..=case.span.end_char - atoms[start].span.start_char {
                if index == object {
                    return true;
                }
                let Some(s) = doc.syntax.iter().find(|s| s.token == index && phrase(s)) else {
                    return false;
                };
                if !matches!(
                    s.deprel.as_str(),
                    "nmod" | "compound" | "amod" | "det" | "nummod" | "case"
                ) {
                    return false;
                }
                let Some(parent) = s.head else { return false };
                index = parent;
            }
            false
        };
        if !(day.head == Some(owner) || day.head.is_some_and(owns))
            || !doc
                .syntax
                .iter()
                .filter(|s| phrase(s))
                .all(|s| owns(s.token))
        {
            continue;
        }
        if shared {
            atoms[idx - 1].head = atoms[idx].doc_tok_idx;
        }
        atoms[idx].head = Some(owner);
        atoms[idx].deprel = "obl".to_owned();
        if pause {
            atoms.remove(idx + 1);
        }
        if topic_pause {
            atoms.remove(count_start - 1);
        }
    }
}

pub(super) fn asks_interval(atom: &Atom) -> bool {
    is_interval(atom) && atom.surface == "每隔幾天"
}

// 開催する may be a single verb, or a verbal noun joined to a separate する.
pub(super) fn is_held_event(source: &[Atom], owner: usize) -> bool {
    let Some(index) = source
        .iter()
        .position(|a| !a.is_locked && a.doc_tok_idx == Some(owner))
    else {
        return false;
    };
    let verb = &source[index];
    (verb.upos == Some(Upos::Verb)
        && matches!(
            verb.lemma.as_deref().unwrap_or(&verb.surface),
            "開催する" | "開催"
        ))
        || (verb.lemma.as_deref().unwrap_or(&verb.surface) == "する"
            && matches!(&source[..index], [.., noun] if !noun.is_locked && noun.surface == "開催"
                && noun.head == Some(owner) && noun.span.end_char == verb.span.start_char))
}

pub(super) fn count_modifier(atom: &Atom) -> Option<&'static str> {
    match atom.lemma.as_deref()? {
        REPETITIONS | MEASURED_SUBJECT => Some(""),
        ONLY_REPETITIONS | ONLY_MEASURED_SUBJECT => Some("只"),
        DISTRIBUTED_SUBJECT => Some("每次"),
        EACH_SUBJECT | EACH_COUNT => Some("各"),
        _ => None,
    }
}

pub(super) fn is_date(atom: &Atom) -> bool {
    matches!(atom.lemma.as_deref(), Some(DATE | DATE_LIST))
}

pub(super) fn question(atom: &Atom) -> Option<&str> {
    (is_date(atom) && (atom.surface.ends_with("幾號") || atom.surface == "星期幾"))
        .then_some(atom.surface.as_str())
}

pub(super) fn asks_event_argument(groups: &[super::Bunsetsu], source: &[Atom], ka: &Atom) -> bool {
    groups
        .iter()
        .flat_map(|group| &group.atoms)
        .any(|date| is_date(date) && date.head == ka.head)
        && source.windows(2).any(|pair| {
            let [wh, case] = pair else { return false };
            !wh.is_locked
                && wh.head == ka.head
                && matches!(
                    wh.surface.as_str(),
                    "誰" | "だれ" | "どなた" | "何" | "なに" | "どこ"
                )
                && matches!(wh.deprel.as_str(), "nsubj" | "obj" | "iobj" | "obl")
                && matches!(
                    case.surface.as_str(),
                    "が" | "に" | "へ" | "を" | "で" | "から"
                )
                && linked_case(case, wh, wh, &case.surface)
                && !source.iter().any(|p| {
                    p.doc_tok_idx != case.doc_tok_idx
                        && p.head == wh.doc_tok_idx
                        && matches!(p.surface.as_str(), "も" | "か" | "でも")
                })
        })
}

fn is_numeral(atom: &Atom) -> bool {
    !atom.is_locked
        && (atom.upos == Some(Upos::Num)
            || atom.upos == Some(Upos::Noun)
                && !atom.surface.is_empty()
                && atom.surface.chars().all(|c| ('０'..='９').contains(&c)))
}

fn calendar_number(text: &str) -> Option<u32> {
    if text.is_empty()
        || text.chars().count() > 7
        || !text.chars().all(|c| {
            matches!(c, '0'..='9' | '０'..='９' | '〇' | '零' | '一' | '二' | '三' | '四' |
            '五' | '六' | '七' | '八' | '九' | '十' | '百' | '千')
        })
    {
        return None;
    }
    parse_cjk_or_ascii_num(text)
}

// In 二十五日, 二 can attach to 十, while 十 and 五 both attach to 日.
fn number_before(atoms: &[Atom], end: usize) -> Option<(usize, String)> {
    let head = atoms[end].doc_tok_idx?;
    let mut start = end;
    while let Some(i) = start.checked_sub(1) {
        let part = &atoms[i];
        if !is_numeral(part)
            || part.span.end_char != atoms[start].span.start_char
            || !matches!(part.deprel.as_str(), "nmod" | "nummod" | "compound")
            || !part
                .head
                .is_some_and(|h| part.doc_tok_idx.is_some_and(|p| p < h && h <= head))
        {
            break;
        }
        start = i;
    }
    (start != end).then(|| {
        (
            start,
            atoms[start..end]
                .iter()
                .map(|a| a.surface.as_str())
                .collect(),
        )
    })
}

fn month_text(surface: &str) -> Option<String> {
    let number = surface.strip_suffix('月')?;
    if matches!(
        surface,
        "今月" | "先月" | "来月" | "翌月" | "前月" | "同月" | "当月" | "毎月"
    ) {
        return Some(super::translate_single_ja_term(
            surface,
            None,
            Some(Upos::Noun),
        ));
    }
    (1..=12)
        .contains(&calendar_number(number)?)
        .then(|| format!("{}月", normalize_decimal_digits(number)))
}

fn month_before(atoms: &[Atom], start: usize, day: &Atom) -> Option<(usize, String)> {
    let i = start.checked_sub(1)?;
    let month = &atoms[i];
    if month.is_locked
        || month.span.end_char != atoms[start].span.start_char
        || month.head != day.head
        || month.deprel != "advmod"
    {
        return None;
    }
    if month.surface != "月" {
        return Some((i, month_text(&month.surface)?));
    }
    let n = i.checked_sub(1)?;
    let num = &atoms[n];
    if !is_numeral(num)
        || num.span.end_char != month.span.start_char
        || !(num.head == month.doc_tok_idx || num.head == day.head)
    {
        return None;
    }
    Some((n, month_text(&format!("{}月", num.surface))?))
}

fn linked_case(case: &Atom, day: &Atom, adjacent_to: &Atom, surface: &str) -> bool {
    !case.is_locked
        && case.upos == Some(Upos::Adp)
        && case.surface == surface
        && case.deprel == "case"
        && day.doc_tok_idx.is_some()
        && case.head == day.doc_tok_idx
        && case.span.start_char == adjacent_to.span.end_char
}

fn adjacent_count(atoms: &[Atom], case_idx: usize) -> Option<(usize, &str)> {
    let case = &atoms[case_idx];
    let first_idx = case_idx + 1;
    let first = atoms.get(first_idx)?;
    if first.is_locked || first.span.start_char != case.span.end_char {
        return None;
    }
    if let Some(amount) = first.surface.strip_suffix(['度', '回'])
        && calendar_number(amount).is_some()
    {
        Some((first_idx, amount))
    } else if let Some(unit) = atoms.get(first_idx + 1)
        && is_numeral(first)
        && calendar_number(&first.surface).is_some()
        && !unit.is_locked
        && first.span.end_char == unit.span.start_char
        && matches!(unit.surface.as_str(), "度" | "回")
        && (first.head == unit.doc_tok_idx || first.head == unit.head)
    {
        Some((first_idx + 1, first.surface.as_str()))
    } else {
        None
    }
}

struct DayInterval {
    start: usize,
    dated: bool,
    atom: Atom,
}

fn genitive_month(atoms: &[Atom], start: usize, day: &Atom) -> bool {
    matches!(&atoms[..start], [.., month, no] if !month.is_locked && month.surface.ends_with('月')
        && month.head == day.doc_tok_idx && linked_case(no, month, month, "の"))
}

fn countable_day(day: &Atom) -> bool {
    !day.is_locked && day.surface == "日"
        || is_date(day)
        || day.lemma.as_deref() == Some(PROTECTED_INTERVAL)
}

fn day_interval(atoms: &[Atom], end: usize) -> Option<DayInterval> {
    let day = &atoms[end];
    if day.lemma.as_deref() == Some(PROTECTED_INTERVAL) {
        let mut interval = day.clone();
        interval.surface = format!("每{}", day.surface);
        interval.lemma = Some(INTERVAL.to_owned());
        interval.span.end_char = atoms.get(end + 1)?.span.end_char;
        return Some(DayInterval {
            start: end,
            dated: false,
            atom: interval,
        });
    }
    if is_date(day) {
        return Some(DayInterval {
            start: end,
            dated: true,
            atom: day.clone(),
        });
    }
    let case = atoms.get(end + 1)?;
    let prev_idx = end.checked_sub(1)?;
    let prev = &atoms[prev_idx];
    let (start, days) = if prev.surface == "何"
        && !prev.is_locked
        && prev.upos == Some(Upos::Pron)
        && prev.deprel == "compound"
        && prev.head == day.doc_tok_idx
        && prev.span.end_char == day.span.start_char
    {
        (prev_idx, "幾".to_owned())
    } else if let Some((start, raw)) = number_before(atoms, end) {
        let n = calendar_number(&raw)?;
        if n == 0 {
            return None;
        }
        (
            start,
            if raw == "二" {
                "兩".to_owned()
            } else {
                normalize_decimal_digits(&raw)
            },
        )
    } else {
        return None;
    };
    // A coordinated argument can hide its month in a protected phrase.
    // Only the positively recognized date list may supply a calendar date.
    if matches!(&atoms[..start], [.., left, join] if !join.is_locked && matches!(join.surface.as_str(),"と"|"や")
        && (left.is_locked || left.surface=="日") && join.head==left.doc_tok_idx)
        || matches!(&atoms[..start], [.., opaque] if opaque.is_locked && opaque.lemma.as_deref()!=Some(quantities::HABIT) && opaque.span.end_char == atoms[start].span.start_char)
        || matches!(&atoms[..start], [.., opaque, no] if opaque.is_locked && no.surface == "の" && no.head == opaque.doc_tok_idx)
    {
        return None;
    }
    let dated = month_before(atoms, start, day).is_some() || genitive_month(atoms, start, day);
    let mut atom = day.clone();
    atom.span.start_char = atoms[start].span.start_char;
    atom.span.end_char = case.span.end_char;
    atom.lemma = Some(INTERVAL.to_owned());
    atom.is_locked = true;
    atom.surface = match days.as_str() {
        "幾" => "每隔幾天".to_owned(),
        "一" | "1" => "每天".to_owned(),
        _ => format!("每{days}天"),
    };
    Some(DayInterval { start, dated, atom })
}

// Count the event in [day に] [number 回/度 (だけ)]. A case-marked argument
// can intervene in the dependency tree: 一度 can modify 薬 which modifies 飲む.
fn bind_repetitions(atoms: &mut Vec<Atom>, end: usize) {
    let day = &atoms[end];
    let Some(case) = atoms.get(end + 1) else {
        return;
    };
    if !countable_day(day)
        || day.upos != Some(Upos::Noun)
        || !matches!(day.deprel.as_str(), "obl" | "iobj")
        || !linked_case(case, day, day, "に")
    {
        return;
    }
    let Some(day_head) = day.head else { return };
    let Some((unit_idx, amount)) = adjacent_count(atoms, end + 1) else {
        return;
    };
    let first_idx = end + 2;
    let first = &atoms[first_idx];
    let unit = &atoms[unit_idx];
    let Some(unit_head) = unit.head else { return };
    let next = atoms.get(unit_idx + 1);
    let focused = next.is_some_and(|a| {
        !a.is_locked
            && a.surface == "だけ"
            && a.upos == Some(Upos::Part)
            && a.deprel == "discourse"
            && a.head == unit.doc_tok_idx
            && a.span.start_char == unit.span.end_char
    });
    // の makes a nominal rate, and も takes scope over the negative event.
    if !focused
        && next.is_some_and(|a| !a.is_locked && matches!(a.upos, Some(Upos::Adp | Upos::Part)))
    {
        return;
    }
    let fragment = atoms[unit_idx + 1 + usize::from(focused)..].is_empty()
        && !atoms.iter().any(|a| a.upos == Some(Upos::Verb));
    if !fragment
        && unit_head != day_head
        && !(matches!(unit.deprel.as_str(), "nmod" | "compound")
            && atoms.iter().any(|a| {
                !a.is_locked
                    && a.doc_tok_idx == Some(unit_head)
                    && a.head == Some(day_head)
                    && matches!(a.upos, Some(Upos::Noun | Upos::Propn))
            }))
    {
        return;
    }
    // Keep the interval, the count and the event inside their clause.
    let root_event = atoms.iter().any(|a| {
        !a.is_locked
            && a.doc_tok_idx == Some(day_head)
            && a.upos == Some(Upos::Verb)
            && a.deprel == "root"
    });
    if !root_event && !fragment {
        return;
    }
    let Some(DayInterval {
        start,
        dated,
        atom: mut interval,
    }) = day_interval(atoms, end)
    else {
        return;
    };
    let mut repeat = unit.clone();
    repeat.surface = format!(
        "{}次",
        if amount == "二" {
            "兩".to_owned()
        } else {
            normalize_decimal_digits(amount)
        }
    );
    repeat.span.start_char = first.span.start_char;
    repeat.span.end_char = if focused {
        atoms[unit_idx + 1].span.end_char
    } else {
        unit.span.end_char
    };
    repeat.lemma = Some(
        if focused {
            ONLY_REPETITIONS
        } else {
            REPETITIONS
        }
        .to_owned(),
    );
    repeat.head = Some(day_head);
    repeat.is_locked = true;
    if fragment {
        if !dated {
            interval.span.end_char = repeat.span.end_char;
            if focused {
                interval.surface.push('僅');
            }
            interval.surface.push_str(&repeat.surface);
            atoms.splice(start..=unit_idx + usize::from(focused), [interval]);
        }
        return;
    }
    atoms.splice(first_idx..=unit_idx + usize::from(focused), [repeat]);
    if !dated {
        atoms.splice(start..=end + 1, [interval]);
    }
}

// A year that modifies this month belongs inside the same date phrase.
// Leave other attached modifiers and era years to their normal renderer.
fn dated_month(atoms: &[Atom], start: usize, month: String) -> Option<(usize, String)> {
    let Some((year, before)) = atoms[..start].split_last() else {
        return Some((start, month));
    };
    if year.head != atoms[start].doc_tok_idx
        || year.span.end_char != atoms[start].span.start_char
        || year.deprel != "advmod"
    {
        return Some((start, month));
    }
    if year.is_locked {
        return None;
    }
    let relative = match year.surface.as_str() {
        "毎年" => "每年",
        "去年" => "去年",
        "今年" => "今年",
        "来年" => "明年",
        "年" => {
            let mut first = before.len();
            while first > 0
                && is_numeral(&atoms[first - 1])
                && atoms[first - 1].span.end_char == atoms[first].span.start_char
            {
                first -= 1;
            }
            let digits: String = before[first..].iter().map(|a| a.surface.as_str()).collect();
            if !(1000..=9999).contains(&calendar_number(&digits)?) {
                return None;
            }
            return Some((
                first,
                format!("{}年{month}", normalize_decimal_digits(&digits)),
            ));
        }
        _ => return None,
    };
    Some((start - 1, format!("{relative}{month}")))
}

fn expression(atoms: &[Atom], end: usize, contrasts_recurrence: bool) -> Option<(usize, String)> {
    let day = &atoms[end];
    if day.is_locked || day.upos != Some(Upos::Noun) {
        return None;
    }
    if day.surface == "日間" {
        let (start, n) = number_before(atoms, end)?;
        let count = if n == "二" {
            "兩".to_owned()
        } else {
            normalize_decimal_digits(&n)
        };
        return Some((start, format!("{count}天")));
    }
    if day.surface != "日" || !matches!(day.deprel.as_str(), "obl" | "iobj") {
        return None;
    }
    let case = atoms.get(end + 1)?;
    let deadline = linked_case(case, day, day, "まで")
        && atoms
            .get(end + 2)
            .is_some_and(|ni| linked_case(ni, day, case, "に"));
    if !deadline && !linked_case(case, day, day, "に") {
        return None;
    }
    let verb = atoms
        .iter()
        .find(|a| a.doc_tok_idx == day.head && !a.is_locked && a.upos == Some(Upos::Verb))?;
    if !deadline && selects_time_value(verb.lemma.as_deref().unwrap_or(&verb.surface)) {
        return None;
    }
    // Counted repetitions turn 何日 into an interval: 何日に一回, 何日に三度.
    // A glossary lock followed by だけ can hide the repetition count.
    // An explicit month still identifies a date independently of that lock.
    let opaque_focus = matches!(&atoms[end + 2..], [locked, only, ..] if locked.is_locked
        && locked.surface.contains("[[PH") && locked.span.start_char == case.span.end_char
        && !only.is_locked && only.upos == Some(Upos::Part) && matches!(only.surface.as_str(), "だけ" | "ずつ")
        && only.span.start_char == locked.span.end_char);
    let frequency = !deadline
        && (contrasts_recurrence
            || atoms
                .iter()
                .any(|a| a.lemma.as_deref() == Some(quantities::HABIT) && a.head == day.head)
            || opaque_focus
            || quantities::has_measure(atoms, end)
            || adjacent_count(atoms, end + 1).is_some()
            || atoms[end + 2..].iter().any(|a| {
                a.head == verb.doc_tok_idx
                    && !a.is_locked
                    && ["回", "度"].iter().any(|suffix| {
                        a.surface
                            .strip_suffix(suffix)
                            .is_some_and(|n| n.is_empty() || calendar_number(n).is_some())
                    })
            }));
    let linked_modifier = |a: &Atom, next: &Atom| {
        !a.is_locked
            && a.deprel == "compound"
            && a.head == day.doc_tok_idx
            && a.span.end_char == next.span.start_char
    };
    let i = end.checked_sub(1)?;
    let prev = &atoms[i];
    let (q, text) =
        if prev.surface == "曜" && prev.upos == Some(Upos::Noun) && linked_modifier(prev, day) {
            (i.checked_sub(1)?, "星期幾")
        } else {
            (i, "幾號")
        };
    if atoms[q].surface == "何"
        && atoms[q].upos == Some(Upos::Pron)
        && linked_modifier(&atoms[q], &atoms[q + 1])
    {
        if text == "幾號"
            && let Some((start, month)) = month_before(atoms, q, day)
        {
            let (start, month) = dated_month(atoms, start, month)?;
            return Some((start, format!("{month}{text}")));
        }
        return (text != "幾號" || !frequency || genitive_month(atoms, q, day))
            .then(|| (q, text.to_owned()));
    }
    let (start, n) = number_before(atoms, end)?;
    let value = calendar_number(&n)?;
    if !(1..=31).contains(&value) {
        return None;
    }
    let n = normalize_decimal_digits(&n);
    if let Some((start, month)) = month_before(atoms, start, day) {
        let (start, month) = dated_month(atoms, start, month)?;
        return Some((start, format!("{month}{n}日")));
    }
    // A bare numbered day followed by a count can express a rate (二日に一回).
    if !genitive_month(atoms, start, day)
        && (frequency || (!deadline && value == 1 && atoms[end + 2..].iter().any(is_numeral)))
    {
        return None;
    }
    Some((start, format!("{n}號")))
}

pub(super) fn normalize(atoms: &mut Vec<Atom>, contrasts_recurrence: bool) {
    // Work from the right: binding a count can collapse both the counted
    // phrase and its day interval. Start the date pass with the resulting spans.
    quantities::normalize_habit(atoms);
    for idx in (0..atoms.len()).rev() {
        let Some(day) = atoms.get(idx).filter(|a| countable_day(a)) else {
            continue;
        };
        let recurring = contrasts_recurrence
            || atoms
                .iter()
                .any(|a| a.lemma.as_deref() == Some(quantities::HABIT) && a.head == day.head)
            || atoms.iter().any(|a| {
                !a.is_locked
                    && a.head == day.head
                    && a.deprel == "advmod"
                    && matches!(a.surface.as_str(), "いつも" | "毎回" | "繰り返し")
            });
        quantities::bind(atoms, idx, recurring);
        if atoms.get(idx).is_some_and(countable_day) {
            bind_repetitions(atoms, idx);
        }
        // The amount may keep its native の or an unsupported classifier.
        // An explicit contrast still resolves the interval independently.
        if recurring
            && atoms.get(idx).is_some_and(|d| {
                atoms.iter().any(|a| {
                    a.doc_tok_idx == d.head && a.deprel == "root" && a.upos == Some(Upos::Verb)
                })
            })
            && atoms
                .get(idx)
                .is_some_and(|a| !a.is_locked && a.surface == "日")
            && atoms
                .get(idx + 1)
                .is_some_and(|ni| linked_case(ni, &atoms[idx], &atoms[idx], "に"))
            && quantities::has_measure(atoms, idx)
            && let Some(DayInterval {
                start,
                dated: false,
                atom,
            }) = day_interval(atoms, idx)
        {
            atoms.splice(start..idx + 2, [atom]);
        }
    }
    // The resolved count already supplies 各 for a coordinated date list.
    let mut i = 0;
    while i < atoms.len() {
        let each = &atoms[i];
        if !each.is_locked
            && each.surface == "それぞれ"
            && each.deprel == "advmod"
            && atoms.iter().any(|a| {
                a.head == each.head && matches!(a.lemma.as_deref(), Some(EACH_COUNT | EACH_SUBJECT))
            })
        {
            atoms.remove(i);
        } else {
            i += 1;
        }
    }
    // Voiced だ after an adnominal verb (読んだ) is the past auxiliary た.
    // The generic noun modifier renderer otherwise treats it as a copula.
    for i in 1..atoms.len() {
        let verb = &atoms[i - 1];
        let past = &atoms[i];
        if !past.is_locked
            && past.surface == "だ"
            && past.upos == Some(Upos::Aux)
            && past.deprel == "aux"
            && past.head == verb.doc_tok_idx
            && !verb.is_locked
            && verb.upos == Some(Upos::Verb)
            && verb.deprel == "acl"
            && atoms[..i - 1].iter().any(|a| {
                (is_interval(a) || count_modifier(a).is_some()) && a.head == verb.doc_tok_idx
            })
        {
            atoms[i].surface = "た".to_owned();
        }
    }
    let mut end = atoms.len();
    while end > 0 {
        end -= 1;
        if let Some((start, text)) = expression(atoms, end, contrasts_recurrence) {
            let mut day = atoms[end].clone();
            let duration = (day.surface == "日間")
                .then(|| counted_days(&atoms[start..=end]))
                .flatten();
            day.lemma = Some(
                if duration.is_some() {
                    DAY_DURATION
                } else {
                    DATE
                }
                .to_owned(),
            );
            day.surface = duration.unwrap_or(text);
            day.is_locked = true;
            day.span.start_char = atoms[start].span.start_char;
            atoms.splice(start..=end, [day]);
            end = start;
        }
    }
}
