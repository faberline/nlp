use super::{after_clause_comma, skip_space, strip_prefix, trim_space_end};
use crate::lock::{LockOrigin, SpanLockTable};
use faberline_nlp::Span;

#[derive(Clone, Copy)]
enum Request {
    ReadAloud { as_written: bool },
    WriteAsWritten,
    Copy { as_written: bool },
    ReadingQuestion,
}

fn after_prefix<'a>(text: &'a [char], prefix: &str) -> Option<&'a [char]> {
    strip_prefix(skip_space(text), prefix).map(skip_space)
}

fn read_aloud(after_object: &[char]) -> Option<(bool, &[char])> {
    let (as_written, manner) =
        after_prefix(after_object, "そのまま").map_or((false, after_object), |text| (true, text));
    let after_voice = after_prefix(manner, "声に出して")?;
    if let Some(tail) = after_prefix(after_voice, "そのまま") {
        return (!as_written).then_some((true, tail));
    }
    Some((as_written, after_voice))
}

fn reading_question(after: &[char]) -> Option<&[char]> {
    let question = after_prefix(after, "は").or_else(|| after_prefix(after, "って"))?;
    let reading = ["何と", "なんと", "何て", "なんて"]
        .iter()
        .find_map(|marker| after_prefix(question, marker))?;
    after_prefix(reading, "読みますか")
}

fn parse(after: &[char]) -> Option<(Request, &[char])> {
    if let Some(tail) = reading_question(after) {
        return Some((Request::ReadingQuestion, tail));
    }
    let object = after_clause_comma(after_prefix(after, "を")?);
    if let Some((as_written, verb)) = read_aloud(object) {
        return after_prefix(verb, "読んでください")
            .map(|tail| (Request::ReadAloud { as_written }, tail));
    }
    let (as_written, verb) =
        after_prefix(object, "そのまま").map_or((false, object), |tail| (true, tail));
    if let Some(tail) = after_prefix(verb, "書き写してください") {
        return Some((Request::Copy { as_written }, tail));
    }
    if as_written {
        return after_prefix(verb, "書いてください").map(|tail| (Request::WriteAsWritten, tail));
    }
    None
}

fn at_boundary(tail: &[char]) -> bool {
    matches!(
        tail.first(),
        None | Some('。' | '！' | '!' | '？' | '?' | '」' | '』' | '\n')
    )
}

pub(super) fn protects(after: &[char]) -> bool {
    if parse(after).is_some_and(|(_, tail)| at_boundary(tail)) {
        return true;
    }
    // Statements explicitly reading the object aloud still preserve the written
    // operand. Their outer clause continues through the ordinary renderer.
    after_prefix(after, "を")
        .and_then(|object| read_aloud(after_clause_comma(object)))
        .is_some_and(|(_, verb)| {
            [
                "読む",
                "読みます",
                "読みました",
                "読んだ",
                "読まない",
                "読みません",
            ]
            .iter()
            .any(|form| after_prefix(verb, form).is_some_and(at_boundary))
        })
}

fn paper_source(before: &[char]) -> Option<&'static str> {
    match before {
        ['こ', 'の', '紙', 'の'] => Some("這張紙"),
        ['そ', 'の', '紙', 'の'] => Some("那張紙"),
        ['あ', 'の', '紙', 'の'] => Some("那張紙"),
        ['紙', 'の'] => Some("紙上"),
        _ => None,
    }
}

pub(crate) fn render(chars: &[char], span: Span, locks: &SpanLockTable) -> Option<String> {
    // Leave a more elaborate clause to the general renderer. In particular,
    // never swallow an opaque fragment or an unrelated caller glossary match.
    let [lock] = locks.locks.as_slice() else {
        return None;
    };
    if lock.origin != LockOrigin::QuotedSource {
        return None;
    }
    let open = lock.source_span.start_char.checked_sub(1)?;
    let close = lock.source_span.end_char;
    if open < span.start_char || close >= span.end_char {
        return None;
    }
    let (opener, closer) = (*chars.get(open)?, *chars.get(close)?);
    if !matches!((opener, closer), ('「', '」') | ('『', '』')) {
        return None;
    }

    let before = trim_space_end(skip_space(chars.get(span.start_char..open)?));
    let after = chars.get(close + 1..span.end_char)?;
    let (request, tail) = parse(after)?;
    if !matches!(trim_space_end(tail), [] | ['。' | '！' | '!' | '？' | '?']) {
        return None;
    }
    let quote = format!("{opener}{}{closer}", lock.placeholder);
    let text = match request {
        Request::ReadingQuestion if before.is_empty() => return Some(format!("{quote}怎麼讀？")),
        Request::ReadAloud { as_written } if before.is_empty() => {
            let manner = if as_written { "照原樣" } else { "" };
            format!("請{manner}把{quote}念出來")
        }
        Request::WriteAsWritten if before.is_empty() => format!("請照原樣寫下{quote}"),
        Request::Copy { as_written } => {
            let manner = if as_written { "原樣" } else { "" };
            if before.is_empty() {
                format!("請把{quote}{manner}抄寫下來")
            } else {
                let source = paper_source(before)?;
                if source == "紙上" {
                    format!("請把紙上的{quote}{manner}抄寫下來")
                } else {
                    format!("請照{source}，把{quote}{manner}抄寫下來")
                }
            }
        }
        _ => return None,
    };
    let punctuation = match tail.first() {
        Some('！' | '!') => "！",
        Some('？' | '?') => "？",
        Some('。') => "。",
        _ => "",
    };
    Some(text + punctuation)
}
