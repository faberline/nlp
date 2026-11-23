//! Read a numbered classroom period without treating 目 as a body part.

use super::{Atom, Upos, normalize_decimal_digits};
use crate::lock::{LockOrigin, SpanLockTable};
use crate::opaque::parse_cjk_or_ascii_num;
use crate::schema::GlossaryKind;

const CLASS_PERIOD: &str = "__ja_class_period";

pub(super) fn is_period(atom: &Atom) -> bool {
    atom.lemma.as_deref() == Some(CLASS_PERIOD)
}

fn is_digit(ch: char) -> bool {
    matches!(ch, '0'..='9' | '０'..='９') || "〇零一二三四五六七八九十百千万萬".contains(ch)
}

fn ordinal(atoms: &[Atom], chars: &[char], locks: &SpanLockTable) -> Option<(Atom, usize)> {
    let first = atoms.first()?;
    let start = first.span.start_char;
    let after_prefix = start + usize::from(chars.get(start) == Some(&'第'));
    let number = chars.get(after_prefix..)?;
    let digits = number.iter().take_while(|&&ch| is_digit(ch)).count();
    if digits == 0 || number.get(digits..digits + 3) != Some(&['時', '限', '目']) {
        return None;
    }
    let end = after_prefix + digits + 3;
    let used = atoms.partition_point(|atom| atom.span.end_char <= end);
    let parts = &atoms[..used];
    let head = parts.last()?;
    if head.span.end_char != end
        || head.upos != Some(Upos::Noun)
        || head.doc_tok_idx.is_none()
        || !parts
            .windows(2)
            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
    {
        return None;
    }
    let numeral = normalize_decimal_digits(&number[..digits].iter().collect::<String>());
    if parse_cjk_or_ascii_num(&numeral)? == 0 {
        return None;
    }
    let mut protected = parts.iter().filter(|atom| atom.is_locked);
    let unit = match (protected.next(), protected.next()) {
        (None, None) => "節課",
        (Some(atom), None)
            if atom.span.start_char == end - 3
                && atom.span.end_char == end - 1
                && locks.locks.iter().any(|lock| {
                    lock.mandatory
                        && lock.source_span == atom.span
                        && lock.placeholder == atom.surface
                        && matches!(
                            lock.origin,
                            LockOrigin::Glossary {
                                kind: GlossaryKind::DomainTerm | GlossaryKind::Object,
                                ..
                            }
                        )
                }) =>
        {
            &atom.surface
        }
        _ => return None,
    };
    let mut period = head.clone();
    period.span.start_char = start;
    period.surface = format!("第{numeral}{unit}");
    period.lemma = Some(CLASS_PERIOD.to_owned());
    period.is_locked = true;
    Some((period, used))
}

pub(super) fn normalize(atoms: &mut Vec<Atom>, chars: &[char], locks: &SpanLockTable) {
    let mut index = 0;
    while index < atoms.len() {
        let start = atoms[index].span.start_char;
        // Do not start inside a numeral, an explicit ordinal or a glossary span.
        let inside_number = start.checked_sub(1).is_some_and(|prev| {
            chars
                .get(prev)
                .is_some_and(|&ch| ch == '第' || is_digit(ch))
        });
        if !atoms[index].is_locked
            && !inside_number
            && let Some((period, used)) = ordinal(&atoms[index..], chars, locks)
        {
            atoms.splice(index..index + used, [period]);
        }
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary,
        analyze_and_translate,
    };

    #[test]
    fn numbered_periods_keep_the_ordinal_and_the_clause_role() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("二時限目は音楽です。", "第二節課是音樂。"),
            ("１時限目に数学を勉強します。", "在第1節課學習數學。"),
            ("十一時限目が始まります。", "第十一節課開始。"),
            ("第２時限目は音楽です。", "第2節課是音樂。"),
            ("三時限目を選びました。", "選擇了第三節課。"),
            ("二時限の授業があります。", "有二時限授業。"),
            ("三時に目が覚めました。", "眼睛在三點甦醒了。"),
            ("`二時限目`は音楽です。", "`二時限目`是音樂。"),
        ] {
            let actual = analyze_and_translate(source, &options, &glossary, None, Backend::Onnx)
                .unwrap()
                .target_text;
            assert_eq!(actual, expected, "{source}");
        }
    }

    #[test]
    fn glossary_locked_period_unit_keeps_the_ordinal_and_exact_hit() {
        let source = "二時限目は音楽です。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary {
            domain_terms: vec![GlossaryEntry {
                canonical: "時限".to_owned(),
                surfaces: vec!["時限".to_owned()],
                targets: [(TargetVariant::ZhHant, "節次".to_owned())].into(),
                ..GlossaryEntry::default()
            }],
            ..TranslationGlossary::new()
        };
        let translated =
            analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
        assert_eq!(translated.target_text, "第二節次是音樂。");
        let [hit] = translated.glossary_hits.as_slice() else {
            panic!("expected one glossary hit");
        };
        assert_eq!(hit.canonical, "時限");
        assert_eq!(
            (hit.source_span.start_char, hit.source_span.end_char),
            (1, 3)
        );
        assert_eq!(
            (hit.target_span.start_char, hit.target_span.end_char),
            (2, 4)
        );
    }
}
