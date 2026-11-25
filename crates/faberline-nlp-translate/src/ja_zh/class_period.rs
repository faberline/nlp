//! Read a classroom period as an ordinal or a quantity of lessons.

use super::{Atom, Document, Upos, normalize_decimal_digits};
use crate::lock::{LockOrigin, SpanLockTable};
use crate::opaque::{parse_cjk_or_ascii_num, to_zh_numeral};
use crate::schema::GlossaryKind;

const CLASS_PERIOD: &str = "__ja_class_period";

pub(super) fn is_period(atom: &Atom) -> bool {
    atom.lemma.as_deref() == Some(CLASS_PERIOD)
}

fn is_digit(ch: char) -> bool {
    matches!(ch, '0'..='9' | '０'..='９') || "〇零一二三四五六七八九十百千万萬".contains(ch)
}

fn is_attached_modifier(
    part: &Atom,
    parts: &[Atom],
    doc: &Document,
    sentence: usize,
    number_start: usize,
) -> bool {
    let Some(index) = part.doc_tok_idx else {
        return false;
    };
    let (Some(token), Some(link)) = (doc.tokens.get(index), doc.syntax.get(index)) else {
        return false;
    };
    let relation = link.deprel.as_str();
    let ordinal_prefix = part.span.end_char == number_start;
    token.sentence == sentence
        && token.span == part.span
        && (link.token, link.sentence, link.span) == (index, sentence, part.span)
        && (matches!(relation, "compound" | "nummod")
            || (ordinal_prefix && matches!(relation, "amod" | "det")))
        // Every modifier reaches the final head through later parts of this phrase.
        && link.head.is_some_and(|head| {
            parts.iter().any(|other| {
                other.doc_tok_idx == Some(head) && other.span.start_char >= part.span.end_char
            })
        })
}

pub(super) fn glossary_fragment_or<'a>(
    parts: &'a [Atom],
    locks: &SpanLockTable,
    start: usize,
    end: usize,
    fallback: &'a str,
) -> Option<&'a str> {
    let mut protected = parts.iter().filter(|atom| atom.is_locked);
    match (protected.next(), protected.next()) {
        (None, None) => Some(fallback),
        (Some(atom), None)
            if atom.span.start_char == start
                && atom.span.end_char == end
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
            Some(&atom.surface)
        }
        _ => None,
    }
}

fn class_phrase(
    atoms: &[Atom],
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<(Atom, usize)> {
    let first = atoms.first()?;
    let start = first.span.start_char;
    let after_prefix = start + usize::from(chars.get(start) == Some(&'第'));
    let number = chars.get(after_prefix..)?;
    let digits = number.iter().take_while(|&&ch| is_digit(ch)).count();
    if digits == 0 {
        return None;
    }
    let suffix = &number[digits..];
    let ordinal = suffix.starts_with(&['時', '限', '目']);
    let suffix_len = if ordinal {
        3
    } else if after_prefix == start && suffix.starts_with(&['時', '限', '分', 'の', '授', '業'])
    {
        6
    } else {
        return None;
    };
    let end = after_prefix + digits + suffix_len;
    let used = atoms.partition_point(|atom| atom.span.end_char <= end);
    let parts = &atoms[..used];
    let (head, modifiers) = parts.split_last()?;
    let index = head.doc_tok_idx?;
    let original = doc.tokens.get(index)?;
    let link = doc.syntax.get(index)?;
    let noun_head = if ordinal {
        head.upos == Some(Upos::Noun)
    } else {
        original.upos == Some(Upos::Noun)
            && matches!(
                link.deprel.as_str(),
                "nsubj" | "obj" | "iobj" | "obl" | "root"
            )
    };
    if (
        link.token,
        link.sentence,
        link.span,
        link.head,
        link.deprel.as_str(),
    ) != (
        index,
        original.sentence,
        head.span,
        head.head,
        head.deprel.as_str(),
    ) || original.span != head.span
        || head.span.end_char != end
        || !noun_head
        || !parts
            .windows(2)
            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
        || !modifiers
            .iter()
            .all(|part| is_attached_modifier(part, parts, doc, original.sentence, after_prefix))
    {
        return None;
    }
    let digits = normalize_decimal_digits(&number[..digits].iter().collect::<String>());
    let number = parse_cjk_or_ascii_num(&digits)?;
    if number == 0 {
        return None;
    }
    let mut period = head.clone();
    period.span.start_char = start;
    if ordinal {
        let unit = glossary_fragment_or(parts, locks, end - 3, end - 1, "節課")?;
        period.surface = format!("第{digits}{unit}");
        period.lemma = Some(CLASS_PERIOD.to_owned());
    } else {
        let lesson = glossary_fragment_or(parts, locks, end - 2, end, "課")?;
        let count = if number == 2 {
            "兩".to_owned()
        } else {
            to_zh_numeral(number)
        };
        period.surface = format!("{count}節{lesson}");
        period.upos = Some(Upos::Noun);
    }
    period.is_locked = true;
    Some((period, used))
}

pub(super) fn normalize(
    atoms: &mut Vec<Atom>,
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) {
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
            && let Some((period, used)) = class_phrase(&atoms[index..], doc, chars, locks)
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
    fn class_quantities_keep_the_time_and_existential_clause() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("今日は二時限分の授業があります。", "今天有兩節課。"),
            ("明日は1時限分の授業があります。", "明天有一節課。"),
            ("来週は３時限分の授業があります。", "下週有三節課。"),
        ] {
            let translation =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(translation.target_text, expected, "{source}");
        }
    }

    #[test]
    fn lesson_duration_and_opaque_text_are_not_class_quantities() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let duration = analyze_and_translate(
            "今日は二時間の授業があります。",
            &options,
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert!(!duration.target_text.contains("節課"));

        let literal = analyze_and_translate(
            "今日は`二時限分の授業`があります。",
            &options,
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_eq!(literal.target_text, "今天有`二時限分の授業`。");
    }

    #[test]
    fn class_quantity_preserves_the_caller_lesson_name() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary {
            domain_terms: vec![GlossaryEntry {
                canonical: "授業".to_owned(),
                surfaces: vec!["授業".to_owned()],
                targets: [(TargetVariant::ZhHant, "課程".to_owned())].into(),
                ..GlossaryEntry::default()
            }],
            ..TranslationGlossary::new()
        };
        let translation = analyze_and_translate(
            "今日は二時限分の授業があります。",
            &options,
            &glossary,
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_eq!(translation.target_text, "今天有兩節課程。");
        let [hit] = translation.glossary_hits.as_slice() else {
            panic!("expected one glossary hit");
        };
        assert_eq!(hit.canonical, "授業");
        assert_eq!(
            (hit.source_span.start_char, hit.source_span.end_char),
            (8, 10)
        );
        assert_eq!(
            (hit.target_span.start_char, hit.target_span.end_char),
            (5, 7)
        );
    }

    #[test]
    fn a_class_period_keeps_its_subject_link() {
        let source = "二時限目は音楽です。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let translation = analyze_and_translate(
            source,
            &options,
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_eq!(translation.target_text, "第二節課是音樂。");
    }

    #[test]
    fn decimal_ordinals_keep_their_clause_roles() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            ("3時限目に数学を勉強します。", "在第3節課學習數學。"),
            ("４時限目を選びました。", "選擇了第4節課。"),
        ] {
            let translation =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(translation.target_text, expected, "{source}");
        }
    }

    #[test]
    fn a_caller_supplied_attachment_can_keep_period_parts_separate() {
        let source = "二時限目は音楽です。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        let document = faberline_nlp::doc::analyze_document(
            source,
            &crate::doc_options_for_translate(&options, &glossary),
        );
        let translate = |doc: &faberline_nlp::doc::Document| {
            crate::translate_document(
                source,
                doc,
                #[cfg(feature = "narrative")]
                None,
                &options,
                &glossary,
                None,
            )
            .unwrap()
            .target_text
        };
        assert_eq!(translate(&document), "第二節課是音樂。");

        let predicate = document
            .syntax
            .iter()
            .find(|arc| arc.deprel == "root")
            .unwrap()
            .token;
        let mut separate_number = document.clone();
        let number = separate_number
            .syntax
            .iter_mut()
            .find(|arc| arc.span.start_char == 0)
            .unwrap();
        number.head = Some(predicate);
        assert!(!translate(&separate_number).contains("第二節課"));

        let mut separate_unit = document;
        let unit = separate_unit
            .syntax
            .iter_mut()
            .find(|arc| arc.span.start_char == 1)
            .unwrap();
        unit.deprel = "obj".to_owned();
        assert!(!translate(&separate_unit).contains("第二節課"));
    }

    #[test]
    fn a_clock_time_stays_a_clock_time() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let translation = analyze_and_translate(
            "会議は十三時に始まります。",
            &options,
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap();
        assert_eq!(translation.target_text, "會議在十三點開始。");
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
