//! Keep a linked elapsed-hour modifier on its course noun.

use faberline_nlp::doc::SyntaxToken;

use super::{
    Atom, Document, Upos, class_period::glossary_fragment_or, elapsed_duration_surface,
    elapsed_number, lookup_ja_lexicon_gloss, starts_inside_elapsed_number,
};
use crate::{lock::SpanLockTable, opaque::to_zh_numeral};

fn source_arc<'a>(atom: &Atom, doc: &'a Document, sentence: usize) -> Option<&'a SyntaxToken> {
    let index = atom.doc_tok_idx?;
    let token = doc.tokens.get(index)?;
    let link = doc.syntax.get(index)?;
    (token.span == atom.span
        && token.sentence == sentence
        && (link.token, link.span, link.sentence) == (index, atom.span, sentence)
        && link.head == atom.head
        && link.deprel == atom.deprel)
        .then_some(link)
}

fn course_phrase(
    atoms: &[Atom],
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<(Atom, usize)> {
    let (time, no, noun, used) = match atoms {
        [_, hours, no, noun, ..] if hours.surface == "時間" => (hours, no, noun, 4),
        [hours, no, noun, ..] => (hours, no, noun, 3),
        _ => return None,
    };
    let parts = &atoms[..used];
    let start = parts[0].span.start_char;
    if parts[..used - 1].iter().any(|atom| atom.is_locked)
        || !parts
            .windows(2)
            .all(|pair| pair[0].span.end_char == pair[1].span.start_char)
        || no.surface != "の"
        || no.upos != Some(Upos::Adp)
        || !matches!(time.upos, Some(Upos::Noun | Upos::Num))
    {
        return None;
    }
    let original = doc.tokens.get(noun.doc_tok_idx?)?;
    let sentence = original.sentence;
    let course = source_arc(noun, doc, sentence)?;
    let duration = source_arc(time, doc, sentence)?;
    let genitive = source_arc(no, doc, sentence)?;
    if original.upos != Some(Upos::Noun)
        || !matches!(
            course.deprel.as_str(),
            "nsubj" | "obj" | "iobj" | "obl" | "root"
        )
        || (duration.deprel.as_str(), duration.head) != ("nmod", Some(course.token))
        || (genitive.deprel.as_str(), genitive.head) != ("case", Some(duration.token))
    {
        return None;
    }
    if used == 4 {
        let number = source_arc(&parts[0], doc, sentence)?;
        if number.head != Some(duration.token)
            || !matches!(number.deprel.as_str(), "nummod" | "compound")
        {
            return None;
        }
    }
    let course_spelling: String = chars
        .get(noun.span.start_char..noun.span.end_char)?
        .iter()
        .collect();
    let default_label = match course_spelling.as_str() {
        "授業" => "課".to_owned(),
        "講習" | "講座" | "研修" => {
            lookup_ja_lexicon_gloss(&course_spelling, None, original.upos)?
        }
        _ => return None,
    };
    let label = glossary_fragment_or(
        parts,
        locks,
        noun.span.start_char,
        noun.span.end_char,
        &default_label,
    )?;
    let source: String = chars.get(start..time.span.end_char)?.iter().collect();
    let hours = elapsed_number(source.strip_suffix("時間")?)?;
    let rendered = elapsed_duration_surface(&format!("{}時間", to_zh_numeral(hours)))?;
    let mut combined = noun.clone();
    combined.span.start_char = start;
    combined.surface = format!("{rendered}的{label}");
    combined.upos = Some(Upos::Noun);
    combined.is_locked = true;
    Some((combined, used))
}

pub(super) fn normalize(
    atoms: &mut Vec<Atom>,
    doc: &Document,
    chars: &[char],
    locks: &SpanLockTable,
) {
    let mut i = 0;
    while i < atoms.len() {
        if !starts_inside_elapsed_number(atoms, i)
            && let Some((course, used)) = course_phrase(&atoms[i..], doc, chars, locks)
        {
            atoms.splice(i..i + used, [course]);
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary,
        analyze_and_translate,
    };

    const SOURCE: &str = "今日は二時間の授業があります。";

    #[test]
    fn duration_modifies_the_lesson() {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = TranslationGlossary::new();
        for (source, expected) in [
            (SOURCE, "今天有兩個小時的課。"),
            ("来週は3時間の授業があります。", "下週有三個小時的課。"),
            ("明日は１時間の講習があります。", "明天有一個小時的講習。"),
            ("会議は十三時に始まります。", "會議在十三點開始。"),
            ("今日は二時限分の授業があります。", "今天有兩節課。"),
            ("今日は`二時間の授業`があります。", "今天有`二時間の授業`。"),
        ] {
            let result =
                analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
            assert_eq!(result.target_text, expected, "{source}");
        }
    }

    #[test]
    fn supplied_syntax_and_course_label_are_preserved() {
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
        let doc = faberline_nlp::doc::analyze_document(
            SOURCE,
            &crate::doc_options_for_translate(&options, &glossary),
        );
        let translate = |document: &faberline_nlp::doc::Document| {
            crate::translate_document(
                SOURCE,
                document,
                #[cfg(feature = "narrative")]
                None,
                &options,
                &glossary,
                None,
            )
            .unwrap()
        };
        let result = translate(&doc);
        assert_eq!(result.target_text, "今天有兩個小時的課程。");
        let [hit] = result.glossary_hits.as_slice() else {
            panic!("expected one glossary hit");
        };
        assert_eq!(hit.canonical, "授業");
        assert_eq!(
            (hit.source_span.start_char, hit.source_span.end_char),
            (7, 9)
        );
        assert_eq!(
            (hit.target_span.start_char, hit.target_span.end_char),
            (8, 10)
        );

        let mut detached = doc;
        let predicate = detached
            .syntax
            .iter()
            .find(|link| link.deprel == "root")
            .unwrap()
            .token;
        let duration = detached
            .syntax
            .iter_mut()
            .find(|link| link.span.start_char == 4)
            .unwrap();
        duration.head = Some(predicate);
        duration.deprel = "obl".to_owned();
        let fallback = translate(&detached);
        assert!(!fallback.target_text.contains("兩個小時的課程"));
        let [hit] = fallback.glossary_hits.as_slice() else {
            panic!("expected the caller's lesson name after detaching the duration");
        };
        assert_eq!(hit.canonical, "授業");
        assert_eq!(
            (hit.source_span.start_char, hit.source_span.end_char),
            (7, 9)
        );
    }
}
