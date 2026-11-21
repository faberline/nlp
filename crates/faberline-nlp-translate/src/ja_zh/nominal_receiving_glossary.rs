use super::*;

// A term override can hide the noun's normal POS or absorb its case marker.
// Run the ordinary nominal-request parser on the source and then substitute
// only the proven object. The segment and lock table retain their own spans.
pub(super) fn render(atoms: &[Atom], chars: &[char]) -> Option<String> {
    let mut locked = atoms.iter().filter(|atom| atom.is_locked);
    let lock = locked.next()?;
    if lock.lemma.as_deref() != Some(LOCKED_OBJECT_ATOM)
        || locked.next().is_some()
        || !matches!(atoms, [.., verb, masu, neg, question]
            if verb.surface == "いただけ" && masu.surface == "ませ"
                && neg.surface == "ん" && question.surface == "か")
    {
        return None;
    }
    let start = atoms.first()?.span.start_char;
    let end = atoms.last()?.span.end_char;
    let end = end
        + usize::from(
            chars
                .get(end)
                .is_some_and(|p| matches!(p, '。' | '？' | '?')),
        );
    let source = chars.get(start..end)?;
    let text: String = source.iter().collect();
    let original = faberline_nlp::doc::analyze_document(
        &text,
        &faberline_nlp::doc::Options {
            normalize_traditional: true,
            enrich_morphology: true,
            cpu_syntax: true,
            ..Default::default()
        },
    );
    if original.segmentation.sentences.len() != 1 {
        return None;
    }
    let tokens: Vec<_> = original.tokens.iter().collect();
    let plain = build_sentence_atoms(0, source, &original, &tokens, &SpanLockTable::default());
    let mut relative = lock.clone();
    relative.span.start_char = lock.span.start_char.checked_sub(start)?;
    relative.span.end_char = lock.span.end_char.checked_sub(start)?;
    render_nominal_receiving_request(
        core_non_punct(&plain),
        &original,
        0,
        Some(&relative),
        source,
    )
}

pub(super) fn render_object(
    phrase: &[Atom],
    lock: &Atom,
    doc: &Document,
    sentence: usize,
    source: &[char],
    time_request: bool,
) -> Option<String> {
    let [.., noun, particle] = phrase else {
        return None;
    };
    let nominal_start = phrase.first()?.span.start_char;
    if lock.span.end_char != noun.span.end_char && lock.span.end_char != particle.span.end_char {
        return None;
    }
    if lock.span.start_char == nominal_start {
        return Some(lock.surface.clone());
    }
    // お only marks courtesy on this already proven time request.
    if time_request
        && source.get(nominal_start..lock.span.start_char) == Some(&['お'])
        && source.get(lock.span.start_char..noun.span.end_char) == Some(&['時', '間'])
    {
        return Some(lock.surface.clone());
    }
    if lock.span.start_char != noun.span.start_char {
        return None;
    }
    let mut object = phrase[..phrase.len() - 1].to_vec();
    let masked_noun = object.last_mut()?;
    masked_noun.surface.clone_from(&lock.surface);
    masked_noun.is_locked = true;
    nominal_possession::choice_phrase(&object, doc, sentence, TargetVariant::ZhHant, source)
}

#[cfg(test)]
mod tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

    fn translate(source: &str, term: &str, label: &str) -> String {
        let mut options = TranslateOptions::new(TargetVariant::ZhHant);
        options.source_hint = Some(TargetVariant::Ja);
        let glossary = serde_json::from_value::<TranslationGlossary>(serde_json::json!({
            "schema": "nlp.translate.glossary.v1",
            "objects": [{
                "canonical": term,
                "surfaces": [term],
                "targets": {"zh-Hant": label}
            }]
        }))
        .unwrap();
        let result =
            analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
        let [hit] = result.glossary_hits.as_slice() else {
            panic!("expected one glossary hit for {term}");
        };
        let slice = |text: &str, span: faberline_nlp::Span| -> String {
            text.chars()
                .skip(span.start_char)
                .take(span.end_char - span.start_char)
                .collect()
        };
        assert_eq!(slice(source, hit.source_span), term);
        assert_eq!(slice(&result.target_text, hit.target_span), label);
        result.target_text
    }

    #[test]
    fn object_glossary_keeps_the_nominal_request() {
        for (term, label) in [
            ("資料", "講義"),
            ("この資料", "這份講義"),
            ("資料を", "講義"),
            ("この資料を", "這份講義"),
        ] {
            assert_eq!(
                translate("この資料をいただけませんか。", term, label),
                "可以給我這份講義嗎？",
                "{term}",
            );
        }
        for term in ["時間", "お時間", "時間を", "お時間を"] {
            assert_eq!(
                translate(
                    "明日の午後、少しお時間をいただけませんか。",
                    term,
                    "談話時間"
                ),
                "明天下午，可以占用您一點談話時間嗎？",
                "{term}",
            );
        }
    }

    #[test]
    fn eligibility_question_keeps_the_restriction_with_a_glossary() {
        let result = translate(
            "この資料は参加者だけに配ります。参加者以外の人は資料をいただけませんか。",
            "資料を",
            "講義",
        );
        assert!(result.contains("參加者") && !result.contains("可以給我"));
    }
}
