use super::*;

// Prove the request against the original tokens before using the placeholder
// as the object. Only an internally produced CPU analysis can recover a token
// boundary hidden by the caller's glossary.
pub(super) fn render(
    atoms: &[Atom],
    chars: &[char],
    doc: &Document,
    sent_tokens: &[&Token],
    sentence: usize,
    locks: &SpanLockTable,
    analysis: NominalRequestAnalysis,
) -> Option<String> {
    let mut locked = atoms.iter().filter(|atom| atom.is_locked);
    let atom = locked.next()?;
    if locked.next().is_some()
        || !matches!(atoms, [.., verb, masu, neg, question]
            if verb.surface == "いただけ" && masu.surface == "ませ"
                && neg.surface == "ん" && question.surface == "か")
    {
        return None;
    }
    let lock = locks.locks.iter().find(|lock| {
        lock.mandatory
            && lock.source_span == atom.span
            && lock.placeholder == atom.surface
            && matches!(
                lock.origin,
                LockOrigin::Glossary {
                    kind: GlossaryKind::Object | GlossaryKind::DomainTerm,
                    ..
                }
            )
    })?;
    let start = atoms.first()?.span.start_char;
    let end = atoms.last()?.span.end_char;
    let follows_context = nominal_request_follows_paragraph_context(doc, sentence);
    let unlocked = SpanLockTable::default();
    let source_atoms = build_sentence_atoms(sentence, chars, doc, sent_tokens, &unlocked);
    let left = source_atoms.partition_point(|a| a.span.end_char <= start);
    let right = source_atoms.partition_point(|a| a.span.start_char < end);
    let clause = source_atoms.get(left..right)?;
    if clause.first().is_some_and(|a| a.span.start_char == start)
        && clause.last().is_some_and(|a| a.span.end_char == end)
        && let Some(request) = render_nominal_receiving_request(
            clause,
            doc,
            sentence,
            Some(lock),
            chars,
            follows_context,
        )
    {
        return Some(request);
    }
    if !matches!(analysis, NominalRequestAnalysis::OwnedCpu)
        || doc
            .segmentation
            .opaque
            .iter()
            .any(|op| op.span.start_char < end && start < op.span.end_char)
    {
        return None;
    }
    let term: String = chars
        .get(lock.source_span.start_char..lock.source_span.end_char)?
        .iter()
        .collect();
    let mut options = doc.options.clone();
    options.lexicon.retain(|entry| *entry != term);
    if options.lexicon.len() == doc.options.lexicon.len() {
        return None;
    }
    let end = end
        + usize::from(
            chars
                .get(end)
                .is_some_and(|p| matches!(p, '。' | '？' | '?')),
        );
    let source = chars.get(start..end)?;
    let text: String = source.iter().collect();
    let original = faberline_nlp::doc::analyze_document(&text, &options);
    if original.segmentation.sentences.len() != 1 || !original.segmentation.opaque.is_empty() {
        return None;
    }
    let tokens: Vec<_> = original.tokens.iter().collect();
    let plain = build_sentence_atoms(0, source, &original, &tokens, &unlocked);
    let mut relative = lock.clone();
    relative.source_span.start_char = lock.source_span.start_char.checked_sub(start)?;
    relative.source_span.end_char = lock.source_span.end_char.checked_sub(start)?;
    render_nominal_receiving_request(
        core_non_punct(&plain),
        &original,
        0,
        Some(&relative),
        source,
        follows_context,
    )
}

pub(super) fn render_object(
    phrase: &[Atom],
    lock: &LockedSpan,
    doc: &Document,
    sentence: usize,
    source: &[char],
    time_request: bool,
) -> Option<String> {
    let [.., noun, particle] = phrase else {
        return None;
    };
    if matches!(
        lock.origin,
        LockOrigin::Glossary {
            kind: GlossaryKind::DomainTerm,
            ..
        }
    ) && !(time_request && has_courtesy_time_prefix(phrase))
        && !matches!(
            translate_single_ja_term(&noun.surface, noun.lemma.as_deref(), noun.upos).as_str(),
            "資料" | "文件" | "報告" | "報紙"
        )
    {
        return None;
    }
    let nominal_start = phrase.first()?.span.start_char;
    if lock.source_span.end_char != noun.span.end_char
        && lock.source_span.end_char != particle.span.end_char
    {
        return None;
    }
    if lock.source_span.start_char == nominal_start {
        return Some(lock.placeholder.clone());
    }
    // お only marks courtesy on this already proven time request.
    if time_request
        && source.get(nominal_start..lock.source_span.start_char) == Some(&['お'])
        && source.get(lock.source_span.start_char..noun.span.end_char) == Some(&['時', '間'])
    {
        return Some(lock.placeholder.clone());
    }
    if lock.source_span.start_char != noun.span.start_char {
        return None;
    }
    let mut object = phrase[..phrase.len() - 1].to_vec();
    let masked_noun = object.last_mut()?;
    masked_noun.surface.clone_from(&lock.placeholder);
    masked_noun.is_locked = true;
    nominal_possession::choice_phrase(&object, doc, sentence, TargetVariant::ZhHant, source)
}

#[cfg(test)]
mod tests {
    use crate::{
        GlossaryKind, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
        doc_options_for_translate, translate_document,
    };
    use faberline_nlp::doc::model::Backend;
    use faberline_nlp::doc::{Document, analyze_document};

    fn glossary(term: &str, label: &str, field: &str) -> TranslationGlossary {
        serde_json::from_value::<TranslationGlossary>(serde_json::json!({
            "schema": "nlp.translate.glossary.v1",
            (field): [{
                "canonical": term,
                "surfaces": [term],
                "targets": {"zh-Hant": label}
            }]
        }))
        .unwrap()
    }

    fn translate(source: &str, term: &str, label: &str, field: &str) -> String {
        let mut options = TranslateOptions::new(TargetVariant::ZhHant);
        options.source_hint = Some(TargetVariant::Ja);
        let glossary = glossary(term, label, field);
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
        let kind = if field == "objects" {
            GlossaryKind::Object
        } else {
            GlossaryKind::DomainTerm
        };
        assert_eq!(hit.kind, kind);
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
                translate("この資料をいただけませんか。", term, label, "objects"),
                "可以給我這份講義嗎？",
                "{term}",
            );
        }
        for term in ["時間", "お時間", "時間を", "お時間を"] {
            assert_eq!(
                translate(
                    "明日の午後、少しお時間をいただけませんか。",
                    term,
                    "談話時間",
                    "objects",
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
            "objects",
        );
        assert!(result.contains("參加者") && !result.contains("可以給我"));
    }

    #[test]
    fn explicit_time_request_survives_earlier_eligibility() {
        let context = "この資料は参加者だけに配ります。参加者以外の人は資料をいただけませんか。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let prior = analyze_and_translate(
            context,
            &options,
            &TranslationGlossary::new(),
            None,
            Backend::Onnx,
        )
        .unwrap()
        .target_text;
        let expected = format!("{prior}明天下午，可以占用您一點談話時間嗎？");
        for request in [
            "明日の午後、少しお時間をいただけませんか。",
            "明日の午後少しお時間をいただけませんか。",
        ] {
            let source = format!("{context}{request}");
            for term in ["時間", "お時間", "時間を", "お時間を"] {
                for field in ["objects", "domain_terms"] {
                    assert_eq!(translate(&source, term, "談話時間", field), expected);
                }
            }
        }
    }

    #[test]
    fn request_respects_grammar_supplied_with_a_document() {
        let source = "この資料をいただけませんか。";
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let glossary = glossary("資料を", "講義", "objects");
        let mut analysis = analyze_document(
            source,
            &doc_options_for_translate(&options, &TranslationGlossary::new()),
        );
        let translate = |input: &Document| {
            translate_document(
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
        assert_eq!(translate(&analysis).target_text, "可以給我這份講義嗎？");
        let object = analysis
            .syntax
            .iter_mut()
            .find(|arc| arc.deprel == "obj")
            .unwrap();
        object.deprel = "obl".to_owned();
        let output = translate(&analysis);
        assert_ne!(output.target_text, "可以給我這份講義嗎？");
        assert_eq!(output.glossary_hits.len(), 1);
    }
}
