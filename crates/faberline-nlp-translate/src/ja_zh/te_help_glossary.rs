use super::*;

// The caller lexicon can combine a direct object and を into one token
// tagged as a verb. Recover the argument from the ordinary clause analysis,
// then run the same request matcher on the original atoms and placeholder.
pub(super) fn restore_object(atoms: &[Atom], doc: &Document, chars: &[char]) -> Option<Vec<Atom>> {
    let body = core_non_punct(atoms);
    if body.last()?.surface != "か"
        || !body.windows(3).any(|group| {
            is_te_form_link(&group[0], &group[1], &group[2])
                && matches!(group[2].lemma.as_deref(), Some("いただける" | "もらえる"))
        })
    {
        return None;
    }
    let mut locks = body.iter().enumerate().filter(|(_, atom)| atom.is_locked);
    let (index, lock) = locks.next()?;
    if lock.lemma.as_deref() != Some(LOCKED_OBJECT_ATOM) || locks.next().is_some() {
        return None;
    }
    let offset = body.first()?.span.start_char;
    let source = chars.get(offset..atoms.last()?.span.end_char)?;
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
    let raw = build_sentence_atoms(0, source, &original, &tokens, &SpanLockTable::default());
    let parsed = te_help_request(core_non_punct(&raw), &original, None)?;
    let action = parsed.action.last()?;
    let original_action = action.doc_tok_idx?;
    let at_source = |atom: &Atom| Span {
        start_char: atom.span.start_char + offset,
        end_char: atom.span.end_char + offset,
    };
    let action_idx = body
        .iter()
        .find(|atom| !atom.is_locked && atom.span == at_source(action))?
        .doc_tok_idx?;
    let (object_index, noun, with_case) =
        parsed
            .action
            .windows(2)
            .enumerate()
            .find_map(|(i, group)| {
                let noun = &group[0];
                let case = &group[1];
                let with_case = at_source(case).end_char == lock.span.end_char;
                (!noun.is_locked
                    && matches!(noun.upos, Some(Upos::Noun | Upos::Propn))
                    && noun.head == Some(original_action)
                    && noun.deprel == "obj"
                    && case.surface == "を"
                    && case.upos == Some(Upos::Adp)
                    && case.head == noun.doc_tok_idx
                    && case.deprel == "case"
                    && noun.span.end_char == case.span.start_char
                    && (with_case || at_source(noun).end_char == lock.span.end_char))
                    .then_some((i, noun, with_case))
            })?;
    let demonstrative = object_index
        .checked_sub(1)
        .map(|i| &parsed.action[i])
        .and_then(|det| attached_demonstrative(det, noun).map(|word| (det, word)));
    let mut restored = lock.clone();
    restored.deprel = "obj".to_owned();
    restored.head = Some(action_idx);
    // The masked token's parser frame still describes a verb. Its only role
    // here is the object whose relation was verified in the ordinary analysis.
    restored.doc_tok_idx = None;
    restored.upos = Some(Upos::Noun);
    let mut start = index;
    match demonstrative {
        Some((det, word)) if lock.span.start_char == at_source(noun).start_char => {
            start = index.checked_sub(1)?;
            if body[start].span != at_source(det) || body[start].is_locked {
                return None;
            }
            let classifier = match noun.surface.as_str() {
                "荷物" => "件",
                "書類" => "份",
                _ => {
                    let translated =
                        translate_single_ja_term(&noun.surface, noun.lemma.as_deref(), noun.upos);
                    demonstrative_classifier(&translated, noun)
                }
            };
            if classifier.is_empty() && !with_case {
                return None;
            }
            restored.span.start_char = body[start].span.start_char;
            restored.surface = format!("{word}{classifier}{}", lock.surface);
        }
        Some((det, _)) if with_case && lock.span.start_char == at_source(det).start_char => {}
        _ if with_case && lock.span.start_char == at_source(noun).start_char => {}
        _ => return None,
    }
    let mut result = atoms.to_vec();
    result.splice(start..=index, [restored]);
    te_help_request(core_non_punct(&result), doc, None)?;
    Some(result)
}

#[cfg(test)]
mod tests {
    use crate::{TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate};
    use faberline_nlp::doc::model::Backend;

    fn translate(source: &str, term: &str, label: &str) -> crate::TranslatedDocument {
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
        let translated =
            analyze_and_translate(source, &options, &glossary, None, Backend::Onnx).unwrap();
        let [hit] = translated.glossary_hits.as_slice() else {
            panic!("expected one glossary hit for {term}");
        };
        let from_span = |text: &str, start, end| -> String {
            text.chars().skip(start).take(end - start).collect()
        };
        assert_eq!(
            from_span(source, hit.source_span.start_char, hit.source_span.end_char),
            term
        );
        assert_eq!(
            from_span(
                &translated.target_text,
                hit.target_span.start_char,
                hit.target_span.end_char
            ),
            label
        );
        translated
    }

    #[test]
    fn object_case_locks_keep_the_assistance_request() {
        for (source, opening) in [
            ("この荷物を持っていただけますか。", "可以請您幫我"),
            ("この荷物を持っていただけませんか。", "可以麻煩您幫我"),
        ] {
            for (term, label) in [
                ("荷物", "器材"),
                ("荷物を", "器材"),
                ("この荷物を", "這件器材"),
            ] {
                let result = translate(source, term, label);
                assert_eq!(
                    result.target_text,
                    format!("{opening}拿著這件器材嗎？"),
                    "{term}"
                );
            }
        }
        assert_eq!(
            translate("写真を撮ってもらえますか。", "写真を", "紀錄照").target_text,
            "可以幫我拍攝紀錄照嗎？"
        );
    }

    #[test]
    fn case_lock_does_not_make_a_nominal_request_an_assistance_request() {
        let result = translate("この資料をいただけませんか。", "この資料を", "這份講義");
        assert!(!result.target_text.contains("幫我"));
    }
}
