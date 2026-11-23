//! Restore a glossary object whose original case belongs to a linked request action.

use super::naide_glossary::sole_object_lock;
use super::*;

pub(super) fn may_restore(text: &str, locks: &SpanLockTable) -> bool {
    let text =
        text.trim_end_matches(|ch: char| ch.is_whitespace() || matches!(ch, '。' | '？' | '?'));
    let Some(prefix) = text
        .strip_suffix("ますか")
        .or_else(|| text.strip_suffix("ませんか"))
    else {
        return false;
    };
    ["ていただけ", "でいただけ", "てもらえ", "でもらえ"]
        .iter()
        .any(|link| prefix.ends_with(link))
        && sole_object_lock(locks).is_some()
}

pub(super) struct AssistanceObject {
    source_span: Span,
    action_span: Span,
    action_token: usize,
    demonstrative: Option<(Span, String)>,
}

pub(super) fn prove(
    atoms: &[Atom],
    doc: &Document,
    locks: &SpanLockTable,
) -> Option<AssistanceObject> {
    let lock = sole_object_lock(locks)?;
    let parsed = te_help_request(core_non_punct(atoms), doc, None)?;
    let action = parsed.action.last()?;
    let action_token = action.doc_tok_idx?;
    let (index, noun, with_case) = parsed.action.windows(2).enumerate().find_map(|(i, pair)| {
        let [noun, case] = pair else { return None };
        let noun_token = noun.doc_tok_idx?;
        let with_case = lock.source_span.end_char == case.span.end_char;
        (matches!(noun.upos, Some(Upos::Noun | Upos::Propn))
            && noun.head == Some(action_token)
            && noun.deprel == "obj"
            && noun.span.end_char == case.span.start_char
            && case.surface == "を"
            && case.upos == Some(Upos::Adp)
            && case.head == Some(noun_token)
            && case.deprel == "case"
            && (with_case || lock.source_span.end_char == noun.span.end_char))
            .then_some((i, noun, with_case))
    })?;
    let demonstrative = index
        .checked_sub(1)
        .map(|i| &parsed.action[i])
        .and_then(|det| attached_demonstrative(det, noun).map(|word| (det, word)));
    let demonstrative = match demonstrative {
        Some((det, word)) if lock.source_span.start_char == noun.span.start_char => {
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
            Some((det.span, format!("{word}{classifier}")))
        }
        Some((det, _)) if with_case && lock.source_span.start_char == det.span.start_char => None,
        _ if with_case && lock.source_span.start_char == noun.span.start_char => None,
        _ => return None,
    };
    Some(AssistanceObject {
        source_span: lock.source_span,
        action_span: action.span,
        action_token,
        demonstrative,
    })
}

impl AssistanceObject {
    pub(super) fn restore(&self, atoms: &[Atom], doc: &Document) -> Option<Vec<Atom>> {
        let body = core_non_punct(atoms);
        let mut locked = body.iter().enumerate().filter(|(_, atom)| atom.is_locked);
        let (index, lock) = locked.next()?;
        if lock.lemma.as_deref() != Some(LOCKED_OBJECT_ATOM)
            || lock.span != self.source_span
            || locked.next().is_some()
            || !body.iter().any(|atom| {
                !atom.is_locked
                    && atom.span == self.action_span
                    && atom.doc_tok_idx == Some(self.action_token)
            })
        {
            return None;
        }

        let mut restored = lock.clone();
        restored.deprel = "obj".to_owned();
        restored.head = Some(self.action_token);
        restored.doc_tok_idx = None;
        restored.upos = Some(Upos::Noun);
        let start = if let Some((span, prefix)) = &self.demonstrative {
            let index = index.checked_sub(1)?;
            let det = &body[index];
            if det.is_locked || det.span != *span {
                return None;
            }
            restored.span.start_char = span.start_char;
            restored.surface.insert_str(0, prefix);
            index
        } else {
            index
        };
        let mut result = atoms.to_vec();
        result.splice(start..=index, [restored]);
        te_help_request(core_non_punct(&result), doc, None)?;
        Some(result)
    }
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
        assert_eq!(result.target_text, "可以給我這份講義嗎？");
    }
}
