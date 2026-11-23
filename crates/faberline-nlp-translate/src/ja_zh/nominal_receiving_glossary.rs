//! Recover a nominal request from the shared unlocked source analysis.

use super::naide_glossary::sole_object_lock;
use super::*;

pub(super) fn may_restore(text: &str, locks: &SpanLockTable) -> bool {
    text.trim_end_matches(|ch: char| ch.is_whitespace() || matches!(ch, '。' | '？' | '?'))
        .ends_with("いただけませんか")
        && sole_object_lock(locks).is_some()
}

pub(super) struct NominalRequest {
    clause: Span,
    object: Span,
    text: String,
}

fn clause_span(atoms: &[Atom]) -> Option<Span> {
    Some(Span {
        start_char: atoms.first()?.span.start_char,
        end_char: atoms.last()?.span.end_char,
    })
}

impl NominalRequest {
    pub(super) fn render(&self, atoms: &[Atom]) -> Option<&str> {
        let mut locked = atoms.iter().filter(|atom| atom.is_locked);
        let lock = locked.next()?;
        (locked.next().is_none()
            && lock.lemma.as_deref() == Some(LOCKED_OBJECT_ATOM)
            && lock.span == self.object
            && clause_span(atoms) == Some(self.clause))
        .then_some(self.text.as_str())
    }
}

pub(super) fn prove(
    atoms: &[Atom],
    chars: &[char],
    doc: &Document,
    sentence: usize,
    locks: &SpanLockTable,
) -> Option<NominalRequest> {
    let lock = sole_object_lock(locks)?;
    if !matches!(core_non_punct(atoms), [.., verb, masu, neg, question]
        if verb.surface == "いただけ" && masu.surface == "ませ"
            && neg.surface == "ん" && question.surface == "か")
    {
        return None;
    }
    let span = doc.segmentation.sentences[sentence].span;
    let segments = split_ja_atoms(atoms, doc, span, Some(chars));
    let [prefix @ .., Segment::Clause { atoms, .. }] = segments.as_slice() else {
        return None;
    };
    if !nominal_receiving_request_position(&segments, prefix.len(), doc, sentence) {
        return None;
    }
    let atoms = core_non_punct(atoms);
    let text = render_nominal_receiving_request(atoms, doc, sentence, Some(lock), chars)?;
    Some(NominalRequest {
        clause: clause_span(atoms)?,
        object: lock.source_span,
        text,
    })
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
        for source in [
            "明日の午後、少しお時間をいただけませんか。",
            "明日の午後少しお時間をいただけませんか。",
        ] {
            for term in ["時間", "お時間", "時間を", "お時間を"] {
                assert_eq!(
                    translate(source, term, "談話時間"),
                    "明天下午，可以占用您一點談話時間嗎？",
                    "{source}: {term}",
                );
            }
        }
    }

    #[test]
    fn object_glossary_keeps_the_demonstrative_and_explicit_count() {
        for term in ["紙", "紙を"] {
            assert_eq!(
                translate("この紙を一枚いただけませんか。", term, "便箋"),
                "可以給我這一張便箋嗎？",
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
