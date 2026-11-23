//! Keep the written-text sense and the caller's genitive wording.

use crate::lock::{LockOrigin, LockedSpan, SpanLockTable};
use crate::schema::GlossaryKind;

use super::{Atom, ClauseDiscourseCtx, Span, Upos, dasu_case_link};

fn is_text(surface: &str) -> bool {
    matches!(
        surface,
        "本" | "書籍" | "小説" | "物語" | "文章" | "段落" | "章" | "手紙"
    )
}

fn is_opening(surface: &str) -> bool {
    matches!(surface, "始め" | "初め")
}

fn caller_term(lock: &LockedSpan) -> bool {
    lock.mandatory
        && matches!(
            lock.origin,
            LockOrigin::Glossary {
                kind: GlossaryKind::Object | GlossaryKind::DomainTerm,
                ..
            }
        )
}

fn source_word(chars: &[char], span: Span) -> Option<String> {
    Some(chars.get(span.start_char..span.end_char)?.iter().collect())
}

fn term_at<'a>(atom: &Atom, locks: &'a [LockedSpan]) -> Option<&'a LockedSpan> {
    locks.iter().find(|lock| {
        caller_term(lock) && lock.source_span == atom.span && lock.placeholder == atom.surface
    })
}

fn genitive_link(text: &Atom, genitive: &Atom, opening: &Atom) -> bool {
    opening.doc_tok_idx.is_some()
        && text.deprel == "nmod"
        && text.head == opening.doc_tok_idx
        && genitive.surface == "の"
        && dasu_case_link(text, genitive)
        && genitive.span.end_char == opening.span.start_char
}

/// Apply the source reading and report whether the caller already supplied 的.
pub(super) fn render(
    opening: &mut Atom,
    modifier: &[Atom],
    ctx: Option<&ClauseDiscourseCtx<'_>>,
) -> bool {
    let [.., text, genitive] = modifier else {
        return false;
    };
    if !genitive_link(text, genitive, opening) {
        return false;
    }
    let opening_is_nominal = if opening.is_locked {
        ctx.is_some_and(|ctx| {
            term_at(opening, ctx.locks).is_some()
                && source_word(ctx.source_chars, opening.span).is_some_and(|word| is_opening(&word))
        })
    } else {
        opening.upos == Some(Upos::Noun) && is_opening(&opening.surface)
    };
    if !opening_is_nominal {
        return false;
    }
    let text_lock = if text.is_locked {
        let Some(lock) = ctx.and_then(|ctx| {
            let word = source_word(ctx.source_chars, text.span)?;
            is_text(&word).then_some(())?;
            term_at(text, ctx.locks)
        }) else {
            return false;
        };
        Some(lock)
    } else {
        if text.upos != Some(Upos::Noun) || !is_text(&text.surface) {
            return false;
        }
        None
    };
    if !opening.is_locked {
        opening.surface = "開頭".to_owned();
        opening.lemma = None;
    }
    text_lock.is_some_and(|lock| lock.replacement.trim_end().ends_with('的'))
}

pub(super) fn may_restore(chars: &[char], locks: &SpanLockTable) -> bool {
    locks.locks.iter().any(|lock| {
        caller_term(lock)
            && source_word(chars, lock.source_span)
                .is_some_and(|word| is_text(word.strip_suffix('の').unwrap_or(&word)))
    })
}

pub(super) fn restore_case(
    atoms: &[Atom],
    chars: &[char],
    locks: &SpanLockTable,
) -> Option<SpanLockTable> {
    if !may_restore(chars, locks) {
        return None;
    }
    let mut render = locks.clone();
    let mut restored = false;
    for window in atoms.windows(3) {
        let [text, genitive, opening] = window else {
            continue;
        };
        if text.upos != Some(Upos::Noun)
            || !is_text(&text.surface)
            || opening.upos != Some(Upos::Noun)
            || !is_opening(&opening.surface)
            || !genitive_link(text, genitive, opening)
        {
            continue;
        }
        let phrase = Span {
            start_char: text.span.start_char,
            end_char: genitive.span.end_char,
        };
        if let Some(lock) = render.locks.iter_mut().find(|lock| {
            caller_term(lock) && (lock.source_span == text.span || lock.source_span == phrase)
        }) {
            // Only the render table exposes の. Keep the original hit extent.
            lock.source_span = text.span;
            restored = true;
        }
    }
    restored.then_some(render)
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::{analyze_document, model::Backend};

    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
        doc_options_for_translate, translate_document,
    };

    fn entry(canonical: &str, source: &str, target: &str) -> GlossaryEntry {
        GlossaryEntry {
            canonical: canonical.into(),
            surfaces: vec![source.into()],
            targets: [(TargetVariant::ZhHant, target.into())].into(),
            ..GlossaryEntry::default()
        }
    }

    fn check(source: &str, glossary: &TranslationGlossary, expected: &str) {
        let options = TranslateOptions::new(TargetVariant::ZhHant);
        let result =
            analyze_and_translate(source, &options, glossary, None, Backend::Onnx).unwrap();
        let doc = analyze_document(source, &doc_options_for_translate(&options, glossary));
        let direct = translate_document(
            source,
            &doc,
            #[cfg(feature = "narrative")]
            None,
            &options,
            glossary,
            None,
        )
        .unwrap();
        assert_eq!(result.target_text, expected, "{source}");
        assert_eq!(direct.target_text, expected);
        assert_eq!(direct.glossary_hits, result.glossary_hits);
        assert_eq!(direct.clause_alignments, result.clause_alignments);
        let entries = glossary.objects.iter().chain(&glossary.domain_terms);
        assert_eq!(result.glossary_hits.len(), entries.clone().count());
        let slice = |text: &str, span: super::Span| -> String {
            text.chars()
                .skip(span.start_char)
                .take(span.end_char - span.start_char)
                .collect()
        };
        for term in entries {
            let hit = result
                .glossary_hits
                .iter()
                .find(|hit| hit.canonical == term.canonical)
                .unwrap();
            assert_eq!(slice(source, hit.source_span), term.surfaces[0]);
            assert_eq!(
                slice(&result.target_text, hit.target_span),
                term.targets[&TargetVariant::ZhHant]
            );
        }
    }

    #[test]
    fn written_opening_uses_the_source_word_under_a_caller_alias() {
        for (lock, target) in [("本", "手冊"), ("本の", "手冊"), ("本の", "手冊的")] {
            let mut glossary = TranslationGlossary::new();
            glossary.objects.push(entry("指定讀物", lock, target));
            check("本の始めは大切です。", &glossary, "手冊的開頭重要。");
        }
    }

    #[test]
    fn written_opening_keeps_the_callers_opening_and_single_genitive() {
        let mut glossary = TranslationGlossary::new();
        glossary.domain_terms.push(entry("本の", "本の", "手冊的"));
        glossary.objects.push(entry("始め", "始め", "起首"));
        check("本の始めを読みます。", &glossary, "讀手冊的起首。");

        let mut combined = TranslationGlossary::new();
        combined.objects.push(entry("本の", "本の", "手冊的"));
        combined
            .domain_terms
            .push(entry("初めに", "初めに", "首先"));
        check(
            "私は初めに本の始めを読みます。",
            &combined,
            "我首先讀手冊的開頭。",
        );
    }

    #[test]
    fn written_opening_leaves_event_nouns_and_actions_on_their_own_paths() {
        let mut glossary = TranslationGlossary::new();
        glossary.objects.push(entry("会議", "会議", "例會"));
        check("会議の初めは大切です。", &glossary, "例會的起初重要。");
        let empty = TranslationGlossary::new();
        check("先生は授業を始めます。", &empty, "老師開始授業。");
        check("私は初めに名前を書きます。", &empty, "我先寫名字。");
    }
}
