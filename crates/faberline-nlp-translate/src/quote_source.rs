use faberline_nlp::Span;

const FORM_NAMES: &[&str] = &[
    "文字数",
    "送り仮名",
    "表記",
    "字形",
    "字面",
    "綴り",
    "つづり",
    "文字",
    "字",
];
const FORM_LABELS: &[&str] = &[
    "日本語の",
    "英語の",
    "中国語の",
    "カタカナの",
    "片仮名の",
    "ひらがなの",
    "平仮名の",
    "漢字の",
    "ローマ字の",
];

/// Find quoted source text followed by an explicit description of its written form.
/// The marks stay outside the lock so the normal quote renderer can handle them.
pub(crate) fn written_form_quotes(chars: &[char], sentence: Span, context_end: usize) -> Vec<Span> {
    let mut openings = Vec::new();
    let mut protected = Vec::new();
    let mut top_level = Vec::new();
    let raw = &chars[sentence.start_char..sentence.end_char];
    for (at, &ch) in raw.iter().enumerate() {
        match ch {
            '「' => openings.push((at, '」')),
            '『' => openings.push((at, '』')),
            close @ ('」' | '』') => {
                let Some((start, expected)) = openings.pop() else {
                    continue;
                };
                if close != expected {
                    openings.clear();
                    continue;
                }
                if start + 1 == at {
                    continue;
                }
                let inner = Span {
                    start_char: sentence.start_char + start + 1,
                    end_char: sentence.start_char + at,
                };
                if describes_written_form(&chars[inner.end_char + 1..context_end]) {
                    protected.push(inner);
                }
                if openings.is_empty()
                    && !chars[inner.start_char..inner.end_char]
                        .iter()
                        .any(|&ch| "「」『』".contains(ch))
                {
                    top_level.push(inner);
                }
            }
            _ => {}
        }
    }
    protect_compared_forms(chars, sentence, &top_level, &mut protected);
    protected
}

fn protect_compared_forms(
    chars: &[char],
    sentence: Span,
    quotes: &[Span],
    protected: &mut Vec<Span>,
) {
    if quotes.len() < 2 {
        return;
    }
    let mut first = 0;
    let mut labelled = false;
    for (index, quote) in quotes.iter().enumerate() {
        let opening = quote.start_char - 1;
        let join = index.checked_sub(1).and_then(|prev| {
            let after_close = quotes[prev].end_char + 1;
            quote_join(&chars[after_close..opening])
        });
        if let Some(right_labelled) = join {
            labelled &= right_labelled;
        } else {
            first = index;
            let before = &chars[sentence.start_char..opening];
            labelled = FORM_LABELS.iter().any(|label| {
                before
                    .iter()
                    .rev()
                    .copied()
                    .skip_while(|&ch| is_horizontal_space(ch))
                    .take(label.chars().count())
                    .eq(label.chars().rev())
            });
        }

        let after = skip_space(&chars[quote.end_char + 1..sentence.end_char]);
        if index > first && (compares_written_forms(after) || (labelled && compares_objects(after)))
        {
            for span in &quotes[first..=index] {
                if !protected.contains(span) {
                    protected.push(*span);
                }
            }
        }
    }
}

// Only the conjunction and an optional label may occur between the two operands.
fn quote_join(between: &[char]) -> Option<bool> {
    let rest = skip_space(skip_space(between).strip_prefix(&['と'])?);
    if rest.is_empty() {
        Some(false)
    } else {
        FORM_LABELS
            .iter()
            .any(|label| strip_prefix(rest, label).is_some_and(|tail| skip_space(tail).is_empty()))
            .then_some(true)
    }
}

fn compares_objects(after: &[char]) -> bool {
    after.strip_prefix(&['を']).is_some_and(|rest| {
        [
            "比べる",
            "比べま",
            "比べた",
            "比べて",
            "比べな",
            "見比べる",
            "見比べま",
            "見比べた",
            "見比べて",
            "見比べな",
            "比較し",
            "比較する",
        ]
        .iter()
        .any(|verb| strip_prefix(skip_space(rest), verb).is_some())
    })
}

fn compares_written_forms(after: &[char]) -> bool {
    let Some((&particle, rest)) = after.split_first() else {
        return false;
    };
    FORM_NAMES.iter().any(|form| {
        let Some(tail) = strip_prefix(skip_space(rest), form).map(skip_space) else {
            return false;
        };
        match particle {
            'の' => compares_objects(tail),
            'は' => tail.strip_prefix(&['が']).is_some_and(|tail| {
                ["違う", "違い", "違っ", "違わ", "異な", "同じ"]
                    .iter()
                    .any(|comparison| strip_prefix(skip_space(tail), comparison).is_some())
            }),
            _ => false,
        }
    })
}

fn describes_written_form(after: &[char]) -> bool {
    let after = skip_space(after);
    if let Some(topic) = after.strip_prefix(&['は']) {
        // A count asserted about the quoted text is evidence of a literal use.
        // A location followed by は alone is also a normal topic.
        let rest = skip_space(topic);
        let rest = ["カタカナ", "片仮名", "ひらがな", "平仮名", "漢字"]
            .iter()
            .find_map(|script| strip_prefix(rest, script))
            .unwrap_or(rest);
        let number = skip_space(rest);
        let rest = without_count(number);
        return rest.len() < number.len()
            && strip_prefix(rest, "文字").is_some_and(|tail| {
                let tail = skip_space(tail);
                ["です", "でした", "である", "だった", "だ"]
                    .iter()
                    .any(|copula| {
                        strip_prefix(tail, copula).is_some_and(|end| {
                            matches!(
                                skip_space(end).first(),
                                None | Some(
                                    '。' | '！' | '？' | 'か' | 'ね' | 'よ' | 'と' | '」' | '』'
                                )
                            )
                        })
                    })
            });
    }

    if let Some(note) = strip_prefix(after, "と書いてある") {
        let note = skip_space(note);
        return ["メモの", "紙の", "札の", "カードの"]
            .iter()
            .find_map(|medium| strip_prefix(note, medium))
            .is_some_and(|rest| {
                ["文字数", "字数"]
                    .iter()
                    .any(|head| strip_prefix(skip_space(rest), head).is_some_and(noun_boundary))
            });
    }

    let nominal = if let Some(nominal) = strip_prefix(after, "という") {
        nominal
    } else if let Some(nominal) = after.strip_prefix(&['の']) {
        let nominal = skip_space(nominal);
        ["最初の", "最後の", "先頭の", "末尾の"]
            .iter()
            .find_map(|modifier| strip_prefix(nominal, modifier))
            .unwrap_or(nominal)
    } else {
        return false;
    };
    let nominal = without_count(skip_space(nominal));
    FORM_NAMES
        .iter()
        .any(|form| strip_prefix(nominal, form).is_some_and(noun_boundary))
}

fn strip_prefix<'a>(mut chars: &'a [char], prefix: &str) -> Option<&'a [char]> {
    for expected in prefix.chars() {
        let (&actual, rest) = chars.split_first()?;
        if actual != expected {
            return None;
        }
        chars = rest;
    }
    Some(chars)
}

fn skip_space(chars: &[char]) -> &[char] {
    let count = chars
        .iter()
        .take_while(|&&ch| is_horizontal_space(ch))
        .count();
    &chars[count..]
}

fn is_horizontal_space(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '　')
}

fn without_count(chars: &[char]) -> &[char] {
    let count = chars
        .iter()
        .take_while(|&&ch| {
            ch.is_ascii_digit()
                || ('０'..='９').contains(&ch)
                || "〇零一二三四五六七八九十百千万".contains(ch)
        })
        .count();
    skip_space(&chars[count..])
}

fn noun_boundary(chars: &[char]) -> bool {
    skip_space(chars)
        .first()
        .is_none_or(|&ch| "はがをにとでのもへだ、。！？」「『』".contains(ch))
}

#[cfg(test)]
mod tests {
    use faberline_nlp::doc::model::Backend;

    use crate::{
        GlossaryEntry, TargetVariant, TranslateOptions, TranslationGlossary, analyze_and_translate,
    };

    fn glossary() -> TranslationGlossary {
        TranslationGlossary {
            places: vec![GlossaryEntry {
                canonical: "ドイツ".to_owned(),
                surfaces: vec!["ドイツ".to_owned()],
                targets: [(TargetVariant::ZhHant, "德國".to_owned())].into(),
                pronoun_class: Default::default(),
                speech_register: Default::default(),
            }],
            ..TranslationGlossary::default()
        }
    }

    #[test]
    fn preserves_the_characters_that_the_sentence_describes() {
        let glossary = glossary();
        for source in [
            "私は「ドイツ」という三文字を丸で囲みました。",
            "私は「ドイツ」の最後の文字を消しました。",
            "『ドイツ』はカタカナ三文字です。",
            "「ドイツ」の表記をノートに書きました。",
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert!(translated.target_text.contains("ドイツ"), "{source}");
            assert!(translated.glossary_hits.is_empty(), "{source}");
        }
    }

    #[test]
    fn written_form_does_not_lock_a_separate_location_or_reply() {
        let glossary = glossary();
        for (source, preserved) in [
            ("出身国を聞かれて、「ドイツです」と答えました。", false),
            ("私は「ドイツ」という国を地図で見ました。", false),
            (
                "私は「ドイツ」という三文字を丸で囲み、「ドイツへ行きます」と言いました。",
                true,
            ),
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert!(translated.target_text.contains("德國"), "{source}");
            assert_eq!(
                translated.target_text.contains("ドイツ"),
                preserved,
                "{source}"
            );
            assert_eq!(translated.glossary_hits.len(), 1, "{source}");
        }
    }

    #[test]
    fn preserves_counted_source_with_terminal_punctuation_inside_the_quote() {
        for source in [
            "「昼までに！」は五文字です。",
            "「昼までに！」と書いてあるメモの文字数を数えました。",
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &TranslationGlossary::default(),
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert!(translated.target_text.contains("昼までに！"), "{source}");
        }
    }

    #[test]
    fn preserves_both_source_forms_in_an_explicit_comparison() {
        let mut glossary = glossary();
        for (source, target) in [("独逸", "獨逸"), ("どいつ", "哪個人"), ("フランス", "法國")]
        {
            glossary.places.push(GlossaryEntry {
                canonical: source.to_owned(),
                surfaces: vec![source.to_owned()],
                targets: [(TargetVariant::ZhHant, target.to_owned())].into(),
                pronoun_class: Default::default(),
                speech_register: Default::default(),
            });
        }
        for (source, other) in [
            ("日本語の「ドイツ」と漢字の「独逸」を比べました。", "独逸"),
            ("「ドイツ」と『どいつ』は表記が違います。", "どいつ"),
            ("『ドイツ』と『フランス』の表記を見比べました。", "フランス"),
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert!(translated.target_text.contains("ドイツ"), "{source}");
            assert!(translated.target_text.contains(other), "{source}");
            assert!(translated.glossary_hits.is_empty(), "{source}");
        }

        for source in [
            "「ドイツ」と「フランス」を比べました。",
            "「ドイツ」と「フランス」の人口を比べました。",
            "「ドイツ」と「フランス」を比べるために旅行しました。",
            "太郎は「ドイツ」と答え、花子は「フランス」と答えました。",
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            assert!(translated.target_text.contains("德國"), "{source}");
            assert!(translated.target_text.contains("法國"), "{source}");
            assert_eq!(translated.glossary_hits.len(), 2, "{source}");
        }
    }
}
