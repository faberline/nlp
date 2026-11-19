use faberline_nlp::Span;
use faberline_nlp::doc::Segmentation;

const FORM_NAMES: &[&str] = &[
    "文字数",
    "字数",
    "送り仮名",
    "表記",
    "字形",
    "字面",
    "綴り",
    "つづり",
    "文字",
    "字",
];
const COMPARED_FORM_NAMES: &[&str] = &["表記", "字形", "字面", "綴り", "つづり"];
const FORM_COUNTERS: &[&str] = &["つの", "種類の"];
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

/// Find explicitly written forms on the source paragraph, then give each
/// analysis sentence only its part of the original text to protect.
pub(crate) fn plan_written_form_quotes(
    chars: &[char],
    segmentation: &Segmentation,
) -> Vec<Vec<Span>> {
    let mut by_sentence = vec![Vec::new(); segmentation.sentences.len()];
    let sentences = &segmentation.sentences;
    for paragraph in &segmentation.paragraphs {
        let mut quotes = written_form_quotes(chars, paragraph.span);
        quotes.sort_unstable_by_key(|span| (span.start_char, std::cmp::Reverse(span.end_char)));
        let mut last_end = paragraph.span.start_char;
        for quote in quotes {
            // An explicitly literal outer quote already contains its inner quote.
            if quote.start_char < last_end {
                continue;
            }
            last_end = quote.end_char;
            let first = sentences.partition_point(|sent| sent.span.end_char <= quote.start_char);
            for (index, sent) in sentences.iter().enumerate().skip(first) {
                if quote.end_char <= sent.span.start_char {
                    break;
                }
                let start_char = quote.start_char.max(sent.span.start_char);
                let end_char = quote.end_char.min(sent.span.end_char);
                if start_char < end_char {
                    by_sentence[index].push(Span {
                        start_char,
                        end_char,
                    });
                }
            }
        }
    }
    by_sentence
}

struct OpenQuote {
    start: usize,
    expected_close: char,
    children: Vec<Span>,
    first_protection: usize,
}

fn written_form_quotes(chars: &[char], paragraph: Span) -> Vec<Span> {
    let mut openings = Vec::new();
    let mut protected = Vec::new();
    let mut top_level = Vec::new();
    let raw = &chars[paragraph.start_char..paragraph.end_char];
    for (offset, &ch) in raw.iter().enumerate() {
        let at = paragraph.start_char + offset;
        match ch {
            '「' | '『' => openings.push(OpenQuote {
                start: at,
                expected_close: if ch == '「' { '」' } else { '』' },
                children: Vec::new(),
                first_protection: protected.len(),
            }),
            close @ ('」' | '』') => {
                let Some(frame) = openings.pop() else {
                    continue;
                };
                if close != frame.expected_close {
                    let first = openings.first().unwrap_or(&frame);
                    protected.truncate(first.first_protection);
                    openings.clear();
                    continue;
                }
                if frame.start + 1 == at {
                    continue;
                }
                let inner = Span {
                    start_char: frame.start + 1,
                    end_char: at,
                };
                protect_sibling_forms(chars, inner, &frame.children, true, &mut protected);
                if let Some(parent) = openings.last_mut() {
                    parent.children.push(inner);
                } else {
                    top_level.push(inner);
                }
            }
            _ => {}
        }
    }
    if let Some(unclosed) = openings.first() {
        protected.truncate(unclosed.first_protection);
    }
    protect_sibling_forms(chars, paragraph, &top_level, false, &mut protected);
    protected
}

fn protection_span(inner: Span, nested: bool) -> Span {
    if nested {
        // Keep the inner quotation as one atom within the outer quotation.
        Span {
            start_char: inner.start_char - 1,
            end_char: inner.end_char + 1,
        }
    } else {
        inner
    }
}

fn protect_sibling_forms(
    chars: &[char],
    scope: Span,
    quotes: &[Span],
    nested: bool,
    protected: &mut Vec<Span>,
) {
    let mut first_for_count = 0;
    let mut first_for_comparison = 0;
    let mut labelled = false;
    for (index, quote) in quotes.iter().enumerate() {
        let opening = quote.start_char - 1;
        let after = skip_space(&chars[quote.end_char + 1..scope.end_char]);
        if describes_written_form(after) {
            protected.push(protection_span(*quote, nested));
        }
        let join = index.checked_sub(1).and_then(|prev| {
            let after_close = quotes[prev].end_char + 1;
            quote_join(&chars[after_close..opening])
        });
        if let Some(QuoteJoin::Conjunction(right_labelled)) = join {
            labelled &= right_labelled;
        } else {
            first_for_comparison = index;
            if join.is_none() {
                first_for_count = index;
            }
            let before = trim_space_end(&chars[scope.start_char..opening]);
            labelled = FORM_LABELS
                .iter()
                .any(|label| strip_suffix(before, label).is_some());
        }

        // Terminal marks inside the operands can split the analysis sentences,
        // but the operands still belong to this one balanced quotation scope.
        let first = if index > first_for_count && counts_each_form(after) {
            first_for_count
        } else if index > first_for_comparison {
            let before = &chars[scope.start_char..quotes[first_for_comparison].start_char - 1];
            let count = index - first_for_comparison + 1;
            if !compares_written_forms(after)
                && !compares_apposed_forms(before, after, count, labelled)
            {
                continue;
            }
            first_for_comparison
        } else {
            continue;
        };
        protected.extend(
            quotes[first..=index]
                .iter()
                .map(|&span| protection_span(span, nested)),
        );
    }
}

fn counts_each_form(after: &[char]) -> bool {
    let Some(rest) = after.strip_prefix(&['の']) else {
        return false;
    };
    ["文字数", "字数"].iter().any(|count| {
        strip_prefix(skip_space(rest), count)
            .and_then(|rest| skip_space(rest).strip_prefix(&['を']))
            .and_then(|rest| strip_prefix(skip_space(rest), "それぞれ"))
            .is_some_and(|rest| strip_prefix(skip_space(rest), "数え").is_some())
    })
}

#[derive(Debug, PartialEq, Eq)]
enum QuoteJoin {
    Conjunction(bool),
    Enumeration,
}

fn quote_join(between: &[char]) -> Option<QuoteJoin> {
    let between = skip_space(between);
    if trim_space_end(between) == ['、'] {
        return Some(QuoteJoin::Enumeration);
    }
    // A single Japanese comma can follow the conjunction と.
    let rest = skip_space(between.strip_prefix(&['と'])?);
    let rest = rest.strip_prefix(&['、']).map(skip_space).unwrap_or(rest);
    if rest.is_empty() {
        Some(QuoteJoin::Conjunction(false))
    } else {
        FORM_LABELS
            .iter()
            .any(|label| strip_prefix(rest, label).is_some_and(|tail| skip_space(tail).is_empty()))
            .then_some(QuoteJoin::Conjunction(true))
    }
}

fn compares_apposed_forms(before: &[char], after: &[char], count: usize, labelled: bool) -> bool {
    if compares_objects(after) && (labelled || has_form_name_before(before, count)) {
        return true;
    }
    let after = strip_prefix(after, "という")
        .or_else(|| after.strip_prefix(&['、']))
        .or_else(|| after.strip_prefix(&['の']))
        .unwrap_or(after);
    let Some((&number, rest)) = skip_space(after).split_first() else {
        return false;
    };
    matches_list_count(number, count)
        && FORM_COUNTERS.iter().any(|counter| {
            strip_prefix(skip_space(rest), counter).is_some_and(|rest| {
                COMPARED_FORM_NAMES.iter().any(|form| {
                    strip_prefix(skip_space(rest), form)
                        .is_some_and(|tail| compares_objects(skip_space(tail)))
                })
            })
        })
}

fn has_form_name_before(before: &[char], count: usize) -> bool {
    COMPARED_FORM_NAMES.iter().any(|form| {
        let Some(before) = strip_suffix(trim_space_end(before), form) else {
            return false;
        };
        FORM_COUNTERS.iter().any(|counter| {
            strip_suffix(trim_space_end(before), counter)
                .and_then(|before| trim_space_end(before).split_last())
                .is_some_and(|(&number, rest)| {
                    matches_list_count(number, count)
                        && !rest.last().is_some_and(|&ch| is_count_character(ch))
                })
        })
    })
}

fn matches_list_count(ch: char, count: usize) -> bool {
    count < 10
        && ("〇一二三四五六七八九".chars().nth(count) == Some(ch)
            || ch.to_digit(10) == Some(count as u32)
            || ch as u32 == '０' as u32 + count as u32)
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

fn strip_suffix<'a>(mut chars: &'a [char], suffix: &str) -> Option<&'a [char]> {
    for expected in suffix.chars().rev() {
        let (&actual, rest) = chars.split_last()?;
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

fn trim_space_end(chars: &[char]) -> &[char] {
    let end = chars
        .iter()
        .rposition(|&ch| !is_horizontal_space(ch))
        .map_or(0, |index| index + 1);
    &chars[..end]
}

fn is_horizontal_space(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '　')
}

fn without_count(chars: &[char]) -> &[char] {
    let count = chars
        .iter()
        .take_while(|&&ch| is_count_character(ch))
        .count();
    skip_space(&chars[count..])
}

fn is_count_character(ch: char) -> bool {
    ch.is_ascii_digit()
        || ('０'..='９').contains(&ch)
        || "〇零一二三四五六七八九十百千万".contains(ch)
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
        GlossaryEntry, TargetVariant, TranslateOptions, TranslateUnknownKind, TranslationGlossary,
        analyze_and_translate,
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
            ("日本語の「ドイツ」と、漢字の「独逸」を比べました。", "独逸"),
            ("「ドイツ」と『どいつ』は表記が違います。", "どいつ"),
            ("「ドイツ」と、『どいつ』は表記が違います。", "どいつ"),
            ("『ドイツ』と『フランス』の表記を見比べました。", "フランス"),
            ("二つの表記「ドイツ」と「独逸」を比べました。", "独逸"),
            ("「ドイツ」と「独逸」という二つの表記を比べました。", "独逸"),
            ("「ドイツ」と「独逸」、二つの表記を比べました。", "独逸"),
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
            "「ドイツ」と、「フランス」の人口を比べました。",
            "太郎は「ドイツ」と、花子は「フランス」と答えました。",
            "今日は「ドイツ」と、「フランス」について話しました。次に地図の表記を比べました。",
            "二つの国「ドイツ」と「フランス」を比べました。",
            "表記を確認したあと、「ドイツ」と「フランス」の人口を比べました。",
            "三つの表記「ドイツ」と「フランス」を比べました。",
            "十二の表記「ドイツ」と「フランス」を比べました。",
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

    #[test]
    fn nested_written_forms_stay_inside_the_outer_quote_and_alignment() {
        let glossary = glossary();
        for (source, written) in [
            (
                "先生は『「ドイツ」という三文字を書きなさい』と言いました。",
                "「ドイツ」",
            ),
            ("父は『「駅」の文字を消しなさい』と言いました。", "「駅」"),
        ] {
            let translated = analyze_and_translate(
                source,
                &TranslateOptions::new(TargetVariant::ZhHant),
                &glossary,
                None,
                Backend::Onnx,
            )
            .unwrap();
            let text = &translated.target_text;
            let quote = text.find(written).unwrap_or_else(|| panic!("{text}"));
            assert_eq!(text.matches(written).count(), 1, "{text}");
            assert!(
                text.find('『').unwrap() < quote && quote < text.find('』').unwrap(),
                "{text}"
            );
            let start = text[..quote].chars().count();
            let end = start + written.chars().count();
            assert!(
                translated.clause_alignments.iter().any(|clause| {
                    clause.target_span.start_char <= start && end <= clause.target_span.end_char
                }),
                "{text}"
            );
            assert!(
                !translated
                    .unknown
                    .iter()
                    .any(|item| { item.kind == TranslateUnknownKind::PlaceholderRepaired }),
                "{text}"
            );
        }
    }

    #[test]
    fn counting_retains_complete_written_forms_across_analysis_sentences() {
        let glossary = glossary();
        for (source, written) in [
            (
                "「来て！すぐ帰る」の文字数を数えます。",
                &["「来て！すぐ帰る」"][..],
            ),
            ("「A!ドイツ？B」の文字数を数えます。", &["「A!ドイツ？B」"]),
            ("「来て！帰る」の字数を数えました。", &["「来て！帰る」"]),
            ("「来て！帰る」の字数を調べました。", &["「来て！帰る」"]),
            (
                "「ドイツ！」と「行く？」の文字数をそれぞれ数えました。",
                &["「ドイツ！」", "「行く？」"],
            ),
            (
                "「ドイツ！」と、「行く？」の文字数をそれぞれ数えました。",
                &["「ドイツ！」", "「行く？」"],
            ),
            (
                "「行く？」と、「ドイツ！」の文字数をそれぞれ数えました。",
                &["「行く？」", "「ドイツ！」"],
            ),
            (
                "「来て！」と、「すぐ帰る」の文字数をそれぞれ数えました。",
                &["「来て！」", "「すぐ帰る」"],
            ),
            (
                "「ドイツ！」、「行く？」、「払える！」の文字数をそれぞれ数えました。",
                &["「ドイツ！」", "「行く？」", "「払える！」"],
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
            for original in written {
                assert!(
                    translated.target_text.contains(original),
                    "{}",
                    translated.target_text
                );
            }
            assert!(translated.glossary_hits.is_empty(), "{source}");
            assert!(
                !translated
                    .unknown
                    .iter()
                    .any(|item| { item.kind == TranslateUnknownKind::PlaceholderRepaired }),
                "{source}"
            );
        }
    }

    #[test]
    fn coordinated_children_keep_their_complete_text_inside_the_parent_quote() {
        let glossary = glossary();
        for (source, expected) in [
            (
                "先生は『「ドイツ！」と、「行く？」の文字数をそれぞれ数えなさい』と言いました。",
                ["「ドイツ！」", "「行く？」"],
            ),
            (
                "父は『「来て！」と、「すぐ帰る」の文字数をそれぞれ数えなさい』と言いました。",
                ["「来て！」", "「すぐ帰る」"],
            ),
            (
                "母は『日本語の「ドイツ」と、漢字の「独逸」を比べなさい』と言いました。",
                ["「ドイツ」", "「独逸」"],
            ),
            (
                "『二つの表記「ドイツ」と「独逸」を比べました』",
                ["「ドイツ」", "「独逸」"],
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
            let text = &translated.target_text;
            let start = text.find('『').unwrap_or_else(|| panic!("{text}"));
            let end = text.rfind('』').unwrap_or_else(|| panic!("{text}"));
            let mut previous = start;
            for original in expected {
                let at = text.find(original).unwrap_or_else(|| panic!("{text}"));
                assert!(previous < at && at + original.len() <= end, "{text}");
                assert_eq!(text.matches(original).count(), 1, "{text}");
                previous = at;
            }
            assert!(
                !translated
                    .unknown
                    .iter()
                    .any(|item| item.kind == TranslateUnknownKind::PlaceholderRepaired),
                "{text}"
            );
        }
    }

    #[test]
    fn a_count_inside_one_quote_does_not_protect_other_utterances() {
        use super::{Span, written_form_quotes};

        for source in [
            "先生は『「行く？」の文字数を数えてから、「ドイツ！」と答えなさい』と言いました。",
            "『「ドイツ！」と叫びました』と『「行く？」の文字数を数えました』を読みました。",
        ] {
            let chars: Vec<_> = source.chars().collect();
            let spans = written_form_quotes(
                &chars,
                Span {
                    start_char: 0,
                    end_char: chars.len(),
                },
            );
            let protected: Vec<String> = spans
                .into_iter()
                .map(|span| chars[span.start_char..span.end_char].iter().collect())
                .collect();
            assert_eq!(protected, ["「行く？」"], "{source}");
        }
    }

    #[test]
    fn quote_join_accepts_only_one_comma_after_the_conjunction() {
        use super::{QuoteJoin, quote_join};

        for (between, expected) in [
            ("と、", Some(QuoteJoin::Conjunction(false))),
            (" と　、 漢字の　", Some(QuoteJoin::Conjunction(true))),
            ("、", Some(QuoteJoin::Enumeration)),
            ("、と", None),
            ("と、、", None),
            ("と、。", None),
            ("と、\n", None),
            ("と、花子は", None),
        ] {
            assert_eq!(
                quote_join(&between.chars().collect::<Vec<_>>()),
                expected,
                "{between:?}"
            );
        }
    }
}
