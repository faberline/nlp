use faberline_nlp::text::{Analyzer, Locale, Sentence, TokenKind};

fn verify_spans(source: &str, sentences: &[Sentence<'_>]) {
    let mut end = 0;
    for sentence in sentences {
        assert!(sentence.bytes.start >= end);
        assert_eq!(&source[sentence.bytes.clone()], sentence.text);
        let mut cursor = sentence.bytes.start;
        for token in &sentence.tokens {
            assert!(token.bytes.start >= cursor && token.bytes.end <= sentence.bytes.end);
            assert!(
                source[cursor..token.bytes.start]
                    .chars()
                    .all(char::is_whitespace)
            );
            assert_eq!(&source[token.bytes.clone()], token.text);
            cursor = token.bytes.end;
        }
        assert!(
            source[cursor..sentence.bytes.end]
                .chars()
                .all(char::is_whitespace)
        );
        end = sentence.bytes.end;
    }
}

#[test]
fn english_preserves_contractions_decimal_numbers_and_source_ranges() -> anyhow::Result<()> {
    let analyzer = Analyzer::new(Locale::English)?;
    let source = " \r\nDr. Lin can’t use 3.14 TB. He said “Done!”  ";
    let sentences = analyzer.analyze(source);
    verify_spans(source, &sentences);
    assert_eq!(
        sentences.iter().map(|s| s.text).collect::<Vec<_>>(),
        ["Dr. Lin can’t use 3.14 TB.", "He said “Done!”"]
    );
    assert_eq!(
        sentences[0]
            .tokens
            .iter()
            .map(|t| t.text)
            .collect::<Vec<_>>(),
        ["Dr", ".", "Lin", "can’t", "use", "3.14", "TB", "."]
    );
    assert!(analyzer.analyze(" \r\n\t").is_empty());
    assert_eq!(Locale::English.tag(), "en");
    Ok(())
}

#[test]
fn each_chinese_locale_applies_its_own_vocabulary() -> anyhow::Result<()> {
    let taiwan = Analyzer::new(Locale::ChineseTaiwan)?;
    let china = Analyzer::new(Locale::ChineseChina)?;
    let traditional = "臺灣的人工智慧使用記憶體。這是第二句！";
    let simplified = "中国大陆的人工智能使用内存。这是第二句！";
    let pairs = [
        (&taiwan, &china, traditional, ["臺灣", "人工智慧", "記憶體"]),
        (
            &china,
            &taiwan,
            simplified,
            ["中国大陆", "人工智能", "内存"],
        ),
    ];
    for (analyzer, other, source, terms) in pairs {
        let sentences = analyzer.analyze(source);
        verify_spans(source, &sentences);
        assert_eq!(sentences.len(), 2);
        assert_eq!(
            sentences[0]
                .tokens
                .iter()
                .filter(|t| t.kind == TokenKind::Term)
                .map(|t| t.text)
                .collect::<Vec<_>>(),
            terms
        );
        assert!(
            other
                .analyze(source)
                .iter()
                .flat_map(|s| &s.tokens)
                .all(|t| t.kind != TokenKind::Term)
        );
    }
    assert_eq!(Locale::ChineseTaiwan.tag(), "zh-Hant-TW");
    assert_eq!(Locale::ChineseChina.tag(), "zh-Hans-CN");
    Ok(())
}

#[test]
fn custom_terms_prefer_longest_matches_at_word_boundaries() -> anyhow::Result<()> {
    let analyzer = Analyzer::with_terms(Locale::English, &["New York", "New York City"])?;
    let source = "New Yorkshire is far from New York City.";
    let sentences = analyzer.analyze(source);
    verify_spans(source, &sentences);
    assert_eq!(
        sentences[0]
            .tokens
            .iter()
            .filter(|t| t.kind == TokenKind::Term)
            .map(|t| t.text)
            .collect::<Vec<_>>(),
        ["New York City"]
    );
    assert!(Analyzer::with_terms(Locale::English, &[" "]).is_err());
    Ok(())
}

#[test]
fn chinese_terms_accept_adjacent_han_text_without_splitting_ascii_words() -> anyhow::Result<()> {
    let analyzer = Analyzer::with_terms(Locale::ChineseTaiwan, &["星際記錄器", "GPU"])?;
    let source = "𠮷使用星際記錄器，GPU2和GPU都可用。";
    let sentences = analyzer.analyze(source);
    verify_spans(source, &sentences);
    assert_eq!(
        sentences[0]
            .tokens
            .iter()
            .filter(|t| t.kind == TokenKind::Term)
            .map(|t| t.text)
            .collect::<Vec<_>>(),
        ["星際記錄器", "GPU"]
    );
    Ok(())
}
