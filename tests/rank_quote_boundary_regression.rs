//! Synthetic regression for a predicate and rank on opposite sides of a quote.

use faberline_nlp::{Lexicon, NarrativeAnalyzer, analyze, text::Locale};

#[test]
fn predicate_before_short_quote_and_rank_after_quote_do_not_panic() {
    let source = "甲抬手「光」突破築基成功。";
    let lexicon = Lexicon {
        actors: &[("甲", "甲")],
        ranks: &[("築基", "築基")],
        ..Lexicon::default()
    };
    let analyzer = NarrativeAnalyzer::new(Locale::ChineseTaiwan, &lexicon).unwrap();
    for analysis in [
        analyze(source, "", &lexicon),
        analyzer.analyze_source_mentions(source, ""),
    ] {
        for event in analysis.events {
            for range in event.evidence {
                assert!(range.start_char < range.end_char);
                assert!(range.end_char <= source.chars().count());
            }
        }
    }
}
