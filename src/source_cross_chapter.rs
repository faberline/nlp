//! Bounded written-form co-occurrence across chapters.
//!
//! These observations do not resolve pronouns, establish people, or attach a
//! participant to an event. The caller must bind every returned range to the
//! exact chapter bytes. A chapter ordinal and ranges are not authentication.
//! Limits can make output partial. Empty output does not prove absence.

use std::{
    collections::{BTreeSet, VecDeque},
    ops::{Range, RangeInclusive},
};

use aho_corasick::AhoCorasick;
use anyhow::{Result, ensure};
use serde::Serialize;

use crate::{
    Lexicon, Span,
    analyze::admitted_lines,
    text::{Analyzer, Locale},
};

/// Caller-selected resource limits. All fields must be positive.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct CrossChapterPolicy {
    pub max_chapter_bytes: usize,
    pub max_terms: usize,
    /// Maximum sum of all supplied actor-surface byte lengths.
    pub max_term_bytes: usize,
    /// Maximum difference in chapter ordinals retained for comparison.
    pub history_distance: u32,
    pub max_names_per_chapter: usize,
    pub max_pronouns_per_chapter: usize,
    pub max_pairs_per_pronoun: usize,
    pub max_pairs_per_chapter: usize,
}

/// Whether the configured written-form operation ran to completion.
/// `Complete` does not establish language coverage or semantic correctness.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Inspection {
    Complete,
    Partial,
    NotExamined,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Unknown {
    Unknown,
}

/// Half-open positions in the exact unmodified source chapter.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SourcePosition {
    pub chapter: u32,
    pub bytes: Range<usize>,
    pub scalars: Span,
}

/// An exact token-shaped written form, not a verified subject or pronoun use.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct WrittenPronoun {
    pub source: SourcePosition,
    pub written_text: String,
    pub interpretation_status: Unknown,
}

/// Two written forms within the configured chapter window, without coreference.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Candidate {
    pub prior_name: SourcePosition,
    pub current_pronoun: SourcePosition,
    pub basis: &'static str,
    pub identity_status: Unknown,
    pub role_status: Unknown,
    pub association_status: Unknown,
    pub occurrence_status: Unknown,
    pub narrative_scope_status: Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ChapterCandidates {
    pub chapter: u32,
    /// The exact configured limits; complete is never a whole-book claim.
    pub policy: CrossChapterPolicy,
    /// BCP 47 locale supplied by the caller; this does not identify the source language.
    pub locale: &'static str,
    /// Scope of the configured written-form scan, not grammatical coverage.
    pub pronoun_inspection_scope: &'static str,
    /// Exact token forms examined by this implementation, or empty if not examined.
    pub pronoun_forms: &'static [&'static str],
    pub history_distance: u32,
    /// Inclusive first ordinal in the configured history window.
    pub history_start_chapter: u32,
    /// Earlier ordinals are outside this window, not necessarily examined.
    pub earlier_chapters_outside_window: bool,
    pub names: Vec<SourcePosition>,
    pub pronouns: Vec<WrittenPronoun>,
    pub candidates: Vec<Candidate>,
    pub names_status: Inspection,
    pub pronouns_status: Inspection,
    pub pairs_status: Inspection,
    /// At least one pair was omitted due to a configured pair limit.
    pub pair_limit_reached: bool,
    /// At least one configured-window chapter is missing or had capped names.
    pub history_incomplete: bool,
    /// Previously processed window chapters whose name scan hit its limit.
    pub partial_name_chapters: Vec<u32>,
    /// Missing or partial history must be recovered separately from this chapter.
    pub history_recheck_required: bool,
    /// Inclusive ordinal ranges absent within the configured history window.
    pub missing_chapter_ranges: Vec<RangeInclusive<u32>>,
    /// Conservative current-chapter recheck start, not a resumable cursor;
    /// rechecking this chapter alone does not repair incomplete history or
    /// remove configured pair limits.
    pub recheck_from_scalar: Option<usize>,
    pub source_binding_status: &'static str,
    pub coverage: Unknown,
    pub identity_status: Unknown,
    pub role_status: Unknown,
    pub association_status: Unknown,
    pub occurrence_status: Unknown,
    pub narrative_scope_status: Unknown,
}

struct HistoricalChapter {
    order: u32,
    names: Vec<SourcePosition>,
    names_status: Inspection,
}

/// Reusable, bounded history of written names. It never stores chapter text.
pub struct CrossChapterCandidates<'a> {
    lexicon: Lexicon<'a>,
    locale: Locale,
    policy: CrossChapterPolicy,
    matcher: Option<AhoCorasick>,
    tokenizer: Option<Analyzer>,
    history: VecDeque<HistoricalChapter>,
    last_order: Option<u32>,
}

// Sparse original-source byte/scalar mapping. Each lookup scans at most 255
// scalars. The number of checkpoints is bounded by max_chapter_bytes / 256.
struct ScalarCheckpoints(Vec<(usize, usize)>);
impl ScalarCheckpoints {
    fn new(text: &str) -> Self {
        let mut checkpoints = vec![(0, 0)];
        for (scalar, (byte, _)) in text.char_indices().enumerate() {
            if scalar != 0 && scalar % 256 == 0 {
                checkpoints.push((byte, scalar));
            }
        }
        Self(checkpoints)
    }

    fn scalar(&self, text: &str, byte: usize) -> usize {
        let index = self.0.partition_point(|&(offset, _)| offset <= byte) - 1;
        let (offset, scalar) = self.0[index];
        scalar + text[offset..byte].chars().count()
    }

    fn position(&self, order: u32, text: &str, start: usize, end: usize) -> SourcePosition {
        SourcePosition {
            chapter: order,
            bytes: start..end,
            scalars: Span {
                start_char: self.scalar(text, start),
                end_char: self.scalar(text, end),
            },
        }
    }
}

const CHINESE_PRONOUN_FORMS: &[&str] = &[
    "他", "她", "它", "他們", "她們", "它們", "他们", "她们", "它们",
];

fn written_pronoun(text: &str) -> bool {
    CHINESE_PRONOUN_FORMS.contains(&text)
}

impl<'a> CrossChapterCandidates<'a> {
    /// Prepare exact caller-provided actor surfaces and a generic tokenizer.
    /// Canonical labels are deliberately not used for matching or grouping.
    pub fn new(locale: Locale, lexicon: Lexicon<'a>, policy: CrossChapterPolicy) -> Result<Self> {
        ensure!(
            policy.max_chapter_bytes > 0
                && policy.max_terms > 0
                && policy.max_term_bytes > 0
                && policy.history_distance > 0
                && policy.max_names_per_chapter > 0
                && policy.max_pronouns_per_chapter > 0
                && policy.max_pairs_per_pronoun > 0
                && policy.max_pairs_per_chapter > 0,
            "all limits must be positive"
        );
        ensure!(
            lexicon.actors.len() <= policy.max_terms,
            "actor term limit exceeded"
        );
        let mut bytes = 0usize;
        for &(surface, _) in lexicon.actors {
            ensure!(
                !surface.is_empty() && surface.trim() == surface && !surface.contains(['\n', '\r']),
                "actor surfaces must be nonempty and have no outer whitespace or line breaks"
            );
            bytes = bytes
                .checked_add(surface.len())
                .ok_or_else(|| anyhow::anyhow!("actor term bytes overflow"))?;
            ensure!(
                bytes <= policy.max_term_bytes,
                "actor term byte limit exceeded"
            );
        }
        let matcher = if lexicon.actors.is_empty() {
            None
        } else {
            Some(AhoCorasick::new(
                lexicon.actors.iter().map(|&(surface, _)| surface),
            )?)
        };
        let tokenizer = match locale {
            Locale::ChineseTaiwan | Locale::ChineseChina => Some(Analyzer::new(locale)?),
            Locale::English => None,
        };
        Ok(Self {
            lexicon,
            locale,
            policy,
            matcher,
            tokenizer,
            history: VecDeque::new(),
            last_order: None,
        })
    }

    /// Written pronoun byte ranges of one line. zh-TW reads `nlp.doc.v1` layer 2 tokens and
    /// falls back to the generic tokenizer when the line has no layer 2 reading.
    fn line_pronouns(&self, tokenizer: &Analyzer, line: &str) -> Vec<Range<usize>> {
        if self.locale == Locale::ChineseTaiwan
            && crate::analyze::doc_applies(line)
            && let Some(tokens) = crate::doc::line_tokens(line)
        {
            return tokens
                .into_iter()
                .filter(|r| written_pronoun(&line[r.clone()]))
                .collect();
        }
        tokenizer
            .analyze(line)
            .into_iter()
            .flat_map(|sentence| sentence.tokens)
            .filter(|token| written_pronoun(token.text))
            .map(|token| token.bytes)
            .collect()
    }

    /// Observe one chapter in strictly increasing ordinal order.
    /// Empty chapters must be pushed too: skipped ordinals are reported missing.
    /// Errors leave history unchanged. All output is provisional written form.
    pub fn push(&mut self, order: u32, text: &str, title: &str) -> Result<ChapterCandidates> {
        ensure!(order > 0, "chapter ordinal must be positive");
        if let Some(last) = self.last_order {
            ensure!(order > last, "chapter ordinals must increase");
        }
        ensure!(
            text.len() <= self.policy.max_chapter_bytes,
            "chapter byte limit exceeded"
        );

        let checkpoints = ScalarCheckpoints::new(text);
        let mut names = BTreeSet::<(usize, usize)>::new();
        let mut pronouns = BTreeSet::<(usize, usize)>::new();
        let mut names_status = Inspection::Complete;
        let mut pronouns_status = if self.tokenizer.is_some() {
            Inspection::Complete
        } else {
            Inspection::NotExamined
        };
        for line in admitted_lines(text, title, &self.lexicon) {
            if names_status == Inspection::Complete
                && let Some(matcher) = &self.matcher
            {
                for hit in matcher.find_overlapping_iter(line.text) {
                    let range = (line.start + hit.start(), line.start + hit.end());
                    if names.contains(&range) {
                        continue;
                    }
                    if names.len() == self.policy.max_names_per_chapter {
                        names_status = Inspection::Partial;
                        break;
                    }
                    names.insert(range);
                }
            }
            if pronouns_status == Inspection::Complete
                && let Some(tokenizer) = &self.tokenizer
            {
                for bytes in self.line_pronouns(tokenizer, line.text) {
                    let range = (line.start + bytes.start, line.start + bytes.end);
                    if pronouns.contains(&range) {
                        continue;
                    }
                    if pronouns.len() == self.policy.max_pronouns_per_chapter {
                        pronouns_status = Inspection::Partial;
                        break;
                    }
                    pronouns.insert(range);
                }
            }
        }
        let names: Vec<_> = names
            .into_iter()
            .map(|(a, b)| checkpoints.position(order, text, a, b))
            .collect();
        let pronouns: Vec<_> = pronouns
            .into_iter()
            .map(|(a, b)| WrittenPronoun {
                source: checkpoints.position(order, text, a, b),
                written_text: text[a..b].to_owned(),
                interpretation_status: Unknown::Unknown,
            })
            .collect();

        let floor = order.saturating_sub(self.policy.history_distance).max(1);
        let mut missing = Vec::new();
        let mut next = floor;
        for prior in self.history.iter().filter(|chapter| chapter.order >= floor) {
            if next < prior.order {
                missing.push(next..=prior.order - 1);
            }
            next = prior.order + 1;
        }
        if next < order {
            missing.push(next..=order - 1);
        }
        let partial_name_chapters: Vec<_> = self
            .history
            .iter()
            .filter(|chapter| {
                chapter.order >= floor && chapter.names_status != Inspection::Complete
            })
            .map(|chapter| chapter.order)
            .collect();
        let history_incomplete = !missing.is_empty() || !partial_name_chapters.is_empty();
        let mut pairs_status = if self.tokenizer.is_none() {
            Inspection::NotExamined
        } else if history_incomplete || pronouns_status == Inspection::Partial {
            Inspection::Partial
        } else {
            Inspection::Complete
        };
        let mut pair_limit_reached = false;
        let mut candidates = Vec::new();
        if self.tokenizer.is_some() {
            for pronoun in &pronouns {
                let mut per_pronoun = 0usize;
                for prior in self
                    .history
                    .iter()
                    .rev()
                    .filter(|chapter| chapter.order >= floor)
                {
                    for name in &prior.names {
                        if per_pronoun == self.policy.max_pairs_per_pronoun
                            || candidates.len() == self.policy.max_pairs_per_chapter
                        {
                            pairs_status = Inspection::Partial;
                            pair_limit_reached = true;
                            break;
                        }
                        candidates.push(Candidate {
                            prior_name: name.clone(),
                            current_pronoun: pronoun.source.clone(),
                            basis: "cooccurrence_in_configured_history_window",
                            identity_status: Unknown::Unknown,
                            role_status: Unknown::Unknown,
                            association_status: Unknown::Unknown,
                            occurrence_status: Unknown::Unknown,
                            narrative_scope_status: Unknown::Unknown,
                        });
                        per_pronoun += 1;
                    }
                    if per_pronoun == self.policy.max_pairs_per_pronoun
                        || candidates.len() == self.policy.max_pairs_per_chapter
                    {
                        // Continue looking only to detect a genuinely omitted pair.
                        // The next nonempty chapter marks the operation partial.
                        continue;
                    }
                }
            }
        }
        let partial = names_status == Inspection::Partial
            || pronouns_status == Inspection::Partial
            || pairs_status == Inspection::Partial;
        let output = ChapterCandidates {
            chapter: order,
            policy: self.policy,
            locale: self.locale.tag(),
            pronoun_inspection_scope: if self.tokenizer.is_some() {
                "chinese_exact_token_form_subset"
            } else {
                "not_examined"
            },
            pronoun_forms: if self.tokenizer.is_some() {
                CHINESE_PRONOUN_FORMS
            } else {
                &[]
            },
            history_distance: self.policy.history_distance,
            history_start_chapter: floor,
            earlier_chapters_outside_window: floor > 1,
            names: names.clone(),
            pronouns,
            candidates,
            names_status,
            pronouns_status,
            pairs_status,
            pair_limit_reached,
            history_incomplete,
            partial_name_chapters,
            history_recheck_required: history_incomplete,
            missing_chapter_ranges: missing,
            recheck_from_scalar: partial.then_some(0),
            source_binding_status: "caller_must_bind_exact_input",
            coverage: Unknown::Unknown,
            identity_status: Unknown::Unknown,
            role_status: Unknown::Unknown,
            association_status: Unknown::Unknown,
            occurrence_status: Unknown::Unknown,
            narrative_scope_status: Unknown::Unknown,
        };

        // Commit only after all fallible validation and result construction.
        while self
            .history
            .front()
            .is_some_and(|chapter| chapter.order < floor)
        {
            self.history.pop_front();
        }
        self.history.push_back(HistoricalChapter {
            order,
            names,
            names_status,
        });
        self.last_order = Some(order);
        Ok(output)
    }
}

#[cfg(test)]
mod doc_pronoun_tests {
    //! zh-TW pronoun tokens come from `nlp.doc.v1` layer 2. They must equal the generic tokenizer.
    use super::*;

    const LINES: &[&str] = &[
        "他走進木屋。她留在港口。\n",
        "他們說：「我們明天出發。」它們不動。\n",
        "他说她们走了，它也走了。\n",
        "哈哈 他先走了 我也要回家了\n",
        "Alice met him at the 港口. 她笑了。\n",
        "他走了很久……終於到了。\n",
        "他讀了 `a。b` 這段，她也讀了。\n",
        "他看了 https://example.org/a 然後她走了。\n",
        "一行沒有標點的文字\n",
        "\n",
        "- 他走進木屋。\n",
        "# 他走了\n",
        "第一章 他們\n",
        "1. 她走了。\n",
        "把 `他` 換成 `她`。\n",
        "$他$ 她\n",
        "他走了。她留下。\n",
        "他說：「我們走。」她點頭。\n",
    ];

    /// Fixed-seed generated lines (the allow-list alphabet, and the same with other chars and
    /// paragraph indents mixed in): layer 2 pronouns must equal the generic tokenizer.
    #[test]
    fn generated_lines_give_the_generic_tokenizer_pronouns() {
        use crate::analyze::doc_line_tests::{Rng, gen_allowed_line, gen_mixed_line};
        let analyzer = Analyzer::new(Locale::ChineseTaiwan).expect("analyzer");
        let candidates = CrossChapterCandidates::new(
            Locale::ChineseTaiwan,
            Lexicon::default(),
            CrossChapterPolicy {
                max_chapter_bytes: 1 << 20,
                max_terms: 8,
                max_term_bytes: 64,
                history_distance: 4,
                max_names_per_chapter: 8,
                max_pronouns_per_chapter: 64,
                max_pairs_per_pronoun: 8,
                max_pairs_per_chapter: 64,
            },
        )
        .expect("new");
        let mut rng = Rng(0x5EED_2023);
        let (mut differ, mut used) = (Vec::new(), 0);
        for index in 0..20_000 {
            let mut line = if index % 2 == 0 {
                gen_allowed_line(&mut rng)
            } else {
                gen_mixed_line(&mut rng)
            };
            line.push('\n');
            used += usize::from(crate::analyze::doc_applies(&line));
            let old: Vec<_> = analyzer
                .analyze(&line)
                .into_iter()
                .flat_map(|s| s.tokens)
                .filter(|t| written_pronoun(t.text))
                .map(|t| t.bytes)
                .collect();
            if candidates.line_pronouns(&analyzer, &line) != old {
                differ.push(line);
            }
        }
        assert!(used >= 5_000, "only {used} generated lines use layer 2");
        assert!(
            differ.is_empty(),
            "{} lines differ, first {:?}",
            differ.len(),
            differ.first()
        );
    }

    #[test]
    fn layer_2_gives_the_generic_tokenizer_pronouns() {
        let lexicon = Lexicon::default();
        let analyzer = Analyzer::new(Locale::ChineseTaiwan).expect("analyzer");
        let policy = CrossChapterPolicy {
            max_chapter_bytes: 1 << 20,
            max_terms: 8,
            max_term_bytes: 64,
            history_distance: 4,
            max_names_per_chapter: 8,
            max_pronouns_per_chapter: 64,
            max_pairs_per_pronoun: 8,
            max_pairs_per_chapter: 64,
        };
        let candidates =
            CrossChapterCandidates::new(Locale::ChineseTaiwan, lexicon, policy).expect("new");
        for line in LINES {
            let old: Vec<_> = analyzer
                .analyze(line)
                .into_iter()
                .flat_map(|s| s.tokens)
                .filter(|t| written_pronoun(t.text))
                .map(|t| t.bytes)
                .collect();
            assert_eq!(candidates.line_pronouns(&analyzer, line), old, "{line:?}");
        }
    }
}
