//! Bounded, source-backed spelling suggestions; never text correction.
//! Vocabulary and token windows come from the caller. Grammar, meaning,
//! unknown vocabulary, and overall coverage are deliberately unresolved.
//! `Analyzer::analyze` materializes sentences and tokens for the entire input;
//! this is not a streaming API or a complete process-memory limit. Analyze
//! bounded chapters rather than assuming this budget makes a whole book safe.
use crate::text::{Analyzer, TokenKind};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceRange {
    pub bytes: Range<usize>,
    pub chars: Range<usize>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VocabularyEntry {
    pub word: String,
    pub left_context: Option<String>,
    pub right_context: Option<String>,
}
#[derive(Clone, Debug)]
pub struct QualityPolicy {
    pub vocabulary: Vec<VocabularyEntry>,
    pub protected_literals: Vec<String>,
    pub protected_bytes: Vec<Range<usize>>,
    pub max_edit_distance: usize,
    /// Deterministic work budget for this module, in abstract work units.
    /// This does not meter initialization or internal work of `Analyzer`.
    pub work_budget: usize,
}
impl Default for QualityPolicy {
    fn default() -> Self {
        Self {
            vocabulary: Vec::new(),
            protected_literals: Vec::new(),
            protected_bytes: Vec::new(),
            max_edit_distance: 0,
            work_budget: 1_000_000,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Unknown {
    Unknown,
}
/// This describes only whether the requested window iteration finished.
/// Even `FinishedRequestedWindows` does not establish correctness or coverage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Examination {
    FinishedRequestedWindows,
    PartialBudget,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Suggestion {
    pub word: String,
    pub edit_distance: usize,
    pub left_evidence: Option<SourceRange>,
    pub right_evidence: Option<SourceRange>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpellingCandidate {
    pub source: SourceRange,
    pub observed: String,
    pub suggestions: Vec<Suggestion>,
    pub correctness: Unknown,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QualityReport {
    pub candidates: Vec<SpellingCandidate>,
    /// Distinct attempted windows; may include one only partly examined when
    /// `examination` is `PartialBudget`.
    pub selected_windows: usize,
    pub inspected_windows: usize,
    pub protected_windows: usize,
    pub coverage: Unknown,
    pub grammar: Unknown,
    pub meaning: Unknown,
    pub examination: Examination,
    pub work_used: usize,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QualityError {
    EmptyVocabularyWord,
    EmptyProtectedLiteral,
    InvalidWindow(Range<usize>),
    InvalidProtectedRange(Range<usize>),
}

struct Meter {
    used: usize,
    limit: usize,
    exhausted: bool,
}
impl Meter {
    fn charge(&mut self, units: usize) -> bool {
        if self.exhausted {
            return false;
        }
        match self.used.checked_add(units) {
            Some(next) if next <= self.limit => {
                self.used = next;
                true
            }
            _ => {
                self.exhausted = true;
                false
            }
        }
    }
}
fn valid(source: &str, range: &Range<usize>) -> bool {
    range.start < range.end
        && range.end <= source.len()
        && source.is_char_boundary(range.start)
        && source.is_char_boundary(range.end)
}
fn located(offsets: &[usize], bytes: Range<usize>) -> Option<SourceRange> {
    Some(SourceRange {
        chars: offsets.binary_search(&bytes.start).ok()?..offsets.binary_search(&bytes.end).ok()?,
        bytes,
    })
}
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substitution = if ca == *cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1)
                .min(curr[j] + 1)
                .min(prev[j] + substitution);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

struct Scan<'a> {
    source: &'a str,
    policy: &'a QualityPolicy,
    meter: Meter,
    offsets: Vec<usize>,
    protected: Vec<Range<usize>>,
    seen: BTreeSet<(usize, usize)>,
    candidates: Vec<SpellingCandidate>,
    inspected: usize,
    protected_count: usize,
    ready: bool,
}
impl<'a> Scan<'a> {
    fn new(source: &'a str, policy: &'a QualityPolicy) -> Result<Self, QualityError> {
        let mut s = Self {
            source,
            policy,
            meter: Meter {
                used: 0,
                limit: policy.work_budget,
                exhausted: false,
            },
            offsets: Vec::new(),
            protected: Vec::new(),
            seen: BTreeSet::new(),
            candidates: Vec::new(),
            inspected: 0,
            protected_count: 0,
            ready: false,
        };
        // Covers the source traversal and bounds the size of the offsets vector.
        if !s.meter.charge(source.len()) {
            return Ok(s);
        }
        for entry in &policy.vocabulary {
            let cost = entry
                .word
                .len()
                .saturating_add(1)
                .saturating_add(entry.left_context.as_ref().map_or(0, String::len))
                .saturating_add(entry.right_context.as_ref().map_or(0, String::len));
            if !s.meter.charge(cost) {
                return Ok(s);
            }
            if entry.word.is_empty() {
                return Err(QualityError::EmptyVocabularyWord);
            }
        }
        for literal in &policy.protected_literals {
            if !s.meter.charge(literal.len().saturating_add(1)) {
                return Ok(s);
            }
            if literal.is_empty() {
                return Err(QualityError::EmptyProtectedLiteral);
            }
        }
        for r in &policy.protected_bytes {
            if !s.meter.charge(1) {
                return Ok(s);
            }
            if !valid(source, r) {
                return Err(QualityError::InvalidProtectedRange(r.clone()));
            }
            s.protected.push(r.clone());
        }
        for literal in &policy.protected_literals {
            // Each comparison is charged before it runs. Checking every scalar
            // boundary also finds overlapping literal occurrences.
            for (start, _) in source.char_indices() {
                if !s.meter.charge(literal.len().saturating_add(1)) {
                    return Ok(s);
                }
                if source[start..].starts_with(literal) {
                    s.protected.push(start..start + literal.len());
                }
            }
        }
        // Charge before sorting, and fail closed if the protected index cannot
        // be completed. Thus partial protection never emits unsafe candidates.
        let n = s.protected.len();
        let levels = (usize::BITS - n.leading_zeros()) as usize;
        if !s.meter.charge(n.saturating_mul(levels.saturating_add(1))) {
            return Ok(s);
        }
        s.protected.sort_unstable_by_key(|r| (r.start, r.end));
        let mut merged: Vec<Range<usize>> = Vec::new();
        for r in s.protected.drain(..) {
            if let Some(last) = merged.last_mut()
                && r.start <= last.end
            {
                last.end = last.end.max(r.end);
            } else {
                merged.push(r);
            }
        }
        s.protected = merged;
        s.offsets = source
            .char_indices()
            .map(|(byte, _)| byte)
            .chain(std::iter::once(source.len()))
            .collect();
        s.ready = true;
        Ok(s)
    }
    fn overlaps_protected(&self, range: &Range<usize>) -> bool {
        // Merged ranges are ordered and disjoint. At most one prior interval
        // can intersect the window.
        let i = self.protected.partition_point(|r| r.start < range.end);
        i > 0 && self.protected[i - 1].end > range.start
    }
    fn push(&mut self, window: Range<usize>) -> Result<bool, QualityError> {
        if !self.ready || !self.meter.charge(1) {
            return Ok(false);
        }
        if !valid(self.source, &window) {
            return Err(QualityError::InvalidWindow(window));
        }
        if !self.seen.insert((window.start, window.end)) {
            return Ok(true);
        }
        // Charge a conservative logarithmic index lookup, not a full scan.
        let levels = (usize::BITS - self.protected.len().leading_zeros()) as usize;
        if !self.meter.charge(levels.saturating_add(1)) {
            return Ok(false);
        }
        if self.overlaps_protected(&window) {
            self.protected_count += 1;
            return Ok(true);
        }
        let observed = &self.source[window.clone()];
        let mut exact_word_seen = false;
        for entry in &self.policy.vocabulary {
            let cost = observed
                .len()
                .saturating_add(entry.word.len())
                .saturating_add(1)
                .saturating_add(entry.left_context.as_ref().map_or(0, String::len))
                .saturating_add(entry.right_context.as_ref().map_or(0, String::len));
            if !self.meter.charge(cost) {
                return Ok(false);
            }
            if entry.word == observed {
                exact_word_seen = true;
                let left_matches = entry.left_context.as_deref().is_none_or(|left| {
                    left.is_empty() || self.source[..window.start].ends_with(left)
                });
                let right_matches = entry.right_context.as_deref().is_none_or(|right| {
                    right.is_empty() || self.source[window.end..].starts_with(right)
                });
                if left_matches && right_matches {
                    // An unconstrained exact entry also remains a conservative
                    // match: lack of context is not evidence that it is wrong.
                    self.inspected += 1;
                    return Ok(true);
                }
            }
        }
        let mut suggestions = BTreeMap::<String, Suggestion>::new();
        for entry in &self.policy.vocabulary {
            let cost = observed
                .len()
                .saturating_add(entry.word.len())
                .saturating_add(1)
                .saturating_add(entry.left_context.as_ref().map_or(0, String::len))
                .saturating_add(entry.right_context.as_ref().map_or(0, String::len));
            if !self.meter.charge(cost) {
                return Ok(false);
            }
            if exact_word_seen
                && entry.left_context.as_deref().is_none_or(str::is_empty)
                && entry.right_context.as_deref().is_none_or(str::is_empty)
            {
                // A real word in a different context needs affirmative literal
                // context evidence, not merely a similar dictionary spelling.
                continue;
            }
            let left = match &entry.left_context {
                None => None,
                Some(v) if v.is_empty() => None,
                Some(v) if self.source[..window.start].ends_with(v) => Some(
                    located(&self.offsets, window.start - v.len()..window.start)
                        .expect("valid context"),
                ),
                Some(_) => continue,
            };
            let right = match &entry.right_context {
                None => None,
                Some(v) if v.is_empty() => None,
                Some(v) if self.source[window.end..].starts_with(v) => Some(
                    located(&self.offsets, window.end..window.end + v.len())
                        .expect("valid context"),
                ),
                Some(_) => continue,
            };
            let a = observed.chars().count();
            let b = entry.word.chars().count();
            if a.abs_diff(b) > self.policy.max_edit_distance {
                continue;
            }
            let cells = a.saturating_add(1).saturating_mul(b.saturating_add(1));
            if !self.meter.charge(cells) {
                return Ok(false);
            }
            let distance = edit_distance(observed, &entry.word);
            if distance == 0 || distance > self.policy.max_edit_distance {
                continue;
            }
            suggestions.entry(entry.word.clone()).or_insert(Suggestion {
                word: entry.word.clone(),
                edit_distance: distance,
                left_evidence: left,
                right_evidence: right,
            });
        }
        if !suggestions.is_empty() {
            let n = suggestions.len();
            let levels = (usize::BITS - n.leading_zeros()) as usize;
            if !self
                .meter
                .charge(n.saturating_mul(levels.saturating_add(1)))
            {
                return Ok(false);
            }
            let mut suggestions: Vec<_> = suggestions.into_values().collect();
            suggestions.sort_by(|a, b| (a.edit_distance, &a.word).cmp(&(b.edit_distance, &b.word)));
            self.candidates.push(SpellingCandidate {
                source: located(&self.offsets, window).expect("validated range"),
                observed: observed.to_owned(),
                suggestions,
                correctness: Unknown::Unknown,
            });
        }
        self.inspected += 1;
        Ok(true)
    }
    fn report(self) -> QualityReport {
        QualityReport {
            candidates: self.candidates,
            selected_windows: self.seen.len(),
            inspected_windows: self.inspected,
            protected_windows: self.protected_count,
            coverage: Unknown::Unknown,
            grammar: Unknown::Unknown,
            meaning: Unknown::Unknown,
            examination: if self.meter.exhausted {
                Examination::PartialBudget
            } else {
                Examination::FinishedRequestedWindows
            },
            work_used: self.meter.used,
        }
    }
}

/// Inspect explicit byte windows. A partial budget is never complete coverage.
pub fn inspect_windows(
    source: &str,
    windows: &[Range<usize>],
    policy: &QualityPolicy,
) -> Result<QualityReport, QualityError> {
    let mut scan = Scan::new(source, policy)?;
    if scan.ready {
        for window in windows {
            if !scan.push(window.clone())? {
                break;
            }
        }
    }
    Ok(scan.report())
}

pub fn inspect_segmented(
    source: &str,
    analyzer: &Analyzer,
    policy: &QualityPolicy,
) -> Result<QualityReport, QualityError> {
    inspect_segmented_with_ngrams(source, analyzer, 1, policy)
}

// Conservative character filter: unknown marks or symbols are skipped rather
// than joining over a possible punctuation boundary.
fn lexical_token(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|ch| ch.is_alphanumeric() ||
        matches!(ch as u32, 0x0300..=0x036f | 0x1ab0..=0x1aff | 0x1dc0..=0x1dff | 0x20d0..=0x20ff | 0xfe20..=0xfe2f))
}

/// Inspect adjacent lexical token n-grams without first allocating windows.
/// Tokens containing spaces, punctuation, or unsupported symbols are excluded,
/// even if the caller registered them as a single `Term`. The Analyzer's own
/// segmentation/allocation is not included in this module's work-unit meter.
/// This is not streaming and the meter is not a process-memory bound.
pub fn inspect_segmented_with_ngrams(
    source: &str,
    analyzer: &Analyzer,
    max_tokens: usize,
    policy: &QualityPolicy,
) -> Result<QualityReport, QualityError> {
    let mut scan = Scan::new(source, policy)?;
    if !scan.ready {
        return Ok(scan.report());
    }
    'sentences: for sentence in analyzer.analyze(source) {
        for (i, first) in sentence.tokens.iter().enumerate() {
            if !scan.meter.charge(first.text.len().saturating_add(1)) {
                break 'sentences;
            }
            if first.kind == TokenKind::Symbol || !lexical_token(first.text) {
                continue;
            }
            let mut end = first.bytes.start;
            for token in sentence.tokens[i..].iter().take(max_tokens) {
                if !scan.meter.charge(token.text.len().saturating_add(1)) {
                    break 'sentences;
                }
                if token.kind == TokenKind::Symbol
                    || !lexical_token(token.text)
                    || token.bytes.start != end
                {
                    break;
                }
                end = token.bytes.end;
                if !scan.push(first.bytes.start..end)? {
                    break 'sentences;
                }
            }
        }
    }
    Ok(scan.report())
}
