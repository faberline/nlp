//! Candidate spelling observations from the bundled Jieba dictionary.
//!
//! This is not a correction or grammar checker. Dictionary absence does not
//! mean the input is wrong; dictionary membership does not prove a suggestion
//! is correct. Only bounded, contiguous Han windows and one basic-CJK
//! substitution are examined. Traditional forms, names, domain vocabulary,
//! real-word errors, and other edits may be missed. All coverage is unknown.
//!
//! The abstract budget meters this module's scans and probes, not the one-time
//! initialization and memory cost of Jieba's bundled dictionary. The input is
//! borrowed and is never changed; callers should process bounded segments.
//! Offsets are valid only for that exact input. This API has no source digest
//! or resume cursor; callers must bind and verify their own source documents.

use std::collections::BTreeSet;
use std::ops::Range;

use crate::text::default_jieba;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Unknown {
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceRange {
    pub bytes: Range<usize>,
    pub scalars: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct AutoPolicy {
    /// Abstract work units; initializing Jieba is outside this budget.
    pub work_budget: usize,
    /// Contiguous Han windows from 2 through this many Unicode scalars.
    pub max_window_scalars: usize,
    pub max_alternatives_per_window: usize,
    pub max_reported_windows: usize,
    /// Exact written literals and ranges are protected, including overlaps.
    pub protected_literals: Vec<String>,
    pub protected_bytes: Vec<Range<usize>>,
}

impl Default for AutoPolicy {
    fn default() -> Self {
        Self {
            work_budget: 200_000,
            max_window_scalars: 3,
            max_alternatives_per_window: 8,
            max_reported_windows: 16,
            protected_literals: Vec::new(),
            protected_bytes: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanStatus {
    /// Window iteration finished; individual alternative lists can still be partial.
    FinishedConfiguredWindows,
    PartialBudget,
    /// The report limit was reached; further results may have been omitted.
    PartialReportedWindowLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlternativeStatus {
    FinishedBasicCjkAlphabet,
    PartialAlternativeLimit,
    PartialBudget,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Suggestion {
    pub spelling: String,
    /// The exact bundled dictionary lookup, not a frequency or confidence.
    pub basis: &'static str,
    pub correctness: Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Candidate {
    pub source: SourceRange,
    pub observed: String,
    pub suggestions: Vec<Suggestion>,
    pub alternatives: AlternativeStatus,
    pub correctness: Unknown,
    pub identity: Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutoReport {
    pub candidates: Vec<Candidate>,
    pub status: ScanStatus,
    pub coverage: Unknown,
    pub grammar: Unknown,
    pub meaning: Unknown,
    pub dictionary_scope: &'static str,
    pub operation_scope: &'static str,
    /// Callers must bind offsets to the exact source before saving or reuse.
    pub source_binding_status: &'static str,
    /// This API does not provide a resumable cursor.
    pub resume_status: &'static str,
    /// Earliest scalar start that was not fully inspected, relative to this
    /// exact input. Rechecking can repeat earlier output and can hit the same
    /// limits again. None does not establish spelling or grammar coverage.
    pub recheck_from_scalar: Option<usize>,
    pub work_used: usize,
    pub dictionary_probes: usize,
    /// Windows whose configured operation ran to completion.
    pub inspected_windows: usize,
    /// Windows stopped during alternative enumeration by a limit or budget.
    pub partial_windows: usize,
    pub protected_windows: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AutoError {
    InvalidLimits,
    EmptyProtectedLiteral,
    InvalidProtectedRange(Range<usize>),
}

struct Budget {
    used: usize,
    limit: usize,
    exhausted: bool,
}
impl Budget {
    fn charge(&mut self, units: usize) -> bool {
        match self.used.checked_add(units) {
            Some(next) if !self.exhausted && next <= self.limit => {
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

fn han(ch: char) -> bool {
    matches!(ch as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff |
        0x20000..=0x2fa1f | 0x30000..=0x323af)
}

fn valid_range(source: &str, range: &Range<usize>) -> bool {
    range.start < range.end
        && range.end <= source.len()
        && source.is_char_boundary(range.start)
        && source.is_char_boundary(range.end)
}

fn blank_report(status: ScanStatus, budget: &Budget) -> AutoReport {
    AutoReport {
        candidates: Vec::new(),
        status,
        coverage: Unknown::Unknown,
        grammar: Unknown::Unknown,
        meaning: Unknown::Unknown,
        dictionary_scope: "jieba_rs_bundled_default_dictionary",
        operation_scope: "contiguous_han_windows_basic_cjk_single_substitution",
        source_binding_status: "caller_must_bind_exact_input",
        resume_status: "unavailable_caller_segments_input",
        recheck_from_scalar: (status != ScanStatus::FinishedConfiguredWindows).then_some(0),
        work_used: budget.used,
        dictionary_probes: 0,
        inspected_windows: 0,
        partial_windows: 0,
        protected_windows: 0,
    }
}

/// Inspect the exact borrowed source; no caller vocabulary is required.
/// A returned suggestion is a dictionary match, not a finding or correction.
/// `FinishedConfiguredWindows` means only the configured bounded operation
/// finished. It does not establish correctness or exhaustive coverage.
pub fn scan(source: &str, policy: &AutoPolicy) -> Result<AutoReport, AutoError> {
    if policy.max_window_scalars < 2
        || policy.max_alternatives_per_window == 0
        || policy.max_reported_windows == 0
    {
        return Err(AutoError::InvalidLimits);
    }
    let mut budget = Budget {
        used: 0,
        limit: policy.work_budget,
        exhausted: false,
    };
    if !budget.charge(source.len()) {
        return Ok(blank_report(ScanStatus::PartialBudget, &budget));
    }
    for literal in &policy.protected_literals {
        if !budget.charge(literal.len().saturating_add(1)) {
            return Ok(blank_report(ScanStatus::PartialBudget, &budget));
        }
        if literal.is_empty() {
            return Err(AutoError::EmptyProtectedLiteral);
        }
    }
    let mut protected = Vec::new();
    for r in &policy.protected_bytes {
        if !budget.charge(1) {
            return Ok(blank_report(ScanStatus::PartialBudget, &budget));
        }
        if !valid_range(source, r) {
            return Err(AutoError::InvalidProtectedRange(r.clone()));
        }
        protected.push(r.clone());
    }
    for literal in &policy.protected_literals {
        for (start, _) in source.char_indices() {
            if !budget.charge(literal.len().saturating_add(1)) {
                // The index is incomplete: never emit an unprotected window.
                return Ok(blank_report(ScanStatus::PartialBudget, &budget));
            }
            if source[start..].starts_with(literal) {
                protected.push(start..start + literal.len());
            }
        }
    }
    let n = protected.len();
    let levels = (usize::BITS - n.leading_zeros()) as usize;
    if !budget.charge(n.saturating_mul(levels.saturating_add(1))) {
        return Ok(blank_report(ScanStatus::PartialBudget, &budget));
    }
    protected.sort_unstable_by_key(|r| (r.start, r.end));
    let mut merged: Vec<Range<usize>> = Vec::new();
    for r in protected {
        if let Some(last) = merged.last_mut()
            && r.start <= last.end
        {
            last.end = last.end.max(r.end);
        } else {
            merged.push(r);
        }
    }
    let mut report = blank_report(ScanStatus::FinishedConfiguredWindows, &budget);
    let mut first_incomplete: Option<usize> = None;
    let mut stopped_at: Option<usize> = None;
    let dictionary = default_jieba();
    'source: for (scalar_start, (byte_start, first)) in source.char_indices().enumerate() {
        if !budget.charge(1) {
            stopped_at = Some(scalar_start);
            break;
        }
        if !han(first) {
            continue;
        }
        for (index, (relative, ch)) in source[byte_start..]
            .char_indices()
            .take(policy.max_window_scalars)
            .enumerate()
        {
            if !budget.charge(1) {
                stopped_at = Some(scalar_start);
                break 'source;
            }
            if !han(ch) {
                break;
            }
            let count = index + 1;
            if count < 2 {
                continue;
            }
            let byte_end = byte_start + relative + ch.len_utf8();
            let window = byte_start..byte_end;
            let levels = (usize::BITS - merged.len().leading_zeros()) as usize;
            if !budget.charge(levels.saturating_add(1)) {
                stopped_at = Some(scalar_start);
                break 'source;
            }
            let i = merged.partition_point(|r| r.start < window.end);
            if i > 0 && merged[i - 1].end > window.start {
                report.protected_windows += 1;
                continue;
            }
            let observed = &source[window.clone()];
            if !budget.charge(observed.len().saturating_add(1)) {
                stopped_at = Some(scalar_start);
                break 'source;
            }
            report.dictionary_probes += 1;
            if dictionary.has_word(observed) {
                report.inspected_windows += 1;
                continue;
            }
            let mut chars: Vec<char> = observed.chars().collect();
            let mut suggestions = BTreeSet::new();
            let mut alternatives = AlternativeStatus::FinishedBasicCjkAlphabet;
            'replace: for position in 0..chars.len() {
                let original = chars[position];
                for code in 0x4e00..=0x9fff {
                    if !budget.charge(observed.len().saturating_add(1)) {
                        alternatives = AlternativeStatus::PartialBudget;
                        chars[position] = original;
                        break 'replace;
                    }
                    let replacement = char::from_u32(code).expect("basic CJK scalar");
                    if replacement == original {
                        continue;
                    }
                    chars[position] = replacement;
                    let proposal: String = chars.iter().collect();
                    report.dictionary_probes += 1;
                    if dictionary.has_word(&proposal) {
                        suggestions.insert(proposal);
                        if suggestions.len() >= policy.max_alternatives_per_window {
                            alternatives = AlternativeStatus::PartialAlternativeLimit;
                            chars[position] = original;
                            break 'replace;
                        }
                    }
                }
                chars[position] = original;
            }
            if alternatives == AlternativeStatus::FinishedBasicCjkAlphabet {
                report.inspected_windows += 1;
            } else {
                report.partial_windows += 1;
                first_incomplete = Some(
                    first_incomplete.map_or(scalar_start, |earlier| earlier.min(scalar_start)),
                );
            }
            if !suggestions.is_empty() {
                report.candidates.push(Candidate {
                    source: SourceRange {
                        bytes: window,
                        scalars: scalar_start..scalar_start + count,
                    },
                    observed: observed.into(),
                    suggestions: suggestions
                        .into_iter()
                        .map(|spelling| Suggestion {
                            spelling,
                            basis: "jieba_rs_bundled_default_dictionary_exact_match",
                            correctness: Unknown::Unknown,
                        })
                        .collect(),
                    alternatives,
                    correctness: Unknown::Unknown,
                    identity: Unknown::Unknown,
                });
                if report.candidates.len() >= policy.max_reported_windows {
                    report.status = ScanStatus::PartialReportedWindowLimit;
                    stopped_at = Some(scalar_start);
                    break 'source;
                }
            }
            if budget.exhausted {
                stopped_at = Some(scalar_start);
                break 'source;
            }
        }
    }
    if budget.exhausted {
        report.status = ScanStatus::PartialBudget;
    }
    report.recheck_from_scalar = match (first_incomplete, stopped_at) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    report.work_used = budget.used;
    Ok(report)
}
