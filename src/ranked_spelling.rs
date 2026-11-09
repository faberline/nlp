//! Budgeted, fair dictionary-candidate exploration. This is an experimental
//! source observation, not a spelling correction or a grammar checker.
//!
//! The ranking key is Jieba's `suggest_freq` heuristic, not the dictionary's
//! raw word frequency or a context model. Ranking covers only explored hits.
//! The budget counts abstract work units, not dictionary initialization or
//! process memory. Offsets bind only to the exact input supplied by the caller.
pub use crate::automatic_spelling::{SourceRange, Unknown};
use crate::text::default_jieba;
use std::collections::VecDeque;
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct RankedPolicy {
    pub work_budget: usize,
    pub max_window_scalars: usize,
    pub max_alternatives_per_window: usize,
    pub max_windows: usize,
    /// Maximum substitution attempts per window in one scheduling round.
    pub quantum: usize,
    pub protected_literals: Vec<String>,
    pub protected_bytes: Vec<Range<usize>>,
}
impl Default for RankedPolicy {
    fn default() -> Self {
        Self {
            work_budget: 200_000,
            max_window_scalars: 3,
            max_alternatives_per_window: 8,
            max_windows: 32,
            quantum: 64,
            protected_literals: Vec::new(),
            protected_bytes: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RankedScanStatus {
    FinishedConfiguredWindows,
    PartialBudget,
    PartialWindowLimit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankedSuggestion {
    pub spelling: String,
    /// `Jieba::suggest_freq` is an estimate and need not equal raw frequency.
    pub heuristic_score: usize,
    pub basis: &'static str,
    pub correctness: Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowObservation {
    pub source: SourceRange,
    pub observed: String,
    pub suggestions: Vec<RankedSuggestion>,
    /// True only when every configured basic-CJK substitution was explored.
    pub search_complete: bool,
    /// True when explored dictionary matches exceeded the returned top-k.
    pub alternatives_truncated: bool,
    pub visited_substitutions: usize,
    pub total_substitutions: usize,
    pub correctness: Unknown,
    pub identity: Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankedReport {
    pub windows: Vec<WindowObservation>,
    pub status: RankedScanStatus,
    pub coverage: Unknown,
    pub grammar: Unknown,
    pub meaning: Unknown,
    pub work_used: usize,
    pub dictionary_probes: usize,
    pub known_windows: usize,
    pub protected_windows: usize,
    pub partial_windows: usize,
    pub window_limit_reached: bool,
    pub budget_exhausted: bool,
    pub protection_index_complete: bool,
    pub dictionary_scope: &'static str,
    pub ranking_basis: &'static str,
    pub source_binding_status: &'static str,
    pub resume_status: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RankedError {
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
    fn charge(&mut self, amount: usize) -> bool {
        match self.used.checked_add(amount) {
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
fn valid(source: &str, range: &Range<usize>) -> bool {
    range.start < range.end
        && range.end <= source.len()
        && source.is_char_boundary(range.start)
        && source.is_char_boundary(range.end)
}
fn han(ch: char) -> bool {
    matches!(ch as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff |
        0x20000..=0x2fa1f | 0x30000..=0x323af)
}
fn report(budget: &Budget) -> RankedReport {
    RankedReport {
        windows: Vec::new(),
        status: RankedScanStatus::FinishedConfiguredWindows,
        coverage: Unknown::Unknown,
        grammar: Unknown::Unknown,
        meaning: Unknown::Unknown,
        work_used: budget.used,
        dictionary_probes: 0,
        known_windows: 0,
        protected_windows: 0,
        partial_windows: 0,
        window_limit_reached: false,
        budget_exhausted: budget.exhausted,
        protection_index_complete: false,
        dictionary_scope: "jieba_rs_bundled_default_dictionary",
        ranking_basis: "jieba_suggest_freq_heuristic_over_explored_matches_only",
        source_binding_status: "caller_must_bind_exact_input",
        resume_status: "unavailable_caller_segments_input",
    }
}
struct WindowState {
    span: SourceRange,
    chars: Vec<char>,
    next: usize,
    total: usize,
    matches: usize,
    kept: Vec<RankedSuggestion>,
}

/// Explore candidate windows with round-robin scheduling. Dictionary labels,
/// even high-scoring ones, do not establish correctness or identity.
pub fn scan_ranked(source: &str, policy: &RankedPolicy) -> Result<RankedReport, RankedError> {
    if policy.max_window_scalars < 2
        || policy.max_alternatives_per_window == 0
        || policy.max_windows == 0
        || policy.quantum == 0
    {
        return Err(RankedError::InvalidLimits);
    }
    let mut budget = Budget {
        used: 0,
        limit: policy.work_budget,
        exhausted: false,
    };
    if !budget.charge(source.len()) {
        let mut r = report(&budget);
        r.status = RankedScanStatus::PartialBudget;
        return Ok(r);
    }
    let mut protected = Vec::new();
    for range in &policy.protected_bytes {
        if !budget.charge(1) {
            let mut r = report(&budget);
            r.status = RankedScanStatus::PartialBudget;
            return Ok(r);
        }
        if !valid(source, range) {
            return Err(RankedError::InvalidProtectedRange(range.clone()));
        }
        protected.push(range.clone());
    }
    for literal in &policy.protected_literals {
        if literal.is_empty() {
            return Err(RankedError::EmptyProtectedLiteral);
        }
        for (start, _) in source.char_indices() {
            if !budget.charge(literal.len().saturating_add(1)) {
                let mut r = report(&budget);
                r.status = RankedScanStatus::PartialBudget;
                return Ok(r);
            }
            if source[start..].starts_with(literal) {
                protected.push(start..start + literal.len());
            }
        }
    }
    let n = protected.len();
    let levels = (usize::BITS - n.leading_zeros()) as usize;
    if !budget.charge(n.saturating_mul(levels.saturating_add(1))) {
        let mut r = report(&budget);
        r.status = RankedScanStatus::PartialBudget;
        return Ok(r);
    }
    protected.sort_unstable_by_key(|r| (r.start, r.end));
    let mut merged: Vec<Range<usize>> = Vec::new();
    for range in protected {
        if let Some(last) = merged.last_mut()
            && range.start <= last.end
        {
            last.end = last.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    let mut result = report(&budget);
    result.protection_index_complete = true;
    let dictionary = default_jieba();
    let scalars: Vec<(usize, char)> = source.char_indices().collect();
    let mut offsets: Vec<usize> = scalars.iter().map(|(byte, _)| *byte).collect();
    offsets.push(source.len());
    let mut states = Vec::<WindowState>::new();
    // Short windows get a discovery opportunity before longer windows.
    'discover: for width in 2..=policy.max_window_scalars.min(scalars.len()) {
        for start in 0..=scalars.len() - width {
            if !budget.charge(width.saturating_add(1)) {
                break 'discover;
            }
            if !scalars[start..start + width].iter().all(|(_, ch)| han(*ch)) {
                continue;
            }
            let bytes = offsets[start]..offsets[start + width];
            let lookup_cost = (usize::BITS - merged.len().leading_zeros()) as usize + 1;
            if !budget.charge(lookup_cost) {
                break 'discover;
            }
            let pos = merged.partition_point(|range| range.start < bytes.end);
            if pos > 0 && merged[pos - 1].end > bytes.start {
                result.protected_windows += 1;
                continue;
            }
            let observed = &source[bytes.clone()];
            if !budget.charge(observed.len().saturating_add(1)) {
                break 'discover;
            }
            result.dictionary_probes += 1;
            if dictionary.has_word(observed) {
                result.known_windows += 1;
                continue;
            }
            if states.len() == policy.max_windows {
                result.window_limit_reached = true;
                break 'discover;
            }
            let total = width.saturating_mul(0x9fffu32 as usize - 0x4e00usize + 1);
            states.push(WindowState {
                span: SourceRange {
                    bytes,
                    scalars: start..start + width,
                },
                chars: scalars[start..start + width]
                    .iter()
                    .map(|(_, ch)| *ch)
                    .collect(),
                next: 0,
                total,
                matches: 0,
                kept: Vec::new(),
            });
        }
    }
    let mut queue: VecDeque<usize> = (0..states.len()).collect();
    while !budget.exhausted {
        let Some(index) = queue.pop_front() else {
            break;
        };
        let state = &mut states[index];
        for _ in 0..policy.quantum {
            if state.next >= state.total {
                break;
            }
            let position = state.next % state.chars.len();
            let code = 0x4e00u32 + (state.next / state.chars.len()) as u32;
            if !budget.charge(state.span.bytes.len().saturating_add(1)) {
                break;
            }
            state.next += 1;
            let replacement = char::from_u32(code).expect("basic CJK scalar");
            if replacement == state.chars[position] {
                continue;
            }
            let original = state.chars[position];
            state.chars[position] = replacement;
            let proposed: String = state.chars.iter().collect();
            state.chars[position] = original;
            result.dictionary_probes += 1;
            if !dictionary.has_word(&proposed) {
                continue;
            }
            // suggest_freq calls a tokenizer; charge a conservative size-based
            // work unit, but do not claim a CPU or process-memory guarantee.
            let scoring = state
                .chars
                .len()
                .saturating_mul(state.chars.len())
                .saturating_mul(4)
                .saturating_add(proposed.len())
                .saturating_add(1)
                .saturating_add(state.kept.len().saturating_add(1).saturating_mul(
                    (usize::BITS - state.kept.len().saturating_add(1).leading_zeros()) as usize + 1,
                ));
            if !budget.charge(scoring) {
                // The lookup completed, but this hit was not scored or kept.
                // Do not report the window as completely examined.
                state.next -= 1;
                break;
            }
            state.matches += 1;
            let heuristic_score = dictionary.suggest_freq(&proposed);
            state.kept.push(RankedSuggestion {
                spelling: proposed,
                heuristic_score,
                basis: "jieba_suggest_freq_heuristic_not_raw_frequency",
                correctness: Unknown::Unknown,
            });
            state.kept.sort_unstable_by(|a, b| {
                b.heuristic_score
                    .cmp(&a.heuristic_score)
                    .then_with(|| a.spelling.cmp(&b.spelling))
            });
            state.kept.truncate(policy.max_alternatives_per_window);
        }
        if state.next < state.total {
            queue.push_back(index);
        }
    }
    result.partial_windows = states
        .iter()
        .filter(|state| state.next < state.total)
        .count();
    result.windows = states
        .into_iter()
        .map(|state| WindowObservation {
            observed: source[state.span.bytes.clone()].to_owned(),
            source: state.span,
            suggestions: state.kept,
            search_complete: state.next == state.total,
            alternatives_truncated: state.matches > policy.max_alternatives_per_window,
            visited_substitutions: state.next,
            total_substitutions: state.total,
            correctness: Unknown::Unknown,
            identity: Unknown::Unknown,
        })
        .collect();
    result.work_used = budget.used;
    result.budget_exhausted = budget.exhausted;
    result.status = if budget.exhausted {
        RankedScanStatus::PartialBudget
    } else if result.window_limit_reached {
        RankedScanStatus::PartialWindowLimit
    } else {
        RankedScanStatus::FinishedConfiguredWindows
    };
    Ok(result)
}
