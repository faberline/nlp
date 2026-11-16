use std::collections::HashMap;

const MAX_NGRAM: usize = 6;
const BETA: f64 = 2.0;

fn ngram_counts(chars: &[char], n: usize) -> HashMap<&[char], usize> {
    let mut counts = HashMap::new();
    if chars.len() >= n {
        for w in chars.windows(n) {
            *counts.entry(w).or_insert(0) += 1;
        }
    }
    counts
}

/// Sentence-level chrF (character n-grams 1..=6, beta 2), 0-100. Whitespace is ignored.
/// Precision and recall are averaged over the orders the reference has, then combined.
pub fn chrf(hyp: &str, reference: &str) -> f64 {
    let h: Vec<char> = hyp.chars().filter(|c| !c.is_whitespace()).collect();
    let r: Vec<char> = reference.chars().filter(|c| !c.is_whitespace()).collect();
    let (mut p_sum, mut r_sum, mut orders) = (0.0, 0.0, 0usize);
    for n in 1..=MAX_NGRAM {
        if r.len() < n {
            break;
        }
        let hc = ngram_counts(&h, n);
        let rc = ngram_counts(&r, n);
        let matched: usize = rc
            .iter()
            .map(|(g, &c)| c.min(hc.get(g).copied().unwrap_or(0)))
            .sum();
        let h_total = h.len().saturating_sub(n - 1);
        let r_total = r.len() - (n - 1);
        p_sum += if h_total == 0 {
            0.0
        } else {
            matched as f64 / h_total as f64
        };
        r_sum += matched as f64 / r_total as f64;
        orders += 1;
    }
    if orders == 0 {
        return if h.is_empty() { 100.0 } else { 0.0 };
    }
    let p = p_sum / orders as f64;
    let rc = r_sum / orders as f64;
    if p + rc == 0.0 {
        return 0.0;
    }
    let b2 = BETA * BETA;
    100.0 * (1.0 + b2) * p * rc / (b2 * p + rc)
}
