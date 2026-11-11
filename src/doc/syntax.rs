//! Layer 4 of `nlp.doc.v1`: Universal Dependencies v2 syntax trees and contextual UPOS upgrade.
//! See `docs/doc-v1.md`, section "Syntax (layer 4)".

use crate::Span;
#[cfg(any(feature = "onnx", feature = "coreml"))]
use crate::doc::model::{Backend, SubwordTokenizer};
use crate::doc::model::{Batch, Model, ModelError, Subwords, no_model_unknown};
use crate::doc::{
    Document, EntityKind, Options, Reason, Unknown, Upos, analyze_document, sort_unknown,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(any(feature = "onnx", feature = "coreml"))]
use std::path::Path;

/// Default model identifier for Layer 4 syntax decoding.
pub const SYNTAX_MODEL_ID: &str = "doc-syntax-v1";

/// Minimum quantized confidence (`0.35`, i.e. `350 / 1000`) to emit a [`SyntaxToken`].
pub const SYNTAX_THRESHOLD: f32 = 0.35;

/// Quantized [`SYNTAX_THRESHOLD`] in thousandths (`0..=1000`).
pub const SYNTAX_THRESHOLD_MILLI: u16 = 350;

/// Minimum quantized confidence (`0.20`, i.e. `200 / 1000`) for an intermediate ancestor to stay
/// in `Document.syntax` when connecting a `>= 0.35` descendant to the sentence `"root"`.
pub const ANCESTOR_MIN_MILLI: u16 = 200;

/// Soft subword window budget for long paratactic sentences (`>= 4` clauses and `> 64` tokens).
pub const SOFT_WINDOW_MAX_LEN: usize = 96;

/// Minimum UPOS probability (`0.50`) required to upgrade an undecided Layer-2 token's `upos`.
pub const UPOS_THRESHOLD: f64 = 0.50;

/// Index of `"parataxis"` inside [`DEPRELS`].
const PARATAXIS_DEPREL_INDEX: usize = 31;

/// Index of `"root"` inside [`DEPRELS`].
pub const ROOT_DEPREL_INDEX: usize = 34;

/// Exact 17 Universal POS tags in `Upos` enum order (`0..17` of the packed output tensor).
pub const UPOS_TAGS: [Upos; 17] = [
    Upos::Adj,
    Upos::Adp,
    Upos::Adv,
    Upos::Aux,
    Upos::Cconj,
    Upos::Det,
    Upos::Intj,
    Upos::Noun,
    Upos::Num,
    Upos::Part,
    Upos::Pron,
    Upos::Propn,
    Upos::Punct,
    Upos::Sconj,
    Upos::Sym,
    Upos::Verb,
    Upos::X,
];

/// Exact 37 alphabetical UD v2 base dependency relation labels (`"root"` is index `34`).
pub const DEPRELS: [&str; 37] = [
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
];

/// Quantized confidence score in thousandths (`0..=1000`), serialized as `(milli as f64) / 1000.0`.
///
/// Keeping integer thousandths preserves `Eq` on [`Document`] and gives byte-identical JSON across
/// ONNX and Core ML backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Score(pub u16);

impl Score {
    /// Quantizes a probability in `[0.0, 1.0]` to nearest thousandths (`0..=1000`).
    pub fn from_confidence(conf: f64) -> Self {
        if !conf.is_finite() {
            return Self(0);
        }
        Self((conf.clamp(0.0, 1.0) * 1000.0).round() as u16)
    }

    /// Raw thousandths value (`0..=1000`).
    pub const fn milli(self) -> u16 {
        self.0
    }

    /// Floating-point value in `[0.0, 1.0]`.
    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / 1000.0
    }
}

impl Serialize for Score {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.as_f64())
    }
}

impl<'de> Deserialize<'de> for Score {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = f64::deserialize(deserializer)?;
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(serde::de::Error::custom(
                "score must be a finite number in 0.0..=1.0",
            ));
        }
        Ok(Self((v * 1000.0).round() as u16))
    }
}

/// One token's dependency attachment in `Document.syntax` (Layer 4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntaxToken {
    /// 0-based index into `Document.tokens`.
    pub token: usize,
    /// 0-based index into `Document.segmentation.sentences`.
    pub sentence: usize,
    pub span: Span,
    /// `None` for the sentence `"root"`, or `Some(global_token_idx)` into `Document.tokens`.
    #[serde(default)]
    pub head: Option<usize>,
    pub deprel: String,
    pub score: Score,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Weight {
    root_pen: i64,
    score: f64,
}

impl Weight {
    fn sub(self, other: Self) -> Self {
        Self {
            root_pen: self.root_pen - other.root_pen,
            score: self.score - other.score,
        }
    }

    fn gt(self, other: Self) -> bool {
        self.root_pen > other.root_pen
            || (self.root_pen == other.root_pen && self.score > other.score)
    }
}

#[derive(Clone, Copy, Debug)]
struct MsaEdge {
    u: usize,
    v: usize,
    weight: Weight,
    parent_idx: usize,
}

fn clean_logit(x: f32) -> f64 {
    if x.is_nan() {
        f64::from(f32::MIN)
    } else {
        f64::from(x).clamp(f64::from(f32::MIN), f64::from(f32::MAX))
    }
}

/// Exact single-root Chu-Liu/Edmonds (Maximum Spanning Arborescence) over nodes `0..=m` (`0` = root).
///
/// `arc_scores` has `m` rows (one per dependent token `u` in `1..=m`), and each row has `m + 1`
/// entries for candidate heads `v` in `0..=m`. Returns a vector `heads` of length `m` where
/// `heads[u - 1]` is in `0..=m`, with **exactly one** `0` (root) when `m >= 1`, no self-loops,
/// and no cycles.
pub fn chu_liu_edmonds(arc_scores: &[Vec<f32>]) -> Vec<usize> {
    let m = arc_scores.len();
    if m == 0 {
        return Vec::new();
    }
    for (i, row) in arc_scores.iter().enumerate() {
        assert_eq!(
            row.len(),
            m + 1,
            "arc_scores row {i} has len {}, expected {}",
            row.len(),
            m + 1
        );
    }
    if m == 1 {
        return vec![0];
    }

    let num_nodes = m + 1;
    let mut edges = Vec::with_capacity(m * m);
    for (v_idx, row) in arc_scores.iter().enumerate() {
        let v = v_idx + 1;
        for (u, &raw) in row.iter().enumerate() {
            if u == v {
                continue;
            }
            edges.push(MsaEdge {
                u,
                v,
                weight: Weight {
                    root_pen: if u == 0 { -1 } else { 0 },
                    score: clean_logit(raw),
                },
                parent_idx: 0,
            });
        }
    }

    let chosen = msa(num_nodes, 0, &edges);
    chosen.into_iter().map(|idx| edges[idx].u).collect()
}

fn msa(num_nodes: usize, root: usize, edges: &[MsaEdge]) -> Vec<usize> {
    let mut best_in_idx: Vec<Option<usize>> = vec![None; num_nodes];
    for (idx, e) in edges.iter().enumerate() {
        if e.v == root || e.u == e.v {
            continue;
        }
        match best_in_idx[e.v] {
            None => best_in_idx[e.v] = Some(idx),
            Some(prev) => {
                if e.weight.gt(edges[prev].weight) {
                    best_in_idx[e.v] = Some(idx);
                }
            }
        }
    }

    let mut visited = vec![usize::MAX; num_nodes];
    let mut comp = vec![usize::MAX; num_nodes];
    let mut in_cycle = vec![false; num_nodes];
    let mut num_cycles = 0usize;

    for start in 0..num_nodes {
        let mut cur = start;
        while cur != root && comp[cur] == usize::MAX && visited[cur] == usize::MAX {
            visited[cur] = start;
            let idx = best_in_idx[cur].expect("every non-root node has an incoming edge");
            cur = edges[idx].u;
        }
        if cur != root && comp[cur] == usize::MAX && visited[cur] == start {
            let cid = num_cycles;
            num_cycles += 1;
            let mut walk = cur;
            loop {
                comp[walk] = cid;
                in_cycle[walk] = true;
                let idx = best_in_idx[walk].expect("cycle node has an incoming edge");
                walk = edges[idx].u;
                if walk == cur {
                    break;
                }
            }
        }
    }

    if num_cycles == 0 {
        return (0..num_nodes)
            .filter(|&v| v != root)
            .map(|v| best_in_idx[v].expect("every non-root node has an incoming edge"))
            .collect();
    }

    let mut total_comps = num_cycles;
    for c in &mut comp {
        if *c == usize::MAX {
            *c = total_comps;
            total_comps += 1;
        }
    }
    let new_root = comp[root];

    let mut best_pair: Vec<Option<MsaEdge>> = vec![None; total_comps * total_comps];
    for (idx, e) in edges.iter().enumerate() {
        let cu = comp[e.u];
        let cv = comp[e.v];
        if cu == cv || cv == new_root {
            continue;
        }
        let weight = if in_cycle[e.v] {
            let cyc_idx = best_in_idx[e.v].expect("cycle node has an incoming edge");
            e.weight.sub(edges[cyc_idx].weight)
        } else {
            e.weight
        };
        let candidate = MsaEdge {
            u: cu,
            v: cv,
            weight,
            parent_idx: idx,
        };
        let slot = &mut best_pair[cu * total_comps + cv];
        match slot {
            None => *slot = Some(candidate),
            Some(prev) => {
                if candidate.weight.gt(prev.weight) {
                    *slot = Some(candidate);
                }
            }
        }
    }

    let new_edges: Vec<MsaEdge> = best_pair.into_iter().flatten().collect();
    let sub_chosen = msa(total_comps, new_root, &new_edges);

    let mut chosen_for_node: Vec<usize> = (0..num_nodes)
        .map(|v| {
            if v == root {
                usize::MAX
            } else {
                best_in_idx[v].expect("every non-root node has an incoming edge")
            }
        })
        .collect();
    for ne_idx in sub_chosen {
        let parent_idx = new_edges[ne_idx].parent_idx;
        let target_v = edges[parent_idx].v;
        chosen_for_node[target_v] = parent_idx;
    }

    (0..num_nodes)
        .filter(|&v| v != root)
        .map(|v| chosen_for_node[v])
        .collect()
}

fn argmax_slice(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate().skip(1) {
        let best_v = values[best];
        if v > best_v || (best_v.is_nan() && !v.is_nan()) {
            best = i;
        }
    }
    best
}

fn argmax_non_root_deprel(rel_logits: &[f32; 37]) -> usize {
    let mut best = 0usize;
    for (c, &v) in rel_logits.iter().enumerate().skip(1) {
        if c == ROOT_DEPREL_INDEX {
            continue;
        }
        let best_v = rel_logits[best];
        if v > best_v || (best_v.is_nan() && !v.is_nan()) {
            best = c;
        }
    }
    best
}

fn softmax_prob(logits: &[f32], chosen: usize) -> f64 {
    if logits.is_empty() || chosen >= logits.len() {
        return 0.0;
    }
    let max_v = logits
        .iter()
        .map(|&x| clean_logit(x))
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits.iter().map(|&x| (clean_logit(x) - max_v).exp()).sum();
    if sum <= 0.0 || !sum.is_finite() {
        0.0
    } else {
        ((clean_logit(logits[chosen]) - max_v).exp() / sum).clamp(0.0, 1.0)
    }
}

fn is_unsupported_sentence(doc: &Document, span: Span) -> bool {
    doc.unknown.iter().any(|u| {
        u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage && u.span == span
    })
}

/// Emits `no_model` `unknown` entries (`layer: "syntax"`) for every supported sentence in `doc`.
pub fn mark_no_model(doc: &mut Document) {
    let spans: Vec<Span> = doc
        .segmentation
        .sentences
        .iter()
        .map(|s| s.span)
        .filter(|&span| !is_unsupported_sentence(doc, span))
        .collect();
    for span in spans {
        doc.unknown.push(no_model_unknown("syntax", span));
    }
    sort_unknown(&mut doc.unknown);
}

fn char_to_byte_table(text: &str) -> Vec<usize> {
    let mut bytes = Vec::with_capacity(text.chars().count() + 1);
    for (byte, _) in text.char_indices() {
        bytes.push(byte);
    }
    bytes.push(text.len());
    bytes
}

struct PreparedWindow {
    token_indices: Vec<usize>,
    subword_spans: Vec<Vec<usize>>,
    unmapped_token_indices: Vec<usize>,
    ids: Vec<i64>,
}

struct PreparedSentence {
    sent_idx: usize,
    windows: Vec<PreparedWindow>,
}

enum WindowFit {
    Fits(PreparedWindow),
    TooLong,
    Unmappable,
}

fn partition_tokens_by_subwords(
    doc: &Document,
    token_indices: &[usize],
    base_start_char: usize,
    subwords: &Subwords,
) -> (Vec<usize>, Vec<Vec<usize>>, Vec<usize>) {
    let mut mapped_tokens = Vec::with_capacity(token_indices.len());
    let mut subword_spans = Vec::with_capacity(token_indices.len());
    let mut unmapped_tokens = Vec::new();
    for &tok_idx in token_indices {
        let tok_span = doc.tokens[tok_idx].span;
        // Subword 0 is reserved for [CLS] = UD root (`s_0 = 0`), so tokens map into `1..len`.
        let overlapping: Vec<usize> = subwords
            .spans
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(s_idx, sub)| {
                if sub.start_char >= sub.end_char {
                    return None;
                }
                let doc_start = base_start_char + sub.start_char;
                let doc_end = base_start_char + sub.end_char;
                (doc_start < tok_span.end_char && tok_span.start_char < doc_end).then_some(s_idx)
            })
            .collect();
        if overlapping.is_empty() {
            unmapped_tokens.push(tok_idx);
        } else {
            mapped_tokens.push(tok_idx);
            subword_spans.push(overlapping);
        }
    }
    (mapped_tokens, subword_spans, unmapped_tokens)
}

fn clause_boundaries_for_sentence(
    doc: &Document,
    token_indices: &[usize],
    clause_spans: &[Span],
) -> Vec<usize> {
    let m = token_indices.len();
    if clause_spans.len() <= 1 {
        return vec![0, m];
    }
    let mut boundaries = Vec::with_capacity(clause_spans.len() + 1);
    boundaries.push(0);
    let mut tok_pos = 0usize;
    for next_clause in &clause_spans[1..] {
        while tok_pos < m
            && doc.tokens[token_indices[tok_pos]].span.start_char < next_clause.start_char
        {
            tok_pos += 1;
        }
        if let Some(&last) = boundaries.last()
            && tok_pos > last
            && tok_pos < m
        {
            boundaries.push(tok_pos);
        }
    }
    if boundaries.last().copied().unwrap_or(0) < m {
        boundaries.push(m);
    }
    boundaries
}

struct WindowCtx<'a, F> {
    doc: &'a Document,
    text: &'a str,
    char_to_byte: &'a [usize],
    budget_max_len: usize,
    hard_max_len: usize,
    encode: &'a F,
}

impl<F> WindowCtx<'_, F>
where
    F: Fn(&str) -> Result<Subwords, ModelError>,
{
    fn try_encode(&self, slice: &[usize], limit_len: usize) -> Result<WindowFit, ModelError> {
        let first_tok = slice[0];
        let last_tok = *slice.last().expect("window slice is non-empty");
        let win_start_char = self.doc.tokens[first_tok].span.start_char;
        let win_end_char = self.doc.tokens[last_tok].span.end_char;
        let Some(&byte_start) = self.char_to_byte.get(win_start_char) else {
            return Err(ModelError::Run(format!(
                "window start_char {win_start_char} outside text"
            )));
        };
        let Some(&byte_end) = self.char_to_byte.get(win_end_char) else {
            return Err(ModelError::Run(format!(
                "window end_char {win_end_char} outside text"
            )));
        };
        if byte_start > byte_end {
            return Err(ModelError::Run(format!(
                "window span {win_start_char}..{win_end_char} inverted"
            )));
        }
        let win_sub = (self.encode)(&self.text[byte_start..byte_end])?;
        if win_sub.ids.len() != win_sub.spans.len() {
            return Err(ModelError::Run(
                "subword ids and spans differ in length".to_owned(),
            ));
        }
        if win_sub.ids.is_empty() {
            return Ok(WindowFit::Unmappable);
        }
        if win_sub.ids.len() > limit_len {
            return Ok(WindowFit::TooLong);
        }
        let (token_indices, subword_spans, unmapped_token_indices) =
            partition_tokens_by_subwords(self.doc, slice, win_start_char, &win_sub);
        Ok(WindowFit::Fits(PreparedWindow {
            token_indices,
            subword_spans,
            unmapped_token_indices,
            ids: win_sub.ids,
        }))
    }

    /// Greedily packs as many consecutive `boundaries` segments starting at `start_b` as fit in
    /// `limit_len` subwords (`<= limit_len - 2` content subwords plus `[CLS]` and `[SEP]`).
    fn extend_boundary(
        &self,
        token_indices: &[usize],
        boundaries: &[usize],
        start_b: usize,
        limit_len: usize,
    ) -> Result<Option<(usize, PreparedWindow)>, ModelError> {
        let limit = boundaries.len() - 1;
        let first_slice = &token_indices[boundaries[start_b]..boundaries[start_b + 1]];
        let WindowFit::Fits(mut best_win) = self.try_encode(first_slice, limit_len)? else {
            return Ok(None);
        };
        let mut lo = start_b + 1;
        if lo == limit {
            return Ok(Some((lo, best_win)));
        }

        // Exponential (galloping) search followed by binary search to stay O(log K) on mega-clauses.
        let mut step = 1usize;
        let mut hi = limit;
        let mut found_upper = false;
        while lo < limit {
            let probe = (lo + step).min(limit);
            let slice = &token_indices[boundaries[start_b]..boundaries[probe]];
            match self.try_encode(slice, limit_len)? {
                WindowFit::Fits(win) => {
                    lo = probe;
                    best_win = win;
                    if lo == limit {
                        return Ok(Some((lo, best_win)));
                    }
                    step = step.saturating_mul(2);
                }
                WindowFit::TooLong | WindowFit::Unmappable => {
                    hi = probe;
                    found_upper = true;
                    break;
                }
            }
        }

        if found_upper {
            while lo + 1 < hi {
                let mid = lo + (hi - lo) / 2;
                let slice = &token_indices[boundaries[start_b]..boundaries[mid]];
                match self.try_encode(slice, limit_len)? {
                    WindowFit::Fits(win) => {
                        lo = mid;
                        best_win = win;
                    }
                    WindowFit::TooLong | WindowFit::Unmappable => {
                        hi = mid;
                    }
                }
            }
        }

        Ok(Some((lo, best_win)))
    }

    /// Partitions a sentence into contiguous clause-aligned windows under `budget_max_len`
    /// (allowing a single clause up to `hard_max_len` intact before falling back to token
    /// boundaries inside any mega-clause `> hard_max_len`). Returns `Ok(None)` only when a single
    /// token alone exceeds `hard_max_len - 2` subwords or cannot map to any subword.
    fn partition(
        &self,
        token_indices: &[usize],
        clause_spans: &[Span],
    ) -> Result<Option<Vec<PreparedWindow>>, ModelError> {
        if self.hard_max_len < 3 {
            return Ok(None);
        }
        let clause_bounds = clause_boundaries_for_sentence(self.doc, token_indices, clause_spans);
        let mut windows = Vec::new();
        let mut b_idx = 0usize;
        while b_idx + 1 < clause_bounds.len() {
            if let Some((next_b, win)) =
                self.extend_boundary(token_indices, &clause_bounds, b_idx, self.budget_max_len)?
            {
                windows.push(win);
                b_idx = next_b;
            } else if self.budget_max_len < self.hard_max_len
                && let WindowFit::Fits(win) = self.try_encode(
                    &token_indices[clause_bounds[b_idx]..clause_bounds[b_idx + 1]],
                    self.hard_max_len,
                )?
            {
                windows.push(win);
                b_idx += 1;
            } else {
                // Single mega-clause exceeds `hard_max_len`: split cleanly at token boundaries.
                let tok_bounds: Vec<usize> =
                    (clause_bounds[b_idx]..=clause_bounds[b_idx + 1]).collect();
                let mut t_idx = 0usize;
                while t_idx + 1 < tok_bounds.len() {
                    let Some((next_t, win)) =
                        self.extend_boundary(token_indices, &tok_bounds, t_idx, self.hard_max_len)?
                    else {
                        return Ok(None);
                    };
                    windows.push(win);
                    t_idx = next_t;
                }
                b_idx += 1;
            }
        }
        Ok(Some(windows))
    }
}

fn is_zh_numeral_char(ch: char) -> bool {
    "〇零一二三四五六七八九十百千萬万億亿兩两".contains(ch)
}

/// Returns `true` when `tok_str` is a multi-character Chinese numeral-initial lexical phrase or
/// idiom (`一會兒`, `一時間`, `一清二楚`, `五顏六色`, `百合花`, `一路上`) rather than a UD `NUM`
/// numeral-classifier/quantity expression (`一個`, `兩株`, `三萬個`, `兩小時`, `百分之十`).
fn should_mean_pool_upos(doc: &Document, tok_span: Span, tok_str: &str) -> bool {
    if tok_str.chars().count() < 3 {
        return false;
    }
    if !tok_str.chars().next().is_some_and(is_zh_numeral_char) {
        return false;
    }
    for prefix in ["百分之", "千分之", "萬分之", "万分之"] {
        if tok_str.starts_with(prefix) {
            return false;
        }
    }
    if doc.entities.iter().any(|e| {
        e.span.start_char <= tok_span.start_char
            && tok_span.end_char <= e.span.end_char
            && matches!(e.kind, EntityKind::Quantity | EntityKind::Number)
    }) {
        return false;
    }
    let rest = tok_str.trim_start_matches(is_zh_numeral_char);
    let rest = rest
        .strip_prefix('個')
        .or_else(|| rest.strip_prefix('个'))
        .or_else(|| rest.strip_prefix('整'))
        .unwrap_or(rest);
    if rest.chars().count() < 2 {
        return false;
    }
    !matches!(
        rest,
        "小時"
            | "小时"
            | "分鐘"
            | "分钟"
            | "個月"
            | "个月"
            | "星期"
            | "禮拜"
            | "礼拜"
            | "時辰"
            | "时辰"
            | "刻鐘"
            | "刻钟"
            | "盞茶"
            | "盏茶"
            | "炷香"
            | "柱香"
    )
}

fn mean_upos_logits(
    doc: &Document,
    data: &[f32],
    cols: usize,
    subword_span: &[usize],
    tok_span: Span,
    tok_str: &str,
) -> [f32; 17] {
    let mut out = [0.0f32; 17];
    let first = subword_span[0];
    let first_row = &data[first * cols..first * cols + 17];
    if subword_span.len() == 1 || !should_mean_pool_upos(doc, tok_span, tok_str) {
        out.copy_from_slice(first_row);
        return out;
    }
    let mut sums = [0.0f64; 17];
    for &s_idx in subword_span {
        let row = &data[s_idx * cols..s_idx * cols + 17];
        for (slot, &val) in sums.iter_mut().zip(row) {
            *slot += clean_logit(val);
        }
    }
    let inv_n = 1.0 / (subword_span.len() as f64);
    for (dst, &sum) in out.iter_mut().zip(&sums) {
        *dst = (sum * inv_n) as f32;
    }
    out
}

fn verify_sentence_tree_invariants(emitted: &[SyntaxToken]) {
    if emitted.is_empty() {
        return;
    }
    let roots = emitted.iter().filter(|s| s.head.is_none()).count();
    assert_eq!(
        roots, 1,
        "non-empty sentence syntax must have exactly 1 root, got {roots}"
    );
    for s in emitted {
        if let Some(h) = s.head {
            assert_ne!(h, s.token, "self-loop in sentence syntax at token {h}");
            assert!(
                emitted.iter().any(|p| p.token == h),
                "orphan head {h} referenced by token {}",
                s.token
            );
        } else {
            assert_eq!(s.deprel, "root");
        }
    }
    let max_hops = emitted.len();
    for s in emitted {
        let mut cur = s;
        let mut hops = 0usize;
        while let Some(h) = cur.head {
            hops += 1;
            assert!(
                hops <= max_hops,
                "cycle detected in sentence syntax starting at token {}",
                s.token
            );
            cur = emitted
                .iter()
                .find(|p| p.token == h)
                .expect("head checked present");
        }
    }
}

/// Runs Layer 4 syntax decoding and contextual UPOS upgrade on an existing Layer 1–3/7 [`Document`]
/// using `model` and a sentence-level subword encoder `encode`.
pub fn annotate_with_encoder<F>(
    doc: &mut Document,
    text: &str,
    model: &dyn Model,
    encode: F,
) -> Result<(), ModelError>
where
    F: Fn(&str) -> Result<Subwords, ModelError>,
{
    let char_to_byte = char_to_byte_table(text);
    let max_len = model.info().max_len;
    let expected_cols = 17 + max_len + 4 * DEPRELS.len();

    let mut tokens_by_sentence: Vec<Vec<usize>> =
        vec![Vec::new(); doc.segmentation.sentences.len()];
    for (tok_idx, tok) in doc.tokens.iter().enumerate() {
        if let Some(bucket) = tokens_by_sentence.get_mut(tok.sentence) {
            bucket.push(tok_idx);
        }
    }
    let mut clauses_by_sentence: Vec<Vec<Span>> =
        vec![Vec::new(); doc.segmentation.sentences.len()];
    for clause in &doc.segmentation.clauses {
        if let Some(bucket) = clauses_by_sentence.get_mut(clause.sentence) {
            bucket.push(clause.span);
        }
    }

    let mut prepared = Vec::new();
    for (sent_idx, sentence) in doc.segmentation.sentences.iter().enumerate() {
        if is_unsupported_sentence(doc, sentence.span) {
            continue;
        }
        let token_indices = std::mem::take(&mut tokens_by_sentence[sent_idx]);
        let clause_spans = std::mem::take(&mut clauses_by_sentence[sent_idx]);
        if token_indices.is_empty() {
            continue;
        }
        let Some(&byte_start) = char_to_byte.get(sentence.span.start_char) else {
            return Err(ModelError::Run(format!(
                "sentence start_char {} outside text",
                sentence.span.start_char
            )));
        };
        let Some(&byte_end) = char_to_byte.get(sentence.span.end_char) else {
            return Err(ModelError::Run(format!(
                "sentence end_char {} outside text",
                sentence.span.end_char
            )));
        };
        if byte_start > byte_end {
            return Err(ModelError::Run(format!(
                "sentence span {}..{} inverted",
                sentence.span.start_char, sentence.span.end_char
            )));
        }
        let sentence_str = &text[byte_start..byte_end];
        let subwords = encode(sentence_str)?;
        if subwords.ids.len() != subwords.spans.len() {
            return Err(ModelError::Run(
                "subword ids and spans differ in length".to_owned(),
            ));
        }
        if subwords.ids.is_empty() {
            doc.unknown.push(Unknown {
                layer: "syntax".to_owned(),
                span: sentence.span,
                reason: Reason::NotApplicable,
            });
            continue;
        }
        let needs_hard_window = subwords.ids.len() > max_len;
        let needs_soft_window = !needs_hard_window
            && clause_spans.len() >= 4
            && token_indices.len() > 64
            && max_len >= 3;
        if needs_hard_window || needs_soft_window {
            let budget_max_len = if needs_soft_window {
                SOFT_WINDOW_MAX_LEN.min(max_len)
            } else {
                max_len
            };
            let ctx = WindowCtx {
                doc,
                text,
                char_to_byte: &char_to_byte,
                budget_max_len,
                hard_max_len: max_len,
                encode: &encode,
            };
            match ctx.partition(&token_indices, &clause_spans)? {
                Some(windows) if windows.iter().any(|w| !w.token_indices.is_empty()) => {
                    prepared.push(PreparedSentence { sent_idx, windows });
                }
                _ => doc.unknown.push(Unknown {
                    layer: "syntax".to_owned(),
                    span: sentence.span,
                    reason: Reason::NotApplicable,
                }),
            }
            continue;
        }

        let (mapped_tokens, subword_spans, unmapped_token_indices) =
            partition_tokens_by_subwords(doc, &token_indices, sentence.span.start_char, &subwords);
        if mapped_tokens.is_empty() {
            doc.unknown.push(Unknown {
                layer: "syntax".to_owned(),
                span: sentence.span,
                reason: Reason::NotApplicable,
            });
            continue;
        }

        prepared.push(PreparedSentence {
            sent_idx,
            windows: vec![PreparedWindow {
                token_indices: mapped_tokens,
                subword_spans,
                unmapped_token_indices,
                ids: subwords.ids,
            }],
        });
    }

    if prepared.is_empty() {
        sort_unknown(&mut doc.unknown);
        return Ok(());
    }

    // Run sentence by sentence (identical to batch in both backends, and allows isolating TooLong).
    for item in &prepared {
        for win in &item.windows {
            for &u_tok in &win.unmapped_token_indices {
                doc.unknown.push(Unknown {
                    layer: "syntax".to_owned(),
                    span: doc.tokens[u_tok].span,
                    reason: Reason::NotApplicable,
                });
            }
        }
        let non_empty_windows: Vec<&PreparedWindow> = item
            .windows
            .iter()
            .filter(|w| !w.token_indices.is_empty())
            .collect();
        if non_empty_windows.is_empty() {
            continue;
        }

        let sentence_span = doc.segmentation.sentences[item.sent_idx].span;
        let mut window_tensors = Vec::with_capacity(non_empty_windows.len());
        let mut too_long = false;
        for win in &non_empty_windows {
            let output = match model.run(&Batch {
                ids: vec![win.ids.clone()],
            }) {
                Ok(out) => out,
                Err(ModelError::TooLong { .. }) => {
                    too_long = true;
                    break;
                }
                Err(err) => return Err(err),
            };
            let Some(tensor) = output.logits.into_iter().next() else {
                return Err(ModelError::Run(
                    "model returned no output tensor".to_owned(),
                ));
            };
            let seq_len = win.ids.len();
            if tensor.shape != [seq_len, expected_cols]
                || tensor.data.len() != seq_len * expected_cols
            {
                return Err(ModelError::Run(format!(
                    "unexpected syntax tensor shape {:?}, expected [{seq_len}, {expected_cols}]",
                    tensor.shape
                )));
            }
            window_tensors.push((*win, tensor));
        }
        if too_long {
            doc.unknown.push(Unknown {
                layer: "syntax".to_owned(),
                span: sentence_span,
                reason: Reason::NotApplicable,
            });
            continue;
        }

        // 1. Contextual UPOS upgrade via multi-subword arithmetic mean pooling.
        let mut undecided_before = Vec::new();
        for &(win, ref tensor) in &window_tensors {
            for (&tok_idx, s_span) in win.token_indices.iter().zip(&win.subword_spans) {
                if doc.tokens[tok_idx].upos.is_none() {
                    undecided_before.push(tok_idx);
                    let tok_span = doc.tokens[tok_idx].span;
                    let tok_str = char_to_byte
                        .get(tok_span.start_char)
                        .zip(char_to_byte.get(tok_span.end_char))
                        .filter(|&(bs, be)| bs <= be)
                        .map_or("", |(&bs, &be)| &text[bs..be]);
                    let upos_logits = mean_upos_logits(
                        doc,
                        &tensor.data,
                        expected_cols,
                        s_span,
                        tok_span,
                        tok_str,
                    );
                    let best_upos = argmax_slice(&upos_logits);
                    let upos_prob = softmax_prob(&upos_logits, best_upos);
                    if upos_prob >= UPOS_THRESHOLD {
                        doc.tokens[tok_idx].upos = Some(UPOS_TAGS[best_upos]);
                        doc.unknown
                            .retain(|u| !(u.layer == "pos" && u.span == tok_span));
                    } else if let Some(existing) = doc
                        .unknown
                        .iter_mut()
                        .find(|u| u.layer == "pos" && u.span == tok_span)
                    {
                        existing.reason = Reason::LowConfidence;
                    } else {
                        doc.unknown.push(Unknown {
                            layer: "pos".to_owned(),
                            span: tok_span,
                            reason: Reason::LowConfidence,
                        });
                    }
                }
            }
        }

        // Deterministic disambiguation for `什麼`/`什么` when Layer 2 left `upos == None` and the
        // model predicted `PART` or fell below `UPOS_THRESHOLD`.
        for tok_idx in undecided_before {
            if !matches!(doc.tokens[tok_idx].upos, None | Some(Upos::Part)) {
                continue;
            }
            let tok_span = doc.tokens[tok_idx].span;
            let Some(&b_start) = char_to_byte.get(tok_span.start_char) else {
                continue;
            };
            let Some(&b_end) = char_to_byte.get(tok_span.end_char) else {
                continue;
            };
            if b_start <= b_end && matches!(&text[b_start..b_end], "什麼" | "什么") {
                let followed_by_nominal = doc.tokens.get(tok_idx + 1).is_some_and(|next| {
                    next.sentence == item.sent_idx
                        && matches!(next.upos, Some(Upos::Noun | Upos::Propn))
                });
                doc.tokens[tok_idx].upos = Some(if followed_by_nominal {
                    Upos::Det
                } else {
                    Upos::Pron
                });
                doc.unknown
                    .retain(|u| !(u.layer == "pos" && u.span == tok_span));
            }
        }

        // 2. Decode each window with Chu-Liu/Edmonds and stitch window `k >= 1` local roots to
        //    window `0`'s root with `deprel = "parataxis"`.
        let total_tokens: usize = window_tensors
            .iter()
            .map(|(w, _)| w.token_indices.len())
            .sum();
        let mut sent_tokens = Vec::with_capacity(total_tokens);
        let mut head_pos: Vec<Option<usize>> = Vec::with_capacity(total_tokens);
        let mut deprel_indices = Vec::with_capacity(total_tokens);
        let mut scores = Vec::with_capacity(total_tokens);
        let mut ancestor_ok = Vec::with_capacity(total_tokens);
        let mut global_root_pos: Option<usize> = None;
        let rel_offset = 17 + max_len;
        let ancestor_cap = f64::from(ANCESTOR_MIN_MILLI) / 1000.0;

        for (win, tensor) in &window_tensors {
            let row = |s: usize| &tensor.data[s * expected_cols..(s + 1) * expected_cols];
            let m = win.token_indices.len();
            let base_pos = sent_tokens.len();
            let min_ancestor_arc = (1.05 / ((m + 1) as f64)).min(ancestor_cap);

            let mut arc_scores = Vec::with_capacity(m);
            for s_span in &win.subword_spans {
                let s_u = s_span[0];
                let r_u = row(s_u);
                let mut scores_u = Vec::with_capacity(m + 1);
                scores_u.push(r_u[17]); // v = 0 -> s_0 = 0 ([CLS])
                for s_v_span in &win.subword_spans {
                    scores_u.push(r_u[17 + s_v_span[0]]);
                }
                arc_scores.push(scores_u);
            }

            let heads = chu_liu_edmonds(&arc_scores);
            for (u_zero, (&tok_idx, s_span)) in
                win.token_indices.iter().zip(&win.subword_spans).enumerate()
            {
                let s_u = s_span[0];
                let v = heads[u_zero];
                let p_arc = softmax_prob(&arc_scores[u_zero], v);
                let best_v = argmax_slice(&arc_scores[u_zero]);
                let p_best_arc = softmax_prob(&arc_scores[u_zero], best_v);
                let arc_score = Score::from_confidence(p_arc);
                ancestor_ok.push(p_best_arc >= min_ancestor_arc);

                if v == 0 {
                    if let Some(r0) = global_root_pos {
                        head_pos.push(Some(r0));
                        deprel_indices.push(PARATAXIS_DEPREL_INDEX);
                    } else {
                        global_root_pos = Some(base_pos + u_zero);
                        head_pos.push(None);
                        deprel_indices.push(ROOT_DEPREL_INDEX);
                    }
                    scores.push(arc_score);
                } else {
                    let s_v = win.subword_spans[v - 1][0];
                    let r_u = row(s_u);
                    let r_v = row(s_v);
                    let r_dep = &r_u[rel_offset..rel_offset + 37];
                    let r_head = &r_v[rel_offset + 37..rel_offset + 74];
                    let r_bi_dep = &r_u[rel_offset + 74..rel_offset + 111];
                    let r_bi_head = &r_v[rel_offset + 111..rel_offset + 148];

                    let mut rel_logits = [0.0f32; 37];
                    for (c, slot) in rel_logits.iter_mut().enumerate() {
                        *slot = r_dep[c] + r_head[c] + r_bi_dep[c] * r_bi_head[c];
                    }
                    let best_c = argmax_non_root_deprel(&rel_logits);

                    let mut non_root = [0.0f32; 36];
                    let mut dst = 0;
                    for (c, &val) in rel_logits.iter().enumerate() {
                        if c != ROOT_DEPREL_INDEX {
                            non_root[dst] = val;
                            dst += 1;
                        }
                    }
                    let chosen_non_root = if best_c < ROOT_DEPREL_INDEX {
                        best_c
                    } else {
                        best_c - 1
                    };
                    let p_rel = softmax_prob(&non_root, chosen_non_root);
                    head_pos.push(Some(base_pos + (v - 1)));
                    deprel_indices.push(best_c);
                    scores.push(Score::from_confidence(p_arc * p_rel));
                }
                sent_tokens.push(tok_idx);
            }
        }

        // 3. Tree-consistent root & ancestor retention (0 zero-root sentences, 0 orphan arcs).
        let n = sent_tokens.len();
        let r_pos = global_root_pos.expect("non-empty sentence has a single global root");
        let mut retained = vec![false; n];
        for (u, &sc) in scores.iter().enumerate() {
            if sc.milli() < SYNTAX_THRESHOLD_MILLI {
                continue;
            }
            let mut path = Vec::new();
            let mut cur = u;
            let mut connected = false;
            for _ in 0..=n {
                path.push(cur);
                match head_pos[cur] {
                    None => {
                        connected = cur == r_pos;
                        break;
                    }
                    Some(p) => {
                        if p != r_pos && !ancestor_ok[p] {
                            break;
                        }
                        cur = p;
                    }
                }
            }
            if connected {
                for node in path {
                    retained[node] = true;
                }
            }
        }

        let syntax_start = doc.syntax.len();
        for u in 0..n {
            let tok_idx = sent_tokens[u];
            let tok_span = doc.tokens[tok_idx].span;
            if retained[u] {
                let head = head_pos[u].map(|p| sent_tokens[p]);
                doc.syntax.push(SyntaxToken {
                    token: tok_idx,
                    sentence: item.sent_idx,
                    span: tok_span,
                    head,
                    deprel: DEPRELS[deprel_indices[u]].to_owned(),
                    score: scores[u],
                });
            } else {
                doc.unknown.push(Unknown {
                    layer: "syntax".to_owned(),
                    span: tok_span,
                    reason: Reason::LowConfidence,
                });
            }
        }
        verify_sentence_tree_invariants(&doc.syntax[syntax_start..]);
    }

    sort_unknown(&mut doc.unknown);
    Ok(())
}

/// Runs Layer 4 syntax decoding on `doc` with an already-loaded `model` and `tokenizer`.
#[cfg(any(feature = "onnx", feature = "coreml"))]
pub fn annotate(
    doc: &mut Document,
    text: &str,
    model: &dyn Model,
    tokenizer: &SubwordTokenizer,
) -> Result<(), ModelError> {
    annotate_with_encoder(doc, text, model, |s| tokenizer.encode(s))
}

/// Analyzes `text` with Layers 1–3/7 and runs Layer 4 using an already-loaded `model` and
/// `tokenizer`.
#[cfg(any(feature = "onnx", feature = "coreml"))]
pub fn analyze_document_with_loaded_model(
    text: &str,
    options: &Options,
    model: &dyn Model,
    tokenizer: &SubwordTokenizer,
) -> Result<Document, ModelError> {
    let mut doc = analyze_document(text, options);
    annotate(&mut doc, text, model, tokenizer)?;
    Ok(doc)
}

/// Analyzes `text` with Layers 1–3/7 and loads [`SYNTAX_MODEL_ID`] (`doc-syntax-v1`) from
/// `models_dir` on `backend` to populate Layer 4 (`syntax`) and upgrade undecided Layer-2 `upos`.
///
/// When `backend` is not compiled in, or the model or its files are absent (`NoBackend`,
/// `UnknownModel`, `Missing`), `Document.syntax` stays empty and each supported sentence gets a
/// `no_model` `unknown` entry for `"syntax"`. Integrity and runtime errors (`HashMismatch`,
/// `Manifest`, `Load`, `Run`) are returned to the caller.
pub fn analyze_document_with_model(
    text: &str,
    options: &Options,
    models_dir: &std::path::Path,
    backend: crate::doc::model::Backend,
) -> Result<Document, ModelError> {
    let mut doc = analyze_document(text, options);
    if !crate::doc::model::compiled(backend) {
        let _ = models_dir;
        mark_no_model(&mut doc);
        return Ok(doc);
    }
    #[cfg(any(feature = "onnx", feature = "coreml"))]
    {
        run_verified_syntax(doc, text, models_dir, backend)
    }
    #[cfg(not(any(feature = "onnx", feature = "coreml")))]
    {
        let _ = models_dir;
        mark_no_model(&mut doc);
        Ok(doc)
    }
}

#[cfg(any(feature = "onnx", feature = "coreml"))]
fn run_verified_syntax(
    mut doc: Document,
    text: &str,
    models_dir: &Path,
    backend: Backend,
) -> Result<Document, ModelError> {
    let verified = match crate::doc::model::verify(models_dir, SYNTAX_MODEL_ID) {
        Ok(v) => v,
        Err(err) if err.into_reason() == Some(Reason::NoModel) => {
            mark_no_model(&mut doc);
            return Ok(doc);
        }
        Err(err) => return Err(err),
    };
    let loaded = crate::doc::model::load_verified(&verified, backend)?;
    let tokenizer = SubwordTokenizer::from_verified(&verified)?;
    annotate(&mut doc, text, &*loaded, &tokenizer)?;
    Ok(doc)
}
