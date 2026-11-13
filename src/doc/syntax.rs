//! Layer 4 of `nlp.doc.v1`: Universal Dependencies v2 syntax trees and contextual UPOS upgrade.
//! See `docs/doc-v1.md`, section "Syntax (layer 4)".

use crate::Span;
#[cfg(any(feature = "onnx", feature = "coreml"))]
use crate::doc::model::{Backend, SubwordTokenizer};
use crate::doc::model::{Batch, Model, ModelError, Subwords, no_model_unknown};
use crate::doc::{
    Document, EntityKind, Lang, Options, Reason, Unknown, Upos, analyze_document, sort_unknown,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::collections::HashSet;
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

/// Index of `"dep"` inside [`DEPRELS`].
const DEP_DEPREL_INDEX: usize = 14;

/// Index of `"parataxis"` inside [`DEPRELS`].
const PARATAXIS_DEPREL_INDEX: usize = 31;

/// Index of `"root"` inside [`DEPRELS`].
pub const ROOT_DEPREL_INDEX: usize = 34;

/// Calibrated logit penalty subtracted from `0 -> tok` (`root`) for non-predicative tokens,
/// prenominal `…的 + NP` relative predicates, and non-final subordinate clauses in multi-clause
/// Chinese sentences when at least one viable main-clause predicate exists.
const ROOT_PENALTY_LOGIT: f32 = 2.5;

/// Minimum local head probability (`0.22`) to re-attach a low-confidence (`< 0.35`) cross-clause
/// closure token onto an emitted `VERB`/`ADJ` predicate within its own clause.
const LOCAL_REATTACH_MIN_PROB: f64 = 0.22;

/// Scraper anti-censorship radical/stroke infixes stripped from temporary subword/UPOS strings
/// when `options.normalize_traditional` is enabled (without altering `Token.span`).
const SCRAPER_INFIX_CHARS: [char; 5] = ['丅', '丵', '灬', '堊', '冇'];

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
    if doc.unknown.iter().any(|u| {
        u.layer == "segmentation" && u.reason == Reason::UnsupportedLanguage && u.span == span
    }) {
        return true;
    }
    let Some((sent_idx, sent)) = doc
        .segmentation
        .sentences
        .iter()
        .enumerate()
        .find(|(_, s)| s.span == span)
    else {
        return false;
    };
    if matches!(sent.lang, Lang::Ja | Lang::Ko) {
        return true;
    }
    doc.tokens
        .iter()
        .any(|t| t.sentence == sent_idx && matches!(t.lang, Lang::Ja | Lang::Ko))
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
    clause_spans: Vec<Span>,
    windows: Vec<PreparedWindow>,
}

enum WindowFit {
    Fits(PreparedWindow),
    TooLong,
    Unmappable,
}

fn is_scraper_infix_char(ch: char) -> bool {
    SCRAPER_INFIX_CHARS.contains(&ch)
}

fn is_clean_han(ch: char) -> bool {
    super::segment::is_han(ch) && !is_scraper_infix_char(ch)
}

fn clean_token_surface(tok_str: &str, normalize_traditional: bool) -> Cow<'_, str> {
    if !normalize_traditional || !tok_str.chars().any(is_scraper_infix_char) {
        return Cow::Borrowed(tok_str);
    }
    let chars: Vec<char> = tok_str.chars().collect();
    if chars.len() < 3 {
        return Cow::Borrowed(tok_str);
    }
    let mut out = String::with_capacity(tok_str.len());
    let mut stripped = false;
    for i in 0..chars.len() {
        if i > 0
            && i + 1 < chars.len()
            && is_scraper_infix_char(chars[i])
            && is_clean_han(chars[i - 1])
            && is_clean_han(chars[i + 1])
        {
            stripped = true;
            continue;
        }
        out.push(chars[i]);
    }
    if stripped {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(tok_str)
    }
}

/// Encodes `raw` (starting at document char `base_start_char`) with `encode`, stripping single
/// scraper anti-censorship infixes (`丅`, `丵`, `灬`, `堊`, `冇`) sandwiched between Han chars
/// inside multi-char tokens when `doc.options.normalize_traditional` is true, and remapping
/// returned subword spans back to `raw`-relative char offsets so `Token.span` alignment stays
/// exact.
fn encode_slice_clean<F>(
    doc: &Document,
    raw: &str,
    base_start_char: usize,
    token_indices: &[usize],
    encode: &F,
) -> Result<Subwords, ModelError>
where
    F: Fn(&str) -> Result<Subwords, ModelError>,
{
    if !doc.options.normalize_traditional || !raw.chars().any(is_scraper_infix_char) {
        return encode(raw);
    }
    let chars: Vec<char> = raw.chars().collect();
    let mut strip = vec![false; chars.len()];
    let mut any_strip = false;
    for &tok_idx in token_indices {
        let ts = doc.tokens[tok_idx].span;
        if ts.end_char.saturating_sub(ts.start_char) < 3 || ts.start_char < base_start_char {
            continue;
        }
        let rel_s = ts.start_char - base_start_char;
        let rel_e = (ts.end_char - base_start_char).min(chars.len());
        if rel_s + 2 >= rel_e {
            continue;
        }
        for k in (rel_s + 1)..(rel_e - 1) {
            if is_scraper_infix_char(chars[k])
                && is_clean_han(chars[k - 1])
                && is_clean_han(chars[k + 1])
            {
                strip[k] = true;
                any_strip = true;
            }
        }
    }
    if !any_strip {
        return encode(raw);
    }
    let mut clean_str = String::with_capacity(raw.len());
    let mut orig_pos = Vec::with_capacity(chars.len());
    for (k, &ch) in chars.iter().enumerate() {
        if !strip[k] {
            clean_str.push(ch);
            orig_pos.push(k);
        }
    }
    let mut sub = encode(&clean_str)?;
    for sp in &mut sub.spans {
        if sp.start_char < sp.end_char && sp.end_char <= orig_pos.len() {
            sp.start_char = orig_pos[sp.start_char];
            sp.end_char = orig_pos[sp.end_char - 1] + 1;
        } else if sp.start_char == sp.end_char && sp.start_char > 0 {
            let mapped = orig_pos.get(sp.start_char).copied().unwrap_or(chars.len());
            sp.start_char = mapped;
            sp.end_char = mapped;
        }
    }
    Ok(sub)
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
        let win_sub = encode_slice_clean(
            self.doc,
            &self.text[byte_start..byte_end],
            win_start_char,
            slice,
            self.encode,
        )?;
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
    if matches!(
        tok_str,
        "一五一十" | "三三兩兩" | "三三两两" | "千千萬萬" | "千千万万"
    ) {
        return true;
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

fn token_str<'a>(text: &'a str, char_to_byte: &[usize], span: Span) -> &'a str {
    char_to_byte
        .get(span.start_char)
        .zip(char_to_byte.get(span.end_char))
        .filter(|&(bs, be)| bs <= be)
        .map_or("", |(&bs, &be)| &text[bs..be])
}

fn clause_index_for_span(clause_spans: &[Span], span: Span) -> usize {
    if clause_spans.len() <= 1 {
        return 0;
    }
    clause_spans
        .partition_point(|cl| cl.start_char <= span.start_char)
        .saturating_sub(1)
}

fn clear_pos_unknown(doc: &mut Document, span: Span) {
    doc.unknown
        .retain(|u| !(u.layer == "pos" && u.span == span));
}

fn sentence_has_following_predicate(doc: &Document, tok_idx: usize, sent_idx: usize) -> bool {
    let mut k = tok_idx + 1;
    while let Some(t) = doc.tokens.get(k) {
        if t.sentence != sent_idx {
            break;
        }
        if matches!(t.upos, Some(Upos::Verb | Upos::Adj)) {
            return true;
        }
        k += 1;
    }
    false
}

fn clause_has_following_de_or_predicate(
    doc: &Document,
    tok_idx: usize,
    sent_idx: usize,
    char_to_byte: &[usize],
    text: &str,
) -> bool {
    let mut k = tok_idx + 1;
    while let Some(t) = doc.tokens.get(k) {
        if t.sentence != sent_idx || t.upos == Some(Upos::Punct) {
            break;
        }
        let s = token_str(text, char_to_byte, t.span);
        if k == tok_idx + 1 && matches!(s, "地" | "的") {
            return true;
        }
        if matches!(t.upos, Some(Upos::Verb | Upos::Adj)) {
            return true;
        }
        k += 1;
    }
    false
}

/// Resolves 4-character numeral-initial idioms (`二話不說`, `一動不動`, `四面八方`, `一般無二`,
/// `一五一十`, etc.) whose heterogeneous subwords otherwise dilute arithmetic-mean softmax below
/// [`UPOS_THRESHOLD`] (`0.50`) or leave all-numeral idioms on `Upos::Num`.
fn fallback_four_char_numeral_idiom(
    doc: &Document,
    tok_idx: usize,
    sent_idx: usize,
    char_to_byte: &[usize],
    text: &str,
    clean_tok: &str,
    upos_logits: &[f32; 17],
) -> Option<Upos> {
    match clean_tok {
        "四面八方" | "五湖四海" | "三頭六臂" | "三头六臂" | "千軍萬馬" | "千军万马" =>
        {
            return Some(Upos::Noun);
        }
        "一般無二" | "一般无二" | "一模一樣" | "一模一样" | "一乾二淨" | "一干二净"
        | "一干二凈" | "五顏六色" | "五颜六色" | "亂七八糟" | "乱七八糟" | "七零八落" =>
        {
            return Some(Upos::Adj);
        }
        "一五一十" | "三三兩兩" | "三三两两" | "一時半刻" | "一时半刻" => {
            return Some(Upos::Adv);
        }
        "二話不說" | "二话不说" => {
            return Some(
                if sentence_has_following_predicate(doc, tok_idx, sent_idx) {
                    Upos::Adv
                } else {
                    Upos::Verb
                },
            );
        }
        "一動不動" | "一动不动" | "一聲不吭" | "一声不吭" | "一言不發" | "一言不发" =>
        {
            return Some(
                if clause_has_following_de_or_predicate(doc, tok_idx, sent_idx, char_to_byte, text)
                {
                    Upos::Adv
                } else {
                    Upos::Verb
                },
            );
        }
        _ => {}
    }

    // Generic non-`NUM` open-class recovery for any other 4-char `should_mean_pool_upos` idiom.
    let mut non_num = [0.0f32; 16];
    let mut dst = 0usize;
    for (idx, &val) in upos_logits.iter().enumerate() {
        if UPOS_TAGS[idx] != Upos::Num {
            non_num[dst] = val;
            dst += 1;
        }
    }
    let open_candidates = [0usize, 2, 7, 15]; // Adj, Adv, Noun, Verb
    let mut best_open = open_candidates[0];
    for &cand in &open_candidates[1..] {
        if upos_logits[cand] > upos_logits[best_open] {
            best_open = cand;
        }
    }
    let non_num_idx = if best_open < 8 {
        best_open
    } else {
        best_open - 1
    };
    (softmax_prob(&non_num, non_num_idx) >= 0.35).then_some(UPOS_TAGS[best_open])
}

fn has_non_nominal_lexicon_shape(clean_tok: &str) -> bool {
    let chars: Vec<char> = clean_tok.chars().collect();
    if chars
        .iter()
        .any(|&c| matches!(c, '不' | '無' | '无' | '沒' | '没' | '勿' | '莫'))
    {
        return true;
    }
    if chars.len() >= 3
        && (chars[0] == chars[1] || chars[chars.len() - 2] == chars[chars.len() - 1])
    {
        return true;
    }
    for suffix in [
        "起來", "起来", "出來", "出来", "出去", "進去", "进去", "進來", "进来", "下去", "下來",
        "下来", "過來", "过来", "過去", "过去", "開來", "开来", "不得", "不了", "不住",
    ] {
        if clean_tok.ends_with(suffix) {
            return true;
        }
    }
    chars
        .last()
        .is_some_and(|&ch| matches!(ch, '然' | '地' | '著' | '着' | '了' | '過' | '过'))
}

fn is_disposal_coverb_context(
    doc: &Document,
    tok_idx: usize,
    sent_idx: usize,
    clause_spans: &[Span],
    char_to_byte: &[usize],
    text: &str,
) -> bool {
    let my_clause = clause_index_for_span(clause_spans, doc.tokens[tok_idx].span);
    if tok_idx > 0 {
        let prev = &doc.tokens[tok_idx - 1];
        if prev.sentence == sent_idx
            && prev.upos != Some(Upos::Punct)
            && clause_index_for_span(clause_spans, prev.span) == my_clause
        {
            let prev_str = clean_token_surface(
                token_str(text, char_to_byte, prev.span),
                doc.options.normalize_traditional,
            );
            if prev_str.chars().last().is_some_and(|ch| {
                matches!(
                    ch,
                    '一' | '兩'
                        | '两'
                        | '三'
                        | '四'
                        | '五'
                        | '六'
                        | '七'
                        | '八'
                        | '九'
                        | '十'
                        | '百'
                        | '千'
                        | '數'
                        | '数'
                        | '幾'
                        | '几'
                        | '眾'
                        | '众'
                        | '諸'
                        | '诸'
                        | '名'
                        | '員'
                        | '员'
                        | '位'
                        | '個'
                        | '个'
                        | '大'
                        | '老'
                        | '小'
                        | '神'
                        | '魔'
                        | '古'
                        | '悍'
                        | '猛'
                        | '勇'
                        | '主'
                        | '副'
                        | '守'
                        | '敵'
                        | '敌'
                        | '敗'
                        | '败'
                        | '女'
                )
            }) {
                return false;
            }
        }
    }

    let Some(next1) = doc.tokens.get(tok_idx + 1) else {
        return false;
    };
    if next1.sentence != sent_idx
        || next1.upos == Some(Upos::Punct)
        || clause_index_for_span(clause_spans, next1.span) != my_clause
    {
        return false;
    }
    let next1_str = clean_token_surface(
        token_str(text, char_to_byte, next1.span),
        doc.options.normalize_traditional,
    );
    for blocked in [
        "軍", "军", "領", "领", "帥", "帅", "要", "近", "來", "来", "會", "会", "士", "令", "官",
        "校", "門", "门", "就", "至", "及",
    ] {
        if next1_str.starts_with(blocked) {
            return false;
        }
    }
    if matches!(
        next1.upos,
        Some(Upos::Aux | Upos::Adp | Upos::Adv | Upos::Cconj | Upos::Sconj | Upos::Part)
    ) {
        return false;
    }

    // If `tok + 1` is `VERB`, only allow when it is a prenominal relative modifier (`將擊敗的對手擒住`).
    let mut min_np_offset = 1usize;
    if next1.upos == Some(Upos::Verb) {
        let mut rel_de_offset = None;
        for d in 2..=3 {
            let Some(cand) = doc.tokens.get(tok_idx + d) else {
                break;
            };
            if cand.sentence != sent_idx
                || cand.upos == Some(Upos::Punct)
                || clause_index_for_span(clause_spans, cand.span) != my_clause
            {
                break;
            }
            if token_str(text, char_to_byte, cand.span) == "的" {
                rel_de_offset = Some(d);
                break;
            }
        }
        let Some(de_off) = rel_de_offset else {
            return false;
        };
        min_np_offset = de_off + 1;
    }

    let mut found_np_at: Option<usize> = None;
    for d in 1..=6 {
        let Some(t) = doc.tokens.get(tok_idx + d) else {
            break;
        };
        if t.sentence != sent_idx
            || t.upos == Some(Upos::Punct)
            || clause_index_for_span(clause_spans, t.span) != my_clause
        {
            break;
        }
        if found_np_at.is_none() {
            if d >= min_np_offset
                && matches!(
                    t.upos,
                    Some(Upos::Noun | Upos::Propn | Upos::Pron | Upos::Det)
                )
            {
                found_np_at = Some(d);
            }
        } else if matches!(t.upos, Some(Upos::Verb | Upos::Adj)) {
            return true;
        }
    }
    false
}

fn detect_subordinate_non_final_clauses(
    doc: &Document,
    sent_tokens: &[usize],
    clause_spans: &[Span],
    char_to_byte: &[usize],
    text: &str,
) -> Vec<bool> {
    let num_clauses = clause_spans.len();
    let mut sub = vec![false; num_clauses];
    if num_clauses < 2 {
        return sub;
    }
    let mut clause_non_punct: Vec<Vec<usize>> = vec![Vec::new(); num_clauses];
    for &tok_idx in sent_tokens {
        if doc.tokens[tok_idx].upos == Some(Upos::Punct) {
            continue;
        }
        let c_idx = clause_index_for_span(clause_spans, doc.tokens[tok_idx].span);
        if let Some(bucket) = clause_non_punct.get_mut(c_idx) {
            bucket.push(tok_idx);
        }
    }

    const LEADING_SUBORDINATORS: &[&str] = &[
        "雖然", "虽然", "儘管", "尽管", "即使", "縱然", "纵然", "哪怕", "如果", "若是", "倘若",
        "要是", "假如", "一旦", "別看", "别看", "既然", "無論", "无论", "不管",
    ];
    const TRAILING_FRAMES: &[&str] = &["之際", "之际", "之時", "之时", "的時候", "的时候"];
    const CIRCUM_OPENERS: &[&str] = &["當", "当", "在", "自從", "自从"];
    const CIRCUM_CLOSERS: &[&str] = &[
        "時",
        "时",
        "後",
        "后",
        "之後",
        "之后",
        "之際",
        "之际",
        "之時",
        "之时",
        "的時候",
        "的时候",
    ];
    const KINSHIP_AFTER: &[&str] = &[
        "故人之後",
        "故人之后",
        "高人之後",
        "高人之后",
        "仙人之後",
        "仙人之后",
        "名門之後",
        "名门之后",
        "將門之後",
        "将门之后",
    ];

    for c_idx in 0..(num_clauses - 1) {
        let toks = &clause_non_punct[c_idx];
        let has_leading = toks.iter().take(3).any(|&ti| {
            let s = clean_token_surface(
                token_str(text, char_to_byte, doc.tokens[ti].span),
                doc.options.normalize_traditional,
            );
            LEADING_SUBORDINATORS.contains(&s.as_ref())
        });
        if has_leading {
            sub[c_idx] = true;
            continue;
        }

        let raw_cl = token_str(text, char_to_byte, clause_spans[c_idx]);
        let trimmed = raw_cl.trim_end_matches(|ch: char| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    '。' | '！'
                        | '？'
                        | '，'
                        | '；'
                        | '：'
                        | '」'
                        | '』'
                        | '”'
                        | '’'
                        | '）'
                        | ')'
                        | '"'
                        | '\''
                )
        });
        if trimmed.chars().count() >= 3 {
            if TRAILING_FRAMES.iter().any(|f| trimmed.ends_with(f)) {
                sub[c_idx] = true;
                continue;
            }
            if (trimmed.ends_with("之後") || trimmed.ends_with("之后"))
                && !KINSHIP_AFTER.iter().any(|k| trimmed.ends_with(k))
            {
                sub[c_idx] = true;
                continue;
            }
            let has_circum_open = toks.iter().take(2).any(|&ti| {
                let s = clean_token_surface(
                    token_str(text, char_to_byte, doc.tokens[ti].span),
                    doc.options.normalize_traditional,
                );
                CIRCUM_OPENERS.contains(&s.as_ref())
            });
            if has_circum_open && CIRCUM_CLOSERS.iter().any(|f| trimmed.ends_with(f)) {
                sub[c_idx] = true;
            }
        }
    }
    sub
}

fn is_non_predicative_token(doc: &Document, tok_idx: usize, clean_tok: &str) -> bool {
    matches!(
        doc.tokens[tok_idx].upos,
        Some(
            Upos::Adp
                | Upos::Cconj
                | Upos::Sconj
                | Upos::Part
                | Upos::Det
                | Upos::Punct
                | Upos::Adv
                | Upos::Num
        )
    ) || matches!(clean_tok, "終於" | "终于" | "除了" | "直到")
}

fn is_prenominal_rel_pred(
    doc: &Document,
    tok_idx: usize,
    sent_idx: usize,
    c_idx: usize,
    clause_spans: &[Span],
    char_to_byte: &[usize],
    text: &str,
) -> bool {
    if !matches!(doc.tokens[tok_idx].upos, Some(Upos::Verb | Upos::Adj)) {
        return false;
    }
    let Some(t1) = doc.tokens.get(tok_idx + 1) else {
        return false;
    };
    if t1.sentence != sent_idx
        || t1.upos == Some(Upos::Punct)
        || clause_index_for_span(clause_spans, t1.span) != c_idx
    {
        return false;
    }
    let s1 = token_str(text, char_to_byte, t1.span);
    let de_idx = if s1 == "的" {
        tok_idx + 1
    } else if matches!(s1, "了" | "過" | "过" | "著" | "着")
        && let Some(t2) = doc.tokens.get(tok_idx + 2)
        && t2.sentence == sent_idx
        && clause_index_for_span(clause_spans, t2.span) == c_idx
        && token_str(text, char_to_byte, t2.span) == "的"
    {
        tok_idx + 2
    } else {
        return false;
    };

    for d in 1..=3 {
        let Some(tn) = doc.tokens.get(de_idx + d) else {
            break;
        };
        if tn.sentence != sent_idx
            || tn.upos == Some(Upos::Punct)
            || clause_index_for_span(clause_spans, tn.span) != c_idx
        {
            break;
        }
        let sn = token_str(text, char_to_byte, tn.span);
        if matches!(sn, "是" | "有") || tn.upos == Some(Upos::Verb) {
            break;
        }
        if matches!(tn.upos, Some(Upos::Noun | Upos::Propn | Upos::Pron)) {
            return true;
        }
    }
    false
}

fn compute_biaffine_deprel_and_score(
    data: &[f32],
    expected_cols: usize,
    rel_offset: usize,
    s_u: usize,
    s_v: usize,
    p_arc: f64,
) -> (usize, Score) {
    let r_u = &data[s_u * expected_cols..(s_u + 1) * expected_cols];
    let r_v = &data[s_v * expected_cols..(s_v + 1) * expected_cols];
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
    (best_c, Score::from_confidence(p_arc * p_rel))
}

fn is_ancestor_of(
    candidate_anc: usize,
    mut node: usize,
    head_pos: &[Option<usize>],
    limit: usize,
) -> bool {
    for _ in 0..=limit {
        if node == candidate_anc {
            return true;
        }
        match head_pos[node] {
            Some(p) => node = p,
            None => return false,
        }
    }
    true
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
        let subwords = encode_slice_clean(
            doc,
            sentence_str,
            sentence.span.start_char,
            &token_indices,
            &encode,
        )?;
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
                    prepared.push(PreparedSentence {
                        sent_idx,
                        clause_spans,
                        windows,
                    });
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
            clause_spans,
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

    let lexicon_set: HashSet<String> = doc
        .options
        .lexicon
        .iter()
        .filter(|w| w.chars().count() >= 3)
        .map(|w| clean_token_surface(w, doc.options.normalize_traditional).into_owned())
        .collect();
    let names_set: HashSet<String> = if lexicon_set.is_empty() {
        HashSet::new()
    } else {
        doc.options
            .names
            .iter()
            .filter(|w| w.chars().count() >= 2)
            .map(|w| clean_token_surface(w, doc.options.normalize_traditional).into_owned())
            .collect()
    };

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
        let mut diluted_idioms = Vec::new();
        for &(win, ref tensor) in &window_tensors {
            for (&tok_idx, s_span) in win.token_indices.iter().zip(&win.subword_spans) {
                if doc.tokens[tok_idx].upos.is_none() {
                    undecided_before.push(tok_idx);
                    let tok_span = doc.tokens[tok_idx].span;
                    let raw_tok = token_str(text, &char_to_byte, tok_span);
                    let clean_tok = clean_token_surface(raw_tok, doc.options.normalize_traditional);
                    let upos_logits = mean_upos_logits(
                        doc,
                        &tensor.data,
                        expected_cols,
                        s_span,
                        tok_span,
                        &clean_tok,
                    );
                    let best_upos = argmax_slice(&upos_logits);
                    let upos_prob = softmax_prob(&upos_logits, best_upos);
                    let is_four_char_idiom = clean_tok.chars().count() == 4
                        && should_mean_pool_upos(doc, tok_span, &clean_tok);
                    if is_four_char_idiom
                        && (upos_prob < UPOS_THRESHOLD || UPOS_TAGS[best_upos] == Upos::Num)
                    {
                        let has_confident_subword = s_span.iter().any(|&s_idx| {
                            let r = &tensor.data[s_idx * expected_cols..s_idx * expected_cols + 17];
                            softmax_prob(r, argmax_slice(r)) >= 0.35
                        });
                        if has_confident_subword {
                            diluted_idioms.push((tok_idx, clean_tok.into_owned(), upos_logits));
                        }
                    }
                    if upos_prob >= UPOS_THRESHOLD {
                        doc.tokens[tok_idx].upos = Some(UPOS_TAGS[best_upos]);
                        clear_pos_unknown(doc, tok_span);
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
                } else if doc.tokens[tok_idx].upos == Some(Upos::Num) {
                    let tok_span = doc.tokens[tok_idx].span;
                    let raw_tok = token_str(text, &char_to_byte, tok_span);
                    let clean_tok = clean_token_surface(raw_tok, doc.options.normalize_traditional);
                    if matches!(clean_tok.as_ref(), "一五一十" | "三三兩兩" | "三三两两")
                    {
                        doc.tokens[tok_idx].upos = Some(Upos::Adv);
                    }
                }
            }
        }

        // 1a. Recover diluted 4-character numeral idioms (`二話不說`, `一動不動`, `四面八方`, etc.)
        //     after neighboring tokens in the sentence have received their model UPOS.
        for (tok_idx, clean_tok, upos_logits) in diluted_idioms {
            if let Some(recovered) = fallback_four_char_numeral_idiom(
                doc,
                tok_idx,
                item.sent_idx,
                &char_to_byte,
                text,
                &clean_tok,
                &upos_logits,
            ) {
                let tok_span = doc.tokens[tok_idx].span;
                doc.tokens[tok_idx].upos = Some(recovered);
                clear_pos_unknown(doc, tok_span);
            }
        }

        // 1b. Deterministic disambiguation for `什麼`/`什么` when Layer 2 left `upos == None` and the
        //     model predicted `PART` or fell below `UPOS_THRESHOLD`.
        for &tok_idx in &undecided_before {
            if !matches!(doc.tokens[tok_idx].upos, None | Some(Upos::Part)) {
                continue;
            }
            let tok_span = doc.tokens[tok_idx].span;
            if matches!(token_str(text, &char_to_byte, tok_span), "什麼" | "什么") {
                let followed_by_nominal = doc.tokens.get(tok_idx + 1).is_some_and(|next| {
                    next.sentence == item.sent_idx
                        && matches!(next.upos, Some(Upos::Noun | Upos::Propn))
                });
                doc.tokens[tok_idx].upos = Some(if followed_by_nominal {
                    Upos::Det
                } else {
                    Upos::Pron
                });
                clear_pos_unknown(doc, tok_span);
            }
        }

        // 1c. Lexicon compound nominal prior (`>= 3` Han chars in `doc.options.lexicon`).
        if !lexicon_set.is_empty() {
            for &tok_idx in &undecided_before {
                let tok_span = doc.tokens[tok_idx].span;
                let raw_tok = token_str(text, &char_to_byte, tok_span);
                let clean_tok = clean_token_surface(raw_tok, doc.options.normalize_traditional);
                let is_propn =
                    names_set.contains(raw_tok) || names_set.contains(clean_tok.as_ref());
                let override_eligible = if is_propn {
                    matches!(
                        doc.tokens[tok_idx].upos,
                        None | Some(
                            Upos::Noun
                                | Upos::Adj
                                | Upos::Verb
                                | Upos::Adp
                                | Upos::Adv
                                | Upos::Sconj
                                | Upos::Part
                        )
                    )
                } else {
                    matches!(
                        doc.tokens[tok_idx].upos,
                        None | Some(
                            Upos::Adj
                                | Upos::Verb
                                | Upos::Adp
                                | Upos::Adv
                                | Upos::Sconj
                                | Upos::Part
                        )
                    )
                };
                let followed_by_adverbial_de = doc.tokens.get(tok_idx + 1).is_some_and(|next| {
                    next.sentence == item.sent_idx
                        && token_str(text, &char_to_byte, next.span) == "地"
                });
                if override_eligible
                    && !followed_by_adverbial_de
                    && clean_tok.chars().count() >= 3
                    && clean_tok.chars().all(is_clean_han)
                    && !clean_tok.chars().next().is_some_and(is_zh_numeral_char)
                    && (is_propn || !has_non_nominal_lexicon_shape(&clean_tok))
                    && (lexicon_set.contains(raw_tok) || lexicon_set.contains(clean_tok.as_ref()))
                {
                    doc.tokens[tok_idx].upos =
                        Some(if is_propn { Upos::Propn } else { Upos::Noun });
                    clear_pos_unknown(doc, tok_span);
                }
            }
        }

        // 1d. Disposal coverb `將` / `将` -> `ADP` disambiguation (`normalize_traditional == true`).
        if doc.options.normalize_traditional {
            for &tok_idx in &undecided_before {
                if !matches!(
                    doc.tokens[tok_idx].upos,
                    None | Some(Upos::Noun | Upos::Verb | Upos::Propn | Upos::Adv | Upos::Sconj)
                ) {
                    continue;
                }
                let tok_span = doc.tokens[tok_idx].span;
                if matches!(token_str(text, &char_to_byte, tok_span), "將" | "将")
                    && is_disposal_coverb_context(
                        doc,
                        tok_idx,
                        item.sent_idx,
                        &item.clause_spans,
                        &char_to_byte,
                        text,
                    )
                {
                    doc.tokens[tok_idx].upos = Some(Upos::Adp);
                    clear_pos_unknown(doc, tok_span);
                }
            }
        }

        // 2. Decode each window with Chu-Liu/Edmonds (with multi-clause Chinese root-stealing
        //    penalties when a viable main-clause predicate exists) and stitch multi-window local
        //    roots with `deprel = "parataxis"`.
        let total_tokens: usize = window_tensors
            .iter()
            .map(|(w, _)| w.token_indices.len())
            .sum();
        let all_sent_tokens: Vec<usize> = window_tensors
            .iter()
            .flat_map(|(w, _)| w.token_indices.iter().copied())
            .collect();
        let is_zh_multiclause = item.clause_spans.len() >= 2
            && doc.segmentation.sentences[item.sent_idx].lang != Lang::En;
        let subordinate_clauses = if is_zh_multiclause {
            detect_subordinate_non_final_clauses(
                doc,
                &all_sent_tokens,
                &item.clause_spans,
                &char_to_byte,
                text,
            )
        } else {
            Vec::new()
        };

        let mut sent_tokens = Vec::with_capacity(total_tokens);
        let mut clause_pos = Vec::with_capacity(total_tokens);
        let mut win_of_pos = Vec::with_capacity(total_tokens);
        let mut local_in_win = Vec::with_capacity(total_tokens);
        let mut win_arc_matrices: Vec<Vec<Vec<f32>>> = Vec::with_capacity(window_tensors.len());
        let mut head_pos: Vec<Option<usize>> = Vec::with_capacity(total_tokens);
        let mut deprel_indices = Vec::with_capacity(total_tokens);
        let mut scores = Vec::with_capacity(total_tokens);
        let mut ancestor_ok = Vec::with_capacity(total_tokens);
        let mut global_root_pos: Option<usize> = None;
        let mut global_root_was_penalized = false;
        let rel_offset = 17 + max_len;
        let ancestor_cap = f64::from(ANCESTOR_MIN_MILLI) / 1000.0;

        for (w_idx, (win, tensor)) in window_tensors.iter().enumerate() {
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

            let mut penalized = vec![false; m];
            let mut viable_main = vec![false; m];
            if is_zh_multiclause {
                for (u_zero, &tok_idx) in win.token_indices.iter().enumerate() {
                    let c_idx = clause_index_for_span(&item.clause_spans, doc.tokens[tok_idx].span);
                    let clean_tok = clean_token_surface(
                        token_str(text, &char_to_byte, doc.tokens[tok_idx].span),
                        doc.options.normalize_traditional,
                    );
                    let non_pred = is_non_predicative_token(doc, tok_idx, &clean_tok);
                    let rel_pred = is_prenominal_rel_pred(
                        doc,
                        tok_idx,
                        item.sent_idx,
                        c_idx,
                        &item.clause_spans,
                        &char_to_byte,
                        text,
                    );
                    let sub_cl = subordinate_clauses.get(c_idx).copied().unwrap_or(false);
                    penalized[u_zero] = non_pred || rel_pred || sub_cl;
                    viable_main[u_zero] =
                        matches!(doc.tokens[tok_idx].upos, Some(Upos::Verb | Upos::Adj))
                            && !penalized[u_zero];
                }
            }
            let win_has_viable = viable_main.iter().any(|&v| v);
            let heads = if is_zh_multiclause && win_has_viable && penalized.iter().any(|&p| p) {
                let mut decode_scores = arc_scores.clone();
                for (u_zero, &is_pen) in penalized.iter().enumerate() {
                    if is_pen {
                        decode_scores[u_zero][0] -= ROOT_PENALTY_LOGIT;
                    }
                }
                chu_liu_edmonds(&decode_scores)
            } else {
                chu_liu_edmonds(&arc_scores)
            };

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
                    let cur_pos = base_pos + u_zero;
                    if let Some(r0) = global_root_pos {
                        if global_root_was_penalized && viable_main[u_zero] {
                            // Window 0's local root came from a subordinate/non-predicative clause;
                            // promote this window's viable main predicate to global sentence root.
                            head_pos[r0] = Some(cur_pos);
                            deprel_indices[r0] = PARATAXIS_DEPREL_INDEX;
                            for prev_h in &mut head_pos[..base_pos] {
                                if *prev_h == Some(r0) {
                                    *prev_h = Some(cur_pos);
                                }
                            }
                            global_root_pos = Some(cur_pos);
                            global_root_was_penalized = false;
                            head_pos.push(None);
                            deprel_indices.push(ROOT_DEPREL_INDEX);
                        } else {
                            head_pos.push(Some(r0));
                            deprel_indices.push(PARATAXIS_DEPREL_INDEX);
                        }
                    } else {
                        global_root_pos = Some(cur_pos);
                        global_root_was_penalized = is_zh_multiclause && penalized[u_zero];
                        head_pos.push(None);
                        deprel_indices.push(ROOT_DEPREL_INDEX);
                    }
                    scores.push(arc_score);
                } else {
                    let s_v = win.subword_spans[v - 1][0];
                    let (best_c, combined_score) = compute_biaffine_deprel_and_score(
                        &tensor.data,
                        expected_cols,
                        rel_offset,
                        s_u,
                        s_v,
                        p_arc,
                    );
                    head_pos.push(Some(base_pos + (v - 1)));
                    deprel_indices.push(best_c);
                    scores.push(combined_score);
                }
                sent_tokens.push(tok_idx);
                clause_pos.push(clause_index_for_span(
                    &item.clause_spans,
                    doc.tokens[tok_idx].span,
                ));
                win_of_pos.push(w_idx);
                local_in_win.push(u_zero);
            }
            win_arc_matrices.push(arc_scores);
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

        // 4. WS5-v3 Clause-local argument guard for low-confidence (`< 0.35`) cross-clause
        //    closure arcs: prefer a retained local `VERB`/`ADJ` predicate (`prob >= 0.22`,
        //    cycle-free) within the same clause, and relabel remaining `< 0.35` cross-clause core
        //    argument bridges to `"parataxis"` (predicates) or `"dep"` (nominals/adjuncts).
        if is_zh_multiclause {
            let mut clause_has_pred = vec![false; item.clause_spans.len()];
            for u in 0..n {
                if matches!(
                    doc.tokens[sent_tokens[u]].upos,
                    Some(Upos::Verb | Upos::Adj)
                ) && let Some(slot) = clause_has_pred.get_mut(clause_pos[u])
                {
                    *slot = true;
                }
            }

            for u in 0..n {
                if !retained[u] || u == r_pos || scores[u].milli() >= SYNTAX_THRESHOLD_MILLI {
                    continue;
                }
                if doc.tokens[sent_tokens[u]].upos == Some(Upos::Punct) {
                    continue;
                }
                let Some(cur_h) = head_pos[u] else {
                    continue;
                };
                if clause_pos[u] == clause_pos[cur_h] {
                    continue;
                }

                // Step A: Try cycle-free re-attachment to an emitted `VERB`/`ADJ` in the same clause.
                let w_idx = win_of_pos[u];
                let u_loc = local_in_win[u];
                let mut best_local: Option<(usize, f64)> = None;
                for cand_p in 0..n {
                    if cand_p == u
                        || !retained[cand_p]
                        || clause_pos[cand_p] != clause_pos[u]
                        || win_of_pos[cand_p] != w_idx
                        || !matches!(
                            doc.tokens[sent_tokens[cand_p]].upos,
                            Some(Upos::Verb | Upos::Adj)
                        )
                    {
                        continue;
                    }
                    if is_ancestor_of(u, cand_p, &head_pos, n) {
                        continue;
                    }
                    let p_loc = local_in_win[cand_p];
                    let p_prob = softmax_prob(&win_arc_matrices[w_idx][u_loc], p_loc + 1);
                    if p_prob >= LOCAL_REATTACH_MIN_PROB
                        && best_local.is_none_or(|(_, best_p)| p_prob > best_p)
                    {
                        best_local = Some((cand_p, p_prob));
                    }
                }

                if let Some((local_p, p_prob)) = best_local {
                    let (win, tensor) = &window_tensors[w_idx];
                    let s_u = win.subword_spans[u_loc][0];
                    let s_v = win.subword_spans[local_in_win[local_p]][0];
                    let (best_c, new_sc) = compute_biaffine_deprel_and_score(
                        &tensor.data,
                        expected_cols,
                        rel_offset,
                        s_u,
                        s_v,
                        p_prob,
                    );
                    head_pos[u] = Some(local_p);
                    deprel_indices[u] = best_c;
                    scores[u] = new_sc;
                    continue;
                }

                // Step B: Relabel cross-clause `< 0.35` closure arcs so they never hallucinate
                // long-distance core agency (`nsubj`/`obj`/`iobj`/`obl`).
                let c_u = clause_pos[u];
                let c_h = clause_pos[cur_h];
                if clause_has_pred.get(c_u).copied().unwrap_or(false)
                    || c_u.abs_diff(c_h) >= 2
                    || u.abs_diff(cur_h) > 6
                {
                    let cur_rel = DEPRELS[deprel_indices[u]];
                    if matches!(
                        doc.tokens[sent_tokens[u]].upos,
                        Some(Upos::Verb | Upos::Adj)
                    ) {
                        if matches!(
                            cur_rel,
                            "nsubj"
                                | "obj"
                                | "iobj"
                                | "obl"
                                | "csubj"
                                | "compound"
                                | "nmod"
                                | "amod"
                        ) {
                            deprel_indices[u] = PARATAXIS_DEPREL_INDEX;
                        }
                    } else if matches!(cur_rel, "nsubj" | "obj" | "iobj" | "obl" | "csubj") {
                        deprel_indices[u] = DEP_DEPREL_INDEX;
                    }
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
