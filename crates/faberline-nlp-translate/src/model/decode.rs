//! Deterministic greedy (`beam_size = 1`) and fixed-width beam (`beam_size = 2..=4`) decoding
//! with quantized confidence scoring (`Score`).

use faberline_nlp::doc::model::ModelError;

use super::Seq2SeqModel;
use crate::schema::Score;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeResult {
    /// Generated token IDs excluding the leading `bos_id` and trailing `eos_id`.
    pub token_ids: Vec<i64>,
    /// Quantized geometric-mean token probability (`0..=1000`).
    pub confidence: Score,
}

#[derive(Clone, Debug)]
struct BeamHyp {
    ids: Vec<i64>,
    log_prob_sum: f32,
    steps: usize,
    finished: bool,
}

impl BeamHyp {
    fn normalized_score(&self) -> f32 {
        let len = self.steps.max(1) as f32;
        let lp = ((5.0 + len) / 6.0).powf(0.6);
        self.log_prob_sum / lp
    }
}

/// Decode one encoded source window into target token IDs using deterministic greedy or fixed-beam
/// search (`beam_size` clamped to `1..=4`).
pub fn decode_sequence(
    model: &dyn Seq2SeqModel,
    input_ids: &[i64],
    beam_size: usize,
    max_new_tokens: Option<usize>,
) -> Result<DecodeResult, ModelError> {
    let info = model.info();
    let enc_state = model.encode(input_ids)?;
    let beam_k = beam_size.clamp(1, 4);
    let max_steps = max_new_tokens
        .unwrap_or(info.tgt_max_len.saturating_sub(1))
        .min(info.tgt_max_len.saturating_sub(1))
        .max(1);

    if beam_k == 1 {
        return decode_greedy(model, &enc_state, max_steps);
    }

    let mut beams = vec![BeamHyp {
        ids: vec![info.bos_id],
        log_prob_sum: 0.0,
        steps: 0,
        finished: false,
    }];

    for _step in 0..max_steps {
        if beams.iter().all(|b| b.finished) {
            break;
        }
        let mut next_candidates: Vec<BeamHyp> = Vec::with_capacity(beam_k * beam_k);
        for hyp in &beams {
            if hyp.finished || hyp.ids.len() >= info.tgt_max_len {
                next_candidates.push(hyp.clone());
                continue;
            }
            let logits = model.decode_step(&enc_state, &hyp.ids)?;
            let top_k = top_k_log_probs(&logits, info.pad_id, beam_k);
            for (tok_id, log_p) in top_k {
                let is_eos = tok_id == info.eos_id;
                let mut new_ids = hyp.ids.clone();
                if !is_eos {
                    new_ids.push(tok_id);
                }
                next_candidates.push(BeamHyp {
                    ids: new_ids,
                    log_prob_sum: hyp.log_prob_sum + log_p,
                    steps: hyp.steps + 1,
                    finished: is_eos,
                });
            }
        }
        next_candidates.sort_by(|a, b| {
            b.normalized_score()
                .total_cmp(&a.normalized_score())
                .then_with(|| a.ids.cmp(&b.ids))
        });
        next_candidates.truncate(beam_k);
        beams = next_candidates;
    }

    let best = beams.into_iter().next().unwrap();
    let geom_prob = if best.steps == 0 {
        1.0
    } else {
        (best.log_prob_sum / (best.steps as f32)).exp()
    };
    Ok(DecodeResult {
        token_ids: best.ids.into_iter().skip(1).collect(),
        confidence: Score::from_f32(geom_prob),
    })
}

fn decode_greedy(
    model: &dyn Seq2SeqModel,
    enc_state: &super::EncoderState,
    max_steps: usize,
) -> Result<DecodeResult, ModelError> {
    let info = model.info();
    let mut ids = vec![info.bos_id];
    let mut log_prob_sum = 0.0f32;
    let mut steps = 0usize;

    for _ in 0..max_steps {
        if ids.len() >= info.tgt_max_len {
            break;
        }
        let logits = model.decode_step(enc_state, &ids)?;
        let (best_id, best_log_p) = top_k_log_probs(&logits, info.pad_id, 1)
            .into_iter()
            .next()
            .ok_or_else(|| ModelError::Run("empty vocabulary logits".to_owned()))?;
        log_prob_sum += best_log_p;
        steps += 1;
        if best_id == info.eos_id {
            break;
        }
        ids.push(best_id);
    }

    let geom_prob = if steps == 0 {
        1.0
    } else {
        (log_prob_sum / (steps as f32)).exp()
    };
    Ok(DecodeResult {
        token_ids: ids.into_iter().skip(1).collect(),
        confidence: Score::from_f32(geom_prob),
    })
}

/// Numerically stable log-softmax top-K selection (excluding `pad_id`), breaking exact score ties
/// by lowest token ID for cross-backend determinism.
pub fn top_k_log_probs(logits: &[f32], pad_id: i64, k: usize) -> Vec<(i64, f32)> {
    if logits.is_empty() || k == 0 {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
    let log_denom = max_val + sum_exp.ln();

    let mut scored: Vec<(i64, f32)> = logits
        .iter()
        .enumerate()
        .filter_map(|(idx, &v)| {
            let id = idx as i64;
            if id == pad_id && logits.len() > 1 {
                None
            } else {
                Some((id, v - log_denom))
            }
        })
        .collect();

    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}
