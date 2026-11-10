//! ONNX backend on `tract-onnx`: pure Rust, CPU, f32, one sequence per run.

use super::{Batch, Model, ModelError, ModelInfo, Output, Tensor, Verified};
use std::sync::Arc;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::ops::array::Gather;
use tract_onnx::tract_hir::infer::Factoid;

type Plan = Arc<TypedRunnableModel>;

pub(super) struct OnnxModel {
    info: ModelInfo,
    plan: Plan,
    /// Rows of the embedding table. Every id must be in `0..vocab`.
    vocab: usize,
}

/// Parses the model from the verified bytes. The embedding table is the constant input of the
/// one `Gather` node; its row count is the vocabulary size.
pub(super) fn load(v: &Verified) -> Result<OnnxModel, ModelError> {
    let max_len = v.info.max_len;
    let inference = tract_onnx::onnx()
        .model_for_read(&mut v.model.as_slice())
        .map_err(|e| ModelError::Load(format!("model: {e}")))?;
    check_input_len(&inference, max_len)?;
    let inference = inference
        .with_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), [1, max_len]))
        .map_err(|e| ModelError::Load(format!("model: {e}")))?;
    let typed = inference
        .into_optimized()
        .map_err(|e| ModelError::Load(format!("model: {e}")))?;
    let vocab = embedding_rows(&typed)?;
    if usize::try_from(v.info.pad_id).map_or(true, |p| p >= vocab) {
        return Err(ModelError::Load(format!(
            "pad_id {} is outside the vocabulary of {vocab}",
            v.info.pad_id
        )));
    }
    let plan = typed
        .into_runnable()
        .map_err(|e| ModelError::Load(format!("model: {e}")))?;
    Ok(OnnxModel {
        info: v.info.clone(),
        plan,
        vocab,
    })
}

/// Refuses a model whose declared input length (the second dimension of input 0) is fixed and
/// differs from `max_len`. A symbolic or unknown dimension is accepted.
fn check_input_len(model: &InferenceModel, max_len: usize) -> Result<(), ModelError> {
    let inputs = model
        .input_outlets()
        .map_err(|e| ModelError::Load(format!("model: {e}")))?;
    if inputs.is_empty() {
        return Err(ModelError::Load("model: no graph input".to_owned()));
    }
    let fact = model
        .input_fact(0)
        .map_err(|e| ModelError::Load(format!("model: {e}")))?;
    let declared = fact
        .shape
        .dim(1)
        .and_then(|d| d.concretize())
        .and_then(|d| d.to_i64().ok())
        .and_then(|n| usize::try_from(n).ok());
    match declared {
        Some(n) if n != max_len => Err(ModelError::Load(format!(
            "max_len is {max_len}, but the model declares input length {n}"
        ))),
        _ => Ok(()),
    }
}

fn embedding_rows(model: &TypedModel) -> Result<usize, ModelError> {
    let mut rows = Vec::new();
    for node in model.nodes() {
        if !node.op_is::<Gather>() {
            continue;
        }
        let table = node
            .inputs
            .first()
            .ok_or_else(|| ModelError::Load("Gather node without a data input".to_owned()))?;
        let fact = model
            .outlet_fact(*table)
            .map_err(|e| ModelError::Load(format!("model: {e}")))?;
        let konst = fact
            .konst
            .as_ref()
            .ok_or_else(|| ModelError::Load("embedding table is not a constant".to_owned()))?;
        let n = konst
            .shape()
            .first()
            .copied()
            .ok_or_else(|| ModelError::Load("embedding table has no rows".to_owned()))?;
        rows.push(n);
    }
    match rows.as_slice() {
        [n] => Ok(*n),
        [] => Err(ModelError::Load(
            "no embedding table (Gather) in model".to_owned(),
        )),
        _ => Err(ModelError::Load(
            "more than one embedding table in model".to_owned(),
        )),
    }
}

impl OnnxModel {
    /// Runs one sequence, padded to `max_len`, and keeps the first `ids.len()` positions.
    fn run_one(&self, ids: &[i64]) -> Result<Tensor, ModelError> {
        let max_len = self.info.max_len;
        let mut padded = vec![self.info.pad_id; max_len];
        padded[..ids.len()].copy_from_slice(ids);
        let input = tract_onnx::prelude::Tensor::from_shape(&[1, max_len], &padded)
            .map_err(|e| ModelError::Run(e.to_string()))?;
        let outputs = self
            .plan
            .run(tvec!(input.into()))
            .map_err(|e| ModelError::Run(e.to_string()))?;
        let view = outputs[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| ModelError::Run(e.to_string()))?;
        let shape = view.shape().to_vec();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != max_len {
            return Err(ModelError::Run(format!(
                "unexpected output shape {shape:?}"
            )));
        }
        let labels = shape[2];
        let data: Vec<f32> = view.iter().take(ids.len() * labels).copied().collect();
        Ok(Tensor {
            name: "logits".to_owned(),
            shape: vec![ids.len(), labels],
            data,
        })
    }
}

impl Model for OnnxModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn run(&self, batch: &Batch) -> Result<Output, ModelError> {
        let max = self.info.max_len;
        if let Some(len) = batch.ids.iter().map(Vec::len).find(|&n| n > max) {
            return Err(ModelError::TooLong { len, max });
        }
        // Checked before any run: tract panics on an out-of-range gather index.
        let in_vocab = |id: i64| usize::try_from(id).is_ok_and(|u| u < self.vocab);
        if let Some(&bad) = batch.ids.iter().flatten().find(|&&id| !in_vocab(id)) {
            return Err(ModelError::Run(format!(
                "id {bad} is outside the vocabulary of {} entries",
                self.vocab
            )));
        }
        let logits = batch
            .ids
            .iter()
            .map(|ids| self.run_one(ids))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Output { logits })
    }
}
