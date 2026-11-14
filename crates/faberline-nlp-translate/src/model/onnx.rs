//! Pure-Rust ONNX Seq2Seq backend on `tract-onnx` (CPU, deterministic f32).

use std::sync::Arc;
use tract_onnx::prelude::*;

use super::{EncoderState, ModelError, Seq2SeqModel, Seq2SeqModelInfo, VerifiedSeq2Seq};

type Plan = Arc<TypedRunnableModel>;

pub(super) struct OnnxSeq2Seq {
    info: Seq2SeqModelInfo,
    encoder: Plan,
    decoder: Plan,
}

pub(super) fn load(v: &VerifiedSeq2Seq) -> Result<OnnxSeq2Seq, ModelError> {
    let s_len = v.info.src_max_len;
    let t_len = v.info.tgt_max_len;
    let d_mod = v.info.d_model;

    let enc_inf = tract_onnx::onnx()
        .model_for_read(&mut v.encoder_onnx.as_slice())
        .map_err(|e| ModelError::Load(format!("encoder: {e}")))?
        .with_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), [1, s_len]))
        .map_err(|e| ModelError::Load(format!("encoder input: {e}")))?;
    let encoder = enc_inf
        .into_optimized()
        .map_err(|e| ModelError::Load(format!("encoder opt: {e}")))?
        .into_runnable()
        .map_err(|e| ModelError::Load(format!("encoder runnable: {e}")))?;

    let dec_inf = tract_onnx::onnx()
        .model_for_read(&mut v.decoder_onnx.as_slice())
        .map_err(|e| ModelError::Load(format!("decoder: {e}")))?
        .with_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), [1, t_len]))
        .map_err(|e| ModelError::Load(format!("decoder input 0: {e}")))?
        .with_input_fact(
            1,
            InferenceFact::dt_shape(f32::datum_type(), [1, s_len, d_mod]),
        )
        .map_err(|e| ModelError::Load(format!("decoder input 1: {e}")))?;
    let decoder = dec_inf
        .into_optimized()
        .map_err(|e| ModelError::Load(format!("decoder opt: {e}")))?
        .into_runnable()
        .map_err(|e| ModelError::Load(format!("decoder runnable: {e}")))?;

    Ok(OnnxSeq2Seq {
        info: v.info.clone(),
        encoder,
        decoder,
    })
}

impl OnnxSeq2Seq {
    fn check_ids(&self, ids: &[i64], max_len: usize) -> Result<(), ModelError> {
        if ids.len() > max_len {
            return Err(ModelError::TooLong {
                len: ids.len(),
                max: max_len,
            });
        }
        let vocab = self.info.vocab;
        if let Some(&bad) = ids
            .iter()
            .find(|&&id| usize::try_from(id).map_or(true, |u| u >= vocab))
        {
            return Err(ModelError::Run(format!(
                "id {bad} outside vocabulary 0..{vocab}"
            )));
        }
        Ok(())
    }
}

impl Seq2SeqModel for OnnxSeq2Seq {
    fn info(&self) -> &Seq2SeqModelInfo {
        &self.info
    }

    fn encode(&self, input_ids: &[i64]) -> Result<EncoderState, ModelError> {
        let s_len = self.info.src_max_len;
        let d_mod = self.info.d_model;
        self.check_ids(input_ids, s_len)?;

        let mut padded = vec![self.info.pad_id; s_len];
        padded[..input_ids.len()].copy_from_slice(input_ids);

        let inp = tract_onnx::prelude::Tensor::from_shape(&[1, s_len], &padded)
            .map_err(|e| ModelError::Run(format!("encoder tensor: {e}")))?;
        let out = self
            .encoder
            .run(tvec!(inp.into()))
            .map_err(|e| ModelError::Run(format!("encoder run: {e}")))?;
        let view = out[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| ModelError::Run(format!("encoder view: {e}")))?;
        let shape = view.shape();
        if shape != [1, s_len, d_mod] {
            return Err(ModelError::Run(format!(
                "unexpected encoder shape {shape:?}, expected [1, {s_len}, {d_mod}]"
            )));
        }
        let mut hidden: Vec<f32> = view.iter().copied().collect();
        // Zero-mask padded source positions (`input_ids.len()..s_len`) for exact pad invariance.
        let active_elems = input_ids.len() * d_mod;
        hidden[active_elems..].fill(0.0);

        Ok(EncoderState {
            src_len: input_ids.len(),
            src_max_len: s_len,
            d_model: d_mod,
            hidden,
        })
    }

    fn decode_step(
        &self,
        encoder: &EncoderState,
        decoder_ids: &[i64],
    ) -> Result<Vec<f32>, ModelError> {
        let t_len = self.info.tgt_max_len;
        let s_len = self.info.src_max_len;
        let d_mod = self.info.d_model;
        let vocab = self.info.vocab;
        if decoder_ids.is_empty() {
            return Err(ModelError::Run(
                "decoder_ids must contain at least bos_id".to_owned(),
            ));
        }
        self.check_ids(decoder_ids, t_len)?;
        if encoder.hidden.len() != s_len * d_mod {
            return Err(ModelError::Run(
                "encoder hidden buffer shape mismatch".to_owned(),
            ));
        }

        let mut padded_dec = vec![self.info.pad_id; t_len];
        padded_dec[..decoder_ids.len()].copy_from_slice(decoder_ids);

        let dec_inp = tract_onnx::prelude::Tensor::from_shape(&[1, t_len], &padded_dec)
            .map_err(|e| ModelError::Run(format!("decoder ids tensor: {e}")))?;
        let enc_inp = tract_onnx::prelude::Tensor::from_shape(&[1, s_len, d_mod], &encoder.hidden)
            .map_err(|e| ModelError::Run(format!("encoder hidden tensor: {e}")))?;

        let out = self
            .decoder
            .run(tvec!(dec_inp.into(), enc_inp.into()))
            .map_err(|e| ModelError::Run(format!("decoder run: {e}")))?;
        let view = out[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| ModelError::Run(format!("decoder view: {e}")))?;
        let shape = view.shape();
        if shape != [1, t_len, vocab] {
            return Err(ModelError::Run(format!(
                "unexpected decoder shape {shape:?}, expected [1, {t_len}, {vocab}]"
            )));
        }

        let step = decoder_ids.len() - 1;
        let offset = step * vocab;
        let all: Vec<f32> = view.iter().copied().collect();
        Ok(all[offset..offset + vocab].to_vec())
    }
}
