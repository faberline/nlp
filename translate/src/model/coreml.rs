//! macOS Core ML Seq2Seq backend on `objc2-core-ml` (`CPUOnly` deterministic reference and
//! `All` Apple Silicon CPU+GPU+ANE fast mode).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};

use super::{EncoderState, ModelError, Seq2SeqModel, Seq2SeqModelInfo, VerifiedSeq2Seq};

const ENC_INPUT: &str = "input_ids";
const ENC_OUTPUT: &str = "encoder_hidden";
const DEC_INPUT_IDS: &str = "decoder_input_ids";
const DEC_INPUT_ENC: &str = "encoder_hidden";
const DEC_OUTPUT: &str = "logits";

pub(super) struct CoremlSeq2Seq {
    info: Seq2SeqModelInfo,
    encoder: Mutex<Retained<MLModel>>,
    decoder: Mutex<Retained<MLModel>>,
    compiled_enc_dir: Option<PathBuf>,
    compiled_dec_dir: Option<PathBuf>,
}

impl Drop for CoremlSeq2Seq {
    fn drop(&mut self) {
        if let Some(p) = &self.compiled_enc_dir {
            let _ = std::fs::remove_dir_all(p);
        }
        if let Some(p) = &self.compiled_dec_dir {
            let _ = std::fs::remove_dir_all(p);
        }
    }
}

// SAFETY: `MLModel` instances are accessed strictly under `Mutex` locks; all other fields are
// plain owned data.
unsafe impl Send for CoremlSeq2Seq {}
unsafe impl Sync for CoremlSeq2Seq {}

pub(super) fn load_cpu(v: &VerifiedSeq2Seq) -> Result<CoremlSeq2Seq, ModelError> {
    load_with_units(v, MLComputeUnits::CPUOnly)
}

pub(super) fn load_fast(v: &VerifiedSeq2Seq) -> Result<CoremlSeq2Seq, ModelError> {
    load_with_units(v, MLComputeUnits::All)
}

fn load_err(what: &str, e: &NSError) -> ModelError {
    ModelError::Load(format!("{what}: {}", e.localizedDescription()))
}

fn run_err(what: &str, e: &NSError) -> ModelError {
    ModelError::Run(format!("{what}: {}", e.localizedDescription()))
}

fn file_url(path: &Path) -> Result<Retained<NSURL>, ModelError> {
    let s = path
        .to_str()
        .ok_or_else(|| ModelError::Load("temporary path is not UTF-8".to_owned()))?;
    Ok(NSURL::fileURLWithPath(&NSString::from_str(s)))
}

fn compile_one(
    bytes: &[u8],
    name: &str,
    units: MLComputeUnits,
) -> Result<(Retained<MLModel>, Option<PathBuf>), ModelError> {
    let dir = tempfile::tempdir().map_err(|e| ModelError::Load(format!("{name} temp dir: {e}")))?;
    let source = dir.path().join(format!("{name}.mlmodel"));
    std::fs::write(&source, bytes).map_err(|e| ModelError::Load(format!("write {name}: {e}")))?;

    autoreleasepool(|_| {
        let url = file_url(&source)?;
        // SAFETY: `url` points to the verified `.mlmodel` file just written.
        #[allow(deprecated)]
        let compiled = unsafe { MLModel::compileModelAtURL_error(&url) }
            .map_err(|e| load_err(&format!("compile {name}"), &e))?;
        let compiled_dir = compiled.path().map(|p| PathBuf::from(p.to_string()));
        drop(dir);

        // SAFETY: valid `MLModelConfiguration` and `MLComputeUnits`.
        let config = unsafe {
            let c = MLModelConfiguration::new();
            c.setComputeUnits(units);
            c
        };
        // SAFETY: `compiled` is the URL returned by `compileModelAtURL_error`.
        let model = match unsafe {
            MLModel::modelWithContentsOfURL_configuration_error(&compiled, &config)
        } {
            Ok(m) => m,
            Err(e) => {
                if let Some(p) = &compiled_dir {
                    let _ = std::fs::remove_dir_all(p);
                }
                return Err(load_err(&format!("load {name}"), &e));
            }
        };
        Ok((model, compiled_dir))
    })
}

fn load_with_units(
    v: &VerifiedSeq2Seq,
    units: MLComputeUnits,
) -> Result<CoremlSeq2Seq, ModelError> {
    let cm = v
        .coreml
        .as_ref()
        .ok_or_else(|| ModelError::Load("the manifest lists no Core ML models".to_owned()))?;

    let (encoder, compiled_enc_dir) = compile_one(&cm.encoder, "encoder", units)?;
    let (decoder, compiled_dec_dir) = match compile_one(&cm.decoder, "decoder_step", units) {
        Ok(pair) => pair,
        Err(e) => {
            if let Some(p) = &compiled_enc_dir {
                let _ = std::fs::remove_dir_all(p);
            }
            return Err(e);
        }
    };

    let loaded = CoremlSeq2Seq {
        info: v.info.clone(),
        encoder: Mutex::new(encoder),
        decoder: Mutex::new(decoder),
        compiled_enc_dir,
        compiled_dec_dir,
    };

    // Smoke pad-run both graphs at load time so shape mismatches fail immediately at `load`.
    let state = loaded
        .encode(&vec![v.info.pad_id; v.info.src_max_len])
        .map_err(|e| ModelError::Load(format!("encoder pad run: {e}")))?;
    loaded
        .decode_step(&state, &[v.info.bos_id])
        .map_err(|e| ModelError::Load(format!("decoder pad run: {e}")))?;

    Ok(loaded)
}

impl CoremlSeq2Seq {
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

impl Seq2SeqModel for CoremlSeq2Seq {
    fn info(&self) -> &Seq2SeqModelInfo {
        &self.info
    }

    #[allow(deprecated)]
    fn encode(&self, input_ids: &[i64]) -> Result<EncoderState, ModelError> {
        let s_len = self.info.src_max_len;
        let d_mod = self.info.d_model;
        self.check_ids(input_ids, s_len)?;

        let pad = i32::try_from(self.info.pad_id)
            .map_err(|_| ModelError::Run("pad_id does not fit in i32".to_owned()))?;
        let mut padded = vec![pad; s_len];
        for (slot, &id) in padded.iter_mut().zip(input_ids) {
            *slot = i32::try_from(id)
                .map_err(|_| ModelError::Run(format!("id {id} does not fit in i32")))?;
        }

        autoreleasepool(|_| {
            let shape = NSArray::from_retained_slice(&[
                NSNumber::new_isize(1),
                NSNumber::new_isize(s_len as isize),
            ]);
            // SAFETY: newly allocated `[1, s_len]` Int32 array holds `s_len` i32 elements.
            let input_arr = unsafe {
                let arr = MLMultiArray::initWithShape_dataType_error(
                    MLMultiArray::alloc(),
                    &shape,
                    MLMultiArrayDataType::Int32,
                )
                .map_err(|e| run_err("encoder input array", &e))?;
                let dst = arr.dataPointer().as_ptr().cast::<i32>();
                std::ptr::copy_nonoverlapping(padded.as_ptr(), dst, s_len);
                arr
            };

            // SAFETY: valid dictionary mapping `ENC_INPUT` -> `MLFeatureValue`.
            let features = unsafe {
                let val = MLFeatureValue::featureValueWithMultiArray(&input_arr);
                let key = NSString::from_str(ENC_INPUT);
                let dict: Retained<NSDictionary<NSString, AnyObject>> =
                    NSDictionary::from_retained_objects(
                        &[&*key],
                        &[Retained::into_super(val).into()],
                    );
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &dict,
                )
                .map_err(|e| run_err("encoder features", &e))?
            };
            let provider: &ProtocolObject<dyn MLFeatureProvider> =
                ProtocolObject::from_ref(&*features);

            let guard = self
                .encoder
                .lock()
                .map_err(|_| ModelError::Run("encoder lock poisoned".to_owned()))?;
            // SAFETY: `provider` matches the encoder's input signature.
            let result = unsafe { guard.predictionFromFeatures_error(provider) }
                .map_err(|e| run_err("encoder predict", &e))?;
            drop(guard);

            // SAFETY: `result` was just produced by `predictionFromFeatures_error`.
            let mut hidden = unsafe {
                let val = result
                    .featureValueForName(&NSString::from_str(ENC_OUTPUT))
                    .ok_or_else(|| ModelError::Run(format!("no output named {ENC_OUTPUT}")))?;
                let arr = val
                    .multiArrayValue()
                    .ok_or_else(|| ModelError::Run("encoder output is not an array".to_owned()))?;
                read_3d_tensor(&arr, s_len, d_mod)?
            };
            hidden[input_ids.len() * d_mod..].fill(0.0);

            Ok(EncoderState {
                src_len: input_ids.len(),
                src_max_len: s_len,
                d_model: d_mod,
                hidden,
            })
        })
    }

    #[allow(deprecated)]
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

        let pad = i32::try_from(self.info.pad_id)
            .map_err(|_| ModelError::Run("pad_id does not fit in i32".to_owned()))?;
        let mut padded_dec = vec![pad; t_len];
        for (slot, &id) in padded_dec.iter_mut().zip(decoder_ids) {
            *slot = i32::try_from(id)
                .map_err(|_| ModelError::Run(format!("id {id} does not fit in i32")))?;
        }

        autoreleasepool(|_| {
            let dec_shape = NSArray::from_retained_slice(&[
                NSNumber::new_isize(1),
                NSNumber::new_isize(t_len as isize),
            ]);
            let enc_shape = NSArray::from_retained_slice(&[
                NSNumber::new_isize(1),
                NSNumber::new_isize(s_len as isize),
                NSNumber::new_isize(d_mod as isize),
            ]);

            // SAFETY: freshly allocated contiguous arrays of exact declared capacities.
            let (dec_arr, enc_arr) = unsafe {
                let da = MLMultiArray::initWithShape_dataType_error(
                    MLMultiArray::alloc(),
                    &dec_shape,
                    MLMultiArrayDataType::Int32,
                )
                .map_err(|e| run_err("decoder_input_ids array", &e))?;
                std::ptr::copy_nonoverlapping(
                    padded_dec.as_ptr(),
                    da.dataPointer().as_ptr().cast::<i32>(),
                    t_len,
                );

                let ea = MLMultiArray::initWithShape_dataType_error(
                    MLMultiArray::alloc(),
                    &enc_shape,
                    MLMultiArrayDataType::Float32,
                )
                .map_err(|e| run_err("encoder_hidden array", &e))?;
                std::ptr::copy_nonoverlapping(
                    encoder.hidden.as_ptr(),
                    ea.dataPointer().as_ptr().cast::<f32>(),
                    s_len * d_mod,
                );
                (da, ea)
            };

            // SAFETY: valid 2-entry dictionary mapping decoder inputs.
            let features = unsafe {
                let v0 = MLFeatureValue::featureValueWithMultiArray(&dec_arr);
                let v1 = MLFeatureValue::featureValueWithMultiArray(&enc_arr);
                let k0 = NSString::from_str(DEC_INPUT_IDS);
                let k1 = NSString::from_str(DEC_INPUT_ENC);
                let dict: Retained<NSDictionary<NSString, AnyObject>> =
                    NSDictionary::from_retained_objects(
                        &[&*k0, &*k1],
                        &[
                            Retained::into_super(v0).into(),
                            Retained::into_super(v1).into(),
                        ],
                    );
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &dict,
                )
                .map_err(|e| run_err("decoder features", &e))?
            };
            let provider: &ProtocolObject<dyn MLFeatureProvider> =
                ProtocolObject::from_ref(&*features);

            let guard = self
                .decoder
                .lock()
                .map_err(|_| ModelError::Run("decoder lock poisoned".to_owned()))?;
            // SAFETY: `provider` supplies both declared decoder inputs.
            let result = unsafe { guard.predictionFromFeatures_error(provider) }
                .map_err(|e| run_err("decoder predict", &e))?;
            drop(guard);

            let step = decoder_ids.len() - 1;
            // SAFETY: `result` was returned by `predictionFromFeatures_error`.
            unsafe {
                let val = result
                    .featureValueForName(&NSString::from_str(DEC_OUTPUT))
                    .ok_or_else(|| ModelError::Run(format!("no output named {DEC_OUTPUT}")))?;
                let arr = val
                    .multiArrayValue()
                    .ok_or_else(|| ModelError::Run("decoder output is not an array".to_owned()))?;
                read_3d_row(&arr, step, t_len, vocab)
            }
        })
    }
}

#[allow(deprecated)]
unsafe fn read_3d_tensor(
    arr: &MLMultiArray,
    dim1: usize,
    dim2: usize,
) -> Result<Vec<f32>, ModelError> {
    unsafe {
        let dims: Vec<usize> = arr.shape().iter().map(|n| n.as_isize() as usize).collect();
        let strides: Vec<usize> = arr
            .strides()
            .iter()
            .map(|n| n.as_isize() as usize)
            .collect();
        if dims != [1, dim1, dim2] || strides.len() != 3 {
            return Err(ModelError::Run(format!(
                "unexpected tensor shape {dims:?}, expected [1, {dim1}, {dim2}]"
            )));
        }
        let ptr = arr.dataPointer().as_ptr();
        let kind = arr.dataType();
        let mut out = Vec::with_capacity(dim1 * dim2);
        for r in 0..dim1 {
            for c in 0..dim2 {
                let idx = r * strides[1] + c * strides[2];
                out.push(read_elem(ptr, idx, kind)?);
            }
        }
        Ok(out)
    }
}

#[allow(deprecated)]
unsafe fn read_3d_row(
    arr: &MLMultiArray,
    row: usize,
    dim1: usize,
    dim2: usize,
) -> Result<Vec<f32>, ModelError> {
    unsafe {
        let dims: Vec<usize> = arr.shape().iter().map(|n| n.as_isize() as usize).collect();
        let strides: Vec<usize> = arr
            .strides()
            .iter()
            .map(|n| n.as_isize() as usize)
            .collect();
        if dims != [1, dim1, dim2] || strides.len() != 3 || row >= dim1 {
            return Err(ModelError::Run(format!(
                "unexpected tensor shape {dims:?} at step {row}"
            )));
        }
        let ptr = arr.dataPointer().as_ptr();
        let kind = arr.dataType();
        let mut out = Vec::with_capacity(dim2);
        for c in 0..dim2 {
            let idx = row * strides[1] + c * strides[2];
            out.push(read_elem(ptr, idx, kind)?);
        }
        Ok(out)
    }
}

unsafe fn read_elem(
    ptr: *mut std::ffi::c_void,
    idx: usize,
    kind: MLMultiArrayDataType,
) -> Result<f32, ModelError> {
    unsafe {
        Ok(match kind {
            MLMultiArrayDataType::Float32 => *ptr.cast::<f32>().add(idx),
            MLMultiArrayDataType::Float16 => f16_to_f32(*ptr.cast::<u16>().add(idx)),
            MLMultiArrayDataType::Float64 => *ptr.cast::<f64>().add(idx) as f32,
            other => {
                return Err(ModelError::Run(format!(
                    "unexpected array data type {other:?}"
                )));
            }
        })
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127u32 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}
