//! Core ML backend on `objc2-core-ml`: macOS only, f32, one sequence per run.
//!
//! The manifest lists a `.mlmodel` file. Core ML loads compiled models from a file URL, so the
//! verified bytes are written to a private temporary directory, compiled there, and loaded
//! with `MLComputeUnits::CPUOnly` (`Backend::Coreml`) or `MLComputeUnits::All`
//! (`Backend::CoremlFast`). The compiled bundle lives until the model is dropped. Calls into the
//! framework are `unsafe` in the bindings. Each call below passes objects that this module owns.

use super::{Batch, Model, ModelError, ModelInfo, Output, Tensor, Verified};
use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const INPUT: &str = "input_ids";
const OUTPUT: &str = "logits";

pub(super) struct CoremlModel {
    info: ModelInfo,
    vocab: usize,
    model: Mutex<Retained<MLModel>>,
    /// Compiled `.mlmodelc` bundle path kept alive while `MLModel` is loaded, removed on drop.
    compiled_dir: Option<PathBuf>,
}

impl Drop for CoremlModel {
    fn drop(&mut self) {
        if let Some(path) = &self.compiled_dir {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

// SAFETY: the `MLModel` is reached only through the mutex, so at most one thread calls it at a
// time. The other fields are plain data.
unsafe impl Send for CoremlModel {}
unsafe impl Sync for CoremlModel {}

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

pub(super) fn load(v: &Verified) -> Result<CoremlModel, ModelError> {
    load_with_units(v, MLComputeUnits::CPUOnly)
}

pub(super) fn load_with_units(
    v: &Verified,
    units: MLComputeUnits,
) -> Result<CoremlModel, ModelError> {
    let file = v
        .coreml
        .as_ref()
        .ok_or_else(|| ModelError::Load("the manifest lists no Core ML model".to_owned()))?;
    if usize::try_from(v.info.pad_id).map_or(true, |p| p >= file.vocab) {
        return Err(ModelError::Load(format!(
            "pad_id {} is outside the vocabulary of {}",
            v.info.pad_id, file.vocab
        )));
    }
    let dir = tempfile::tempdir().map_err(|e| ModelError::Load(format!("temp dir: {e}")))?;
    let source = dir.path().join("model.mlmodel");
    std::fs::write(&source, &file.model)
        .map_err(|e| ModelError::Load(format!("write model: {e}")))?;
    let (model, compiled_dir) = autoreleasepool(|_| -> Result<_, ModelError> {
        let url = file_url(&source)?;
        // SAFETY: `url` is a valid file URL. The synchronous compile call is deprecated in favour
        // of a block-based one; the sync call keeps this loader free of run loops and threads.
        #[allow(deprecated)]
        let compiled = unsafe { MLModel::compileModelAtURL_error(&url) }
            .map_err(|e| load_err("compile model", &e))?;
        let compiled_dir = compiled.path().map(|p| PathBuf::from(p.to_string()));
        // Drop the uncompiled `.mlmodel` directory right away now that compilation finished.
        drop(dir);
        // SAFETY: plain configuration object with a valid enum value.
        let config = unsafe {
            let config = MLModelConfiguration::new();
            config.setComputeUnits(units);
            config
        };
        // SAFETY: `compiled` is the URL that the compile call returned.
        let model = match unsafe {
            MLModel::modelWithContentsOfURL_configuration_error(&compiled, &config)
        } {
            Ok(m) => m,
            Err(e) => {
                if let Some(p) = &compiled_dir {
                    let _ = std::fs::remove_dir_all(p);
                }
                return Err(load_err("load model", &e));
            }
        };
        Ok((model, compiled_dir))
    })?;
    let loaded = CoremlModel {
        info: v.info.clone(),
        vocab: file.vocab,
        model: Mutex::new(model),
        compiled_dir,
    };
    // One sequence of `max_len` pad ids. A `max_len` that differs from the model's input length
    // fails here, at load, and not at the first real run.
    loaded
        .run_one(&vec![v.info.pad_id; v.info.max_len])
        .map_err(|e| ModelError::Load(format!("pad run: {e}")))?;
    Ok(loaded)
}

impl CoremlModel {
    // `dataPointer` is deprecated for the block-based accessors. For a freshly made, contiguous
    // array of known type, a direct pointer is the simplest correct access.
    #[allow(deprecated)]
    fn run_one(&self, ids: &[i64]) -> Result<Tensor, ModelError> {
        let max_len = self.info.max_len;
        let to_i32 = |id: i64| {
            i32::try_from(id).map_err(|_| ModelError::Run(format!("id {id} does not fit in i32")))
        };
        let pad = to_i32(self.info.pad_id)?;
        let mut padded = vec![pad; max_len];
        for (slot, &id) in padded.iter_mut().zip(ids) {
            *slot = to_i32(id)?;
        }

        autoreleasepool(|_| {
            let shape = NSArray::from_retained_slice(&[
                NSNumber::new_isize(1),
                NSNumber::new_isize(max_len as isize),
            ]);
            // SAFETY: the array is freshly allocated with an Int32 type and a contiguous buffer, so
            // `max_len` i32 values fit in it. The buffer is written before the array is used.
            let input = unsafe {
                let arr = MLMultiArray::initWithShape_dataType_error(
                    MLMultiArray::alloc(),
                    &shape,
                    MLMultiArrayDataType::Int32,
                )
                .map_err(|e| run_err("input array", &e))?;
                let dst = arr.dataPointer().as_ptr().cast::<i32>();
                std::ptr::copy_nonoverlapping(padded.as_ptr(), dst, max_len);
                arr
            };

            // SAFETY: valid objects owned here. The dictionary maps one string to one feature value.
            let features = unsafe {
                let value = MLFeatureValue::featureValueWithMultiArray(&input);
                let key = NSString::from_str(INPUT);
                let dict: Retained<NSDictionary<NSString, AnyObject>> =
                    NSDictionary::from_retained_objects(
                        &[&*key],
                        &[Retained::into_super(value).into()],
                    );
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &dict,
                )
                .map_err(|e| run_err("input features", &e))?
            };
            let provider: &ProtocolObject<dyn MLFeatureProvider> =
                ProtocolObject::from_ref(&*features);

            let guard = self
                .model
                .lock()
                .map_err(|_| ModelError::Run("model lock poisoned".to_owned()))?;
            // SAFETY: `provider` holds the one input the model declares.
            let result = unsafe { guard.predictionFromFeatures_error(provider) }
                .map_err(|e| run_err("predict", &e))?;
            drop(guard);

            // SAFETY: `result` is the provider the model returned.
            unsafe {
                let value = result
                    .featureValueForName(&NSString::from_str(OUTPUT))
                    .ok_or_else(|| ModelError::Run(format!("no output named {OUTPUT}")))?;
                let arr = value
                    .multiArrayValue()
                    .ok_or_else(|| ModelError::Run("output is not an array".to_owned()))?;
                read_logits(&arr, ids.len(), max_len)
            }
        })
    }
}

/// Reads the first `len` rows of an output of shape `[1, max_len, labels]`, through the
/// array's own strides and data type.
///
/// # Safety
/// `arr` must be a valid `MLMultiArray`.
#[allow(deprecated)]
unsafe fn read_logits(
    arr: &MLMultiArray,
    len: usize,
    max_len: usize,
) -> Result<Tensor, ModelError> {
    // SAFETY: the caller guarantees that `arr` is valid. Every index is below
    // `strides[1] * max_len`, which is inside the array for the shape checked here.
    unsafe {
        let dims: Vec<usize> = arr.shape().iter().map(|n| n.as_isize() as usize).collect();
        let strides: Vec<usize> = arr
            .strides()
            .iter()
            .map(|n| n.as_isize() as usize)
            .collect();
        if dims.len() != 3 || dims[0] != 1 || dims[1] != max_len || strides.len() != 3 {
            return Err(ModelError::Run(format!("unexpected output shape {dims:?}")));
        }
        let labels = dims[2];
        let ptr = arr.dataPointer().as_ptr();
        let kind = arr.dataType();
        let mut data = Vec::with_capacity(len * labels);
        for pos in 0..len {
            for label in 0..labels {
                let i = pos * strides[1] + label * strides[2];
                data.push(match kind {
                    MLMultiArrayDataType::Float32 => *ptr.cast::<f32>().add(i),
                    MLMultiArrayDataType::Float16 => f16_to_f32(*ptr.cast::<u16>().add(i)),
                    MLMultiArrayDataType::Float64 => *ptr.cast::<f64>().add(i) as f32,
                    other => {
                        return Err(ModelError::Run(format!("unexpected output type {other:?}")));
                    }
                });
            }
        }
        Ok(Tensor {
            name: OUTPUT.to_owned(),
            shape: vec![len, labels],
            data,
        })
    }
}

/// Converts IEEE-754 binary16 bits to `f32` (needed if GPU/Neural Engine emits fp16 tensors).
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

impl Model for CoremlModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn run(&self, batch: &Batch) -> Result<Output, ModelError> {
        let max = self.info.max_len;
        if let Some(len) = batch.ids.iter().map(Vec::len).find(|&n| n > max) {
            return Err(ModelError::TooLong { len, max });
        }
        // Checked before any run: Core ML does not promise a clean error for a bad gather index.
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
