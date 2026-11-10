//! Core ML backend on `objc2-core-ml`: macOS only, CPU only, f32, one sequence per run.
//!
//! The manifest lists a `.mlmodel` file. Core ML loads compiled models from a file URL, so the
//! verified bytes are written to a private temporary directory, compiled there, and loaded
//! with `MLComputeUnitsCPUOnly`. The directory lives as long as the model. Calls into the
//! framework are `unsafe` in the bindings. Each call below passes objects that this module owns.

use super::{Batch, Model, ModelError, ModelInfo, Output, Tensor, Verified};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use std::path::Path;
use std::sync::Mutex;

const INPUT: &str = "input_ids";
const OUTPUT: &str = "logits";

pub(super) struct CoremlModel {
    info: ModelInfo,
    vocab: usize,
    model: Mutex<Retained<MLModel>>,
    /// Holds the model files for as long as the model is loaded.
    _dir: tempfile::TempDir,
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
    let url = file_url(&source)?;
    // SAFETY: `url` is a valid file URL. The synchronous compile call is deprecated in favour
    // of a block-based one; the sync call keeps this loader free of run loops and threads.
    #[allow(deprecated)]
    let compiled = unsafe { MLModel::compileModelAtURL_error(&url) }
        .map_err(|e| load_err("compile model", &e))?;
    // SAFETY: plain configuration object with a valid enum value.
    let config = unsafe {
        let config = MLModelConfiguration::new();
        config.setComputeUnits(MLComputeUnits::CPUOnly);
        config
    };
    // SAFETY: `compiled` is the URL that the compile call returned.
    let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&compiled, &config) }
        .map_err(|e| load_err("load model", &e))?;
    // Core ML puts the compiled copy in the system temporary area. Remove it, now that the
    // model is loaded.
    if let Some(path) = compiled.path() {
        let _ = std::fs::remove_dir_all(path.to_string());
    }
    let loaded = CoremlModel {
        info: v.info.clone(),
        vocab: file.vocab,
        model: Mutex::new(model),
        _dir: dir,
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
        let provider: &ProtocolObject<dyn MLFeatureProvider> = ProtocolObject::from_ref(&*features);

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
