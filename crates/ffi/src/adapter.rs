//! Owned native LoRA GGUF and control-vector C interface.
//! Independent original-library captures qualify dense Llama CPU execution.
use crate::model::{ModelData, ModelHandle};
use crate::model_metadata::{MetadataCache, copy_optional};
use crate::native_metadata_policy::{Profile, pointer_map_order};
use crate::*;
use llama_gguf::{GgufSet, RawValue, ShardLimits};
use llama_model_runtime::{AdapterSet, Lora, Matrix};
use std::collections::BTreeMap;
use std::sync::{Arc, MutexGuard, Weak};

const MAX_ADAPTER_BYTES: usize = 512 * 1024 * 1024;
const MAX_ADAPTERS: usize = 256;
const MAX_MODEL_ADAPTERS: usize = 1024;

pub struct AdapterHandle {
    model: Weak<ModelData>,
    data: Arc<AdapterData>,
}
struct AdapterData {
    weights: Vec<Lora>,
    metadata: MetadataCache,
    invocation_tokens: Vec<i32>,
    bytes: usize,
}
#[derive(Clone)]
struct Active {
    identity: usize,
    scale: f32,
    data: Arc<AdapterData>,
}
#[derive(Default)]
pub(crate) struct ContextAdapters {
    active: Vec<Active>,
    // Incomplete updates leave previous complete layer rows unchanged.
    retained_control: Vec<Vec<f32>>,
    control_range: Option<(i32, i32)>,
}
fn reserve<T>(n: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| "adapter allocation failed")?;
    Ok(out)
}
fn copy<T: Copy>(values: &[T]) -> Result<Vec<T>> {
    let mut out = reserve(values.len())?;
    out.extend_from_slice(values);
    Ok(out)
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| "adapter size overflow".into())
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| "adapter size overflow".into())
}
fn require(bytes: usize) -> Result<()> {
    if bytes > MAX_ADAPTER_BYTES {
        Err("adapter owned payload exceeds 512 MiB".into())
    } else {
        Ok(())
    }
}
fn registry(model: &ModelData) -> Result<MutexGuard<'_, BTreeMap<usize, usize>>> {
    model
        .adapters
        .lock()
        .map_err(|_| "model adapter registry poisoned".into())
}
pub(crate) fn release_model_adapters(model: &mut ModelData) {
    let pointers = match model.adapters.get_mut() {
        Ok(pointers) => std::mem::take(pointers),
        Err(poison) => std::mem::take(poison.into_inner()),
    };
    for (pointer, _) in pointers {
        // SAFETY: each pointer was created by load, registered once, and not
        // manually freed. Model destruction owns the final handle release.
        unsafe {
            drop(Box::from_raw(pointer as *mut AdapterHandle));
        }
    }
}
fn expected_string(files: &GgufSet, key: &[u8], expected: &[u8]) -> Result<()> {
    let value = files
        .raw_metadata_native(key)
        .and_then(RawValue::as_c_string);
    if value == Some(expected) {
        Ok(())
    } else {
        Err(format!(
            "adapter metadata {} has an incompatible string value",
            String::from_utf8_lossy(key)
        ))
    }
}
fn matrix(files: &mut GgufSet, name: &str) -> Result<Matrix> {
    let info = files.tensor(name).ok_or("missing adapter tensor")?;
    if info.dimensions.len() > 2 || info.dimensions.is_empty() {
        return Err("adapter tensors must be matrices with at most two dimensions".into());
    }
    let cols = usize::try_from(info.dimensions[0]).map_err(|_| "adapter width overflow")?;
    let rows = usize::try_from(*info.dimensions.get(1).unwrap_or(&1))
        .map_err(|_| "adapter height overflow")?;
    let dtype = info.dtype;
    let bytes = files.read_tensor(name).map_err(|e| e.to_string())?;
    Matrix::from_quantized(rows, cols, dtype, bytes).map_err(|e| e.to_string())
}
unsafe fn model_data(model: *mut ModelHandle) -> Result<Arc<ModelData>> {
    let data = unsafe { reference(model)? }.data.clone();
    if registry(&data)?.len() >= MAX_MODEL_ADAPTERS {
        return Err("model already owns 1024 adapters".into());
    }
    Ok(data)
}
fn load_limits() -> ShardLimits {
    let mut limits = ShardLimits {
        max_shards: 1,
        ..ShardLimits::default()
    };
    limits.gguf.max_metadata_bytes = 32 * 1024 * 1024;
    limits.max_metadata_storage_bytes = 64 * 1024 * 1024;
    limits.gguf.max_tensors = 32_768;
    limits.gguf.max_tensor_read_bytes = MAX_ADAPTER_BYTES;
    limits
}
unsafe fn load_file_pointer(
    model: *mut ModelHandle,
    file: *mut std::ffi::c_void,
) -> Result<*mut AdapterHandle> {
    let data = unsafe { model_data(model)? };
    let mut source = unsafe { crate::file_pointer::Source::new(file)? };
    let origin = source.origin();
    let files = GgufSet::from_file_at_with_limits(source.take_file()?, origin, load_limits())
        .map_err(|e| e.to_string())?;
    let result = load_files(data, files);
    if let Err(error) = source.finish() {
        if let Ok(adapter) = result {
            // SAFETY: this call owns the registered adapter before exposure.
            unsafe {
                free(adapter)?;
            }
        }
        return Err(error);
    }
    result
}
unsafe fn load(model: *mut ModelHandle, path: *const c_char) -> Result<*mut AdapterHandle> {
    let data = unsafe { model_data(model)? };
    if path.is_null() {
        return Err("null adapter filename".into());
    }
    let raw = unsafe { CStr::from_ptr(path) }.to_bytes();
    #[cfg(unix)]
    let filename = {
        use std::os::unix::ffi::OsStrExt;
        std::path::Path::new(std::ffi::OsStr::from_bytes(raw))
    };
    #[cfg(not(unix))]
    let filename = std::path::Path::new(
        std::str::from_utf8(raw).map_err(|_| "adapter filename is not UTF-8")?,
    );
    let files =
        GgufSet::open_splits_with_limits(&[filename], load_limits()).map_err(|e| e.to_string())?;
    load_files(data, files)
}
fn load_files(data: Arc<ModelData>, mut files: GgufSet) -> Result<*mut AdapterHandle> {
    if files.tensor_endian(files.tensors().first().map_or("", |t| t.name.as_str()))
        == Some(llama_gguf::Endian::Big)
    {
        return Err("big-endian adapter tensors are outside the qualified native profile".into());
    }
    expected_string(&files, b"general.type", b"adapter")?;
    expected_string(
        &files,
        b"general.architecture",
        data.architecture.as_bytes(),
    )?;
    expected_string(&files, b"adapter.type", b"lora")?;
    let alpha = match files.raw_metadata_native(b"adapter.lora.alpha") {
        None => 0.0,
        Some(RawValue::F32(value)) if value.is_finite() => *value,
        _ => return Err("adapter.lora.alpha must be a finite FLOAT32".into()),
    };
    let invocation_tokens = match files.raw_metadata_native(b"adapter.alora.invocation_tokens") {
        None => Vec::new(),
        Some(RawValue::Array {
            element_type: 4,
            values,
        }) => {
            let mut tokens = reserve(values.len())?;
            for value in values {
                let RawValue::U32(value) = value else {
                    return Err("invalid invocation-token element".into());
                };
                tokens.push(*value as i32);
            }
            tokens
        }
        _ => return Err("aLoRA invocation tokens must be an array of UINT32".into()),
    };
    let metadata = MetadataCache::build(&files)?;
    let mut pairs = BTreeMap::<String, (Option<String>, Option<String>)>::new();
    let mut owned = add(
        add(metadata.owned_bytes()?, files.metadata_storage_bytes())?,
        mul(invocation_tokens.capacity(), 4)?,
    )?;
    for tensor in files.tensors() {
        if tensor.name.len() >= 64 || tensor.name.as_bytes().contains(&0) {
            return Err(
                "adapter tensor names exceeding native GGML name storage are unsupported".into(),
            );
        }
        let suffix = if tensor.name.ends_with(".lora_a") {
            ".lora_a"
        } else if tensor.name.ends_with(".lora_b") {
            ".lora_b"
        } else if tensor.name.ends_with("_norm.weight") {
            continue;
        } else {
            return Err(format!("unexpected adapter tensor suffix: {}", tensor.name));
        };
        owned = add(owned, add(256, mul(tensor.name.len(), 2)?)?)?;
        let target = tensor.name.replace(suffix, "");
        let pair = pairs.entry(target).or_default();
        if suffix == ".lora_a" {
            pair.0 = Some(tensor.name.clone());
        } else {
            pair.1 = Some(tensor.name.clone());
        }
        owned = add(
            owned,
            add(
                mul(tensor.elements().map_err(|e| e.to_string())?, 4)?,
                usize::try_from(tensor.byte_len).map_err(|_| "adapter byte length")?,
            )?,
        )?;
        require(owned)?;
    }
    let runtime = data
        .model
        .as_ref()
        .map(|model| model.runtime_model())
        .transpose()
        .map_err(|e| e.to_string())?;
    let mut weights = reserve(pairs.len())?;
    for (target, (a, b)) in pairs {
        let a = a.ok_or_else(|| format!("missing A component for {target}"))?;
        let b = b.ok_or_else(|| format!("missing B component for {target}"))?;
        if !data.tensor_names.contains(&target) {
            return Err(format!(
                "adapter target is absent from the base model: {target}"
            ));
        }
        let runtime = runtime
            .as_ref()
            .ok_or("vocabulary-only model has no adapter matrix targets")?;
        let weight = runtime
            .weight(&target)
            .ok_or_else(|| format!("unsupported non-matrix adapter target: {target}"))?;
        let a = matrix(&mut files, &a)?;
        let b = matrix(&mut files, &b)?;
        let valid = if target == "token_embd.weight" {
            weight.cols() == b.rows() && weight.rows() == a.rows() && a.cols() == b.cols()
        } else {
            weight.cols() == a.cols() && weight.rows() == b.rows() && a.rows() == b.cols()
        };
        if !valid {
            return Err(format!("adapter matrix dimensions do not match {target}"));
        }
        weights.push(Lora {
            target,
            a,
            b,
            alpha,
            scale: 1.0,
        });
    }
    let mut resident = add(
        metadata.owned_bytes()?,
        mul(invocation_tokens.capacity(), 4)?,
    )?;
    resident = add(
        resident,
        mul(weights.capacity(), std::mem::size_of::<Lora>())?,
    )?;
    for weight in &weights {
        resident = add(
            resident,
            add(
                weight.target.capacity(),
                add(
                    weight.a.owned_bytes().map_err(|e| e.to_string())?,
                    weight.b.owned_bytes().map_err(|e| e.to_string())?,
                )?,
            )?,
        )?;
    }
    require(resident)?;
    let handle = Box::new(AdapterHandle {
        model: Arc::downgrade(&data),
        data: Arc::new(AdapterData {
            weights,
            metadata,
            invocation_tokens,
            bytes: resident,
        }),
    });
    let mut handles = registry(&data)?;
    if handles.len() >= MAX_MODEL_ADAPTERS {
        return Err("model adapter capacity reached".into());
    }
    let aggregate = handles
        .values()
        .try_fold(resident, |total, &bytes| add(total, bytes))?;
    require(aggregate)?;
    let pointer = Box::into_raw(handle);
    handles.insert(pointer as usize, resident);
    Ok(pointer)
}
unsafe fn free(adapter: *mut AdapterHandle) -> Result<()> {
    if adapter.is_null() {
        return Ok(());
    }
    let handle = unsafe { reference(adapter)? };
    if let Some(model) = handle.model.upgrade()
        && registry(&model)?.remove(&(adapter as usize)).is_none()
    {
        return Err("adapter handle is not registered".into());
    }
    // SAFETY: caller passes the live owned registered handle exactly once.
    unsafe {
        drop(Box::from_raw(adapter));
    }
    Ok(())
}
fn profile() -> Result<Profile> {
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "windows",
        target_env = "msvc"
    )) {
        Ok(Profile::Msvc)
    } else if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        Ok(Profile::Gnu)
    } else {
        Err("native adapter iteration order requires measured x64 MSVC or GNU".into())
    }
}
impl ContextAdapters {
    fn compose(&self, runtime: &llama_model_runtime::NativeContext) -> Result<AdapterSet> {
        let mut bytes = mul(self.active.capacity(), std::mem::size_of::<Active>())?;
        bytes = add(
            bytes,
            mul(
                self.retained_control.len(),
                128 + std::mem::size_of::<Vec<f32>>(),
            )?,
        )?;
        let mut count = 0;
        for active in &self.active {
            bytes = add(bytes, active.data.bytes)?;
            count = add(count, active.data.weights.len())?;
        }
        for values in &self.retained_control {
            bytes = add(bytes, mul(values.len(), 4)?)?;
        }
        // AdapterData includes metadata and all weight storage, so doubling
        // conservatively preflights the original plus its execution copy.
        let prospective = add(bytes, mul(count, std::mem::size_of::<Lora>())?)?;
        require(add(bytes, prospective)?)?;
        runtime
            .preflight_adapter_storage(prospective)
            .map_err(|e| e.to_string())?;
        let mut result = AdapterSet {
            lora: reserve(count)?,
            control_vectors: BTreeMap::new(),
        };
        for active in &self.active {
            for item in &active.data.weights {
                result.lora.push(Lora {
                    target: item.target.clone(),
                    a: item.a.try_clone().map_err(|e| e.to_string())?,
                    b: item.b.try_clone().map_err(|e| e.to_string())?,
                    alpha: item.alpha,
                    scale: active.scale,
                });
            }
        }
        if let Some((start, end)) = self.control_range {
            for (layer, values) in self.retained_control.iter().enumerate().skip(1) {
                if layer as i32 >= start && layer as i32 <= end {
                    result.control_vectors.insert(layer, copy(values)?);
                }
            }
        }
        require(result.owned_bytes().map_err(|e| e.to_string())?)?;
        Ok(result)
    }
    fn clone_control(&self) -> Result<Vec<Vec<f32>>> {
        let mut result = reserve(self.retained_control.len())?;
        for row in &self.retained_control {
            result.push(copy(row)?);
        }
        Ok(result)
    }
}
unsafe fn set_lora(
    context: *mut ContextHandle,
    adapters: *const *mut AdapterHandle,
    count: usize,
    scales: *const f32,
) -> Result<i32> {
    if count > MAX_ADAPTERS {
        return Err("native context adapter list exceeds 256 handles".into());
    }
    let context = unsafe { reference_mut(context)? };
    let handles = unsafe { input_slice(adapters, count)? };
    let scales = unsafe { input_slice(scales, count)? };
    if scales.iter().any(|scale| !scale.is_finite()) {
        return Err("adapter scales must be finite".into());
    }
    // Preserve the frozen no-change fast path, including container iteration.
    let mut same = scales.iter().filter(|scale| **scale != 0.0).count()
        == context.adapters.active.len()
        && handles
            .iter()
            .zip(scales)
            .filter(|(_, scale)| **scale != 0.0)
            .all(|(&pointer, &scale)| {
                context
                    .adapters
                    .active
                    .iter()
                    .any(|a| a.identity == pointer as usize && a.scale == scale)
            });
    if same {
        // Active execution owns data after manual free. If an allocator reuses
        // that handle address, a new data identity must replace the old weights.
        for (&pointer, &scale) in handles
            .iter()
            .zip(scales)
            .filter(|(_, scale)| **scale != 0.0)
        {
            let handle = unsafe { reference(pointer)? };
            let current = context
                .adapters
                .active
                .iter()
                .find(|a| a.identity == pointer as usize && a.scale == scale)
                .ok_or("active adapter identity changed")?;
            if !Arc::ptr_eq(&current.data, &handle.data) {
                same = false;
                break;
            }
        }
        if same {
            return Ok(0);
        }
    }
    let mut active = reserve(count)?;
    for (&pointer, &scale) in handles.iter().zip(scales) {
        if scale == 0.0
            || active
                .iter()
                .any(|a: &Active| a.identity == pointer as usize)
        {
            continue;
        }
        let handle = unsafe { reference(pointer)? };
        let model = handle
            .model
            .upgrade()
            .ok_or("adapter model has been released")?;
        if !Arc::ptr_eq(&model, &context.model_handle.data) {
            return Err("adapter belongs to a different base model".into());
        }
        if !model.tensor_names.contains("output.weight")
            && handle
                .data
                .weights
                .iter()
                .any(|w| w.target == "token_embd.weight")
        {
            return Err("embedding LoRA with a tied output head reaches an incompatible frozen matmul; use an untied output model".into());
        }
        active.push(Active {
            identity: pointer as usize,
            scale,
            data: handle.data.clone(),
        });
    }
    let ids: Vec<_> = active.iter().map(|a| a.identity).collect();
    let order = pointer_map_order(&ids, profile()?)?;
    let mut ordered = reserve(active.len())?;
    for index in order {
        ordered.push(active[index].clone());
    }
    let replacement = ContextAdapters {
        active: ordered,
        retained_control: context.adapters.clone_control()?,
        control_range: context.adapters.control_range,
    };
    let composed = replacement.compose(context.decoder()?)?;
    context
        .decoder_mut()?
        .set_adapters(composed)
        .map_err(|e| e.to_string())?;
    context.adapters = replacement;
    Ok(0)
}
unsafe fn set_control(
    context: *mut ContextHandle,
    data: *const f32,
    len: usize,
    width: i32,
    start: i32,
    end: i32,
) -> Result<i32> {
    let context = unsafe { reference_mut(context)? };
    if data.is_null() {
        let replacement = ContextAdapters {
            active: context.adapters.active.clone(),
            retained_control: context.adapters.clone_control()?,
            control_range: None,
        };
        let composed = replacement.compose(context.decoder()?)?;
        context
            .decoder_mut()?
            .set_adapters(composed)
            .map_err(|e| e.to_string())?;
        context.adapters = replacement;
        return Ok(0);
    }
    if width != context.model_handle.data.width {
        return Err("control-vector width does not match model".into());
    }
    let values = unsafe { input_slice(data, len)? };
    // Only complete source rows are consumed. Trailing/extra bytes are ignored.
    let layers = context.model_handle.data.layers as usize;
    let width = usize::try_from(width).map_err(|_| "negative control-vector width")?;
    require(mul(mul(layers, width)?, 4)?)?;
    let mut retained = context.adapters.clone_control()?;
    if retained.is_empty() {
        retained = reserve(layers)?;
        for _ in 0..layers {
            let mut row = reserve(width)?;
            row.resize(width, 0.0);
            retained.push(row);
        }
    }
    for (layer, row) in retained.iter_mut().enumerate().skip(1) {
        let offset = mul(layer - 1, width)?;
        if let Some(source) = values.get(offset..add(offset, width)?) {
            if source.iter().any(|x| !x.is_finite()) {
                return Err("control-vector active row contains nonfinite values".into());
            }
            row.copy_from_slice(source);
        }
    }
    let replacement = ContextAdapters {
        active: context.adapters.active.clone(),
        retained_control: retained,
        control_range: Some((start, end)),
    };
    let composed = replacement.compose(context.decoder()?)?;
    context
        .decoder_mut()?
        .set_adapters(composed)
        .map_err(|e| e.to_string())?;
    context.adapters = replacement;
    Ok(0)
}
ffi!(
    llama_adapter_lora_init(model: *mut ModelHandle, path: *const c_char) -> *mut AdapterHandle,
    ptr::null_mut(),
    { unsafe { load(model, path) } }
);
ffi!(
    llama_adapter_lora_init_from_file_ptr(
        model: *mut ModelHandle,
        file: *mut std::ffi::c_void,
    ) -> *mut AdapterHandle,
    ptr::null_mut(),
    { unsafe { load_file_pointer(model, file) } }
);
ffi!(
    llama_adapter_lora_free(adapter: *mut AdapterHandle) -> (),
    (),
    { unsafe { free(adapter) } }
);
ffi!(
    llama_adapter_meta_count(adapter: *const AdapterHandle) -> i32,
    -1,
    { unsafe { reference(adapter)? }.data.metadata.count() }
);
ffi!(
    llama_adapter_meta_val_str(
        adapter: *const AdapterHandle,
        key: *const c_char,
        output: *mut c_char,
        capacity: usize,
    ) -> i32,
    -1,
    {
        let data = &unsafe { reference(adapter)? }.data;
        if key.is_null() {
            return Err("null adapter metadata key".into());
        }
        let key = unsafe { CStr::from_ptr(key) }.to_bytes();
        unsafe { copy_optional(data.metadata.value(key)?, output, capacity) }
    }
);
ffi!(
    llama_adapter_meta_key_by_index(
        adapter: *const AdapterHandle,
        index: i32,
        output: *mut c_char,
        capacity: usize,
    ) -> i32,
    -1,
    {
        let data = &unsafe { reference(adapter)? }.data;
        unsafe { copy_optional(data.metadata.key_at(index)?, output, capacity) }
    }
);
ffi!(
    llama_adapter_meta_val_str_by_index(
        adapter: *const AdapterHandle,
        index: i32,
        output: *mut c_char,
        capacity: usize,
    ) -> i32,
    -1,
    {
        let data = &unsafe { reference(adapter)? }.data;
        unsafe { copy_optional(data.metadata.value_at(index)?, output, capacity) }
    }
);
ffi!(
    llama_adapter_get_alora_n_invocation_tokens(adapter: *const AdapterHandle) -> u64,
    0,
    {
        if adapter.is_null() {
            return Ok(0);
        }
        Ok(unsafe { reference(adapter)? }.data.invocation_tokens.len() as u64)
    }
);
ffi!(
    llama_adapter_get_alora_invocation_tokens(adapter: *const AdapterHandle) -> *const i32,
    ptr::null(),
    {
        let tokens = &unsafe { reference(adapter)? }.data.invocation_tokens;
        Ok(if tokens.is_empty() {
            ptr::null()
        } else {
            tokens.as_ptr()
        })
    }
);
ffi!(
    llama_set_adapters_lora(
        context: *mut ContextHandle,
        adapters: *mut *mut AdapterHandle,
        count: usize,
        scales: *mut f32,
    ) -> i32,
    -1,
    { unsafe { set_lora(context, adapters, count, scales) } }
);
ffi!(
    llama_set_adapter_cvec(
        context: *mut ContextHandle,
        data: *const f32,
        len: usize,
        width: i32,
        start: i32,
        end: i32,
    ) -> i32,
    -1,
    { unsafe { set_control(context, data, len, width, start, end) } }
);

#[cfg(test)]
#[path = "adapter_reference_tests.rs"]
mod tests;
