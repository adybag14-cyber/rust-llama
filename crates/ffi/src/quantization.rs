//! Typed, owned model-file quantization through the frozen public C structs.
use crate::{Result, boolean, c_string, ffi, input_slice, reference};
use llama_model_quant::{
    FileType, MetadataOverride, OverrideValue, QuantizeParams, TensorTypeOverride,
};
use llama_quant::DType;
use std::{
    collections::BTreeMap,
    ffi::{CStr, c_char},
    ptr,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub union LlamaModelKvOverrideValue {
    pub val_i64: i64,
    pub val_f64: f64,
    pub val_bool: u8,
    pub val_str: [u8; 128],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlamaModelKvOverride {
    pub tag: i32,
    pub key: [u8; 128],
    pub value: LlamaModelKvOverrideValue,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaModelTensorOverride {
    pub pattern: *const c_char,
    pub dtype: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaModelImatrixData {
    pub name: *const c_char,
    pub data: *const f32,
    pub size: usize,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlamaModelQuantizeParams {
    pub nthread: i32,
    pub ftype: i32,
    pub output_tensor_type: i32,
    pub token_embedding_type: i32,
    pub allow_requantize: u8,
    pub quantize_output_tensor: u8,
    pub only_copy: u8,
    pub pure: u8,
    pub keep_split: u8,
    pub dry_run: u8,
    pub imatrix: *const LlamaModelImatrixData,
    pub kv_overrides: *const LlamaModelKvOverride,
    pub tt_overrides: *const LlamaModelTensorOverride,
    pub prune_layers: *const i32,
    pub max_buf_size: usize,
}
impl Default for LlamaModelQuantizeParams {
    fn default() -> Self {
        Self {
            nthread: 0,
            ftype: 7,
            output_tensor_type: 44,
            token_embedding_type: 44,
            allow_requantize: 0,
            quantize_output_tensor: 1,
            only_copy: 0,
            pure: 0,
            keep_split: 0,
            dry_run: 0,
            imatrix: ptr::null(),
            kv_overrides: ptr::null(),
            tt_overrides: ptr::null(),
            prune_layers: ptr::null(),
            max_buf_size: 8 * 1024 * 1024 * 1024,
        }
    }
}
fn dtype(value: i32) -> Result<DType> {
    let id = u32::try_from(value).map_err(|_| "negative quantization tensor type")?;
    DType::from_id(id).map_err(|e| e.to_string())
}
fn optional_dtype(value: i32) -> Result<Option<DType>> {
    if value >= 44 {
        Ok(None)
    } else {
        dtype(value).map(Some)
    }
}
fn fixed_string(bytes: &[u8]) -> Result<Vec<u8>> {
    let end = bytes
        .iter()
        .position(|&b| b == 0)
        .ok_or("fixed metadata field lacks a NUL terminator")?;
    Ok(bytes[..end].to_vec())
}
unsafe fn string_bytes(pointer: *const c_char) -> Result<Vec<u8>> {
    // SAFETY: caller supplies a live NUL-terminated C string.
    unsafe {
        reference(pointer)?;
    }
    let bytes = unsafe { CStr::from_ptr(pointer) }.to_bytes();
    if bytes.len() > 64 * 1024 {
        return Err("quantization string exceeds byte budget".into());
    }
    Ok(bytes.to_vec())
}
unsafe fn owned(p: &LlamaModelQuantizeParams) -> Result<QuantizeParams> {
    let mut result = QuantizeParams {
        nthread: p.nthread,
        ftype: FileType::from_id(u32::try_from(p.ftype).map_err(|_| "negative file type")?)
            .map_err(|e| e.to_string())?,
        output_tensor_type: optional_dtype(p.output_tensor_type)?,
        token_embedding_type: optional_dtype(p.token_embedding_type)?,
        allow_requantize: boolean(p.allow_requantize)?,
        quantize_output_tensor: boolean(p.quantize_output_tensor)?,
        only_copy: boolean(p.only_copy)?,
        pure: boolean(p.pure)?,
        keep_split: boolean(p.keep_split)?,
        dry_run: boolean(p.dry_run)?,
        max_buf_size: p.max_buf_size,
        ..QuantizeParams::default()
    };
    const LIMIT: usize = 4096;
    if !p.imatrix.is_null() {
        let mut entries = BTreeMap::new();
        let mut finished = false;
        let mut total = 0usize;
        for i in 0..=LIMIT {
            // SAFETY: caller provides a live array terminated by a null name.
            let value = unsafe { reference(p.imatrix.add(i))? };
            if value.name.is_null() {
                finished = true;
                break;
            }
            if i == LIMIT {
                return Err("importance entry count exceeds budget".into());
            }
            let name = unsafe { c_string(value.name)? }.to_owned();
            let data = unsafe { input_slice(value.data, value.size)? };
            total = total
                .checked_add(data.len())
                .ok_or("importance value count overflow")?;
            if total > 128 * 1024 * 1024 {
                return Err("importance value budget exceeded".into());
            }
            if let std::collections::btree_map::Entry::Vacant(entry) = entries.entry(name) {
                let mut copy = Vec::new();
                copy.try_reserve_exact(data.len())
                    .map_err(|_| "importance allocation failed")?;
                copy.extend_from_slice(data);
                entry.insert(copy);
            }
        }
        if !finished {
            return Err("importance array lacks a sentinel".into());
        }
        result.imatrix = Some(entries);
    }
    if !p.tt_overrides.is_null() {
        let mut finished = false;
        for i in 0..=LIMIT {
            // SAFETY: caller supplies the null-pattern-terminated typed array.
            let value = unsafe { reference(p.tt_overrides.add(i))? };
            if value.pattern.is_null() {
                finished = true;
                break;
            }
            if i == LIMIT {
                return Err("tensor override count exceeds budget".into());
            }
            result.tt_overrides.push(TensorTypeOverride {
                pattern: unsafe { string_bytes(value.pattern)? },
                dtype: dtype(value.dtype)?,
            });
        }
        if !finished {
            return Err("tensor override array lacks sentinel".into());
        }
    }
    if !p.kv_overrides.is_null() {
        let mut finished = false;
        for i in 0..=LIMIT {
            // SAFETY: caller supplies the empty-key-terminated typed array.
            let value = unsafe { reference(p.kv_overrides.add(i))? };
            if value.key[0] == 0 {
                finished = true;
                break;
            }
            if i == LIMIT {
                return Err("metadata override count exceeds budget".into());
            }
            let key = fixed_string(&value.key)?;
            let typed = match value.tag {
                0 => OverrideValue::Integer(unsafe { value.value.val_i64 }),
                1 => OverrideValue::Float(unsafe { value.value.val_f64 }),
                2 => OverrideValue::Bool(boolean(unsafe { value.value.val_bool })?),
                3 => OverrideValue::String(fixed_string(unsafe { &value.value.val_str })?),
                _ => continue, // frozen quantizer warns and ignores unknown tags
            };
            result
                .kv_overrides
                .push(MetadataOverride { key, value: typed });
        }
        if !finished {
            return Err("metadata override array lacks sentinel".into());
        }
    }
    if !p.prune_layers.is_null() {
        let mut finished = false;
        for i in 0..=LIMIT {
            // SAFETY: caller supplies the -1-terminated layer array.
            let value = *unsafe { reference(p.prune_layers.add(i))? };
            if value == -1 {
                finished = true;
                break;
            }
            if i == LIMIT {
                return Err("pruning list exceeds budget".into());
            }
            result.prune_layers.push(value);
        }
        if !finished {
            return Err("pruning list lacks sentinel".into());
        }
    }
    Ok(result)
}
ffi!(llama_model_quantize_default_params()->LlamaModelQuantizeParams,LlamaModelQuantizeParams::default(),{
    Ok(LlamaModelQuantizeParams::default())
});
ffi!(
    llama_model_quantize(
        input: *const c_char,
        output: *const c_char,
        params: *const LlamaModelQuantizeParams,
    ) -> u32,
    1,
    {
        let input = unsafe { c_string(input)? };
        let output = unsafe { c_string(output)? };
        let params = unsafe { owned(reference(params)?)? };
        llama_model_quant::quantize_file(input, output, &params).map_err(|e| e.to_string())?;
        Ok(0)
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_and_layout_match_independent_original_header() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<LlamaModelQuantizeParams>(), 64);
        assert_eq!(align_of::<LlamaModelQuantizeParams>(), 8);
        assert_eq!(offset_of!(LlamaModelQuantizeParams, imatrix), 24);
        assert_eq!(offset_of!(LlamaModelQuantizeParams, kv_overrides), 32);
        assert_eq!(offset_of!(LlamaModelQuantizeParams, tt_overrides), 40);
        assert_eq!(offset_of!(LlamaModelQuantizeParams, prune_layers), 48);
        assert_eq!(offset_of!(LlamaModelQuantizeParams, max_buf_size), 56);
        let p = unsafe { llama_model_quantize_default_params() };
        assert_eq!(p.nthread, 0);
        assert_eq!(p.ftype, 7);
        assert_eq!((p.output_tensor_type, p.token_embedding_type), (44, 44));
        assert_eq!(p.quantize_output_tensor, 1);
        assert_eq!(
            [
                p.allow_requantize,
                p.only_copy,
                p.pure,
                p.keep_split,
                p.dry_run
            ],
            [0; 5]
        );
        assert_eq!(p.max_buf_size, 8 * 1024 * 1024 * 1024);
        assert!(
            p.imatrix.is_null()
                && p.kv_overrides.is_null()
                && p.tt_overrides.is_null()
                && p.prune_layers.is_null()
        );
    }
    #[test]
    fn typed_arrays_are_owned_and_duplicate_importance_names_keep_first() {
        let mut data = [1.0, 2.0];
        let other = [3.0, 4.0];
        let im = [
            LlamaModelImatrixData {
                name: c"matrix.weight".as_ptr(),
                data: data.as_ptr(),
                size: 2,
            },
            LlamaModelImatrixData {
                name: c"matrix.weight".as_ptr(),
                data: other.as_ptr(),
                size: 2,
            },
            LlamaModelImatrixData {
                name: ptr::null(),
                data: ptr::null(),
                size: 0,
            },
        ];
        let p = LlamaModelQuantizeParams {
            imatrix: im.as_ptr(),
            ..Default::default()
        };
        let copied = unsafe { owned(&p) }.unwrap();
        data[0] = 9.0;
        assert_eq!(data, [9.0, 2.0]);
        assert_eq!(copied.imatrix.unwrap()["matrix.weight"], [1.0, 2.0]);
        let p = LlamaModelQuantizeParams {
            pure: 2,
            ..Default::default()
        };
        assert!(unsafe { owned(&p) }.is_err());
    }
}
