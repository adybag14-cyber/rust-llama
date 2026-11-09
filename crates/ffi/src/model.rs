use crate::*;
use llama_gguf::{Gguf, Value};
use llama_inference::{LoadOptions, Model};
use llama_tokenizer::Tokenizer;
use std::collections::BTreeMap;
use std::ffi::c_char;
use std::sync::Arc;

pub struct VocabHandle {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) texts: Vec<Vec<u8>>,
    pub(crate) scores: Vec<f32>,
    pub(crate) attrs: Vec<i32>,
    pub(crate) add_sep: bool,
    pub(crate) special: BTreeMap<&'static str, i32>,
}
pub struct ModelHandle {
    pub(crate) data: Arc<ModelData>,
    pub(crate) owned: bool,
}
pub(crate) struct ModelData {
    pub model: Option<Model>,
    pub vocab: VocabHandle,
    pub train_context: i32,
    pub width: i32,
    pub layers: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub rope_base: f32,
    pub bytes: u64,
    pub parameters: u64,
    pub ftype: i32,
    pub description: String,
    pub strings: BTreeMap<String, Vec<u8>>,
}
fn integer(meta: &BTreeMap<String, Value>, key: &str, default: i32) -> Result<i32> {
    match meta.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_i64()
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| format!("invalid integer metadata: {key}")),
    }
}
fn attr(kind: i32) -> Result<i32> {
    match kind {
        0 => Ok(0),
        1 => Ok(4),
        2 => Ok(1),
        3 => Ok(8),
        4 => Ok(16),
        5 => Ok(2),
        6 => Ok(32),
        _ => Err("invalid tokenizer token type".into()),
    }
}
fn nul_bytes(value: &str) -> Vec<u8> {
    let mut out = value.as_bytes().to_vec();
    out.push(0);
    out
}
fn file_type_name(ftype: i32) -> String {
    let name = match ftype & !1024 {
        0 => "all F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K - Medium",
        11 => "Q3_K - Small",
        12 => "Q3_K - Medium",
        13 => "Q3_K - Large",
        14 => "Q4_K - Small",
        15 => "Q4_K - Medium",
        16 => "Q5_K - Small",
        17 => "Q5_K - Medium",
        18 => "Q6_K",
        19 => "IQ2_XXS - 2.0625 bpw",
        20 => "IQ2_XS - 2.3125 bpw",
        21 => "Q2_K - Small",
        22 => "IQ3_XS - 3.3 bpw",
        23 => "IQ3_XXS - 3.0625 bpw",
        24 => "IQ1_S - 1.5625 bpw",
        25 => "IQ4_NL - 4.5 bpw",
        26 => "IQ3_S - 3.4375 bpw",
        27 => "IQ3_S mix - 3.66 bpw",
        28 => "IQ2_S - 2.5 bpw",
        29 => "IQ2_M - 2.7 bpw",
        30 => "IQ4_XS - 4.25 bpw",
        31 => "IQ1_M - 1.75 bpw",
        32 => "BF16",
        36 => "TQ1_0 - 1.69 bpw ternary",
        37 => "TQ2_0 - 2.06 bpw ternary",
        38 => "MXFP4 MoE",
        39 => "NVFP4",
        40 => "Q1_0",
        41 => "Q2_0",
        42 => "F8_E4M3",
        _ => "unknown, may not work",
    };
    format!(
        "{}{name}",
        if ftype & 1024 != 0 { "(guessed) " } else { "" }
    )
}
fn guess_ftype<R: std::io::Read + std::io::Seek>(file: &Gguf<R>) -> i32 {
    let mut counts = BTreeMap::new();
    let mut dominant = 0;
    let mut largest = 0;
    let mut ordered: Vec<_> = file.tensors.iter().collect();
    ordered.sort_by(|a, b| a.name.cmp(&b.name));
    for tensor in ordered {
        let n = counts.entry(tensor.dtype as u32).or_insert(0usize);
        *n += 1;
        if *n > largest {
            largest = *n;
            dominant = tensor.dtype as u32;
        }
    }
    let ty = match dominant {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        6 => 8,
        7 => 9,
        8 => 7,
        10 => 10,
        11 => 12,
        12 => 15,
        13 => 17,
        14 => 18,
        16 => 19,
        17 => 20,
        18 => 23,
        19 => 24,
        20 => 25,
        21 => 26,
        22 => 28,
        23 => 30,
        29 => 31,
        30 => 32,
        34 => 36,
        35 => 37,
        40 => 39,
        41 => 40,
        42 => 41,
        43 => 42,
        _ => 0,
    };
    ty | 1024
}
fn validate_params(p: &LlamaModelParams) -> Result<bool> {
    for value in [
        p.vocab_only,
        p.check_tensors,
        p.use_extra_bufts,
        p.no_host,
        p.no_alloc,
        p.load_mtp,
    ] {
        boolean(value)?;
    }
    if !p.devices.is_null()
        || !p.tensor_buft_overrides.is_null()
        || !p.tensor_split.is_null()
        || !p.kv_overrides.is_null()
    {
        return Err(
            "device lists, tensor overrides, tensor split and metadata overrides are not supported"
                .into(),
        );
    }
    if !(-1..=0).contains(&p.n_gpu_layers) || p.main_gpu != 0 || !(0..=3).contains(&p.split_mode) {
        return Err("this C ABI currently executes on the scalar CPU; requested GPU placement is unavailable".into());
    }
    if ![-1, 0].contains(&p.load_mode) || ![0, 1].contains(&p.lazy_mode) {
        return Err("this C ABI supports eager file reads, not mmap/mlock/direct I/O or forced lazy loading".into());
    }
    if p.no_host != 0 || p.no_alloc != 0 || p.load_mtp != 0 {
        return Err("no_host, no_alloc and MTP loading are not implemented".into());
    }
    Ok(p.vocab_only != 0)
}
unsafe fn progress(p: &LlamaModelParams, amount: f32) -> Result<()> {
    if let Some(callback) = p.progress_callback {
        // SAFETY: foreign caller supplies a valid, non-reentrant callback and userdata.
        if !unsafe { callback(amount, p.progress_callback_user_data) } {
            return Err("model loading cancelled by progress callback".into());
        }
    }
    Ok(())
}
pub(crate) unsafe fn load(path: *const c_char, p: LlamaModelParams) -> Result<*mut ModelHandle> {
    let vocab_only = validate_params(&p)?;
    let path = unsafe { c_string(path)? };
    unsafe {
        progress(&p, 0.0)?;
    }
    let file = Gguf::open(path).map_err(|e| e.to_string())?;
    let m = &file.metadata;
    if !vocab_only && m.get("general.architecture").and_then(Value::as_str) != Some("llama") {
        return Err("this C ABI model loader currently supports dense Llama only".into());
    }
    let tokenizer = Tokenizer::from_metadata(m).map_err(|e| e.to_string())?;
    let mut texts = Vec::with_capacity(tokenizer.vocabulary_size());
    let mut scores = Vec::with_capacity(texts.capacity());
    let mut attrs = Vec::with_capacity(texts.capacity());
    let original_kinds = m.get("tokenizer.ggml.token_type").and_then(Value::as_array);
    for id in 0..tokenizer.vocabulary_size() {
        let (text, score, kind) = tokenizer
            .token_info(id as i32)
            .ok_or("missing vocabulary item")?;
        let original = original_kinds
            .and_then(|v| v.get(id))
            .and_then(Value::as_i64)
            .unwrap_or(1);
        let mut bits = attr(i32::try_from(original).map_err(|_| "invalid token type")?)?;
        if kind == 3 {
            bits |= 8;
        }
        texts.push(nul_bytes(text));
        scores.push(score);
        attrs.push(bits);
    }
    let mut special = BTreeMap::new();
    special.insert("bos", tokenizer.bos_id);
    special.insert("eos", tokenizer.eos_id);
    for (name, key) in [
        ("eot", "tokenizer.ggml.eot_token_id"),
        ("sep", "tokenizer.ggml.seperator_token_id"),
        ("pad", "tokenizer.ggml.padding_token_id"),
        ("mask", "tokenizer.ggml.mask_token_id"),
    ] {
        // Native loading retains the default when a metadata id is out of range,
        // including the UINT32_MAX sentinel used by stories260K.
        let value = match m.get(key) {
            None => -1,
            Some(value) => {
                let id = value
                    .as_u64()
                    .ok_or_else(|| format!("special id must be unsigned: {key}"))?;
                if id < tokenizer.vocabulary_size() as u64 {
                    id as i32
                } else {
                    -1
                }
            }
        };
        special.insert(name, value);
    }
    // Native SPM byte_to_token uses the literal byte token, falling back to PAD.
    let nl = texts
        .iter()
        .position(|t| t.as_slice() == b"<0x0A>\0")
        .map_or(special["pad"], |i| i as i32);
    special.insert("nl", nl);
    // The source auto-detects EOT spellings. Multiple matching spellings have no
    // stable precedence in its unordered map; reject ambiguity rather than guess.
    if special["eot"] == -1 {
        let spellings = [
            "<|eot_id|>",
            "<|im_end|>",
            "<|end|>",
            "<end_of_turn>",
            "<|endoftext|>",
            "<|end_of_text|>",
            "<EOT>",
            "_<EOT>",
            "[EOT]",
            "<｜end▁of▁sentence｜>",
            "<end_of_utterance>",
        ];
        let matches: Vec<_> = (0..tokenizer.vocabulary_size())
            .filter(|&id| spellings.contains(&tokenizer.token_info(id as i32).unwrap().0))
            .collect();
        if matches.len() > 1 {
            return Err("ambiguous auto-detected EOT ids; explicit eot_token_id required".into());
        }
        if let Some(&id) = matches.first() {
            special.insert("eot", id as i32);
        }
    }
    let train_context = integer(m, "llama.context_length", 0)?;
    let width = integer(m, "llama.embedding_length", 0)?;
    let layers = integer(m, "llama.block_count", 0)?;
    let heads = integer(m, "llama.attention.head_count", 0)?;
    let kv_heads = integer(m, "llama.attention.head_count_kv", heads)?;
    let rope_base = m
        .get("llama.rope.freq_base")
        .and_then(Value::as_f64)
        .unwrap_or(10000.0) as f32;
    if [train_context, width, layers, heads, kv_heads]
        .iter()
        .any(|v| *v < 0)
    {
        return Err("negative model dimension metadata".into());
    }
    let mut bytes = 0u64;
    let mut parameters = 0u64;
    for tensor in &file.tensors {
        bytes = bytes
            .checked_add(tensor.byte_len)
            .ok_or("model size overflow")?;
        parameters = parameters
            .checked_add(tensor.elements().map_err(|e| e.to_string())? as u64)
            .ok_or("parameter count overflow")?;
    }
    let ftype = integer(m, "general.file_type", guess_ftype(&file))?;
    let family = match layers {
        16 | 22 => "1B",
        26 | 28 => "3B",
        30 => "256M",
        32 if tokenizer.vocabulary_size() == 49152 => "3B",
        32 if tokenizer.vocabulary_size() < 40000 => "7B",
        32 | 36 => "8B",
        40 => "13B",
        48 => "34B",
        60 => "30B",
        80 if heads == kv_heads => "65B",
        80 => "70B",
        _ => "?B",
    };
    let description = format!("llama {family} {}", file_type_name(ftype));
    let strings = m
        .iter()
        .filter_map(|(key, value)| value.as_str().map(|v| (key.clone(), nul_bytes(v))))
        .collect();
    let model = if vocab_only {
        None
    } else {
        if train_context <= 0 {
            return Err("missing positive training context".into());
        }
        Some(
            Model::load_with_options(
                path,
                &LoadOptions {
                    context_length: train_context as usize,
                    ..LoadOptions::default()
                },
            )
            .map_err(|e| e.to_string())?,
        )
    };
    unsafe {
        progress(&p, 1.0)?;
    }
    let data = Arc::new(ModelData {
        model,
        vocab: VocabHandle {
            add_sep: m
                .get("tokenizer.ggml.add_sep_token")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            tokenizer,
            texts,
            scores,
            attrs,
            special,
        },
        train_context,
        width,
        layers,
        heads,
        kv_heads,
        rope_base,
        bytes,
        parameters,
        ftype,
        description,
        strings,
    });
    Ok(Box::into_raw(Box::new(ModelHandle { data, owned: true })))
}

ffi!(
    llama_model_load_from_file(path: *const c_char, p: LlamaModelParams) -> *mut ModelHandle,
    ptr::null_mut(),
    { unsafe { load(path, p) } }
);
ffi!(
    llama_load_model_from_file(path: *const c_char, p: LlamaModelParams) -> *mut ModelHandle,
    ptr::null_mut(),
    { unsafe { load(path, p) } }
);
pub(crate) unsafe fn free(model: *mut ModelHandle) -> Result<()> {
    if model.is_null() {
        return Ok(());
    }
    if !unsafe { reference(model)? }.owned {
        return Err("cannot free a borrowed context model handle".into());
    }
    // SAFETY: owning handle is returned once by load and must be freed exactly once.
    unsafe {
        drop(Box::from_raw(model));
    }
    Ok(())
}
ffi!(llama_model_free(model: *mut ModelHandle) -> (), (), {
    unsafe { free(model) }
});
ffi!(llama_free_model(model: *mut ModelHandle) -> (), (), {
    unsafe { free(model) }
});
ffi!(
    llama_model_get_vocab(model: *const ModelHandle) -> *const VocabHandle,
    ptr::null(),
    { Ok(&unsafe { reference(model)? }.data.vocab) }
);
macro_rules! dimension {
    ($name:ident,$field:ident) => {
        ffi!($name(model: *const ModelHandle) -> i32, 0, {
            Ok(unsafe { reference(model)? }.data.$field)
        });
    };
}
dimension!(llama_model_n_ctx_train, train_context);
dimension!(llama_model_n_embd, width);
dimension!(llama_model_n_embd_inp, width);
dimension!(llama_model_n_embd_out, width);
dimension!(llama_model_n_layer, layers);
dimension!(llama_model_n_head, heads);
dimension!(llama_model_n_head_kv, kv_heads);
dimension!(llama_n_ctx_train, train_context);
dimension!(llama_n_embd, width);
dimension!(llama_n_layer, layers);
dimension!(llama_n_head, heads);
ffi!(
    llama_model_n_layer_nextn(model: *const ModelHandle) -> i32,
    0,
    {
        unsafe {
            reference(model)?;
        }
        Ok(0)
    }
);
ffi!(llama_model_n_swa(model: *const ModelHandle) -> i32, 0, {
    unsafe {
        reference(model)?;
    }
    Ok(0)
});
ffi!(
    llama_model_rope_type(model: *const ModelHandle) -> i32,
    -1,
    {
        unsafe {
            reference(model)?;
        }
        Ok(0)
    }
);
ffi!(
    llama_model_rope_freq_scale_train(model: *const ModelHandle) -> f32,
    0.0,
    {
        unsafe {
            reference(model)?;
        }
        Ok(1.0)
    }
);
ffi!(
    llama_model_desc(model: *const ModelHandle, buf: *mut c_char, size: usize) -> i32,
    -1,
    { unsafe { copy_string(&reference(model)?.data.description, buf, size) } }
);
ffi!(llama_model_ftype(model: *const ModelHandle) -> i32, -1, {
    Ok(unsafe { reference(model)? }.data.ftype)
});
ffi!(llama_model_size(model: *const ModelHandle) -> u64, 0, {
    Ok(unsafe { reference(model)? }.data.bytes)
});
ffi!(llama_model_n_params(model: *const ModelHandle) -> u64, 0, {
    Ok(unsafe { reference(model)? }.data.parameters)
});
ffi!(
    llama_model_chat_template(model: *const ModelHandle, name: *const c_char) -> *const c_char,
    ptr::null(),
    {
        let key = if name.is_null() {
            "tokenizer.chat_template".into()
        } else {
            format!("tokenizer.chat_template.{}", unsafe { c_string(name)? })
        };
        Ok(unsafe { reference(model)? }
            .data
            .strings
            .get(&key)
            .map_or(ptr::null(), |v| v.as_ptr().cast()))
    }
);
macro_rules! flag {
    ($name:ident,$value:expr) => {
        ffi!($name(model: *const ModelHandle) -> bool, false, {
            unsafe {
                reference(model)?;
            }
            Ok($value)
        });
    };
}
flag!(llama_model_has_encoder, false);
flag!(llama_model_has_decoder, true);
flag!(llama_model_is_recurrent, false);
flag!(llama_model_is_hybrid, false);
flag!(llama_model_is_diffusion, false);
ffi!(
    llama_model_decoder_start_token(model: *const ModelHandle) -> i32,
    -1,
    {
        unsafe {
            reference(model)?;
        }
        Ok(-1)
    }
);
