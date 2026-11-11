use crate::*;
use llama_gguf::{GgufSet, Value};
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
    pub architecture: String,
    pub vocab: VocabHandle,
    pub train_context: i32,
    pub width: i32,
    pub layers: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub rope_base: f32,
    pub causal_attention: bool,
    pub classifier_labels: Vec<Vec<u8>>,
    pub bytes: u64,
    pub parameters: u64,
    pub ftype: i32,
    pub description: String,
    pub metadata: crate::model_metadata::MetadataCache,
    pub adapters: std::sync::Mutex<BTreeMap<usize, usize>>,
    pub tensor_names: std::collections::BTreeSet<String>,
}
impl Drop for ModelData {
    fn drop(&mut self) {
        crate::adapter::release_model_adapters(self);
    }
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
fn guess_ftype(files: &GgufSet) -> i32 {
    let mut counts = BTreeMap::new();
    let mut dominant = 0;
    let mut largest = 0;
    let mut ordered: Vec<_> = files.tensors().iter().collect();
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
fn owned_path(path: &std::path::Path) -> Result<std::path::PathBuf> {
    let mut owned = std::path::PathBuf::new();
    owned
        .try_reserve_exact(path.as_os_str().len())
        .map_err(|_| "foreign model path allocation failed")?;
    owned.push(path);
    Ok(owned)
}
unsafe fn owned_split_path(path: *const c_char) -> Result<std::path::PathBuf> {
    if path.is_null() {
        return Err("null path in explicit model split list".into());
    }
    // SAFETY: each caller-supplied path is a live NUL-terminated string.
    let bytes = unsafe { CStr::from_ptr(path) }.to_bytes();
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        owned_path(std::path::Path::new(std::ffi::OsStr::from_bytes(bytes)))
    }
    #[cfg(not(unix))]
    {
        let text = std::str::from_utf8(bytes).map_err(|_| "model split path is not UTF-8")?;
        owned_path(std::path::Path::new(text))
    }
}
pub(crate) unsafe fn load(path: *const c_char, p: LlamaModelParams) -> Result<*mut ModelHandle> {
    let vocab_only = validate_params(&p)?;
    // Retain an owned path before invoking foreign callbacks, as the original
    // C++ entry point retains its std::string. The existing UTF-8 single-file
    // entry contract is unchanged.
    let path = owned_path(std::path::Path::new(unsafe { c_string(path)? }))?;
    unsafe {
        progress(&p, 0.0)?;
    }
    let files = GgufSet::open(path).map_err(|e| e.to_string())?;
    unsafe { load_set(files, p, vocab_only) }
}
unsafe fn load_splits(
    paths: *const *const c_char,
    n_paths: usize,
    p: LlamaModelParams,
) -> Result<*mut ModelHandle> {
    if n_paths == 0 {
        return Err("explicit model split list is empty".into());
    }
    let vocab_only = validate_params(&p)?;
    // SAFETY: the foreign caller supplies n_paths live pointer elements.
    let supplied = unsafe { input_slice(paths, n_paths)? };
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(supplied.len())
        .map_err(|_| "explicit model split path list allocation failed")?;
    for &path in supplied {
        owned.push(unsafe { owned_split_path(path)? });
    }
    unsafe {
        progress(&p, 0.0)?;
    }
    let files = GgufSet::open_splits(&owned).map_err(|e| e.to_string())?;
    unsafe { load_set(files, p, vocab_only) }
}
unsafe fn load_set(
    files: GgufSet,
    p: LlamaModelParams,
    vocab_only: bool,
) -> Result<*mut ModelHandle> {
    // Vocabulary-only loading also validates every declared shard. Metadata is
    // taken only from the primary file; statistics cover the validated set.
    let m = files.metadata();
    let architecture = match m.get("general.architecture") {
        None => "llama",
        Some(value) => value
            .as_str()
            .ok_or("general.architecture must be a UTF-8 STRING")?,
    };
    // These frozen architectures share the implemented tokenizer families. Their
    // vocabulary-only source path does not load tensor hyperparameters.
    if ![
        "llama",
        "bert",
        "gpt2",
        "gptneox",
        "mpt",
        "falcon",
        "command-r",
        "refact",
        "starcoder2",
        "qwen2",
        "gemma4",
        "nomic-bert-moe",
        "baichuan",
        "phi3",
    ]
    .contains(&architecture)
    {
        return Err(format!(
            "unsupported C ABI model architecture: {architecture}"
        ));
    }
    if !vocab_only && !["llama", "bert"].contains(&architecture) {
        return Err(
            "this C ABI currently executes Llama and BERT; this architecture supports vocabulary-only loading"
                .into(),
        );
    }
    let tokenizer = Tokenizer::from_metadata(m).map_err(|e| e.to_string())?;
    // NONE has a nominal vocabulary count but no token data in the frozen API.
    let stored_tokens = if tokenizer.vocab_type() as i32 == 0 {
        0
    } else {
        tokenizer.vocabulary_size()
    };
    let mut texts = Vec::with_capacity(stored_tokens);
    let mut scores = Vec::with_capacity(texts.capacity());
    let mut attrs = Vec::with_capacity(texts.capacity());
    for id in 0..stored_tokens {
        let (text, score, _) = tokenizer
            .token_info(id as i32)
            .ok_or("missing vocabulary item")?;
        let bits = tokenizer
            .token_attributes(id as i32)
            .ok_or("missing token attributes")? as i32;
        texts.push(nul_bytes(text));
        scores.push(score);
        attrs.push(bits);
    }
    let mut special = BTreeMap::new();
    special.insert("bos", tokenizer.bos_id);
    special.insert("eos", tokenizer.eos_id);
    special.insert("sep", tokenizer.sep_id);
    special.insert("pad", tokenizer.pad_id);
    special.insert("mask", tokenizer.mask_id);
    special.insert("eot", tokenizer.eot_id);
    special.insert("eom", tokenizer.eom_id);
    special.insert("fim_pre", tokenizer.fim_pre_id);
    special.insert("fim_suf", tokenizer.fim_suf_id);
    special.insert("fim_mid", tokenizer.fim_mid_id);
    special.insert("fim_pad", tokenizer.fim_pad_id);
    special.insert("fim_rep", tokenizer.fim_rep_id);
    special.insert("fim_sep", tokenizer.fim_sep_id);
    special.insert("nl", tokenizer.newline_id());
    let train_context = if vocab_only {
        0
    } else {
        integer(m, &format!("{architecture}.context_length"), 0)?
    };
    let width = if vocab_only {
        0
    } else {
        integer(m, &format!("{architecture}.embedding_length"), 0)?
    };
    let layers = if vocab_only {
        0
    } else {
        integer(m, &format!("{architecture}.block_count"), 0)?
    };
    let heads = if vocab_only {
        0
    } else {
        integer(m, &format!("{architecture}.attention.head_count"), 0)?
    };
    let kv_heads = if vocab_only {
        0
    } else {
        integer(m, &format!("{architecture}.attention.head_count_kv"), heads)?
    };
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
    let causal_attention = if vocab_only {
        true
    } else {
        match m.get(&format!("{architecture}.attention.causal")) {
            None => true,
            Some(Value::Bool(value)) => *value,
            Some(_) => return Err("attention.causal must be a boolean".into()),
        }
    };
    let classifier_labels = if vocab_only {
        Vec::new()
    } else {
        match m.get(&format!("{architecture}.classifier.output_labels")) {
            None => Vec::new(),
            Some(value) => {
                let labels = value
                    .as_array()
                    .ok_or("classifier labels must be an array")?;
                if labels.len() > 65_536 {
                    return Err("classifier label count exceeds 65536".into());
                }
                let mut owned = Vec::new();
                owned
                    .try_reserve_exact(labels.len())
                    .map_err(|_| "classifier labels allocation failed")?;
                let mut bytes = 0usize;
                for label in labels {
                    let label = label
                        .as_bytes()
                        .ok_or("classifier labels must be byte strings")?;
                    bytes = bytes
                        .checked_add(label.len())
                        .and_then(|n| n.checked_add(1))
                        .ok_or("classifier labels size overflow")?;
                    if bytes > 16 * 1024 * 1024 {
                        return Err("classifier labels exceed 16 MiB".into());
                    }
                    let mut terminated = Vec::new();
                    terminated
                        .try_reserve_exact(label.len() + 1)
                        .map_err(|_| "classifier label allocation failed")?;
                    terminated.extend_from_slice(label);
                    terminated.push(0);
                    owned.push(terminated);
                }
                owned
            }
        }
    };
    let bytes = files.tensor_bytes();
    let parameters = files.tensor_elements();
    let ftype = if vocab_only {
        0
    } else {
        integer(m, "general.file_type", guess_ftype(&files))?
    };
    let family = if architecture == "bert" {
        match (layers, width) {
            (3, _) => "17M",
            (6, _) => "22M",
            (12, 384) => "33M",
            (12, 768) => "109M",
            (24, _) => "335M",
            _ => "?B",
        }
    } else {
        match layers {
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
        }
    };
    let description = if vocab_only {
        String::new()
    } else {
        format!("{architecture} {family} {}", file_type_name(ftype))
    };
    let metadata = crate::model_metadata::MetadataCache::build(&files)?;
    let tensor_names = files
        .tensors()
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();
    let architecture = architecture.to_owned();
    let model = if vocab_only {
        None
    } else {
        if train_context <= 0 {
            return Err("missing positive training context".into());
        }
        Some(
            // Consume the same validated handles used for the getters above.
            Model::load_from_gguf(
                files,
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
        architecture,
        vocab: VocabHandle {
            add_sep: tokenizer.add_sep,
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
        causal_attention,
        classifier_labels,
        bytes,
        parameters,
        ftype,
        description,
        metadata,
        adapters: std::sync::Mutex::new(BTreeMap::new()),
        tensor_names,
    });
    Ok(Box::into_raw(Box::new(ModelHandle { data, owned: true })))
}

ffi!(
    llama_model_load_from_file(path: *const c_char, p: LlamaModelParams) -> *mut ModelHandle,
    ptr::null_mut(),
    { unsafe { load(path, p) } }
);
ffi!(
    llama_model_load_from_splits(
        paths: *const *const c_char,
        n_paths: usize,
        p: LlamaModelParams,
    ) -> *mut ModelHandle,
    ptr::null_mut(),
    { unsafe { load_splits(paths, n_paths, p) } }
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

dimension!(llama_n_ctx_train, train_context);
dimension!(llama_n_embd, width);
dimension!(llama_n_layer, layers);

macro_rules! attention_dimension {
    ($name:ident,$field:ident) => {
        ffi!($name(model: *const ModelHandle) -> i32, 0, {
            let data = &unsafe { reference(model)? }.data;
            if data.model.is_none() {
                return Err(
                    "attention head dimensions are unavailable on a vocabulary-only model".into(),
                );
            }
            Ok(data.$field)
        });
    };
}
attention_dimension!(llama_model_n_head, heads);
attention_dimension!(llama_n_head, heads);
attention_dimension!(llama_model_n_head_kv, kv_heads);
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
        // Exact frozen public architecture classification; these values also
        // apply when the source loader stops after vocabulary-only loading.
        Ok(
            match unsafe { reference(model)? }.data.architecture.as_str() {
                "gpt2" | "mpt" | "refact" => -1,
                "llama" | "command-r" | "baichuan" => 0,
                "bert" | "gptneox" | "falcon" | "starcoder2" | "qwen2" | "gemma4"
                | "nomic-bert-moe" | "phi3" => 2,
                _ => return Err("unsupported model RoPE classification".into()),
            },
        )
    }
);
ffi!(
    llama_model_rope_freq_scale_train(model: *const ModelHandle) -> f32,
    0.0,
    {
        Ok(if unsafe { reference(model)? }.data.model.is_none() {
            0.0
        } else {
            1.0
        })
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
        let name = if name.is_null() {
            None
        } else {
            // SAFETY: the foreign caller supplies a live NUL-terminated name.
            Some(unsafe { CStr::from_ptr(name) }.to_bytes())
        };
        let prefix = b"tokenizer.chat_template";
        let length = prefix
            .len()
            .checked_add(name.map_or(0, |name| name.len().saturating_add(1)))
            .ok_or("chat template name length overflow")?;
        if length > MAX_FOREIGN_ELEMENTS {
            return Err("chat template name exceeds safety limit".into());
        }
        let mut key = Vec::new();
        key.try_reserve_exact(length)
            .map_err(|_| "chat template name allocation")?;
        key.extend_from_slice(prefix);
        if let Some(name) = name {
            key.push(b'.');
            key.extend_from_slice(name);
        }
        Ok(unsafe { reference(model)? }
            .data
            .metadata
            .chat_template(&key)
            .unwrap_or(ptr::null()))
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
// Frozen public has_encoder/has_decoder flags describe encode/decode API routing:
// BERT runs through decode in the original and reports false/true like Llama.
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

ffi!(
    llama_model_n_cls_out(model: *const ModelHandle) -> u32,
    0,
    {
        let model = unsafe { reference(model)? };
        Ok(model.data.classifier_labels.len().max(1) as u32)
    }
);
ffi!(
    llama_model_cls_label(model: *const ModelHandle, index: u32) -> *const c_char,
    ptr::null(),
    {
        let model = unsafe { reference(model)? };
        Ok(usize::try_from(index)
            .ok()
            .and_then(|index| model.data.classifier_labels.get(index))
            .map_or(ptr::null(), |label| label.as_ptr().cast()))
    }
);
