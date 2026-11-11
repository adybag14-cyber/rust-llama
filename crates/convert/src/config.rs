//! Frozen dense Llama configuration precedence and GGUF metadata.
use crate::frozen::{AUTO_CONFIG_PATTERNS, LLAMA_DEFAULTS};
use crate::qwen_tables::{QWEN2_DEFAULTS, QWEN3_DEFAULTS};
use crate::util::{
    JsonBudget, Metadata, integer, object, optional_text, present, real, text, truth, u32_value,
};
use crate::{Error, Limits, OutputType, Result, string};
use llama_checkpoint::{Object, Value as Json};
use llama_gguf::RawValue as Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Architecture {
    Llama,
    Qwen2,
    Qwen3,
    Gpt2,
    GptNeoX,
}
impl Architecture {
    pub fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Gpt2 => "gpt2",
            Self::GptNeoX => "gptneox",
        }
    }
    pub fn tensor_names(self) -> &'static [crate::frozen::TensorName] {
        match self {
            Self::Llama => crate::frozen::TENSOR_NAMES,
            Self::Qwen2 => crate::qwen_tables::QWEN2_TENSOR_NAMES,
            Self::Qwen3 => crate::qwen_tables::QWEN3_TENSOR_NAMES,
            Self::Gpt2 => crate::gpt_tables::GPT2_TENSOR_NAMES,
            Self::GptNeoX => crate::gpt_tables::GPTNEOX_TENSOR_NAMES,
        }
    }
}
pub(crate) struct Config {
    raw: Json,
    defaults: Option<Json>,
    computed_head: Option<Json>,
    nested_key: Option<&'static str>,
    nested_thinker: bool,
    pub architecture: String,
    pub family: Architecture,
    pub layers: usize,
}
impl Config {
    pub fn new(
        raw: Json,
        input_path: &str,
        budget: &mut JsonBudget,
        limits: &Limits,
    ) -> Result<Self> {
        let root = object(&raw)?;
        let model_type = optional_text(root.get("model_type"))?;
        let selected = match model_type {
            Some(name) => Some(name),
            // AutoConfig tests key presence; explicit null does not enable
            // directory-pattern inference and falls back to raw hparams.
            None if root.get("model_type").is_some() => None,
            None => AUTO_CONFIG_PATTERNS
                .iter()
                .copied()
                .find(|name| input_path.contains(name)),
        };
        if selected.is_some_and(|name| {
            !matches!(name, "llama" | "qwen2" | "qwen3" | "gpt2" | "gpt_neox")
                && AUTO_CONFIG_PATTERNS.contains(&name)
        }) {
            return Err(Error::Unsupported(
                "AutoConfig class outside the admitted dense converter",
            ));
        }
        let mut normalized = matches!(
            selected,
            Some("llama" | "qwen2" | "qwen3" | "gpt2" | "gpt_neox")
        );
        let mut head = None;
        if normalized {
            normalized = auto_config_valid(root, selected)
                && (!matches!(selected, Some("qwen2" | "qwen3"))
                    || crate::qwen::valid_auto_config(root))
                && (selected != Some("gpt_neox") || crate::gpt::valid_neox_config(root));
            if normalized && selected == Some("llama") && present(root.get("head_dim")).is_none() {
                let hidden = root
                    .get("hidden_size")
                    .map(integer)
                    .transpose()?
                    .unwrap_or(4096);
                let heads = root
                    .get("num_attention_heads")
                    .map(integer)
                    .transpose()?
                    .unwrap_or(32);
                if heads == 0 {
                    normalized = false;
                } else {
                    // Python // rounds toward negative infinity.
                    let quotient = floor_div(hidden, heads)?;
                    head = Some(budget.parse(quotient.to_string().as_bytes())?);
                }
            }
        }
        let defaults = if normalized {
            Some(
                budget.parse(
                    match selected {
                        Some("qwen2") => QWEN2_DEFAULTS,
                        Some("qwen3") => QWEN3_DEFAULTS,
                        Some("gpt2") => crate::gpt_tables::GPT2_DEFAULTS,
                        Some("gpt_neox") => crate::gpt_tables::GPTNEOX_DEFAULTS,
                        _ => LLAMA_DEFAULTS,
                    }
                    .as_bytes(),
                )?,
            )
        } else {
            None
        };
        if !normalized {
            head = None;
        }
        let mut config = Self {
            raw,
            defaults,
            computed_head: head,
            nested_key: None,
            nested_thinker: false,
            architecture: String::new(),
            family: Architecture::Llama,
            layers: 0,
        };
        // This is the actual order of the source's config-key rewrites.
        for key in [
            "text_config",
            "llm_config",
            "lm_config",
            "thinker_config",
            "language_config",
            "lfm",
        ] {
            if config.root().get(key).is_some() {
                config.nested_key = Some(key);
                config.nested_thinker = key == "thinker_config";
            }
        }
        let mut architecture = config
            .root_value("architectures")
            .filter(|v| !matches!(v, Json::Null))
            .map(first_architecture)
            .transpose()?
            .flatten();
        if let Some(value) = config
            .nested()?
            .and_then(|o| present(o.get("architectures")))
        {
            architecture = Some(
                first_architecture(value)?.ok_or(Error::Invalid("empty nested architectures"))?,
            );
        }
        let architecture = architecture.ok_or(Error::Invalid("missing model architecture"))?;
        let family = match architecture {
            "LlamaForCausalLM" | "LLaMAForCausalLM" | "LlamaModel" => Architecture::Llama,
            "Qwen2ForCausalLM" | "Qwen2Model" => Architecture::Qwen2,
            "Qwen3ForCausalLM" | "Qwen3Model" => Architecture::Qwen3,
            "GPT2LMHeadModel" => Architecture::Gpt2,
            "GPTNeoXForCausalLM" => Architecture::GptNeoX,
            _ => {
                return Err(Error::Unsupported(
                    "architecture is not an admitted dense conversion",
                ));
            }
        };
        config.architecture = string(architecture)?;
        config.family = family;
        for key in [
            "quantization_config",
            "transformer_layer_config",
            "mtp",
            "mtp_num_hidden_layers",
            "ssm_cfg",
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
        ] {
            if config.get(key)?.is_some_and(truth) {
                return Err(Error::Unsupported(
                    "prequantized, speculative, recurrent or expert conversion",
                ));
            }
        }
        config.layers = usize::try_from(integer(config.required(&[
            "n_layers",
            "num_hidden_layers",
            "n_layer",
            "num_layers",
        ])?)?)
        .map_err(|_| Error::Invalid("negative layer count"))?;
        if config.layers > limits.max_layers {
            return Err(Error::Limit("layer count"));
        }
        Ok(config)
    }
    pub fn root(&self) -> &Object {
        self.raw.as_object().expect("validated root")
    }
    pub fn auto_tokenizer_fallback(&self) -> Result<&str> {
        if self.defaults.is_none()
            || self.root().get("configuration_files").is_some()
            || self.root().get("auto_map").is_some()
        {
            return Err(Error::Unsupported(
                "AutoTokenizer requires an admitted resolved root AutoConfig without indirection",
            ));
        }
        if let Some(value) = present(self.root().get("tokenizer_class")) {
            return text(value);
        }
        match self.root_value_model_type() {
            Some("qwen2" | "qwen3") => Ok("Qwen2TokenizerFast"),
            Some("llama") => Ok("LlamaTokenizerFast"),
            Some("gpt2") => Ok("GPT2TokenizerFast"),
            Some("gpt_neox") => Ok("GPTNeoXTokenizerFast"),
            _ => Err(Error::Unsupported("AutoTokenizer root configuration class")),
        }
    }
    fn root_value_model_type(&self) -> Option<&str> {
        self.defaults
            .as_ref()?
            .as_object()?
            .get("model_type")?
            .as_text()?
            .as_str()
    }
    fn root_value(&self, key: &str) -> Option<&Json> {
        if self.root_value_model_type() == Some("gpt2") {
            // GPT2Config initializes n_* arguments first, then applies the
            // inherited attribute-map aliases from **kwargs. These aliases
            // therefore overwrite an explicitly supplied n_* value.
            let alias = match key {
                "n_embd" => Some("hidden_size"),
                "n_positions" => Some("max_position_embeddings"),
                "n_head" => Some("num_attention_heads"),
                "n_layer" => Some("num_hidden_layers"),
                _ => None,
            };
            if let Some(value) = alias.and_then(|name| self.root().get(name)) {
                return Some(value);
            }
        }
        if self.defaults.is_some() {
            if key == "head_dim"
                && let Some(value) = self.computed_head.as_ref()
            {
                return Some(value);
            }
            if key == "num_key_value_heads"
                && (matches!(self.root_value_model_type(), Some("llama"))
                    && present(self.root().get(key)).is_none()
                    || matches!(self.root().get(key), Some(Json::Null)))
            {
                return self.root().get("num_attention_heads").or_else(|| {
                    self.defaults
                        .as_ref()?
                        .as_object()?
                        .get("num_attention_heads")
                });
            }
        }
        self.root()
            .get(key)
            .or_else(|| self.defaults.as_ref()?.as_object()?.get(key))
    }
    fn nested(&self) -> Result<Option<&Object>> {
        let Some(key) = self.nested_key else {
            return Ok(None);
        };
        let value = self
            .root()
            .get(key)
            .ok_or(Error::Invalid("missing nested config"))?;
        let value = if self.nested_thinker {
            object(value)?
                .get("text_config")
                .ok_or(Error::Invalid("thinker text config is missing"))?
        } else {
            value
        };
        Ok(Some(object(value)?))
    }
    pub fn get(&self, key: &str) -> Result<Option<&Json>> {
        Ok(self
            .nested()?
            .and_then(|o| o.get(key))
            .or_else(|| self.root_value(key)))
    }
    pub fn find(&self, names: &[&str]) -> Result<Option<&Json>> {
        for name in names {
            if let Some(value) = self.get(name)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }
    pub fn required(&self, names: &[&str]) -> Result<&Json> {
        self.find(names)?
            .ok_or(Error::Invalid("missing required configuration parameter"))
    }
    pub fn head_count(&self) -> Result<usize> {
        usize::try_from(integer(
            self.required(&["n_heads", "num_attention_heads"])?,
        )?)
        .map_err(|_| Error::Invalid("negative query head count"))
    }
    pub fn kv_head_count(&self) -> Result<Option<usize>> {
        self.find(&["n_kv_heads", "num_key_value_heads"])?
            .filter(|v| !matches!(v, Json::Null))
            .map(|v| {
                usize::try_from(integer(v)?).map_err(|_| Error::Invalid("negative KV head count"))
            })
            .transpose()
    }
    pub fn rope(&self) -> Result<Rope<'_>> {
        let root = match self.get("rope_parameters")? {
            Some(value) => value,
            None => self.get("rope_scaling")?.unwrap_or(&Json::Null),
        };
        let root = if truth(root) {
            Some(object(root)?)
        } else {
            None
        };
        let isolated = root.is_some_and(|r| {
            r.get("full_attention").is_some() || r.get("sliding_attention").is_some()
        });
        let full = match root.and_then(|r| r.get("full_attention")) {
            Some(value) => Some(object(value)?),
            None => root,
        };
        let from_normalized_root = self.defaults.is_some()
            && self
                .nested()?
                .is_none_or(|n| n.get("rope_scaling").is_none())
            && self.get("rope_parameters")?.is_none();
        Ok(Rope {
            config: self,
            root,
            full,
            isolated,
            from_normalized_root,
        })
    }
    pub fn metadata(&self, out: &mut Metadata, output: OutputType) -> Result<()> {
        if matches!(self.family, Architecture::Gpt2 | Architecture::GptNeoX) {
            return crate::gpt::metadata(self, out, output);
        }
        let mut fields = Metadata::default();
        self.common_metadata(&mut fields, output)?;
        for (name, value) in fields.0 {
            if let Some(suffix) = name.strip_prefix("llama.") {
                let mut key = string(self.family.name())?;
                key.try_reserve(suffix.len() + 1)
                    .map_err(|_| Error::Limit("metadata key"))?;
                key.push('.');
                key.push_str(suffix);
                out.put(&key, value)?;
            } else {
                out.put(&name, value)?;
            }
        }
        Ok(())
    }
    fn common_metadata(&self, out: &mut Metadata, output: OutputType) -> Result<()> {
        out.uint(
            "llama.block_count",
            u32::try_from(self.layers).map_err(|_| Error::Limit("layer count"))?,
        )?;
        for (key, fields) in [
            (
                "llama.context_length",
                &[
                    "max_position_embeddings",
                    "n_ctx",
                    "n_positions",
                    "max_length",
                    "max_sequence_length",
                    "model_max_length",
                ][..],
            ),
            (
                "llama.embedding_length",
                &["hidden_size", "n_embd", "dim"][..],
            ),
            (
                "llama.feed_forward_length",
                &[
                    "prefix_dense_intermediate_size",
                    "dense_intermediate_size",
                    "intermediate_size",
                    "n_inner",
                    "hidden_dim",
                ][..],
            ),
            (
                "llama.attention.head_count",
                &["num_attention_heads", "n_head", "n_heads"][..],
            ),
            (
                "llama.attention.head_count_kv",
                &["num_key_value_heads", "n_kv_heads"][..],
            ),
        ] {
            if let Some(value) = present(self.find(fields)?) {
                if matches!(key, "llama.context_length" | "llama.embedding_length") {
                    out.json_uint(key, value)?;
                } else {
                    out.json_number_or_array(key, value)?;
                }
            }
        }
        if matches!(self.get("is_causal")?, Some(Json::Bool(false)))
            || matches!(
                self.get("use_bidirectional_attention")?,
                Some(Json::Bool(true))
            )
        {
            out.boolean("llama.attention.causal", false)?;
        }
        let rope = self.rope()?;
        if let Some(kind) = present(rope.full("rope_type")?) {
            let kind = text(kind)?;
            match (kind, present(rope.full("factor")?)) {
                ("linear" | "yarn", Some(factor)) => {
                    out.string("llama.rope.scaling.type", kind)?;
                    out.json_float("llama.rope.scaling.factor", factor)?;
                    if kind == "yarn" {
                        out.json_uint(
                            "llama.rope.scaling.original_context_length",
                            rope.full("original_max_position_embeddings")?
                                .ok_or(Error::Invalid("YaRN original context is missing"))?,
                        )?;
                        if let Some(value) = present(rope.full("truncate")?) {
                            out.boolean("llama.rope.scaling.yarn_truncate", truth(value))?;
                        }
                        for (field, key) in [
                            ("extrapolation_factor", "llama.rope.scaling.yarn_ext_factor"),
                            ("attention_factor", "llama.rope.scaling.yarn_attn_factor"),
                            ("beta_fast", "llama.rope.scaling.yarn_beta_fast"),
                            ("beta_slow", "llama.rope.scaling.yarn_beta_slow"),
                        ] {
                            let value = if field == "attention_factor" {
                                rope.full(field)?.or(rope.full("attn_factor")?)
                            } else {
                                rope.full(field)?
                            };
                            if let Some(value) = present(value) {
                                out.json_float(key, value)?;
                            }
                        }
                    }
                }
                ("su" | "longrope", _) => out.string("llama.rope.scaling.type", "longrope")?,
                _ => {}
            }
        }
        if let Some(sections) = rope.root.and_then(|r| r.get("mrope_section")) {
            let values = sections
                .as_array()
                .ok_or(Error::Invalid("mrope sections require an array"))?;
            let mut dims = Vec::new();
            crate::reserve(&mut dims, 4, "mrope sections")?;
            for index in 0..4 {
                dims.push(Value::I32(match values.get(index) {
                    Some(v) => i32::try_from(integer(v)?)
                        .map_err(|_| Error::Invalid("mrope section overflow"))?,
                    None => 0,
                }));
            }
            out.put(
                "llama.rope.dimension_sections",
                Value::Array {
                    element_type: 5,
                    values: dims,
                },
            )?;
        }
        if let Some(value) = present(rope.full("rope_theta")?) {
            out.json_float("llama.rope.freq_base", value)?;
        }
        if let Some(value) = present(rope.local_theta()?) {
            out.json_float("llama.rope.freq_base_swa", value)?;
        }
        for (key, fields) in [
            (
                "llama.attention.layer_norm_rms_epsilon",
                &["rms_norm_eps", "norm_eps"][..],
            ),
            (
                "llama.attention.layer_norm_epsilon",
                &["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"][..],
            ),
        ] {
            if let Some(value) = present(self.find(fields)?) {
                out.json_float(key, value)?;
            }
        }
        for (key, fields) in [
            (
                "llama.expert_count",
                &["num_local_experts", "num_experts", "n_routed_experts"][..],
            ),
            (
                "llama.expert_used_count",
                &[
                    "num_experts_per_tok",
                    "num_experts_per_token",
                    "top_k_experts",
                ][..],
            ),
            ("llama.expert_group_count", &["n_group"][..]),
            ("llama.expert_group_used_count", &["topk_group"][..]),
        ] {
            if let Some(value) = present(self.find(fields)?) {
                out.json_uint(key, value)?;
            }
        }
        if let Some(value) = present(self.get("head_dim")?) {
            out.json_uint("llama.attention.key_length", value)?;
            out.json_uint("llama.attention.value_length", value)?;
        }
        out.uint(
            "general.file_type",
            output
                .file_type()
                .ok_or(Error::Invalid("unresolved auto type"))?,
        )?;
        if self.family != Architecture::Llama {
            return Ok(());
        }
        out.json_uint("llama.vocab_size", self.required(&["vocab_size"])?)?;
        let dims = if let Some(value) = present(self.get("head_dim")?) {
            u32_value(value)?
        } else {
            let hidden = integer(self.required(&["hidden_size"])?)?;
            let heads = integer(self.required(&["num_attention_heads"])?)?;
            if heads == 0 {
                return Err(Error::Invalid("zero attention heads"));
            }
            u32::try_from(floor_div(hidden, heads)?)
                .map_err(|_| Error::Invalid("rope dimension count"))?
        };
        out.uint("llama.rope.dimension_count", dims)
    }
}
fn first_architecture(value: &Json) -> Result<Option<&str>> {
    let values = value
        .as_array()
        .ok_or(Error::Invalid("architectures must be an array"))?;
    values.first().map(text).transpose()
}
fn auto_config_valid(root: &Object, selected: Option<&str>) -> bool {
    for key in ["label2id", "id2label"] {
        if present(root.get(key)).is_some_and(|v| v.as_object().is_none()) {
            return false;
        }
    }
    if let Some(value) = present(root.get("id2label")).and_then(Json::as_object) {
        if value
            .entries()
            .any(|(k, _)| k.as_str().is_none_or(|s| s.trim().parse::<i64>().is_err()))
        {
            return false;
        }
    } else if let Some(value) = present(root.get("num_labels"))
        && integer(value).is_err()
    {
        return false;
    }
    if let Some(value) = present(root.get("problem_type"))
        && !matches!(
            text(value),
            Ok("regression" | "single_label_classification" | "multi_label_classification")
        )
    {
        return false;
    }
    let dtype = present(root.get("dtype")).or_else(|| present(root.get("torch_dtype")));
    if let Some(Json::Text(name)) = dtype
        && !matches!(
            name.as_str(),
            Some(
                "float16"
                    | "half"
                    | "float32"
                    | "float"
                    | "float64"
                    | "double"
                    | "bfloat16"
                    | "int8"
                    | "uint8"
                    | "int16"
                    | "short"
                    | "int32"
                    | "int"
                    | "int64"
                    | "long"
                    | "bool"
                    | "float8_e4m3fn"
                    | "float8_e5m2"
                    | "float8_e4m3fnuz"
                    | "float8_e5m2fnuz"
                    | "float8_e8m0fnu"
            )
        )
    {
        return false;
    }
    if selected == Some("gpt2") {
        // GPT2Config does not validate a rotary configuration or head division.
        return true;
    }
    if present(root.get("head_dim")).is_none()
        && (root.get("hidden_size").is_some_and(|v| integer(v).is_err())
            || root
                .get("num_attention_heads")
                .is_some_and(|v| integer(v).is_err() || matches!(integer(v), Ok(0))))
    {
        return false;
    }
    let Some(value) = present(root.get("rope_scaling")) else {
        return true;
    };
    let Some(rope) = value.as_object() else {
        return false;
    };
    let kind = rope.get("type").or_else(|| rope.get("rope_type"));
    let kind = match kind {
        None => "default",
        Some(v) => match text(v) {
            Ok(s) => s,
            Err(_) => return false,
        },
    };
    let required: &[&str] = match kind {
        "default" => &["rope_type"],
        "linear" | "dynamic" | "yarn" => &["rope_type", "factor"],
        "longrope" => &["rope_type", "short_factor", "long_factor"],
        "llama3" => &[
            "rope_type",
            "factor",
            "original_max_position_embeddings",
            "low_freq_factor",
            "high_freq_factor",
        ],
        _ => &[],
    };
    for key in required {
        if *key == "rope_type" {
            if kind == "default" && rope.get("type").is_none() && rope.get("rope_type").is_none() {
                return false;
            }
        } else if rope.get(key).is_none() {
            return false;
        }
    }
    match kind {
        "yarn" => {
            for key in ["beta_fast", "beta_slow"] {
                if present(rope.get(key)).is_some_and(|v| truth(v) && real(v).is_err()) {
                    return false;
                }
            }
            if let Some(value) = present(rope.get("original_max_position_embeddings"))
                && (real(value).is_err()
                    || matches!(real(value), Ok(0.0))
                    || root
                        .get("max_position_embeddings")
                        .is_some_and(|v| real(v).is_err()))
            {
                return false;
            }
        }
        "llama3" => {
            for key in [
                "low_freq_factor",
                "high_freq_factor",
                "original_max_position_embeddings",
            ] {
                if rope.get(key).is_none_or(|v| real(v).is_err()) {
                    return false;
                }
            }
            if root
                .get("max_position_embeddings")
                .is_some_and(|v| real(v).is_err())
            {
                return false;
            }
        }
        "longrope" => {
            for key in ["short_factor", "long_factor"] {
                if rope
                    .get(key)
                    .is_none_or(|v| !matches!(v, Json::Array(_) | Json::Object(_) | Json::Text(_)))
                {
                    return false;
                }
            }
        }
        _ => {}
    }
    true
}
pub(crate) struct Rope<'a> {
    config: &'a Config,
    root: Option<&'a Object>,
    full: Option<&'a Object>,
    isolated: bool,
    from_normalized_root: bool,
}
impl<'a> Rope<'a> {
    pub fn full(&self, key: &str) -> Result<Option<&'a Json>> {
        if key == "rope_type"
            && self.from_normalized_root
            && let Some(value) = self.full.and_then(|r| r.get("type"))
        {
            return Ok(Some(value));
        }
        if let Some(value) = self.full.and_then(|r| r.get(key)) {
            return Ok(Some(value));
        }
        if self.isolated {
            return Ok(None);
        }
        match key {
            "rope_type" => Ok(self.full.and_then(|r| present(r.get("type")))),
            "rope_theta" => self
                .config
                .find(&[
                    "global_rope_theta",
                    "rope_global_theta",
                    "rope_theta_global",
                    "rope_theta",
                    "rotary_emb_base",
                ])
                .map(present),
            "partial_rotary_factor" => self
                .config
                .find(&["partial_rotary_factor", "rope_pct", "rope_percent"])
                .map(present),
            "original_max_position_embeddings" => self
                .config
                .get("original_max_position_embeddings")
                .map(present),
            _ => Ok(None),
        }
    }
    fn local_theta(&self) -> Result<Option<&'a Json>> {
        if let Some(value) = self.root.and_then(|r| r.get("sliding_attention")) {
            return Ok(object(value)?.get("rope_theta"));
        }
        if self.isolated {
            return Ok(None);
        }
        self.config
            .find(&[
                "local_rope_theta",
                "rope_local_theta",
                "rope_theta_local",
                "swa_rope_theta",
                "rope_local_base_freq",
            ])
            .map(present)
    }
}

pub(crate) fn floor_div(left: i64, right: i64) -> Result<i64> {
    let quotient = left.checked_div(right).ok_or(Error::Invalid(
        "configuration division overflow or zero divisor",
    ))?;
    let remainder = left.checked_rem(right).ok_or(Error::Invalid(
        "configuration division overflow or zero divisor",
    ))?;
    if remainder != 0 && (remainder < 0) != (right < 0) {
        quotient
            .checked_sub(1)
            .ok_or(Error::Invalid("configuration floor division overflow"))
    } else {
        Ok(quotient)
    }
}
