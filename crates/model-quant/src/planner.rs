//! Frozen mixed-precision heuristics; tensor order is semantically significant.
use crate::{DType, Error, QuantizeParams, Result, invalid, is_quantized};
use llama_behavior::native_regex::{NativeRegex, RegexProfile};

#[derive(Clone, Debug)]
pub struct ModelDescriptor {
    pub architecture: String,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_expert: u32,
    pub n_embd_head_k: u32,
    pub n_embd_head_v: u32,
    /// Real file-loader classification when known. None preserves the frozen
    /// low-level descriptor helper's simplified layer-count classification.
    pub llama_70b: Option<bool>,
}
impl Default for ModelDescriptor {
    fn default() -> Self {
        Self {
            architecture: "llama".into(),
            n_embd: 256,
            n_ff: 512,
            n_layer: 8,
            n_head: 8,
            n_head_kv: 2,
            n_expert: 0,
            n_embd_head_k: 32,
            n_embd_head_v: 32,
            llama_70b: None,
        }
    }
}
#[derive(Clone, Debug)]
pub struct TensorSpec {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub dtype: DType,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Category {
    Token,
    Output,
    Q,
    K,
    V,
    Qkv,
    KvB,
    AttentionOutput,
    AttentionGate,
    Up,
    Gate,
    Down,
    Latent,
    Other,
}
fn category(n: &str) -> Category {
    use Category::*;
    if n == "output.weight" {
        return Output;
    }
    if token(n) {
        return Token;
    }
    for (fragment, cat) in [
        ("attn_qkv.weight", Qkv),
        ("attn_qkv_gate.weight", Qkv),
        ("attn_kv_b.weight", KvB),
        ("attn_v.weight", V),
        ("attn_k.weight", K),
        ("attn_q.weight", Q),
        ("attn_output.weight", AttentionOutput),
        ("ffn_up", Up),
        ("ffn_gate", Gate),
        ("ffn_latent", Latent),
        ("attn_gate.weight", AttentionGate),
        ("ffn_down", Down),
    ] {
        if n.contains(fragment) {
            return cat;
        }
    }
    Other
}
fn token(n: &str) -> bool {
    matches!(n, "token_embd.weight" | "per_layer_token_embd.weight")
}
fn v_like(c: Category) -> bool {
    matches!(c, Category::V | Category::Qkv | Category::KvB)
}
fn contains_any(n: &str, fragments: &[&str]) -> bool {
    fragments.iter().any(|s| n.contains(s))
}
pub fn allows_quantization(p: &QuantizeParams, arch: &str, t: &TensorSpec) -> bool {
    let rank = t
        .dimensions
        .iter()
        .rposition(|&v| v > 1)
        .map_or(1, |i| i + 1);
    if p.only_copy || rank < 2 {
        return false;
    }
    let n = t.name.as_str();
    if !n.ends_with("weight") || (!p.quantize_output_tensor && n == "output.weight") {
        return false;
    }
    if contains_any(
        n,
        &[
            "_norm.weight",
            "ffn_gate_inp.weight",
            "ffn_gate_tid2eid.weight",
            "engram_gate.weight",
            "altup",
            "laurel",
            "per_layer_model_proj",
            "ssm_conv1d",
            "shortconv.conv.weight",
            "conv2d_spatial.weight",
            "conv_dw",
            "indexer.k_proj.weight",
            "indexer.q_proj.weight",
            "attn_rel_b.weight",
            ".position_embd",
            "sam.pos_embd",
            "sam.neck.",
            "sam.net_",
            ".rel_pos",
            ".patch_embd",
            ".patch_merger",
            "a.rvq.codebook",
            "mm.a.code_embd",
        ],
    ) {
        return false;
    }
    if matches!(n, "position_embd.weight" | "token_types.weight")
        || n.starts_with("a.conv1d.")
        || n.starts_with("a.gen.wav.quant.")
    {
        return false;
    }
    if [
        "first",
        "w0",
        "w1",
        "w2",
        "v0",
        "v1",
        "v2",
        "a0",
        "a1",
        "a2",
        "g1",
        "g2",
        "decay_w1",
        "decay_w2",
        "lerp_fused",
    ]
    .iter()
    .any(|v| n.contains(&format!("time_mix_{v}.weight")))
    {
        return false;
    }
    match arch {
        "limite" if contains_any(n, &["attn_gate.weight", "attn_ve_gate.weight", "mudd_"]) => false,
        "metis16"
            if n.starts_with("depth_mem_")
                || n.starts_with("cont_")
                || contains_any(n, &["stream_embd", "attn_pass_lora", "ffn_exp_embd", "hc_"]) =>
        {
            false
        }
        "kimi-k31" if contains_any(n, &["indexer.attn_q_b", "indexer.attn_k", "indexer.proj"]) => {
            false
        }
        "glm5-next"
            if contains_any(
                n,
                &[
                    "hc_",
                    "indexer.attn_q_b",
                    "indexer.attn_k",
                    "indexer.proj",
                    "indexer_compressor_gate",
                    "indexer_compressor_ape",
                    "ssm_f_a.weight",
                    "ssm_f_b.weight",
                    "ssm_g_a.weight",
                    "ssm_g_b.weight",
                    "ssm_beta.weight",
                    "attn_kv_a_mqa.weight",
                    "attn_k_b.weight",
                    "attn_v_b.weight",
                ],
            ) =>
        {
            false
        }
        "xing4_0" if n.contains("hc_") => false,
        "vllm-decision2" if n.starts_with("decision.") => false,
        _ => true,
    }
}
pub(crate) fn native_layer_index(name: &str) -> Option<i32> {
    let s = name
        .strip_prefix("blk.")?
        .trim_start_matches([' ', '\t', '\n', '\r', '\u{b}', '\u{c}']);
    let mut end = 0;
    for (i, b) in s.bytes().enumerate() {
        if b.is_ascii_digit() || (i == 0 && matches!(b, b'+' | b'-')) {
            end = i + 1
        } else {
            break;
        }
    }
    s.get(..end)?.parse().ok()
}
fn more(i: i32, n: i32) -> bool {
    i < n / 8 || i >= 7 * n / 8 || (i - n / 8) % 3 == 2
}
struct State<'a> {
    model: &'a ModelDescriptor,
    params: &'a QuantizeParams,
    patterns: Vec<(NativeRegex, DType)>,
    n_v: i32,
    i_v: i32,
    i_down: i32,
    i_gate: i32,
    i_up: i32,
    tied: bool,
    has_imatrix: bool,
}
impl State<'_> {
    fn layer(&self, index: i32, name: &str) -> Result<i32> {
        if self.model.n_expert <= 1 {
            return Ok(index);
        }
        let i = native_layer_index(name)
            .ok_or_else(|| invalid(format!("cannot determine expert layer for {name}")))?;
        if i < 0 || i >= self.model.n_layer as i32 {
            return Err(invalid(format!("expert layer {i} outside model blocks")));
        }
        Ok(i)
    }
    fn standard(&mut self, mut ty: DType, t: &TensorSpec, c: Category) -> Result<DType> {
        use DType::*;
        let f = self.params.ftype.id();
        let m = self.model;
        let n = t.name.as_str();
        let arch = m.architecture.as_str();
        let gqa = m.n_head.checked_div(m.n_head_kv).unwrap_or(0);
        let layers = m.n_layer as i32;
        if arch == "glm5-next"
            && contains_any(n, &["attn_q_a", "attn_q_b", "nextn.eh_proj"])
            && !matches!(ty, F32 | F16 | BF16)
        {
            return Ok(Q8_0);
        }
        if arch == "muse-spark"
            && matches!(c, Category::Latent | Category::AttentionGate)
            && is_quantized(ty)
            && ty.type_size() * Q8_0.block_size() < Q8_0.type_size() * ty.block_size()
        {
            ty = Q8_0
        }
        if c == Category::Output || (self.tied && c == Category::Token) {
            ty = if let Some(output) = self.params.output_tensor_type {
                output
            } else if f == 38
                || arch == "falcon"
                || !t.dimensions[0].is_multiple_of(ty.block_size() as u64)
            {
                Q8_0
            } else if matches!(f, 19 | 20 | 23 | 24 | 28 | 29 | 31) {
                Q5_K
            } else if ty != Q8_0 {
                Q6_K
            } else {
                ty
            };
        } else if f == 38 {
            ty = if n.contains("_exps.") || n.contains("_chexps.") {
                MXFP4
            } else {
                Q8_0
            };
        } else if c == Category::Token {
            ty = if let Some(token) = self.params.token_embedding_type {
                token
            } else if matches!(f, 19 | 20 | 24 | 31) {
                Q2_K
            } else if matches!(f, 28 | 29 | 23) {
                IQ3_S
            } else if matches!(f, 36 | 37 | 41) {
                Q4_K
            } else {
                ty
            };
        } else if matches!(f, 19 | 20 | 24 | 28 | 29 | 31) {
            if v_like(c) {
                ty = if gqa >= 4 || m.n_expert >= 4 {
                    Q4_K
                } else if matches!(f, 28 | 29) {
                    IQ3_S
                } else {
                    Q2_K
                };
                self.i_v += 1;
            } else if m.n_expert == 8 && c == Category::K {
                ty = Q4_K
            } else if c == Category::Down {
                if self.i_down < layers / 8 {
                    ty = if matches!(f, 28 | 29) { IQ3_S } else { Q2_K }
                }
                self.i_down += 1;
            } else if c == Category::AttentionOutput {
                if m.n_expert == 8 {
                    ty = Q5_K
                } else if matches!(f, 24 | 31) {
                    ty = IQ2_XXS
                } else if matches!(f, 28 | 29) {
                    ty = IQ3_S
                }
            }
        } else if v_like(c) {
            match f {
                10 => ty = if gqa >= 4 { Q4_K } else { Q3_K },
                21 if gqa >= 4 => ty = Q4_K,
                23 => {
                    ty = if gqa >= 4 {
                        Q4_K
                    } else if !self.has_imatrix {
                        IQ3_S
                    } else {
                        IQ3_XXS
                    }
                }
                22 | 26 if gqa >= 4 => ty = Q4_K,
                27 => ty = Q4_K,
                12 => ty = if self.i_v < 2 { Q5_K } else { Q4_K },
                13 => ty = Q5_K,
                25 | 30 if gqa >= 4 => ty = Q5_K,
                15 | 17 if more(self.i_v, self.n_v) => ty = Q6_K,
                14 if self.i_v < 4 => ty = Q5_K,
                _ => {}
            }
            if m.llama_70b
                .unwrap_or(arch == "llama" && m.n_layer == 80 && m.n_head != m.n_head_kv)
                && matches!(ty, Q3_K | Q4_K)
            {
                ty = Q5_K
            }
            if m.n_expert == 8 {
                ty = Q8_0
            }
            self.i_v += 1;
        } else if c == Category::K {
            if m.n_expert == 8 {
                ty = Q8_0
            } else if f == 22 {
                ty = IQ3_XXS
            } else if f == 23 {
                ty = IQ2_S
            }
        } else if c == Category::Q {
            if f == 22 {
                ty = IQ3_XXS
            } else if f == 23 {
                ty = IQ2_S
            }
        } else if c == Category::Down {
            let i = self.layer(self.i_down, n)?;
            match f {
                10 => ty = Q3_K,
                21 if i < layers / 8 => ty = Q4_K,
                23 if !self.has_imatrix => ty = if i < layers / 8 { Q4_K } else { Q3_K },
                12 => {
                    ty = if i < layers / 16 {
                        Q5_K
                    } else if arch != "falcon" || more(i, layers) {
                        Q4_K
                    } else {
                        Q3_K
                    }
                }
                27 if i < layers / 8 || (m.n_expert == 8 && more(i, layers)) => ty = Q4_K,
                13 => ty = if arch == "falcon" { Q4_K } else { Q5_K },
                15 => {
                    if arch == "falcon" {
                        ty = if i < layers / 16 {
                            Q6_K
                        } else if more(i, layers) {
                            Q5_K
                        } else {
                            Q4_K
                        }
                    } else if more(i, layers) {
                        ty = Q6_K
                    }
                }
                25 | 30 if i < layers / 8 && !self.has_imatrix => ty = Q5_K,
                17 if more(i, layers) => ty = Q6_K,
                14 if arch != "falcon" && i < layers / 8 => ty = Q5_K,
                2 | 8 if self.has_imatrix && i < layers / 8 => {
                    ty = if f == 2 { Q4_1 } else { Q5_1 }
                }
                _ => {}
            }
            self.i_down += 1;
        } else if c == Category::AttentionOutput {
            if arch != "falcon" {
                if m.n_expert == 8 {
                    if matches!(f, 10 | 22 | 23 | 11 | 12 | 25 | 14 | 15 | 26 | 27 | 30) {
                        ty = Q5_K
                    }
                } else {
                    match f {
                        10 => ty = Q3_K,
                        23 => ty = IQ3_S,
                        12 | 27 => ty = Q4_K,
                        13 => ty = Q5_K,
                        _ => {}
                    }
                }
            } else if f == 13 {
                ty = Q4_K
            }
        } else if c == Category::Gate || c == Category::Up {
            let idx = if c == Category::Gate {
                self.i_gate
            } else {
                self.i_up
            };
            let i = self.layer(idx, n)?;
            if f == 22 && i >= layers / 8 && i < 7 * layers / 8 {
                ty = IQ3_XXS
            }
            if c == Category::Gate {
                self.i_gate += 1
            } else {
                self.i_up += 1
            }
        }
        Ok(ty)
    }
    fn tensor(&mut self, t: &TensorSpec) -> Result<DType> {
        let p = self.params;
        let c = category(&t.name);
        if !allows_quantization(p, &self.model.architecture, t) {
            return Ok(t.dtype);
        }
        let mut selected = None;
        for (regex, dtype) in &self.patterns {
            if regex
                .find(t.name.as_bytes())
                .map_err(|e| invalid(e.to_string()))?
                .is_some()
            {
                selected = Some(*dtype);
                break;
            }
        }
        if let Some(dtype) = p.token_embedding_type
            && c == Category::Token
            && (t.name != "per_layer_token_embd.weight" || selected.is_none())
        {
            return Ok(dtype);
        }
        if c == Category::Output
            && let Some(dtype) = p.output_tensor_type
        {
            return Ok(dtype);
        }
        let default = p.ftype.default_type();
        if !is_quantized(default) {
            return Ok(default);
        }
        let ty = match selected {
            Some(v) => v,
            None if !p.pure => self.standard(default, t, c)?,
            None => default,
        };
        fallback(ty, t.dimensions[0])
    }
}
fn fallback(t: DType, width: u64) -> Result<DType> {
    use DType::*;
    if width.is_multiple_of(t.block_size() as u64) {
        return Ok(t);
    }
    let other = match t {
        IQ1_S | IQ1_M | IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S | IQ4_XS => IQ4_NL,
        Q2_0 | Q2_K | Q3_K | TQ1_0 | TQ2_0 => Q4_0,
        Q4_K => Q5_0,
        Q5_K => Q5_1,
        Q6_K => Q8_0,
        _ if t.block_size() <= 32 => t,
        _ => {
            return Err(invalid(format!(
                "no tensor type fallback is defined for {t:?}"
            )));
        }
    };
    Ok(if width.is_multiple_of(other.block_size() as u64) {
        other
    } else {
        F16
    })
}
/// Assign types in the supplied order, counting every descriptor (including
/// ineligible output weights) exactly as the source file driver does.
pub fn assign_types(
    model: &ModelDescriptor,
    params: &QuantizeParams,
    tensors: &[TensorSpec],
    has_imatrix: bool,
) -> Result<Vec<DType>> {
    if !crate::architectures::KNOWN.contains(&model.architecture.as_str()) {
        return Err(Error::Unsupported(format!(
            "model architecture {}",
            model.architecture
        )));
    }
    if model.n_layer == 0 || model.n_layer > 512 {
        return Err(invalid(
            "model block count is outside the frozen 1..=512 limit",
        ));
    }
    let mut patterns = Vec::new();
    if params.tt_overrides.len() > 4096 {
        return Err(Error::Limit("tensor override count".into()));
    }
    for p in &params.tt_overrides {
        patterns.push((compile_regex(&p.pattern)?, p.dtype));
    }
    for t in tensors {
        if t.dimensions.is_empty() || t.dimensions.len() > 4 || t.dimensions.contains(&0) {
            return Err(invalid("tensor must have one to four positive dimensions"));
        }
        if t.name.as_bytes().contains(&0) || t.name.len() >= 64 {
            return Err(invalid(
                "tensor name exceeds the checked native name domain",
            ));
        }
        t.dtype.row_bytes(
            usize::try_from(t.dimensions[0]).map_err(|_| invalid("row width exceeds usize"))?,
        )?;
    }
    let n_v = i32::try_from(tensors.iter().filter(|t| v_like(category(&t.name))).count())
        .map_err(|_| invalid("tensor count exceeds INT32"))?;
    let mut state = State {
        model,
        params,
        patterns,
        n_v,
        i_v: 0,
        i_down: 0,
        i_gate: 0,
        i_up: 0,
        tied: !tensors.iter().any(|t| t.name == "output.weight"),
        has_imatrix,
    };
    tensors.iter().map(|t| state.tensor(t)).collect()
}
fn compile_regex(pattern: &[u8]) -> Result<NativeRegex> {
    NativeRegex::parse_with_profile(
        pattern,
        if cfg!(target_env = "gnu") {
            RegexProfile::Gnu
        } else {
            RegexProfile::Msvc
        },
    )
    .map_err(|e| invalid(e.to_string()))
}
