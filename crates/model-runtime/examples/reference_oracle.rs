//! Independent, opt-in comparison against the *separately built* frozen GGML.
//! Run with a path to libggml-reference.so; no production crate links GGML.
//! All computation here uses GGML operators, not this crate's scalar functions.
//! The Rust source remains authoritative for this test harness. Native reference
//! code and its compiled objects stay outside the delivered product workspace.
use llama_model_runtime::{BatchToken, Context, ContextOptions, Matrix, Model, Pooling, fixtures};
use llama_platform_loader::DynamicLibrary;
use std::{ffi::c_void, ptr, sync::Arc};

type P = *mut c_void;
type Unary = unsafe extern "C" fn(P, P) -> P;
type Binary = unsafe extern "C" fn(P, P, P) -> P;
type Normalize = unsafe extern "C" fn(P, P, f32) -> P;
#[repr(C)]
struct InitParams {
    mem_size: usize,
    mem_buffer: P,
    no_alloc: bool,
}

macro_rules! native_api {
    ($($field:ident: $ty:ty => $symbol:expr),* $(,)?)=> {
        struct Api { $($field: $ty),* }
        impl Api {
            fn load(library:&DynamicLibrary)->Result<Self,Box<dyn std::error::Error>> {
                // SAFETY: signatures are copied from the frozen ggml.h and
                // ggml-cpu.h C ABI. Oracle owns the library until after free().
                Ok(Self { $($field:unsafe {library.symbol::<$ty>($symbol)?}),* })
            }
        }
    }
}
native_api! {
    init:unsafe extern "C" fn(InitParams)->P=>c"ggml_init",
    free:unsafe extern "C" fn(P)->()=>c"ggml_free",
    new1:unsafe extern "C" fn(P,i32,i64)->P=>c"ggml_new_tensor_1d",
    new2:unsafe extern "C" fn(P,i32,i64,i64)->P=>c"ggml_new_tensor_2d",
    new3:unsafe extern "C" fn(P,i32,i64,i64,i64)->P=>c"ggml_new_tensor_3d",
    new4:unsafe extern "C" fn(P,i32,i64,i64,i64,i64)->P=>c"ggml_new_tensor_4d",
    data:unsafe extern "C" fn(P)->P=>c"ggml_get_data",
    add:Binary=>c"ggml_add",
    mul:Binary=>c"ggml_mul",
    mm:Binary=>c"ggml_mul_mat",
    rows:Binary=>c"ggml_get_rows",
    rms:Normalize=>c"ggml_rms_norm",
    norm:Normalize=>c"ggml_norm",
    silu:Unary=>c"ggml_silu",
    gelu:Unary=>c"ggml_gelu",
    cont:Unary=>c"ggml_cont",
    transpose:Unary=>c"ggml_transpose",
    reshape2:unsafe extern "C" fn(P,P,i64,i64)->P=>c"ggml_reshape_2d",
    reshape3:unsafe extern "C" fn(P,P,i64,i64,i64)->P=>c"ggml_reshape_3d",
    reshape4:unsafe extern "C" fn(P,P,i64,i64,i64,i64)->P=>c"ggml_reshape_4d",
    view1:unsafe extern "C" fn(P,P,i64,usize)->P=>c"ggml_view_1d",
    view2:unsafe extern "C" fn(P,P,i64,i64,usize,usize)->P=>c"ggml_view_2d",
    view3:unsafe extern "C" fn(P,P,i64,i64,i64,usize,usize,usize)->P=>c"ggml_view_3d",
    view4:unsafe extern "C" fn(P,P,i64,i64,i64,i64,usize,usize,usize,usize)->P=>c"ggml_view_4d",
    concat:unsafe extern "C" fn(P,P,P,i32)->P=>c"ggml_concat",
    softmax:unsafe extern "C" fn(P,P,P,f32,f32)->P=>c"ggml_soft_max_ext",
    rope:unsafe extern "C" fn(P,P,P,P,i32,i32,i32,f32,f32,f32,f32,f32,f32)->P=>c"ggml_rope_ext",
    convolution:Binary=>c"ggml_ssm_conv",
    scan:unsafe extern "C" fn(P,P,P,P,P,P,P,P,i64)->P=>c"ggml_ssm_scan",
    graph:unsafe extern "C" fn(P)->P=>c"ggml_new_graph",
    expand:unsafe extern "C" fn(P,P)->()=>c"ggml_build_forward_expand",
    compute:unsafe extern "C" fn(P,P,i32)->i32=>c"ggml_graph_compute_with_ctx",
}

struct Oracle {
    api: Api,
    ctx: P,
    _library: DynamicLibrary,
}
impl Drop for Oracle {
    fn drop(&mut self) {
        // SAFETY: ctx is the sole live handle returned by init; the library and
        // all operator symbols are retained throughout this call.
        unsafe { (self.api.free)(self.ctx) };
    }
}
impl Oracle {
    fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let library = DynamicLibrary::open(&[path])?;
        let api = Api::load(&library)?;
        // SAFETY: exact repr(C) parameters; GGML owns the requested allocation.
        let ctx = unsafe {
            (api.init)(InitParams {
                mem_size: 64 * 1024 * 1024,
                mem_buffer: ptr::null_mut(),
                no_alloc: false,
            })
        };
        if ctx.is_null() {
            return Err("reference ggml_init failed".into());
        }
        Ok(Self {
            api,
            ctx,
            _library: library,
        })
    }
    fn tensor(&self, cols: usize, rows: usize, values: &[f32]) -> P {
        assert_eq!(cols * rows, values.len());
        // SAFETY: this test constructs bounded dense tensors in a live context;
        // the exact allocation length equals the copied number of F32 values.
        unsafe {
            let t = (self.api.new2)(self.ctx, 0, cols as i64, rows as i64);
            assert!(!t.is_null());
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                (self.api.data)(t).cast::<f32>(),
                values.len(),
            );
            t
        }
    }
    fn matrix(&self, m: &Matrix) -> P {
        self.tensor(m.cols(), m.rows(), m.values())
    }
    fn vector(&self, v: &[f32]) -> P {
        self.tensor(v.len(), 1, v)
    }
    fn integer(&self, value: i32) -> P {
        // SAFETY: source GGML_TYPE_I32=26; a one-element tensor owns four bytes.
        unsafe {
            let t = (self.api.new1)(self.ctx, 26, 1);
            *(self.api.data)(t).cast::<i32>() = value;
            t
        }
    }
    fn binary(&self, f: Binary, a: P, b: P) -> P {
        // SAFETY: all tensor handles below belong to this live context; graph
        // construction uses the checked fixture model's exact dimensions.
        unsafe { f(self.ctx, a, b) }
    }
    fn unary(&self, f: Unary, a: P) -> P {
        unsafe { f(self.ctx, a) }
    }
    fn bias(&self, a: P, b: &Option<Vec<f32>>) -> P {
        match b {
            Some(b) => self.binary(self.api.add, a, self.vector(b)),
            None => a,
        }
    }
    fn mm(&self, w: &Matrix, x: P) -> P {
        self.binary(self.api.mm, self.matrix(w), x)
    }
    fn rms(&self, x: P, w: &[f32], epsilon: f32) -> P {
        // SAFETY: positive finite epsilon, valid tensor owned by this context.
        self.binary(
            self.api.mul,
            unsafe { (self.api.rms)(self.ctx, x, epsilon) },
            self.vector(w),
        )
    }
    fn norm(&self, x: P, w: &[f32], b: &[f32], epsilon: f32) -> P {
        let normalized = unsafe { (self.api.norm)(self.ctx, x, epsilon) };
        self.binary(
            self.api.add,
            self.binary(self.api.mul, normalized, self.vector(w)),
            self.vector(b),
        )
    }
    fn embedding(&self, w: &Matrix, token: u32) -> P {
        self.binary(self.api.rows, self.matrix(w), self.integer(token as i32))
    }
    fn rope(&self, x: P, heads: usize, dim: usize, position: usize, base: f32) -> P {
        // SAFETY: shapes and position are from the validated finite test model.
        unsafe {
            let x = (self.api.reshape3)(self.ctx, x, dim as i64, heads as i64, 1);
            let x = (self.api.rope)(
                self.ctx,
                x,
                self.integer(position as i32),
                ptr::null_mut(),
                dim as i32,
                0,
                32,
                base,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            );
            (self.api.reshape2)(self.ctx, x, (heads * dim) as i64, 1)
        }
    }
    fn concat(&self, a: P, b: P, dimension: i32) -> P {
        unsafe { (self.api.concat)(self.ctx, a, b, dimension) }
    }
    fn attention(&self, q: P, k: P, v: P, shape: (usize, usize, usize, usize), scale: f32) -> P {
        let (heads, kv_heads, dim, count) = shape;
        let mut output: P = ptr::null_mut();
        for head in 0..heads {
            let kv_head = head / (heads / kv_heads);
            // SAFETY: byte strides address an in-bounds head of the allocated
            // query and the dense token-major KV matrices; all lifetimes live
            // through graph execution in this context.
            let (qh, kh, vh) = unsafe {
                (
                    (self.api.view2)(self.ctx, q, dim as i64, 1, dim * 4, head * dim * 4),
                    (self.api.view2)(
                        self.ctx,
                        k,
                        dim as i64,
                        count as i64,
                        kv_heads * dim * 4,
                        kv_head * dim * 4,
                    ),
                    (self.api.view2)(
                        self.ctx,
                        v,
                        dim as i64,
                        count as i64,
                        kv_heads * dim * 4,
                        kv_head * dim * 4,
                    ),
                )
            };
            let kh = self.unary(self.api.cont, kh);
            let scores = self.binary(self.api.mm, kh, qh);
            let probs =
                unsafe { (self.api.softmax)(self.ctx, scores, ptr::null_mut(), scale, 0.0) };
            let vt = self.unary(self.api.cont, self.unary(self.api.transpose, vh));
            let weighted = self.binary(self.api.mm, vt, probs);
            output = if output.is_null() {
                weighted
            } else {
                self.concat(output, weighted, 0)
            };
        }
        output
    }
    fn read(&self, tensor: P, len: usize) -> Vec<f32> {
        // SAFETY: final tensor is a contiguous F32 result with len elements.
        // GGML graph construction/compute is synchronous and keeps it alive.
        unsafe {
            let graph = (self.api.graph)(self.ctx);
            (self.api.expand)(graph, tensor);
            assert_eq!(
                (self.api.compute)(self.ctx, graph, 1),
                0,
                "reference GGML graph failed"
            );
            std::slice::from_raw_parts((self.api.data)(tensor).cast::<f32>(), len).to_vec()
        }
    }
    fn llama(&self, tokens: &[u32]) -> Vec<(Vec<f32>, Vec<f32>)> {
        let model = fixtures::tiny_llama().unwrap();
        let c = model.config();
        let w = model.weights();
        let mut keys = vec![ptr::null_mut(); w.layers.len()];
        let mut vals = keys.clone();
        let mut result = Vec::new();
        for (position, token) in tokens.iter().enumerate() {
            let mut hidden = self.embedding(&w.token_embedding, *token);
            for (i, layer) in w.layers.iter().enumerate() {
                let norm = self.rms(hidden, &layer.attention_norm, c.rms_epsilon);
                let q = self.bias(self.mm(&layer.query, norm), &layer.query_bias);
                let k = self.bias(self.mm(&layer.key, norm), &layer.key_bias);
                let v = self.bias(self.mm(&layer.value, norm), &layer.value_bias);
                let q = self.rope(
                    q,
                    c.heads,
                    c.head_dim().expect("validated fixture"),
                    position,
                    c.rope_base,
                );
                let k = self.rope(
                    k,
                    c.kv_heads,
                    c.head_dim().expect("validated fixture"),
                    position,
                    c.rope_base,
                );
                keys[i] = if position == 0 {
                    k
                } else {
                    self.concat(keys[i], k, 1)
                };
                vals[i] = if position == 0 {
                    v
                } else {
                    self.concat(vals[i], v, 1)
                };
                let attention = self.attention(
                    q,
                    keys[i],
                    vals[i],
                    (
                        c.heads,
                        c.kv_heads,
                        c.head_dim().expect("validated fixture"),
                        position + 1,
                    ),
                    (c.head_dim().expect("validated fixture") as f32)
                        .sqrt()
                        .recip(),
                );
                let projected = self.bias(
                    self.mm(&layer.attention_output, attention),
                    &layer.attention_output_bias,
                );
                hidden = self.binary(self.api.add, hidden, projected);
                let norm = self.rms(hidden, &layer.feed_forward_norm, c.rms_epsilon);
                let gate = self.bias(self.mm(&layer.gate, norm), &layer.gate_bias);
                let up = self.bias(self.mm(&layer.up, norm), &layer.up_bias);
                let activated = self.binary(self.api.mul, self.unary(self.api.silu, gate), up);
                let ffn = self.bias(self.mm(&layer.down, activated), &layer.down_bias);
                hidden = self.binary(self.api.add, hidden, ffn);
            }
            let embedding = self.rms(hidden, &w.output_norm, c.rms_epsilon);
            let logits = self.mm(w.output.as_ref().unwrap_or(&w.token_embedding), embedding);
            result.push((
                self.read(embedding, c.width),
                self.read(logits, c.vocabulary),
            ));
        }
        result
    }
    fn mamba(&self, tokens: &[u32]) -> Vec<(Vec<f32>, Vec<f32>)> {
        let model = fixtures::tiny_mamba().unwrap();
        let c = model.config();
        let w = model.weights();
        let mut conv = Vec::new();
        let mut ssm = Vec::new();
        for _ in &w.layers {
            // SAFETY: exact model-sized buffers, initialized before computation.
            unsafe {
                let t = (self.api.new3)(
                    self.ctx,
                    0,
                    (c.convolution_width - 1) as i64,
                    c.inner as i64,
                    1,
                );
                ptr::write_bytes(
                    (self.api.data)(t),
                    0,
                    (c.convolution_width - 1) * c.inner * 4,
                );
                conv.push(t);
                let t = (self.api.new4)(self.ctx, 0, c.state_size as i64, 1, c.inner as i64, 1);
                ptr::write_bytes((self.api.data)(t), 0, c.state_size * c.inner * 4);
                ssm.push(t);
            }
        }
        let mut result = Vec::new();
        for token in tokens {
            let mut hidden = self.embedding(&w.token_embedding, *token);
            for (i, layer) in w.layers.iter().enumerate() {
                let norm = self.rms(hidden, &layer.norm, c.rms_epsilon);
                let xz = self.mm(&layer.input, norm);
                // SAFETY: split the 2*inner projection, preserving its backing
                // allocation and the source's time-major convolution strides.
                let (x, z) = unsafe {
                    (
                        (self.api.view2)(self.ctx, xz, c.inner as i64, 1, 2 * c.inner * 4, 0),
                        (self.api.view2)(
                            self.ctx,
                            xz,
                            c.inner as i64,
                            1,
                            2 * c.inner * 4,
                            c.inner * 4,
                        ),
                    )
                };
                let cx = self.concat(conv[i], self.unary(self.api.transpose, x), 0);
                conv[i] = unsafe {
                    (self.api.view3)(
                        self.ctx,
                        cx,
                        (c.convolution_width - 1) as i64,
                        c.inner as i64,
                        1,
                        c.convolution_width * 4,
                        c.convolution_width * c.inner * 4,
                        4,
                    )
                };
                let conv_out =
                    self.binary(self.api.convolution, cx, self.matrix(&layer.convolution));
                let conv_out =
                    self.binary(self.api.add, conv_out, self.vector(&layer.convolution_bias));
                let x = self.unary(self.api.silu, conv_out);
                let projected = self.mm(&layer.x_projection, x);
                let dsize = c.state_size;
                let rank = c.time_step_rank;
                let full = rank + 2 * dsize;
                let (dt, b, cc) = unsafe {
                    (
                        (self.api.view2)(self.ctx, projected, rank as i64, 1, full * 4, 0),
                        (self.api.view4)(
                            self.ctx,
                            projected,
                            dsize as i64,
                            1,
                            1,
                            1,
                            dsize * 4,
                            full * 4,
                            full * 4,
                            rank * 4,
                        ),
                        (self.api.view4)(
                            self.ctx,
                            projected,
                            dsize as i64,
                            1,
                            1,
                            1,
                            dsize * 4,
                            full * 4,
                            full * 4,
                            (rank + dsize) * 4,
                        ),
                    )
                };
                let dt = self.mm(&layer.dt_projection, self.unary(self.api.cont, dt));
                let dt = self.binary(self.api.add, dt, self.vector(&layer.dt_bias));
                let reshaped = unsafe { (self.api.reshape4)(self.ctx, x, 1, c.inner as i64, 1, 1) };
                let combined = unsafe {
                    (self.api.scan)(
                        self.ctx,
                        ssm[i],
                        reshaped,
                        dt,
                        self.matrix(&layer.a),
                        b,
                        cc,
                        self.integer(0),
                        1,
                    )
                };
                ssm[i] = unsafe {
                    (self.api.view4)(
                        self.ctx,
                        combined,
                        dsize as i64,
                        1,
                        c.inner as i64,
                        1,
                        dsize * 4,
                        dsize * 4,
                        dsize * c.inner * 4,
                        c.inner * 4,
                    )
                };
                let y = unsafe { (self.api.view1)(self.ctx, combined, c.inner as i64, 0) };
                let skip = self.binary(self.api.mul, x, self.vector(&layer.d));
                let y = self.binary(self.api.add, y, skip);
                let y = self.binary(
                    self.api.mul,
                    y,
                    self.unary(self.api.silu, self.unary(self.api.cont, z)),
                );
                let output = self.mm(&layer.output, y);
                hidden = self.binary(self.api.add, hidden, output);
            }
            let embedding = self.rms(hidden, &w.output_norm, c.rms_epsilon);
            let logits = self.mm(w.output.as_ref().unwrap_or(&w.token_embedding), embedding);
            result.push((
                self.read(embedding, c.width),
                self.read(logits, c.vocabulary),
            ));
        }
        result
    }
    fn bert(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let model = fixtures::tiny_bert().unwrap();
        let c = model.config();
        let w = model.weights();
        let mut hidden = Vec::new();
        for (position, token) in tokens.iter().enumerate() {
            let mut input = self.embedding(&w.token_embedding, *token);
            input = self.binary(
                self.api.add,
                input,
                self.embedding(&w.position_embedding, position as u32),
            );
            if let Some(types) = &w.token_type_embedding {
                input = self.binary(self.api.add, input, self.embedding(types, 0));
            }
            hidden.push(self.norm(
                input,
                &w.embedding_norm,
                &w.embedding_norm_bias,
                c.layer_norm_epsilon,
            ));
        }
        for layer in &w.layers {
            let mut queries = Vec::new();
            let mut keys = ptr::null_mut();
            let mut values = ptr::null_mut();
            for (position, input) in hidden.iter().enumerate() {
                queries.push(self.bias(self.mm(&layer.query, *input), &layer.query_bias));
                let key = self.bias(self.mm(&layer.key, *input), &layer.key_bias);
                let value = self.bias(self.mm(&layer.value, *input), &layer.value_bias);
                keys = if position == 0 {
                    key
                } else {
                    self.concat(keys, key, 1)
                };
                values = if position == 0 {
                    value
                } else {
                    self.concat(values, value, 1)
                };
            }
            let mut next = Vec::new();
            for (query, input) in queries.iter().zip(hidden) {
                let attention = self.attention(
                    *query,
                    keys,
                    values,
                    (
                        c.heads,
                        c.kv_heads,
                        c.head_dim().expect("validated fixture"),
                        tokens.len(),
                    ),
                    (c.head_dim().expect("validated fixture") as f32)
                        .sqrt()
                        .recip(),
                );
                let attention = self.bias(
                    self.mm(&layer.attention_output, attention),
                    &layer.attention_output_bias,
                );
                let residual = self.binary(self.api.add, attention, input);
                let norm = self.norm(
                    residual,
                    &layer.attention_norm,
                    &layer.attention_norm_bias,
                    c.layer_norm_epsilon,
                );
                let up = self.bias(self.mm(&layer.up, norm), &layer.up_bias);
                let activated = self.unary(self.api.gelu, up);
                let down = self.bias(self.mm(&layer.down, activated), &layer.down_bias);
                let residual = self.binary(self.api.add, norm, down);
                next.push(self.norm(
                    residual,
                    &layer.output_norm,
                    &layer.output_norm_bias,
                    c.layer_norm_epsilon,
                ));
            }
            hidden = next;
        }
        hidden.iter().map(|t| self.read(*t, c.width)).collect()
    }
}

fn compare(name: &str, actual: &[f32], reference: &[f32]) {
    assert_eq!(actual.len(), reference.len());
    let mut maximum = 0.0f32;
    for (i, (a, b)) in actual.iter().zip(reference).enumerate() {
        let error = (a - b).abs();
        maximum = maximum.max(error);
        assert!(
            a.is_finite() && b.is_finite() && error <= 3e-6 + 3e-6 * b.abs(),
            "{name}[{i}]: Rust {a}, frozen GGML {b}, error {error}"
        );
    }
    eprintln!(
        "verified {name}: {} values, maximum_abs_error={maximum:.9e}",
        actual.len()
    );
    print!("{name}");
    for value in reference {
        print!("\t{value:.9e}");
    }
    println!();
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: reference_oracle /absolute/path/libggml-reference.so")?;
    let tokens = [1, 2, 3, 4];
    let batch: Vec<_> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| BatchToken::token(0, i, *t))
        .collect();
    for name in ["llama", "mamba"] {
        let reference = Oracle::new(&path)?;
        let (model, expected) = if name == "llama" {
            (
                Model::Llama(fixtures::tiny_llama()?),
                reference.llama(&tokens),
            )
        } else {
            (
                Model::Mamba(fixtures::tiny_mamba()?),
                reference.mamba(&tokens),
            )
        };
        let mut context = Context::new(Arc::new(model), ContextOptions::default())?;
        let actual = context.decode_batch(&batch)?;
        for (i, (output, (embedding, logits))) in actual.iter().zip(expected).enumerate() {
            compare(
                &format!("{name}.embedding.{i}"),
                output.embedding.as_ref().unwrap(),
                &embedding,
            );
            compare(
                &format!("{name}.logits.{i}"),
                output.logits.as_ref().unwrap(),
                &logits,
            );
        }
    }
    let oracle = Oracle::new(&path)?;
    let expected = oracle.bert(&tokens);
    let actual = fixtures::tiny_bert()?.encode(&tokens, Pooling::None)?;
    for (i, (actual, reference)) in actual.tokens.iter().zip(expected).enumerate() {
        compare(&format!("bert.embedding.{i}"), actual, &reference);
    }
    Ok(())
}
