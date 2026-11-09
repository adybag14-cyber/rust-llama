//! Mamba-1 recurrent graph slice from `mamba.cpp` / `mamba-base.cpp`.
use crate::{AdapterSet, Error, Fingerprint, Matrix, Result, check_epsilon, check_positive, math};

#[derive(Debug, Clone)]
pub struct MambaConfig {
    pub vocabulary: usize,
    pub width: usize,
    pub inner: usize,
    pub state_size: usize,
    pub convolution_width: usize,
    pub time_step_rank: usize,
    pub max_context: usize,
    pub rms_epsilon: f32,
    pub normalize_dt_b_c: bool,
}
impl MambaConfig {
    /// Check raw dimensions and normalization before deriving sizes or allocating weights.
    pub fn validate(&self) -> Result<()> {
        check_positive(&[
            ("vocabulary", self.vocabulary),
            ("width", self.width),
            ("inner", self.inner),
            ("state_size", self.state_size),
            ("convolution_width", self.convolution_width),
            ("time_step_rank", self.time_step_rank),
            ("max_context", self.max_context),
        ])?;
        check_epsilon(self.rms_epsilon)?;
        if self.width.checked_mul(2) != Some(self.inner) {
            return Err(Error::InvalidConfig(
                "the source Mamba-1 path requires inner == 2 * width".into(),
            ));
        }
        self.inner
            .checked_mul(self.state_size)
            .ok_or_else(|| Error::InvalidConfig("SSM state size overflow".into()))?;
        self.inner
            .checked_mul(self.convolution_width)
            .ok_or_else(|| Error::InvalidConfig("convolution state size overflow".into()))?;
        self.state_size
            .checked_mul(2)
            .and_then(|v| v.checked_add(self.time_step_rank))
            .ok_or_else(|| Error::InvalidConfig("SSM projection size overflow".into()))?;
        self.inner
            .checked_mul(2)
            .ok_or_else(|| Error::InvalidConfig("SSM input size overflow".into()))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MambaLayer {
    pub norm: Vec<f32>,
    pub input: Matrix,
    /// Per-channel causal convolution: inner rows, convolution_width columns.
    pub convolution: Matrix,
    pub convolution_bias: Vec<f32>,
    pub x_projection: Matrix,
    pub dt_projection: Matrix,
    pub dt_bias: Vec<f32>,
    /// Already transformed negative A values, not the checkpoint's A_log.
    pub a: Matrix,
    pub d: Vec<f32>,
    pub output: Matrix,
    pub dt_norm: Option<Vec<f32>>,
    pub b_norm: Option<Vec<f32>>,
    pub c_norm: Option<Vec<f32>>,
}
impl MambaLayer {
    fn validate(&self, c: &MambaConfig) -> Result<()> {
        math::vector(&self.norm, c.width, "Mamba RMS norm")?;
        self.input.shape(2 * c.inner, c.width, "Mamba input")?;
        self.convolution
            .shape(c.inner, c.convolution_width, "Mamba convolution")?;
        math::vector(&self.convolution_bias, c.inner, "Mamba convolution bias")?;
        self.x_projection.shape(
            c.time_step_rank + 2 * c.state_size,
            c.inner,
            "Mamba x projection",
        )?;
        self.dt_projection
            .shape(c.inner, c.time_step_rank, "Mamba dt projection")?;
        math::vector(&self.dt_bias, c.inner, "Mamba dt bias")?;
        self.a.shape(c.inner, c.state_size, "Mamba A")?;
        math::vector(&self.d, c.inner, "Mamba D")?;
        self.output.shape(c.width, c.inner, "Mamba output")?;
        math::optional_bias(&self.dt_norm, c.time_step_rank, "Mamba dt norm")?;
        math::optional_bias(&self.b_norm, c.state_size, "Mamba B norm")?;
        math::optional_bias(&self.c_norm, c.state_size, "Mamba C norm")?;
        let number = usize::from(self.dt_norm.is_some())
            + usize::from(self.b_norm.is_some())
            + usize::from(self.c_norm.is_some());
        if number != 0 && number != 3 {
            return Err(Error::InvalidConfig(
                "Mamba dt/B/C norm weights must be supplied together".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MambaWeights {
    pub token_embedding: Matrix,
    pub layers: Vec<MambaLayer>,
    pub output_norm: Vec<f32>,
    pub output: Option<Matrix>,
}
#[derive(Debug, Clone)]
pub struct MambaModel {
    config: MambaConfig,
    weights: MambaWeights,
    fingerprint: u64,
}
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecurrentLayer {
    pub convolution: Vec<f32>,
    pub ssm: Vec<f32>,
}

impl MambaModel {
    pub fn new(config: MambaConfig, weights: MambaWeights) -> Result<Self> {
        config.validate()?;
        check_positive(&[("layers", weights.layers.len())])?;
        weights
            .token_embedding
            .shape(config.vocabulary, config.width, "Mamba token embedding")?;
        math::vector(&weights.output_norm, config.width, "Mamba output norm")?;
        if let Some(output) = &weights.output {
            output.shape(config.vocabulary, config.width, "Mamba output")?;
        }
        for layer in &weights.layers {
            layer.validate(&config)?;
        }
        let mut result = Self {
            config,
            weights,
            fingerprint: 0,
        };
        result.fingerprint = result.calculate_fingerprint();
        Ok(result)
    }
    pub fn config(&self) -> &MambaConfig {
        &self.config
    }
    pub fn weights(&self) -> &MambaWeights {
        &self.weights
    }
    pub fn layer_count(&self) -> usize {
        self.weights.layers.len()
    }
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
    pub fn weight(&self, name: &str) -> Option<&Matrix> {
        if name == "output.weight" {
            return Some(
                self.weights
                    .output
                    .as_ref()
                    .unwrap_or(&self.weights.token_embedding),
            );
        }
        let (index, field) = name.strip_prefix("blk.")?.split_once('.')?;
        let parsed = index.parse::<usize>().ok()?;
        if parsed.to_string() != index {
            return None;
        }
        let layer = self.weights.layers.get(parsed)?;
        match field {
            "ssm_in.weight" => Some(&layer.input),
            "ssm_x.weight" => Some(&layer.x_projection),
            "ssm_dt.weight" => Some(&layer.dt_projection),
            "ssm_out.weight" => Some(&layer.output),
            _ => None,
        }
    }
    pub(crate) fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= self.config.vocabulary {
            return Err(Error::InvalidToken {
                token,
                vocabulary: self.config.vocabulary,
            });
        }
        Ok(self.weights.token_embedding.row(token as usize)?.to_vec())
    }
    pub(crate) fn empty_state(&self) -> Vec<RecurrentLayer> {
        self.weights
            .layers
            .iter()
            .map(|_| RecurrentLayer {
                convolution: vec![0.0; self.config.inner * (self.config.convolution_width - 1)],
                ssm: vec![0.0; self.config.inner * self.config.state_size],
            })
            .collect()
    }
    pub(crate) fn step(
        &self,
        mut hidden: Vec<f32>,
        state: &mut [RecurrentLayer],
        adapters: &AdapterSet,
        cancel: &mut impl FnMut() -> bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = &self.config;
        for (index, (layer, memory)) in self.weights.layers.iter().zip(state).enumerate() {
            if cancel() {
                return Err(Error::Cancelled);
            }
            let normalized = math::rms_norm(&hidden, &layer.norm, c.rms_epsilon)?;
            let xz = adapters.multiply(
                &format!("blk.{index}.ssm_in.weight"),
                &layer.input,
                &normalized,
            )?;
            let (x, z) = xz.split_at(c.inner);
            let mut convolved = vec![0.0; c.inner];
            let prior = c.convolution_width - 1;
            for channel in 0..c.inner {
                let old = &mut memory.convolution[channel * prior..(channel + 1) * prior];
                let kernel = layer.convolution.row(channel)?;
                let mut value = 0.0;
                for (sample, weight) in old.iter().zip(kernel) {
                    value += sample * weight;
                }
                value += x[channel] * kernel[prior];
                value += layer.convolution_bias[channel];
                convolved[channel] = math::silu(value);
                if prior != 0 {
                    old.rotate_left(1);
                    old[prior - 1] = x[channel];
                }
            }
            let projected = adapters.multiply(
                &format!("blk.{index}.ssm_x.weight"),
                &layer.x_projection,
                &convolved,
            )?;
            let mut dt = projected[..c.time_step_rank].to_vec();
            let mut b = projected[c.time_step_rank..c.time_step_rank + c.state_size].to_vec();
            let mut cc = projected[c.time_step_rank + c.state_size..].to_vec();
            if c.normalize_dt_b_c || layer.dt_norm.is_some() {
                dt = math::rms_norm(
                    &dt,
                    layer
                        .dt_norm
                        .as_deref()
                        .unwrap_or(&vec![1.0; c.time_step_rank]),
                    c.rms_epsilon,
                )?;
                b = math::rms_norm(
                    &b,
                    layer.b_norm.as_deref().unwrap_or(&vec![1.0; c.state_size]),
                    c.rms_epsilon,
                )?;
                cc = math::rms_norm(
                    &cc,
                    layer.c_norm.as_deref().unwrap_or(&vec![1.0; c.state_size]),
                    c.rms_epsilon,
                )?;
            }
            let mut dt = adapters.multiply(
                &format!("blk.{index}.ssm_dt.weight"),
                &layer.dt_projection,
                &dt,
            )?;
            math::add(&mut dt, &layer.dt_bias);
            let mut y = vec![0.0; c.inner];
            for channel in 0..c.inner {
                let delta = math::softplus(dt[channel]);
                let input_delta = convolved[channel] * delta;
                let mut sum = 0.0;
                for i in 0..c.state_size {
                    let at = channel * c.state_size + i;
                    memory.ssm[at] =
                        memory.ssm[at] * (delta * layer.a.values()[at]).exp() + b[i] * input_delta;
                    sum += memory.ssm[at] * cc[i];
                }
                y[channel] = (sum + convolved[channel] * layer.d[channel]) * math::silu(z[channel]);
            }
            let output =
                adapters.multiply(&format!("blk.{index}.ssm_out.weight"), &layer.output, &y)?;
            math::add(&mut hidden, &output);
            adapters.control(index, &mut hidden);
            math::finite(&hidden, "Mamba layer output")?;
            math::finite(&memory.ssm, "Mamba recurrent state")?;
        }
        let embedding = math::rms_norm(&hidden, &self.weights.output_norm, c.rms_epsilon)?;
        let output = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embedding);
        let logits = adapters.multiply("output.weight", output, &embedding)?;
        Ok((embedding, logits))
    }
    fn calculate_fingerprint(&self) -> u64 {
        let mut h = Fingerprint::new();
        h.bytes(b"mamba1-f32-v1");
        let c = &self.config;
        for x in [
            c.vocabulary,
            c.width,
            c.inner,
            c.state_size,
            c.convolution_width,
            c.time_step_rank,
            c.max_context,
            usize::from(c.normalize_dt_b_c),
            self.layer_count(),
        ] {
            h.usize(x);
        }
        h.floats(&[c.rms_epsilon]);
        h.matrix(&self.weights.token_embedding);
        h.floats(&self.weights.output_norm);
        h.usize(usize::from(self.weights.output.is_some()));
        if let Some(output) = &self.weights.output {
            h.matrix(output);
        }
        for layer in &self.weights.layers {
            for m in [
                &layer.input,
                &layer.convolution,
                &layer.x_projection,
                &layer.dt_projection,
                &layer.a,
                &layer.output,
            ] {
                h.matrix(m);
            }
            for v in [
                &layer.norm,
                &layer.convolution_bias,
                &layer.dt_bias,
                &layer.d,
            ] {
                h.floats(v);
            }
            for v in [&layer.dt_norm, &layer.b_norm, &layer.c_norm] {
                h.usize(usize::from(v.is_some()));
                if let Some(v) = v {
                    h.floats(v);
                }
            }
        }
        h.finish()
    }
}
