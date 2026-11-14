//! Scalar sampler behavior ported from `src/llama-sampler.cpp`.
//!
//! Explicit seeds use MT19937 and the libstdc++ canonical uniform conversion.
//! Uniform conversion is a platform contract: C++ does not promise identical
//! `uniform_real_distribution` or `discrete_distribution` across vendors.
//! Equal logits use stable input order here; the original's unstable sort may
//! order ties differently. No backend/device sampling is claimed.

mod gnu_order;

use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub type TokenId = i32;
pub const MAX_CANDIDATES: usize = 1_000_000;
pub const MAX_HISTORY: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenData {
    pub id: TokenId,
    pub logit: f32,
    pub p: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidates {
    pub data: Vec<TokenData>,
    pub selected: Option<usize>,
    pub sorted: bool,
}

impl Candidates {
    pub fn from_logits(logits: &[f32]) -> Result<Self> {
        if logits.len() > MAX_CANDIDATES {
            return Err(Error::new("candidate limit exceeded"));
        }
        Self::new(
            logits
                .iter()
                .enumerate()
                .map(|(id, &logit)| TokenData {
                    id: id as i32,
                    logit,
                    p: 0.0,
                })
                .collect(),
        )
    }
    pub fn new(data: Vec<TokenData>) -> Result<Self> {
        let result = Self {
            data,
            selected: None,
            sorted: false,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn selected_token(&self) -> Option<TokenId> {
        self.selected.and_then(|i| self.data.get(i)).map(|x| x.id)
    }
    pub fn validate(&self) -> Result<()> {
        if self.data.len() > MAX_CANDIDATES {
            return Err(Error::new("candidate limit exceeded"));
        }
        let mut ids = BTreeSet::new();
        for token in &self.data {
            if token.id < 0 || !ids.insert(token.id) {
                return Err(Error::new("candidate IDs must be unique and non-negative"));
            }
            if token.logit.is_nan() || token.logit == f32::INFINITY {
                return Err(Error::new("logits must be finite or negative infinity"));
            }
        }
        if self.sorted && self.data.windows(2).any(|x| x[0].logit < x[1].logit) {
            return Err(Error::new("sorted candidate flag disagrees with logits"));
        }
        if self.selected.is_some_and(|i| i >= self.data.len()) {
            return Err(Error::new("selected candidate index outside array"));
        }
        Ok(())
    }
    /// Native candidate arrays may contain repeated IDs and stale selection or
    /// ordering metadata. Keep those source-defined cases separate from validate.
    pub fn validate_native(&self) -> Result<()> {
        if self.data.len() > MAX_CANDIDATES {
            return Err(Error::new("candidate limit exceeded"));
        }
        for token in &self.data {
            if token.id < 0 {
                return Err(Error::new("negative native candidate ID"));
            }
            if token.logit.is_nan() || token.logit == f32::INFINITY {
                return Err(Error::new("logits must be finite or negative infinity"));
            }
        }
        Ok(())
    }
    fn sort(&mut self) {
        if !self.sorted {
            self.data.sort_by(|a, b| b.logit.total_cmp(&a.logit));
            self.sorted = true;
        }
    }
    pub fn softmax(&mut self, sort: bool) -> Result<()> {
        self.validate()?;
        softmax(self, sort)
    }
}

/// Standard 32-bit Mersenne Twister engine with clonable, fully owned state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }
    pub fn next_u32(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let joined =
                    (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (joined >> 1)
                    ^ if joined & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
    /// `generate_canonical<double,53>` for a 32-bit MT engine on libstdc++.
    pub fn uniform_f64(&mut self) -> f64 {
        let low = self.next_u32() as f64;
        let high = self.next_u32() as f64;
        let result = (low + high * 4294967296.0) / 18446744073709551616.0;
        result.min(f64::from_bits(1.0f64.to_bits() - 1))
    }
    /// `generate_canonical<float,24>` for a 32-bit MT engine on libstdc++.
    pub fn uniform_f32(&mut self) -> f32 {
        let result = self.next_u32() as f32 / 4294967296.0f32;
        result.min(f32::from_bits(1.0f32.to_bits() - 1))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Config {
    Greedy,
    Dist {
        seed: u32,
    },
    TopK {
        k: i32,
    },
    TopP {
        p: f32,
        min_keep: usize,
    },
    MinP {
        p: f32,
        min_keep: usize,
    },
    Typical {
        p: f32,
        min_keep: usize,
    },
    Temperature {
        temperature: f32,
    },
    DynamicTemperature {
        temperature: f32,
        delta: f32,
        exponent: f32,
    },
    Xtc {
        probability: f32,
        threshold: f32,
        min_keep: usize,
        seed: u32,
    },
    Penalties {
        last_n: usize,
        repeat: f32,
        frequency: f32,
        presence: f32,
    },
    TopNSigma {
        n: f32,
    },
    /// Breakers must already be tokenized by the T04 tokenizer.
    Dry {
        multiplier: f32,
        base: f32,
        allowed_length: usize,
        last_n: usize,
        breakers: Vec<Vec<TokenId>>,
    },
    Mirostat {
        n_vocab: usize,
        seed: u32,
        tau: f32,
        eta: f32,
        m: usize,
    },
    MirostatV2 {
        seed: u32,
        tau: f32,
        eta: f32,
    },
    AdaptiveP {
        target: f32,
        decay: f32,
        seed: u32,
    },
    LogitBias {
        biases: Vec<(TokenId, f32)>,
    },
}

impl Config {
    /// Explicit diagnostic/configuration schema for this Rust checkpoint. This
    /// is not a claim of full common/arg.cpp command-line compatibility.
    pub fn from_json(value: &crate::json::Value) -> Result<Self> {
        use crate::json::Value;
        let object = value
            .as_object()
            .ok_or_else(|| Error::new("sampler configuration must be an object"))?;
        let kind = object
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::new("sampler configuration requires string kind"))?;
        let number = |key: &str| -> Result<f32> {
            let number = object
                .get(key)
                .and_then(Value::as_f64)
                .ok_or_else(|| Error::new(format!("sampler parameter {key} must be numeric")))?;
            if !number.is_finite() || number.abs() > f32::MAX as f64 {
                return Err(Error::new(format!(
                    "sampler parameter {key} is outside finite f32 range"
                )));
            }
            Ok(number as f32)
        };
        let size = |key: &str| -> Result<usize> {
            let number = object.get(key).and_then(Value::as_u64).ok_or_else(|| {
                Error::new(format!(
                    "sampler parameter {key} must be an unsigned integer"
                ))
            })?;
            usize::try_from(number)
                .map_err(|_| Error::new(format!("sampler parameter {key} is too large")))
        };
        let seed = || -> Result<u32> {
            u32::try_from(size("seed")?).map_err(|_| Error::new("seed exceeds u32"))
        };
        let config = match kind {
            "greedy" => Self::Greedy,
            "dist" => Self::Dist { seed: seed()? },
            "top-k" => Self::TopK {
                k: i32::try_from(
                    object
                        .get("k")
                        .and_then(Value::as_i64)
                        .ok_or_else(|| Error::new("top-k k must be integer"))?,
                )
                .map_err(|_| Error::new("top-k k exceeds i32"))?,
            },
            "top-p" => Self::TopP {
                p: number("p")?,
                min_keep: size("min_keep")?,
            },
            "min-p" => Self::MinP {
                p: number("p")?,
                min_keep: size("min_keep")?,
            },
            "typical" => Self::Typical {
                p: number("p")?,
                min_keep: size("min_keep")?,
            },
            "temp" => Self::Temperature {
                temperature: number("temperature")?,
            },
            "temp-ext" => Self::DynamicTemperature {
                temperature: number("temperature")?,
                delta: number("delta")?,
                exponent: number("exponent")?,
            },
            "xtc" => Self::Xtc {
                probability: number("probability")?,
                threshold: number("threshold")?,
                min_keep: size("min_keep")?,
                seed: seed()?,
            },
            "penalties" => Self::Penalties {
                last_n: size("last_n")?,
                repeat: number("repeat")?,
                frequency: number("frequency")?,
                presence: number("presence")?,
            },
            "top-n-sigma" => Self::TopNSigma { n: number("n")? },
            "mirostat" => Self::Mirostat {
                n_vocab: size("n_vocab")?,
                seed: seed()?,
                tau: number("tau")?,
                eta: number("eta")?,
                m: size("m")?,
            },
            "mirostat-v2" => Self::MirostatV2 {
                seed: seed()?,
                tau: number("tau")?,
                eta: number("eta")?,
            },
            "adaptive-p" => Self::AdaptiveP {
                target: number("target")?,
                decay: number("decay")?,
                seed: seed()?,
            },
            "dry" => {
                let arrays = object
                    .get("breakers")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Error::new("dry breakers must be an array of token arrays"))?;
                let mut breakers = Vec::new();
                for array in arrays {
                    let values = array
                        .as_array()
                        .ok_or_else(|| Error::new("dry breaker must be an array"))?;
                    breakers.push(
                        values
                            .iter()
                            .map(|x| {
                                x.as_i64()
                                    .and_then(|t| i32::try_from(t).ok())
                                    .ok_or_else(|| Error::new("dry breaker token exceeds i32"))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    );
                }
                Self::Dry {
                    multiplier: number("multiplier")?,
                    base: number("base")?,
                    allowed_length: size("allowed_length")?,
                    last_n: size("last_n")?,
                    breakers,
                }
            }
            "logit-bias" => {
                let arrays = object
                    .get("biases")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Error::new("biases must be an array of [id,bias] pairs"))?;
                let mut biases = Vec::new();
                for array in arrays {
                    let pair = array
                        .as_array()
                        .filter(|x| x.len() == 2)
                        .ok_or_else(|| Error::new("bias entry must have two values"))?;
                    let id = pair[0]
                        .as_i64()
                        .and_then(|x| i32::try_from(x).ok())
                        .ok_or_else(|| Error::new("bias token exceeds i32"))?;
                    let value = pair[1]
                        .as_f64()
                        .ok_or_else(|| Error::new("bias must be numeric"))?;
                    if !value.is_finite() || value.abs() > f32::MAX as f64 {
                        return Err(Error::new("bias exceeds finite f32 range"));
                    }
                    biases.push((id, value as f32));
                }
                Self::LogitBias { biases }
            }
            _ => return Err(Error::new(format!("unsupported sampler kind {kind}"))),
        };
        validate_config(&config)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sampler {
    config: Config,
    rng: Option<Mt19937>,
    history: VecDeque<TokenId>,
    counts: BTreeMap<TokenId, usize>,
    mu: f32,
    weighted_sum: f32,
    total_weight: f32,
    pending: Option<(TokenId, f32)>,
}

impl Sampler {
    pub fn new(mut config: Config) -> Result<Self> {
        validate_config(&config)?;
        if let Config::AdaptiveP { decay, .. } = &mut config {
            *decay = decay.clamp(0.0, 0.99);
        }
        let mut result = Self {
            config,
            rng: None,
            history: VecDeque::new(),
            counts: BTreeMap::new(),
            mu: 0.0,
            weighted_sum: 0.0,
            total_weight: 0.0,
            pending: None,
        };
        result.reset();
        Ok(result)
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn name(&self) -> &'static str {
        match self.config {
            Config::Greedy => "greedy",
            Config::Dist { .. } => "dist",
            Config::TopK { .. } => "top-k",
            Config::TopP { .. } => "top-p",
            Config::MinP { .. } => "min-p",
            Config::Typical { .. } => "typical",
            Config::Temperature { .. } => "temp",
            Config::DynamicTemperature { .. } => "temp-ext",
            Config::Xtc { .. } => "xtc",
            Config::Penalties { .. } => "penalties",
            Config::TopNSigma { .. } => "top-n-sigma",
            Config::Dry { .. } => "dry",
            Config::Mirostat { .. } => "mirostat",
            Config::MirostatV2 { .. } => "mirostat-v2",
            Config::AdaptiveP { .. } => "adaptive-p",
            Config::LogitBias { .. } => "logit-bias",
        }
    }
    pub fn reset(&mut self) {
        self.history.clear();
        self.counts.clear();
        self.pending = None;
        self.mu = 0.0;
        self.weighted_sum = 0.0;
        self.total_weight = 0.0;
        self.rng = match self.config {
            Config::Dist { seed } | Config::Xtc { seed, .. } => Some(Mt19937::new(seed)),
            Config::Mirostat { seed, tau, .. } | Config::MirostatV2 { seed, tau, .. } => {
                self.mu = 2.0 * tau;
                Some(Mt19937::new(seed))
            }
            Config::AdaptiveP {
                seed,
                target,
                decay,
            } => {
                self.weighted_sum = target / (1.0 - decay);
                self.total_weight = 1.0 / (1.0 - decay);
                Some(Mt19937::new(seed))
            }
            _ => None,
        };
    }
    pub fn accept(&mut self, token: TokenId) -> Result<()> {
        if token < 0 {
            return Err(Error::new("cannot accept negative token ID"));
        }
        match self.config {
            Config::Penalties { last_n, .. } => {
                if last_n > 0 {
                    if self.history.len() == last_n {
                        let old = self.history.pop_front().expect("full history");
                        let count = self.counts.get_mut(&old).expect("history frequency exists");
                        *count -= 1;
                        if *count == 0 {
                            self.counts.remove(&old);
                        }
                    }
                    self.history.push_back(token);
                    *self.counts.entry(token).or_default() += 1;
                }
            }
            Config::Dry { last_n, .. } => {
                if last_n > 0 {
                    if self.history.len() == last_n {
                        self.history.pop_front();
                    }
                    self.history.push_back(token);
                }
            }
            Config::AdaptiveP { decay, .. } => {
                if let Some((pending_token, p)) = self.pending.take()
                    && pending_token == token
                {
                    self.weighted_sum = p + decay * self.weighted_sum;
                    self.total_weight = 1.0 + decay * self.total_weight;
                }
            }
            _ => {}
        }
        Ok(())
    }
    /// On an error neither RNG/history nor the caller's candidates are changed.
    pub fn apply(&mut self, candidates: &mut Candidates) -> Result<()> {
        candidates.validate()?;
        let mut next = self.clone();
        let mut values = candidates.clone();
        next.apply_inner(&mut values, false)?;
        values.validate()?;
        *self = next;
        *candidates = values;
        Ok(())
    }
    /// Execute the frozen public C sampler contract without changing the stable
    /// Rust API's documented defaults. Preserves nonterminal selection and source
    /// sorted flags. Returned rows were removed from the *front* by XTC; a C
    /// adapter must advance its data pointer by that count after writing them.
    ///
    /// Native Windows draws use the MSVC 19.51 canonical conversion; other
    /// targets use the existing libstdc++ conversion. Invalid logits and source
    /// undefined numeric cases remain checked errors.
    pub fn apply_native(&mut self, candidates: &mut Candidates) -> Result<Vec<TokenData>> {
        let mut values = candidates.clone();
        let selected = values.selected.take();
        let sorted = std::mem::replace(&mut values.sorted, false);
        values.validate_native()?;
        values.selected = selected;
        values.sorted = sorted;
        let mut next = self.clone();
        let prefix = next.apply_inner(&mut values, true)?;
        let selected = values.selected.take();
        let sorted = std::mem::replace(&mut values.sorted, false);
        values.validate_native()?;
        values.selected = selected;
        values.sorted = sorted;
        *self = next;
        *candidates = values;
        Ok(prefix)
    }

    /// Frozen adaptive-p clone resets its RNG rather than copying it. Its
    /// original clone also omits the pending probability buffer; cloning while a
    /// token awaits acceptance would permit an invalid read in the reference.
    /// Reject that case explicitly, while reproducing defined post-accept clones.
    pub fn clone_native(&self) -> Result<Self> {
        let mut result = self.clone();
        if let Config::AdaptiveP { seed, .. } = self.config {
            if self.pending.is_some() {
                return Err(Error::new(
                    "native adaptive-p clone requires pending token acceptance or reset",
                ));
            }
            result.rng = Some(Mt19937::new(seed));
        }
        Ok(result)
    }

    /// Replace only the native RNG seed/state while preserving adaptive EMA.
    pub fn reseed_native(&mut self, value: u32) -> Result<()> {
        if value == u32::MAX {
            return Err(Error::new("reserved random seed sentinel"));
        }
        match &mut self.config {
            Config::Dist { seed }
            | Config::Xtc { seed, .. }
            | Config::Mirostat { seed, .. }
            | Config::MirostatV2 { seed, .. }
            | Config::AdaptiveP { seed, .. } => *seed = value,
            _ => return Err(Error::new("sampler has no RNG")),
        }
        self.rng = Some(Mt19937::new(value));
        Ok(())
    }

    fn apply_inner(&mut self, c: &mut Candidates, native: bool) -> Result<Vec<TokenData>> {
        // A prior terminal selection cannot survive subsequent reordering.
        if !native {
            c.selected = None;
        }
        let mut prefix = Vec::new();
        if c.data.is_empty() {
            if native {
                match self.config {
                    Config::Greedy => c.selected = Some(0),
                    Config::Dist { .. } => c.selected = None,
                    Config::TopP { p, .. } | Config::Typical { p, .. } if p < 1.0 => {
                        return Err(Error::new("native sampler requires nonempty candidates"));
                    }
                    Config::Mirostat { .. }
                    | Config::MirostatV2 { .. }
                    | Config::AdaptiveP { .. } => {
                        return Err(Error::new("native sampler requires nonempty candidates"));
                    }
                    _ => {}
                }
            }
            return Ok(prefix);
        }
        match &self.config {
            Config::Greedy => {
                let mut best = 0;
                for i in 1..c.data.len() {
                    if c.data[i].logit > c.data[best].logit {
                        best = i;
                    }
                }
                if !native && c.data[best].logit == f32::NEG_INFINITY {
                    return Err(Error::new("all candidate logits are masked"));
                }
                c.selected = Some(best);
            }
            Config::Dist { .. } => {
                sample_logits(c, self.rng.as_mut().expect("distribution RNG"), native)?
            }
            Config::TopK { k } => top_k(c, *k, native),
            Config::TopP { p, min_keep } => {
                if *p < 1.0 {
                    softmax_mode(c, false, native)?;
                    if native && !c.sorted && c.data.len() > 1024 {
                        // The frozen source first sorts 256 items into a separate
                        // buffer. A boundary tied logit can have a different token
                        // order than a full sort, so this optimization is observable.
                        let mut ordered = Candidates {
                            data: c.data.clone(),
                            selected: None,
                            sorted: false,
                        };
                        sort_mode(&mut ordered, 256, true);
                        let mut sum = 0.0f32;
                        let mut keep = c.data.len();
                        for i in 0..c.data.len() {
                            sum += ordered.data[i].p;
                            if sum >= *p && i + 1 >= *min_keep {
                                keep = i + 1;
                                break;
                            }
                            if i == 255 {
                                ordered.data = c.data.clone();
                                ordered.sorted = false;
                                sort_mode(&mut ordered, c.data.len(), true);
                            }
                        }
                        ordered.data.truncate(keep);
                        c.data = ordered.data;
                        c.sorted = true;
                    } else {
                        sort_mode(c, c.data.len(), native);
                        let mut sum = 0.0f32;
                        let mut keep = c.data.len();
                        for (i, token) in c.data.iter().enumerate() {
                            sum += token.p;
                            if sum >= *p && i + 1 >= *min_keep {
                                keep = i + 1;
                                break;
                            }
                        }
                        c.data.truncate(keep);
                    }
                }
            }
            Config::MinP { p, min_keep } => {
                if *p > 0.0 {
                    let max = c
                        .data
                        .iter()
                        .map(|x| x.logit)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let threshold = max + p.ln();
                    let enough = c.data.iter().filter(|x| x.logit >= threshold).count();
                    if !c.sorted && enough > 0 && enough >= *min_keep {
                        c.data.retain(|x| x.logit >= threshold);
                    } else {
                        sort_mode(c, c.data.len(), native);
                        let threshold = c.data[0].logit + p.ln();
                        let keep = (1..c.data.len())
                            .find(|&i| c.data[i].logit < threshold && i >= *min_keep)
                            .unwrap_or(c.data.len());
                        c.data.truncate(keep);
                    }
                }
            }
            Config::Typical { p, min_keep } => {
                if *p < 1.0 {
                    softmax_mode(c, true, native)?;
                    if native && c.data.iter().any(|x| x.p == 0.0) {
                        return Err(Error::new(
                            "native typical entropy is undefined for zero probabilities",
                        ));
                    }
                    // 0 log 0 is defined as zero; the C++ source propagates NaN
                    // on an underflowed p here. This is an intentional correction.
                    let entropy: f32 = c
                        .data
                        .iter()
                        .filter(|x| x.p > 0.0)
                        .map(|x| -x.p * x.p.ln())
                        .sum();
                    let scores: Vec<f32> =
                        c.data.iter().map(|x| (-x.p.ln() - entropy).abs()).collect();
                    let mut indices: Vec<usize> = (0..c.data.len()).collect();
                    if native {
                        native_unstable_sort(&mut indices, |&i| scores[i]);
                    } else {
                        indices.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
                    }
                    let mut sum = 0.0f32;
                    let mut keep = indices.len();
                    for (i, &index) in indices.iter().enumerate() {
                        sum += c.data[index].p;
                        if sum > *p && i + 1 >= *min_keep {
                            keep = i + 1;
                            break;
                        }
                    }
                    c.data = indices[..keep].iter().map(|&i| c.data[i]).collect();
                    c.sorted = false;
                }
            }
            Config::Temperature { temperature } => temperature_impl(c, *temperature),
            Config::DynamicTemperature {
                temperature: temp,
                delta,
                exponent,
            } => {
                if *delta <= 0.0 {
                    temperature_impl(c, *temp);
                } else if c.data.len() > 1 {
                    let min = (temp - delta).max(0.0);
                    let max = temp + delta;
                    if max < 0.0 {
                        return Err(Error::new("dynamic temperature range is negative"));
                    }
                    softmax_mode(c, true, native)?;
                    let max_entropy = -(1.0f32 / c.data.len() as f32).ln();
                    let entropy: f32 = c
                        .data
                        .iter()
                        .filter(|x| x.p > 0.0)
                        .map(|x| -x.p * x.p.ln())
                        .sum();
                    let temp = min + (max - min) * (entropy / max_entropy).powf(*exponent);
                    temperature_impl(c, temp);
                    let max_logit = c.data[0].logit as f64;
                    let mut sum = 0.0f64;
                    for t in &mut c.data {
                        let p = (t.logit as f64 - max_logit).exp();
                        t.p = p as f32;
                        sum += p;
                    }
                    if sum <= 0.0 || !sum.is_finite() {
                        return Err(Error::new("invalid dynamic-temperature distribution"));
                    }
                    for t in &mut c.data {
                        t.p = (t.p as f64 / sum) as f32;
                    }
                }
            }
            Config::Xtc {
                probability,
                threshold,
                min_keep,
                ..
            } => {
                if *probability > 0.0
                    && *threshold <= 0.5
                    && c.data.len() >= 2
                    && draw_f32(self.rng.as_mut().expect("XTC RNG"), native) <= *probability
                {
                    softmax_mode(c, true, native)?;
                    let mut last = 0;
                    for (i, t) in c.data.iter().enumerate() {
                        if t.p >= *threshold {
                            last = i;
                        } else {
                            break;
                        }
                    }
                    if last > 0 && c.data.len() - last >= *min_keep {
                        if native {
                            prefix.extend(c.data.drain(..last));
                        } else {
                            c.data.drain(..last);
                        }
                    }
                }
            }
            Config::Penalties {
                last_n,
                repeat,
                frequency,
                presence,
            } => {
                if *last_n > 0 && (*repeat != 1.0 || *frequency != 0.0 || *presence != 0.0) {
                    for t in &mut c.data {
                        if let Some(&count) = self.counts.get(&t.id) {
                            if t.logit <= 0.0 {
                                t.logit *= repeat;
                            } else {
                                t.logit /= repeat;
                            }
                            t.logit -= count as f32 * frequency + presence;
                        }
                    }
                    c.sorted = false;
                }
            }
            Config::TopNSigma { n } => {
                if *n > 0.0 && c.data.len() > 1 {
                    let valid: Vec<f32> = c
                        .data
                        .iter()
                        .filter(|x| x.logit != f32::NEG_INFINITY)
                        .map(|x| x.logit)
                        .collect();
                    if !valid.is_empty() {
                        let max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mean = valid.iter().sum::<f32>() / valid.len() as f32;
                        let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                            / valid.len() as f32;
                        let threshold = max - n * variance.sqrt();
                        for t in &mut c.data {
                            if t.logit < threshold {
                                t.logit = f32::NEG_INFINITY;
                            }
                        }
                    }
                }
            }
            Config::Dry {
                multiplier,
                base,
                allowed_length,
                last_n,
                breakers,
            } => {
                dry(
                    c,
                    &self.history,
                    *multiplier,
                    *base,
                    *allowed_length,
                    *last_n,
                    breakers,
                );
            }
            Config::Mirostat {
                n_vocab,
                tau,
                eta,
                m,
                ..
            } => {
                softmax_mode(c, true, native)?;
                if c.data.len() > 1 {
                    let mut numerator = 0.0f32;
                    let mut denominator = 0.0f32;
                    for i in 0..(m - 1).min(c.data.len() - 1) {
                        let t = ((i + 2) as f32 / (i + 1) as f32).ln();
                        let b = (c.data[i].p / c.data[i + 1].p).ln();
                        numerator += t * b;
                        denominator += t * t;
                    }
                    let s = numerator / denominator;
                    let epsilon = s - 1.0;
                    // The source casts a possibly NaN/overflowed k to int.
                    // Reject these undefined cases rather than choose silently.
                    let k = ((epsilon * 2.0f32.powf(self.mu))
                        / (1.0 - (*n_vocab as f32).powf(-epsilon)))
                    .powf(1.0 / s);
                    if !k.is_finite() || k > i32::MAX as f32 {
                        return Err(Error::new(
                            "Mirostat v1 slope is degenerate or outside supported range",
                        ));
                    }
                    top_k(c, (k as i32).max(1), native);
                }
                softmax_mode(c, true, native)?;
                let selected = sample_probs(c, self.rng.as_mut().expect("Mirostat RNG"), native)?;
                c.selected = Some(selected);
                self.mu -= eta * (-c.data[selected].p.log2() - tau);
            }
            Config::MirostatV2 { tau, eta, .. } => {
                softmax_mode(c, true, native)?;
                let keep = c
                    .data
                    .iter()
                    .position(|x| -x.p.log2() > self.mu)
                    .unwrap_or(c.data.len())
                    .max(1);
                c.data.truncate(keep);
                softmax_mode(c, true, native)?;
                let selected = sample_probs(c, self.rng.as_mut().expect("Mirostat RNG"), native)?;
                c.selected = Some(selected);
                self.mu -= eta * (-c.data[selected].p.log2() - tau);
            }
            Config::AdaptiveP { target, .. } => {
                softmax_mode(c, false, native)?;
                self.pending = None;
                if *target < 0.0 {
                    c.selected = Some(sample_probs(
                        c,
                        self.rng.as_mut().expect("adaptive RNG"),
                        native,
                    )?);
                } else {
                    let probs: Vec<f32> = c.data.iter().map(|x| x.p).collect();
                    let target = target.clamp(0.0, 1.0);
                    let adapted = if self.total_weight == 0.0 {
                        target
                    } else {
                        2.0 * target - self.weighted_sum / self.total_weight
                    }
                    .clamp(0.0, 1.0);
                    for t in &mut c.data {
                        if t.logit != f32::NEG_INFINITY {
                            let distance = ((t.p - adapted) * (1.0 / 0.3f32)).abs();
                            t.logit = 5.0 - 10.0 * distance * distance / (1.0 + distance);
                        }
                    }
                    // Rust API repairs stale ordering; the C contract retains it.
                    if !native {
                        c.sorted = false;
                    }
                    softmax_mode(c, false, native)?;
                    let index = sample_probs(c, self.rng.as_mut().expect("adaptive RNG"), native)?;
                    c.selected = Some(index);
                    self.pending = Some((c.data[index].id, probs[index]));
                }
            }
            Config::LogitBias { biases } => {
                for t in &mut c.data {
                    if let Some((_, bias)) = biases.iter().find(|(id, _)| *id == t.id) {
                        t.logit += bias;
                    }
                }
                // Source leaves the sorted bit unchanged; clearing it prevents
                // using a stale maximum after a bias reverses rank order.
                if !native && !biases.is_empty() {
                    c.sorted = false;
                }
            }
        }
        Ok(prefix)
    }
}

fn validate_config(config: &Config) -> Result<()> {
    let finite = |values: &[f32]| -> Result<()> {
        if values.iter().all(|x| x.is_finite()) {
            Ok(())
        } else {
            Err(Error::new("sampler parameters must be finite"))
        }
    };
    match config {
        Config::TopP { p, min_keep }
        | Config::MinP { p, min_keep }
        | Config::Typical { p, min_keep } => {
            finite(&[*p])?;
            if *min_keep > MAX_CANDIDATES {
                return Err(Error::new("min_keep exceeds candidate limit"));
            }
        }
        Config::Temperature { temperature } => finite(&[*temperature])?,
        Config::DynamicTemperature {
            temperature,
            delta,
            exponent,
        } => {
            finite(&[*temperature, *delta, *exponent])?;
            if *exponent < 0.0 {
                return Err(Error::new(
                    "negative dynamic temperature exponent is not supported",
                ));
            }
        }
        Config::Xtc {
            probability,
            threshold,
            min_keep,
            ..
        } => {
            finite(&[*probability, *threshold])?;
            if *min_keep > MAX_CANDIDATES {
                return Err(Error::new("min_keep exceeds candidate limit"));
            }
        }
        Config::Penalties {
            last_n,
            repeat,
            frequency,
            presence,
        } => {
            finite(&[*repeat, *frequency, *presence])?;
            if *last_n > MAX_HISTORY || *repeat <= 0.0 {
                return Err(Error::new("invalid penalty window or repeat factor"));
            }
        }
        Config::TopNSigma { n } => finite(&[*n])?,
        Config::Dry {
            multiplier,
            base,
            last_n,
            allowed_length,
            breakers,
        } => {
            finite(&[*multiplier, *base])?;
            if *last_n > MAX_HISTORY
                || *allowed_length > MAX_HISTORY
                || breakers.len() > 4096
                || breakers
                    .iter()
                    .any(|x| x.is_empty() || x.len() > 256 || x.iter().any(|t| *t < 0))
            {
                return Err(Error::new(
                    "DRY history/breaker bounds exceeded or empty breaker",
                ));
            }
        }
        Config::Mirostat {
            n_vocab,
            tau,
            eta,
            m,
            ..
        } => {
            finite(&[*tau, *eta])?;
            if *n_vocab == 0
                || *n_vocab > MAX_CANDIDATES
                || *m < 2
                || *m > MAX_CANDIDATES
                || *eta < 0.0
            {
                return Err(Error::new("invalid Mirostat parameters"));
            }
        }
        Config::MirostatV2 { tau, eta, .. } => {
            finite(&[*tau, *eta])?;
            if *eta < 0.0 {
                return Err(Error::new("negative Mirostat learning rate"));
            }
        }
        Config::AdaptiveP { target, decay, .. } => finite(&[*target, *decay])?,
        Config::LogitBias { biases } => {
            if biases.len() > MAX_CANDIDATES {
                return Err(Error::new("bias count exceeds limit"));
            }
            let mut ids = BTreeSet::new();
            for &(id, bias) in biases {
                if id < 0 || !ids.insert(id) || bias.is_nan() || bias == f32::INFINITY {
                    return Err(Error::new("invalid or duplicate logit bias"));
                }
            }
        }
        _ => {}
    }
    match config {
        Config::Dist { seed }
        | Config::Xtc { seed, .. }
        | Config::Mirostat { seed, .. }
        | Config::MirostatV2 { seed, .. }
        | Config::AdaptiveP { seed, .. }
            if *seed == u32::MAX =>
        {
            Err(Error::new(
                "default random-seed sentinel requires caller-provided entropy; pass an explicit seed",
            ))
        }
        _ => Ok(()),
    }
}

/// Match the current MSVC canonical bit extraction without changing the Rust
/// API's established libstdc++ stream. Both consume two MT words for f64.
fn draw_f64(rng: &mut Mt19937, native: bool) -> f64 {
    if native && cfg!(target_env = "msvc") {
        let low = rng.next_u32() as u64;
        let high = rng.next_u32() as u64;
        (((low >> 11) + (high << 21)) as f64) * (1.0 / 9007199254740992.0)
    } else {
        rng.uniform_f64()
    }
}
fn draw_f32(rng: &mut Mt19937, native: bool) -> f32 {
    if native && cfg!(target_env = "msvc") {
        ((rng.next_u32() >> 8) as f32) * (1.0 / 16777216.0)
    } else {
        rng.uniform_f32()
    }
}

// A minimum heap retains the largest k logits. Choosing the right child on
// equal keys reproduces the tested native partial-sort tie order. Comparison
// deliberately uses >, so signed zero remains a tie as in the C++ comparator.
fn heap_repair(data: &mut [TokenData], start: usize, end: usize, value: TokenData) {
    let mut hole = start;
    while hole * 2 + 1 < end {
        let left = hole * 2 + 1;
        let right = left + 1;
        let child = if right < end && data[right].logit <= data[left].logit {
            right
        } else {
            left
        };
        data[hole] = data[child];
        hole = child;
    }
    while hole > start {
        let parent = (hole - 1) / 2;
        if data[parent].logit <= value.logit {
            break;
        }
        data[hole] = data[parent];
        hole = parent;
    }
    data[hole] = value;
}
fn native_partial_sort(data: &mut [TokenData], k: usize) {
    if k == 0 {
        return;
    }
    for root in (0..k / 2).rev() {
        let value = data[root];
        heap_repair(data, root, k, value);
    }
    for candidate in k..data.len() {
        if data[candidate].logit > data[0].logit {
            let value = data[candidate];
            data[candidate] = data[0];
            heap_repair(data, 0, k, value);
        }
    }
    for end in (1..k).rev() {
        let value = data[end];
        data[end] = data[0];
        heap_repair(data, 0, end, value);
    }
}

// Ordering compatibility with the pinned MSVC 14.51 std::sort implementation.
// This independently expressed Rust adaptation preserves its median selection,
// three-way partition and insertion cutoff because tied token IDs are observable.
// Algorithm provenance: Microsoft STL (Apache-2.0 WITH LLVM-exception); see
// this crate's THIRD_PARTY_NOTICES.md. The rest of this file keeps its own license.
fn native_unstable_sort<T: Copy>(values: &mut [T], key: impl Fn(&T) -> f32 + Copy) {
    if cfg!(target_env = "msvc") {
        msvc_sort_range(values, 0, values.len(), values.len(), key);
    } else {
        if cfg!(all(target_os = "linux", target_env = "gnu")) {
            gnu_order::sort(values, key);
            return;
        }
        // Tie ordering on other standard-library profiles is not claimed.
        values.sort_by(|a, b| {
            key(a)
                .partial_cmp(&key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}
fn median_three<T: Copy>(v: &mut [T], a: usize, b: usize, c: usize, key: impl Fn(&T) -> f32) {
    if key(&v[b]) < key(&v[a]) {
        v.swap(a, b);
    }
    if key(&v[c]) < key(&v[b]) {
        v.swap(b, c);
        if key(&v[b]) < key(&v[a]) {
            v.swap(a, b);
        }
    }
}
fn sort_heap_repair<T: Copy>(
    v: &mut [T],
    root: usize,
    end: usize,
    item: T,
    key: impl Fn(&T) -> f32,
) {
    let mut hole = root;
    while hole * 2 + 1 < end {
        let left = hole * 2 + 1;
        let right = left + 1;
        let child = if right < end && key(&v[right]) >= key(&v[left]) {
            right
        } else {
            left
        };
        v[hole] = v[child];
        hole = child;
    }
    while hole > root {
        let parent = (hole - 1) / 2;
        if key(&v[parent]) >= key(&item) {
            break;
        }
        v[hole] = v[parent];
        hole = parent;
    }
    v[hole] = item;
}
fn msvc_sort_range<T: Copy>(
    v: &mut [T],
    mut lo: usize,
    mut hi: usize,
    mut budget: usize,
    key: impl Fn(&T) -> f32 + Copy,
) {
    loop {
        if hi - lo <= 32 {
            for next in lo + usize::from(lo < hi)..hi {
                let item = v[next];
                let mut at = next;
                while at > lo && key(&item) < key(&v[at - 1]) {
                    v[at] = v[at - 1];
                    at -= 1;
                }
                v[at] = item;
            }
            return;
        }
        if budget == 0 {
            let part = &mut v[lo..hi];
            for root in (0..part.len() / 2).rev() {
                let item = part[root];
                sort_heap_repair(part, root, part.len(), item, key);
            }
            for end in (1..part.len()).rev() {
                let item = part[end];
                part[end] = part[0];
                sort_heap_repair(part, 0, end, item, key);
            }
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let last = hi - 1;
        if last - lo > 40 {
            let step = (hi - lo) / 8;
            median_three(v, lo, lo + step, lo + 2 * step, key);
            median_three(v, mid - step, mid, mid + step, key);
            median_three(v, last - 2 * step, last - step, last, key);
            median_three(v, lo + step, mid, last - step, key);
        } else {
            median_three(v, lo, mid, last, key);
        }
        let mut equal_lo = mid;
        let mut equal_hi = mid + 1;
        while equal_lo > lo && key(&v[equal_lo - 1]) == key(&v[equal_lo]) {
            equal_lo -= 1;
        }
        while equal_hi < hi && key(&v[equal_hi]) == key(&v[equal_lo]) {
            equal_hi += 1;
        }
        let mut scan_hi = equal_hi;
        let mut scan_lo = equal_lo;
        loop {
            while scan_hi < hi {
                if key(&v[equal_lo]) < key(&v[scan_hi]) {
                    scan_hi += 1;
                } else if key(&v[scan_hi]) < key(&v[equal_lo]) {
                    break;
                } else {
                    v.swap(equal_hi, scan_hi);
                    equal_hi += 1;
                    scan_hi += 1;
                }
            }
            while scan_lo > lo {
                let prev = scan_lo - 1;
                if key(&v[prev]) < key(&v[equal_lo]) {
                    scan_lo -= 1;
                } else if key(&v[equal_lo]) < key(&v[prev]) {
                    break;
                } else {
                    equal_lo -= 1;
                    v.swap(equal_lo, prev);
                    scan_lo -= 1;
                }
            }
            if scan_lo == lo && scan_hi == hi {
                break;
            }
            if scan_lo == lo {
                if equal_hi != scan_hi {
                    v.swap(equal_lo, equal_hi);
                }
                equal_hi += 1;
                v.swap(equal_lo, scan_hi);
                equal_lo += 1;
                scan_hi += 1;
            } else if scan_hi == hi {
                scan_lo -= 1;
                equal_lo -= 1;
                v.swap(scan_lo, equal_lo);
                equal_hi -= 1;
                v.swap(equal_lo, equal_hi);
            } else {
                scan_lo -= 1;
                v.swap(scan_hi, scan_lo);
                scan_hi += 1;
            }
        }
        budget = (budget / 2) + (budget / 4);
        if equal_lo - lo < hi - equal_hi {
            msvc_sort_range(v, lo, equal_lo, budget, key);
            lo = equal_hi;
        } else {
            msvc_sort_range(v, equal_hi, hi, budget, key);
            hi = equal_lo;
        }
    }
}

fn sort_mode(c: &mut Candidates, k: usize, native: bool) {
    if !native {
        c.sort();
        return;
    }
    if c.sorted {
        return;
    }
    if k <= 128 {
        native_partial_sort(&mut c.data, k);
    } else {
        // Frozen llama.cpp groups larger sorts into 128 logarithmic buckets.
        // Bucket insertion preserves source input order.
        let mut buckets: Vec<Vec<TokenData>> = (0..128).map(|_| Vec::new()).collect();
        for &item in &c.data {
            let bucket = ((6.4f32 * item.logit + 64.0f32) as i32).clamp(0, 127) as usize;
            buckets[bucket].push(item);
        }
        let mut output = Vec::with_capacity(k);
        for bucket in buckets.iter_mut().rev() {
            if output.len() + bucket.len() < k {
                native_unstable_sort(bucket, |t| -t.logit);
                output.extend_from_slice(bucket);
            } else {
                let needed = k - output.len();
                native_partial_sort(bucket, needed);
                output.extend_from_slice(&bucket[..needed]);
                break;
            }
        }
        c.data[..k].copy_from_slice(&output);
    }
    c.data.truncate(k);
    c.sorted = true;
}

fn top_k(c: &mut Candidates, k: i32, native: bool) {
    if k > 0 {
        sort_mode(c, (k as usize).min(c.data.len()), native);
        c.data.truncate(k as usize);
    }
}

fn temperature_impl(c: &mut Candidates, t: f32) {
    if t <= 0.0 {
        let mut best = 0;
        for i in 1..c.data.len() {
            if c.data[i].logit > c.data[best].logit {
                best = i;
            }
        }
        for (i, token) in c.data.iter_mut().enumerate() {
            if i != best {
                token.logit = f32::NEG_INFINITY;
            }
        }
    } else {
        for token in &mut c.data {
            token.logit /= t;
        }
    }
}

fn softmax(c: &mut Candidates, sort: bool) -> Result<()> {
    softmax_mode(c, sort, false)
}
fn softmax_mode(c: &mut Candidates, sort: bool, native: bool) -> Result<()> {
    if c.data.is_empty() {
        return Err(Error::new("cannot normalize an empty candidate set"));
    }
    if sort {
        sort_mode(c, c.data.len(), native);
    }
    let max = if c.sorted {
        c.data[0].logit
    } else {
        c.data
            .iter()
            .map(|x| x.logit)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    if !max.is_finite() {
        return Err(Error::new("all candidate logits are masked"));
    }
    let mut sum = 0.0f32;
    for t in &mut c.data {
        t.p = (t.logit - max).exp();
        sum += t.p;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return Err(Error::new("invalid softmax normalization"));
    }
    for t in &mut c.data {
        t.p /= sum;
    }
    Ok(())
}

fn sample_logits(c: &mut Candidates, rng: &mut Mt19937, native: bool) -> Result<()> {
    if c.data.len() == 1 {
        if c.data[0].logit == f32::NEG_INFINITY {
            return Err(Error::new("all candidate logits are masked"));
        }
        let _ = draw_f64(rng, native);
        c.data[0].p = 1.0;
        c.selected = Some(0);
        return Ok(());
    }
    let max = if native && c.sorted {
        c.data[0].logit
    } else {
        c.data
            .iter()
            .map(|x| x.logit)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    if !max.is_finite() {
        return Err(Error::new("all candidate logits are masked"));
    }
    let mut sum = 0.0f64;
    for t in &mut c.data {
        t.p = (t.logit - max).exp();
        sum += t.p as f64;
    }
    let target = sum * draw_f64(rng, native);
    let mut running = 0.0f64;
    let mut selected = None;
    for (i, t) in c.data.iter_mut().enumerate() {
        if selected.is_none() {
            running += t.p as f64;
            if running >= target {
                selected = Some(i);
            }
        }
        t.p = (t.p as f64 / sum) as f32;
    }
    c.selected = Some(selected.unwrap_or(c.data.len() - 1));
    Ok(())
}

/// Mirrors libstdc++'s discrete-distribution draw after a float softmax.
fn sample_probs(c: &Candidates, rng: &mut Mt19937, native: bool) -> Result<usize> {
    if c.data.is_empty() {
        return Err(Error::new("empty discrete distribution"));
    }
    if c.data.len() == 1 {
        if native && cfg!(target_env = "msvc") {
            let _ = draw_f64(rng, true);
        }
        return Ok(0);
    }
    let sum: f64 = c.data.iter().map(|x| x.p as f64).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return Err(Error::new("invalid probability sum"));
    }
    let target = draw_f64(rng, native);
    let mut running = 0.0f64;
    for (i, t) in c.data.iter().enumerate() {
        running += t.p as f64 / sum;
        if running >= target {
            return Ok(i);
        }
    }
    Ok(c.data.len() - 1)
}

fn dry(
    c: &mut Candidates,
    history: &VecDeque<TokenId>,
    multiplier: f32,
    base: f32,
    allowed: usize,
    last_n: usize,
    breakers: &[Vec<TokenId>],
) {
    if multiplier == 0.0 || base < 1.0 || last_n == 0 || history.len() <= allowed {
        return;
    }
    let reversed: Vec<TokenId> = history.iter().rev().take(last_n).copied().collect();
    let n = reversed.len();
    let mut limit = n;
    for i in 0..n {
        let longest = breakers
            .iter()
            .filter(|seq| {
                seq[0] == reversed[i]
                    && seq.len() - 1 <= i
                    && seq
                        .iter()
                        .enumerate()
                        .skip(1)
                        .all(|(j, token)| *token == reversed[i - j])
            })
            .map(|x| x.len() - 1)
            .max();
        if let Some(tail) = longest {
            limit = i - tail;
            break;
        }
    }
    if limit < allowed {
        return;
    }
    // Reverse Z values give each previous occurrence's common suffix length.
    let mut z = vec![0usize; n];
    let (mut left, mut right) = (0usize, 0usize);
    for i in 1..n {
        if i <= right {
            z[i] = z[i - left].min(right - i + 1);
        }
        while i + z[i] < n && reversed[z[i]] == reversed[i + z[i]] {
            z[i] += 1;
        }
        if z[i] > 0 && i + z[i] - 1 > right {
            left = i;
            right = i + z[i] - 1;
        }
    }
    let mut repeats = BTreeMap::<TokenId, usize>::new();
    for i in 1..n {
        let count = z[i].min(limit);
        if count >= allowed {
            repeats
                .entry(reversed[i - 1])
                .and_modify(|v| *v = (*v).max(count))
                .or_insert(count);
        }
    }
    let max_exponent = if base > 1.000001 {
        (88.72284f32 / base.ln()) as usize
    } else {
        0
    };
    for t in &mut c.data {
        if breakers.iter().any(|x| x.len() == 1 && x[0] == t.id) {
            continue;
        }
        if let Some(&count) = repeats.get(&t.id) {
            let exponent = if max_exponent > 0 {
                (count - allowed).min(max_exponent)
            } else {
                count - allowed
            };
            t.logit -= multiplier * base.powi(exponent as i32);
        }
    }
    c.sorted = false;
}

/// CPU DRY state for the frozen C contract. Raw breaker preprocessing belongs
/// to the vocabulary adapter; each map value excludes its head token.
#[derive(Debug, Clone)]
pub struct NativeDry {
    multiplier: f32,
    base: f32,
    allowed_length: i32,
    last_n: usize,
    breakers: BTreeMap<TokenId, Vec<Vec<TokenId>>>,
    history: VecDeque<TokenId>,
}
impl NativeDry {
    pub fn new(
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        last_n: usize,
        breakers: BTreeMap<TokenId, Vec<Vec<TokenId>>>,
    ) -> Result<Self> {
        if !multiplier.is_finite() || !base.is_finite() || last_n > MAX_HISTORY {
            return Err(Error::new(
                "invalid native DRY parameters or history budget",
            ));
        }
        let mut sequences = 0usize;
        let mut tokens = 0usize;
        for (&head, tails) in &breakers {
            if head < 0 {
                return Err(Error::new("negative DRY breaker head"));
            }
            for tail in tails {
                sequences = sequences
                    .checked_add(1)
                    .ok_or_else(|| Error::new("DRY breaker count overflow"))?;
                tokens = tokens
                    .checked_add(tail.len())
                    .ok_or_else(|| Error::new("DRY breaker token count overflow"))?;
                if tail.len() > 20 || tail.iter().any(|&t| t < 0) {
                    return Err(Error::new("native DRY breaker tail exceeds source limit"));
                }
            }
        }
        if sequences > MAX_CANDIDATES || tokens > 8 * MAX_CANDIDATES {
            return Err(Error::new("native DRY processed-breaker budget exceeded"));
        }
        Ok(Self {
            multiplier,
            base,
            allowed_length,
            last_n,
            breakers,
            history: VecDeque::new(),
        })
    }
    pub fn accept(&mut self, token: TokenId) -> Result<()> {
        if token < 0 {
            return Err(Error::new("cannot accept negative token ID"));
        }
        if self.multiplier == 0.0 || self.base < 1.0 || self.last_n == 0 {
            return Ok(());
        }
        if self.history.len() == self.last_n {
            self.history.pop_front();
        }
        self.history.push_back(token);
        Ok(())
    }
    pub fn reset(&mut self) {
        self.history.clear();
    }
    pub fn apply(&self, c: &mut Candidates) -> Result<()> {
        if self.multiplier == 0.0 || self.base < 1.0 || self.last_n == 0 {
            return Ok(());
        }
        let reversed: Vec<TokenId> = self
            .history
            .iter()
            .rev()
            .take(self.last_n)
            .copied()
            .collect();
        let n = reversed.len();
        if n as i64 <= i64::from(self.allowed_length) {
            return Ok(());
        }
        let mut limit = n;
        for i in 0..n {
            let Some(tails) = self.breakers.get(&reversed[i]) else {
                continue;
            };
            let longest = tails
                .iter()
                .filter(|tail| {
                    tail.len() <= i
                        && tail
                            .iter()
                            .enumerate()
                            .all(|(j, &token)| token == reversed[i - j - 1])
                })
                .map(Vec::len)
                .max();
            if let Some(tail) = longest {
                limit = i - tail;
                break;
            }
        }
        if (limit as i64) < i64::from(self.allowed_length) {
            return Ok(());
        }
        // This is the source Z-box algorithm, including clamped cached values.
        let mut counts = vec![0usize; n];
        let (mut right, mut left) = (0usize, 0usize);
        for k in 1..n {
            if k > right {
                let mut length = 0;
                while length + k < n && reversed[length] == reversed[length + k] {
                    length += 1;
                }
                counts[k] = length.min(limit);
                if length > 0 {
                    left = k;
                    right = k + length - 1;
                }
            } else {
                let p = k - left;
                let remaining = right - k + 1;
                if counts[p] < remaining {
                    counts[k] = counts[p].min(limit);
                } else {
                    let mut i = right + 1;
                    while i < n && reversed[i] == reversed[i - k] {
                        i += 1;
                    }
                    counts[k] = (i - k).min(limit);
                    left = k;
                    right = i - 1;
                }
            }
        }
        let mut repeats = BTreeMap::<TokenId, usize>::new();
        for k in 1..n {
            let count = counts[k];
            if count as i64 >= i64::from(self.allowed_length) {
                repeats
                    .entry(reversed[k - 1])
                    .and_modify(|v| *v = (*v).max(count))
                    .or_insert(count);
            }
        }
        let max_exponent = if self.base > 1.000001 {
            (88.722_84f32 / self.base.ln()) as i32
        } else {
            0
        };
        let mut updated = Vec::with_capacity(c.data.len());
        for token in &c.data {
            let mut logit = token.logit;
            if let Some(&count) = repeats.get(&token.id) {
                let single = self
                    .breakers
                    .get(&token.id)
                    .is_some_and(|tails| tails.iter().any(Vec::is_empty));
                if !single {
                    let exponent = i64::try_from(count).unwrap() - i64::from(self.allowed_length);
                    let exponent = i32::try_from(exponent).map_err(|_| {
                        Error::new("native DRY exponent exceeds defined source integer range")
                    })?;
                    let exponent = if max_exponent > 0 {
                        exponent.min(max_exponent)
                    } else {
                        exponent
                    };
                    // std::pow(float, int) promotes both operands to double.
                    let penalty = (f64::from(self.multiplier)
                        * f64::from(self.base).powf(f64::from(exponent)))
                        as f32;
                    logit -= penalty;
                    if logit.is_nan() || logit == f32::INFINITY {
                        return Err(Error::new("native DRY produced an invalid logit"));
                    }
                }
            }
            updated.push(logit);
        }
        for (token, logit) in c.data.iter_mut().zip(updated) {
            token.logit = logit;
        }
        c.sorted = false;
        Ok(())
    }
}

/// Frozen infill filtering, including retained selection/sorted flags and the
/// fallback token's unmodified probability payload. Vocabulary callbacks supply
/// actual special=false pieces and normalized EOG metadata.
pub fn infill_native(
    candidates: &mut Candidates,
    mut piece: impl FnMut(TokenId) -> Result<Vec<u8>>,
    mut is_eog: impl FnMut(TokenId) -> bool,
    fallback: TokenId,
) -> Result<()> {
    if candidates.data.is_empty() {
        return Err(Error::new("native infill requires candidates"));
    }
    let mut c = candidates.clone();
    softmax_mode(&mut c, true, true)?;
    let (mut text_sum, mut eog_sum) = (0.0f32, 0.0f32);
    for t in &c.data {
        if is_eog(t.id) {
            eog_sum += t.p;
        } else {
            text_sum += t.p;
        }
    }
    if 3.0 * eog_sum * c.data.len() as f32 > text_sum {
        let mut sum = 0.0f32;
        c.data.retain(|t| {
            let keep = is_eog(t.id);
            if keep {
                sum += t.p;
            }
            keep
        });
        for t in &mut c.data {
            t.p /= sum;
        }
        *candidates = c;
        return Ok(());
    }
    if c.data.len() > 4096 {
        return Err(Error::new(
            "native infill prefix comparison budget exceeds 4096 candidates",
        ));
    }
    let mut pieces = BTreeMap::new();
    for t in &c.data {
        if let std::collections::btree_map::Entry::Vacant(entry) = pieces.entry(t.id) {
            entry.insert(piece(t.id)?);
        }
    }
    for i0 in 0..c.data.len() {
        for i1 in 0..c.data.len() {
            if c.data[i0].logit == f32::NEG_INFINITY {
                break;
            }
            if i0 == i1 || c.data[i1].logit == f32::NEG_INFINITY {
                continue;
            }
            let a = &pieces[&c.data[i0].id];
            let b = &pieces[&c.data[i1].id];
            if !a.is_empty() && b.starts_with(a) {
                let (dst, src) = if c.data[i1].p > c.data[i0].p {
                    (i1, i0)
                } else {
                    (i0, i1)
                };
                c.data[dst].p += c.data[src].p;
                c.data[src].logit = f32::NEG_INFINITY;
                c.data[src].p = 0.0;
            }
        }
    }
    let (mut non_eog, mut sum) = (0usize, 0.0f32);
    let original_first = c.data[0];
    c.data.retain(|t| {
        let end = is_eog(t.id);
        if t.p < 0.2 && !end {
            return false;
        }
        if !end {
            non_eog += 1;
        }
        sum += t.p;
        true
    });
    if non_eog == 0 {
        if fallback < 0 {
            return Err(Error::new(
                "native infill requires an EOT or EOS fallback token",
            ));
        }
        let mut first = c.data.first().copied().unwrap_or(original_first);
        first.id = fallback;
        first.logit = 1.0;
        c.data.clear();
        c.data.push(first);
        *candidates = c;
        return Ok(());
    }
    for t in &mut c.data {
        t.p /= sum;
    }
    let threshold = (1.0f64 / (non_eog + 1) as f64) as f32;
    let mut sum = 0.0f32;
    c.data.retain(|t| {
        if t.p < threshold && !is_eog(t.id) {
            return false;
        }
        sum += t.p;
        true
    });
    for t in &mut c.data {
        t.p /= sum;
    }
    *candidates = c;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Chain {
    samplers: Vec<Sampler>,
}

impl Chain {
    pub fn new(configs: impl IntoIterator<Item = Config>) -> Result<Self> {
        let mut samplers = Vec::new();
        for config in configs {
            if samplers.len() >= 128 {
                return Err(Error::new("sampler chain exceeds 128 stages"));
            }
            samplers.push(Sampler::new(config)?);
        }
        Ok(Self { samplers })
    }
    pub fn samplers(&self) -> &[Sampler] {
        &self.samplers
    }
    /// Conservative state-copy accounting used to bound speculative snapshots.
    /// Includes inline RNG state, vector capacities, owned configuration and a
    /// 512-byte allowance per BTreeMap entry (including node/slack overhead).
    /// Candidate/logit arrays are separately bounded by MAX_CANDIDATES.
    pub fn snapshot_size_bound(&self) -> Result<usize> {
        fn allocation(total: &mut usize, count: usize, element: usize) -> Result<()> {
            let payload = count
                .checked_mul(element)
                .ok_or_else(|| Error::new("sampler state size overflow"))?;
            *total = total
                .checked_add(payload)
                .and_then(|n| n.checked_add(if count > 0 { 64 } else { 0 }))
                .ok_or_else(|| Error::new("sampler state size overflow"))?;
            Ok(())
        }
        let mut bytes = std::mem::size_of::<Self>();
        allocation(
            &mut bytes,
            self.samplers.capacity(),
            std::mem::size_of::<Sampler>(),
        )?;
        for sampler in &self.samplers {
            allocation(
                &mut bytes,
                sampler.history.capacity(),
                std::mem::size_of::<TokenId>(),
            )?;
            allocation(&mut bytes, sampler.counts.len(), 512)?;
            match &sampler.config {
                Config::Dry { breakers, .. } => {
                    allocation(
                        &mut bytes,
                        breakers.capacity(),
                        std::mem::size_of::<Vec<TokenId>>(),
                    )?;
                    for breaker in breakers {
                        allocation(
                            &mut bytes,
                            breaker.capacity(),
                            std::mem::size_of::<TokenId>(),
                        )?;
                    }
                }
                Config::LogitBias { biases } => allocation(
                    &mut bytes,
                    biases.capacity(),
                    std::mem::size_of::<(TokenId, f32)>(),
                )?,
                _ => {}
            }
        }
        Ok(bytes)
    }
    pub fn reset(&mut self) {
        for s in &mut self.samplers {
            s.reset();
        }
    }
    pub fn accept(&mut self, token: TokenId) -> Result<()> {
        if token < 0 {
            return Err(Error::new("cannot accept negative token ID"));
        }
        for s in &mut self.samplers {
            s.accept(token)?;
        }
        Ok(())
    }
    pub fn apply(&mut self, candidates: &mut Candidates) -> Result<()> {
        candidates.validate()?;
        let mut next = self.clone();
        let mut values = candidates.clone();
        for sampler in &mut next.samplers {
            sampler.apply_inner(&mut values, false)?;
            values.validate()?;
        }
        *self = next;
        *candidates = values;
        Ok(())
    }
    pub fn sample(&mut self, logits: &[f32]) -> Result<TokenId> {
        let mut next = self.clone();
        let mut c = Candidates::from_logits(logits)?;
        next.apply(&mut c)?;
        let token = c.selected_token().ok_or_else(|| {
            Error::new("chain did not select a token; end with greedy, dist or adaptive sampler")
        })?;
        *self = next;
        Ok(token)
    }
    pub fn sample_and_accept(&mut self, logits: &[f32]) -> Result<TokenId> {
        let token = self.sample(logits)?;
        self.accept(token)?;
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn probs(values: &[f32]) -> Candidates {
        Candidates::from_logits(&values.iter().map(|x| x.ln()).collect::<Vec<_>>()).unwrap()
    }
    fn compare(values: &[f32], config: Config, expected: &[f32]) {
        let mut c = probs(values);
        Sampler::new(config).unwrap().apply(&mut c).unwrap();
        c.softmax(false).unwrap();
        assert_eq!(c.data.len(), expected.len());
        for (token, &p) in c.data.iter().zip(expected) {
            assert!((token.p - p).abs() <= 2e-5, "{} versus {p}", token.p);
        }
    }
    #[test]
    fn mt19937_standard_seed_reference_vector() {
        let mut rng = Mt19937::new(5489);
        assert_eq!(
            (0..10).map(|_| rng.next_u32()).collect::<Vec<_>>(),
            vec![
                3499211612, 581869302, 3890346734, 3586334585, 545404204, 4161255391, 3922919429,
                949333985, 2715962298, 1323567403
            ]
        );
    }
    #[test]
    fn reference_sampling_probability_cases() {
        // Expected values are from the frozen tests/test-sampling.cpp corpus.
        compare(
            &[0.1, 0.2, 0.3, 0.4],
            Config::TopK { k: 3 },
            &[0.44444, 0.33333, 0.22222],
        );
        compare(
            &[0.1, 0.2, 0.3, 0.4],
            Config::TopP {
                p: 0.7,
                min_keep: 1,
            },
            &[0.571429, 0.428571],
        );
        compare(
            &[0.1, 0.2, 0.3, 0.4],
            Config::MinP {
                p: 0.26,
                min_keep: 1,
            },
            &[0.2 / 0.9, 0.3 / 0.9, 0.4 / 0.9],
        );
        compare(
            &[0.1, 0.2, 0.3, 0.4],
            Config::MinP {
                p: 1.05,
                min_keep: 1,
            },
            &[1.0],
        );
        compare(
            &[0.97, 0.01, 0.01, 0.01],
            Config::Typical {
                p: 0.5,
                min_keep: 1,
            },
            &[1.0],
        );
        compare(
            &[0.4, 0.2, 0.2, 0.2],
            Config::Typical {
                p: 0.5,
                min_keep: 1,
            },
            &[1.0 / 3.0; 3],
        );
        compare(
            &[0.1, 0.2, 0.3, 0.4],
            Config::Temperature { temperature: 0.0 },
            &[0.0, 0.0, 0.0, 1.0],
        );
        compare(
            &[0.1, 0.2, 0.3, 0.4],
            Config::TopNSigma { n: 1.0 },
            &[0.0, 0.0, 0.428571, 0.571429],
        );
    }
    #[test]
    fn xtc_source_case() {
        compare(
            &[0.4, 0.3, 0.2, 0.1],
            Config::Xtc {
                probability: 1.0,
                threshold: 0.19,
                min_keep: 1,
                seed: 1,
            },
            &[2.0 / 3.0, 1.0 / 3.0],
        );
    }
    #[test]
    fn penalty_window_positive_negative_and_reset() {
        let mut sampler = Sampler::new(Config::Penalties {
            last_n: 3,
            repeat: 2.0,
            frequency: 0.5,
            presence: 0.25,
        })
        .unwrap();
        for t in [0, 1, 0, 2] {
            sampler.accept(t).unwrap();
        }
        let mut c = Candidates::from_logits(&[2.0, -1.0, 1.0, 0.0]).unwrap();
        sampler.apply(&mut c).unwrap();
        assert_eq!(
            c.data.iter().map(|t| t.logit).collect::<Vec<_>>(),
            vec![0.25, -2.75, -0.25, 0.0]
        );
        sampler.reset();
        let mut c = Candidates::from_logits(&[2.0, -1.0]).unwrap();
        sampler.apply(&mut c).unwrap();
        assert_eq!(c.data[0].logit, 2.0);
        assert_eq!(c.data[1].logit, -1.0);
    }
    #[test]
    fn dry_source_cases_and_breakers() {
        let mut s = Sampler::new(Config::Dry {
            multiplier: 1.0,
            base: 1.1,
            allowed_length: 2,
            last_n: 5,
            breakers: vec![],
        })
        .unwrap();
        for t in [0, 1, 2, 0, 1] {
            s.accept(t).unwrap();
        }
        let mut c = probs(&[0.25; 4]);
        s.apply(&mut c).unwrap();
        c.softmax(false).unwrap();
        for (t, p) in c.data.iter().zip([0.296923, 0.296923, 0.109232, 0.296923]) {
            assert!((t.p - p).abs() < 2e-6);
        }
        let mut s = Sampler::new(Config::Dry {
            multiplier: 1.0,
            base: 1.1,
            allowed_length: 2,
            last_n: 6,
            breakers: vec![vec![3]],
        })
        .unwrap();
        for t in [0, 1, 3, 4, 0, 1] {
            s.accept(t).unwrap();
        }
        let mut c = probs(&[0.2; 5]);
        s.apply(&mut c).unwrap();
        c.softmax(false).unwrap();
        assert!(c.data.iter().all(|x| (x.p - 0.2).abs() < 1e-6));
    }
    #[test]
    fn distribution_clone_reset_singleton_consumption() {
        let mut a = Sampler::new(Config::Dist { seed: 42 }).unwrap();
        let mut b = a.clone();
        let mut singleton = Candidates::from_logits(&[0.0]).unwrap();
        a.apply(&mut singleton).unwrap();
        let mut dummy = Candidates::from_logits(&[0.0, 1.0, 2.0]).unwrap();
        b.apply(&mut dummy).unwrap();
        for _ in 0..20 {
            let mut x = Candidates::from_logits(&[0.0, 1.0, 2.0]).unwrap();
            let mut y = x.clone();
            a.apply(&mut x).unwrap();
            b.apply(&mut y).unwrap();
            assert_eq!(x, y);
        }
        a.reset();
        assert_eq!(a, Sampler::new(Config::Dist { seed: 42 }).unwrap());
    }
    #[test]
    fn errors_do_not_consume_rng_or_mutate_candidates() {
        let mut sampler = Sampler::new(Config::Dist { seed: 42 }).unwrap();
        let initial = sampler.clone();
        let mut c = Candidates::from_logits(&[f32::NEG_INFINITY; 3]).unwrap();
        let old = c.clone();
        assert!(sampler.apply(&mut c).is_err());
        assert_eq!(initial, sampler);
        assert_eq!(c, old);
        assert!(Candidates::from_logits(&[f32::NAN]).is_err());
        assert!(Sampler::new(Config::Dist { seed: u32::MAX }).is_err());
    }
    #[test]
    fn chain_order_is_observable() {
        let logits = [2.0, 1.0, 0.0];
        let mut before = Chain::new([
            Config::Temperature { temperature: 2.0 },
            Config::TopP {
                p: 0.7,
                min_keep: 1,
            },
            Config::Greedy,
        ])
        .unwrap();
        let mut after = Chain::new([
            Config::TopP {
                p: 0.7,
                min_keep: 1,
            },
            Config::Temperature { temperature: 2.0 },
            Config::Greedy,
        ])
        .unwrap();
        let mut x = Candidates::from_logits(&logits).unwrap();
        let mut y = x.clone();
        before.apply(&mut x).unwrap();
        after.apply(&mut y).unwrap();
        assert_ne!(x.data[0].p, y.data[0].p);
        assert_eq!(x.selected_token(), Some(0));
    }
    #[test]
    fn adaptive_accept_only_commits_matching_pending_token() {
        let mut s = Sampler::new(Config::AdaptiveP {
            target: 0.3,
            decay: 0.8,
            seed: 7,
        })
        .unwrap();
        let old = s.weighted_sum;
        let mut c = probs(&[0.1, 0.2, 0.3, 0.4]);
        s.apply(&mut c).unwrap();
        let mut rejected = s.clone();
        rejected.accept(999).unwrap();
        assert_eq!(rejected.weighted_sum, old);
        s.accept(c.selected_token().unwrap()).unwrap();
        assert_ne!(s.weighted_sum, old);
        s.reset();
        assert_eq!(
            s,
            Sampler::new(Config::AdaptiveP {
                target: 0.3,
                decay: 0.8,
                seed: 7
            })
            .unwrap()
        );
    }
}
