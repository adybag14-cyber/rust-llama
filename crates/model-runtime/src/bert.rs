//! Dense bidirectional BERT encoder from `src/models/bert.cpp`.
use crate::{Error, Matrix, Result, check_epsilon, check_positive, math};

pub mod native;

#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocabulary: usize,
    pub width: usize,
    pub feed_forward: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub max_positions: usize,
    pub layer_norm_epsilon: f32,
}
impl BertConfig {
    fn validate(&self) -> Result<()> {
        check_positive(&[
            ("vocabulary", self.vocabulary),
            ("width", self.width),
            ("feed_forward", self.feed_forward),
            ("heads", self.heads),
            ("kv_heads", self.kv_heads),
            ("max_positions", self.max_positions),
        ])?;
        check_epsilon(self.layer_norm_epsilon)?;
        if !self.width.is_multiple_of(self.heads) || !self.heads.is_multiple_of(self.kv_heads) {
            return Err(Error::InvalidConfig(
                "BERT width/head/KV grouping mismatch".into(),
            ));
        }
        Ok(())
    }
    /// Checked dimensions are available even before constructing a model.
    pub fn head_dim(&self) -> Result<usize> {
        crate::checked_head_dimension(self.width, self.heads)
    }
    pub fn kv_width(&self) -> Result<usize> {
        crate::checked_kv_width(self.width, self.heads, self.kv_heads)
    }
}

#[derive(Debug, Clone)]
pub struct BertLayer {
    pub query: Matrix,
    pub key: Matrix,
    pub value: Matrix,
    pub query_bias: Option<Vec<f32>>,
    pub key_bias: Option<Vec<f32>>,
    pub value_bias: Option<Vec<f32>>,
    pub attention_output: Matrix,
    pub attention_output_bias: Option<Vec<f32>>,
    pub attention_norm: Vec<f32>,
    pub attention_norm_bias: Vec<f32>,
    pub up: Matrix,
    pub up_bias: Option<Vec<f32>>,
    pub down: Matrix,
    pub down_bias: Option<Vec<f32>>,
    pub output_norm: Vec<f32>,
    pub output_norm_bias: Vec<f32>,
}
impl BertLayer {
    fn validate(&self, c: &BertConfig) -> Result<()> {
        self.query.shape(c.width, c.width, "BERT query")?;
        self.key.shape(c.kv_width()?, c.width, "BERT key")?;
        self.value.shape(c.kv_width()?, c.width, "BERT value")?;
        self.attention_output
            .shape(c.width, c.width, "BERT attention output")?;
        self.up.shape(c.feed_forward, c.width, "BERT FFN up")?;
        self.down.shape(c.width, c.feed_forward, "BERT FFN down")?;
        for (name, v) in [
            ("attention norm", &self.attention_norm),
            ("attention norm bias", &self.attention_norm_bias),
            ("output norm", &self.output_norm),
            ("output norm bias", &self.output_norm_bias),
        ] {
            math::vector(v, c.width, name)?;
        }
        for (name, v, len) in [
            ("query bias", &self.query_bias, c.width),
            ("key bias", &self.key_bias, c.kv_width()?),
            ("value bias", &self.value_bias, c.kv_width()?),
            (
                "attention output bias",
                &self.attention_output_bias,
                c.width,
            ),
            ("up bias", &self.up_bias, c.feed_forward),
            ("down bias", &self.down_bias, c.width),
        ] {
            math::optional_bias(v, len, name)?;
        }
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct BertWeights {
    pub token_embedding: Matrix,
    pub position_embedding: Matrix,
    /// The frozen BERT graph uses row 0 for every token.
    pub token_type_embedding: Option<Matrix>,
    pub embedding_norm: Vec<f32>,
    pub embedding_norm_bias: Vec<f32>,
    pub layers: Vec<BertLayer>,
}
#[derive(Debug, Clone)]
pub struct BertModel {
    config: BertConfig,
    weights: BertWeights,
    classifier: BertClassifier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    None,
    Mean,
    First,
    Last,
    /// Source classifier/reranking head, including its optional dense tanh layer.
    Rank,
}
/// Optional classifier tensors from the frozen BERT graph. Biases without their
/// corresponding matrix are retained and validated, but do not execute.
#[derive(Debug, Clone)]
pub struct BertClassifier {
    pub dense: Option<Matrix>,
    pub dense_bias: Option<Vec<f32>>,
    pub output: Option<Matrix>,
    pub output_bias: Option<Vec<f32>>,
    pub output_count: usize,
    pub mean_pooling: bool,
}

impl Default for BertClassifier {
    fn default() -> Self {
        Self {
            dense: None,
            dense_bias: None,
            output: None,
            output_bias: None,
            output_count: 1,
            mean_pooling: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    pub tokens: Vec<Vec<f32>>,
    pub pooled: Option<Vec<f32>>,
}

impl BertModel {
    pub fn new(config: BertConfig, weights: BertWeights) -> Result<Self> {
        config.validate()?;
        check_positive(&[("layers", weights.layers.len())])?;
        weights
            .token_embedding
            .shape(config.vocabulary, config.width, "BERT token embedding")?;
        weights.position_embedding.shape(
            config.max_positions,
            config.width,
            "BERT position embedding",
        )?;
        if let Some(types) = &weights.token_type_embedding {
            types.shape(types.rows(), config.width, "BERT token types")?;
        }
        math::vector(&weights.embedding_norm, config.width, "BERT embedding norm")?;
        math::vector(
            &weights.embedding_norm_bias,
            config.width,
            "BERT embedding norm bias",
        )?;
        for layer in &weights.layers {
            layer.validate(&config)?;
        }
        Ok(Self {
            config,
            weights,
            classifier: BertClassifier {
                output_count: 1,
                ..BertClassifier::default()
            },
        })
    }
    pub fn with_classifier(mut self, classifier: BertClassifier) -> Result<Self> {
        check_positive(&[("BERT classifier output count", classifier.output_count)])?;
        if let Some(matrix) = &classifier.dense {
            matrix.shape(
                self.config.width,
                self.config.width,
                "BERT classifier dense",
            )?;
        }
        if let Some(matrix) = &classifier.output {
            matrix.shape(
                classifier.output_count,
                self.config.width,
                "BERT classifier output",
            )?;
        }
        math::optional_bias(
            &classifier.dense_bias,
            self.config.width,
            "BERT classifier dense bias",
        )?;
        math::optional_bias(
            &classifier.output_bias,
            classifier.output_count,
            "BERT classifier output bias",
        )?;
        self.classifier = classifier;
        Ok(self)
    }
    pub fn classifier(&self) -> &BertClassifier {
        &self.classifier
    }
    fn classify(&self, mut pooled: Vec<f32>) -> Result<Vec<f32>> {
        if let Some(matrix) = &self.classifier.dense {
            pooled = matrix.multiply(&pooled)?;
            math::bias(&mut pooled, &self.classifier.dense_bias);
            for value in &mut pooled {
                *value = value.tanh();
            }
        }
        if let Some(matrix) = &self.classifier.output {
            pooled = matrix.multiply(&pooled)?;
            math::bias(&mut pooled, &self.classifier.output_bias);
        }
        math::finite(&pooled, "BERT classifier output")?;
        Ok(pooled)
    }
    pub fn config(&self) -> &BertConfig {
        &self.config
    }
    pub fn weights(&self) -> &BertWeights {
        &self.weights
    }
    pub fn encode(&self, tokens: &[u32], pooling: Pooling) -> Result<Encoded> {
        let positions: Vec<usize> = (0..tokens.len()).collect();
        self.encode_positions_with_cancel(tokens, &positions, pooling, || false)
    }
    pub fn encode_positions(
        &self,
        tokens: &[u32],
        positions: &[usize],
        pooling: Pooling,
    ) -> Result<Encoded> {
        self.encode_positions_with_cancel(tokens, positions, pooling, || false)
    }
    pub fn encode_positions_with_cancel(
        &self,
        tokens: &[u32],
        positions: &[usize],
        pooling: Pooling,
        mut cancel: impl FnMut() -> bool,
    ) -> Result<Encoded> {
        let c = &self.config;
        if tokens.is_empty() {
            return Err(Error::InvalidConfig(
                "BERT encoding needs at least one token".into(),
            ));
        }
        if tokens.len() != positions.len() {
            return Err(Error::Shape("BERT token/position lengths differ".into()));
        }
        if tokens.len() > c.max_positions {
            return Err(Error::ContextFull {
                maximum: c.max_positions,
            });
        }
        let mut hidden = Vec::with_capacity(tokens.len());
        for (&token, &position) in tokens.iter().zip(positions) {
            if cancel() {
                return Err(Error::Cancelled);
            }
            if token as usize >= c.vocabulary {
                return Err(Error::InvalidToken {
                    token,
                    vocabulary: c.vocabulary,
                });
            }
            if position >= c.max_positions {
                return Err(Error::ContextFull {
                    maximum: c.max_positions,
                });
            }
            let mut embedding = self.weights.token_embedding.row(token as usize)?.to_vec();
            // Preserve the frozen graph's grouping: token + type, then position.
            // Reordering three F32 additions can change a pretrained model output.
            if let Some(types) = &self.weights.token_type_embedding {
                math::add(&mut embedding, types.row(0)?);
            }
            math::add(
                &mut embedding,
                self.weights.position_embedding.row(position)?,
            );
            hidden.push(math::layer_norm(
                &embedding,
                &self.weights.embedding_norm,
                &self.weights.embedding_norm_bias,
                c.layer_norm_epsilon,
            )?);
        }
        let scale = (c.head_dim()? as f32).sqrt().recip();
        for layer in &self.weights.layers {
            if cancel() {
                return Err(Error::Cancelled);
            }
            let mut queries = Vec::with_capacity(tokens.len());
            let mut keys = Vec::with_capacity(tokens.len() * c.kv_width()?);
            let mut values = Vec::with_capacity(tokens.len() * c.kv_width()?);
            for input in &hidden {
                let mut q = layer.query.multiply(input)?;
                math::bias(&mut q, &layer.query_bias);
                queries.push(q);
                let mut k = layer.key.multiply(input)?;
                math::bias(&mut k, &layer.key_bias);
                keys.extend(k);
                let mut v = layer.value.multiply(input)?;
                math::bias(&mut v, &layer.value_bias);
                values.extend(v);
            }
            let mut next = Vec::with_capacity(tokens.len());
            for (input, query) in hidden.iter().zip(queries) {
                if cancel() {
                    return Err(Error::Cancelled);
                }
                // Every token sees all positions in this sequence. There is no
                // decoder causal mask or reused decoder KV state here.
                let attention = math::attention(
                    &query,
                    &keys,
                    &values,
                    c.heads,
                    c.kv_heads,
                    c.head_dim()?,
                    scale,
                )?;
                let mut out = layer.attention_output.multiply(&attention)?;
                math::bias(&mut out, &layer.attention_output_bias);
                math::add(&mut out, input);
                let norm = math::layer_norm(
                    &out,
                    &layer.attention_norm,
                    &layer.attention_norm_bias,
                    c.layer_norm_epsilon,
                )?;
                let mut up = layer.up.multiply(&norm)?;
                math::bias(&mut up, &layer.up_bias);
                for x in &mut up {
                    *x = math::gelu(*x);
                }
                let mut ffn = layer.down.multiply(&up)?;
                math::bias(&mut ffn, &layer.down_bias);
                math::add(&mut ffn, &norm);
                let output = math::layer_norm(
                    &ffn,
                    &layer.output_norm,
                    &layer.output_norm_bias,
                    c.layer_norm_epsilon,
                )?;
                math::finite(&output, "BERT layer output")?;
                next.push(output);
            }
            hidden = next;
        }
        if cancel() {
            return Err(Error::Cancelled);
        }
        let selected_pooling = if pooling == Pooling::Rank {
            if self.classifier.mean_pooling {
                Pooling::Mean
            } else {
                Pooling::First
            }
        } else {
            pooling
        };
        let pooled = match selected_pooling {
            Pooling::None => None,
            Pooling::First | Pooling::Rank => Some(hidden[0].clone()),
            Pooling::Last => Some(hidden[hidden.len() - 1].clone()),
            Pooling::Mean => {
                // The frozen pooling graph multiplies by a repeated F32
                // reciprocal, then uses the scalar F32 dot product's F64 sum.
                // Summing first and dividing afterward changes rounding.
                let reciprocal = 1.0f32 / hidden.len() as f32;
                let mut sum = vec![0.0f64; c.width];
                for token in &hidden {
                    for (sum, value) in sum.iter_mut().zip(token) {
                        *sum += f64::from(*value * reciprocal);
                    }
                }
                Some(sum.into_iter().map(|value| value as f32).collect())
            }
        };
        let pooled = if pooling == Pooling::Rank {
            pooled
                .map(|values| {
                    let mut values = self.classify(values)?;
                    if self.classifier.output_count > values.len() {
                        return Err(Error::Unsupported(
                            "BERT classifier extraction exceeds its graph width".into(),
                        ));
                    }
                    values.truncate(self.classifier.output_count);
                    Ok(values)
                })
                .transpose()?
        } else {
            pooled
        };
        if let Some(values) = &pooled {
            math::finite(values, "BERT pooled embedding")?;
        }
        Ok(Encoded {
            tokens: hidden,
            pooled,
        })
    }
}
