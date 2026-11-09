//! Ordinary LoRA composition and source-style post-layer control vectors.
use crate::{Error, Fingerprint, Matrix, Model, Result, math};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct Lora {
    pub target: String,
    /// A: rank output rows by input columns.
    pub a: Matrix,
    /// B: output rows by rank columns.
    pub b: Matrix,
    /// Zero preserves upstream's unscaled-rank convention; nonzero divides by rank.
    pub alpha: f32,
    pub scale: f32,
}

#[derive(Debug, Clone, Default)]
pub struct AdapterSet {
    pub lora: Vec<Lora>,
    /// Layer 0 has no control vector in the frozen source; keys start at 1.
    pub control_vectors: BTreeMap<usize, Vec<f32>>,
}
impl AdapterSet {
    pub fn validate(&self, model: &Model) -> Result<()> {
        for adapter in &self.lora {
            let weight = model.weight(&adapter.target).ok_or_else(|| {
                Error::InvalidAdapter(format!("unknown or non-matrix target {}", adapter.target))
            })?;
            if !adapter.alpha.is_finite() || !adapter.scale.is_finite() {
                return Err(Error::InvalidAdapter("LoRA scales must be finite".into()));
            }
            adapter.a.shape(adapter.a.rows(), weight.cols(), "LoRA A")?;
            adapter.b.shape(weight.rows(), adapter.a.rows(), "LoRA B")?;
        }
        for (&layer, vector) in &self.control_vectors {
            if layer == 0 || layer >= model.layer_count() {
                return Err(Error::InvalidAdapter(format!(
                    "control vector layer {layer} must be in 1..{}",
                    model.layer_count()
                )));
            }
            math::vector(vector, model.width(), "control vector")?;
        }
        Ok(())
    }
    pub(crate) fn multiply(&self, name: &str, weight: &Matrix, input: &[f32]) -> Result<Vec<f32>> {
        let mut output = weight.multiply(input)?;
        for adapter in self.lora.iter().filter(|x| x.target == name) {
            let intermediate = adapter.a.multiply(input)?;
            let correction = adapter.b.multiply(&intermediate)?;
            let factor = if adapter.alpha != 0.0 {
                adapter.scale * adapter.alpha / adapter.a.rows() as f32
            } else {
                adapter.scale
            };
            for (value, delta) in output.iter_mut().zip(correction) {
                *value += factor * delta;
            }
        }
        math::finite(&output, "adapted matrix output")?;
        Ok(output)
    }
    pub(crate) fn control(&self, layer: usize, hidden: &mut [f32]) {
        if let Some(vector) = self.control_vectors.get(&layer) {
            math::add(hidden, vector);
        }
    }
    pub(crate) fn fingerprint(&self) -> u64 {
        let mut hash = Fingerprint::new();
        hash.usize(self.lora.len());
        for item in &self.lora {
            hash.usize(item.target.len());
            hash.bytes(item.target.as_bytes());
            hash.matrix(&item.a);
            hash.matrix(&item.b);
            hash.floats(&[item.alpha, item.scale]);
        }
        hash.usize(self.control_vectors.len());
        for (&layer, values) in &self.control_vectors {
            hash.usize(layer);
            hash.floats(values);
        }
        hash.finish()
    }
}
