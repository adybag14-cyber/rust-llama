//! Context-owned sampled output storage and the frozen eight-function API.
use super::*;
use crate::LlamaSampler;
use crate::sampler::backend::{self, Plan, Row};
use std::collections::BTreeMap;

const MAX_SAMPLING_BYTES: usize = 256 * 1024 * 1024;

pub(crate) struct Attachment {
    sampler: *mut LlamaSampler,
    plan: Plan,
}

pub(crate) struct Sampling {
    attached: BTreeMap<i32, Attachment>,
    pub(crate) outputs: Outputs,
    pub(crate) full_vocabulary: Vec<i32>,
}

#[derive(Default)]
pub(crate) struct Outputs {
    pub(crate) allocated: bool,
    pub(crate) rows: Vec<Option<Row>>,
}

impl Sampling {
    pub(crate) fn new(vocabulary: usize) -> Result<Self> {
        if vocabulary == 0 || vocabulary > 1_000_000 {
            return Err("backend sampling vocabulary must contain 1..=1000000 entries".into());
        }
        Ok(Self {
            attached: BTreeMap::new(),
            outputs: Outputs::default(),
            full_vocabulary: (0..vocabulary as i32).collect(),
        })
    }
    pub(crate) fn active(&self) -> bool {
        !self.attached.is_empty()
    }
    pub(crate) fn initialize_outputs(&mut self) {
        self.outputs.allocated = self.active();
    }
    pub(crate) fn preflight(&self, batch: &[NativeBatchToken]) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        let mut counts = BTreeMap::<i32, usize>::new();
        let mut bytes = 0usize;
        for row in batch {
            if row.output == OutputSelection::None {
                continue;
            }
            for sequence in &row.sequences {
                if let Some(attachment) = self.attached.get(sequence) {
                    // SAFETY: exclusive context access, with the lifetime lease
                    // preventing sampler destruction while attached.
                    unsafe {
                        backend::validate_live(attachment.sampler)?;
                    }
                    let count = counts.entry(*sequence).or_default();
                    *count += 1;
                    if *count > 1 {
                        return Err(
                            "backend sampling supports at most one output per attached sequence"
                                .into(),
                        );
                    }
                }
            }
            if let Some(attachment) = row.sequences.first().and_then(|id| self.attached.get(id)) {
                bytes = bytes
                    .checked_add(
                        attachment
                            .plan
                            .output_width()
                            .checked_mul(12)
                            .ok_or("sampling output size overflow")?,
                    )
                    .ok_or("sampling allocation overflow")?;
            }
        }
        // Include one transient raw candidate tensor and its graph copy.
        bytes = bytes
            .checked_add(
                self.full_vocabulary
                    .len()
                    .checked_mul(16)
                    .ok_or("sampling scratch overflow")?,
            )
            .ok_or("sampling allocation overflow")?;
        if bytes > MAX_SAMPLING_BYTES {
            return Err("backend sampling output/scratch budget exceeds 256 MiB".into());
        }
        Ok(())
    }
    pub(crate) fn native_outputs(
        &self,
        batch: &[NativeBatchToken],
        outputs: &[llama_model_runtime::TokenOutput],
    ) -> Result<Outputs> {
        if !self.active() {
            return Ok(Outputs::default());
        }
        let mut rows = Vec::new();
        for output in outputs {
            let Some(logits) = &output.logits else {
                continue;
            };
            let attachment = batch[output.batch_index]
                .sequences
                .first()
                .and_then(|id| self.attached.get(id));
            rows.push(match attachment {
                Some(attachment) => Some(backend::execute(&attachment.plan, logits.clone())?),
                None => None,
            });
        }
        Ok(Outputs {
            allocated: true,
            rows,
        })
    }
}

impl Drop for Sampling {
    fn drop(&mut self) {
        for entry in self.attached.values() {
            backend::release(entry.sampler);
        }
    }
}

pub(crate) unsafe fn set(
    c: &mut ContextHandle,
    sequence: i32,
    sampler: *mut LlamaSampler,
) -> Result<bool> {
    if sampler.is_null() {
        if let Some(old) = c.sampling.attached.remove(&sequence) {
            backend::release(old.sampler);
        }
        return Ok(true);
    }
    c.decoder()?;
    if c.sampling.attached.len() >= 256 && !c.sampling.attached.contains_key(&sequence) {
        return Err("backend sampling exceeds 256 attached sequences".into());
    }
    let plan = unsafe { backend::initialize(sampler, c.model_handle.data.vocab.texts.len())? };
    if let Some(old) = c.sampling.attached.remove(&sequence) {
        backend::release(old.sampler);
    }
    let Some(plan) = plan else {
        return Ok(false);
    };
    c.sampling
        .attached
        .insert(sequence, Attachment { sampler, plan });
    Ok(true)
}

ffi!(
    llama_set_sampler(
        context: *mut ContextHandle,
        sequence: i32,
        sampler: *mut LlamaSampler,
    ) -> bool,
    false,
    { unsafe { set(reference_mut(context)?, sequence, sampler) } }
);

fn row(c: &ContextHandle, index: i32) -> Option<usize> {
    output_index(c, index)
        .ok()
        .filter(|&i| c.sampling.outputs.rows.get(i).is_some_and(Option::is_some))
}
fn data(c: &ContextHandle, index: i32) -> Option<&Row> {
    row(c, index).and_then(|i| c.sampling.outputs.rows[i].as_ref())
}
fn data_mut(c: &mut ContextHandle, index: i32) -> Option<&mut Row> {
    let i = row(c, index)?;
    c.sampling.outputs.rows[i].as_mut()
}

ffi!(
    llama_get_sampled_token_ith(context: *mut ContextHandle, index: i32) -> i32,
    -1,
    {
        let c = unsafe { reference_mut(context)? };
        c.performance.synchronize()?;
        Ok(data(c, index).and_then(|v| v.token).unwrap_or(-1))
    }
);
ffi!(
    llama_get_sampled_logits_ith(context: *mut ContextHandle, index: i32) -> *mut f32,
    ptr::null_mut(),
    {
        let c = unsafe { reference_mut(context)? };
        c.performance.synchronize()?;
        Ok(data_mut(c, index)
            .filter(|v| !v.logits.is_empty())
            .map_or(ptr::null_mut(), |v| v.logits.as_mut_ptr()))
    }
);
ffi!(
    llama_get_sampled_probs_ith(context: *mut ContextHandle, index: i32) -> *mut f32,
    ptr::null_mut(),
    {
        let c = unsafe { reference_mut(context)? };
        c.performance.synchronize()?;
        Ok(data_mut(c, index)
            .filter(|v| !v.probs.is_empty())
            .map_or(ptr::null_mut(), |v| v.probs.as_mut_ptr()))
    }
);
ffi!(
    llama_get_sampled_candidates_ith(context: *mut ContextHandle, index: i32) -> *mut i32,
    ptr::null_mut(),
    {
        let c = unsafe { reference_mut(context)? };
        c.performance.synchronize()?;
        let selected = row(c, index);
        if let Some(v) = selected
            .and_then(|i| c.sampling.outputs.rows[i].as_mut())
            .filter(|v| !v.candidates.is_empty())
        {
            return Ok(v.candidates.as_mut_ptr());
        }
        Ok(c.sampling.full_vocabulary.as_mut_ptr())
    }
);
ffi!(
    llama_get_sampled_logits_count_ith(context: *mut ContextHandle, index: i32) -> u32,
    0,
    {
        let c = unsafe { reference_mut(context)? };
        c.performance.synchronize()?;
        Ok(if !c.sampling.outputs.allocated {
            c.sampling.full_vocabulary.len() as u32
        } else {
            data(c, index).map_or(0, |v| v.logits.len() as u32)
        })
    }
);
ffi!(
    llama_get_sampled_probs_count_ith(context: *mut ContextHandle, index: i32) -> u32,
    0,
    {
        let c = unsafe { reference_mut(context)? };
        c.performance.synchronize()?;
        Ok(data(c, index).map_or(0, |v| v.probs.len() as u32))
    }
);
ffi!(
    llama_get_sampled_candidates_count_ith(context: *mut ContextHandle, index: i32) -> u32,
    0,
    {
        let c = unsafe { reference_mut(context)? };
        c.performance.synchronize()?;
        Ok(data(c, index).map_or(0, |v| v.candidates.len() as u32))
    }
);

pub(crate) fn logits_pointer(c: &mut ContextHandle, index: i32) -> Option<*mut f32> {
    data_mut(c, index)
        .filter(|r| !r.logits.is_empty())
        .map(|r| r.logits.as_mut_ptr())
}

pub(crate) unsafe fn candidates(
    context: *mut ContextHandle,
    index: i32,
) -> Result<(Option<i32>, Vec<LlamaTokenData>)> {
    let c = unsafe { reference_mut(context)? };
    c.performance.synchronize()?;
    c.decoder()?;
    if let Some(output) = data(c, index) {
        if let Some(token) = output.token {
            return Ok((Some(token), Vec::new()));
        }
        let ids = if output.candidates.is_empty() {
            &c.sampling.full_vocabulary
        } else {
            &output.candidates
        };
        let values = output
            .logits
            .iter()
            .enumerate()
            .map(|(i, &v)| LlamaTokenData {
                id: ids[i],
                logit: v,
                p: output.probs.get(i).copied().unwrap_or(0.0),
            })
            .collect();
        return Ok((None, values));
    }
    let row = output_index(c, index)?;
    let width = c.model_handle.data.vocab.texts.len();
    let offset = row.checked_mul(width).ok_or("sampling offset overflow")?;
    let values = c
        .logits
        .get(offset..offset + width)
        .ok_or("sampling logits are unavailable")?
        .iter()
        .enumerate()
        .map(|(i, &v)| LlamaTokenData {
            id: i as i32,
            logit: v,
            p: 0.0,
        })
        .collect();
    Ok((None, values))
}

#[cfg(test)]
mod tests;
