//! Native CPU sampling-prefix execution. The old GGML graph is not linked.
//! TOP_K and ARGMAX use the shared Rust tensor operators because their tie and
//! ordering behavior deliberately differs from the standalone CPU samplers.
use super::*;
use llama_tensor::{Graph, Op, Tensor};

#[derive(Clone)]
pub(crate) enum Stage {
    Empty,
    Greedy,
    TopK(usize),
    Temperature(f32),
}

pub(crate) struct Plan {
    stages: Vec<Stage>,
    width: usize,
}

#[derive(Default)]
pub(crate) struct Row {
    pub(crate) token: Option<i32>,
    pub(crate) logits: Vec<f32>,
    pub(crate) probs: Vec<f32>,
    pub(crate) candidates: Vec<i32>,
}

impl Plan {
    pub(crate) fn output_width(&self) -> usize {
        self.width
    }
}

fn indices(values: &[f32], op: Op) -> Result<Vec<i32>> {
    let mut graph = Graph::new();
    let input =
        graph.input(Tensor::from_f32([values.len(), 1, 1, 1], values).map_err(|e| e.to_string())?);
    let selected = graph.op(op, &[input]).map_err(|e| e.to_string())?;
    graph
        .evaluate(selected)
        .and_then(|v| v.to_i32())
        .map_err(|e| e.to_string())
}

pub(crate) fn execute(plan: &Plan, logits: Vec<f32>) -> Result<Row> {
    let mut row = Row {
        logits,
        ..Row::default()
    };
    for stage in &plan.stages {
        match *stage {
            Stage::Empty => {}
            Stage::Greedy => {
                let index = indices(&row.logits, Op::ArgMax)?[0] as usize;
                row.token = Some(if row.candidates.is_empty() {
                    index as i32
                } else {
                    row.candidates[index]
                });
            }
            Stage::TopK(count) => {
                let selected = indices(&row.logits, Op::TopK { count })?;
                gather(&mut row, &selected);
            }
            Stage::Temperature(t) if t <= 0.0 => {
                let selected = indices(&row.logits, Op::ArgMax)?;
                gather(&mut row, &selected);
            }
            Stage::Temperature(t) => {
                let scale = 1.0_f32 / t;
                // ggml_scale executes scale*value + zero, not value/temp.
                for value in &mut row.logits {
                    *value = *value * scale + 0.0;
                }
            }
        }
    }
    Ok(row)
}

fn gather(row: &mut Row, selected: &[i32]) {
    let candidates = selected
        .iter()
        .map(|&i| {
            if row.candidates.is_empty() {
                i
            } else {
                row.candidates[i as usize]
            }
        })
        .collect();
    row.logits = selected.iter().map(|&i| row.logits[i as usize]).collect();
    row.candidates = candidates;
}

/// No mutations occur until the whole admitted prefix has been checked and the
/// required native sort/numerical profile has executed successfully.
pub(crate) unsafe fn initialize(
    sampler: *mut LlamaSampler,
    vocabulary: usize,
) -> Result<Option<Plan>> {
    let h = unsafe { handle(sampler)? };
    if native_index(h.iface) != Some(CHAIN) {
        return Ok(None);
    }
    let chain = unsafe { chain_state(sampler)? };
    if chain.children.is_empty() {
        return Ok(None);
    }
    if chain.backend_prefix.is_some() {
        return Err("a sampler chain can be initialized for backend sampling only once".into());
    }
    if registry()
        .lock()
        .map_err(|_| "sampler registry poisoned")?
        .get(&(sampler as usize))
        .ok_or("missing sampler")?
        .owner
        .is_some()
    {
        return Err("an attached sampler must be an unowned top-level chain".into());
    }
    let children = chain.children.clone();
    let mut stages = Vec::new();
    let mut width = vocabulary;
    for child in &children {
        let h = unsafe { handle(*child)? };
        if native_index(h.iface).is_none() {
            let iface = unsafe { reference(h.iface)? };
            if iface.backend_init.is_some() {
                return Err(
                    "custom GGML backend sampler callbacks require an unimplemented tensor C ABI"
                        .into(),
                );
            }
            break;
        }
        let stage = match unsafe { state(*child)? } {
            State::Empty(_) => Stage::Empty,
            State::Builtin(b) => match *b.sampler.config() {
                Config::Greedy => Stage::Greedy,
                Config::TopK { k } => {
                    let count = usize::try_from(k).map_err(|_| "negative backend top-k")?;
                    if count == 0 || count > width {
                        return Err("backend top-k must fit its current candidate width".into());
                    }
                    width = count;
                    Stage::TopK(count)
                }
                Config::Temperature { temperature } => {
                    if temperature <= 0.0 {
                        width = 1;
                    }
                    Stage::Temperature(temperature)
                }
                Config::Dist { .. }
                | Config::TopP { .. }
                | Config::MinP { .. }
                | Config::DynamicTemperature { .. }
                | Config::Penalties { .. }
                | Config::LogitBias { .. } => {
                    return Err("this native backend sampler stage is not yet implemented; keep the complete chain on the CPU".into());
                }
                _ => break,
            },
            State::Bias(_) => return Err("backend logit-bias graph is not yet implemented".into()),
            State::Chain(_) => {
                return Err("nested backend sampler chains are not yet implemented".into());
            }
            // These original interfaces have no backend_init; the remaining
            // chain is the source-defined CPU suffix, including later stages.
            State::Dry(_) | State::Infill(_) | State::Grammar(_) => break,
        };
        stages.push(stage);
    }
    let plan = Plan { stages, width };
    // This also validates the platform-specific TOP_K implementation and catches
    // incompatible shape sequences before inference state can be changed.
    execute(&plan, vec![0.0; vocabulary])?;
    for child in children.iter().take(plan.stages.len()) {
        if let State::Builtin(value) = unsafe { state_mut(*child)? } {
            value.name = CString::new(format!("+{}", value.sampler.name()))
                .map_err(|_| "invalid sampler name")?;
        }
    }
    unsafe { chain_state_mut(sampler)? }.backend_prefix = Some(plan.stages.len());
    registry()
        .lock()
        .map_err(|_| "sampler registry poisoned")?
        .get_mut(&(sampler as usize))
        .ok_or("missing sampler registry entry")?
        .attachments += 1;
    Ok(Some(plan))
}

pub(crate) fn release(sampler: *mut LlamaSampler) {
    if let Ok(mut registry) = registry().lock()
        && let Some(entry) = registry.get_mut(&(sampler as usize))
    {
        entry.attachments = entry.attachments.saturating_sub(1);
    }
}

pub(crate) unsafe fn validate_live(sampler: *mut LlamaSampler) -> Result<()> {
    unsafe {
        handle(sampler)?;
    }
    Ok(())
}
